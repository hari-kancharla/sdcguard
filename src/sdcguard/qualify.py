"""Qualify a change to a vLLM server: start the stack you trust and each candidate, run every check on each, and
give one decision per candidate with its evidence.

    sdcguard qualify --out qual/ --policy policy.json \\
        --baseline "Qwen/Qwen2.5-7B-Instruct" \\
        --candidate awq="Qwen/Qwen2.5-7B-Instruct-AWQ" \\
        --candidate triton="Qwen/Qwen2.5-7B-Instruct --attention-backend TRITON_ATTN"

Each configuration is what follows `vllm serve` (the model and its flags). One at a time, on the same GPUs and port,
each is started with the sdcguard plugin on (SDCGUARD=1, its metrics and alert log; --share, default 1: every layer,
since a qualification run is a test and decide's default policy counts the plugin's silence only when it checks at
least half the weights; --no-plugin leaves it off) and, once it answers. The plugin doesn't refuse to serve a model
whose layers it can't all check (no SDCGUARD_REQUIRE, unless --require): an FP8 or MoE candidate is still started
and judged, and its decision says what wasn't checked (NOT DECIDED rather than ACCEPT when too little was).

  baseline     the quality reference (`gate model-ref --url`): greedy completions with their top-20 log-probabilities
  again        with --repeat-baseline, the baseline started a second time: its quality check is the noise floor
  candidates   the quality check against the reference (teacher-forced, `gate model-check --url`)
  every one    `sdcguard isolation` (primer dependence and canaries), `sdcguard consistency` (prefix cache cold vs
               hit, decode vs prefill), `sdcguard perf` (the same load each time), and
               each GPU worker's sdcguard metrics at the end (what the plugin checked), then the alerts it logged;
               with the plugin on, also the stack it read from inside the server (sdcguard.stack: attention and MoE
               backends, compilation and CUDA graph modes, versions, GPU, ...), recorded with the decision, so that
               `sdcguard ready --qualified decision_<name>.json` can later tell whether a running server is that stack

then stopped. Each candidate gets `sdcguard decide` against the baseline under the policy: decision_<name>.json and
.txt, and REPORT.md has one line per candidate. Each candidate not accepted also gets repro_<name>/: why, how its stack
differs from the baseline's, the evidence files, and rerun.sh, which starts it again and reruns only the checks that
failed. A server that doesn't start is reported, and its candidate is NOT DECIDED. Exit code 0 when every candidate is
ACCEPT, 1 when one is REJECT, 2 otherwise.

What this doesn't do: run your real traffic (the prompts are fixed sets; --quality-prompts and --prompts-file put your
own prompts in the quality reference and the speed load), or check what the plugin doesn't (see each decision's
scope). Stdlib only, apart from what vLLM itself needs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Optional


def _parse_config(text: str, default_name: str) -> tuple:
    name, sep, rest = text.partition("=")
    if sep and name and " " not in name and "/" not in name:
        return name, rest.strip()
    return default_name, text.strip()


def _tp(args: list) -> int:
    for i, a in enumerate(args):
        for flag in ("--tensor-parallel-size", "-tp"):
            if a == flag and i + 1 < len(args):
                return int(args[i + 1])
            if a.startswith(flag + "="):
                return int(a.split("=", 1)[1])
    return 1


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


class Server:
    def __init__(self, name: str, config: str, out: str, port: int, metrics_port: int, plugin: bool, share: str,
                 serve_cmd: list, extra_env: dict, require: Optional[str] = None):
        self.name, self.config, self.port, self.metrics_port = name, config, port, metrics_port
        self.args = shlex.split(config)
        self.workers = _tp(self.args)
        self.url = f"http://127.0.0.1:{port}"
        self.alerts = os.path.join(out, f"{name}_alerts.jsonl")
        env = {k: v for k, v in os.environ.items() if not k.startswith("SDCGUARD")}
        env.update(extra_env)
        if plugin:
            env.update(SDCGUARD="1", SDCGUARD_SHARE=share, SDCGUARD_METRICS_PORT=str(metrics_port),
                       SDCGUARD_LOG=self.alerts, SDCGUARD_EVIDENCE_DIR=os.path.join(out, f"{name}_evidence"))
            if require:
                env["SDCGUARD_REQUIRE"] = require
        self.cmd = [*serve_cmd, *self.args, "--port", str(port)]
        self.log_path = os.path.join(out, f"{name}_server.log")
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def wait(self, timeout: float) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                if _get(self.url + "/health")[0] == 200:
                    return True
            except OSError:
                pass
            time.sleep(2)
        return False

    def metrics(self) -> list:
        texts = []
        for i in range(self.workers):
            try:
                texts.append(_get(f"http://127.0.0.1:{self.metrics_port + i}/metrics", 10)[1].decode())
            except OSError:
                texts.append("")
        return texts

    def stop(self, timeout: float = 120.0) -> dict:
        t0 = time.monotonic()
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
                self.log.close()
                return {"exited_in_s": None, "killed": True}
        self.log.close()
        return {"exited_in_s": round(time.monotonic() - t0, 1), "killed": False, "code": self.proc.returncode}


def _write(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def run_one(role: str, name: str, config: str, a, out: str, ref: str, noise: Optional[str]) -> dict:
    from . import consistency, gate_model, isolation, perf
    res: dict = {"name": name, "role": role, "config": config}
    srv = Server(name, config, out, a.port, a.metrics_port, not a.no_plugin, a.share, a.serve_cmd, a.env_map,
                 a.require)
    res["command"] = srv.cmd
    try:
        t0 = time.monotonic()
        res["ready"] = srv.wait(a.startup_timeout)
        res["startup_s"] = round(time.monotonic() - t0, 1)
        if not res["ready"]:
            res["exit_code"] = srv.proc.poll()
            return res
        stack = {"config": config, **a.stack_map}
        if not a.no_plugin:                           # what the server runs, as the plugin read it from inside
            from .stack import from_metrics
            stacks = [from_metrics(t) for t in srv.metrics()]
            fps = {s.get("fingerprint") for s in stacks if s}
            if stacks and stacks[0]:
                stack["deployed"] = stacks[0]
                res["deployed_stack"] = stacks[0]
            if len(fps) > 1:
                res["stack_note"] = f"the workers publish different stacks: {sorted(fps)}"
        try:
            if role == "baseline":
                res["model_ref"] = gate_model.make_ref_url(srv.url, ref, a.tokens, extra_stack=stack,
                                                           prompts=_lines(a.quality_prompts))
            else:
                q = gate_model.check_url(ref, os.path.join(out, f"{name}_quality.json"), srv.url,
                                         noise_path=noise, extra_stack=stack)
                res["quality"] = q.get("verdict")
        except Exception as e:  # noqa: BLE001  (a quality check that couldn't run leaves the approximation unmeasured)
            res["quality_error"] = f"{type(e).__name__}: {e}"[:500]
        if not a.no_isolation:
            try:
                iso = isolation.run(srv.url)
                _write(os.path.join(out, f"{name}_isolation.json"), iso)
                res["isolation"] = iso["verdict"]
            except Exception as e:  # noqa: BLE001
                res["isolation_error"] = f"{type(e).__name__}: {e}"[:500]
        if not a.no_consistency:
            try:
                con = consistency.run(srv.url)
                _write(os.path.join(out, f"{name}_consistency.json"), con)
                res["consistency"] = con["verdict"]
            except Exception as e:  # noqa: BLE001
                res["consistency_error"] = f"{type(e).__name__}: {e}"[:500]
        if not a.no_perf:
            try:
                texts = None
                if a.prompts_file:
                    texts = [line.rstrip("\n") for line in open(a.prompts_file) if line.strip()]
                p = perf.measure(srv.url, texts=texts, concurrency=a.concurrency, max_tokens=a.max_tokens,
                                 rounds=a.rounds)
                _write(os.path.join(out, f"{name}_perf.json"), p)
                res["throughput"] = p["median"].get("output_tokens_per_s")
            except Exception as e:  # noqa: BLE001
                res["perf_error"] = f"{type(e).__name__}: {e}"[:500]
        if not a.no_plugin:
            time.sleep(3)                                 # the plugin reads its results at the next step after 2 s
            try:
                perf.one_request(srv.url, perf.server_model(srv.url) or "", "One more request.", 8)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
            texts = srv.metrics()
            for i, t in enumerate(texts):
                with open(os.path.join(out, f"{name}_metrics{i}.txt"), "w") as f:
                    f.write(t)
            res["metrics_read"] = sum(bool(t) for t in texts)
    finally:
        res["stop"] = srv.stop()
    return res


def _lines(path: Optional[str]) -> Optional[list]:
    if not path:
        return None
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


REPRO_FILES = ("decision_{n}.json", "decision_{n}.txt", "{n}_quality.json", "{n}_isolation.json",
               "{n}_consistency.json", "baseline_consistency.json", "{n}_alerts.jsonl",
               "{n}_perf.json", "{n}_server.log", "ref.json", "again_quality.json", "baseline_perf.json")


def write_repro(out: str, name: str, configs: dict, runs: dict, row: dict, a) -> Optional[str]:
    """For a candidate that wasn't accepted: repro_<name>/ with what failed and why, the two stacks and how they
    differ, the evidence files, and rerun.sh, which starts the candidate as qualify did and reruns only the checks
    that failed (against the baseline's reference and speed). Returns the folder."""
    import shutil
    from .stack import compare
    d = os.path.join(out, f"repro_{name}")
    os.makedirs(d, exist_ok=True)
    for pat in REPRO_FILES:
        src = os.path.join(out, pat.format(n=name))
        if os.path.exists(src):
            shutil.copy2(src, d)
    for i in range(16):
        src = os.path.join(out, f"{name}_metrics{i}.txt")
        if os.path.exists(src):
            shutil.copy2(src, d)
    ev = os.path.join(out, f"{name}_evidence")
    if os.path.isdir(ev):
        shutil.copytree(ev, os.path.join(d, "evidence"), dirs_exist_ok=True)
    base, cand = (runs.get("baseline") or {}).get("deployed_stack"), (runs.get(name) or {}).get("deployed_stack")
    diff = compare(base, cand) if base and cand else {"differences": [], "why": "no stack recorded for both"}
    serve = " ".join(shlex.quote(x) for x in a.serve_cmd)
    req = getattr(a, "require", None)
    env = {"SDCGUARD": "1", "SDCGUARD_SHARE": a.share, **({"SDCGUARD_REQUIRE": req} if req else {}),
           **a.env_map} if not a.no_plugin \
        else dict(a.env_map)
    failing = {"approximation": row.get("approximation") == "OUTSIDE" or row.get("approximation") == "UNMEASURED",
               "isolation": row.get("isolation") in ("LEAK", "UNMEASURED") and not a.no_isolation,
               "consistency": row.get("consistency") in ("INCONSISTENT", "UNMEASURED") and
               not getattr(a, "no_consistency", False),
               "arithmetic": row.get("arithmetic") not in ("NO FAULT OBSERVED", None) or
               row.get("weights") == "CHANGED",
               "quantization": row.get("quantization") == "WRONG",
               "performance": row.get("performance") == "MISSES" and bool(row.get("performance_required"))}
    steps = []
    if failing["approximation"]:
        noise = ' --noise "$HERE/again_quality.json"' if os.path.exists(os.path.join(d, "again_quality.json")) else ""
        steps.append(f'"$PY" -m sdcguard gate model-check "$HERE/ref.json" --url "$URL" --out "$HERE/rerun_quality.json"'
                     f"{noise}")
    if failing["isolation"]:
        steps.append('"$PY" -m sdcguard isolation --url "$URL" --out "$HERE/rerun_isolation.json"')
    if failing["consistency"]:
        steps.append('"$PY" -m sdcguard consistency --url "$URL" --out "$HERE/rerun_consistency.json"')
    if failing["performance"]:
        load = f"--concurrency {a.concurrency} --max-tokens {a.max_tokens} --rounds {a.rounds}"
        if a.prompts_file and os.path.exists(a.prompts_file):
            shutil.copy2(a.prompts_file, os.path.join(d, "perf_prompts.txt"))
            load += ' --prompts-file "$HERE/perf_prompts.txt"'
        steps.append(f'"$PY" -m sdcguard perf --url "$URL" {load} --out "$HERE/rerun_perf.json" && '
                     '"$PY" -m sdcguard perf --compare "$HERE/baseline_perf.json" "$HERE/rerun_perf.json"')
    if failing["quantization"]:                 # the FP8 check fires on traffic: short requests one at a time
        if not failing["isolation"]:
            steps.append('"$PY" -m sdcguard isolation --url "$URL" --canaries 16 --out "$HERE/rerun_isolation.json"')
        steps.append('grep -c serve_fp8 "$HERE/rerun_alerts.jsonl" 2>/dev/null | sed "s/^/FP8 quantization alerts: /"')
    if failing["arithmetic"]:
        steps.append('ls "$HERE/rerun_alerts.jsonl" 2>/dev/null && echo "alerts logged: $HERE/rerun_alerts.jsonl"')
        steps.append('for f in "$HERE"/evidence/*.pt; do [ -e "$f" ] && "$PY" -m sdcguard replay "$f"; done')
    envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
    script = f"""#!/bin/sh
# Rerun what failed for candidate {name}: start it as `sdcguard qualify` did, run the failing checks, stop it.
# PORT and MPORT pick the ports (default 8000 and 9400); PY the Python with vLLM and sdcguard installed.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PORT=${{PORT:-8000}}
MPORT=${{MPORT:-9400}}
PY=${{PY:-python3}}
URL=http://127.0.0.1:$PORT
{envs} SDCGUARD_METRICS_PORT=$MPORT SDCGUARD_LOG="$HERE/rerun_alerts.jsonl" SDCGUARD_EVIDENCE_DIR="$HERE/rerun_evidence" \\
  {serve} {configs[name]} --port $PORT > "$HERE/rerun_server.log" 2>&1 &
PID=$!
# (a background job of a script starts with SIGINT ignored, so it is stopped with SIGTERM, then SIGKILL after 60 s)
stop() {{ kill -TERM $PID 2>/dev/null; i=0; while kill -0 $PID 2>/dev/null && [ $i -lt 60 ]; do sleep 1; i=$((i+1)); done
  kill -KILL $PID 2>/dev/null; }}
trap stop EXIT
until "$PY" -c "import urllib.request,sys; urllib.request.urlopen(sys.argv[1] + '/health', timeout=5)" "$URL" 2>/dev/null; do
  kill -0 $PID 2>/dev/null || {{ echo "the server exited; see $HERE/rerun_server.log"; exit 3; }}
  sleep 3
done
{chr(10).join(steps) if steps else 'echo "nothing to rerun: see the decision for why it was not accepted"'}
"""
    with open(os.path.join(d, "rerun.sh"), "w") as f:
        f.write(script)
    os.chmod(os.path.join(d, "rerun.sh"), 0o755)
    lines = [f"# Candidate {name}: {row.get('decision')}", "",
             f"Baseline: `{serve} {configs['baseline']}`", "", f"Candidate: `{serve} {configs[name]}`", "",
             "## Why", "", *[f"- {w}" for w in row.get("why") or []], "",
             "## What differs between the two stacks (read from inside each server)", "",
             *([f"- {x}" for x in diff.get("differences") or []] or [f"- {diff.get('why')}"]), "",
             "## Rerun", "",
             "`./rerun.sh` starts the candidate the way qualify did and reruns only what failed: " +
             (", ".join(k for k, v in failing.items() if v) or "nothing (see the decision)") + ". Evidence files:",
             "", *[f"- {f}" for f in sorted(os.listdir(d)) if f not in ("README.md", "rerun.sh")], ""]
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write("\n".join(lines))
    return d


def _load(path: Optional[str]):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def tested_conditions(out: str, name: str, a=None) -> dict:
    """What a candidate's qualification exercised, for `sdcguard ready --qualified` to compare a deployment with: the
    longest prompt and context the checks sent, the most requests at once, whether a prefix-cache hit and the decode
    path past 128 tokens of KV were tested, and which probes ran."""
    t: dict = {"probes": []}
    prompts, contexts = [], []
    ref = _load(os.path.join(out, "ref.json"))
    if ref and _load(os.path.join(out, f"{name}_quality.json")):
        recs = ref.get("records") or []
        prompts += [len(r.get("prompt_ids") or []) for r in recs]
        contexts += [len(r.get("prompt_ids") or []) + len(r.get("completion_ids") or []) for r in recs]
        t["probes"].append("quality")
    con = _load(os.path.join(out, f"{name}_consistency.json"))
    if con:
        for c in (con.get("decode_prefill") or {}).get("cases") or []:
            prompts.append(c.get("prompt_tokens") or 0)
            contexts.append((c.get("prompt_tokens") or 0) + (c.get("tokens") or 0))
        for c in (con.get("prefix_cache") or {}).get("cases") or []:
            prompts.append(c.get("prompt_tokens") or 0)
            contexts.append((c.get("prompt_tokens") or 0) + len(c.get("cold_tokens") or []))
        t["prefix_cache_hit_tested"] = (con.get("prefix_cache") or {}).get("cache_hits_confirmed")
        t["decode_past_128_tested"] = any((c.get("prompt_tokens") or 0) + (c.get("tokens") or 0) > 128
                                          for c in (con.get("decode_prefill") or {}).get("cases") or [])
        t["probes"].append("consistency")
    conc = []
    if _load(os.path.join(out, f"{name}_isolation.json")):
        conc.append(32)
        t["probes"].append("isolation")
    perf_ = _load(os.path.join(out, f"{name}_perf.json"))
    if perf_:
        conc.append(int((perf_.get("load") or {}).get("concurrency") or getattr(a, "concurrency", 0) or 0))
        t["probes"].append("perf")
    t["max_prompt_tokens"] = max(prompts) if prompts else None
    t["max_context_tokens"] = max(contexts) if contexts else None
    t["max_concurrency"] = max(conc) if conc else None
    return t


OWN_FILES = ("ref.json", "RUNS.json", "QUALIFY.json", "REPORT.md")
OWN_PER_RUN = re.compile(r"^(?P<n>.+)_(alerts\.jsonl|evidence|server\.log|quality\.json|isolation\.json|"
                         r"consistency\.json|perf\.json|metrics\d+\.txt)$")


def set_aside_earlier(out: str, names: list) -> Optional[str]:
    """Move the files an earlier qualify left in `out` for these runs (the baseline, again and each candidate named:
    their results, logs, decisions and repro folders, and the reference and reports) into out/earlier_<time>/, so
    that this run decides only from what it measured: a candidate that doesn't start now can't be decided from an
    earlier run's results, and a check that fails to write now can't be read from an earlier one. Nothing is deleted,
    and other files (a policy or prompts kept there, say) stay where they are. Returns where the files went, or None
    when there were none."""
    runs = set(names) | {"baseline", "again"}
    mine = []
    for n in sorted(os.listdir(out)):
        m = OWN_PER_RUN.match(n)
        if n in OWN_FILES or (m and m.group("n") in runs) or \
                any(n in (f"decision_{r}.json", f"decision_{r}.txt", f"repro_{r}") for r in runs):
            mine.append(n)
    if not mine:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest, k = os.path.join(out, f"earlier_{stamp}"), 1
    while os.path.exists(dest):
        dest, k = os.path.join(out, f"earlier_{stamp}_{k}"), k + 1
    os.makedirs(dest)
    for n in mine:
        os.rename(os.path.join(out, n), os.path.join(dest, n))
    return dest


def decide_all(out: str, names: list, pol: Optional[dict], plugin: bool, runs: Optional[dict] = None) -> dict:
    """A decision per candidate from what this run measured. With `runs`, a candidate that didn't start in this run
    (or whose baseline didn't) is NOT DECIDED, whatever files are on disk."""
    from . import decide as dc
    base_perf = _load(os.path.join(out, "baseline_perf.json"))
    rows = {}
    for name in names:
        if runs is not None and not ((runs.get(name) or {}).get("ready") and (runs.get("baseline") or {}).get("ready")):
            why = (runs.get(name) or {}).get("skipped") or ("the baseline didn't start in this run"
                                                             if not (runs.get("baseline") or {}).get("ready") else
                                                             "the candidate didn't start in this run")
            rows[name] = {"decision": "NOT DECIDED", "arithmetic": None, "weights": None, "approximation": None,
                          "quality_check": None, "quantization": None, "isolation": None, "consistency": None,
                          "performance": None, "performance_required": False, "throughput_change": None,
                          "why": [f"no evidence from this run: {why}"]}
            continue
        alerts, cov = None, None
        if plugin:
            path = os.path.join(out, f"{name}_alerts.jsonl")
            alerts = dc._read_jsonl(path) if os.path.exists(path) else []
            mets = sorted(p for p in os.listdir(out) if p.startswith(f"{name}_metrics") and p.endswith(".txt"))
            texts = [open(os.path.join(out, p)).read() for p in mets]
            cov = dc.plugin_coverage(texts) if texts and all(texts) else None
        d = dc.decide(pol, quality=_load(os.path.join(out, f"{name}_quality.json")), alerts=alerts,
                      perf_before=base_perf, perf_after=_load(os.path.join(out, f"{name}_perf.json")), plugin=cov,
                      isolation=_load(os.path.join(out, f"{name}_isolation.json")),
                      consistency=_load(os.path.join(out, f"{name}_consistency.json")),
                      consistency_baseline=_load(os.path.join(out, "baseline_consistency.json")))
        d["scope"]["tested"] = tested_conditions(out, name)
        _write(os.path.join(out, f"decision_{name}.json"), d)
        with open(os.path.join(out, f"decision_{name}.txt"), "w") as f:
            f.write(dc.render(d) + "\n")
        tp = ((d["performance"].get("metrics") or {}).get("output_tokens_per_s") or {}).get("change")
        rows[name] = {"decision": d["decision"], "arithmetic": d["arithmetic"]["verdict"],
                      "weights": (d.get("weights") or {}).get("verdict"),
                      "approximation": d["approximation"]["verdict"], "quality_check": d["approximation"].get("check"),
                      "quantization": d["quantization"]["verdict"],
                      "isolation": d["isolation"]["verdict"], "consistency": d["consistency"]["verdict"],
                      "performance": d["performance"]["verdict"],
                      "performance_required": bool(d["performance"].get("required")),
                      "throughput_change": tp, "why": d["why"]}
    return rows


def report(out: str, configs: dict, runs: dict, rows: dict) -> str:
    lines = ["# Qualification", "", f"Baseline: `vllm serve {configs['baseline']}`", "",
             "| Candidate | Decision | Arithmetic | Weights | Approximation | Isolation | Consistency | Performance | "
             "Throughput |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        tp = "" if r["throughput_change"] is None else f"{r['throughput_change']:+.1%}"
        q = (r.get("approximation") or "") + (f" ({r['quality_check']})" if r.get("quality_check") else "")
        lines.append(f"| {name} | {r['decision']} | {r.get('arithmetic') or ''} | {r.get('weights') or ''} | {q} | "
                     f"{r.get('isolation') or ''} | "
                     f"{r.get('consistency') or ''} | {r.get('performance') or ''} | {tp} |")
    lines.append("")
    for name, r in rows.items():
        lines.append(f"- **{name}** (`vllm serve {configs[name]}`): {r['decision']}: " + "; ".join(r["why"]))
    not_started = [n for n, r in runs.items() if not r.get("ready")]
    if not_started:
        lines += ["", "Servers that didn't start: " + ", ".join(not_started) + " (their logs are here)."]
    lines += ["", "Each candidate's full decision, with what was and wasn't checked, is in decision_<name>.txt."]
    repros = [n for n, r in rows.items() if r.get("repro")]
    if repros:
        lines.append("For each one not accepted, repro_<name>/ has the evidence, the two stacks' differences and "
                     "rerun.sh, which reruns only what failed: " + ", ".join(repros) + ".")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "REPORT.md"), "w") as f:
        f.write(text)
    return text


def _pairs(items) -> dict:
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        out[k] = v
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sdcguard qualify", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, help="what follows `vllm serve` for the stack you trust")
    p.add_argument("--candidate", action="append", required=True, help="[name=]what follows `vllm serve` (repeat)")
    p.add_argument("--policy", help="decide's policy file (JSON)")
    p.add_argument("--out", required=True)
    p.add_argument("--repeat-baseline", action="store_true", help="start the baseline twice: the noise floor")
    p.add_argument("--no-plugin", action="store_true", help="serve without the sdcguard plugin (no arithmetic source)")
    p.add_argument("--no-isolation", action="store_true")
    p.add_argument("--no-consistency", action="store_true", help="skip `sdcguard consistency` on every server")
    p.add_argument("--no-perf", action="store_true")
    p.add_argument("--share", default="1", help="SDCGUARD_SHARE for every server (default 1: every layer; decide's "
                   "default policy counts the plugin's silence only when it checks at least half the weights)")
    p.add_argument("--require", default=None, metavar="FLOOR",
                   help="SDCGUARD_REQUIRE for every server: refuse to serve with less checked (default: not set)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--metrics-port", type=int, default=9400)
    p.add_argument("--tokens", type=int, default=64, help="tokens per reference completion")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--prompts-file", default=None, help="prompts for the speed load, one per line")
    p.add_argument("--quality-prompts", default=None, help="your prompts for the quality reference, one per line "
                   "(default: a fixed built-in set of 32)")
    p.add_argument("--startup-timeout", type=float, default=1800.0)
    p.add_argument("--serve-cmd", default=None, help="the command that serves (default: this Python's vllm serve)")
    p.add_argument("--env", action="append", default=[], help="KEY=VALUE for every server (repeat)")
    p.add_argument("--stack", action="append", default=[], help="KEY=VALUE recorded with each stack (gpu=H100, say)")
    a = p.parse_args(argv)
    a.serve_cmd = shlex.split(a.serve_cmd) if a.serve_cmd else [sys.executable, "-m", "vllm.entrypoints.cli.main",
                                                                 "serve"]
    a.env_map, a.stack_map = _pairs(a.env), _pairs(a.stack)
    os.makedirs(a.out, exist_ok=True)
    configs = {"baseline": a.baseline}
    for i, c in enumerate(a.candidate):
        name, cfg = _parse_config(c, f"candidate{i + 1}")
        if name in configs or name == "again":
            p.error(f"two configurations are named {name}")
        configs[name] = cfg
    if a.policy and not os.path.isfile(a.policy):
        p.error(f"--policy {a.policy}: no such file (decide's default policy would be used in its place)")
    pol = _load(a.policy)
    moved = set_aside_earlier(a.out, list(configs))
    if moved:
        print(f"sdcguard qualify: {a.out} held an earlier run's files; moved them to {moved}", file=sys.stderr)
    ref = os.path.join(a.out, "ref.json")
    runs = {"baseline": run_one("baseline", "baseline", a.baseline, a, a.out, ref, None)}
    noise = None
    if runs["baseline"].get("ready") and a.repeat_baseline:
        runs["again"] = run_one("again", "again", a.baseline, a, a.out, ref, None)
        if os.path.exists(os.path.join(a.out, "again_quality.json")):
            noise = os.path.join(a.out, "again_quality.json")
    names = [n for n in configs if n != "baseline"]
    for name in names:
        if not runs["baseline"].get("ready"):
            runs[name] = {"name": name, "ready": None, "skipped": "the baseline didn't start"}
            continue
        runs[name] = run_one("candidate", name, configs[name], a, a.out, ref, noise)
    _write(os.path.join(a.out, "RUNS.json"), {"configs": configs, "runs": runs})
    rows = decide_all(a.out, names, pol, not a.no_plugin, runs)
    for name, row in rows.items():
        if row["decision"] != "ACCEPT" and (runs.get(name) or {}).get("ready"):
            try:
                row["repro"] = write_repro(a.out, name, configs, runs, row, a)
            except Exception as e:  # noqa: BLE001  (the decision stands without its repro folder)
                row["repro_error"] = f"{type(e).__name__}: {e}"[:300]
    _write(os.path.join(a.out, "QUALIFY.json"), {"configs": configs, "runs": runs, "decisions": rows})
    print(report(a.out, configs, runs, rows))
    ds = [r["decision"] for r in rows.values()]
    return 1 if "REJECT" in ds else (0 if ds and all(d == "ACCEPT" for d in ds) else 2)


if __name__ == "__main__":
    sys.exit(main())
