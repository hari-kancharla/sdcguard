"""Qualify a change to a vLLM server: start the stack you trust and each candidate, run every check on each, and
give one decision per candidate with its evidence.

    sdcguard qualify --out qual/ --policy policy.json \\
        --baseline "Qwen/Qwen2.5-7B-Instruct" \\
        --candidate awq="Qwen/Qwen2.5-7B-Instruct-AWQ" \\
        --candidate triton="Qwen/Qwen2.5-7B-Instruct --attention-backend TRITON_ATTN"

Each configuration is what follows `vllm serve` (the model and its flags). One at a time, on the same GPUs and port,
each is started with the sdcguard plugin on (SDCGUARD=1, SDCGUARD_REQUIRE=1, its metrics and alert log; --share,
default 1: every layer, since a qualification run is a test and decide's default policy counts the plugin's silence
only when it checks at least half the weights; --no-plugin leaves it off) and, once it answers:

  baseline     the quality reference (`gate model-ref --url`): greedy completions with their top-20 log-probabilities
  again        with --repeat-baseline, the baseline started a second time: its quality check is the noise floor
  candidates   the quality check against the reference (teacher-forced, `gate model-check --url`)
  every one    `sdcguard isolation` (primer dependence and canaries), `sdcguard perf` (the same load each time), and
               each GPU worker's sdcguard metrics at the end (what the plugin checked), then the alerts it logged

then stopped. Each candidate gets `sdcguard decide` against the baseline under the policy: decision_<name>.json and
.txt, and REPORT.md has one line per candidate. A server that doesn't start is reported, and its candidate is NOT
DECIDED. Exit code 0 when every candidate is ACCEPT, 1 when one is REJECT, 2 otherwise.

What this doesn't do: run your real traffic (the prompts are fixed sets, or --prompts-file for the speed load), or
check what the plugin doesn't (see each decision's scope). Stdlib only, apart from what vLLM itself needs.
"""
from __future__ import annotations

import argparse
import json
import os
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
                 serve_cmd: list, extra_env: dict):
        self.name, self.config, self.port, self.metrics_port = name, config, port, metrics_port
        self.args = shlex.split(config)
        self.workers = _tp(self.args)
        self.url = f"http://127.0.0.1:{port}"
        self.alerts = os.path.join(out, f"{name}_alerts.jsonl")
        env = {k: v for k, v in os.environ.items() if not k.startswith("SDCGUARD")}
        env.update(extra_env)
        if plugin:
            env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_SHARE=share, SDCGUARD_METRICS_PORT=str(metrics_port),
                       SDCGUARD_LOG=self.alerts, SDCGUARD_EVIDENCE_DIR=os.path.join(out, f"{name}_evidence"))
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
    from . import gate_model, isolation, perf
    res: dict = {"name": name, "role": role, "config": config}
    srv = Server(name, config, out, a.port, a.metrics_port, not a.no_plugin, a.share, a.serve_cmd, a.env_map)
    res["command"] = srv.cmd
    try:
        t0 = time.monotonic()
        res["ready"] = srv.wait(a.startup_timeout)
        res["startup_s"] = round(time.monotonic() - t0, 1)
        if not res["ready"]:
            res["exit_code"] = srv.proc.poll()
            return res
        stack = {"config": config, **a.stack_map}
        try:
            if role == "baseline":
                res["model_ref"] = gate_model.make_ref_url(srv.url, ref, a.tokens, extra_stack=stack)
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


def _load(path: Optional[str]):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def decide_all(out: str, names: list, pol: Optional[dict], plugin: bool) -> dict:
    from . import decide as dc
    base_perf = _load(os.path.join(out, "baseline_perf.json"))
    rows = {}
    for name in names:
        alerts, cov = None, None
        if plugin:
            path = os.path.join(out, f"{name}_alerts.jsonl")
            alerts = dc._read_jsonl(path) if os.path.exists(path) else []
            mets = sorted(p for p in os.listdir(out) if p.startswith(f"{name}_metrics") and p.endswith(".txt"))
            texts = [open(os.path.join(out, p)).read() for p in mets]
            cov = dc.plugin_coverage(texts) if texts and all(texts) else None
        d = dc.decide(pol, quality=_load(os.path.join(out, f"{name}_quality.json")), alerts=alerts,
                      perf_before=base_perf, perf_after=_load(os.path.join(out, f"{name}_perf.json")), plugin=cov,
                      isolation=_load(os.path.join(out, f"{name}_isolation.json")))
        _write(os.path.join(out, f"decision_{name}.json"), d)
        with open(os.path.join(out, f"decision_{name}.txt"), "w") as f:
            f.write(dc.render(d) + "\n")
        tp = ((d["performance"].get("metrics") or {}).get("output_tokens_per_s") or {}).get("change")
        rows[name] = {"decision": d["decision"], "arithmetic": d["arithmetic"]["verdict"],
                      "approximation": d["approximation"]["verdict"], "quality_check": d["approximation"].get("check"),
                      "isolation": d["isolation"]["verdict"], "performance": d["performance"]["verdict"],
                      "throughput_change": tp, "why": d["why"]}
    return rows


def report(out: str, configs: dict, runs: dict, rows: dict) -> str:
    lines = ["# Qualification", "", f"Baseline: `vllm serve {configs['baseline']}`", "",
             "| Candidate | Decision | Arithmetic | Approximation | Isolation | Performance | Throughput |",
             "|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        tp = "" if r["throughput_change"] is None else f"{r['throughput_change']:+.1%}"
        q = r["approximation"] + (f" ({r['quality_check']})" if r.get("quality_check") else "")
        lines.append(f"| {name} | {r['decision']} | {r['arithmetic']} | {q} | {r['isolation']} | {r['performance']} "
                     f"| {tp} |")
    lines.append("")
    for name, r in rows.items():
        lines.append(f"- **{name}** (`vllm serve {configs[name]}`): {r['decision']}: " + "; ".join(r["why"]))
    not_started = [n for n, r in runs.items() if not r.get("ready")]
    if not_started:
        lines += ["", "Servers that didn't start: " + ", ".join(not_started) + " (their logs are here)."]
    lines += ["", "Each candidate's full decision, with what was and wasn't checked, is in decision_<name>.txt."]
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
    p.add_argument("--no-perf", action="store_true")
    p.add_argument("--share", default="1", help="SDCGUARD_SHARE for every server (default 1: every layer; decide's "
                   "default policy counts the plugin's silence only when it checks at least half the weights)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--metrics-port", type=int, default=9400)
    p.add_argument("--tokens", type=int, default=64, help="tokens per reference completion")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--prompts-file", default=None, help="prompts for the speed load, one per line")
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
    pol = _load(a.policy)
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
    rows = decide_all(a.out, names, pol, not a.no_plugin)
    _write(os.path.join(a.out, "QUALIFY.json"), {"configs": configs, "runs": runs, "decisions": rows})
    print(report(a.out, configs, runs, rows))
    ds = [r["decision"] for r in rows.values()]
    return 1 if "REJECT" in ds else (0 if ds and all(d == "ACCEPT" for d in ds) else 2)


if __name__ == "__main__":
    sys.exit(main())
