"""Why the compiled server's answers changed with norm checks on (docs/METHOD.md, "Norm checks in serving", the
amendment written October 25 before this run).

The October 24 norms run found 5 of Qwen2.5-1.5B's 64 greedy completions different from plain vLLM's with
SDCGUARD_NORMS=1 on a compiled server. That comparison changed three things at once: the norm checks, the linear
checks, and PyTorch's cuBLAS reduced-precision setting (the plugin keeps bf16/fp16 partial sums in fp32), and plain
vLLM compiled afresh can change its own answers (Round 3: 28 of 256 on Qwen2.5-7B-AWQ, from Inductor re-tuning a
fused add+RMSNorm kernel). This run separates them. One H100, vLLM 0.32.0, Qwen2.5-1.5B-Instruct in bf16, prefix
caching off, every server compiled from empty caches of its own (VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR and
TRITON_CACHE_DIR under its folder, the compiled code saved unpacked so it can be read). Servers, one after another:

  plain_a       SDCGUARD=0
  plain_b       SDCGUARD=0 again: what a fresh compile alone changes
  plain_exact   SDCGUARD=0 with the plugin's cuBLAS setting (SDCGUARD_TEST_EXACT=1)
  nonorms       the plugin, linear checks on every layer, SDCGUARD_NORMS=0 (attention checks off, as in that run)
  guarded       the same with SDCGUARD_NORMS=1 (that run's guarded server)
  plain_det     plain_a with Inductor's deterministic mode, set in vLLM's inductor_compile_config (on torch 2.13
                TORCHINDUCTOR_DETERMINISTIC=1 alone is reset by the first compiled frame, pytorch#198563)
  guarded_det   guarded with the same

Each answers the same 64 prompts greedily (32 tokens, the top 5 log-probabilities of each token kept). Reported for
each pair below: how many completions differ, and for each difference, at its first differing token, whether each
side preferred its own token by at most 0.25 nat (vLLM's own near-tie rule in its tests); and from each server's
compiled code, its Triton kernels that compute an rsqrt (the norms), each by a hash of its source with the kernel's
name taken out, the set of configurations Inductor's autotuning chose (all kernels), and which configuration each
kernel got (by its source hash, from the .best_config file Inductor keys to that kernel's file), so a kernel tuned
differently on two servers is named. These locate where two servers differ; they point to a cause, they don't prove
one.

The run is complete only when every server started, answered all 64 prompts with a usable completion (its text, 32
token ids and 32 top-5 log-probability tables), and left compiled norm kernels to read; a server whose startup
would run past the run's time isn't started.

    python experiments/vllm_norm_diag.py --out runs/norm_diag
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_norm_check as vn  # noqa: E402
import vllm_serve_check as vs  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ROLES = ("plain_a", "plain_b", "plain_exact", "nonorms", "guarded", "plain_det", "guarded_det")
PAIRS = (("plain_a", "plain_b"), ("plain_a", "plain_exact"), ("plain_exact", "nonorms"), ("nonorms", "guarded"),
         ("plain_a", "guarded"), ("plain_det", "guarded_det"), ("plain_a", "plain_det"))
IDENTITY = 64
TOKENS = 32
NEAR_TIE = 0.25
RUN_S = 55 * 60                          # inside the Modal phase's 60 minutes (7 compiles from empty caches)
ANSWER_S = 4 * 60                        # 64 completions, reading the compiled code, stopping the server
MIN_START_S = 8 * 60                     # a fresh compile (2 to 4 minutes on an H100) and ANSWER_S


def server(role: str, out: str, port: int, metrics_port: int):
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in [k for k in env if k.startswith("SDCGUARD")]:
        del env[k]
    root = os.path.join(out, role)
    os.makedirs(root, exist_ok=True)
    env.update(VLLM_CACHE_ROOT=os.path.join(root, "vllm_cache"), TORCHINDUCTOR_CACHE_DIR=os.path.join(root, "inductor"),
               TRITON_CACHE_DIR=os.path.join(root, "triton"), VLLM_COMPILE_CACHE_SAVE_FORMAT="unpacked")
    if role in ("nonorms", "guarded", "guarded_det"):
        env["SDCGUARD_ATTN"] = "0"           # as in the norms run
        env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_LOG=os.path.join(out, f"{role}_alerts.jsonl"),
                   SDCGUARD_METRICS_PORT=str(metrics_port), SDCGUARD_NORMS="0" if role == "nonorms" else "1")
    if role == "plain_exact":
        env["SDCGUARD_TEST_EXACT"] = "1"
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--no-enable-prefix-caching", "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    if role.endswith("_det"):
        cmd.append("-cc.inductor_compile_config=" + json.dumps({"deterministic": True}))
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def complete(port: int, prompt: str) -> dict:
    """Greedy, TOKENS tokens, the top 5 log-probabilities of each: {"text", "tokens", "top"} or {"error"}."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": TOKENS, "temperature": 0.0, "ignore_eos": True,
            "logprobs": 5, "return_tokens_as_token_ids": True}     # (tokens by id: two with the same text don't merge)
    try:
        status, r = vn._post(port, "/v1/completions", body)
        if status != 200:
            return {"error": f"HTTP {status}: {str(r)[:200]}"}
        c = r["choices"][0]
        lp = c.get("logprobs") or {}
        out = {"text": c["text"], "tokens": lp.get("tokens") or [], "top": lp.get("top_logprobs") or []}
        why = unusable(out)
        return {"error": f"unusable completion: {why}", **out} if why else out
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def unusable(c: dict):
    """Why a completion can't be compared (None if it can): its text, TOKENS token ids, TOKENS top-5 tables."""
    if not isinstance(c.get("text"), str):
        return "no text"
    toks, top = c.get("tokens"), c.get("top")
    if not isinstance(toks, list) or len(toks) != TOKENS:
        return f"{len(toks) if isinstance(toks, list) else 'no'} tokens, not {TOKENS}"
    if not isinstance(top, list) or len(top) != TOKENS or not all(isinstance(t, dict) and t for t in top):
        return f"top log-probabilities for {len(top) if isinstance(top, list) else 'no'} tokens, not {TOKENS}"
    return None


def first_divergence(a: dict, b: dict) -> dict:
    """Where two completions first differ and whether it is a near-tie: at the first differing token, each side
    preferred its own token over the other's by at most NEAR_TIE nat (by each side's top-5 log-probabilities; a token
    missing from the other side's top 5 counts as a gap larger than NEAR_TIE)."""
    ta, tb = a.get("tokens") or [], b.get("tokens") or []
    i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), None)
    if i is None:
        return {"index": None, "near_tie": len(ta) == len(tb)}
    gaps = []
    for side, own, other in ((a, ta[i], tb[i]), (b, tb[i], ta[i])):
        top = (side.get("top") or [])[i] if i < len(side.get("top") or []) else {}
        top = top or {}
        if own in top and other in top:
            gaps.append(round(float(top[own]) - float(top[other]), 4))
        else:
            gaps.append(None)
    near = all(g is not None and 0 <= g <= NEAR_TIE for g in gaps)
    return {"index": i, "tokens": [ta[i], tb[i]], "gaps": gaps, "near_tie": near}


def compare(x: list, y: list) -> dict:
    """compared: both servers gave IDENTITY usable completions (else the count of those that failed, and no
    comparison: a failed request isn't a different answer)."""
    if len(x) != IDENTITY or len(y) != IDENTITY:
        return {"compared": False, "why": f"{len(x)} and {len(y)} completions, not {IDENTITY} each"}
    bad = [k for k, (a, b) in enumerate(zip(x, y)) if a.get("error") or b.get("error")]
    if bad:
        return {"compared": False, "failed": len(bad), "why": f"{len(bad)} prompts without a usable completion on one "
                f"side (first: prompt {bad[0]}: {x[bad[0]].get('error') or y[bad[0]].get('error')})"}
    diffs = [{"prompt": k, **first_divergence(a, b)} for k, (a, b) in enumerate(zip(x, y))
             if a.get("text") != b.get("text")]
    return {"compared": True, "differ": len(diffs), "all_near_ties": all(d.get("near_tie") for d in diffs),
            "diffs": diffs}


_NAME = re.compile(r"\btriton_[A-Za-z0-9_]+")


def norm_kernels(root: str, keep: Optional[str] = None) -> dict:
    """The Triton kernels of a server's compiled code that compute an rsqrt: {source hash with names taken out:
    {"names": [...]}}. Inductor's wrapper modules (which embed every kernel's source and call the graph's ops,
    sdcguard's included) are left out. With keep, each kernel's source is copied there as <hash>.py (the caches
    themselves are removed after, so the run's folder stays small)."""
    out: dict = {}
    for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        if keep and os.path.commonpath([os.path.abspath(path), os.path.abspath(keep)]) == os.path.abspath(keep):
            continue
        try:
            src = open(path, errors="replace").read()
        except OSError:
            continue
        if "@triton.jit" not in src or "rsqrt" not in src or "async_compile.triton(" in src or "def call(" in src:
            continue
        body = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
        names = sorted(set(_NAME.findall(body)))
        norm = _NAME.sub("KERNEL", body)
        norm = re.sub(r"'kernel_name': '[^']*'|'backend_hash': '[^']*'|'cache_key': '[^']*'", "", norm)
        h = hashlib.sha256(norm.encode()).hexdigest()[:16]
        if keep:
            os.makedirs(keep, exist_ok=True)
            with open(os.path.join(keep, f"{h}.py"), "w") as f:
                f.write(src)
        e = out.setdefault(h, {"names": []})
        e["names"] = sorted(set(e["names"]) | set(names))
    return out


def tuned_configs(root: str) -> list:
    """Every configuration Inductor's autotuning chose in a server's compiled code (its .best_config files, which are
    named by a cache key, not by kernel), without timings, as a sorted list of JSON strings."""
    out = set()
    for bc in glob.glob(os.path.join(root, "**", "*.best_config"), recursive=True):
        try:
            c = json.load(open(bc))
        except (OSError, ValueError):
            continue
        if isinstance(c, dict):
            c = {k: v for k, v in c.items() if k not in ("time_taken_ms", "triton_cache_hash", "found_by_coordesc")}
        out.add(json.dumps(c, sort_keys=True))
    return sorted(out)


def best_config_path(path: str) -> Optional[str]:
    """Where Inductor keeps the tuned configuration of the kernel in the file `path`, by PyTorch's own code
    (AutotuneCache: a hash of the file's name and torch's own key, in the kernel's folder; the function that names it
    changed between versions), or None where it can't be told."""
    try:
        from torch._inductor.runtime.autotune_cache import AutotuneCache as A
        key = A._prepare_key(path)
        if hasattr(A, "_make_local_cache_key"):
            return A._make_local_cache_key(os.path.dirname(path), key)
        c = A.__new__(A)
        c.local_cache = None
        c._setup_local_cache({"autotune_local_cache": True}, os.path.dirname(path), key)
        return c.local_cache[1] if c.local_cache else None
    except Exception:  # noqa: BLE001
        return None


def configs_by_kernel(root: str) -> dict:
    """{source hash with names taken out (as norm_kernels): the sorted tuned configurations of the kernels with that
    source} for every Triton kernel of a server's compiled code whose .best_config file is found (best_config_path):
    two kernels can differ by name only, each tuned on its own."""
    out = {}
    for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        try:
            src = open(path, errors="replace").read()
            if "@triton.jit" not in src or "async_compile.triton(" in src or "def call(" in src:
                continue
            bc = best_config_path(path)
            if not bc or not os.path.exists(bc):
                continue
            c = json.load(open(bc))
        except Exception:  # noqa: BLE001  (a file that can't be read or paired is left out)
            continue
        body = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
        norm = _NAME.sub("KERNEL", body)
        norm = re.sub(r"'kernel_name': '[^']*'|'backend_hash': '[^']*'|'cache_key': '[^']*'", "", norm)
        if isinstance(c, dict):
            c = {k: v for k, v in c.items() if k not in ("time_taken_ms", "triton_cache_hash", "found_by_coordesc")}
        h = hashlib.sha256(norm.encode()).hexdigest()[:16]
        out[h] = sorted(out.get(h, []) + [json.dumps(c, sort_keys=True)])
    return out


def tuned_by_kernel_diff(a: dict, b: dict) -> dict:
    """The kernels both servers compiled (same source) that autotuning gave different configurations (compared as
    the sorted list of configurations of the kernels with that source on each server)."""
    if not a or not b:
        return {"compared": False}
    both = sorted(set(a) & set(b))
    diff = [{"kernel": h, "first": a[h], "second": b[h]} for h in both if a[h] != b[h]]
    return {"compared": True, "kernels_in_both": len(both), "tuned_differently": diff, "same": not diff}


def kernel_diff(a: dict, b: dict) -> dict:
    if not a or not b:
        return {"compared": False}
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    return {"compared": True, "kernels": [len(a), len(b)], "only_first": only_a, "only_second": only_b,
            "same": not only_a and not only_b}


def config_diff(a, b) -> dict:
    if a is None or b is None:
        return {"compared": False}
    return {"compared": True, "configs": [len(a), len(b)], "only_first": sorted(set(a) - set(b)),
            "only_second": sorted(set(b) - set(a)), "same": set(a) == set(b)}


def run(out: str, port: int, metrics_port: int) -> dict:
    os.makedirs(out, exist_ok=True)
    deadline = time.monotonic() + RUN_S
    ps = vs.prompts(IDENTITY)
    res: dict = {"model": MODEL}
    for role in ROLES:
        # startup (a fresh compile) plus 64 completions must fit before the deadline: the wait for readiness ends
        # ANSWER_S before it, and a server isn't started with less than MIN_START_S left
        left = deadline - time.monotonic()
        if left < MIN_START_S:
            res[role] = {"skipped": f"the run's time was nearly up ({left / 60:.1f} minutes left)"}
            continue
        proc, log = server(role, out, port, metrics_port)
        r: dict = {}
        res[role] = r
        try:
            r["ready"] = vs.wait_ready(port, proc, timeout=max(60.0, min(1200.0, left - ANSWER_S)))
            if not r["ready"]:
                continue
            r["reductions"] = vn.rpc(port, "sdcguard_test_reductions")
            r["answers"] = [complete(port, p) for p in ps]
            if role in ("nonorms", "guarded", "guarded_det"):
                r["alerts"] = len(vn.alerts(os.path.join(out, f"{role}_alerts.jsonl")))
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            r["stop"] = vs.stop(proc)
            log.close()
            root = os.path.join(out, role)
            r["norm_kernels"] = norm_kernels(root, keep=os.path.join(root, "norm_kernels"))
            r["tuned_configs"] = tuned_configs(root)
            r["tuned_by_kernel"] = configs_by_kernel(root)
            for d in ("vllm_cache", "inductor", "triton"):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
            with open(os.path.join(out, "PARTIAL.json"), "w") as f:
                json.dump(res, f, indent=1, default=str)
    return res


def summarize(res: dict) -> dict:
    pairs = {}
    for a, b in PAIRS:
        ra, rb = res.get(a) or {}, res.get(b) or {}
        c = compare(ra.get("answers") or [], rb.get("answers") or [])
        pairs[f"{a}_vs_{b}"] = {"answers": {k: v for k, v in c.items() if k != "diffs"},
                                "first_divergences": c.get("diffs"),
                                "norm_kernels": kernel_diff(ra.get("norm_kernels") or {}, rb.get("norm_kernels") or {}),
                                "tuned_configs": config_diff(ra.get("tuned_configs"), rb.get("tuned_configs")),
                                "tuned_by_kernel": tuned_by_kernel_diff(ra.get("tuned_by_kernel") or {},
                                                                        rb.get("tuned_by_kernel") or {})}
    roles = {}
    for k in ROLES:
        r = res.get(k) or {}
        ans = r.get("answers") or []
        roles[k] = {"ready": r.get("ready"), "skipped": r.get("skipped"), "error": r.get("error"),
                    "reductions": r.get("reductions"), "alerts": r.get("alerts"),
                    "usable_answers": sum(1 for a in ans if not a.get("error")), "answers": len(ans),
                    "norm_kernels": len(r.get("norm_kernels") or {}),
                    "tuned_by_kernel": len(r.get("tuned_by_kernel") or {})}
    why = [f"{k}: " + ("didn't start" + (f" ({v['skipped']})" if v["skipped"] else "") if not v["ready"] else
                       f"stopped with an error ({v['error']})" if v["error"] else
                       f"{v['usable_answers']} of {IDENTITY} usable completions" if v["usable_answers"] != IDENTITY
                       else "no compiled norm kernel found")
           for k, v in roles.items()
           if not v["ready"] or v["error"] or v["usable_answers"] != IDENTITY or not v["norm_kernels"]]
    why += [f"{name}: not compared ({p['answers'].get('why')})" for name, p in pairs.items()
            if not p["answers"].get("compared")]
    return {"complete": not why, "incomplete_because": why, "roles": roles, "pairs": pairs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--metrics-port", type=int, default=9400)
    a = ap.parse_args(argv)
    res = run(a.out, a.port, a.metrics_port)
    s = summarize(res)
    res["summary"] = s
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(s, indent=1, default=str)[:20000])
    return 0 if s["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
