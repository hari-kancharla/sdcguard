"""sdcguard's vLLM plugin on a real GPU.

Runs vLLM twice in separate processes (the plugin patches vLLM when vLLM loads it): once without
sdcguard (baseline) and once with SDCGUARD=1. Same prompts, greedy decoding. Reports:
  - same outputs?  the first 64 completions must be identical with and without sdcguard
  - cost           output tokens per second, both runs (default settings: torch.compile + CUDA graphs)
  - quiet          alerts during the guarded run's clean traffic (must be 0)
  - catches        then one high exponent bit is flipped in one weight of one layer (what a memory
                   error would do); after a few more prompts the alert must name that layer, and
                   verify_weights() must find the changed weights

    python experiments/vllm_check.py --out runs/vllm            # both runs, then the comparison
    python experiments/vllm_check.py --role guarded --out ...   # one run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"     # bf16, public, has biases on q/k/v (tests the bias path)
FLIP_LAYER = "model.layers.10.mlp.down_proj"


def _guarded_failures(result: dict) -> list[str]:
    """Validate observed behavior independently of whether the serving process exited normally."""
    errors = []
    clean, after, flipped = (result.get(k) for k in ("clean", "after_flip", "flipped"))
    if not all(isinstance(v, list) and v for v in (clean, after, flipped)):
        return ["missing clean, injected-fault or worker reports"]
    if len(clean) != len(after) or len(clean) != len(flipped):
        return ["worker report counts differ"]
    for i, (c, a, target) in enumerate(zip(clean, after, flipped)):
        if not isinstance(c, dict) or not isinstance(a, dict):
            errors.append(f"worker {i}: malformed reports")
            continue
        layers = c.get("layers", 0)
        if not isinstance(layers, int) or layers <= 0 or c.get("layers_checked_since_last_read") != layers:
            errors.append(f"worker {i}: not every registered layer demonstrated a check")
        if c.get("alerts") != [] or c.get("changed_weights") != []:
            errors.append(f"worker {i}: clean serving raised alerts or changed weights")
        if not isinstance(target, str) or not target:
            errors.append(f"worker {i}: no weight was corrupted")
            continue
        locations = {e.get("where") for e in a.get("alerts", []) if isinstance(e, dict)}
        if target not in locations or target not in a.get("changed_weights", []):
            errors.append(f"worker {i}: the corrupted layer was not identified by both checks")
    return errors


def _comparison_failures(baseline: dict, guarded: dict) -> list[str]:
    errors = _guarded_failures(guarded)
    before, after = baseline.get("first_outputs"), guarded.get("first_outputs")
    if not isinstance(before, list) or not before or before != after:
        errors.append("compared completions are missing or differ")
    for field in ("model", "vllm", "enforce_eager"):
        if field not in baseline or baseline.get(field) != guarded.get(field):
            errors.append(f"baseline and guarded {field} do not match")
    for name, result in (("baseline", baseline), ("guarded", guarded)):
        value = result.get("output_tokens_per_s")
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            errors.append(f"{name} throughput is missing or invalid")
    return errors


def _prompts(n: int):
    words = ("the quick brown fox jumps over a lazy dog while engineers check every matrix multiply "
             "on thousands of graphics processors to make sure the numbers are right").split()
    out = []
    for i in range(n):
        k = 20 + (i * 37) % 400
        body = " ".join(words[(i + j) % len(words)] for j in range(k))
        out.append(f"Summarize this in one sentence: {body}")
    return out


def _flip(model):
    """Flip one high exponent bit of one weight in a checked layer: FLIP_LAYER, or (when only a share of
    the layers is checked and it isn't one of them) the middle checked down_proj."""
    import torch
    from sdcguard import vllm_plugin
    g = vllm_plugin.guard()
    names = [lay["name"] for lay in g.layers] if g is not None else []
    target = FLIP_LAYER
    if names and not any(n.endswith(FLIP_LAYER) for n in names):
        downs = [n for n in names if n.endswith("down_proj")] or names
        target = downs[len(downs) // 2]
    for name, mod in model.named_modules():
        if name.endswith(target):
            w = mod.weight
            with torch.no_grad():
                w.data.view(torch.int16)[123, 456] ^= (1 << 14)      # high exponent bit of one weight
            torch.cuda.synchronize()
            return name
    return None


def _guard_report(_model):
    from sdcguard import vllm_plugin
    g = vllm_plugin.guard()
    if g is None:
        return {"layers": 0}
    vllm_plugin.poll(wait=True)
    worst = dict(g.worst_seen)     # positive control: worst score each layer ever showed (0 = never checked)
    finite = [z for z in worst.values() if z == z and z != float("inf")]
    return {"layers": g.stats["layers"], "layers_checked_since_last_read": sum(1 for z in worst.values() if z > 0),
            "triton_layers": sum(g.uses_fast(s) for s in range(len(g.layers))),
            "worst_clean_z": max(finite) if finite else None,
            "alerts": [e.detail | {"gpu": e.identity.get("gpu_uuid"), "gpu_name": e.identity.get("gpu_name")}
                       for e in g.log.events],
            "changed_weights": g.verify_weights(), "names_sample": [lay["name"] for lay in g.layers[:6]]}


def run_role(role: str, out: str, n_prompts: int, max_tokens: int, model: str, eager: bool = False,
             tag: str | None = None):
    os.environ["SDCGUARD"] = "1" if role == "guarded" else "0"
    os.environ["SDCGUARD_POLL_EVERY"] = "1000000000"      # this script reads the results itself
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"           # engine in this process: apply_model reaches it
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    # vLLM caches compiled graphs on disk; a graph compiled in the baseline run has no checks in it, and
    # the guarded run must not reuse it. Both runs compile from scratch.
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    # vLLM 0.31's warm-up runs FlashInfer's top-k/top-p sampler, which JIT-compiles with nvcc; a pip install
    # without the CUDA toolkit or flashinfer-cubin crashes there (round 3: both runs died at start-up).
    # 0 selects vLLM's own PyTorch/Triton sampler (vllm/envs.py). Decoding here is greedy either way.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # The plugin keeps bf16 matmul partial sums in fp32 (as the training guard does). Do the same in the
    # baseline, so "same outputs" and its speed compare the checks alone. "vanilla" keeps PyTorch's default
    # (what a deployment without sdcguard runs): speed only, so the total cost of turning sdcguard on is known.
    import torch
    if role != "vanilla":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if role == "guarded":
        from sdcguard import vllm_plugin
        vllm_plugin.register()        # also loaded by vLLM through the entry point; calling it twice is fine
    import vllm
    from vllm import LLM, SamplingParams
    res = {"role": role, "model": model, "vllm": vllm.__version__, "enforce_eager": eager,
           "opaque_op": os.environ.get("SDCGUARD_OPAQUE") == "1", "share": os.environ.get("SDCGUARD_SHARE", "1")}
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048, seed=0,
              enforce_eager=eager)
    res["load_seconds"] = round(time.time() - t0, 1)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    prompts = _prompts(n_prompts)
    llm.generate(prompts[:16], sp, use_tqdm=False)                 # warm-up
    times = []
    for _ in range(3):                                             # three timed runs, median
        t1 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        times.append(time.time() - t1)
    dt = sorted(times)[1]
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    n_in = sum(len(o.prompt_token_ids) for o in outs)
    res.update(seconds=round(dt, 3), seconds_each_run=[round(t, 3) for t in times], output_tokens=n_out,
               prompt_tokens=n_in, output_tokens_per_s=round(n_out / dt, 1),
               total_tokens_per_s=round((n_in + n_out) / dt, 1), first_outputs=[o.outputs[0].text for o in outs[:64]])
    if role == "guarded":
        res["clean"] = llm.apply_model(_guard_report)
        res["flipped"] = llm.apply_model(_flip)
        llm.generate(prompts[:8], SamplingParams(temperature=0.0, max_tokens=16), use_tqdm=False)
        res["after_flip"] = llm.apply_model(_guard_report)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{tag or role}.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "first_outputs"}, indent=1, default=str), flush=True)
    failures = _guarded_failures(res) if role == "guarded" else []
    if failures:
        print("VALIDATION FAILED: " + "; ".join(failures), flush=True)
    return 1 if failures else 0


def compare(out: str):
    b = json.load(open(os.path.join(out, "baseline.json")))
    g = json.load(open(os.path.join(out, "guarded.json")))
    same = sum(x == y for x, y in zip(b["first_outputs"], g["first_outputs"]))
    clean = g.get("clean", [{}])[0]
    after = g.get("after_flip", [{}])[0]
    flagged = sorted({a["where"] for a in after.get("alerts", [])})
    summary = {
        "model": g["model"], "vllm": g["vllm"],
        "guarded_mode": "enforce_eager" if g.get("enforce_eager") else "torch.compile + CUDA graphs",
        "triton_layers": clean.get("triton_layers"),
        "same_outputs": f"{same}/{len(b['first_outputs'])}",
        "output_tokens_per_s": {"without": b["output_tokens_per_s"], "with": g["output_tokens_per_s"]},
        "slowdown_pct": round(100 * (b["output_tokens_per_s"] / g["output_tokens_per_s"] - 1), 2),
        "layers_checked": clean.get("layers"),
        "layers_that_actually_ran_checks": clean.get("layers_checked_since_last_read"),
        "worst_clean_z": clean.get("worst_clean_z"),
        "clean_alerts": len(clean.get("alerts", [])),
        "flipped_layer": g.get("flipped"), "flagged_after_flip": flagged,
        "verify_weights_after_flip": after.get("changed_weights"),
        "alert_example": (after.get("alerts") or [None])[0],
    }
    old = {}
    path = os.path.join(out, "SUMMARY.json")
    if os.path.exists(path):
        old = json.load(open(path))
    summary = {**old, **summary}
    failures = _comparison_failures(b, g)
    summary["validation"] = {"passed": not failures, "errors": failures}
    with open(os.path.join(out, "SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str), flush=True)
    return failures


def add_vanilla(out: str):
    """Slowdowns against a vLLM without sdcguard and with PyTorch's default reductions: the full cost of
    turning sdcguard on (the main comparison isolates the checks: both runs use fp32 reductions)."""
    v = json.load(open(os.path.join(out, "vanilla.json")))
    b = json.load(open(os.path.join(out, "baseline.json")))
    path = os.path.join(out, "SUMMARY.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}
    summary["vanilla_output_tokens_per_s"] = v["output_tokens_per_s"]
    summary["fp32_reductions_alone_slowdown_pct"] = round(100 * (v["output_tokens_per_s"] / b["output_tokens_per_s"] - 1), 2)
    g = os.path.join(out, "guarded.json")
    if os.path.exists(g):
        gt = json.load(open(g))["output_tokens_per_s"]
        summary["slowdown_vs_vanilla_pct"] = round(100 * (v["output_tokens_per_s"] / gt - 1), 2)
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps({k: summary[k] for k in summary if "vanilla" in k or "fp32" in k}, indent=1), flush=True)


def add_share(out: str, tag: str):
    b = json.load(open(os.path.join(out, "baseline.json")))
    g = json.load(open(os.path.join(out, f"{tag}.json")))
    path = os.path.join(out, "SUMMARY.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}
    clean = g.get("clean", [{}])[0]
    after = g.get("after_flip", [{}])[0]
    same = sum(x == y for x, y in zip(b["first_outputs"], g["first_outputs"]))
    summary.setdefault("shares", {})[g["share"]] = {
        "same_outputs": f"{same}/{len(b['first_outputs'])}",
        "output_tokens_per_s": g["output_tokens_per_s"],
        "slowdown_pct": round(100 * (b["output_tokens_per_s"] / g["output_tokens_per_s"] - 1), 2),
        "layers_checked": clean.get("layers"), "clean_alerts": len(clean.get("alerts", [])),
        "worst_clean_z": clean.get("worst_clean_z"),
        "flagged_after_flip": sorted({x["where"] for x in after.get("alerts", [])}),
        "verify_weights_after_flip": after.get("changed_weights")}
    failures = _comparison_failures(b, g)
    summary["shares"][g["share"]]["validation"] = {"passed": not failures, "errors": failures}
    with open(path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary["shares"][g["share"]], indent=1, default=str), flush=True)
    return failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["baseline", "guarded", "vanilla"])
    p.add_argument("--out", default="runs/vllm")
    p.add_argument("--prompts", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--tag", default=None, help="file name for this run's results (default: the role)")
    p.add_argument("--shares", default="", help="after the main pair, extra guarded runs checking only this "
                   "share of layers, e.g. 0.125,0.25")
    a = p.parse_args()
    if a.role:
        return run_role(a.role, a.out, a.prompts, a.max_tokens, a.model, a.enforce_eager, a.tag)
    shares = [float(x) for x in a.shares.split(",") if x.strip()]
    def go(role, env=None, extra=()):
        cmd = [sys.executable, __file__, "--role", role, "--out", a.out, "--prompts", str(a.prompts),
               "--max-tokens", str(a.max_tokens), "--model", a.model, *extra]
        return subprocess.run(cmd, env={**os.environ, **(env or {})}).returncode

    # Default: vLLM's torch.compile + CUDA graphs, the check as one custom op per layer (the Triton kernel).
    # If a run fails, fall back so one paid run still gives a matched pair: the PyTorch check traced into the
    # compiled graph, then both runs without torch.compile / CUDA graphs. The baseline always runs in the
    # same mode as the guarded run.
    attempts = [({}, ()), ({"SDCGUARD_SERVE_FAST": "0"}, ()), ({}, ("--enforce-eager",))]
    baseline_ok: dict = {}
    mode = attempts[-1]
    matched_pair = False
    for env, extra in attempts:
        if extra not in baseline_ok:
            baseline_ok[extra] = go("baseline", extra=extra) == 0
            print(f"baseline run {' '.join(extra)}: {'ok' if baseline_ok[extra] else 'FAILED'}", flush=True)
        if not baseline_ok[extra]:
            continue                       # vLLM itself doesn't start in this mode: try the next one
        code = go("guarded", env, extra)
        print(f"guarded run {env or ''} {' '.join(extra)}: exit code {code}", flush=True)
        if code == 0 and baseline_ok[extra]:
            mode = (env, extra)
            matched_pair = True
            break
    if not matched_pair:
        print("VALIDATION FAILED: no successful baseline/guarded pair; stale files are not accepted", flush=True)
        return 1
    failures = []
    try:
        failures.extend(compare(a.out))
    except Exception as e:  # noqa: BLE001
        print(f"comparison failed: {e}", flush=True)
        failures.append(f"comparison failed: {e}")
    code = go("vanilla", extra=mode[1])          # same mode, PyTorch's default reductions, no sdcguard
    print(f"vanilla run {' '.join(mode[1])}: exit code {code}", flush=True)
    try:
        if code:
            raise RuntimeError(f"vanilla process exited {code}")
        add_vanilla(a.out)
    except Exception as e:  # noqa: BLE001
        print(f"vanilla comparison failed: {e}", flush=True)
        failures.append(f"vanilla comparison failed: {e}")
    for share in shares:            # same mode as the main pair, fewer layers checked
        tag = f"guarded_share_{share}"
        code = go("guarded", {**mode[0], "SDCGUARD_SHARE": str(share)}, (*mode[1], "--tag", tag))
        print(f"guarded run, share {share} of layers: exit code {code}", flush=True)
        try:
            if code:
                raise RuntimeError(f"share {share} process exited {code}")
            failures.extend(add_share(a.out, tag))
        except Exception as e:  # noqa: BLE001
            print(f"share comparison failed: {e}", flush=True)
            failures.append(f"share comparison failed: {e}")
    if failures:
        print("VALIDATION FAILED: " + "; ".join(failures), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
