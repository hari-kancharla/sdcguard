"""Which part of sdcguard's Machete patch changes a model's completions in vLLM (a diagnostic).

On October 22, 2026, on one H100 with vLLM 0.31, Qwen2.5-1.5B-Instruct-GPTQ-Int4, -AWQ and Qwen2.5-7B-Instruct-AWQ
on Machete gave 1 to 4 of 64 greedy completions different with sdcguard than without (the same ones in two runs a
day apart), while Marlin and the bf16 models gave every completion the same. This runs the same model in fresh
processes: once without sdcguard, then with sdcguard's plugin and its Machete apply_weights replaced by each variant
below, each of which adds one piece of what sdcguard does, and compares every completion with the run without it.

  wrap_only        sdcguard loads (reads each layer's weights before Machete repacks them, probes the kernel,
                   registers the layer); apply_weights is vLLM's own, unchanged: no check
  split_inplace    Machete without the bias, then the bias added in place (vLLM's own code, split in two): no check
  split_oop        Machete without the bias, then y + bias: no check
  split_rounded    Machete without the bias, then the bias added by one opaque call (sdcguard's add_bias, rounded
                   to bf16 once, as Marlin's patch adds it): no check
  check_no_bias    sdcguard's patch, but the check only on layers without a bias
  check_bias       sdcguard's patch, but the check only on layers with a bias
  full             sdcguard's patch as it ships
  full_rounded     sdcguard's patch, with the bias added by the opaque call instead of y + bias (the candidate fix
                   if vLLM's own graph rounds the bias sum before what follows)

    python experiments/machete_diag.py --out DIR [--model M]          # every variant, then the comparison
    python experiments/machete_diag.py --variant V --out DIR           # one run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
VARIANTS = ("wrap_only", "split_inplace", "split_oop", "split_rounded", "check_no_bias", "check_bias", "full",
            "full_rounded")
PROMPTS = 64
TOKENS = 128


def _install(variant: str) -> None:
    """Replace sdcguard's patched MacheteLinearKernel.apply_weights with the variant (after the plugin registered)."""
    from vllm.model_executor.kernels.linear.mixed_precision import machete
    from sdcguard import _quant, vllm_plugin
    vllm_plugin.register()
    add_bias = _quant.add_bias_op()
    K = machete.MacheteLinearKernel
    shipped = K.apply_weights
    names = shipped.__code__.co_varnames[:shipped.__code__.co_argcount]
    defaults = dict(zip(names[-len(shipped.__defaults__):], shipped.__defaults__))
    orig = defaults["_orig"]                                  # vLLM's own apply_weights
    if variant == "full":
        return

    def apply_weights(self, layer, x, bias=None):
        a = getattr(layer, "_sdcguard_args", None)
        if variant == "wrap_only" or a is None:
            return orig(self, layer, x, bias)
        if variant in ("split_inplace", "split_oop", "split_rounded"):
            y = orig(self, layer, x, None)
            if bias is not None:
                if variant == "split_inplace":
                    y.add_(bias)
                elif variant == "split_oop":
                    y = y + bias
                else:
                    y = add_bias(y, bias)
            return y
        if variant == "full_rounded":
            y = shipped(self, layer, x, None)              # the check, on the output without the bias
            return add_bias(y, bias) if bias is not None else y
        # check_no_bias / check_bias: the shipped patch, with the check skipped on the other kind of layer
        want = (bias is None) if variant == "check_no_bias" else (bias is not None)
        if want:
            return shipped(self, layer, x, bias)
        y = orig(self, layer, x, None)
        return y + bias if bias is not None else y

    K.apply_weights = apply_weights


def run(variant: str, out: str, model: str) -> int:
    sys.path.insert(0, HERE)
    import vllm_check
    os.environ["SDCGUARD"] = "0" if variant == "baseline" else "1"
    os.environ["SDCGUARD_POLL_EVERY"] = "1000000000"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False      # (as vllm_check's baseline)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if variant != "baseline":
        _install(variant)
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=TOKENS, ignore_eos=True)
    prompts = vllm_check._prompts(PROMPTS)
    llm.generate(prompts[:16], sp, use_tqdm=False)                              # warm-up, as vllm_check
    outs = llm.generate(prompts, sp, use_tqdm=False)
    light = llm.generate(prompts[:16], sp, use_tqdm=False)
    res = {"variant": variant, "model": model, "seconds": round(time.time() - t0, 1),
           "outputs": [o.outputs[0].text for o in outs], "light_outputs": [o.outputs[0].text for o in light]}
    if variant != "baseline":
        from sdcguard import vllm_plugin
        res["checked_layers"] = llm.apply_model(lambda _m: len(vllm_plugin._STATE["guard"].layers)
                                                if vllm_plugin._STATE.get("guard") else 0)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{variant}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if "outputs" not in k}), flush=True)
    return 0


def compare(out: str) -> dict:
    """Every variant's completions against the run without sdcguard: how many of each set are the same."""
    base = json.load(open(os.path.join(out, "baseline.json")))
    summary = {"model": base["model"], "variants": {}}
    for v in VARIANTS:
        p = os.path.join(out, f"{v}.json")
        if not os.path.exists(p):
            summary["variants"][v] = {"error": "no result"}
            continue
        r = json.load(open(p))
        same = sum(a == b for a, b in zip(base["outputs"], r["outputs"]))
        lsame = sum(a == b for a, b in zip(base["light_outputs"], r["light_outputs"]))
        summary["variants"][v] = {"same": f"{same}/{len(base['outputs'])}",
                                  "light_same": f"{lsame}/{len(base['light_outputs'])}",
                                  "all_same": same == len(base["outputs"]) and lsame == len(base["light_outputs"]),
                                  "checked_layers": r.get("checked_layers")}
    with open(os.path.join(out, "DIAG.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--variant", choices=("baseline",) + VARIANTS)
    a = p.parse_args()
    if a.variant:
        return run(a.variant, a.out, a.model)
    codes = {}
    for v in ("baseline",) + VARIANTS:
        cmd = [sys.executable, os.path.abspath(__file__), "--variant", v, "--out", a.out, "--model", a.model]
        print("$", " ".join(cmd), flush=True)
        codes[v] = subprocess.run(cmd, timeout=15 * 60).returncode
        print(f"{v}: exit code {codes[v]}", flush=True)
    s = compare(a.out)
    s["exit_codes"] = codes
    with open(os.path.join(a.out, "DIAG.json"), "w") as f:
        json.dump(s, f, indent=1)
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
