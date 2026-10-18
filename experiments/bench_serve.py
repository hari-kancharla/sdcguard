"""ServeGuard on a GPU: does it agree with the training check, what does it cost, does it stay quiet
on clean inference, and does it catch faults? Writes a JSON report (SERVE_OUT, default serve.json).

    python experiments/bench_serve.py          # NVIDIA GPU; about 20 minutes on an H100
    python experiments/bench_serve.py --smoke  # CPU, tiny model, a minute: tests this script

1. agreement   on clean and corrupted linear layers of many shapes, bf16 and fp16: the serving check
               (Triton kernel, and the PyTorch version) vs the training check (plain PyTorch)
2. cost        Llama-shaped models with random weights (nothing downloaded): 1.1B (TinyLlama's shape)
               and 7B (Llama-2-7B's shape). Two regimes: decode-like (64 sequences x 1 token) and
               prefill (4 x 1024 tokens). torch.compile with CUDA graphs ("reduce-overhead"; how vLLM
               and other engines serve) and, for the 1.1B model, eager. Timing: 3 rounds of 20 passes
               back to back, median. Configurations, each against a baseline in the same mode:
                 off             no sdcguard, PyTorch's default matmul reductions
                 exact only      no sdcguard, fp32 split-K reductions (what the guard switches on)
                 every layer     ServeGuard().attach(model)
                 1/8 of layers   ServeGuard().attach(model, share=0.125)
               Round 3 measured the PyTorch version on the same 1.1B model and regimes: +159% (decode)
               and +122% (prefill) under CUDA graphs.
3. quiet       alerts during all guarded passes, plus 200 eager passes of random shapes: must be 0;
               every checked layer must show a nonzero score (the checks ran)
4. catches     a flipped exponent bit in one weight (eager and compiled); software-injected compute
               faults in two MLPs (eager); a simulated bad GPU (faults in every matmul) with only 1/8
               of the layers checked
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sdcguard import FaultInjector  # noqa: E402
from sdcguard import _serve_fast  # noqa: E402
from sdcguard import checksum as cs  # noqa: E402
from sdcguard.inject import fault_model  # noqa: E402
from sdcguard.serve import LayerChecksums, ServeGuard, check_static_impl  # noqa: E402

OUT = os.environ.get("SERVE_OUT", "serve.json")
report: dict = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__}
SMOKE = False


def log(*a):
    print(*a, flush=True)


def agreement(dev):
    rows, worst_rel_clean, bad = [], 0.0, []
    shapes = [(64, 256, 512, False), (7, 300, 1000, True), (1, 64, 40, True), (512, 2048, 5632, False),
              (33, 5632, 2048, True), (2048, 2048, 32000, False), (3, 4096, 384, True)]
    if SMOKE:
        shapes = shapes[:3]
    for dt in (torch.bfloat16, torch.float16):
        for m, k, n, bias in shapes:
            for corrupt in (None, "zero", "nan", "big", "bitflip"):
                torch.manual_seed(m + k + n)
                lin = nn.Linear(k, n, bias=bias, device=dev, dtype=dt)
                x = torch.randn(m, k, device=dev, dtype=dt)
                y = lin(x).detach()
                if corrupt == "zero":
                    y[3 % m, 5:9] = 0
                elif corrupt == "nan":
                    y[1 % m, n - 1] = float("nan")
                elif corrupt == "big":
                    y[2 % m, n // 2] += 40 * y.float().abs().mean().to(dt)
                elif corrupt == "bitflip":
                    y.view(torch.int16)[m // 2, n // 3] ^= (1 << 13)
                g = ServeGuard()
                slot = g.add_layer(lin, "lin")
                g.check(slot, x, y)                                    # Triton kernel where it passed its probe
                z, flat, _ = g.read_slot(slot)
                c = g.layers[slot]["cs"]
                loc = divmod(int(flat) // 2, c.nb)
                st = torch.zeros(1, 2, dtype=torch.int64, device=dev)  # the PyTorch serving check
                check_static_impl(x, y, *LayerChecksums(lin.weight, lin.bias, 128).tensors(True), st, 0, 6.0,
                                  cs.ABS_FLOOR)
                tz, tflat, _ = _serve_fast.decode(*st[0].tolist())
                r = cs.check_matmul(x, lin.weight.t(), y, lin.bias)   # the training check
                ref_z, ref_loc = r.max_z, (r.row, r.col_block)

                def close(a, b):
                    if math.isinf(a) or math.isinf(b):
                        return math.isinf(a) and math.isinf(b)
                    return abs(a - b) / max(1.0, abs(b)) < (2e-2 if corrupt is None else 1e-3)
                if corrupt is None and math.isfinite(z) and math.isfinite(ref_z):
                    worst_rel_clean = max(worst_rel_clean, abs(z - ref_z) / max(1.0, abs(ref_z)))
                same_verdict = (z > 6) == (ref_z > 6) == (tz > 6)
                loc_ok = corrupt is None or loc == ref_loc == divmod(int(tflat) // 2, c.nb)
                row = {"dtype": str(dt)[6:], "m": m, "k": k, "n": n, "bias": bias, "corrupt": corrupt,
                       "serve_z": z, "serve_torch_z": tz, "train_z": ref_z, "serve_loc": loc, "train_loc": ref_loc,
                       "used_triton": g.uses_fast(slot),
                       "agree": bool(close(z, ref_z) and close(tz, ref_z) and same_verdict and loc_ok)}
                rows.append(row)
                if not row["agree"]:
                    bad.append(row)
    used = sum(r["used_triton"] for r in rows)
    log(f"agreement: {len(rows) - len(bad)}/{len(rows)} cases agree (serving Triton, serving PyTorch, training); "
        f"Triton kernel used in {used}/{len(rows)}; worst relative difference of clean scores {worst_rel_clean:.2e}")
    for b in bad:
        log("  DISAGREE", b)
    report["agreement"] = {"cases": len(rows), "agree": len(rows) - len(bad), "disagree": bad,
                           "triton_used": used, "worst_rel_clean": worst_rel_clean,
                           "first_use_comparison": {str(k): v for k, v in _serve_fast.STATUS.items()}}


def llama(dev, size: str):
    from transformers import LlamaConfig, LlamaForCausalLM
    if SMOKE:
        cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=344, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    elif size == "1.1B":
        cfg = LlamaConfig(vocab_size=32000, hidden_size=2048, intermediate_size=5632, num_hidden_layers=22,
                          num_attention_heads=32, num_key_value_heads=4, max_position_embeddings=2048)
    else:
        cfg = LlamaConfig(vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
                          num_attention_heads=32, num_key_value_heads=32, max_position_embeddings=2048)
    torch.manual_seed(0)
    with torch.device(dev):
        m = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    return m, cfg.vocab_size


def timed(fn, n=20, warm=5, rounds=3):
    """ms per call: `rounds` rounds of n calls back to back between one pair of CUDA events (a serving
    engine doesn't wait for the GPU after every pass); returns (median round, [all rounds])."""
    if SMOKE:
        n, warm, rounds = 2, 1, 2
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    per = []
    for _ in range(rounds):
        if torch.cuda.is_available():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(n):
                fn()
            b.record()
            torch.cuda.synchronize()
            per.append(a.elapsed_time(b) / n)
        else:
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            per.append(1000 * (time.perf_counter() - t0) / n)
    return statistics.median(per), [round(x, 4) for x in per]


def regimes():
    if SMOKE:
        return {"decode-like (4 x 1 token)": (4, 1), "prefill (2 x 32 tokens)": (2, 32)}
    return {"decode-like (64 x 1 token)": (64, 1), "prefill (4 x 1024 tokens)": (4, 1024)}


def _reductions(exact: bool) -> None:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not exact
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = not exact


COUNT = {"passes": 0, "checks": 0, "alerts": []}


def _skips() -> int:
    try:
        from torch._dynamo.utils import counters
        return int(sum(counters["inductor"].get(k, 0) for k in counters["inductor"] if "cudagraph_skip" in k))
    except Exception:  # noqa: BLE001
        return -1


def measure(model, vocab, dev, mode: str, config: str):
    """Median ms per forward pass for each regime, with sdcguard set up as `config`."""
    guard = None
    torch._dynamo.reset()
    skips0 = _skips()
    if config in ("every layer", "1/8 of layers"):
        guard = ServeGuard().attach(model, share=1.0 if config == "every layer" else 0.125)
    else:
        _reductions(config == "exact only")
    fwd = torch.compile(model, mode="reduce-overhead", dynamic=False) if mode == "compiled + CUDA graphs" else model
    out = {}
    for name, (b, t) in regimes().items():
        ids = torch.randint(0, vocab, (b, t), device=dev)

        def go():
            with torch.no_grad():
                fwd(input_ids=ids, use_cache=False)
        t0 = time.time()
        try:
            ms, rounds = timed(go)
            out[name] = {"ms": ms, "rounds_ms": rounds, "seconds_incl_compile": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            log(f"  {mode} {config} {name}: FAILED {out[name]['error']}")
        if guard is not None and "ms" in out[name]:
            n = 5 if SMOKE else 65
            COUNT["passes"] += n
            COUNT["checks"] += n * guard.stats["layers"]
    if guard is not None:
        ws = guard.worst_z()
        COUNT["alerts"] += [e.detail for e in guard.poll(wait=True)]
        out["layers_checked"] = guard.stats["layers"]
        out["layers_with_nonzero_scores"] = sum(1 for z in ws.values() if z > 0)
        out["triton_layers"] = sum(guard.uses_fast(s) for s in range(len(guard.layers)))
        out["worst_clean_z"] = max([z for z in ws.values() if math.isfinite(z)] or [0.0])
        guard.detach()
    # CUDA graphs that torch.compile had to skip (0 = every compiled pass ran as a CUDA graph)
    out["cudagraph_skips"] = _skips() - skips0 if mode == "compiled + CUDA graphs" else None
    _reductions(False)
    torch._dynamo.reset()
    return out


def cost(dev):
    res = {}
    for size in ("1.1B", "7B"):
        model, vocab = llama(dev, size)
        info = {"params": sum(p.numel() for p in model.parameters())}
        modes = ["compiled + CUDA graphs"] + (["eager"] if size == "1.1B" or SMOKE else [])
        for mode in modes:
            # "exact only" separates the cost of fp32 split-K reductions (switched on with the guard) from
            # the cost of the checks themselves
            configs = ["off", "exact only", "every layer", "1/8 of layers"]
            rows = {c: measure(model, vocab, dev, mode, c) for c in configs}
            for name in regimes():
                base = rows["off"].get(name, {}).get("ms")
                for c in configs[1:]:
                    on = rows[c].get(name, {}).get("ms")
                    if base and on:
                        rows[c][name]["overhead_pct"] = round(100 * (on - base) / base, 2)
                        log(f"cost: {size}, {mode}, {name}, {c}: {base:.3f} ms -> {on:.3f} ms "
                            f"({rows[c][name]['overhead_pct']:+.2f}%)")
            info[mode] = rows
            res[size] = info
            report["cost"] = res
            _save()
        res[size] = info
        report["cost"] = res
        if size == "1.1B":
            quiet_and_catches(model, vocab, dev)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def quiet_and_catches(model, vocab, dev):
    guard = ServeGuard().attach(model)
    g2 = torch.Generator().manual_seed(1)
    for _ in range(10 if SMOKE else 200):                        # eager passes of random shapes
        b = int(torch.randint(1, 33, (1,), generator=g2))
        t = int(torch.randint(1, 257, (1,), generator=g2))
        with torch.no_grad():
            model(input_ids=torch.randint(0, vocab, (b, t), device=dev), use_cache=False)
        COUNT["passes"] += 1
        COUNT["checks"] += guard.stats["layers"]
    ws = guard.worst_z()
    COUNT["alerts"] += [e.detail for e in guard.poll(wait=True)]
    finite = [z for z in ws.values() if math.isfinite(z)]
    report["quiet"] = {"alerts": COUNT["alerts"], "passes": COUNT["passes"], "layer_checks": COUNT["checks"],
                       "worst_clean_z_random_shapes": max(finite) if finite else None,
                       "layers_with_nonzero_scores": sum(1 for z in ws.values() if z > 0), "layers": len(ws)}
    log(f"quiet: {len(COUNT['alerts'])} alerts in {COUNT['passes']} guarded passes ({COUNT['checks']:,} layer "
        f"checks); random-shape passes: {report['quiet']['layers_with_nonzero_scores']}/{len(ws)} layers showed "
        f"nonzero scores, worst clean z {report['quiet']['worst_clean_z_random_shapes']:.2f}")
    out = {}
    b, t = list(regimes().values())[0]
    ids = torch.randint(0, vocab, (b, t), device=dev)
    layers = model.model.layers
    target = layers[len(layers) // 2].mlp.down_proj
    r, c = target.weight.shape[0] // 3, target.weight.shape[1] // 5
    with torch.no_grad():
        target.weight.view(torch.int16)[r, c] ^= (1 << 14)
        model(input_ids=ids, use_cache=False)
    ev = guard.poll(wait=True)
    out["weight_flip_eager"] = {"flagged": sorted({e.detail["where"] for e in ev}),
                                "first": ev[0].detail if ev else None, "verify_weights": guard.verify_weights()}
    try:
        torch._dynamo.reset()
        cm = torch.compile(model, mode="reduce-overhead", dynamic=False)
        with torch.no_grad():
            for _ in range(4):
                cm(input_ids=ids, use_cache=False)
        ev = guard.poll(wait=True)
        out["weight_flip_compiled"] = {"flagged": sorted({e.detail["where"] for e in ev})}
    except Exception as e:  # noqa: BLE001
        out["weight_flip_compiled"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    torch._dynamo.reset()
    with torch.no_grad():
        target.weight.view(torch.int16)[r, c] ^= (1 << 14)      # undo
    guard.poll(wait=True)
    picks = sorted({len(layers) // 7, (2 * len(layers)) // 3})
    mlps = [layers[i].mlp for i in picks]
    inj = FaultInjector(rate=0.05, fault_model=fault_model("nvidia", "warp"), seed=5).arm_on(mlps)
    with torch.no_grad(), inj:
        for _ in range(10):
            model(input_ids=torch.randint(0, vocab, (4, 64), device=dev), use_cache=False)
    inj.remove_hooks()
    ev = guard.poll(wait=True)
    flagged = sorted({e.detail["where"] for e in ev})
    out["injected_compute_faults"] = {"injected": len(inj.events), "faulty_mlps": picks, "layers_flagged": flagged,
                                      "only_faulty_layers": all(any(f".layers.{i}.mlp" in f for i in picks)
                                                                for f in flagged)}
    guard.detach()
    # a bad GPU corrupts every matmul it runs; with 1/8 of the layers checked, is it flagged in the first pass?
    eighth = ServeGuard().attach(model, share=0.125)
    inj = FaultInjector(rate=1.0, fault_model=fault_model("nvidia", "warp"), seed=7).arm_on(list(layers))
    with torch.no_grad(), inj:
        model(input_ids=torch.randint(0, vocab, (4, 64), device=dev), use_cache=False)
    inj.remove_hooks()
    ev = eighth.poll(wait=True)
    out["bad_gpu_one_eighth_of_layers"] = {"layers_checked": eighth.stats["layers"],
                                           "faults_injected_in_the_pass": len(inj.events),
                                           "layers_flagged_in_first_pass": sorted({e.detail["where"] for e in ev})}
    eighth.detach()
    for k, v in out.items():
        log(f"catches: {k}: {v}")
    report["catches"] = out
    _save()


def _save():
    """Write the report so far (a hard time limit then loses only the section that was running)."""
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1, default=str)


def main():
    global SMOKE
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="CPU, tiny model: a quick test of this script")
    a = p.parse_args()
    SMOKE = a.smoke
    if not torch.cuda.is_available() and not SMOKE:
        print("needs an NVIDIA GPU (or --smoke)")
        return 2
    dev = torch.device("cuda" if torch.cuda.is_available() and not SMOKE else "cpu")
    t0 = time.time()
    try:
        agreement(dev)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        report["agreement"] = {"error": f"{type(e).__name__}: {e}"}
    _save()
    try:
        cost(dev)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        report["error"] = f"{type(e).__name__}: {e}"
    report["seconds"] = round(time.time() - t0, 1)
    report["serving_triton_fallbacks"] = {"count": _serve_fast.FALLBACKS[0], "why": _serve_fast.WHY}
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1, default=str)
    log(f"wrote {OUT} ({report['seconds']} s)")
    return 0 if "error" not in report and "error" not in report.get("agreement", {}) else 1


if __name__ == "__main__":
    sys.exit(main())
