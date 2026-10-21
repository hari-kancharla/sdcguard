"""The recompute check (src/sdcguard/recompute.py) on real checkpointed models: does it stay silent on clean
steps, catch a transient fault and name the block, leave gradients unchanged, and what does it cost? Criteria are
in docs/METHOD.md ("The recompute check on GPUs"), written before the first GPU run.

Models: Hugging Face GPT-2 and Llama configurations with random weights (no downloads), gradient checkpointing
turned on before the check is installed (the order that used to leave it blind), dropout on with torch's default
preserved RNG; on a GPU in bf16 autocast.

1. clean: every checkpointed block of every step compared, no mismatch (calibrate()).
2. faults: a value change in one call of a module inside one block (forward run or recompute), once per trial:
   caught, and the block named.
3. gradients: bit-identical with and without the check, same seeds.
4. coverage: blocks compared per step, and the share of parameters inside compared blocks.
5. cost: seconds per step with plain checkpointing, with early stop off only (the check makes the recompute run
   the whole block), and with the full check.
6. GPU only, legitimate nondeterminism: a block with index_add_ (atomics on CUDA) differs between its runs with
   deterministic algorithms off, which is why calibrate() exists; with them on, it should be clean.

    python experiments/recompute_check.py --device cuda --out results/recompute
    python experiments/recompute_check.py --device cpu --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.utils.checkpoint as tuc  # noqa: E402

from sdcguard.recompute import RecomputeCheck  # noqa: E402

SIZES = {"cuda": dict(layers=8, width=768, heads=12, seq=512, batch=8, vocab=32000),
         "cpu": dict(layers=2, width=64, heads=4, seq=32, batch=2, vocab=128)}


def make(kind: str, dev, sz: dict):
    import transformers
    torch.manual_seed(0)
    if kind == "gpt2":
        cfg = transformers.GPT2Config(n_layer=sz["layers"], n_embd=sz["width"], n_head=sz["heads"],
                                      vocab_size=sz["vocab"], n_positions=sz["seq"], resid_pdrop=0.1,
                                      embd_pdrop=0.1, attn_pdrop=0.1)
        m = transformers.GPT2LMHeadModel(cfg)
    else:
        cfg = transformers.LlamaConfig(hidden_size=sz["width"], intermediate_size=4 * sz["width"],
                                       num_hidden_layers=sz["layers"], num_attention_heads=sz["heads"],
                                       num_key_value_heads=max(1, sz["heads"] // 4), vocab_size=sz["vocab"],
                                       max_position_embeddings=sz["seq"], attention_dropout=0.1)
        m = transformers.LlamaForCausalLM(cfg)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return m.to(dev).train()


def step_fn(m, dev, sz, seed):
    def f():
        torch.manual_seed(seed[0])
        seed[0] += 1
        ids = torch.randint(0, sz["vocab"], (sz["batch"], sz["seq"]), device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = m(input_ids=ids, labels=ids).loss
        loss.backward()
        m.zero_grad(set_to_none=True)
    return f


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _interleaved_medians(variants: dict, dev, rounds: int) -> dict:
    """Seconds per call of each variant (name: (call, setup or None, teardown or None)): the median of `rounds`,
    one call of each per round in an order that rotates, so whatever drifts while they're measured (the GPU's
    clocks, the allocator's caches) falls on all of them alike. Timed in blocks one after another, as until
    October 21, 2026, it fell on whichever block came last: the plain step's own time moved by up to 30% between
    runs, and the same code's cost read 6.8% to 16.1%. Round 0 warms every variant up and isn't counted; setup and
    teardown aren't timed."""
    names = list(variants)
    times: dict = {n: [] for n in names}
    for r in range(rounds + 1):
        for n in names[r % len(names):] + names[:r % len(names)]:
            call, setup, teardown = variants[n]
            if setup is not None:
                setup()
            try:
                _sync(dev)
                t0 = time.perf_counter()
                call()
                _sync(dev)
                dt = time.perf_counter() - t0
            finally:
                if teardown is not None:
                    teardown()
            if r:
                times[n].append(dt)
    return {n: round(sorted(ts)[len(ts) // 2], 5) for n, ts in times.items()}


def _host_profile(variants: dict, dev, top: int = 15) -> dict:
    """Where one step's host time goes (torch.profiler, one step of each variant after a warm-up step): the total
    and the operations with the most self CPU time. Reported only; no criterion reads it."""
    out = {}
    try:
        from torch.profiler import ProfilerActivity, profile
        for name, (call, setup, teardown) in variants.items():
            if setup is not None:
                setup()
            try:
                call()
                _sync(dev)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    call()
                    _sync(dev)
            finally:
                if teardown is not None:
                    teardown()
            rows = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)
            out[name] = {"self_cpu_ms_total": round(sum(e.self_cpu_time_total for e in rows) / 1e3, 3),
                         "top": [{"op": e.key[:80], "calls": e.count,
                                  "self_cpu_ms": round(e.self_cpu_time_total / 1e3, 3)} for e in rows[:top]]}
    except Exception as e:  # noqa: BLE001  (a profile that can't be taken costs the run nothing)
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def _fault(module, call, delta):
    n = {"i": 0}

    def hook(mod, inp, out):
        n["i"] += 1
        with torch.no_grad():
            corr = torch.zeros_like(out)
            if n["i"] == call:
                corr += delta * (out.abs() > out.abs().median()).to(out.dtype)
        return out + corr
    return module.register_forward_hook(hook)


def _inner(m, kind, layer):
    return m.transformer.h[layer].mlp.c_fc if kind == "gpt2" else m.model.layers[layer].mlp.up_proj


def one_model(kind, dev, sz, steps, trials, say=print) -> dict:
    m = make(kind, dev, sz)
    f = step_fn(m, dev, sz, [0])
    r = {"model": kind, "layers": sz["layers"]}
    rc = RecomputeCheck().install(m)
    try:
        r["clean"] = rc.calibrate(f, steps=steps)
        r["coverage"] = rc.coverage(m, f)
    finally:
        rc.uninstall()
    # faults: one per trial, in a different layer each time, in the forward run (call 1) or the recompute (2)
    caught = []
    for t in range(trials):
        layer, call = t % sz["layers"], 1 + (t % 2)
        h = _fault(_inner(m, kind, layer), call, 1e-2 if dev.type == "cuda" else 1e-3)
        seen = []
        rc = RecomputeCheck(on_alert=seen.append).install(m)
        try:
            f()
            rc.flush()
        finally:
            rc.uninstall()
            h.remove()
        caught.append({"layer": layer, "run": "forward" if call == 1 else "recompute",
                       "alerts": len(seen), "named_block": seen[0].detail["block_call"] if seen else None,
                       "right_block": bool(seen) and seen[0].detail["block_call"] == layer + 1})
    r["faults"] = {"trials": trials, "caught": sum(c["alerts"] == 1 for c in caught),
                   "right_block": sum(c["right_block"] for c in caught), "detail": caught}
    # gradients with and without the check, same seed
    def grads(check):
        torch.manual_seed(7)
        ids = torch.randint(0, sz["vocab"], (sz["batch"], sz["seq"]), device=dev)
        rc2 = RecomputeCheck().install(m) if check else None
        try:
            torch.manual_seed(8)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                m(input_ids=ids, labels=ids).loss.backward()
            if rc2:
                rc2.flush()
        finally:
            if rc2:
                rc2.uninstall()
        g = [p.grad.clone() for p in m.parameters() if p.grad is not None]
        m.zero_grad(set_to_none=True)
        return g
    a, b = grads(False), grads(True)
    r["gradients_bit_identical"] = len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))
    # cost: plain checkpointing, early stop off only (the one change the check makes to checkpointing, by itself:
    # no faster recompute that stops early, which on small blocks is slightly faster, since stopping early raises
    # an exception), the full check; one step of each in turn (_interleaved_medians)
    rc = RecomputeCheck()

    def no_early():
        with tuc.set_checkpoint_early_stop(False):
            f()

    def checked():
        f()
        rc.flush()
    secs = _interleaved_medians({"plain checkpointing": (f, None, None), "early stop off only": (no_early, None, None),
                                 "full check": (checked, lambda: rc.install(m), rc.uninstall)},
                                dev, max(5, 3 * steps))
    plain, full_recompute, full = (secs[k] for k in ("plain checkpointing", "early stop off only", "full check"))
    r["seconds_per_step"] = secs
    r["timed_rounds"] = max(5, 3 * steps)
    if dev.type == "cuda":
        r["host_profile"] = _host_profile({"plain checkpointing": (f, None, None),
                                           "full check": (checked, lambda: rc.install(m), rc.uninstall)}, dev)
    r["overhead_pct"] = {"early stop off only": round(100 * (full_recompute / plain - 1), 1),
                         "full check": round(100 * (full / plain - 1), 1)}
    say(f"{kind}: clean {r['clean']['mismatches']} mismatches in {r['clean']['recomputes_compared']} compares, "
        f"faults {r['faults']['caught']}/{trials} caught ({r['faults']['right_block']} named right), gradients "
        f"identical {r['gradients_bit_identical']}, overhead {r['overhead_pct']}")
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return r


class _Scatter(nn.Module):
    """A block whose forward uses index_add_ (atomic adds on CUDA: order, and so rounding, can vary)."""

    def __init__(self, width=1024, rows=1 << 16):
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.rows = rows

    def forward(self, x):
        h = self.lin(x)
        idx = torch.randint(0, 64, (h.shape[0],), device=h.device, generator=None)
        out = torch.zeros(64, h.shape[1], device=h.device, dtype=h.dtype)
        return out.index_add(0, idx, h).sum() + h.sum()


def nondeterminism(dev, say=print) -> dict:
    """In a child process: deterministic algorithms need CUBLAS_WORKSPACE_CONFIG set before CUDA starts, and it
    shouldn't change the cuBLAS kernels the other measurements in this process use."""
    if dev.type != "cuda":
        return {"skipped": "CPU kernels here are deterministic"}
    import subprocess
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--device", "cuda", "--nondeterminism-only"],
                       env=env, capture_output=True, text=True, timeout=900)
    try:
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        res = {"error": f"exit {p.returncode}", "output": (p.stdout + p.stderr)[-2000:]}
    say(f"nondeterminism: {json.dumps(res, default=str)[:400]}")
    return res


def _nondeterminism_here(dev) -> dict:
    res = {}
    for det in (False, True):
        torch.use_deterministic_algorithms(det, warn_only=False)
        try:
            torch.manual_seed(0)
            blk = _Scatter().to(dev)
            x = torch.randn(1 << 16, 1024, device=dev, requires_grad=True)

            def f():
                tuc.checkpoint(blk, x, use_reentrant=False).backward()
            res["deterministic algorithms on" if det else "deterministic algorithms off"] = \
                RecomputeCheck().calibrate(f, steps=5)
        except Exception as e:  # noqa: BLE001
            res["deterministic algorithms on" if det else "deterministic algorithms off"] = {
                "error": f"{type(e).__name__}: {str(e)[:200]}"}
        finally:
            torch.use_deterministic_algorithms(False)
    return res


def criteria(r: dict) -> dict:
    ms = [m for m in r.get("models", []) if isinstance(m, dict) and "error" not in m]
    ok_models = bool(ms) and len(ms) == len(r.get("models", []))
    c1 = ok_models and all(m["clean"]["mismatches"] == 0 and m["clean"]["recomputes_compared"] > 0 for m in ms)
    c2 = ok_models and all(m["faults"]["caught"] == m["faults"]["trials"] and
                           m["faults"]["right_block"] == m["faults"]["trials"] for m in ms)
    c3 = ok_models and all(m["gradients_bit_identical"] for m in ms)
    return {"1 clean: no mismatch on any clean step": c1,
            "2 every injected fault caught and its block named": c2,
            "3 gradients bit-identical with and without the check": c3,
            "all met": bool(c1 and c2 and c3)}


def run_all(dev, quick=False, say=print, deadline=None, reserve=120.0) -> dict:
    """deadline: time.monotonic() value after which no model starts (recorded as not run, never as passed)."""
    kind = "cuda" if dev.type == "cuda" else "cpu"
    sz = SIZES[kind]
    r = {"sizes": sz, "models": []}
    if dev.type == "cuda":
        r.update(gpu=torch.cuda.get_device_name(dev), torch=torch.__version__)
    try:
        import transformers
        r["transformers"] = transformers.__version__
    except Exception as e:  # noqa: BLE001
        r["error"] = f"transformers is needed: {e}"
        r["criteria"] = criteria(r)
        return r
    for model in ("gpt2", "llama"):
        if deadline is not None and time.monotonic() > deadline - reserve:
            r["models"].append({"model": model, "error": "not run: no time left before the function's limit"})
            continue
        try:
            r["models"].append(one_model(model, dev, sz, 2 if quick else 10, 2 if quick else 8, say))
        except Exception as e:  # noqa: BLE001
            import traceback
            r["models"].append({"model": model, "error": f"{type(e).__name__}: {str(e)[:300]}",
                                "trace": traceback.format_exc()[-1500:]})
    r["nondeterminism"] = nondeterminism(dev, say)
    r["criteria"] = criteria(r)
    return r


def markdown(r: dict) -> str:
    lines = [f"## recompute check {r.get('gpu', 'cpu')}", ""]
    for k, v in r.get("criteria", {}).items():
        lines.append(f"- {k}: {'met' if v else 'not met'}")
    for m in r.get("models", []):
        if "error" in m:
            lines.append(f"- {m['model']}: {m['error']}")
            continue
        lines.append(f"- {m['model']}: {m['clean']['recomputes_compared']} clean compares, "
                     f"{m['clean']['mismatches']} mismatches; faults {m['faults']['caught']}/{m['faults']['trials']}"
                     f" caught; parameters inside compared blocks "
                     f"{m['coverage']['parameter_share_inside_compared_blocks']:.0%}; overhead {m['overhead_pct']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nondeterminism-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.nondeterminism_only:
        print(json.dumps(_nondeterminism_here(torch.device(a.device)), default=str))
        return 0
    r = run_all(torch.device(a.device), a.quick)
    print(markdown(r))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"recompute_{a.device}.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
    return 0 if r["criteria"]["all met"] else 1


if __name__ == "__main__":
    sys.exit(main())
