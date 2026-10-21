"""The 0.4.0 features on a GPU, against criteria written in docs/METHOD.md before the run
("0.4.0 on GPUs").

1. fingerprints: the one-pass Triton fingerprint kernel against the PyTorch version, bit for bit
   (tests/_fingerprint_compare.py), and its speed on a large tensor.
2. compile: protect() under torch.compile(fullgraph=True) with Inductor: checks per checked step equal eager's,
   no alarm on clean steps, a wrong output inside a compiled checked step caught, and the time of unchecked and
   checked compiled steps.
3. grouped: torch._grouped_mm (MoE experts) in bf16, forward and backward, checked per expert: every expert with
   tokens checked in all three matmuls of every clean call, no alarm on them, one wrong value in one expert's
   output caught, and the cost.
4. blas: `sdcguard blas` on its default shapes with the GPU to itself (no MPS): every algorithm right. A wrong
   algorithm is also matched against NVIDIA's listed cuBLAS issues (sdcguard.known_issues), so a library bug
   NVIDIA already lists is reported as that, apart from sdcguard's own results.
5. capture: original-event capture in serving (serve.CAPTURE_SLOTS) inside a CUDA graph: a one-time wrong value
   kept as it alarmed and a fault in the check's own side not blamed on the output, and the capture's clean cost
   (METHOD.md's criteria 6 and 7).
6. capture_stress: capture under contention (criterion 8, added October 21, 2026, before its first run): every
   program of a layer's check over the line at once, several layers at once, more alarms than records, and reads
   still in flight.

The injected wrong values of 2 and 3 change their 128-column block by 10% of that block's L2 norm: the harm
study's measure of a fault's size (inject._rel_change), at which the default check caught 99% of faults there.
The ones in 5 and 6 are as large as their block's L2 norm: those phases test what is kept, not how small a fault
is caught.

    python experiments/v040_check.py --device cuda --out results/v040
    python experiments/v040_check.py --device cpu --quick         # smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sdcguard import SDCGuard  # noqa: E402
from sdcguard.screen import ScreenModel  # noqa: E402

CFG = {"cuda": dict(vocab=32000, layers=8, width=1024, heads=16, seq=512, batch=8),
       "cpu": dict(vocab=256, layers=2, width=128, heads=4, seq=32, batch=2)}
FAULT = 0.1                  # the size of every injected wrong value, on the harm study's scale


def add_fault_(row: torch.Tensor, col: int) -> float:
    """Add one wrong value to row[col]: FAULT times the L2 norm of its 128-column block, so the block changes by
    10% on the harm study's measure (||change|| / ||block||, inject._rel_change), the size at which the default
    check caught 99% of faults there. Returns the value added."""
    start = (col // 128) * 128
    delta = FAULT * float(torch.linalg.vector_norm(row[start:start + 128].float()))
    row[col] += delta
    return delta


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def fingerprints(dev) -> dict:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"),
                                                       os.environ.get("PYTHONPATH", "")]))
    if dev.type == "cpu":
        env.update(TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1")
    p = subprocess.run([sys.executable, os.path.join(HERE, "..", "tests", "_fingerprint_compare.py"), dev.type],
                       env=env, capture_output=True, text=True, timeout=1800)
    out = {"agree": p.returncode == 0 and "ALL AGREE" in p.stdout, "output": (p.stdout + p.stderr)[-3000:]}
    if dev.type == "cuda":
        from sdcguard import _fingerprint
        from sdcguard.recompute import _fingerprint_torch
        x = torch.randn(64 * 1024 * 1024 // 2, device=dev, dtype=torch.bfloat16)       # 64 MiB of bf16
        bits = x.view(torch.int16)

        def ms(fn, reps=20):
            fn()
            _sync(dev)
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            _sync(dev)
            return round((time.perf_counter() - t0) / reps * 1e3, 3)
        out["ms_64mib_kernel"] = ms(lambda: _fingerprint.fingerprint_bits(bits))
        out["ms_64mib_torch"] = ms(lambda: _fingerprint_torch(bits))
        out["kernel_used"] = _fingerprint.usable(dev, _fingerprint_torch)
    return out


def _model(dev, cfg, seed=0):
    torch.manual_seed(seed)
    return ScreenModel(cfg["vocab"], cfg["layers"], cfg["width"], cfg["heads"], cfg["seq"]).to(dev)


def compile_check(dev, steps=40, say=print) -> dict:
    cfg = CFG[dev.type]
    low = torch.bfloat16
    gen = torch.Generator().manual_seed(3)
    batches = [torch.randint(0, cfg["vocab"], (cfg["batch"], cfg["seq"] + 1), generator=gen).to(dev)
               for _ in range(4)]

    def step(model, f, g, n, timer):
        idx = batches[n % len(batches)]
        _sync(dev)
        t0 = time.perf_counter()
        with (g.on_step(n) if g is not None else _null()):
            with torch.autocast(dev.type, dtype=low):
                logits = f(idx[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
            loss.backward()
        model.zero_grad(set_to_none=True)
        _sync(dev)
        timer.append(time.perf_counter() - t0)

    out = {}
    # eager, every step checked: the reference count of checks per step
    m = _model(dev, cfg)
    g = SDCGuard(every=1).protect(m)
    t = []
    step(m, m, g, 0, t)
    out["eager_checks_per_step"] = g.stats["checked"]
    g.unprotect()
    # compiled: unguarded, then guarded every 10th step
    torch._dynamo.reset()
    m = _model(dev, cfg)
    f = torch.compile(m, fullgraph=True)
    plain = []
    for n in range(steps):
        step(m, f, None, n, plain)
    torch._dynamo.reset()
    m = _model(dev, cfg)
    g = SDCGuard(every=10).protect(m)
    f = torch.compile(m, fullgraph=True)
    times, counts = [], []
    for n in range(steps):
        before = g.stats["checked"]
        step(m, f, g, n, times)
        counts.append(g.stats["checked"] - before)
    checked_counts = [c for n, c in enumerate(counts) if n % 10 == 0]
    warm = 10                               # compile happens in the first steps of each version
    un = [t for n, t in enumerate(times) if n % 10 != 0 and n >= warm]
    ch = [t for n, t in enumerate(times) if n % 10 == 0 and n >= warm]
    pl = plain[warm:]
    med = lambda v: sorted(v)[len(v) // 2] if v else None      # noqa: E731
    out.update(checks_per_checked_step=checked_counts, checks_on_unchecked_steps=sum(counts) - sum(checked_counts),
               alerts=g.stats["alerts"], unjudged=g.stats["unjudged"],
               ms_unguarded=round(med(pl) * 1e3, 2) if pl else None,
               ms_unchecked=round(med(un) * 1e3, 2) if un else None,
               ms_checked=round(med(ch) * 1e3, 2) if ch else None)
    # a wrong output inside a compiled checked step
    real = torch.nn.functional.linear

    def faulty(x, w, b=None):                 # one wrong value (add_fault_)
        y = real(x, w, b)
        add_fault_(y.view(-1, y.shape[-1])[0], 5)
        return y
    caught = 0
    trials = 5
    for n in range(trials):
        before = g.stats["alerts"]
        torch.nn.functional.linear = faulty
        try:
            step(m, f, g, 10 * (steps // 10 + 1 + n), [])
        finally:
            torch.nn.functional.linear = real
        caught += g.stats["alerts"] > before
    out.update(fault_trials=trials, faults_caught=caught)
    g.unprotect()
    say(f"compile: {out}")
    return out


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def grouped(dev, calls=200, say=print) -> dict:
    if not hasattr(torch, "_grouped_mm"):
        return {"error": "this PyTorch has no torch._grouped_mm"}
    experts, k, n, tokens = (64, 1024, 2048, 8192) if dev.type == "cuda" else (4, 64, 128, 96)
    gen = torch.Generator().manual_seed(7)
    g = SDCGuard()
    out = {"experts": experts, "shape": [tokens, k, n], "calls": calls}
    alarms = unjudged = expected = 0
    t_plain, t_checked = [], []
    for c in range(calls):
        counts = torch.distributions.Dirichlet(torch.ones(experts)).sample() * tokens
        counts = counts.floor().long()
        counts[-1] += tokens - int(counts.sum())
        # every expert with tokens is checked in the forward matmul and in both backward ones (dA and dB); one with
        # no tokens only in its weight gradient dB, an (k, n) output that must be all zeros (a k = 0 product)
        expected += 3 * int((counts > 0).sum()) + int((counts == 0).sum())
        offs = torch.cumsum(counts, 0).to(torch.int32).to(dev)
        a = torch.randn(tokens, k, generator=gen).to(torch.bfloat16).to(dev).requires_grad_(True)
        b = (torch.randn(experts, k, n, generator=gen) / k ** 0.5).to(torch.bfloat16).to(dev).requires_grad_(True)
        _sync(dev)
        t0 = time.perf_counter()
        y = torch._grouped_mm(a, b, offs=offs)
        y.float().pow(2).mean().backward()
        _sync(dev)
        t_plain.append(time.perf_counter() - t0)
        a.grad = b.grad = None
        before = (g.stats["alerts"], g.stats["unjudged"])
        t0 = time.perf_counter()
        with g.at_step(c):
            y = torch._grouped_mm(a, b, offs=offs)
            y.float().pow(2).mean().backward()
        _sync(dev)
        t_checked.append(time.perf_counter() - t0)
        alarms += g.stats["alerts"] - before[0]
        unjudged += g.stats["unjudged"] - before[1]
    out.update(clean_alarms=alarms, unjudged=unjudged, checks=g.stats["checked"], expected_checks=expected,
               ms_plain=round(sorted(t_plain)[len(t_plain) // 2] * 1e3, 2),
               ms_checked=round(sorted(t_checked)[len(t_checked) // 2] * 1e3, 2))
    # one wrong value in one expert's output (add_fault_)
    from torch.utils._python_dispatch import TorchDispatchMode
    caught, trials = 0, (100 if dev.type == "cuda" else 10)

    class Faulty(TorchDispatchMode):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out_ = func(*args, **(kwargs or {}))
            if "grouped_mm" in str(func) and self.idx is not None:
                r, c = self.idx
                add_fault_(out_[r], c)
                self.idx = None                       # the forward output only
            return out_
    for t in range(trials):
        a = torch.randn(tokens, k, generator=gen).to(torch.bfloat16).to(dev)
        b = (torch.randn(experts, k, n, generator=gen) / k ** 0.5).to(torch.bfloat16).to(dev)
        offs = torch.linspace(tokens / experts, tokens, experts).round().to(torch.int32).to(dev)
        before = g.stats["alerts"]
        idx = (int(torch.randint(0, tokens, (1,), generator=gen)), int(torch.randint(0, n, (1,), generator=gen)))
        with Faulty(idx):
            with g.at_step(10_000 + t):
                torch._grouped_mm(a, b, offs=offs)
        caught += g.stats["alerts"] > before
    out.update(fault_trials=trials, faults_caught=caught)
    say(f"grouped: {out}")
    return out


def listed_issues_for(wrong: dict, matches: list) -> list:
    """The listed cuBLAS issues (sdcguard.known_issues.match() for this machine) that could explain one wrong
    algorithm: those that apply or may apply here and that NVIDIA ties to the same algorithm (and, where it says
    so, to clusters of 4 or more CTAs, or to split-K: an algorithm that ran with SPLITK_NUM 1 can't have hit a
    split-K bug). A candidate explanation, not a diagnosis."""
    out = []
    for m in matches or []:
        if not isinstance(m, dict) or m.get("status") not in ("applies", "may apply"):
            continue
        what = m.get("what") or ""
        named = re.search(r"algorithm (\d+)", what)
        if named is None or int(named.group(1)) != wrong.get("algo_id"):
            continue
        if "4 or more CTAs" in what and (wrong.get("cluster_ctas") or 0) < 4:
            continue
        if "split-K" in what and wrong.get("splitk") is not None and wrong["splitk"] <= 1:
            continue
        out.append(m.get("id"))
    return out


def _config(w: dict) -> str:
    """What tells one configuration of an algorithm from another in a report line."""
    parts = [f"{w['cluster_ctas']}-CTA clusters"] if w.get("cluster_ctas") else []
    if w.get("splitk") is not None:
        parts.append(f"split-K {w['splitk']}" if w["splitk"] > 1 else "no split-K")
    return f" ({', '.join(parts)})" if parts else ""


def wrong_with_listed_issues(sweeps: list, matches: list) -> list:
    """Every wrong algorithm of some sweeps, with the listed cuBLAS issues that could explain it (none: nothing
    NVIDIA lists accounts for it)."""
    return [{"shape": sw.get("shape"), "dtype": sw.get("dtype"), "index": w.get("index"), "algo_id": w.get("algo_id"),
             "cluster_ctas": w.get("cluster_ctas"), "splitk": w.get("splitk"),
             "listed_issues": listed_issues_for(w, matches)}
            for sw in sweeps or [] if isinstance(sw, dict)
            for w in (sw.get("wrong") or sw.get("wrong_algorithms") or [])]


def blas_sweep(dev, say=print) -> dict:
    from sdcguard import blas, known_issues
    ok, why = blas.available()
    if not ok:
        return {"error": why}
    res = []
    for shape in blas.DEFAULT_SHAPES:
        for dt in (torch.bfloat16, torch.float16):
            r = blas.sweep(*shape, dtype=dt, device=dev)
            res.append({"shape": list(shape), "dtype": str(dt).replace("torch.", ""), "verdict": r["verdict"],
                        "algorithms_tested": r.get("algorithms_tested"), "wrong": r.get("wrong_algorithms"),
                        "error": r.get("error")})
            say(f"blas {shape} {dt}: {r['verdict']} ({r.get('algorithms_tested')} algorithms)")
    out = {"sweeps": res}
    try:
        out["known_issues"] = known_issues.report(dev)
    except Exception as e:  # noqa: BLE001  (the report is context for a failure, never a reason for one)
        out["known_issues"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    out["wrong_with_listed_issues"] = wrong_with_listed_issues(res, out["known_issues"].get("matches"))
    return out


def capture_check(dev, replays=200, say=print) -> dict:
    """Original-event capture (serve.CAPTURE_SLOTS) where it has to work: inside a CUDA graph (on the CPU, the same
    sequence without a graph, through the PyTorch check). A bf16 stack of linear layers attached to a ServeGuard is
    captured once; every replay gets fresh inputs copied into the graph's input buffer, so each replay overwrites the
    memory of the one before. On one replay the graph makes one wrong value in layer 1's output (a fault tensor it
    adds, zero on every other replay), as large as that block's L2 norm; on another, layer 2's check has a wrong
    expected side (its block sums changed for that replay only). Then the clean cost of capture: the Triton check of
    one layer with and without the capture buffer."""
    import statistics
    import tempfile
    from sdcguard import _serve_fast
    from sdcguard.serve import ServeGuard
    cuda = dev.type == "cuda"
    width, batch, L = (4096, 8, 4) if cuda else (256, 4, 3)
    out = {"layers": L, "width": width, "batch": batch, "replays": replays, "graph": cuda}
    if cuda:
        out["first_use_comparison"] = {str(dt).replace("torch.", ""): bool(_serve_fast.verified(dev, dt))
                                       for dt in (torch.bfloat16, torch.float16)}
    torch.manual_seed(0)
    model = torch.nn.Sequential(*[torch.nn.Linear(width, width) for _ in range(L)]).to(torch.bfloat16).to(dev)
    ev_dir = tempfile.mkdtemp(prefix="sdcguard_capture_")
    g = ServeGuard(evidence_dir=ev_dir).attach(model)
    out["uses_fast"] = [g.uses_fast(s) for s in range(L)]
    fault = torch.zeros(batch, width, dtype=torch.bfloat16, device=dev)
    target, real = model[1].weight, F.linear

    def linear(x, w, b=None):
        y = real(x, w, b)
        return y + fault if w is target else y
    x_static = torch.zeros(batch, width, dtype=torch.bfloat16, device=dev)

    def forward():
        with torch.no_grad():
            return model(x_static)
    bad, chk, r, c = replays // 3, (2 * replays) // 3, 5 % batch, 1000 % width
    blk = c // 128
    cs2 = g.layers[2]["cs"]
    col = 2 * blk                                             # layer 2's plain checksum of that block (first part)
    found, gen = [], torch.Generator().manual_seed(1)
    F.linear = linear
    try:
        if cuda:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(s)
            g.poll(wait=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            run = graph.replay
        else:
            run = forward
        for i in range(replays):
            x_static.copy_(torch.randn(batch, width, generator=gen).to(torch.bfloat16).to(dev))
            if i == bad:
                with torch.no_grad():
                    clean = real(model[0](x_static), target, model[1].bias)    # this replay's clean layer-1 output
                delta = float(torch.linalg.vector_norm(clean[r, blk * 128:(blk + 1) * 128].float()))
                fault[r, c] = delta
                out["fault"] = {"replay": i, "layer": "1", "row": r, "col": c, "added": delta}
            if i == chk:
                saved = cs2.parts[:, col].clone()
                cs2.parts[:, col] += 1.0
                out["checker_fault"] = {"replay": i, "layer": "2", "block": blk}
            run()
            if i == bad:
                fault.zero_()
            if i == chk:
                cs2.parts[:, col] = saved
            found += [(i, e) for e in g.poll(wait=True)]
    finally:
        F.linear = real
    out["alerts"] = [{"replay": i, "where": e.detail["where"], "captured": e.detail.get("captured"),
                      "verdict": (e.detail.get("recheck") or {}).get("verdict"),
                      "outside_envelope": ((e.detail.get("recheck") or {}).get("cpu") or {}).get(
                          "outputs_outside_rounding_envelope"),
                      "evidence": e.detail.get("evidence")} for i, e in found]
    one = next((a for a in out["alerts"] if a["replay"] == bad and a["where"] == "1"), None)
    if one and isinstance(one["evidence"], str) and os.path.exists(one["evidence"] + ".pt"):
        p = torch.load(one["evidence"] + ".pt")
        info = json.load(open(one["evidence"] + ".json"))
        out["evidence_holds_the_wrong_value"] = bool(
            info.get("captured_at_check") and abs(float(p["C"][0, c % 128]) - float(
                (p["A"].double() @ p["B"].double())[0, c % 128] + (p["bias"].double()[c % 128]
                                                                   if p["bias"] is not None else 0.0))) > 0.5 * delta)
    out["stats"] = {k: g.stats[k] for k in ("alerts", "captured", "captures_dropped", "checks_executed")}
    g.detach()
    if cuda and all(out["uses_fast"]):
        # the clean cost: the same Triton check of one layer, with and without a capture buffer (median of 5 rounds
        # of 200 calls, by CUDA events)
        from sdcguard.serve import LayerChecksums
        cs = LayerChecksums(target.detach(), None, 128)
        x = torch.randn(batch, width, device=dev).to(torch.bfloat16)
        y = real(x, target)
        st = torch.zeros(1, 4, dtype=torch.int64, device=dev)
        cm, cd = _serve_fast._capture_buffers(dev, width)
        times = {}
        for name, kw in (("without", {}), ("with", {"cap_meta": cm, "cap_data": cd, "cap_w": 128, "cap_tag": 2})):
            rounds = []
            for _ in range(5):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(200):
                    _serve_fast.check(x, y, *cs.tensors(False), st, 0, 6.0, 1e-20, **kw)
                b.record()
                b.synchronize()
                rounds.append(a.elapsed_time(b) / 200 * 1000)
            times[name] = statistics.median(rounds)
        out["check_us"] = times
    say(f"capture: {len(found)} alerts, {out['stats']['captured']} captured")
    return out


def capture_stress(dev, say=print) -> dict:
    """Capture under contention (criterion 8, added October 21, 2026, after the first run and before this check's
    first run, as the outside review of that day asked: labeled apart from criteria 6 and 7, which stay as
    written). 6 bf16 linear layers side by side, 4096 to 4096, each applied to the same input of batch 8, attached
    to a ServeGuard and captured once in a CUDA graph (on the CPU: 256 to 768 through the PyTorch check, a size this
    PyTorch's CPU matmul rounds correctly; PyTorch 2.8's on arm64 doesn't at 768 inputs), replayed with fresh inputs
    and read after
    every replay unless said. Each wrong value is as large as its 128-column block's L2 norm in that replay's clean
    output, made by a fault tensor the graph adds to that layer's output (zero on every other replay):
      a. every row of layer 3 wrong once in every block: every program of that layer's check over the line at once,
         racing for the layer's one capture;
      b. one wrong value in each of layers 0 to 3 (row i, block i);
      c. one in each of the 6 layers (more than the 4 records), then a replay with one in layer 5 only;
      d. one in layer 0, read without waiting, then one in layer 1, read without waiting, then a read that waits
         (on a GPU a read without waiting reports the one before it once that copy has arrived)."""
    from sdcguard.serve import CAPTURE_SLOTS, ServeGuard
    cuda = dev.type == "cuda"
    k, n, batch, L = (4096, 4096, 8, 6) if cuda else (256, 768, 8, 6)     # (6 blocks at least: c uses blocks 0 to 5)
    nb = n // 128
    out = {"layers": L, "k": k, "n": n, "batch": batch, "graph": cuda, "slots": CAPTURE_SLOTS}
    torch.manual_seed(0)
    # side by side, not stacked: one layer's wrong value doesn't become the next layer's input
    model = torch.nn.ModuleList([torch.nn.Linear(k, n) for _ in range(L)]).to(torch.bfloat16).to(dev)
    g = ServeGuard().attach(model)
    out["uses_fast"] = [g.uses_fast(s) for s in range(L)]
    faults = [torch.zeros(batch, n, dtype=torch.bfloat16, device=dev) for _ in range(L)]
    which = {id(m.weight): i for i, m in enumerate(model)}
    real = F.linear

    def linear(x, w, b=None):
        y = real(x, w, b)
        i = which.get(id(w))
        return y + faults[i] if i is not None else y
    x_static = torch.zeros(batch, k, dtype=torch.bfloat16, device=dev)
    gen = torch.Generator().manual_seed(2)

    def forward():
        with torch.no_grad():
            return [m(x_static) for m in model]

    def clean_outputs():                    # this replay's clean output of every layer, block norms per (row, block)
        with torch.no_grad():
            return [torch.linalg.vector_norm(real(x_static, m.weight, m.bias).float().view(batch, nb, 128), dim=-1)
                    .cpu() for m in model]

    def wrong(layer, r, blk, norms):        # one wrong value in (row r, block blk) of a layer's output
        c = blk * 128 + (r * 37 + blk * 11) % 128
        faults[layer][r, c] = float(norms[layer][r, blk])
        return (layer, r, blk)

    def replay(plan, wait=True):
        """plan(norms) makes the wrong values; returns (made, alerts read now)."""
        x_static.copy_(torch.randn(batch, k, generator=gen).to(torch.bfloat16).to(dev))
        made = plan(clean_outputs()) if plan else []
        run()
        for f in faults:
            f.zero_()
        evs = g.poll(wait=wait)
        return made, [{"where": int(e.detail["where"]), "row": e.detail["row"], "col_block": e.detail["col_block"],
                       "captured": e.detail.get("captured"), "capture_missing": e.detail.get("capture_missing"),
                       "verdict": (e.detail.get("recheck") or {}).get("verdict"),
                       "outside_envelope": ((e.detail.get("recheck") or {}).get("cpu") or {}).get(
                           "outputs_outside_rounding_envelope")} for e in evs]
    F.linear = linear
    try:
        if cuda:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(s)
            g.poll(wait=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            run = graph.replay
        else:
            run = forward
        out["clean_before"] = sum(len(replay(None)[1]) for _ in range(3))
        made, alerts = replay(lambda n: [wrong(3, r, b, n) for r in range(batch) for b in range(nb)])
        out["a"] = {"made": len(made), "alerts": alerts}
        out["a"]["captured_was_made_wrong"] = bool(alerts and alerts[0]["captured"]) and (
            3, alerts[0]["captured"]["row"], alerts[0]["captured"]["col_block"]) in set(made)
        out["clean_between"] = [len(replay(None)[1])]
        made, alerts = replay(lambda n: [wrong(i, i, i, n) for i in range(4)])
        out["b"] = {"made": made, "alerts": alerts}
        out["clean_between"].append(len(replay(None)[1]))
        made, alerts = replay(lambda n: [wrong(i, i % batch, i, n) for i in range(L)])
        out["c"] = {"made": made, "alerts": alerts}
        made, alerts = replay(lambda n: [wrong(5, 1, 2, n)])
        out["c"]["then"] = {"made": made, "alerts": alerts}
        out["c"]["dropped"] = g.stats["captures_dropped"]
        out["clean_between"].append(len(replay(None)[1]))
        m1, a1 = replay(lambda n: [wrong(0, 2, 3, n)], wait=False)
        m2, a2 = replay(lambda n: [wrong(1, 3, 4, n)], wait=False)
        a3 = g.poll(wait=True)
        out["d"] = {"made": m1 + m2, "alerts_first_read": a1, "alerts_second_read": a2,
                    "alerts_last_read": [{"where": int(e.detail["where"]), "captured": e.detail.get("captured"),
                                          "verdict": (e.detail.get("recheck") or {}).get("verdict"),
                                          "outside_envelope": ((e.detail.get("recheck") or {}).get("cpu") or {}).get(
                                              "outputs_outside_rounding_envelope")} for e in a3]}
    finally:
        F.linear = real
        g.detach()
    out["stats"] = {k: g.stats[k] for k in ("alerts", "captured", "captures_dropped", "checks_executed")}
    say(f"capture stress: {out['stats']}")
    return out


def stress_met(st: dict, gpu: bool = True) -> Optional[bool]:
    """Criterion 8 (capture_stress) from its report: on a GPU, in a CUDA graph with the Triton check on every layer
    (gpu=False judges a CPU run of the same sequence, for the tests). None when there is no report."""
    if not st:
        return None
    if "error" in st or gpu and not (st.get("graph") is True
                                     and all(x is True for x in st.get("uses_fast") or [False])):
        return False

    def kept(a, layer, row, blk):           # the alert on `layer` kept exactly that row and block, right as rechecked
        c = a.get("captured") or {}
        return (a["where"] == layer and c.get("row") == row and c.get("col_block") == blk
                and a.get("verdict") == "output_wrong" and a.get("outside_envelope") == 1)
    # no alert on any clean replay, and none but the 14 the stages make (a 1, b 4, c 6 and then 1, d 2)
    ok = st.get("clean_before") == 0 and st.get("clean_between") == [0, 0, 0] \
        and (st.get("stats") or {}).get("alerts") == 14
    a = st.get("a") or {}
    al = a.get("alerts") or []
    ok = ok and len(al) == 1 and al[0]["where"] == 3 and a.get("captured_was_made_wrong") is True \
        and al[0]["verdict"] == "output_wrong" and al[0]["outside_envelope"] == 1
    b = st.get("b") or {}
    bl = sorted(b.get("alerts") or [], key=lambda x: x["where"])
    ok = ok and len(bl) == 4 and all(kept(x, i, i, i) for i, x in enumerate(bl))
    c = st.get("c") or {}
    cl = sorted(c.get("alerts") or [], key=lambda x: x["where"])
    kept_c = [x for x in cl if x.get("captured")]
    full = [x for x in cl if not x.get("captured") and "buffer was full" in str(x.get("capture_missing"))]
    ok = ok and len(cl) == 6 and len(kept_c) == st.get("slots") == 4 and len(full) == 2 \
        and all(kept(x, x["where"], x["where"] % st["batch"], x["where"]) for x in kept_c)
    then = (c.get("then") or {}).get("alerts") or []
    ok = ok and len(then) == 1 and kept(then[0], 5, 1, 2) and c.get("dropped") == 2
    # d: whichever read reports them (a read without waiting reports the one before it once that copy has arrived,
    # on a GPU; on a CPU it reports at once), the two alerts come in order, each with its own row
    d = st.get("d") or {}
    reads = [d.get(k) for k in ("alerts_first_read", "alerts_second_read", "alerts_last_read")]
    seq = [x for r in reads if isinstance(r, list) for x in r]
    ok = ok and all(isinstance(r, list) for r in reads) and len(seq) == 2 and kept(seq[0], 0, 2, 3) \
        and kept(seq[1], 1, 3, 4)
    return bool(ok)


def criteria(r: dict) -> dict:
    """The criteria in docs/METHOD.md ("0.4.0 on GPUs"). Each one is met only by a report of work that ran in
    full, as registered: a missing or partial stage (no kernel in use, no timing, fewer steps, calls, trials or
    sweeps than the protocol runs, no CUDA graph) doesn't meet it. (Tightened October 21, 2026, after an outside
    review showed the first version met several criteria on reports with such gaps; the real H100 report of that
    morning gets the same verdicts on criteria 1 to 7 from both versions.)"""
    def positive(d, k):
        return isinstance(d.get(k), (int, float)) and math.isfinite(d[k]) and d[k] > 0
    fp = r.get("fingerprints") or {}
    c1 = fp.get("agree") is True and fp.get("kernel_used") is True
    cc = r.get("compile") or {}
    steps = cc.get("checks_per_checked_step")
    eager = cc.get("eager_checks_per_step")
    c2 = (isinstance(steps, list) and len(steps) == 4 and isinstance(eager, int) and eager > 0   # 40 steps, every 10th
          and all(x == eager for x in steps)
          and cc.get("checks_on_unchecked_steps") == 0 and cc.get("alerts") == 0 and cc.get("unjudged") == 0
          and cc.get("fault_trials") == 5 and cc.get("faults_caught") == 5)
    c3 = None
    if positive(cc, "ms_unchecked") and positive(cc, "ms_unguarded"):
        c3 = cc["ms_unchecked"] <= 1.02 * cc["ms_unguarded"] + 0.5
    gr = r.get("grouped") or {}
    # "no alarm and nothing unjudged in 200 calls" says something only about calls that were checked: every
    # expert with tokens, in all three matmuls of every call (3 checks per expert with tokens, 1 per expert with
    # none: between 64 + 2 and 3 * 64 per call when any expert has tokens)
    c4 = ("error" not in gr and gr.get("clean_alarms") == 0 and gr.get("unjudged") == 0
          and gr.get("calls") == 200 and gr.get("experts") == 64
          and isinstance(gr.get("expected_checks"), int) and 66 * 200 <= gr["expected_checks"] <= 192 * 200
          and gr.get("checks") == gr["expected_checks"]
          and gr.get("fault_trials") == 100 and gr.get("faults_caught", 0) >= 99)
    bl = r.get("blas") or {}
    # every registered shape in both dtypes, each with algorithms actually run: a sweep that's missing, errored or
    # ran nothing doesn't count as right
    from sdcguard.blas import DEFAULT_SHAPES
    want = {(tuple(shape), dt) for shape in DEFAULT_SHAPES for dt in ("bfloat16", "float16")}
    sweeps = bl.get("sweeps") or []
    got = [(tuple(s.get("shape") or ()), s.get("dtype")) for s in sweeps]
    c5 = (len(got) == len(want) and set(got) == want
          and all(s.get("verdict") == "PASS" and not s.get("error") and not s.get("wrong")
                  and isinstance(s.get("algorithms_tested"), int) and s["algorithms_tested"] > 0 for s in sweeps))
    # a wrong algorithm that a listed NVIDIA issue accounts for still fails criterion 5, but is reported as that
    # library issue, apart from sdcguard's own results
    cp = r.get("capture") or {}
    alerts = cp.get("alerts") or []
    flt, chk = cp.get("fault") or {}, cp.get("checker_fault") or {}
    first = [a for a in alerts if a["replay"] == flt.get("replay") and a["where"] == "1"]
    second = [a for a in alerts if a["replay"] == chk.get("replay") and a["where"] == "2"]
    use = cp.get("first_use_comparison") or {}
    c6 = ("error" not in cp and cp.get("graph") is True and cp.get("replays") == 200
          and use.get("bfloat16") is True and use.get("float16") is True
          and bool(cp.get("uses_fast")) and all(x is True for x in cp["uses_fast"])
          and len(alerts) == 2 and len(first) == 1 and len(second) == 1
          and (first[0]["captured"] or {}).get("row") == flt.get("row")
          and (first[0]["captured"] or {}).get("col_block") == flt.get("col", -1) // 128
          and first[0]["verdict"] == "output_wrong" and first[0]["outside_envelope"] == 1
          and cp.get("evidence_holds_the_wrong_value") is True and second[0]["verdict"] == "output_agrees")
    c8 = stress_met(r.get("capture_stress") or {}) if "capture_stress" in r else False
    timing = cp.get("check_us") or {}
    c7 = timing["with"] <= 1.05 * timing["without"] if positive(timing, "with") and positive(timing, "without") \
        else None
    library = [f"{'x'.join(map(str, w['shape'] or []))} {w['dtype']} algorithm {w['algo_id']}{_config(w)}: "
               + (f"listed cuBLAS issue {', '.join(w['listed_issues'])}" if w["listed_issues"]
                  else "no listed cuBLAS issue")
               for w in (bl.get("wrong_with_listed_issues") or [])]
    return {"1 fingerprint kernel equals the PyTorch version": c1,
            "2 compiled fullgraph: same checks as eager, clean, faults caught": c2,
            "3 compiled unchecked steps within 2% of an unguarded compiled step": c3,
            "4 grouped GEMM: clean, at least 99% of single wrong values caught": c4,
            "5 blas: every algorithm right on every default shape (no MPS)": c5,
            "5 wrong algorithms and the listed cuBLAS issues that match them": library,
            "6 capture in a CUDA graph: the one-time fault kept as it alarmed, a checker fault not blamed": c6,
            "7 capture costs the clean check at most 5%": c7,
            "8 capture under contention: racing programs, every layer at once, a full buffer, reads in flight": c8,
            "all met": bool(c1 and c2 and c3 and c4 and c5 and c6) and c7 is True and c8 is True}


def run_all(dev, quick=False, say=print) -> dict:
    r = {"torch": torch.__version__}
    if dev.type == "cuda":
        r.update(gpu=torch.cuda.get_device_name(dev), cc=list(torch.cuda.get_device_capability(dev)))
    parts = [("fingerprints", lambda: fingerprints(dev)),
             ("compile", lambda: compile_check(dev, 12 if quick else 40, say)),
             ("grouped", lambda: grouped(dev, 4 if quick else 200, say)),
             ("capture", lambda: capture_check(dev, 12 if quick else 200, say)),
             ("capture_stress", lambda: capture_stress(dev, say)),
             ("blas", lambda: blas_sweep(dev, say))]
    for name, fn in parts:
        t0 = time.time()
        try:
            r[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            r[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-1500:]}
        say(f"{name}: done in {time.time() - t0:.0f} s")
    r["criteria"] = criteria(r)
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    r = run_all(torch.device(a.device), a.quick)
    print(json.dumps(r["criteria"], indent=2))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"v040_{a.device}.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
    return 0 if r["criteria"]["all met"] else 1


if __name__ == "__main__":
    sys.exit(main())
