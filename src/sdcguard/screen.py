"""Screen GPUs for silent corruption with a real workload, and keep evidence you can replay.

    python -m sdcguard.screen --minutes 10            # every GPU this machine has, in parallel
    python -m sdcguard.screen --device cuda:3         # one GPU
    python -m sdcguard.screen --replay screen_out/GPU-.../evidence_0.pt    # on this or another GPU

For GPU clouds and anyone accepting, re-validating or retiring GPUs. Offline diagnostics run
synthetic patterns and miss many bad GPUs (in ByteDance's fleet, vendor diagnostics caught 2 of 8
known-bad machines). This runs what the GPUs are bought for, a transformer, training and serving,
with every matrix multiply checked:

  training steps   forward and backward of a GPT-style model (bf16 autocast), every matmul of every
                   step checked by SDCGuard, including the attention matmuls
  serving passes   the same model in bf16, every linear layer checked by ServeGuard

Everything is seeded (weights, data), so the exact workload can be rerun on another GPU.

When a check fails, the inputs and the wrong output of that matmul are saved (evidence_N.pt plus
a readable evidence_N.json with the GPU's UUID, serial number and PCI slot). --replay recomputes
that matmul many times and checks every result: on the faulty GPU the error tends to come back; on
a healthy GPU the same inputs give the right answer. That is the "failure follows the board"
evidence NVIDIA's RMA process asks for when its field diagnostic passes.

Exit code: 0 = no corrupted results, 1 = corruption found (evidence saved), 2 = could not run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import __version__
from .checksum import check_matmul
from . import verdict
from .events import device_identity
from .guard import SDCGuard
from .serve import ServeGuard

SIZES = {   # layers, width, heads, sequence, batch
    "tiny": (2, 128, 4, 64, 4),
    "small": (8, 1024, 16, 512, 8),
    "1b": (16, 2048, 16, 1024, 8),
    "3b": (24, 3072, 24, 1024, 8),
}


class _Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        # attention written out (not fused), so its two matmuls are checked too
        att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(C // self.h))
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class ScreenModel(nn.Module):
    def __init__(self, vocab: int, n_layer: int, d: int, h: int, seq: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList([_Block(d, h) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def _nvidia_smi(index) -> dict:
    """Serial number, VBIOS and driver from nvidia-smi (for RMA paperwork), when available. `index` is the
    GPU's UUID (preferred: nvidia-smi's index order isn't CUDA's) or its nvidia-smi index."""
    if index is None:
        return {}
    try:
        out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=serial,vbios_version,driver_version,uuid",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        line = [ln for ln in out.splitlines() if ln.strip()][-1]
        serial, vbios, driver, uuid = [s.strip() for s in line.split(",")][:4]
        return {"serial": serial, "vbios": vbios, "driver": driver, "smi_uuid": uuid}
    except Exception:  # noqa: BLE001
        return {}


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Recorder(SDCGuard):
    """SDCGuard that keeps the operands of every check of the current step, so a failed check's
    inputs and output can be saved after the step's results are read back."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.kept: list = []
        self.failed: list = []

    def _queue(self, name, where, node, A, B, C, bias, alpha, beta):
        n0 = len(self._pending)
        super()._queue(name, where, node, A, B, C, bias, alpha, beta)
        if len(self._pending) > n0:
            self.kept.append((A, B, C, bias, alpha, beta, name, where))

    def _flush(self):
        kept, self.kept = self.kept, []
        results = super()._flush()
        if len(results) == len(kept):           # (a check skipped for lack of memory breaks the pairing)
            for res, ops in zip(results, kept):
                if not res.ok:
                    self.failed.append((res, ops))
        return results


def _save_evidence(out_dir: str, n: int, res, ops, ident: dict, meta: dict) -> str:
    A, B, C, bias, alpha, beta, name, where = ops
    path = os.path.join(out_dir, f"evidence_{n}")
    torch.save({"A": A.detach().cpu(), "B": B.detach().cpu(), "C": C.detach().cpu(),
                "bias": None if bias is None else bias.detach().cpu(), "alpha": alpha, "beta": beta,
                "op": name, "where": where}, path + ".pt")
    info = {"what": "a matrix multiply whose output failed its checksum", "op": name, "where": where,
            "shape_A": list(A.shape), "shape_B": list(B.shape), "dtype": str(C.dtype).replace("torch.", ""),
            "max_z": res.max_z, "alarm_line": 6.0, "row": res.row, "col_block": res.col_block, "batch": res.batch,
            "nonfinite": res.nonfinite, "gpu": ident, **meta,
            "replay": f"sdcguard replay {path}.pt"}
    with open(path + ".json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    return path


def screen_one(device: torch.device, minutes: float, size: str, out_root: str, seed: int = 0,
               max_evidence: int = 5, simulate_faults: float = 0.0, max_steps: Optional[int] = None,
               tag: Optional[str] = None) -> dict:
    """Screen one device. Returns a summary dict (also written to out_root/<id>/summary.json).

    simulate_faults: for demos and tests only, corrupt this share of the training matmul outputs in
    software (NVIDIA's fault patterns), to see what a failing GPU's report and evidence look like."""
    L, d, h, seq, batch = SIZES[size]
    vocab = 32000 if size != "tiny" else 512
    ident = device_identity(device)
    idx = device.index if device.type == "cuda" else None
    if device.type == "cuda" and idx is None:
        idx = torch.cuda.current_device()
    smi_id = ident.get("gpu_uuid") or (ident.get("gpu_index") if idx is not None else None)
    ident.update(_nvidia_smi(smi_id))
    gid = ident.get("gpu_uuid") or f"{device.type}{tag or ident.get('gpu_index') or (idx if idx is not None else '')}"
    out_dir = os.path.join(out_root, str(gid))
    os.makedirs(out_dir, exist_ok=True)
    # bf16 where the GPU has bf16 tensor cores (Ampere and newer); fp16 on older GPUs (T4, RTX 20xx)
    dtype = torch.bfloat16
    if device.type == "cuda":
        with torch.cuda.device(device):
            try:
                native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:                      # older PyTorch
                native_bf16 = torch.cuda.get_device_capability()[0] >= 8
            if not native_bf16:
                dtype = torch.float16
    torch.manual_seed(seed)
    model = ScreenModel(vocab, L, d, h, seq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    serve_model = ScreenModel(vocab, L, d, h, seq).to(device)
    serve_model.load_state_dict(model.state_dict())
    serve_model = serve_model.to(dtype).eval()
    sguard = ServeGuard().attach(serve_model)
    # read results back every 24 checks, so only that many matmuls' operands are kept for evidence
    tguard = _Recorder(sample_rate=1.0, seed=seed, max_pending=24)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    t_end = time.time() + 60.0 * minutes
    steps = passes = 0
    flops = 0.0
    evidence, serve_alerts = [], []
    meta = {"sdcguard": __version__, "torch": torch.__version__, "seed": seed, "size": size,
            "dtype": str(dtype).replace("torch.", "")}
    inj = None
    if simulate_faults > 0:
        from .inject import FaultInjector, fault_model
        inj = FaultInjector(rate=simulate_faults, fault_model=fault_model("nvidia", "warp"), seed=seed + 3)
        meta["SIMULATED_FAULTS"] = f"yes: {simulate_faults} of matmul outputs corrupted in software (a demo, not a GPU fault)"
    t0 = time.time()
    counts = {"serve_nonfinite": 0}
    try:
        steps, passes, flops = _loop(model, opt, serve_model, sguard, tguard, inj, gen, device, dtype, vocab, batch,
                                     seq, L, d, t_end, max_steps, evidence, serve_alerts, max_evidence, out_dir,
                                     ident, meta, counts)
    finally:
        sguard.detach()      # the serving check changes process-wide cuBLAS settings; put them back
    seconds = time.time() - t0
    bad = tguard.stats["alerts"] + len(serve_alerts)
    unjudged = tguard.stats.get("unjudged", 0)
    # PASS needs judged checks: training checks that couldn't be judged, or serving passes whose output wasn't
    # finite (the serving check scores blocks it can't judge as 0), make a clean run INCOMPLETE
    result = verdict.from_counts(bad, tguard.stats["checked"], unjudged)
    if result == "PASS" and verdict.from_counts(0, passes, counts["serve_nonfinite"]) == "INCOMPLETE":
        result = "INCOMPLETE"
    summary = {"device": str(device), "gpu": ident, "result": result, "minutes": round(seconds / 60, 2),
               "training_steps": steps, "serving_passes": passes, "matmuls_checked": tguard.stats["checked"],
               "linear_layer_checks_while_serving": passes * sguard.stats["layers"],
               "approx_matmul_tflop": round(flops / 1e12, 1), "corrupted_results_training": tguard.stats["alerts"],
               "corrupted_results_serving": len(serve_alerts),
               "serving_alerts": [e.detail for e in serve_alerts[:20]],
               "worst_clean_z": round(tguard.max_z_seen, 3), "alarm_line": 6.0, "evidence": evidence,
               "unjudged_training_checks": unjudged,
               "serving_passes_with_nonfinite_output": counts["serve_nonfinite"],
               "out_dir": out_dir, **meta}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def _loop(model, opt, serve_model, sguard, tguard, inj, gen, device, dtype, vocab, batch, seq, L, d, t_end, max_steps,
          evidence, serve_alerts, max_evidence, out_dir, ident, meta, counts):
    steps = passes = 0
    flops = 0.0
    while True:
        x = torch.randint(0, vocab, (batch, seq), generator=gen).to(device)
        y = torch.randint(0, vocab, (batch, seq), generator=gen).to(device)
        with (inj if inj is not None else _Null()):
            with tguard.at_step(steps):
                with torch.autocast(device.type, dtype=dtype):
                    loss = F.cross_entropy(model(x).float().view(-1, vocab), y.view(-1))
                loss.backward()
        # evidence first: the failed matmuls' operands are saved before the optimizer step runs
        for res, ops in tguard.failed:
            if len(evidence) < max_evidence:
                evidence.append(_save_evidence(out_dir, len(evidence), res, ops, ident, {**meta, "step": steps,
                                                                                         "phase": "training"}))
        tguard.failed.clear()
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps += 1
        with torch.no_grad():
            served = serve_model(x)
        passes += 1
        serve_alerts += sguard.poll(wait=True)         # (waits for the GPU anyway)
        counts["serve_nonfinite"] += int(not bool(torch.isfinite(served).all()))
        flops += 3 * 2 * batch * seq * (12 * L * d * d + vocab * d) + 2 * batch * seq * (12 * L * d * d + vocab * d)
        if time.time() >= t_end or (max_steps is not None and steps >= max_steps):
            break
    return steps, passes, flops


def replay(path: str, times: int = 200, device: Optional[str] = None) -> dict:
    """Recompute a saved matmul `times` times on this device and check each result."""
    ev = torch.load(path, map_location="cpu")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if ev.get("kind") == "math_mode":           # from sdcguard check's math-mode check
        from .modes import replay as mode_replay
        r = mode_replay(ev, min(times, 50), dev)
        bad = r["recomputations_that_failed_the_check"]
        return {"evidence": path, "replayed_on": device_identity(dev), **r,
                "reading": (f"the {r['mode']} result came out wrong again on this device ({bad} of {r['recomputations']})"
                            if bad else f"every {r['mode']} recomputation on this device matched the CPU's answer")}
    A, B, C = ev["A"].to(dev), ev["B"].to(dev), ev["C"].to(dev)
    bias = ev["bias"].to(dev) if ev["bias"] is not None else None
    alpha, beta = ev["alpha"], ev["beta"]
    from .guard import _exact_reductions_off, _exact_reductions_on
    restore = _exact_reductions_on()          # the screen ran its checks this way
    try:
        return _replay(path, times, dev, A, B, C, bias, alpha, beta)
    finally:
        _exact_reductions_off(restore)


def _replay(path, times, dev, A, B, C, bias, alpha, beta) -> dict:
    recorded = check_matmul(A, B, C, bias, alpha=alpha, beta=beta)
    bad, unjudged, worst, first_diff = 0, 0, 0.0, None
    def recompute():
        """The same operation the model ran (so a correct GPU reproduces the recorded output except where
        it was corrupted, up to kernel choice on a different GPU model)."""
        with torch.no_grad():
            if A.dim() == 3 and B.dim() == 3:
                out = torch.bmm(A, B)
            elif A.dim() == 3 or B.dim() == 3:
                out = torch.matmul(A, B)
            elif bias is not None and beta != 0:
                return torch.addmm(bias.to(A.dtype), A, B, beta=beta, alpha=alpha).to(C.dtype)
            else:
                out = torch.mm(A, B)
            return (out * alpha if alpha != 1 else out).to(C.dtype)

    for i in range(times):
        out = recompute()
        r = check_matmul(A, B, out, bias, alpha=alpha, beta=beta)
        if r is not None and not r.ok:
            bad += 1
        elif r is None or not r.judged:
            unjudged += 1
            continue
        worst = max(worst, r.max_z if math.isfinite(r.max_z) else float("inf"))
        if first_diff is None:
            diff = (out.float() - C.float()).abs()
            scale = out.float().abs().clamp_min(1e-30)
            first_diff = {"elements_that_differ_from_recorded_output": int((diff > 0).sum()),
                          "elements_off_by_more_than_1_percent": int(((diff / scale) > 1e-2).sum()),
                          "largest_difference": float(torch.nan_to_num(diff, nan=float("inf")).max())}
    ident = device_identity(dev)
    rec_judged = recorded is not None and recorded.judged
    if bad:
        reading = ("the error came back on this device: it follows the matmul's inputs on this GPU too (a "
                   "replay doesn't by itself rule out a software or configuration cause)")
    elif unjudged == times:
        reading = "no recomputation could be judged (the inputs hold values that aren't finite): no evidence either way"
    else:
        reading = "every recomputation on this device that could be judged was correct" + (
            f" ({unjudged} couldn't be judged)" if unjudged else "")
    if recorded is not None and not rec_judged:
        reading += "; the recorded output itself couldn't be judged, so this file doesn't show a wrong result"
    return {"evidence": path, "replayed_on": ident, "recorded_output_check": None if recorded is None else
            {"max_z": recorded.max_z, "ok": recorded.ok, "judged": rec_judged, "row": recorded.row,
             "col_block": recorded.col_block},
            "recomputations": times, "recomputations_that_failed_the_check": bad,
            "recomputations_that_couldnt_be_judged": unjudged, "worst_z_of_recomputations": worst,
            "this_device_vs_recorded_output": first_diff, "reading": reading}


def _all_gpus(args) -> int:
    n = torch.cuda.device_count()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = [v.strip() for v in vis.split(",") if v.strip()][:n] if vis else [str(i) for i in range(n)]
    procs = []
    for i in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=ids[i] if i < len(ids) else str(i))
        cmd = [sys.executable, "-m", "sdcguard.screen", "--device", "cuda:0", "--minutes", str(args.minutes),
               "--size", args.size, "--out", args.out, "--seed", str(args.seed), "--quiet",
               "--simulate-faults", str(args.simulate_faults)]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    print(f"sdcguard screen: {n} GPUs, {sum(c == 0 for c in codes)} PASS, {sum(c == 1 for c in codes)} FAIL, "
          f"{sum(c not in (0, 1) for c in codes)} incomplete or could not run. Details: {args.out}/<GPU UUID>/summary.json")
    return 1 if any(c == 1 for c in codes) else (2 if any(c not in (0, 1) for c in codes) else 0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sdcguard.screen", description=__doc__.split("\n\n")[0])
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--device", default=None, help="e.g. cuda:0 (default: every visible GPU, or the CPU)")
    p.add_argument("--size", default=None, choices=sorted(SIZES), help="model size (default 1b on GPUs)")
    p.add_argument("--out", default="screen_out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replay", default=None, help="an evidence_N.pt file to recompute and check here")
    p.add_argument("--times", type=int, default=200)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--simulate-faults", type=float, default=0.0,
                   help="demo only: corrupt this share of matmul outputs in software")
    args = p.parse_args(argv)
    if args.replay:
        r = replay(args.replay, args.times, args.device)
        print(json.dumps(r, indent=2, default=str))
        return 1 if r["recomputations_that_failed_the_check"] else 0
    if args.size is None:
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            args.size = "1b" if gb >= 70 else ("small" if gb >= 20 else "tiny")
        else:
            args.size = "tiny"
    if args.device is None and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return _all_gpus(args)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        s = screen_one(dev, args.minutes, args.size, args.out, args.seed, simulate_faults=args.simulate_faults)
    except Exception as e:  # noqa: BLE001
        print(f"sdcguard screen: could not run on {dev}: {type(e).__name__}: {e}")
        return 2
    g = s["gpu"]
    if args.simulate_faults:
        print("  (SIMULATED faults were injected in software for this run: a demo, not a real GPU fault)")
    print(f"sdcguard screen: {s['result']}  {g.get('gpu_name', s['device'])} {g.get('gpu_uuid') or ''} "
          f"PCI {g.get('pci_bus_id')} serial {g.get('serial')}")
    print(f"  {s['minutes']} min, {s['training_steps']} training steps, {s['serving_passes']} serving passes, "
          f"{s['matmuls_checked']:,} matmuls checked in training, {s['linear_layer_checks_while_serving']:,} "
          f"layer checks while serving, about {s['approx_matmul_tflop']:,} TFLOP of matmul work")
    print(f"  corrupted results: {s['corrupted_results_training']} in training, {s['corrupted_results_serving']} while "
          f"serving; worst clean score {s['worst_clean_z']} (alarm line 6)")
    if s.get("unjudged_training_checks") or s.get("serving_passes_with_nonfinite_output"):
        print(f"  couldn't be judged: {s.get('unjudged_training_checks', 0)} training checks, "
              f"{s.get('serving_passes_with_nonfinite_output', 0)} serving passes with output that wasn't finite")
    for e in s["evidence"]:
        print(f"  evidence: {e}.json   replay: sdcguard replay {e}.pt")
    return {"FAIL": 1, "INCOMPLETE": 2}.get(s["result"], 0)


if __name__ == "__main__":
    sys.exit(main())
