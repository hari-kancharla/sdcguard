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
import uuid
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
            self.kept.append((self._pending[-1], (A, B, C, bias, alpha, beta, name, where, self._epoch)))

    def _flush(self, raise_alerts: bool = True):
        ops_of = {id(entry): ops for entry, ops in self.kept}
        self.kept = []
        results = super()._flush(raise_alerts=raise_alerts)
        # paired by queue entry, not by position: checks skipped at the read-back (out of memory) are
        # left out of the results, and the others must still get their own operands
        for entry, res in zip(self._flushed, results):
            if not res.ok and id(entry) in ops_of:
                self.failed.append((res, ops_of[id(entry)]))
        return results


def _block_slices(A, B, C, bias, row, col_block, batch, block):
    A, B, C = A.detach().cpu(), B.detach().cpu(), C.detach().cpu()
    # Bias follows the output's broadcast rules, including scalars, singleton
    # rows and a batch dimension (baddbmm). Slice it only after broadcasting.
    if bias is not None:
        bias = torch.broadcast_to(bias.detach().cpu(), C.shape)
    if A.dim() == 3:
        A = A[(batch or 0) if A.shape[0] > 1 else 0]
    if B.dim() == 3:
        B = B[(batch or 0) if B.shape[0] > 1 else 0]
    if C.dim() == 3:
        C = C[batch or 0]
        if bias is not None:
            bias = bias[batch or 0]
    lo, hi = col_block * block, min(C.shape[-1], (col_block + 1) * block)
    b = None
    if bias is not None:
        b = bias[row:row + 1, lo:hi]
    return A[row:row + 1], B[:, lo:hi], C[row:row + 1, lo:hi], b


def exact_block(a, b_blk, c, bias, alpha, beta) -> dict:
    """One row of one block against a float64 recomputation. An output counts as definitely wrong when it is
    outside the worst-case rounding envelope of an fp32-accumulating GPU, with a 2x margin: a confirmation that
    needs no noise model. Smaller errors (which the checksums' statistical test can still flag) aren't
    confirmed this way.

    Arithmetic assumed (what NVIDIA GPUs do for bf16/fp16 matmuls through cuBLAS): products exact in fp32 and
    at most k fp32 additions (the classic bound k x 2^-23 x sum|a_t b_t|); fp32 values below its normal range
    may be flushed to zero (up to 2^-126 per addition); one round-to-nearest conversion to the output dtype,
    which can round a tiny result to a subnormal or to zero (up to half the smallest subnormal step) and a
    result beyond the dtype's largest value to infinity; alpha, and beta x bias added in fp32. A kernel that
    accumulates in lower precision, or rounds more than once, can exceed this envelope without being wrong."""
    from .checksum import UNIT_ROUNDOFF
    k = a.shape[-1]
    ref = alpha * (a.double() @ b_blk.double())
    env = abs(alpha) * k * 2.0 ** -23 * (a.double().abs() @ b_blk.double().abs())
    if bias is not None and beta != 0:
        ref = ref + beta * bias.double()
        env = env + abs(beta) * bias.double().abs() * 2.0 ** -23
    u = UNIT_ROUNDOFF.get(c.dtype, 2.0 ** -24)
    fi = torch.finfo(c.dtype) if c.dtype.is_floating_point else torch.finfo(torch.float32)
    # underflow: half the smallest subnormal step of the output, plus fp32 flush-to-zero on each addition
    # (an fp16 output of 1e-4 x 1e-4 is a correct 0; Oct 19 outside review)
    tiny = 0.5 * fi.tiny * fi.eps + (k + 1) * torch.finfo(torch.float32).tiny
    tol = 2 * (u * ref.abs() + env + tiny) + 1e-300
    cd = c.double()
    diff = (cd - ref).abs()
    # overflow: an infinite output of the right sign is correct when the exact result is at or beyond the
    # dtype's largest value, within the envelope
    overflow_ok = torch.isinf(cd) & (torch.sign(cd) == torch.sign(ref)) & (ref.abs() + tol >= fi.max)
    bad = (~torch.isfinite(cd) & torch.isfinite(ref) & ~overflow_ok) | (torch.isfinite(cd) & (diff > tol))
    ratio = torch.where(torch.isfinite(diff), diff / tol, torch.full_like(diff, float("inf")))
    ratio = torch.where(overflow_ok, torch.zeros_like(ratio), ratio)
    return {"outputs_outside_rounding_envelope": int(bad.sum()), "outputs_in_block": int(c.numel()),
            "worst_error_over_envelope": float(ratio.max()) if ratio.numel() else 0.0}


def cpu_recheck(A, B, C, bias, alpha, beta, row, col_block, batch=None, epoch: int = 0, block: int = 128,
                seed: Optional[int] = None, z: float = 6.0, abs_floor: Optional[float] = None,
                nfma: Optional[int] = None) -> dict:
    """The flagged block judged again on the CPU from the saved operands, with the same checksums (same sign
    patterns, block width, alarm line and noise model, fp32 accumulators included) as the GPU check, plus
    exact_block's rounding envelope. It separates an output outside the envelope, a statistical mismatch, and a
    mismatch the CPU doesn't reproduce; the last one can come from the check's own GPU arithmetic (on Oct 19,
    2026 an H200 with the cuBLASLt MPS bug alerted while the outputs it checked matched the CPU, consistent with
    the checksum matmul itself being hit), from numerical differences, or from a checker error. seed: the sign
    patterns of the process that ran the check (default: this process's)."""
    if row is None or col_block is None:
        return {"done": False, "why": "the check didn't record which block failed"}
    try:
        from .checksum import ABS_FLOOR, acc_supported, matmul_zscores, matmul_zscores_acc, resolve, sign_seed
        floor = ABS_FLOOR if abs_floor is None else abs_floor
        a, b_blk, c, b = _block_slices(A, B, C, bias, row, col_block, batch, block)
        with sign_seed(seed):
            if acc_supported(a, c):          # precise mode's evidence: the fp32 accumulator of bf16/fp16 inputs
                st = matmul_zscores_acc(a, b_blk, c, b, alpha=alpha, beta=beta, block=block, abs_floor=floor, z=z,
                                        nfma=nfma, epoch=epoch)
            else:
                st = matmul_zscores(a, b_blk, c, b, alpha=alpha, beta=beta, block=block, abs_floor=floor, z=z,
                                    epoch=epoch)
            r = None if st is None else resolve([st], z)[0]
        exact = exact_block(a, b_blk, c, b, alpha, beta)
    except Exception as e:  # noqa: BLE001
        return {"done": False, "why": error_text(e)}
    out = {"done": True, **exact}
    confirmed = exact["outputs_outside_rounding_envelope"] > 0
    if r is None or not r.judged:
        if confirmed:                        # the envelope needs no checksum: it decides on its own
            return {**out, "judged": True, "output_wrong": True, "output_mismatch_confirmed": True,
                    "reading": "the CPU recomputation places the recorded output outside the stated rounding "
                               "envelope"}
        why = (f"the CPU has no checksum for a {str(c.dtype).replace('torch.', '')} output of "
               f"{str(a.dtype).replace('torch.', '')} inputs" if r is None else
               "values that aren't finite in the inputs")
        return {**out, "judged": False, "reading": f"the CPU couldn't judge this block either ({why})"}
    wrong = not r.ok
    if confirmed:
        reading = "the CPU recomputation places the recorded output outside the stated rounding envelope"
    elif wrong:
        reading = ("the CPU checksum also flags this output; its error remains within the deterministic rounding "
                   "envelope, so this is a statistical mismatch rather than an independently confirmed fault")
    else:
        reading = ("the CPU did not reproduce the checksum mismatch; the check's own GPU arithmetic, differences "
                   "in numerical execution, or a checker error may explain it. This does not prove the output "
                   "correct or identify a hardware fault")
    return {**out, "judged": True, "max_z": r.max_z, "output_wrong": wrong or confirmed,
            "output_mismatch_confirmed": confirmed, "reading": reading}


def _claim(out_dir: str, prefix: str, n: int) -> str:
    """The first free <prefix>_<n> name, claimed atomically: ranks or processes sharing one evidence folder (every
    rank of a torchrun job started in the same directory) never write over each other's evidence."""
    while True:
        path = os.path.join(out_dir, f"{prefix}_{n}")
        try:
            fd = os.open(path + ".pt", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            n += 1
            continue
        os.close(fd)
        return path


def _write_atomic(path: str, write) -> None:
    """Write through a temporary file and rename it into place: a reader never sees half a file."""
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _save_evidence(out_dir: str, n: int, res, ops, ident: dict, meta: dict) -> str:
    A, B, C, bias, alpha, beta, name, where = ops[:8]
    epoch = ops[8] if len(ops) > 8 else 0
    from .checksum import _SIGN_SEED
    meta = dict(meta)
    # the GEMM settings in force when the checked matmul ran (the guard reads them during the step; its read-back,
    # and this save, come after the step put them back), else as they are now
    gemm = meta.pop("gemm_settings", None)
    # how the check judged it (the guard's block width, alarm line and floor; tensor-core steps for an fp32
    # accumulator): the CPU recheck and every replay judge it the same way
    cfg = {"block": 128, "z": 6.0, "abs_floor": None, "nfma": None, **(meta.pop("check_config", None) or {})}
    alarm = cfg["z"]
    # the GPU it ran on (with evidence_copy the operands are already host copies)
    device = meta.pop("record_device", None) or C.device
    in_job = meta.pop("in_job_replay", None)
    software = _software_for_evidence(device, gemm)
    # The stress phase finishes before the workload starts. Both used evidence_0,
    # so a later training fault silently replaced the saved stress incident.
    prefix = "evidence_stress" if meta.get("phase") == "stress" else "evidence"
    path = _claim(out_dir, prefix, n)
    payload = {"A": A.detach().cpu(), "B": B.detach().cpu(), "C": C.detach().cpu(),
               "bias": None if bias is None else bias.detach().cpu(), "alpha": alpha, "beta": beta,
               "op": name, "where": where, "row": res.row, "col_block": res.col_block, "batch": res.batch,
               "epoch": epoch, "sign_seed": _SIGN_SEED, "check_config": cfg,
               "gemm_settings": software.get("matmul") if isinstance(software.get("matmul"), dict) else None}
    _write_atomic(path + ".pt", lambda t: torch.save(payload, t))
    from .casefile import _sha256
    info = {"what": "a matrix multiply whose output failed its checksum", "op": name, "where": where,
            "evidence_sha256": _sha256(path + ".pt"),        # this .json describes that .pt and no other
            "shape_A": list(A.shape), "shape_B": list(B.shape), "dtype": str(C.dtype).replace("torch.", ""),
            "max_z": res.max_z, "alarm_line": alarm, "check_config": cfg, "row": res.row,
            "col_block": res.col_block, "batch": res.batch, "nonfinite": res.nonfinite,
            "cpu_recheck": cpu_recheck(A, B, C, bias, alpha, beta, res.row, res.col_block, res.batch, epoch,
                                       block=cfg["block"], z=cfg["z"], abs_floor=cfg["abs_floor"], nfma=cfg["nfma"]),
            "sign_epoch": epoch,
            "gpu": ident, "software": software, **meta,
            "replay": f"sdcguard replay {path}.pt"}
    if in_job:
        # the guard's recomputations right after the alert (SDCGuard(confirm=...)): a replay on the same GPU, in
        # the same process, with the step's settings; the casefile counts it with the others
        info["in_job_replays"] = [{**{k: v for k, v in in_job.items() if k not in ("evidence", "replayed_on")},
                                   "evidence": path + ".pt", "evidence_sha256": info["evidence_sha256"],
                                   "replay_id": uuid.uuid4().hex, "replayed_on": ident, "software": software,
                                   "sign_patterns": "recorded", "in_job": True}]

    def dump(t):
        with open(t, "w") as f:
            json.dump(info, f, indent=2, default=str)
    _write_atomic(path + ".json", dump)
    return path


EXECUTION_ENV = ("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE")


def execution_settings(device=None) -> dict:
    """The settings that change which GEMM kernels run, besides the library versions: the MPS share, the SMs this
    process can use, cuBLAS's workspace and TF32 overrides (cuBLAS known issue 6862950 depends on the first two),
    and the cuBLASLt this process actually loaded. Recorded with evidence and with every replay, together with
    PyTorch's matmul flags (check.gemm_settings), so a peer counts as a matched control only when these match."""
    env = {k: os.environ[k] for k in EXECUTION_ENV if k in os.environ}
    out: dict = {"environment": env}
    try:
        dev = torch.device(device) if device is not None else None
        if dev is not None and dev.type == "cuda":
            out["sms_visible"] = torch.cuda.get_device_properties(dev).multi_processor_count
            out["compute_capability"] = list(torch.cuda.get_device_capability(dev))
            from .blas import _library
            lib = _library()
            if lib is not None:
                v = int(lib.cublasLtGetVersion())
                out["cublaslt_loaded"] = f"{v // 10000}.{(v // 100) % 100}.{v % 100}"
    except Exception:  # noqa: BLE001
        pass
    return out


def _software_for_evidence(device=None, matmul: Optional[dict] = None) -> dict:
    """The software stack an evidence file was recorded with (casefile matches peers against it). `matmul`: the
    GEMM settings read when the matmul ran, when the caller has them."""
    try:
        from .check import software
        sw = dict(software())
        if isinstance(matmul, dict):
            sw["matmul"] = matmul
        sw.update(execution_settings(device))
        return sw
    except Exception as e:  # noqa: BLE001  (never lose evidence over this)
        return {"error": error_text(e)}


def device_dir(device: torch.device, out_root: str, tag: Optional[str] = None) -> tuple:
    """(out_root/<GPU UUID>, the GPU's identity): where one GPU's results and evidence go (created)."""
    ident = device_identity(device)
    idx = device.index if device.type == "cuda" else None
    if device.type == "cuda" and idx is None:
        idx = torch.cuda.current_device()
    smi_id = ident.get("gpu_uuid") or (ident.get("gpu_index") if idx is not None else None)
    ident.update(_nvidia_smi(smi_id))
    gid = ident.get("gpu_uuid") or f"{device.type}{tag or ident.get('gpu_index') or (idx if idx is not None else '')}"
    out_dir = os.path.join(out_root, _safe_name(str(gid)))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, ident


def _safe_name(s: str) -> str:
    """A string usable as one path component (a UUID read from a device or a file can't escape the folder)."""
    s = "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip(".")
    return s or "gpu"


def error_text(e: BaseException) -> str:
    msg = str(e).strip().splitlines()[0][:300] if str(e).strip() else ""
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def screen_one(device: torch.device, minutes: float, size: str, out_root: str, seed: int = 0,
               max_evidence: int = 5, simulate_faults: float = 0.0, max_steps: Optional[int] = None,
               tag: Optional[str] = None, run_id: Optional[str] = None) -> dict:
    """Screen one device. Returns a summary dict (also written to out_root/<id>/summary.json).

    simulate_faults: for demos and tests only, corrupt this share of the training matmul outputs in
    software (NVIDIA's fault patterns), to see what a failing GPU's report and evidence look like."""
    L, d, h, seq, batch = SIZES[size]
    vocab = 32000 if size != "tiny" else 512
    out_dir, ident = device_dir(device, out_root, tag)
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
            "run_id": run_id or uuid.uuid4().hex,
            "dtype": str(dtype).replace("torch.", "")}
    inj = None
    if simulate_faults > 0:
        from .inject import FaultInjector, fault_model
        inj = FaultInjector(rate=simulate_faults, fault_model=fault_model("nvidia", "warp"), seed=seed + 3)
        meta["SIMULATED_FAULTS"] = f"yes: {simulate_faults} of matmul outputs corrupted in software (a demo, not a GPU fault)"
    t0 = time.time()
    counts = {"serve_nonfinite": 0, "steps": 0, "passes": 0, "flops": 0.0}
    error = None
    # until this run finishes, its folder says so: a screen stopped at its deadline (a hung GPU) or killed must
    # not leave the previous run's PASS for the fleet report to read
    _write_summary(out_dir, {"device": str(device), "gpu": ident, "result": "INCOMPLETE", "running": True,
                             "error": "this screen hadn't finished (still running, or stopped before it could "
                                      "report)", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "matmuls_checked": 0, "out_dir": out_dir, **meta})
    try:
        _loop(model, opt, serve_model, sguard, tguard, inj, gen, device, dtype, vocab, batch, seq, L, d, t_end,
              max_steps, evidence, serve_alerts, max_evidence, out_dir, ident, meta, counts)
    except Exception as e:  # noqa: BLE001
        # A failing GPU often computes wrong results and then raises a CUDA error: what was found before the
        # error (alerts, saved evidence) still counts, and the run is reported, never dropped
        error = error_text(e)
        for res, ops in tguard.failed:               # the step that raised still had its checks read back
            if len(evidence) < max_evidence:
                try:
                    evidence.append(_save_evidence(out_dir, len(evidence), res, ops, ident,
                                                   {**meta, "step": counts["steps"], "phase": "training"}))
                except Exception:  # noqa: BLE001  (the alert is counted either way)
                    pass
        tguard.failed.clear()
    finally:
        serving = None
        try:                 # what the serving checks actually did, counted on the GPU (before detach drops it)
            serve_alerts += sguard.poll(wait=True)       # the last read can hold an alert too
            serving = sguard.coverage_report()
        except Exception as e:  # noqa: BLE001
            error = error or error_text(e)
        sguard.detach()      # the serving check changes process-wide cuBLAS settings; put them back
        # detach() reads the GPU once more: an alert only that read got (the polls above raised) is in the
        # guard's log, and counts like the others
        seen = {id(ev) for ev in serve_alerts}
        serve_alerts += [ev for ev in sguard.log.events if id(ev) not in seen]
    steps, passes, flops = counts["steps"], counts["passes"], counts["flops"]
    seconds = time.time() - t0
    serve_bad = max(len(serve_alerts), int(sguard.stats.get("alerts", 0)))
    bad = tguard.stats["alerts"] + serve_bad
    unjudged = tguard.stats.get("unjudged", 0)
    # checks skipped for lack of memory or for an unsupported dtype are checks that didn't happen
    skipped = tguard.stats.get("skipped_oom", 0) + tguard.stats.get("skipped_dtype", 0)
    # PASS needs judged checks: training checks that couldn't be judged or were skipped, serving passes whose
    # output wasn't finite (the serving check scores blocks it can't judge as 0), or a run stopped by an error
    # make a clean run INCOMPLETE
    result = verdict.from_counts(bad, tguard.stats["checked"], unjudged + skipped)
    if result == "PASS" and (error or verdict.from_counts(0, passes, counts["serve_nonfinite"]) == "INCOMPLETE"):
        result = "INCOMPLETE"
    # serving: every attached layer must have run its check, and every checksum must have been judged
    serve_gap = passes > 0 and (serving is None or bool(serving["never_checked"]) or serving["unjudged_checksums"] > 0)
    if result == "PASS" and serve_gap:
        result = "INCOMPLETE"
    summary = {"device": str(device), "gpu": ident, "result": result, "minutes": round(seconds / 60, 2),
               "training_steps": steps, "serving_passes": passes, "matmuls_checked": tguard.stats["checked"],
               "serving_linear_layers": None if serving is None else serving["layers"],
               "linear_layer_checks_while_serving": None if serving is None else serving["checks_executed"],
               "serving_layers_never_checked": None if serving is None else serving["never_checked"],
               "unjudged_serving_checksums": None if serving is None else serving["unjudged_checksums"],
               "approx_matmul_tflop": round(flops / 1e12, 1), "corrupted_results_training": tguard.stats["alerts"],
               "corrupted_results_serving": serve_bad,          # (the count the verdict above is made from)
               "serving_alerts": [e.detail for e in serve_alerts[:20]],
               "worst_clean_z": round(tguard.max_z_seen, 3), "alarm_line": 6.0, "evidence": evidence,
               "unjudged_training_checks": unjudged, "skipped_training_checks": skipped,
               "serving_passes_with_nonfinite_output": counts["serve_nonfinite"],
               "error": error, "out_dir": out_dir, **meta}
    _write_summary(out_dir, summary)
    return summary


def _write_summary(out_dir: str, summary: dict) -> None:
    def dump(t):
        with open(t, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    _write_atomic(os.path.join(out_dir, "summary.json"), dump)


class _Flag:
    """The parts of a check result evidence needs, from a serving alert."""
    def __init__(self, d: dict):
        self.max_z, self.nonfinite = d.get("max_z"), bool(d.get("nonfinite"))
        self.row, self.col_block, self.batch = d.get("row"), d.get("col_block"), None


def _serving_evidence(serve_model, x, alerts, out_dir, n0, room, ident, meta) -> list:
    """Evidence for serving alerts the check captured no row for (its buffer was full). The flagged layer's input
    and output are captured from one more pass over the same tokens. A fault that recurs is in it; a transient one
    isn't, and a weight corrupted in memory makes the recaptured output agree with the corrupted weight. So the
    file is marked as recaptured, and the casefile never reads a clean recapture as "the check was wrong"."""
    mods = dict(serve_model.named_modules())
    paths = []
    for ev in alerts:
        if len(paths) >= room:
            break
        name = ev.detail.get("where")
        mod = mods.get(name)
        if mod is None or not isinstance(getattr(mod, "weight", None), torch.Tensor):
            continue
        box = {}

        def keep(m, inp, out):
            box["x"], box["y"] = inp[0].detach(), out.detach()
        h = mod.register_forward_hook(keep)
        try:
            with torch.no_grad():
                serve_model(x)
        except Exception:  # noqa: BLE001  (a failing GPU may raise: the alert itself still counts)
            continue
        finally:
            h.remove()
        if "x" not in box:
            continue
        W, b = mod.weight.detach(), getattr(mod, "bias", None)
        A, C = box["x"].reshape(-1, W.shape[1]), box["y"].reshape(-1, W.shape[0])
        try:
            paths.append(_save_evidence(out_dir, n0 + len(paths), _Flag(ev.detail),
                                        (A, W.t(), C, None if b is None else b.detach(), 1.0, 1.0,
                                         "aten::addmm" if b is not None else "aten::mm", f"{name} (serving)"),
                                        ident, {**meta, "phase": "serving",
                                                "captured": "from a second pass over the same tokens right after "
                                                            "the alert; a transient fault isn't in it",
                                                "weights_changed_since_load":
                                                    ev.detail.get("weights_changed_since_load"),
                                                "alert": {k: ev.detail.get(k) for k in
                                                          ("max_z", "row", "col_block", "checksum", "reading")}}))
        except Exception:  # noqa: BLE001
            continue
    return paths


def _loop(model, opt, serve_model, sguard, tguard, inj, gen, device, dtype, vocab, batch, seq, L, d, t_end, max_steps,
          evidence, serve_alerts, max_evidence, out_dir, ident, meta, counts):
    """The workload. Progress goes into `counts` as it happens (steps, passes, flops), so a run that stops on
    an error still reports what it did."""
    steps = passes = 0
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
        counts["steps"] = steps
        counts["flops"] += 3 * 2 * batch * seq * (12 * L * d * d + vocab * d)
        with torch.no_grad():
            served = serve_model(x)
        # the serving check keeps the row that alarmed (serve.CAPTURE_SLOTS): saved as evidence by the guard, with
        # this run's tags, within the room left
        sguard.evidence_dir, sguard.evidence_meta = out_dir, {**meta, "step": steps}
        sguard.max_evidence = len(sguard.evidence) + max(0, max_evidence - len(evidence))
        new_alerts = sguard.poll(wait=True)            # (waits for the GPU anyway)
        serve_alerts += new_alerts
        saved = [a.detail["evidence"] for a in new_alerts if isinstance(a.detail.get("evidence"), str)
                 and not a.detail["evidence"].startswith("not saved")]
        evidence += saved
        rest = [a for a in new_alerts if a.detail.get("evidence") not in saved]
        if rest and len(evidence) < max_evidence:
            evidence += _serving_evidence(serve_model, x, rest, out_dir, len(evidence),
                                          max_evidence - len(evidence), ident, {**meta, "step": steps})
        passes += 1
        counts["passes"] = passes
        counts["serve_nonfinite"] += int(not bool(torch.isfinite(served).all()))
        counts["flops"] += 2 * batch * seq * (12 * L * d * d + vocab * d)
        if time.time() >= t_end or (max_steps is not None and steps >= max_steps):
            break


def _evidence_sha256(path: str) -> Optional[str]:
    from .casefile import _sha256
    return _sha256(path)


def _recorded_gemm_settings(path: str, ev: dict) -> Optional[dict]:
    """The GEMM settings the evidence was recorded with: in the .pt since Oct 21, 2026, else its .json."""
    rec = ev.get("gemm_settings")
    if isinstance(rec, dict) and rec:
        return rec
    info_path = path[:-3] + ".json" if path.endswith(".pt") else path + ".json"
    try:
        with open(info_path) as f:
            mm = (json.load(f).get("software") or {}).get("matmul")
        return mm if isinstance(mm, dict) and mm else None
    except (OSError, ValueError, AttributeError):
        return None


def _replay_record(path: str, digest: Optional[str], dev, out: dict, software: Optional[dict]) -> dict:
    """What every replay result carries: which evidence file it replayed (its SHA-256: a file renamed on the way to
    a peer is still the same file, and a replay of another file can't be counted for this one), a replay id (the
    same result passed twice counts once), when it ran, and the software and settings it ran with."""
    from . import __version__
    rec = {"evidence": path, "evidence_sha256": digest, "replay_id": uuid.uuid4().hex,
           "replayed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "replayed_on": device_identity(dev),
           **{k: v for k, v in out.items() if k not in ("evidence", "replayed_on")}}
    if software is not None:
        rec["software"] = software
    rec.setdefault("software", {"sdcguard": __version__})
    return rec


def _software_here(dev) -> Optional[dict]:
    try:
        from .casefile import software_stack
        return software_stack(dev)
    except Exception as e:  # noqa: BLE001
        return {"error": error_text(e)}


def replay(path: str, times: int = 200, device: Optional[str] = None) -> dict:
    """Recompute a saved matmul `times` times on this device and check each result. The recomputation runs with
    the GEMM settings recorded with the evidence (TF32, reduced-precision and split-K reductions, fp16
    accumulation, BLAS library), so it runs the same kernels where the GPU, the libraries and the environment
    (MPS share, cuBLAS variables) match too; the result records what it ran with, for the casefile to compare."""
    if isinstance(times, bool) or not isinstance(times, int) or times < 1:
        raise ValueError("--times must be at least 1")
    digest = _evidence_sha256(path)
    ev = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ev, dict):
        raise ValueError(f"{path} is not an sdcguard evidence file")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if ev.get("kind") == "math_mode":           # from sdcguard check's math-mode check
        from .modes import ProbeError, available, replay as mode_replay
        try:
            why = available(ev["mode"], dev)
        except ProbeError as e:
            why = str(e)
        if why:
            return _replay_record(path, digest, dev, {
                "mode": ev["mode"], "recomputations": 0, "recomputations_that_failed_the_check": 0,
                "reading": f"this device can't run the {ev['mode']} math mode ({why}): replay it on one that can"},
                _software_here(dev))
        r = mode_replay(ev, min(times, 50), dev)
        bad = r["recomputations_that_failed_the_check"]
        return _replay_record(path, digest, dev, {
            **r, "reading": (f"the {r['mode']} result came out wrong again on this device ({bad} of "
                             f"{r['recomputations']})" if bad else
                             f"every {r['mode']} recomputation on this device matched the CPU's answer")},
            _software_here(dev))
    A, B, C = ev["A"].to(dev), ev["B"].to(dev), ev["C"].to(dev)
    bias = ev["bias"].to(dev) if ev["bias"] is not None else None
    alpha, beta = ev["alpha"], ev["beta"]
    from .check import gemm_settings_applied
    from .checksum import sign_seed
    from .guard import _exact_reductions_off, _exact_reductions_on
    seed, epoch = ev.get("sign_seed"), int(ev.get("epoch") or 0)
    recorded = _recorded_gemm_settings(path, ev)
    # with recorded settings, the recomputation runs as the original did; without (an older file), the way the
    # screen and the guard's checked steps run by default (exact bf16/fp16 reductions)
    restore = None if recorded else _exact_reductions_on()
    try:
        with gemm_settings_applied(recorded) as applied:
            # judged with the sign patterns of the check that flagged the output: a false alarm tied to those
            # patterns comes back on a correct recomputation, instead of vanishing under this process's patterns
            with sign_seed(seed):
                out = _replay(path, times, dev, A, B, C, bias, alpha, beta, epoch, ev.get("check_config"))
            software = _software_here(dev)         # read inside: the settings the recomputations ran with
    finally:
        _exact_reductions_off(restore)
    out = _replay_record(path, digest, dev, out, software)
    out["gemm_settings"] = ({"source": "recorded with the evidence", "applied": sorted(applied)} if recorded else
                            {"source": "this process's: the evidence file doesn't record them (an older file)"})
    out["sign_patterns"] = "recorded" if seed is not None else "this process's"
    out["sign_epoch"] = epoch
    if seed is None:
        out["reading"] += ("; this evidence file doesn't record the check's sign patterns, so the replays were "
                           "judged with other patterns than the alert")
    if ev.get("row") is not None and ev.get("col_block") is not None:
        # the recorded output's flagged block, judged with the patterns of the check that flagged it
        cfg = ev.get("check_config") or {}
        rc = cpu_recheck(ev["A"], ev["B"], ev["C"], ev["bias"], alpha, beta, ev["row"], ev["col_block"],
                         ev.get("batch"), ev.get("epoch") or 0, block=cfg.get("block") or 128,
                         seed=ev.get("sign_seed"), z=cfg.get("z") or 6.0, abs_floor=cfg.get("abs_floor"),
                         nfma=cfg.get("nfma"))
        out["recorded_block_on_cpu"] = rc
        if rc.get("judged") and not rc.get("output_wrong"):
            out["reading"] += ("; the CPU did not reproduce the recorded checksum mismatch. The check's own "
                               "arithmetic, numerical execution differences or a checker error remain possible; "
                               "this result alone does not identify the cause")
    elif out.get("recorded_output_check") and out["recorded_output_check"]["judged"] and \
            out["recorded_output_check"]["ok"]:
        out["reading"] += ("; the recorded output passes the check here (an older evidence file, without the "
                           "check's sign patterns): this file doesn't show a wrong output by itself")
    return out


def _replay(path, times, dev, A, B, C, bias, alpha, beta, epoch: int = 0, cfg: Optional[dict] = None) -> dict:
    from .checksum import ABS_FLOOR, acc_supported, check_matmul_acc
    cfg = cfg or {}
    block, z = int(cfg.get("block") or 128), float(cfg.get("z") or 6.0)
    floor = ABS_FLOOR if cfg.get("abs_floor") is None else cfg["abs_floor"]
    acc = acc_supported(A, C)                 # precise mode: C is the fp32 accumulator of bf16/fp16 inputs
    nfma = None
    if acc and dev.type == "cuda":
        from .modes import n_fma
        nfma = n_fma("bf16" if A.dtype == torch.bfloat16 else "fp16", torch.cuda.get_device_capability(dev))

    def judge(out):
        if acc:
            return check_matmul_acc(A, B, out, bias, alpha=alpha, beta=beta, block=block, z=z, abs_floor=floor,
                                    nfma=nfma, epoch=epoch)
        return check_matmul(A, B, out, bias, alpha=alpha, beta=beta, block=block, z=z, abs_floor=floor, epoch=epoch)

    if acc:
        recorded = check_matmul_acc(A, B, C, bias, alpha=alpha, beta=beta, block=block, z=z, abs_floor=floor,
                                    nfma=cfg.get("nfma"), epoch=epoch)
    else:
        recorded = judge(C)
    bad, unjudged, worst, first_diff = 0, 0, 0.0, None
    def recompute():
        """The same operation the model ran (so a correct GPU reproduces the recorded output except where
        it was corrupted, up to kernel choice on a different GPU model)."""
        with torch.no_grad():
            if acc and A.dim() == 2 and B.dim() == 2:
                from .guard import _acc_mm
                out = _acc_mm(A, B)
                out = out * alpha if alpha != 1 else out
                if bias is not None and beta != 0:
                    out = out + beta * bias.float()
                return out
            if A.dim() == 3 and B.dim() == 3:
                if bias is not None and beta != 0:
                    return torch.baddbmm(bias.to(A.dtype), A, B, beta=beta, alpha=alpha).to(C.dtype)
                out = torch.bmm(A, B)
            elif A.dim() == 3 or B.dim() == 3:
                out = torch.matmul(A, B)
            elif bias is not None and beta != 0:
                return torch.addmm(bias.to(A.dtype), A, B, beta=beta, alpha=alpha).to(C.dtype)
            else:
                out = torch.mm(A, B)
            out = out * alpha if alpha != 1 else out
            if bias is not None and beta != 0:
                out = out + beta * bias
            return out.to(C.dtype)

    for i in range(times):
        out = recompute()
        r = judge(out)
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


def _uuid_of(gid: str, order: Optional[str]) -> Optional[str]:
    """The UUID of the GPU a child gets with CUDA_VISIBLE_DEVICES=gid, without touching CUDA here (a hung GPU
    can hang this process too): given as a UUID, or an index nvidia-smi numbers the same way (PCI bus order)."""
    if gid.startswith("GPU-"):
        return gid
    if order != "PCI_BUS_ID" or not gid.isdigit():
        return None
    try:
        out = subprocess.run(["nvidia-smi", f"--id={gid}", "--query-gpu=uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return out[-1].strip() if out and out[-1].strip().startswith("GPU-") else None
    except Exception:  # noqa: BLE001
        return None


def _not_finished(out_root: str, uuid: Optional[str], why: str, extra: Optional[dict] = None) -> None:
    """Mark a GPU's folder INCOMPLETE (the child's own result replaces it when it finishes)."""
    if not uuid:
        return
    d = os.path.join(out_root, _safe_name(uuid))
    try:
        os.makedirs(d, exist_ok=True)
        _write_summary(d, {"gpu": {"gpu_uuid": uuid}, "result": "INCOMPLETE", "running": False, "error": why,
                           "matmuls_checked": 0, "out_dir": d,
                           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})})
    except OSError:
        pass


def _all_gpus(args) -> int:
    n = torch.cuda.device_count()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = [v.strip() for v in vis.split(",") if v.strip()][:n] if vis else [str(i) for i in range(n)]
    # with no CUDA_VISIBLE_DEVICES, the children count GPUs in PCI bus order, as nvidia-smi does: then each
    # child's GPU is known here by its UUID, and a child that hangs before it reports (a classic sign of a bad
    # GPU) can't leave the previous run's PASS in its GPU's folder
    order = os.environ.get("CUDA_DEVICE_ORDER") if vis else "PCI_BUS_ID"
    procs, uuids = [], []
    for i in range(n):
        gid = ids[i] if i < len(ids) else str(i)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gid)
        if not vis:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        uuids.append(_uuid_of(gid, order))
        _not_finished(args.out, uuids[-1], "this screen hadn't finished (still running, or stopped before it "
                                           "could report)", {"running": True})
        cmd = [sys.executable, "-m", "sdcguard.screen", "--device", "cuda:0", "--minutes", str(args.minutes),
               "--size", args.size, "--out", args.out, "--seed", str(args.seed), "--quiet",
               "--simulate-faults", str(args.simulate_faults)]
        procs.append(subprocess.Popen(cmd, env=env))
    # a hung GPU mustn't hold the others (and the node) forever: past its budget plus room for loading, a
    # child is stopped and its GPU counts as not screened (exit 2)
    deadline = time.time() + 60.0 * args.minutes * 1.5 + 600.0
    codes = []
    for p, uuid in zip(procs, uuids):
        try:
            codes.append(p.wait(timeout=max(1.0, deadline - time.time())))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            codes.append(2)
            _not_finished(args.out, uuid, "this GPU's screen didn't finish in time and was stopped (a hung GPU, or "
                                          "a hung driver call): not screened")
            print(f"sdcguard screen: a GPU's screen didn't finish in time and was stopped (pid {p.pid})")
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
        try:
            r = replay(args.replay, args.times, args.device)
        except Exception as e:  # noqa: BLE001
            print(f"sdcguard replay: could not run: {error_text(e)}")
            return 2
        print(json.dumps(r, indent=2, default=str))
        return verdict.replay_exit(r)
    if args.size is None:
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            args.size = "1b" if gb >= 70 else ("small" if gb >= 20 else "tiny")
        else:
            args.size = "tiny"
    if args.device is None and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return _all_gpus(args)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:                                 # (a device name torch doesn't know is "couldn't run", 2, not a FAIL)
        dev = torch.device(dev)
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
          f"{s['matmuls_checked']:,} matmuls checked in training, {s['linear_layer_checks_while_serving'] or 0:,} "
          f"layer checks while serving, about {s['approx_matmul_tflop']:,} TFLOP of matmul work")
    print(f"  corrupted results: {s['corrupted_results_training']} in training, {s['corrupted_results_serving']} while "
          f"serving; worst clean score {s['worst_clean_z']} (alarm line 6)")
    if s.get("error"):
        print(f"  stopped by an error: {s['error']} (what was found before it is reported here)")
    if s.get("skipped_training_checks"):
        print(f"  {s['skipped_training_checks']:,} training checks skipped (not enough GPU memory for the check)")
    if s.get("unjudged_training_checks") or s.get("serving_passes_with_nonfinite_output") or \
            s.get("unjudged_serving_checksums"):
        print(f"  couldn't be judged: {s.get('unjudged_training_checks', 0)} training checks, "
              f"{s.get('unjudged_serving_checksums') or 0} serving checksums, "
              f"{s.get('serving_passes_with_nonfinite_output', 0)} serving passes with output that wasn't finite")
    if s.get("serving_layers_never_checked"):
        never = s["serving_layers_never_checked"]
        print(f"  {len(never)} serving layers never ran their check: {', '.join(never[:5])}"
              f"{' ...' if len(never) > 5 else ''}")
    for e in s["evidence"]:
        print(f"  evidence: {e}.json   replay: sdcguard replay {e}.pt")
    return {"FAIL": 1, "INCOMPLETE": 2}.get(s["result"], 0)


if __name__ == "__main__":
    sys.exit(main())
