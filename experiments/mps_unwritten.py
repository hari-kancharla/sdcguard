"""NVIDIA's MPS cuBLASLt bug on a real GPU: does sdcguard catch it?

On H200s shared through MPS with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE below 100, some cuBLASLt fp16/bf16
kernels return success but leave part of the output unwritten (NVIDIA developer forum thread 384873,
Oct 2, 2026; NVIDIA reproduced it on Oct 6 and said a fix is coming in a later cuBLAS). The reporter's
repository (github.com/Yeosangho/mps-cublaslt-fp16-corruption-repro) lists the affected shares and shapes.

This runs the reported layer, 16x2048x1000 (ResNet-50's last layer at batch 16), in this process, which has
to be an MPS client with a share set, and prints one JSON line:
  - how many outputs differ from a float64 answer on the CPU (a plain call, and one whose output buffer is
    filled with NaN first, so unwritten outputs can't look right by accident)
  - the same for the layer's two backward matmuls (dX and dW), called the way training calls them, with
    PyTorch's default reduction settings and with the ones the training check runs under
  - for the training check's own step: how many of the outputs it judged were wrong (forward, dX, dW), so
    every alert, and every silence, can be matched to the actual results
  - what sdcguard's training check and serving check said about the same layer
  - what `sdcguard check`'s math-mode screen said in 10 seconds

    nvidia-cuda-mps-control -d
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14 python experiments/mps_unwritten.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sdcguard import SDCGuard, modes  # noqa: E402
from sdcguard.serve import ServeGuard  # noqa: E402

M, K, N = 16, 2048, 1000


def wrong_outputs(y: torch.Tensor, ref: torch.Tensor) -> int:
    """Outputs that are non-finite or off by more than 1% of their size plus 1% of a typical output
    (fp16/bf16 rounding is far smaller; an unwritten output is off by about a whole output)."""
    y = y.double().cpu()
    tol = 1e-2 * (ref.abs() + ref.pow(2).mean().sqrt())
    return int((~torch.isfinite(y) | ((y - ref).abs() > tol)).sum())


def layer(dt, dev):
    torch.manual_seed(0)
    lin = nn.Linear(K, N).to(dev, dt)
    x = torch.randn(M, K, generator=torch.Generator().manual_seed(1)).to(dev, dt)
    ref = x.double().cpu() @ lin.weight.detach().double().cpu().t() + lin.bias.detach().double().cpu()
    return lin, x, ref


def one_dtype(dt, dev) -> dict:
    lin, x, ref = layer(dt, dev)
    r = {}
    with torch.no_grad():
        y = lin(x)
        pre = torch.full((M, N), float("nan"), device=dev, dtype=dt)
        torch.addmm(lin.bias, x, lin.weight.t(), out=pre)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    r["wrong_outputs"] = wrong_outputs(y, ref)
    r["wrong_outputs_nan_prefilled"] = wrong_outputs(pre, ref)
    r["outputs"] = M * N

    # the two backward matmuls, called the way the training check's backward calls them, against float64
    gy = (torch.randn(M, N, generator=torch.Generator().manual_seed(2)) * 1e-3).to(dev, dt)
    with torch.no_grad():
        gx = gy.mm(lin.weight)                          # dX = dY W        (16 x 1000 by 1000 x 2048)
        gw = gy.t().mm(x)                               # dW = dY^T X      (1000 x 16 by 16 x 2048)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    w64 = lin.weight.detach().double().cpu()
    r["backward_dX_wrong_outputs"] = wrong_outputs(gx, gy.double().cpu() @ w64)
    r["backward_dW_wrong_outputs"] = wrong_outputs(gw, gy.double().cpu().t() @ x.double().cpu())
    r["backward_outputs"] = {"dX": M * K, "dW": N * K}
    # the same two calls with the reduction settings the training check runs under (cuBLAS may pick other
    # kernels with them)
    from sdcguard.guard import _exact_reductions_off, _exact_reductions_on
    restore = _exact_reductions_on()
    try:
        with torch.no_grad():
            gx_e, gw_e = gy.mm(lin.weight), gy.t().mm(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    finally:
        _exact_reductions_off(restore)
    r["exact_reductions_dX_wrong_outputs"] = wrong_outputs(gx_e, gy.double().cpu() @ w64)
    r["exact_reductions_dW_wrong_outputs"] = wrong_outputs(gw_e, gy.double().cpu().t() @ x.double().cpu())

    # the training check: forward, dX and dW of this layer
    events = []
    g = SDCGuard(every=1, on_alert=events.append).protect(nn.Sequential(lin))
    xr = x.clone().requires_grad_(True)
    with g.on_step(0):
        y_prot = lin(xr)
        y_prot.float().pow(2).mean().backward()
    g.unprotect()
    # were the outputs the check judged actually wrong? The loss gradient is 2 y / count, rounded to the
    # layer's dtype the way autograd rounds it, so the exact dX and dW follow from the protected forward output
    if dev.type == "cuda":
        torch.cuda.synchronize()
    gy_p = (2 * y_prot.detach().float() / y_prot.numel()).to(dt).double().cpu()
    r["checked_outputs_wrong"] = {
        "forward": wrong_outputs(y_prot.detach(), ref),
        "dX": wrong_outputs(xr.grad, gy_p @ w64),
        "dW": wrong_outputs(lin.weight.grad, gy_p.t() @ x.double().cpu()),
        "non_finite_in_forward": int((~torch.isfinite(y_prot.detach())).sum())}
    lin.zero_grad(set_to_none=True)
    worst: dict = {}
    for e in events:
        z = e.detail.get("max_z")
        worst[e.detail["where"]] = max(worst.get(e.detail["where"], 0.0), float("inf") if z is None else float(z))
    r["training_check"] = {"checked": g.stats["checked"], "alerts": len(events),
                           "alerts_by_matmul": sorted({e.detail["where"] for e in events}),
                           "worst_score_by_matmul": worst,
                           "first": {k: events[0].detail.get(k) for k in ("where", "row", "col_block", "max_z",
                                                                         "nonfinite")} if events else None}

    # the serving check, as a server would run it
    lin2, x2, _ = layer(dt, dev)
    model = nn.Sequential(lin2).eval()
    sg = ServeGuard().attach(model)
    with torch.no_grad():
        model(x2)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ev = sg.poll(wait=True)
    r["serving_check"] = {"layers": len(sg.layers), "alerts": len(ev),
                          "first": {k: ev[0].detail.get(k) for k in ("where", "row", "col_block", "max_z",
                                                                     "nonfinite", "reading")} if ev else None}
    sg.detach()
    return r


def main() -> int:
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    out = {"share": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"), "gpu": props.name,
           "sms_reported": props.multi_processor_count, "torch": torch.__version__, "cuda": torch.version.cuda}
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        try:
            out[name] = one_dtype(dt, dev)
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-1500:]}
    try:
        s = modes.screen(dev, 10.0)
        out["math_modes"] = {m: {k: v for k, v in rec.items() if k in ("matmuls", "wrong", "worst_z", "skipped",
                                                                      "error")} for m, rec in s["modes"].items()}
    except Exception as e:  # noqa: BLE001
        out["math_modes"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    print(json.dumps(out, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
