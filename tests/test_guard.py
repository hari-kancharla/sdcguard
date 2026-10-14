import pytest
import torch
import torch.nn as nn

from sdcguard import FaultInjector, SDCError, SDCGuard


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256), nn.LayerNorm(256))


def step(model, x):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(x).float().pow(2).mean()
    loss.backward()
    return loss


def test_clean_training_step_has_no_alerts_and_sees_backward():
    model, x = tiny_model(), torch.randn(64, 256)
    fwd_only = SDCGuard(sample_rate=1.0)
    with fwd_only, torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(x)
    guard = SDCGuard(sample_rate=1.0)
    with guard.at_step(0):
        step(model, x)
    assert guard.stats["alerts"] == 0
    assert guard.stats["checked"] > 0
    # backward launches extra matmuls (grad wrt inputs and weights)
    assert guard.stats["matmuls_seen"] > fwd_only.stats["matmuls_seen"]


def test_detects_injected_faults_when_guard_is_on_top():
    model, x = tiny_model(), torch.randn(64, 256)
    inj = FaultInjector(rate=1.0, seed=3).arm_on([model[0]])
    guard = SDCGuard(sample_rate=1.0)
    with inj:              # faulty "hardware"
        with guard.at_step(7):   # detector entered last -> sees corrupted outputs
            step(model, x)
    inj.remove_hooks()
    assert len(inj.events) >= 1
    assert guard.stats["alerts"] >= 1
    ev = guard.log.events[0]
    assert ev.step == 7 and ev.detector == "checksum" and "host" in ev.identity


def test_wrong_order_cannot_see_faults():
    # documents the ordering rule: if the injector is on top, the guard checks clean outputs
    model, x = tiny_model(), torch.randn(64, 256)
    inj = FaultInjector(rate=1.0, seed=3).arm_on([model[0]])
    guard = SDCGuard(sample_rate=1.0)
    with guard:
        with inj:
            step(model, x)
    inj.remove_hooks()
    assert len(inj.events) >= 1 and guard.stats["alerts"] == 0


def test_raise_on_alert():
    model, x = tiny_model(), torch.randn(64, 256)
    inj = FaultInjector(rate=1.0, seed=5).arm_on([model[2]])
    with pytest.raises(SDCError):
        with inj, SDCGuard(sample_rate=1.0, raise_on_alert=True):
            step(model, x)
    inj.remove_hooks()


def test_apple_gpu_linear_ops_are_understood():
    # On Apple GPUs (MPS), nn.Linear runs as aten::_mps_linear and its backward as
    # aten::linear_backward. Those ops only exist in macOS builds, so test the mapping directly.
    from sdcguard.checksum import check_matmul
    from sdcguard.guard import matmul_checks

    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 32, 256, generator=g).bfloat16()
    w = (torch.randn(512, 256, generator=g) / 16).bfloat16()
    b = (torch.randn(512, generator=g) * 0.02).bfloat16()
    y = torch.nn.functional.linear(x, w, b)
    (A, B, C, bias, al, be), = matmul_checks("aten::_mps_linear", (x, w, b), {}, y)
    assert check_matmul(A, B, C, bias, alpha=al, beta=be).ok
    y_bad = y.clone()
    y_bad[1, 3, 64:96] = 0
    (A, B, C, bias, al, be), = matmul_checks("aten::_mps_linear", (x, w, b), {}, y_bad)
    assert not check_matmul(A, B, C, bias, alpha=al, beta=be).ok

    gy = torch.randn(4, 32, 512, generator=g).bfloat16()
    gx = (gy.reshape(-1, 512).float() @ w.float()).bfloat16().reshape(4, 32, 256)   # dX = dY W
    gw = (gy.reshape(-1, 512).float().t() @ x.reshape(-1, 256).float()).bfloat16()  # dW = dY^T X
    checks = matmul_checks("aten::linear_backward", (x, gy, w, [True, True, False]), {}, (gx, gw, None))
    assert len(checks) == 2 and all(check_matmul(*c[:3]).ok for c in checks)
    gw_bad = gw.clone()
    gw_bad[7, 0:32] = 0
    checks = matmul_checks("aten::linear_backward", (x, gy, w, [True, True, False]), {}, (gx, gw_bad, None))
    assert check_matmul(*checks[0][:3]).ok and not check_matmul(*checks[1][:3]).ok


def test_sampling_reduces_checks():
    model, x = tiny_model(), torch.randn(64, 256)
    guard = SDCGuard(sample_rate=0.0)
    with guard:
        step(model, x)
    assert guard.stats["checked"] == 0 and guard.stats["matmuls_seen"] > 0


def test_exact_reductions_on_checked_steps_only(monkeypatch):
    # On NVIDIA the guard keeps bf16/fp16 partial sums in fp32 while active, then restores.
    import torch.nn as nn
    from sdcguard import SDCGuard
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    m = torch.backends.cuda.matmul
    saved = (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
    try:
        m.allow_bf16_reduced_precision_reduction = True
        m.allow_fp16_reduced_precision_reduction = True
        g = SDCGuard()
        with g:
            assert not m.allow_bf16_reduced_precision_reduction
            assert not m.allow_fp16_reduced_precision_reduction
        assert m.allow_bf16_reduced_precision_reduction and m.allow_fp16_reduced_precision_reduction
        # restored even if the step raises
        try:
            with SDCGuard():
                raise ValueError("boom")
        except ValueError:
            pass
        assert m.allow_bf16_reduced_precision_reduction
        # a setting the user already turned off stays off
        m.allow_bf16_reduced_precision_reduction = False
        with SDCGuard():
            pass
        assert not m.allow_bf16_reduced_precision_reduction and m.allow_fp16_reduced_precision_reduction
        # opt-out leaves everything alone
        m.allow_bf16_reduced_precision_reduction = True
        with SDCGuard(exact_reductions=False):
            assert m.allow_bf16_reduced_precision_reduction
    finally:
        m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction = saved


def test_on_result_sees_every_check():
    import torch.nn as nn
    from sdcguard import SDCGuard
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    seen = []
    g = SDCGuard(on_result=lambda d, r: seen.append((d["pass"], d["op"], r.ok)))
    with g:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(torch.randn(32, 64)).float().pow(2).mean().backward()
    assert len(seen) == g.stats["checked"] > 0
    assert {p for p, _, _ in seen} == {"forward", "backward"} and all(ok for _, _, ok in seen)


def test_schedule_and_restart_window():
    from sdcguard import SDCGuard
    g = SDCGuard(every=10)
    assert [s for s in range(25) if g.should_check(s)] == [0, 10, 20]
    g.after_restart(201, steps=5)
    assert [s for s in range(195, 225) if g.should_check(s)] == [200, 201, 202, 203, 204, 205, 210, 220]


def test_on_step_checks_only_scheduled_steps():
    import torch
    import torch.nn as nn
    from sdcguard import SDCGuard
    model = nn.Linear(32, 64)
    g = SDCGuard(every=3)
    seen = []
    for step in range(7):
        with g.on_step(step) as checked:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model(torch.randn(8, 32)).float().sum().backward()
        seen.append(checked)
    assert seen == [True, False, False, True, False, False, True]
    assert g.stats["checked"] > 0 and g.stats["alerts"] == 0


def test_fp16_loss_scaling_overflow_is_not_an_alarm():
    """fp16 training with loss scaling overflows to Inf on purpose now and then (GradScaler then
    skips the step). Those Infs come from the inputs' size, not from the hardware: no alert."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64)).half()
    x = torch.randn(32, 64).half()
    g = SDCGuard()
    with g:
        loss = model(x).float().pow(2).mean()
        (loss * 2.0**28).backward()           # a loss scale that is too high: gradients overflow
    # it really overflowed, inside the matmuls (the weight gradients are matmul outputs)
    assert not any(torch.isfinite(m.weight.grad).all() for m in (model[0], model[2]))
    assert g.stats["checked"] == 5 and g.stats["alerts"] == 0, g.log.events


def test_inf_from_small_inputs_is_still_caught():
    """The same Inf, when the inputs are too small to produce it, is corruption."""
    from sdcguard.checksum import check_matmul
    torch.manual_seed(0)
    A, B = torch.randn(32, 64).half(), torch.randn(64, 256).half()
    C = A @ B
    C[3, 200] = float("inf")
    r = check_matmul(A, B, C)
    assert not r.ok and r.nonfinite and (r.row, r.col_block) == (3, 1)
    A2 = A.clone()
    A2[3] = 30000.0                           # now row 3 really overflows: not judged
    r2 = check_matmul(A2, B, A2 @ B)
    assert r2.ok
