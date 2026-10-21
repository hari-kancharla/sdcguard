"""Reproductions against the b778d27 snapshot: incomplete numerical checks and canary lifecycle.

Run the Triton cases on a GPU, or separately with TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1.
The interpreter checks the kernel's logic, not generated CUDA code or hardware behavior.
"""
import os

import pytest
import torch

from sdcguard import SDCGuard, checksum as cs
from sdcguard.canary import Canary


def _mixed(case, *, device="cpu", dtype=torch.bfloat16, corrupt=False):
    # Include rows in different _group tiles and a partial final column block. Unknown coverage
    # has to survive both the local reduction and the final reduction across tiles/spans.
    a = torch.ones(70, 8, dtype=dtype, device=device)
    b = torch.ones(8, 260, dtype=dtype, device=device)
    if case == "row":
        a[0] = float("nan")
    elif case == "block":
        b[:, :128] = float("nan")
    elif case == "remainder":
        b[:, 256:] = float("nan")
    elif case == "batch":
        a = a.repeat(2, 1, 1)
        b = b.repeat(2, 1, 1)
        a[0] = float("nan")
    c = a.float() @ b.float()
    if corrupt:
        c.reshape(-1, c.shape[-1])[-1, 200] = 1000.0
    return a, b, c


@pytest.mark.parametrize("case", ["row", "block", "remainder", "batch"])
@pytest.mark.parametrize("precise", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
def test_partial_unjudged_survives_plain_reduction(monkeypatch, case, precise, corrupt):
    monkeypatch.setattr(cs, "USE_FAST", False)
    a, b, c = _mixed(case, corrupt=corrupt)
    r = cs.check_matmul_acc(a, b, c) if precise else cs.check_matmul(a, b, c.bfloat16())
    if corrupt:
        assert not r.ok and r.max_z > 6, r       # a real alarm has priority over incomplete coverage
    else:
        assert r.ok and not r.judged and r.max_z == cs.UNJUDGED, r


@pytest.mark.parametrize("precise", [False, True])
def test_training_counts_a_partially_unjudged_check(monkeypatch, precise):
    monkeypatch.setattr(cs, "USE_FAST", False)
    model = torch.nn.Linear(8, 128, bias=False).bfloat16()
    guard = SDCGuard(every=1, precise=precise).protect(model)
    a = torch.ones(2, 8, dtype=torch.bfloat16)
    a[0] = float("nan")
    try:
        with guard.on_step(0):
            model(a)
        assert guard.stats["checked"] == guard.stats["unjudged"] == 1
        assert guard.stats["alerts"] == 0
    finally:
        guard.unprotect()


@pytest.mark.parametrize("precise", [False, True])
def test_finite_zero_and_tiny_results_remain_judged(monkeypatch, precise):
    monkeypatch.setattr(cs, "USE_FAST", False)
    for scale in (0.0, 1e-24, 1.0):
        a = torch.full((3, 8), scale, dtype=torch.bfloat16)
        b = torch.ones(8, 260, dtype=torch.bfloat16)
        c = a.float() @ b.float()
        r = cs.check_matmul_acc(a, b, c) if precise else cs.check_matmul(a, b, c.bfloat16())
        assert r.ok and r.judged, r


@pytest.mark.parametrize("case", ["row", "block", "remainder", "batch"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_partial_unjudged_survives_triton_reduction(case, corrupt):
    from sdcguard import _group
    interpreter = os.environ.get("TRITON_INTERPRET") == "1"
    if not _group.HAVE_TRITON or not (torch.cuda.is_available() or interpreter):
        pytest.skip("requires CUDA or an explicit Triton interpreter run")
    dev = torch.device("cuda" if torch.cuda.is_available() and not interpreter else "cpu")
    a, b, c = _mixed(case, device=dev, corrupt=corrupt)
    grp = _group.Group(dev, torch.bfloat16, 128, cs.ABS_FLOOR)
    grp.add(a, b, c.bfloat16(), None, 1.0, 1.0)
    grp.launch()
    raw = grp.result[0].tolist()
    score, _, _, _, _ = grp.row(0, raw)
    assert raw[4] == 1.0                        # metadata survived padded tiles and final reduction
    assert score > 6 if corrupt else score == cs.UNJUDGED
    torch_row = grp._worst_torch()[0].tolist()
    assert grp.row(0, torch_row) == grp.row(0, raw)


def test_training_counts_partially_unjudged_triton_checks(monkeypatch):
    from sdcguard import _group
    interpreter = os.environ.get("TRITON_INTERPRET") == "1"
    if not _group.HAVE_TRITON or not (torch.cuda.is_available() or interpreter):
        pytest.skip("requires CUDA or an explicit Triton interpreter run")
    dev = torch.device("cuda" if torch.cuda.is_available() and not interpreter else "cpu")
    # Individual kernels are tested above. Bypass the expensive first-use probe here to verify only
    # the protected-model -> Group -> guard.flush plumbing, not to substitute for that probe in use.
    monkeypatch.setattr(_group, "available", lambda _: True)
    model = torch.nn.Linear(8, 260, bias=False, device=dev, dtype=torch.bfloat16)
    guard = SDCGuard(every=1).protect(model)
    a = torch.ones(70, 8, device=dev, dtype=torch.bfloat16)
    a[0] = float("nan")
    used = _group.USED[0]
    try:
        with guard.on_step(0):
            model(a)
        assert _group.USED[0] == used + 1
        assert guard.stats["checked"] == guard.stats["unjudged"] == 1
        assert guard.stats["alerts"] == 0
    finally:
        guard.unprotect()


def test_canary_includes_a_root_leaf_layer():
    layer = torch.nn.Linear(8, 4, bias=False)
    canary = Canary(layer)
    result = canary.fingerprints(torch.ones(2, 8))
    assert result.shape == (1, 2) and canary.last_names == ["Linear"]


@pytest.mark.parametrize("raise_in_forward", [False, True])
def test_canary_restores_each_modules_training_mode(raise_in_forward):
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Dropout(), torch.nn.Linear(8, 4))
    model.train()
    model[0].eval()
    before = [mod.training for mod in model.modules()]

    def forward(m, x):
        result = m(x)
        if raise_in_forward:
            raise RuntimeError("synthetic forward failure")
        return result

    canary = Canary(model, forward_fn=forward)
    if raise_in_forward:
        with pytest.raises(RuntimeError, match="synthetic forward failure"):
            canary.fingerprints(torch.ones(2, 8))
    else:
        canary.fingerprints(torch.ones(2, 8))
    assert [mod.training for mod in model.modules()] == before
    assert not any(mod._forward_hooks for mod in model.modules())
