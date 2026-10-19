"""Which GPU generations get the Triton kernels, and with what pipeline depth (no GPU needed)."""
import pytest
import torch

from sdcguard import _group


def _fake_gpu(monkeypatch, major, minor=0):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda idx=None: (major, minor))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


@pytest.mark.parametrize("major,expected", [(7, True), (8, True), (9, True), (10, True), (12, False)])
def test_triton_kernels_run_only_on_validated_generations(monkeypatch, major, expected):
    monkeypatch.delenv("SDCGUARD_TRITON_NEW_ARCH", raising=False)
    _fake_gpu(monkeypatch, major)
    assert _group.arch_ok(torch.device("cuda", 0)) is expected


def test_new_arch_switch(monkeypatch):
    _fake_gpu(monkeypatch, 10)
    monkeypatch.setenv("SDCGUARD_TRITON_NEW_ARCH", "0")
    assert not _group.arch_ok(torch.device("cuda", 0))          # Blackwell back on the plain checks
    _fake_gpu(monkeypatch, 12)
    monkeypatch.setenv("SDCGUARD_TRITON_NEW_ARCH", "1")
    assert _group.arch_ok(torch.device("cuda", 0))              # opt in on an unvalidated generation


@pytest.mark.parametrize("bj", [16, 32, 64, 128])
def test_blackwell_launches_without_software_pipelining(monkeypatch, bj):
    _fake_gpu(monkeypatch, 9)
    hopper = _group._cfg(bj, torch.device("cuda", 0))
    _fake_gpu(monkeypatch, 10)
    blackwell = _group._cfg(bj, torch.device("cuda", 0))
    assert hopper[1] > 1 and blackwell[1] == 1
    assert (hopper[0], hopper[2]) == (blackwell[0], blackwell[2])   # same tiles and warps, only the depth
    assert _group._cfg(bj) == _group._cfg(bj, torch.device("cpu")) == hopper


def test_pipeline_depth_is_left_alone_off_blackwell(monkeypatch):
    _fake_gpu(monkeypatch, 8)
    assert _group.pipeline_stages(torch.device("cuda", 0), None) is None
    assert _group.pipeline_stages(torch.device("cpu"), 3) == 3
    _fake_gpu(monkeypatch, 10)
    assert _group.pipeline_stages(torch.device("cuda", 0), None) == 1
