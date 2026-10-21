"""Evidence is what was checked: an operand that changes before the read-back isn't saved as if it were the
checked value, and evidence_copy keeps a copy at check time. z=1e-9 makes every check alarm, standing in for a
check-side false alarm on a correct output (so a faithful CPU recheck must find the saved output inside the rounding
envelope; judged with the guard's own alarm line it is still "statistical", as the guard's z asks)."""
import json
import os

import pytest
import torch

from sdcguard import SDCGuard, casefile


def _residual(x, w, r):
    return (x @ w.t()).add_(r)


def _relu(x, w, r):
    return torch.relu_(x @ w.t())


def _operands():
    torch.manual_seed(0)
    return (torch.randn(64, 256, dtype=torch.bfloat16), torch.randn(256, 256, dtype=torch.bfloat16),
            torch.randn(64, 256, dtype=torch.bfloat16))


@pytest.mark.parametrize("fn", [_residual, _relu])
def test_an_in_place_op_after_the_matmul_isnt_saved_as_the_checked_output(tmp_path, fn):
    g = SDCGuard(z=1e-9, evidence_dir=str(tmp_path), max_evidence=2)
    x, w, r = _operands()
    with g.at_step(0):
        fn(x, w, r)
    ev = g.log.events[0].detail["evidence"]
    assert ev.startswith("not saved") and "C" in ev and g.stats["evidence_errors"] == 1 and not g.evidence
    assert g.evidence_copy                     # from now on operands are copied at check time
    with g.at_step(1):
        fn(x, w, r)
    assert len(g.evidence) == 1
    inc = casefile.load_incident(g.evidence[0])
    assert inc["cpu"]["outputs_outside_rounding_envelope"] == 0         # the saved output is the correct one
    assert casefile.cpu_state(inc["cpu"]) != "outside envelope"


def test_evidence_copy_from_the_start(tmp_path):
    g = SDCGuard(z=1e-9, evidence_dir=str(tmp_path), max_evidence=1, evidence_copy=True)
    x, w, r = _operands()
    with g.at_step(0):
        _residual(x, w, r)
    assert len(g.evidence) == 1 and not g.stats.get("evidence_errors")
    saved = torch.load(g.evidence[0], weights_only=True)
    assert torch.equal(saved["C"], x @ w.t())


def test_fast_mode_adds_a_separate_bias_out_of_place(tmp_path):
    """A non-contiguous 3D input: the matmul is checked without the bias, then the bias is added. In place, that
    add rewrote the checked output before the evidence was saved."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 256).bfloat16()
    g = SDCGuard(z=1e-9, evidence_dir=str(tmp_path), max_evidence=1).protect(lin)
    x = torch.randn(8, 16, 256, dtype=torch.bfloat16).transpose(0, 1)
    with g.at_step(0):
        y = lin(x)
    g.unprotect()
    assert torch.equal(y, torch.nn.functional.linear(x, lin.weight, lin.bias))
    assert len(g.evidence) == 1
    inc = casefile.load_incident(g.evidence[0])
    assert inc["cpu"]["outputs_outside_rounding_envelope"] == 0
    assert casefile.cpu_state(inc["cpu"]) != "outside envelope"


def _fsdp_worker(rank, world, port, out, copy):
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.fsdp import fully_shard
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Linear(256, 256)).bfloat16()
        for layer in m:
            fully_shard(layer)
        fully_shard(m)
        d = os.path.join(out, f"rank{rank}")
        g = SDCGuard(z=1e-9, evidence_dir=d, max_evidence=6, evidence_copy=copy).protect(m)
        with g.at_step(0):
            m(torch.randn(64, 256, dtype=torch.bfloat16)).float().sum().backward()
        loaded = 0
        for p in g.evidence:
            torch.load(p, weights_only=True)
            loaded += 1
        notes = [e.detail["evidence"] for e in g.log.events if str(e.detail.get("evidence", "")).startswith("not")]
        with open(os.path.join(out, f"r{rank}.json"), "w") as f:
            json.dump({"loaded": loaded, "saved": len(g.evidence), "not_saved": notes, "alerts": g.stats["alerts"],
                       "errors": g.stats.get("evidence_errors", 0)}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("copy", [False, True])
def test_fsdp2_evidence_is_loadable_or_marked_not_saved(tmp_path, copy):
    pytest.importorskip("torch.distributed.fsdp")
    import socket
    import torch.multiprocessing as mp
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(_fsdp_worker, args=(2, port, str(tmp_path), copy), nprocs=2, join=True)
    for rank in range(2):
        r = json.loads((tmp_path / f"r{rank}.json").read_text())
        assert r["loaded"] == r["saved"]                   # nothing saved that can't be loaded
        if copy:
            assert r["saved"] == r["alerts"] == 5 and not r["not_saved"]     # 2 forward, 2 dW, 1 dX
        else:
            assert r["not_saved"] and r["errors"] == len(r["not_saved"])
            assert all("freed" in n for n in r["not_saved"])


def test_a_change_through_data_is_noticed_too(tmp_path):
    """x.data.add_() doesn't bump the version counter (nor does a write by faulty hardware): the contents'
    fingerprint catches it, so the changed operand isn't saved as the checked one."""
    g = SDCGuard(z=1e-9, evidence_dir=str(tmp_path), max_evidence=2)
    x, w, r = _operands()
    with g.at_step(0):
        y = x @ w.t()
        y.data.add_(1.0)
    ev = g.log.events[0].detail["evidence"]
    assert ev.startswith("not saved") and "C" in ev and not g.evidence


@pytest.mark.parametrize("protect", [True, False])
def test_an_interrupted_step_lets_go_of_its_evidence_operands(tmp_path, protect):
    """A step stopped by KeyboardInterrupt (or another BaseException) drops its queued checks without judging them.
    The operands kept for their evidence (GPU memory) must go with them, not stay referenced through later checked
    steps until one of them queues a check."""
    import warnings
    torch.manual_seed(0)
    m = torch.nn.Linear(64, 64).bfloat16()
    g = SDCGuard(evidence_dir=str(tmp_path))
    if protect:
        g.protect(m)
    with pytest.raises(KeyboardInterrupt):
        with g.at_step(0):
            m(torch.randn(8, 64, dtype=torch.bfloat16))
            assert g._kept
            raise KeyboardInterrupt
    assert g._kept == [] and g.stats["checked"] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with g.at_step(1):
            pass
    assert g._kept == []
