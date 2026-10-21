"""Replica and canary detectors across 3 CPU processes (gloo). Rank 2 is the 'bad GPU'."""
import json
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from sdcguard import Canary, FaultInjector, check_replicas

WORLD = 3
BAD = 2


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64), nn.LayerNorm(64))


def _worker(rank, init_file, out_dir):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    model = _model()
    result = {}

    # Replica check: the bad rank silently flips one bit in one weight.
    if rank == BAD:
        with torch.no_grad():
            w = model[2].weight.view(-1)
            w.view(torch.int32)[5] ^= 1 << 3
    suspects, _ = check_replicas(model.parameters(), step=100)
    result["replica_suspects"] = suspects

    # Canary: fresh identical weights; the bad rank's hardware corrupts the 2nd Linear.
    model = _model()
    batch = torch.randn(32, 64, generator=torch.Generator().manual_seed(42))

    def fwd(m, b):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return m(b)

    canary = Canary(model, forward_fn=fwd)
    if rank == BAD:
        inj = FaultInjector(rate=1.0, seed=9).arm_on([model[2]])
        with inj:
            suspects, layer, _ = canary.check(batch, step=200)
        inj.remove_hooks()
    else:
        suspects, layer, _ = canary.check(batch, step=200)
    result["canary_suspects"], result["canary_layer"] = suspects, layer

    # Clean canary: no faults anywhere -> no suspects.
    suspects, layer, _ = Canary(model, forward_fn=fwd).check(batch, step=300)
    result["clean_canary_suspects"] = suspects

    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(result, f)
    dist.barrier()
    # PyTorch 2.5's gloo teardown can hang one rank after all the work is done; the process exits anyway
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 6):
        dist.destroy_process_group()


def test_replica_and_canary_name_the_bad_rank():
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        mp.spawn(_worker, args=(init_file, d), nprocs=WORLD, join=True)
        results = [json.load(open(os.path.join(d, f"rank{r}.json"))) for r in range(WORLD)]
    for res in results:
        assert res["replica_suspects"] == [BAD]
        assert res["canary_suspects"] == [BAD]
        assert res["canary_layer"] == "2"          # the second Linear in the Sequential
        assert res["clean_canary_suspects"] == []


def _ddp_fast_worker(rank, init_file, out_dir):
    """Fast mode under DistributedDataParallel: same gradients as without sdcguard, faults on the bad
    rank caught and only there."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    from sdcguard import SDCGuard
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    x = torch.randn(16, 64, generator=torch.Generator().manual_seed(rank))
    res = {}
    for mode in ("plain", "fast"):
        model = DDP(_model())
        guard = SDCGuard().protect(model) if mode == "fast" else None
        with (guard.at_step(0) if guard else torch.enable_grad()):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(x).float().pow(2).mean()
            loss.backward()
        res[mode] = [p.grad.clone() for p in model.parameters()]
        if guard:
            res["fast_checks"], res["fast_alerts"] = guard.stats["checked"], guard.stats["alerts"]
    res["same_grads"] = all(torch.equal(a, b) for a, b in zip(res.pop("plain"), res.pop("fast")))
    # faulty rank: corrupt the first Linear's outputs
    model = DDP(_model())
    guard = SDCGuard().protect(model)
    inj = FaultInjector(rate=1.0, seed=5).arm_on([model.module[0]]) if rank == BAD else None
    with (inj if inj else torch.enable_grad()):
        with guard.at_step(1):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(x).float().pow(2).mean()
            loss.backward()
    if inj:
        inj.remove_hooks()
    res["faulty_alerts"] = guard.stats["alerts"]
    res["faulty_where"] = sorted({e.detail["where"] for e in guard.log.events})
    with open(os.path.join(out_dir, f"ddp{rank}.json"), "w") as f:
        json.dump(res, f)
    dist.barrier()
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 6):
        dist.destroy_process_group()


def test_fast_mode_under_ddp():
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        mp.spawn(_ddp_fast_worker, args=(init_file, d), nprocs=WORLD, join=True)
        results = [json.load(open(os.path.join(d, f"ddp{r}.json"))) for r in range(WORLD)]
    for r, res in enumerate(results):
        assert res["same_grads"], r
        assert res["fast_checks"] > 0 and res["fast_alerts"] == 0
        if r == BAD:
            assert res["faulty_alerts"] >= 1 and res["faulty_where"][0].startswith("module.0")
        else:
            assert res["faulty_alerts"] == 0


def _fsdp_fast_worker(rank, init_file, out_dir):
    """Fast mode with FSDP2 (fully_shard) and activation checkpointing: checks run, no false alarms,
    and the same loss and gradients as without sdcguard."""
    from torch.distributed.fsdp import fully_shard
    from torch.utils.checkpoint import checkpoint

    from sdcguard import SDCGuard
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    x = torch.randn(16, 64, generator=torch.Generator().manual_seed(rank))
    res = {}
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = _model()

        def forward(self, t):
            s = self.seq
            h = checkpoint(lambda u: s[1](s[0](u)), t, use_reentrant=False)     # recomputed in backward
            return s[3](s[2](h))

    # An explicit CPU mesh: without one, FSDP2 builds its mesh on CUDA whenever a GPU is visible, which
    # with gloo and CPU tensors crashed (SIGSEGV) on the H100 box in round 3
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cpu", (2,))
    for mode in ("plain", "fast"):
        model = Net()
        fully_shard(model.seq[0], mesh=mesh)
        fully_shard(model.seq[2], mesh=mesh)
        fully_shard(model, mesh=mesh)
        guard = SDCGuard().protect(model) if mode == "fast" else None
        with (guard.at_step(0) if guard else torch.enable_grad()):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(x).float().pow(2).mean()
            loss.backward()
        res[mode] = ([p.grad.full_tensor().clone() for p in model.parameters()], loss.item())
        if guard:
            res["checks"], res["alerts"] = guard.stats["checked"], guard.stats["alerts"]
    (gp, lp), (gf, lf) = res.pop("plain"), res.pop("fast")
    res["same_loss"] = lp == lf
    res["same_grads"] = all(torch.equal(a, b) for a, b in zip(gp, gf))
    with open(os.path.join(out_dir, f"fsdp{rank}.json"), "w") as f:
        json.dump(res, f)
    dist.barrier()
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 6):
        dist.destroy_process_group()


def test_fast_mode_with_fsdp2_and_checkpointing():
    try:
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("this PyTorch has no FSDP2")
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        mp.spawn(_fsdp_fast_worker, args=(init_file, d), nprocs=2, join=True)
        results = [json.load(open(os.path.join(d, f"fsdp{r}.json"))) for r in range(2)]
    for res in results:
        assert res["checks"] > 0 and res["alerts"] == 0
        assert res["same_loss"] and res["same_grads"]


def _guard_replica_worker(rank, init_file, out_dir):
    """SDCGuard(replicas=True) under DDP: clean checked steps agree; after one rank's optimizer update goes
    wrong (one flipped bit), the next checked step names that rank on every rank."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    from sdcguard import SDCGuard
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    model = DDP(_model())
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    seen = []
    guard = SDCGuard(every=2, replicas=True, on_alert=seen.append).protect(model)
    x = torch.randn(16, 64, generator=torch.Generator().manual_seed(rank))
    res = {"alerts_by_step": {}}
    for step in range(6):
        before = len(guard.log.events)
        with guard.on_step(step):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(x).float().pow(2).mean()
            loss.backward()
        opt.step()
        opt.zero_grad()
        if step == 2 and rank == BAD:                # this rank's weight update went wrong
            with torch.no_grad():
                model.module[2].weight.view(-1).view(torch.int32)[11] ^= 1 << 5
        res["alerts_by_step"][step] = [e.detail.get("suspect_ranks") for e in guard.log.events[before:]
                                       if e.detector == "replica"]
    res["stats"] = {k: guard.stats[k] for k in ("replica_checks", "replica_mismatches", "replica_skipped")}
    rep = [e for e in guard.log.events if e.detector == "replica"]
    res["named_rank"] = [e.identity.get("rank") for e in rep]
    res["suspects"] = [e.detail.get("suspects") for e in rep]
    res["delivered"] = len([e for e in seen if e.detector == "replica"])
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(res, f)
    dist.barrier()
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 6):
        dist.destroy_process_group()


def test_guard_replica_check_names_the_rank_whose_update_went_wrong():
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        mp.spawn(_guard_replica_worker, args=(init_file, d), nprocs=WORLD, join=True)
        results = [json.load(open(os.path.join(d, f"rank{r}.json"))) for r in range(WORLD)]
    for res in results:
        alerts = res["alerts_by_step"]
        assert alerts["0"] == [] and alerts["2"] == []          # clean checked steps agree
        assert alerts["4"] == [[BAD]]                            # the first checked step after the bad update
        assert res["stats"] == {"replica_checks": 3, "replica_mismatches": 1, "replica_skipped": 0}
        assert res["named_rank"] == [BAD] and res["suspects"][0][0]["rank"] == BAD   # names the bad GPU
    assert sum(r["delivered"] for r in results) == 1             # one rank delivers it, not every rank


def test_sharded_weights_are_recognized():
    from sdcguard.replica import is_sharded

    class FakeDTensor(torch.Tensor):
        placements = ("Shard(0)",)

        def to_local(self):
            return self
    assert is_sharded(torch.zeros(2).as_subclass(FakeDTensor)) and not is_sharded(torch.zeros(2))


def test_alert_lines_speak_for_their_detector():
    from sdcguard.alerts import describe
    from sdcguard.events import SDCEvent
    rep = SDCEvent(detector="replica", step=4, detail={"suspect_ranks": [2], "world": 3, "clear_majority": True,
                                                       "where": "data-parallel weights"},
                   identity={"gpu_uuid": "GPU-bad"})
    assert "drifted from the other ranks (rank 2 of 3" in describe(rep) and "GPU-bad" in describe(rep)
    rc = SDCEvent(detector="recompute", step=1, detail={"where": "block.3", "outputs_that_differ": [0, 1]},
                  identity={})
    assert "recomputed this checkpointed block" in describe(rc) and "2 of its outputs" in describe(rc)
    mm = SDCEvent(detector="checksum", step=1, detail={"where": "fc", "max_z": 7.5, "alarm_line": 4.0}, identity={})
    assert "alarm line 4" in describe(mm)
