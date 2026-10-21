"""replicas=True (the data-parallel weight comparison) across real gloo process groups: it runs only on the checked
steps every rank shares, so a loss spike one rank saw alone can't hang the job in mismatched collectives; a guard
that wasn't given the model says so instead of passing as checked; and with replica groups the alert names the
job's global rank that drifted."""
import datetime
import json
import os
import socket
import time
import warnings

import torch
import torch.multiprocessing as mp

from sdcguard import SDCGuard


def _bf16_mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16)).bfloat16()


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _spawn(fn, world, *args, limit=120.0):
    """mp.spawn that can't hang the test run: processes still alive after `limit` seconds are killed (their
    missing result file then fails the test)."""
    ctx = mp.spawn(fn, args=(world, *args), nprocs=world, join=False)
    t0 = time.time()
    while True:
        try:
            if ctx.join(timeout=2):
                return
        except Exception:  # noqa: BLE001  (a worker raised: its result file says what happened)
            return
        if time.time() - t0 > limit:
            for p in ctx.processes:
                p.kill()
            return


def _flip_a_weight_bit(model):
    with torch.no_grad():
        model[0].weight.view(-1).view(torch.int16)[3] ^= 1 << 2


def test_only_steps_every_rank_checks_run_the_replica_check():
    """The schedule by itself: a window observe() opened is checked, but it's this rank's own, so it isn't a step
    every rank shares; the `every` schedule and an after_restart() window are."""
    g = SDCGuard(every=10)
    for step in range(1, 30):
        g.observe(step, loss=2.0)
    assert g.observe(30, loss=float("nan"))
    assert g.should_check(31) and not g._every_rank_checks(31)
    assert g.should_check(40) and g._every_rank_checks(40)
    g.after_restart(100, steps=5)
    assert all(g.should_check(s) and g._every_rank_checks(s) for s in range(100, 105))


def _observe_worker(rank, world, port, out):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=15))
    res = {"done": False}
    try:
        torch.manual_seed(0)
        model = DDP(torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Linear(32, 16)))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        g = SDCGuard(every=10, replicas=True).protect(model)
        x = torch.randn(8, 16, generator=torch.Generator().manual_seed(rank))
        checked = []
        for step in range(1, 30):
            with g.on_step(step) as c:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = model(x).float().pow(2).mean()
                loss.backward()
            checked.append(c)
            opt.step()
            opt.zero_grad()
            # README: guard.observe(step, loss=...), with this rank's own loss; rank 1's GPU goes bad at step 15
            g.observe(step, loss=float("nan") if (rank == 1 and step == 15) else loss.item())
        res = {"done": True, "checked": sum(checked), "replica_checks": g.stats["replica_checks"],
               "replica_skipped": g.stats["replica_skipped"]}
    except Exception as e:  # noqa: BLE001
        res = {"done": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    finally:
        with open(os.path.join(out, f"ob{rank}.json"), "w") as f:
            json.dump(res, f)


def test_a_loss_spike_on_one_rank_does_not_deadlock_replica_checks(tmp_path):
    """In DDP each rank logs its own loss, and the rank with the bad GPU is the one that spikes: observe() opens a
    checked window on that rank alone. The replica check is a collective; run at the start of that window's
    steps, it would wait in all_gather while the other ranks wait in DDP's all-reduce, until the process group's
    timeout. Rank 1 checks every step after its spike; the replica check runs on steps 10 and 20 on both."""
    _spawn(_observe_worker, 2, _port(), str(tmp_path), limit=90.0)
    res = []
    for r in range(2):
        p = tmp_path / f"ob{r}.json"
        res.append(json.loads(p.read_text()) if p.exists() else {"done": False, "error": "hung"})
        assert res[r]["done"], (r, res[r])
        assert res[r]["replica_checks"] == 2 and res[r]["replica_skipped"] == 0, res[r]
    assert res[0]["checked"] == 2 and res[1]["checked"] == 15, res      # rank 1: step 10, then 16..29


def _replica_default_mode_worker(rank, world, port, out):
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=30))
    res = {}
    try:
        model = _bf16_mlp()
        if rank == 1:                                            # this rank's weights drifted
            _flip_a_weight_bit(model)
        g = SDCGuard(replicas=True)                              # the default mode, no watch()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for step in (0, 10):
                with g.at_step(step):
                    model(torch.randn(4, 16, dtype=torch.bfloat16)).float().sum().backward()
        res["unwatched"] = {"replica_checks": g.stats["replica_checks"], "skipped": g.stats["replica_skipped"],
                            "warned": [str(x.message) for x in w if "replica" in str(x.message)]}
        g2 = SDCGuard(replicas=True).watch(model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for step in (0, 5):                                  # 5: at_step() off the `every` schedule
                with g2.at_step(step):
                    model(torch.randn(4, 16, dtype=torch.bfloat16)).float().sum().backward()
        res["watched"] = {"replica_checks": g2.stats["replica_checks"], "alerts": g2.stats["alerts"],
                          "skipped": g2.stats["replica_skipped"],
                          "warned": [str(x.message) for x in w if "replica" in str(x.message)]}
    finally:
        with open(os.path.join(out, f"dm{rank}.json"), "w") as f:
            json.dump(res, f)
        dist.destroy_process_group()


def test_replicas_true_in_the_default_mode_is_not_silently_off(tmp_path):
    """The default mode sees operations, not the model, so the replica check needs the model from watch(). Without
    it every checked step's comparison is counted as skipped, with one warning that says what to do; with it,
    rank 1's drifted weights are caught. at_step() on a step that isn't on the `every` schedule doesn't compare
    (every rank must run the comparison on the same steps), and says so instead of passing as checked."""
    _spawn(_replica_default_mode_worker, 2, _port(), str(tmp_path))
    for r in range(2):
        res = json.loads((tmp_path / f"dm{r}.json").read_text())
        un = res["unwatched"]
        assert un["replica_checks"] == 0 and un["skipped"] == 2 and len(un["warned"]) == 1, res
        assert "watch()" in un["warned"][0]
        wa = res["watched"]
        assert (wa["replica_checks"], wa["alerts"], wa["skipped"], len(wa["warned"])) == (1, 1, 1, 1), res
        assert "step 5" in wa["warned"][0] and "every=10" in wa["warned"][0]


def _replica_group_worker(rank, world, port, out):
    import torch.distributed as dist
    from sdcguard.alerts import describe
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=30))
    res = {}
    try:
        groups = [dist.new_group([0]), dist.new_group([1, 2, 3])]     # HSDP-style replica groups
        model = _bf16_mlp()
        if rank == 3:                                                  # global rank 3 drifted
            _flip_a_weight_bit(model)
        g = SDCGuard(replicas=True, replica_group=groups[0] if rank == 0 else groups[1]).protect(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with g.at_step(0):
                pass
        ev = [e for e in g.log.events if e.detector == "replica"]
        res = {"suspect_ranks": [e.detail["suspect_ranks"] for e in ev],
               "group_ranks": [e.detail.get("suspect_group_ranks") for e in ev],
               "who": [[s["rank"] for s in e.detail["suspects"]] for e in ev],
               "identity_rank": [e.identity.get("rank") for e in ev], "text": [describe(e) for e in ev]}
    finally:
        with open(os.path.join(out, f"rg{rank}.json"), "w") as f:
            json.dump(res, f)
        dist.destroy_process_group()


def test_replica_group_alerts_name_the_global_rank_that_drifted(tmp_path):
    """With replica_group set, the comparison runs within the group, whose ranks are numbered from 0. Global rank
    3 (rank 2 of its group) drifted: the alert must name rank 3, the rank event.identity, the launcher and the
    logs use, not rank 2 (another, healthy GPU). The group's own numbering is kept as suspect_group_ranks."""
    _spawn(_replica_group_worker, 4, _port(), str(tmp_path))
    for r in (1, 2, 3):
        res = json.loads((tmp_path / f"rg{r}.json").read_text())
        assert res["identity_rank"] == [3] and res["suspect_ranks"] == [[3]] and res["who"] == [[3]], res
        assert res["group_ranks"] == [[2]], res
        assert "rank 3 in a group of 3" in res["text"][0], res
    assert json.loads((tmp_path / "rg0.json").read_text())["suspect_ranks"] == []     # a group of one: no comparison
