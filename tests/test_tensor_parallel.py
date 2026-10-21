"""Tensor parallel (DTensor): each rank checks its own shard of every column- and row-parallel matmul, a wrong
value in one rank's shard is caught on that rank, gradients are unchanged, and shardings the local check can't
cover are skipped and counted instead of crashing the step. Two CPU ranks over gloo."""
import json
import os
import socket

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

pytest.importorskip("torch.distributed.tensor.parallel")


class _CorruptLocal(TorchDispatchMode):
    """Faulty hardware on one rank: adds 8 to one element of the local shard of every matmul output."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if str(func).startswith(("aten.mm", "aten.addmm")) and hasattr(out, "to_local"):
            with torch.no_grad():
                out.to_local().view(-1)[3] += 8.0
        return out


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256)).bfloat16()


def _worker(rank, world, port, out, mode, faulty):
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
        from sdcguard import SDCGuard
        mesh = init_device_mesh("cpu", (world,))
        plan = {"0": ColwiseParallel(), "2": RowwiseParallel()}
        ref = parallelize_module(_model(), mesh, plan)          # the same tensor-parallel model, unguarded
        m = parallelize_module(_model(), mesh, plan)
        g = SDCGuard()
        if mode == "fast":
            g.protect(m)
        torch.manual_seed(1)
        x = torch.randn(4, 16, 256, dtype=torch.bfloat16)
        inj = _CorruptLocal() if faulty and rank == 1 else None
        if inj is not None:
            inj.__enter__()
        try:
            with g.at_step(0):
                y = m(x)
                y.float().pow(2).mean().backward()
        finally:
            if inj is not None:
                inj.__exit__(None, None, None)
        ref(x).float().pow(2).mean().backward()
        same = all(torch.equal(a.grad.full_tensor(), b.grad.full_tensor())
                   for a, b in zip(m.parameters(), ref.parameters())) if not faulty else None
        with open(os.path.join(out, f"{mode}_{faulty}_{rank}.json"), "w") as f:
            json.dump({"stats": {k: v for k, v in g.stats.items() if isinstance(v, int)}, "grads_same": same}, f)
    finally:
        dist.destroy_process_group()


def _run(tmp_path, mode, faulty):
    import torch.multiprocessing as mp
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(_worker, args=(2, port, str(tmp_path), mode, faulty), nprocs=2, join=True)
    return [json.loads((tmp_path / f"{mode}_{faulty}_{r}.json").read_text()) for r in range(2)]


@pytest.mark.parametrize("mode", ["fast", "default"])
def test_each_rank_checks_its_shard(tmp_path, mode):
    for r in _run(tmp_path, mode, False):
        st = r["stats"]
        assert st.get("checked_dtensor", 0) >= 4 and st["alerts"] == 0, st          # 2 forward + backward
        assert not st.get("skipped_dtensor"), st
        assert r["grads_same"] is True


@pytest.mark.parametrize("mode", ["fast", "default"])
def test_a_wrong_value_in_one_shard_is_caught_on_that_rank(tmp_path, mode):
    clean, bad = _run(tmp_path, mode, True)
    assert clean["stats"]["alerts"] == 0
    assert bad["stats"]["alerts"] >= 1


def test_unsupported_placements_are_skipped_not_raised():
    """The placement rule by itself: a pattern whose local pieces aren't a matmul is a reason, not a crash."""
    from sdcguard.guard import _placement_role
    from torch.distributed.tensor import Partial, Replicate, Shard
    assert _placement_role(Replicate(), ("m", "k")) == "R"
    assert _placement_role(Shard(1), ("m", "k")) == ("S", "k")
    assert _placement_role(Shard(-1), ("k", "n")) == ("S", "n")
    assert _placement_role(Partial(), ("m", "n")) == "P"
    assert _placement_role(Partial("max"), ("m", "n")) is None


def test_a_local_bias_that_doesnt_broadcast_to_the_shard_is_rejected(monkeypatch):
    """A replicated (4, 3) bias on a row-sharded (2, 3) output: the old check looked at the last dimension only."""
    from torch.distributed.tensor import Replicate, Shard
    from sdcguard import guard

    class Mesh:
        ndim = 1

    class Fake:
        def __init__(self, x, pl, mesh):
            self.x, self.placements, self.device_mesh = x, (pl,), mesh

        def to_local(self):
            return self.x

        def dim(self):
            return self.x.dim()

        @property
        def shape(self):
            return self.x.shape
    monkeypatch.setattr(guard, "_is_dtensor", lambda t: isinstance(t, Fake))
    mesh = Mesh()
    a, b, c = Fake(torch.ones(2, 2), Shard(0), mesh), Fake(torch.ones(2, 3), Replicate(), mesh), \
        Fake(torch.ones(2, 3), Shard(0), mesh)
    assert isinstance(guard._dtensor_local(a, b, c, Fake(torch.ones(4, 3), Replicate(), mesh)), str)
    assert not isinstance(guard._dtensor_local(a, b, c, Fake(torch.ones(3), Replicate(), mesh)), str)


def _collectives(graphs) -> list:
    """The communication ops in captured graphs, in order (the guarded and the plain model must match)."""
    return [str(n.target) for g in graphs for n in g.graph.nodes
            if n.op == "call_function" and "c10d" in str(n.target)]


def _compiled_worker(rank, world, port, out, precise, faulty):
    import warnings
    import torch.distributed as dist
    import torch.nn.functional as F
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch._dynamo.backends.common import aot_autograd
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
        from sdcguard import SDCGuard

        def capture(into):
            def keep(gm, _inputs):
                into.append(gm)
                return gm.forward
            return aot_autograd(fw_compiler=keep, bw_compiler=keep)
        mesh = init_device_mesh("cpu", (world,))
        plan = {"0": ColwiseParallel(), "2": RowwiseParallel()}
        plain = parallelize_module(_model(), mesh, plan)
        m = parallelize_module(_model(), mesh, plan)
        g = SDCGuard(every=1, precise=precise).protect(m)
        plain_graphs, graphs = [], []
        fp = torch.compile(plain, backend=capture(plain_graphs), fullgraph=True)
        f = torch.compile(m, backend=capture(graphs), fullgraph=True)
        torch.manual_seed(1)
        x = torch.randn(4, 16, 256, dtype=torch.bfloat16)
        real = F.linear

        def wrong(a, w, b=None):                  # rank 1's GPU computes one output wrong (inside the operator)
            y = real(a, w, b)
            if rank == 1:
                y = y.clone()
                y.view(-1, y.shape[-1])[3, 5] += 50.0
            return y
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for step in range(2):
                fp(x).float().sum().backward()
                if faulty and step == 1:
                    F.linear = wrong
                try:
                    with g.on_step(step):
                        f(x).float().sum().backward()
                finally:
                    F.linear = real
        msgs = [str(m.message) for m in w if "sdcguard" in str(m.message)]
        same = all(torch.equal(a.grad.full_tensor(), b.grad.full_tensor())
                   for a, b in zip(m.parameters(), plain.parameters()))
        close = all(torch.allclose(a.grad.full_tensor().float(), b.grad.full_tensor().float(), rtol=2e-2, atol=2e-2)
                    for a, b in zip(m.parameters(), plain.parameters()))
        y, yp = f(x), fp(x)                      # (RowwiseParallel hands back a plain, replicated tensor)
        with open(os.path.join(out, f"c{rank}.json"), "w") as fh:
            json.dump({"stats": {k: v for k, v in g.stats.items() if isinstance(v, int)}, "warnings": msgs,
                       "grads_equal": same, "outputs_equal": bool(torch.equal(y, yp)), "grads_close": close,
                       "outputs_close": bool(torch.allclose(y.float(), yp.float(), rtol=2e-2, atol=2e-2)),
                       "collectives": _collectives(graphs[:2]), "plain_collectives": _collectives(plain_graphs[:2]),
                       "alert_steps": [e.step for e in g.log.events]}, fh)
    finally:
        dist.destroy_process_group()


def _spawn_compiled(tmp_path, world, precise=False, faulty=False):
    import torch.multiprocessing as mp
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(_compiled_worker, args=(world, port, str(tmp_path), precise, faulty), nprocs=world, join=True)
    return [json.loads((tmp_path / f"c{r}.json").read_text()) for r in range(world)]


@pytest.mark.parametrize("precise", [False, True])
def test_tensor_parallel_inside_a_compiled_graph_is_checked_on_each_ranks_shards(tmp_path, precise):
    """Real DTensors (column then row parallel, with biases) under torch.compile(fullgraph=True) on 2 ranks: the
    checked operator runs on each rank's local shards. Every layer's matmuls are checked (forward, dX where
    needed, dW: 5 per step), nothing is skipped, the outputs and gradients are bit-identical to the same model
    compiled without the guard (precise mode rounds the checked fp32 result once, so there they are close), and
    DTensor adds no communication to the graph."""
    for r in _spawn_compiled(tmp_path, 2, precise=precise):
        st = r["stats"]
        assert st["checked"] == 10 and st["checked_dtensor"] == 10, st
        assert not st.get("skipped_dtensor_compiled") and not st.get("skipped_dtensor") and st["alerts"] == 0
        if precise:
            assert r["grads_close"] and r["outputs_close"]
        else:
            assert r["grads_equal"] and r["outputs_equal"]
        assert r["collectives"] == r["plain_collectives"] and r["collectives"]
        assert not any("weren't checked" in m or "ran no checks" in m for m in r["warnings"])


def test_a_wrong_local_shard_inside_a_compiled_graph_alerts_on_its_own_rank(tmp_path):
    r0, r1 = _spawn_compiled(tmp_path, 2, faulty=True)
    assert r0["stats"]["alerts"] == 0
    assert r1["stats"]["alerts"] == 2 and set(r1["alert_steps"]) == {1}   # both layers' forward outputs on rank 1


def _fp32_worker(rank, world, port, out):
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        import warnings
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
        from sdcguard import SDCGuard
        mesh = init_device_mesh("cpu", (world,))
        torch.manual_seed(0)
        m = parallelize_module(torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(),
                                                   torch.nn.Linear(128, 64)), mesh,
                               {"0": ColwiseParallel(), "2": RowwiseParallel()})
        g = SDCGuard()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with g.at_step(0):
                m(torch.randn(4, 64)).sum().backward()            # float32: nothing can be checked
        with open(os.path.join(out, f"tp{rank}.json"), "w") as f:
            json.dump({k: v for k, v in g.stats.items() if isinstance(v, int)}, f)
    finally:
        dist.destroy_process_group()


def test_checked_dtensor_counts_only_matmuls_that_were_checked(tmp_path):
    """stats["checked_dtensor"] is the proof that shards were checked (the tests above and experiments/v040_tp.py
    rely on it), so it counts a tensor-parallel matmul only once its check was queued, not one that was then left
    out (float32 here, an empty output, out of memory)."""
    import torch.multiprocessing as mp
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(_fp32_worker, args=(1, port, str(tmp_path)), nprocs=1, join=True)
    st = json.loads((tmp_path / "tp0.json").read_text())
    assert st["matmuls_seen"] > 0 and st["skipped_dtype"] > 0, st
    assert st["checked"] == 0 and st.get("checked_dtensor", 0) == 0, st


def _gate_worker(rank, world, port, out, mode):
    """The 4-block column/row-parallel stack experiments/v040_tp.py runs on two H100s, small, with the grouped GPU
    check made available and a spy on its gate: what reaches it, and whether each tensor has a real address."""
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
        from sdcguard import SDCGuard, _group
        from sdcguard import checksum as _cs
        mesh = init_device_mesh("cpu", (world,))
        torch.manual_seed(0)
        mods, plan = [], {}
        for i in range(4):
            mods += [torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)]
            plan[str(3 * i)], plan[str(3 * i + 2)] = ColwiseParallel(), RowwiseParallel()
        m = parallelize_module(torch.nn.Sequential(*mods).bfloat16(), mesh, plan)
        seen = []

        def spy(A, B, C, bias, beta, block=128):
            seen.append({"types": sorted({type(t).__name__ for t in (A, B, C, bias) if t is not None}),
                         "addressed": all(t is None or t.numel() == 0 or t.data_ptr() != 0 for t in (A, B, C, bias))})
            return False                        # the plain check then runs, as on a CPU
        _cs.USE_FAST = True
        _group.available = lambda C: True
        _group.supports = spy
        g = SDCGuard(every=1)
        if mode == "fast":
            g.protect(m)
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        for step in range(2):
            with g.on_step(step):
                m(x).float().pow(2).mean().backward()
            m.zero_grad(set_to_none=True)
        with open(os.path.join(out, f"gate_{mode}_{rank}.json"), "w") as f:
            json.dump({"seen": seen, "stats": {k: v for k, v in g.stats.items() if isinstance(v, int)}}, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["fast", "default"])
def test_every_shard_reaching_the_gpu_check_has_its_own_memory(tmp_path, mode):
    """The two-H100 run (October 21, 2026) ended in an illegal memory access at address 0 on both GPUs: in this
    stack DTensor hands some local operands over as AsyncCollectiveTensors (no memory until their collective is
    done, data_ptr() 0), and the grouped Triton check was handed them. Every one is now waited for first: what
    reaches the GPU check is the tensor itself, every matmul is checked as a local shard, and nothing alarms."""
    import torch.multiprocessing as mp
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    mp.spawn(_gate_worker, args=(2, port, str(tmp_path), mode), nprocs=2, join=True)
    for r in range(2):
        d = json.loads((tmp_path / f"gate_{mode}_{r}.json").read_text())
        per_step = 3 * 8 - 1                    # forward, dW and dX of 8 layers, no dX for the first
        assert len(d["seen"]) == 2 * per_step and all(x["addressed"] for x in d["seen"]), d["seen"]
        assert all(x["types"] == ["Tensor"] for x in d["seen"]), d["seen"]
        st = d["stats"]
        assert st["checked"] == st["checked_dtensor"] == 2 * per_step and st["alerts"] == 0, st
