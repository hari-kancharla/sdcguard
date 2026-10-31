"""Protocol-shaped stand-ins; real GPU acceptance remains a separate test."""
from types import SimpleNamespace as NS
import sys
import types

import pytest

from sdcguard.lifetime import vllm_nixl as V
from sdcguard.lifetime.adapter import JsonlDirSink, Probe, merge


class Req:
    def __init__(self, rid):
        self.request_id = rid
        self.status = "WAITING_FOR_REMOTE_KVS"
        self.num_computed_tokens = 16


class Manager:
    def __init__(self, engine_id, n=8):
        self.engine_id, self.free_q, self.held = engine_id, list(range(n)), {}

    def get_block_ids(self, rid):
        return (list(self.held.get(rid, [])),)

    def allocate_slots(self, request, num_blocks, new_computed_blocks=None):
        got = [self.free_q.pop(0) for _ in range(num_blocks)]
        self.held.setdefault(request.request_id, []).extend(got)
        return ("blocks", got)

    def free(self, request):
        self.free_q = self.held.pop(request.request_id, []) + self.free_q


class ConnScheduler:
    engine_id = "P"
    def request_finished(self, request, block_ids):
        return True, {"remote_block_ids": block_ids}


class Agent:
    def __init__(self, agent_name):
        self.batch = {}
    def get_new_notifs(self):
        out, self.batch = self.batch, {}
        return out


class WorkerBase:
    engine_id = "D"
    def __init__(self):
        self._recving_transfers = {}
    def _pop_done_transfers(self, transfers):
        done = set(transfers)
        transfers.clear()
        return done, set()
    def _reap_expired_send_leases(self, done_sending):
        done_sending.add("X")


class PullWorker(WorkerBase):
    Agent = Agent
    def __init__(self):
        super().__init__()
        self.nixl_wrapper = self.Agent("decode-agent")
        self._lease_slots = None
        self._reqs_to_send = {}
        self._remote_agents = {"P": {(0, 0): "prefill-agent"}}
        self.dcp_size = self.pcp_size = self.world_size = self.pp_size = self._physical_blocks_per_logical_kv_block = 1
        self.tp_rank = 0
        self._has_mamba = self._mixed_mem_types = False
        self.block_size = 16
        self.transfer_topo = NS(get_engine_info=lambda _: NS(remote_physical_blocks_per_logical=1,
                                                           remote_dcp_size=1, remote_tp_size=1, remote_block_size=16))
        self.tp_mappings = {"P": NS(all_source_ranks=[0])}

    def _read_blocks_for_req(self, req_id, meta):
        if getattr(self, "early_return", False):
            return None
        self._recving_transfers.setdefault(req_id, [])
        self._read_blocks(read_spec=NS(local_block_ids=meta.local_block_ids,
                                     remote_block_ids=meta.remote.block_ids, remote_rank=0),
                          request_id=req_id)
        return "read-started"

    def _read_blocks(self, read_spec, request_id):
        if getattr(self, "failed_read", False):
            return False
        self._recving_transfers[request_id].append(1)
        return True

    def _get_new_notifs(self):
        released = set()
        for notifs in self.nixl_wrapper.get_new_notifs().values():
            for payload in notifs:
                msg = payload.decode()
                if msg.startswith("HB:"):
                    continue
                if self._lease_slots is None:
                    rid, count = msg.rsplit(":", 1)
                else:
                    rid, count, nonce = _parse_fixed(msg)
                    if nonce is not None and nonce != self._lease_slots.nonce(rid):
                        continue
                if int(count) == 1 and rid in self._reqs_to_send:
                    self._reqs_to_send.pop(rid)
                    released.add(rid)
        return released


def _parse_fixed(msg):
    rid, count, nonce = msg.rsplit(":", 2)
    return rid, int(count), int(nonce) if nonce else None


class Remote:
    def __init__(self, engine_id, request_id, block_ids):
        self.engine_id, self.request_id, self.block_ids = engine_id, request_id, block_ids


class LeasedRemote(Remote):
    def __init__(self, engine_id, request_id, block_ids, slot=0, nonce=7):
        super().__init__(engine_id, request_id, block_ids)
        self.lease_slot, self.lease_nonce = slot, nonce


class Meta:
    def __init__(self, local, remote):
        self.local_block_ids, self.remote = local, remote


class VSched:
    def __init__(self, mgr):
        self.kv_cache_manager = mgr
        self.failed_recving_kv_req_ids = set()
        self.finished_recving_kv_req_ids = set()
        self.recompute_kv_load_failures = True
    def _update_waiting_for_remote_kv(self, request):
        self.failed_recving_kv_req_ids.discard(request.request_id)
        self.finished_recving_kv_req_ids.remove(request.request_id)
        # vLLM 0.32 returns None here, not a readiness boolean.
    def _handle_blocked_waiting_request(self, request):
        if request.status == "WAITING_FOR_REMOTE_KVS":
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            request.status = "WAITING"
        return True
    def _handle_failed_recving(self, failed_req_ids):
        ids = set(failed_req_ids)
        if self.recompute_kv_load_failures:
            self.failed_recving_kv_req_ids |= ids
            return set()
        return ids


class Verifier:
    def __init__(self, observations=None):
        self._cells = [0] * 8
        self.checks, self.verified, self.failures = {}, set(), {}
        self.observations = observations or {}
    def add_check(self, req_id, engine_id, remote_rank, slot, nonce):
        self.checks.setdefault(req_id, []).append((engine_id, remote_rank, slot, nonce))
    def _resolve(self, remote, entries, done, reusable):
        for e in entries:
            if not done or self._cells[e.cell] != e.nonce:
                self.failures[e.req_id] = "completely arbitrary wording"
    def step(self, reads_done):
        for rid in reads_done:
            for i, (eng, rank, slot, nonce) in enumerate(self.checks.pop(rid)):
                observation = self.observations.get((rid, rank), nonce)
                self._cells[i] = observation or 0
                self._resolve((eng, rank), [NS(req_id=rid, slot=slot, cell=i, nonce=nonce)],
                              observation is not None, True)
            if rid not in self.failures:
                self.verified.add(rid)
        out = self.verified, self.failures
        self.verified, self.failures = set(), {}
        return out


@pytest.fixture
def wrapped(tmp_path, monkeypatch):
    d = str(tmp_path / "ev")
    monkeypatch.setenv("SDCGUARD_LIFETIME_DIR", d)
    V.reset()
    # Deterministic synchronous fixture; production uses bounded async recording.
    V._PROBE = Probe(JsonlDirSink(d))
    V._PROBE.clock_id = "fixture-host:one-boot"  # modeled shared clock, including macOS unit tests
    classes = {name: type(name, (cls,), {}) for name, cls in dict(Manager=Manager, ConnScheduler=ConnScheduler,
               WorkerBase=WorkerBase, VSched=VSched, Verifier=Verifier, Agent=Agent).items()}
    classes["PullWorker"] = type("PullWorker", (classes["WorkerBase"], PullWorker), {"Agent": classes["Agent"]})
    # Explicit initialization avoids the stand-in's artificial diamond MRO.
    classes["PullWorker"].__init__ = PullWorker.__init__
    names = V.install_manager(classes["Manager"])
    names += V.install_connector(classes["ConnScheduler"], classes["WorkerBase"], classes["PullWorker"])
    names += V.install_scheduler(classes["VSched"])
    names += V.install_verifier(classes["Verifier"])
    names += V.install_nixl_agent(classes["Agent"])
    V._PROBE._emit("note", "hook coverage", hooks=names)
    mod = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata")
    mod.parse_read_notif = _parse_fixed
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    yield classes, names, d
    V.reset()


def _events(d):
    p = V.probe()
    p.evidence("recording_health", healthy=p._dropped + getattr(p.sink, "errors", 0) == 0)
    p.sink.flush()
    return JsonlDirSink(d).history()


def _evidence(d, what):
    return [e["args"] for e in _events(d) if e["kind"] == "evidence" and e["args"]["what"] == what]


def _findings(d, rule):
    from sdcguard.lifetime.obligations import evaluate
    return [f for f in evaluate(_events(d), merge(d, track_content=False).violations) if f.rule == rule]


def run_pair(c, race=False, leased=False):
    pm, dm = c["Manager"]("P"), c["Manager"]("D")
    psched, worker, sched = c["ConnScheduler"](), c["PullWorker"](), c["VSched"](dm)
    req = Req("X")
    pm.allocate_slots(req, 2)
    psched.request_finished(req, pm.get_block_ids("X"))
    dm.allocate_slots(req, 2)
    if race:
        pm.free(req)
        pm.allocate_slots(Req("Y"), 2)
    worker._read_blocks_for_req("X", Meta(dm.get_block_ids("X"), (LeasedRemote if leased else Remote)("P", "X", [0, 1])))
    worker._pop_done_transfers(worker._recving_transfers)
    return pm, worker, sched, req


def admit(sched, req):
    sched.finished_recving_kv_req_ids.add(req.request_id)
    return sched._handle_blocked_waiting_request(req)


def test_installs_once_and_preserves_results(wrapped):
    c, names, _ = wrapped
    assert len(names) == 16
    assert V.install_manager(c["Manager"]) == []
    m = c["Manager"]("P")
    assert m.allocate_slots(Req("X"), 2) == ("blocks", [0, 1])
    assert c["ConnScheduler"]().request_finished(Req("X"), [0, 1]) == (True, {"remote_block_ids": [0, 1]})
    assert c["PullWorker"]()._read_blocks_for_req("X", Meta([5], Remote("P", "X", [0]))) == "read-started"


def test_normal_pull_records_real_submission_then_completion_then_admission(wrapped):
    c, _, d = wrapped
    _, _, sched, req = run_pair(c)
    assert not _evidence(d, "admit")
    assert admit(sched, req) is True
    a = _evidence(d, "attempt")[0]
    assert a["mapping_complete"] and a["source_indices"] == [0, 1]
    assert a["reader_agent"] == "decode-agent"
    assert a["source_regions"] == [["P", 0], ["P", 1]]
    assert _evidence(d, "read_submission_result")[0]["submitted"] is True
    assert _evidence(d, "transfer_done")[0]["op_ids"] == a["op_ids"]
    assert _evidence(d, "admit")[0]["transferred"] is True
    assert merge(d, track_content=False).violations == []
    (f,) = _findings(d, "lease_valid_at_use")
    assert f.verdict == "satisfied", f.as_dict()


def test_stale_read_at_submission_is_detected(wrapped):
    c, _, d = wrapped
    _, _, sched, req = run_pair(c, race=True)
    admit(sched, req)
    assert merge(d, track_content=False).violations[0].rule == "R1"
    (f,) = _findings(d, "lease_valid_at_use")
    assert f.verdict == "violated", f.as_dict()


def test_failed_or_early_read_does_not_claim_transport_submission(wrapped):
    c, _, d = wrapped
    w = c["PullWorker"]()
    w.early_return = True
    w._read_blocks_for_req("X", Meta([4], Remote("P", "X", [0])))
    assert not _evidence(d, "read_submission_result")
    w.early_return, w.failed_read = False, True
    w._read_blocks_for_req("Y", Meta([4], Remote("P", "Y", [0])))
    assert _evidence(d, "read_submission_result")[0]["submitted"] is False


@pytest.mark.parametrize("local,remote", [([4], [0, 1]), ([], [0]), ([[4], [5]], [[0], [1]]), ([4, 4], [0, 1])])
def test_unsupported_mapping_never_fabricates_source_submits(wrapped, local, remote):
    c, _, d = wrapped
    c["PullWorker"]()._read_blocks_for_req("X", Meta(local, Remote("P", "X", remote)))
    assert _evidence(d, "attempt")[0]["mapping_complete"] is False
    assert not [e for e in _events(d) if e["kind"] == "submit"]


def test_repeated_internal_request_is_ambiguous(wrapped):
    c, _, d = wrapped
    w = c["PullWorker"]()
    for _ in range(2):
        w._read_blocks_for_req("X", Meta([4], Remote("P", "X", [0])))
    assert _evidence(d, "attempt")[-1]["mapping_complete"] is False


def test_scheduler_boolean_not_update_none_controls_admission(wrapped):
    c, _, d = wrapped
    s, r = c["VSched"](c["Manager"]("D")), Req("X")
    assert s._handle_blocked_waiting_request(r) is False
    assert not _evidence(d, "admit")
    s.finished_recving_kv_req_ids.add("X")
    assert s._update_waiting_for_remote_kv(r) is None
    assert not _evidence(d, "admit")
    assert admit(s, r) is True
    assert len(_evidence(d, "admit")) == 1


@pytest.mark.parametrize("observed,cause", [(7, "matched"), (0, "lease_mismatch"), (None, "check_unavailable")])
def test_verifier_uses_structured_cpu_cells_not_reason_words(wrapped, observed, cause):
    c, _, d = wrapped
    v = c["Verifier"]({("X", 0): observed})
    v.add_check("X", "P", 0, 3, 7)
    v.step(iter(["X"]))
    a = _evidence(d, "lease_check")[0]
    assert a["cause"] == cause and a["details_complete"] is True
    detail = _evidence(d, "lease_check_detail")[0]
    assert detail["observed_nonce"] == observed
    assert detail["remote_rank"] == 0 and detail["lease_slot"] == 3


def test_verifier_all_ranks_and_io_failure_not_inferred_stale(wrapped):
    c, _, d = wrapped
    v = c["Verifier"]({("X", 1): None})
    v.add_check("X", "P", 0, 3, 7)
    v.add_check("X", "P", 1, 3, 7)
    v.step({"X"})
    a = _evidence(d, "lease_check")[0]
    assert a["cause"] == "check_unavailable"
    assert a["expected_ranks"] == [["P", 0], ["P", 1]]


def test_fixed_recompute_reports_rejection_not_unobserved_recovery(wrapped):
    c, _, d = wrapped
    _, _, s, r = run_pair(c, race=True, leased=True)
    v = c["Verifier"]({("X", 0): 0})
    v.add_check("X", "P", 0, 0, 7)
    v.step({"X"})
    s._handle_failed_recving({"X"})
    admit(s, r)
    assert _evidence(d, "admit")[0]["transferred"] is False
    assert _evidence(d, "admit")[0]["recompute_pending"] is True
    assert not _evidence(d, "recompute_done")
    (f,) = _findings(d, "lease_valid_at_use")
    assert f.verdict == "correctly_rejected", f.as_dict()


def test_fail_policy_does_not_claim_closure_before_the_caller_closes(wrapped):
    c, _, d = wrapped
    s = c["VSched"](c["Manager"]("D"))
    s.recompute_kv_load_failures = False
    assert s._handle_failed_recving({"X"}) == {"X"}
    assert _evidence(d, "load_failed")[0]["policy"] == "fail"
    assert not _evidence(d, "admit")


def test_failed_ids_iterator_not_drained_before_original(wrapped):
    c, _, _ = wrapped
    s = c["VSched"](c["Manager"]("D"))
    s._handle_failed_recving(iter(["X", "Y"]))
    assert s.failed_recving_kv_req_ids == {"X", "Y"}


def test_notification_has_wire_sender_and_nonce_not_current_nonce(wrapped, monkeypatch):
    c, _, d = wrapped
    w = c["PullWorker"]()
    # The PR implementation imports parse_read_notif into its method's globals.
    original = c["PullWorker"]._get_new_notifs.__wrapped__
    monkeypatch.setitem(original.__globals__, "parse_read_notif", _parse_fixed)
    calls = []
    w._lease_slots = NS(nonce=lambda rid: calls.append(rid) or 9)
    w._reqs_to_send = {str(i): 1 for i in range(10000)}
    w._reqs_to_send["X"] = 1
    w.nixl_wrapper.batch = {"old-reader": [b"X:1:7"]}
    assert w._get_new_notifs() == set()
    got = _evidence(d, "read_notification_received")[0]
    assert got["notification_nonce"] == 7 and got["current_nonce"] == 9 and got["sender_agent"] == "old-reader"
    assert calls == ["X", "X"]  # one observer read, one original read; no scan of pending requests
    assert not _evidence(d, "released_by_read_notification")
    w.nixl_wrapper.batch = {"new-reader": [b"X:1:9"]}
    assert w._get_new_notifs() == {"X"}
    r = _evidence(d, "released_by_read_notification")[0]
    assert r["exact_mapping"] and r["transport_complete"]
    assert r["notification_nonce"] == 9 and r["sender_agent"] == "new-reader"


def test_multiple_matching_notifications_are_ambiguous(wrapped):
    c, _, d = wrapped
    w = c["PullWorker"]()
    w._reqs_to_send["X"] = 1
    w.nixl_wrapper.batch = {"a": [b"X:1"], "b": [b"X:1"]}
    assert w._get_new_notifs() == {"X"}
    r = _evidence(d, "released_by_read_notification")[0]
    assert not r["exact_mapping"] and r["sender_agent"] is None


def test_recorder_exceptions_never_call_original_twice_or_hide_its_error(wrapped, monkeypatch):
    c, _, d = wrapped
    p = V.probe()
    def boom(*a, **k):
        raise OSError("disk unavailable")
    monkeypatch.setattr(p, "alloc", boom)
    m = c["Manager"]("P")
    assert m.allocate_slots(Req("X"), 1) == ("blocks", [0])
    assert m.held["X"] == [0] and p._dropped > 0
    # Original exception is still raised, and does not become a recorder error.
    before = p._dropped
    with pytest.raises(IndexError):
        m.allocate_slots(Req("Y"), 100)
    assert p._dropped == before


def test_free_event_is_after_successful_actual_free(wrapped, monkeypatch):
    c, _, _ = wrapped
    m, r = c["Manager"]("P"), Req("X")
    m.allocate_slots(r, 1)
    original = V.probe().free
    def inspect_free(region, by):
        assert "X" not in m.held
        original(region, by)
    monkeypatch.setattr(V.probe(), "free", inspect_free)
    m.free(r)


def test_attempt_is_after_the_hold(wrapped, monkeypatch):
    c, _, d = wrapped
    def hold(point, rid, **ctx):
        V.probe().evidence("hold_witness", request=rid)
    monkeypatch.setattr(V.probe(), "checkpoint", hold)
    c["PullWorker"]()._read_blocks_for_req("X", Meta([4], Remote("P", "X", [0])))
    events = _events(d)
    h = next(e["seq"] for e in events if e["args"].get("what") == "hold_witness")
    a = next(e["seq"] for e in events if e["args"].get("what") == "attempt")
    assert h < a


def test_disabled_has_no_probe(monkeypatch):
    monkeypatch.delenv("SDCGUARD_LIFETIME_DIR", raising=False)
    V.reset()
    assert V.probe() is None and V.install() == []
def test_ucx_params_reach_every_nixl_ucx_backend(monkeypatch):
    """The test-environment UCX setting (vLLM PR #62905 proposes a key for it) is merged into the UCX backend's
    init parameters, leaves other backends and explicit values alone, and is applied once."""
    import sys
    import types

    calls = []

    class Agent:
        def create_backend(self, backend, initParams={}):  # noqa: B006  (NIXL's own signature)
            calls.append((backend, dict(initParams)))

    nixl = types.ModuleType("nixl")
    api = types.ModuleType("nixl._api")
    api.nixl_agent = Agent
    nixl._api = api
    monkeypatch.setitem(sys.modules, "nixl", nixl)
    monkeypatch.setitem(sys.modules, "nixl._api", api)
    want = {"ucx_error_handling_mode": "none"}
    assert V.apply_ucx_params(want) and V.apply_ucx_params(want)          # idempotent
    Agent().create_backend("UCX", {"num_threads": "4"})
    Agent().create_backend("UCX", {"ucx_error_handling_mode": "peer"})
    Agent().create_backend("GDS", {})
    assert calls == [("UCX", {"num_threads": "4", "ucx_error_handling_mode": "none"}),
                     ("UCX", {"ucx_error_handling_mode": "peer"}), ("GDS", {})]
    assert not V.apply_ucx_params({})



def test_completed_read_notification_precedes_free_and_late_completion_poll(wrapped):
    """H1: producer safely reuses after the READ notifies, before decode polls DONE."""
    c, _, d = wrapped
    pm, dm = c["Manager"]("P"), c["Manager"]("D")
    r = Req("X")
    pm.allocate_slots(r, 1)
    c["ConnScheduler"]().request_finished(r, pm.get_block_ids("X"))
    dm.allocate_slots(r, 1)
    dworker, pworker = c["PullWorker"](), c["PullWorker"]()
    pworker.engine_id = "P"
    dworker._read_blocks_for_req("X", Meta(dm.get_block_ids("X"), Remote("P", "X", [0])))
    pworker._reqs_to_send["X"] = 1
    pworker.nixl_wrapper.batch = {"decode-agent": [b"X:1"]}
    assert pworker._get_new_notifs() == {"X"}
    pm.free(r)
    pm.allocate_slots(Req("Y"), 1)
    dworker._pop_done_transfers(dworker._recving_transfers)
    admit(c["VSched"](dm), r)
    assert any(v.rule == "R2" for v in merge(d, track_content=False).violations)
    (f,) = _findings(d, "lease_valid_at_use")
    assert f.verdict == "satisfied", f.as_dict()


def test_failed_scheduler_promotion_does_not_admit(wrapped):
    c, _, d = wrapped
    class Broken(VSched):
        def _handle_blocked_waiting_request(self, request):
            self._update_waiting_for_remote_kv(request)
            return False
    V.install_scheduler(Broken)
    s, r = Broken(c["Manager"]("D")), Req("X")
    s.finished_recving_kv_req_ids.add("X")
    assert s._handle_blocked_waiting_request(r) is False
    assert not _evidence(d, "admit")


def test_observer_failure_cannot_escape_any_event_wrapper(wrapped, monkeypatch):
    c, _, _ = wrapped
    p = V.probe()
    def boom(*a, **k):
        raise OSError("evidence full")
    monkeypatch.setattr(p, "evidence", boom)
    pm, worker, sched, req = run_pair(c)
    assert admit(sched, req)
    v = c["Verifier"]()
    v.add_check("X", "P", 0, 0, 7)
    assert v.step({"X"}) == ({"X"}, {})
    worker._reqs_to_send["X"] = 1
    worker.nixl_wrapper.batch = {"decode-agent": [b"X:1"]}
    assert worker._get_new_notifs() == {"X"}
    worker._reap_expired_send_leases(set())
    sched._handle_failed_recving({"X"})
    assert p._dropped > 0


def test_original_transport_exception_is_not_swallowed_or_retried(wrapped):
    c, _, _ = wrapped
    class Broken(PullWorker):
        calls = 0
        def _read_blocks(self, read_spec, request_id):
            self.calls += 1
            raise RuntimeError("original transport failure")
    V.install_connector(PullWorker=Broken)
    worker = Broken()
    with pytest.raises(RuntimeError, match="original transport failure"):
        worker._read_blocks_for_req("X", Meta([4], Remote("P", "X", [0])))
    assert worker.calls == 1


def test_actual_hold_precedes_attempt_and_source_submission(wrapped, tmp_path, monkeypatch):
    import json
    import threading
    import time
    from sdcguard.lifetime.realsearch import candidates
    c, _, d = wrapped
    V.probe().record_checkpoints = True
    _, _, s, r = run_pair(c)
    admit(s, r)
    baseline = _events(d)
    candidate = next(x for x in candidates(baseline, symbolic=True)
                     if x.kind == "realloc" and "desc_block" in json.dumps(x.spec))
    path = tmp_path / "holds.json"
    path.write_text(json.dumps([candidate.hold_spec(3.0)]))
    newdir = str(tmp_path / "held")
    monkeypatch.setenv("SDCGUARD_LIFETIME_DIR", newdir)
    monkeypatch.setenv("SDCGUARD_LIFETIME_HOLDS", str(path))
    V.reset()
    pm, dm = c["Manager"]("P"), c["Manager"]("D")
    req = Req("X2")
    pm.allocate_slots(req, 1)
    c["ConnScheduler"]().request_finished(req, pm.get_block_ids("X2"))
    dm.allocate_slots(req, 1)
    w = c["PullWorker"]()
    thread = threading.Thread(target=w._read_blocks_for_req,
                              args=("X2", Meta(dm.get_block_ids("X2"), Remote("P", "X2", [0]))))
    thread.start()
    time.sleep(0.1)
    assert thread.is_alive()
    pm.free(req)
    pm.allocate_slots(Req("Y2"), 1)
    thread.join(4)
    assert not thread.is_alive()
    events = _events(newdir)
    att = next(e for e in events if e["args"].get("what") == "attempt")
    released = next(e for e in events if e["kind"] == "released" and "condition reached" in e["text"])
    assert att["mono"] > released["mono"]
    assert merge(newdir, track_content=False).violations[0].rule == "R1"


def test_first_peer_handshake_is_not_a_recording_error_or_duplicate_attempt(wrapped):
    c, _, d = wrapped
    class Handshake(PullWorker):
        def _read_blocks_for_req(self, req_id, meta):
            if meta.remote.engine_id not in self._remote_agents:
                self.handshake_queued = True
                return None
            return super()._read_blocks_for_req(req_id, meta)
    V.install_connector(PullWorker=Handshake)
    w = Handshake()
    w._remote_agents = {}
    w.tp_mappings = {}
    w.transfer_topo = NS(get_engine_info=lambda _: (_ for _ in ()).throw(KeyError("unknown peer")))
    meta = Meta([4], Remote("P", "X", [0]))
    assert w._read_blocks_for_req("X", meta) is None
    assert w.handshake_queued and not _evidence(d, "attempt")
    assert V.probe()._dropped == 0
    w._remote_agents["P"] = {(0, 0): "prefill-agent"}
    w.tp_mappings["P"] = NS(all_source_ranks=[0])
    w.transfer_topo = NS(get_engine_info=lambda _: NS(remote_physical_blocks_per_logical=1,
                                                    remote_dcp_size=1, remote_tp_size=1, remote_block_size=16))
    assert w._read_blocks_for_req("X", meta) == "read-started"
    assert len(_evidence(d, "attempt")) == 1
    assert _evidence(d, "attempt")[0]["mapping_complete"] is True
    assert V.probe()._dropped == 0


def test_fork_replays_inherited_hook_coverage_in_child_stream(wrapped, tmp_path, monkeypatch):
    """Model a pid change without forking pytest's live writer/runner threads."""
    c, _, _ = wrapped
    parent = V.probe()
    parent.evidence("parent_only")
    parent.sink.flush()
    parent_pid = V.os.getpid()
    V._PROBE_PID = parent_pid
    child_dir = str(tmp_path / "child")
    monkeypatch.setenv("SDCGUARD_LIFETIME_DIR", child_dir)
    monkeypatch.setattr(V.os, "getpid", lambda: parent_pid + 100000)
    child = V.probe()
    assert child is not parent and child.pid == parent_pid + 100000
    assert child.session != parent.session
    assert V.install_manager(c["Manager"]) == []  # inherited wrappers are already installed
    mgr = c["Manager"]("P")
    mgr.allocate_slots(Req("X"), 1)
    c["ConnScheduler"]().request_finished(Req("X"), mgr.get_block_ids("X"))
    records = _events(child_dir)
    notes = [e for e in records if e["kind"] == "note" and "hooks" in e["args"]]
    pub = next(e for e in records if e["kind"] == "publish")
    assert notes and notes[0]["seq"] < pub["seq"]
    assert {"KVCacheManager.allocate_slots", "KVCacheManager.free"} <= set(notes[0]["args"]["hooks"])
    assert all(e["pid"] == child.pid and e["session"] == child.session for e in records)
    assert not any(e["args"].get("what") == "parent_only" for e in records)
    # Real fork leaves parent's recorder running in the parent. This simulation
    # must explicitly close the original writer, which is no longer V._PROBE.
    parent.close()


def test_registry_does_not_claim_a_replaced_wrapper(wrapped, monkeypatch):
    c, _, _ = wrapped
    import weakref
    # Isolate this identity check from other stand-in classes in the fixture.
    registry = weakref.WeakKeyDictionary({c["Manager"]: dict(V._INSTALLED_HOOKS[c["Manager"]])})
    monkeypatch.setattr(V, "_INSTALLED_HOOKS", registry)
    assert "KVCacheManager.free" in V._installed_hook_names()
    monkeypatch.setattr(c["Manager"], "free", Manager.free)
    assert "KVCacheManager.free" not in V._installed_hook_names()
    assert "KVCacheManager.allocate_slots" in V._installed_hook_names()
