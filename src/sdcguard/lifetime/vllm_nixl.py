"""Read-only observations of the vLLM 0.32 NIXL pull path.

Facts are deliberately narrower than the transport's capabilities. Block correspondence
is supported for one cache group/source rank, equal physical block sizes, and no local
prefix hit or DCP. Other layouts remain unknown. Transport completion, lease validation,
and scheduler admission are separate facts; attention's actual use is not observed.
Recorder failures poison evidence health but never escape into vLLM. Wrapped methods
are always called exactly once, with their original arguments and exception behavior.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import threading
import weakref

from .adapter import JsonlDirSink, Probe, desc_ref, holds_from_file

_PROBE = None
_PROBE_PID = None
_STATE = {"remote": {}, "owned": {}, "agents": {}, "checks": {}, "resolved": {}, "failed_terminal": {}}
_CONTEXT = threading.local()
# Wrappers survive fork/reset, unlike per-process recording state. Keep their
# identities as well as names so a subsequently replaced method is not claimed
# as covered. Weak class keys avoid retaining temporary stand-in classes.
_INSTALLED_HOOKS = weakref.WeakKeyDictionary()
# Only explicitly registered, exact observer compositions can extend coverage.
_COMPOSED_HOOKS = weakref.WeakKeyDictionary()


def _closure(function):
    if not inspect.isfunction(function):
        return None
    return dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in (function.__closure__ or ()))))


def _nested_code(function, name):
    values = [value for value in function.__code__.co_consts
              if type(value) is type(function.__code__) and value.co_name == name]
    if len(values) != 1:
        raise ValueError("source-reviewed composition code is unavailable")
    return values[0]


def _composition_matches(cls, name, inner, outer):
    entry = _COMPOSED_HOOKS.get(cls, {}).get(name)
    if entry is None:
        return False
    if getattr(cls, name, None) is not entry["outer"]:
        entry["healthy"] = False
    if entry["inner"] is not inner or entry["outer"] is not outer or not entry["healthy"]:
        return False
    try:
        valid = (getattr(cls, name, None) is outer and getattr(outer, "__wrapped__", None) is inner
                and inner.__code__ is entry["inner_code"] and outer.__code__ is entry["outer_code"]
                and entry["supplied"].__code__ is entry["supplied_code"]
                and entry["native"].__code__ is entry["native_code"]
                and _closure(inner).get("fn") is entry["native"]
                and getattr(inner, "__wrapped__", None) is entry["native"]
                and _closure(outer).get("original") is inner
                and _closure(outer).get("supplied") is entry["supplied"]
                and _closure(outer).get("self") is entry["mapping"]()
                and _closure(entry["supplied"]).get("original") is inner
                and _closure(entry["supplied"]).get("self") is entry["recipient"]())
    except (AttributeError, ValueError, TypeError):
        valid = False
    if not valid:
        entry["healthy"] = False
    return valid


def register_composition(cls, name, inner, outer):
    """Track only the exact reviewed recipient/mapping delegation around alloc.

    This validates local callable bookkeeping. It is not remote authentication.
    No arbitrary marker or generic unwrapping establishes coverage.
    """
    from .mapping_observer import MappingObserver
    from .allocation_recipient_observer import AllocationRecipientObserver
    tracked = _INSTALLED_HOOKS.get(cls, {}).get(name)
    outer_cells, inner_cells = _closure(outer), _closure(inner)
    if (name != "allocate_slots" or tracked is None or tracked[0] is not inner
            or getattr(cls, name, None) is not outer or getattr(outer, "__wrapped__", None) is not inner
            or name in _COMPOSED_HOOKS.get(cls, {}) or outer_cells is None or inner_cells is None):
        raise ValueError("lifetime composition is not an exact tracked wrapper")
    mapping, supplied, native = outer_cells.get("self"), outer_cells.get("supplied"), inner_cells.get("fn")
    recipient_cells = _closure(supplied)
    recipient = None if recipient_cells is None else recipient_cells.get("self")
    legacy_code = next(value for value in install_manager.__code__.co_consts
                       if type(value) is type(install_manager.__code__) and value.co_name == "mk_alloc")
    legacy_alloc = next(value for value in legacy_code.co_consts
                        if type(value) is type(legacy_code) and value.co_name == "alloc")
    if (type(mapping) is not MappingObserver or type(recipient) is not AllocationRecipientObserver
            or mapping.recipient is not recipient or outer_cells.get("original") is not inner
            or recipient_cells.get("original") is not inner
            or inner.__code__ is not legacy_alloc
            or outer.__code__ is not _nested_code(MappingObserver._install, "installed")
            or supplied.__code__ is not _nested_code(AllocationRecipientObserver.wrap, "observed")
            or not inspect.isfunction(native) or getattr(inner, "__wrapped__", None) is not native):
        raise ValueError("lifetime composition delegation is not source-reviewed")
    _COMPOSED_HOOKS.setdefault(cls, {})[name] = dict(inner=inner, outer=outer, native=native,
        inner_code=inner.__code__, outer_code=outer.__code__, supplied=supplied,
        supplied_code=supplied.__code__, native_code=native.__code__,
        mapping=weakref.ref(mapping), recipient=weakref.ref(recipient), healthy=True)


def unregister_composition(cls, name, inner, outer):
    entries = _COMPOSED_HOOKS.get(cls, {})
    entry = entries.get(name)
    if entry is not None and entry["inner"] is inner and entry["outer"] is outer:
        del entries[name]
        if not entries:
            _COMPOSED_HOOKS.pop(cls, None)
        return True
    return False


def _installed_hook_names():
    return sorted({label for cls, methods in list(_INSTALLED_HOOKS.items())
                   for name, (wrapper, label) in list(methods.items())
                   if getattr(cls, name, None) is wrapper
                   or _composition_matches(cls, name, wrapper, getattr(cls, name, None))})


def _record_hook_coverage(p):
    names = _installed_hook_names()
    from ..assessment import installed_engine_identity
    from importlib.metadata import version, PackageNotFoundError
    try:
        nixl_version = version("nixl")
    except PackageNotFoundError:
        nixl_version = None
    p._emit("note", "sdcguard lifetime hooks: " + (", ".join(names) or "none"), hooks=names,
            engine_identity=installed_engine_identity(), nixl_version=nixl_version)


def reset():
    global _PROBE, _PROBE_PID
    p, _PROBE, _PROBE_PID = _PROBE, None, None
    if p is not None and hasattr(p, "close"):
        try:
            p.close()
        except Exception:
            pass
    for state in _STATE.values():
        state.clear()


def probe() -> Probe | None:
    global _PROBE, _PROBE_PID
    pid = os.getpid()
    if _PROBE is not None and _PROBE_PID not in (None, pid):
        # A fork must not inherit another process's writer, locks or sequence.
        _PROBE = None
        for state in _STATE.values():
            state.clear()
    if _PROBE is None and os.environ.get("SDCGUARD_LIFETIME_DIR"):
        hp = os.environ.get("SDCGUARD_LIFETIME_HOLDS")
        holds = holds_from_file(hp) if hp and os.path.exists(hp) else []
        _PROBE = Probe(JsonlDirSink(os.environ["SDCGUARD_LIFETIME_DIR"], asynchronous=True),
                       holds=holds, holds_path=hp or None, health_interval=0.1,
                       record_checkpoints=os.environ.get("SDCGUARD_LIFETIME_RECORD") == "1")
        _PROBE_PID = pid
        # In fork-mode vLLM does not necessarily rerun plugin installation.
        # This receipt belongs to the child's own stream and precedes events
        # recorded by any inherited wrapper.
        _record_hook_coverage(_PROBE)
    return _PROBE


def _record(callback, default=None):
    """Only recorder work belongs in callback; never a call to the wrapped method."""
    p = None
    try:
        p = probe()
        return callback(p) if p is not None else default
    except Exception as exc:
        if p is not None:
            try:
                p.recording_error(f"NIXL observer: {type(exc).__name__}: {exc}")
            except Exception:
                pass
        return default


def _engine(obj=None) -> str:
    return str(getattr(obj, "engine_id", None) or os.environ.get("SDCGUARD_LIFETIME_ENGINE", "engine"))


def _flat(block_ids) -> list:
    if block_ids is None:
        return []
    if not isinstance(block_ids, (list, tuple)):
        raise TypeError("block ids are not a concrete sequence")
    if block_ids and isinstance(block_ids[0], (list, tuple)):
        return list(block_ids[0])
    return list(block_ids)


def _groups(block_ids):
    if not isinstance(block_ids, (list, tuple)):
        return None
    return len(block_ids) if block_ids and isinstance(block_ids[0], (list, tuple)) else 1


def _ids(manager, request_id):
    return _flat(manager.get_block_ids(request_id))


def _concrete(value):
    # Inspect only documented concrete containers, never drain a generator passed
    # to vLLM, or one returned to its caller.
    return value if isinstance(value, (list, tuple, set, frozenset, dict)) else ()


def _wrap(cls, name, make, label=None):
    fn = getattr(cls, name, None)
    if fn is None or getattr(fn, "_sdcguard_lifetime", False):
        return False
    w = make(fn)
    w._sdcguard_lifetime = True
    setattr(cls, name, functools.wraps(fn)(w))
    _INSTALLED_HOOKS.setdefault(cls, {})[name] = (w, label or f"{cls.__name__}.{name}")
    return True


def _ops(rid):
    return list(_STATE["remote"].get(rid, {}).get("op_ids", ()))


def install_manager(KVCacheManager) -> list:
    done = []
    def mk_alloc(fn):
        signature = inspect.signature(fn)
        def alloc(self, request, *a, **k):
            out = fn(self, request, *a, **k)
            def record(p):
                if out is None:
                    return
                rid = request.request_id
                bound = signature.bind(self, request, *a, **k)
                nc = bound.arguments.get("new_computed_blocks")
                hits = set(_flat(nc.get_block_ids())) if nc is not None else set()
                owned = _STATE["owned"].setdefault((_engine(self), rid), set())
                for b in _ids(self, rid):
                    if b not in owned:
                        (p.share if b in hits else p.alloc)((_engine(self), b), rid)
                        owned.add(b)
            _record(record)
            return out
        return alloc
    if _wrap(KVCacheManager, "allocate_slots", mk_alloc, "KVCacheManager.allocate_slots"):
        done.append("KVCacheManager.allocate_slots")
    def mk_free(fn):
        def free(self, request, *a, **k):
            before = _record(lambda p: _ids(self, request.request_id))
            out = fn(self, request, *a, **k)
            def record(p):
                if before is None:
                    p.recording_error("free block identity unavailable")
                    return
                for b in before:
                    p.free((_engine(self), b), request.request_id)
                _STATE["owned"].pop((_engine(self), request.request_id), None)
            _record(record)
            return out
        return free
    if _wrap(KVCacheManager, "free", mk_free, "KVCacheManager.free"):
        done.append("KVCacheManager.free")
    return done


def _mapping(self, meta):
    remote = meta.remote
    local, sources = _flat(meta.local_block_ids), _flat(remote.block_ids)
    reason = None
    if _groups(meta.local_block_ids) != 1 or _groups(remote.block_ids) != 1:
        reason = "multiple cache groups"
    elif not local or len(local) != len(sources):
        reason = "empty transfer or partial prefix hit"
    elif len(set(local)) != len(local) or len(set(sources)) != len(sources):
        reason = "duplicate blocks"
    elif (getattr(self, "dcp_size", None) != 1 or getattr(self, "pcp_size", None) != 1
          or getattr(self, "world_size", None) != 1 or getattr(self, "pp_size", None) != 1):
        reason = "TP/PP/DCP/PCP topology unavailable or unsupported"
    elif getattr(self, "_physical_blocks_per_logical_kv_block", None) != 1:
        reason = "local physical block mapping unavailable or unsupported"
    elif getattr(self, "_has_mamba", True) or getattr(self, "_mixed_mem_types", True):
        reason = "mixed memory or state-space cache"
    else:
        info = self.transfer_topo.get_engine_info(remote.engine_id)
        plan = self.tp_mappings[remote.engine_id]
        if (info.remote_physical_blocks_per_logical != 1 or info.remote_dcp_size != 1
                or info.remote_tp_size != 1
                or self.block_size != info.remote_block_size or len(plan.all_source_ranks) != 1):
            reason = "remote block/rank mapping unsupported"
        elif set(self._remote_agents[remote.engine_id]) != {(0, 0)}:
            reason = "remote pipeline/rank mapping unsupported"
    return local, sources, reason


def install_connector(SchedulerCls=None, WorkerBase=None, PullWorker=None) -> list:
    done = []
    if SchedulerCls is not None:
        def mk_finished(fn):
            def request_finished(self, request, block_ids, *a, **k):
                out = fn(self, request, block_ids, *a, **k)
                def record(p):
                    if not isinstance(out, tuple) or not out or not out[0]:
                        return
                    eng, rid = _engine(self), request.request_id
                    params = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
                    sent = params.get("remote_block_ids", block_ids)
                    regions = [(eng, b) for b in _flat(sent)]
                    p.publish(f"{rid}@{eng}", regions, rid)
                    p.evidence("lease_granted", request=rid, engine=eng, desc=f"{rid}@{eng}",
                               source_regions=regions, published_count=len(regions),
                               mapping_complete=_groups(sent) == 1,
                               lease_fields="remote_lease_nonce" in params,
                               lease_slot=params.get("remote_lease_slot"), lease_nonce=params.get("remote_lease_nonce"))
                _record(record)
                return out
            return request_finished
        if _wrap(SchedulerCls, "request_finished", mk_finished):
            done.append(f"{SchedulerCls.__name__}.request_finished")
    if WorkerBase is not None:
        def mk_reap(fn):
            def reap(self, done_sending, *a, **k):
                before = _record(lambda p: set(_concrete(done_sending)))
                out = fn(self, done_sending, *a, **k)
                def record(p):
                    if before is not None:
                        for rid in set(_concrete(done_sending)) - before:
                            p._emit("note", f"lease expired for {rid} on {_engine(self)}", lease_expired=rid,
                                    engine=_engine(self))
                            p.evidence("lease_expired", request=rid, engine=_engine(self))
                _record(record)
                return out
            return reap
        if _wrap(WorkerBase, "_reap_expired_send_leases", mk_reap):
            done.append(f"{WorkerBase.__name__}._reap_expired_send_leases")
        def mk_pop(fn):
            def pop(self, transfers, *a, **k):
                is_recv = _record(lambda p: transfers is self._recving_transfers, False)
                out = fn(self, transfers, *a, **k)
                def record(p):
                    if is_recv:
                        ids = out[0] if isinstance(out, tuple) else out
                        for rid in _concrete(ids):
                            ops = _ops(rid)
                            p.evidence("transfer_done", request=rid, op_ids=ops)
                            destinations = _STATE["remote"].get(rid, {}).get("local", ())
                            for op, dst in zip(ops, destinations):
                                p.complete(op, marks_ready=(_engine(self), dst))
                _record(record)
                return out
            return pop
        if _wrap(WorkerBase, "_pop_done_transfers", mk_pop):
            done.append(f"{WorkerBase.__name__}._pop_done_transfers")
    if PullWorker is not None:
        def mk_read(fn):
            def read(self, req_id, meta, *a, **k):
                def record(p):
                    remote = meta.remote
                    reng, rreq = str(remote.engine_id), str(remote.request_id)
                    # vLLM queues an asynchronous handshake and calls this
                    # method again when the peer is ready. No READ or attempt
                    # has happened yet; probing topology now would both taint
                    # the recorder and mislabel that retry as a second reader.
                    agents = getattr(self, "_remote_agents", None)
                    if agents is not None and remote.engine_id not in agents:
                        return
                    if remote.engine_id in getattr(self, "_invalid_remote_engines", ()):
                        return
                    desc = f"{rreq}@{reng}"
                    p.checkpoint("read_submit", req_id, desc=desc)
                    local, sources, reason = _mapping(self, meta)
                    if req_id in _STATE["remote"]:
                        reason = "repeated attempt id; publication/attempt binding ambiguous"
                    op_ids = [f"read-{req_id}-{i}" for i in range(len(local))] if reason is None else []
                    state = dict(local=local, source=sources, desc=desc, op_ids=op_ids,
                                 mapping_complete=reason is None, submitted=False, expected_rank=None)
                    _STATE["remote"][req_id] = state
                    p.evidence("attempt", request=req_id, remote_request=rreq, remote_engine=reng, desc=desc,
                               blocks=len(local), op_ids=op_ids, source_indices=list(range(len(sources))),
                               source_regions=[(reng, b) for b in sources],
                               destination_regions=[(_engine(self), b) for b in local], published_count=len(sources),
                               mapping_complete=reason is None, mapping_reason=reason,
                               reader_rank=getattr(self, "tp_rank", None),
                               reader_agent=_STATE["agents"].get(id(getattr(self, "nixl_wrapper", None))),
                               lease_fields=hasattr(remote, "lease_nonce"),
                               lease_slot=getattr(remote, "lease_slot", None),
                               lease_nonce=getattr(remote, "lease_nonce", None))
                _record(record)
                return fn(self, req_id, meta, *a, **k)
            return read
        if _wrap(PullWorker, "_read_blocks_for_req", mk_read):
            done.append(f"{PullWorker.__name__}._read_blocks_for_req")
        def mk_blocks(fn):
            signature = inspect.signature(fn)
            def blocks(self, *a, **k):
                def before(p):
                    args = signature.bind(self, *a, **k).arguments
                    rid, spec = args["request_id"], args["read_spec"]
                    state = _STATE["remote"].get(rid)
                    if not state or not state["mapping_complete"]:
                        return None
                    # These equalities establish the actual spec agrees with the
                    # published logical blocks. Prefix trimming is intentionally unsupported.
                    if (getattr(spec, "block_ids_by_region", False)
                            or _flat(spec.local_block_ids) != state["local"]
                            or _flat(spec.remote_block_ids) != state["source"] or state["submitted"]):
                        p.recording_error("READ spec differs from the observed block correspondence")
                        return None
                    state["submitted"] = True
                    p.accept(state["desc"], f"decode {rid}")
                    for i, (op, dst) in enumerate(zip(state["op_ids"], state["local"])):
                        p.submit(op, "read", f"decode {rid}", src=desc_ref(state["desc"], i),
                                 dst=(_engine(self), dst))
                    return rid, list(state["op_ids"]), len(self._recving_transfers.get(rid, ())), int(spec.remote_rank)
                context = _record(before)
                out = fn(self, *a, **k)
                def after(p):
                    if context is not None:
                        rid, ops, count, rank = context
                        submitted = out is True and len(self._recving_transfers.get(rid, ())) > count
                        p.evidence("read_submission_result", request=rid, op_ids=ops, remote_rank=rank,
                                   submitted=submitted)
                _record(after)
                return out
            return blocks
        if _wrap(PullWorker, "_read_blocks", mk_blocks):
            done.append(f"{PullWorker.__name__}._read_blocks")
        def mk_notifs(fn):
            def notifs(self, *a, **k):
                context = {"owner": self, "records": [], "complete": False}
                # The fixed connector uses the three-field wire format even if
                # this producer does not export lease slots. Select its parser
                # by the actual wrapped implementation, not slots presence.
                _record(lambda p: context.update(parser=getattr(fn, "__globals__", {}).get("parse_read_notif")))
                previous = getattr(_CONTEXT, "notifications", None)
                _CONTEXT.notifications = context
                try:
                    _record(lambda p: install_nixl_agent(type(self.nixl_wrapper)))
                    out = fn(self, *a, **k)
                finally:
                    _CONTEXT.notifications = previous
                def record(p):
                    for rid in _concrete(out):
                        records = [r for r in context["records"] if r["request"] == rid and r["nonce_matches"]]
                        exact = context["complete"] and len(records) == 1 and records[0]["expected_consumers"] == 1
                        rec = records[0] if exact else {}
                        p.evidence("released_by_read_notification", request=rid, engine=_engine(self),
                                   notification_nonce=rec.get("notification_nonce"),
                                   current_nonce=rec.get("current_nonce"), sender_agent=rec.get("sender_agent"),
                                   exact_mapping=exact, transport_complete=exact, notifications=records)
                _record(record)
                return out
            return notifs
        if _wrap(PullWorker, "_get_new_notifs", mk_notifs):
            done.append(f"{PullWorker.__name__}._get_new_notifs")
    return done


def install_nixl_agent(Agent) -> list:
    """Observe constructor identity and the returned notification batch unchanged."""
    names = []
    def mk_init(fn):
        def init(self, *a, **k):
            out = fn(self, *a, **k)
            def record(p):
                name = a[0] if a else k.get("agent_name")
                if isinstance(name, str):
                    _STATE["agents"][id(self)] = name
            _record(record)
            return out
        return init
    if _wrap(Agent, "__init__", mk_init):
        names.append("NixlAgent.__init__")
    def mk_get(fn):
        def get(self, *a, **k):
            out = fn(self, *a, **k)
            def record(p):
                context = getattr(_CONTEXT, "notifications", None)
                if context is None or context["owner"].nixl_wrapper is not self:
                    return
                if not isinstance(out, dict) or any(not isinstance(v, (list, tuple)) for v in out.values()):
                    raise TypeError("unknown notification batch shape")
                owner = context["owner"]
                fixed_parser = context.get("parser")
                for sender, notifications in out.items():
                    for payload in notifications:
                        msg = payload.decode("utf-8")
                        if msg.startswith("HB:"):
                            continue
                        if fixed_parser is None:
                            rid, count = msg.rsplit(":", 1)
                            count, nonce = int(count), None
                        else:
                            rid, count, nonce = fixed_parser(msg)
                        slots = getattr(owner, "_lease_slots", None)
                        current = slots.nonce(rid) if slots is not None else None
                        rec = dict(request=rid, sender_agent=str(sender), notification_nonce=nonce,
                                   current_nonce=current, expected_consumers=count,
                                   nonce_matches=nonce is None or nonce == current)
                        context["records"].append(rec)
                        p.evidence("read_notification_received", engine=_engine(owner), **rec)
                context["complete"] = True
            _record(record)
            return out
        return get
    if _wrap(Agent, "get_new_notifs", mk_get):
        names.append("NixlAgent.get_new_notifs")
    return names


def install_verifier(LeaseVerifier) -> list:
    names = []
    def mk_add(fn):
        def add(self, req_id, engine_id, remote_rank, slot, nonce, *a, **k):
            out = fn(self, req_id, engine_id, remote_rank, slot, nonce, *a, **k)
            def record(p):
                _STATE["checks"].setdefault((id(self), req_id), []).append((engine_id, remote_rank, slot, nonce))
            _record(record)
            return out
        return add
    if _wrap(LeaseVerifier, "add_check", mk_add):
        names.append(f"{LeaseVerifier.__name__}.add_check")
    def mk_resolve(fn):
        def resolve(self, remote, entries, done, reusable, *a, **k):
            def snapshot(p):
                if not isinstance(entries, (list, tuple)):
                    raise TypeError("unknown lease entries container")
                facts = []
                for entry in entries:
                    observed = int(self._cells[entry.cell]) if done else None
                    cause = "check_unavailable" if not done else "matched" if observed == entry.nonce else "lease_mismatch"
                    facts.append(dict(request=entry.req_id, remote_engine=remote[0], remote_rank=int(remote[1]),
                                      lease_slot=int(entry.slot), expected_nonce=int(entry.nonce),
                                      observed_nonce=observed, done=bool(done), cause=cause))
                return facts
            facts = _record(snapshot)
            out = fn(self, remote, entries, done, reusable, *a, **k)
            def record(p):
                for fact in facts or ():
                    _STATE["resolved"].setdefault((id(self), fact["request"]), []).append(fact)
                    p.evidence("lease_check_detail", op_ids=_ops(fact["request"]), **fact)
            _record(record)
            return out
        return resolve
    if _wrap(LeaseVerifier, "_resolve", mk_resolve):
        names.append(f"{LeaseVerifier.__name__}._resolve")
    def mk_step(fn):
        def step(self, reads_done, *a, **k):
            out = fn(self, reads_done, *a, **k)
            def record(p):
                if not isinstance(out, tuple) or len(out) != 2:
                    raise TypeError("unknown lease verifier result")
                verified, failures = out
                for rid in list(_concrete(verified)) + list(_concrete(failures)):
                    facts = _STATE["resolved"].pop((id(self), rid), [])
                    expected = _STATE["checks"].pop((id(self), rid), [])
                    complete = (bool(expected) and len(facts) == len(expected)
                                and sorted((f["remote_engine"], f["remote_rank"], f["lease_slot"], f["expected_nonce"])
                                           for f in facts) == sorted(expected))
                    cause = "check_unavailable"
                    if complete:
                        causes = {f["cause"] for f in facts}
                        cause = "lease_mismatch" if "lease_mismatch" in causes else "check_unavailable" if "check_unavailable" in causes else "matched"
                    p.evidence("lease_check", request=rid, op_ids=_ops(rid),
                               result="passed" if rid in verified else "failed", cause=cause,
                               details_complete=complete,
                               expected_ranks=[[e[0], e[1]] for e in expected],
                               observed_ranks=[[f["remote_engine"], f["remote_rank"]] for f in facts])
            _record(record)
            return out
        return step
    if _wrap(LeaseVerifier, "step", mk_step):
        names.append(f"{LeaseVerifier.__name__}.step")
    return names


def install_scheduler(Scheduler) -> list:
    names = []
    def mk_upd(fn):
        def upd(self, request, *a, **k):
            def before(p):
                return request.request_id in self.failed_recving_kv_req_ids
            failed = _record(before)
            out = fn(self, request, *a, **k)
            def record(p):
                active = getattr(_CONTEXT, "admission", None)
                if active is not None and active["scheduler"] is self and active["request"] is request:
                    active.update(updated=True, failed=failed)
            _record(record)
            return out
        return upd
    if _wrap(Scheduler, "_update_waiting_for_remote_kv", mk_upd):
        names.append("Scheduler._update_waiting_for_remote_kv")
    def mk_blocked(fn):
        def blocked(self, request, *a, **k):
            context = {"scheduler": self, "request": request, "updated": False, "failed": None}
            previous = getattr(_CONTEXT, "admission", None)
            _CONTEXT.admission = context
            try:
                out = fn(self, request, *a, **k)
            finally:
                _CONTEXT.admission = previous
            def record(p):
                if out is True and context["updated"]:
                    rid, failed = request.request_id, context["failed"]
                    p.evidence("admit", request=rid, transferred=None if failed is None else not failed,
                               closed=False, recompute_pending=failed is True,
                               computed_tokens=getattr(request, "num_computed_tokens", None))
                    # Legacy search event: scheduler admission, NOT GPU execution.
                    if failed is False:
                        mgr = self.kv_cache_manager
                        regions = [(_engine(mgr), b) for b in _ids(mgr, rid)]
                        p.consume(rid, regions)
            _record(record)
            return out
        return blocked
    if _wrap(Scheduler, "_handle_blocked_waiting_request", mk_blocked):
        names.append("Scheduler._handle_blocked_waiting_request")
    def mk_failed(fn):
        def failed(self, failed_req_ids, *a, **k):
            # Do not consume an iterable before the original function.
            ids = _record(lambda p: set(_concrete(failed_req_ids)), set())
            out = fn(self, failed_req_ids, *a, **k)
            def record(p):
                recompute = getattr(self, "recompute_kv_load_failures", None)
                affected = self.failed_recving_kv_req_ids if recompute is True else _concrete(out)
                for rid in ids & set(affected):
                    policy = "recompute" if recompute is True else "fail" if recompute is False else "unknown"
                    p.evidence("load_failed", request=rid, policy=policy)
                    if policy == "fail":
                        _STATE["failed_terminal"][(id(self), rid)] = True
                # Returned IDs are only *scheduled* for closure by the caller;
                # recording closed=True here would claim an unobserved action.
            _record(record)
            return out
        return failed
    if _wrap(Scheduler, "_handle_failed_recving", mk_failed):
        names.append("Scheduler._handle_failed_recving")
    def mk_free_request(fn):
        def free_request(self, request, *a, **k):
            out = fn(self, request, *a, **k)
            def record(p):
                rid = request.request_id
                if _STATE["failed_terminal"].pop((id(self), rid), False):
                    if getattr(getattr(request, "status", None), "name", None) == "FINISHED_ERROR":
                        p.evidence("request_closed", request=rid, status="FINISHED_ERROR")
                        p.evidence("admit", request=rid, transferred=False, closed=True)
                    else:
                        p.recording_error("failed-load request closure status was not established")
            _record(record)
            return out
        return free_request
    if _wrap(Scheduler, "_free_request", mk_free_request):
        names.append("Scheduler._free_request")
    return names


def apply_ucx_params(params: dict) -> bool:
    """Pass extra UCX backend init parameters to every NIXL agent this process creates. A test-environment setting:
    vLLM 0.32.0 has no key for it (vLLM PR #62905 proposes kv_connector_extra_config.ucx_backend_params), and UCX's
    environment variables cannot set it. Needed where NIXL's default peer error handling leaves UCX no transport it
    may use: a container sandbox that hides the network device list disables UCX's TCP transport, and the
    shared-memory ones cannot detect a failed peer. ucx_error_handling_mode=none lets UCX use shared memory and CUDA
    IPC on one machine, and gives up detecting a failed peer."""
    if not params:
        return False
    os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")     # what vLLM sets before it imports nixl
    try:
        importlib.import_module("nixl")
        api = importlib.import_module("nixl._api")
        cls = api.nixl_agent
    except Exception:  # noqa: BLE001  (no NIXL here: nothing to pass the parameters to)
        return False
    fn = cls.create_backend
    if getattr(fn, "_sdcguard_ucx", None) == params:
        return True
    orig = getattr(fn, "_sdcguard_orig", fn)

    def create_backend(self, backend, initParams=None, *a, **k):
        merged = dict(initParams or {})
        if backend == "UCX":
            for key, value in params.items():
                merged.setdefault(key, str(value))
        return orig(self, backend, merged, *a, **k)
    create_backend._sdcguard_ucx = dict(params)
    create_backend._sdcguard_orig = orig
    cls.create_backend = create_backend
    return True


def install() -> list:
    """Wrap what this vLLM has; return the names wrapped (empty when vLLM or the NIXL connector is absent)."""
    if not os.environ.get("SDCGUARD_LIFETIME_DIR"):
        return []
    done = []
    ucx = os.environ.get("SDCGUARD_NIXL_UCX_PARAMS")
    if ucx:
        try:
            params = json.loads(ucx)
        except ValueError:
            params = {}
        if isinstance(params, dict) and apply_ucx_params(params):
            done.append("NIXL UCX backend params " + ", ".join(f"{k}={v}" for k, v in sorted(params.items())))
    try:
        from vllm.distributed.nixl_utils import NixlWrapper
        if NixlWrapper is not None:
            done += install_nixl_agent(NixlWrapper)
    except Exception:  # optional NIXL library
        pass
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        done += install_manager(KVCacheManager)
    except Exception:  # noqa: BLE001  (not this vLLM layout)
        pass
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker, pull_worker
        try:      # v0.32.0: the base class only declares request_finished; the pull scheduler implements it
            from vllm.distributed.kv_transfer.kv_connector.v1.nixl import pull_scheduler
            sched = pull_scheduler.NixlPullConnectorScheduler
        except Exception:  # noqa: BLE001
            from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_scheduler
            sched = base_scheduler.NixlBaseConnectorScheduler
        done += install_connector(sched, base_worker.NixlBaseConnectorWorker, pull_worker.NixlPullConnectorWorker)
    except Exception:  # noqa: BLE001
        pass
    try:          # present only on builds with PR #62065's lease slots
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl import lease_slots
        done += install_verifier(lease_slots.LeaseVerifier)
    except Exception:  # noqa: BLE001
        pass
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        done += install_scheduler(Scheduler)
    except Exception:  # noqa: BLE001
        pass
    _record(_record_hook_coverage)
    return done
