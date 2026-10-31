"""Evaluate observed KV handoff obligations, never model answers.

Positive verdicts require complete same-host recordings with health receipts. A
ledger overlap ending at a delayed completion poll is not proof of a physical
READ race. The scope is source lifetime and scheduler admission, not GPU use or
answer correctness. Recovery additionally needs a recompute-completion fact.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field

RULES = {
    "lease_valid_at_use": "transferred KV is admitted only with an established source lifetime or lease check",
}
VERDICTS = ("satisfied", "violated", "correctly_rejected", "recovered", "refused", "unknown", "not_exercised")
PRODUCER_HOLDS = "producer keeps the blocks until the read completes"
CONSUMER_CHECKS = "consumer checks the lease after the transfer"


@dataclass
class Finding:
    rule: str
    request: str
    verdict: str = "unknown"
    protection: str | None = None
    assumption: str = "recorded identities, ownership and transport facts describe this attempt"
    relationship: str = "not enough evidence to judge the rule"
    evidence: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    detail: str = ""
    ordering: str = "same-host monotonic observations and explicit transport evidence"
    not_observed: tuple = ("use by the model itself (admission is the closest event)",
                           "correctness of the generated answer")

    def as_dict(self) -> dict:
        return asdict(self)


def _args(e):
    a = e.get("args") if isinstance(e, dict) else None
    return a if isinstance(a, dict) else {}


def _mono(e):
    n = e.get("mono") if isinstance(e, dict) else None
    return n if type(n) in (int, float) and math.isfinite(n) else None


def _stream(e):
    session, pid = e.get("session"), e.get("pid")
    return (session, pid) if isinstance(session, str) and session and type(pid) in (int, str) else None


def _ref(e):
    return {"session": e.get("session"), "pid": e.get("pid"), "seq": e.get("seq"),
            "kind": e.get("kind"), "what": _args(e).get("what")}


def _region(r):
    if (isinstance(r, (list, tuple)) and len(r) == 2 and isinstance(r[0], str)
            and r[0] and type(r[1]) is int and r[1] >= 0):
        return tuple(r)
    return None


def _ops(e):
    ids = _args(e).get("op_ids")
    return ids if isinstance(ids, list) and all(isinstance(i, str) and i for i in ids) else []


def _regions(value):
    return [_region(r) for r in value] if isinstance(value, (list, tuple)) else []


class Facts:
    def __init__(self, events):
        original = list(events or ())
        self.events = [e for e in original if isinstance(e, dict)]
        self.invalid = len(original) != len(self.events)
        self.by = defaultdict(list)
        self.published = defaultdict(list)
        self.streams = defaultdict(list)
        self.attempts = []
        for e in self.events:
            a = _args(e)
            # Every observed event can influence lifecycle reconstruction. Do
            # not let an unsequenced free/allocation bypass receipt validation.
            if (type(e.get("seq")) is not int or e["seq"] < 1
                    or _stream(e) is None or _mono(e) is None
                    or not isinstance(e.get("clock_id"), str) or not e["clock_id"]
                    or type(e.get("dropped")) is not int or e["dropped"] < 0):
                self.invalid = True
            if (not isinstance(e.get("args"), dict) or not isinstance(e.get("kind"), str)
                    or any(a.get(k) is not None and not isinstance(a[k], str)
                           for k in ("what", "request", "desc"))):
                self.invalid = True
                continue
            owner_field = {"alloc": "owner", "free": "by", "share": "request"}.get(e["kind"])
            if owner_field and (_region(a.get("region")) is None
                                or not isinstance(a.get(owner_field), str) or not a[owner_field]):
                self.invalid = True
            self.streams[_stream(e)].append(e)
            if e.get("kind") == "evidence":
                self.by[(a.get("what"), a.get("request"))].append(e)
                if a.get("what") == "attempt" and a.get("request") is not None:
                    if a["request"] not in self.attempts:
                        self.attempts.append(a["request"])
            elif e.get("kind") == "publish":
                self.published[a.get("desc")].append(e)

    def first(self, what, request):
        es = self.all(what, request)
        return es[0] if es else None

    def all(self, what, request):
        return list(self.by.get((what, request), ()))

    def unique(self, what, request, finding):
        es = self.all(what, request)
        if not es:
            finding.missing.append(what)
            return None
        if len(es) != 1:
            finding.missing.append(f"unambiguous {what}")
            return None
        return es[0]

    def coverage(self, required, terminal, finding):
        """Each relevant stream has a contiguous healthy prefix through terminal."""
        clock = terminal.get("clock_id") if terminal else None
        end = _mono(terminal)
        if self.invalid or any(_args(e).get("what") == "recording_error" for e in self.events):
            finding.missing.append("recording without parse or truncation errors")
        if not clock or end is None:
            finding.missing.append("terminal monotonic clock identity")
            return
        for key in dict.fromkeys(_stream(e) for e in required if e):
            es = self.streams.get(key, [])
            if key is None:
                finding.missing.append("recording stream identity")
                continue
            receipts = [e for e in es if e.get("kind") == "evidence"
                        and _args(e).get("what") == "recording_health"
                        and _args(e).get("healthy") is True
                        and e.get("clock_id") == clock and _mono(e) is not None and _mono(e) >= end
                        and type(e.get("seq")) is int]
            if not receipts:
                finding.missing.append(f"healthy recording through terminal for {key}")
                continue
            receipt = min(receipts, key=lambda e: e["seq"])
            prefix = [e for e in es if type(e.get("seq")) is int and e["seq"] <= receipt["seq"]]
            seqs = sorted(e["seq"] for e in prefix)
            if len(seqs) != receipt["seq"] or any(seq != i for i, seq in enumerate(seqs, 1)):
                finding.missing.append(f"contiguous recording prefix for {key}")
            if any(type(e.get("seq")) is not int or e["seq"] > receipt["seq"]
                   for e in required if e and _stream(e) == key):
                finding.missing.append(f"receipt covering every required event for {key}")
            if any(e.get("dropped") != 0 or e.get("clock_id") != clock or _mono(e) is None
                   for e in prefix):
                finding.missing.append(f"same-clock lossless recording for {key}")
            ordered = sorted(prefix, key=lambda e: e["seq"])
            if any(_mono(a) is not None and _mono(b) is not None and _mono(a) > _mono(b)
                   for a, b in zip(ordered, ordered[1:])):
                finding.missing.append(f"monotonic recording for {key}")
        if any(e.get("clock_id") != clock or _mono(e) is None for e in required if e):
            finding.missing.append("one known monotonic clock for related observations")


def _context(facts, request, finding):
    att = facts.unique("attempt", request, finding)
    adm = facts.unique("admit", request, finding)
    done = facts.unique("transfer_done", request, finding)
    sent = facts.unique("read_submission_result", request, finding)
    if not att or not adm or not done or not sent:
        return None
    a = _args(att)
    if not all(isinstance(a.get(k), str) and a[k] for k in ("desc", "remote_request", "remote_engine")):
        finding.missing.append("explicit remote request and engine identities")
        return None
    if type(_args(adm).get("transferred")) is not bool:
        finding.missing.append("explicit admission decision")
    pubs = facts.published.get(a.get("desc"), [])
    if len(pubs) != 1:
        finding.missing.append("one unambiguous producer publication")
        return None
    pub = pubs[0]
    regions = _regions(_args(pub).get("regions"))
    count, ids = a.get("blocks"), a.get("op_ids")
    if (not regions or None in regions or len(set(regions)) != len(regions)
            or type(count) is not int or count != len(regions) or not isinstance(ids, list)
            or len(ids) != count or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != count):
        finding.missing.append("nonempty complete published-to-read mapping and operation identities")
        return None
    if a.get("mapping_complete") is not True:
        finding.missing.append("verified complete block mapping")
    if (_regions(a.get("source_regions")) != regions or a.get("source_indices") != list(range(count))
            or a.get("published_count") != count):
        finding.missing.append("actual read sources match every published source block")
    if _args(pub).get("by") != a["remote_request"] or any(r[0] != a["remote_engine"] for r in regions):
        finding.missing.append("publication bound to the remote owner and engine")
    if _args(sent).get("submitted") is not True or set(_ops(sent)) != set(ids):
        finding.missing.append("actual transport submission of declared operations")
    if set(_ops(done)) != set(ids):
        finding.missing.append("completion of declared operations")
    if _stream(done) != _stream(att) or _stream(sent) != _stream(att):
        finding.missing.append("submission and completion bound to the attempt worker")
    subs = [e for e in facts.events if e.get("kind") == "submit" and _args(e).get("op") in ids]
    if len(subs) != count or {_args(e).get("op") for e in subs} != set(ids):
        finding.missing.append("one submission per declared read operation")
        return None
    indices, destinations = [], []
    expected_destinations = _regions(a.get("destination_regions"))
    for e in subs:
        s, src = _args(e), _args(e).get("src")
        if s.get("kind") != "read" or not isinstance(src, dict) or src.get("desc") != a.get("desc"):
            finding.missing.append("source identity of each read submission")
            continue
        index = src.get("i")
        if type(index) is not int or index < 0 or index >= count:
            finding.missing.append("explicit valid source block index")
            continue
        indices.append(index)
        destinations.append(_region(s.get("dst")))
        if (len(expected_destinations) != count or None in expected_destinations
                or _region(s.get("dst")) != expected_destinations[ids.index(s["op"])]):
            finding.missing.append("submission destinations match actual read mapping")
        if _stream(e) != _stream(att):
            finding.missing.append("read submission bound to attempt stream")
    if set(indices) != set(range(count)) or None in destinations or len(set(destinations)) != count:
        finding.missing.append("complete source and destination mapping")
    hooks = [e for e in facts.streams.get(_stream(pub), ()) if e.get("kind") == "note"
             and isinstance(_args(e).get("hooks"), list) and all(isinstance(h, str) for h in _args(e)["hooks"])
             and {"KVCacheManager.free", "KVCacheManager.allocate_slots"}.issubset(_args(e)["hooks"])
             and _mono(e) is not None and _mono(pub) is not None and _mono(e) <= _mono(pub)]
    if not hooks:
        finding.missing.append("producer allocation/free hooks before publication")
    checks = facts.all("lease_check", request)
    required = [att, pub, done, adm, sent, *subs, *checks, *hooks]
    facts.coverage(required, adm, finding)
    if not finding.missing:
        if not (_mono(pub) < min(_mono(e) for e in subs) and _mono(att) < min(_mono(e) for e in subs)
                and max(_mono(e) for e in subs) < _mono(sent) < _mono(done) < _mono(adm)):
            finding.missing.append("publication, submission, completion and admission order")
        if any(not (_mono(done) < _mono(e) < _mono(adm)) for e in checks):
            finding.missing.append("lease checking after transfer and before admission")
    finding.evidence.extend(_ref(e) for e in required)
    return att, pub, subs, done, adm, regions, checks


def _source_history(facts, context, finding):
    att, pub, subs, done, adm, regions, _ = context
    owner = _args(att).get("remote_request")
    foreign = [e for e in facts.events if e.get("kind") in ("alloc", "free", "share")
               and _region(_args(e).get("region")) in regions and _stream(e) != _stream(pub)]
    if foreign:
        # Engine/block names do not identify an allocator incarnation. A second
        # process using those names may be a restart or collision; ignoring its
        # writes would make an apparently complete publishing stream look safe.
        finding.missing.append("one allocator recording incarnation for every source region")
        finding.evidence.extend(_ref(e) for e in foreign)
        return [], []
    changes = [e for e in facts.streams.get(_stream(pub), ()) if e.get("kind") in ("alloc", "free")
               and _region(_args(e).get("region")) in regions]
    facts.coverage([e for e in changes if _mono(e) is not None and _mono(e) <= _mono(done)], adm, finding)
    if finding.missing:
        return [], []
    bounds = {_mono(pub), _mono(done), *(_mono(e) for e in subs)}
    if any(_mono(e) in bounds for e in changes):
        finding.missing.append("strict source lifecycle ordering at publication/read boundaries")
        return [], []
    before = [e for e in changes if _mono(e) is not None and _mono(e) <= _mono(pub)]
    for region in regions:
        prior = [e for e in before if _region(_args(e).get("region")) == region]
        if (not prior or max(prior, key=_mono).get("kind") != "alloc"
                or _args(max(prior, key=_mono)).get("owner") != owner):
            finding.missing.append(f"source ownership at publication for {region}")
    retired_before, uncertain = [], []
    for sub in subs:
        region = regions[_args(sub)["src"].get("i", 0)]
        for e in changes:
            if (_region(_args(e).get("region")) != region or _mono(e) is None
                    or not (_mono(pub) < _mono(e) <= _mono(done))):
                continue
            # alloc is recorded after allocation. Free-only evidence does not prove
            # data was replaced, and completion polling can lag the physical READ.
            if e.get("kind") == "alloc" and _mono(e) < _mono(sub):
                frees = [x for x in changes if x.get("kind") == "free"
                         and _region(_args(x).get("region")) == region and _args(x).get("by") == owner
                         and _mono(x) is not None and _mono(pub) < _mono(x) < _mono(e)]
                if frees:
                    retired_before.append(e)
                else:
                    # Allocation alone might be a repeated/shared ownership record.
                    # The supported stale-source proof needs retirement and reuse.
                    uncertain.append(e)
            else:
                uncertain.append(e)
    return retired_before, uncertain


def _notification_completion(facts, context, finding):
    att, pub, subs, done, adm, _, _ = context
    a = _args(att)
    readers = [x for x in facts.attempts if _args(facts.first("attempt", x)).get("desc") == a.get("desc")]
    if readers != [a.get("request")]:
        return None
    candidates = [e for e in facts.all("released_by_read_notification", a.get("remote_request"))
                  if _args(e).get("engine") == a.get("remote_engine")
                  and _args(e).get("transport_complete") is True and _args(e).get("exact_mapping") is True
                  and a.get("reader_agent") is not None and _args(e).get("sender_agent") == a["reader_agent"]]
    if len(candidates) != 1:
        return None
    e = candidates[0]
    facts.coverage([e], adm, finding)
    if finding.missing or not (max(_mono(s) for s in subs) < _mono(e) < _mono(done)):
        return None
    return e


def lease_valid_at_use(facts: Facts, violations, request: str) -> Finding:
    f = Finding("lease_valid_at_use", request)
    context = _context(facts, request, f)
    if context is None or f.missing:
        return f
    att, pub, subs, done, adm, regions, checks = context
    f.protection = CONSUMER_CHECKS if checks else PRODUCER_HOLDS
    admitted = _args(adm)["transferred"]
    if any(not isinstance(_args(e).get(k), str) for e in checks for k in ("result", "cause")):
        f.missing.append("normalized lease-check result and cause")
        return f
    statuses = {(_args(e).get("result"), _args(e).get("cause")) for e in checks}
    passed = any(_args(e).get("result") == "passed" for e in checks)
    failed = any(_args(e).get("result") == "failed" for e in checks)
    if (passed and failed or statuses - {("passed", "matched"), ("failed", "lease_mismatch"),
                                        ("failed", "check_unavailable")}):
        f.detail = "contradictory or unrecognized lease-check results"
        return f
    if any(set(_ops(e)) != set(_args(att)["op_ids"]) or _stream(e) != _stream(att) for e in checks):
        f.missing.append("lease-check operation coverage")
        return f
    retired, uncertain = _source_history(facts, context, f)
    if f.missing:
        return f
    mismatch = failed and statuses == {("failed", "lease_mismatch")}
    if mismatch and admitted:
        f.verdict = "violated"
        f.relationship = "the engine admitted transferred KV after its recorded lease check failed"
        return f
    if failed and admitted:
        f.detail = "the check was unavailable; invalidity was not established"
        return f
    if not admitted:
        if not mismatch:
            f.detail = "rejection observed, but a normalized invalid-lease check was not established"
            return f
        f.verdict = "correctly_rejected"
        f.relationship = "the lease check found an invalid lease and the engine rejected the transferred state"
        recoveries, policies = facts.all("recompute_done", request), facts.all("load_failed", request)
        if _args(adm).get("closed") is True:
            closed = facts.unique("request_closed", request, f)
            if (closed is None or len(policies) != 1 or _args(policies[0]).get("policy") != "fail"
                    or _args(closed).get("status") != "FINISHED_ERROR"
                    or _stream(closed) != _stream(adm) or _stream(policies[0]) != _stream(adm)):
                f.verdict = "unknown"
                f.missing.append("attributed failed-load terminal closure")
                return f
            facts.coverage([closed, *policies], adm, f)
            if f.missing or not (_mono(done) < _mono(policies[0]) < _mono(closed) < _mono(adm)) or recoveries:
                f.verdict = "unknown"
                return f
            f.verdict = "refused"
            f.relationship += "; scheduler terminal error closure was observed"
            return f
        if recoveries:
            recovery = facts.unique("recompute_done", request, f)
            if (_args(adm).get("closed") is not False or len(policies) != 1
                    or _args(policies[0]).get("policy") != "recompute"):
                f.verdict = "unknown"
                f.detail = "recompute completion contradicts failure policy or closed request"
                return f
            if recovery is None:
                f.verdict = "unknown"
                return f
            # A request label alone cannot bind a recovery from another process
            # or attempt. The scheduler must name the exact rejected transfer.
            rejected_ops = _args(recovery).get("rejected_op_ids")
            if (_stream(recovery) != _stream(adm) or _stream(policies[0]) != _stream(adm)
                    or not isinstance(rejected_ops, list)
                    or any(not isinstance(op, str) for op in rejected_ops)
                    or len(rejected_ops) != len(_ops(att)) or set(rejected_ops) != set(_ops(att))):
                f.verdict = "unknown"
                f.missing.append("recompute completion bound to the rejecting scheduler and exact transfer")
                return f
            facts.coverage([att, pub, adm, recovery, *policies], recovery, f)
            if (f.missing or not (_mono(done) < _mono(policies[0]) < _mono(adm) < _mono(recovery))):
                f.verdict = "unknown"
                return f
            f.verdict = "recovered"
            f.relationship += "; local recomputation completion was independently recorded"
        elif any(_args(e).get("policy") == "recompute" for e in policies):
            f.detail = "recompute requested; completion not observed"
        return f
    if retired:
        f.verdict = "violated" if not passed else "unknown"
        f.relationship = "source blocks were reallocated before read submission and transferred state was admitted"
        f.detail = "passed lease check contradicts ownership history" if passed else "no lease check was recorded"
        f.evidence.extend(_ref(e) for e in retired)
        return f
    if passed:
        f.verdict = "satisfied"
        f.relationship = "the completed pull passed its recorded lease check before transferred-state admission"
        return f
    if uncertain:
        notification = _notification_completion(facts, context, f)
        if notification is None or any(_mono(e) <= _mono(notification) for e in uncertain):
            f.detail = "source lifetime changed before completion poll; actual READ ordering is not established"
            return f
        f.evidence.append(_ref(notification))
    if f.missing:
        return f
    f.verdict = "satisfied"
    f.relationship = "recorded source ownership held through a completion witness before transferred-state admission"
    return f




def evaluate(events, violations, requests=None, *, rules=None) -> list:
    """Evaluate exact request IDs. Ledger strings are never used as identity/proof."""
    facts = Facts(events)
    selected = facts.attempts if requests is None else list(dict.fromkeys(requests))
    enabled = set(RULES) if rules is None else set(rules)
    if enabled - set(RULES):
        raise ValueError("unrecognized lifetime rule")
    findings = []
    for rid in selected:
        if "lease_valid_at_use" in enabled:
            findings.append(lease_valid_at_use(facts, violations, rid))
    return findings
