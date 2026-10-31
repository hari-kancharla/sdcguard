"""A sufficient destination-incarnation retirement proof from bounded observations.

Formal rule: for each bound operation o, destination resource r, incarnation g,
and observed release R(o,r,g), every writer in the complete bound writer set
must have an observed terminal fence F(o,r,g,w) before R. A fence is a transport
completion or a drain acknowledged for this incarnation; a deadline is not one.
The binding, writer set, fence, and release must share a complete, lossless,
closed observation domain with a comparable monotonic clock. No rule branch
depends on an engine, request spelling, or a particular failure.

Adapter contract: facts describe actual method inputs/results, not intended
configuration. Resource identities must include the pool/engine namespace;
incarnations come from observed allocation boundaries or an explicitly labelled
protocol hold. The adapter must reject ambiguous mappings, unobserved startup,
missing streams, reused operation IDs, and unsupported layouts. A protocol-hold
incarnation is NOT a physical allocator generation. Completeness receipts are
observer declarations, not cryptographic attestation of an untrusted producer.

This initial rule certifies sufficient CPU component observations. It returns
UNKNOWN for incomplete or contradictory receipts, including timeout releases;
it does not turn absent evidence into a defect. It does not certify physical
DMA, accelerator writes, deployed-engine behavior, or the existing source-lease
rule. Existing evaluators are deliberately independent of this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


SCHEMA = "sdcguard.destination-generation-observations.v1"
SCOPE = "source_fragment_cpu"


@dataclass(frozen=True)
class GenerationAssessment:
    status: str
    reasons: tuple[str, ...]
    obligations: tuple[dict, ...] = ()
    scope: str = SCOPE
    physical_write_safety: str = "unknown"

    def as_dict(self):
        return {"rule": "destination_incarnation_retired_after_writers",
                "status": self.status, "scope": self.scope,
                "physical_write_safety": self.physical_write_safety,
                "reasons": list(self.reasons), "obligations": list(self.obligations)}


def _text(value):
    return isinstance(value, str) and bool(value)


def _integer(value):
    return type(value) is int and value >= 0


def _key(value):
    if (not isinstance(value, dict) or set(value) != {"operation", "resource", "generation"}
            or not _text(value.get("operation")) or not _text(value.get("resource"))
            or not _integer(value.get("generation"))):
        raise ValueError("missing or malformed operation/resource/generation attribution")
    return value["operation"], value["resource"], value["generation"]


def evaluate_generation_obligations(receipt) -> GenerationAssessment:
    """Evaluate normalized observations; malformed input is always UNKNOWN.

    Every raw observation retains its envelope even when it contains no rule
    facts. Thus a sequence hole cannot disappear during normalization. All
    claimed streams must be closed and cover sequence 1 through their last
    observation, without drops, under one actual comparable observer clock.
    """
    reasons = []
    states = {}
    try:
        if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
            raise ValueError("unsupported observation schema")
        if receipt.get("scope") != SCOPE:
            raise ValueError("this rule supports only source fragment CPU observations")
        issues = receipt.get("issues")
        if not isinstance(issues, list) or any(not _text(x) for x in issues):
            raise ValueError("malformed adapter issues")
        reasons.extend(issues)
        coverage = receipt.get("coverage")
        if (not isinstance(coverage, dict) or coverage.get("complete_from_start") is not True
                or coverage.get("closed") is not True):
            raise ValueError("complete startup-to-close coverage was not recorded")
        streams = coverage.get("streams")
        if not isinstance(streams, list) or not streams:
            raise ValueError("required observation streams were not recorded")
        by_stream = {}
        for stream in streams:
            if (not isinstance(stream, dict) or not _text(stream.get("id"))
                    or not _text(stream.get("clock_id")) or type(stream.get("first_seq")) is not int
                    or stream["first_seq"] != 1
                    or not _integer(stream.get("last_seq")) or stream["last_seq"] < 1
                    or type(stream.get("dropped")) is not int or stream["dropped"] != 0
                    or stream.get("closed") is not True):
                raise ValueError("missing, open, or lossy stream coverage")
            if stream["id"] in by_stream:
                raise ValueError("duplicate stream identity")
            by_stream[stream["id"]] = stream
        if len({s["clock_id"] for s in streams}) != 1:
            raise ValueError("stream clocks are not known comparable")
        events = receipt.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError("no actual observations")
        sequences = {name: [] for name in by_stream}
        previous_time = {}
        for event in events:
            if not isinstance(event, dict) or event.get("stream") not in by_stream:
                raise ValueError("uncovered observation stream")
            stream_id = event["stream"]
            stream = by_stream[stream_id]
            seq, mono = event.get("seq"), event.get("mono")
            if (type(seq) is not int or seq < 1 or type(mono) not in (int, float)
                    or not math.isfinite(mono) or event.get("clock_id") != stream["clock_id"]
                    or type(event.get("dropped")) is not int or event["dropped"] != 0):
                raise ValueError("malformed or unhealthy observation envelope")
            if stream_id in previous_time and mono < previous_time[stream_id]:
                raise ValueError("stream monotonic clock went backwards")
            previous_time[stream_id] = mono
            sequences[stream_id].append(seq)
            if not isinstance(event.get("facts"), list):
                raise ValueError("missing observation facts")
        for stream_id, seqs in sequences.items():
            if (len(seqs) != by_stream[stream_id]["last_seq"]
                    or any(seq != expected for expected, seq in enumerate(seqs, 1))):
                raise ValueError("observation stream has missing, duplicate, or reordered receipts")
        # Comparable clocks plus per-stream sequence establish this ordering.
        for event in sorted(events, key=lambda e: (e["mono"], e["stream"], e["seq"])):
            for fact in event["facts"]:
                if not isinstance(fact, dict):
                    raise ValueError("malformed rule fact")
                key = _key(fact.get("key"))
                kind = fact.get("kind")
                if kind == "bind":
                    peers = fact.get("writers")
                    if (not isinstance(peers, list) or not peers or any(not _text(p) for p in peers)
                            or len(set(peers)) != len(peers)
                            or fact.get("generation_basis") not in ("observed_allocation", "protocol_hold")):
                        raise ValueError("complete writer set or generation basis unavailable")
                    if key in states:
                        raise ValueError("operation/incarnation binding was repeated")
                    states[key] = {"writers": set(peers), "fenced": set(), "released": False,
                                   "generation_basis": fact["generation_basis"], "stream": event["stream"]}
                elif kind in ("fence", "release"):
                    state = states.get(key)
                    if state is None:
                        raise ValueError("fence/release has no observed binding")
                    if state["stream"] != event["stream"]:
                        raise ValueError("cross-stream operation/resource mapping was not established")
                    if state["released"]:
                        raise ValueError("receipt follows an already released incarnation")
                    if kind == "fence":
                        writer = fact.get("writer")
                        if (writer not in state["writers"] or writer in state["fenced"]
                                or fact.get("terminal") is not True
                                or fact.get("semantics") not in ("matching_drain_ack", "transport_done")):
                            raise ValueError("terminal writer fence was not established")
                        state["fenced"].add(writer)
                    else:
                        if fact.get("cause") != "observed_release":
                            raise ValueError("a timeout or policy decision is not a terminal fence")
                        if state["fenced"] != state["writers"]:
                            reasons.append("release lacks an observed terminal fence for every bound writer")
                        state["released"] = True
                else:
                    raise ValueError("unsupported rule fact")
        if not states:
            reasons.append("no attributed destination-incarnation operation was observed")
        for state in states.values():
            if not state["released"]:
                reasons.append("destination release was not observed")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        reasons.append(str(exc))
    obligations = tuple({"operation": key[0], "resource": key[1], "generation": key[2],
                         "generation_basis": value["generation_basis"],
                         "writers": sorted(value["writers"]), "fenced": sorted(value["fenced"]),
                         "release_observed": value["released"]}
                        for key, value in states.items())
    return GenerationAssessment("unknown" if reasons else "satisfied",
                                tuple(dict.fromkeys(reasons)), obligations)
