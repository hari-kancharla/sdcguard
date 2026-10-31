"""Offline coverage of recorded native transfer comparisons.

Owner: Hari Krishna Kancharla.
This is a separate obligation. It never relaxes the full-prompt release gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from . import checksum_client as client, checksum_observer_report as checks
from . import evidence_io as io, kv_checksum_observer as observer
from . import prefix_cache_coverage as cache, prefix_cache_observer as cache_observer
from . import required_gate as gate

SCHEMA = "sdcguard.transfer-coverage.v1"
SCOPE = "observed native comparison of planned transfer token-chunk positions"
LIMITS = [
    "Only the pinned same-host eager checksum profile with native cache accounting is supported.",
    "Cached decode tokens are outside this transfer obligation and still hold the full-prompt gate.",
    "The checker validates recorded comparison scope and native results, not checksum arrays offline.",
    "Token-chunk positions are not physical bytes, all model layers, ownership or physical reads.",
    "Cross-machine correctness and runtime qualification remain unknown.",
    "Input hashes show local consistency, not execution attestation or recorder authenticity.",
]


def _unique(values):
    return list(dict.fromkeys(values))


def _integer(value, low=0, high=client.MAX_PROMPT_TOKENS):
    return type(value) is int and low <= value <= high


def _cache_usage(facts, raw, row):
    """Join native accounting without using any checksum coverage verdict."""
    values, issues = {}, []
    for role in ("prefill", "decode"):
        fact = facts[role]
        issues.extend(fact["issues"])
        request_id = row["prefill_request_id"] if role == "prefill" else row["client_request_id"]
        matches = [b for b in fact["bindings"]
                   if b.get("external_request_id") == "cmpl-" + request_id + "-0"]
        if len(matches) != 1:
            issues.append("native_cache_request_join_unverified:" + role)
            continue
        binding = matches[0]
        internal = binding.get("internal_request_id")
        if (type(internal) is not str or not internal
                or sum(b.get("internal_request_id") == internal for b in fact["bindings"]) != 1
                or binding.get("prompt_sha256") != row["prompt_sha256"]
                or type(binding.get("prompt_tokens")) is not int
                or binding["prompt_tokens"] != row["prompt_tokens"]
                or role == "decode" and internal != row.get("internal_request_id")):
            issues.append("native_cache_request_join_unverified:" + role)
            continue
        fact["joined"].add(internal)
        stage = raw.get(role, {}) if type(raw) is dict else {}
        response = stage.get("response_body", {}) if type(stage) is dict else {}
        response = response if type(response) is dict else {}
        usage = response.get("usage", {})
        usage = usage if type(usage) is dict else {}
        value, errors = cache_observer.request_usage(fact["records"], internal,
            row["prompt_tokens"], row["prompt_sha256"], usage.get("completion_tokens"))
        issues.extend(reason + ":" + role for reason in errors)
        if fact["issues"] or errors or type(value) is not dict:
            continue
        if role == "prefill":
            descriptor = response.get("kv_transfer_params", {})
            if (type(descriptor) is not dict or descriptor.get("remote_request_id") != internal
                    or value["num_external_cached_tokens"] != 0
                    or type(descriptor.get("remote_prefill_cached_tokens")) is not int
                    or descriptor["remote_prefill_cached_tokens"] != value["num_cached_tokens"]):
                issues.append("producer_cache_descriptor_counts_unverified")
                continue
        values[role] = value
    return values, _unique(issues)


def transfer_scope(plans, records, *, prompt_tokens, local_tokens, remote_tokens):
    """Derive expected transfer positions from a receive plan, never an intersection.

    The caller must validate the snapshot, native cache accounting and client join.
    This function alone grants no source or recording authority.
    """
    result = dict(status="UNOBSERVABLE", reasons=[], expected_units=None, compared_units=None,
                  expected_tokens=remote_tokens, compared_tokens=None, native_result="unverified",
                  plan_sequence=None, update_sequence=None, verification_sequence=None)
    reasons = result["reasons"]
    if (not _integer(prompt_tokens, 1) or not _integer(local_tokens)
            or not _integer(remote_tokens) or local_tokens + remote_tokens != prompt_tokens):
        reasons.append("native_transfer_token_accounting_unverified")
        return result
    if remote_tokens == 0:
        if plans or records:
            reasons.append("zero_transfer_has_checksum_activity")
        else:
            result.update(status="NOT_EXERCISED", expected_units=[], compared_units=[], compared_tokens=0)
        return result
    if len(plans) != 1 or len(records) != 1 or records[0].get("kind") != "verification":
        reasons.append("transfer_plan_or_verification_missing_or_ambiguous")
        return result
    plan, record = plans[0], records[0]
    load, proof = plan.get("load_plan"), record.get("proof")
    if type(load) is not dict or type(proof) is not dict:
        reasons.append("transfer_plan_metadata_missing")
        return result
    plan_id, plan_event = plan.get("plan_sequence"), plan.get("sequence")
    update, verified = record.get("origin_update_sequence"), record.get("sequence")
    result.update(plan_sequence=plan_id, update_sequence=update, verification_sequence=verified)
    if (any(not _integer(n, 1, 2**63 - 1) for n in (plan_id, plan_event, update, verified))
            or not plan_id < plan_event < update < verified
            or type(proof.get("plan_sequence")) is not int or proof["plan_sequence"] != plan_id
            or not plan.get("request_id") or plan.get("request_id") != record.get("request_id")):
        reasons.append("receive_plan_to_verification_join_unverified")
    if not checks._planned_scope_matches(plan, proof):
        reasons.append("planned_chunk_scope_disagrees_with_verification")
    # These are independently joined native cache counts, not the old gate's
    # expected/compared sets or a count derived from the successful intersection.
    if (type(load.get("num_tokens")) is not int or load["num_tokens"] != prompt_tokens
            or type(load.get("num_local_computed_tokens")) is not int
            or load["num_local_computed_tokens"] != local_tokens):
        reasons.append("receive_plan_disagrees_with_native_cache_accounting")
    groups = load.get("groups")
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict:
        reasons.append("transfer_group_layout_unsupported")
    else:
        group = groups[0]
        gid, size, start, count = (group.get(k) for k in ("group_id", "block_size", "start", "block_count"))
        if (not _integer(gid) or not _integer(size, 1) or not _integer(start)
                or not _integer(count, 1, checks.MAX_UNITS) or local_tokens % size
                or start != local_tokens // size
                or count != (prompt_tokens + size - 1) // size - start):
            reasons.append("receive_plan_does_not_cover_transfer_span")
        else:
            result["expected_units"] = [f"group:{gid}:chunk:{n}" for n in range(start, start + count)]
    derived = observer.classify_verification(record, max_units=checks.MAX_UNITS)
    # Reuse the source-bound primitive validator. The sole irrelevant obligation
    # here is full-prompt coverage; transfer expectations above are independent.
    reasons.extend(r for r in derived["reasons"] if r != observer.CACHED_PREFIX_REASON)
    result.update(native_result=derived["native_result"], compared_units=derived["compared_units"])
    if not result["expected_units"] or result["compared_units"] != result["expected_units"]:
        reasons.append("transfer_comparison_scope_incomplete")
    if derived["native_result"] == "mismatch":
        reasons.append("native_checksum_mismatch")
    elif derived["native_result"] != "passed" or record.get("native_return") is not True:
        reasons.append("native_passing_checksum_not_recorded")
    result["reasons"] = _unique(reasons)
    if not reasons:
        result.update(status="VERIFIED", compared_tokens=remote_tokens)
    elif derived["native_result"] == "mismatch":
        result["status"] = "MISMATCH"
    return result


def evaluate(result_raw, archive_raw, preflight_raw):
    """Inspect all planned requests. Original full-prompt policy stays intact."""
    full_prompt = gate.checksum_gate(result_raw, archive_raw, preflight_raw, ["checksum"])
    # This contains global identity, source, archive, journal and closure issues.
    # No per-request gate verdict is treated as transfer evidence.
    issues = [r for r in full_prompt["hold_reasons"] if r != "required_checksum_coverage_incomplete"]
    files = gate.read_archive(archive_raw)
    plan = gate.object_at(files, "client/REQUEST_PLAN.json")
    inventory = gate.object_at(files, "client/CLIENT_INVENTORY.json")
    captures = gate.object_at(files, "CAPTURES.json")["captures"]
    joined = client.evaluate(files["client/REQUEST_PLAN.json"], files["client/CLIENT_INVENTORY.json"], files["CAPTURES.json"])
    if plan.get("prefix_caching") is not True:
        issues.append("transfer_profile_requires_native_cache_accounting")
    cache_facts = {}
    for arm in plan["arms"]:
        matches = [c for c in captures if c.get("arm_id") == arm["arm_id"]]
        if arm["observer_enabled"] and len(matches) == 1 and plan.get("prefix_caching") is True:
            cache_facts[arm["arm_id"]] = cache.capture_facts(matches[0], plan, arm)
    rows = []
    known_requests = {(row["arm_id"], row["internal_request_id"]) for row in joined["requests"]
                      if type(row["internal_request_id"]) is str}
    for capture in captures:
        for snapshot in capture.get("checksum_snapshots", []):
            for record in snapshot.get("records", []):
                if (type(record) is not dict
                        or (capture["arm_id"], record.get("request_id")) not in known_requests):
                    issues.append("unjoined_native_transfer_record")
    for row in joined["requests"]:
        errors, accounting = [], {}
        if row["client_status"] != "complete":
            errors.append("client_request_completion_unverified")
        if row["binding_status"] != "matched":
            errors.append("client_to_native_request_join_unverified")
        if not row["observer_enabled"]:
            errors.append("observer_disabled")
        raw_rows = [r for r in inventory["requests"] if type(r) is dict
                    and r.get("arm_id") == row["arm_id"] and r.get("slot") == row["slot"]]
        facts = cache_facts.get(row["arm_id"])
        if facts is None or len(raw_rows) != 1:
            errors.append("native_cache_accounting_unavailable")
        else:
            accounting, cache_errors = _cache_usage(facts, raw_rows[0], row)
            errors.extend(cache_errors)
        selected = [c for c in captures if c.get("arm_id") == row["arm_id"]]
        snapshots = selected[0].get("checksum_snapshots", []) if len(selected) == 1 else []
        records = [r for s in snapshots for r in s.get("records", []) if type(r) is dict
                   and type(row["internal_request_id"]) is str and r.get("request_id") == row["internal_request_id"]]
        plans = [r for r in records if r.get("kind") == "recv_plan"]
        other = [r for r in records if r.get("kind") != "recv_plan"]
        decode, prefill = accounting.get("decode", {}), accounting.get("prefill", {})
        local, remote = decode.get("num_local_cached_tokens"), decode.get("num_external_cached_tokens")
        scoped = transfer_scope(plans, other, prompt_tokens=row["prompt_tokens"],
                                local_tokens=local, remote_tokens=remote)
        if decode.get("num_computed_tokens") != 0:
            errors.append("locally_computed_prompt_scope_unsupported")
        if len(accounting) != 2:
            errors.append("both_native_cache_accounts_required")
        for call in row["checksum_calls"]:
            errors.extend(call["recording_issues"])
        for capture in joined["captures"]:
            if capture["arm_id"] == row["arm_id"]:
                errors.extend(capture["issues"])
        status = scoped["status"]
        # A proven zero is visible even if the old capture profile cannot qualify
        # an entirely empty checksum recorder. It never grants a passing check.
        empty_recorder_errors = {"unambiguous_active_checksum_recorder_missing",
                                 "client_to_native_request_join_unverified"}
        if errors and (status != "NOT_EXERCISED" or set(errors) - empty_recorder_errors):
            status = "UNOBSERVABLE"
        rows.append(dict(client_request_id=row["client_request_id"], arm_id=row["arm_id"], slot=row["slot"],
            internal_request_id=row["internal_request_id"], status=status,
            reasons=_unique(errors + scoped["reasons"]), comparison=scoped,
            tokens=dict(prompt=row["prompt_tokens"], expected_transfer=remote,
                compared_transfer=scoped["compared_tokens"] if not errors else None,
                decode_cache_outside_transfer=local, prefill_cache_reused=prefill.get("num_local_cached_tokens"))))
    for facts in cache_facts.values():
        issues.extend(cache.unexpected(facts))
    issues = _unique(issues)
    for row in rows:
        row["reasons"] = _unique(row["reasons"] + issues)
        row["decision"] = "HOLD" if row["reasons"] or row["status"] not in ("VERIFIED", "NOT_EXERCISED") else (
            "NOT_EXERCISED" if row["status"] == "NOT_EXERCISED" else "PASS")
    counts = Counter(row["status"] for row in rows)
    if not rows or not counts["VERIFIED"]:
        issues.append("no_verified_transfer_requests")
    if any(row["decision"] == "HOLD" for row in rows):
        issues.append("transfer_comparison_obligation_incomplete")
    def total(name):
        values = [row["tokens"][name] for row in rows]
        return sum(values) if values and all(type(v) is int for v in values) else None
    expected_sets = [r["comparison"]["expected_units"] for r in rows]
    expected_count = (sum(len(units) for units in expected_sets)
                      if expected_sets and all(type(units) is list for units in expected_sets) else None)
    return dict(schema=SCHEMA, owner="Hari Krishna Kancharla", scope=SCOPE,
        transfer_decision="HOLD" if issues else "PASS", release_decision=full_prompt["decision"],
        full_prompt_gate=full_prompt, runtime_qualified=False, engine=full_prompt["engine"],
        inputs_sha256=dict(result=io.digest(result_raw), archive=io.digest(archive_raw),
                           preflight=io.digest(preflight_raw) if preflight_raw is not None else None),
        summary=dict(planned_requests=len(rows), verified_requests=sum(r["decision"] == "PASS" for r in rows),
            not_exercised_requests=counts["NOT_EXERCISED"], held_requests=sum(r["decision"] == "HOLD" for r in rows),
            statuses=dict(counts), expected_chunk_positions=expected_count,
            prompt_tokens=total("prompt"), expected_transfer_tokens=total("expected_transfer"),
            compared_transfer_tokens=total("compared_transfer"), decode_cached_tokens_outside_transfer=total("decode_cache_outside_transfer")),
        hold_reasons=_unique(issues), requests=rows, limits=LIMITS)


def inspect(path):
    path = Path(path)
    names = ["RESULT.json", "ordinary-evidence.zip"]
    if (path / "CPU_PREFLIGHT.json").exists():
        names.append("CPU_PREFLIGHT.json")
    raw = {name: io.read_file(path, name) for name in names}
    answer = evaluate(raw["RESULT.json"], raw["ordinary-evidence.zip"], raw.get("CPU_PREFLIGHT.json"))
    if any(raw[name] != io.read_file(path, name) for name in names):
        raise io.EvidenceError("owner inputs changed during inspection")
    return answer


def render(answer):
    lines = ["Transfer comparison: " + answer["transfer_decision"],
             "Full-prompt policy: " + answer["release_decision"], SCOPE]
    if "summary" in answer:
        s = answer["summary"]
        lines.append(f"{s['verified_requests']} / {s['planned_requests']} requests verified for this transfer obligation")
        lines.append(f"Transferred tokens compared: {s['compared_transfer_tokens']}; decode-cache tokens outside transfer: {s['decode_cached_tokens_outside_transfer']}")
    lines.extend("HOLD: " + reason for reason in answer["hold_reasons"])
    for row in answer.get("requests", []):
        t = row["tokens"]
        lines.append(f"{row['client_request_id']} | {row['status']} | {row['decision']} | " +
                     ("; ".join(row["reasons"]) or "complete recorded transfer comparison"))
        lines.append(f"  transfer tokens: {t['compared_transfer']} / {t['expected_transfer']}; cached outside transfer: {t['decode_cache_outside_transfer']}")
    lines.extend(answer["limits"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard transfer-check", description=__doc__)
    parser.add_argument("input", help="saved owner-result directory; no runtime execution")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        answer = inspect(args.input)
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError) as error:
        answer = dict(schema=SCHEMA, transfer_decision="HOLD", release_decision="HOLD",
                      hold_reasons=["evidence_unverifiable:" + str(error)], limits=LIMITS)
    print(io.json_bytes(answer).decode() if args.json else render(answer), end="")
    return 0 if answer["transfer_decision"] == "PASS" else 2
