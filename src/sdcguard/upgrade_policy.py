"""Pure preservation policy for already recomputed canonical evidence reports.

Owner: Hari Krishna Kancharla.
The caller must verify and snapshot both packets first. This module performs no
I/O, authenticates no receipt, and never establishes a runtime upgrade witness.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from . import evidence_catalog as catalog

POLICY_NAME = "preserve-named-contracts.v1"
MAX_REPORT_JSON_NODES = 1_000_000
_SCHEMA = "sdcguard.evidence-report.v1"
_VLLM = "5df63a7203cdb9328d824a27b682e77c7b9928be"
_PATCH = "a669acdbc188d59439ca3ba74bba9a18447578b997be3f61278d45fd6a5f337d"
_SGLANG = "baa9befa381511cd4d56f83980951e5a45fffc6d"
_MISSING = object()
_POSITIVE = frozenset({"satisfied", "correctly_rejected", "recovered", "refused", "not_applicable"})
_UNKNOWN = frozenset({"unknown", "not_exercised"})
_CAPACITY_FACTS = ("advertised_context_tokens", "configured_context_tokens", "block_or_page_size_tokens")
_SCOPE_NOTES = (
    "Only the selected named static, recorded-event and source-fragment contracts are compared.",
    "The caller must verify captured inputs and recompute both reports before this pure comparison.",
    "Static identity and startup facts are operator-supplied consistency evidence, not deployment attestation.",
    "Fresh trusted CPU component execution concerns its own pinned fragments, not an upgraded deployment build.",
    "Recorded-event scope ends at source lifetime and scheduler admission; actual model reads and kernel identities are not observed.",
    "A copied recording is unchanged evidence, not a candidate execution witness.",
    "Local hashes do not establish authenticity, runtime isolation, loaded weights or whole-model correctness.",
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def policy_descriptor():
    """Return an independent JSON-safe description of the fixed installed policy."""
    return {
        "name": POLICY_NAME,
        "semantic_revision": 1,
        "catalog_sha256": hashlib.sha256((json.dumps(catalog.catalog(), sort_keys=True, indent=2,
                                                       allow_nan=False) + "\n").encode()).hexdigest(),
        "input_contract": "already verified, snapshotted and recomputed canonical reports",
        "runtime_upgrade_witness": "not established",
        "clauses": [
            "preserve semantic identity, required rules, subjects, scopes and resource meanings",
            "allow ordinary pool targets only when both worker startup contracts are satisfied",
            "preserve enabled checksum and configured canary action under exact source semantics",
            "do not order timeout changes, real-KV mode changes or distinct client outcomes",
            "unknown facts and unmapped input changes remain comparison gaps",
            "a repaired negative baseline is not silently approved",
        ],
        "reviewed_source_transitions": [
            {"id": "vllm-032-saved-overlay-add", "engine": "vllm", "version": "0.32.0",
             "build_commit": _VLLM, "from_runtime_patch_sha256": None,
             "to_runtime_patch_sha256": _PATCH, "scope": "first-direction NIXL pull"},
            {"id": "vllm-032-saved-overlay-remove", "engine": "vllm", "version": "0.32.0",
             "build_commit": _VLLM, "from_runtime_patch_sha256": _PATCH,
             "to_runtime_patch_sha256": None, "scope": "first-direction NIXL pull"},
        ],
    }


def _identity(finding):
    return {key: deepcopy(finding[key]) for key in
            ("rule_id", "semantic_revision", "evidence_kind", "scope", "resource_kind", "subject", "scenario")}


def _key(finding):
    return tuple(_json(finding.get(key)) for key in
                 ("rule_id", "semantic_revision", "evidence_kind", "scope", "resource_kind", "subject", "scenario"))


def _gap(result, code, section, message, **details):
    row = dict(code=code, section=section, message=message, **deepcopy(details))
    if row not in result["gaps"]:
        result["gaps"].append(row)


def _finding(report, rule, worker):
    return next((row for row in report["findings"] if row["rule_id"] == rule
                 and row["evidence_kind"] == "static_config" and row["subject"] == {"worker": worker}), None)


def _fact(report, rule, worker, key):
    row = _finding(report, rule, worker)
    return row.get("facts", {}).get(key, _MISSING) if row else _MISSING


def _diff(before, after, path=()):
    if before is _MISSING and isinstance(after, dict) and after:
        for key in sorted(after):
            yield from _diff(_MISSING, after[key], path + (key,))
        return
    if after is _MISSING and isinstance(before, dict) and before:
        for key in sorted(before):
            yield from _diff(before[key], _MISSING, path + (key,))
        return
    if type(before) is dict and type(after) is dict:
        for key in sorted(set(before) | set(after)):
            yield from _diff(before.get(key, _MISSING), after.get(key, _MISSING), path + (key,))
    elif before is _MISSING or after is _MISSING or type(before) is not type(after) or before != after:
        yield path, before, after


def _delta(result, section, path, before, after, classification, reason):
    result["input_deltas"].append({
        "section": section, "path": list(path),
        "baseline_present": before is not _MISSING, "candidate_present": after is not _MISSING,
        "baseline": None if before is _MISSING else deepcopy(before),
        "candidate": None if after is _MISSING else deepcopy(after),
        "classification": classification, "reason": reason,
    })


def _source_equal(before, after):
    return all(before.get(key) == after.get(key) and before.get(key) is not None
               for key in ("engine", "version", "build_commit")) and before.get("components", {}) == after.get("components", {})


def _sglang(before, after):
    return (_source_equal(before, after) and before.get("engine") == "sglang"
            and before.get("version") == "0.5.22" and before.get("build_commit") == _SGLANG)


def _overlay_transition(before, after):
    for worker in (before, after):
        connector = worker.get("connector", {})
        extra = connector.get("extra_config", {})
        if not isinstance(extra, dict):
            return None
        if (worker.get("engine") != "vllm" or worker.get("version") != "0.32.0"
                or worker.get("build_commit") != _VLLM or connector.get("name") != "NixlConnector"
                or connector.get("transfer_mode") != "pull" or connector.get("bidirectional") is not False
                or extra.get("bidirectional_kv_xfer") is not False):
            return None
    left, right = (worker.get("components", {}).get("runtime_patch_sha256", _MISSING)
                   for worker in (before, after))
    if (left, right) == (None, _PATCH):
        return "vllm-032-saved-overlay-add"
    if (left, right) == (_PATCH, None):
        return "vllm-032-saved-overlay-remove"
    return None


def _default_equivalent(base, candidate, left, right, worker, path, old, new):
    if old is not _MISSING and new is not _MISSING:
        return False
    if not _source_equal(left, right):
        return False
    spec = {
        ("parameters", "disaggregation_enable_kv_checksum"): ("protection.checksum_configured", "state"),
        ("parameters", "kv_canary"): ("protection.canary_configured", "mode"),
        ("parameters", "kv_canary_real_data"): ("protection.canary_configured", "real_kv_mode"),
        ("environment", "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE"): ("protection.deferred_release", "state"),
        ("environment", "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT"): ("protection.deferred_release", "deferred_timeout_seconds"),
        ("connector", "extra_config", "kv_lease_verification"): ("protection.retirement", "state"),
        ("connector", "extra_config", "bidirectional_kv_xfer"): ("protection.retirement", "bidirectional_kv_xfer"),
    }.get(path)
    if not spec:
        return False
    a, b = (_finding(report, spec[0], worker) for report in (base, candidate))
    if not a or not b or a["verdict"] in _UNKNOWN or b["verdict"] in _UNKNOWN:
        return False
    x, y = (row["facts"].get(spec[1], _MISSING) for row in (a, b))
    return x is not _MISSING and x == y


def _complete_static(document):
    model = document.get("model", {})
    if not (all(isinstance(model.get(key), str) and model[key] for key in ("id", "revision", "architecture"))
            and re.fullmatch(r"[0-9a-fA-F]{40}", model["revision"])
            and type(document.get("advertised_context_tokens")) is int and document["advertised_context_tokens"] > 0):
        return False
    for worker in document.get("workers", []):
        if (not all(isinstance(worker.get(key), str) and worker[key] for key in ("id", "engine", "version", "build_commit", "role"))
                or not re.fullmatch(r"[0-9a-fA-F]{40}", worker["build_commit"])
                or worker["role"] not in ("prefill", "decode")):
            return False
    return bool(document.get("workers"))


def _static_inputs(base, candidate, result):
    documents = [report["native_sections"].get("static", {}).get("input", {}) for report in (base, candidate)]
    left, right = documents
    global_ok = _complete_static(left) and _complete_static(right)
    if not global_ok:
        _gap(result, "incomplete_baseline_or_candidate_identity", "static", "Both semantic identities must be complete.")
    workers = [{worker["id"]: worker for worker in doc.get("workers", [])} for doc in documents]
    worker_ok = {name: True for name in set(workers[0]) & set(workers[1])}
    orders = [[worker["id"] for worker in doc.get("workers", [])] for doc in documents]
    if orders[0] != orders[1]:
        _delta(result, "static", ("worker_order",), orders[0], orders[1],
               "representation" if set(orders[0]) == set(orders[1]) else "incomparable",
               "Worker order is representational only when worker membership and roles are preserved.")
    if set(workers[0]) != set(workers[1]):
        global_ok = False
        _gap(result, "worker_inventory_changed", "static", "Worker IDs may not be added, removed or renamed.")
    roots = [{key: value for key, value in doc.items() if key != "workers"} for doc in documents]
    for path, old, new in _diff(*roots):
        harmless = path == ("deployment_id",)
        _delta(result, "static", path, old, new, "representation" if harmless else "incomparable",
               "Display label changed." if harmless else "The deployment contract identity changed.")
        if not harmless:
            global_ok = False
            _gap(result, "deployment_identity_changed", "static", "Model and context contracts must remain unchanged.", path=list(path))
    for name in sorted(set(workers[0]) | set(workers[1])):
        a, b = (collection.get(name, _MISSING) for collection in workers)
        if a is _MISSING or b is _MISSING:
            _delta(result, "static", ("workers", name), a, b, "incomparable", "Worker inventory changed.")
            continue
        transition = _overlay_transition(a, b)
        for path, old, new in _diff(a, b):
            classification, reason = "incomparable", "No installed policy classifies this input change."
            if path and path[0] == "startup_log":
                classification, reason = "evidence_receipt", "Both startup receipts must be independently verified by the caller."
            elif path == ("components", "runtime_patch_sha256") and transition:
                classification, reason = "reviewed_source_transition", transition
            elif _default_equivalent(base, candidate, a, b, name, path, old, new):
                classification, reason = "same_resolved_default", "The existing evaluator resolves the same mapped value."
            elif path in (("parameters", "num_gpu_blocks_override"), ("parameters", "max_total_tokens")):
                expected = "max_total_tokens" if a.get("engine") == "sglang" else "num_gpu_blocks_override"
                startup = [_finding(report, "capacity.startup", name) for report in (base, candidate)]
                if path[-1] == expected and all(row and row["verdict"] == "satisfied" for row in startup):
                    classification, reason = "preserved_capacity_target", "Both bound startup contracts retain the same capacity property."
                elif (path[-1] == expected and all(row and row["verdict"] in ("satisfied", "violated") for row in startup)
                      and all(startup[0]["facts"].get(key, _MISSING) == startup[1]["facts"].get(key, _MISSING)
                              and key in startup[0]["facts"] for key in _CAPACITY_FACTS)):
                    classification, reason = "native_capacity_failure_or_remediation", "The same mapped capacity contract has a known negative result on at least one side."
            elif _sglang(a, b) and path in (("parameters", "disaggregation_enable_kv_checksum"), ("parameters", "kv_canary")):
                classification, reason = "configured_protection_state", "The explicit configured-state policy compares this setting."
            elif _sglang(a, b) and path == ("environment", "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE"):
                classification, reason = "native_protection_setting", "The native retention policy judges the enabled state."
            elif (a.get("engine") == "vllm" and _source_equal(a, b)
                  and path == ("connector", "extra_config", "kv_lease_verification")):
                classification, reason = "native_protection_setting", "The native verifier policy judges the enabled state."
            _delta(result, "static", ("workers", name) + path, old, new, classification, reason)
            if classification == "incomparable":
                worker_ok[name] = False
                _gap(result, "unclassified_input_delta", "static", reason, subject={"worker": name}, path=list(path))
    return {"global": global_ok, "workers": worker_ok, "documents": documents, "source_workers": workers}


def _other_identities(base, candidate, result, section):
    before, after = (report["identities"].get(section, {}) for report in (base, candidate))
    comparable = bool(before) and bool(after)
    if not comparable:
        _gap(result, "missing_section_identity", section, "Both section identities are required.")
    for path, old, new in _diff(before, after):
        classification = "incomparable"
        reason = "No installed identity transition covers this change."
        if section == "recorded" and path == ("packet_snapshot_sha256",):
            reason = "A changed recording has no bound reusable stimulus identity in this schema."
        _delta(result, section, path, old, new, classification, reason)
        comparable = False
        _gap(result, "section_identity_changed", section, reason, path=list(path))
    if section == "recorded":
        for identity in (before, after):
            model = identity.get("model", {})
            if (not all(identity.get(key) for key in ("engine", "engine_version", "nixl_version", "topology", "source_identity", "configuration", "packet_snapshot_sha256"))
                    or not model.get("id") or not isinstance(model.get("revision"), str)
                    or not re.fullmatch(r"[0-9a-fA-F]{40}", model["revision"])):
                comparable = False
                _gap(result, "incomplete_recorded_identity", section, "Recorded source, immutable model and scenario identities must be complete.")
    return comparable


def _resolved(row, index):
    if (row["rule_id"] != "capacity.planned" or row["evidence_kind"] != "static_config"
            or row.get("required") is not False
            or row["verdict"] != "unknown" or row.get("resolved_by") != "capacity.startup"
            or row["facts"].get("state") not in ("upper_bound_only", "automatic_pool")):
        return False
    return any(other["rule_id"] == "capacity.startup" and other["subject"] == row["subject"]
               and other["evidence_kind"] == "static_config" and other["verdict"] == "satisfied"
               and other.get("required") is True for other in index.values())


def _comparable(row, contexts):
    section = {"static_config": "static", "source_fragment_cpu": "components", "recorded_engine_events": "recorded"}.get(row["evidence_kind"])
    if section is None:
        return False
    value = contexts.get(section, False)
    if section != "static":
        return value
    worker = row["subject"].get("worker") if isinstance(row["subject"], dict) else None
    return value["global"] and (value["workers"].get(worker, False) if worker is not None else all(value["workers"].values()))


def _state_policy(base, candidate, before, after, result, context, comparable):
    if after["evidence_kind"] != "static_config":
        if after["rule_id"] == "lease_valid_at_use":
            left = before["facts"].get("native_finding", {}).get("protection")
            right = after["facts"].get("native_finding", {}).get("protection")
            if left != right:
                _gap(result, "recorded_protection_changed", "recorded", "Recorded protection mechanisms are not automatically ordered.", finding=_identity(after))
        return
    rule = after["rule_id"]
    worker = after["subject"].get("worker") if isinstance(after["subject"], dict) else None
    a, b = before["facts"], after["facts"]
    if rule.startswith("capacity."):
        for key in _CAPACITY_FACTS:
            if key in a and key in b and a[key] != b[key]:
                _gap(result, "capacity_contract_changed", "static", "Capacity target exceptions do not change context or geometry.", finding=_identity(after), fact=key)
    if worker is None or rule not in ("protection.checksum_configured", "protection.canary_configured"):
        return
    source_left, source_right = (items.get(worker, {}) for items in context["source_workers"])
    if not _sglang(source_left, source_right):
        return
    changed = False
    clause = None
    if rule == "protection.checksum_configured":
        old, new = a.get("state"), b.get("state")
        changed = old != new
        if old == "on" and new == "off":
            clause = "enabled_checksum_removed"
        enabling = old == "off" and new == "on"
    else:
        old, new = a.get("mode"), b.get("mode")
        changed = old != new
        if (old == "raise" and new in ("log", "none")) or (old == "log" and new == "none"):
            clause = "configured_canary_action_weakened"
        enabling = (old == "none" and new in ("log", "raise")) or (old == "log" and new == "raise")
        if a.get("real_kv_mode", _MISSING) != b.get("real_kv_mode", _MISSING):
            _gap(result, "real_kv_mode_changed", "static", "Configured real-KV modes have no automatic detection ordering.", finding=_identity(after))
    if clause:
        result["policy_violations"].append(dict(clause=clause, finding=_identity(after),
            baseline=old, candidate=new, native_baseline_verdict=before["verdict"], native_candidate_verdict=after["verdict"],
            regression_attribution="regressed" if comparable else "not established",
            scope="configured protection preservation only; runtime verification is not established"))
    elif changed:
        coverage = _finding(candidate, "detection.architecture_coverage", worker)
        if not enabling or not coverage or coverage["verdict"] != "satisfied":
            _gap(result, "protection_transition_unestablished", "static", "Enabling a setting requires exact source semantics and established source coverage.", finding=_identity(after))


def _validate(report):
    if not isinstance(report, dict) or catalog._json_errors(report, max_nodes=MAX_REPORT_JSON_NODES):
        raise ValueError("comparison needs finite JSON canonical reports")
    if (report.get("schema") != _SCHEMA or not isinstance(report.get("findings"), list)
            or not isinstance(report.get("identities"), dict) or not isinstance(report.get("native_sections"), dict)
            or not isinstance(report.get("profile"), dict)):
        raise ValueError("unsupported canonical evidence report")
    profile = report["profile"]
    if (set(profile) != {"static", "components", "recorded"} or type(profile["static"]) is not bool
            or type(profile["recorded"]) is not bool or profile["components"] not in ("none", "all")):
        raise ValueError("unsupported evidence profile")
    for row in report["findings"]:
        if (not isinstance(row, dict) or type(row.get("required")) is not bool
                or any(key not in row for key in ("rule_id", "semantic_revision", "evidence_kind", "scope", "resource_kind", "subject", "scenario", "verdict", "facts", "missing"))
                or not all(isinstance(row[key], str) for key in ("rule_id", "evidence_kind", "scope", "resource_kind", "verdict"))
                or not isinstance(row["facts"], dict)):
            raise ValueError("malformed canonical finding")
    if profile["static"]:
        section = report["native_sections"].get("static")
        document = section.get("input") if isinstance(section, dict) else None
        if (not isinstance(document, dict) or not isinstance(document.get("workers"), list)
                or not isinstance(document.get("model", {}), dict)
                or any(not isinstance(worker, dict) or not isinstance(worker.get("id"), str)
                       for worker in document["workers"])
                or len({worker["id"] for worker in document["workers"]}) != len(document["workers"])):
            raise ValueError("malformed captured static input")


def compare(base_report, candidate_report):
    """Compare trusted recomputed reports without I/O or a final gate decision.

    Input flags such as status, trusted or verified never authorize a finding.
    The caller owns packet integrity and current CPU-execution verification.
    Empty policy gaps do not establish authenticity or runtime upgrade evidence.
    """
    for report in (base_report, candidate_report):
        _validate(report)
    base, candidate = deepcopy(base_report), deepcopy(candidate_report)
    descriptor = policy_descriptor()
    result = {"policy": descriptor, "policy_sha256": hashlib.sha256(_json(descriptor).encode()).hexdigest(),
              "matched_outcomes": [], "input_deltas": [], "native_candidate_violations": [],
              "policy_violations": [], "gaps": [], "newly_established": [], "remediations": [],
              "scope_notes": list(_SCOPE_NOTES), "runtime_upgrade_witness": "not established"}
    if base["profile"] != candidate["profile"]:
        _delta(result, "profile", (), base["profile"], candidate["profile"], "incomparable", "Selected evidence sections changed.")
        _gap(result, "profile_changed", "profile", "The default policy preserves the same selected evidence scopes.")
    for label, report in (("baseline", base), ("candidate", candidate)):
        if (report.get("catalog_schema") != catalog.SCHEMA or report.get("catalog_semantic_revision") != catalog.SEMANTIC_REVISION
                or report.get("catalog_sha256") != descriptor["catalog_sha256"]):
            _gap(result, "catalog_identity_changed", label, "The report does not name the installed catalog semantics.")
    catalog_ok = not any(row["code"] == "catalog_identity_changed" for row in result["gaps"])
    contexts = {}
    selected = {"static": candidate["profile"]["static"] or base["profile"]["static"],
                "recorded": candidate["profile"]["recorded"] or base["profile"]["recorded"],
                "components": candidate["profile"]["components"] == "all" or base["profile"]["components"] == "all"}
    for section, enabled in selected.items():
        if enabled:
            contexts[section] = (_static_inputs(base, candidate, result) if section == "static"
                                 else _other_identities(base, candidate, result, section))
            if section == "static":
                contexts[section]["global"] = contexts[section]["global"] and catalog_ok
            else:
                contexts[section] = contexts[section] and catalog_ok
    indexes = []
    invalid = []
    for label, report in (("baseline", base), ("candidate", candidate)):
        index = {}
        invalid_keys = set()
        for row in report["findings"]:
            errors = catalog.validate_finding(row, row["evidence_kind"])
            if errors:
                _gap(result, "finding_contract_changed", label, "; ".join(errors), finding=_identity(row))
                invalid_keys.add(_key(row))
            key = _key(row)
            if key in index:
                _gap(result, "duplicate_finding_identity", label, "A finding identity occurs more than once.", finding=_identity(row))
            index[key] = row
        if not any(row["required"] for row in index.values()):
            _gap(result, "no_required_findings", label, "A comparison cannot pass vacuously.")
        indexes.append(index)
        invalid.append(invalid_keys)
    left, right = indexes
    for key in sorted(set(left) | set(right)):
        before, after = left.get(key), right.get(key)
        if before is None or after is None:
            row = before or after
            _gap(result, "finding_identity_missing", "comparison", "An exact rule, subject, scenario, scope or resource identity is absent on one side.",
                 finding=_identity(row), missing_from="baseline" if before is None else "candidate")
            if after and key not in invalid[1] and after["required"] and after["verdict"] == "violated":
                result["native_candidate_violations"].append(dict(finding=_identity(after), regression_attribution="not established", baseline_verdict=None))
            continue
        comparable = (_comparable(after, contexts) and base["profile"] == candidate["profile"]
                      and key not in invalid[0] and key not in invalid[1])
        required = before["required"] or after["required"]
        resolution = _resolved(after, right)
        if before["required"] and not after["required"] and not resolution:
            comparable = False
            _gap(result, "required_finding_downgraded", "comparison", "A required finding cannot become optional.", finding=_identity(after))
        if not comparable:
            _gap(result, "finding_incomparable", "comparison", "The finding's semantic identity or an input delta is not comparable.", finding=_identity(after))
        state_gap_start, state_violation_start = len(result["gaps"]), len(result["policy_violations"])
        if key not in invalid[0] and key not in invalid[1]:
            _state_policy(base, candidate, before, after, result, contexts.get("static", {}), comparable)
        old, new = before["verdict"], after["verdict"]
        if new == "violated":
            attribution = ("regressed" if comparable and old in _POSITIVE else
                           "still_violated" if comparable and old == "violated" else "not established")
            if after["required"] and key not in invalid[1]:
                result["native_candidate_violations"].append(dict(finding=_identity(after), regression_attribution=attribution, baseline_verdict=old))
            outcome = attribution if attribution != "not established" else "candidate_violated"
        elif resolution:
            outcome = "resolved_by_startup"
        elif new in _UNKNOWN:
            outcome = "inconclusive"
            if required:
                _gap(result, "required_candidate_unknown", "comparison", "A required candidate fact remains unknown or unexercised.", finding=_identity(after))
        elif old in _UNKNOWN and new in _POSITIVE:
            outcome = "newly_established" if comparable else "inconclusive"
            if comparable:
                result["newly_established"].append(dict(finding=_identity(after), baseline_verdict=old, candidate_verdict=new))
        elif old == "violated" and new in _POSITIVE:
            outcome = "remediated" if comparable else "inconclusive"
            result["remediations"].append(dict(finding=_identity(after), baseline_verdict=old, candidate_verdict=new, comparable=comparable))
            _gap(result, "baseline_not_approved", "comparison", "A repaired negative baseline needs a separate expected-negative criterion.", finding=_identity(after))
        elif old != new:
            outcome = "inconclusive"
            _gap(result, "outcome_changed_without_ordering", "comparison", "Distinct terminal outcomes and applicability states are not ranked.", finding=_identity(after), baseline_verdict=old, candidate_verdict=new)
        else:
            outcome = "preserved" if comparable else "inconclusive"
        if old == new:
            if len(result["policy_violations"]) > state_violation_start:
                outcome = "regressed" if comparable else "policy_violated"
            elif len(result["gaps"]) > state_gap_start:
                outcome = "inconclusive"
        fact_deltas = [{"path": list(path), "baseline_present": old_fact is not _MISSING,
                       "candidate_present": new_fact is not _MISSING,
                       "baseline": None if old_fact is _MISSING else deepcopy(old_fact),
                       "candidate": None if new_fact is _MISSING else deepcopy(new_fact)}
                      for path, old_fact, new_fact in _diff(before["facts"], after["facts"])]
        result["matched_outcomes"].append(dict(finding=_identity(after), baseline_verdict=old, candidate_verdict=new,
                                              outcome=outcome, comparable=comparable, required=required, fact_deltas=fact_deltas))
    return result
