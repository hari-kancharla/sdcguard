"""Compare saved transfer coverage without changing either release policy.

Owner: Hari Krishna Kancharla.
This command performs no engine execution and qualifies no new engine build.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

from . import evidence_io as io, required_gate as gate, transfer_coverage as transfer

SCHEMA = "sdcguard.transfer-upgrade.v1"
POLICY = "preserve-recorded-transfer-coverage.v1"
ERRORS = (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError)
LIMITS = [
    "Changes describe recorded coverage, not a cause or an engine correctness failure.",
    "Only matching workloads and recorded configuration can support coverage preservation.",
    "Missing evidence is UNVERIFIABLE; REGRESSED requires a recorded native mismatch after a verified baseline.",
    "Identical inputs, different paths, run IDs or packet hashes do not prove independent builds.",
    "Only one engine revision is qualified by the installed transfer checker.",
    "No changed-build transition is qualified; the upgrade decision remains HOLD.",
    "Input hashes establish local consistency, not execution or recorder authenticity.",
    "Full-prompt release decisions are recomputed unchanged. Cached tokens remain outside transfer scope.",
]
_STARTUP_VOLATILE = {
    "run_id", "packet_sha256", "engine_session_id", "process_id", "actual_model",
    "actual_tokenizer", "snapshot", "source_revision", "source_identity",
    "source_profile", "native_transport",
}
_REQUIRED_CONFIG = {
    "model", "model_revision", "prefix_caching", "checksum_enabled", "parallel",
    "dtype", "kv_dtype", "block_size", "group_specs", "gpu_blocks", "max_model_len",
    "max_num_seqs", "connector", "role", "transfer_role", "transfer_extra",
    "transport_profile", "model_runner_v2", "async_scheduling", "chunked_prefill",
    "speculative_disabled", "flashinfer_sampler_enabled", "multiprocess_method",
    "request_id_randomization_enabled",
}
_TRANSFER_VOLATILE = {
    "remote_request_id", "remote_block_ids", "remote_engine_id", "remote_host",
    "remote_port", "remote_blocks_expiry_time", "remote_registration_generation",
    "remote_prefill_cached_tokens", "remote_num_tokens", "kv_checksums",
}


def _hash(value):
    return io.digest(io.json_bytes(value))


def _without(value, names):
    if type(value) is not dict:
        raise io.EvidenceError("comparison metadata must be an object")
    return {k: v for k, v in value.items() if k not in names}


def _key(row):
    arm, slot = row["arm_id"], row["slot"]
    if type(arm) is not str or not arm or type(slot) is not int or slot < 0:
        raise io.EvidenceError("invalid comparison slot")
    return arm, slot


def _index(rows):
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise io.EvidenceError("duplicate comparison slot")
        result[key] = row
    return result


def _engine_identity(preflight):
    """Hash engine bytes, not run-specific witness or observer package fields."""
    installed = preflight["installed_engine"]
    result = {k: installed[k] for k in ("source_revision", "source_files", "compiled_members")}
    payload = installed["precompiled_payload"]
    # Generated metadata and bytecode are not the installed native payload.
    result["extracted_members"] = payload["extracted_members"]
    if not all(result[k] for k in result):
        raise io.EvidenceError("engine identity is incomplete")
    return result


def _runtime_identity(preflight):
    installed = preflight["installed_engine"]
    py = installed["python_identity"]
    versions = installed["versions"]
    required = {"vllm", "torch", "triton", "nixl", "nixl-cu12", "nixl-cu13"}
    version = py.get("version")
    if (type(versions) is not dict or not required <= versions.keys()
            or any(type(v) is not str or re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,127}", v) is None
                   for k, v in versions.items() if k != "uv")
            or py.get("implementation") != "cpython" or py.get("platform") != "linux"
            or py.get("soabi") != "cpython-312-x86_64-linux-gnu"
            or type(version) is not list or len(version) != 3
            or any(type(n) is not int or not 0 <= n <= 999 for n in version)
            or version[:2] != [3, 12]):
        raise io.EvidenceError("recorded runtime identity is missing, unknown or unsupported")
    return dict(versions={k: v for k, v in versions.items() if k != "uv"},
                python={k: py[k] for k in ("implementation", "version", "platform", "soabi")})


def _contracts(files, preflight):
    plan = gate.object_at(files, "client/REQUEST_PLAN.json")
    inventory = gate.object_at(files, "client/CLIENT_INVENTORY.json")
    captures = gate.object_at(files, "CAPTURES.json")["captures"]
    raw_rows = _index(inventory["requests"])
    workload = _without(plan, {"run_id", "packet_sha256", "source_revision", "arms"})
    workload["arms"] = []
    slots, sessions, configs = {}, [], []
    for arm in plan["arms"]:
        normalized = _without(arm, {"engine_session_id", "requests"})
        normalized["requests"] = []
        for row in arm["requests"]:
            key = _key(dict(arm_id=arm["arm_id"], slot=row["slot"]))
            if key in slots:
                raise io.EvidenceError("duplicate planned comparison slot")
            spec = _without(row, {"client_request_id", "prefill_request_id"})
            spec["client_inputs"] = {}
            raw = raw_rows.get(key)
            if raw is None:
                raise io.EvidenceError("client input missing from comparison slot")
            for role in ("prefill", "decode"):
                stage = raw[role]
                body = _without(stage["request_body"], {"request_id", "kv_transfer_params"})
                body["kv_transfer_params"] = _without(stage["request_body"]["kv_transfer_params"], _TRANSFER_VOLATILE)
                headers = _without(stage["request_headers"], {"X-Request-Id"})
                spec["client_inputs"][role] = dict(method=stage["method"], path=stage["path"],
                                                    body_sha256=_hash(body), headers_sha256=_hash(headers))
            slots[key] = _hash(spec)
            normalized["requests"].append(spec)
        workload["arms"].append(normalized)
    if set(raw_rows) != set(slots):
        raise io.EvidenceError("client inventory and planned comparison slots differ")
    for capture in captures:
        session = capture["engine_session_id"]
        if type(session) is not str or not session:
            raise io.EvidenceError("engine session missing")
        sessions.append(session)
        for role, value in (("decode", capture), ("prefill", capture["prefill_capture"])):
            startup = value["startup"]
            if not _REQUIRED_CONFIG <= startup.keys():
                raise io.EvidenceError("recorded comparison configuration is incomplete")
            configs.append(dict(arm_id=capture["arm_id"], role=role,
                                config=_without(startup, _STARTUP_VOLATILE)))
    model = preflight["model_cache"]
    if not all(model.get(k) for k in ("model", "revision", "files")):
        raise io.EvidenceError("model and tokenizer file identities are missing")
    configuration = dict(engines=configs, model={k: model[k] for k in ("model", "revision", "files")},
                         runtime=_runtime_identity(preflight))
    return dict(workload_sha256=_hash(workload), config_sha256=_hash(configuration),
                slots=slots, sessions=sorted(sessions), run_id=plan["run_id"])


def _side(raw):
    """Recompute from raw inputs. Saved verdicts are never comparison authority."""
    hashes = {k: io.digest(v) if v is not None else None for k, v in
              zip(("result", "archive", "preflight"), raw)}
    side = dict(inputs_sha256=hashes, errors=[], assessment=None, contracts=None, engine_identity_sha256=None)
    try:
        side["assessment"] = transfer.evaluate(*raw)
        files = gate.read_archive(raw[1])
        preflight = io.parse_json(raw[2]) if raw[2] is not None else {}
        side["contracts"] = _contracts(files, preflight)
        global_gaps = [r for r in side["assessment"]["hold_reasons"] if r not in {
            "required_checksum_coverage_incomplete", "transfer_comparison_obligation_incomplete",
            "no_verified_transfer_requests"}]
        side["errors"].extend("input_qualification:" + r for r in global_gaps)
        if not global_gaps:
            side["engine_identity_sha256"] = _hash(_engine_identity(preflight))
    except ERRORS as error:
        side["errors"].append("evidence_unverifiable:" + str(error))
    return side


def _view(side):
    assessment, contract = side["assessment"], side["contracts"]
    return dict(inputs_sha256=side["inputs_sha256"], errors=side["errors"],
        engine=assessment["engine"] if assessment else None,
        engine_identity_sha256=side["engine_identity_sha256"],
        recording={k: contract[k] for k in ("run_id", "sessions")} if contract else None,
        workload_sha256=contract["workload_sha256"] if contract else None,
        config_sha256=contract["config_sha256"] if contract else None,
        transfer_decision=assessment["transfer_decision"] if assessment else "HOLD",
        full_prompt_gate=assessment["full_prompt_gate"] if assessment else None,
        summary=assessment["summary"] if assessment else None,
        hold_reasons=assessment["hold_reasons"] if assessment else side["errors"])


def _row_view(row):
    if row is None:
        return None
    return dict({k: row[k] for k in ("client_request_id", "status", "decision", "reasons", "tokens")},
                expected_chunk_positions=row["comparison"]["expected_units"],
                compared_chunk_positions=row["comparison"]["compared_units"])


def _change(before, after, comparable, same_slot):
    if not comparable or not same_slot:
        return "UNVERIFIABLE", ["workload_or_configuration_not_comparable"]
    if before is None or after is None:
        return "UNVERIFIABLE", ["request_missing_from_one_side"]
    if before["decision"] != "PASS":
        return "UNVERIFIABLE", ["baseline_transfer_not_verified"]
    b, a = before["tokens"], after["tokens"]
    scope_keys = ("prompt", "expected_transfer", "decode_cache_outside_transfer", "prefill_cache_reused")
    if (any(type(t[k]) is not int for t in (b, a) for k in scope_keys)
            or any(type(r["comparison"]["expected_units"]) is not list or not r["comparison"]["expected_units"] for r in (before, after))):
        return "UNVERIFIABLE", ["transfer_scope_unverifiable"]
    if (any(b[k] != a[k] for k in scope_keys)
            or before["comparison"]["expected_units"] != after["comparison"]["expected_units"]):
        return "UNVERIFIABLE", ["transfer_exercise_changed"]
    if after["decision"] != "PASS":
        comparison = after["comparison"]
        if (after["status"] == "MISMATCH" and comparison["native_result"] == "mismatch"
                and set(after["reasons"]) == {"native_checksum_mismatch"}
                and set(comparison["reasons"]) == {"native_checksum_mismatch"}
                and comparison["compared_units"] == comparison["expected_units"]):
            return "REGRESSED", ["candidate_recorded_native_mismatch"]
        return "UNVERIFIABLE", ["candidate_transfer_evidence_unverifiable"]
    return "PRESERVED", []


def evaluate(baseline_raw, candidate_raw):
    """Compare two bounded retained recordings, including failed evidence inputs."""
    baseline, candidate = _side(baseline_raw), _side(candidate_raw)
    b, c = baseline["contracts"], candidate["contracts"]
    errors = [f"{label}:{e}" for label, side in (("baseline", baseline), ("candidate", candidate)) for e in side["errors"]]
    same_workload = b is not None and c is not None and b["workload_sha256"] == c["workload_sha256"]
    same_config = b is not None and c is not None and b["config_sha256"] == c["config_sha256"]
    if not same_workload:
        errors.append("workload_not_comparable")
    if not same_config:
        errors.append("configuration_not_comparable")
    comparable = not errors
    bi, ci = baseline["engine_identity_sha256"], candidate["engine_identity_sha256"]
    build_relation = "unknown" if bi is None or ci is None else ("same_recorded_build" if bi == ci else "different_recorded_build")
    same_bytes = baseline["inputs_sha256"] == candidate["inputs_sha256"]
    relation = "unknown"
    if same_bytes:
        relation = "same_evidence"
    elif b is not None and c is not None and not baseline["errors"] and not candidate["errors"]:
        relation = ("distinct_recorded_sessions" if b["run_id"] != c["run_id"] and not set(b["sessions"]) & set(c["sessions"])
                    else "overlapping_recording_identity")
    before = _index(baseline["assessment"]["requests"]) if baseline["assessment"] else {}
    after = _index(candidate["assessment"]["requests"]) if candidate["assessment"] else {}
    rows = []
    for key in sorted(before.keys() | after.keys()):
        left, right = before.get(key), after.get(key)
        same_slot = b is not None and c is not None and key in b["slots"] and b["slots"].get(key) == c["slots"].get(key)
        status, reasons = _change(left, right, comparable, same_slot)
        rows.append(dict(arm_id=key[0], slot=key[1], coverage_change=status, reasons=reasons,
                         lost_passing_evidence=bool(left and left["decision"] == "PASS" and (not right or right["decision"] != "PASS")),
                         baseline=_row_view(left), candidate=_row_view(right)))
    counts = Counter(r["coverage_change"] for r in rows)
    holds = list(errors)
    if counts["REGRESSED"]:
        holds.append("recorded_native_comparison_regressed")
    if counts["UNVERIFIABLE"] or not rows:
        holds.append("coverage_preservation_unverifiable")
    if relation != "distinct_recorded_sessions":
        holds.append("independent_recordings_not_established")
    if build_relation != "different_recorded_build":
        holds.append("changed_engine_build_not_established")
    for label, side in (("baseline", baseline), ("candidate", candidate)):
        assessment = side["assessment"]
        if assessment is None or assessment["transfer_decision"] != "PASS":
            holds.append(label + "_transfer_check_held")
        if assessment is None or assessment["release_decision"] != "PASS":
            holds.append(label + "_full_prompt_policy_held")
    # There is no reviewed second revision or independent build witness in the
    # supported profile. A different digest must not manufacture that authority.
    holds.append("changed_build_transition_not_qualified")
    return dict(schema=SCHEMA, owner="Hari Krishna Kancharla", policy=POLICY,
        scope=transfer.SCOPE, upgrade_decision="HOLD", runtime_upgrade_witness="not established",
        comparison_status="COMPARABLE" if comparable else "UNVERIFIABLE",
        build_relation=build_relation, recording_relation=relation,
        summary=dict(compared_slots=len(rows), preserved=counts["PRESERVED"], regressed=counts["REGRESSED"], unverifiable=counts["UNVERIFIABLE"],
                     lost_passing_evidence=sum(r["lost_passing_evidence"] for r in rows)),
        baseline=_view(baseline), candidate=_view(candidate), requests=rows,
        hold_reasons=list(dict.fromkeys(holds)), limits=LIMITS)


def _read(path):
    names = ["RESULT.json", "ordinary-evidence.zip"]
    if (path / "CPU_PREFLIGHT.json").exists():
        names.append("CPU_PREFLIGHT.json")
    return {name: io.read_file(path, name) for name in names}


def inspect(baseline, candidate):
    paths = (Path(baseline), Path(candidate))
    raw = [_read(path) for path in paths]
    answer = evaluate(*(tuple(item.get(n) for n in ("RESULT.json", "ordinary-evidence.zip", "CPU_PREFLIGHT.json")) for item in raw))
    if any(_read(path) != item for path, item in zip(paths, raw)):
        raise io.EvidenceError("comparison inputs changed during inspection")
    return answer


def render(answer):
    lines = ["Offline upgrade comparison: " + answer["upgrade_decision"]]
    if "summary" in answer:
        s = answer["summary"]
        lines.extend(["Recorded builds: " + answer["build_relation"], "Recordings: " + answer["recording_relation"],
            f"Coverage: {s['preserved']} preserved, {s['regressed']} regressed, {s['unverifiable']} unverifiable",
            "Runtime upgrade witness: " + answer["runtime_upgrade_witness"]])
        for label in ("baseline", "candidate"):
            side = answer[label]
            full = side["full_prompt_gate"]
            lines.append(f"{label}: transfer {side['transfer_decision']}; full-prompt {full['decision'] if full else 'HOLD'}")
        for row in answer["requests"]:
            if row["coverage_change"] != "PRESERVED":
                lines.append(f"{row['arm_id']} slot {row['slot']}: {row['coverage_change']} | " + "; ".join(row["reasons"]))
                for label in ("baseline", "candidate"):
                    value = row[label]
                    if value:
                        lines.append(f"  {label} {value['client_request_id']}: {value['decision']} | " + "; ".join(value["reasons"]))
    lines.extend("HOLD: " + reason for reason in answer["hold_reasons"])
    lines.extend(LIMITS)
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard transfer-compare", description=__doc__)
    parser.add_argument("baseline", help="retained baseline owner-result directory")
    parser.add_argument("candidate", help="retained candidate owner-result directory")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        answer = inspect(args.baseline, args.candidate)
        output = io.json_bytes(answer).decode() if args.json else render(answer)
    except ERRORS as error:
        answer = dict(schema=SCHEMA, upgrade_decision="HOLD", hold_reasons=["evidence_unverifiable:" + str(error)], limits=LIMITS)
        output = io.json_bytes(answer).decode() if args.json else render(answer)
    print(output, end="")
    return 2  # No qualified changed-build transition is supported yet.
