"""Required native-check coverage for retained client requests.

Owner: Hari Krishna Kancharla.
PASS applies to the selected recorded check and scope, not deployment safety.
No engine, SDK, traffic or device work is started by this command.
"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import stat
import zipfile
import zlib

from . import checksum_client as checksum, native_checks as lease, evidence_io as io
from . import evidence_json_budget as budget
from . import checksum_profiles as profiles

CHECKS = ("lease", "checksum")
CHECKSUM_VERSION = "0.33.0rc1.dev205+g29d2f4e03.precompiled"
CONTEXT = ("run_id", "packet_sha256", "arm_id", "engine_session_id")
PROFILE_LIMITS = {"runtime_qualification_pending", "joint_release_profile_not_qualified"}


def read_archive(raw):
    """Bound all archive members; parse only evidence needed by this gate."""
    if type(raw) is not bytes or len(raw) > io.MAX_FILE_BYTES:
        raise io.EvidenceError("owner archive exceeds the input bound")
    files, total = {}, 0
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) > 1024:
                raise io.EvidenceError("owner archive exceeds 1024 files")
            for item in entries:
                io.relative(item.filename)
                if (item.filename in files or item.is_dir() or item.flag_bits & 1
                        or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG)
                        or item.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED)
                        or item.file_size > 32 * 1024 * 1024):
                    raise io.EvidenceError("unsafe or unsupported owner archive member")
                total += item.file_size
                if total > 128 * 1024 * 1024:
                    raise io.EvidenceError("owner archive exceeds 128 MiB expanded bytes")
                files[item.filename] = archive.read(item)
                if len(files[item.filename]) != item.file_size:
                    raise io.EvidenceError("owner member size differs")
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, NotImplementedError, zlib.error) as error:
        raise io.EvidenceError("unreadable owner archive: " + str(error)) from error
    for name in files:
        if any("/".join(name.split("/")[:i]) in files for i in range(1, len(name.split("/")))):
            raise io.EvidenceError("owner archive parent path overlaps a file")
    return files


def same(a, b):
    return io.json_bytes(a) == io.json_bytes(b)


def object_at(files, name):
    value = io.parse_json(files[name])
    if type(value) is not dict:
        raise io.EvidenceError("expected object: " + name)
    return value


def decide(requests, required, issues=(), *, total=None, engine=None, arm=None):
    """Apply one coverage policy; never merge checks from separate runs."""
    if not required or any(check not in CHECKS for check in required) or len(set(required)) != len(required):
        raise io.EvidenceError("require lease, checksum, or both once each")
    holds = list(issues)
    if len(required) != 1:
        holds.append("combined_lease_checksum_not_supported")
    if not requests:
        holds.append("no_verified_request_denominator")
    rows = []
    for request in requests:
        reasons = list(request.get("reasons", []))
        for name in required:
            item = request.get("checks", {}).get(name)
            if type(item) is not dict:
                reasons.append(name + ":not_recorded")
                continue
            if item.get("check", name) != name:
                reasons.append(name + ":different_check_recorded")
                continue
            # Recompute from facts rather than trusting a supplied check_status.
            assessed = lease.assess_check(name, native_result=item.get("native_result", "unverified"),
                expected_units=item.get("expected_units"), compared_units=item.get("compared_units"),
                completed=item.get("completed") is True, evidence_complete=item.get("evidence_complete") is True,
                applicable=item.get("applicable"), reason=item.get("reason"))
            if assessed["check_status"] != "passed":
                reasons.append(name + ":" + assessed["reason"])
        all_reasons = list(dict.fromkeys(reasons + holds))
        rows.append(dict(request, decision="HOLD" if all_reasons else "PASS", reasons=all_reasons))
    failed = [row for row in rows if row["decision"] == "HOLD"]
    return dict(schema="sdcguard.required-check-gate.v1", owner="Hari Krishna Kancharla",
        decision="HOLD" if holds or failed else "PASS", required_checks=list(required),
        scope="selected recorded client requests only; no deployment or isolation guarantee",
        engine=engine or {"name": "vllm", "version": "unknown"}, selected_arm=arm,
        total_planned_requests=total, selected_requests=len(rows),
        excluded_requests=total - len(rows) if type(total) is int else None,
        passed_requests=len(rows) - len(failed), hold_reasons=list(dict.fromkeys(holds)), requests=rows,
        runtime_qualified=False, combined_checks_supported=False)


def _cleanup_valid(value):
    if (value.get("complete") is not True or value.get("errors") != []
            or value.get("remaining_pids") != [] or value.get("missing_terminal_evidence") != []
            or value.get("roots_without_exact_exit_status") != []):
        return False
    members, terminal = value.get("process_membership"), value.get("terminal_handles")
    if type(members) is not list or not members or type(terminal) is not list:
        return False
    def identity(row):
        if (type(row) is not dict or any(type(row.get(k)) is not int or row[k] <= 0 for k in ("pid", "start"))):
            raise ValueError("invalid cleanup process identity")
        return row["pid"], row["start"]
    expected = [identity(row) for row in members]
    observed = [identity(row) for row in terminal if type(row) is dict and (
        (row.get("basis") == "same_pidfd_exit_readiness" and type(row.get("poll")) is dict
         and row["poll"].get("status") == "exit_observed") or
        (row.get("basis") == "exact_pidfd_wait" and type(row.get("returncode")) is int))]
    groups, supervisors = value.get("groups"), value.get("supervisor_receipts")
    if (type(groups) is not list or len(groups) != 2 or len(set(groups)) != 2
            or any(type(pid) is not int or pid <= 0 for pid in groups)
            or type(supervisors) is not list or len(supervisors) != 2):
        return False
    roots = []
    terminal_by_id = {identity(row): row for row in terminal}
    reaped = value.get("reaped_leaders")
    if type(reaped) is not list or len(reaped) != len(groups):
        return False
    reaped_by_id = {identity(row): row for row in reaped}
    for row in supervisors:
        if (row.get("complete") is not True or type(row.get("anchor_returncode")) is not int
                or row["anchor_returncode"] != 0 or type(row.get("engine_returncode")) is not int
                or row["engine_returncode"] != 0):
            return False
        root = identity(row)
        ended, waited = terminal_by_id.get(root, {}), reaped_by_id.get(root, {})
        if (ended.get("basis") != "exact_pidfd_wait"
                or type(ended.get("returncode")) is not int
                or ended["returncode"] != row["anchor_returncode"]
                or type(waited.get("returncode")) is not int
                or waited["returncode"] != row["anchor_returncode"]):
            return False
        roots.append(root)
    return (len(set(expected)) == len(expected) and len(set(observed)) == len(observed)
            and set(expected) == set(observed)
            and same(sorted(value.get("observed_pids", [])), sorted(pid for pid, _ in expected))
            and sorted(pid for pid, _ in roots) == sorted(groups) and set(roots) <= set(expected))


def _startup(value, plan, arm, role):
    context = dict(run_id=plan["run_id"], packet_sha256=plan["packet_sha256"],
                   arm_id=arm["arm_id"], engine_session_id=arm["engine_session_id"])
    revision = plan["source_revision"]
    expected = dict(context, role=role, source_revision=revision,
        source_profile=checksum.source_profile(revision), checksum_enabled=True,
        request_id_randomization_enabled=True, flashinfer_sampler_enabled=False,
        model_runner_v2=True, multiprocess_method="spawn", prefix_caching=plan.get("prefix_caching", False), chunked_prefill=True,
        async_scheduling=False, speculative_disabled=True, kv_dtype="auto", connector="NixlConnector",
        transfer_role="kv_producer" if role == "prefill" else "kv_consumer",
        block_size=16, max_model_len=2048, max_num_seqs=16,
        parallel=dict(tp=1, pp=1, dp=1, dcp=1, pcp=1),
        group_specs=[dict(type="FullAttentionSpec", block_size=16, sliding_window=None,
                          attention_chunk_size=None, non_causal=False)])
    if "gpu_blocks_override" in plan:
        expected["gpu_blocks"] = plan["gpu_blocks_override"]
    return (profiles.startup_supported(value, revision)
            and all(same(value.get(k), v) for k, v in expected.items())
            and type(value.get("process_id")) is int and value["process_id"] > 0
            and value.get("model") == plan["model"] == "Qwen/Qwen3-0.6B"
            and value.get("model_revision") == "c1899de289a04d12100db370d81485cdf75e47ca"
            and value.get("dtype") in ("torch.bfloat16", "bfloat16"))


def _raw_checksum_evidence(files, plan, captures):
    """Bind assembled inputs to original startup and close receipts."""
    issues = []
    expected_native_paths = set()
    for arm in plan["arms"]:
        arm_id = arm["arm_id"]
        startup = {}
        for role in ("prefill", "decode"):
            prefix = arm_id + "/" + role + "/"
            paths = [name for name in files if name.startswith(prefix + "startup-") and name.endswith(".json")]
            if len(paths) != 1:
                issues.append("startup_receipt_missing_or_ambiguous:" + prefix)
                continue
            value = object_at(files, paths[0])
            if value.get("checksum_enabled") is not True:
                issues.append("checksum_check_not_enabled:" + arm_id + ":" + role)
            if (not _startup(value, plan, arm, role)
                    or paths[0] != prefix + "startup-" + str(value.get("process_id")) + ".json"):
                issues.append("unsupported_recorded_engine_config:" + prefix)
            startup[role] = value
        cleanup = object_at(files, arm_id + "/CLEANUP.json")
        if not _cleanup_valid(cleanup):
            issues.append("cleanup_evidence_incomplete:" + arm_id)
        member_ids = {row["pid"] for row in cleanup.get("process_membership", [])
                      if type(row) is dict and type(row.get("pid")) is int}
        for role, value in startup.items():
            if value.get("process_id") not in member_ids:
                issues.append("startup_process_membership_unverified:" + arm_id + ":" + role)
        if not arm["observer_enabled"]:
            continue
        selected = [v for v in captures["captures"] if v.get("arm_id") == arm_id]
        if len(selected) != 1:
            issues.append("capture_missing_or_ambiguous:" + arm_id)
            continue
        capture = selected[0]
        role_captures = [("decode", capture)]
        if plan.get("prefix_caching") is True:
            role_captures.append(("prefill", capture.get("prefill_capture", {})))
        for role, envelope in role_captures:
            if type(envelope) is not dict:
                envelope = {}
            assembled = envelope.get("startup", {})
            assembled = assembled if type(assembled) is dict else {}
            raw = {k: v for k, v in assembled.items() if k != "native_transport"}
            if not same(raw, startup.get(role)):
                issues.append("startup_assembly_disagrees:" + arm_id + ":" + role)
            if envelope.get("capture_complete") is not True or envelope.get("capture_errors") != []:
                issues.append("native_capture_incomplete:" + arm_id + ":" + role)
            prefix = arm_id + "/" + role + "/"
            kinds = [("bindings", "binding_snapshots")]
            if role == "decode":
                kinds.insert(0, ("checksum", "checksum_snapshots"))
            if plan.get("prefix_caching") is True:
                kinds.append(("cache", "cache_snapshots"))
            for kind, key in kinds:
                paths = sorted(name for name in files if name.startswith(prefix + kind + "-")
                               and name.endswith(".json") and "-disabled-" not in name)
                originals = []
                for name in paths:
                    expected_native_paths.add(name)
                    value = object_at(files, name)
                    context = value.get("context", {})
                    pid = context.get("process_id") if type(context) is dict else None
                    expected_context = {k: plan[k] if k in ("run_id", "packet_sha256") else arm[k] for k in CONTEXT}
                    expected_context["process_id"] = pid
                    if (type(pid) is not int or pid <= 0 or name != prefix + kind + "-" + str(pid) + ".json"
                            or not same(context, expected_context) or value.get("role") != role
                            or value.get("schema") != "sdcguard.owner-checksum-close.v1"
                            or value.get("original_shutdown_completed") is not True
                            or value.get("original_shutdown_error") is not None or value.get("recorder_errors") != []):
                        issues.append("native_close_receipt_unverified:" + name)
                    snapshot = value.get("snapshot")
                    if type(snapshot) is not dict or snapshot.get("qualification") != "source_bound_candidate":
                        issues.append("non_runtime_native_evidence:" + name)
                    snapshot_context = (snapshot.get("capture_context") if kind == "checksum" else
                        {k: snapshot.get(k) for k in (*CONTEXT, "process_id")}) if type(snapshot) is dict else None
                    if not same(snapshot_context, expected_context):
                        issues.append("native_snapshot_context_disagrees:" + name)
                    if kind in ("checksum", "cache") and pid != startup.get(role, {}).get("process_id"):
                        issues.append("native_scheduler_process_disagrees:" + name)
                    if kind == "cache" and (type(snapshot) is not dict or snapshot.get("role") != role):
                        issues.append("native_cache_role_disagrees:" + name)
                    if pid not in member_ids:
                        issues.append("native_recorder_process_membership_unverified:" + name)
                    originals.append(snapshot)
                if not originals or not same(originals, envelope.get(key)):
                    issues.append("native_snapshot_assembly_disagrees:" + arm_id + ":" + role + ":" + kind)
    for name in files:
        basename = name.rsplit("/", 1)[-1]
        if basename.startswith("errors-") and name.endswith(".json"):
            issues.append("native_recorder_error:" + name)
        if (basename.startswith(("checksum-", "bindings-", "cache-")) and name.endswith(".json")
                and "-disabled-" not in basename and name not in expected_native_paths):
            issues.append("unexpected_native_receipt:" + name)
    return issues


def _journal(files):
    inventory = object_at(files, "client/CLIENT_INVENTORY.json")
    expected = {(row["arm_id"], row["slot"], role): row[role]
                for row in inventory["requests"] for role in ("prefill", "decode")}
    started, finished = {}, {}
    lines = files.get("client/CLIENT_RECEIPTS.jsonl", b"").splitlines()
    if len(lines) != 2 * len(expected):
        return ["client_journal_stage_count_disagrees"]
    for sequence, raw in enumerate(lines, 1):
        value = io.parse_json(raw)
        if (type(value) is not dict or type(value.get("sequence")) is not int or value["sequence"] != sequence
                or type(value.get("slot")) is not int or type(value.get("arm_id")) is not str
                or value.get("role") not in ("prefill", "decode")):
            return ["client_journal_sequence_unverified"]
        key = value["arm_id"], value["slot"], value["role"]
        receipt = value.get("receipt")
        if key not in expected or type(receipt) is not dict:
            return ["client_journal_unplanned_stage"]
        if value.get("kind") == "stage_started":
            if (key in started or receipt.get("completed") is not False or receipt.get("http_status") is not None
                    or receipt.get("error") is not None or receipt.get("response_body") is not None
                    or any(not same(receipt.get(k), expected[key].get(k))
                           for k in ("method", "path", "request_body", "request_headers"))):
                return ["client_journal_started_stage_disagrees"]
            started[key] = sequence
        elif value.get("kind") == "stage_finished":
            if key not in started or key in finished or not same(receipt, expected[key]):
                return ["client_journal_finished_stage_disagrees"]
            finished[key] = sequence
        else:
            return ["client_journal_kind_unverified"]
    if set(started) != set(expected) or set(finished) != set(expected):
        return ["client_journal_incomplete"]
    if any(finished[arm, slot, "prefill"] >= started[arm, slot, "decode"]
           for arm, slot, role in expected if role == "prefill"):
        return ["client_journal_handoff_order_unverified"]
    return []


def checksum_gate(result_raw, archive_raw, preflight_raw, required, arm=None):
    result, files = io.parse_json(result_raw), read_archive(archive_raw)
    consumed = {name: raw for name, raw in files.items() if name in
        ("client/REQUEST_PLAN.json", "client/CLIENT_INVENTORY.json", "client/CLIENT_RECEIPTS.jsonl", "CAPTURES.json", "FINISHED.json", "BUILD_WITNESS.json")
        or name.endswith("/CLEANUP.json") or name.rsplit("/", 1)[-1].startswith(("startup-", "checksum-", "bindings-", "cache-"))}
    if preflight_raw is not None:
        consumed["CPU_PREFLIGHT.json"] = preflight_raw
    consumed["RESULT.json"] = result_raw
    budget.count_members(consumed, max_nodes=budget.MAX_TOTAL_JSON_NODES)
    plan = object_at(files, "client/REQUEST_PLAN.json")
    captures = object_at(files, "CAPTURES.json")
    report = checksum.evaluate(files["client/REQUEST_PLAN.json"], files["client/CLIENT_INVENTORY.json"], files["CAPTURES.json"])
    coverage_hold = "required_checksum_coverage_incomplete"
    # Missing coverage holds the release, but does not invalidate a different
    # request's complete comparison. Identity and recording issues still do.
    issues = [x for x in report["hold_reasons"] if x not in PROFILE_LIMITS and x != coverage_hold]
    finished = object_at(files, "FINISHED.json")
    if (not same(result.get("report"), finished) or result.get("archive_sha256") != io.digest(archive_raw)
            or any(not same(result.get(k), plan[k]) for k in ("run_id", "packet_sha256"))
            or any(not same(finished.get(k), plan[k]) for k in ("run_id", "packet_sha256"))):
        issues.append("owner_evidence_identity_disagrees")
    if (finished.get("complete") is not True or finished.get("failure") is not None
            or finished.get("stage") != "complete" or finished.get("cleanup_complete") is not True
            or finished.get("client_inventory_complete") is not True):
        issues.append("owner_run_completion_unverified")
    version, recorded_revision = "unknown", "unknown"
    if preflight_raw is None:
        issues.append("recorded_engine_build_missing")
    else:
        preflight = io.parse_json(preflight_raw)
        installed = preflight.get("installed_engine", {})
        version = installed.get("versions", {}).get("vllm", "unknown")
        recorded_revision = installed.get("source_revision", "unknown")
        witness = object_at(files, "BUILD_WITNESS.json")
        payload = witness.get("witness", {})
        revision = plan["source_revision"]
        if (version != profiles.get(revision)["version"] or installed.get("source_revision") != revision
                or preflight.get("source_revision") != revision):
            issues.append("unsupported_recorded_engine_version")
        if preflight.get("schema") == "sdcguard.installed-checksum-preflight.v1":
            from .native_operator.installed import validate, WITNESS
            if (not validate(preflight, plan, files["client/REQUEST_PLAN.json"])
                    or payload.get("schema") != WITNESS
                    or not same(payload, preflight.get("build_witness"))
                    or witness.get("sha256") != preflight.get("build_witness_sha256")
                    or result.get("build_witness_sha256") != witness.get("sha256")):
                issues.append("installed_runtime_preflight_unverified")
            snapshot_path = preflight.get("model_cache", {}).get("snapshot")
            for planned_arm in plan["arms"]:
                for role in ("prefill", "decode"):
                    prefix = planned_arm["arm_id"] + "/" + role + "/startup-"
                    starts = [object_at(files, n) for n in files if n.startswith(prefix) and n.endswith(".json")]
                    if (len(starts) != 1 or type(snapshot_path) is not str
                            or any(starts[0].get(k) != snapshot_path for k in ("actual_model", "actual_tokenizer", "snapshot"))):
                        issues.append("installed_model_startup_binding_unverified:" + planned_arm["arm_id"] + ":" + role)
        elif revision != profiles.BASELINE:
            issues.append("candidate_requires_installed_runtime_preflight")
        elif (witness.get("sha256") != io.digest(io.json_bytes(payload))
                or result.get("build_witness_sha256") != witness.get("sha256")
                or payload.get("schema") != "sdcguard.owner-checksum-build-witness.v1"
                or payload.get("installed_identity_sha256") != io.digest(io.json_bytes(installed))
                or payload.get("packet_sha256") != plan["packet_sha256"]
                or preflight.get("packet_sha256") != plan["packet_sha256"]
                or preflight.get("run_id") != plan["run_id"] or preflight.get("status") != "PASS"
                or preflight.get("schema") != "sdcguard.owner-checksum-cpu-preflight.v1"
                or installed.get("schema") != "sdcguard.owner-checksum-installed.v1"
                or preflight.get("failure") is not None
                or preflight.get("plan_sha256") != io.digest(files["client/REQUEST_PLAN.json"])
                or not same(preflight.get("build_witness"), payload)
                or preflight.get("build_witness_sha256") != witness.get("sha256")
                or any(preflight.get(k) != payload.get(k) for k in ("build_result_sha256", "build_archive_sha256"))
                or payload.get("pinned_source_guard_sha256") != io.digest(io.json_bytes(preflight.get("pinned_source_guard")))
                or payload.get("python_identity_sha256") != io.digest(io.json_bytes(preflight.get("python_runtime", {}).get("identity")))):
            issues.append("recorded_engine_build_unverified")
    issues.extend(_raw_checksum_evidence(files, plan, captures))
    issues.extend(_journal(files))
    domains = {value["arm_id"]: "source_bound_candidate" if value["observer_enabled"] else "observer_disabled"
               for value in plan["arms"]}
    if any(capture["qualification"] != domains.get(capture["arm_id"]) for capture in report["captures"]):
        issues.append("runtime_capture_source_unverified")
    rows = []
    for row in report["requests"]:
        if arm is not None and row["arm_id"] != arm:
            continue
        calls = row["checksum_calls"]
        proof = calls[0]["conditional_scope"] if len(calls) == 1 else {}
        check = dict(native_result=row["native_result"], expected_units=proof.get("expected_units"),
            compared_units=proof.get("compared_units"), completed=len(calls) == 1 and row["check_status"] == "passed",
            evidence_complete=row["coverage"] == "complete" and not row["reasons"], applicable=row["observer_enabled"],
            reason=row["reasons"][0] if row["reasons"] else "complete_recorded_comparison")
        rows.append(dict(client_request_id=row["client_request_id"], arm_id=row["arm_id"], slot=row["slot"],
            checks={"checksum": check}, token_coverage=row["token_coverage"], reasons=list(row["reasons"])))
    if arm is not None and arm not in {value["arm_id"] for value in plan["arms"]}:
        issues.append("selected_arm_not_in_plan")
    answer = decide(rows, required, issues, total=len(report["requests"]),
                    engine=dict(name="vllm", version=version, source_revision=recorded_revision,
                                supported_source_revision=plan["source_revision"]), arm=arm)
    if coverage_hold in report["hold_reasons"]:
        answer["decision"] = "HOLD"
        answer["hold_reasons"].insert(0, coverage_hold)
    return answer


def lease_gate(result_raw, archive_raw, required, arm=None):
    report = lease.evaluate_owner(result_raw, archive_raw, lease.policy(["lease"]))
    issues = list(report["hold_reasons"])
    for role in ("prefill", "decode"):
        facts = report["context"].get("recorded_startup", {}).get(role, {})
        if facts.get("transfer", {}).get("lease_verification") is not True:
            issues.append("lease_check_not_enabled:" + role)
    if arm is not None:
        issues.append("arm_selection_not_supported_for_lease_capture")
    rows = []
    for row in report["requests"]:
        reasons = [] if row["request_outcome"] == "satisfied" else ["request_outcome_not_satisfied"]
        rows.append(dict(client_request_id=row["client_request_id"], slot=row["slot"],
                         checks=row["checks"], reasons=reasons))
    return decide(rows, required, issues, total=len(rows), engine=report["context"], arm=arm)


def inspect(path, required, arm=None):
    path = Path(path)
    if path.is_file():
        io.parse_json(io.read_file(path.parent, path.name))
        return decide([], required, ["legacy_summary_has_no_complete_per_request_check_evidence"], arm=arm)
    result = io.read_file(path, "RESULT.json")
    archive = io.read_file(path, "ordinary-evidence.zip")
    files = read_archive(archive)
    if "client/REQUEST_PLAN.json" in files:
        preflight = io.read_file(path, "CPU_PREFLIGHT.json") if (path / "CPU_PREFLIGHT.json").exists() else None
        answer = checksum_gate(result, archive, preflight, required, arm)
        if preflight is not None and preflight != io.read_file(path, "CPU_PREFLIGHT.json"):
            raise io.EvidenceError("engine build evidence changed during inspection")
    elif lease.CAPTURE + "CAPTURE.json" in files:
        answer = lease_gate(result, archive, required, arm)
    else:
        answer = decide([], required, ["unsupported_recorded_engine_profile"], arm=arm)
    if result != io.read_file(path, "RESULT.json") or archive != io.read_file(path, "ordinary-evidence.zip"):
        raise io.EvidenceError("owner inputs changed during inspection")
    return answer


def render(answer):
    lines = [answer["decision"] + " | required: " + ", ".join(answer["required_checks"]), answer["scope"],
        str(answer["passed_requests"]) + " / " + str(answer["selected_requests"]) + " selected requests passed",
        "Total planned: " + str(answer["total_planned_requests"]) + "; explicitly excluded: " + str(answer["excluded_requests"])]
    lines.extend("HOLD: " + reason for reason in answer["hold_reasons"])
    for row in answer["requests"]:
        lines.append(str(row["client_request_id"]) + " | " + row["decision"] + " | " +
                     ("; ".join(row["reasons"]) or "required check passed with complete recorded coverage"))
        tokens = row.get("token_coverage")
        if type(tokens) is dict:
            def count(name):
                value = tokens.get(name)
                return str(value) if type(value) is int else "unknown"
            lines.append("  Prompt tokens: " + count("prompt_tokens") +
                "; transferred tokens checked: " + count("transferred_tokens_checked") +
                "; decode cache reused unchecked: " + count("decode_cached_tokens_unchecked") +
                "; prefill cache reused: " + count("prefill_cached_tokens"))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard gate", description=__doc__)
    parser.add_argument("input", help="saved owner-result directory or historical RESULT.json")
    parser.add_argument("--require", choices=CHECKS, action="append", required=True)
    parser.add_argument("--arm", help="explicit checksum workload arm; default includes all planned requests")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        answer = inspect(args.input, args.require, args.arm)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        answer = decide([], list(dict.fromkeys(args.require)), ["evidence_unverifiable:" + str(error)], arm=args.arm)
    print(io.json_bytes(answer).decode() if args.json else render(answer), end="")
    return 0 if answer["decision"] == "PASS" else 2
