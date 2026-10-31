"""Saved ordinary recordings and offline evidence controls only.

Owner: Hari Krishna Kancharla.
Edited copies are synthetic missing/contradictory evidence, not engine runs.
"""
from collections import Counter
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from sdcguard import cli, evidence_io as io, required_gate as gate, transfer_coverage as transfer

SAVED = Path(__file__).parent / "data/checksum_owner_3a2ad847"


@pytest.fixture(scope="module")
def saved():
    source = json.loads((SAVED / "SOURCE.json").read_bytes())
    for name, expected in source["files"].items():
        raw = (SAVED / name).read_bytes()
        assert dict(bytes=len(raw), sha256=io.digest(raw)) == expected
    return tuple((SAVED / name).read_bytes() for name in
                 ("RESULT.json", "ordinary-evidence.zip", "CPU_PREFLIGHT.json"))


@pytest.fixture(scope="module")
def answer(saved):
    return transfer.evaluate(*saved)


@pytest.fixture(scope="module")
def raw_files(saved):
    return gate.read_archive(saved[1])


@pytest.fixture
def pair(raw_files):
    records = json.loads(raw_files["CAPTURES.json"])["captures"][0]["checksum_snapshots"][0]["records"]
    plans = [r for r in records if r["kind"] == "recv_plan"]
    plan = plans[1]
    record = next(r for r in records if r["kind"] == "verification" and r["request_id"] == plan["request_id"])
    return deepcopy(plan), deepcopy(record)


def scope(pair, **overrides):
    plan, record = pair
    inputs = dict(prompt_tokens=166, local_tokens=144, remote_tokens=22)
    inputs.update(overrides)
    return transfer.transfer_scope([plan], [record], **inputs)


def repack(saved, mutate, *, sync=True):
    """Edit only an in-memory copy and retain every client slot by default."""
    files = gate.read_archive(saved[1])
    captures = json.loads(files["CAPTURES.json"])
    mutate(captures, files)
    if sync:
        for capture in captures["captures"]:
            for snapshot in capture["checksum_snapshots"]:
                records = snapshot["records"]
                budget = snapshot["budgets"]
                budget["retained_records"] = len(records)
                budget["retained_bytes"] = sum(len(json.dumps(r, sort_keys=True, allow_nan=False).encode()) for r in records)
                budget["reserved_record_bytes"] = max(budget["reserved_record_bytes"], budget["retained_bytes"])
                budget["retained_units"] = sum(len(r.get(k) or []) for r in records for k in
                    ("expected_units", "compared_units", "missing_units"))
                name = capture["arm_id"] + "/decode/checksum-" + str(snapshot["capture_context"]["process_id"]) + ".json"
                receipt = json.loads(files[name])
                receipt["snapshot"] = snapshot
                files[name] = io.json_bytes(receipt)
    files["CAPTURES.json"] = io.json_bytes(captures)
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    result = json.loads(saved[0])
    result["archive_sha256"] = io.digest(output.getvalue())
    return io.json_bytes(result), output.getvalue(), saved[2]


def test_saved_transfer_obligation_and_full_prompt_policy_are_distinct(saved, answer):
    assert answer["transfer_decision"] == "PASS"
    assert answer["release_decision"] == "HOLD"
    assert answer["full_prompt_gate"] == gate.checksum_gate(*saved, ["checksum"])
    assert answer["full_prompt_gate"]["passed_requests"] == 9
    assert Counter(r["decision"] for r in answer["full_prompt_gate"]["requests"]) == {"PASS": 9, "HOLD": 144}
    assert answer["summary"] == dict(planned_requests=153, verified_requests=153,
        not_exercised_requests=0, held_requests=0, statuses={"VERIFIED": 153},
        expected_chunk_positions=688, prompt_tokens=109790, expected_transfer_tokens=9822,
        compared_transfer_tokens=9822, decode_cached_tokens_outside_transfer=99968)
    assert answer["hold_reasons"] == [] and answer["runtime_qualified"] is False
    assert len({r["client_request_id"] for r in answer["requests"]}) == 153
    assert all(r["comparison"]["expected_units"] == r["comparison"]["compared_units"] for r in answer["requests"])


def test_partial_final_blocks_count_tokens_without_padding(pair, answer):
    value = scope(pair)
    assert value["status"] == "VERIFIED"
    assert value["expected_units"] == ["group:0:chunk:9", "group:0:chunk:10"]
    assert value["compared_tokens"] == 22  # Not two padded blocks, or 32 tokens.
    for row in answer["requests"]:
        tokens = row["tokens"]
        assert tokens["prompt"] % 16 != 0
        assert tokens["expected_transfer"] == tokens["prompt"] - tokens["decode_cache_outside_transfer"]
        assert tokens["compared_transfer"] < len(row["comparison"]["compared_units"]) * 16


@pytest.mark.parametrize("change", ["short_intersection", "empty_replica", "final_token_disagreement",
    "no_compare", "skipped", "unsupported_layout", "different_fingerprint", "missing_rank",
    "producer_short", "native_exception", "incomplete_record", "wrong_plan_start",
    "wrong_plan_count", "wrong_request", "wrong_order", "boolean_count"])
def test_native_success_is_not_sufficient(pair, change):
    plan, record = pair
    proof = record["proof"]
    group = proof["groups"][0]
    consumer = group["consumer_reports"][0]
    if change == "short_intersection":
        consumer.update(group_start=10, value_shape=[1, 1])
    elif change == "empty_replica":
        consumer.update(value_shape=[2, 0], replica_keys=[])
    elif change == "final_token_disagreement":
        proof["expected_num_tokens"] -= 1
    elif change == "no_compare":
        group["compare_called"] = False
    elif change == "skipped":
        consumer["skipped"] = True
    elif change == "unsupported_layout":
        group["spec_type"] = "SlidingWindowSpec"
    elif change == "different_fingerprint":
        consumer["fingerprint"] += 1
    elif change == "missing_rank":
        proof["worker_ranks"] = []
    elif change == "producer_short":
        group.update(sent_start=1, sent_count=10, sent_shape=[10])
    elif change == "native_exception":
        record["engine_exception"] = "recorded error"
    elif change == "incomplete_record":
        record["recording_complete"] = False
    elif change == "wrong_plan_start":
        plan["load_plan"]["groups"][0]["start"] += 1
    elif change == "wrong_plan_count":
        plan["load_plan"]["groups"][0]["block_count"] -= 1
    elif change == "wrong_request":
        plan["request_id"] = "synthetic-different-request"
    elif change == "wrong_order":
        record["origin_update_sequence"] = plan["sequence"]
    else:
        plan["load_plan"]["groups"][0]["block_count"] = True
    assert record["native_return"] is True
    result = scope(pair)
    assert result["status"] == "UNOBSERVABLE"
    assert result["reasons"] and result["compared_tokens"] is None


def test_recorded_mismatch_is_distinct_from_unobservable(pair):
    plan, record = pair
    record["native_return"] = False
    record["proof"]["groups"][0]["native_group_result"] = [9]
    record["groups"][0].update(native_result="mismatch", returned_positions=[9])
    value = scope(pair)
    assert value["status"] == "MISMATCH"
    assert "native_checksum_mismatch" in value["reasons"]


@pytest.mark.parametrize("counts", [dict(local_tokens=None), dict(remote_tokens=None),
    dict(prompt_tokens=True), dict(local_tokens=True), dict(remote_tokens=-1), dict(remote_tokens=21),
    dict(local_tokens=10**30), dict(prompt_tokens=10**30)])
def test_transfer_needs_independent_bounded_native_accounting(pair, counts):
    assert scope(pair, **counts)["status"] == "UNOBSERVABLE"


@pytest.mark.parametrize("which", ["plan", "verification"])
def test_missing_or_duplicate_record_cannot_pass(pair, which):
    plan, record = pair
    for n in (0, 2):
        plans, records = ([plan] * n, [record]) if which == "plan" else ([plan], [record] * n)
        value = transfer.transfer_scope(plans, records, prompt_tokens=166, local_tokens=144, remote_tokens=22)
        assert value["status"] == "UNOBSERVABLE"


def test_zero_transfer_is_not_exercised_only_with_complete_accounting(pair):
    assert transfer.transfer_scope([], [], prompt_tokens=160, local_tokens=160, remote_tokens=0)["status"] == "NOT_EXERCISED"
    assert transfer.transfer_scope([], [], prompt_tokens=160, local_tokens=None, remote_tokens=0)["status"] == "UNOBSERVABLE"
    assert transfer.transfer_scope([], [], prompt_tokens=160, local_tokens=144, remote_tokens=16)["status"] == "UNOBSERVABLE"
    assert transfer.transfer_scope([pair[0]], [pair[1]], prompt_tokens=160, local_tokens=160, remote_tokens=0)["status"] == "UNOBSERVABLE"


def test_missing_verification_in_an_offline_copy_keeps_the_request_in_denominator(saved):
    before = {p.name: io.digest(p.read_bytes()) for p in SAVED.iterdir() if p.is_file()}
    def remove(captures, files):
        snapshot = captures["captures"][0]["checksum_snapshots"][0]
        target = [r for r in snapshot["records"] if r["kind"] == "verification"][1]
        snapshot["records"].remove(target)
    value = transfer.evaluate(*repack(saved, remove))
    assert value["transfer_decision"] == "HOLD"
    assert value["summary"]["planned_requests"] == 153
    assert value["summary"]["verified_requests"] == 152
    row = value["requests"][1]
    assert row["decision"] == "HOLD" and row["status"] == "UNOBSERVABLE"
    assert "transfer_plan_or_verification_missing_or_ambiguous" in row["reasons"]
    assert row["tokens"]["expected_transfer"] == 22 and row["tokens"]["compared_transfer"] is None
    assert value["summary"]["expected_chunk_positions"] is None
    assert before == {p.name: io.digest(p.read_bytes()) for p in SAVED.iterdir() if p.is_file()}


@pytest.mark.parametrize("change", ["raw_disagrees", "missing_client", "journal_gap", "dropped_record",
                                    "unclosed", "unknown_build", "unexpected_record", "no_receipt"])
def test_global_integrity_gaps_hold_and_preserve_denominator(saved, change):
    def mutate(captures, files):
        snapshot = captures["captures"][0]["checksum_snapshots"][0]
        if change == "raw_disagrees":
            snapshot["records"][0]["request_id"] = "synthetic-unjoined"
        elif change == "missing_client":
            value = json.loads(files["client/CLIENT_INVENTORY.json"])
            value["requests"].pop()
            files["client/CLIENT_INVENTORY.json"] = io.json_bytes(value)
        elif change == "journal_gap":
            files["client/CLIENT_RECEIPTS.jsonl"] = b"\n".join(files["client/CLIENT_RECEIPTS.jsonl"].splitlines()[:-1]) + b"\n"
        elif change == "dropped_record":
            snapshot["health"]["dropped_records"] = 1
        elif change == "unclosed":
            snapshot["closed"] = False
        elif change == "unexpected_record":
            snapshot["records"].append(dict(kind="not_verified", sequence=10**8, request_id="synthetic-unjoined", reason="missing"))
        elif change == "no_receipt":
            name = next(n for n in files if n.startswith("observer-on/decode/checksum-") and "-disabled-" not in n)
            del files[name]
    values = repack(saved, mutate, sync=change not in ("raw_disagrees", "no_receipt"))
    if change == "unknown_build":
        preflight = json.loads(values[2])
        preflight["installed_engine"]["versions"]["vllm"] = "unknown"
        values = (*values[:2], io.json_bytes(preflight))
    result = transfer.evaluate(*values)
    assert result["transfer_decision"] == "HOLD"
    assert result["release_decision"] == "HOLD"
    assert result["summary"]["planned_requests"] == 153
    assert result["summary"]["held_requests"] > 0


def test_cache_counts_are_independent_of_each_other(raw_files):
    plan = json.loads(raw_files["client/REQUEST_PLAN.json"])
    capture = json.loads(raw_files["CAPTURES.json"])["captures"][0]
    inventory = json.loads(raw_files["client/CLIENT_INVENTORY.json"])
    joined = transfer.client.evaluate(raw_files["client/REQUEST_PLAN.json"], raw_files["client/CLIENT_INVENTORY.json"], raw_files["CAPTURES.json"])
    row, raw = joined["requests"][1], deepcopy(inventory["requests"][1])
    facts = transfer.cache.capture_facts(capture, plan, plan["arms"][0])
    internal = raw["prefill"]["response_body"]["kv_transfer_params"]["remote_request_id"]
    initial = next(r for r in facts["prefill"]["records"] if r["request_id"] == internal and r["stats"] is not None)
    initial["stats"].update(num_local_cached_tokens=0, num_cached_tokens=0, num_computed_tokens=166)
    raw["prefill"]["response_body"]["kv_transfer_params"]["remote_prefill_cached_tokens"] = 0
    values, errors = transfer._cache_usage(facts, raw, row)
    assert errors == []
    assert values["prefill"]["num_local_cached_tokens"] == 0
    assert values["decode"]["num_local_cached_tokens"] == 144


@pytest.mark.parametrize("all_zero", [False, True])
def test_proven_zero_transfer_rows_remain_visible_and_never_create_a_vacuous_pass(saved, all_zero):
    def mutate(captures, files):
        capture = captures["captures"][0]
        snapshot = capture["checksum_snapshots"][0]
        request_ids = [r["request_id"] for r in snapshot["records"] if r["kind"] == "recv_plan"]
        zero = set(request_ids if all_zero else request_ids[1:2])
        snapshot["records"] = [r for r in snapshot["records"] if r["request_id"] not in zero]
        cache_snapshot = capture["cache_snapshots"][0]
        for row in cache_snapshot["records"]:
            if row["request_id"] in zero and row["stats"] is not None:
                counts = row["stats"]
                counts.update(num_external_cached_tokens=0, num_local_cached_tokens=row["prompt_tokens"],
                              num_cached_tokens=row["prompt_tokens"], num_computed_tokens=0)
        cache_snapshot["budgets"]["retained_bytes"] = sum(len(io.json_bytes(r)) for r in cache_snapshot["records"])
        name = "observer-on/decode/cache-" + str(cache_snapshot["process_id"]) + ".json"
        original = json.loads(files[name])
        original["snapshot"] = cache_snapshot
        files[name] = io.json_bytes(original)
    value = transfer.evaluate(*repack(saved, mutate))
    assert value["summary"]["planned_requests"] == 153
    assert value["summary"]["not_exercised_requests"] == (153 if all_zero else 1)
    assert value["requests"][1]["status"] == "NOT_EXERCISED"
    assert value["requests"][1]["comparison"]["compared_tokens"] == 0
    if all_zero:
        assert value["transfer_decision"] == "HOLD"
        assert value["summary"]["verified_requests"] == 0
        assert "no_verified_transfer_requests" in value["hold_reasons"]
    else:
        assert value["transfer_decision"] == "PASS"
        assert value["summary"]["verified_requests"] == 152
        assert value["requests"][1]["decision"] == "NOT_EXERCISED"


def test_cli_keeps_release_hold_visible(answer, capsys):
    assert cli.main(["transfer-check", str(SAVED), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == answer
    text = transfer.render(answer)
    assert text.startswith("Transfer comparison: PASS\nFull-prompt policy: HOLD\n")
    assert "153 / 153" in text and "99968" in text


def test_cli_missing_input_holds_without_runtime_import(tmp_path):
    code = """
import sys
from sdcguard.cli import main
status = main(['transfer-check', sys.argv[1], '--json'])
assert not any(n in sys.modules for n in ('torch', 'vllm', 'modal', 'nixl'))
raise SystemExit(status)
"""
    p = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert p.returncode == 2, p.stderr
    assert json.loads(p.stdout)["transfer_decision"] == "HOLD"
