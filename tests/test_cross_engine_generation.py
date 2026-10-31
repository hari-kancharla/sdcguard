"""Bounded source-fragment CPU conformance, with no server, accelerator or forcer.

The ordinary SGLang hold/two matching ACK/release sequence is its upstream CPU
test sequence. vLLM runs its original homogeneous single-READ and DONE poll
bodies plus original allocation/free bodies, with vendor-equivalent CPU mocks.
Both receipts feed the SAME generic destination-incarnation rule. Mocks replace
transport, descriptors, collective, cache coordinator and memory, so these are
component conformance witnesses, never deployed or physical-write validation.
Negative cases damage inert receipts only; no unsafe engine path is executed.
"""

from __future__ import annotations

from copy import deepcopy
import builtins
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from sdcguard.lifetime.generation_obligations import evaluate_generation_obligations
from sdcguard.lifetime.generation_observations import (
    SOURCE_PINS, sglang_component_observations, vllm_probe_observations,
)
from sdcguard.lifetime.component_conformance import _sglang_receipt, _vllm_receipt, _evaluate
from sdcguard.lifetime import component_conformance as component


FIXTURES = Path(__file__).parent / "fixtures" / "engine_generation"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text())


@pytest.fixture(scope="module")
def component_receipts():
    return {"sglang": _sglang_receipt(), "vllm": _vllm_receipt()}


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
def test_same_generic_rule_proves_ordinary_component_retirement(component_receipts, engine):
    events, metadata = component_receipts[engine]
    result = _evaluate(engine, events, metadata)
    assert result.status == "satisfied", result.as_dict()
    assert result.physical_write_safety == "unknown"
    assert len(result.obligations) == 1
    assert result.obligations[0]["release_observed"] is True
    assert result.obligations[0]["generation_basis"] == ("protocol_hold" if engine == "sglang" else "observed_allocation")


@pytest.mark.parametrize("engine", ["sglang", "vllm"])
@pytest.mark.parametrize("damage", ["source", "startup", "close", "clock", "gap", "duplicate", "dropped", "stream", "no_events", "nonfinite", "backwards"])
def test_missing_coverage_is_unknown_never_an_alarm(component_receipts, engine, damage):
    events, metadata = deepcopy(component_receipts[engine])
    if damage == "source":
        metadata["source"]["commit"] = "0" * 40
    elif damage == "startup":
        metadata["complete_from_start"] = False
    elif damage == "close":
        metadata["closed"] = False
    elif damage == "clock":
        events[2]["clock_id"] = None
    elif damage == "gap":
        del events[2]
    elif damage == "duplicate":
        events.insert(2, deepcopy(events[2]))
    elif damage == "dropped":
        events[-1]["dropped"] = 1
    elif damage == "stream":
        events[2]["session"] = "unmapped-other-process"
    elif damage == "nonfinite":
        events[2]["mono"] = float("nan")
    elif damage == "backwards":
        events[2]["mono"] = events[1]["mono"] - 1
    else:
        events = []
    result = _evaluate(engine, events, metadata)
    assert result.status == "unknown" and result.reasons


@pytest.mark.parametrize("field", ["generation", "room", "metadata_index", "pool", "outstanding", "admission_failed", "ack_accepted", "peer"])
def test_missing_ack_attribution_is_unknown(component_receipts, field):
    events, metadata = deepcopy(component_receipts["sglang"])
    next(e for e in events if e["kind"] == "drain_ack")["args"].pop(field)
    assert _evaluate("sglang", events, metadata).status == "unknown"


def test_timeout_receipt_is_unknown_without_executing_timeout(component_receipts):
    events, metadata = deepcopy(component_receipts["sglang"])
    next(e for e in events if e["kind"] == "release_outcome")["args"]["outcome"] = "timeout"
    result = _evaluate("sglang", events, metadata)
    assert result.status == "unknown" and any("timeout" in reason for reason in result.reasons)


def test_boolean_is_not_a_recorded_wire_generation(component_receipts):
    events, metadata = deepcopy(component_receipts["sglang"])
    next(e for e in events if e["kind"] == "drain_ack")["args"]["generation"] = True
    assert _evaluate("sglang", events, metadata).status == "unknown"


@pytest.mark.parametrize("kind,what", [("alloc", None), ("free", None), ("submit", None), ("complete", None),
                                     ("evidence", "attempt"), ("evidence", "read_submission_result"),
                                     ("evidence", "transfer_done"), ("evidence", "recording_health"),
                                     ("evidence", "destination_transfer_handles"),
                                     ("evidence", "destination_transport_poll")])
def test_missing_destination_lifecycle_receipt_is_unknown(component_receipts, kind, what):
    events, metadata = deepcopy(component_receipts["vllm"])
    # Remove the meaningful fields without renumbering or executing any engine.
    record = next(e for e in events if e["kind"] == kind and (what is None or e["args"].get("what") == what)
                  and (what != "destination_transport_poll" or e["args"].get("state") == "DONE"))
    record["args"] = {}
    assert _evaluate("vllm", events, metadata).status == "unknown"


@pytest.mark.parametrize("field", ["request", "handle", "state", "op_ids"])
def test_unattributed_transport_done_is_unknown(component_receipts, field):
    events, metadata = deepcopy(component_receipts["vllm"])
    record = next(e for e in events if e["args"].get("what") == "destination_transport_poll"
                  and e["args"].get("state") == "DONE")
    del record["args"][field]
    assert _evaluate("vllm", events, metadata).status == "unknown"


@pytest.mark.parametrize("field", ["manager_engine", "worker_engine", "pool", "single_cache_group"])
def test_unmapped_manager_worker_pool_is_unknown(component_receipts, field):
    events, metadata = deepcopy(component_receipts["vllm"])
    del metadata[field]
    assert _evaluate("vllm", events, metadata).status == "unknown"


def _additional_destination_receipts(events, missing):
    """Damage inert receipts to describe partial coverage of a second operation.

    No additional engine operation runs. Cloned observations solely isolate
    missing per-operation facts from stream/sequence validation. Every caller
    removes at least one required fact, and no synthetic positive is claimed.
    """
    assert missing in {"allocation", "submit", "completion", "free", "submission_result",
                       "registered_handles", "poll", "transfer_done"}
    result = []
    op = "unaccounted-destination-operation"
    covered = {"attempt", "read_submission_result", "transfer_done",
               "destination_transfer_handles", "destination_transport_poll"}
    omit_op = {"submission_result": "read_submission_result",
               "registered_handles": "destination_transfer_handles", "poll": "destination_transport_poll",
               "transfer_done": "transfer_done"}.get(missing)
    for original in deepcopy(events):
        args = original["args"]
        what = args.get("what")
        if what in covered and what != omit_op:
            args["op_ids"].append(op)
        if what == "attempt":
            args["destination_regions"].append(["D", 8])
            args["blocks"] += 1
        result.append(original)
        kind = original["kind"]
        clone = deepcopy(original)
        if kind == "alloc" and missing != "allocation":
            clone["args"]["region"] = ["D", 8]
        elif kind == "submit" and missing != "submit":
            clone["args"].update(op=op, dst=["D", 8])
        elif kind == "complete" and missing not in ("completion", "submit"):
            clone["args"].update(op=op, marks_ready=["D", 8])
        elif kind == "free" and missing != "allocation" and missing != "free":
            clone["args"]["region"] = ["D", 8]
        else:
            continue
        result.append(clone)
    for seq, event in enumerate(result, 1):
        event["seq"] = seq
    return result


def test_successful_mapping_cannot_ignore_an_unrecorded_destination_operation(component_receipts):
    events, metadata = deepcopy(component_receipts["vllm"])
    for event in events:
        args = event["args"]
        if args.get("what") in ("attempt", "read_submission_result", "destination_transfer_handles",
                                "destination_transport_poll", "transfer_done"):
            args["op_ids"].append("missing-destination-operation")
        if args.get("what") == "attempt":
            args["destination_regions"].append(["D", 8])
            args["blocks"] += 1
    result = _evaluate("vllm", events, metadata)
    assert result.status == "unknown"
    assert any("allocation" in reason for reason in result.reasons)


@pytest.mark.parametrize("missing", ["allocation", "submit", "completion", "free", "submission_result",
                                     "registered_handles", "poll", "transfer_done"])
def test_each_mapped_destination_needs_complete_individual_coverage(component_receipts, missing):
    events, metadata = component_receipts["vllm"]
    damaged = _additional_destination_receipts(events, missing)
    result = _evaluate("vllm", damaged, metadata)
    assert result.status == "unknown", result.as_dict()
    if missing == "submit":
        assert any("unaccounted destination operations" in reason for reason in result.reasons)


@pytest.mark.parametrize("damage", ["duplicate_operation", "duplicate_region", "duplicate_done",
                                    "duplicate_completion", "unaccounted_request", "orphan_submit"])
def test_duplicate_or_unaccounted_mappings_are_unknown(component_receipts, damage):
    events, metadata = deepcopy(component_receipts["vllm"])
    attempt = next(e for e in events if e["args"].get("what") == "attempt")
    if damage in ("duplicate_operation", "duplicate_region"):
        args = attempt["args"]
        args["op_ids"].append(args["op_ids"][0] if damage == "duplicate_operation" else "another-operation")
        args["destination_regions"].append(args["destination_regions"][0])
        args["blocks"] += 1
    else:
        source = next(e for e in events if (
            e["args"].get("what") == "transfer_done" if damage == "duplicate_done" else
            e["kind"] == "complete" if damage == "duplicate_completion" else
            e["args"].get("what") == "attempt" if damage == "unaccounted_request" else e["kind"] == "submit"))
        extra = deepcopy(source)
        if damage == "unaccounted_request":
            extra["args"]["request"] = "unaccounted-request"
        elif damage == "orphan_submit":
            extra["args"]["op"] = "unmapped-operation"
        events.insert(events.index(source) + 1, extra)
        for seq, event in enumerate(events, 1):
            event["seq"] = seq
    assert _evaluate("vllm", events, metadata).status == "unknown"


@pytest.mark.parametrize("receipt", [None, {}, [], {"schema": "wrong"}])
def test_malformed_receipts_fail_closed(receipt):
    assert evaluate_generation_obligations(receipt).status == "unknown"


def test_oversized_sequence_claim_is_unknown_without_allocating_a_range(component_receipts):
    events, metadata = component_receipts["sglang"]
    normalized = sglang_component_observations(events, metadata)
    normalized["coverage"]["streams"][0]["last_seq"] = 10**30
    assert evaluate_generation_obligations(normalized).status == "unknown"


def test_source_fragment_manifest_and_licenses_are_pinned():
    assert len(MANIFEST["fragments"]) == 25
    for engine, expected in SOURCE_PINS.items():
        assert MANIFEST["engines"][engine] == expected["commit"]
        actual = {f["source_path"]: f["source_sha256"] for f in MANIFEST["fragments"] if f["engine"] == engine}
        assert actual == expected["sha256"]
    for item in MANIFEST["fragments"] + MANIFEST["licenses"]:
        raw = (FIXTURES / item["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    assert {item["engine"] for item in MANIFEST["licenses"]} == {"sglang", "vllm"}


def test_generic_rule_contains_no_engine_specific_branch():
    from sdcguard.lifetime import generation_obligations
    source = Path(generation_obligations.__file__).read_text().lower()
    assert "sglang" not in source and "vllm" not in source


@pytest.mark.parametrize("engine", ["vllm", "sglang", "all"])
def test_component_main_prints_inspectable_scoped_json(engine, capsys):
    assert component.main(["--engine", engine, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["scope"] == "source_fragment_cpu" and result["physical_write_safety"] == "unknown"
    assert result["status"] == "satisfied"
    assert set(result["engines"]) == ({"vllm", "sglang"} if engine == "all" else {engine})
    for name, item in result["engines"].items():
        assert item["source"] == SOURCE_PINS[name]
        assert item["assessment"]["physical_write_safety"] == "unknown"
        assert _evaluate(name, item["receipt"]["observations"], item["receipt"]["metadata"]).status == "satisfied"


def test_component_main_defaults_to_both_and_labels_cpu_limits(capsys):
    assert component.main([]) == 0
    output = capsys.readouterr().out
    assert "source_fragment_cpu" in output and "Physical GPU/write safety: unknown" in output
    assert "vllm: satisfied" in output and "sglang: satisfied" in output


def test_component_helpers_do_not_launch_processes_network_or_engine_imports(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU component attempted an external action")

    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("system", "fork", "posix_spawn", "execv", "execve"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for name in ("connect", "connect_ex", "send", "sendall", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    original_import = builtins.__import__

    def cpu_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in ("vllm", "sglang", "torch", "nixl", "mooncake"):
            raise AssertionError("CPU component attempted a serving/accelerator import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_import)
    assert component.run_component_conformance()["status"] == "satisfied"


@pytest.mark.parametrize("damage", ["fragment", "manifest", "license", "missing", "extra", "traversal", "unknown_method"])
def test_untrusted_packaged_data_is_refused_before_execution(tmp_path, monkeypatch, capsys, damage):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for source in component._data_root().iterdir():
        (bundle / source.name).write_bytes(source.read_bytes())
    manifest_path = bundle / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "fragment":
        path = bundle / manifest["fragments"][0]["file"]
        path.write_bytes(path.read_bytes() + b"\n# changed excerpt\n")
    elif damage == "manifest":
        manifest_path.write_text("{}")
    elif damage == "license":
        path = bundle / manifest["licenses"][0]["file"]
        path.write_bytes(path.read_bytes() + b"changed license\n")
    elif damage == "missing":
        (bundle / manifest["fragments"][0]["file"]).unlink()
    elif damage == "extra":
        (bundle / "extra.py.txt").write_text("# unrecognized data\n")
    else:
        if damage == "traversal":
            manifest["fragments"][0]["file"] = "../outside.py.txt"
        else:
            manifest["fragments"][0]["method"] = "unreviewed_method"
        raw = (json.dumps(manifest, indent=2) + "\n").encode()
        manifest_path.write_bytes(raw)
        # Also challenge the independent inventory/path guards when an inert
        # test changes the trusted anchor. Normal CLI callers cannot do this.
        monkeypatch.setattr(component, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(component, "_data_root", lambda: bundle)
    executed = []
    monkeypatch.setattr(component, "_vllm_receipt", lambda: executed.append("vllm"))
    monkeypatch.setattr(component, "_sglang_receipt", lambda: executed.append("sglang"))
    assert component.main(["--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unknown" and result["physical_write_safety"] == "unknown"
    assert executed == []


def test_packaged_component_files_match_original_reviewed_fixtures():
    packaged = component._verified_bundle()
    for name in ["MANIFEST.json", *[i["file"] for i in MANIFEST["fragments"] + MANIFEST["licenses"]]]:
        assert component._data_root().joinpath(name).read_bytes() == (FIXTURES / name).read_bytes()
    assert len(packaged["fragments"]) == 25


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_component_refuses_a_loaded_serving_engine(engine, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, engine, object())
    assert component.main(["--json"]) == 2
    assert "fresh process" in json.loads(capsys.readouterr().out)["reason"]
