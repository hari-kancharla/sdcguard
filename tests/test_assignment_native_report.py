"""Retained native close receipts must gate portable mapping results.

Owner: Hari Krishna Kancharla.
These are host recorder and local file controls, not a running engine.
"""
import json

import pytest

from sdcguard import assignment_report as A
from sdcguard.lifetime import mapping_closure as C
from test_assignment_init import primary
from test_mapping_activation import active, isolated_activation
from test_mapping_closure import finished, rewrite_pair


def reseal(packet):
    captured = A.io.snapshot(packet)
    manifest = A.io.parse_json(captured.pop("manifest.json"))
    manifest["files"] = A.io.file_receipts(captured)
    (packet / "manifest.json").write_bytes(A.io.json_bytes(manifest))


def test_native_command_preserves_exact_receipts_and_rechecks_without_source(active, tmp_path, capsys):
    sink, receipt = finished(active)
    before = {p.name:p.read_bytes() for p in sink.path.parent.iterdir()}
    packet = tmp_path / "native-packet"
    assert A.main(["capture", str(sink.path.parent), "--out", str(packet)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["recording_bookkeeping"] == "locally consistent"
    assert result["client_attribution"] == result["ownership_verdict"] == "unknown"
    assert (packet / "inputs" / sink.path.name).read_bytes() == before[sink.path.name]
    assert (packet / "closures" / receipt.name).read_bytes() == before[receipt.name]
    assert A.io.parse_json((packet / "manifest.json").read_bytes())["schema"] == A.NATIVE_FORMAT
    moved = tmp_path / "moved-packet"
    packet.rename(moved)
    sink.path.parent.rename(tmp_path / "original-recording")
    checked = A.verify_packet(moved)
    assert checked["integrity"] == "verified" and checked["verdict"] == "satisfied"
    assert checked["recording_bookkeeping"] == "locally consistent"
    assert checked["report"]["schema"] == A.NATIVE_REPORT
    assert checked["report"]["closure_receipts"][receipt.name]["sha256"] == A.io.digest(before[receipt.name])


@pytest.mark.parametrize("verdict,code", [("violated", 1), ("unknown", 2)])
def test_native_bookkeeping_does_not_turn_mapping_outcomes_green(active, tmp_path, capsys, verdict, code):
    sink, path = finished(active)
    records = A.parse_records(sink.path.read_bytes())
    if verdict == "violated":
        next(r for r in records if r["kind"] == "model_entry")["layers"][0]["rows"][0][0] = 0
    else:
        next(r for r in records if r["kind"] == "assignments")["rows"][0]["blocks"] = None
    rewrite_pair(sink, path, records, json.loads(path.read_bytes()))
    packet = tmp_path / "packet"
    assert A.main(["capture", str(sink.path.parent), "--out", str(packet)]) == code
    assert json.loads(capsys.readouterr().out)["recording_bookkeeping"] == "locally consistent"
    checked = A.verify_packet(packet)
    assert checked["integrity"] == "verified" and checked["verdict"] == verdict


@pytest.mark.parametrize("kind", ["negative_close", "missing", "counter", "source", "unfinished", "extra"])
def test_native_bad_closure_refuses_output_and_reports_unknown(active, tmp_path, capsys, kind):
    sink, path = finished(active)
    receipt = json.loads(path.read_bytes())
    if kind == "negative_close":
        receipt["close_succeeded"] = False
    elif kind == "counter":
        receipt["journal"]["records"] = True
    elif kind == "source":
        receipt["identity"]["source_sha256"] = {}
    if kind == "missing":
        path.unlink()
    elif kind == "unfinished":
        path.write_bytes(b'{"incomplete":')
    elif kind == "extra":
        (path.parent / "unfinished.tmp").write_bytes(b"not closed")
    else:
        path.write_bytes(A.io.json_bytes(receipt))
    packet = tmp_path / "packet"
    assert A.main(["capture", str(sink.path.parent), "--out", str(packet)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["verdict"] == "unknown" and result["integrity"] == "invalid"
    assert result["recording_bookkeeping"] != "locally consistent"
    assert not packet.exists()


@pytest.mark.parametrize("kind", ["negative_close", "missing", "counter", "unexpected", "html", "report", "evaluator"])
def test_resealed_native_packet_cannot_bypass_close_or_recomputation(active, tmp_path, kind):
    sink, path = finished(active)
    packet = tmp_path / "packet"
    A.create_packet(path.parent, packet, native=True)
    retained = packet / "closures" / path.name
    if kind in ("negative_close", "counter"):
        receipt = json.loads(retained.read_bytes())
        if kind == "negative_close":
            receipt["close_succeeded"] = False
        else:
            receipt["observer"]["events"] = 0
        retained.write_bytes(A.io.json_bytes(receipt))
    elif kind == "missing":
        retained.unlink()
    elif kind == "unexpected":
        (packet / "closures/extra.json").write_bytes(b"{}\n")
    elif kind == "html":
        (packet / "index.html").write_text("<h1>safe</h1>")
    elif kind == "report":
        report = json.loads((packet / "report.json").read_bytes())
        report["ownership_verdict"] = "satisfied"
        (packet / "report.json").write_bytes(A.io.json_bytes(report))
    else:
        (packet / "metadata/evaluators.json").write_bytes(A.io.json_bytes(A.evaluator_hashes()))
    reseal(packet)
    checked = A.verify_packet(packet)
    assert checked["integrity"] == "invalid" and checked["verdict"] == "unknown"
    assert checked["recording_bookkeeping"] != "locally consistent"


def test_raw_inspection_cannot_claim_native_close_check(active, tmp_path):
    sink, path = finished(active)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / sink.path.name).write_bytes(sink.path.read_bytes())
    packet = tmp_path / "packet"
    A.create_packet(raw, packet)
    checked = A.verify_packet(packet)
    assert checked["integrity"] == "verified" and checked["verdict"] == "satisfied"
    assert checked["recording_bookkeeping"] == "not checked"
    assert "closure_receipts" not in checked["report"]
    assert A.io.parse_json((packet / "manifest.json").read_bytes())["schema"] == A.FORMAT


@pytest.mark.parametrize("which", ["input", "output"])
def test_native_nested_output_is_rejected_before_capture(active, tmp_path, which):
    sink, path = finished(active)
    source = path.parent
    output = source / "nested" if which == "input" else source.parent
    with pytest.raises(A.io.EvidenceError, match="trees must be separate"):
        A.create_packet(source, output, native=True)


@pytest.mark.parametrize("kind", ["nonbytes", "receipt_count", "oversized", "unknown_name", "alias"])
def test_retained_pair_validator_keeps_same_bounds_as_filesystem_capture(active, monkeypatch, kind):
    sink, path = finished(active)
    captured = C.capture_closed(path.parent)
    payload, receipts = captured["payload"], captured["receipts"]
    if kind == "nonbytes":
        payload[sink.path.name] = bytearray(payload[sink.path.name])
    elif kind == "receipt_count":
        receipts.clear()
    elif kind == "oversized":
        monkeypatch.setattr(C, "MAX_RECEIPT_BYTES", 1)
    elif kind == "unknown_name":
        raw = payload.pop(sink.path.name)
        payload["wrong.jsonl"] = raw
        receipts["wrong.closure.json"] = receipts.pop(path.name)
    else:
        payload["../" + sink.path.name] = payload.pop(sink.path.name)
    with pytest.raises((ValueError, OSError)):
        C.validate_captured(payload, receipts)
