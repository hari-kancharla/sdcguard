"""Recorded mapping results must survive capture without gaining broader claims."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from sdcguard import assignment_report as A
from test_assignment_obligations import ordinary_records, one


def _raw(records):
    return b"".join(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
                    for record in records)


def _input(tmp_path, raw=None):
    root = tmp_path / "input"
    root.mkdir()
    (root / "worker.jsonl").write_bytes(_raw(ordinary_records()) if raw is None else raw)
    return root


def _reseal(root):
    members = {name: raw for name, raw in A.io.snapshot(root).items() if name != "manifest.json"}
    (root / "manifest.json").write_bytes(A.io.json_bytes({"schema": A.FORMAT, "files": A.io.file_receipts(members)}))


def _limits(result):
    assert result["ownership_verdict"] == result["physical_read_verdict"] == "unknown"
    assert result["content_provenance"] == "unknown"
    assert result["runtime_isolation"] == result["authenticity"] == "not established"


@pytest.mark.parametrize("case,verdict,code", [("match", "satisfied", 0),
                                               ("mismatch", "violated", 1),
                                               ("missing", "unknown", 2)])
def test_roundtrip_named_rule_and_exit_codes(tmp_path, capsys, case, verdict, code):
    records = ordinary_records()
    if case == "mismatch":
        one(records, "model_entry")["layers"][0]["rows"][0][0] = 31
    if case == "missing":
        records.pop()
    source = _input(tmp_path, _raw(records))
    before = (source / "worker.jsonl").read_bytes()
    output = tmp_path / "packet"
    assert A.main(["inspect", str(source), "--out", str(output)]) == code
    created = json.loads(capsys.readouterr().out)
    assert created["verdict"] == verdict
    _limits(created)
    assert A.main(["verify", str(output)]) == code
    verified = json.loads(capsys.readouterr().out)
    assert verified["integrity"] == "verified" and verified["verdict"] == verdict
    _limits(verified)
    assert (output / "inputs/worker.jsonl").read_bytes() == before
    assert (source / "worker.jsonl").read_bytes() == before
    assert verified["report"]["sessions"][0]["recorded_identity"]["profile"] == records[0]["profile"]


@pytest.mark.parametrize("raw", [b"", b"{}", b"{}\n\n", b"\xff\n", b'{"x":NaN}\n',
                                    b'{"x":1,"x":2}\n', b'{"x":1e999}\n', b'{"x":"\\ud800"}\n'])
def test_unusable_bytes_remain_retained_and_unknown(tmp_path, raw):
    source = _input(tmp_path, raw)
    packet = tmp_path / "packet"
    assert A.create_packet(source, packet)["verdict"] == "unknown"
    checked = A.verify_packet(packet)
    assert checked["integrity"] == "verified" and checked["verdict"] == "unknown"
    assert (packet / "inputs/worker.jsonl").read_bytes() == raw
    _limits(checked)


@pytest.mark.parametrize("target", ["report", "html", "metadata", "receipt_type", "extra"])
def test_resealed_derived_tamper_cannot_gain_a_verdict(tmp_path, target):
    packet = tmp_path / "packet"
    A.create_packet(_input(tmp_path), packet)
    if target == "report":
        report = json.loads((packet / "report.json").read_text())
        report["ownership_verdict"] = "satisfied"
        (packet / "report.json").write_bytes(A.io.json_bytes(report))
    elif target == "html":
        (packet / "index.html").write_text("<h1>Safe</h1>")
    elif target == "metadata":
        (packet / "metadata/evaluators.json").write_text("{}\n")
    elif target == "receipt_type":
        report = json.loads((packet / "report.json").read_text())
        r = report["input_files"]["worker.jsonl"]
        r["bytes"] = float(r["bytes"])
        (packet / "report.json").write_bytes(A.io.json_bytes(report))
    else:
        (packet / "extra.txt").write_text("unselected evidence")
    _reseal(packet)
    checked = A.verify_packet(packet)
    assert checked["integrity"] == "invalid" and checked["verdict"] == "unknown"
    _limits(checked)


def test_duplicate_session_is_unknown_including_a_negative_copy():
    records = ordinary_records()
    one(records, "model_entry")["layers"][0]["rows"][0][0] = 31
    report = A.inspect_payload({"one.jsonl": _raw(records), "two.jsonl": _raw(records)})
    assert report["verdict"] == "unknown"
    assert all(entry["assessment"]["verdict"] == "unknown" for entry in report["sessions"])


@pytest.mark.parametrize("tail", [b'{"unfinished":', b'not-json\n', b''])
def test_invalid_tail_does_not_hide_a_valid_duplicate_header(tail):
    records = ordinary_records()
    one(records, "model_entry")["layers"][0]["rows"][0][0] = 31
    duplicate = _raw(records[:1]) + tail
    report = A.inspect_payload({"complete.jsonl": _raw(records), "broken.jsonl": duplicate})
    assert report["verdict"] == "unknown"
    assert all(entry["assessment"]["verdict"] == "unknown" for entry in report["sessions"])
    assert len({entry["session"] for entry in report["sessions"]}) == 1


def test_independent_complete_negative_survives_unrelated_unknown():
    records = ordinary_records()
    one(records, "model_entry")["layers"][0]["rows"][0][0] = 0
    report = A.inspect_payload({"negative.jsonl": _raw(records), "legacy.jsonl": b'{"kind":"consume"}\n'})
    assert report["verdict"] == "violated"
    assert report["scope_verdict"] == "not fine"
    _limits(report)


def test_sessions_do_not_supply_each_others_missing_initialization():
    complete = ordinary_records()
    missing = deepcopy(complete)
    missing.pop(1)
    for index, record in enumerate(missing):
        record["session"] = "a" * 32
        record["process"] = "b" * 32
        record["seq"] = index
    missing[-1]["records"] = len(missing)
    report = A.inspect_payload({"complete.jsonl": _raw(complete), "missing.jsonl": _raw(missing)})
    assert report["verdict"] == "unknown"
    assert [s["assessment"]["verdict"] for s in report["sessions"]] == ["satisfied", "unknown"]


def test_html_escapes_recorded_names_and_identity():
    report = A.inspect_payload({'<script>alert(1)</script>.jsonl': b'{}\n'})
    rendered = A.render_report(report).decode()
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "Runtime isolation: not established" in rendered


@pytest.mark.parametrize("case", ["nested_output", "symlink", "foreign", "existing_output"])
def test_unsafe_capture_rejected_before_writing(tmp_path, case):
    source = _input(tmp_path)
    target = tmp_path / "packet"
    if case == "nested_output":
        target = source / "packet"
    elif case == "symlink":
        (source / "alias.jsonl").symlink_to(source / "worker.jsonl")
    elif case == "foreign":
        (source / "notes.txt").write_text("not a selected recording")
    else:
        target.mkdir()
        (target / "keep.txt").write_text("preserve")
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    with pytest.raises((OSError, ValueError)):
        A.create_packet(source, target)
    assert {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()} == before
    if case == "existing_output":
        assert (target / "keep.txt").read_text() == "preserve"
    else:
        assert not target.exists()


def test_cumulative_capture_budget_checked_before_output(tmp_path, monkeypatch):
    source = _input(tmp_path, b"{}\n")
    (source / "second.jsonl").write_bytes(b"{}\n")
    monkeypatch.setattr(A, "MAX_INPUT_BYTES", 5)
    with pytest.raises(ValueError, match="limit"):
        A.create_packet(source, tmp_path / "packet")
    assert not (tmp_path / "packet").exists()


def test_record_count_limit_is_unknown_not_a_partial_pass(monkeypatch):
    monkeypatch.setattr(A, "MAX_RECORDS", 1)
    assert A.inspect_payload({"worker.jsonl": _raw(ordinary_records())})["verdict"] == "unknown"


def test_cli_dispatch_stays_offline(tmp_path):
    source = _input(tmp_path)
    packet = tmp_path / "packet"
    script = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'modal', 'requests', 'httpx'}:
            raise AssertionError('unexpected runtime import')
sys.meta_path.insert(0, Guard())
from sdcguard.cli import main
raise SystemExit(main(sys.argv[1:]))
'''
    for args in [["assignment", "inspect", str(source), "--out", str(packet)],
                 ["assignment", "verify", str(packet)]]:
        completed = subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert json.loads(completed.stdout)["verdict"] == "satisfied"
