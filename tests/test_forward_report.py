"""Offline capture must not promote metadata into an isolation verdict."""
import json
from pathlib import Path

import pytest

from sdcguard import forward_report as F


def _input(tmp_path, raw=b'{"kind":"consume","request":"R"}\n'):
    root = tmp_path / "input"
    root.mkdir()
    (root / "worker.jsonl").write_bytes(raw)
    return root


def test_legacy_admission_keeps_physical_reads_unknown(tmp_path):
    source = _input(tmp_path)
    before = (source / "worker.jsonl").read_bytes()
    output = tmp_path / "packet"
    result = F.create_packet(source, output)
    assert result["observation_status"] == "unknown"
    verified = F.verify_packet(output)
    assert verified["integrity"] == "verified"
    assert verified["scope_verdict"] == "inconclusive"
    assert verified["physical_read_verdict"] == "unknown"
    assert verified["ownership_verdict"] == "unknown"
    assert (source / "worker.jsonl").read_bytes() == before
    assert (output / "inputs/worker.jsonl").read_bytes() == before


@pytest.mark.parametrize("raw", [b'', b'{}', b'{"x":1,"x":2}\n', b'{"x":NaN}\n',
                                    b'{"x":1e999}\n', b'{"x":"\\ud800"}\n', b'{}\n\n', b'\xff\n'])
def test_malformed_stream_remains_retained_and_unknown(tmp_path, raw):
    source = _input(tmp_path, raw)
    output = tmp_path / "packet"
    result = F.create_packet(source, output)
    assert result["observation_status"] == "unknown"
    assert F.verify_packet(output)["integrity"] == "verified"
    assert (output / "inputs/worker.jsonl").read_bytes() == raw


def test_resealing_a_forged_report_does_not_change_verdict(tmp_path):
    output = tmp_path / "packet"
    F.create_packet(_input(tmp_path), output)
    report = json.loads((output / "report.json").read_text())
    report["physical_read_verdict"] = "satisfied"
    (output / "report.json").write_bytes(F.io.json_bytes(report))
    manifest = json.loads((output / "manifest.json").read_text())
    raw = (output / "report.json").read_bytes()
    manifest["files"]["report.json"] = {"sha256": F.io.digest(raw), "bytes": len(raw)}
    (output / "manifest.json").write_bytes(F.io.json_bytes(manifest))
    assert F.verify_packet(output)["integrity"] == "invalid"


@pytest.mark.parametrize("target", ["report", "manifest"])
def test_equal_valued_float_cannot_replace_integer_receipt(tmp_path, target):
    output = tmp_path / "packet"
    F.create_packet(_input(tmp_path), output)
    manifest = json.loads((output / "manifest.json").read_text())
    if target == "report":
        report = json.loads((output / "report.json").read_text())
        receipt = report["input_files"]["worker.jsonl"]
        receipt["bytes"] = float(receipt["bytes"])
        raw = F.io.json_bytes(report)
        (output / "report.json").write_bytes(raw)
        manifest["files"]["report.json"] = {"sha256": F.io.digest(raw), "bytes": len(raw)}
    else:
        receipt = manifest["files"]["inputs/worker.jsonl"]
        receipt["bytes"] = float(receipt["bytes"])
    (output / "manifest.json").write_bytes(F.io.json_bytes(manifest))
    assert F.verify_packet(output)["integrity"] == "invalid"


def test_output_cannot_enter_input_tree(tmp_path):
    source = _input(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        F.create_packet(source, source / "packet")
    assert list(source.iterdir()) == [source / "worker.jsonl"]


def test_symlink_input_is_rejected_before_output(tmp_path):
    source = _input(tmp_path)
    (source / "alias.jsonl").symlink_to(source / "worker.jsonl")
    with pytest.raises(ValueError):
        F.create_packet(source, tmp_path / "packet")
    assert not (tmp_path / "packet").exists()


def test_extra_member_and_html_tamper_are_invalid(tmp_path):
    output = tmp_path / "packet"
    F.create_packet(_input(tmp_path), output)
    (output / "index.html").write_text("<p>Everything is safe</p>")
    assert F.verify_packet(output)["integrity"] == "invalid"


def test_foreign_files_are_not_silently_ignored(tmp_path):
    source = _input(tmp_path)
    (source / "notes.txt").write_text("unselected evidence")
    with pytest.raises(ValueError, match="JSONL"):
        F.create_packet(source, tmp_path / "packet")


def test_cli_unknown_has_nonzero_exit(tmp_path, capsys):
    source = _input(tmp_path)
    output = tmp_path / "packet"
    assert F.main(["inspect", str(source), "--out", str(output)]) == 2
    assert json.loads(capsys.readouterr().out)["scope_verdict"] == "inconclusive"
    assert F.main(["verify", str(output)]) == 2
    assert json.loads(capsys.readouterr().out)["integrity"] == "verified"


def test_report_escapes_untrusted_recording_names():
    report = F.inspect_payload({'<script>alert(1)</script>.jsonl': b'{}\n'})
    rendered = F.render_report(report).decode()
    assert '<script>' not in rendered
    assert '&lt;script&gt;' in rendered


def test_line_limit_is_fail_closed(monkeypatch):
    monkeypatch.setattr(F, "MAX_RECORDS", 1)
    with pytest.raises(ValueError, match="record count"):
        F.parse_records(b'{}\n{}\n')


def test_complete_metadata_remains_separate_from_isolation(tmp_path):
    from test_forward_inputs import ordinary_records
    raw = b''.join((json.dumps(record) + '\n').encode() for record in ordinary_records())
    source = _input(tmp_path, raw)
    output = tmp_path / "packet"
    result = F.create_packet(source, output)
    assert result["observation_status"] == "observed"
    checked = F.verify_packet(output)
    assert checked["integrity"] == "verified"
    assert checked["physical_read_verdict"] == checked["ownership_verdict"] == "unknown"
    assert checked["scope_verdict"] == "inconclusive"


def test_duplicate_process_sessions_are_not_independent_observations(tmp_path):
    from test_forward_inputs import ordinary_records
    raw = b''.join((json.dumps(record) + '\n').encode() for record in ordinary_records())
    source = _input(tmp_path, raw)
    (source / "duplicate.jsonl").write_bytes(raw)
    output = tmp_path / "packet"
    assert F.create_packet(source, output)["observation_status"] == "unknown"
    report = F.verify_packet(output)["report"]
    assert all(s["assessment"]["observation_status"] == "unknown" for s in report["sessions"])


def test_capture_enforces_cumulative_budget_before_output(tmp_path, monkeypatch):
    source = _input(tmp_path, b'{}\n')
    (source / "second.jsonl").write_bytes(b'{}\n')
    monkeypatch.setattr(F, "MAX_INPUT_BYTES", 5)
    with pytest.raises(ValueError, match="limit"):
        F.create_packet(source, tmp_path / "packet")
    assert not (tmp_path / "packet").exists()


def test_long_session_is_portable_without_losing_snapshot_summaries(tmp_path):
    from copy import deepcopy
    from test_forward_inputs import ordinary_records
    header, template, footer = ordinary_records()
    count = 2048
    records = [header]
    for i in range(1, count + 1):
        item = deepcopy(template)
        item.update(seq=i, forward_id=i, snapshot_id=f"{i:032x}")
        records.append(item)
    footer.update(seq=count + 1, snapshots=count)
    records.append(footer)
    raw = b''.join((json.dumps(record) + '\n').encode() for record in records)
    source = _input(tmp_path, raw)
    output = tmp_path / "packet"
    assert F.create_packet(source, output)["observation_status"] == "observed"
    verified = F.verify_packet(output)
    assert verified["integrity"] == "verified"
    session = verified["report"]["sessions"][0]["assessment"]
    assert session["snapshot_count"] == session["valid_snapshot_count"] == count
    assert len(session["snapshots"]) == count
    assert session["physical_read_verdict"] == session["ownership_verdict"] == "unknown"
