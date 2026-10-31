"""Archive read errors remain inconclusive. Owner: Hari Krishna Kancharla."""
from io import BytesIO
from pathlib import Path
import zipfile
import zlib

import pytest

from sdcguard import cli, evidence_io, native_checks


@pytest.mark.parametrize("error", [EOFError, zlib.error])
def test_decompressor_read_error_becomes_evidence_error(monkeypatch, error):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("record.json", b"{}")

    def unavailable(self, name, *args, **kwargs):
        raise error("saved archive could not be decompressed")

    monkeypatch.setattr(zipfile.ZipFile, "read", unavailable)
    with pytest.raises(evidence_io.EvidenceError, match="unsupported evidence archive"):
        native_checks._archive(stream.getvalue())


@pytest.mark.parametrize("error", [EOFError, zlib.error])
def test_capture_reports_inconclusive_for_decompressor_error(tmp_path, monkeypatch, capsys, error):
    fixture = Path(__file__).parent / "data/native_check_owner_r3"
    source = tmp_path / "owner"
    source.mkdir()
    for name in ("RESULT.json", "ordinary-evidence.zip"):
        (source / name).write_bytes((fixture / name).read_bytes())

    def unavailable(self, name, *args, **kwargs):
        raise error("saved archive could not be decompressed")

    monkeypatch.setattr(zipfile.ZipFile, "read", unavailable)
    output = tmp_path / "coverage"
    assert cli.main(["native-checks", "capture", "--owner-result", str(source),
                     "--out", str(output)]) == 2
    text = capsys.readouterr().out
    assert "Native-check evidence is inconclusive" in text
    assert "unsupported evidence archive" in text
    assert not output.exists()
