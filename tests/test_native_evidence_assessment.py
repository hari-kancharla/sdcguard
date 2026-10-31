"""Assessment packet boundaries only. Owner: Hari Krishna Kancharla.

Use the existing clean CPU metadata fixture. No engine source is executed.
"""
import pytest

from sdcguard import assessment_bundle
from sdcguard import evidence_combined as combined
from sdcguard import evidence_envelope as envelope
from sdcguard import evidence_io as io
from sdcguard import evidence_native_io as bounded
from test_assessment_bundle import supported


def test_selected_assessment_keeps_its_original_bytes_and_verifies(supported, tmp_path):
    packet = tmp_path / "assessment"
    assessment_bundle.create_bundle(supported["path"], packet, run_ready=False)
    before = bounded.capture_snapshot(packet)
    result = combined.create_bundle(tmp_path / "combined", recorded_packet=packet, require_native=["decode"])
    assert result["report"]["legacy"]["integrity"] == "verified"
    assert result["report"]["status"] == "inconclusive"
    assert combined.verify_bundle(tmp_path / "combined")["integrity"] == "verified"
    assert bounded.capture_snapshot(packet) == before


def test_original_assessment_is_rechecked_after_output_completion(supported, tmp_path, monkeypatch):
    packet = tmp_path / "assessment"
    assessment_bundle.create_bundle(supported["path"], packet, run_ready=False)
    actual = envelope.complete
    def complete(*args, **kwargs):
        result = actual(*args, **kwargs)
        (packet / "late-local-member").write_bytes(b"changed after snapshot")
        return result
    monkeypatch.setattr(envelope, "complete", complete)
    with pytest.raises(io.EvidenceError, match="membership changed"):
        combined.create_bundle(tmp_path / "combined", recorded_packet=packet, require_native=["decode"])


def test_assessment_directory_admission_precedes_any_recorded_reconstruction(tmp_path, monkeypatch):
    packet = tmp_path / "oversized-shape"
    packet.mkdir()
    for index in range(10):
        (packet / ("empty-" + str(index))).mkdir()
    actual = envelope.materialize
    materialized_recorded = []
    def materialize(root, *args, **kwargs):
        if root.name == "recorded":
            materialized_recorded.append(root)
        return actual(root, *args, **kwargs)
    monkeypatch.setattr(envelope, "materialize", materialize)
    monkeypatch.setattr(io, "MAX_FILES", 12)
    result = combined.create_bundle(tmp_path / "combined", recorded_packet=packet, require_native=["decode"])
    assert materialized_recorded == []
    assert result["report"]["status"] == "inconclusive"
    assert "directory" in result["report"]["legacy"]["errors"][0]
