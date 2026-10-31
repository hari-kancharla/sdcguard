"""Two real H100 recordings, one on each supported engine build. Owner: Hari Krishna Kancharla.

Both were recorded on Oct 31, 2026 with the same workload, model and effective startup settings, including
--gpu-blocks 8192. These tests replay them offline; nothing here starts an engine.
"""
import json
from pathlib import Path

import pytest

from sdcguard import checksum_profiles as profiles, cli, evidence_io as io, transfer_upgrade as upgrade

DATA = Path(__file__).parent / "data"
BASELINE = DATA / "checksum_owner_9b42f5ab"
CANDIDATE = DATA / "checksum_owner_b340ee03"
NAMES = ("RESULT.json", "ordinary-evidence.zip", "CPU_PREFLIGHT.json")
SUMMARY = dict(compared_slots=153, preserved=153, regressed=0, unverifiable=0, lost_passing_evidence=0)
FULL_PROMPT_HOLDS = ["baseline_full_prompt_policy_held", "candidate_full_prompt_policy_held"]


def load(folder):
    pin = json.loads((folder / "SOURCE.json").read_bytes())
    for name, identity in pin["files"].items():
        raw = (folder / name).read_bytes()
        assert {"bytes": len(raw), "sha256": io.digest(raw)} == identity
    return tuple((folder / n).read_bytes() for n in NAMES)


@pytest.fixture(scope="module")
def before():
    return load(BASELINE)


@pytest.fixture(scope="module")
def after():
    return load(CANDIDATE)


@pytest.fixture(scope="module")
def compared(before, after):
    return upgrade.evaluate(before, after)


@pytest.mark.parametrize("side,revision", [("baseline", profiles.BASELINE), ("candidate", profiles.CANDIDATE)])
def test_each_build_passes_transfer_and_holds_the_full_prompt(compared, side, revision):
    answer = compared[side]
    assert answer["transfer_decision"] == "PASS", answer["hold_reasons"]
    assert answer["errors"] == [] and answer["hold_reasons"] == []
    assert answer["engine"]["source_revision"] == revision
    assert answer["full_prompt_gate"]["decision"] == "HOLD"
    assert answer["full_prompt_gate"]["passed_requests"] == 9
    summary = answer["summary"]
    assert summary["verified_requests"] == 153 and summary["statuses"] == {"VERIFIED": 153}
    assert summary["compared_transfer_tokens"] == summary["expected_transfer_tokens"] == 9822
    assert summary["decode_cached_tokens_outside_transfer"] == 99968
    assert summary["prompt_tokens"] == 109790 and summary["expected_chunk_positions"] == 688


def test_upgrade_keeps_every_transfer_check(compared):
    assert compared["transfer_preservation_decision"] == "PASS", compared["hold_reasons"]
    assert compared["comparison_status"] == "COMPARABLE"
    assert compared["summary"] == SUMMARY
    assert compared["recording_relation"] == "distinct_recorded_sessions"
    assert compared["build_relation"] == "different_recorded_build"
    assert compared["runtime_upgrade_witness"] == "distinct source-bound recordings"
    assert compared["baseline"]["recording"]["run_id"] != compared["candidate"]["recording"]["run_id"]


def test_full_prompt_hold_still_blocks_the_upgrade(compared):
    assert compared["upgrade_decision"] == "HOLD"
    assert compared["hold_reasons"] == FULL_PROMPT_HOLDS


def test_reversed_order_is_not_a_qualified_upgrade(before, after):
    answer = upgrade.evaluate(after, before)
    assert answer["comparison_status"] == "COMPARABLE"
    assert answer["transfer_preservation_decision"] == "HOLD"
    assert answer["upgrade_decision"] == "HOLD"
    assert answer["hold_reasons"] == FULL_PROMPT_HOLDS + ["changed_build_transition_not_qualified"]


def test_cli_replays_both_recordings(compared, monkeypatch, capsys):
    assert cli.main(["transfer-compare", str(BASELINE), str(CANDIDATE), "--json"]) == 2
    assert json.loads(capsys.readouterr().out) == compared
    monkeypatch.setattr(upgrade, "inspect", lambda *a: compared)
    assert cli.main(["transfer-compare", str(BASELINE), str(CANDIDATE)]) == 2
    text = capsys.readouterr().out
    assert text.startswith("Offline upgrade comparison: HOLD\nTransfer preservation: PASS\n")
    assert "153 preserved, 0 regressed, 0 unverifiable" in text
    assert "different_recorded_build" in text and "full-prompt HOLD" in text
