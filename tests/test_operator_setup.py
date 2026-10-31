"""Installed setup and inert local preflight; no serving or device work."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from sdcguard import assessment
from sdcguard import evidence_io as io
from sdcguard import operator_setup as operator
from test_assessment_bundle import supported
from test_evidence_bundle import static_config


def rehash(root):
    path = root / "manifest.json"
    value = io.parse_json(path.read_bytes())
    value["files"] = io.file_receipts({name: raw for name, raw in io.snapshot(root).items() if name != "manifest.json"})
    value["directories"] = io.names(root, directories=True)
    path.write_bytes(io.json_bytes(value))


def static(tmp_path, name, **kwargs):
    source = tmp_path / name
    source.mkdir()
    return static_config(source, **kwargs)


def recorded_failure(supported):
    rows = supported["make_events"](stale=True, no_check=True)
    supported["events"].write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_export_has_distinct_incomplete_inputs_and_exact_installed_resources(tmp_path):
    out = tmp_path / "setup"
    result = operator.export_setup(out)
    assert result["scope_verdict"] == "inconclusive"
    source = io.parse_json((out / "static-deployment.example.json").read_bytes())
    recorded = io.parse_json((out / "recording-deployment.example.json").read_bytes())
    assert source["schema"] == "sdcguard.config-safety.v1"
    assert "model" not in source and all("engine" not in worker for worker in source["workers"])
    assert recorded["schema"] == 1 and recorded["model_revision"] is None
    support = io.parse_json((out / "SUPPORT.json").read_bytes())
    assert {row["engine"] for row in support["static_profiles"]} == {"vllm", "sglang", "vllm-ascend", "vllm-lmcache"}
    assert next(row for row in support["static_profiles"] if row["engine"] == "sglang")["version"] == "0.5.22"
    assert support["upgrade_policy"] == "preserve-named-contracts.v1"
    assert operator.verify_output(out)["integrity"] == "verified"
    assert not (out / "guides/SGLANG_TRANSITION.md").exists()


def test_packaged_guides_and_profile_shapes_match_the_current_source_inputs():
    payload, manifest = operator._resources()
    root = Path(__file__).resolve().parents[1]
    for name, receipt in manifest["source_inputs"].items():
        assert payload[name] == (root / receipt["source"]).read_bytes(), name


@pytest.mark.parametrize("member", ["README.md", "SUPPORT.json", "metadata/resources.json", "metadata/evaluators.json"])
def test_rehashed_edited_export_is_not_the_installed_setup(tmp_path, member):
    out = tmp_path / "setup"
    operator.export_setup(out)
    (out / member).write_bytes(b"edited\n")
    rehash(out)
    result = operator.verify_output(out)
    assert result["integrity"] == "failed" and result["scope_verdict"] == "inconclusive"


def test_export_fixed_directory_inventory_rejects_rehashed_empty_addition(tmp_path):
    out = tmp_path / "setup"
    operator.export_setup(out)
    (out / "extra").mkdir()
    rehash(out)
    assert operator.verify_output(out)["integrity"] == "failed"


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_static_preflight_recomputes_and_survives_relocation(tmp_path, engine):
    config = static(tmp_path, "inputs", engine=engine)
    before = io.snapshot(config.parent)
    out = tmp_path / "preflight"
    result = operator.preflight(out, static=config)
    assert result["scope_verdict"] == "fine", result
    assert result["report"]["components"] == "not selected"
    assert result["report"]["readiness_network"] == "disabled"
    assert result["report"]["runtime_isolation"] == "not established"
    assert result["report"]["selection"] == {"schema": operator.PREFLIGHT_FORMAT, "static": True, "recording": False}
    assert io.snapshot(config.parent) == before
    moved = tmp_path / "moved"
    shutil.copytree(out, moved)
    shutil.rmtree(out)
    shutil.rmtree(config.parent)
    verified = operator.verify_output(moved)
    assert verified["integrity"] == "verified" and verified["scope_verdict"] == "fine", verified


def test_unavailable_saved_startup_stays_unknown(tmp_path):
    config = static(tmp_path, "inputs", logs=False)
    out = tmp_path / "preflight"
    result = operator.preflight(out, static=config)
    assert result["scope_verdict"] == "inconclusive"
    assert result["report"]["sections"]["static"]["missing"]
    assert operator.verify_output(out)["scope_verdict"] == "inconclusive"


def test_recording_uses_common_conservative_contract_and_no_original_paths_after_move(supported, tmp_path):
    before = io.snapshot(supported["root"])
    out = tmp_path / "preflight"
    result = operator.preflight(out, recording=supported["path"])
    assert result["scope_verdict"] == "fine", result
    assert result["report"]["sections"]["recording"]["findings"][0]["verdict"] == "satisfied"
    assert result["report"]["sections"]["recording"]["proposed_fix_source"]["status"] == "present"
    assert (out / "sections/recording/inputs/recorded/manifest.json").is_file()
    assert not (out / "legacy").exists()
    assert io.snapshot(supported["root"]) == before
    moved = tmp_path / "moved"
    shutil.copytree(out, moved)
    shutil.rmtree(out)
    shutil.rmtree(supported["root"])
    verified = operator.verify_output(moved)
    assert verified["integrity"] == "verified" and verified["scope_verdict"] == "fine", verified


def test_global_recording_gap_does_not_become_an_authoritative_native_violation(supported, tmp_path):
    recorded_failure(supported)
    config = deepcopy(supported["config"])
    config["model_revision"] = "main"
    supported["path"].write_bytes(io.json_bytes(config))
    out = tmp_path / "preflight"
    result = operator.preflight(out, recording=supported["path"])
    assert result["scope_verdict"] == "inconclusive", result
    section = result["report"]["sections"]["recording"]
    assert section["findings"] and all(row["verdict"] == "unknown" for row in section["findings"])
    assert section["proposed_fix_source"]["status"] == "present"
    assert operator.verify_output(out)["scope_verdict"] == "inconclusive"


def test_configured_readiness_is_not_fetched_or_fabricated(supported, tmp_path, monkeypatch):
    config = deepcopy(supported["config"])
    config["ready"] = {"metrics_urls": ["http://127.0.0.1:1/metrics"]}
    supported["path"].write_bytes(io.json_bytes(config))
    for name in ("run", "call", "Popen"):
        monkeypatch.setattr(subprocess, name, lambda *a, **k: pytest.fail("no process or readiness request allowed"))
    out = tmp_path / "preflight"
    result = operator.preflight(out, recording=supported["path"])
    assert result["scope_verdict"] == "inconclusive"
    assert any("readiness" in item for item in result["report"]["sections"]["recording"]["missing"])
    assert operator.verify_output(out)["scope_verdict"] == "inconclusive"


def test_known_static_failure_survives_missing_recording(tmp_path):
    config = static(tmp_path, "inputs", short=True)
    out = tmp_path / "preflight"
    result = operator.preflight(out, static=config, recording=tmp_path / "missing-recording.json")
    assert result["scope_verdict"] == "not fine"
    assert result["report"]["selection"]["recording"] is True
    assert result["report"]["sections"]["recording"]["scope_verdict"] == "inconclusive"
    verified = operator.verify_output(out)
    assert verified["integrity"] == "verified" and verified["scope_verdict"] == "not fine", verified


def test_known_recording_failure_survives_invalid_static_input(supported, tmp_path):
    recorded_failure(supported)
    config = tmp_path / "invalid-static.json"
    config.write_bytes(b'{"schema":"unsupported"}\n')
    out = tmp_path / "preflight"
    result = operator.preflight(out, static=config, recording=supported["path"])
    assert result["scope_verdict"] == "not fine", result
    assert result["report"]["sections"]["static"]["scope_verdict"] == "inconclusive"
    assert operator.verify_output(out)["scope_verdict"] == "not fine"


def test_incomplete_legacy_capture_is_retained_as_unknown_without_common_promotion(supported, tmp_path):
    config = deepcopy(supported["config"])
    config["events"] = "missing-events"
    supported["path"].write_bytes(io.json_bytes(config))
    out = tmp_path / "preflight"
    result = operator.preflight(out, recording=supported["path"])
    section = result["report"]["sections"]["recording"]
    assert result["scope_verdict"] == "inconclusive"
    assert section["receipt"]["retained_kind"] == "legacy_capture"
    assert section["findings"] == []
    assert operator.verify_output(out)["integrity"] == "verified"


@pytest.mark.parametrize("kind", ["static", "recording"])
@pytest.mark.parametrize("raw", [b'{"schema":1,"schema":1}\n', b'{"schema":NaN}\n'])
def test_duplicate_and_nonfinite_supplied_configs_are_unknown_receipts(tmp_path, kind, raw):
    config = tmp_path / "input.json"
    config.write_bytes(raw)
    out = tmp_path / "preflight"
    result = operator.preflight(out, **{kind: config})
    assert result["scope_verdict"] == "inconclusive"
    assert result["report"]["sections"][kind]["receipt"]["retained_kind"] is None
    assert operator.verify_output(out)["integrity"] == "verified"
    assert config.read_bytes() == raw


@pytest.mark.parametrize("kind", ["static", "recording"])
def test_linked_config_is_not_followed(tmp_path, kind):
    original = tmp_path / "real.json"
    original.write_bytes(b"{}\n")
    link = tmp_path / "linked.json"
    link.symlink_to(original)
    out = tmp_path / "preflight"
    result = operator.preflight(out, **{kind: link})
    assert result["scope_verdict"] == "inconclusive"
    assert result["report"]["sections"][kind]["receipt"]["input_sha256"] is None
    assert original.read_bytes() == b"{}\n"


def test_nonregular_config_is_refused_without_blocking(tmp_path):
    path = tmp_path / "pipe"
    os.mkfifo(path)
    result = operator.preflight(tmp_path / "preflight", static=path)
    assert result["scope_verdict"] == "inconclusive"
    assert result["report"]["sections"]["static"]["receipt"]["retained_kind"] is None


def test_event_directory_does_not_follow_a_symlink(supported, tmp_path):
    alias = supported["root"] / "alias"
    alias.symlink_to(supported["root"] / "events", target_is_directory=True)
    config = deepcopy(supported["config"])
    config["events"] = "alias"
    supported["path"].write_bytes(io.json_bytes(config))
    result = operator.preflight(tmp_path / "preflight", recording=supported["path"])
    assert result["scope_verdict"] == "inconclusive"
    assert result["report"]["sections"]["recording"]["receipt"]["retained_kind"] is None


@pytest.mark.parametrize("key", ["events", "engine_root"])
def test_output_under_recording_input_tree_is_refused_without_writes(supported, tmp_path, key):
    tree = supported["root"] / supported["config"][key]
    before = io.snapshot(supported["root"])
    out = tree / "preflight-output"
    with pytest.raises(io.EvidenceError, match="intersects"):
        operator.preflight(out, recording=supported["path"])
    assert not out.exists()
    assert io.snapshot(supported["root"]) == before
    sibling = supported["root"] / "preflight-output"
    assert operator.preflight(sibling, recording=supported["path"])["scope_verdict"] == "fine"


def test_known_failure_html_names_failed_contract_without_satisfaction_claim(tmp_path):
    config = static(tmp_path, "inputs", short=True)
    result = operator.preflight(tmp_path / "preflight", static=config)
    content = operator.render(result["report"])
    assert b"violate named contracts" in content and b"capacity." in content
    assert b"retained facts satisfy" not in content


def test_one_section_evaluation_error_cannot_erase_other_verified_failure(tmp_path, monkeypatch):
    config = static(tmp_path, "inputs", short=True)
    original = operator._section

    def fail_recording(section, receipt, root):
        if section == "recording":
            raise ValueError("section unavailable")
        return original(section, receipt, root)

    monkeypatch.setattr(operator, "_section", fail_recording)
    out = tmp_path / "preflight"
    result = operator.preflight(out, static=config, recording=tmp_path / "missing.json")
    assert result["scope_verdict"] == "not fine"
    assert result["report"]["sections"]["recording"]["scope_verdict"] == "inconclusive"
    assert operator.verify_output(out)["scope_verdict"] == "not fine"


@pytest.mark.parametrize("member", ["report.json", "index.html", "metadata/evaluators.json"])
def test_edited_rehashed_summary_cannot_override_selected_facts(tmp_path, member):
    config = static(tmp_path, "inputs", short=True)
    out = tmp_path / "preflight"
    operator.preflight(out, static=config)
    if member == "report.json":
        value = io.parse_json((out / member).read_bytes())
        value["scope_verdict"] = "fine"
        value["sections"]["static"]["scope_verdict"] = "fine"
        (out / member).write_bytes(io.json_bytes(value))
    else:
        (out / member).write_bytes(b"edited\n")
    rehash(out)
    result = operator.verify_output(out)
    assert result["integrity"] == "failed" and result["scope_verdict"] == "inconclusive"


def test_missing_selected_section_cannot_be_replaced_by_bare_success_flag(tmp_path):
    config = static(tmp_path, "inputs")
    out = tmp_path / "preflight"
    operator.preflight(out, static=config, recording=tmp_path / "missing.json")
    path = out / "receipts/recording.json"
    receipt = io.parse_json(path.read_bytes())
    receipt["scope_verdict"] = "fine"
    path.write_bytes(io.json_bytes(receipt))
    rehash(out)
    assert operator.verify_output(out)["integrity"] == "failed"


def test_substituted_component_profile_cannot_start_unselected_cpu_execution(tmp_path, monkeypatch):
    from sdcguard import evidence_bundle
    config = static(tmp_path, "inputs")
    out = tmp_path / "preflight"
    operator.preflight(out, static=config)
    alternate = tmp_path / "alternate"
    evidence_bundle.create_bundle(alternate, config=config, components=True)
    shutil.rmtree(out / "sections/static")
    shutil.copytree(alternate, out / "sections/static")
    rehash(out)
    monkeypatch.setattr(evidence_bundle, "_collect_components", lambda: pytest.fail("unselected component execution"))
    result = operator.verify_output(out)
    assert result["integrity"] == "failed" and result["scope_verdict"] == "inconclusive"


def test_capture_binding_receipt_must_match_retained_original_config(tmp_path):
    config = static(tmp_path, "inputs")
    out = tmp_path / "preflight"
    operator.preflight(out, static=config)
    path = out / "receipts/static.json"
    receipt = io.parse_json(path.read_bytes())
    receipt["input_sha256"] = "f" * 64
    path.write_bytes(io.json_bytes(receipt))
    rehash(out)
    assert operator.verify_output(out)["integrity"] == "failed"


def test_manifest_count_requires_integer_not_boolean(tmp_path):
    out = tmp_path / "setup"
    operator.export_setup(out)
    path = out / "manifest.json"
    value = io.parse_json(path.read_bytes())
    value["files"]["README.md"]["bytes"] = True
    path.write_bytes(io.json_bytes(value))
    assert operator.verify_output(out)["integrity"] == "failed"


def test_existing_output_refused_before_any_capture_and_files_preserved(tmp_path, monkeypatch):
    out = tmp_path / "keep"
    out.mkdir()
    (out / "keep.txt").write_bytes(b"keep")
    monkeypatch.setattr(operator, "_capture", lambda *a, **k: pytest.fail("existing output must be refused first"))
    with pytest.raises(FileExistsError):
        operator.preflight(out, static=tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        operator.export_setup(out)
    assert io.snapshot(out) == {"keep.txt": b"keep"}


def test_failed_summary_write_retains_explicit_incomplete_output(tmp_path, monkeypatch):
    config = static(tmp_path, "inputs")
    out = tmp_path / "preflight"
    original = io.write_file

    def fail_index(root, name, raw):
        if Path(root) == out and name == "index.html":
            raise OSError("write unavailable")
        return original(root, name, raw)

    monkeypatch.setattr(io, "write_file", fail_index)
    with pytest.raises(io.EvidenceError, match="incomplete"):
        operator.preflight(out, static=config)
    manifest = io.parse_json((out / "manifest.json").read_bytes())
    assert manifest["complete"] is False
    assert operator.verify_output(out)["integrity"] == "failed"


def test_larger_unretained_recording_cannot_erase_retained_static_failure(supported, tmp_path, monkeypatch):
    config = static(tmp_path, "static-inputs", short=True)
    static_only = tmp_path / "static-only"
    recording_only = tmp_path / "recording-only"
    operator.preflight(static_only, static=config)
    operator.preflight(recording_only, recording=supported["path"])
    sizes = [len(io.snapshot(root / "sections" / side))
             for root, side in ((static_only, "static"), (recording_only, "recording"))]
    dirs = [len(io.names(root / "sections" / side, directories=True))
            for root, side in ((static_only, "static"), (recording_only, "recording"))]
    monkeypatch.setattr(io, "MAX_FILES", max(sizes + dirs) + 8)
    out = tmp_path / "combined"
    result = operator.preflight(out, static=config, recording=supported["path"])
    assert result["scope_verdict"] == "not fine", result
    assert result["report"]["sections"]["static"]["integrity"] == "verified"
    assert result["report"]["sections"]["recording"]["receipt"]["retained_kind"] is None
    assert any("aggregate" in item for item in result["report"]["sections"]["recording"]["missing"])
    assert operator.verify_output(out)["scope_verdict"] == "not fine"


def test_readable_report_escapes_input_errors(tmp_path):
    result = operator.preflight(tmp_path / "preflight", static=tmp_path / "missing.json")
    report = deepcopy(result["report"])
    report["sections"]["static"]["missing"] = ["<img src=x onerror=alert(1)>"]
    rendered = operator.render(report)
    assert b"<img src=x" not in rendered and b"&lt;img src=x" in rendered


def test_cli_dispatch_export_verify_and_preflight_exit_codes(tmp_path):
    from sdcguard.cli import main
    out = tmp_path / "setup"
    assert main(["operator", "export", str(out)]) == 0
    assert main(["operator", "verify", str(out)]) == 0
    assert main(["operator", "preflight", "--static", str(out / "static-deployment.example.json"),
                 "--out", str(tmp_path / "unknown")]) == 2
    config = static(tmp_path, "inputs", short=True)
    assert main(["operator", "preflight", "--static", str(config), "--out", str(tmp_path / "failed")]) == 1
    assert main(["operator", "verify", str(tmp_path / "failed")]) == 1
