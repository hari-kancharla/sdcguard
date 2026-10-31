"""Offline report and command acceptance checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from sdcguard import config_safety


def document():
    return {"schema": "sdcguard.config-safety.v1", "deployment_id": "offline-example",
            "advertised_context_tokens": 1024,
            "model": {"id": "example/model", "revision": "a" * 40, "architecture": "LlamaForCausalLM"},
            "workers": [{"id": role, "role": role, "engine": "unmapped-engine", "version": "1.0",
                         "build_commit": "b" * 40, "parameters": {}, "layout": {}}
                        for role in ("prefill", "decode")]}


def test_unsupported_engine_never_fine(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document()))
    assert config_safety.main([str(path), "--out", str(tmp_path / "report")]) == 2
    saved = json.loads((tmp_path / "report/report.json").read_text())
    assert saved["status"] == "inconclusive"
    assert all(r["verdict"] in ("unknown", "not_applicable")
               for w in saved["workers"] for r in w["rules"])
    assert "does not establish request isolation" in saved["scope"]
    assert saved["model"]["revision"] == "a" * 40
    assert all(w["config_sha256"] for w in saved["workers"])
    assert saved["evaluator_sha256"]["config_capacity.py"]
    assert saved["provenance"]["raw_config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_identity_invalidates_even_positive_or_negative_claims(tmp_path, monkeypatch):
    doc = document()
    del doc["model"]["revision"]
    monkeypatch.setattr(config_safety.config_capacity, "evaluate", lambda *args: [
        dict(id="capacity.planned", verdict="satisfied", message="yes", facts={}, basis="input", sources=[]),
        dict(id="capacity.startup", verdict="violated", message="no", facts={}, basis="input", sources=[])])
    report = config_safety.assess(doc, {}, tmp_path)
    assert report["status"] == "inconclusive"
    assert all(r["verdict"] == "unknown" for w in report["workers"] for r in w["rules"])


def test_missing_role_and_advertisement_cannot_fine(tmp_path):
    doc = document()
    del doc["advertised_context_tokens"]
    doc["workers"] = doc["workers"][:1]
    report = config_safety.assess(doc, {}, tmp_path)
    assert {r["id"] for r in report["deployment_rules"]} == {"deployment.roles", "deployment.context", "deployment.peer_profile"}
    assert report["status"] == "inconclusive"


def test_render_escapes_input_and_has_portable_json_link(tmp_path):
    doc = document()
    doc["workers"][0]["id"] = '<img src=x onerror="alert(1)">'
    report = config_safety.assess(doc, {}, tmp_path)
    rendered = config_safety.render(report)
    assert '<img src=x' not in rendered
    assert '&lt;img src=x' in rendered
    assert 'href="report.json"' in rendered
    assert 'No server was started.' in rendered


def test_existing_output_is_preserved(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document()))
    out = tmp_path / "report"
    out.mkdir()
    marker = out / "report.json"
    marker.write_text("preserve")
    assert config_safety.main([str(path), "--out", str(out)]) == 2
    assert marker.read_text() == "preserve"


def test_no_engine_torch_network_or_subprocess_in_static_command(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document()))
    source = Path(__file__).parents[1] / "src"
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','vllm','sglang','lmcache','modal','socket','subprocess'}:
            raise RuntimeError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Block())
from sdcguard.config_safety import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run([sys.executable, "-S", "-c", script, str(path), "--json"],
                            env=dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["status"] == "inconclusive"


def test_top_level_cli_dispatch_and_module_command_match(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document()))
    from sdcguard.cli import main
    assert main(["config-safety", str(path)]) == 2


def test_mixed_engine_peers_do_not_inherit_protocol_compatibility(tmp_path):
    doc = document()
    doc["workers"][1]["engine"] = "another-engine"
    report = config_safety.assess(doc, {}, tmp_path)
    assert next(r for r in report["deployment_rules"] if r["id"] == "deployment.peer_profile")["verdict"] == "unknown"


def test_sglang_peer_checksum_and_admission_policy_disagreement():
    doc = document()
    for w in doc["workers"]:
        w.update(engine="sglang", version="0.5.22", build_commit=config_safety.config_capacity.SGLANG,
                 connector={"name": "mooncake", "module_path": None},
                 layout={"parameters_complete": True})
        w["parameters"]["disaggregation_mode"] = w["role"]
    doc["workers"][0]["parameters"]["disaggregation_enable_kv_checksum"] = True
    doc["workers"][1]["parameters"]["disaggregation_decode_allocation_policy"] = "prefill_complete"
    rules = config_safety._peer_rules(doc)
    assert next(r for r in rules if "checksum" in r["id"])["verdict"] == "violated"
    assert next(r for r in rules if "allocation_policy" in r["id"])["verdict"] == "violated"


def test_unsupported_model_geometry_cannot_claim_capacity(tmp_path):
    doc = document()
    doc["model"]["architecture"] = "GlmMoeDsaForCausalLM"
    report = config_safety.assess(doc, {}, tmp_path)
    assert all(r["verdict"] == "unknown" for w in report["workers"] for r in w["rules"]
               if r["id"].startswith("capacity."))


def test_custom_sglang_connector_cannot_inherit_peer_policy_failure(tmp_path):
    doc = document()
    for w in doc["workers"]:
        w.update(engine="sglang", version="0.5.22", build_commit=config_safety.config_capacity.SGLANG,
                 connector={"name": "mooncake", "module_path": "custom.module"},
                 layout={"parameters_complete": True})
        w["parameters"]["disaggregation_mode"] = w["role"]
    doc["workers"][0]["parameters"]["disaggregation_enable_kv_checksum"] = True
    report = config_safety.assess(doc, {}, tmp_path)
    assert report["status"] == "inconclusive"
    assert all(r["verdict"] == "unknown" for r in report["deployment_rules"])


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_complete_bound_static_contract_can_be_fine_without_serving(engine, tmp_path):
    from sdcguard.config_inputs import binding_digest
    from sdcguard.config_protections import VLLM_PATCH
    doc = document()
    doc["advertised_context_tokens"] = 512
    for w in doc["workers"]:
        w.update(engine=engine, version="0.32.0" if engine == "vllm" else "0.5.22",
                 build_commit=config_safety.config_capacity.VLLM if engine == "vllm" else config_safety.config_capacity.SGLANG,
                 layout=dict(attention="full", cache_groups=1, tp=1, pp=1, dp=1, dcp=1,
                             offload=False, speculative=False, dynamic=False, remote_model_code=False,
                             parameters_complete=True, environment_complete=True, kv_pool_class="MHATokenToKVPool"))
        if engine == "vllm":
            w.update(components={"runtime_patch_sha256": VLLM_PATCH},
                     connector=dict(name="NixlConnector", module_path=None, transfer_mode="pull", bidirectional=False,
                                    extra_config={"bidirectional_kv_xfer": False}),
                     parameters=dict(max_model_len=512, block_size=16, num_gpu_blocks_override=33))
            saved = ("Overriding num_gpu_blocks=200 with num_gpu_blocks_override=33\n"
                     "GPU KV cache size: 528 tokens, Maximum concurrency for 512 tokens per request: 1.03x\n")
        else:
            w.update(components={}, connector=dict(name="mooncake", module_path=None),
                     parameters=dict(context_length=512, page_size=16, max_total_tokens=1024,
                                     disaggregation_mode=w["role"], kv_canary="log", kv_canary_real_data="all"))
            saved = ("max_total_num_tokens=512, chunked_prefill_size=128, max_prefill_tokens=16384, "
                     "max_running_requests=16, context_len=512, available_gpu_mem=1.25 GB\n")
        log = tmp_path / (w["id"] + ".log")
        log.write_text(saved)
        w["startup_log"] = dict(path=log.name, sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                                config_sha256=binding_digest(w, doc), worker_id=w["id"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(doc))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    out = tmp_path / "result"
    assert config_safety.main([str(path), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "fine"
    assert report["input"] == doc
    assert all((tmp_path / name).read_bytes() == raw for name, raw in before.items())
    if engine == "sglang":
        for w in report["workers"]:
            planned = next(r for r in w["rules"] if r["id"] == "capacity.planned")
            assert planned["verdict"] == "unknown"
            assert planned["required_for_status"] is False and planned["resolved_by"] == "capacity.startup"


def test_unmapped_sglang_paths_keep_every_expected_rule_unknown(tmp_path):
    doc = document()
    for w in doc["workers"]:
        w.update(engine="sglang", version="0.5.22", build_commit=config_safety.config_capacity.SGLANG,
                 connector={"name": "custom-backend"}, parameters={"disaggregation_mode": w["role"]})
    report = config_safety.assess(doc, {}, tmp_path)
    expected = set(config_safety._PROTECTION_RULES["sglang"])
    for w in report["workers"]:
        actual = {r["id"]: r for r in w["rules"]}
        assert expected <= set(actual)
        assert all(actual[name]["verdict"] == "unknown" for name in expected)
