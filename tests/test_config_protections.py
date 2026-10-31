"""Static policy cases use only small declared configurations, without engines."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sdcguard import config_protections as p


MODEL = {"architecture": "LlamaForCausalLM"}


def worker(engine):
    value = {"engine": engine, "role": "decode", "parameters": {}, "environment": {},
             "layout": {"parameters_complete": True, "environment_complete": True,
                        "kv_pool_class": "MHATokenToKVPool"},
             "components": {}, "connector": {}}
    if engine == "vllm":
        value.update(version="0.32.0", build_commit=p.VLLM)
        value["connector"] = {"name": "NixlConnector", "transfer_mode": "pull",
                              "bidirectional": False, "extra_config": {"bidirectional_kv_xfer": False}}
        value["components"]["runtime_patch_sha256"] = p.VLLM_PATCH
    elif engine == "sglang":
        value.update(version="0.5.22", build_commit=p.SGLANG)
        value["connector"]["name"] = "mooncake"
        value["parameters"].update(disaggregation_mode="decode", kv_canary="log", kv_canary_real_data="all")
    elif engine == "vllm-ascend":
        value.update(version="source-snapshot", build_commit=p.ASCEND)
        value["components"]["vllm_commit"] = p.ASCEND_BASE
        value["connector"]["name"] = "MooncakeConnectorV2"
    elif engine == "vllm-lmcache":
        value.update(version="0.26.0", build_commit=p.VLLM_LMCACHE)
        value["components"].update(lmcache_version="0.5.2", lmcache_commit=p.LMCACHE)
        value["connector"] = {"name": "LMCacheConnectorV1", "extra_config": {"use_native": False}}
        value["layout"]["kv_layout"] = "separate"
    value["connector"]["module_path"] = None
    return value


def rules(value, model=MODEL):
    result = p.evaluate(value, model)
    assert len({r["id"] for r in result}) == len(result)
    assert all(r["verdict"] in ("satisfied", "violated", "unknown", "not_applicable") for r in result)
    assert all(r["facts"]["runtime_execution"] == "not established" for r in result)
    assert all(url.startswith("https://") for r in result for url in r["sources"])
    return {r["id"]: r for r in result}


@pytest.mark.parametrize("engine", ["vllm", "sglang", "vllm-ascend", "vllm-lmcache"])
def test_unknown_build_never_becomes_known_bad_or_good(engine):
    value = worker(engine)
    value["build_commit"] = "f" * 40
    assert {r["verdict"] for r in rules(value).values()} == {"unknown"}


@pytest.mark.parametrize("engine", ["vllm", "sglang", "vllm-ascend", "vllm-lmcache"])
def test_matching_version_without_build_is_inconclusive(engine):
    value = worker(engine)
    del value["build_commit"]
    assert {r["verdict"] for r in rules(value).values()} == {"unknown"}


def test_vllm_stock_and_saved_overlay_are_distinct_named_policy_results():
    fixed = worker("vllm")
    rule = rules(fixed)["protection.retirement"]
    assert rule["verdict"] == "satisfied"
    assert rule["facts"]["state"] == "on"
    assert "open proposal" in rule["facts"]["fix_status"]
    stock = copy.deepcopy(fixed)
    stock["components"]["runtime_patch_sha256"] = None
    rule = rules(stock)["protection.retirement"]
    assert rule["verdict"] == "violated"
    assert "not a demonstrated leak" in rule["message"]


@pytest.mark.parametrize("overlay", ["missing", "b" * 64, "", " ", "not-a-digest"])
def test_unrecorded_or_unreviewed_overlay_is_unknown(overlay):
    value = worker("vllm")
    if overlay == "missing":
        value["components"].pop("runtime_patch_sha256")
    else:
        value["components"]["runtime_patch_sha256"] = overlay
    assert rules(value)["protection.retirement"]["verdict"] == "unknown"


@pytest.mark.parametrize("field,changed", [("name", "NixlPushConnector"), ("name", "OtherConnector"),
                                         ("transfer_mode", "push"), ("bidirectional", True),
                                         ("bidirectional", None)])
def test_vllm_does_not_apply_pull_policy_to_unreviewed_paths(field, changed):
    value = worker("vllm")
    value["connector"][field] = changed
    assert rules(value)["protection.retirement"]["verdict"] == "unknown"


def test_complete_input_required_for_default_and_explicit_false_is_not_erased():
    value = worker("vllm")
    value["layout"]["parameters_complete"] = False
    assert rules(value)["protection.retirement"]["verdict"] == "unknown"
    value["connector"]["extra_config"]["kv_lease_verification"] = False
    r = rules(value)["protection.retirement"]
    assert r["verdict"] == "violated" and r["basis"] == "explicit configuration"
    value["connector"]["extra_config"]["kv_lease_verification"] = "false"
    assert rules(value)["protection.retirement"]["verdict"] == "unknown"


def test_sglang_optional_checks_off_are_inventory_facts_without_false_violation():
    value = worker("sglang")
    value["parameters"] = {"disaggregation_mode": "decode"}
    r = rules(value)
    assert not any(x["verdict"] == "violated" for x in r.values())
    for name in ("protection.checksum_configured", "protection.canary_configured"):
        assert r[name]["verdict"] == "satisfied"
        assert r[name]["facts"]["state"] == "off"
        assert r[name]["basis"].startswith("pinned source default")
    assert r["detection.architecture_coverage"]["verdict"] == "unknown"


@pytest.mark.parametrize("raw,state", [("0", "off"), ("false", "off"), ("N", "off"),
                                     ("1", "on"), ("TRUE", "on"), ("yes", "on")])
def test_sglang_env_boolean_source_grammar(raw, state):
    value = worker("sglang")
    value["environment"]["SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE"] = raw
    r = rules(value)["protection.deferred_release"]
    assert r["facts"]["state"] == state
    assert r["verdict"] == ("violated" if state == "off" else "satisfied")
    assert r["facts"]["timeout_is_drain_proof"] is False


@pytest.mark.parametrize("raw", [False, 0, " false ", "disabled"])
def test_invalid_environment_does_not_coerce_to_a_default(raw):
    value = worker("sglang")
    value["environment"]["SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE"] = raw
    assert rules(value)["protection.deferred_release"]["verdict"] == "unknown"


def test_missing_environment_and_parameter_facts_do_not_supply_defaults():
    value = worker("sglang")
    value["layout"].update(parameters_complete=False, environment_complete=False)
    value["parameters"] = {"disaggregation_mode": "decode"}
    r = rules(value)
    assert r["protection.retirement"]["verdict"] == "satisfied"
    assert all(r[k]["verdict"] == "unknown" for k in
               ("protection.deferred_release", "protection.checksum_configured",
                "protection.canary_configured", "detection.architecture_coverage"))


@pytest.mark.parametrize("backend", ["nixl", "mori"])
def test_sglang_022_is_not_a_blanket_safe_cutoff(backend):
    value = worker("sglang")
    value["connector"]["name"] = backend
    r = rules(value)
    assert r["protection.retirement"]["verdict"] == "satisfied"
    assert r["protection.failed_send_drain"]["verdict"] == "violated"
    assert r["protection.failed_send_drain"]["facts"]["missing_commit"].startswith("6e667c")


@pytest.mark.parametrize("architecture", ["GlmMoeDsaForCausalLM", "Glm5NextForConditionalGeneration",
                                         "DeepseekV4ForCausalLM"])
def test_exact_architecture_gaps_never_inherit_ordinary_pool_pass(architecture):
    value = worker("sglang")
    r = rules(value, {"architecture": architecture})["detection.architecture_coverage"]
    assert r["verdict"] == "unknown"
    assert r["facts"]["architecture"] == architecture
    if architecture == "DeepseekV4ForCausalLM":
        assert "partial" in r["message"].lower()


@pytest.mark.parametrize("architecture", ["GLM-5.3", "DeepSeek-V4", "UnknownForCausalLM", None, []])
def test_model_brands_and_unmapped_architecture_do_not_pass(architecture):
    assert rules(worker("sglang"), {"architecture": architecture})["detection.architecture_coverage"]["verdict"] == "unknown"


def test_known_canary_attacher_is_only_a_source_capability_result():
    r = rules(worker("sglang"))["detection.architecture_coverage"]
    assert r["verdict"] == "satisfied"
    assert r["facts"]["scope"] == "source attacher availability only"
    assert "runtime coverage are not established" in r["message"]


def test_ascend_pair_connector_and_proposed_source_are_distinct():
    value = worker("vllm-ascend")
    assert rules(value)["protection.retirement"]["verdict"] == "violated"
    value["build_commit"] = p.ASCEND_CANDIDATE
    value["components"]["vllm_commit"] = p.ASCEND_CANDIDATE_BASE
    r = rules(value)["protection.retirement"]
    assert r["verdict"] == "satisfied"
    assert r["facts"]["fix_status"] == "proposal open"
    value["connector"]["name"] = "MooncakeConnectorV1"
    assert rules(value)["protection.retirement"]["verdict"] == "unknown"


def test_ascend_wrong_upstream_pair_does_not_inherit_candidate_capability():
    value = worker("vllm-ascend")
    value["build_commit"] = p.ASCEND_CANDIDATE
    assert rules(value)["protection.retirement"]["verdict"] == "unknown"


def test_lmcache_native_selection_is_not_a_fix_claim():
    value = worker("vllm-lmcache")
    value["connector"]["extra_config"]["use_native"] = True
    r = rules(value)["compatibility.pairing"]
    assert r["verdict"] == "unknown" and r["facts"]["use_native"] is True


def test_lmcache_default_adapter_needs_complete_parameters():
    value = worker("vllm-lmcache")
    value["connector"]["extra_config"] = {}
    value["layout"]["parameters_complete"] = False
    assert rules(value)["compatibility.pairing"]["verdict"] == "unknown"
    value["layout"]["parameters_complete"] = True
    assert rules(value)["compatibility.pairing"]["verdict"] == "satisfied"


def test_lmcache_same_pair_distinguishes_packed_separate_and_unknown_layout():
    value = worker("vllm-lmcache")
    assert rules(value)["compatibility.pairing"]["verdict"] == "satisfied"
    value["layout"]["kv_layout"] = "packed"
    r = rules(value)["compatibility.pairing"]
    assert r["verdict"] == "violated"
    assert "proposal open" in r["facts"]["fix_status"]
    value["layout"].pop("kv_layout")
    assert rules(value)["compatibility.pairing"]["verdict"] == "unknown"


def test_mp_and_inprocess_fixes_are_not_conflated():
    value = worker("vllm-lmcache")
    value["connector"] = {"name": "LMCacheMPConnector", "module_path": None}
    value["layout"].update(kv_layout="packed", hybrid=True, has_mamba_layers=True,
                           kv_spec_kind="FULL_ATTENTION", slot_compression=False,
                           tensor_rank=4, kernel_block_tokens=32, scheduler_block_tokens=544)
    r = rules(value)["compatibility.pairing"]
    assert r["verdict"] == "violated"
    assert any("/4253" in source for source in r["sources"])
    assert not any("/4467" in source for source in r["sources"])
    value["layout"]["has_mamba_layers"] = False
    assert rules(value)["compatibility.pairing"]["verdict"] == "satisfied"
    value["layout"].pop("has_mamba_layers")
    assert rules(value)["compatibility.pairing"]["verdict"] == "unknown"


def test_unresolved_disk_report_is_unknown_not_claimed_fixed_or_demonstrated():
    value = worker("vllm-lmcache")
    value["parameters"]["local_disk"] = "file:///operator/path"
    r = rules(value)["compatibility.pairing"]
    assert r["verdict"] == "unknown"
    assert "root cause and fix are not established" in r["message"]


@pytest.mark.parametrize("field", ["components", "connector", "parameters", "environment", "layout"])
def test_malformed_fact_container_does_not_become_a_default(field):
    value = worker("sglang")
    value[field] = []
    assert {r["verdict"] for r in rules(value).values()} == {"unknown"}


@pytest.mark.parametrize("engine", ["vllm", "sglang", "vllm-ascend", "vllm-lmcache"])
def test_input_is_retained_exactly_and_result_is_json_serializable(engine):
    value, model = worker(engine), copy.deepcopy(MODEL)
    before = copy.deepcopy((value, model))
    result = p.evaluate(value, model)
    json.dumps(result, allow_nan=False)
    assert (value, model) == before


def test_standalone_module_import_does_not_load_engine_or_network_modules():
    path = Path(p.__file__)
    script = """
import importlib.abc, importlib.util, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'sglang', 'lmcache', 'socket'}:
            raise RuntimeError('unexpected engine or network import')
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('static_protections', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.evaluate({}, {})[0]['verdict'] == 'unknown'
"""
    result = subprocess.run([sys.executable, "-S", "-B", "-c", script, str(path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("engine", ["vllm", "sglang", "vllm-ascend", "vllm-lmcache"])
@pytest.mark.parametrize("location,key", [("connector", "module_path"),
                                         ("connector", "kv_connector_module_path"),
                                         ("parameters", "kv_connector_module_path"),
                                         ("extra_config", "kv_connector_module_path")])
def test_custom_connector_overrides_never_inherit_native_policy(engine, location, key):
    value = worker(engine)
    target = (value["connector"].setdefault("extra_config", {})
              if location == "extra_config" else value[location])
    target[key] = "operator.custom_connector"
    assert {r["verdict"] for r in rules(value).values()} == {"unknown"}


def test_unknown_connector_fields_and_unrecorded_module_selector_are_unknown():
    value = worker("vllm")
    value["connector"]["implementation_class"] = "other"
    assert {r["verdict"] for r in rules(value).values()} == {"unknown"}
    value["connector"].pop("implementation_class")
    value["connector"].pop("module_path")
    value["layout"]["parameters_complete"] = False
    value["connector"]["extra_config"]["kv_lease_verification"] = True
    assert {r["verdict"] for r in rules(value).values()} == {"unknown"}
    value["connector"]["module_path"] = None
    assert rules(value)["protection.retirement"]["verdict"] == "satisfied"


def test_effective_deferred_timeout_override_is_retained_with_its_basis():
    value = worker("sglang")
    key = "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT"
    value["environment"][key] = "90.5"
    r = rules(value)["protection.deferred_release"]
    assert r["verdict"] == "satisfied"
    assert r["facts"]["deferred_timeout_seconds"] == 90.5
    assert r["facts"]["deferred_timeout_basis"] == "explicit configuration"
    assert r["facts"]["source_default_timeout_seconds"] == 30.0


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "broken", 30, True])
def test_invalid_or_nonpositive_timeout_cannot_be_a_known_good_hold(raw):
    value = worker("sglang")
    value["environment"]["SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT"] = raw
    r = rules(value)["protection.deferred_release"]
    assert r["verdict"] == "unknown"
    assert r["facts"]["deferred_timeout_seconds"] is None


@pytest.mark.parametrize("actual", [True, "false", None, 0, []])
def test_actual_bidirectional_selector_cannot_contradict_normalized_scope(actual):
    value = worker("vllm")
    value["connector"]["extra_config"]["bidirectional_kv_xfer"] = actual
    assert rules(value)["protection.retirement"]["verdict"] == "unknown"


def test_absent_actual_direction_requires_complete_parameters_or_explicit_false():
    value = worker("vllm")
    value["connector"]["extra_config"].pop("bidirectional_kv_xfer")
    value["layout"]["parameters_complete"] = False
    value["connector"]["extra_config"]["kv_lease_verification"] = True
    assert rules(value)["protection.retirement"]["verdict"] == "unknown"
    value["layout"]["parameters_complete"] = True
    r = rules(value)["protection.retirement"]
    assert r["verdict"] == "satisfied"
    assert r["facts"]["bidirectional_kv_xfer"] is False
    assert r["facts"]["direction_basis"].startswith("pinned source default")


@pytest.mark.parametrize("actual", ["null", "prefill", None, True, "missing"])
def test_sglang_actual_mode_cannot_contradict_declared_decode_role(actual):
    value = worker("sglang")
    if actual == "missing":
        value["parameters"].pop("disaggregation_mode")
    else:
        value["parameters"]["disaggregation_mode"] = actual
    assert {r["verdict"] for r in rules(value).values()} == {"unknown"}


def mp_subpaged():
    value = worker("vllm-lmcache")
    value["connector"] = {"name": "LMCacheMPConnector", "module_path": None}
    value["layout"].update(kv_layout="packed", has_mamba_layers=True,
                           kv_spec_kind="FULL_ATTENTION", slot_compression=False,
                           tensor_rank=4, kernel_block_tokens=32, scheduler_block_tokens=544)
    return value


def test_packed_hybrid_is_insufficient_and_equal_paging_is_not_a_false_alarm():
    value = mp_subpaged()
    value["layout"]["hybrid"] = True
    value["layout"].pop("has_mamba_layers")
    assert rules(value)["compatibility.pairing"]["verdict"] == "unknown"
    value["layout"]["has_mamba_layers"] = True
    value["layout"]["kernel_block_tokens"] = 544
    r = rules(value)["compatibility.pairing"]
    assert r["verdict"] == "satisfied"
    assert r["facts"]["kernel_block_tokens"] == r["facts"]["scheduler_block_tokens"]
    value["layout"]["has_mamba_layers"] = False
    value["layout"]["kernel_block_tokens"] = 32
    assert rules(value)["compatibility.pairing"]["verdict"] == "satisfied"


@pytest.mark.parametrize("field,value", [("kv_spec_kind", "MLA"), ("kv_spec_kind", []), ("slot_compression", True),
                                        ("tensor_rank", 5), ("tensor_rank", True),
                                        ("kernel_block_tokens", 0), ("kernel_block_tokens", 60),
                                        ("kernel_block_tokens", 1024), ("scheduler_block_tokens", None)])
def test_mp_structural_prerequisites_cannot_be_guessed(field, value):
    config = mp_subpaged()
    config["layout"][field] = value
    assert rules(config)["compatibility.pairing"]["verdict"] == "unknown"


@pytest.mark.parametrize("engine", ["vllm", "sglang", "vllm-ascend", "vllm-lmcache"])
def test_blank_overlay_digest_is_missing_evidence_not_a_stock_declaration(engine):
    config = worker(engine)
    config["components"]["runtime_patch_sha256"] = ""
    assert {r["verdict"] for r in rules(config).values()} == {"unknown"}
