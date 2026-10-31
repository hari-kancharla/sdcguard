"""Static geometry and inert saved-log checks only; no serving engine is used."""
from copy import deepcopy
import json

import pytest

from sdcguard import config_capacity as capacity


def worker(engine="vllm"):
    result = dict(engine=engine, version="0.32.0", build_commit=capacity.VLLM,
                  role="prefill", components={}, connector={"name":"NixlConnector", "module_path":None, "extra_config": {}},
                  parameters=dict(max_model_len=512, block_size=16, num_gpu_blocks_override=33),
                  layout=dict(attention="full", cache_groups=1, tp=1, pp=1, dp=1, dcp=1,
                              offload=False, speculative=False, dynamic=False, remote_model_code=False))
    if engine == "sglang":
        result.update(version="0.5.22", build_commit=capacity.SGLANG,
                      parameters=dict(context_length=512, page_size=16, max_total_tokens=512),
                      connector={"name":"mooncake", "module_path":None, "extra_config": {}})
    elif engine == "vllm-ascend":
        result.update(version="source-snapshot", build_commit=capacity.ASCEND,
                      components={"vllm_commit": capacity.ASCEND_BASE},
                      parameters=dict(max_model_len=512, block_size=128, num_gpu_blocks_override=5),
                      connector={"name":"MooncakeConnectorV2", "module_path":None, "extra_config": {}})
    elif engine == "vllm-lmcache":
        result.update(version="0.26.0", build_commit=capacity.VLLM_LMCACHE,
                      components=dict(lmcache_version="0.5.2", lmcache_commit=capacity.LMCACHE),
                      connector={"name":"LMCacheConnectorV1", "module_path":None, "extra_config": {"use_native": False}})
    return result


def startup(item):
    p = item["parameters"]
    if item["engine"] == "sglang":
        return ("[TP0] max_total_num_tokens=512, chunked_prefill_size=128, "
                "max_prefill_tokens=16384, max_running_requests=16, context_len=512, "
                "available_gpu_mem=1.25 GB\n")
    blocks, context, block = (p[k] for k in ("num_gpu_blocks_override", "max_model_len", "block_size"))
    total = blocks * context // ((context + block - 1) // block)
    device = "NPU" if item["engine"] == "vllm-ascend" else "GPU"
    first = f"Overriding num_gpu_blocks=200 with num_gpu_blocks_override={blocks}\n"
    separator = "\n" if item["engine"] == "vllm-lmcache" else ", "
    concurrency = blocks / ((context + block - 1) // block)
    return first + f"{device} KV cache size: {total:,} tokens" + separator + f"Maximum concurrency for {context:,} tokens per request: {concurrency:.2f}x\n"


def verdicts(result):
    assert [r["id"] for r in result] == ["capacity.planned", "capacity.startup"]
    assert all(set(r) == {"id", "verdict", "message", "basis", "facts", "sources"} for r in result)
    json.dumps(result, allow_nan=False)
    return [r["verdict"] for r in result]


@pytest.mark.parametrize("engine", ["vllm", "vllm-ascend", "vllm-lmcache"])
def test_planned_capacity_never_claims_runtime_without_log(engine):
    result = capacity.evaluate(worker(engine), 512)
    assert verdicts(result) == ["satisfied", "unknown"]
    assert "allocation not established" in result[0]["basis"]


@pytest.mark.parametrize("engine", ["vllm", "vllm-ascend", "vllm-lmcache"])
def test_native_log_corroborates_exact_ordinary_geometry(engine):
    item = worker(engine)
    result = capacity.evaluate(item, 512, startup(item))
    assert verdicts(result) == ["satisfied", "satisfied"]
    assert "no runtime or isolation attestation" in result[1]["basis"]
    assert result[1]["sources"]


@pytest.mark.parametrize("blocks,context,expected", [(32,512,"violated"), (33,512,"satisfied"),
    (33,513,"violated"), (34,513,"satisfied"), (1,1,"violated"), (2,1,"satisfied")])
def test_rounding_and_reserved_null_block_boundaries(blocks, context, expected):
    item = worker()
    item["parameters"].update(num_gpu_blocks_override=blocks, max_model_len=context)
    assert verdicts(capacity.evaluate(item, context, startup(item))) == [expected, expected]


def test_advertised_context_is_not_multiplied_by_concurrency_or_required_equal_engine_context():
    item = worker()
    item["parameters"].update(max_model_len=1024, max_num_seqs=1000, num_gpu_blocks_override=65)
    assert verdicts(capacity.evaluate(item, 512, startup(item))) == ["satisfied", "satisfied"]


def test_advertised_context_above_engine_limit_is_a_declared_contract_violation():
    item = worker()
    item["parameters"]["num_gpu_blocks_override"] = 200
    assert verdicts(capacity.evaluate(item, 513, startup(item))) == ["violated", "violated"]


@pytest.mark.parametrize("engine", ["vllm", "sglang", "vllm-ascend", "vllm-lmcache"])
@pytest.mark.parametrize("field", ["version", "build_commit"])
def test_version_string_or_unreviewed_build_never_passes(engine, field):
    item = worker(engine); item[field] = "unverified"
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown", "unknown"]


@pytest.mark.parametrize("key,value", [("attention","sliding"),("attention","hybrid"),
    ("cache_groups",2),("cache_groups",True),("tp",2),("pp",2),("dp",2),("dcp",2),
    ("offload",True),("speculative",True),("dynamic",True),("remote_model_code",True)])
def test_unsupported_geometry_is_unknown_not_false_alarm(key, value):
    item = worker();item["layout"][key] = value
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown", "unknown"]


@pytest.mark.parametrize("key", list(worker()["layout"]))
def test_every_geometry_declaration_is_required(key):
    item = worker();del item["layout"][key]
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown", "unknown"]


@pytest.mark.parametrize("key", ["sliding_window", "attention_chunk_size", "kv_offloading_size"])
def test_conflicting_engine_special_pool_parameter_is_unknown(key):
    item = worker();item["parameters"][key] = 64
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown", "unknown"]


@pytest.mark.parametrize("value", [None,True,False,0,-1,1.5,"512","auto",float("nan"),float("inf"),2**31])
def test_invalid_advertised_length_is_serializable_unknown(value):
    assert verdicts(capacity.evaluate(worker(),value)) == ["unknown", "unknown"]


@pytest.mark.parametrize("key", ["max_model_len", "block_size", "num_gpu_blocks_override"])
@pytest.mark.parametrize("value", [None,True,0,-1,"auto",16.0])
def test_unknown_or_coerced_parameters_do_not_pass(key,value):
    item=worker();item["parameters"][key]=value
    assert verdicts(capacity.evaluate(item,512)) == ["unknown", "unknown"]


def test_memory_fraction_and_byte_budget_are_not_token_capacity():
    item=worker();item["parameters"]={"max_model_len":512,"block_size":16,
        "gpu_memory_utilization":0.99,"kv_cache_memory_bytes":10**12}
    assert verdicts(capacity.evaluate(item,512)) == ["unknown", "unknown"]


@pytest.mark.parametrize("cap,expected", [(511,"violated"),(512,"unknown"),(100000,"unknown"),(None,"unknown")])
def test_sglang_configured_cap_only_proves_an_upper_bound(cap,expected):
    item=worker("sglang");item["parameters"]["max_total_tokens"]=cap
    assert verdicts(capacity.evaluate(item,512)) == [expected,"unknown"]


def test_sglang_page_floor_can_make_configured_cap_insufficient():
    item=worker("sglang");item["parameters"].update(context_length=513,max_total_tokens=527)
    assert verdicts(capacity.evaluate(item,513)) == ["violated","unknown"]


def test_sglang_actual_pool_supports_bounded_capacity_without_universal_admission_claim():
    item=worker("sglang")
    result=capacity.evaluate(item,512,startup(item))
    assert verdicts(result) == ["unknown","satisfied"]
    assert result[1]["facts"]["startup_pool_tokens"] == 512
    assert result[0]["facts"]["state"] == "upper_bound_only"
    assert result[0]["facts"]["resolution_rule"] == "capacity.startup"


def test_sglang_actual_pool_resolves_automatic_sizing_without_a_config_only_pass():
    item=worker("sglang");item["parameters"].pop("max_total_tokens")
    result=capacity.evaluate(item,512,startup(item))
    assert verdicts(result) == ["unknown","satisfied"]
    assert result[0]["facts"]["state"] == "automatic_pool"


def test_sglang_actual_pool_smaller_than_high_configured_cap_is_violation():
    item=worker("sglang");item["parameters"]["max_total_tokens"]=4096
    text=startup(item).replace("max_total_num_tokens=512","max_total_num_tokens=496")
    assert verdicts(capacity.evaluate(item,512,text)) == ["unknown","violated"]


@pytest.mark.parametrize("change", [("context_len=512","context_len=1024"),
    ("max_total_num_tokens=512","max_total_num_tokens=513"),
    ("max_total_num_tokens=512","max_total_num_tokens=528")])
def test_sglang_conflicting_or_unaligned_log_does_not_alarm_or_pass(change):
    item=worker("sglang")
    assert verdicts(capacity.evaluate(item,512,startup(item).replace(*change))) == ["unknown","unknown"]


@pytest.mark.parametrize("engine", ["vllm","sglang","vllm-ascend","vllm-lmcache"])
def test_repeated_or_mixed_startups_are_ambiguous(engine):
    item=worker(engine);text=startup(item)
    assert capacity.evaluate(item,512,text+text)[1]["verdict"] == "unknown"


@pytest.mark.parametrize("change", [("override=33","override=34"),
    ("for 512 tokens","for 1024 tokens"),("528 tokens,","512 tokens,"),("GPU KV","NPU KV")])
def test_vllm_conflicting_logs_return_unknown_for_both_rules(change):
    item=worker()
    assert verdicts(capacity.evaluate(item,512,startup(item).replace(*change))) == ["unknown","unknown"]


def test_banner_or_rounded_concurrency_alone_cannot_prove_capacity():
    item=worker();text=startup(item).split("\n",1)[1]
    assert verdicts(capacity.evaluate(item,512,text)) == ["satisfied","unknown"]


def test_rounded_concurrency_contradiction_is_unknown_and_never_inverted():
    item=worker();text=startup(item).replace("1.03x","2.00x")
    assert verdicts(capacity.evaluate(item,512,text)) == ["unknown","unknown"]


@pytest.mark.parametrize("engine", ["vllm","sglang"])
def test_duplicate_facts_on_one_line_are_not_silently_ignored(engine):
    item=worker(engine);text=startup(item)
    if engine=="sglang":
        text=text.rstrip()+" "+text
    else:
        first,rest=text.split("\n",1);text=first+" "+first+"\n"+rest
    assert capacity.evaluate(item,512,text)[1]["verdict"] == "unknown"


@pytest.mark.parametrize("engine", ["vllm","sglang"])
def test_unbounded_integer_log_field_is_refused_without_conversion(engine):
    item=worker(engine);text=startup(item)
    text=text.replace("override=33","override="+"9"*5000) if engine=="vllm" else text.replace("max_total_num_tokens=512","max_total_num_tokens="+"9"*5000)
    assert capacity.evaluate(item,512,text)[1]["verdict"] == "unknown"


def test_ansi_prefix_is_allowed_without_changing_the_receipts():
    item=worker();text="\x1b[32m"+startup(item)+"\x1b[0m"
    assert verdicts(capacity.evaluate(item,512,text)) == ["satisfied","satisfied"]


def test_known_runtime_patch_preserves_only_capacity_mapping():
    item=worker();item["components"]["runtime_patch_sha256"]=capacity.REVIEWED_RUNTIME_PATCH
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["satisfied","satisfied"]
    item["components"]["runtime_patch_sha256"]="a"*64
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


def test_lmcache_storage_and_transport_bytes_are_never_added_to_gpu_pool():
    item=worker("vllm-lmcache");item["parameters"].update(num_gpu_blocks_override=32,
        max_local_cpu_size=1000,pd_buffer_size=10**12)
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["violated","violated"]


@pytest.mark.parametrize("value", [True,None,"false",0])
def test_lmcache_native_or_unidentified_adapter_does_not_inherit_external_mapping(value):
    item=worker("vllm-lmcache");item["connector"]["extra_config"]["use_native"]=value
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


def test_ascend_base_and_plugin_are_both_required():
    item=worker("vllm-ascend");item["components"]["vllm_commit"]=capacity.VLLM
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


def test_evaluation_does_not_mutate_operator_inputs():
    item=worker();before=deepcopy(item)
    capacity.evaluate(item,512,startup(item))
    assert item==before


def test_capacity_path_imports_no_engine_or_accelerator(monkeypatch):
    import builtins
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        assert name.split(".")[0] not in {"torch","vllm","sglang","lmcache","modal"}
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    for engine in ("vllm","sglang","vllm-ascend","vllm-lmcache"):
        item=worker(engine);capacity.evaluate(item,512,startup(item))


@pytest.mark.parametrize("location", ["connector","parameters","extra_config"])
@pytest.mark.parametrize("key", ["module_path","kv_connector_module_path"])
def test_custom_module_selector_never_inherits_native_capacity(location,key):
    item=worker()
    target=item["connector"]["extra_config"] if location=="extra_config" else item[location]
    target[key]="custom.path"
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


def test_unknown_connector_selector_field_is_unknown():
    item=worker();item["connector"]["module"]="custom.path"
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


def test_unrecorded_module_default_requires_complete_parameters():
    item=worker();item["connector"].pop("module_path")
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]
    item["layout"]["parameters_complete"]=True
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["satisfied","satisfied"]


@pytest.mark.parametrize("name", [None,[],{},"custom", "MooncakeConnector", "NixlConnector2"])
def test_unmapped_connector_names_do_not_inherit_vllm_profile(name):
    item=worker();item["connector"]["name"]=name
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


@pytest.mark.parametrize("key", ["tensor_parallel_size","pipeline_parallel_size", "data_parallel_size",
    "decode_context_parallel_size","prefill_context_parallel_size"])
@pytest.mark.parametrize("value", [2,True,"1"])
def test_actual_vllm_parallel_fields_cannot_contradict_declared_single_rank(key,value):
    item=worker();item["parameters"][key]=value
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


@pytest.mark.parametrize("key", ["tp_size","pp_size","dp_size","dcp_size","attn_cp_size",
    "attn_dp_size","attn_tp_size","attn_dcp_size","moe_dp_size"])
def test_actual_sglang_parallel_fields_cannot_contradict_declared_single_rank(key):
    item=worker("sglang");item["parameters"][key]=2
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


@pytest.mark.parametrize("key", ["enable_hisparse","enable_hierarchical_cache","enable_unified_memory",
    "enable_prefill_cp","enable_kv_cache_sharding","enable_dp_attention","enable_pdmux",
    "disaggregation_decode_enable_offload_kvcache"])
@pytest.mark.parametrize("value", [True,"false",None,0])
def test_actual_sglang_special_pool_fields_cannot_hide_in_normalized_layout(key,value):
    item=worker("sglang");item["parameters"][key]=value
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


@pytest.mark.parametrize("engine,key,value", [("vllm","speculative_config",{}),
    ("sglang","speculative_algorithm","EAGLE"),("sglang","dllm_algorithm","x"),
    ("sglang","hybrid_gdn_config",{}),("vllm","hisparse_config",{})])
def test_actual_speculative_or_special_state_configuration_is_unknown(engine,key,value):
    item=worker(engine);item["parameters"][key]=value
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


def test_explicit_matching_actual_engine_fields_remain_supported():
    item=worker();item["parameters"].update(tensor_parallel_size=1,pipeline_parallel_size=1,
        data_parallel_size=1,decode_context_parallel_size=1,speculative_config=None,num_gpu_blocks=33)
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["satisfied","satisfied"]
    item=worker("sglang");item["parameters"].update(tp_size=1,pp_size=1,dp_size=1,dcp_size=1,
        enable_hisparse=False,enable_unified_memory=False,speculative_algorithm=None)
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","satisfied"]


def test_resolved_block_count_disagreement_is_unknown():
    item=worker();item["parameters"]["num_gpu_blocks"]=34
    assert verdicts(capacity.evaluate(item,512,startup(item))) == ["unknown","unknown"]


@pytest.mark.parametrize("engine", ["vllm", "sglang", "vllm-lmcache", "vllm-ascend"])
def test_blank_overlay_identity_is_unknown(engine):
    item = worker(engine)
    item["components"]["runtime_patch_sha256"] = ""
    assert verdicts(capacity.evaluate(item, 512)) == ["unknown", "unknown"]


@pytest.mark.parametrize("engine", ["vllm", "vllm-lmcache"])
def test_contradictory_component_base_identity_is_unknown(engine):
    item = worker(engine)
    item["components"]["vllm_commit"] = "f" * 40
    assert verdicts(capacity.evaluate(item, 512)) == ["unknown", "unknown"]
