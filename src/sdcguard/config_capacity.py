"""Pinned static capacity contracts for supplied configuration and startup text.

The geometry is explicitly declared by the operator. It is not model, build,
allocation or isolation attestation. No engine code is imported or executed.
Only a single ordinary full-attention device pool with all parallel sizes one
is mapped. Planned capacity and saved startup evidence are separate rules.
"""
from __future__ import annotations

import re

VLLM = "5df63a7203cdb9328d824a27b682e77c7b9928be"
SGLANG = "baa9befa381511cd4d56f83980951e5a45fffc6d"
VLLM_LMCACHE = "568afb3a13806beb53bb2e6bd518269357b237c0"
LMCACHE = "cd2c0d6a6a982ec5e334bae7704e1029c06d3c97"
ASCEND = "399c30d3ae5b1b91e2820e6cd405e1a689a797d1"
ASCEND_BASE = "a98247ab4db686ee03c66d5feb3c761e52a2f8ab"
REVIEWED_RUNTIME_PATCH = "a669acdbc188d59439ca3ba74bba9a18447578b997be3f61278d45fd6a5f337d"

_REQUIRED_LAYOUT = dict(attention="full", cache_groups=1, tp=1, pp=1, dp=1,
                        dcp=1, offload=False, speculative=False, dynamic=False,
                        remote_model_code=False)
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_COUNT = r"(?:[1-9]\d{0,19}|[1-9]\d{0,2}(?:,\d{3}){1,6})"
_OVERRIDE = re.compile(r"Overriding num_gpu_blocks=(\d{1,10}) with num_gpu_blocks_override=(\d{1,10})\s*$")
_BANNER = re.compile(r"(GPU|NPU) KV cache size: (" + _COUNT + r") tokens(?:, Maximum concurrency for (" + _COUNT + r") tokens per request: (\d{1,20}\.\d{2})x)?\s*$")
_CONCURRENCY = re.compile(r"Maximum concurrency for (" + _COUNT + r") tokens per request: (\d{1,20}\.\d{2})x\s*$")
_SG = re.compile(r"max_total_num_tokens=(\d{1,10}), chunked_prefill_size=(?:None|-?\d{1,10}), max_prefill_tokens=\d{1,10}, max_running_requests=\d{1,10}, context_len=(\d{1,10}), available_(?:gpu|cpu)_mem=\d{1,10}(?:\.\d{1,10})? GB\s*$")
_CONNECTOR_KEYS = {"name", "transfer_mode", "bidirectional", "extra_config",
                   "module_path", "kv_connector_module_path"}
_CONNECTOR_NAMES = {"vllm": {"NixlConnector"}, "sglang": {"mooncake", "nixl"},
                    "vllm-ascend": {"MooncakeConnectorV2", "MooncakePullConnector"},
                    "vllm-lmcache": {"LMCacheConnectorV1"}}


def _connector_known(worker):
    connector = worker.get("connector")
    if not isinstance(connector, dict) or set(connector) - _CONNECTOR_KEYS:
        return False
    engine = worker.get("engine")
    name = connector.get("name")
    if not isinstance(engine, str) or not isinstance(name, str) or name not in _CONNECTOR_NAMES.get(engine, set()):
        return False
    params, layout = worker.get("parameters"), worker.get("layout")
    extra = connector.get("extra_config", {})
    if not isinstance(params, dict) or not isinstance(layout, dict) or not isinstance(extra, dict):
        return False
    selectors = ("module_path", "kv_connector_module_path")
    if not (layout.get("parameters_complete") is True
            or any(key in connector and connector[key] is None for key in selectors)):
        return False
    for container in (connector, params, extra):
        if any(container.get(key) is not None for key in selectors):
            return False
    return True


def supported_pin(worker):
    """Whether the supplied identity names one exact reviewed source profile."""
    if not isinstance(worker, dict):
        return False
    if not _connector_known(worker):
        return False
    c = worker.get("components", {})
    if not isinstance(c, dict):
        return False
    engine, version, commit = (worker.get(k) for k in ("engine", "version", "build_commit"))
    patch = c.get("runtime_patch_sha256")
    if patch is not None and not (engine == "vllm" and patch == REVIEWED_RUNTIME_PATCH):
        return False
    if engine == "vllm":
        return version == "0.32.0" and commit == VLLM and c.get("vllm_commit", VLLM) == VLLM
    if engine == "sglang":
        return version == "0.5.22" and commit == SGLANG
    if engine == "vllm-lmcache":
        connector = worker.get("connector", {})
        extra = connector.get("extra_config", {}) if isinstance(connector, dict) else None
        return (version == "0.26.0" and commit == VLLM_LMCACHE
                and c.get("vllm_commit", VLLM_LMCACHE) == VLLM_LMCACHE
                and c.get("lmcache_version") == "0.5.2" and c.get("lmcache_commit") == LMCACHE
                and isinstance(extra, dict) and extra.get("use_native") is False)
    if engine == "vllm-ascend":
        return (version == "source-snapshot" and commit == ASCEND
                and c.get("vllm_commit") == ASCEND_BASE
                and c.get("vllm_ascend_commit", ASCEND) == ASCEND)
    return False


def _sources(engine):
    if engine not in ("vllm", "sglang", "vllm-ascend", "vllm-lmcache"):
        return []
    if engine == "sglang":
        base = "https://github.com/sgl-project/sglang/blob/" + SGLANG + "/python/sglang/srt/"
        return [base + "mem_cache/kv_cache_configurator.py#L2771-L2798",
                base + "model_executor/pool_configurator.py#L614-L630",
                base + "managers/scheduler.py#L1228-L1240",
                base + "managers/tp_worker.py#L543-L568"]
    pin = ASCEND_BASE if engine == "vllm-ascend" else VLLM_LMCACHE if engine == "vllm-lmcache" else VLLM
    base = "https://github.com/vllm-project/vllm/blob/" + pin + "/vllm/v1/"
    result = [base + "core/block_pool.py", base + "kv_cache_interface.py",
              base + "core/kv_cache_utils.py"]
    if engine == "vllm-ascend":
        result += ["https://github.com/vllm-project/vllm-ascend/blob/" + ASCEND + "/vllm_ascend/utils.py#L1423-L1480",
                   "https://github.com/vllm-project/vllm-ascend/blob/" + ASCEND + "/vllm_ascend/patch/platform/patch_kv_cache_utils.py#L632-L665"]
    if engine == "vllm-lmcache":
        result.append("https://github.com/LMCache/LMCache/blob/" + LMCACHE + "/lmcache/v1/config.py#L228-L301")
    return result


def _rule(name, verdict, message, basis, facts, sources):
    return dict(id="capacity." + name, verdict=verdict, message=message,
                basis=basis, facts=dict(facts), sources=list(sources))


def _positive(value):
    return type(value) is int and 0 < value <= 2**31 - 1


def _layout_reason(layout):
    if not isinstance(layout, dict):
        return "Missing explicit ordinary full-attention layout declarations."
    for key, expected in _REQUIRED_LAYOUT.items():
        if key not in layout or type(layout[key]) is not type(expected) or layout[key] != expected:
            return "Unsupported or missing layout declaration: " + key + "."
    return None


def _parameter_reason(engine, params):
    """Contradictory actual engine fields cannot hide behind declared geometry."""
    if engine == "sglang":
        parallel = ("tp_size", "pp_size", "dp_size", "dcp_size", "attn_cp_size",
                    "attn_dp_size", "attn_tp_size", "attn_dcp_size", "moe_dp_size")
        disabled = ("enable_hisparse", "enable_hierarchical_cache", "enable_unified_memory",
                    "enable_prefill_cp", "enable_kv_cache_sharding", "enable_dp_attention",
                    "enable_pdmux", "disaggregation_decode_enable_offload_kvcache")
        empty = ("speculative_algorithm", "dllm_algorithm", "hybrid_gdn_config",
                 "mambaish_config", "hybrid_kda_config")
    else:
        parallel = ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                    "decode_context_parallel_size", "prefill_context_parallel_size")
        disabled = ()
        empty = ("speculative_config", "hisparse_config")
    for key in parallel:
        if key in params and (type(params[key]) is not int or params[key] != 1):
            return "Actual engine parallel setting is outside the declared one-rank layout: " + key + "."
    for key in disabled:
        if key in params and params[key] is not False:
            return "Actual engine setting is outside the ordinary device pool: " + key + "."
    for key in empty:
        if params.get(key) is not None:
            return "Actual engine setting selects an unmapped state or speculative pool: " + key + "."
    for key in ("sliding_window", "attention_chunk_size", "kv_offloading_size",
                "cpu_offload_gb", "offload_group_size"):
        value = params.get(key)
        if value is not None and not (type(value) in (int, float) and value == 0):
            return "Parameter conflicts with the ordinary full-attention pool: " + key + "."
    return None


def _startup_vllm(text, engine, context, block_size, blocks):
    """Return (facts, conflict, missing). Never infer blocks from rounded logs."""
    override_lines = [line for line in text.splitlines() if "Overriding num_gpu_blocks=" in line]
    banner_lines = [line for line in text.splitlines() if "KV cache size:" in line]
    if len(override_lines) != 1 or len(banner_lines) != 1:
        return {}, False, "One unambiguous override receipt and one capacity banner are required."
    override = _OVERRIDE.search(override_lines[0])
    banner = _BANNER.search(banner_lines[0])
    if (not override or not banner or override_lines[0].count("Overriding num_gpu_blocks=") != 1
            or banner_lines[0].count("KV cache size:") != 1):
        return {}, False, "Saved startup capacity messages do not match this pinned format."
    recorded_blocks = int(override.group(2))
    facts = dict(startup_override_blocks=recorded_blocks,
                 logged_equivalent_capacity_tokens=int(banner.group(2).replace(",", "")))
    if recorded_blocks != blocks:
        return facts, True, "Startup override disagrees with the supplied configuration."
    expected_device = "NPU" if engine == "vllm-ascend" else "GPU"
    if banner.group(1) != expected_device:
        return facts, True, "Startup device label disagrees with the selected source profile."
    concurrency_lines = [line for line in text.splitlines() if "Maximum concurrency for " in line]
    if len(concurrency_lines) != 1:
        return facts, False, "One unambiguous resolved-context startup message is required."
    match = _CONCURRENCY.search(concurrency_lines[0])
    if not match:
        return facts, False, "Saved resolved-context message is malformed."
    logged_context = int(match.group(1).replace(",", ""))
    facts["startup_context_tokens"] = logged_context
    if logged_context != context:
        return facts, True, "Startup context disagrees with the supplied configuration."
    if engine != "vllm-lmcache" and banner.group(3) is None:
        return facts, False, "This pinned source uses a combined capacity and context message."
    if engine == "vllm-lmcache" and banner.group(3) is not None:
        return facts, False, "This pinned source uses separate capacity and context messages."
    # This corroborates exact N from the override, it does not invert a float.
    # A disagreement is inconclusive, including a source floating-rounding edge.
    required = (context + block_size - 1) // block_size
    if facts["logged_equivalent_capacity_tokens"] != blocks * context // required:
        return facts, True, "Startup equivalent capacity disagrees with the declared ordinary pool geometry."
    facts["logged_rounded_max_concurrency"] = match.group(2)
    if match.group(2) != f"{blocks / required:.2f}":
        return facts, True, "Startup rounded concurrency contradicts the exact declared pool geometry."
    return facts, False, None


def evaluate(worker, advertised_context_tokens, startup_text=None):
    """Return exactly planned/startup capacity rules, with honest evidence scope."""
    worker = worker if isinstance(worker, dict) else {}
    engine = worker.get("engine")
    sources = _sources(engine)
    facts = {"engine": engine if isinstance(engine, str) else None,
             "advertised_context_tokens": advertised_context_tokens if _positive(advertised_context_tokens) else None,
             "identity_basis": "operator supplied", "geometry_basis": "operator declared"}

    def unknown(message, basis="missing or unsupported facts"):
        return [_rule(name, "unknown", message, basis, facts, sources)
                for name in ("planned", "startup")]

    if not supported_pin(worker):
        return unknown("Engine, version, component or patch identity is not a reviewed capacity profile.")
    reason = _layout_reason(worker.get("layout"))
    if reason:
        return unknown(reason)
    params = worker.get("parameters")
    if not isinstance(params, dict) or not _positive(advertised_context_tokens):
        return unknown("Positive integer advertised context and engine parameters are required.")
    reason = _parameter_reason(engine, params)
    if reason:
        return unknown(reason)
    context_key, page_key = ("context_length", "page_size") if engine == "sglang" else ("max_model_len", "block_size")
    context, page = params.get(context_key), params.get(page_key)
    if not _positive(context) or not _positive(page):
        return unknown("Explicit positive resolved context and block/page size are required; auto is unknown.")
    facts.update(configured_context_tokens=context, block_or_page_size_tokens=page)
    if engine == "vllm-ascend" and page != 128:
        return unknown("Initial ordinary Ascend capacity support requires resolved block size 128.")
    if engine == "sglang":
        cap = params.get("max_total_tokens")
        if cap is not None and not _positive(cap):
            return unknown("Configured token ceiling must be a positive integer or absent.")
        if cap is not None:
            facts["configured_pool_upper_bound_tokens"] = cap // page * page
        planned_verdict = "violated" if cap is not None and cap // page * page < advertised_context_tokens else "unknown"
        planned_message = ("Configured pool ceiling cannot hold the advertised context."
                           if planned_verdict == "violated" else "Configured ceiling or memory fraction does not establish actual pool capacity.")
        if planned_verdict == "unknown":
            facts.update(state="upper_bound_only" if cap is not None else "automatic_pool",
                         resolution_rule="capacity.startup")
    else:
        blocks = params.get("num_gpu_blocks_override")
        if not _positive(blocks):
            return unknown("An exact positive num_gpu_blocks_override is required; automatic sizing is unknown.")
        if "num_gpu_blocks" in params and (type(params["num_gpu_blocks"]) is not int or params["num_gpu_blocks"] != blocks):
            return unknown("Resolved num_gpu_blocks conflicts with the exact configured override.")
        facts.update(configured_total_blocks=blocks, reserved_null_blocks=1,
                     usable_capacity_tokens=(blocks - 1) * page,
                     required_request_blocks=(advertised_context_tokens + page - 1) // page,
                     required_engine_context_blocks=(context + page - 1) // page)
        planned_verdict = "satisfied" if blocks - 1 >= facts["required_request_blocks"] else "violated"
        planned_message = ("Planned ordinary pool can hold one advertised-context request."
                           if planned_verdict == "satisfied" else "Planned ordinary pool cannot hold one advertised-context request after null-block reservation.")
        if blocks - 1 < facts["required_engine_context_blocks"]:
            planned_verdict, planned_message = "violated", "Planned pool cannot meet the pinned engine's configured-context startup requirement."
    if advertised_context_tokens > context:
        planned_verdict, planned_message = "violated", "Advertised context exceeds the configured engine context."
    planned = _rule("planned", planned_verdict, planned_message,
                    "declared configuration; successful allocation not established", facts, sources)
    if not isinstance(startup_text, str) or not startup_text.strip():
        return [planned, _rule("startup", "unknown", "Saved startup evidence was not supplied.",
                               "missing startup evidence", facts, sources)]
    text = _ANSI.sub("", startup_text)
    if engine == "sglang":
        lines = [line for line in text.splitlines() if "max_total_num_tokens=" in line]
        if len(lines) != 1 or lines[0].count("max_total_num_tokens=") != 1 or not (match := _SG.search(lines[0])):
            return [planned, _rule("startup", "unknown", "One exact, unambiguous pinned SGLang startup summary is required.",
                                   "missing or ambiguous startup facts", facts, sources)]
        capacity, logged_context = int(match.group(1)), int(match.group(2))
        facts.update(startup_pool_tokens=capacity, startup_context_tokens=logged_context)
        if (not _positive(capacity) or capacity % page or logged_context != context
                or (cap is not None and capacity > cap // page * page)):
            return unknown("Startup capacity or context conflicts with the supplied configuration.", "conflicting startup/config facts")
        verdict = "satisfied" if capacity >= advertised_context_tokens and context >= advertised_context_tokens else "violated"
        message = "Saved ordinary startup pool is consistent with the advertised context." if verdict == "satisfied" else "Saved ordinary startup pool or engine context is below the advertised context."
    else:
        observed, conflict, missing = _startup_vllm(text, engine, context, page, blocks)
        facts.update(observed)
        if conflict:
            return unknown(missing, "conflicting startup/config facts")
        if missing:
            return [planned, _rule("startup", "unknown", missing, "missing or ambiguous startup facts", facts, sources)]
        verdict, message = planned_verdict, "Saved planning receipts corroborate configured capacity; completed physical allocation is not established."
    return [planned, _rule("startup", verdict, message,
                           ("saved finalized scheduler capacity; no runtime or isolation attestation" if engine == "sglang" else
                            "saved planning receipts; completed physical allocation not established; no runtime or isolation attestation"), facts, sources)]
