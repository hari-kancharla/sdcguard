"""Pinned static safeguard inventory, without importing or contacting an engine.

Owner: Hari Krishna Kancharla.
A satisfied rule establishes only its named configuration or source contract.
It does not establish request isolation, build authenticity or runtime execution.
"""
from __future__ import annotations

import math

VLLM = "5df63a7203cdb9328d824a27b682e77c7b9928be"
VLLM_PATCH = "a669acdbc188d59439ca3ba74bba9a18447578b997be3f61278d45fd6a5f337d"
SGLANG = "baa9befa381511cd4d56f83980951e5a45fffc6d"
ASCEND = "399c30d3ae5b1b91e2820e6cd405e1a689a797d1"
ASCEND_BASE = "a98247ab4db686ee03c66d5feb3c761e52a2f8ab"
ASCEND_CANDIDATE = "f3afd042ec1486b5262a0194dec1a7efe07f2239"
ASCEND_CANDIDATE_BASE = "c0b6f74bee518aeaf5d72b200a0216e72fbb5e7c"
VLLM_LMCACHE = "568afb3a13806beb53bb2e6bd518269357b237c0"
LMCACHE = "cd2c0d6a6a982ec5e334bae7704e1029c06d3c97"

_V_PR = "https://github.com/vllm-project/vllm/pull/62065"
_A_PR = "https://github.com/vllm-project/vllm-ascend/pull/19282"
_S = "https://github.com/sgl-project/sglang/blob/" + SGLANG + "/python/sglang/srt/"
_V_CONFIG = ("https://github.com/vadiklyutiy/vllm/blob/"
             "03b2648165d24128611cd15818f4ecf51c0a0205/"
             "vllm/distributed/kv_transfer/kv_connector/v1/nixl/lease_slots.py#L54-L62")
_L_SELECTOR = ("https://github.com/vllm-project/vllm/blob/" + VLLM_LMCACHE
               + "/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py#L93-L112")
_L_ISSUE = "https://github.com/ai-dynamo/dynamo/issues/14416"
_MISSING = object()
_MHA_ARCHITECTURES = {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
                      "MistralForCausalLM"}
_GLM_DSA = {"GlmMoeDsaForCausalLM", "Glm5NextForConditionalGeneration"}


def _dict(value):
    return value if isinstance(value, dict) else {}


def _rule(name, verdict, message, basis, facts=None, sources=()):
    observed = {"runtime_execution": "not established", "identity_basis": "operator supplied",
                "source_status_checked_on": "2026-10-27"}
    observed.update(facts or {})
    return dict(id=name, verdict=verdict, message=message, basis=basis,
                facts=observed, sources=list(sources))


def _unknown(name, message, sources=(), facts=None):
    return _rule(name, "unknown", message, "missing or unreviewed evidence", facts, sources)


def _setting(values, name, default, complete):
    if name in values:
        return values[name], "explicit configuration"
    if complete is True:
        return default, "pinned source default; complete input declared"
    return _MISSING, "input completeness not established"


def _env_bool(value):
    # Exact EnvBool grammar of the pinned SGLang source, without coercing numbers.
    if not isinstance(value, str):
        return _MISSING
    text = value.lower()
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n"):
        return False
    return _MISSING


def _no_extra_patch(components):
    return components.get("runtime_patch_sha256") is None


def _connector_problem(worker):
    connector = _dict(worker.get("connector"))
    allowed = {"name", "transfer_mode", "bidirectional", "extra_config",
               "module_path", "kv_connector_module_path"}
    if set(connector) - allowed:
        return "Unreviewed connector fields cannot inherit a native implementation mapping."
    found_null = False
    for container in (connector, _dict(worker.get("parameters")),
                      _dict(connector.get("extra_config"))):
        for key in ("module_path", "kv_connector_module_path"):
            if key in container:
                if container[key] is not None:
                    return "A custom connector module selector overrides the reviewed native mapping."
                found_null = True
    if not found_null and _dict(worker.get("layout")).get("parameters_complete") is not True:
        return "Connector module selection was not recorded and parameter completeness is unknown."
    return None


def _vllm(worker):
    connector, layout = _dict(worker.get("connector")), _dict(worker.get("layout"))
    components = _dict(worker.get("components"))
    name = connector.get("name")
    sources = [_V_PR, _V_CONFIG]
    if name != "NixlConnector":
        return [_unknown("protection.retirement",
                         "This mapping covers the reviewed NIXL pull path only.", sources)]
    if connector.get("transfer_mode") != "pull" or connector.get("bidirectional") is not False:
        return [_unknown("protection.retirement",
                         "An explicit one-direction pull profile is required; turn-2 reads are not covered.",
                         sources)]
    if worker.get("version") != "0.32.0" or worker.get("build_commit") != VLLM:
        return [_unknown("protection.retirement", "NIXL source profile is unreviewed.", sources)]
    if components.get("vllm_commit", VLLM) != VLLM:
        return [_unknown("protection.retirement", "Contradictory upstream source identity.", sources)]
    extra = _dict(connector.get("extra_config"))
    bidirectional, direction_basis = _setting(extra, "bidirectional_kv_xfer", False,
                                             layout.get("parameters_complete"))
    if type(bidirectional) is not bool or bidirectional is not connector.get("bidirectional"):
        return [_unknown("protection.retirement",
                         "Actual bidirectional_kv_xfer and normalized direction must be known and consistent.",
                         sources)]
    if "runtime_patch_sha256" not in components:
        return [_unknown("protection.retirement", "Runtime overlay identity was not recorded.", sources)]
    patch = components["runtime_patch_sha256"]
    facts = {"source_commit": VLLM, "runtime_patch_sha256": patch, "scope": "first-direction NIXL pull",
             "bidirectional_kv_xfer": bidirectional, "direction_basis": direction_basis}
    if patch is None:
        facts["state"] = "verifier absent from pinned stock source"
        return [_rule("protection.retirement", "violated",
                      "Pinned stock source lacks the required post-read lease verifier. This is a policy gap, not a demonstrated leak.",
                      "pinned stock source", facts, sources)]
    if patch != VLLM_PATCH:
        return [_unknown("protection.retirement", "Runtime overlay is unreviewed.", sources)]
    enabled, basis = _setting(extra, "kv_lease_verification", True,
                              layout.get("parameters_complete"))
    if type(enabled) is not bool:
        return [_unknown("protection.retirement",
                         "Boolean lease-verification setting or complete parameter input is required.",
                         sources, {"source_commit": VLLM, "fix_status": "saved backport of open proposal"})]
    facts.update(state="on" if enabled else "off",
                 fix_status="saved backport of open proposal; not a released upstream fix")
    return [_rule("protection.retirement", "satisfied" if enabled else "violated",
                  "Post-read lease verification is configured on the reviewed source overlay."
                  if enabled else
                  "Post-read lease verification is explicitly disabled. This violates the named policy, not evidence of a leak.",
                  basis, facts, sources)]


def _ascend(worker):
    c, conn = _dict(worker.get("components")), _dict(worker.get("connector"))
    sources = ["https://github.com/vllm-project/vllm-ascend/issues/19281", _A_PR]
    if conn.get("name") not in ("MooncakeConnectorV2", "MooncakePullConnector"):
        return [_unknown("protection.retirement", "Only the reviewed Mooncake V2 pull implementation is mapped.", sources)]
    if conn.get("transfer_mode") not in (None, "pull") or conn.get("bidirectional") is True:
        return [_unknown("protection.retirement", "Requested transport direction is unreviewed.", sources)]
    if worker.get("version") != "source-snapshot" or not _no_extra_patch(c):
        return [_unknown("protection.retirement", "Ascend source snapshot or overlay is unreviewed.", sources)]
    commit = worker.get("build_commit")
    if c.get("vllm_ascend_commit", commit) != commit:
        return [_unknown("protection.retirement", "Contradictory Ascend source identity.", sources)]
    if commit == ASCEND and c.get("vllm_commit") == ASCEND_BASE:
        return [_rule("protection.retirement", "violated",
                      "Pinned Mooncake V2 source lacks the proposed post-read hold verifier. This is a policy gap, not a demonstrated leak.",
                      "pinned plugin and upstream source pair",
                      {"state": "verifier absent", "plugin_commit": commit, "vllm_commit": ASCEND_BASE,
                       "fix_status": "proposal open"}, sources)]
    if commit == ASCEND_CANDIDATE and c.get("vllm_commit") == ASCEND_CANDIDATE_BASE:
        return [_rule("protection.retirement", "satisfied",
                      "Pinned candidate source contains post-read hold verification; both roles require the changed protocol.",
                      "pinned proposed source capability",
                      {"state": "source capability present", "plugin_commit": commit,
                       "vllm_commit": ASCEND_CANDIDATE_BASE, "fix_status": "proposal open",
                       "peer_protocol": "not established by this worker inventory"}, sources)]
    return [_unknown("protection.retirement", "Ascend plugin and upstream source pair is unreviewed.", sources)]


def _sglang(worker, model):
    sources = [_S + "environ.py#L724-L725", _S + "disaggregation/common/conn.py#L308-L311"]
    c = _dict(worker.get("components"))
    if (worker.get("version") != "0.5.22" or worker.get("build_commit") != SGLANG
            or not _no_extra_patch(c)):
        return [_unknown(name, "SGLang source profile or overlay is unreviewed.", sources)
                for name in ("protection.retirement", "protection.deferred_release",
                             "protection.checksum_configured", "protection.canary_configured",
                             "detection.architecture_coverage")]
    conn, params = _dict(worker.get("connector")), _dict(worker.get("parameters"))
    layout, environment = _dict(worker.get("layout")), _dict(worker.get("environment"))
    if (worker.get("role") not in ("prefill", "decode")
            or params.get("disaggregation_mode") != worker.get("role")):
        return [_unknown("protection.retirement",
                         "An explicit disaggregation_mode must match the declared prefill/decode role; the source default is null.",
                         [_S + "arg_groups/fields/disagg.py#L55-L58"])]
    backend = conn.get("name")
    if backend not in ("mooncake", "nixl", "mori"):
        return [_unknown("protection.retirement", "A reviewed PD transfer backend must be identified.", sources)]
    param_backend = params.get("disaggregation_transfer_backend", backend)
    if param_backend != backend:
        return [_unknown("protection.retirement", "Connector and transfer-backend parameters contradict.", sources)]
    results = [_rule("protection.retirement", "satisfied",
                     "Pinned source contains the intake-rejection retirement backstop.",
                     "pinned source capability", {"state": "source capability present", "source_commit": SGLANG},
                     ["https://github.com/sgl-project/sglang/pull/38935"])]
    raw, basis = _setting(environment, "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE",
                          "1", layout.get("environment_complete"))
    enabled = _env_bool(raw)
    raw_timeout, timeout_basis = _setting(
        environment, "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT",
        "30.0", layout.get("environment_complete"))
    try:
        timeout = float(raw_timeout) if isinstance(raw_timeout, str) else None
    except ValueError:
        timeout = None
    timeout_known = timeout is not None and math.isfinite(timeout) and timeout > 0
    release_facts = {"state": "unknown" if enabled is _MISSING else "on" if enabled else "off",
                     "backend": backend, "deferred_timeout_seconds": timeout if timeout_known else None,
                     "deferred_timeout_basis": timeout_basis,
                     "source_default_timeout_seconds": 30.0, "timeout_is_drain_proof": False}
    if enabled is _MISSING or not timeout_known:
        results.append(_unknown("protection.deferred_release",
                                "Deferred-release state and a finite positive timeout must both be recorded or derived from complete input.",
                                sources, release_facts))
    else:
        results.append(_rule("protection.deferred_release", "satisfied" if enabled else "violated",
                             "Deferred decode KV release is configured for this worker; peer configuration is not established."
                             if enabled else
                             "Deferred decode KV release is disabled; the named retention policy is not met.",
                             basis, release_facts, sources))
    if backend in ("nixl", "mori"):
        results.append(_rule("protection.failed_send_drain", "violated",
                             "This exact tag lacks the later failed-send drain hardening for the selected backend. No exposure is demonstrated.",
                             "pinned source lacks named hardening commit",
                             {"backend": backend, "state": "hardening absent",
                              "missing_commit": "6e667c431009dfd70d88ea2dd9f8784bd0787d89"},
                             ["https://github.com/sgl-project/sglang/pull/45973"]))
    value, basis = _setting(params, "disaggregation_enable_kv_checksum", False,
                            layout.get("parameters_complete"))
    checksum_sources = [_S + "arg_groups/fields/disagg.py#L72-L77",
                        _S + "disaggregation/decode.py#L3184-L3211"]
    if type(value) is bool:
        results.append(_rule("protection.checksum_configured", "satisfied",
                             "Optional handoff checksum is configured " + ("on." if value else "off.")
                             + " No request verification is established.",
                             basis, {"state": "on" if value else "off", "both_roles_required": True,
                                     "architecture_coverage": "not established"}, checksum_sources))
    else:
        results.append(_unknown("protection.checksum_configured",
                                "Boolean checksum setting or complete parameter input is required.", checksum_sources))
    mode, basis = _setting(params, "kv_canary", "none", layout.get("parameters_complete"))
    mode = mode.strip().lower() if isinstance(mode, str) else _MISSING
    canary_sources = [_S + "arg_groups/fields/observability.py#L198-L209",
                      _S + "kv_canary/pool_patcher/api.py#L28-L59"]
    if mode not in ("none", "log", "raise"):
        results.append(_unknown("protection.canary_configured",
                                "Canary mode or complete parameter input is not established.", canary_sources))
        results.append(_unknown("detection.architecture_coverage", "Canary configuration is unknown.", canary_sources))
        return results
    real, real_basis = _setting(params, "kv_canary_real_data", "none", layout.get("parameters_complete"))
    real = real.strip().lower() if isinstance(real, str) else _MISSING
    if real not in ("none", "partial", "all"):
        results.append(_unknown("protection.canary_configured", "Real-KV canary mode is unknown or invalid.", canary_sources))
        results.append(_unknown("detection.architecture_coverage", "Canary configuration is incomplete.", canary_sources))
        return results
    results.append(_rule("protection.canary_configured", "satisfied",
                         "Optional canary mode is " + mode + "; real-KV mode is " + real
                         + ". Log mode does not reject requests.",
                         basis, {"state": "off" if mode == "none" else "on", "mode": mode,
                                 "real_kv_mode": real, "real_kv_basis": real_basis}, canary_sources))
    architecture = model.get("architecture")
    architecture = architecture if isinstance(architecture, str) else None
    pool = layout.get("kv_pool_class")
    facts = {"architecture": architecture, "declared_pool": pool, "mode": mode,
             "real_kv_mode": real, "scope": "source attacher availability only"}
    if mode == "none":
        results.append(_unknown("detection.architecture_coverage", "Optional canary is off; no detection coverage is established.",
                                canary_sources, facts))
    elif architecture in _GLM_DSA:
        facts["coverage"] = "no native DSA or MLA pool attacher in this registry"
        results.append(_unknown("detection.architecture_coverage",
                                "The exact GLM architecture uses MLA/DSA; its native pool has no reviewed canary attacher.",
                                canary_sources + [_S + "configs/model_config.py#L191-L210"], facts))
    elif architecture == "DeepseekV4ForCausalLM":
        facts["coverage"] = "partial SWA only; compressed, indexer and state pools uncovered; no real-KV fingerprint"
        results.append(_unknown("detection.architecture_coverage", "DeepSeek V4 canary coverage is partial in this source.",
                                canary_sources + [_S + "kv_canary/pool_patcher/adapters/dsv4.py#L17-L27"], facts))
    elif architecture in _MHA_ARCHITECTURES and pool in ("MHATokenToKVPool", "MHATokenToKVPoolFP4", "SWAKVPool"):
        facts["coverage"] = "source attacher present for declared pool; not a whole-model integrity guarantee"
        results.append(_rule("detection.architecture_coverage", "satisfied",
                             "A source attacher exists for the declared ordinary pool. Full tensor and runtime coverage are not established.",
                             "pinned registry and operator-declared pool", facts, canary_sources))
    else:
        results.append(_unknown("detection.architecture_coverage",
                                "The exact architecture and declared pool combination has not been mapped.", canary_sources, facts))
    return results


def _lmcache(worker):
    c, conn = _dict(worker.get("components")), _dict(worker.get("connector"))
    sources = [_L_ISSUE, _L_SELECTOR]
    if (worker.get("version") != "0.26.0" or worker.get("build_commit") != VLLM_LMCACHE
            or c.get("lmcache_version") != "0.5.2" or c.get("lmcache_commit") != LMCACHE
            or not _no_extra_patch(c)):
        return [_unknown("compatibility.pairing", "Engine and LMCache source pair or overlay is unreviewed.", sources)]
    params, layout = _dict(worker.get("parameters")), _dict(worker.get("layout"))
    extra = _dict(conn.get("extra_config"))
    name = conn.get("name")
    facts = {"vllm_commit": VLLM_LMCACHE, "lmcache_commit": LMCACHE, "connector": name}
    if name == "LMCacheConnectorV1":
        native, basis = _setting(extra, "use_native", False, layout.get("parameters_complete"))
        if type(native) is not bool:
            return [_unknown("compatibility.pairing", "Native/external connector selection is not established.", sources)]
        facts.update(use_native=native, selector_basis=basis)
        if native:
            return [_unknown("compatibility.pairing",
                             "The native adapter is a different integration path; no equivalent compatibility result is established.",
                             sources, facts)]
    elif name != "LMCacheMPConnector":
        return [_unknown("compatibility.pairing", "LMCache connector implementation is unreviewed.", sources)]
    kv_layout = layout.get("kv_layout")
    facts["kv_layout"] = kv_layout
    if kv_layout not in ("packed", "separate"):
        return [_unknown("compatibility.pairing", "Actual packed or separate KV layout was not recorded.", sources, facts)]
    if name == "LMCacheConnectorV1" and kv_layout == "packed":
        facts.update(state="known reported compatibility gap", fix_status="in-process proposal open")
        return [_rule("compatibility.pairing", "violated",
                      "The exact external in-process pair and packed layout match the reported compatibility gap. No local corruption is demonstrated.",
                      "pinned source pair and declared layout", facts,
                      sources + ["https://github.com/LMCache/LMCache/issues/4463",
                                 "https://github.com/LMCache/LMCache/pull/4467"])]
    if name == "LMCacheMPConnector":
        mamba = layout.get("has_mamba_layers")
        if type(mamba) is not bool:
            return [_unknown("compatibility.pairing", "Actual has_mamba_layers state was not recorded.", sources, facts)]
        facts["has_mamba_layers"] = mamba
        subpaged = False
        if kv_layout == "packed" and mamba:
            kind, compression = layout.get("kv_spec_kind"), layout.get("slot_compression")
            rank = layout.get("tensor_rank")
            kernel, logical = layout.get("kernel_block_tokens"), layout.get("scheduler_block_tokens")
            facts.update(kv_spec_kind=kind, slot_compression=compression, tensor_rank=rank,
                         kernel_block_tokens=kernel, scheduler_block_tokens=logical)
            eligible = {"FULL_ATTENTION", "SLIDING_WINDOW", "CHUNKED_LOCAL_ATTENTION", "SINK_FULL_ATTENTION"}
            if (not isinstance(kind, str) or kind not in eligible or compression is not False or type(rank) is not int or rank != 4
                    or type(kernel) is not int or type(logical) is not int
                    or kernel <= 0 or logical <= 0 or kernel > logical or logical % kernel != 0):
                return [_unknown("compatibility.pairing",
                                 "Packed MP group-edit applicability needs the exact uncompressed spec kind, rank and physical/logical paging facts.",
                                 sources + ["https://github.com/LMCache/LMCache/pull/4253"], facts)]
            subpaged = kernel < logical
        if subpaged:
            facts.update(state="known reported group-edit compatibility gap",
                         fix_status="MP group-edit fix merged after this pinned LMCache source")
            return [_rule("compatibility.pairing", "violated",
                          "This exact MP pair and recorded packed Mamba subpaged group lack the later group-edit fix.",
                          "pinned source pair and declared layout", facts,
                          sources + ["https://github.com/LMCache/LMCache/issues/4247",
                                     "https://github.com/LMCache/LMCache/pull/4253"])]
    disk, _ = _setting(params, "local_disk", None, layout.get("parameters_complete"))
    if disk is _MISSING:
        return [_unknown("compatibility.pairing", "Disk-tier configuration completeness was not established.", sources, facts)]
    if disk is not None and not isinstance(disk, str):
        return [_unknown("compatibility.pairing", "Local-disk configuration has an unsupported type.", sources, facts)]
    if disk is not None and disk != "":
        return [_unknown("compatibility.pairing",
                         "The exact pair has an unresolved local-disk report; root cause and fix are not established.",
                         sources + ["https://github.com/LMCache/LMCache/issues/4385"], facts)]
    facts["state"] = "outside the named packed-layout and disk-tier conditions"
    return [_rule("compatibility.pairing", "satisfied",
                  "Recorded connector/layout facts do not match these named reported gaps. This is not general cache correctness evidence.",
                  "bounded compatibility-policy applicability", facts, sources)]


def evaluate(worker, model):
    """Return named rules from supplied JSON facts, without mutating input."""
    worker, model = _dict(worker), _dict(model)
    for key in ("components", "connector", "parameters", "environment", "layout"):
        if key in worker and not isinstance(worker[key], dict):
            return [_unknown("protection.retirement", "Malformed " + key + " facts cannot establish a safeguard.")]
    connector = _dict(worker.get("connector"))
    if "extra_config" in connector and not isinstance(connector["extra_config"], dict):
        return [_unknown("protection.retirement", "Malformed connector extra_config cannot establish a safeguard.")]
    problem = _connector_problem(worker)
    if problem:
        return [_unknown("protection.retirement", problem,
                         ["https://github.com/vllm-project/vllm/blob/" + VLLM_LMCACHE
                          + "/vllm/distributed/kv_transfer/kv_connector/factory.py#L102-L107"])]
    engine = worker.get("engine")
    if engine == "vllm":
        return _vllm(worker)
    if engine == "sglang":
        return _sglang(worker, model)
    if engine == "vllm-ascend":
        return _ascend(worker)
    if engine == "vllm-lmcache":
        return _lmcache(worker)
    return [_unknown("protection.retirement", "Engine protection mapping is unsupported.")]
