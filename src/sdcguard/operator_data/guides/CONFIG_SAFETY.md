# Static deployment contracts

Owner: Hari Krishna Kancharla.

`sdcguard config-safety` reads a normalized deployment JSON and optional saved startup logs. It does not import an engine, start a server, call an endpoint, allocate a device or download a model. It writes a compact HTML page and a detailed JSON report, including the supplied config and saved log text. Keep the packet local until its contents have been reviewed for sharing. This check concerns configuration contracts. It does not establish request isolation.

```sh
sdcguard config-safety deployment.json --out new-report
```

Exit 0 means every required applicable contract was satisfied. A planned SGLang pool may remain unknown when its sufficient recorded startup pool resolves that uncertainty; that row is explicitly marked resolved by startup capacity. All other required unknowns block exit 0. Exit 1 means at least one named contract was violated. Exit 2 means no violation was established but a required fact or source mapping was missing. A malformed input or unreadable output also exits 2 and explains the error. A violation of the declared full-context capacity or protection policy is not a claim that a request leaked. Existing reports are never overwritten.

Use the examples in `examples/config-safety/` as input shapes. They are explicitly synthetic arithmetic examples, not measurements or deployment recipes. Copy the actual normalized settings from both roles. Export the effective settings, not only the flags changed from a default. Do not mark a parameter or environment set complete unless all relevant settings were captured.

## Inputs and identity

The root schema is `sdcguard.config-safety.v1`. It contains a deployment ID, an advertised endpoint context in tokens, a model ID, an immutable model revision, its exact architecture ID, and a list of prefill/decode workers. Each worker has an ID, role, engine, version, build commit, component pins, connector, parameters, environment and layout. Both roles are needed for a complete report. Workers may have different capacities and valid parallel settings. The first capacity mappings cover one rank and one homogeneous full-attention cache group for explicitly declared ordinary Llama, Qwen2 and Qwen3 geometry; unsupported architectures and topologies stay unknown.

This release reads the normalized JSON wrapper, not arbitrary YAML, shell commands or Kubernetes manifests. Environment values retain the engine's string form. Connector `module_path: null` explicitly records built-in selection; a custom module stays unknown.

The engine names are `vllm`, `sglang`, `vllm-ascend` and `vllm-lmcache`. The current inspected source profiles are pinned below. A version label alone is insufficient. The report retains full component pins and a hash of each worker config, the model identity and endpoint context. Operator-supplied source identities are declarations, not authenticated build attestations. Custom modules, unknown patches and unreviewed releases cannot inherit a protection claim from their version string.

`parameters` uses the engine's normalized underscore names. Supply positive integer counts, not shell expressions or human-readable aliases. Memory fractions and byte budgets are retained as input but never treated as observed token pools. `layout` explicitly identifies attention, cache groups, parallel ranks and alternate storage/allocation paths. Missing layout facts stay unknown. Other parameter fields are retained for provenance; only the fields named by a rule are evaluated. This is not a complete engine configuration validator. An architecture string or layout declaration does not verify the model weights or runtime pool.

## Parameters checked

| Profile | Capacity inputs | Protection and compatibility inputs |
| --- | --- | --- |
| vLLM 0.32.0 | `max_model_len`, `block_size`, `num_gpu_blocks_override`; saved exact allocation override log | `NixlConnector`, pull mode, bidirectional state, `extra_config.kv_lease_verification`, exact runtime patch hash |
| SGLang 0.5.22 | `context_length`, `page_size`, `max_total_tokens`; saved `max_total_num_tokens` and `context_len` startup facts | transfer backend, `SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE`, its timeout, `disaggregation_enable_kv_checksum`, `kv_canary`, `kv_canary_real_data`, actual declared pool class |
| vLLM-Ascend source snapshot | exact plugin/base pair, resolved block size and ordinary cache geometry | Mooncake V2 connector and post-read hold-verification source capability |
| vLLM 0.26.0 + LMCache 0.5.2 | vLLM device pool only; LMCache chunk/storage settings do not add device capacity | exact LMCache commit, connector branch, `extra_config.use_native`, packed/hybrid layout |

The vLLM simple capacity contract uses `ceil(context / block_size)` request blocks plus the permanently reserved null block. A configured override is a planned allocation, not proof startup succeeded. Rounded token or concurrency banners alone cannot establish the physical pool.

SGLang `max_total_tokens` is a ceiling on a memory-profiled pool. A page-aligned ceiling below the advertised context cannot meet the named full-context contract. A high ceiling cannot prove sufficient actual capacity. A matching saved startup capacity can establish that bounded startup fact. Request admission has separate headroom and can safely reject long requests. A smaller operational pool is not, by itself, evidence of leakage.

Sliding-window, hybrid/recurrent, compressed, speculative, offloaded, dynamically resized and context-parallel pools need separate mappings. They stay unknown for capacity. No rule requires equal P/D pool sizes or multiplies context by maximum concurrency.

## Saved startup logs

A worker's optional `startup_log` object supplies `path`, `sha256`, `config_sha256` and optionally `worker_id`. The path is local and must be relative to the deployment JSON; absolute paths and traversal are unsupported. The SHA256 covers the exact UTF-8 log bytes. The config binding covers that worker excluding `startup_log`, plus the root model identity and advertised endpoint context. Use the package helper to compute it:

```python
from sdcguard.config_inputs import binding_digest
worker["startup_log"]["config_sha256"] = binding_digest(worker, deployment)
```

This binds already saved facts consistently. It does not authenticate who collected them or prove the process used that configuration. Record logs at collection time; do not re-label an old log to fit a new build. Missing, conflicting, unreadable, oversized or wrongly bound logs cannot support a satisfied startup contract. Do not combine startup sessions or roles in one file. The reader rejects symlink inputs, duplicate JSON keys and non-finite JSON numbers.

## Protection inventory and limits

Known on/off settings are inventory facts. Optional checksum or canary being off is not automatically an engine defect. A separate named retirement policy requires the mapped hold/lease protection. Its gap is a policy violation, not proof that the deployment exercised the defect.

The vLLM #62065 proposal remains open at the recorded research date. The supported saved overlay is a backport of that proposal. A missing overlay declaration stays unknown. A configured enabled verifier does not prove it checked a particular request.

There is no universal SGLang safe-version threshold in this report. The 0.5.22 tag has default deferred release, but later backend-specific hardening is separate. Canary `log` mode records findings while serving continues; `raise` has different behavior. The native DSA pool used by GLM-5.3 lacks a canary adapter at this pin. The DeepSeek V4 adapter covers only its SWA subpool, with no real-KV fingerprint. Architecture coverage stays unknown for partial or unsupported paths. Detection settings never certify full-model coverage.

Dynamo #14416 describes connector/layout-specific reports for the vLLM/LMCache pair. The external in-process packed layout and the multiprocess packed hybrid path have separate issues and fixes. Native selection, unknown layout, custom modules or unknown backports remain unknown. A LocalDisk report is not assigned an established root cause here.

## Sources and support pins

Research checked October 27, 2026. These are inspected mappings, not a claim that all four engines were run locally.

| Profile | Exact source |
| --- | --- |
| vLLM 0.32.0 | `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Saved vLLM runtime overlay | SHA256 `a669acdbc188d59439ca3ba74bba9a18447578b997be3f61278d45fd6a5f337d` |
| SGLang 0.5.22 | `baa9befa381511cd4d56f83980951e5a45fffc6d` |
| Ascend inspected snapshot / upstream vLLM | `399c30d3ae5b1b91e2820e6cd405e1a689a797d1` / `a98247ab4db686ee03c66d5feb3c761e52a2f8ab` |
| vLLM 0.26.0 / LMCache 0.5.2 | `568afb3a13806beb53bb2e6bd518269357b237c0` / `cd2c0d6a6a982ec5e334bae7704e1029c06d3c97` |

- [Cloudflare incident and remediation](https://blog.cloudflare.com/kv-cache-exposure-remediation/), [SGLang incident analysis](https://www.lmsys.org/blog/2026-10-26-cf-security/).
- [vLLM block reserve](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py#L184-L188), [existing startup validator](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L2605-L2647).
- [SGLang pool cap](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/mem_cache/kv_cache_configurator.py#L2771-L2798), [startup capacity log](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/managers/scheduler.py#L1230-L1238).
- [vLLM #61229](https://github.com/vllm-project/vllm/issues/61229), [proposed #62065 fix](https://github.com/vllm-project/vllm/pull/62065).
- [SGLang deferred-release defaults](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/environ.py#L724-L725), [canary pool registry](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/kv_canary/pool_patcher/api.py#L28-L34), [DeepSeek V4 scope](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/kv_canary/pool_patcher/adapters/dsv4.py#L17-L27).
- [Ascend #19281](https://github.com/vllm-project/vllm-ascend/issues/19281), [proposed #19282 fix](https://github.com/vllm-project/vllm-ascend/pull/19282).
- [Dynamo #14416](https://github.com/ai-dynamo/dynamo/issues/14416), [vLLM LMCache adapter selection](https://github.com/vllm-project/vllm/blob/568afb3a13806beb53bb2e6bd518269357b237c0/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py#L93-L112).

Capacity validators and cross-engine cache tooling already exist upstream. This command contributes one independent, portable contract report with pinned mappings. It does not claim to invent the arithmetic, replace vendor safeguards or prove production isolation. The named rule results are intended to feed the upgrade comparison gate; a new source profile must be reviewed before it can inherit a result.
