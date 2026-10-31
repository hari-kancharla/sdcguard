# Allocator-to-model mapping

Owner: Hari Krishna Kancharla.

This experimental offline rule asks one question: does a request's positive, current allocator assignment match the attention block table recorded for that request at model entry? The rule is `allocator_to_model_entry_mapping_consistency`. It is a mapping relation, not a complete ownership proof.

The passive native adapter is implemented for one pinned profile. Its [activation and closure guide](ASSIGNMENT_NATIVE.md) describes the observed call chain and unsupported paths. CPU examples and source checks do not establish that a real engine emitted these records. Native GPU validation remains pending. The older saved #61229 and H100 recordings lack the necessary initialization and model-entry facts. Their existing lifetime results remain separate.

## Commands

For native recordings, use the dedicated mapping directory containing completed JSONL sessions and their paired `.closure.json` files. The command requires successful settled-close bookkeeping, exact source and profile association, matching raw byte hashes and actual event counts. A failed or missing receipt produces unknown and no packet. It cannot be replaced with a healthy-looking footer.

```sh
sdcguard assignment capture /path/completed-native-mapping --out /new/native-mapping-packet
sdcguard assignment verify /new/native-mapping-packet
```

The native packet retains the exact journals and separate close receipts. Offline verification repeats their bounds, identities and accounting checks before evaluating the mapping relation. Its `recording_bookkeeping` is `locally consistent`; this is not runtime authentication, client attribution or ownership proof. A locally consistent native packet can still contain a violated or unknown mapping relation.

Raw-format inspection accepts a dedicated directory of completed JSONL files without close receipts. Its `recording_bookkeeping` stays `not checked`. Use it for offline format inputs, not as proof that the native recorder closed successfully.

```sh
sdcguard assignment inspect /path/completed-mapping-records --out /new/mapping-packet
sdcguard assignment verify /new/mapping-packet
```

All output directories must be new, have an existing parent and be separate from their input tree. Both packet forms retain all selected input bytes and write `report.json`, `index.html`, evaluator hashes and a manifest. Their distinct manifest schemas keep raw inspection separate from native capture. Verification checks the exact inventory and hashes, then recomputes both reports from the retained inputs with the installed checker. Resealing an edited report cannot change the rederived result. Local hashes establish consistency, not remote authenticity.

| Exit | Named mapping rule |
| --- | --- |
| 0 | Satisfied for every supplied session |
| 1 | A complete recorded interval establishes a mapping contradiction |
| 2 | Unknown, incomplete, unsupported or invalid |

A verified packet can contain an unknown or violated result. File integrity and rule verdict are separate. A missing unrelated session does not erase a complete contradiction in an independent session. Duplicate session identities are unknown. The command never joins identities, pools or initialization chains across files.

## Evidence needed

The format binds the recorded exact source pins and profile, the native initialization call chain, the copied worker and scheduler configurations, the current worker cache, the scheduler's actual attachment to its core, the same direct dispatch object, request lifetimes and a stable mutation interval. Sequence numbers establish recorded causal order. Wall-clock timestamps and matching log text do not supply these edges.

The positive allocator row must contain a current observed acquisition ordinal for the expected block. A present assignment B and a model table entry C at the same logical position establish a mapping contradiction when B differs from C. An unavailable or short allocator row stays unknown. The checker does not infer absent rights from an empty map or a zero reference count. Acquisition ordinals record native allocation observations; they do not prove that block contents changed and are not native lease nonces.

Each session is evaluated independently. A known contradiction can remain visible alongside a missing row. Missing global association, unhealthy recording, incomplete mutation coverage or overlapping capture prevents a trusted mapping verdict.

## Pinned interpretation

The interpretation is tied to vLLM 0.32.0 source commit `5df63a7203cdb9328d824a27b682e77c7b9928be`, the exact UniProc executor, the eager V2 Qwen3 profile, FLASH_ATTN 3, LBHNC, one full-attention cache group and no prefix caching, speculation or alternate mapping. Native capture additionally requires the observed `kernel_config.enable_jit_warmup` to be false. The installed adapter description contains the complete relevant source hashes and exact class restrictions. Unsupported profiles stay unknown.

The source basis is the [core initialization call chain](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core.py#L326-L377), [worker config selection](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/worker_base.py#L340-L344), [runner config copy and backend selection](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py#L605-L674), [direct UniProc dispatch](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/executor/uniproc_executor.py#L90-L132) and [non-mutating ordered assignment getter](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_coordinator.py#L452-L457).

These pins cover the mapping dependencies, including actual configuration derivation, dispatch and attention-group initialization helpers. They do not classify the lease fix. Mapping-only fix status remains unknown. The separate existing fix classifier requires its complete nine-file engine profile. A match or mismatch in one transfer file is insufficient. The mapping source inventory does not require unchanged code on an unsupported multiprocess path or a specific transfer patch.

## Limits

Matching maps can share an erroneous allocator assignment. A correctly assigned destination can also contain bytes copied from an incorrect source. This rule does not settle either problem. Complete read rights, content provenance, physical GPU reads and runtime isolation remain unknown or not established. It grants no new upgrade preservation policy. Actual native engine hook coverage, owner-run validation and overhead measurements remain pending.

Inspection and verification start no engine, request, device work or network operation. HTML is static and escapes recorded strings. Its generated bytes are tested; browser rendering is not a validated runtime surface.
