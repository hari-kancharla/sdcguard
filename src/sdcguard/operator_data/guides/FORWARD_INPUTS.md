# Recorded forward inputs

Owner: Hari Krishna Kancharla.

This observation records an attention input plan on a narrowly supported eager vLLM path. It does not observe physical GPU loads, prove block ownership, or establish that an answer used particular bytes. Runtime validation of this recorder is pending a separate owner-run check. The earlier H100 result validated different lifetime hooks.

## Inspect saved observations

Use a dedicated directory of completed process JSONL recordings. The commands run offline and start no engine or requests:

```sh
sdcguard forward-inputs inspect /path/forward-recording --out /new/forward-packet
sdcguard forward-inputs verify /new/forward-packet
```

The destination must be new, its parent must exist, and it must be outside the input tree. The packet retains the complete input bytes, source-pinned checker identity, recomputed JSON and a readable HTML report. Verification recalculates the report from those retained bytes. Changing a report and recalculating its manifest cannot establish a different verdict.

Exit 0 means the supported metadata observations are complete under the format. Exit 2 means invalid, incomplete or unsupported input. Neither exit code certifies isolation. Physical-read and ownership verdicts remain unknown, and the overall isolation scope remains inconclusive. Hashes establish local consistency, not remote authenticity.

Admission events from the saved #61229 recordings do not contain forward metadata. Inspecting them cannot manufacture new observations or deeper evidence. Malformed or unfinished records, gaps, missing closure, duplicate session identities and unsupported profiles remain unknown. The original input files are not edited.

## What is recorded

Each observation binds the actual request row ordering to a process-local forward identity and cache-pool incarnation. Its observation point is model entry, not the later backend or device invocation. It preserves the immutable block-table rows, exact sequence lengths, query boundaries and supported per-layer layout visible there. The slot mapping is retained as a separate write map. The checker derives bounded logical context intervals mapped to physical block numbers without expanding one object per context token. Later runtime hooks or mutations are not excluded by this observation.

A model step may write one new token while attending over many earlier tokens. This is why its write slots cannot stand in for its context access plan. The pinned [cache-write implementation](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/attention/backends/flash_attn.py#L1427-L1460) and [attention metadata inputs](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/attention/backends/flash_attn.py#L1276-L1281) are separate.

An immutable snapshot needs ordered copies. Keeping a tensor reference or detached view leaves mutable engine storage underneath it. Snapshot-copy completion is distinct from forward-device completion. A host forward return or stream footer does not establish device completion. The recorder adds work and can change timing; overhead needs measurement on its own supported runtime.

## Support boundary

Support is checked against exact relevant source hashes from vLLM v0.32.0 commit `5df63a7203cdb9328d824a27b682e77c7b9928be`, plus effective runtime configuration and metadata. A version string, model family or method name alone is insufficient. The profile is eager V2 FLASH_ATTN with ordinary dense causal Qwen/Qwen3-0.6B attention, one cache group and one rank, no prefix caching, speculative decoding, LoRA, unreviewed sharing, masks or windows. Actual backend and dependency identities are retained separately from the source pin. Unsupported facts remain unknown.

The raw model hook is inside the [V2 forward context](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py#L2015-L2044). Full graph replay uses a [different path](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py#L1997-L2006); ordinary Python hooks are not per-replay witnesses. Dummy and profiling execution cannot count as real request observations.

The [pinned engine requires Python >=3.11 and <3.15](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/pyproject.toml#L34). Python 3.10 tests cover the offline package, not a vLLM 0.32 runtime. The recorder captures its actual interpreter and backend identities. A build log or a CPU test cannot supply a missing GPU runtime observation.

The opt-in recorder is separate from the lifetime recorder and arithmetic checks. Recording metadata never enables failure scheduling, holds, retries or request mutation. Ordinary observer errors become recording gaps. Original engine exceptions retain their behavior. Recovery from device faults is not promised.

The activation setting is `SDCGUARD_FORWARD_DIR`, pointing to a dedicated recording directory before the supported engine starts. GPU validation is owner-run. The setting enables the source-gated forward observer; the separate lifetime observer uses `SDCGUARD_LIFETIME_DIR`.

Normal native model-runner shutdown closes that runner's observer, removes its hook and attempts a bounded writer flush. It preserves the engine's return value or exception. A shutdown error or an overlapping capture makes the recording incomplete. Abrupt termination, lost output or a missing footer remains unknown. A closed observer session does not establish per-forward device completion.

HF offline mode can replace the requested repository ID with a local model and tokenizer snapshot path in the [pinned argument normalization](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/engine/arg_utils.py#L907-L931). The optional `sdcguard.observed-model-reference.v1` profile extension retains that actual path in `profile.model`. Its separate `model_reference.reference.model` is the canonical repository ID. A served-model alias or a matching directory basename is not sufficient.

For this local-path case, the owner preflight must produce a `sdcguard.model-reference.v1` capsule once in the actual GPU pod, before engine startup, by hashing all ten files against the installed `MODEL_PIN.json`. It records the configured and resolved snapshot paths, exact model revision, pin digest, per-file digests and stat receipts, owner run ID and packet digest. CPU preflight stat values from another pod or mount cannot substitute. The per-role environment must supply `SDCGUARD_MODEL_REFERENCE_FILE` as the exact canonical capsule file path, `SDCGUARD_MODEL_REFERENCE_RUN_ID` and `SDCGUARD_MODEL_REFERENCE_PACKET_SHA256`.

Each process binds the capsule to the actual model-config object and its observed model/tokenizer paths and revisions. Setup checks the local file receipt associations. Later observations check the same config fields, environment and capsule file identity without rehashing the weights or rereading all model files on every forward. Missing, changed or conflicting proof produces a gap. This is file consistency at setup and recorded local provenance, not continuous file monitoring, loaded-weight attestation, read permission or ownership proof. The existing exact repository-ID branch and old raw profile shape remain supported. Completed packets still require their exact recorded evaluator identity; this extension does not waive old packet hashes.

## Ownership is a separate rule

An observed invalid request-to-block binding can be a useful finding once a rule proves the conflicting rights in the same decoder pool and at the observation. The metadata snapshot alone supplies neither that ownership proof nor its ordering. Pool UUIDs label local recorder incarnations; they do not prove a join to another process's allocator.

Do not infer invalid ownership from a zero reference count. vLLM explicitly allows some [unpinned blocks to stay readable](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/block_pool.py#L708-L727), and cached-prefix sharing has separate rights. The later ownership rule needs allocation and lease identity, allowed sharing and causal joins. It must name a binding violation separately from physical reads or disclosure.

NIXL also has separate remote-source and decoder-destination block lists. Wrong source bytes can be copied into a valid decoder-owned destination. A valid decoder attention table therefore does not disprove #61229. The existing transfer-lifetime rule remains separate.

This observation format does not grant a new upgrade-safety contract. The existing upgrade gate keeps its named contracts and coverage requirements. Adding a binding rule to that gate requires reviewed ownership evidence and explicit compatibility policy.
