# Passive checksum observations

Owner: Hari Krishna Kancharla.

This candidate observes the consumer checksum scheduler in vLLM checksum PR #62211 at `29d2f4e03740acb14439ce22d24b06d8095d17d3`. It does not backport an engine change or combine that branch with the older lease build. Installation is explicit and must occur before constructing the scheduler. No plugin activates it automatically.

The observer delegates each original callback once, with the original arguments, result and exception behavior. It adds no checksum calculation, engine comparison, device copy, device synchronization, transport action or request ordering control. It retains bounded host metadata and original native comparison outcomes. CPU source-function validation and actual deployment validation are separate.

## API

```python
from sdcguard.kv_checksum_observer import ChecksumObserver

observer = ChecksumObserver()
observer.install(KVChecksumScheduler)
# The existing engine invokes its own ordinary callbacks.
snapshot = observer.close()
```

`install` checks seven exact source files from the installed pins and the loaded callback and helper code. It refuses unknown source files, a foreign source path or a changed callback. It does not start an engine. `install_cpu_source_functions(cls, source_root=...)` is the separate test seam. That seam checks the original functions against the pins and labels all observations `cpu_source_functions`. It cannot qualify a deployment.

`source_profile()` returns the profile, revision and expected source hashes without importing an engine. `classify_verification(record, max_units=65536)` re-derives conditional chunk scope from primitive retained metadata. It does not authenticate the record. `snapshot()` returns a detached JSON-compatible object. `close()` restores only the observer's own wrappers and returns a closed snapshot. `close` and `uninstall` are idempotent. A foreign wrapper is preserved and marks source support unknown.

An open snapshot, a recording error, any budget loss, changed callbacks or a close while a callback is active makes effective verification coverage unknown. An original native pass remains visible separately. The snapshot always reports `runtime_qualified: false` and `release_qualifying: false`. A source observation cannot itself promote a release.

## Observation boundary

The causal path is the original `on_new_request` and `add_recv` load plan, followed by the original `update_from_output`, `_verify` and `_compare_group` callbacks for the same scheduler and pending object. Direct verifier calls do not establish that path. A call that did not occur is not inferred from a zero counter. Planned requests without verification remain explicit.

The observer retains the plan when native `add_recv` schedules it. At verification it reads bounded metadata before and after the original call, without re-running a comparison. Native group results are retained as not compared, passed, mismatch or unverified. Native aggregate `False` is not automatically called a mismatch: the engine also returns it for absent or unsupported comparison evidence.

The first complete conditional scope is deliberately narrow:

- TP, PP, DP, DCP and PCP are one, as observed in constructor configuration, with native TP and worker counts one.
- Speculative decoding is disabled in the observed constructor configuration.
- There is exactly one configured and planned ordinary `FullAttentionSpec`, bound to the pinned loaded class. Sliding-window and attention-chunk fields are disabled.
- The local prefix is zero. Producer and consumer token counts, block sizes, group identities, row counts and fingerprints agree.
- The producer starts at zero and has one digest per full-prompt checksum chunk, including any final partial chunk.
- The only worker rank is `(0, 0)`, its record agrees, and its `uint64` array has shape `(N, 1)` with replica key `(1, 0)`. Empty replica arrays cannot establish coverage.
- The actual native group callback ran, its result agrees with the aggregate result and no unverified reason contradicts it.

Units such as `group:0:chunk:1` denote native checksum chunk comparisons. They are not physical block identities, observed GPU reads or evidence that every model layer or byte was checksummed. Unsupported layouts, partial spans, absent payloads, missing calls or recording gaps remain unknown.

An unknown compared set is `null`, with an unknown missing set. It is not an empty set or a claim that every planned unit was missed. An empty compared list is reserved for source-proven absence of comparison. If the exact compared set was observed before a later recording gap, that set remains visible while the overall coverage result becomes unknown.

Unknown compared and missing unit sets are `null`. Known empty sets are empty lists. A partial span may include native comparisons even though this profile cannot establish the full compared set. A later recording gap keeps any independently established comparison set visible while holding overall coverage unknown.

Two native-pass gaps have explicit reasons in the report:

- `partial_checksum_span`: the original verifier returned true after a strict producer/consumer chunk intersection or valid-token filter left planned positions outside the compared span. The native pass remains visible, while full coverage stays unknown.
- `empty_replica_inputs`: the original verifier returned true with an empty replica-key list and an `(N, 0)` consumer array. Its zero-array arithmetic does not represent a consumer replica. This is not described as no comparison call.

Both are labeled **Engine pass, not covered**. Neither is reported as engine corruption. Different block sizes alone do not establish partial coverage; they remain unsupported by the narrow profile.

## Bounds and closure

Defaults are 4,096 retained records, 65,536 total retained unit entries and 8 MiB of metadata. Records and pending request/configuration state share the byte budget. The observer reserves 16 KiB for fixed metadata and 8 KiB per pending state entry. The minimum byte budget is 64 KiB. Each retained record reserves enough space for its later incomplete-recording classification. Arrays and checksum values are not copied into the trace.

Returned mismatch-position lists also share one in-flight unit and byte reservation across all groups and active verification calls. A copy is reserved before it is made, and the reservation is released in the verification callback's final cleanup. An unsupported multi-group call cannot multiply the copy allowance by the number of groups.

Exhausting a bound records sticky loss and makes coverage unknown. Snapshot counters describe the actual rendered record bytes and unit entries. Native calls continue unchanged. Closure requires no active callbacks; it does not stop or wait for an engine process and does not establish process teardown.

## Offline inspection

Write the JSON-compatible object returned by `observer.close()` to a new local file. The installed reader starts no engine and imports no device library:

```sh
sdcguard checksum-observer inspect /path/snapshot.json
sdcguard checksum-observer capture --snapshot /path/snapshot.json --out /new/checksum-packet
sdcguard checksum-observer verify /new/checksum-packet
```

Use `--json` for structured output. These commands return exit code two for this candidate, including a correctly written and verified packet. The report always holds release because runtime qualification, an independent client inventory join and a combined lease/checksum source build are missing. A hold is not a checksum mismatch or a command crash.

The summary counts recorded verification calls, not all client requests. It keeps native outcomes, conditional chunk scope, missing verification calls and recording health separate. A receive plan without a later verification remains visible. The reader checks source declarations against installed pins and re-derives chunk scope from primitive metadata. This is a consistency check, not authenticated execution evidence.

Capture retains the exact snapshot, source profile, evaluator hashes and JSON, text and HTML reports. Verification recomputes every report from the retained snapshot. It works after the external snapshot file is removed. Editing a derived verdict and recomputing the outer hash does not pass verification.

The first reader supports at most 4,096 records, 65,536 retained unit entries and an 8 MiB observer byte budget. Larger library budgets need a separate reader profile. Inputs use the shared bounded regular-file contract. Outputs must be new directories and cannot overwrite the snapshot.

## Primary source basis

The exact source pins are installed in `case_data/checksum-observer-pins.json`.

- [Consumer scheduling and completed output processing, scheduler.py:280-362](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/distributed/kv_transfer/kv_checksum/scheduler.py#L280-L362).
- [Aggregate and per-group comparison, scheduler.py:465-623](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/distributed/kv_transfer/kv_checksum/scheduler.py#L465-L623). Comparisons intersect the producer and consumer chunk ranges; an empty mismatch list alone does not prove full planned coverage.
- [Replica aggregation, scheduler.py:673-701](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/distributed/kv_transfer/kv_checksum/scheduler.py#L673-L701). The native helper can produce a zero check when replica keys are empty, so this profile requires a real nonempty replica column.
- [Checksum record types, checksum.py:71-126](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/distributed/kv_transfer/kv_checksum/checksum.py#L71-L126).
- [Full-attention scope fields, kv_cache_interface.py:619-640](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/v1/kv_cache_interface.py#L619-L640).
- [Host output synchronization, async_utils.py:275-300](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/v1/worker/gpu/async_utils.py#L275-L300). Enqueue and worker scheduling are not themselves completed verification.

The checksum and lease branches have different metadata histories. Their integer protocol versions cannot qualify a combined build. Historical lease positives are never borrowed for this checksum candidate. Client inventory completeness, router forwarding, loaded weights, ownership, read rights, physical reads, full immutable engine identity, GPU behavior and overhead remain unverified.
