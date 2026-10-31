# Passive native mapping collector

Owner: Hari Krishna Kancharla.

This assembler connects the existing forward metadata observer, initialization observer and bounded assignment journal. Importing it starts no engine, device work or network operation. Explicit passive activation is available through the existing plugin. Native runtime compatibility, owner-run validation and overhead measurements remain pending.

Set SDCGUARD_ASSIGNMENT_DIR and SDCGUARD_FORWARD_DIR to separate dedicated directories before the supported engine starts. The former requests mapping facts in addition to the existing forward stream. Neither setting enables the arithmetic checks. The plugin supplies one stable mapping factory to the existing forward installation. It does not add another model hook or runner wrapper. Installing the package or importing mapping_activation alone activates nothing.

The general-plugin loader runs before EngineCore calls _initialize_kv_caches. That is the method the initialization observer wraps. The native profile pins the loader and core sources. A late attempt to add mapping to an existing forward-only installation stays unknown. Changed directories, inherited process state, unsupported source bytes and missing effective profile fields also stay unknown. Registration is not runtime validation or a completed mapping verdict.

The mapping_activation.status() accessor returns independent host status and journal artifact paths for the current process. Its outer health includes the journal's own health, hook coverage while open, and the actual sealed outcome after close. Interval health and export readiness are separate. A raw final-looking file or a healthy footer cannot substitute for the actual close result and settled healthy status.

The opt-in factory supplies one post-close callback. After admission stops and producers are accounted for, it writes an exclusive session.closure.json receipt. That artifact records local recorder bookkeeping, not a native engine event. It binds the actual close result, sealed healthy status, session, process, exact profile, source hashes, dependency observations, journal filename, byte count and SHA256. A negative close is retained as negative. Duplicate close does not retry or replace the receipt. Receipt publication failure does not change the actual historical interval outcome; it leaves export readiness false. Storage latency can delay receipt availability.

Use `sdcguard assignment capture MAPPING_DIR --out NEW_PACKET` to export these native recordings. It uses mapping_closure.capture_closed and retains the exact raw payload and separate receipt bytes. It checks strict bounded JSON, no-follow regular files, exact paired inventory, actual positive closure fields and the captured journal bytes. Every recorded envelope must retain contiguous integer sequence numbers and the same session/process identity. The helper derives body-event, positive-schedule, admitted-record, frame-end, acquisition and node counts from those bytes. Footer and receipt counters must be exact integers, and settled queued bytes must be zero. Missing, negative, partial, stale, replaced or mismatched receipts are rejected and must produce unknown. `sdcguard assignment verify NEW_PACKET` repeats the closure checks offline before recomputing the named mapping verdict. Complete bookkeeping can contain a violated or unknown mapping relation; it does not require mapping agreement. Raw-format `assignment inspect` does not establish this cross-process closure check. Local consistency does not authenticate the recorder or deployment. Client attribution remains unknown.

The narrow owner configuration must explicitly disable and verify kernel_config.enable_jit_warmup. In the pinned source, gpu_worker.py:952-957 otherwise invokes gpu.warmup.warmup_kernels with the ordinary worker execute callback. That helper submits hand-built SchedulerOutput objects without the actual Scheduler.schedule association. They correctly make this observer unknown. The explicit _dummy_run calls use dummy_run=True and are excluded before observation; unbound warmup callbacks are not suppressed or relabeled. The exact owner startup command and runtime confirmation remain pending.

The collector uses the existing four runner wrappers and one model pre-hook. It adds observation wrappers on the pinned scheduler and fifteen allocator methods. It does not wrap the legacy KVCacheManager allocation/free slots. Every native argument, return and exception passes through unchanged, and no native work is retried or rescheduled.

## Observed relation

The rule is `allocator_to_model_entry_mapping_consistency`. The collector binds the actual core initialization, copied configs, constructed allocator and current runner cache. It then follows the actual SchedulerOutput object and current Scheduler.requests entries into the direct UniProc call. Request names, clock times and local labels alone are insufficient. Resumable or streaming requests are unsupported.

One process interval and one cache initialization are supported. Native objects use one bounded weak identity registry. KVCacheBlock is not weak-referenceable in this source. The collector therefore keeps no block object outside a dynamic call: it labels successful pool acquisitions by the proven pool association and native index, and verifies each observed block is the exact current pool.blocks[index] with that pool pointer. These ordinals are not byte generations or native lease nonces.

The coordinator's concrete ordered getter is copied without indexing or inserting a missing defaultdict row. An absent or short row stays incomplete. A row with no observed allocation ordinal is represented as unavailable, rather than receiving a guessed ordinal. A zero reference count does not establish absent rights.

Each native mutation advances a recorded epoch at entry and exit, including no-op calls. Cache method entry alone is not an unsupported branch. Actual counter changes, CoW completion, range replacement and successful prefix reset are distinct from the source's no-op returns. Incoming free, unpin and replay iterables remain untouched by the observer. A successful zero-block pool acquisition still emits its empty return fact.

The model-entry callback receives the same immutable host copy already made by the existing hook. Each layer is converted independently and clipped by its exact context length. The collector makes no second device copy and does not normalize a contradictory table to its expected assignment. A known allocator block B compared with table slot C remains a mapping contradiction when B differs from C, including a null or unacquired C.

## Source interpretation and recording

`vllm.0.32.uniproc.eager.native.v1` adds the verified Request source and exact Request/SchedulerOutput classes to the common source interpretation. The accepted default `vllm.0.32.uniproc.eager.v1` dictionary remains unchanged with its 35-file inventory. Existing default-profile raw records can still be reinspected. The native profile records all 36 actual source hashes in its header and does not classify the lease fix.

Initialization facts are bounded host copies until the actual supported profile exists. They retain their causal order. Model revision, backend selector, topology, model and dependency observations are read from actual state. A missing field or mutable revision such as main does not receive an expected default.

Closing stops admission and removes only owned wrappers before the journal samples shared health. Open producers, mutations, frames and incomplete positive dispatches make closure unhealthy. A caller must check close success and the independent journal status, including healthy publication and footer completion. A final-looking filename is insufficient. An unfinished artifact must remain visible in an unknown result. Later activity is outside the sealed historical interval and cannot reopen it.

A local model path uses the explicit versioned model-reference extension described in [FORWARD_INPUTS.md](FORWARD_INPUTS.md). The native header keeps the actual observed `profile.model`, with the canonical repository ID and GPU-pod file receipts separate under `profile.model_reference`. It reuses the same process-scoped binding for the actual runner, or an explicitly supplied binding for that same actual model-config object. The extension changes no engine source pins, allocator semantics or runtime guarantees. Its setup file proof cannot establish loaded weights, ownership or a candidate execution witness.

The native bridge passes a weak installation capability before initialization begins. The initialization and header observers recheck its process, runner class and engine registration, then use the same runner binding. The model-config reference is weak too. A collected or changed installation or config, missing capsule, or changed path remains unsupported. No model file is rehashed during each forward.

## Limits

The result covers the recorded mapping relation. Matching maps can share an erroneous allocator assignment. A legitimately assigned destination can also contain data copied from a stale source. Complete ownership, read rights, content provenance, physical GPU reads and runtime isolation remain unknown or not established. Local source hashes do not authenticate loaded code or a remote deployment.

The older saved #61229 and H100 recordings contain no new initialization, assignment or model-entry facts. Their existing lifetime results remain separate. This collector adds no new failure ordering, retry or reused-ID mechanism, and drives no live leak condition.

CPU checks use ordinary host objects and retained primary Python method bodies. They are not a running engine or GPU compatibility result. Fresh independent combined review, full native CPU suites, packaging and an owner run are required before this candidate is ready.
