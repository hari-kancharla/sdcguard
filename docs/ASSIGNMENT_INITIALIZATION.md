# Passive initialization component

Owner: Hari Krishna Kancharla.

`sdcguard.lifetime.assignment_init` implements the initialization part of a future native mapping observer. It has no plugin, command or automatic activation. It does not start a server, import an engine or device library, inspect model contents, acquire a GPU, add a model hook or wrap any runner method.

The component observes the pinned vLLM 0.32.0 same-process UniProc startup chain. A future assembler must explicitly provide already imported native classes/modules, the source root, the existing forward bridge, a shared health object and a bounded sink. The complete allocation/frame assembler is absent. These initialization facts alone cannot produce a complete mapping verdict, ownership verdict, physical-read claim or runtime-isolation claim.

## Implemented interface

`InitializationObserver(sink, health)` uses `sink(InitializationFact) -> True` for immutable host-only facts. `fact.record()` returns an independent ordinary copy for a writer. A rejected or failing sink marks sticky observer/health errors. The sink must be a bounded, nonblocking recorder admission function. This component creates no writer, thread, file, footer or asynchronous queue.

`attach(NativeBindings(...), source_root, version)` checks the refined 35 source hashes, exact supported class names and helper owners, then installs its own wrappers transactionally. The supplied native objects are an internal integration input, not remote attestation. Source presence and class names do not authenticate loaded code. Current wrapper identity is rechecked throughout initialization and before returning a later binding. Detach restores only wrappers still owned by this installation.

The future bridge assembler delegates these two existing callbacks:

- `retire_runner(runner, reason)` before native cache initialization/replacement or shutdown.
- `runner_initialized(boundary)` after the existing forward wrapper successfully returns from real, supported runner initialization. The boundary supplies the actual runner, bound original arguments, copied runner config, actual block tables and a label. No native object is serialized.

The component is not itself the bridge's full callback protocol. It supplies no invocation, preparation, frame, mutation, writer-close or runner-close implementation. Registering it directly as that complete observer would be incorrect. Those remaining responsibilities must be implemented and reviewed in the future assembler before activation.

`binding_for(core, scheduler, runner)` checks the complete observed initialization and the actual later `core.scheduler` pointer. It returns immutable tokens/dimensions or `None`. It emits no schedule or frame. Missing initialization, stale resources, changed hooks, observation errors or unsupported runtime state remain unavailable.

## Observed chain

1. Wrap the later `EngineCore._initialize_kv_caches` call after plugin loading, rather than a constructor already in progress. Bind the actual core/executor/wrapper/worker/runner objects and reserve their current generations.
2. Observe the actual core-module scheduler-config deepcopy and granularity resolver aliases. Retain the native list only within the parent call. Emit config dimensions at the actual executor initialization entry after hash/alignment fields have been finalized.
3. Follow the actual executor call, wrapper rank-zero selection and worker config argument. Variadic native keyword arguments are interpreted for observation only. Original calls receive their original arguments unchanged.
4. Consume the existing runner callback. Require nonprofiling input, a distinct actual copied runner config, fresh table/call incarnation, full matching layers and observed kernel size. Every attention group must have no alternate block-table remapping. Config events contain no future kernel facts.
5. Emit the outer init return only after native worker initialization and warmup return successfully. Match its actual returned scheduler-config object.
6. Follow successful outer Scheduler or AsyncScheduler construction, actual allocator factories and manager/coordinator/pool/group constructors. Require the real pointer chain, capacity, block sizes and null block identity. The base constructor return alone does not complete AsyncScheduler construction.

The existing strict record format calls the fresh runner cache-call label `worker_pool`. Internally it is `worker_cache`. Neither name proves physical pool identity. The label is not joined to a scheduler pool by equality, and is not a native lease nonce or content generation.

## Bounds and lifecycle

Native identity uses guarded weak references keyed by `id` and never invokes native equality or hashing. Tokens are never reused within the bounded observer. Historical config-copy edges remain usable after the original worker config is legitimately collected. The observer retains no native tensor or strong core/executor/worker/runner graph. Unsupported non-weak-referenceable identities remain unavailable.

Shared participants and overlapping initializations invalidate previous and pending generations. An older call returning later cannot revive its binding. Profiling can retire an old cache, but cannot emit a ready real-cache observation. Replacement, changed native pointers, detach, fork/PID change, missing returns and exceptions prevent future bindings. This component's internal retirement is not a substitute for the future assembler's complete lifecycle records and session accounting.

Original functions run once with unchanged arguments and return values. Original exceptions propagate unchanged. Observation failures mark incomplete evidence; they do not retry native work. No observer lock spans native calls. No generator argument is drained for observation. Ordinary process-control exceptions remain process-control signals; cleanup cannot replace an original native exception.

The initial profile requires the exact supported classes, Qwen3 dense causal full attention, eager execution, one group/rank, prefix caching off, automatic KV dtype, no speculation, no extension, no graph capture and no per-group kernel subdivision. Source-native `worker_extension_cls` is the empty string; `ubatch_size` zero and one both disable microbatching when DBO is off. The existing forward recorder still supplies its independent, stricter model-entry validation. Transfer/lease correctness, arbitrary custom allocator compatibility and full read rights are not established by this component.

## Primary source and CPU evidence

All references use commit `5df63a7203cdb9328d824a27b682e77c7b9928be`:

- [Core startup and final hash/alignment writes](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/engine/core.py#L326-L406).
- [Actual scheduler-config deepcopy](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/kv_cache_utils.py#L2276-L2297).
- [Native direct dispatch](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/serial_utils.py#L485-L508) and [rank selection](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/worker_base.py#L340-L344).
- [Runner copy and selected kernel dimensions](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/gpu/model_runner.py#L605-L704), [per-group kernel remapping](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/worker/utils.py#L284-L395), and [actual group-manager factory](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/v1/core/single_type_kv_cache_manager.py#L2585-L2630).
- [Native extension default](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/parallel.py#L305-L312), [microbatch default](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/parallel.py#L257-L260) and [effective microbatch predicate](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/config/parallel.py#L640-L646).

The tests retain the verified 35 primary files in a small compressed fixture and execute selected pinned Python method fragments with ordinary host dependencies. They run no inference engine, device kernel or server. Negative cases are observer/profile/lifecycle controls. Fixture SHA-256 and Git blob checks establish source identity; these CPU tests do not establish native deployment compatibility or overhead. Existing saved transfer evidence cannot be upgraded to initialization/mapping evidence.

Mapping-only fix status stays unknown. Fix presence remains the separate complete nine-file source classifier. Full assembler review, complete CPU suites, packaging and an owner-run proposal remain required before activation or native support claims.
