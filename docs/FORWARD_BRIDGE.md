# Optional forward callback bridge

Owner: Hari Krishna Kancharla.

This is an internal callback interface. It does not supply a native assignment adapter, record allocator events, prove initialization, or change an isolation verdict. The existing forward metadata observer still reports physical reads and ownership as unknown.

`vllm_forward.install(assignment_factory=None)` keeps the ordinary path. When the forward recorder is disabled, it returns before importing the bridge or calling a factory. There is no new environment switch, CLI command or automatic adapter activation.

An explicit factory receives an immutable `Registration` with the runner class and recorded engine identity, plus an `AssignmentHealth` object. It returns an observer implementing these methods:

```python
runner_initialized(boundary)
invocation_enter(boundary)  # bounded string token, or None when unsupported
prepared(invocation_token, rows)
begin_copy(invocation_token, boundary)  # bounded string frame token
end_copy(frame_token, frozen_copy)
abandon_copy(invocation_token, frame_token, reason)
invocation_leave(invocation_token, outcome)
retire_runner(runner, reason)
runner_closed(runner, exception_type)  # True only when selected closure completed
close(timeout=...)  # True only when process observer closure completed
```

The factory must roll back its own partial setup if it raises. An incomplete returned protocol receives a bounded cleanup attempt. Setup failure leaves the separately requested forward recorder usable and returns an explicit unknown bridge status. Register before initialization. Requesting a different factory on an existing installation is rejected and leaves that installation intact.

The bridge reuses the existing four runner wrappers and the single model pre-hook. Preparation captures only immutable row IDs and token count. The pre-hook brackets the existing metadata copy and passes an immutable CPU view to `end_copy` before the forward semantic validator runs. A structurally copied contradictory table can therefore reach a separate checker even when the forward checker rejects its own plan. No second tensor copy is issued for the callback. Copies contain no prompt tokens, KV bytes, model output or tensor aliases.

The callback view uses read-only mappings and tuples. Its raw source is checked against the existing per-record and aggregate byte, node and cell bounds. Callback count, live invocation count and weak runner registry are bounded. Opaque callback tokens must be short printable ASCII strings and cannot be reused within an installation, even after an invocation ends. Internal tickets are checked against live object membership, PID, weak runner identity and dynamic invocation context. A stale ticket, cross-runner use or wrong frame owner cannot deliver a copy. No token by itself proves a native association.

Hook and wrapper identity, global pre-hook absence and process identity are checked at copy boundaries. Missing coverage, changed callbacks, incomplete capture and callback errors set sticky assignment health. A future assignment writer must combine that health with its own dropped/error counters and footer completeness. `AssignmentHealth.healthy` alone is not evidence of an initialization, a complete recording, or a mapping result.

Ordinary callback errors do not replace native returns or discard a valid forward copy. Cleanup contains callback exceptions, including process-control exceptions, to preserve the original native return or exception. Process-control exceptions during ordinary observation remain process-control signals. No native call is retried, serialized, rescheduled or modified by the bridge.

Reinitialization and profiling retire the old callback association before the native initializer runs. A successful non-profiling return exposes the actual bound arguments, runner config and table object to the callback. These are opaque native identities, not serialized or independently authenticated facts. The supplied native adapter must still observe the complete core-to-worker call chain and kernel configuration before emitting any initialization evidence.

Shutdown retires the selected runner before native teardown and calls `runner_closed` after the original method and forward writer cleanup. Repeated shutdown does not reopen an observer or repeat that completion callback. Another runner remains open. `close` is reserved for installation detach. A later explicit initialization may use the same process observer, but the bridge never supplies missing association facts for it.

After fork, the bridge does not touch inherited observer locks, files or callbacks and does not call the factory again. The child may lazily establish the existing forward-only metadata recording. Assignment remains unavailable until a fresh process installation can observe the required initialization. Spawn and forkserver need their own explicit registration and complete initialization chain.

The CPU controls use structural stand-ins and one retained native initialization fragment. They test this transport and its error paths. They are not evidence of a running engine or GPU compatibility. Native initialization binding, allocator/mutation observation and a reviewed owner run remain separate work.
