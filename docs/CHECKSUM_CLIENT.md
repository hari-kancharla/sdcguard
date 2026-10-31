# Client checksum reconciliation

Owner: Hari Krishna Kancharla.

This candidate joins an independently written client plan to ordinary HTTP receipts, actual frontend request-ID assignments and the checksum scheduler's recorded callbacks. It does not attest a deployment or qualify a release. Every result remains HOLD until the separate runtime and release requirements are met.

## Input contract

`evaluate(plan_raw, inventory_raw, captures_raw)` accepts three strict JSON byte strings. All three share `run_id` (32 lowercase hex characters) and `packet_sha256` (64 lowercase hex characters). Inventory and captures also carry `plan_sha256`, the SHA256 of the exact original plan bytes. Request IDs and engine session IDs are opaque strings, not timestamps or ownership identities.

`REQUEST_PLAN.json` uses schema `sdcguard.checksum-request-plan.v1`:

- `source_revision` is the pinned checksum revision and `model` is the API model name.
- `phases` is a list of `{phase_id, concurrency}` declarations.
- `arms` is a list of `{arm_id, engine_session_id, observer_enabled, requests}`.
- Each planned request has `slot`, `phase_id`, `kind` (`warmup` or `measured`), `concurrency`, `prefill_request_id`, `client_request_id`, `prompt_tokens`, `prompt_sha256` and `max_tokens`.
- Slots are consecutive within each arm. All warmups remain in the list. Off and on arms use different IDs. The same arm session may span several client concurrency phases; joins never depend on phase timing.
- `prompt_sha256` is `prompt_digest(token_ids)`, the SHA256 of `evidence_io.json_bytes(token_ids)`. The recorded API prompt must be that exact nonempty integer token list.

`CLIENT_INVENTORY.json` uses schema `sdcguard.checksum-client-inventory.v1`. It has `completed` and a flat `requests` list. Each row has `arm_id`, `slot`, `client_request_id`, `prefill` and `decode`. Every HTTP stage has `completed`, `method` (`POST`), `path` (`/v1/completions`), `http_status`, `request_headers`, `request_body`, `response_body` and `error`. Timing fields may be retained alongside these facts.

Both stage bodies explicitly use the planned model, integer prompt, `n: 1`, `stream: false`, and `add_special_tokens: false`. Their `request_id` and `X-Request-Id` header are the distinct planned stage ID. Prefill requests one output token; decode uses the planned `max_tokens`. Decode forwards the producer's actual `kv_transfer_params` unchanged. The first-direction descriptor requires `do_remote_prefill: true`, `do_remote_decode: false`, one nonempty typed block group, a remote engine/request/host/port, TP/DCP/PP one, `remote_num_tokens` equal to the prompt count, a registration generation and a nonempty base64 checksum payload. Successful responses have status 200, the exact pinned response-ID form and the planned prompt count in `usage.prompt_tokens`. A missing receipt stays in the planned denominator.

`CAPTURES.json` uses schema `sdcguard.checksum-client-captures.v1`. It has a `captures` list. Each on-arm capture has `arm_id`, `engine_session_id`, `source_revision`, `capture_complete`, `startup`, `startup_sha256`, `binding_snapshots` and `checksum_snapshots`. Startup is retained as an object; its digest is SHA256 of `evidence_io.json_bytes(startup)`. Startup contains the shared run, packet, arm and engine session IDs, `role: decode`, the source revision, model, `source_profile: source_profile()`, `checksum_enabled: true` and `request_id_randomization_enabled: true`. These are recorded local declarations, not an attestation.

Each checksum snapshot is the original closed observer result plus a `capture_context` object containing `run_id`, `packet_sha256`, `arm_id`, `engine_session_id` and positive `process_id`. The original observer fields remain unchanged. The binding snapshot carries that context at its root and a separate binding source profile. API and engine-core processes may have different process IDs. Healthy empty process snapshots can be retained; this narrow profile permits one active binding recorder and one active checksum recorder per engine session. All recorded checksum calls remain visible, including unmatched calls.

Off arms have `observer_enabled: false` in the plan. Their HTTP rows remain visible, with reason `observer_disabled`. They cannot borrow checks from the on arm and are not counted as verified by an observer.

## Passive request-ID observation

`RequestBindingObserver(run_id=..., packet_sha256=..., arm_id=..., engine_session_id=...)` installs explicitly with `install(InputProcessor, source_root=...)` and returns a closed snapshot with `close()`. `snapshot()` is open until close. The source-function CPU seam is `install_cpu_source_functions(...)` and is always labeled as such.

The original static method runs exactly once with the original arguments, mutation, return and exception behavior. Only after that method returns does the observer record its actual external and internal IDs. It also records the count and canonical SHA256 of the existing bounded CPU `EngineCoreRequest.prompt_token_ids` list, with a maximum of 8,192 tokens. Token values are not retained in the binding trace. A missing, unsupported or different engine prompt keeps the client join unknown. The bounded CPU hashing cost is part of the measured observer overhead; no device data is read. The observer does not choose an ID, disable native randomization, modify a request or add an engine operation. Source drift, foreign wrappers, unsupported arguments, recording loss or active calls at close makes the recording incomplete.

## Source basis

All references use checksum revision `29d2f4e03740acb14439ce22d24b06d8095d17d3`.

- [Base request ID, serving.py:133-142](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/entrypoints/serve/engine/serving.py#L133-L142): a supplied X-Request-Id header takes precedence, then the request's default ID.
- [Completion dispatch, serving.py:147-214](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/entrypoints/openai/completion/serving.py#L147-L214): response IDs use `cmpl-`; a single rendered prompt adds `-0` before engine submission.
- [Native request assignment, input_processor.py:363-380](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/v1/engine/input_processor.py#L363-L380): the engine retains the external ID and normally appends eight random characters to its internal ID. The join uses the recorded actual mapping. It does not guess this suffix.
- [Request conversion, request.py:244-270](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/v1/request.py#L244-L270): scheduler Request objects do not retain external_req_id, so the checksum request ID alone cannot establish this join.

The binding source profile also pins [EngineCoreRequest fields, engine/__init__.py:137-177](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/v1/engine/__init__.py#L137-L177). The recorded token list is CPU input metadata at assignment, not a physical read observation. Ordinary transfer fields follow [pull_scheduler.py:278-292](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py#L278-L292). In this supported no-speculation profile the pending checksum span is the full prompt count. A different observed span stays unknown rather than being normalized to a convenient count.

Source hashes establish local consistency. They do not authenticate a recorder, establish complete immutable engine identity, prove ownership or observe physical reads. The existing checksum source profile and its native outcomes remain separate from the historical lease build.

The reader bounds each capture to 4,096 total binding records and 4,096 checksum records across its process snapshots. Missing or duplicate capture envelopes cannot hide readable native calls: they remain explicit unexpected or unverified observations. Router forwarding beyond the recorded client handoff is not established by copying a descriptor in the client log.
