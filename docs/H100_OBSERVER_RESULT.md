# H100 observer validation

Owner: Hari Krishna Kancharla. Run date: October 27, 2026 in the owner's local time; server logs use October 28 UTC.

The owner-run packet passed its saved-condition and ordinary-traffic checks. An offline recomputation reproduced the measurements and lifecycle judgments from the retained raw records on Python 3.10 and 3.13. This supports the revised source-lifetime and scheduler-admission hooks on the stack below. It does not establish whole-deployment isolation.

## Tested stack

- vLLM distribution 0.32.0 with the exact local backport of proposed PR #62065. Its nine inspected relevant source files match the packaged fixed variant. This is not a complete engine binary or kernel identity.
- Qwen/Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`; all ten pinned model files matched before and after the run. The performance identity's `weights_sha256` identifies the aggregate model pin, not the individual weight file. These are local file observations, not live-weight attestation.
- One observed NVIDIA H100 80GB HBM3, two eager instances on the same host, NIXL 1.4.1 pull mode, prefix caching off. The pinned image and retained runtime paths identify Python 3.12 for the GPU engine; its exact patch release was not recorded. The two offline review interpreters are separate.
- Accepted observer wheel SHA256 `bf98ea485b67061b2a91d555e1d1312da44db500b01733e9eb2747b1171d1fb5`.

## Saved condition and controls

Both clean controls satisfied the recorded rule. All five held trials reached the existing saved condition, recorded a `lease_mismatch` refusal of the stale transfer matched by request ID, and returned the exact clean-reference answer after recovery. The protocol outcome is `correctly_rejected`; the client outcome is `recovered`. This run does not establish a terminal client `refused` outcome.

Offline re-evaluation covered 5,448 regression events. Every held trial records the original allocation, a different owner's allocation and free before the READ, and mismatched expected/observed nonces. The hook assigns incarnation identity at real allocation/free boundaries; it does not reconstruct those incarnations from timestamps alone or read a hardware generation counter. The source-lifetime rule does not observe actual attention consumption.

This was a fresh fixed-arm replay. It did not rerun stock, prove deletion-minimality, add another condition, or exercise SGLang. The historical stock/backport result remains unchanged.

## Ordinary-traffic measurement

Both arms used the same fixed engine with lease verification enabled. The change below measures the lifetime observer and recording overhead. Offline rule evaluation ran after measurement and is excluded. It does not measure the cost of the lease fix itself.

Each arm has three clean references and three serial rounds of 128 timed prefill/decode calls. Each timed decode returned 24 completion tokens, with exact reference text. The observer-on records contain 18,644 events and satisfied findings for all 387 calls, including all 384 timed calls. Each arm recorded 384 timed transfers, zero failed-transfer increments, zero lease-check-failure increments and zero declines. These observed clean outcomes are not a universal no-false-alarm guarantee.

| Saved metric | Observer off | Observer on | Observed change |
| --- | --- | --- | --- |
| Median round decode tokens/s | 68.64692128702099 | 65.88348614581668 | -4.025577679806001% |
| Median round end-to-end p99, ms | 375.94122400003016 | 396.342048000065 | +5.426599345229883% |

Rate is server-reported decode completion tokens divided by summed client request duration, including prefill and decode but excluding receipt writes between calls. Tail is nearest-rank p99 within each round, then the median of the three round p99 values. It is not pooled p99, TTFT or inter-token latency. The off arm ran before the on arm. These grouped serial observations have no confidence interval and do not measure saturated throughput or establish a causal overhead bound for other traffic.

## Retained errors and provenance limits

All six server logs contain UCX network-interface enumeration errors. Five contain engine-death traces after requested shutdown and resource teardown. The regression producer also records five expected invalid-KV expiry errors. The completed timed requests and transfer counters passed; recorded process cleanup completed with no remaining PIDs. The logs are not error-free and this single-host run does not validate cross-host transport.

The response fingerprint is `vllm-0.32.0-96cb1967`. It is a recorded response identifier, not a verified full Git commit. File hashes and recorded GPU identity are not remote attestation. Actual billed cost was not recorded.

## Evidence locations

Run ID: `8250642462a045a588e890ea0e79c669`.

The raw run directory (`SUMMARY.json`, `regression/RESULT.json`, both `PERFORMANCE.json` files, events, logs and cleanup receipts) is kept offline and is not part of this repository.

[H100_OBSERVER_RESULT.json](H100_OBSERVER_RESULT.json) carries the exact measured aggregates, per-round observations and raw-file hashes. It is a derived result note. It does not bundle the raw run or grant a new deployment verdict. Raw recordings must be supplied separately for a new evaluation. The historical regression export keeps its original harness and saved artifacts; `operator export` supplies this current result note separately.
