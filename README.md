# sdcguard

Hi, I'm Hari. sdcguard checks whether each request in a prefill/decode inference setup was actually covered by the
engine's own KV checksum.

vLLM's KV-transfer checksum (the open PR [#62211](https://github.com/vllm-project/vllm/pull/62211)) compares the
blocks a decode server loads from the prefill server. When the decode server reuses prompt tokens from its own
prefix cache, those tokens can sit outside that comparison while the check still returns success. sdcguard joins
each client request to the engine's recorded checksum results and tells you, per request, how much of the prompt
was compared. A request passes only if its whole prompt was covered. Otherwise it gets HOLD, with the number of
unchecked tokens.

## Replay a real H100 run

From a source checkout:

```sh
python -m pip install --no-deps .
sdcguard gate tests/data/checksum_owner_3a2ad847 --require checksum
```

This replays a real recording from an H100 run on your machine. It doesn't start an engine, make a network request
or use a GPU. The recording stays in the source checkout and isn't included in the wheel or source distribution.
The offline gate is tested on Python 3.10, 3.12 and 3.13.

You should get **HOLD**, with **9 of 153 requests passing**. All 153 native checksum calls returned success, but 144
requests reused decode-cache tokens outside the comparison. The run has 153 prefill/decode pairs (306 HTTP stages).
Of its 109,790 prompt tokens, 9,822 were inside a recorded checksum comparison and 99,968 were reused from the
decode cache without one.

This is a coverage result, not evidence of corruption. It's also per request: it doesn't tell you whether a cached
block was checked during an earlier request.

Add `--json` for request IDs, check results, reasons and token counts. Exit code 0 means the policy passed and 2
means HOLD, so this run exits 2.

## Check the transfers on their own

```sh
sdcguard transfer-check tests/data/checksum_owner_3a2ad847
```

This offline command checks something narrower: whether each recorded prefill-to-decode transfer got a complete
native comparison of the token chunks it was planned to move. The expected chunks come from the receive plan, tied
to the engine's cache accounting and the client's request, and the command checks the recorded comparison and the
order things happened in. A successful native return doesn't count as coverage, and neither does the overlap between the
planned and compared chunks on its own.

For the saved run this passes for all 153 requests, covering 688 chunk positions and 9,822 transferred tokens.
**The full-prompt gate is still HOLD: 9 PASS and 144 HOLD.** The 99,968 decode-cache tokens are still outside the
transfer comparison. Cache reuse on the prefill side is recorded separately. These numbers count token uses across
requests, not unique stored tokens or physical bytes.

`transfer-check` exits 0 only for the transfer check. It doesn't replace `gate --require checksum` and can't approve
a full-prompt release. Its JSON includes the unchanged full-prompt gate, every planned request, source identity,
input hashes and any gaps. Missing receipts, unknown builds, broken joins and incomplete recordings all hold. A
request that provably transferred nothing is `NOT_EXERCISED`, never a pass, and a recording where nothing was
exercised holds. This first profile needs prefix caching on and recorded native accounting in both engine roles.
Anything else holds as unsupported.

The consumer-side checksum arrays aren't kept, so this verifies that the engine's comparison was complete, not the
checksum values themselves. It makes no claim about all layers, ownership, physical reads or cross-machine safety.
The pinned engine checks only the load span, on purpose, in
[`add_recv`](https://github.com/sawsa307/vllm/blob/29d2f4e03740acb14439ce22d24b06d8095d17d3/vllm/distributed/kv_transfer/kv_checksum/scheduler.py#L280-L315).

## Compare two recordings

```sh
sdcguard transfer-compare baseline-run candidate-run
```

This takes two saved recordings and reruns the transfer check on both, offline. It matches requests by workload
slot and actual request parameters, checks that the model files and startup configuration match, and shows which
coverage was kept, which regressed and which can't be verified. A native mismatch after a verified baseline counts as a
regression. Missing evidence stays unverifiable and names the request. Changed cache reuse or comparison spans
don't count as kept coverage. Neither full-prompt decision changes.

`--json` gives both full-prompt gates, input hashes, build and recording identities, token counts and every slot.
In this release the command always exits 2 (HOLD): only one engine revision is qualified so far, so there's no
qualified before-and-after build pair yet. Comparing a recording with itself can show equal coverage, but it's
labeled as the same evidence, not an upgrade. Changing paths, packet hashes or run IDs doesn't make it a different
engine build, and an unsupported revision stays unverifiable. All of this checks that the local evidence is
consistent. It doesn't authenticate where the evidence came from or explain why two runs differ.

## Record your own run

Recording needs a prepared Linux x86_64 host with one H100, the CPython 3.12 environment at `/opt/sdcguard-python`
and the pinned editable vLLM install at `/opt/vllm-checksum`. Installing sdcguard doesn't set these up. A CPU
preflight checks the interpreter, headers, dependencies, engine source, compiled payload, model files, transport
and cleanup support before anything is recorded.

| Component | Supported recording profile |
| --- | --- |
| Engine | `sawsa307/vllm` at `29d2f4e03740acb14439ce22d24b06d8095d17d3`, from the open checksum PR [#62211](https://github.com/vllm-project/vllm/pull/62211) |
| Compiled payload | Exact base wheel at `2b678ac739166bd0356883048673bbd1c34a3b0e`; [source and dependency lock](src/sdcguard/native_operator/profile_data/artifacts/ENGINE_BUILD_LOCK.json) |
| Model | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, with the [recorded file hashes](src/sdcguard/native_operator/profile_data/artifacts/MODEL_PIN.json) |
| Execution | Two engine instances on one host, NIXL pull, eager `uni`, prefix caching on, 16-token blocks, same-host UCX profile with peer error handling disabled |

Use the prepared host's Python environment. The output directories must not exist yet:

```sh
# CPU preflight only. No server or GPU work.
sdcguard operator run --require checksum --out checksum-preflight --preflight-only

# Owner-run GPU capture: starts both engines, records ordinary traffic and verifies it.
sdcguard operator run --require checksum --out checksum-run
```

The model has to be cached already. Pass `--model-cache /path/to/the/pinned/snapshot` to point at it. The run sends
153 ordinary prefill/decode pairs in nine shared-prefix groups at concurrency 1, 4 and 16. It keeps `RESULT.json`,
`ordinary-evidence.zip`, the CPU preflight and the final `GATE.json`. Replay it later with:

```sh
sdcguard gate checksum-run --require checksum
```

## What PASS and HOLD mean

PASS means every selected client request joined to a recorded passing check with complete coverage, both engine
roles had the check enabled, and the identity and recording checks passed. Any unchecked reused prefix gives HOLD,
with its token counts. Missing facts, unknown builds and incomplete recordings can't pass. An incomplete native
comparison and a native mismatch stay distinguishable in the per-request evidence.

## Limits

This profile is checksum-only and same-host eager, not all vLLM defaults. It doesn't establish ownership, read
rights, physical reads, cross-machine correctness, recovery from peer failures or that a deployment is safe in
general. Recording lease and checksum checks together isn't supported. A newer head of the PR needs its own
qualification, because matching protocol version numbers don't prove compatibility.

The checksum gate admits up to 2,000,000 JSON lexical nodes in total. Each JSON document keeps its 1,000,000-node
limit, along with the byte, depth and archive limits. Evidence over a limit is refused, not truncated.

The offline comparison works today. Qualifying it on two independently built and recorded engines is still to do.
Older interfaces, and what their evidence covers, are in [earlier workflows](docs/EARLIER_WORKFLOWS.md).
