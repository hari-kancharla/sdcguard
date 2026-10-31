# sdcguard

Hi, I'm Hari. sdcguard checks whether each request in a prefill/decode inference setup was actually covered by the
engine's own KV checksum.

The KV-transfer checksum proposed for vLLM in the open PR [#62211](https://github.com/vllm-project/vllm/pull/62211)
compares the blocks a decode server loads from the prefill server. By design, prompt tokens the decode server reuses
from its own prefix cache aren't part of that load, so they sit outside the comparison while the check still returns
success. sdcguard's vLLM plugin records each checksum call and the chunks it compared, and sdcguard joins each client
request to those records. It tells you, per request, how much of the prompt was compared. A request passes only if
its whole prompt was covered. Otherwise it gets HOLD, with the number of unchecked tokens.

In the saved H100 run below, all 153 checksum calls returned success, but 99,968 of the 109,790 prompt tokens (91%)
came from the decode cache and were outside the check for that request.

sdcguard started as a GPU math checker. That's version 0.4.0 on PyPI (`pip install sdcguard`), and its code and
results are at the [v0.4.0 tag](https://github.com/hari-kancharla/sdcguard/tree/v0.4.0). The checksum gate on this
page isn't on PyPI yet, so install it from a clone of this repo. A clone still reports version 0.4.0.

## Replay a real H100 run

```sh
git clone https://github.com/hari-kancharla/sdcguard && cd sdcguard
python3 -m venv .venv && . .venv/bin/activate
python -m pip install --no-deps .
sdcguard gate tests/data/checksum_owner_3a2ad847 --require checksum
```

This replays, on your machine, a real recording from an H100 run. It needs Python 3.10 or newer and no other
packages, and it doesn't start an engine, make a network request or use a GPU. The recording stays in the source
checkout and isn't included in the wheel or source distribution. CI runs the test suite on Python 3.10 and 3.12.

You should get **HOLD**, with **9 of 153 requests passing**. All 153 native checksum calls returned success, but 144
requests reused decode-cache tokens outside the comparison. The run has 153 prefill/decode pairs (306 HTTP stages)
in nine shared-prefix groups. The 9 that pass are the first request in each group, which loaded its whole prompt
from the prefill server. Of the run's 109,790 prompt tokens, 9,822 were inside a recorded checksum comparison and
99,968 were reused from the decode cache without one.

This is a coverage result, not evidence of corruption. In this run every cached block was compared once, when it was
first loaded from the prefill server. The gate counts per request and doesn't credit that earlier check, because
nothing checks the block again when a later request reuses it.

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

`transfer-check` exits 0 when every planned transfer was fully compared, as it does for the saved run. That exit
code covers the transfers only. It doesn't replace `gate --require checksum` and can't approve a full-prompt
release. Its JSON includes the unchanged full-prompt gate, every planned request, source identity, input hashes and
any gaps. Missing receipts, unknown builds, broken joins and incomplete recordings all hold. A
request that provably transferred nothing is `NOT_EXERCISED`, never a pass, and a recording where nothing was
exercised holds. This first profile needs prefix caching on and recorded native accounting in both engine roles.
Anything else holds as unsupported.

The consumer-side checksum arrays aren't kept, so this verifies that the engine's comparison was complete, not the
checksum values themselves. It makes no claim that every layer was checked, or about ownership, physical reads or
cross-machine safety. By design, the pinned engine checks only the load span, in
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
The command always exits 2, because the full-prompt HOLD still applies. Separately, it reports transfer
preservation for one exact engine upgrade, `29d2f4e` to `1a0c6954` (below), but only when both recordings pass
their own source and coverage checks and have matching workloads and effective configuration. Comparing a recording
with itself is labeled as the same evidence, not an upgrade. Changing paths, packet hashes or run IDs doesn't make
it a different engine build, and an unsupported revision stays unverifiable. All of this checks that the local
evidence is consistent. It doesn't authenticate where the evidence came from or explain why two runs differ.

On Oct 31, 2026 I recorded the same workload on both builds, each on a single H100, with `--gpu-blocks 8192` on
both. Replay the comparison from a source checkout with:

```sh
sdcguard transfer-compare tests/data/checksum_owner_9b42f5ab tests/data/checksum_owner_b340ee03
```

The output starts with `Offline upgrade comparison: HOLD`, then `Transfer preservation: PASS` with 153 preserved,
0 regressed and 0 unverifiable. Each build on its own gives the same full-prompt HOLD as the saved run above: 9 of
153 requests pass, 9,822 tokens are compared and 99,968 decode-cache tokens are outside the comparison. The command
exits 2. The `kv_checksum` module files are identical in both builds, but the PR head merges in 163 newer commits,
almost all from vLLM main, which change 763 files, including the v1 scheduler, the NIXL worker and the precompiled
wheel. The one new scheduler branch (deferred loads, used only with HiSparse) is off in this profile. So this shows
transfer coverage held across the rest of that upgrade for this workload. It doesn't show the decode-cache gap
closing.

## Record your own run

Recording needs a prepared Linux x86_64 host with one H100, the CPython 3.12 environment at `/opt/sdcguard-python`
and the pinned editable vLLM install at `/opt/vllm-checksum`. Installing sdcguard doesn't set these up, and this
repo doesn't include a script that builds that host yet. A CPU preflight checks the interpreter, headers,
dependencies, engine source, compiled payload, model files, transport and cleanup support before anything is
recorded.

| Component | Supported recording profile |
| --- | --- |
| Engine | `sawsa307/vllm` at `29d2f4e03740acb14439ce22d24b06d8095d17d3`, from the open checksum PR [#62211](https://github.com/vllm-project/vllm/pull/62211) |
| Compiled payload | Exact base wheel at `2b678ac739166bd0356883048673bbd1c34a3b0e`; [source and dependency lock](src/sdcguard/native_operator/profile_data/artifacts/ENGINE_BUILD_LOCK.json) |
| Model | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, with the [recorded file hashes](src/sdcguard/native_operator/profile_data/artifacts/MODEL_PIN.json) |
| Execution | Two engine instances on one host, NIXL pull, eager `uni`, prefix caching on, 16-token blocks, same-host UCX profile with peer error handling disabled |

The second supported build is
[`1a0c6954a9d881c81f85fc1a6cbf51c76f3fcba8`](https://github.com/sawsa307/vllm/commit/1a0c6954a9d881c81f85fc1a6cbf51c76f3fcba8),
the head of the same PR as of Oct 31, 2026, with its matching base wheel at
`255431b5ed8777b7524c65e9d3dfadeb2319abfc` and its own [source and dependency lock](src/sdcguard/native_operator/profile_data/builds/1a0c6954a9d881c81f85fc1a6cbf51c76f3fcba8/artifacts/ENGINE_BUILD_LOCK.json).
The checksum functions are unchanged between the two, so this compares the surrounding engine upgrade, not a new
checksum fix. HiSparse host blocks must be recorded as unset and deferred asynchronous loads as disabled.
Missing facts hold the check.

Use the prepared host's Python environment. The output directories must not exist yet:

```sh
# CPU preflight only. No server or GPU work.
sdcguard operator run --require checksum --out checksum-preflight --preflight-only

# Owner-run GPU capture: starts both engines, records ordinary traffic and verifies it.
sdcguard operator run --require checksum --out checksum-run
```

Pick the second build with `--engine-revision 1a0c6954a9d881c81f85fc1a6cbf51c76f3fcba8`. This only selects the
verification profile; it doesn't install or change the engine. Short SHAs and branch names are refused. For a
paired run, pass `--gpu-blocks 8192` on both builds. That passes `--num-gpu-blocks-override 8192` to both engines
in each run and requires that exact count at startup, because automatic memory profiling can otherwise give the
two builds different capacities and make a strict comparison unverifiable.

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

Evidence over the gate's size limits is refused, not truncated.

The two-build comparison has been checked on real recordings for one pair, `29d2f4e` to `1a0c6954`, with one
recording per build, each on a single H100. That's one sample, not runtime qualification of the newer build.

## Earlier work

On vLLM bug [#61229](https://github.com/vllm-project/vllm/issues/61229), where a decode server can read KV blocks
the prefill server has already freed and reused, every held run on the unfixed build returned a wrong answer as a
normal response, and an earlier recorder flagged the stale read each time. With the proposed lease fix
([#62065](https://github.com/vllm-project/vllm/pull/62065)), vLLM rejected the stale read in all five replays on an
H100 and the client got the right answer. See the [H100 result](docs/H100_OBSERVER_RESULT.md).
