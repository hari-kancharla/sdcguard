# Lifetime checks for inference state

`sdcguard lifetime` finds orderings of a serving stack's internal events in which a request reads or writes KV memory it no longer owns, or uses state before it is ready. These bugs leave the arithmetic correct and the cache keys correct, and still compute on the wrong data:

1. Request A starts transferring its KV state to a decode worker.
2. A times out, and the decode worker gives A's blocks to request B.
3. A's unfinished transfer writes into those blocks.
4. B runs on partly overwritten state, with HTTP 200.

Z.ai described this sequence in GLM-5 serving (April 2026: about 0.1% abnormal outputs, several weeks to find, not reproducible by replaying single requests). vLLM issue 61229 describes the pull-side version: a producer's lease expires, it reuses the blocks, and a delayed consumer reads the replacement contents.

## What it checks

An independent ledger records every allocation of a KV region with a generation number, so "block 42 holding A" and "block 42 later holding B" are different handles. Asynchronous operations (writes, reads, loads) refer to handles and stay outstanding until they complete or the transport confirms they stopped. A cancel request alone does not stop an operation. Four rules are checked as events arrive:

| Rule | Broken when |
| --- | --- |
| R1 stale_handle | an operation is submitted, or a descriptor accepted, for an allocation that was freed or reallocated |
| R2 reuse_outstanding | a region is reallocated while an older operation can still write into it or read from it |
| R3 consume_unready | a request runs on state that is not current, not its own or shared, still being written, or holding another request's content |
| R4 stale_completion | a completion is applied to an allocation its operation did not refer to |

Allowed outcomes are not violations: a request that fails after losing its lease, a rejected transfer, a free with nothing outstanding. Permitted sharing (a prefix-cache hit) is declared and allowed.

## How it searches

A protocol model (or, through an adapter, a real server) lists the actions that may happen next: a worker issuing a transfer, a transfer completing, a timeout firing, time passing. The normal order takes the first one each time. The explorer departs from it on purpose: choosing the k-th action instead costs k departures, and it searches every schedule within a budget of departures, fewest first, so the first schedule it reports for a rule is the shortest way to break it. Only conflicting choices branch (actions that touch the same region, request or operation; the clock conflicts with everything). A finding is reduced to the departures it needs and to the events that caused it, and the report shows what happens next if nothing stops it.

```
sdcguard lifetime list
sdcguard lifetime explore pd-lease               # exit 1: a rule broke
sdcguard lifetime explore pd-lease --variant fixed --max-delays 5 --json out.json
sdcguard lifetime selfcheck                      # every buggy model breaks a rule, every fixed one does not
```

## The models

| Model | Protocol | Buggy variant | Fixed variant | Rules the buggy variant breaks |
| --- | --- | --- | --- | --- |
| pd-abort | prefill side pushes KV into decode blocks | decode frees a timed-out request's blocks at once | reclaim only after the prefill side says no write was issued, or every issued write completed or stopped | R1, R2 |
| pd-abort, release-timeout | the fixed handshake with a liveness fallback | decode frees the blocks anyway after a drain timeout | (this is the variant under study) | R2 with the write's cancel already requested |
| pd-lease | decode side pulls KV from leased prefill blocks | the lease expires under a delayed or in-flight read | the consumer pins before reading; a late pin gets "expired" and the request fails | R1, R2 (and R3 as the consequence) |
| pd-notify | completion notifications and control messages on separate channels | notifications matched by a reusable request id | notifications carry the transfer id and generation; stale ones are dropped | R4 |
| offload-load | host tier to GPU load before the forward pass | forward may start once the load is issued | forward waits for the load's completion | R3 |

Each model lists what it assumes (for example, that a write's bytes land when it completes and never after the transport confirms it stopped) and what it does not cover (partial landing across blocks, CUDA graph replay). The report prints both.

The release-timeout variant is the shape of a real design choice: SGLang v0.5.22's decode side defers releasing an aborted request's KV until the prefill side has drained, and for device destinations releases it after a timeout anyway, with a warning (disaggregation/decode.py, `resolve_deferred_releases`); for host destinations the same code notes that "A timeout cannot prove that a remote write has stopped." The explorer shows the ordering the fallback permits: a cancelled write still in flight when the timeout fires lands in the next request's blocks. Whether a write can still be in flight after the timeout on a given transport is exactly what a run on that transport has to settle; the model only says the protocol does not rule it out.

## Searching real code, with no model

A model written after studying a failure partly supplies the answer, so the models above are development references. The search that matters runs on real code: `realsearch.search(make, run)` is given only a way to build the instrumented system and run it. It is not told where any bug is.

1. It runs the code in normal timing and records every checkpoint (a point where a thread could legally be slow) and which blocks the same thread touches right after it.
2. It makes candidate delays from two sources. From the lifetime rules: hold a checkpoint until one of its blocks is freed, reallocated, run on by another request, or until an operation landing in it completes. From the normal run itself: an event of another thread on one of those blocks that came later (an allocation, a free, a write, a completion), held for so that the observed order flips. Most real timing bugs need one or two such flips. Candidates whose flipped order never appeared in a normal run go first.
3. It runs the code afresh under each delay (a delay whose condition never comes expires) and checks the rules on what was recorded. A new finding is replayed until three runs in all, and counts only if at least two of them break the same rule; the rest are listed as unconfirmed. Findings are grouped by rule and shape, keep every delay that produced them, and prefer, as the shown cause, a delay whose condition was actually reached.
4. Liveness: once the delays lift, the run must still finish. A run that raises, or does not finish in time, is a liveness finding.
5. Every finding carries its delays as a holds file. `sdcguard lifetime replay-toy` (or `realsearch.replay`) reruns it, so a finding becomes a regression check that the fix keeps passing.

Optional extras: pairs of delays (`depth=2`) and runs with short random pauses at every checkpoint, as a check on what the targeted delays miss.

`sdcguard lifetime search-toy pull`, `push` and `wakeup` run it on three small real implementations (real threads, real timers, a byte store the transfers really copy). On the buggy ones it finds, with one delay each, the stale read after a lease expires (R1, and the request then runs on the other request's bytes), the late write after an abort (R1 and R2), and a lost wakeup that hangs the request (liveness); on the fixed ones it finds nothing. These implementations were written knowing the bugs, so this shows the search does not need a model; it does not show discovery in someone else's code. That is what the vLLM wrappers are for.

A real server numbers its blocks and requests differently each run, so for it the candidates name blocks relative to the held point: the i-th block of the descriptor the held read uses, or of the held request (`candidates(events, symbolic=True)`), resolved against what was recorded when the hold starts. `experiments/lifetime_pd_check.py` runs exactly this on a real vLLM 0.32 prefill/decode pair: record a normal run, make the candidates from it, and try them one at a time under ordinary load on the prefill side. The candidates come from the recording, not from an issue report, but the experiment itself chooses the lease length, the load and a hold longer than the lease, so reproducing a known defect this way is an automatic search reproducing it, not discovery. Each run is reported in separate levels (lease expired; stale access recorded; output changed; another request's secret word in the output), with two controls (the same load with no hold, and a hold shorter than the lease), and vLLM's own late-read counter as corroboration of the ordering only.

What Python hooks miss: a Python call returning does not mean the GPU or transport work it started has finished. Recording the actual completion of transfers and the stream dependencies between copies, allocation, release and reuse is the next capability; until then a finding on paths where that matters needs byte evidence from the device.

## What a finding means

A finding in a model says the protocol as modeled allows that ordering. It is a candidate. It becomes a finding about a server only when the same ordering is driven in the real implementation and the effect is observed there, on the real transport. Simulated, transport-confirmed and serving-impact results are kept separate.

## Next: real servers

The adapter interface records the same events from a running engine's hooks (descriptor issuance, transfer submission, completion, cancellation, lease expiry, block release and reuse, admission) and can hold one event while others proceed, so the explorer's schedules are driven in the real code. The first target is vLLM's NIXL prefill/decode path. Holding must only produce orderings the real transport allows; an impossible order is a harness bug, not a server bug.
