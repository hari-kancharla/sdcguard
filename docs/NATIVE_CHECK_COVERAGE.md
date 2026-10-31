# Engine pass vs actually covered

Owner: Hari Krishna Kancharla.

This offline command asks which native checks have a recorded result and complete eligible comparison scope for each declared request. A configured flag, an empty error log, checksum production, scheduler admission or a rising counter does not establish that a request was compared. A native result is retained separately from coverage. Missing evidence holds the release policy. It does not establish engine corruption.

The first runtime adapter reads an existing owner-result directory directly. It supports the recorded vLLM 0.32.0 NIXL pull lease backport with one cache group and TP, PP, DP, DCP and PCP all one. It uses exact descriptor identities to join the independently recorded ordinary client batch to the consumer's explicit native `lease_check` events. It checks the named source fingerprints, recorded transfer operations, expected and observed remote ranks, consumer stream, causal sequence and healthy lossless recording prefix. It does not require a new ownership proof. An old lifetime invariant can be satisfied by producer retention without any native check, so that invariant alone cannot supply a passing native result.

## Capture and verify

```sh
sdcguard native-checks capture --owner-result /path/H100-owner-result --out /new/check-coverage
sdcguard native-checks verify /new/check-coverage
```

The default policy requires both lease and checksum. `--require lease` selects a distinct policy that requires lease only. Repeat `--require` to name multiple checks. Every policy has its own identity in the packet. Policy schema `sdcguard.native-check-policy.v2` also requires every request outcome to be `satisfied`. Unknown, recovered, refused, correctly rejected and violated outcomes remain distinct and hold this policy. Older policies are not silently reinterpreted. Supported names are `lease`, `checksum`, `bootstrap` and `canary`; unsupported observations stay unverified. A capture with no checksum observation cannot satisfy a checksum policy.

The packet retains the exact original `RESULT.json` and `ordinary-evidence.zip`, the policy and evaluator identities, and JSON, text and HTML reports. `verify` recomputes the result and all report bytes from the retained raw inputs. Changing a derived verdict and recomputing its outer hash does not pass verification. Output directories must be new, outside the input directory, with existing parents. Inputs are bounded to 512 regular archive members, 512 MiB expanded total and 64 MiB per member. Unsafe, duplicate, encrypted or non-regular archive members are refused. No archive path is directly extracted.

Both commands exit zero only when the selected release policy passes. They exit two for a hold, unsupported evidence or an invalid packet. Native mismatches and missing coverage have separate report fields and reasons; both hold a required check. A successfully written packet can legitimately exit two. `--json` prints the structured result. Neither command imports a cloud SDK, opens a network connection, starts an engine or runs a workload.

## What each request shows

- Native observation: passed, mismatch, unverified, admitted, activity or scheduled. Admission and activity do not mean a native comparison passed.
- Coverage: complete, partial, not covered or unknown, with expected units, compared units and every missing unit.
- Effective check result: passed, mismatch or unverified, including the reason and retained event references.
- Exact client join and separately retained request outcome. Recovered and refused remain distinct.
- Client-sent lease fields and exact prefill-to-decode descriptor handoff when recorded. Router forwarding remains unknown without a recorded ingress and egress relation.

Lease units are remote request leases by rank. They are not checksum comparisons over every layer or model cache group. The recorded one-group configuration is context, not evidence that KV contents were checked. Eligible expected units must be nonempty and established independently of the units reported as compared. Extra and duplicate normalized units are rejected. An empty comparison plan cannot produce a complete passing check.

A passing lease summary is insufficient by itself. The supported single-rank path requires one detailed comparison for that rank, on the same consumer stream and between transfer completion and the aggregate check. Its expected nonce and slot must match both the recorded attempt and the independently joined client descriptor. A matched cause requires a completed read and equal observed and expected nonces. Detail operations and remote identity must also agree. Missing or contradictory details make effective coverage unverified even when the native summary says passed.

## Saved ordinary run

The retained October 29 R3 owner result contains 19 logical ordinary prefill/decode requests, 38 client stages and 19 uniquely joined explicit passing native lease checks. Its archive SHA256 is `edb494c39cc5cf8bfa333d0294b390c0bc9455c5ca2dae5343f2ccb9e8a25b6b`. All 19 checksum results are unverified because no checksum observation was recorded. Zero of 19 requests are verified by both checks.

That owner run remains incomplete because its capture teardown evidence was missing. This is independent of the recorded lease positives. Even a lease-only policy holds that run. Its original result and status are retained unchanged. This command does not promote the run to a successful release or claim it recorded all later measurement traffic. There were no completed measurement segments in that result. The supported denominator is the declared `ordinary/ordinary-capture` client batch.

## Source basis and limits

The runtime lease adapter binds the nine fixed file hashes in installed `case_data/engine-pins.json`, for the exact proposed [lease-check backport](https://github.com/vllm-project/vllm/pull/62065/commits/42b8f230676796445107ec3f50241e871fe9b9b6). The passive recorder records the native verifier's result and the expected/observed ranks in `src/sdcguard/lifetime/vllm_nixl.py`, lines 556 to 578. The consumer calls that emit the result are the evidence basis, not a model response or a producer checksum value.

Separate CPU controls exercise pinned SGLang and vLLM functions with synthetic ordinary metadata. They show the limits of interpreting native signals as full coverage. They do not establish three engine defects: SGLang admission is not an assertion that a checksum was compared, a canary activity counter is not a per-request pass, and a vLLM aggregate return value concerns its compared groups. Their exact sources and hashes are retained with those controls. They cannot qualify an actual runtime release.

Full immutable engine identity, authenticity, loaded weights, ownership, read rights, physical reads, answer correctness and isolation remain unknown. This is an execution-coverage report for selected native checks, not a safety certificate. Native SGLang runtime collection and checksum runtime collection are not implemented in this delivery. No engine source is backported or altered by this command.
