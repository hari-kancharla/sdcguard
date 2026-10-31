# Product assessment and build gates

Updated October 27, 2026.

## Actual product and intended distinction

The product direction is a vendor-independent acceptance and regression gate for inference state: exact deployment/configuration → passive lifecycle evidence → bounded rules → portable, inspectable decision → retained regression case. Arithmetic can be correct over the wrong request's KV data. The current proven deployment case is the pinned vLLM #61229 experiment. One shared destination-retirement rule now evaluates original pinned vLLM and SGLang component functions on CPU through the installed `sdcguard lifetime conformance --json` command. This is component evidence; SGLang deployment coverage remains unproved.

The company-sized hypothesis is an independent, versioned set of proof obligations and adapters that keeps the same meaning as operators change engines, configurations and source versions. Vendor patches are valuable dependencies, not competitors to dismiss. Independence, repeatable acceptance and evidence continuity are possible product advantages; neither an exclusive moat nor technical novelty is established by describing them.

## What is strong

- A concrete serving-state failure and preserved stock/backport contrast exist, with clean controls. Historical outputs are separate from revised observer validation.
- Conservative bookkeeping uses stream/operation/generation identity and rejects missing coverage. Recovered, refused, correctly rejected and unknown remain distinct observations.
- The new workflow now includes packaged regression assets, offline packet verification, passive collection instructions, assessment JSON/HTML and bound performance receipts. Targeted CPU checks are evidence for these local functions; a combined independent review/full run is still required for each new source revision.
- Model recovery made real progress: the persistent cache's sole commit-named snapshot and all ten file hashes were verified read-only against immutable upstream file identities. Future replays record an explicit model/tokenizer revision; original artifacts still say main.
- The shared CPU rule requires complete, attributable writer fences before observed destination retirement. It has no engine branches. The adapters preserve vLLM allocation-generation and SGLang protocol-hold semantics rather than equating them. Independent deletion probes and per-operation coverage regressions reject missing evidence as unknown.

## Why now, with limits

[vLLM #61229](https://github.com/vllm-project/vllm/issues/61229) and [the proposed PR #62065](https://github.com/vllm-project/vllm/pull/62065) are open as checked today. The saved local backport must not be described as a released official fix, nor does issue status prove every later release is vulnerable.

[Cloudflare's report](https://blog.cloudflare.com/kv-cache-exposure-remediation/) and [SGLang's update](https://www.lmsys.org/blog/2026-10-26-cf-security/) establish a production incident involving GLM-5.3 Workers AI, disaggregated prefill/decode and Mooncake. Cloudflare reports 155 affected responses in August 27–September 9, 2026 and successful HTTP responses carrying cross-request data. Those incident facts are not this project's detection evidence.

[Micron's September 30, 2026 prepared remarks](https://s25.q4cdn.com/621799436/files/doc_financials/2026/q4/Q4-FY26-Prepared-Remarks.pdf) describe tight memory supply, rising context/concurrency demand and growth in KV-cache offload, and forecast continued supply constraints. These are manufacturer observations/forecasts. The inference that memory efficiency becomes more valuable is reasonable; a claim that every operator is forced into sharing/disaggregation is not established.

## What is weak or unproved

[GRIEF](https://arxiv.org/html/2605.11202v1) already includes multiple engines, replay confirmation and structural KV forensics. A second adapter alone does not prove differentiation. The operational acceptance workflow must be demonstrated end to end and compared honestly.

[TensorRT-LLM's pinned lifecycle design](https://github.com/NVIDIA/TensorRT-LLM/blob/0a03c3ed057fe8c488e221413640ca87f64e99d2/tensorrt_llm/_torch/disaggregation/lifecycle/README.md) already separates physical-operation ownership, retirement evidence and logical outcomes. [Dynamo KVBM](https://github.com/ai-dynamo/dynamo/blob/45e30e07a4f4cef37ab7bfa739f7bee0c89b3057/lib/kvbm-engine/docs/architecture.md) supplies distributed cache coordination. Vendor-independent acceptance remains a hypothesis worth testing; a claim that vendors will never build shared lifecycle infrastructure is unsupported. These sources do not add a third supported engine here.

Current source matching supports only pinned layouts. An unknown upgrade is inconclusive until its adapter contract is reviewed. The product does not automatically prove every future upgrade safe.

SGLang's deferred-release path protects destination buffers from outstanding writes. The existing vLLM source-lease rule concerns producer lifetime and admission after reads. The new shared destination-retirement rule is separate and supports only source-fragment CPU evidence. vLLM needs actual registered handles and individual DONE observations beyond existing runtime receipts; cross-process resource mapping remains unsupported. SGLang's metadata-slot protocol hold is not a physical KV allocator generation. Timeout, configured policy and absent records cannot become completion proof.

The saved harness is not verified minimal. Revised hooks have not been GPU-validated here. Real supported-config overhead is unmeasured. The recovered shared-cache pin is corroborative historical evidence, not an immutable pin captured by the original run.

## Next work and concrete acceptance gates

1. Complete the integrated local workflow. A clean-installed wheel must export/verify the case without a checkout, prepare observation instructions without activation, save and verify a portable scoped assessment, and bind ordinary perf output without JSON edits. Tampering, unsupported versions, missing facts and interrupted captures must not pass.
2. Preserve and independently review the shared-rule component proof using exact pinned vendor source and existing ordinary CPU handler fixtures. Extend observable facts only within the authorized passive scope. No engine-specific exceptions in the rule, new unsafe flow or retry/reused-ID forcing. Missing corresponding facts must produce unknown. The current command proves only the scoped handler property, not deployed isolation.
3. Extend to a third engine only when its real protocol and observable identities support the same obligations. Count demonstrated semantics, not adapter names.
4. Validate the existing saved condition on the pinned snapshot, revised hooks, both controls and five replays; then validate deletion candidates without changing the failure class. New failure orderings and gated forcing remain excluded.
5. Measure actual observer overhead against the same existing supported serving path with unchanged engine source and matched workload. Receipts must preserve failures and declarations; fake HTTP timing is not NIXL performance evidence.
6. Full Python 3.13/3.10 CPU suites, a clean install and before/after source hashes for every revision. Keep all failed attempts and unsupported environments explicit.

## What this does not claim

No universal isolation guarantee; no automatically safe future engine version; no production leak discovered by this package; no naturally defective GPU diagnosed by this software case; no validated minimality or GPU performance without the owner evidence; no inference-state coverage inferred from arithmetic checks; no remote attestation inferred from a manifest; no exclusive novelty from cross-engine adapters or replay.
