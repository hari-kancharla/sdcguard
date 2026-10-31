# Compare the same named contracts

Owner: Hari Krishna Kancharla.

The upgrade gate compares two portable evidence packets. It verifies captured inputs, recomputes both reports with the installed rules, and refreshes the existing CPU component checks when selected. It does not start a server or run a serving workload.

```
sdcguard evidence collect --config baseline.json --components --out baseline-evidence
sdcguard evidence collect --config candidate.json --components --out candidate-evidence
sdcguard upgrade compare baseline-evidence candidate-evidence --out upgrade-result
sdcguard upgrade verify upgrade-result
sdcguard upgrade policy
```

Both configs use the normalized static schema in [CONFIG_SAFETY.md](CONFIG_SAFETY.md). Supply each build's own saved startup logs and exact source identity. See [EVIDENCE_REPORT.md](EVIDENCE_REPORT.md) for optional recorded-event inputs. Output directories must be new and their parents must exist.

The result has a JSON report, readable HTML, both complete captured input packets, and a file manifest. It can be moved and verified without the original paths. The saved report has a `captured_scope_verdict` for deterministic saved facts and grants no current gate approval. Each compare or verify command independently refreshes selected CPU profiles and returns an effective `scope_verdict` with its current execution receipts. Missing fresh proof blocks a clean gate as inconclusive; it never erases a known candidate failure or makes unchanged captured bytes corrupt. The combined artifact is limited to 512 regular files, 64 MiB per file, and 512 MiB total, including its manifest. Unsupported size is inconclusive. Local hashes check consistency, not authenticity.

| Exit | Scoped verdict | Meaning |
| --- | --- | --- |
| 0 | fine | The installed policy permits preservation of the named contracts. |
| 1 | not fine | A verified candidate contract fails, or an explicit preservation clause fails. |
| 2 | inconclusive | Verification, coverage, identity, source mapping, or comparison facts are missing. |

A known failure remains a failure when another comparison fact is missing. The report separately says whether that failure is a regression, was already present, or cannot be attributed. A bad baseline repaired by the candidate is shown as a remediation. The default policy does not approve an expected-negative baseline.

The installed `preserve-named-contracts.v1` policy preserves required rule identities, model and immutable revision, context, page geometry, pool class, role, topology, connector, and protocol. Source changes need an exact installed mapping. The first mapping covers only the saved vLLM 0.32.0 base with its exact proposed-fix overlay added or removed. Ordinary engine version/build upgrades remain inconclusive until a separate transition is reviewed. Unknown config changes also remain inconclusive, even when both standalone reports appear clean. The policy has no caller-supplied ignore list.

An ordinary pool target may change when each worker's independently bound startup contract still covers the same context. For example, with context 512 and block size 16, vLLM targets of 33 and 65 ordinary blocks preserve usable capacities of 512 and 1024 after the null block. A target of 32 has usable capacity 496 and fails that contract. These numbers describe a synthetic local control, not a GPU measurement or concurrency promise. A SGLang configured ceiling alone does not establish its final pool.

The policy preserves enabled checksum settings and configured canary action. Removing a checksum, or changing canary action from raise to log or none, fails that explicit clause even when the native inventory correctly reports a known off state. Timeout changes and real-KV mode changes are inconclusive. Recovery, refusal, and correct rejection are distinct outcomes without a general safety ordering.

Every result says `runtime_upgrade_witness: not established`. Static inputs are declarations and saved startup facts. The two CPU checks execute pinned source fragments with controlled mocks; they do not execute a changed deployment build. A copied recording is the same evidence, not a new candidate run. Current recorded packets lack a reusable stimulus manifest, so changed recording identities remain incomparable. Actual model reads, GPU/DMA behavior, loaded weights, and unobserved kernel identities are outside these narrow contracts.

Re-verification uses the installed evaluator and refreshes selected CPU checks again. A changed evaluator, changed inputs, edited derived report, or unavailable current proof blocks verification. A local report is not an approval to deploy.

Whole evidence reports use the bounded evidence JSON capacity described in [EVIDENCE_REPORT.md](EVIDENCE_REPORT.md). This allows a complete recorded session to reach the same existing policy without dropping findings. Comparing a recording with itself exercises packet verification and policy plumbing; it does not validate a new engine version or establish a runtime upgrade witness.
