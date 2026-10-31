# Pinned support and evidence matrix

| Component | Exact scope | Evidence status |
| --- | --- | --- |
| Source baseline | Snapshot 576ac8b; all 3,973 manifest files matched | Verified locally |
| Engine | vLLM 0.32.0, commit 5df63a7203cdb9328d824a27b682e77c7b9928be | Relevant complete files fetched from pinned upstream source |
| Fix | Exact runtime-only backport of proposed PR #62065 (OPEN / unmerged on October 27, 2026) at 42b8f230676796445107ec3f50241e871fe9b9b6; patch SHA256 a669acdbc188d59439ca3ba74bba9a18447578b997be3f61278d45fd6a5f337d | All hunks applied to pinned base with git apply; nine complete relevant file hashes packaged. No claim about other commits or runtime enablement from source presence alone. |
| Model | Qwen/Qwen3-0.6B | Original run artifacts record main. Subsequent read-only shared-cache recovery verified revision c1899de289a04d12100db370d81485cdf75e47ca and all ten files against immutable upstream identities. Future replays pin that snapshot. The historical association is corroborated, not an original run manifest. |
| Transport/topology | NIXL 1.4.1, UCX; two eager instances on one H100/host; TP/PP/DCP/PCP 1; one cache group; equal block layouts; prefix caching off | Historical result only. No in-flight RDMA, multi-host, mixed-cache, partial-prefix, or retry/reused-id claim. |
| Existing condition | Saved read_submit hold released on designated block's next-owner free; timeout 14 seconds | Exported unchanged; wrapper exposes no search arguments. |
| Revised evaluator/hooks | Source lifetime and scheduler admission | CPU validation is separate from owner-run GPU validation. Actual attention consumption is not observed. |
| Shared-rule component conformance | vLLM 5df63a7203cdb9328d824a27b682e77c7b9928be allocation/READ/DONE and SGLang 41fb719cf25966e6155f3e3d91f1157bc1a15123 matching-generation ACK/deferred release | Original pinned functions with controlled CPU mocks; one generic destination-retirement rule. Offline installed command includes source/file pins, licenses and receipts. source_fragment_cpu only; physical_write_safety unknown. This adds no SGLang deployment support. |
| Terminal outcomes | recovered: attributed rejection then exact reference output; refused: attributed rejection then terminal client error | Client observations, separate from internal recomputation completion. Policy alone proves neither. |
| Operator | sdcguard assess deployment.json | Scoped evaluator and optional existing metrics readiness check. No isolation probes, replay or remediation. Fine covers recorded attempts, not future traffic. |
| Python lanes | 3.13 / CPU PyTorch 2.14.1 / Triton 3.8.0; 3.10 / CPU PyTorch 2.14.1 | These are the tested native ARM64 CPU backend pins. Exact status, skips and source hashes are recorded per test run; new revisions need full validation. ARM64 PyTorch 2.8.0 failed 13 unchanged baseline tests and is excluded from this validated CPU matrix; QEMU changes process-name semantics. CPU evidence does not establish GPU execution. |
| Minimality | Original workload exported with saved condition | NOT ESTABLISHED. Requests/tokens need owner-run deletion verification. A topology or timing change introducing another behavior is out of scope. |
| Check overhead | Supported configuration with revised recorder | NOT MEASURED. Experiment wall time is not throughput or p99. Older arithmetic-check results do not transfer. |

Source matching checks nine complete relevant files, not class names or patch filenames. Changed/unknown sources are inconclusive. It does not attest an unrelated remote server. Recorded lease-check facts establish execution separately.

The deployment config declares a model revision; this check does not independently verify live model weights. No whole-deployment correctness or security certificate is issued.
