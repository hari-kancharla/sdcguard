# Why sdcguard

Primary sources opened October 19, 2026. **VERIFIED** below means the linked source was opened and supports
that description. It does not mean sdcguard has independently reproduced another project's results.
Product choices and buyer demand remain hypotheses to test.
Implementation status below was updated October 20 against the supplied `12e23f5` snapshot and local review
fixes. The patched candidate then passed the GPU test gate on an L4, an H100 and a B200 on October 20 (code
fingerprint 74b844ee7ca8, [results](../results/gpu_tests)), and later code passed the vLLM gate in real vLLM on
an H100 the same day (fingerprint bbdb51c58ac0, [results](../results/vllm_oct20)). The 10-GPU `sdcguard check`
sweep hasn't been rerun since October 18.

## The problem is documented; our detection rate is not

- **VERIFIED, Meta, July 22, 2025:** Meta reports approximately one silent-corruption fault per thousand
  devices. The post does not provide a time denominator or a GPU-only breakdown. Do not turn it into an
  annual GPU failure rate or a market-size calculation.
  [Meta engineering](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/).
- **VERIFIED, AEGIS, OSDI, July 2026:** ByteDance reports 18 incidents on 13 GPUs across 35 million GPU-hours;
  15 did not cause visible training failure. The 2-of-8 vendor-diagnostic result is a selected, previously
  screened set, not general DCGM recall. The paper also cites approximately 70% offline recall from earlier
  production work. Its effective system combines detectors and subsequent verification.
  [AEGIS](https://www.usenix.org/system/files/osdi26-lei.pdf).
- **VERIFIED, SDCHunter, OSDI, July 2026:** four studied GEMM-faulty GPUs passed BF16 and FP16 and failed
  different subsets of FP32, FP64, TF32 and INT8. The broader study also found faults first exposed by
  operations other than GEMM. Testing more arithmetic modes addresses one coverage gap, not every defect.
  [SDCs in the Wild](https://www.usenix.org/system/files/osdi26-zheng.pdf), Tables 2 and 3.
- **VERIFIED, Crusoe, updated June 30, 2026:** its customer support guide says a clean DCGM result does not
  exclude SDC and describes isolating nodes and collecting reproduction evidence.
  [Crusoe support](https://support.crusoecloud.com/hc/en-us/articles/52229568610715-How-To-Detect-Silent-Data-Corruption-in-Distributed-Training-Jobs).

These studies establish a reliability problem. They do not establish sdcguard's recall, a five-minute test's
probability of finding a defect, or willingness to pay for this product.

## Existing alternatives

Each row is VERIFIED from the linked primary source. Undated live documentation is dated by this review's
access date, October 19, 2026. Comparisons describe the documented mechanism, not an exhaustive absence claim.

| Alternative and source date | Documented mechanism | Consequence for sdcguard |
|---|---|---|
| [NVIDIA DCGM diagnostic plugin](https://docs.nvidia.com/datacenter/dcgm/latest/reference/diagnostics/plugins/diagnostic.html), updated September 10, 2026 | "It verifies matrix results and framebuffer data": S/D GEMM stress (180 seconds by default) with half, single or double precision selections, plus hardware-error monitoring. No bf16 or FP8 option or comparison tolerance is documented. How it verifies, from its source (4.6.1, [compare.cu](https://github.com/NVIDIA/DCGM/blob/64df9f894541e426e416131a9820cae97aa4dd81/nvvs/plugin_src/diagnostic/compare.cu#L30-L136)): the same GEMM of the same inputs many times per pass, each result compared with the pass's first result (tolerance 0.001, 1e-7 in FP64) and counted for NaNs; there is no independent reference answer. Its memory plugin and memtest do compare reads with known written patterns. | Do not claim DCGM only reads counters or never checks numerical results. Say how: by consistency between repeated GEMMs, so an error that repeats exactly is invisible to it, and only the GEMMs and settings it runs itself are exercised. Compare added workload/mode coverage and operational cost on the same hardware. |
| [NVIDIA NVSentinel](https://github.com/NVIDIA/NVSentinel), accessed October 19, 2026 | Connects health events to cordon, drain and remediation (GPU reset or reboot); those actions are opt-in, and by default it only monitors. | A dashboard, alerts and quarantine wiring already exist. A useful sdcguard integration supplies additional evidence to the operator's workflow. |
| [NVIDIA Cluster Readiness Engine](https://github.com/NVIDIA/cluster-readiness-engine), accessed October 19, 2026 | Runs real training and communication workloads across node groups and reports every bad node with a reason; it never cordons or modifies a node, and its introduction doesn't mention numerical correctness. | Running a small training job for acceptance testing is not unique. |
| [NVIDIA Fleet Intelligence reports](https://docs.nvidia.com/fleet-intel/reports/), accessed October 19, 2026 | Produces inventory reports, optionally with a Sigstore signature bundle. | A signature establishes provenance/integrity, not numerical health. Signing a report is not a defensible differentiator by itself. |
| [TorchTitan SDC replayer](https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/SILENT_DATA_CORRUPTION.md), accessed October 19, 2026 | Re-executes a deterministic forward/backward on the same device and compares hashes before the optimizer step; off by default, and by default only the first step after a start or restart. Its documentation notes the order-insensitive hash can miss permutations. (Our inference: a fault that repeats identically on the same GPU would pass a same-device replay.) | Public in-training detection already exists. Compare integration burden, coverage and cost, including deterministic requirements. |
| [Megatron Core rerun state machine](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.rerun_state_machine.html), accessed October 19, 2026; source read October 20 at [ebab48f](https://github.com/NVIDIA/Megatron-LM/blob/ebab48fc40c2b6f0a8768f765b9364656a2dccb9/megatron/core/rerun_state_machine.py) | Accepts caller-defined rejection/comparison functions with a tolerance, and reruns an iteration on the same GPU and on a different one; the default comparison handles scalar values such as the loss. Wired in by default: NaN and Inf in the loss and in each gradient bucket's norm; opt-in: spiky loss and large gradients (10 times the largest seen); a stats mode reruns every iteration to measure nondeterminism. Outcomes: correct result, transient error, persistent error. Its own disclaimer calls it alpha-level and says flagged nodes need standard diagnostics for a definitive diagnosis. | It is extensible beyond NaN/Inf or loss spikes. Do not infer its detection rate from a paper's NaN frequency. What it checks by default are scalar summaries (loss, gradient norms); sdcguard checks each matmul against its inputs and runs outside Megatron. |
| [SCOUT / lm-resiliency](https://github.com/LMResiliency/lm-resiliency), accessed October 19, 2026; [paper](https://arxiv.org/abs/2608.11034), August 11, 2026 | Public in-job failure localization (SDC, stragglers, hangs): between iterations it replays a sampled layer on equivalent replicas and finds the rank that disagrees by majority; it certifies clean checkpoints rather than measuring recovery. Integrations: PyTorch, TorchTitan, Megatron Core, DeepSpeed. Evaluated with software fault injection on 16 A100s. | Treat it as a serious competing approach. sdcguard's single-device checksum path has different coverage and integration tradeoffs. |
| [Google MaxText TPU checker flags](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/xla_flags_library.py), accessed October 19, 2026 | Exposes repeated-program checking for TPU execution. | Replay is an established alternative. An NVIDIA-specific checker is not vendor-neutral simply because its license is MIT. |
| [assay-gpu](https://github.com/GautamTalksDev/assay-gpu), accessed October 19, 2026 | Public GPU acceptance testing with checksums and signed reports. | This is close overlap. There is no completed side-by-side evaluation supporting a superiority claim. |
| [Clockwork TorchPass](https://clockwork.io/blog/torchpass-workload-fault-tolerance/), accessed October 19, 2026 | Describes workload fault tolerance including network failover and GPU migration. | Relevant competing spending, even where the advertised mechanism differs from sdcguard's numerical checks. |

The AEGIS paper describes an internal ByteDance integration; this review did not verify a public implementation
of AEGIS itself. That does not make all public in-workload detection absent. Nor does an unsuccessful search
prove that vLLM, SGLang or another framework has no related checking mechanism.

## The proposed contribution

**Hypothesis:** a deployable checksum sensor inside supported PyTorch and vLLM operations can produce useful
incident evidence for teams that cannot reproduce a numerical failure with their existing checks.

The current implementation offers single-GPU inline checks, layer/GPU localization, screening across supported
math modes, portable training/stress evidence, replay, a public results board and fleet integration. Serving
alerts do not currently save the original operands. None of those features alone establishes a moat.

AEGIS reports from its production deployment that it "did not successfully locate SDC incidents using checksums
accumulated from bfloat16 outputs"; all its confirmed checksum detections came from fp32 accumulators (section 8
of the [OSDI '26 paper](https://www.usenix.org/system/files/osdi26-lei.pdf)). Its fault-injection study on a
healthy machine shows why: over bf16 outputs, injected bit flips are buried in round-off noise (section 5.1.1).
sdcguard's default check works on bf16/fp16 outputs, so on AEGIS's evidence it should be expected to catch large
errors (exponent-bit changes, outputs that were never written) rather than the small ones real defects often
produce; its experimental precise mode checks fp32 outputs instead. AEGIS's deterministic detection, which compares a
fingerprint of an operation's output with the recomputation training already performs, flagged more production
incidents (12) than its sparsely sampled checksum detection and outlier warnings (6); the paper credits its wider
operator coverage (normalizations among them) and the sparse sampling (section 7.2). That motivates measuring precise mode against the default on the same
matmuls, not assuming it inherits AEGIS's results, and it means checksums alone shouldn't be expected to find
most real incidents. The revised noise
model already had an October 19 held-out run (fingerprint `1265b3e0a385`): no alarms in 17,424 checks, but the
A100 and L4 missed the pre-registered margin, and the T4 was incomplete. Precise mode remains experimental.
The paired default/precise sensitivity and cost measurement hasn't run on the current source.
See [METHOD.md](METHOD.md) and the [historical held-out results](../results/precise_validate/SUMMARY.md).

## What would establish operator value

**VERIFIED, SF Compute cluster requirements:** v2 (effective August 24, 2026, for B300/GB300) requires at
least 24 hours of GPU soak with `dcgmi diag -r 4` on every GPU before and after; v1 (effective July 7, 2026)
still applies to H100, H200 and B200. Both require a signed acceptance certificate, and neither mentions a
numerical-correctness or SDC test. [v2](https://sfcompute.com/requirements/clusters/v2),
[v1](https://sfcompute.com/requirements/clusters/v1). A five-minute sdcguard result is a complementary
observation, not a replacement acceptance process.

**Hypothesis:** the first paying user is an operator dealing with a current unexplained numerical incident who
can install the checker and act on the result. Evidence of value would be a reproducible diagnosis, reduced
investigation time, a changed operational decision and repeat use. Provider names, stars and free PASS cards
are not evidence of a purchase.

## Limits that matter to that user

- Checksum sensors can miss small errors, cancelling patterns, unselected layers and unsupported operations.
  A large legitimate value in a block can mask a smaller error even in precise mode.
- Attention internals, normalization, convolution and fused MoE are not covered by the inline GEMM check.
  Repeatability and canary fingerprints can reveal some discrepancies, but do not verify every such operation.
  An experimental attention check (the sum of dV over keys against the sum of dO over queries, the invariant
  AEGIS uses in section 5.1.1, plus forward outputs within the values' range) met its clean GPU
  calibration criteria on an L4, an H100 and a B200 in bf16 (October 20) but stays off by default: it catches
  large errors only, doesn't cover dQ and dK, works only in the default mode, and its cost isn't measured.
- A checksum alert can originate in the checked operation, the checking computation, the software stack or
  a violated numerical assumption. Compare against independent evidence before attributing it to hardware.
- A CPU emulation or an NVBit injection is not a test on a GPU removed from service for a naturally occurring
  defect. No such GPU is established in the supplied evidence.
- A signature verifies an issuer and declared artifacts. NVIDIA's [RMA guide, March 15, 2023](https://docs.nvidia.com/deploy/rma-process/index.html)
  calls for additional diagnostics and evidence that a failure follows a particular board. A simulation and a
  clean replay do not establish that.
- Earlier clean-run and performance numbers belong to their original workloads and software. New code needs
  new validation; [METHOD.md](METHOD.md) records which code fingerprint each run used.
