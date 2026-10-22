# Why sdcguard

Every source below was opened between October 19 and 21, 2026, and says what its row says. These sources show
the problem is real. They don't show how often sdcguard catches it: sdcguard's own measurements, negative ones
included, are in [EVIDENCE.md](EVIDENCE.md) and [METHOD.md](METHOD.md).

## Wrong answers that don't crash

- **Meta, July 22, 2025:** about one silent-corruption fault per thousand devices. The post gives no time
  period and no GPU-only figure.
  [Meta engineering](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/)
- **ByteDance, AEGIS (OSDI '26):** 18 incidents on 13 GPUs over 35 million GPU-hours; 15 caused no visible
  training failure. Vendor diagnostics found 2 of a selected set of 8. [AEGIS](https://www.usenix.org/system/files/osdi26-lei.pdf)
- **ByteDance and SJTU, SDCHunter (OSDI '26):** four GPUs with faulty GEMMs computed BF16 and FP16 correctly and
  failed different subsets of FP32, FP64, TF32 and INT8; on other GPUs a fault first showed up in operations
  other than GEMM. [SDCs in the Wild](https://www.usenix.org/system/files/osdi26-zheng.pdf), Tables 2 and 3
- **Crusoe, June 30, 2026:** its support guide says a clean DCGM result does not rule out silent data corruption.
  [Crusoe support](https://support.crusoecloud.com/hc/en-us/articles/52229568610715-How-To-Detect-Silent-Data-Corruption-in-Distributed-Training-Jobs)
- **Libraries, not only hardware:** NVIDIA's cuBLAS known issue 6862950 gives wrong FP16, BF16 and FP8 GEMMs on
  Hopper GPUs under MPS below 100% of the SMs. NVIDIA says it was introduced in CUDA 12.6 Update 2; it was
  [reported on NVIDIA's forum](https://forums.developer.nvidia.com/t/cublaslt-returns-silently-wrong-fp16-bf16-fp8-gemm-results-on-hopper-under-cuda-mps-active-thread-percentage-algoid-66-cluster-kernels/384873)
  on October 2, 2026. [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- **Acceptance tests don't look for it:** SF Compute's cluster requirements ask for at least 24 hours of
  `dcgmi diag -r 4` on every GPU and a signed acceptance certificate, and mention no numerical-correctness test
  ([v2](https://sfcompute.com/requirements/clusters/v2), effective August 24, 2026, for B300 and GB300;
  [v1](https://sfcompute.com/requirements/clusters/v1), July 7, 2026, for H100, H200 and B200).

## What already exists

| Tool | How it works | What it can't see |
|---|---|---|
| [NVIDIA DCGM diagnostic](https://docs.nvidia.com/datacenter/dcgm/latest/reference/diagnostics/plugins/diagnostic.html) (docs updated September 10, 2026) | The same half, single or double precision GEMM of the same inputs many times per pass, each result compared with the pass's first (tolerance 0.001, 1e-7 in FP64), and NaNs counted ([compare.cu](https://github.com/NVIDIA/DCGM/blob/64df9f894541e426e416131a9820cae97aa4dd81/nvvs/plugin_src/diagnostic/compare.cu#L30-L136), 4.6.1). Its memory tests compare reads with patterns they wrote. | An error that repeats exactly; BF16 and FP8; the GEMMs and settings of the job you actually run |
| [gpu-burn](https://github.com/wilicc/gpu-burn) (commit 0d19c94) | Repeated GEMMs (FP32 by default, FP16 tensor-core math with `-tc`, FP64 with `-d`), each result compared with its own first result | The same |
| [NVIDIA NVSentinel](https://github.com/NVIDIA/NVSentinel) | Connects health events to cordon, drain and remediation (GPU reset or reboot); those actions are opt-in, and by default it only monitors | It acts on the health events it gets; checking results isn't its job, and `sdcguard nvsentinel` feeds it sdcguard's |
| [Megatron Core rerun state machine](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.rerun_state_machine.html) ([source](https://github.com/NVIDIA/Megatron-LM/blob/ebab48fc40c2b6f0a8768f765b9364656a2dccb9/megatron/core/rerun_state_machine.py) read October 20) | Reruns an iteration on the same GPU and on another when a check rejects it. Built in: NaN and Inf in the loss and each gradient bucket's norm; opt-in: loss spikes and large gradients. Its documentation calls it alpha. | Wrong results that leave the loss and gradient norms looking normal; jobs outside Megatron |
| [TorchTitan SDC replayer](https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/SILENT_DATA_CORRUPTION.md) | Re-executes a deterministic forward and backward on the same device and compares hashes before the optimizer step; off by default, and by default only the first step after a start or restart | A fault that repeats identically on the same GPU; serving |
| [SCOUT / lm-resiliency](https://github.com/LMResiliency/lm-resiliency) ([paper](https://arxiv.org/abs/2608.11034), August 2026) | Between iterations, replays a sampled layer on equivalent replicas and finds the rank that disagrees by majority | An error every replica shares, like a library bug; jobs with no replicas; serving |
| AEGIS and SDCHunter (ByteDance, OSDI '26) | Production detectors inside ByteDance: fp32-accumulator checksums, fingerprints compared with training's own recompute, and replay of the exact workload | Not public |
| [assay-gpu](https://github.com/GautamTalksDev/assay-gpu) | GPU acceptance testing with bf16 checksums and signed reports | A standalone test, not a check inside a running job |

## What sdcguard adds

- Checks of each matmul against its own inputs, inside the training job or the vLLM server you already run, on
  one GPU: no replicas, no replay, no deterministic mode.
- When a serving check alarms, the row that alarmed is kept, inside CUDA graphs too, and rechecked in float64 on
  the CPU, so the alert says whether the output was wrong for its weights or the check's own side disagreed.
- A five-minute `sdcguard check` that tests every precision the GPU has (the SDCHunter finding) against float64,
  with a public board of results by GPU model.
- `sdcguard blas` and `sdcguard issues`: every cuBLASLt algorithm for a shape checked, and the machine matched
  against NVIDIA's listed wrong-result bugs.
- Evidence files, replay on other GPUs and the CPU, and a casefile that weighs them; fleet reports, certificates
  and Kubernetes and Slurm health checks.

## What it doesn't do yet

- **Small errors.** AEGIS found none of its production incidents with checksums of bf16 outputs; every checksum
  detection it confirmed came from fp32 accumulators (section 8). sdcguard's default check works on bf16 and
  fp16 outputs, so it catches large errors (exponent bits, outputs never written, a block off by 10% or more)
  rather than small ones. Its fp32-accumulator check (`precise="fused"`) is experimental.
- **Operations outside matmuls.** AEGIS's fingerprint detection, which compares an operation's output with the
  recompute training already does, found 12 of its 18 incidents. sdcguard's recompute check works the same way
  and is experimental; attention internals, normalization, convolutions and fused MoE are otherwise not covered.
- **A natural defect.** No naturally defective GPU has been tested yet. The evidence is injected faults (in
  software, and in real tensor-core instructions with NVBit) and a real library bug.
- **Diagnosis.** An alert can come from the checked operation, the check's own computation, the software stack
  or a broken numerical assumption. NVIDIA's [RMA process](https://docs.nvidia.com/deploy/rma-process/index.html)
  asks for evidence that a failure follows a particular board, which an alert alone doesn't give.
