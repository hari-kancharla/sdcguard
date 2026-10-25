# Why sdcguard

Every source below was opened between October 19 and 24, 2026, and says what its row says. These sources show
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
- **Serving software, not only hardware and libraries:** in October 2026 alone, vLLM issues report a short
  prompt's greedy answer changing with the request before it under FP8 per-tensor MoE quantization and CUDA graphs
  ([#62124](https://github.com/vllm-project/vllm/issues/62124), October 17, open), a request decoding against
  another request's KV prefix with cascade attention and CPU offload
  ([#61673](https://github.com/vllm-project/vllm/issues/61673), October 14), and answers containing other requests'
  content under prefix caching in a disaggregated deployment
  ([#61160](https://github.com/vllm-project/vllm/issues/61160), October 12; a maintainer confirmed it on the v0.31.0
  image and couldn't reproduce it on main). Each returns fluent text with HTTP 200, and each matmul in it is right
  for the numbers it was given.
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
| [Huawei Ascend PyTorch backend](https://gitcode.com/Ascend/pytorch/blob/master/torch_npu/asd/checksum.py) (created June 20, 2025) and [MindSpore SDC detection](https://www.mindspore.cn/tutorials/en/master/debug/sdc.html) | A matmul checksum shipped by a chip vendor: row sums predicted from the inputs, a rounding-error bound with a safety factor, and a recompute when a block flags; MindSpore escalates to it from gradient-norm statistics. Its documentation gives "roughly 100% slowdown for a single 4096-sized MatMul and about 90% for a Llama2-7B setup" and checks bf16 MatMul only | Ascend NPUs and training only: no NVIDIA GPUs, no serving, no evidence kept, no gate. The checksum method itself is not new, here or in sdcguard (Huang and Abraham, 1984) |
| [Flipless](https://flipless.space) (prototype runs dated October 7 to 8, 2026) | Checksums every linear layer in Hugging Face inference, computes norms twice bit for bit, keeps a spare KV-cache copy, and rolls back and replays a token when a check fails | Not vLLM yet (its site says next); no public code; its replay-to-repair assumes a transient fault, and a library bug such as cuBLAS 6862950 comes back on every replay |
| [vLLM Trace Replay](https://docs.vllm.ai/en/latest/serving/online_serving/trace_replay/) (in vLLM 0.32) | Forces the engine to emit a fixed token sequence through its decode path while returning the real log-probabilities, so two configurations (quantization, parallelism, attention backend) can be compared position by position | Arithmetic it can't tell from an approximation of the same size; anything that depends on the order or mix of requests (a fixed sequence doesn't run it); offline only, not in the server API |
| [AMD Cluster Validation Suite](https://rocm.docs.amd.com/projects/cvs/en/latest/how-to/test-suites/burn-in-diag/health.html) (repo updated October 23, 2026) | Hardware burn-in, plus suites that stand up real vLLM and SGLang servers and fail them on lm-eval accuracy thresholds (GSM8K, HellaSwag) and a long-context retrieval probe | Benchmark scores, not each operation's arithmetic; no evidence of which computation went wrong; AMD Instinct GPUs |
| [cudaprof](https://github.com/tmc/lib/blob/main/nvidia/x/cmd/cudaprof/README.md#gemms) and the [cuBLAS 6862950 reporter's repo](https://github.com/Yeosangho/mps-cublaslt-fp16-corruption-repro) (October 2, 2026) | Record the cuBLASLt kernel every call ran from cuBLASLt's own log and compare runs; the second scans shapes against float64 under MPS and ships an LD_PRELOAD guard against the failing cluster shapes | No check of a live job's outputs; no verdict on a serving change |
| [Racer](https://arxiv.org/abs/2610.28429) (arXiv, October 20, 2026) | Symbolic equivalence checking of GPU kernels at the PTX level: 82 bugs in 23 libraries across 2,408 production kernels; argues that comparing outputs within a tolerance has blind spots more test cases can't remove | Checks a kernel's code, offline; not the conditions a deployment runs it under (the MPS share behind 6862950, the shapes a server picks) and not hardware faults |
| [Microsoft LLM-42](https://arxiv.org/abs/2601.17768) (January 2026) | Speculates with fast nondeterministic kernels and verifies tokens against a deterministic reference inside serving | Its reference is a deterministic kernel, not a right answer: a fault both share passes |
| [GPU Smith](https://gpusmith.com/articles/en/triton-26-08-upgrade-runbook) (September 23, updated October 14, 2026) | Sells inference-stack upgrade runbooks: freeze a representative corpus with expected outputs and require a match "within the model's declared numerical tolerance" | A manual procedure with a tolerance picked by hand |
| [CoreWeave HPC verification](https://docs.coreweave.com/platform/fleet-management/hpc-verification) | A 20 to 30 minute test at the top of each hour on idle nodes: memory errors, and FP8, FP16 and BF16 kernels benchmarked for slowdowns | Idle nodes only; the kernel part measures speed, not results |
| [SF Tensor](https://www.ycombinator.com/companies/sf-tensor/jobs) (YC F25) | Its job posts describe a proof engine that proves the kernels it generates correct before they ship, and it published a bit-exact model of Blackwell tensor-core arithmetic | Its own kernels, before shipping: not a live job, not someone else's vLLM stack |

## What sdcguard adds

Searched October 24, 2026 (56 web searches, 32 arXiv queries since April 2025, 11 repositories read, and YC's
company pages and job posts for every batch, complete from W25 through Spring 2026): no tool found does these.

- **One decision per change to a serving stack**, with its questions kept apart: is the arithmetic right against
  float64 (and what checked it), does one request's answer depend on another's, is the change in the model's
  outputs inside the team's budget, is it as fast as it has to be, and where does the answer hold.
  `sdcguard decide`. Each part has its own tool: the vLLM plugin, `sdcguard isolation`, `gate model-check`,
  `sdcguard perf`, `sdcguard kernels`.
- **The cuBLASLt kernels a real workload ran, and those exact calls run again**: an inventory read from
  cuBLASLt's own log of the job or server, matched against NVIDIA's listed wrong-result issues, and every
  inventoried linear-layer call rerun at its shape against float64 (`sdcguard kernels test`).
- **A check inside the vLLM server that names the requests an alert concerns**: checks of the server's own
  matmuls (bf16 and INT4 on Marlin and Machete), the failing row kept inside CUDA graphs and rechecked in float64,
  and the request that held it named, with the requests potentially exposed kept apart. `sdcguard ready` says from
  outside whether those checks are running.
- **A failure kept, replayed and its fix tested**: `sdcguard gate` keeps a wrong matmul as a case with its exact
  call and conditions, reruns it on every new stack with verdicts that never call a clean rerun a fix, and checks
  a workaround (a TunableOp pin for 6862950) on that case.

What's not new, and isn't claimed: the checksum method (1984; Huawei ships one on Ascend), logprob comparison of
configurations (vLLM Trace Replay, Fireworks), accuracy thresholds on a served model (AMD CVS), idle-node GEMM
tests (DCGM, gpu-burn). Which of these sdcguard does better on the same failures is measured, not assumed:
[EVIDENCE.md](EVIDENCE.md).

## What it doesn't do yet

- **Small errors.** AEGIS found none of its production incidents with checksums of bf16 outputs; every checksum
  detection it confirmed came from fp32 accumulators (section 8). sdcguard's default check works on bf16 and
  fp16 outputs, so it catches large errors (exponent bits, outputs never written, a block off by 10% or more)
  rather than small ones. Its fp32-accumulator check (`precise="fused"`) reaches errors of about 0.7% to 0.9% of a
  typical output and passed 40 minutes of clean training on an H100 and an L4 (October 23); it is for training,
  not serving.
- **Operations outside matmuls.** AEGIS's fingerprint detection, which compares an operation's output with the
  recompute training already does, found 12 of its 18 incidents. sdcguard's recompute check works the same way
  and is experimental; in vLLM serving, attention is spot-checked by default (one layer a step recomputes four of
  its outputs from the KV cache; gross errors only), and RMSNorm layers and fused MoE experts have serving checks
  you turn on (SDCGUARD_NORMS=1, SDCGUARD_MOE=1; both ran on a GPU with limits, docs/STATUS.md). Beyond these,
  attention, normalization, convolutions and fused MoE are not covered.
- **FP8 and fused MoE experts in vLLM.** The plugin checks bf16 and INT4 (Marlin, Machete) linear layers by
  default. Fused MoE experts have an opt-in sampled check (2 rows of one layer a step), and FP8 layers an opt-in check
  of their input quantization only (CPU-tested); nothing checks an FP8 matmul's arithmetic, so a decision on an FP8
  change gets no arithmetic source from it and says so.
- **A natural defect.** No naturally defective GPU has been tested yet. The evidence is injected faults (in
  software, and in real tensor-core instructions with NVBit) and a real library bug.
- **Diagnosis.** An alert can come from the checked operation, the check's own computation, the software stack
  or a broken numerical assumption. NVIDIA's [RMA process](https://docs.nvidia.com/deploy/rma-process/index.html)
  asks for evidence that a failure follows a particular board, which an alert alone doesn't give.
