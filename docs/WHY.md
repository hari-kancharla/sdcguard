# Why sdcguard

Sources checked October 19, 2026.

## The problem

- **It's common at scale.** Meta: "silent data corruptions now occur at about one fault per thousand
  devices" ([Meta, July 2025](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/)).
  ByteDance found 18 incidents on 13 faulty GPUs in about 35 million GPU-hours of training
  ([AEGIS, OSDI '26](https://www.usenix.org/system/files/osdi26-lei.pdf)).
- **Offline tests miss a lot of it.** In AEGIS, vendor diagnostics that take hours caught 2 of 8 known-bad
  machines (the paper puts NVIDIA's offline EUD diagnostics at about 70% recall in its production data, at
  several hours a run). ByteDance's companion study found synthetic tests "miss over 60% of defective
  devices", only 25% of defects showed up in burn-in, and 40% appeared around a year after deployment
  ([SDCs in the Wild, OSDI '26](https://www.usenix.org/system/files/osdi26-zheng.pdf)).
- **Which math mode breaks depends on the GPU.** In the same study, four GPUs that got matrix multiplies
  wrong all computed BF16 and FP16 correctly; each failed at least one of FP32, FP64, TF32 and INT8 (one only
  FP32, one only FP64, two all four; Table 3). A check of bf16 matmuls alone, in training or anywhere else,
  would have passed all four, which is why `sdcguard check` tests every mode the GPU supports.
- **In training it hides, in inference it ships.** 15 of the 18 AEGIS incidents caused no visible training
  failure. Meta: in inference, SDCs "affect thousands of inference consumers" (same Meta post).

## What exists, and what it misses

| Option | What it misses |
|---|---|
| [TorchTitan SDC replayer](https://github.com/pytorch/torchtitan/blob/main/torchtitan/observability/SILENT_DATA_CORRUPTION.md) | Re-runs checked steps (each costs "`1 + num_replays` forward/backward executions"), "requires `debug.deterministic=True`", replays on the same GPU, TorchTitan only |
| [Megatron-LM rerun state machine](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/rerun_state_machine.py) | Only reacts to NaN/Inf or loss spikes, and NaN/Inf are "only 1.01% of SDC outcomes" in NVIDIA's own fault study ([arXiv 2605.04213](https://arxiv.org/abs/2605.04213)) |
| [Google's TPU SDC checker](https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/xla_flags_library.py) | Re-runs the whole program, TPUs only |
| [SCOUT / lm-resiliency](https://github.com/LMResiliency/lm-resiliency) | Needs deterministic replicas to compare against, training only |
| AEGIS (ByteDance) | Checks the math in place, like sdcguard, but inside ByteDance's own Megatron-LM, with no code released |
| Offline tests: NVIDIA DCGM, TinyMeg2, CoreWeave's idle-node tests | Run on idle GPUs, between jobs, and miss many bad ones (above). CoreWeave's stop "as soon as a customer workload is scheduled" ([docs](https://docs.coreweave.com/platform/fleet-management/hpc-verification.md)) |
| vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo | As far as I can find, none of them check matmul results while serving |
| [NVIDIA NVSentinel](https://developer.nvidia.com/blog/automate-kubernetes-ai-cluster-health-with-nvsentinel/) and other fleet tools | React to XID, ECC and thermal signals and to failed burn-in tests, not to wrong results in the running job |
| [assay-gpu](https://github.com/GautamTalksDev/assay-gpu) | The closest open project: GPU acceptance tests with signed attestations ([paper](https://arxiv.org/abs/2610.02240)). A different design; no side-by-side comparison yet |
| [gpu-burn](https://github.com/wilicc/gpu-burn), [gpu-fryer](https://github.com/huggingface/gpu-fryer), memory testers | gpu-burn compares repeated fp32/fp64 matmuls with each other, so a GPU that repeats the same wrong answer passes. gpu-fryer measures speed. Memory testers test memory |

## What sdcguard does differently

- **Checks the math instead of redoing it.** Algorithm-based fault tolerance on every checked matmul, so no
  replay, no deterministic mode and no replicas. It works on one GPU, with sharded weights and with
  nondeterministic kernels. The Open Compute Project's SDC white paper (NVIDIA, Meta, Google, AMD, Intel,
  Arm and Microsoft, August 2025) points to "emerging research on algorithm-based fault tolerance (ABFT,)
  especially when integrated directly into core AI kernels" as a direction, though not in its call to action
  ([OCP, 2025](https://www.opencompute.org/documents/sdc-in-ai-ocp-whitepaper-final-pdf), section 5.4), and
  Huawei ships a matmul checksum for its Ascend NPUs in `torch_npu`
  ([source](https://gitcode.com/Ascend/pytorch/blob/master/torch_npu/asd/checksum.py)). AEGIS showed checksums
  work at production scale on NVIDIA GPUs, with a finding that matters here: it located its incidents only
  with checksums over the fp32 accumulators. "We did not successfully locate SDC incidents using checksums
  accumulated from bfloat16 outputs" (section 8); those need roughly 10,000 times larger errors to be
  detected (section 7.5). sdcguard's default checks read the bf16/fp16 outputs (cheap, and in serving the
  only option without changing cuBLAS); its precise mode checks the fp32 accumulators the way AEGIS does,
  and is still being validated on GPUs ([METHOD.md](METHOD.md)). AEGIS's code isn't released.
- **Names the chip and the layer:** GPU UUID, PCI slot, layer, and the row and column block of the bad output.
- **Covers serving**, through a vLLM plugin and a Hugging Face integration.
- **Saves evidence** you can replay on another GPU, which is what NVIDIA's RMA process asks for when its own
  diagnostic passes: data "illustrating that the failure follows the particular NVIDIA Tesla GPU board"
  ([NVIDIA RMA guide](https://docs.nvidia.com/deploy/rma-process/index.html)).

## Limits

- The checksums cover matrix multiplies, which is most of the math. Attention internals, norms and the rest
  are covered only by `sdcguard check`'s repeatability test and, in training, the cross-GPU canary.
- Errors under about 1% of an output block hide in bf16 rounding noise. TU Berlin, injecting faults into LLM
  training, found "Only exponent bits exhibited measurable impact"
  ([Altenbernd et al., 2026](https://arxiv.org/html/2604.00726)), but AEGIS's production incidents were
  found only by fp32-accumulator checksums (above): on real defective GPUs the default checks may be too
  coarse, which is what precise mode is for.
- All faults so far were simulated in software or injected into a real GPU with NVBit. No GPU pulled from a
  fleet for silent corruption has been tested yet.
