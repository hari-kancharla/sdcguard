# Why sdcguard

Sources checked October 18, 2026.

## The problem

- **It's common at scale.** Meta: "silent data corruptions now occur at about one fault per thousand
  devices" ([Meta, July 2025](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/)).
  ByteDance found 18 incidents on 13 faulty GPUs in about 35 million GPU-hours of training
  ([AEGIS, OSDI '26](https://www.usenix.org/system/files/osdi26-lei.pdf)).
- **Offline tests miss most of it.** In AEGIS the vendor's offline diagnostics caught 2 of 8 known-bad
  machines. ByteDance's companion study found synthetic tests "miss over 60% of defective devices", and only
  25% of defects showed up in burn-in ([SDCs in the Wild, OSDI '26](https://www.usenix.org/system/files/osdi26-zheng.pdf)).
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
| Offline tests: NVIDIA DCGM, TinyMeg2, CoreWeave's idle-node tests | Run on idle GPUs and miss most bad ones (above). CoreWeave's stop "as soon as a customer workload is scheduled" ([docs](https://docs.coreweave.com/platform/fleet-management/hpc-verification.md)) |
| vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo | Nothing checks the math while serving |
| [NVIDIA NVSentinel](https://developer.nvidia.com/blog/automate-kubernetes-ai-cluster-health-with-nvsentinel/) and other fleet tools | React to XID, ECC and thermal signals, not to wrong results |
| [gpu-burn](https://github.com/wilicc/gpu-burn), [gpu-fryer](https://github.com/huggingface/gpu-fryer), memory testers | gpu-burn compares repeated fp32/fp64 matmuls with each other, so a GPU that repeats the same wrong answer passes. gpu-fryer measures speed. Memory testers test memory |

## What sdcguard does differently

- **Checks the math instead of redoing it.** Algorithm-based fault tolerance on every checked matmul, so no
  replay, no deterministic mode and no replicas. It works on one GPU, with sharded weights and with
  nondeterministic kernels. The Open Compute Project's SDC white paper, written with NVIDIA, Meta, Google,
  AMD, Intel, Arm and Microsoft, recommends exactly this
  ([OCP, 2025](https://www.opencompute.org/documents/sdc-in-ai-ocp-whitepaper-final-pdf)), and Huawei ships
  a matmul checksum for its Ascend NPUs in `torch_npu`
  ([source](https://gitcode.com/Ascend/pytorch/blob/master/torch_npu/asd/checksum.py)). Nothing like it ships for NVIDIA GPUs.
- **Names the chip and the layer:** GPU UUID, PCI slot, layer, and the row and column block of the bad output.
- **Covers serving**, through a vLLM plugin and a Hugging Face integration.
- **Saves evidence** you can replay on another GPU, which is what NVIDIA's RMA process asks for when its own
  diagnostic passes: data "illustrating that the failure follows the particular NVIDIA Tesla GPU board"
  ([NVIDIA RMA guide](https://docs.nvidia.com/deploy/rma-process/index.html)).

## Limits

- The checksums cover matrix multiplies, which is most of the math. Attention internals, norms and the rest
  are covered only by `sdcguard check`'s repeatability test and, in training, the cross-GPU canary.
- Errors under about 1% of an output block hide in bf16 rounding noise. Those seem to matter least: TU
  Berlin, injecting faults into LLM training, found "Only exponent bits exhibited measurable impact"
  ([Altenbernd et al., 2026](https://arxiv.org/html/2604.00726)).
- All faults so far were simulated in software or injected into a real GPU with NVBit. No GPU pulled from a
  fleet for silent corruption has been tested yet.
