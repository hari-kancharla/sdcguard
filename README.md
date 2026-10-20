# sdcguard

Catch GPUs that compute wrong answers, inside your own PyTorch training and vLLM serving.

A GPU can return wrong numbers without crashing. Sometimes the hardware is at fault, sometimes a library, and
from the outside the two look the same. ByteDance found 18 such incidents over 3.5 x 10^7 GPU-hours, and 15 of
them were silent ([AEGIS, OSDI '26](https://www.usenix.org/system/files/osdi26-lei.pdf)). The usual health
checks run a test on an idle GPU. sdcguard checks the math of your actual job while it runs, with the kernels,
precisions and settings you really use.

## A real bug the usual tools missed

NVIDIA lists cuBLAS known issue 6862950 in the
[CUDA 13.4 Update 2 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html): on
Hopper GPUs, a process running under MPS with less than 100% of the SMs can get wrong FP16, BF16 and FP8
matmul results. On October 20 we ran a head-to-head on an H200, with the rules written before the run:

| MPS share | `sdcguard check` | DCGM diagnostic | gpu-burn |
|---|---|---|---|
| 100% | PASS | pass | pass |
| 40% | **FAIL** | pass | pass |
| 75% | **FAIL** | pass | pass |
| 14% | **FAIL** | pass | pass |

DCGM and gpu-burn didn't malfunction. Their own test matmuls came out right at every share, so the bug never
reached them. A test of the GPU isn't a test of your job. Rules, predictions and raw logs:
[METHOD.md](docs/METHOD.md#head-to-head-on-a-real-cublas-bug-written-october-20-2026-before-any-run),
[results/h2h](results/h2h/h200_20261020_060928.md).

## Install

```bash
pip install "sdcguard[all]"
```

Python 3.10+, PyTorch 2.5+, an NVIDIA GPU. The demo and self-test also run on a CPU.

## Check a GPU

```bash
sdcguard check            # every GPU on this machine, about 5 minutes
sdcguard check --share    # same, plus a link to add your result to the public board
```

It trains and serves a small model with checks on, runs matmuls in every precision the GPU supports (BF16,
FP16, TF32, FP32, FP64, INT8, FP8) checked in float64, pattern-tests most of the memory, measures speed, and
runs a fixed model three times to see that the output stays bit-identical. The precisions matter: in
ByteDance's study, four GPUs with bad matmuls all got BF16 and FP16 right
([SDCs in the Wild, OSDI '26](https://www.usenix.org/system/files/osdi26-zheng.pdf)).

Each GPU gets PASS, FAIL, SUSPECT (output that should repeat didn't) or INCOMPLETE (part of the check couldn't
run or couldn't judge its results). A FAIL saves the inputs and the wrong output, so `sdcguard replay` can
recompute them on another GPU. PASS means these checks found nothing, not that the GPU is flawless. Shared
results go on the [board](https://hari-kancharla.github.io/sdcguard/board/).

<img alt="An example sdcguard check result card" src="docs/img/example_card.png" width="600">

## In training

```python
from sdcguard import SDCGuard

guard = SDCGuard(every=20).protect(model)    # checks nn.Linear layers: forward, dX and dW
for step, (x, y) in enumerate(loader):
    with guard.on_step(step):                # only every 20th step is checked
        loss = model(x, y); loss.backward()
    optimizer.step(); optimizer.zero_grad()
    if guard.log.events:                     # stop and look at the evidence
        print(guard.log.events[0].to_json()); break
```

Checking every 20th step cost 1.7% on GPT-2 small and 1.3% on a 0.9B model (H100). With
`exact_reductions=False` the gradients are bit-identical to a run without sdcguard.

## In serving

```bash
SDCGUARD=1 SDCGUARD_SHARE=0.125 vllm serve Qwen/Qwen2.5-7B-Instruct
```

This checks one linear layer in eight. With Qwen2.5-7B on an H100 (vLLM 0.31, compiled mode) it cost about 4%
of throughput, every compared answer was word for word the same, and a weight bit flipped in a checked layer
was caught and named. Checking every layer costs about 23% for now
([details](docs/METHOD.md#vllm-again-october-20-2026-with-the-executed-count-gate)).

Any other PyTorch model:

```python
from sdcguard.serve import ServeGuard

guard = ServeGuard().attach(model, share=0.125)   # bf16 or fp16 model on its GPU
...                                                # serve as usual, eager or torch.compile
for event in guard.poll():                         # doesn't stall the GPU
    print(event.to_json())
```

`guard.coverage_report()` shows which layers actually ran their check and how many times.

## Across a fleet

```bash
sdcguard screen --minutes 10                         # screen every GPU on the node
sdcguard replay screen_out/GPU-.../evidence_0.pt     # recompute a failure on another GPU
sdcguard report node*/sdcguard_check --out fleet_report
```

There's also `sdcguard nhc`, a node health check for Kubernetes and Slurm ([deploy/](deploy/)), signed
per-GPU certificates (`sdcguard keygen`, `sdcguard check --key`, `sdcguard verify`), and alerts to Slack,
Discord or Prometheus. An alert is where an investigation starts. It doesn't prove the board is bad.

## How it works

For `C = A @ B`, the sum of each block of 128 output columns has to equal `A` times the matching column sums
of `B`. Both sides are cheap to compute, so a wrong value in `C` shows up as a mismatch. This is
algorithm-based fault tolerance (Huang and Abraham, 1984). Each block also gets a randomly signed sum, so most
errors can't cancel out. Normal bf16 rounding makes the two sides differ a little, so sdcguard estimates each
block's rounding noise and alarms only past 6 times that. On NVIDIA GPUs the checks run as Triton kernels.

## What it catches, and what it misses

It catches:

- wrong results in the matmuls it checks, during real training and serving, on one GPU, with no replicas or
  replay
- errors that repeat exactly every time, because each result is checked against the math, not an earlier run
- in a fault-injection study on GPT-2, 99% of faults that moved a 128-column block by 10% or more, and 70% of
  those between 1% and 10% ([harm study](results/harm_study_gpt2/REPORT.md))
- a mantissa bit flip injected into real L4 tensor-core instructions with NVBit: the loss looked normal, and
  sdcguard flagged 30 of 30 steps ([results](results/round3_oct18/nvbit))

It misses:

- small errors: nothing under about 1% of a block. ByteDance found that checksums of bf16 outputs located none
  of their incidents and fp32 accumulators found them (AEGIS, section 8). sdcguard's fp32 check (precise
  mode) is still experimental.
- anything outside the checked matmuls: attention internals, normalization, MoE experts, the optimizer,
  communication
- correct math on wrong inputs, like the wrong weights or a broken quantization
- steps and layers that sampling skips

No naturally defective GPU has been caught yet. The evidence so far is injected faults and a real library bug.
On clean GPUs it raised no false alarm in 24.6 million checks of GPT-2 training on an H100 or in 1.68 million
checks across 10 cloud GPUs.

## Compared with what you may already run

| Tool | What it does | What it can't see |
|---|---|---|
| DCGM diagnostic, gpu-burn | Run the same test matmul many times on an idle GPU and compare the results with each other | An error that repeats exactly; BF16 and FP8; your job's own kernels and settings |
| Megatron Core rerun state machine | Reruns a step when a check rejects it; built in: NaN or Inf in the loss or gradient norms, opt-in spikes | Wrong results that leave the loss and gradients looking normal; jobs outside Megatron |
| TorchTitan's replayer, SCOUT | Bitwise replay of a step, or exact comparison across 3 or more replicas | An error every replica shares, like a library bug; serving |

Sources for each row are in [WHY.md](docs/WHY.md).

## Coverage

- Tested: PyTorch training in bf16 and fp16 (DDP on 4 H100s, FSDP2 on CPU), Hugging Face and vLLM serving,
  `sdcguard check` on B200, H200, H100, A100, L40S, L4 and T4.
- Not checked: fp32, TF32 and quantized matmuls in training and serving (`check` tests them one matmul at a
  time), convolutions, fused MoE experts, vLLM's LM head, the inside of fused attention.
- Not tested yet: Megatron tensor and pipeline parallel, DeepSpeed, multi-node, AMD.

## No GPU? Try the demo

```bash
python -m sdcguard.demo       # simulated corruption during training, clearly labeled as a demo
python -m sdcguard.selftest   # checks that sdcguard works on this machine
```

Or run it in [Colab](https://colab.research.google.com/github/hari-kancharla/sdcguard/blob/main/examples/colab_demo.ipynb).
The same kind of simulation on 4 H100s:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/demo_zoom_dark.png">
  <img alt="Training loss of the same run on 4 H100s with and without sdcguard. GPU 2 starts returning wrong results at step 205. sdcguard flags it at step 210 and the run finishes all 1,000 steps on GPUs 0, 1 and 3. Without sdcguard the loss looks normal until it turns NaN at step 215, and restarting crashes again." src="docs/img/demo_zoom.png">
</picture>

Logs and raw numbers: [results/demo_4gpu](results/demo_4gpu).

## More

- [docs/METHOD.md](docs/METHOD.md): how every number was measured, with pass/fail criteria written before the
  GPU runs
- [docs/EVIDENCE.md](docs/EVIDENCE.md): what others published and what sdcguard measured, kept apart,
  negative results included
- [docs/WHY.md](docs/WHY.md): what already exists and what it misses
- [results/](results/): raw output of the GPU runs
- To reproduce: `pip install -e ".[experiments,dev]" && pytest -q`, then the Modal scripts in
  [scripts/](scripts/)

## Status

Alpha (0.3.0), MIT license. Built by Hari Krishna Kancharla. Bug reports and results from your own GPUs are
very welcome, most of all from a GPU you think is bad.
