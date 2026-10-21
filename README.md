# sdcguard

Catch GPUs that compute wrong answers, inside your own PyTorch training and vLLM serving.

A GPU can return wrong numbers without crashing. Sometimes the hardware is at fault, sometimes a library, and
from the outside the two look the same. ByteDance found 18 such incidents over 3.5 x 10^7 GPU-hours, and 15 of
them were silent ([AEGIS, OSDI '26](https://www.usenix.org/system/files/osdi26-lei.pdf)). The usual health
checks run a test on an idle GPU. sdcguard checks the math of your actual job while it runs, with the kernels,
precisions and settings you really use.

## A real bug the usual tools couldn't reach

NVIDIA lists cuBLAS known issue 6862950 in the
[CUDA 13.4 Update 2 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html): on
Hopper GPUs, a process running under MPS with less than 100% of the SMs can get wrong FP16, BF16 and FP8
matmul results. On October 20 we ran a head-to-head on one H200, with the rules written before the run. Each
tool was judged against its own matmuls at that share: DETECTED if they were wrong and it failed, NOT EXERCISED
if they were right.

| MPS share | `sdcguard check` | DCGM diagnostic | gpu-burn |
|---|---|---|---|
| 100% | PASS (not exercised) | pass (not exercised) | pass (not exercised) |
| 40% | **FAIL (detected)** | pass (not exercised) | pass (not exercised) |
| 75% | **FAIL (detected)** | pass (not exercised) | pass (not exercised) |
| 14% | **FAIL (detected)** | pass (not exercised) | pass (not exercised) |

DCGM and gpu-burn didn't malfunction: the bug never reached their tests. A probe making their matmul calls
got right answers at every share (for DCGM through the system cuBLAS; DCGM's own copy was logged only at 40%,
where it didn't use the affected algorithm). A test of the GPU isn't a test of your job. Rules, predictions and
raw logs:
[METHOD.md](docs/METHOD.md#head-to-head-on-a-real-cublas-bug-written-october-20-2026-before-any-run),
[results/h2h](results/h2h/h200_20261020_060928.md).

## Install

```bash
pip install "sdcguard[all]"
```

Python 3.10+, PyTorch 2.8+, an NVIDIA GPU. The demo and self-test also run on a CPU.

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

Checking every 20th step cost 1.7% on GPT-2 small and 1.3% on a 0.9B model (H100, October 18 code; later
changes haven't been timed). With `exact_reductions=False` the gradients are bit-identical to a run without
sdcguard.

It works with autocast, DDP, FSDP2, tensor parallel (each rank checks its own shard) and `torch.compile`,
`fullgraph=True` included, tensor parallel inside a compiled graph too (new in 0.4.0 and tested on CPU only so
far). With the Hugging Face Trainer it's one line:

```python
from sdcguard.hf import SDCGuardCallback
trainer = Trainer(..., callbacks=[SDCGuardCallback(every=20, stop_on_alert=True)])
```

`guard.observe(step, loss=..., grad_norm=...)` checks every step for a while after the loss or gradient norm
spikes (the callback does this from the Trainer's logs), and `SDCGuard(replicas=True)` also compares an exact
checksum of the weights across data-parallel ranks on the checked steps every rank shares (the `every`
schedule and `after_restart()` windows, not the steps a spike on one rank added).

Two experimental options, off by default:

- `SDCGuard(recompute=True)` fingerprints the output of every activation-checkpointed block and compares it
  with the recompute your training already does in the backward pass. That's how AEGIS made 12 of its 18
  production catches, and it covers every operation inside those blocks, attention and norms included. On an
  H100 and an L4, with the earlier PyTorch fingerprints, it raised no false alarm in 320 clean compares, caught
  32 of 32 injected faults and named the block each time, and didn't change the gradients; checking every step
  cost 10% to 20%
  ([details](docs/METHOD.md#the-recompute-check-on-gpus-written-october-20-2026-before-any-gpu-run)). 0.4.0
  fingerprints with a Triton kernel instead, tested on CPU only so far.
- `SDCGuard(precise="fused")` checks each matmul's fp32 result before it's rounded, inside a Triton kernel. The
  first version, which scored that result in float64 outside the kernel, caught errors of 0.5% of a value that
  the default check never sees, but cost 3x to 4.7x the time of each checked step
  ([details](docs/METHOD.md#what-the-run-found-october-20-2026-as-sensitive-as-precise-mode-not-cheaper)).
  0.4.0 scores inside the kernel instead, tested on CPU only so far.

## In serving

```bash
SDCGUARD=1 SDCGUARD_SHARE=0.125 vllm serve Qwen/Qwen2.5-7B-Instruct
```

This checks one linear layer in eight. With Qwen2.5-7B on an H100 (vLLM 0.31, compiled mode) it cost about 4%
of throughput, every compared answer was word for word the same, and a weight bit flipped in a checked layer
was caught and named. Checking every layer costs about 23% of throughput for now
([details](docs/METHOD.md#vllm-again-october-20-2026-with-the-executed-count-gate)).

New in 0.4.0, tested on CPU and with injected faults in a CUDA graph on one H100 so far, not yet inside vLLM: when
a layer's check alarms, the check itself keeps the numbers that alarmed (a row of the layer's input and that block
of its output: one row per layer per read, at most four per GPU), copied on the GPU before vLLM reuses the memory,
inside CUDA graphs too. So a fault that never happens again is still on record, and the alert says what a float64
recheck of those numbers found against the layer's weights as they are when the alert is read: the output was
wrong for those weights, or many small errors add up, or it was right for them and the check's own side disagreed;
or the weights changed, or were written in place, since load, and the recheck can't say. The capture keeps the
input and output, not the weights, so the verdict is about the weights the matmul used only as far as nothing
changed them in between. `SDCGUARD_EVIDENCE_DIR=<folder>` also saves them as evidence files for `sdcguard replay`
and `sdcguard casefile`. The Triton check copies only when a row alarms; the PyTorch check, used where the Triton
one isn't, writes every call to a spare record it then discards, which keeps it free of branches.

Quantized layers (FP8, AWQ, GPTQ and the like) and fused MoE experts aren't checked yet. The first model step
logs how many linear layers are checked (and what share of their weights) and why the others aren't, and
`SDCGUARD_REQUIRE=1` refuses to serve unless every layer that `SDCGUARD_SHARE` selected is checked and no
quantized layers or fused MoE experts are left over (the LM head, convolutions and fp32 layers don't count).

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
sdcguard casefile screen_out/GPU-.../evidence_0.pt   # what the evidence supports
sdcguard report node*/sdcguard_check --out fleet_report
```

A failed check saves the matmul's inputs and the wrong output, and so does
`SDCGuard(evidence_dir="sdc_evidence")` in your own job. `sdcguard casefile` (experimental) weighs that evidence
with replays on the same GPU, another GPU and the CPU, and says which explanation it supports (a transient or
recurring GPU fault, a software defect, or the check itself) or that it's inconclusive and which replay would
settle it. A replay runs with the PyTorch matmul settings recorded with the evidence, and counts only if it
replayed that exact file and ran the way the incident did (library versions, MPS share, visible SMs, cuBLAS
settings): a clean replay at another MPS share proves nothing about a bug like 6862950. In a job, a saved
failure is also recomputed three times right away on the same GPU, and the alert says whether it came back.

There's also `sdcguard nhc`, a node health check for Kubernetes and Slurm that runs the full check with a
deadline ([deploy/](deploy/)), signed per-GPU certificates (`sdcguard certify`, `sdcguard verify`), a fleet
report that keeps a GPU in quarantine after a failure even if a later run passes, and alerts to Slack, Discord
or Prometheus. On Kubernetes with NVIDIA's NVSentinel, `sdcguard nvsentinel` hands check results and job
alerts to its fault pipeline as health events: a job's alert is stored as a lead, and a check FAIL asks for
action only if you pass `--remediate`. New in 0.4.0 and not yet run on a GPU: `sdcguard blas` runs every
cuBLASLt algorithm for a matmul shape and checks each one (every output written, a sample of rows against
float64), the kind of sweep a separate probe used to pin 6862950 to one algorithm in the head-to-head;
`sdcguard issues` says which of NVIDIA's listed cuBLAS wrong-result bugs apply to the machine (its cuBLAS
version, GPU and MPS setting). An alert is where an investigation starts. It doesn't prove the board is bad.

## How it works

For `C = A @ B`, the sum of each block of 128 output columns has to equal `A` times the matching column sums
of `B`. Both sides are cheap to compute, so a wrong value in `C` shows up as a mismatch. This is
algorithm-based fault tolerance (Huang and Abraham, 1984). Each block also gets a randomly signed sum, so most
errors can't cancel out. Normal bf16 rounding makes the two sides differ a little, so sdcguard estimates each
block's rounding noise and alarms only past 6 times that. On NVIDIA GPUs up to the B200 the checks run as Triton
kernels; newer generations use plain PyTorch until they're tested.

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

- small errors: nothing under about 1% of a block. ByteDance found none of their incidents with checksums of
  bf16 outputs; every checksum detection they confirmed came from fp32 accumulators (AEGIS, section 8).
  sdcguard's fp32 check (precise mode) is still experimental.
- anything outside the checked matmuls (attention internals, normalization, the optimizer, communication),
  unless the experimental recompute check covers that block; FP8 and other quantized matmuls
- correct math on wrong inputs, like the wrong weights or a broken quantization
- steps and layers that sampling skips

No naturally defective GPU has been caught yet. The evidence so far is injected faults and a real library bug.
On clean GPUs it raised no false alarm in 24.6 million checks of GPT-2 training on an H100 or in 1.68 million
checks across 10 cloud GPUs.

## Compared with what you may already run

| Tool | What it does | What it can't see |
|---|---|---|
| DCGM diagnostic, gpu-burn | Run the same test matmul many times on an idle GPU and compare the results with each other (DCGM also tests memory with patterns it writes and reads back) | A matmul error that repeats exactly; BF16 and FP8; your job's own kernels and settings |
| Megatron Core rerun state machine | Reruns a step when a check rejects it; built in: NaN or Inf in the loss or gradient norms, opt-in spikes | Wrong results that leave the loss and gradients looking normal; jobs outside Megatron |
| TorchTitan's replayer, SCOUT | Bitwise replay of a step, or exact comparison across 3 or more replicas | An error every replica shares, like a library bug; serving |

Sources for each row are in [WHY.md](docs/WHY.md).

## Coverage

- Tested: PyTorch training in bf16 and fp16 (DDP on 4 H100s; FSDP2, tensor parallel and `torch.compile`
  fullgraph on CPU), Hugging Face and vLLM serving, `sdcguard check` on B200, H200, H100, A100, L40S, L4 and T4.
- Checked in the default mode (without `protect()`): MoE experts that run through `torch._grouped_mm`, one
  check per expert (new in 0.4.0, tested on CPU only so far).
- Not checked: fp32, TF32 and quantized matmuls in training and serving (`check` tests them one matmul at a
  time), convolutions, vLLM's fused MoE experts, LM head, MoE router gates computed by vLLM's own GEMM and the
  matmuls of MLA decode, the inside of fused attention. `coverage()` says how much of a model's matmul work that
  leaves out.
- Not tested yet: Megatron pipeline parallel, DeepSpeed, multi-node, AMD.

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

Alpha (0.4.0), MIT license. Built by Hari Krishna Kancharla. Bug reports and results from your own GPUs are
very welcome, most of all from a GPU you think is bad.
