# sdcguard

Catch silent wrong math inside your own PyTorch and vLLM jobs, whether a bad GPU or a buggy CUDA library caused
it, and save the evidence to investigate it.

GPUs can return wrong numbers without crashing. Hardware faults and software defects can both cause this.
ByteDance documented incidents that caused no visible training failure
([AEGIS, OSDI, July 2026](https://www.usenix.org/system/files/osdi26-lei.pdf)); Crusoe documents how to
investigate suspected corruption even after a clean DCGM run
([support guide, June 30, 2026](https://support.crusoecloud.com/hc/en-us/articles/52229568610715-How-To-Detect-Silent-Data-Corruption-in-Distributed-Training-Jobs)).

sdcguard checks selected matrix multiplies against checksum identities and a statistical rounding-noise
model. An alert identifies the GPU (UUID, PCI slot), the layer and an output row/block. The inline checksum
does not require replay, deterministic execution or a framework fork. Detection is limited by the operations
actually checked and the noise model; an alert starts an investigation, rather than proving a hardware defect.

**What it catches, and what it doesn't**

| | |
|---|---|
| Catches | Wrong results in the matmuls it checks, inside real training and vLLM serving, on one GPU, with no replicas or deterministic replay. When a check looked at a corrupted matmul, it caught 99% of injected faults that moved a 128-column output block by 10% or more and 70% of those between 1% and 10% ([harm study](results/harm_study_gpt2/REPORT.md)). Errors that repeat exactly every time (a consistently wrong GPU or library), because each result is checked against the math, not against an earlier run. A real NVIDIA library bug: cuBLAS known issue 6862950 left part of FP16/BF16 outputs unwritten on an H200 under MPS, and sdcguard alerted ([results](results/mps_bug)). In a pre-registered head-to-head on that bug (October 20), `sdcguard check` failed the GPU at all three affected MPS shares and passed it at 100%, while DCGM's diagnostic and gpu-burn passed everywhere because their own test GEMMs never ran the affected kernel ([METHOD.md](docs/METHOD.md#head-to-head-on-a-real-cublas-bug-written-october-20-2026-before-any-run)). |
| Misses | Small errors: none of the injected faults under 1% of a block. ByteDance reports that in deployment, checksums taken from bf16 outputs located none of its incidents, and every confirmed checksum detection used fp32 accumulators ([AEGIS, section 8](https://www.usenix.org/system/files/osdi26-lei.pdf)); sdcguard's default check works on bf16/fp16 outputs, and its fp32 path (precise mode) is experimental. Operations outside the checked matmuls (attention internals, normalization, fused MoE experts, the optimizer, collectives). Software bugs that feed wrong inputs into correct math (wrong weights, wrong expert, a broken template or quantization). Steps and layers that sampling skips. No naturally defective GPU has been caught yet. |

**How it differs from what you may already run** (from each tool's source and docs, October 2026)

| Tool | What it checks | What it can't see |
|---|---|---|
| DCGM diagnostic, gpu-burn | Verify matrix results by repetition: the same FP16/FP32/FP64 GEMM many times on an idle GPU, and any element that differs from the first result of the pass (DCGM: by more than 0.001, or 1e-7 in FP64) or is NaN counts as faulty; no independent reference answer ([DCGM](https://github.com/NVIDIA/DCGM/blob/64df9f894541e426e416131a9820cae97aa4dd81/nvvs/plugin_src/diagnostic/compare.cu#L30-L136), [gpu-burn](https://github.com/wilicc/gpu-burn/blob/0d19c94dcb2e9b6858b7586a4c327674cd67113b/compare.cu#L30-L61)). DCGM's memory tests do compare reads with known written patterns | A wrong answer that repeats exactly; BF16 and FP8; your job's own kernels and configuration (MPS share, library version) |
| Megatron Core rerun state machine | Validates a result with any rejection function ([validate_result](https://github.com/NVIDIA/Megatron-LM/blob/ebab48fc40c2b6f0a8768f765b9364656a2dccb9/megatron/core/rerun_state_machine.py#L499-L530)). Built in: NaN or Inf in the loss and in each gradient bucket's norm, and, opt-in, a loss or gradient norm 10 times the largest seen ([loss](https://github.com/NVIDIA/Megatron-LM/blob/ebab48fc40c2b6f0a8768f765b9364656a2dccb9/pretrain_gpt.py#L265-L293), [gradients](https://github.com/NVIDIA/Megatron-LM/blob/ebab48fc40c2b6f0a8768f765b9364656a2dccb9/megatron/core/distributed/param_and_grad_buffer.py#L402-L437)). On a rejection it reruns the step on the same GPU, then from a checkpoint on another, and labels it transient, persistent or correct; a stats mode reruns every step to measure nondeterminism. Marked experimental by NVIDIA | Wrong results that leave the loss and gradient norms finite and unremarkable (NaN/Inf were about 1% of corruptions in an NVIDIA-coauthored study, [arXiv 2605.04213](https://arxiv.org/abs/2605.04213)); jobs outside Megatron |
| TorchTitan SDC replayer, lm-resiliency (SCOUT) | Bitwise replay on the same GPU (needs full determinism), or exact comparison across 3 or more matching replicas ([TorchTitan](https://github.com/pytorch/torchtitan/blob/e61df55b600331defa37e6997a4fdfc91cfaec3b/torchtitan/observability/SILENT_DATA_CORRUPTION.md), [SCOUT](https://github.com/LMResiliency/lm-resiliency/blob/eabcc5f2a38ca671aca0c21a757b5f4636cf311c/docs/scout.md#L12-L59)) | An error every replica or rerun shares (a library bug); serving; one or two GPUs. In exchange they see errors of any size on the steps they check |
| NVSentinel, cloud health checks | Health counters, ECC, Xid, and DCGM/gpu-burn/NCCL tests before, after or between jobs | The math of your job while it runs |

## Check a GPU

```bash
python -m pip install "sdcguard[all]"
sdcguard check            # every GPU on this machine, about 5 minutes
sdcguard check --share    # same, plus an issue link to review and submit to the public board
```

Version 0.3.0 on PyPI is built by this repository's publish workflow from the `v0.3.0` tag. Its library code
(`src/`) is the code that passed the GPU tests on an L4, an H100 and a B200 ([results](results/gpu_tests), code
fingerprint 74b844ee7ca8) and then ran inside real vLLM in its default compiled mode on an H100, passing its
execution checks ([results](results/vllm_oct20)). Precise mode and attention checking remain experimental.

<img alt="An example sdcguard check result card" src="docs/img/example_card.png" width="600">

It trains and serves a small GPT-style model with its supported linear matmuls checked, runs large matmuls, checks
matmuls in BF16, FP16, TF32, FP32, FP64, INT8 and FP8 (whichever the GPU supports) using block projections
computed in float64 on the CPU, pattern-tests most of the free memory, measures speed against the datasheet, and runs one fixed model
three times to see if the output stays bit-identical. The math modes matter because a bad GPU can get one
wrong and the rest right: in ByteDance's study, four GPUs that got matrix multiplies wrong all passed BF16 and
FP16 ([SDCs in the Wild, OSDI '26](https://www.usenix.org/system/files/osdi26-zheng.pdf)). The verdict is
PASS (no failure detected under the completed checks), FAIL (a failure criterion was met; failing matmuls' inputs and outputs are saved where possible so `sdcguard replay` can recompute
them on another GPU), SUSPECT (no checksum failed, but output that should repeat exactly didn't) or
INCOMPLETE (nothing wrong was found, but part of the check couldn't run or couldn't judge its results). Handy
before a long run on a rented GPU or while investigating a used GPU. PASS is a bounded screening result, not
proof that the GPU is fault-free or that a cloud provider will accept it as a replacement for its diagnostics.

Shared results go on the [board](https://hari-kancharla.github.io/sdcguard/board/), by GPU model and
provider. The earlier October 18 sweep reported 10 cloud GPUs (B200, H200, two H100s, A100, L40S, three L4s, T4 on AWS, Google
Cloud, Azure and Oracle machines): all PASS, 1.68 million results checked, and GPUs of the same model gave
bit-identical output on different machines and clouds. Those historical results do not validate later checker
changes. Compare fingerprints only under compatible software, hardware and execution settings.

## No GPU? Try the demo

```bash
python -m sdcguard.demo       # software-injected corruption during training, clearly labeled as a demo
python -m sdcguard.selftest   # checks that sdcguard works on this machine
```

Or run it in [Colab](https://colab.research.google.com/github/hari-kancharla/sdcguard/blob/main/examples/colab_demo.ipynb).
An earlier simulation of a training incident ran on 4 H100s:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/demo_zoom_dark.png">
  <img alt="Training loss of the same run on 4 H100s with and without sdcguard. GPU 2 starts returning wrong results at step 205. sdcguard flags it at step 210 and the run finishes all 1,000 steps on GPUs 0, 1 and 3. Without sdcguard the loss looks normal until it turns NaN at step 215, and restarting crashes again." src="docs/img/demo_zoom.png">
</picture>

Logs and raw numbers: [results/demo_4gpu](results/demo_4gpu).

## In training

```python
from sdcguard import SDCGuard

guard = SDCGuard(every=20).protect(model)    # eligible nn.Linear: forward and needed dX/dW
for step, (x, y) in enumerate(loader):
    with guard.on_step(step):                # only every 20th step is checked
        loss = model(x, y); loss.backward()
    optimizer.step(); optimizer.zero_grad()
    if guard.log.events:                     # pause and investigate the evidence
        print(guard.log.events[0].to_json()); break
```

October 18 H100 measurements reported 1.7% overhead on GPT-2 small and 1.3% on a 0.9B model when checking every
20th step. These measurements apply to the tested default mode and software. The checks only
read; with `exact_reductions=False` the gradients are bit-identical to a run without sdcguard (one exception:
a protected layer's weight also used directly by another op in the same autocast region, where the two uses
get separate bf16 copies of the weight).

`SDCGuard(every=20, precise=True)` checks the fp32 GEMM result consumed by a protected layer before it is
rounded to bf16/fp16. It is experimental: on October 19 its frozen noise model (fingerprint `1265b3e0a385`) raised no false alarm in 17,424
held-out checks on four GPUs but missed the margin set before the run on the A100 and L4 (worst score 4.27
and 4.26 against 4), and checking every step made each step 260% to 790% slower. It does not inherit the
default mode's overhead or false-alarm results, and it keeps the block-level masking and cancellation limits;
see [METHOD.md](docs/METHOD.md).

`guard.coverage(forward_fn)` estimates potential coverage for the observed dtype and operations. It does not
prove that checks ran. Inspect `guard.stats` for executed checks, unjudged checks and skips.

## In serving (vLLM, Hugging Face)

```bash
SDCGUARD=1 SDCGUARD_SHARE=0.125 vllm serve Qwen/Qwen2.5-7B-Instruct
```

`SDCGUARD_SHARE=0.125` checks one linear layer in eight, spread over every kind of layer and depth. In
the October 20 vLLM offline benchmark with Qwen2.5-7B on an H100 the checks cost 4% of throughput (21,697 vs 22,649 tokens/s;
2% on October 18) and the 64 answers compared were word for word the same as plain vLLM's. Checking every layer cuts
throughput by 23% for now ([METHOD.md](docs/METHOD.md#vllm-again-october-20-2026-with-the-executed-count-gate)). Weight checksums are computed
once at load, which can reveal later memory changes affecting selected layers. Checking one layer in eight
reduces coverage as well as cost; the fault must affect a selected layer and exceed its detection threshold.

Any PyTorch model:

```python
from sdcguard.serve import ServeGuard

guard = ServeGuard().attach(model, share=0.125)   # bf16 or fp16 model on its GPU
...                                                # serve as usual, eager or torch.compile
for event in guard.poll():                         # asynchronous state readback; handles ready alerts
    print(event.to_json())
```

`guard.coverage_report()` says what actually ran: layers attached, layers whose check executed at least
once, layers never checked, checks executed, and checksums that couldn't be judged (Inf/NaN or huge values
the inputs can explain). The counts are kept on the GPU next to each layer's worst score and read back with
it. A worst score of 0 alone (`worst_z()`) can't tell a clean layer from one that never ran; use the counts
for that. `coverage_report()`, `counts()` and `worst_z()` wait for GPU readback; `poll()` doesn't. (The
counts are new on October 19. On October 20 they passed on an L4, an H100 and a B200: the PyTorch and Triton
checks gave equal counts, and in a compiled model a test counted CUDA graph replays with PyTorch's replay hook
and saw every layer's check count advance once per call ([results](results/gpu_tests)). Inside real vLLM on October 20,
every layer's count rose by the same 384 checks during the measured requests ([results](results/vllm_oct20)).)
When validating traffic, compare each layer's counts before and after the measured requests. Startup
profiling can advance lifetime counts without establishing that a check ran on those requests.

## Across a fleet

```bash
sdcguard screen --minutes 10                         # screen every GPU on the node
sdcguard replay screen_out/GPU-.../evidence_0.pt     # recompute a failure on another GPU
sdcguard report node*/sdcguard_check --out fleet_report
```

A failed training, stress or math-mode check saves the matmul's inputs and the wrong output with the GPU's
UUID, serial and PCI slot. Replay helps compare behavior, but diagnosis requires controlling the software and
execution settings and testing whether a failure follows a particular board. A simulated failure followed by
a clean replay is a workflow demonstration, not hardware fault isolation. NVIDIA's
[RMA guide (March 15, 2023)](https://docs.nvidia.com/deploy/rma-process/index.html) describes additional
diagnostics and evidence requirements. Serving alerts
name the layer and position but don't save the inputs. `sdcguard nhc` is a node
health check for Kubernetes node-problem-detector and Slurm ([deploy/](deploy/)). `sdcguard keygen` plus
`sdcguard check --key` gives each GPU a signed certificate, for handing GPUs between a provider and a
customer. A signature authenticates the issuer and declared artifacts; it does not certify that all GPU
computations are correct. Use a separately trusted public key with `sdcguard verify`. Simulated health checks
do not update the actual node health cache or produce fleet quarantine candidates. While serving, alerts can
go to a Slack or Discord webhook or to Prometheus.

## How it works

For `C = A @ B`, the sum of each block of 128 output columns has to equal `A` times the matching column sums
of `B`. Both sides are cheap, so a wrong value in `C` shows up as a mismatch. This is algorithm-based fault
tolerance (Huang and Abraham, 1984). Each block gets a plain sum and a randomly signed one, so most errors that
cancel in one show in the other. Not all: two outputs of a block swapped where the signs match cancel in both.
So `check` draws fresh sign patterns for every matmul it tests, and training checks take a new pattern every
checked step; a fault that keeps swapping the same two outputs then shows on about half the checked steps
instead of never. Serving computes its weight checksums once at load, so it keeps one pattern.

Normal bf16 rounding can make the two sides differ. sdcguard estimates each block's rounding
noise and only alarms past 6 times that. Six is a calibrated threshold, not a universal six-sigma guarantee.
In 24.6 million checks over 5 GPU-hours of clean training (Oct 14-15,
the default mode of that version) it never false-alarmed (worst score 3.54); the board sweep with every phase of
`check` added 1.68 million more on 10 GPUs (Oct 18, worst 3.67). These are historical observations on particular
workloads, not independent trials establishing a fleet-wide false-alarm rate. The bf16 block checks can miss
small errors, with roughly 1% a measured scale in earlier experiments rather than a universal boundary. On NVIDIA
GPUs the checks run as Triton kernels, which first have to match the plain PyTorch check or the plain check
is used instead. Serving reads results back without stalling the GPU; training reads them once at the end of
each checked step.

Some older timing and soak artifacts do not record a source commit or fingerprint, so their exact revision
cannot be recovered from this review archive. Their dates and linked results identify historical observations;
they do not validate the patched candidate.

A current software example: NVIDIA's [CUDA 13.4 Update 2 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
(updated October 12, 2026) list cuBLAS known issue 6862950: on Hopper, GEMM kernels can produce incorrect FP16,
BF16 and FP8 results when the process is an MPS client with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` below 100
(algorithm configuration 66 with clusters of 4 or more CTAs; present since CUDA 12.6 Update 2). A user had
reported it on NVIDIA's forum on October 2 ([thread](https://forums.developer.nvidia.com/t/384873),
[repro](https://github.com/Yeosangho/mps-cublaslt-fp16-corruption-repro)). It is a software defect, not a
physically defective GPU, but from the outside it looks the same. We reproduced it on an H200 on Modal on
October 19 ([results](results/mps_bug)): with MPS limiting the process to 14%, 40% or 75% of the SMs, the
reported layer (16x2048x1000) computed wrong forward outputs or weight gradients and the training check
alerted at each of those shares; at 50% and 100% it computed correctly. At 20% every output was right but the
check still alerted, consistent with the bug hitting the check's own matmul.
[tests/test_unwritten_output.py](tests/test_unwritten_output.py) emulates an unwritten-output pattern on CPU;
it does not reproduce the GPU bug.

Saved training/stress evidence includes a CPU recheck. It distinguishes an output outside a stated rounding
envelope from a statistical checksum mismatch or a mismatch not reproduced on CPU. A passing CPU checksum
alone does not establish that the original GPU checker was corrupted or that the output was correct.

An earlier NVBit experiment reported faults injected into real GPU execution. Flipping a mantissa bit in every tensor-core
instruction on one lane of an L4 (about a 10% error in one input) left the loss untouched, and sdcguard
flagged 30 of 30 steps and named the layer ([results](results/round3_oct18/nvbit)).

## What it covers

| | |
|---|---|
| Tested integrations | Earlier source: PyTorch training in bf16/fp16 (DDP on 4 H100s, FSDP2 on CPU), Hugging Face and vLLM serving, `sdcguard check` on B200, H200, H100, A100, L40S, L4 and T4. Current source (October 20, code fingerprint 74b844ee7ca8): the GPU test suite, both kernel comparisons, the installed wheel, `sdcguard check` and a compiled CUDA-graph serving test on an L4, an H100 and a B200. vLLM ran this library code on October 20 (H100, compiled mode) |
| Not checked | fp32/TF32 and quantized (FP8, INT4) matmuls in training and serving (`check` tests fp32, TF32 and FP8 one matmul at a time), convolutions, fused MoE experts, vLLM's LM head, the inside of fused attention (only `check`'s repeatability test covers it; an experimental attention check, off by default, met its clean calibration criteria on October 20 but catches only large errors, see [METHOD.md](docs/METHOD.md)), `nn.MultiheadAttention`'s output projection and `nn.TransformerEncoderLayer`'s fused inference path (both listed as unchecked, not reported as checked) |
| Not supported on checked steps | `torch.func` transforms (vmap, grad) and `torch.compile(fullgraph=True)`: fast mode runs checked steps' linear layers eagerly |
| Not tested yet | Megatron tensor/pipeline parallel, DeepSpeed, multi-node, AMD, Apple GPUs (only the operator mapping is unit-tested) |

On data-center Blackwell (compute capability 10, including B200), the Triton dot-product comparison paths
use `num_stages=1`. The reported pipelined comparison hit a "misaligned address" fault whose cause isn't
known yet ([METHOD.md](docs/METHOD.md#blackwell-the-triton-kernel-fault-bisected-october-19-2026)).
With this change the test suite and both kernel comparisons passed on a B200 on October 20
([results](results/gpu_tests/b200_20261019_230435.txt), timings in METHOD.md); `SDCGUARD_TRITON_NEW_ARCH=0`
switches back to the plain PyTorch checks.

## More

- [docs/METHOD.md](docs/METHOD.md): how every number here was measured, with the pass/fail criteria written
  before the GPU runs
- [docs/EVIDENCE.md](docs/EVIDENCE.md): what others published and what sdcguard measured, kept apart, with every
  negative result and pre-registered criterion
- [docs/WHY.md](docs/WHY.md): what already exists (DCGM, gpu-burn, TorchTitan's replayer, AEGIS and others)
  and what it misses
- [results/](results/): raw output of the GPU runs (runs since October 19 record the code fingerprint they ran)
- To reproduce: `pip install -e ".[experiments,dev]" && pytest -q`, then the Modal scripts in
  [scripts/](scripts/). Runner prices are estimates from earlier runs; confirm current rates and the hard
  time limits before allocating GPUs. CPU/memory requests and execution timeouts do not form a strict bill cap.

## Status

Alpha (0.3.0), MIT license. Built by Hari Krishna Kancharla. Bug reports and results from your own GPUs are
very welcome, most of all from a GPU you think is bad.
