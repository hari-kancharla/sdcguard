# Method

How the numbers in the README were measured. Raw output for all of it is in [results/](../results/).
Unless noted: NVIDIA H100 80GB HBM3 on Modal, PyTorch 2.14.1 (CUDA 13.0), Triton 3.8.0.

## Pass/fail criteria (written before the first GPU runs)

1. **Noise:** zero false alarms over about 5 clean bf16 GPU-hours, at a slowdown of 3% or less.
2. **Harm:** simulated faults must make training measurably worse than the normal difference between two
   clean runs: a crash, or a final loss worse than every clean run by more than both the seed-to-seed spread
   and the GPU's own same-seed noise.
3. **Catch:** sdcguard must flag the faults that caused harm in (2) and name the right GPU.

If 1 or 3 failed, the project would stop. Outcome (October 14-15, 2026):

1. **Met.** 0 false alarms in 24,580,560 checks over 5.14 GPU-hours, every matmul checked on every step
   (GPT-2 small). Under 3% slowdown checking every 40 steps on GPT-2 small (+2.82%) and every 20 on a 0.9B
   model (+2.60%).
2. **Met for the crashing fault types.** 15 of 24 faulty runs crashed. The other 9 (zeroing-only and
   mantissa-only faults) ended within normal run-to-run variation, so they don't count as harmed.
3. **Partly met at the time.** Checking every 10 steps caught 14 of 15 harmed runs before the crash, but the
   first version of the check cost +10.9% at that rate; every 40 steps (inside the budget) caught 8 of 15.
   The fast training mode built afterwards brings every 10th step down to +3.9% on GPT-2 small (below). GPU
   naming was tested in the 4-GPU demo: the right GPU by UUID and PCI slot in both runs.

## Models and runs

- GPT-style transformers on TinyStories with the GPT-2 tokenizer, bf16 autocast, AdamW: 124M parameters
  (12 layers, width 768) for the harm study and soak, 0.91B (16 layers, width 2048) for slowdown at size,
  51M for the 4-GPU demo.
- **Harm study:** 4,000 steps per run. 3 clean seeds, plus seed 0 twice to measure the GPU's own run-to-run
  noise. 8 fault settings x 3 fault seeds = 24 faulty runs, each paired with clean seed 0. The settings vary
  where (the attention query and key projections, which [TrainSDC](https://arxiv.org/html/2608.30769) found
  most harmful, or every projection and MLP layer), how often (0.1% or 1% of those matmuls), the shape
  (warp-sized chunks or large parts of a 128x128 tile) and the kind (NVIDIA's mix, zeroing only, or
  mantissa-only bit flips). Full list in [REPORT.md](../results/harm_study_gpt2/REPORT.md). The study
  checked every 10th step; the every-20 and every-50 numbers keep only the checks on those steps.
- **Soak:** 5 workers training clean GPT-2 small with every matmul checked on every step, about 62 minutes
  each.
- **Slowdown:** per-step timing on a dedicated GPU, plain steps against checked steps.

## The fault model

`sdcguard.inject.FaultInjector` corrupts matmul outputs in software, following NVIDIA's gate-level
fault-injection study of a data-center GPU ([arXiv 2605.04213](https://arxiv.org/abs/2605.04213);
simulated stuck-at faults, not field data): about 51% of corrupted values are zeroed, 48% are bit flips (low
bits more often than high), about 1% become NaN or Inf, in periodic patterns aligned to the 32-thread warp.
Control-logic faults corrupt 20% to 75% of an SM's output. The injector models chunks of up to 32 values and
parts of a 128x128 tile, and logs every fault so each alert can be matched to the fault it found. It's a
model of a faulty GPU, not a faulty GPU, which is why the NVBit runs below inject into real hardware.

## False alarms

| run | checks | false alarms |
|---|---|---|
| soak: GPT-2 small training, every matmul on every step, 5.14 GPU-hours | 24,580,560 | 0 (worst score 3.54, alarm at 6) |
| round 3 training runs | 48,705 | 0 |
| serving check on clean inputs | 76,500 layer checks | 0 |
| vLLM on clean traffic | every request | 0 |
| `sdcguard check` sweep, 10 GPUs | 1,683,363 | 0 (worst score 3.67) |

## Training cost

Fast mode (`SDCGuard.protect(model)`), every matmul of the checked steps
([GPT-2](../results/round3_oct18/training/fast_gpt2_CALIBRATION.md),
[0.9B](../results/round3_oct18/training/fast_900m_CALIBRATION.md)):

| checked steps | GPT-2 small (124M) | 0.9B model |
|---|---|---|
| every 2nd | +19.5% | +12.5% |
| every 5th | +7.6% | +5.0% |
| every 10th | +3.9% | +2.6% |
| every 20th | +1.7% | +1.3% |

Fast mode wraps each `nn.Linear`. The first version hooked every PyTorch operation of a checked step
(1,500 to 1,800 per GPT-2 small step) through Python, which is where most of its cost went. Matmuls outside
`nn.Linear`, like attention written with `torch.bmm`, aren't checked in fast mode; `coverage()` reports the
share of a model's matmul work it covers.

The default keeps bf16 matmul partial sums in fp32 on checked steps, which changes rounding slightly on
those steps only. With `exact_reductions=False`, gradients are bit-identical with and without sdcguard (3
bf16 steps of a GPT-2-sized model, [identity.json](../results/round3_oct18/training/identity.json)).

## Serving cost

Llama-shaped models with random weights, torch.compile + CUDA graphs, against no sdcguard
([serve.json](../results/round3_oct18/serving/serve.json)):

| | 1 in 8 layers | every layer |
|---|---|---|
| 7B, decoding (64 sequences) | +6.6% | +44% |
| 7B, prompt (4 x 1024 tokens) | +4.4% | +28% |
| 1.1B, decoding | +17.6% | +88% |
| 1.1B, prompt | +7.7% | +43% |

vLLM 0.31, Qwen2.5-7B-Instruct, vLLM's default compiled mode, offline API, 256 prompts x 128 output tokens,
median of 3 runs ([SUMMARY.json](../results/round3_oct18/vllm/qwen7b/SUMMARY.json)):

| layers checked | output tokens/s | slower than plain vLLM |
|---|---|---|
| none (plain vLLM) | 23,294 | |
| 1 in 8 | 22,612 | 3.0% (1.7% the checks, 1.3% the fp32 reduction setting) |
| every layer | 18,156 | 28% |

64 of the 256 answers were compared and were word for word the same in every configuration. A weight bit
flipped in a checked layer was caught in that layer as "weights changed in GPU memory". The runs set
`VLLM_USE_FLASHINFER_SAMPLER=0`, since FlashInfer's sampler compiles with nvcc at warm-up (decoding is
greedy either way).

How it works: weights don't change while serving, so the weight side of each check is computed once at
load. Each checked layer then costs one extra cuBLAS call and one Triton kernel, which reads the layer's input
and output once and keeps the worst score in a small GPU tensor (an integer atomic max) that is read back
without a sync. CUDA graphs capture both. Before first use the kernel has to match the PyTorch version on
clean and corrupted layers, or the PyTorch version runs; it agreed with the training check in 70 of 70 cases
on the H100. Layer shares use golden-ratio spacing, so any share spreads over every layer type and depth.

## Harm study results

15 of 24 faulty runs crashed after a normal-looking loss curve. When a check looked at a corrupted matmul it
caught 99% of faults that changed their block by 10% or more, 70% of those between 1% and 10%, and none under
1% ([REPORT.md](../results/harm_study_gpt2/REPORT.md)). Whether a fault lands on a checked step is partly
chance. Recomputing from the fault logs ([schedule_analysis.py](../experiments/schedule_analysis.py)),
checking every 10 steps catches about 10 of the 15 crashing runs before the crash on average (14 did in the
study), every 2 steps about 14.5, every step all 15.

## Hardware fault injection (NVBit)

NVBit with TU Berlin's injector ([llm-sdc-training](https://github.com/aaltenbernd/llm-sdc-training)),
patched to switch on and off while running, flipped one bit in an input register of every tensor-core
instruction on lane 0 of SM 0 of an NVIDIA L4 while a small transformer trained and served
([nvbit_check.py](../experiments/nvbit_check.py), [results](../results/round3_oct18/nvbit)):

| flipped bit | effect on the loss | steps flagged (fault on) | alerts (clean, fault off) |
|---|---|---|---|
| exponent (bit 14) | NaN while the fault is on | 30 of 30 | 0 |
| mantissa (bit 4, about 10% error in one input) | none, within 0.0001 of a run without the injector | 30 of 30 | 0 |
| mantissa (bit 1, about 1% error in one input) | none | 0 of 30, inside the noise margin | 0 |

Each run compares a control matmul before and after switching the injector on, and a run where it doesn't
change counts for nothing. With bit 4 the bad lane hit most layers at once (14 to 17 of 17 serving layers
flagged per step), which is why checking 1 layer in 8 can still catch a bad GPU on the first pass. Real
defects can be narrower; that's what checking every layer is for.

## What the first GPU runs changed

- Training without warmup produced attention gradients around 1e-24. Squaring them underflowed and the
  noise estimate collapsed to zero. Differences below 1e-20 are no longer judged: 366 false alarms in 32,850
  checks before, 0 after ([DIAG.md](../results/h100_check_run/runs/diag/ts-gpt2/DIAG.md)).
- One line created a tensor on the GPU inside every check, which made the CPU wait. Fixed, and the GPU
  self-test now runs a checked step under PyTorch's sync debugger.
- fp16 loss scaling overflows to Inf on purpose. An Inf or NaN output now counts as corruption only when a
  bound on the inputs shows a finite computation couldn't produce it.
- Blocks whose outputs are all equal round the same way, which inflated the plain sum's error (clean scores
  up to 10.7). The plain sum now also has to exceed the worst possible rounding.
- Found in review: a Triton reshape before a reduction could report the right score at the wrong row on a
  real GPU. The kernel no longer reshapes, and the first-use comparison checks the location too.
- Every check reads A, B and C right after its matmul, in the same CUDA stream, because FSDP frees gathered
  weights right after each layer.

## Math modes, sensitivity and precise mode (criteria written before the GPU run)

Written October 19, 2026, before `scripts/modal_modes.py` ran on a T4, L4, A100 80GB, H100 and B200 (raw
output will go to `results/math_modes/`), together with the GPU test suite (`scripts/modal_check.py --phase
tests`).

1. **Clean math modes:** 0 wrong results in every mode each GPU supports, and a worst clean score under 3
   (half the alarm line). A mode that reaches 3 on any GPU gets a looser noise model, and a smaller
   sensitivity claim, before launch. FP8's noise model is new: it comes from a published model of the tensor
   cores (Khattak and Mikaitis, arXiv 2512.07004: 13 bits kept below each step's largest term on Ada and
   Hopper, 25 on Blackwell), not from a GPU measurement, so this run is its first real test. Emulating that
   model on the CPU, its worst clean score stays under 0.5 at the inner size it uses.
2. **Unsupported isn't an error:** FP8 runs on the L4, H100 and B200 and shows as unsupported on the T4 and
   A100; BF16 and TF32 show as unsupported on the T4.
3. **Sensitivity:** on the same matmuls, the check of the fp32 accumulator flags errors at least 100 times
   smaller than the check of the bf16 result. (On the CPU the gap is about 1,000 times.)
4. **Precise mode:** 0 false alarms on clean training steps, and more of the injected faults caught than the
   ordinary bf16 checks catch on the same model. Its cost is reported as measured, with no pass line.
5. **GPU test suite:** every test passes.

If any of these fails, launch waits for the fix and a rerun.

## sdcguard check

What it runs: the same checksum checks as training and the screen (alarm at 6); a memory test that writes a
multiplicative-hash pattern into most of the free memory, reads it back, then repeats with the bitwise
inverse; sustained 8,192 x 8,192 bf16 matmul TFLOPS (fp16 on GPUs without bf16) in half-second windows; a
device-to-device copy for bandwidth; and a repeatability test. That last one runs a fixed model (6 layers,
width 1,024, seeded weights) plus one fused attention call three times (before the speed tests, after the
stress test, at the end) with deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and hashes the
outputs. The three runs must match, GPUs of the same model in one machine must match, and a result that
differs from at least 3 agreeing GPUs of the identical setup (from at least 2 sources) on the board is
SUSPECT. Datasheet peaks are dense numbers from nvidia.com, listed with their pages in `gpuinfo.py`.

Criteria for the first sweep, written before it ran: every GPU PASS (a FAIL or SUSPECT on a healthy cloud GPU
is a false alarm to fix before launch); 3 of 3 identical fingerprints on every GPU; the two H100s match and
all L4s match (if not, cross-GPU comparisons become notes instead of verdicts); speeds reported as measured,
with no pass line.

First sweep, October 18, 2026, on Modal (machines on AWS, GCP, Azure and OCI; cards in
[docs/board/seed](board/seed)):

| GPU | cloud | result | results checked | worst clean score | fingerprint | bf16 matmul TFLOPS (of datasheet) | GB/s (of datasheet) |
|---|---|---|---|---|---|---|---|
| B200 (rerun, plain checks) | AWS | PASS | 192,805 | 3.29 | c27adbfea641 | 1,326.3 (59%) | 6,415 (80%) |
| H200 | Azure | PASS | 268,032 | 3.41 | 5613afa2605e | 669.8 (68%) | 4,240 (88%) |
| H100 80GB HBM3 (a) | not stated | PASS | 241,583 | 3.41 | 5613afa2605e | 685.2 (69%) | 3,032 (90%) |
| H100 80GB HBM3 (b) | not stated | PASS | 243,711 | 3.41 | 5613afa2605e | 683.1 (69%) | 3,033 (90%) |
| A100 80GB PCIe | Azure | PASS | 101,834 | 3.12 | a10587e20440 | 229.7 (74%) | 1,670 (86%) |
| L40S | OCI | PASS | 302,747 | 3.10 | a10587e20440 | 172.7 (48%) | 644 (75%) |
| L4 | AWS | PASS | 97,573 | 2.97 | a10587e20440 | 56.9 (47%) | 231 (77%) |
| L4 (pair, GPU 1) | GCP | PASS | 84,531 | 2.97 | a10587e20440 | 53.7 (44%) | 231 (77%) |
| L4 (pair, GPU 2) | GCP | PASS | 84,917 | 2.97 | a10587e20440 | 55.1 (46%) | 230 (77%) |
| T4 (fp16) | AWS | PASS | 65,630 | 3.67 | f70d4a7ce612 | 23.2 (36%) | 233 (73%) |

All four criteria met: 0 wrong results in 1,683,363 checks and 532.6 GiB of memory tests, 3 of 3 matching
fingerprints everywhere, and the two H100s matched, as did all three L4s across two clouds. H100 and H200
even gave identical bits, and so did A100, L40S and L4 (the board still only compares identical setups). Big
matmuls reached 36% to 74% of the datasheet peaks, mostly because power limits hold the clocks back, and
memory copies 73% to 90%. The AWS L4 and the T4 ran their PCIe link at x8 though the cards support x16; their
result cards say so.

The B200's first run stopped in the stress step with a CUDA "misaligned address" fault inside sdcguard's
Triton training-check kernels. A diagnostic on a second B200 hit the same fault in those kernels and nowhere
else (cuBLAS, the plain check and the Triton serving kernel were all fine), so it follows the code, not the
GPU. On compute capability 10 and newer sdcguard now uses the plain PyTorch checks
(`SDCGUARD_TRITON_NEW_ARCH=1` opts back in). The B200 row above is that rerun.
