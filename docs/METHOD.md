# Method

How the numbers in the README were measured. Raw output is in [results/](../results/); runs since October 19
record the code fingerprint they ran.
Unless noted: NVIDIA H100 80GB HBM3 on Modal, PyTorch 2.14.1 (CUDA 13.0), Triton 3.8.0.

Measurements below belong to the implementations and workloads that produced them. In particular, the
October 14-15 soak and October 18 overhead measurements do not validate the later precise-mode refit or
changes to coverage reporting. Those changes need their own checks on the exact release candidate.

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

## Incomplete numerical checks (October 19 review)

The training checksum APIs distinguish a judged result from an incomplete check. If a row, column block,
remainder block or batch contains a checksum that cannot be judged, and no other checksum detects
corruption, the result is `UNJUDGED` (`max_z = -1`, `CheckResult.judged = False`). Examples include nonfinite
inputs and values outside the range the check's arithmetic can evaluate. `CheckResult.ok = True` in this
case means no corruption was established; it does not certify the result as clean. A detected corruption
elsewhere in the matmul retains priority and produces an alert.

The original reduction kept only the maximum score. One finite row could therefore hide another row's
negative `UNJUDGED` score. The current [plain and precise checks](../src/sdcguard/checksum.py) retain a
separate coverage flag. The [Triton training reduction](../src/sdcguard/_group.py) carries that flag through
its tiles and final reduction too. The guard counts the resulting incomplete matmul in `stats["unjudged"]`,
which the screening verdict uses. The flag reports that coverage is incomplete; it does not enumerate all
unjudged blocks. Zero and tiny finite outputs remain judged, subject to the documented absolute noise floor.

[Regression tests](../tests/test_numeric_review_b778.py) cover mixed rows, blocks, remainder blocks and
batches, including a genuine alarm alongside an unjudged result. They passed on CPU and in the Triton
interpreter. The changed Triton buffer layout still needs validation on real GPUs; these tests do not
establish CUDA compilation behavior or the overhead of the change.

Serving has a separate limitation: its packed worst-score state currently maps unjudged numerical cases to
zero, so `ServeGuard` does not provide the same per-check completeness signal. Screening also examines
nonfinite final serving outputs, but that does not establish that every internal serving checksum was
judged. Training's coverage fix does not remove this limitation.

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
potential share of the observed forward matmul work eligible for checking at its actual dtype. It does not
establish executed coverage; inspect the separate guard counters for executed checks, skips and unjudged results.

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

| layers checked | output tokens/s | throughput lost against plain vLLM |
|---|---|---|
| none (plain vLLM) | 23,294 | |
| 1 in 8 | 22,612 | 2.9% (about 1.7% the checks, 1.3% the fp32 reduction setting) |
| every layer | 18,156 | 22% (each request takes 28% longer) |

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

### What the run found (October 19, 2026)

Raw output: [results/math_modes](../results/math_modes), [results/gpu_tests](../results/gpu_tests),
[results/precise_diag](../results/precise_diag), [results/precise_models](../results/precise_models).

1. **Clean math modes: met, after one fix.** 0 wrong results in every mode that ran, on all five GPUs; the
   worst clean score was 1.20 (A100 TF32 and B200 TF32), far under 3. INT8 didn't run anywhere: the check
   summed INT8 results with an integer matmul on the GPU, which CUDA doesn't have, so every card said
   INCOMPLETE. Fixed (float64 sums, exact here); the second run had INT8 at 0 wrong on the H100, A100 and L4.
   FP8 on real GPUs matched what the tensor-core model predicted: worst clean score 0.27 (L4), 0.30 (H100)
   and 0.06 (B200).
2. **Unsupported isn't an error: met.** FP8 unsupported on the T4 and A100, BF16 and TF32 on the T4.
3. **Sensitivity: met on the four bf16 GPUs, not on the T4.** On the same matmuls, half of the errors were
   caught from about 20% of a typical output by the bf16-result check and from 0.055% by the fp32-accumulator
   check: about 360 times smaller on the L4, A100, H100 and B200. The T4 has no bf16, and its fp16 result
   keeps three more bits: 5.5% against 0.17%, 32 times. (That is the math-mode check's noise model on random
   matrices of inner size 256. On real training matmuls with thousands of tokens, precise mode needs a more
   cautious model, and is less sensitive: see item 4 and the next section.)
4. **Precise mode: failed.** 70 to 78 false alarms in 2,970 clean checks on the A100, H100 and B200 (worst
   score 17.9 to 20.3), 3 on the L4, none on the T4. Taken apart check by check: all the high scores were in
   the backward dW matmuls of the attention input and the output layer (4,096 tokens), where per-output errors
   were 3 to 5 times the model's allowance. The model scaled the random part of the rounding with the final
   output, but every product is rounded against the running sum, and there the running sums wander far beyond
   where they end. A third run on the same workload reproduced the failure (72 to 84 alarms on the A100, H100
   and B200, 6 on the L4) and scored two candidate fixes on the same 30 steps on five GPUs: tracking the
   running sums (16, 8 or 4 points along k) still alarmed 6 times on the A100's output layer (worst 7.8) and
   needs extra matmuls; scaling that term by sqrt(k / N) had no alarm anywhere (worst 5.2 on the A100), which
   is too close to 6 to ship. Precise mode stays off by default and is documented as experimental; the next section has
   the model chosen from these runs and the test it has to pass.
5. **GPU test suite: failed, then met.** 247 of 251 passed: two failures were the INT8 bug, two were the test
   machine missing the board's seed cards. After the fixes: 252 of 252, and `sdcguard check` PASS on the L4
   with all 7 math modes.

## Precise mode: the refit noise model and its held-out test (criteria written before the run)

Written October 19, 2026, after the runs above and before `scripts/modal_precise_validate.py` ran.

The refit uses a **statistical estimate** for accumulation noise:
`0.6 sqrt(k) u |a_row| |B_block|`, with `u = 2^-24`, the row's L2 norm and the block's Frobenius norm
([implementation](../src/sdcguard/checksum.py), `ACC_C1` and `_acc_zscores`). Cauchy-Schwarz bounds the
magnitude of an exact partial dot product by the full input row and column norms. It does not establish
the distribution or independence of floating-point rounding errors. The estimate additionally assumes
that their random component combines approximately with `sqrt(k)` scaling; the coefficient 0.6 is motivated
by a roughly uniform rounding-error model. The implementation retains a separate term for the accumulation
bias and a more conservative plain-sum test when output rounding errors align.

These assumptions can fail on correlated workloads or different kernel implementations. This formula is
not a deterministic worst-case error bound, and an alarm threshold of 6 does not by itself establish a
universal false-alarm probability. The constant was selected after the calibration failures above and frozen
before the held-out run. The anticipated calibration scores, about 3.2 on the A100 and 1.5 on the H100, and
the anticipated typical sensitivity, about 1% rather than the earlier model's 0.14 to 0.24%, are projections,
not held-out GPU measurements. The historical default bf16 comparison was about 20% on those experiments.
The held-out GPU run below tested the refit.

The held-out run uses workloads none of the candidates was chosen on (`VALIDATE` in
`experiments/precise_models.py`): a wider model with 8,192 tokens per step and a 50,257-token vocabulary, the
calibration shape with a new seed, a narrow 12-layer model with 8,192 tokens per step, two of them fed
word-like (Zipf) token streams, which make the output layer's gradients far more lopsided than the uniform
random tokens of the calibration run, and higher learning rates. The same five GPUs (H100, B200, A100, L4 in
bf16, T4 in fp16).

Criteria: on every GPU, 0 alarms in every clean check and a worst score of at most 4 (a 1.5x margin under the
alarm line). Coverage must also be complete: no unjudged or skipped checks, and both the diagnostic
calculation and the production guard must execute the expected forward, dX and dW checks of the fixed
workload. A clean diagnostic score cannot substitute for a guard check skipped because of memory pressure.
The run records these counts in `guard_stats` and `expected_checks`
([validation code](../experiments/precise_models.py)).

The run also reports the typical error caught (median and 90th percentile) and the cost per step of precise
mode and of the default check, with no overhead pass line. An overhead-measurement error is reported
separately; meeting numerical criteria does not establish performance if that measurement failed. If the
numerical or coverage criteria fail, precise mode stays experimental and the failure is written up here.

### Sensitivity is local to a block, including in precise mode

Both checks use a block-level noise estimate. A large legitimate output channel can raise the tolerance
enough to hide substantial corruption in a smaller neighboring channel. Keeping fp32 accumulators reduces
rounding noise but does not remove this effect.

An October 19 CPU reproduction uses `A = ones(1, 1).bfloat16()` and
`B = ones(1, 128).bfloat16()`, with `B[0, 0] = 1e10`. Changing output `[0, 1]` from 1 to 1000 still passes
the 128-column ordinary and precise checks, at scores about `0.000026` and `0.130` respectively, against
the alarm line of 6. A one-column block catches this specific error, but checks many more checksums and
has a different cost. This constructed example is a limitation test, not an estimate of field failure
frequency. Reported typical sensitivity is not a guarantee for every output position, magnitude pattern,
or cancellation pattern.

### What the held-out run found (October 19, 2026, evening): criteria not met

Code fingerprint 1265b3e0a385 ([results](../results/precise_validate/SUMMARY.md)).

| GPU | held-out checks | alarms | worst score | criteria |
|---|---|---|---|---|
| H100 | 4,356 | 0 | 3.48 | met |
| B200 | 4,356 | 0 | 3.46 | met |
| A100 80GB | 4,356 | 0 | 4.27 | not met (over 4) |
| L4 | 4,356 | 0 | 4.26 | not met (over 4) |
| T4 | not run | | | not met (incomplete) |

- No false alarm in 17,424 held-out checks: the model from the rounding argument removed every alarm the
  earlier model raised on the same data (5 to 88 per workload, worst 9.6 to 40.7). But the margin the
  criteria asked for wasn't there on the A100 and the L4: both peaked at 4.27 and 4.26 on the same layer and
  pass (the MLP down projection's weight gradient in the wide model, 8,192 tokens). H100 and B200 peaked at
  3.48 and 3.46 there.
- The T4 didn't run: the wide workload needs more than its 16 GB, and the runner lost the other two
  workloads with it. Fixed: a workload that doesn't fit is now recorded and the others still run; a run with
  a skipped workload can't meet the criteria. The runner also exits with 1 when the criteria aren't met on
  every GPU (it exited 0 here).
- Sensitivity: the median injected error caught was 0.32% to 0.64% of a typical output, the 90th percentile
  1.1% to 2.7%.
- Cost, checking every step of the calibration-shape model (median seconds per step): precise mode +261%
  (L4), +306% (H100), +416% (A100), +790% (B200); the default check +8.7%, +23%, +40%, and +774% on the B200,
  which ran the plain PyTorch checks then (see the Blackwell section below). Precise mode computes every
  checked matmul with an fp32 output and checks that, which is where its cost goes. AEGIS reports 1.61% to
  2.35% on average (row, and row-and-column checksums; at most 2.6%) and 0.86% in production with sampling
  ([OSDI '26](https://www.usenix.org/system/files/osdi26-lei.pdf), section 7.4); its paper doesn't describe
  the kernel implementation. Getting near those numbers needs the checksum computed inside the matmul
  kernel, not a different noise model.

As written above: precise mode stays experimental, off by default and out of the README. These workloads have
now been seen, so a revised model would need new held-out workloads and criteria written before its run.

## Attention checks: calibration criteria (written October 19, 2026, before any GPU run)

The matmul checksums can't see inside fused attention (`scaled_dot_product_attention`), and AEGIS found 3 of
its 8 faulty machines through FlashAttention faults ([OSDI '26](https://www.usenix.org/system/files/osdi26-lei.pdf),
Table 2). `SDCGuard(attention=True)` adds two checks (`attention.py`): in the backward pass, the sum of dV over
the keys must equal the sum of dO over the queries (every row of the attention probabilities sums to 1; the
invariant AEGIS uses, section 5.1.1, here on the bf16/fp16 tensors the kernel wrote rather than inside the
kernel), and in the forward pass every output must lie within the range of its head's values. The backward
noise model is u x (|dV column|_2 + |dO column|_2), capped by the worst case u x (|dV column|_1 +
|dO column|_1) as in the matmul checks (near-uniform attention makes all of a column's roundings line up).

On the CPU (bf16 and fp16, 4 shapes, causal and not, GQA): worst clean score 1.5. A single dV value off by
10 times the typical size was caught in 3 to 12 of 12 cases depending on sequence length, by 100 times in 10
to 12 of 12; one output multiplied by 256 in 10 to 12 of 12. It is a coarse check: it catches large errors,
like exponent-bit changes, and it doesn't cover dQ and dK.

`experiments/attention_calib.py` trains three models (2,048 and 4,096 tokens, GQA 16:4, head sizes 32 to 128)
with every step checked, on each fused kernel the GPU offers (flash, memory-efficient, cuDNN), then injects
one error per attention call. Criteria: on every GPU and kernel, 0 alarms on clean steps and a worst clean
backward score of at most 4. Detection rates are reported with no pass line. Until the criteria are met,
attention checking stays off by default and out of the README. For now it works in the default mode only,
not with `protect()`.

## Blackwell: the Triton kernel fault, bisected (October 19, 2026)

On B200 the Triton training-check kernels hit "CUDA error: misaligned address" in their first-use
comparison (Oct 18), so Blackwell ran the plain PyTorch checks, which cost far more (the +774% above). The
bisect ran the comparison of tests/_fast_compare.py once per change, each in its own process
([results](../results/blackwell_triton/b200_20261019_125534.json); Triton 3.8.0, PyTorch 2.14.1+cu130):

| change | launches run | Triton and plain checks agree |
|---|---|---|
| none (control) | faulted on launch 2 | no |
| no software pipelining (num_stages=1) | 364 of 364 | yes |
| 4 warps (no change for that launch, which already had 4) | faulted on launch 2 | no |
| fp32 block sums of B with TF32x3 dots instead of three bf16 parts | 364 of 364 | yes |
| plain fp32 dots, no tensor cores | 364 of 364 | yes |
| A's rows padded to 16 elements | faulted on launch 2 | no |

The fault needs software pipelining and the three-part bf16 path together; its root cause is not known (the
pointers were 64-byte aligned). sdcguard now launches the comparison kernels without pipelining on compute
capability 10 (data-center Blackwell), keeping the bf16 tensor-core path, and turns the Triton kernels on
there (SDCGUARD_TRITON_NEW_ARCH=0 turns them off). Capability 11 and up, including consumer Blackwell
(12), keeps the plain checks until tested. Pending: the full test suite, the serving kernel's comparison and
the kernel timings on a B200 with this change.

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
