# Method

How the numbers in the README were measured. Raw output is in [results/](../results/); runs since October 19
record the code fingerprint they ran.
Unless noted: NVIDIA H100 80GB HBM3 on Modal, PyTorch 2.14.1 (CUDA 13.0), Triton 3.8.0.

Measurements below belong to the implementations and workloads that produced them. In particular, the
October 14-15 soak and October 18 overhead measurements do not validate the later precise-mode refit or
changes to coverage reporting. Those changes need their own checks on the exact release candidate.
The October 20 review started from `12e23f5`, fingerprint `750892567560` over 92 Python files. Its patches
changed that fingerprint; the patched code ran the GPU tests on October 20 at fingerprint 74b844ee7ca8. Where a historical artifact does not record a source
commit or fingerprint, its date and linked artifact are the available provenance; do not assign it the
current candidate's revision.

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
| October 18 vLLM clean-traffic runs | layer shares and compared outputs described below; predates the executed-count gate | 0 |
| October 20 vLLM runs, Qwen2.5-1.5B and 7B, every layer and 1 in 8 | 170,688 layer checks, 97,536 of them in measured traffic | 0 (worst score 2.93) |
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
interpreter, and on October 20 on CUDA on an L4, an H100 and a B200 (code fingerprint 74b844ee7ca8). The
overhead of the change is not measured.

Serving scores a checksum it can't judge as zero, since its packed state holds only non-negative scores,
so a worst score alone can't show completeness. Each layer's state row now also counts the checks executed
(empty batches excluded) and the checksums that couldn't be judged, by the same rule as training
(`checksum._judge_nonfinite`); the Triton kernel adds them with integer atomics, the PyTorch version with
one indexed add, both without a host sync and intended to replay inside CUDA graphs. `ServeGuard.coverage_report()`
returns them; `sdcguard screen` reports measured serving checks instead of passes x layers, and calls a run
INCOMPLETE when a serving layer never ran its check or a serving checksum went unjudged. The first-use
comparison requires the Triton counts to equal the PyTorch counts exactly, and
[the comparison script](../tests/_serve_fast_compare.py) adds an infinite input and an fp16 overflow the
inputs explain (both unjudged) to its cases. Status, October 19: passes on CPU and in Triton's interpreter.
Status, October 20 (code fingerprint 74b844ee7ca8, [results](../results/gpu_tests)): on an L4, an H100 and a B200
the comparison script agreed on every case, unjudged ones included; the first-use comparison found equal counts;
and in a compiled model (`torch.compile(mode="reduce-overhead")`, with the Triton check or the PyTorch check
compiled in), three calls replayed at least three CUDA graphs, counted with PyTorch 2.14.1's graph-replay hook,
and advanced every layer's check count by exactly three; afterwards a flipped weight bit was caught. The test
doesn't yet assert exactly one graph per call, and it would pass silently on a PyTorch without the hook (the GPU
image pins 2.14.1, which has it). The counts' own
overhead is not separately measured. The October 20 review adds per-layer count
differences between the end of vLLM warm-up and the end of measured traffic: startup activity cannot satisfy the gate,
and a zero residual is valid when its check count advances. The vLLM gate then still needed validation inside
real vLLM; a compiled call that falls back to eager execution does not establish it. On October 20 it passed
there, in vLLM's default compiled mode, on the first (compiled) attempt
([below](#vllm-again-october-20-2026-with-the-executed-count-gate)).

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

### vLLM again, October 20, 2026, with the executed-count gate

Same benchmark, both models, one H100 80GB, vLLM 0.31.0 in its default mode (torch.compile with piecewise and
full CUDA graphs), on the code of the head-to-head (fingerprint bbdb51c58ac0; the library code in `src/` is
unchanged since the GPU tests at 74b844ee7ca8, only the runners changed), [results](../results/vllm_oct20).
The rules were in [vllm_check.py](../experiments/vllm_check.py) before the run. Each guarded run passes only
if every registered layer executed a check; every layer's count rose between the end of warm-up and the end of
the timed traffic; no checksum went unjudged after warm-up; clean traffic raised no alert and changed no
weight; a weight bit flipped afterwards was named by both the serving check and the weight checksums; and the
64 compared answers matched plain vLLM's. All four guarded runs passed. Function time 11.7 minutes, about
$0.94 at Modal's October 20 prices.

| model, layers checked | output tokens/s, plain vLLM / with checks | throughput lost | each request takes longer by | worst clean score |
|---|---|---|---|---|
| Qwen2.5-7B, every layer (112) | 22,649 / 17,466 | 22.9% | 29.7% | 2.93 |
| Qwen2.5-7B, 1 in 8 (15) | 22,649 / 21,697 | 4.2% | 4.4% | 2.10 |
| Qwen2.5-1.5B, every layer (112) | 37,515 / 33,997 | 9.4% | 10.4% | 2.26 |
| Qwen2.5-1.5B, 1 in 8 (15) | 37,515 / 35,200 | 6.2% | 6.6% | 2.19 |

"Plain vLLM" is vLLM with sdcguard's fp32 reduction setting and no checks, so the table isolates the checks.
vLLM with PyTorch's default reductions was 0.1% (7B) and 0.7% (1.5B) faster than that, within run-to-run
noise. Each figure is the median of three timed runs of 1 to 2 seconds; the first run of a configuration is
up to 6% slower than the other two, and the 1.5B 1-in-8 figure is the noisiest (its runs were 3% to 8% slower
than plain).

What it shows:

- **The checks ran on the measured requests.** In every guarded run every layer's count rose by exactly 384
  during the three timed runs, 128 per run, the number of tokens each request generated: the same for every
  layer, both models and both shares. Startup and warm-up added 288 per layer and are excluded. No checksum
  went unjudged. One check per layer per forward step would give exactly this if each run's prompts were
  processed in its first step; vLLM's own step count wasn't logged, so that match, and whether the decode steps
  ran as CUDA graph replays, are inferences from the counts, not measurements.
- **No false alarms and no changed answers.** 0 alerts in 170,688 layer checks (97,536 in measured traffic),
  worst score 2.93 against the alarm line of 6; 64 of 64 compared answers word for word the same as plain
  vLLM's in all four runs.
- **The injected fault was named.** A high exponent bit flipped in one weight of `model.layers.10.mlp.down_proj`
  (`layers.11` in the 1-in-8 runs, the middle checked down projection) was flagged in that layer by the serving
  check and by the weight checksums. That flip makes outputs non-finite, an easy fault; this run shows nothing
  about subtle faults inside vLLM.
- **Cost.** On the 7B model it matches October 18 for every layer (22.9% against 22%), but the 1-in-8 cost of
  the checks themselves is higher (4.2% against 1.7%). What changed in between includes the per-layer counters
  added on October 19, whose own cost isn't separately measured. Dividing the extra time of a median run by
  128 steps and 112 layers gives roughly 30 microseconds per check on the 7B model and 6 on the 1.5B, whose
  layers are about half as wide: the check's cost grows faster than layer width. That is a tuning target, not
  yet investigated.

Limits: one H100, the offline API, greedy decoding, one batch shape (256 prompts x 128 tokens), two models,
synthetic prompts, one injected fault of one kind.

How it works: weights don't change while serving, so the weight side of each check is computed once at
load. Each checked layer then costs one extra cuBLAS call and one Triton kernel, which reads the layer's input
and output once and keeps the worst score in a small GPU tensor (an integer atomic max), next to counts of
checks executed and checksums that couldn't be judged (integer atomic adds). Nonblocking polls queue
readback; explicit counts and coverage queries wait for it. Before first use the kernel has to match the
PyTorch version on clean and corrupted layers, or the PyTorch version runs. The October 18 version agreed
with the training check in 70 of 70 cases on the H100; that result predates the new count state and does not
validate its CUDA-graph replay. Layer shares use golden-ratio spacing to spread checks over layer types and depth.

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
  ([OSDI, July 13-15, 2026](https://www.usenix.org/system/files/osdi26-lei.pdf), section 7.4). The paper
  describes float32 checksum accumulation on the chip in section 5.1.1, and row or row-and-column verification
  task packing plus dynamic sampling in section 6. Those implementation and scheduling choices differ from
  sdcguard's separate fp32-output GEMM and checker. Its reported overhead cannot be transferred to this path;
  any kernel integration would need its own coverage, correctness and timing evidence.

As written above: precise mode stays experimental and off by default, with that status disclosed in the README. These workloads have
now been seen, so a revised model would need new held-out workloads and criteria written before its run.

## The default check and precise mode on the same matmuls (written October 19, 2026, before any GPU run)

The numbers above come from different runs: the default check's sensitivity from random matrices of inner
size 256 (math modes), precise mode's from its noise model (6 x noise over a typical output, not injected
errors), and the costs from the held-out run. The October 19 outside review asked for both, measured on one
revision. [experiments/mode_compare.py](../experiments/mode_compare.py) does that on the calibration-shape
model: every linear-layer matmul (forward, dX, dW) of two training steps, three random outputs each, an error
of 0.0001% to 100% of a typical output of the block added before rounding, then the default check on the
bf16 result and precise mode's check on the fp32 result of the same corrupted matmul; then seconds per step
unchecked, with the default check and with precise mode. Runner: `scripts/modal_mode_compare.py` (H100, A100,
L4).

What gets reported, fixed now: every error size, both checks, all trials, per GPU, with 95% intervals; the
size at which each check flags half and 90% of the errors, overall and among errors that changed the bf16
output; clean false alarms and worst scores; step times. There is no pass line, and the sizes won't be changed
after the run. Precise mode stays experimental whatever this shows: this says what its sensitivity buys for
its cost. Limits: injected errors in one output of a matmul's result, not hardware faults; the default check
here is the PyTorch version (the Triton version gives the same scores, tests/_fast_compare.py). A CPU smoke
run (no tensor cores, so not a result) flagged half the errors from 19% (default) and 0.068% (precise).

Clarification recorded October 20, before any GPU result from this paired experiment: the reported 95%
Wilson intervals are nominal, pointwise summaries of the sampled positions in these fixed matmuls. Different
matmuls have different detection probabilities; the intervals do not quantify generalization to new training
runs, models or hardware faults. The same positions are reused across sizes, so the intervals are not a
simultaneous confidence band. Three positions per matmul over two steps provide limited within-layer data.
Keep every planned size and trial visible, including missing scales and unjudged checks; these cannot make
an apparently complete comparison. Completed sensitivity results are saved before the separate timing run.

The 50% and 90% crossing estimates describe the sampled curve, with log-linear interpolation between tested
sizes. If the smallest size already reaches the level, the crossing is at or below that size. A non-monotone
curve can later fall below it again; the crossing is not a guarantee for all larger errors. This clarification
and its regression fix change no sizes, alarm thresholds, sampling choices or numerical acceptance criteria.

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
attention checking stays off by default and is described as experimental in the README. For now it works in the default mode only,
not with `protect()`.

Completeness reporting clarified October 20, before the planned GPU calibration: each configured clean run
must record `layers * steps` judged forward checks and the same number of judged backward checks. Partial
execution, skipped or unjudged checks, and unclassified backend errors are INCOMPLETE; a surviving subset
cannot meet the overall criteria. One exclusion has an explicit capability reason, independent of any score:
a fused kernel PyTorch itself reports as not offered on that GPU ("No available kernel", raised before any
check runs) is recorded in the results and not counted; any other error, a CUDA fault or running out of memory
included, makes the calibration INCOMPLETE. The original
0-alarm and worst-score-at-most-4 requirements remain unchanged, and scores are judged before display rounding.

### What the calibration found (October 20, 2026)

Code fingerprint 74b844ee7ca8, PyTorch 2.14.1 (CUDA 13.0), on Modal. Raw output:
[results/attention_calib](../results/attention_calib), and the end of each log in
[results/gpu_tests](../results/gpu_tests). All three fused kernels (flash, memory-efficient, cuDNN) ran on every
GPU; none was reported unavailable, and nothing in the clean runs was partial or unjudged. The detection
columns count alarms, which never exceeded the number of injected calls.

| GPU | clean backward checks | alarms | worst clean score | criteria |
|---|---|---|---|---|
| B200 | 648 | 0 | 0.138 | met |
| H100 80GB | 648 | 0 | 0.139 | met |
| L4 | 648 | 0 | 0.125 | met |

Detection, one error per attention call, no pass line. The counts were the same on all three GPUs and all three
kernels:

| workload | dV + 1, 4 or 16 x rms | dV + 64 x rms | dV + 256 x rms | output x 2 | output x 8 | output x 64 | output x 65,536 |
|---|---|---|---|---|---|---|---|
| GPT, 8 layers, width 1,024, 2,048 tokens | 0 of 16 | 1 of 16 | 16 of 16 | 7 of 16 | 11 of 16 | 15 of 16 | 16 of 16 |
| GQA 16:4, head size 128, 4,096 tokens | 0 of 8 | 0 of 8 | 8 of 8 | 1 of 8 | 5 of 8 | 6 of 8 | 8 of 8 |
| narrow, 32 heads of 32, 1,024 tokens | 0 of 12 | 12 of 12 | 12 of 12 | 2 of 12 | 5 of 12 | 11 of 12 | 12 of 12 |

The criteria are met for the scope that ran: an L4, an H100 and a B200, bf16 only, 12 steps per workload and
kernel from initialization. No A100, no fp16 (which T4 and older GPUs use) and no other GPUs. Within that scope,
this lifts the condition the rules above set for keeping the check off.
Turning it on by default is a separate decision, not made: it catches only large errors (a dV error of 16 times
the typical size or less was never caught), its cost hasn't been measured, it doesn't cover dQ and dK, and it
doesn't work with `protect()` yet. The worst clean score sat about 30 times under the
line of 4, so the noise model is far more cautious than these workloads needed; a tighter model could catch
smaller errors, but it would need new held-out workloads and criteria written before its run.

## Head-to-head on a real cuBLAS bug (written October 20, 2026, before any run)

The question: when NVIDIA's own GEMM library silently computes wrong results on healthy hardware, which tools notice?
The bug is cuBLAS known issue 6862950 (CUDA 13.4 Update 2 release notes, October 12, 2026): on Hopper, FP16, BF16
and FP8 GEMMs can leave part of the output unwritten when the process is an MPS client with
`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` below 100. It is deterministic and reproducible, so every tool faces the same
wrong math. sdcguard reproduced it on a Modal H200 on October 19 ([results](../results/mps_bug)).

**Setup** ([scripts/modal_h2h.py](../scripts/modal_h2h.py), [experiments/h2h_run.py](../experiments/h2h_run.py)):
one H200 on Modal, one container, everything in sequence. MPS shares 100 (control), 40, 75 and 14, then a short 100
recheck. The MPS daemon is restarted at the share before every tool, with the share set at the daemon and in each
tool's environment. At each share:

1. Ground truth for PyTorch (what sdcguard uses): experiments/mps_unwritten.py, the October 19 probe, against a
   float64 CPU reference.
2. Ground truth for the tools' own calls: [experiments/h2h_probe.cpp](../experiments/h2h_probe.cpp) makes DCGM's
   and gpu-burn's cuBLAS calls (Hgemm, Sgemm, Sgemm with tensor-op math and Dgemm, at their shapes, with their
   inputs) and FP16/BF16 GemmEx, into buffers pre-filled with NaN, against float64, with cuBLASLt logging on. It
   links the system cuBLAS that gpu-burn uses. DCGM links its own static cuBLAS (built with a newer CUDA), so for
   DCGM the probe is a close proxy, not its exact library; a short traced DCGM run at 40% logs DCGM's own kernel
   choices if its cuBLAS honors cuBLASLt logging. The probe also measures whether fresh device memory is zeroed.
3. DCGM 4.7.0 (the newest in NVIDIA's apt repository on October 20): `dcgmi diag -r diagnostic`, its only test that
   checks GEMM results, with its defaults. Source reading was of 4.6.1.
4. gpu-burn (commit 0d19c94) with `-tc` for 60 seconds: its only mode that could reach FP16 math (the default is
   FP32 and `-d` is FP64, which the bug does not affect).
5. `sdcguard check` for 5 minutes, offline.

At 40% also: DCGM with FP16 only, the traced DCGM run, and gpu-burn's default FP32 mode.

**Classification** (in code, tested in [tests/test_h2h.py](../tests/test_h2h.py)). Each (tool, share) cell is
judged against that tool's own ground truth at that share:
- DETECTED: its GEMMs were wrong and it flagged them.
- MISSED: they were wrong and it passed.
- NOT EXERCISED: they were right and it passed.
- FALSE ALARM: they were right and it flagged a result error.
- FAIL (NOT ATTRIBUTABLE): it failed for another reason.
- COULD NOT TEST: it didn't run, timed out or couldn't judge.

What counts as flagged:
- DCGM: the diagnostic test fails with error 52 (faulty memory) or 111 (NaN), or reports faulty or NaN elements.
- gpu-burn: `FAULTY` with errors above 0.
- sdcguard check: FAIL with a wrong count above 0 in training, serving, stress or math modes. SUSPECT and INCOMPLETE
  are recorded, not counted.

**Predictions**, from reading each tool's source (judgment, not results):

| share | DCGM diagnostic | gpu-burn | sdcguard check |
|---|---|---|---|
| 100, 50 | pass | OK | PASS |
| 40 | about even odds: a pass, or a "faulty memory" fail if its FP16 GEMMs hit the bug (a real catch with the wrong label) | OK (its FP32 inputs and self-comparison make a hit unlikely and a deterministic hole invisible) | FAIL through math modes (24 of 24 wrong on October 19) |
| 75 | leaning pass | OK | FAIL through math modes |
| 14 | pass | OK | uncertain |

Why: DCGM and gpu-burn compare every result with their own first result, never with a correct answer, so a wrong
answer that repeats exactly passes, and neither runs BF16. DCGM's FP16 buffers overlap earlier FP64 results, which
could turn unwritten regions into mismatches. sdcguard compares with a float64 reference.

**Reporting.** Every cell is published, including any where sdcguard misses or false-alarms. No step is rerun
because its result was unwelcome. A tool that cannot run here is "could not test", never "missed". This is one GPU,
one software bug and sdcguard's own author running the comparison. It says nothing about physically defective GPUs
or about H100.

**Amendments, written October 20 after an outside review and before any run.**
- MPS evidence: DCGM's diagnostic runs its workload in a child process (nvvs). `dcgmi diag` with no host engine
  running starts an embedded one, and the child inherits the environment (DCGM source, `dcgmi/Diag.cpp` and
  `common/ChildProcess`), so it should be an MPS client at the share; that is checked, not assumed. Every tool's
  step records the new connections in the MPS logs while it ran. If gpu-burn or sdcguard check (known MPS clients)
  show connections at a share and DCGM shows none, DCGM's workload ran outside MPS and could not meet the bug: that
  cell is NOT EXERCISED (not an MPS client), never MISSED. Without that evidence the probe rule above applies.
- A share whose MPS daemon did not start is COULD NOT TEST for every tool.
- The probe also runs cuBLASLt directly, every algorithm the heuristic offers in PyTorch's linear-layer layout,
  and prints each algorithm's ID and cluster shape with its result, plus the first one again with a 1x1x1 cluster
  (NVIDIA's workaround). This is evidence of what goes wrong; it does not change any cell.
- The run exits with an error and prints INCOMPLETE if any of the 12 planned cells (three tools at four shares)
  has no measured result. Those cells are COULD NOT TEST, not passes or misses.
- Every step's full output is downloaded to results/h2h/ with the summary, and the summary records the measured
  function time and the cost it implies at the rates below.

**Preflight** (`--phase preflight`, one H200, about 10 minutes, written October 20 before it runs). Shares 100, 40
and 14: the cuBLASLt cases above plus Hgemm and the two GemmEx calls with cuBLASLt logging on; DCGM's software test
and a 20-second diagnostic without MPS; at 40%, a 20-second DCGM diagnostic and a 10-second gpu-burn under MPS with
the MPS logs and DCGM's cuBLASLt log recorded. It answers, from that evidence:
1. Does MPS work here (the daemon starts at every share, fewer SMs are visible at 40% than at 100%)?
2. Is every algorithm right at 100%?
3. Does the bug reproduce as NVIDIA describes it: below 100%, wrong results only from algorithm 66 with 4 or more
   CTAs per cluster, none from 1 or 2 CTA clusters? Does the 1x1x1 workaround give right results?
4. Does DCGM's diagnostic run here and give readable results, without and with MPS? Did it connect as an MPS client?
   Are its cuBLAS kernel choices visible in a cuBLASLt log?
5. Does gpu-burn run under MPS with a readable verdict?

GO for the full run needs 1, 2 and 3. If 4 shows DCGM is not an MPS client, the full run still runs and its DCGM
cells are reported as above; the write-up says so. If 3 fails (the bug doesn't reproduce, or not as published), the
full run is not started and that result is published as it is.

**What the preflight found** (October 20, 2026, 05:58 Chicago time, Modal H200, driver 610.57.04, container cuBLAS
13.0.2.14, code fingerprint 28ac0adccafd, [result](../results/h2h/preflight_h200_20261020_055833.json)): GO on every
question.
- MPS: 132, 52 and 18 SMs visible at 100%, 40% and 14%.
- 100%: all 32 cuBLASLt algorithm runs right (FP16 and BF16, both shapes, including 14 of algorithm 66 with 4 or
  more CTAs per cluster).
- 40%: 20 of 32 wrong, 14%: 10 of 32 wrong. Every wrong one was algorithm 66 with 4 or 8 CTAs per cluster, most
  with outputs never written; no run with a 1 or 2 CTA cluster was wrong, and the first algorithm forced to a
  1x1x1 cluster was right every time. Not every 4+ CTA configuration was wrong (at 40%, 20 of 32). This matches
  NVIDIA's description of known issue 6862950.
- The legacy calls: FP16 and BF16 GemmEx (what PyTorch uses) wrong at 40%, right at 14% and 100%; Hgemm at 2048
  (what DCGM's diagnostic runs) right at all three shares.
- DCGM 4.7.0: its diagnostic ran and passed with readable JSON, without MPS and at 40%; at 40% the MPS log shows
  "NEW CLIENT ... from user 0" while it ran, so its workload was an MPS client; its cuBLASLt log shows
  cublasLtHHHMatmul on 2048 x 2048 FP16. gpu-burn -tc ran under MPS: OK, 0 errors.
- Two runner bugs found and fixed before the full run: the raw-log download stopped at the first folder (Modal's
  file type is an integer enum), and the cuBLASLt log summary cut lines before the algorithm field.

**What the head-to-head found** (October 20, 2026, from 05:09 Chicago time, one Modal H200, GPU-06a92759, driver
610.57.04, code fingerprint bbdb51c58ac0, [summary](../results/h2h/h200_20261020_060928.md),
[result](../results/h2h/h200_20261020_060928.json)). All 12 planned cells measured; the raw logs (3.8 GB) are on the
Modal volume sdcguard-h2h and on Hari's disk, not in git.

| share (SMs) | DCGM diagnostic | gpu-burn -tc | sdcguard check | PyTorch's own matmuls (ground truth) |
|---|---|---|---|---|
| 100 (132) | NOT EXERCISED, pass | NOT EXERCISED, OK | NOT EXERCISED, PASS | right |
| 40 (52) | NOT EXERCISED, pass | NOT EXERCISED, OK | **DETECTED**, FAIL | wrong |
| 75 (98) | NOT EXERCISED, pass | NOT EXERCISED, OK | **DETECTED**, FAIL | wrong (weight gradients) |
| 14 (18) | NOT EXERCISED, pass | NOT EXERCISED, OK | **DETECTED**, FAIL | wrong |

Extras at 40%: DCGM with FP16 only and gpu-burn's default FP32 mode, both NOT EXERCISED (pass, OK).

- Why DCGM and gpu-burn were NOT EXERCISED, not MISSED: the probe made their exact calls (Hgemm, Sgemm, Sgemm with
  tensor-op math, Dgemm, at 2048 and 8192) as MPS clients at every share, and every one was right. DCGM's own
  cuBLASLt log at 40% shows its FP16 GEMMs used algorithm 39 with a 1x2x1 cluster, never algorithm 66, so the bug
  could not reach its test. Every tool connected as an MPS client at every share (4 new MPS connections during
  each step). Neither tool was wrong to pass its own test; what this shows is that a diagnostic that runs its own
  GEMMs can't vouch for the kernels a job runs.
- What went wrong where it was wrong: FP16 and BF16 GemmEx (PyTorch's path) at 40% and 75%; the 16 x 2048 x 1000
  layer at 40% and 14% (outputs never written); weight gradients at 75%; FP16, BF16 and FP8 math modes at 40% and
  75%. cuBLASLt directly: 20 of 32 algorithm runs wrong at 40%, 16 at 75%, 10 at 14%, all algorithm 66 with 4 or
  more CTAs per cluster; the 1x1x1 workaround right every time.
- sdcguard check's counts (wrong results in training, serving, stress, math modes): 40%: 130, 14,680, 2,984, 72;
  75%: 25, 11,980, 0, 72; 14%: 12,288, 3,072, 0, 0. At 40% and 75% most training checks couldn't be judged (the
  wrong outputs made values non-finite: 67,673 of 67,803 at 40%), and the card says so; the verdict came from the
  checks that could be judged. At 100%: PASS, 0 wrong, no false alarm.
- The pre-registered predictions: DCGM at 40% was "about even odds"; it passed, and the probe and its log show
  why. gpu-burn: pass, as predicted. sdcguard: FAIL at 40% and 75% as predicted, and at 14%, which was
  "uncertain". 100%: all pass, as predicted.
- One unexplained event: Modal's host monitor logged an Xid 31 (GPU memory page fault, a write to an unmapped
  address) on this run's GPU between about minutes 25 and 30, when DCGM's FP16-only and traced runs, gpu-burn's
  default mode and the start of the 75% share ran. No step reported an error, sdcguard check's hardware-error field
  stayed empty, and the 100% recheck at the end was right on every call. NVIDIA's Xid catalog says Xid 31 is
  "typically" an application-level bug but can be a driver or hardware bug; its immediate action is restarting
  the application, not a reset or RMA ([catalog](https://docs.nvidia.com/deploy/xid-errors/610/analyzing-xid-catalog.html)).
  Which process caused it isn't known from inside the container; it changes no cell, because each cell is judged
  against that tool's own GEMMs measured at the same share.
- Limits: one GPU, one software bug, one run, sdcguard's own author running it. It says nothing about physically
  defective GPUs or other bugs.

**Cost**, at Modal's prices on October 20: about $5 for a typical hour on the H200. The 80-minute execution limit
is about $6.60 at the requested CPU and memory. That limit is per attempt, excluding queueing; higher actual use,
startup and a restart after preemption can add cost. The preflight is about $1 (about $1.65 at its 20-minute limit).

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
(12), keeps the plain checks until tested.

Ran October 20 with this change (code fingerprint 74b844ee7ca8,
[results](../results/gpu_tests/b200_20261019_230435.txt)): the test suite passed (569 passed, 1 skipped
because the optional Accelerate package wasn't installed; on GPUs, tests/test_fast.py's interpreter runs are
replaced by the same comparison scripts on CUDA), the training and serving Triton checks agreed with the
plain checks, and the installed wheel's selftest and `sdcguard check` passed. Kernel timings for one checked step of
a GPT-2-small-shaped model (16,384 tokens, all 219 linear-layer matmuls of the forward and backward passes, each shape timed on its own as the median
of 10 calls,
`experiments/bench_kernels.py`): with GPT-2's vocabulary of 50,257, the matmuls took 33.9 ms, the Triton check
28.2 ms and the plain check 152.6 ms; with the vocabulary padded to 50,304, the matmuls took 10.4 ms and the
Triton check 23.0 ms. The unpadded output layer is slow on this GPU for cuBLAS and for the check's staging step
alike (its forward pass: 10.2 ms of matmul and 4.2 ms of staging, against 0.8 and 0.6 ms padded). So on a
checked step the Triton check is about 5.4 times faster than the plain check, but with a padded vocabulary it
still takes about 2.2 times as long as the matmuls it checks. The training slowdown depends on how often steps
are checked and on the time spent outside matmuls; it wasn't measured on the B200 in this run. The root cause of
the pipelined fault is still unknown.

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
GPU. At the time of that sweep, compute capability 10 and newer defaulted to the plain PyTorch checks
(`SDCGUARD_TRITON_NEW_ARCH=1` opted back in). The B200 row above is that historical rerun. The later
[Blackwell bisect and current policy](#blackwell-the-triton-kernel-fault-bisected-october-19-2026) enable the
one-stage Triton path on capability 10; the patched candidate passed the GPU test gate on a B200 on October 20,
and the sweep hasn't run on this source yet.
