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
| October 18 training runs ([round3_oct18](../results/round3_oct18/training)) | 48,705 | 0 |
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

[Regression tests](../tests/test_numeric_completeness.py) cover mixed rows, blocks, remainder blocks and
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
errors), and the costs from the held-out run. Here both are measured on one
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

### The expected side's rounding floor (changed October 21, 2026, before any GPU run)

Each checksum's noise has a floor for the fp32 rounding of the expected side, which matters only when the outputs
nearly cancel. It was 1e-5 x |a_row|_2 x |checksum column|_2. For a peaked row of A (attention probabilities with
a sink) that bound is up to sqrt(k) times too large: on CPU, in P @ V with 98% of each query's probability on a
key whose value row is zero, one output off by 50 times a typical output scored 2.9 at 2,048 tokens, under the
line. The floor is now the smaller of that and c(k) x |a_row|_1 x max|checksum column|, with c(k) = 1e-5 up to
k = 1,342 and k x 2^-27 above. The k-scaling is there because on NVIDIA GPUs every check computes the expected
side on tensor cores (A in its own precision against a three-way split of the block sums), and tensor cores
truncate each step's sum, an error that grows with k when the running sum dwarfs the products: in a CPU
emulation of that truncation, a clean dQ = dS K of an attention backward with a sink, at k = 131,072, scored 7.8
against a fixed 1e-5 (a false alarm) and 0.16 against c(k). The cost is at long sequences: the same 50x error
in sink P @ V now scores 23 at 2,048 tokens, but 3.1 at 8,192 and 0.37 at 32,768, under the line, so there this
kind of error in a near-cancelling output is not caught. The training, serving and Triton checks use the same
floor; [tests/test_checksum.py](../tests/test_checksum.py) has these cases. Not yet run on a GPU.

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

**Amendments, written October 20, before any run.**
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

**What the preflight found** (October 20, 2026, 04:58 Chicago time, Modal H200, driver 610.57.04, container cuBLAS
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

- Why DCGM and gpu-burn were NOT EXERCISED, not MISSED: the probe made their calls (Hgemm, Sgemm, Sgemm with
  tensor-op math, Dgemm, at 2048 and 8192) as MPS clients at every share, and every one was right. The probe links
  the system cuBLAS, which gpu-burn uses; DCGM links its own static cuBLAS, so for DCGM at 75% and 14% this rests
  on the probe as a close proxy (setup, step 2). DCGM's own library was logged only at 40%: its cuBLASLt log there
  shows its FP16 GEMMs used algorithm 39 with a 1x2x1 cluster, never algorithm 66, so the bug could not reach its
  test. Every tool connected as an MPS client at every share (4 new MPS connections during each step). Neither
  tool was wrong to pass its own test; what this shows is that a diagnostic that runs its own GEMMs can't vouch
  for the kernels a job runs.
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

## The fused precise check (written October 20, 2026, before any GPU run)

**Why.** AEGIS reports that in deployment, checksums from bf16 outputs located none of its incidents and every
confirmed checksum detection used fp32 accumulators ([OSDI '26](https://www.usenix.org/system/files/osdi26-lei.pdf),
section 8). Precise mode checks the fp32 result, but gets it from a second, fp32-output matmul, and costs +261% to
+790% per checked step (above).

**What it is.** `SDCGuard(precise="fused")` ([fused.py](../src/sdcguard/fused.py)) computes each protected linear
layer's forward, dX and dW matmuls in a Triton GEMM whose epilogue, before rounding, reduces every 128-column block
of each output row of the fp32 accumulator to four float64 numbers: the plain sum, the signed sum, the sum of squares
and the sum of absolute values. The layer's bf16/fp16 output is rounded from the same accumulator. The check then
applies precise mode's frozen noise model, unchanged, to those numbers.

**What it does not change, and what it gives up.** The expected side is still computed in float64 (the inputs times
the block sums of the weights), and that may be a large share of precise mode's cost, mostly on GPUs with slow
float64 such as the L4; this run measures the split. On checked steps those matmuls run in Triton instead of cuBLAS,
so a defect specific to cuBLAS kernels (like known issue 6862950) is not exercised on them; the default check is the
one that watches cuBLAS. Output bits on checked steps can differ from cuBLAS's in the last place (a different
summation order of the same fp32 math).

**Seen so far, in Triton's CPU interpreter only (not a GPU result).** Outputs were the fp32 result rounded to
nearest; the block statistics matched float64 sums; an error of 0.2% of a typical output in one value, under half a
bf16 step, scored 21 with the fused check and 0.96 with the default check of the rounded output; in a two-layer
model every forward, dX and dW matmul went through the fused check, clean, with gradients matching precise=True
([tests](../tests/_fused_compare.py)). The interpreter rounds fp32 to bf16 by truncation, so the kernel rounds to
nearest explicitly there, as GPUs do.

**Run:** `scripts/modal_fused.py` on an H100 and an L4 ([runner](../experiments/fused_check.py)). Criteria, per GPU:

1. Correctness: `tests/_fused_compare.py` passes on the GPU.
2. Clean: on two training workloads not used before (the calibration shape with seed 41 and the wide model with
   seed 43, both with Zipf tokens), every fused check scores at most 4 (the line the precise-mode validation used),
   with no alarms, no unjudged checks and no fallback to precise=True.
3. Same sensitivity: on three real-sized matmuls, the smallest injected error (a share of a typical output) that
   the fused check flags at least half the time is within a factor of 2 of precise mode's.
4. Cost: with every step checked, the fused check's overhead per training step is at most half of precise=True's
   on the same GPU.
5. Gradients (added October 20, before any run): one training step of the
   calibration-shape model with precise=True and with precise="fused", same weights and data; every parameter's
   gradient within 2^-6 of the precise=True one, relative to that parameter's largest gradient (the tolerance the
   precise-mode tests use against a plain step), with every protected matmul going through the kernel.

What this checks is the Triton kernel's own matmuls, not cuBLAS: the run records how many matmuls went through
the kernel, and a pass here says nothing about cuBLAS on those calls.

**Reporting.** All four criteria on both GPUs are published, including failures, and so is the Triton kernel's
speed against cuBLAS, whatever it is. If criterion 4 fails, the timing split (the GEMM against the float64 check)
says which part to work on next. Criterion 2 can fail on the L4 for the same reason precise mode's validation did
(4.26 there). Precise mode and the fused check stay experimental whatever this shows: it is not a held-out
validation of the noise model.

**Cost**, at Modal's prices on October 20 (checked that day): about $2 for both GPUs together with the recompute
check below, at most about $3.80 at the 40-minute limits. Partial results are saved as each part finishes; a run
that stops early or misses any criterion exits with an error.

### What the run found (October 20, 2026): as sensitive as precise mode, not cheaper

Code fingerprint 27b4d3e58291, one H100 80GB and one L4, PyTorch 2.14.1, Triton 3.8.0
([summary](../results/fused/SUMMARY_20261020_073624.md), [H100](../results/fused/h100_20261020_073624.json),
[L4](../results/fused/l4_20261020_073624.json); every line each GPU printed is in the `_log.txt` files). Function
time 75 s on the H100 and 245 s on the L4, about $0.17 in total at the rates above.

| criterion | H100 | L4 |
|---|---|---|
| 1 correctness | met | met |
| 2 clean, worst score at most 4 | met (worst 3.38) | **not met** (worst 4.11; precise mode on the same matmul: 4.11) |
| 3 sensitivity within 2x of precise mode's | met (equal at all three shapes) | met (equal) |
| 4 overhead at most half of precise=True's | **not met** (+366% against +364%) | **not met** (+198% against +261%) |
| 5 gradients within 2^-6 of precise=True's | met (identical) | met (largest 0.0066) |

- **Correctness on the GPUs.** An error of 0.2% of a typical output in one value, under half a bf16 step, scored
  21 with the fused check and 0.91 (H100) or 1.03 (L4) with the default check of the rounded output. Every check
  of both workloads went through the kernel (990 and 306 on each GPU), with no fallback, no unjudged check and no
  alarm.
- **Sensitivity.** At all three shapes the smallest error caught at least half the time was the same for the
  fused check and precise mode: 0.5% of a typical output in one value at two shapes, 3% at the third. The
  default check caught none of the injected errors, up to 10% of a typical output in one value.
- **Where the time goes** (criterion 4 failed; the split was to say which part to work on). The fused GEMM took
  1.7x to 2.1x cuBLAS's bf16 GEMM on the H100 and 1.3x to 2.1x on the L4; precise mode runs cuBLAS's bf16 GEMM
  plus an fp32-output one of 1.1x to 1.3x, so the GEMM side came out about even. The step that turns the block
  statistics into scores, which computes the expected side in float64, took 0.99 to 1.50 ms per matmul on the
  H100 and 3.1 to 23 ms on the L4: 1.6 to 11.5 times the fused GEMM. Precise mode's own scoring took the same
  1.0 ms on the H100 at the four smaller shapes, so the two cost the same per matmul there; at the two largest
  shapes, and on the L4, the fused check cost 51% to 90% of precise mode's time. The scoring step is what to
  work on next. That it stayed near 1 ms on the H100 across four shapes points at a fixed cost per call
  (launches or a host sync) rather than arithmetic; the run did not separate the two.
- **Clean scores.** On the H100 fused and precise scores were equal on the calibration workload (largest gap
  0.0) and within 0.93 on the wide one. On the L4 the fused scores ran higher than precise mode's on the same
  matmuls (up to 2.75 against 0.91 on the calibration workload): the frozen noise model, fitted to cuBLAS's fp32
  accumulators, fits the Triton kernel's on the L4 less well. The L4's worst score, 4.11, came from a matmul where
  both modes scored 4.11, the same kind of miss as precise mode's held-out run (4.26 on an L4). The alarm line is
  6; nothing alarmed.
- **What it checked.** On checked steps all 99 protected matmuls of a step ran in the Triton kernel, not cuBLAS,
  so this says nothing about cuBLAS on those calls.

Outcome: the fused check matches precise mode's sensitivity and stays below the alarm line, but it is not the
cheap fp32 check it was built to be. It stays experimental and off by default.

## The recompute check on GPUs (written October 20, 2026, before any GPU run)

**What it is.** `SDCGuard(recompute=True)` ([recompute.py](../src/sdcguard/recompute.py)) fingerprints the outputs of
every activation-checkpointed block in the forward run and again in the recompute training already does in the
backward pass, and compares the fingerprints: the method behind 12 of AEGIS's 18 production detections (section
7.2; AEGIS uses xorsum fingerprints, sdcguard three int64 sums of the raw 16-bit words (bytes for 1-byte types),
mod 2^64: one plain and two with pseudo-random odd weights of each word's position). Outputs that differ in one
word always get different fingerprints; outputs that differ in several words get the same one only if the
difference cancels in both weighted sums, a chance of at most about 2^-48 for each sum if the weights behave like
random numbers. (The October 20 runs below used the earlier weights, small integers of a row and a column of a
251-wide view, which structured changes could cancel, such as three neighbouring words changed by +1, -2 and +1;
changed October 21.) It covers every operation inside those blocks, not just matmuls. It can't see a fault that
repeats exactly in both runs, blocks that aren't checkpointed, or blocks checkpointed with
`preserve_rng_state=False` (skipped and counted: their dropout differs by design). A mismatch says the two runs
differed, not why.

**Found and fixed on CPU before any GPU run** ([tests](../tests/test_recompute_models.py)). Hugging Face's
`gradient_checkpointing_enable()` and torch's `checkpoint_wrapper` store a partial of `torch.utils.checkpoint`
on each module, so a check installed afterwards compared nothing (shown on GPT-2: 0 blocks). The check now routes
those references when it is given the model (`protect()`, `watch()`, `install(model)`), and a checked step that
compares no block warns. On GPT-2 and Llama configurations (reentrant and not), torch's checkpoint_wrapper and an
explicit `early_stop=True`: every block compared, no false alarm with dropout on, a fault in the forward run or
in the recompute caught and its block named, gradients bit-identical with and without the check.

**Run:** the same Modal run as the fused check (`scripts/modal_fused.py`, H100 and L4;
[runner](../experiments/recompute_check.py)). GPT-2 and Llama configurations with random weights, 8 layers,
width 768, 8 sequences of 512 tokens, bf16 autocast, dropout 0.1, gradient checkpointing turned on before the
check. Criteria, per GPU:

1. Clean: 10 steps per model, every checkpointed block compared, 0 mismatches.
2. Faults: 8 trials per model, a value change in one call of one module inside one block, in the forward run or
   the recompute (alternating), in a different layer each time: all 8 caught, and the block named right.
3. Gradients: bit-identical with and without the check, same seeds.

Reported with no pass line: the share of parameters inside compared blocks; seconds per step with plain
checkpointing, with early stop off only (the check makes the recompute run the whole block) and with the full
check; and a block with `index_add_` (atomic adds on CUDA) calibrated with deterministic algorithms off (expected
to differ between runs: legitimate nondeterminism, which is what `calibrate()` exists for) and on.

**Reporting.** All three criteria on both GPUs, including failures. If 1 fails, the block that differed is
named and the check stays experimental. The recompute check stays experimental and off by default whatever this
shows: two random-weight configurations are not production models.

### What the run found (October 20, 2026): criterion 3 not met, and the test couldn't say why

Same run and files as the fused check above.

| criterion | H100 | L4 |
|---|---|---|
| 1 clean | met: 0 mismatches in 160 compares (80 per model) | met: 0 in 160 |
| 2 faults caught, block named | met: 16 of 16 | met: 16 of 16 |
| 3 gradients bit-identical with and without the check | **not met** (both models) | **not met** (both models) |

- **Coverage.** All 8 checkpointed blocks were compared on every step; 69% of GPT-2's parameters and 58% of
  Llama's sit inside them (the embeddings, the final norm and the LM head are outside).
- **Cost, every step checked.** +20.3% (GPT-2) and +16.0% (Llama) per step on the H100, +10.0% and +16.2% on
  the L4. Turning early stop off alone changed the step time by -2.8% to +1.9%, so the cost is the
  fingerprints. AEGIS's 0.86% is with sampling; checking a share of steps would cut this in proportion (not
  measured).
- **Legitimate nondeterminism.** The `index_add_` block differed in 1 of 5 recomputes with deterministic
  algorithms off (the block named `_Scatter`) and in 0 of 5 with them on, on both GPUs, as expected.
- **Criterion 3.** The runner compared one step with the check against one step without it and kept only
  whether every gradient matched bit for bit. It ran no control (two steps without the check), so it can't tell
  a change made by the check from the GPU's own run-to-run differences, and those exist here: PyTorch 2.14.1
  documents `nn.Embedding`'s backward on CUDA as nondeterministic unless deterministic algorithms are on, and
  treats SDPA's memory-efficient attention as a nondeterministic algorithm
  ([pytorch#91979](https://github.com/pytorch/pytorch/pull/91979)). On CPU, where the kernels are
  deterministic, the gradients matched bit for bit. That was a gap in the test's design, written before the run.
  The criterion stays not met; the follow-up below is written to separate the two causes.

### Follow-up: gradients with a control run (written October 20, 2026, after the run above, before this one)

**Question.** Did the check change the gradients, or do two steps without it already differ on these GPUs?

**Run:** `scripts/modal_grad_control.py` on an H100 and an L4 ([runner](../experiments/recompute_grad_control.py)),
the same models and sizes. Per model and setting: one warm-up backward pass, discarded, then four backward passes
of one step with the same weights, data and seeds: A without the check, B without it again (the control), C with
it, and D without it after it was uninstalled. Two settings, each in its own process: PyTorch's defaults, as in
the first run, and a deterministic one (deterministic algorithms on, with warnings about operations that still
ran nondeterministically recorded; `CUBLAS_WORKSPACE_CONFIG=:4096:8`; cuDNN deterministic; attention on SDPA's
math backend).

**Decision, per GPU and model, from the deterministic setting:** INCONCLUSIVE if C didn't compare every block
cleanly or A and B differ; UNCHANGED if A = B and C and D both equal A; CHANGED if A = B and C or D differs.
Criterion 3 counts as met in the follow-up only if both models are UNCHANGED on both GPUs; the first run's "not
met" stays in the record either way. The default setting is reported with no pass line.

**Predictions.** Default setting: A and B differ, and A and C differ by an amount of the same order. Deterministic
setting: A = B, and both models UNCHANGED. If A and B still differ there, the recorded warnings should name what
didn't run deterministically.

On CPU every comparison is bit-identical, and a check rigged to change one gradient by one bit comes out CHANGED
([tests](../tests/test_grad_control.py)).

**Cost**, at Modal's prices on October 20: about $0.25 for both GPUs, at most about $1.40 at the 15-minute
limits. A run that stops early, or doesn't come out UNCHANGED everywhere, exits with an error.

#### What the follow-up found (October 20, 2026): UNCHANGED on both GPUs, as predicted

Code fingerprint ca941aadb026, one H100 80GB and one L4 ([summary](../results/grad_control/SUMMARY_20261020_082418.md),
[H100](../results/grad_control/h100_20261020_082418.json), [L4](../results/grad_control/l4_20261020_082418.json)).
Function time 30 s and 37 s, about $0.05.

| setting, model | A vs B (control) | A vs C (with the check) | A vs D (after it) | decision |
|---|---|---|---|---|
| default, GPT-2, H100 | 89 of 100 differ, max 7.5e-3 | 90 differ, max 7.2e-3 | 90 differ | |
| default, Llama, H100 | 63 of 75 differ, max 9.4e-3 | 66 differ, max 9.3e-3 | 66 differ | |
| default, GPT-2, L4 | 88 of 100 differ, max 1.1e-2 | 78 differ, max 8.2e-3 | 88 differ | |
| default, Llama, L4 | 66 of 75 differ, max 9.4e-3 | 66 differ, max 9.4e-3 | 66 differ | |
| deterministic, both models, both GPUs | identical | identical | identical | **UNCHANGED** |

(Differences are relative to each parameter's largest gradient.) In PyTorch's default setting two steps without
the check already differ from each other about as much as a step with it, while the losses were identical: the
differences come from the backward kernels, not the check. In the deterministic setting every comparison was
bit-identical, the check compared all 8 blocks of every model with no mismatch, and no nondeterminism warning
was recorded. So the first run's criterion 3 failure came from the test's missing control, not from the check;
the check leaves gradients bit-identical wherever the training itself is deterministic. The first run's "not
met" stays above as it was recorded. The recompute check stays experimental and off by default.

## 0.4.0 on GPUs (written October 20, 2026, before any GPU run)

**What changed since the runs above.**

- The fused check scores its fp32 accumulator inside the kernel: the expected side, the noise model and the
  score, with no float64 and no extra pass, because the October 20 run showed the float64 scoring was the cost.
  That is the default; `SDCGUARD_FUSED_FP64=1` keeps the version the October 20 run measured. The expected side,
  x times the block sums of W, is accumulated in fp32 by a second, narrow dot in the same loop over k, each block
  sum split into three bf16 or fp16 parts that add up to it exactly. In fp16 the block sums are first divided by
  a power of two, also exactly, so sums above fp16's 65504 still fit; that needs the block's largest |W| first,
  so fp16 reads W twice, and the dot's result is multiplied back. The noise is precise mode's frozen model,
  unchanged, plus a term for that dot's own fp32 rounding (its final form is change 2 below); the alarm line is
  unchanged.
- The recompute check fingerprints with one Triton pass instead of several PyTorch reductions, by default on
  CUDA.
- New in 0.4.0, tested before this run on CPU only (the kernels in Triton's CPU interpreter): `protect()` under
  `torch.compile(fullgraph=True)` (a custom operator on checked steps); tensor parallel (each rank checks its own
  shard, inside a compiled graph too, through DTensor sharding rules for the checked operator); MoE grouped GEMMs
  checked per expert; a known-issue report that matches the machine against NVIDIA's cuBLAS known issues;
  `sdcguard blas` (every cuBLASLt algorithm for a shape: every output checked for having been written, a sample
  of up to 32 rows against float64), whose cuBLASLt calls need a GPU; and original-event capture in serving
  (`serve.CAPTURE_SLOTS`: when a layer's check alarms, the check itself copies that row of the layer's input and
  that block of its output into a small buffer on the GPU, and the alert carries `screen.cpu_recheck` of them).

The GPU results above belong to the earlier versions.

**Run:** `scripts/modal_v040.py`: one H100, two H100s in one machine, and one H200 under MPS, in parallel
([runners](../experiments/v040_check.py), [tensor parallel](../experiments/v040_tp.py)). The criteria below are
the ones the runs were judged by. Every change made to them is listed after them, with when it was made.

**H100, the fused check** (`experiments/fused_check.py`, its criteria above, plus the in-kernel scoring):
criteria 1, 2, 4 and 5 as written above; criterion 3 must hold for both the old scoring and the in-kernel one,
each measured in this run (a result with no in-kernel measurement doesn't meet it). What this can't show: the
in-kernel scoring computes the expected side in the same kernel, on the same GPU, as the output it checks, so a
fault that corrupts both the same way stays invisible to it. The default check (cuBLAS's output against a
separate checksum matmul) and the CPU recheck of saved evidence don't share that path, which is one reason the
fused check stays experimental and off by default whatever this run shows. **Prediction:** criterion 4
(overhead at most half of precise=True's) is met now; the Triton GEMM itself took 1.3x to 2.1x cuBLAS's time on
October 20, so the checked step can't get cheaper than that.

**H100, the recompute check** (`experiments/recompute_check.py` and `experiments/recompute_grad_control.py`):
1. clean: 0 mismatches; 2. every injected fault caught and its block named; 3. gradients UNCHANGED against the
deterministic control, as in the follow-up above; 4. with every step checked, at most 10% per step on both
models (it was +20.3% and +16.0% with the PyTorch fingerprints). **Prediction:** 1 to 3 met, and 4 met.

**H100, `experiments/v040_check.py`:**
1. the fingerprint kernel equals the PyTorch version bit for bit on every tensor `tests/_fingerprint_compare.py`
   tries, and is the one in use;
2. a model compiled with `torch.compile(fullgraph=True)` and Inductor runs the same number of checks on each
   checked step as eager, none on unchecked steps, no alarm and nothing unjudged in 40 steps, and a wrong output
   inside a compiled checked step is caught 5 times out of 5;
3. a compiled unchecked step takes at most 2% (plus 0.5 ms) longer than the same step compiled without sdcguard;
4. `torch._grouped_mm` with 64 experts in bf16, forward and backward, random routing: no alarm and nothing
   unjudged in 200 calls, with every expert that got tokens checked in all three matmuls of every call (an expert
   with no tokens adds one check, of its weight gradient, which must be all zeros), and one wrong value in one
   expert's output, changing its 128-column block by 10% of the block's L2 norm, caught in at least 99 of 100
   trials. That is the harm study's measure of a fault's size (the norm of the change over the norm of the
   block, `inject._rel_change`), at which the default check caught 99% of faults there;
5. `sdcguard blas` on its four default shapes, bf16 and fp16, with the GPU to itself: every algorithm right;
6. the serving check's first-use comparison, which also requires the Triton check's captured rows to be the
   PyTorch check's bit for bit, passes in bf16 and fp16, and the Triton check is the one in use; then a bf16
   stack of 4 linear layers, 4096 wide, batch 8, attached to a ServeGuard and captured once in a CUDA graph,
   replayed 200 times with fresh inputs copied into its input buffer. On one replay the graph makes one wrong
   value in layer 1's output, as large as that 128-column block's L2 norm (a fault tensor it adds, zero on every
   other replay); on another, layer 2's check has a wrong expected side for that replay only. Met when there are
   exactly two alerts, those two: the first with its captured row and block where the wrong value was made, the
   recheck finding exactly one output outside the rounding envelope (verdict `output_wrong`), and its evidence
   file holding the wrong value; the second with verdict `output_agrees` (the output isn't blamed);
7. the Triton check of one of those layers takes at most 5% longer with the capture buffer than without it, on
   clean inputs (median of 5 rounds of 200 calls). The copy itself runs only when a row is over the line;
8. capture under contention, in a CUDA graph of 6 bf16 linear layers side by side (4096 to 4096, each applied
   to the same input of batch 8, so no layer's wrong value becomes another's input) with the Triton check on
   every one, read after every replay unless said (each wrong value as large as its block's L2 norm, as in 6):
   (a) every row of one layer wrong once in every block, so every program of that layer's check is over the line
   at once and they race for the layer's one record: exactly one alert, on that layer, keeping one of the rows
   and blocks made wrong, with the recheck finding exactly one output of that block outside the rounding
   envelope; (b) one wrong value in each of 4 layers: four alerts, each keeping its own row and block; (c) one in
   each of the 6 layers, more than the 4 records: four alerts keep theirs, the other two say the buffer was
   full, and a replay after it with one wrong value in one layer keeps that row again; (d) one wrong value in a
   layer, read without waiting, then one in another layer, read without waiting, then a read that waits: the two
   alerts in order, each with its own row. Clean replays before and between raise no alert. Every kept row's
   recheck says `output_wrong` with exactly one output outside the envelope (so the input row and the output
   block kept belong together).

Each of 1 to 8 is met only by its stage run in full as written (all its calls, steps, sweeps, fault trials, the
CUDA graph and the timing).

What 6 to 8 can't show: the wrong values are made by sdcguard's own injection, not by a faulty GPU; the captured
numbers are copied by the GPU being suspected, so a fault that also hit the copy would make a right output look
wrong; the recheck uses the weights as they are when the alert is read (an alert says when they changed since
load); and everything runs on one stream, so races between streams or between graphs replayed concurrently
aren't tested. What 8 (a) checks is a property worth stating: when several rows of a layer alarm in one read, the
Triton check keeps the row of whichever program claims the layer's record first, which need not be the row the
alert names as the worst; the PyTorch check keeps the worst. Related work (checked October 21, 2026): ByteDance's
AEGIS (OSDI '26) keeps a small slice of a failing matmul's inputs and outputs in training and confirms it by
replaying on the GPU; FP-Sketch (arXiv, September 2026) rechecks at fp32 on the same GPU and keeps no copy. As far
as public sources show, neither keeps the failing row inside a CUDA graph during serving and rechecks it in
float64 to tell a wrong output from the check's own side.

**In the next vLLM run** (`experiments/vllm_check.py`, as on October 20), inside vLLM's own compiled graphs: the
alert that names the layer whose weight had a bit flipped must also keep its row, with a recheck that finds the
weights changed since load.

**Two H100s, tensor parallel over NCCL** (`experiments/v040_tp.py`), in both `protect()` and the default mode: no
alarm in 50 clean steps with every step checked and every matmul checked as a local shard: on each rank, the
checked count and the tensor-parallel checked count both equal 50 steps times the 23 local matmuls of a step (8
linear layers, forward, dX and dW, less the first layer's dX), and the skipped counts (tensor-parallel, out of
memory, data type) are all 0. With `exact_reductions=False`, gradients bit-identical to the same tensor-parallel
model without sdcguard. One wrong value in rank 1's local output, changing its 128-column block by 10% of the
block's L2 norm, on 10 checked steps: caught on rank 1 all 10 times and never on rank 0. The same model with
`protect()` under `torch.compile(fullgraph=True)` and Inductor, with the same clean and fault criteria (the wrong
value made inside the checked operator, where the GPU computes the layer, and no compiled tensor-parallel layer
skipped); no gradient comparison there, since Inductor's own kernels may round differently from the operator's
cuBLAS calls.

**H200 under MPS:** at 100% of the SMs, `sdcguard blas` finds every algorithm right on the bug's shape
(1000x2048x16) and on 4096x4096x2048; at 40%, at least one algorithm is wrong, every wrong one is algorithm 66
with a cluster of 4 or more CTAs, and the 1x1x1-cluster workaround, run on every algorithm-66 configuration with 4
or more CTAs, is right; `sdcguard issues` says 6862950 applies at 40% and not at 100%. **Prediction:** all met;
this is the same bug the head-to-head caught.

A wrong algorithm in H100 criterion 5 or at 100% on the H200 fails that criterion even when the cause is
NVIDIA's (sdcguard's own table lists 6580689, a split-K bug in algorithm 66, as matching compute capability 9.0
with the image's cuBLAS). So the runners report each wrong configuration with the listed cuBLAS issues
(`sdcguard.known_issues`) that apply or may apply to the machine and match it (the same algorithm; 4 or more CTAs
where NVIDIA says so; split-K only if it ran with split-K), and the summary lists those apart from wrong
configurations no listed issue accounts for. The criterion stays not met either way; the report says whose bug it
is likely to be, not that it is settled.

**Reporting.** Every criterion on every GPU is published, met or not. If the fused check's cost criterion fails
again, the kernel timing says whether the GEMM or the epilogue is the cost. Everything new stays experimental
and off by default (except where it only adds coverage: tensor parallel, compile, grouped GEMMs, and the
reports) until a run like this one passes.

**Cost**, at Modal's prices on October 20 (H100 $3.95 an hour, H200 $4.54, plus about $0.45 an hour for the CPU
and memory of each one-GPU machine and $0.70 for the pair's 64 GiB): usually about $5, at most about $8.50 at the
functions' limits (60, 20 and 15 minutes). Each function first checks that the code it has is the code the
launcher fingerprinted and stops before any GPU work if not; the launcher records its git commit and whether the
tree had changes (or that it isn't a git checkout). A run that stops early or misses any criterion exits with an
error.

**Changes to these criteria, in order** (none changed what a criterion requires except where said):

1. October 21, before any run: the fused check's term for its dot's rounding used the largest block sum in place
   of the 2-norm, up to sqrt(k) times smaller, and fp16 layers whose block sums passed 65504 went unjudged. Both
   fixed.
2. October 21, after the second run and before the third: each step of that dot, 32 terms of k on the tensor
   cores, is added into a running sum compensated for its rounding (Neumaier's version of Kahan's summation,
   plain fp32 adds), so the term is `0.6 sqrt(32) u |x_row| |block sums|_2 + 4 u |expected|`, sqrt(k/32) times
   smaller than before (5.7 times at k = 1024, 16 at 8192). In the first two runs the in-kernel scoring caught
   single wrong values 2 to 3.3 times as large as precise mode did, where criterion 3 allows 2. **Prediction:**
   criterion 3 met.
3. October 21, before any run: the size of grouped criterion 4's wrong value was corrected twice. The first
   draft used 10% of one typical output, which a CPU smoke test showed is below what the bf16 output check sees;
   the second used 10% of the block's absolute sum, about 90% on the harm study's scale. Both became 10% of the
   block's L2 norm, for every wrong value injected in this section. Experts with no tokens are counted as stated.
4. October 21, before any run: criteria 6 and 7 (capture) and the vLLM requirement were added; the tensor
   parallel reading of "every matmul checked" was spelled out as above (the first version asked for at least one
   checked matmul) and its compiled variant added; the H200 workaround was required on every algorithm-66
   configuration with 4 or more CTAs (`sdcguard blas` had run it on the heuristic's first algorithm only).
5. October 21, after the first run: criterion 8 was added, before its own first run. The runner's judging of 1 to
   7 had met some with a stage missing or cut short (no capture timing, no CUDA graph, a single blas sweep, no
   fault trials, fewer calls or steps than the protocol runs); each is now met only by its stage run in full. The
   first run's H100 report gets the same verdicts on 1 to 7 from both versions.
6. October 21, after the third run and before the fourth: the recompute check's end-of-step comparison takes every
   block's fingerprints in one comparison and one read-back (4 operations for 8 blocks instead of 73), and its cost
   is timed one step of each variant in turn, 30 rounds, medians, instead of three blocks of 10 steps one after
   another, which put whatever drifted during the measurement on the last block. The three runs before had
   measured GPT-2 at 9.2%, 7.2% and 8.8% and Llama at 16.1%, 6.8% and 11.2% with the same check, while the plain
   step's own time moved by up to 30% between runs. **Prediction:** criterion 4 met on both models.
7. October 21, after the fourth run (reporting only): an issue NVIDIA ties to split-K is listed for a wrong
   configuration only when that configuration ran with split-K, and the configurations of one algorithm on one
   shape and data type are one entry, with their count, cluster sizes and split-K. In the fourth run every wrong
   configuration on the H200 at 40% had SPLITK_NUM 1, which leaves 6862950 the one listed issue that accounts for
   them.

### What the runs found (October 21, 2026): every criterion met

Five runs that day, each on code the launcher fingerprinted ([results/v040](../results/v040)). Times are the
launching Mac's clock (UTC-4):

| Run | Code fingerprint | Criteria not met | Then |
|---|---|---|---|
| 08:41 | f500aa2c0eab | H100 1 (fingerprint kernel), fused 3, recompute 4 (9.2%, 16.1%); tensor parallel crashed | the fixes below; change 5 |
| 17:13 | 6907e8f0308a | fused 3 | change 2 |
| 17:43 | 9eb5a762ca46 | recompute 4 (8.8%, 11.2%) | change 6 |
| 19:13 | 5726b69bc7ae | none | change 7 |
| 19:41 | 8d34bf97fcf7 | none | released as 0.4.0 |

After the first run, two fixes. Tensor parallel had ended with an illegal memory access on both GPUs (Xid 31 at
address 0x0): DTensor hands the check AsyncCollectiveTensors while a collective is still running, wrappers with no
memory of their own, and the Triton kernels read them at address 0. The check now waits for such a tensor first,
and every kernel entry point refuses a tensor without memory of its own (`tests/test_wrapper_tensors.py`). And
criterion 1's comparison had looked up the fingerprint kernel's status under `cuda` while the kernel recorded it
under `cuda:0`; both names now mean one device.

In the last run, on the code released as 0.4.0:

- **Fused check:** the smallest single wrong value caught in every trial was 0.5% of a typical output on two
  shapes and 3% on the third, the same as precise=True at each; the default check caught none at the sizes tried.
  Worst clean score 1.53 (precise=True 1.62) in 990 checks. Per step on the calibration workload, every step
  checked: the default check added 46%, precise=True 544%, the fused check 118%. Gradients equal to
  precise=True's.
- **Recompute check:** on 8-layer GPT-2 and Llama models (768 wide): 0 mismatches in 80 clean compares each, 8 of
  8 injected faults caught with the block named, gradients unchanged against the deterministic control, and 5.2%
  and 4.0% per step with every step checked.
- **`v040_check`:** the fingerprint kernel equal to PyTorch on 18 tensors (64 MiB in 0.13 ms against 6.1 ms);
  compiled, 99 checks on every checked step as in eager, none on unchecked steps, no alarm, 5 of 5 faults caught,
  unchecked steps 13.06 ms against 13.04 ms without sdcguard; grouped GEMMs, 38,196 of 38,196 expected checks, no
  alarm, 100 of 100 faults caught (a checked call took 32.6 ms against 1.27 ms plain); `blas`, all 8 algorithms
  right on every default shape in bf16 and fp16; capture, the two alerts as specified, and the check 80.4 us with
  the capture buffer against 79.7 us without; criterion 8 met in all four parts.
- **Two H100s:** every criterion met in `protect()`, the default mode and compiled.
- **H200:** at 100% every algorithm right; at 40%, 20 configurations wrong on the two shapes, every one algorithm
  66 with clusters of 4 or 8 CTAs and no split-K, and the workaround right; `sdcguard issues` said 6862950 applies
  at 40% and is not active at 100%.

The same day, vLLM 0.31 on one H100 with the same code (fingerprint 8d34bf97fcf7, [results/vllm_oct21](../results/vllm_oct21),
13.1 minutes): on Qwen2.5-1.5B and Qwen2.5-7B, every one of 64 answers word for word the same, no alert on clean
traffic, every one of the 112 layers ran its check (75,264 checks, none unjudged), and the layer whose weight had
a bit flipped was named by its alert and by `verify_weights()`; the alert kept its row (captured inside vLLM's
compiled graph) with the recheck verdict `weights_changed`. Checking every layer cost 5.3% (1.5B) and 31.1% (7B)
of output tokens per second against plain vLLM with PyTorch's default reductions; checking 1 in 8 layers was
within the run's noise of plain vLLM (it measured faster on both models). The cost of the five GPU runs was about
$0.65 to $0.74 each by function time, and the vLLM run about $1.06.

## 0.5.0 on GPUs (written October 22, 2026, before any GPU run)

0.5.0 adds four things that need GPUs to validate: checks of INT4 and INT8 weight-only quantized layers in vLLM, the
upgrade gate (`sdcguard gate`) and its output check, and a faster serving kernel. What each run must show is written here first. Every
result is published, including any criterion not met.

### INT4 and INT8 layers in vLLM

How it works (`src/sdcguard/_quant.py`): vLLM 0.31 runs GPTQ, AWQ and compressed-tensors W4A16/W8A16 layers with two
kernels, Machete (H100 and H200, group sizes 64 and 128 and one scale per output) and Marlin (every other NVIDIA GPU,
and Hopper with group size 32). Both dequantize each weight into bf16 or fp16 before an fp32 tensor-core
accumulation, by rules read from vLLM 0.31's CUDA source (the module's docstring has them), so the matrix the GEMM
multiplies by is exactly representable and the existing check applies unchanged. At load, before the kernel repacks
the weights, sdcguard reads them in vLLM's standard layout and builds that matrix; after the repack it runs the
kernel itself on unit vectors (every input when there are at most 1,024, else one input of every group, the
first and the last, and random ones until there are 1,024) and requires every value to equal the matrix it built, bit for bit. A layer that differs
anywhere isn't checked and is listed with the reason. Every call then runs the kernel without the bias, checks that
output and adds the bias in the activation dtype, which is the rounding both kernels use for their own bias add.
Not checked: Marlin with one scale per output (it applies the scale after accumulating, a second rounding of every
output), `VLLM_MARLIN_USE_ATOMIC_ADD=1`, activations quantized inside the kernel, and every other quantized kernel
(FP8 included: on Hopper vLLM's FP8 GEMMs accumulate in reduced precision, which this noise model doesn't cover).

The run (`modal run scripts/modal_vllm.py --phase quant`, one H100, vLLM 0.31.0): Qwen2.5-1.5B-Instruct-AWQ (zero
points) and Qwen2.5-1.5B-Instruct-GPTQ-Int4 (no zero points), each on Machete (vLLM's choice on an H100) and on Marlin
(`VLLM_DISABLED_KERNELS=MacheteLinearKernel`), then Qwen2.5-7B-Instruct-AWQ on Machete, also with 1/8 of the layers
checked. Each model runs as in the vLLM runs above (`experiments/vllm_check.py`): without sdcguard, with it, and
without sdcguard with PyTorch's default reductions; 256 prompts, 128 tokens, greedy, torch.compile and CUDA graphs.

Criteria, for each of the five:
1. Every layer sdcguard registered is quantized and run by the expected kernel, and the start-up count leaves out
   only the LM head: no layer was dropped by the unit-vector probe or for any other reason.
2. The first 64 completions are identical with and without sdcguard.
3. Clean traffic after warm-up: every registered layer executed checks, no alert, no checksum that couldn't be
   judged.
4. A high exponent bit flipped in one group scale of `model.layers.10.mlp.down_proj` (the quantized layer has no
   bf16 weight to flip): the alert names that layer and keeps the row that alarmed, its recheck says the weights
   changed, and `verify_weights()` lists the layer.
5. Throughput with and without sdcguard is reported. There is no threshold.

If the GPTQ model doesn't load in vLLM 0.31 (it refuses act-order with groups), that model's runs are reported as
not run, and the criteria are judged on the others.

**After the first run (October 22, written before the second).** The first run (code fingerprint 38c4e9be3126,
`runs/gpu_r3/quant/`, 24.6 minutes) met every criterion on Marlin, for both models: 112 of 112 layers checked, 64
of 64 completions identical, no alert in clean traffic (highest score 2.2 and 2.2), the flipped scale caught in
`model.layers.10.mlp.down_proj` with the recheck saying the weights changed, and throughput 22.0% (AWQ) and 20.1%
(GPTQ) lower with sdcguard, on this 1.5B model. On Machete, vLLM's choice on an H100, it didn't meet them:
- GPTQ (no zero points): 1, 3 and 4 met (112 layers, highest clean score 2.4, the flip caught; 17.0% slower), 2
  not: 63 of 64 completions identical; prompt 11 changed one word near its end.
- AWQ, 1.5B and 7B (zero points): the unit-vector probe rejected all 112 layers (43% to 51% of each layer's
  values differed from the matrix built), so nothing was checked and the start-up line said so; the experiment then
  stopped on its own error (it tried to flip a bf16 weight the quantized layer doesn't have).

What was found and changed, after seeing that run:
1. Machete's zero points: its source multiplies by the scale and adds the zero term in two steps, read as two
   roundings; but on Hopper both steps are bf16 instructions without a rounding modifier, which the compiler may
   fuse into one fused multiply-add (`_quant`'s docstring has the details); 43% to 51% differing is in the range
   rounding once instead of twice gives for codes near their zero point (45% to 57% in a simulation), though the
   share alone doesn't prove it. `_quant` now reads Machete's zero points as rounded once and
   can read them the other way; the plugin keeps whichever reading the probe confirms bit for bit, and still
   checks nothing on a layer where neither holds.
2. The probe now feeds its unit vectors in batches of 16, 32, 64, 128, 256 and 512, one in each range where
   Machete picks a different tile shape (a different compiled kernel), so a reading is confirmed at every kernel
   serving can use, not only at 256.
3. Machete's bias: the plugin added it in place after the check op, as Machete's own code does. Under
   torch.compile with vLLM's Inductor setting (auto_functionalized v1) that compiled differently from vLLM's own
   graph: in a CPU reproduction of qkv_proj, its bias and the rotary embedding, about 30% of q's and k's values
   differed. Added out of place, it compiles to vLLM's values bit for bit (`tests/test_quant.py` now checks
   this, with the check opaque as on a GPU and traced). This is the likely cause of the changed completion; the
   second run tests it.
4. The experiment no longer stops when no quantized layer is checked; it records that nothing was flipped.

The criteria for the second run stay as written above.

**What the second run found (October 22): Marlin met every criterion; Machete met all but 2, unchanged.** Code
fingerprint 78c664d5b914 (commit d9b8bad), `runs/gpu_r3/quant/`, 26.1 minutes, about $2.10.
- Marlin, AWQ and GPTQ: every criterion met again (112 of 112 layers, 64 of 64 completions and all 16 of the
  light-load ones identical, highest clean score 2.2 and 2.3, the flip caught with the recheck saying the weights
  changed). Each request took 29.2% (AWQ) and 19.2% (GPTQ) longer with every layer checked at 256 requests at a time,
  61.9% and 61.2% at 16.
- Machete, all three models: changes 1 and 2 worked. The probe kept every one of the 112 layers on Machete, AWQ's
  with its zero points read as rounded once, and criteria 1, 3 and 4 were met (no alert in clean traffic, highest
  scores 2.3 to 2.8; the flipped scale named by both checks, its row kept). Criterion 2 was not: 62 of 64
  completions identical (AWQ 1.5B), 63 (GPTQ 1.5B), 60 (AWQ 7B), and the light-load sets differed too. GPTQ's
  completions with and without sdcguard were each exactly those of the first run, so the change is deterministic
  and change 3 above did not change it: it was not the cause, or not the only one.
- AWQ 7B with 1 in 8 layers: the run also failed on this experiment's own rule, which counted the 97 layers the
  share leaves out as "unchecked" (`_quant_failures` now allows them when a share is set); its completions differed
  too (58 of 64).

**A diagnostic before any fix (written October 22, before it runs).** `experiments/machete_diag.py`
(`modal run scripts/modal_vllm.py --phase machete_diag`, one H100, about 20 minutes) runs Qwen2.5-1.5B-Instruct-
GPTQ-Int4 on Machete once without sdcguard and then eight times with it, each time with sdcguard's Machete
apply_weights replaced by a variant: `wrap_only` (sdcguard reads, probes and registers every layer, and
apply_weights is vLLM's own), `split_inplace`, `split_oop` and `split_rounded` (Machete without the bias, then the
bias added in place, out of place, or by one opaque call that rounds the sum to bf16 once, as the Marlin patch adds
it; no check), `check_no_bias` and `check_bias` (the shipped patch with the check only on layers without, or only
with, a bias), `full` (as shipped) and `full_rounded` (as shipped, but the bias added by that opaque call). Each
compares its 64 completions and 16 light-load completions with those without sdcguard. The first of the first seven
variants, in that order, whose completions differ locates the cause: loading if `wrap_only`, calling Machete
without its bias if a `split` variant, and the check op on that kind of layer otherwise. `full_rounded` tests one
candidate fix: that vLLM's own graph rounds Machete's bias sum to bf16 before what follows, which sdcguard's
patch (in place or not) doesn't. It decides no release criterion; a fix follows from it, and the quantized runs
are then repeated with the criteria above.

**What the diagnostic found (October 23).** Code fingerprint 4f75f47ff57b (commit a56db68),
`results/machete_diag_oct23/`, 14.5 minutes, about $1.17. Completions the same as without sdcguard:

| Variant | of 64 | of 16 (light load) |
|---|---|---|
| `wrap_only` | 64 | 16 |
| `split_inplace` | 64 | 16 |
| `split_oop` | 64 | 16 |
| `split_rounded` | 61 | 14 |
| `check_no_bias` | 62 | 15 |
| `check_bias` | 61 | 15 |
| `full` | 61 | 15 |
| `full_rounded` | 61 | 14 |

Applied as written, the rule above names `split_rounded`, the first of the first seven to differ, as "calling
Machete without its bias". That reading is wrong, and so was the rule: `split_rounded` also rounds the bias sum, while
`split_inplace` and `split_oop`, which call Machete without its bias and add it after (in place, or not), gave every
completion. `split_rounded` belonged with `full_rounded`, as a test of rounding, not among the seven. What the runs
show: loading, probing and registering change nothing, and neither does calling Machete without its bias and adding
it afterwards; rounding the bias sum to bf16 does change completions, so vLLM's graph keeps that sum unrounded and
`full_rounded` is not a fix; the check as an op of its own changes completions on layers without a bias
(`check_no_bias`) and with one (`check_bias`, whose 80 completions are all those of `full`); and once the bias sum is
rounded, the 112 checks change nothing more (`full_rounded` and `split_rounded` gave the same 80 completions).

**The fix (written October 23, before the run below).** Under torch.compile, Machete's GEMM and the check are now
one opaque op, `sdcguard::machete_checked`: vLLM's graph has it where it had `machete_mm`, with the same inputs (and
the check's read-only tensors) and the same output, and nothing else. The op finds the buffers the check writes by
the layer's slot (as vLLM's attention ops find the KV cache through vLLM's forward context), so it is declared as
mutating none of its arguments, which is true of them, and it carries `machete_mm`'s layout tags, which decide how
Inductor lays out a custom op's inputs (each run records both ops' tags). The plugin runs a copy of vLLM 0.31's
`MacheteLinearKernel.apply_weights` with only that call replaced, and checks Machete's layers only where vLLM's
`apply_weights` and `_custom_ops.machete_mm` are the code it copies (by fingerprint: the same in 0.31.0, 0.32.0 and on
main); otherwise they run unchecked and are listed with why. On the CPU (`tests/test_quant.py`: qkv_proj with a
bias, its rotary embedding, then o_proj, a residual add and an RMSNorm, compiled with vLLM's Inductor setting):
- the check as an op of its own generated another kernel for the bias and the rotary embedding, which stored the bias
  sum rounded to bf16 where vLLM's kernel keeps it in fp32, and other q and k values (PyTorch 2.14.1). That is the
  effect of rounding the bias, and on the H100 `full` changed the same three prompts as `split_rounded` did (0, 11
  and 33; prompts 0 and 11 to other text than `split_rounded`'s);
- the one op generated the same kernels as vLLM's own code, called in the same order, and the same values, on
  PyTorch 2.14.1 and 2.8.0;
- the one op with the buffers passed to it and declared as mutated gave the same values but other kernel code on
  PyTorch 2.8.0 (the RMSNorm after o_proj wrote to a new buffer instead of reusing Machete's output). On a GPU, kernel
  code that differs is autotuned anew and can get another reduction order, so that form wasn't used.

**The run that tests the fix (written October 23, before it runs).** `modal run scripts/modal_vllm.py --phase
machete_fix`, one H100, vLLM 0.31.0 with the PyTorch it pins (2.13): `experiments/machete_diag.py` (round 2), and
then, only if the shipped fix meets the rule below, the quantized runs above (`--phase quant`) with criteria 1 to 5
as written. About 45 minutes and $3.50 in all (15 minutes and $1.20 if the quantized runs don't start). The
diagnostic runs Qwen2.5-1.5B-Instruct-GPTQ-Int4 on Machete in six fresh processes: `baseline` (no sdcguard),
`separate` (the patch as in the runs above), `fused` (the patch as it is now) and `fused_declared` (the one op with
the buffers passed and declared as mutated), these four sharing one Inductor cache in that order, as vllm_check's
runs do; then `baseline_fresh` and `fused_fresh`, each compiling into an Inductor cache of its own, as a server with
sdcguard does (the plugin keeps its own compile cache). Each records its 64 and 16 completions, Inductor's code for
every graph it compiled (its debug trace: the graph after its passes, the IR after fusion, the generated code), the
launch configs Triton's autotuning saved, and the checks every registered layer ran.

Rule, for `fused`: every completion, 64 and 16, the same as `baseline`'s; where code was captured, the same Triton
kernels (their code with kernel names, comments and the backend hash taken out) and in every compiled graph the same
order of kernel launches and opaque calls (`machete_checked` counted as `machete_mm`); and every registered layer
ran checks. If `fused` meets it, the quantized runs start. If it doesn't and `fused_declared` does, that form becomes
the fix and the quantized runs are a run of their own. If neither does, Machete's layers are not checked in 0.5.0
(they are listed as quantized layers left over, with why), Marlin's are, and the README says so.

Reported, deciding nothing: `separate` (expected to change completions again; its code shows what the separate op
changes on a GPU), `baseline_fresh` (does vLLM, compiling anew, give its own completions? If it doesn't, no plugin
can promise the same completions across fresh compiles, and the claim becomes "the same compiled kernels"),
`fused_fresh` against both baselines, and the kernels each run autotuned anew.

**What the run found (October 23).** Code fingerprint 7685141b635d (commit 2df6b13). The diagnostic
(`results/machete_diag2_oct23/`, 21.7 minutes, about $1.75), against `baseline`, PyTorch 2.13.0 and vLLM 0.31.0:

| Run | of 64 | of 16 | Inductor's code | Autotuned anew |
|---|---|---|---|---|
| `separate` | 61 | 15 | other kernels (2 not in `baseline`, 3 missing) and call orders | 5 kernels |
| `fused` | 64 | 16 | the same in all 29 graphs | 1 kernel |
| `fused_declared` | 61 | 15 | other kernels (1 not in `baseline`, 2 missing) | 1 kernel |
| `baseline_fresh` | 64 | 16 | the same | 8 kernels, all with `baseline`'s configs |
| `fused_fresh` | 64 | 16 | the same in all 29 graphs | 8 kernels: 7 with `baseline`'s configs, 1 it doesn't have |

Every sdcguard run checked all 112 layers. `fused` met the rule, so the quantized runs started. `separate` changed
prompts 0, 11 and 33, as `full` did in the first diagnostic. `fused_declared` behaved on the H100 as on the CPU with
PyTorch 2.8: other kernel code and other completions.

The quantized runs (`results/quant_oct23/`, 24.3 minutes, about $1.96) met criteria 1, 3 and 4 on all five: 112 of
112 layers checked by the expected kernel, only the LM head left out, no alert in clean traffic (highest scores 2.26
to 2.72), and the flipped scale caught in `model.layers.10.mlp.down_proj`, with the recheck saying the weights
changed. Criterion 2:

| Model, kernel | of 64 | Light load (16) | Slower at 256 requests at a time | At 16 |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct-AWQ, Machete | 64 | all the same | 14.9% | 46.9% |
| Qwen2.5-1.5B-Instruct-GPTQ-Int4, Machete | 64 | all the same | 10.2% | 46.4% |
| Qwen2.5-1.5B-Instruct-AWQ, Marlin | 64 | all the same | 13.1% | 56.6% |
| Qwen2.5-1.5B-Instruct-GPTQ-Int4, Marlin | 64 | all the same | 20.2% | 61.5% |
| Qwen2.5-7B-Instruct-AWQ, Machete | **59** | some differed | 24.6% | 38.4% |
| the same, 1 in 8 layers checked | **57** | some differed | 4.5% | 5.3% |

So both 1.5B models now give vLLM's completions on Machete (62 and 63 of 64 before), and the 7B does not.

**What was found after it (written before the next run).** The 7B's changed prompts were mostly the same ones as
on October 22 with the plugin before the fix (prompts 0, 12, 23 and 44 then; 0, 12, 23, 36 and 44 now; prompt 12 at
character 27 and prompt 44 at character 256 both times), also with only 15 layers checked, while `baseline` was the
same on both days: the cause was not how the check is attached. Round 2's traces show something both plugins had in
common: Inductor names the kernels it generates after the ops fused into them, without the op's namespace. Of the 9
distinct kernels in the 1.5B model's graphs, 4 are named after Machete's GEMM in `baseline`
(`triton_red_fused_fused_add_rms_norm_machete_mm_0` and `_2`, the residual adds and RMSNorms; the first piece's
embedding and RMSNorm; and `triton_poi_fused_machete_mm_mul_silu_slice_1`, SiluAndMul), and in `fused` the same 4
came out under `machete_checked` names, their code otherwise the same byte for byte. PyTorch 2.13 writes a kernel's
name into the code it hashes (`config.triton.unique_kernel_names`, on by default), and Triton's autotuning results
are cached by that hash, so a renamed kernel can't reuse `baseline`'s launch config. In `fused`, one kernel was
autotuned anew: a pointwise config (XBLOCK only), so SiluAndMul's, the one renamed pointwise kernel; the 3 renamed
reduction kernels wrote no autotuning result. In `fused_fresh`, 7 of its 8 autotuned kernels had `baseline`'s
cache entries and the 8th didn't. Not known yet: whether the 7B's kernels (hidden size 3,584, a wider MLP) autotune
where the 1.5B's didn't, whether autotuning picked other configs there, and whether that changed the completions.
The next run records which kernels each run autotunes, by name, and tests it.

A second difference both plugins had: the op took each layer's slot (and the check's other arguments) as constants
in vLLM's graph, so no two layers' pieces of the graph (the parts between attention ops) were the same, and each was
compiled on its own: `baseline` compiled 3 graphs, `fused` 29 (and the October 23 quantized run's log shows 7 and 43
seconds of compiling for the 7B). Inductor makes some decisions while compiling by timing candidates (combining
kernels, here: `benchmark_combo_kernel`, which vLLM turns on), and on the 1.5B model all 29 compiles came out the
same; on the 7B, with larger kernels, that isn't known.

The changes, both: the op is now named `machete_mm`, in sdcguard's namespace (`sdcguard::machete_mm`), so Inductor
generates vLLM's kernels byte for byte, names included; and it takes one tensor of the layer's checksums instead of
the slot and the check's arguments, and finds the layer and the rest from it when it runs, so every layer's piece is
the same graph and is compiled once, as without sdcguard. `tests/test_quant.py` checks both on the CPU: the kernels
byte for byte, and the second of two layers a hit in Inductor's graph cache (with a control: Marlin's check, which
still takes the slot, misses).

**The third diagnostic (written October 23, before it runs).** `modal run scripts/modal_vllm.py --phase
machete_fix`, as before: `experiments/machete_diag.py` on Qwen2.5-7B-Instruct-AWQ with 256 requests at a time (as
vllm_check sends; all 256 completions compared, and the first 64 that vllm_check compares counted too), runs
`baseline`, `fused` (the plugin as it is now), `fused_renamed` (the same op named `machete_checked`, as in round 2)
and `fused_slots` (named `machete_mm`, but taking the slot and the check's arguments as constants, as in round 2) in
one Inductor cache in that order, then `baseline_fresh` and `fused_fresh` in caches of their own; then,
only if `fused` meets the rule, the quantized runs with criteria 1 to 5 unchanged.

Rule, for `fused`, as before with two additions: every completion the same as `baseline`'s (256 and 16); where
code was captured, the same kernels and call orders, and now the same kernels byte for byte, names included; for a
run in the shared cache, no kernel autotuned anew; every registered layer ran checks. If `fused` doesn't meet it,
the quantized runs don't start, and Machete's layers go unchecked in 0.5.0 unless a further run finds the cause
(listed as quantized layers left over, with why; Marlin's stay checked); the README says which.

Reported, deciding nothing: `fused_renamed` and `fused_slots`, each with one of the two differences back (whichever
changes completions, and how its code and autotuning differ, says which difference mattered; if neither does, the
cause was something else, and that's written here), `baseline_fresh` (does vLLM, compiling the 7B anew, give its
own completions, and pick the same launch configs?), and `fused_fresh` against both. About 20 minutes for the
diagnostic's six runs of the 7B, about $1.60, so about 45 to 50 minutes and $3.70 in all with the quantized runs.

**What the third diagnostic found (October 23): the rule met, and the quantized runs passed.** Code fingerprint
ce3966db3cc9 (commit adca707), [results/machete_diag3_oct23](../results/machete_diag3_oct23) (7.9 minutes, about
$0.64 by function time) and [results/quant3_oct23](../results/quant3_oct23) (20.7 minutes, about $1.67).

- `fused` met the rule: all 256 completions and all 16 of the light load the same as `baseline`'s (the first 64
  too); the compiled code byte for byte the same in all 3 graphs, names included; no kernel autotuned anew; all 112
  layers ran checks.
- Reported: `fused_renamed` (the op named `machete_checked`) gave every completion the same too. Its 4 kernels were
  named after the op, 3 of them were autotuned anew, and each picked the launch config its `machete_mm` twin has
  in `baseline` (XBLOCK 1024 with 4 warps; R0_BLOCK 4096 with 16 warps). `fused_slots` (the slot as a constant)
  compiled 29 graphs instead of 3, with the same kernels byte for byte and nothing autotuned anew, and every
  completion the same. So in this run neither difference changed a completion.
- `baseline_fresh`, vLLM alone compiled in a cache of its own: 228 of 256 completions the same as `baseline`'s (58
  of the first 64, 15 of 16), with its code byte for byte the same. Of its 12 autotuned kernels, 11 picked
  `baseline`'s configs and one didn't: the fused add + RMSNorm kernel (`triton_red_fused_fused_add_rms_norm_
  machete_mm_0`) took R0_BLOCK 2048 instead of 4096, another order of additions. `fused_fresh` changed the same 28
  completions, and is the same as `baseline_fresh`.
- Reading: vLLM's compiled completions of this model depend on Triton's autotuning, which times its candidates
  and can settle on another reduction block, so other rounding. A kernel with another name is autotuned anew and
  can land either way. That is consistent with the 5 of 64 changed in round 2 (its kernels were renamed), but not
  shown for round 2 itself, whose launch configs weren't recorded by kernel. With kernels identical to vLLM's,
  the check reuses vLLM's own tuning; on a fresh cache it gives what vLLM itself gives there.
- Quantized runs, criteria 1 to 5 unchanged: met on all five (GPTQ and AWQ, 1.5B on Marlin and Machete, 7B on
  Machete). Each: 64 of 64 completions and 16 of 16 light the same as the baseline and as untouched vLLM (PyTorch's
  default reductions, compared from the saved completions), all 112 layers ran checks (118,272 checks, none
  unjudged), no clean alert (worst score 2.89), the layer with the flipped weight bit named by its alert and by
  `verify_weights()`. Every layer checked cost 44.8% of output tokens per second on the 7B (40.7% at 16 requests)
  and 12% to 30% on the 1.5B models (48% to 61% at 16 requests); 1 layer in 8 on the 7B cost 1.0% (5.9% at 16
  requests), with 64 of 64 the same. Machete's layers are checked in 0.5.0.

### The real server, vLLM 0.32, two GPUs, and INT4 on an L4 (written October 23, before any run)

Every vLLM result above used the offline API with the engine in the same process. People deploy `vllm serve`: an
API server, an engine and GPU workers in processes of their own. `experiments/vllm_serve_check.py` starts `vllm
serve` twice, plain and guarded (SDCGUARD=1 with SDCGUARD_REQUIRE=1, alerts posted as JSON to a local webhook,
Prometheus metrics per GPU worker), and talks to it over HTTP only: 64 completions one at a time (greedy, 128
tokens), then streamed requests 1, 16 and 256 at a time, then, guarded only, one weight bit flipped in every worker
through vLLM's /collective_rpc (a test-only worker extension) and more requests. `modal run scripts/modal_vllm.py
--phase serve` runs it on Qwen2.5-1.5B-Instruct three ways at once: vLLM 0.32.0 (the release of October 18) on one
H100, 0.32.0 on two with tensor parallelism, and 0.31.0 on one.

Criteria, for the guarded server, each run:
1. It starts and serves under SDCGUARD_REQUIRE=1 (every registered layer checked; the API server process, which
   never loads the model, mustn't refuse).
2. The 64 one-at-a-time completions are identical to the plain server's.
3. Every GPU worker's metrics: the same number of layers checked on each, 112 with one GPU; no layer whose check
   never ran; results read back; no alert before the flip.
4. No alert reaches the webhook during the clean load.
5. After the flip, an alert naming `model.layers.10.mlp.down_proj` reaches the webhook within 90 seconds, and the
   metrics count it.
6. Both servers exit within 120 seconds of SIGINT.
Reported, deciding nothing: time to first token and request latency (p50, p90, p99) and output tokens per second
at 1, 16 and 256 requests at a time, guarded against plain. About 15 minutes each, about $5 in all.

`--phase quant_l4`: the INT4 runs of `experiments/vllm_check.py` (criteria 1 to 5 of the INT4 section, unchanged)
for the AWQ and GPTQ 1.5B models on one L4, where vLLM has no Machete (it needs an H100) and runs Marlin by itself,
with other shapes of its kernels. About 20 minutes, about $0.50.

**What the runs found (October 23).** Code fingerprint 30f939a08310 (commit 208bba6),
[results/serve_oct23](../results/serve_oct23), [results/quant_l4_oct23](../results/quant_l4_oct23).

- vLLM 0.32.0 on one H100 (6.6 minutes) and 0.31.0 on one (7.6 minutes): every criterion met. The guarded server
  started and served under SDCGUARD_REQUIRE=1; 64 of 64 one-at-a-time completions the same as the plain server's;
  112 layers checked, none never checked, results read back; no alert in clean traffic; the alert naming
  `model.layers.10.mlp.down_proj` reached the webhook while the requests after the flip were still being sent, and
  the metrics counted it; both servers exited at once on SIGINT. Reported, against the plain server, with every
  layer of this 1.5B model checked:

  | vLLM | requests at a time | output tokens/s | request latency p50 / p99 | time to first token p50 |
  |---|---|---|---|---|
  | 0.32.0 | 1 | -34.0% | +51.5% / +47.9% | +8.5% |
  | 0.32.0 | 16 | -30.8% | +44.9% / +43.0% | -1.8% |
  | 0.32.0 | 256 | -10.1% | +12.7% / +11.4% | -1.1% |
  | 0.31.0 | 1 | -29.0% | +41.6% / +31.5% | -5.9% |
  | 0.31.0 | 16 | -22.8% | +41.2% / +11.0% | -12.8% |
  | 0.31.0 | 256 | -2.5% | +8.1% / +4.3% | -33.8% |

  Time to first token is mostly unchanged; the cost is in each decoded token, largest when few requests share the
  GPU (the small model's layers are short, so the check is a large share of each step), as the offline runs found.
- vLLM 0.32.0 on two H100s (4.8 minutes): not judged. Both servers, plain vLLM as well as guarded, died before
  serving: vLLM 0.32 reduces across GPUs with FlashInfer's fused all-reduce, which JIT-compiles with nvcc, and the
  image (vLLM installed with pip) has none. The image now sets VLLM_ALLREDUCE_USE_FLASHINFER=0 (vLLM's own
  all-reduce), and `--phase serve_tp2` reruns this one under the same criteria.
- vLLM 0.32.0 on two H100s again, with VLLM_ALLREDUCE_USE_FLASHINFER=0 (code fingerprint f1664ae2b079, commit
  1143f4d, 6.4 minutes, about $1.03 of function time; the first attempt's results are kept as
  `serve032_tp2_before_20261023_201503`): every criterion met. The guarded server started and served under
  SDCGUARD_REQUIRE=1; 64 of 64 one-at-a-time completions the same as the plain server's; 112 layers checked on
  each of the two workers, none never checked, results read back; no alert in clean traffic; the alert naming
  `model.layers.10.mlp.down_proj` came from both workers (the bit was flipped in each worker's shard) while the
  requests after the flip were still being sent, and the metrics counted it; the servers exited in 4.1 s (plain)
  and 0.6 s (guarded) after SIGINT. Reported, against the plain server, every layer checked:

  | requests at a time | output tokens/s | request latency p50 / p99 | time to first token p50 |
  |---|---|---|---|
  | 1 | -35.7% | +55.4% / +51.4% | +12.3% |
  | 16 | -34.5% | +52.8% / +51.4% | -7.3% |
  | 256 | -16.7% | +28.3% / +21.0% | +0.1% |

  At 256 requests at a time the cost is larger than on one GPU (-10.1%): each worker's matmuls are half as large
  with two GPUs, while the check's fixed cost per layer (the serving kernel section) stays the same, so it is a
  larger share of each step. That explanation fits the numbers but wasn't measured separately.
- INT4 on an L4 (17.2 minutes, about $0.41): criteria 1 to 5 met on both 1.5B models, on Marlin (vLLM's own
  choice there): 112 layers ran checks, 64 of 64 completions the same as the baseline and as untouched vLLM, no
  clean alert (worst 2.37), the flipped scale named. Every layer cost 12% to 15% of throughput at 256 requests at a
  time, 32% to 33% at 16.
- The gate's output check (8.4 minutes, about $0.68, [results/outputs_oct23](../results/outputs_oct23)): all four
  expectations met. Same stack, batches of 4: PASS (mean KL 5.1e-4, top-1 agreement 99.6%). Triton attention: PASS
  (5.1e-4, 99.4%). The AWQ model: FAIL (mean KL 6.1e-2, top-1 agreement 92.9%), past the published 4-bit range for
  an 8B model; this 1.5B model loses more. `rms_norm_eps` set to 1.0: FAIL (mean KL 11.2, top-1 agreement 0.05%),
  while sdcguard's matmul checks, on in the same run, checked all 112 layers 18,032 times and raised no alert: the
  math right, the model wrong, which the output check exists to catch.

### The fused precise check, a long clean run, and the default check against precise mode (written October 23, before any run)

The fused precise check (`precise="fused"`) met every criterion on October 21, but on 990 clean checks on the H100,
and its worst clean score on an L4 on October 20 was 4.11. The default check's false-alarm record is 24.6 million
clean checks. `experiments/fused_soak.py` trains the calibration-shape model (8 layers, 1024 wide, Zipf tokens) in
bf16 with every linear layer's forward, dX and dW checked by the fused check on every step, for 40 minutes, on an
H100 and an L4 at once (`modal run scripts/modal_fused_soak.py`, about $3.80). Criteria, on each GPU: the run
finishes; no alarm; a score read back for at least 90% of the fused checks; no check unjudged; every check run in
the fused kernel (no fallback). Reported: checks, the worst clean score and the histogram of scores. If both GPUs
meet them, the README and WHY.md call the fused check supported for training (with this count), no longer
experimental; if not, it stays experimental and the result is written here.

Then the paired comparison written on October 19 ("The default check and precise mode on the same matmuls"),
unchanged, which hasn't run yet: `modal run scripts/modal_mode_compare.py`, an H100, an A100 and an L4, about
$1.50.

**What the runs found (October 23).** Code fingerprint 9c9e05a04892 (commit cffa950),
[results/fused_soak](../results/fused_soak), [results/mode_compare](../results/mode_compare).

- The long clean run: every criterion met on both GPUs. H100: 46,149 training steps in 40 minutes, 4,568,751
  fused checks, no alarm, none unjudged, a score read back for every one, all in the fused kernel (no fallback),
  worst clean score 2.67 (the alarm line is 6). L4: 6,196 steps, 613,404 fused checks, the same zeros, worst 3.32.
  Scores, H100: 3,674,400 below 0.5, 753,823 from 0.5 to 1, 132,385 from 1 to 1.5, 7,698 from 1.5 to 2, 431 from
  2 to 2.5, 14 from 2.5 to 3, none from 3. L4: 508,777, 50,162, 42,370, 10,130, 1,869, 92, and 4 from 3 to 3.5.
  About $2.93 (H100) and $0.83 (L4) of function time. As written above, the fused check is now supported for
  training on this record (5.2 million clean checks on two GPU types, one model shape), no longer experimental.
  Its cost per step was not measured by this run.
- The paired comparison (mode_compare's own code digest 17c41a10c789; its "precise mode" is `precise=True`, the
  PyTorch fp32 path, not the fused kernel): reported as fixed on October 19, with no pass line. 594 corrupted
  matmuls per error size on each GPU, 198 clean ones.

  | | H100 | A100 80GB | L4 |
  |---|---|---|---|
  | error size flagged half the time: default / precise | 29% / 0.71% | 29% / 0.92% | 29% / 0.92% |
  | flagged 9 times in 10: default / precise | 78% / 2.4% | 78% / 2.4% | 78% / 2.4% |
  | clean false alarms, default / precise (worst score) | 0 (2.27) / 0 (0.52) | 0 (2.11) / 0 (0.73) | 0 (2.13) / 0 (0.25) |
  | unjudged checks | 0 | 0 | 0 |
  | step time unchecked; default check / precise mode, every step | 0.0282 s; +58.5% / +525.9% | 0.0558 s; +31.4% / +337.6% | 0.268 s; +8.1% / +258.4% |

  Error sizes are relative to a typical output of the 128-column block, added before rounding to bf16; among
  errors that changed the bf16 output the crossings are the same to two digits. Below about 0.1% most injected
  errors didn't change the bf16 output at all (39% to 43% did at 0.1%). The default check needs an error of about
  30% of a typical output to flag it half the time; precise mode about 1%. On this small model (0.03 s per step
  on the H100) the fixed cost of each check is a large share of the step, so these step times are an upper end,
  not what a large model pays. The default check runs every 10th step unless told otherwise.

### Verifiable protection status and the requests an alert concerns (written October 24, before any run)

Two things the vLLM plugin couldn't show until now. First, whether a server is protected right now, asked from
outside: the plugin's SDCGUARD_REQUIRE can't report its own absence when it never loads (vLLM's loader logs and
skips a plugin that fails to import). `sdcguard ready` reads every GPU worker's sdcguard metrics twice, a few
seconds apart, and says PROTECTED, NOT PROTECTED or INCONCLUSIVE (its docstring has the rules; it says
INCONCLUSIVE when checks didn't go up and there are no vLLM metrics to show the server was idle). Second, which
requests an alert concerns: before each model step the plugin writes the step's number into the capture
buffer's epoch field (`ServeGuard.mark_step`; the check kernels copy that field into a captured row as it is when
they run, so no kernel changed), and after each step it keeps the step's request ids in batch order with the rows
each held. An alert keeps three answers apart: the directly implicated request (the one that held the captured
row in its step; none for a CUDA-graph padding row, unknown without a captured row); the potentially exposed ones
(every request in the steps its read covered, and every request since the layer's last clean read, which a
persistent fault such as a changed weight concerns; a conservative window, not a proof of harm); and what neither
covers (state that outlives the window, such as prefix-cache blocks, and what the checks don't cover).

`experiments/vllm_ready_attrib_check.py` (`modal run scripts/modal_vllm.py --phase ready_attrib`, vLLM 0.32.0,
Qwen2.5-1.5B-Instruct, one H100, three server starts, about 40 minutes and $3 to $4) runs the scenarios in its
docstring. Criteria:
1. `sdcguard ready`, 3 times each: PROTECTED while the guarded server serves (with --vllm) and while it is idle
   (with --vllm); INCONCLUSIVE when it is idle without --vllm; NOT PROTECTED when its result reads have stopped
   while it serves, when vLLM never loaded the plugin (VLLM_PLUGINS naming no installed plugin, with SDCGUARD=1
   and SDCGUARD_REQUIRE=1 set), when SDCGUARD=0, and after the server stopped. 21 of 21 as expected, and no
   PROTECTED in any of the 15 asked where the server wasn't protected or that couldn't be told.
2. Five times, each in its own layer: one request alone, a weight bit flipped during its decoding and flipped
   back: at least one alert for that layer gives exactly that request as directly implicated, with exact
   attribution, none gives another request (an alert on a CUDA-graph padding row gives none and says so), and the
   requests since the layer's last clean read include it. 5 of 5.
3. A bit flipped while 32 requests run 8 at a time, flipped back after them: at least one alert; every request an
   alert gives as directly implicated is one of the 32; the requests since the last clean read include all 32.
4. The 64 one-at-a-time completions of the guarded server, before any flip, are identical to the SDCGUARD=0
   server's.
This is narrower than the rule the October 24 plan wrote (at least ten injected row-level faults at known steps and
rows): a fault in one row of one step can't be placed from outside a server whose model runs in CUDA graphs. A
request alone in the batch-of-1 graph is the case where the true request is known without trusting the mapping
under test; the mapping itself (rows to requests by the scheduler's token counts, in the runner's batch order) is
tested on stand-ins on the CPU (`tests/test_request_attribution.py`). Not covered by this run: microbatching
(dual-batch overlap) and sequence parallelism, where the plugin says attribution is unavailable, and speculative
decoding.

**What the run found (October 24, 2026): every criterion met.** One H100, vLLM 0.32.0, code fingerprint
a7096a98ba9e, 9.7 minutes (`results/ready_attrib_oct24/`). 1: all 21 verdicts as expected (PROTECTED 3 of 3 while
serving and 3 of 3 while idle with --vllm; INCONCLUSIVE 3 of 3 idle without it; NOT PROTECTED 3 of 3 each with the
reads stopped while vLLM served, with the plugin never loaded, with SDCGUARD=0, and after the server stopped), and
PROTECTED in none of the 15 where it wasn't protected or couldn't be told. 2: 5 of 5 single requests named exactly
as directly implicated (in the first trial all 7 of its alerts named it). 3: 7 alerts while 32 requests ran, 7 of 7
naming a request that was one of the 32, none another, and every alert's requests since the last clean read held
all 32. 4: the 64 completions identical to the SDCGUARD=0 server's.

**An idle server, and the second run (written October 24, after the run above, before the second).** The run
above met its criteria, but criterion 1 had `sdcguard ready` say PROTECTED for an idle server on the strength of
counters from before it went idle, and an idle server whose checks had stopped would have shown the same counters:
the run stopped the reads only while traffic flowed. So `sdcguard ready` now says IDLE when vLLM produced no tokens
between its reads (exit code 0, so a readiness probe on an idle server still passes; nothing was served
unchecked, and nothing shows the next request would be checked), and `--probe URL` sends two short requests 2.5
seconds apart when the server is idle and reads again: PROTECTED if every worker's checks went up for them, NOT
PROTECTED if vLLM counted their tokens and a worker's checks didn't. The second run (same script and server)
expects, 3 times each: R1 PROTECTED; R2 INCONCLUSIVE; R3 (idle, --vllm) IDLE; R3p (idle, --vllm --probe)
PROTECTED; R4 (reads stopped, serving) NOT PROTECTED; R4n (reads stopped, idle, --vllm) IDLE, the limit without a
probe; R4i (reads stopped, idle, --vllm --probe) NOT PROTECTED; R5, R6 and R7 NOT PROTECTED. Criteria: 30 of 30 as
expected, and no PROTECTED in any of the 21 asked where the server wasn't protected, couldn't be told, or wasn't
shown checking (R2, R4, R4n, R4i, R5, R6, R7); criteria 2 to 4 as above. Criterion 3 is about membership: the
requests are sent 8 at a time, and the run checks that every request an alert names is one of the 32, not which of
them held the failing row. So the second run adds criterion 5, exact attribution under concurrency, on a fourth
server (the guarded one with --enforce-eager, so a test hook can act between a layer's GEMM and its check): 5
times, each in its own layer, 32 requests sent at once, one of them with a marker token (a rare character's id)
inserted in its prompt; that token's row of the layer is made wrong after the GEMM, so the request that owns it is
known by construction, not from the plugin's mapping. 5 of 5: at least one alert for the layer, and every alert with
a captured row names exactly the marked request (an alert without a captured row names none and doesn't count
against it). Eager mode is not the default compiled mode: the mapping of rows to requests is the same code in both,
and the CUDA-graph padding case is criterion 2's.

Result of the second run (October 24, on 54c3a69): every criterion met. 30 of 30 verdicts as expected (R1 and R3p
PROTECTED, R2 INCONCLUSIVE, R3 and R4n IDLE, R4, R4i, R5, R6 and R7 NOT PROTECTED, 3 times each), PROTECTED in none of
the 21 where it shouldn't be; 5 of 5 single requests named exactly; 8 alerts during the 32 requests, all 8 naming one
of them and none another; 5 of 5 exact attributions under 32 concurrent requests; completions identical to the
SDCGUARD=0 server's.

### The cuBLAS bug inside the real server (written October 24, before any run)

Every catch of cuBLAS issue 6862950 so far came from the offline check, the gate and single synthetic layers; the
vLLM plugin itself has met only injected faults, and has never run as an MPS client on Hopper, the condition that
triggers the bug. `experiments/vllm_live_catch.py` (`modal run scripts/modal_vllm.py --phase live_catch`, one H200
with the MPS daemon, vLLM 0.32.0, Qwen2.5-1.5B-Instruct in bf16, about 55 to 70 minutes and $5 to $6.50) runs the
guarded server three ways with cuBLASLt's log on: as an MPS client at 100% (the reference), at 40%, and at 40% with
a TunableOp file pinning every shape the 40% inventory flagged to algorithm 66 with a 1x1x1 cluster. Alerts now
carry what a replay at the real shape needs: the layer's output and input features and bias, and the step's rows
with CUDA-graph padding.

First, a finding with no pass line: whether this model's GEMMs reach the affected kernel at all at 40% on this
H200 (`sdcguard kernels` on the server's log: any call on algorithm 66 with 4 or more CTAs per cluster, in FP16 or
BF16). If none does, that is the result, written here as found, and criteria 3 to 5 don't apply. If some do:
1. Both servers start and serve under SDCGUARD_REQUIRE=1.
2. No alert at 100%.
3. At least one alert at 40%, and at least one alert's captured row rechecked in float64 as wrong.
4. For at least one alert: the replay at its real shape (`sdcguard gate capture` with the alert's out and in
   features, rows and bias, in a fresh process) is wrong in at least 19 of 20 runs at 40% and right in 20 of 20 at
   100%, and the kernel cuBLASLt's log shows for it at 40% is one the server's inventory shows for that shape
   (same algorithm and cluster). The replay of the kept row and block alone (`gate add`, today's way) is reported
   beside it, with no line.
5. Reported, deciding nothing: how many of the 64 completions at 40% differ from 100%; and the pinned run: alerts,
   completions differing from 100%, and whether the flagged kernel still ran (whether TunableOp's pins reach
   vLLM's compiled GEMMs is itself unknown before the run). Also each replayed case run at 40% with the pin file
   (`gate workaround`, 20 runs): the fix checked on the very call the server got wrong; and each server's speed
   under the same load (64 requests, 16 at a time): what the pin costs.
6. Reported, deciding nothing, the first GPU run of `sdcguard kernels test`: on each server's inventory at that
   server's share, every logged bf16 linear-layer call run again on its own (random inputs, 3 runs, float64
   reference). Expected: at 40%, the calls flagged for 6862950 WRONG and no other call WRONG; at 100%, no call
   WRONG.

The verdict is one of: CAUGHT (1 to 4 met), NOT EXERCISED (the server never ran the affected kernel at 40%: no
catch, and none claimed; a forced reproduction outside the server is other evidence, with another scope), NOT
CAUGHT (it ran, and a criterion failed) or DIDN'T START. Only CAUGHT is reported as a catch inside vLLM. With any
replay, `reproduce/` keeps the cases, the pin file, the conditions and the commands, so someone else can rerun the
wrong call and its fix without this script.
Not claimed whatever the outcome: anything about other models, GPUs or MPS shares, or about hardware faults.

Result (October 24, on 54c3a69, an H200 with cuBLAS 13.1.1 and driver 610.57.04): CAUGHT. The run hit its 85-minute
limit in the pin step (criterion 5, reported only: it pinned the 215 flagged shapes one process at a time) before it
wrote SUMMARY.json, so criteria 1 to 4 were judged afterwards from the files it had saved, by the experiment's own
judge (`judge`, unchanged), and criterion 5 isn't reported. Both servers served under
SDCGUARD_REQUIRE=1 (1). No alert at 100% (2). At 40%, 419 inventoried calls ran algorithm 66 with 4 or more CTAs
per cluster (345 at 100%, where the bug doesn't show), 14,707 alerts, and all 2,115 alerts with a captured row
rechecked in float64 on the CPU as wrong (3). The replays of three alerts at their real shapes (qkv_proj 2048x1536,
o_proj 1536x1536 and gate_up_proj 17920x1536, 1024 rows with CUDA-graph padding) were wrong in 20 of 20 runs at 40%
and right in 20 of 20 at 100%, each on the kernel the server's inventory shows for that shape at 40% (4); the kept
slice replayed alone was right in 20 of 20 for all three: the real shape is what reproduces it. Reported (criterion
6): `sdcguard kernels test` at 40% gave 214 WRONG and 9 PASSED among the calls it ran, at 100% 241 PASSED and none
WRONG. Not reported, because they weren't saved: the completions at 40% against 100%, the servers' speed, and the
pinned server. Changed for any rerun: the verdict is written before the pin step, and the pin step pins the
most-called shapes first within 15 minutes and lists the rest.

### Decisions on four changes to a vLLM server (written October 24, before any run)

`sdcguard decide` puts four questions about one change to a serving stack side by side and keeps them apart:
whether the checked arithmetic is right against float64 (the vLLM plugin's alerts and what it covered), whether the
model's outputs stay inside a quality budget (`gate model-check` against the trusted stack, now also against a
running server with `--url`), whether the change is as fast as it has to be (`sdcguard perf`, the same load on both
sides, a change inside the spread between rounds counted as noise), and where the answer holds. It has only run on
files made by hand. `experiments/vllm_change_decisions.py` (`modal run scripts/modal_vllm.py --phase decisions`, one
H100, vLLM 0.32.0, about 55 to 70 minutes and $4.40 to $5.70) runs it on a real `vllm serve` of Qwen2.5-7B-Instruct with
sdcguard on as deployed (one layer in eight per step; amended below: every layer), six server launches: the trusted stack (`base`: the
reference and the baseline's speed), the same stack without sdcguard (`plain`: speed only, for what sdcguard at
one layer in eight costs, reported), the same stack started again (`same`), Triton attention (`triton`), the AWQ
INT4 model on Machete (`awq`), and the trusted stack with one weight of model.layers.10.mlp.down_proj multiplied by
2^8 after start (`fault`: what flipping bit 10 of a bf16 weight does when that bit is 0). Each change is decided
against `base` under two policies: `hold` (no quality loss: model-check PASS; throughput no lower, within the noise)
and `speed` (at least 10% more throughput; model-check WARN accepted). The decision's load is 128 prompts at 16 at a
time, 128 tokens each, 3 rounds; one request at a time is reported only.

Criteria:
1. All six servers start and serve (the five with sdcguard under SDCGUARD_REQUIRE=1).
2. `same` under `hold` is ACCEPT; `fault` is REJECT under both policies.
3. The rejection of `fault` gives the arithmetic reason and names model.layers.10.mlp.down_proj.
4. No arithmetic FAULT on `same`, `triton` or `awq`, and no isolation LEAK on any role (`sdcguard isolation` runs on
   each and no policy here requires it: none of these changes is expected to make one request depend on another).
5. Every decision has its arithmetic, approximation and performance sections.

Reported as found, deciding nothing: what sdcguard costs (`base` against `plain`: throughput and latency at both
loads, and the KV cache vLLM reports at startup), the seconds from the fault's injection to its first alert, every
other decision (`triton` and `awq` under both policies, `same` under
`speed`, where no gain is expected), the speed changes, and the model-check verdict on `fault`: whether a quality
check alone would have passed the faulty server. Also reported, after the servers: vLLM's own Trace Replay (vLLM
0.32's feature for comparing configurations along a fixed token sequence, through the decode path) run offline on
the recorded completions for the trusted stack (its noise), the same weight fault and the AWQ model, through the
quality check's metrics and verdict lines: whether that tool, used as documented, would have seen the fault. Not
claimed whatever the outcome: anything about other models,
GPUs, loads or vLLM versions; and a decision covers the prompts and load it ran, the plugin's checked layers and
nothing else (attention, norms and sampling only through the quality check).

Result of the first run (October 24, on 54c3a69): failed criteria 2 and 3, from two mistakes of ours. Every
decision came out REJECT on quality with mean KL about 5e-4 and top-1 agreement 95.5% (the generation_config bug
written up under "Qualify on a real server": Qwen2.5-7B-Instruct's repetition penalty of 1.05 made the server's
reference not greedy), and every arithmetic section INCONCLUSIVE: "one layer in eight per step" above was wrong
about the mechanism. A share of 1/8 checks the same layers every step (15 of 113 here, 13% of the weights), decide's
default policy counts the plugin's silence only at half the weights or more, and model.layers.10.mlp.down_proj, where
the fault went, wasn't among the 15, so the fault drew no alert. Amended before the rerun: sdcguard runs on every
layer in each server with sdcguard (SDCGUARD_SHARE=1, as qualify does), and the cost reported (`base` against
`plain`) is that of every layer. The criteria stand as written.

Result of the rerun (October 24, on 63f01f5, 26.6 minutes): every criterion met. `same` and `triton` ACCEPT under
`hold` (throughput +0.7% and +0.6%, within the noise), REJECT under `speed` (no 10% gain); `awq` REJECT under both on
its quality check (FAIL; 28.9% more throughput than the sdcguard-instrumented bf16 `base`, not than plain vLLM);
`fault` REJECT under both, the reason naming model.layers.10.mlp.down_proj. What the fault's 649 alerts said: the
layer's weights no longer matched their checksums at load (`weights_changed`); none showed an output wrong for the
weights as they were then. That is a corrupted weight, caught: not wrong arithmetic. Reported: the quality check
passed the faulty server (PASS over 32 prompts, 2,048 positions), so this check alone would have accepted it; the
faulty server also lost 27.8% of its throughput, a second reason under these policies; every layer checked cost
22.7% of `base`'s output tokens per second at 16 requests at once against `plain` (2,514 to 1,944); vLLM's Trace
Replay scored no position for any of the three runs (INCONCLUSIVE), so it says nothing here.

Amendment (October 24, after the rerun): `sdcguard decide` reported the changed weight under its arithmetic
section, as a confirmed wrong output. A GPU can multiply a corrupted weight correctly, and an engineer needs to know
which one failed, so a changed weight is now its own section (weights: CHANGED, UNCHANGED or UNMEASURED) and its
own count (`weights_changed`, apart from `confirmed_wrong`), and still rejects. Criterion 3 reads, from now on: the
fault's rejection gives the weight reason (or, on the code before this, the arithmetic one) naming the layer;
criterion 4 also asks for no weights CHANGED on `same`, `triton` and `awq`.

### Answers that depend on the request before (written October 24, before any run)

A server can compute every matmul right and still answer one request with another's data. vLLM issue #62124
(reported October 17, 2026, open, fix PR #62138 open) says that with FP8 per-tensor MoE quantization and CUDA graphs,
a short prompt's greedy answer changes with the request before it: the reporter's [1, 2, 4, 8] capture sizes made 88
of 114 probe pairs differ on an L40S, and 0 of 114 with --enforce-eager. The reported cause is padding rows of the
graph's batch carrying stale data into the per-tensor activation scale. No matmul checksum sees this, because each
matmul is right for the numbers it was given. `sdcguard isolation` runs two probes against a server: the primer
probe (a short probe prompt run alone right after a primer request run alone, every request with its own
cache_salt; the probe's tokens and log-probabilities must not depend on the primer; the probe after the same primer
twice is the control and must agree exactly, or its case isn't judged), and canaries (64 requests with shared
document prefixes and a random code each, 32 at a time, a quarter cancelled after their first chunk; no answer may
hold another request's code). It has run only against stand-ins.

`experiments/vllm_isolation_check.py` (`modal run scripts/modal_vllm.py --phase isolation`, one L4 per vLLM version,
0.31.0 and 0.32.0 at once, about 15 to 25 minutes and $0.40 to $0.70 together) serves
ibm-granite/granite-3.1-1b-a400m-instruct with the reporter's compilation settings (capture sizes [1, 2, 4, 8],
FULL_AND_PIECEWISE) three ways: bf16 (the trusted stack), `--quantization fp8_per_tensor --moe-backend triton` (the
reported configuration) and the same with `--enforce-eager`. Each gets `sdcguard isolation` and `gate model-check
--url` against bf16's reference, then `sdcguard decide` on bf16 -> fp8pt and bf16 -> eager with a policy that
requires the isolation check and accepts a model-check WARN. An L4 is sm_89 like the reporter's L40S; that the bug
shows on an L4 is itself untested.

Criteria, on vLLM 0.31.0 (the reported version):
1. All three servers start and serve.
2. fp8pt: the primer probe says LEAK.
3. eager and bf16: NO LEAK SEEN (both probes judged).
4. No canary holds another request's code on any role.
5. The decision on bf16 -> fp8pt is REJECT, with the isolation reason.
On vLLM 0.32.0 the same is reported as found: whether the bug is still there decides nothing. Also reported:
how many primer pairs differ and at which token, the model-check verdicts, and the decision on bf16 -> eager (its
arithmetic has no source here, so NOT DECIDED is the expected honest answer), and vLLM's own Trace Replay run offline
on bf16's recorded completions under each configuration (bf16 as its noise), as its documentation suggests for
comparing quantization schemes. Expected there, not a criterion: fp8pt and eager score about the same, because a
comparison along a fixed token sequence measures the approximation and doesn't run the request order that
triggers the bug. Not claimed whatever the outcome:
anything about other models, GPUs or serving settings, or that NO LEAK SEEN means a server is isolated beyond these
prompts.

Result (October 24, on 54c3a69, run on vLLM 0.31.0 and 0.32.0 at once): every criterion met on both. fp8pt LEAK: 20
of 192 primer answers changed with the request before them on 0.31.0, 55 of 180 on 0.32.0 (the bug is still there in
0.32.0); bf16 and eager 0 of 192 on both; no canary held another request's code; bf16 -> fp8pt REJECT with the
isolation reason. A LEAK here means answers changed with the request before them (cross-request influence);
no canary held another request's text, so no disclosure of another request's content was shown. Reported: bf16 ->
eager came out REJECT, not the expected NOT DECIDED, on its quality check alone
(top-1 agreement under 97%), from the server-reference bug written up under "Qualify on a real server"; its
isolation was NO LEAK SEEN.

### Bit flips in a served model's weights (written October 24, before any run)

Every sensitivity number so far is from training matmuls (one wrong value in an output: the default check flagged
half at about 29% of a typical output, none of 594 at 10%) or from a few flips chosen to be caught. Nothing says
how much of what goes wrong in a served model's weights the vLLM plugin sees, or whether what it misses matters.
`experiments/vllm_sensitivity.py` (`modal run scripts/modal_vllm.py --phase sensitivity`, one H100, vLLM 0.32.0,
Qwen2.5-1.5B-Instruct in bf16, every linear layer checked, about 20 to 30 minutes and $1.60 to $2.40) flips one bit
of one weight at a time in the running server: every bit position of a bf16 weight (0 to 6 mantissa, 7 to 14
exponent, 15 sign), 5 trials each (80), at layers and positions drawn with seed 0 from the q/k/v, o, gate/up and
down projections of all 28 blocks. With the flip in place: 8 requests, then the quality check (`gate model-check
--url`, 32 prompts teacher-forced against the clean server's reference, with a clean check as noise), then the bit
flipped back. A trial is detected when an alert names its layer between the flip and 5 s after the flip back; it
changed the answers when the quality check says WARN or FAIL.

Criteria (about the run, not about how much is caught): 1. the server starts with SDCGUARD_REQUIRE=1; 2. no alert in
64 requests with no fault; 3. the clean quality check is PASS; 4. every flip is undone (the weight reads back as it
was). Reported as found, by bit and in total: trials detected, trials that changed the answers, and the ones that
changed the answers and weren't detected (the misses that matter); alerts on layers other than the flipped one
(a huge weight can make the next layers' inputs non-finite). Expected from the training numbers, not a criterion:
a flip that makes a typical weight much larger (the exponent's top bit, or a 0 bit that multiplies it by 16 or
more) is caught and changes the answers; a flip that makes it smaller, or changes it by under about 10%, isn't
caught and usually doesn't change them. Not claimed whatever the outcome: rates for hardware faults (a real fault
isn't one bit of one weight at a time), other models, or one layer in eight (a flip in an unchecked layer is never
caught there).

Result (October 24, on 54c3a69): failed criterion 3, met 1, 2 and 4. The clean quality check was FAIL (mean KL 9e-6,
top-1 agreement 90%): the server-reference bug written up under "Qualify on a real server". With the clean check
failing, "changed the answers" (a trial's quality check WARN or FAIL: 79 of 80, bit 0 included) measures nothing
here and isn't taken as a result. What stands: no alert in the 64 clean requests and none on another layer in any trial; detected, by bit:
all 5 flips of bit 14 (the exponent's top bit: a typical weight becomes enormous), 1 of 5 each for bits 8, 9 and 10,
and none for bits 0 to 7, 11 to 13 (in a typical weight these exponent bits are 1, so the flip shrinks it) and 15 (the
sign). Rerun with the fix before anything about changed answers is said.

Result of the rerun (October 25, on 64c9d02): every criterion met. The server started with SDCGUARD_REQUIRE=1, no
alert in the 64 clean requests, the clean quality check PASS (mean KL 8.1e-6, top-1 agreement 100%), every flip
undone. Found, by bit: detected, as in the first run, all 5 flips of bit 14 and 1 of 5 each of bits 8, 9 and 10 (8
of 80), none of the others; no alert on another layer. The 4 trials whose quality check said FAIL ("changed the
answers") were all bit-14 flips, which made one weight about 10^36 to 10^37, and all 4 were detected: one changed
29% of the top-1 tokens; the other three gave the same mean KL, 3.13, on three different layers, with top-1
agreement 100%, which fits a flat output (every token equally likely: the quality check counts a tie for the top as
agreement) rather than answers that stayed the same. The fifth bit-14 flip's quality check was INCONCLUSIVE (it was
detected), and the three detected flips of bits 8 to 10 passed it. The quality check passed all 72 flips the plugin
missed: 70 of them moved the answers measurably (mean KL up to 5.3e-4, against 8.1e-6 with no fault; top-1 agreement
at least 99.9%) but under its line, and two flips of bit 0 changed nothing it measures. For this model, every layer
checked and this quality check, no flip that failed it went undetected.

### FP8 arithmetic on Hopper: what a check would see (written October 24, before any run)

The vLLM plugin doesn't check FP8 layers. Before building a check for them, this measures what it would have to
tell apart from a fault. vLLM's `--quantization fp8` on Hopper quantizes weights to FP8 E4M3 per tensor and
activations per token, and multiplies them with CUTLASS kernels that use FP8 fast accumulation (every Hopper FP8
configuration in vLLM 0.32's `scaled_mm_sm90_fp8_dispatch.cuh`). DeepSeek-V3's report describes Hopper's FP8
tensor-core accumulation as limited to about 14 bits; how far that moves a block sum on real layers is the
question. `experiments/fp8_residuals.py` (`modal run scripts/modal_vllm.py --phase fp8_residuals`, one H100, about 5
to 10 minutes and $0.40 to $0.80) takes the inputs and weights of 16 linear layers of Qwen2.5-1.5B-Instruct on 16
prompts, quantizes them as vLLM does, and compares vLLM's `cutlass_scaled_mm`, and `torch._scaled_mm` with and
without fast accumulation, with exact float64 arithmetic on the same FP8 values and scales: per output, the error
relative to its row's largest value and its sign bias; per 128-column block, the sum's distance from the exact sum
in units of what bf16 rounding of the outputs alone explains.

No pass line; what decides the design, written now: if the fast kernel's clean block scores stay under 3 on every
layer (at the 99.9th percentile), an FP8 check can use the current noise model with the FP8 operands; if they reach
6 or more, the check needs an accumulation term fitted to these numbers, and catches correspondingly less than the
bf16 check (the run gives the size). Either way, no FP8 check is called supported until one meets criteria of its
own on a real server (same answers as plain vLLM, no clean alert, a flipped weight caught).

Result (October 24, on 54c3a69, torch 2.13 on an H100, 148 tokens per layer): the second case. With fast
accumulation (vLLM's CUTLASS kernel, and torch._scaled_mm with it on, which gave the same numbers) block scores reached
29.7 at worst and 25.0 at the 99th percentile on the worst layer (model.layers.27.mlp.down_proj, K = 8960), and 9.6 to
12.8 on others; without it they stayed at 6.4 at worst and 4.1 at the 99th percentile. The error grows with the
inner dimension and with outliers in the inputs. So an FP8 check needs an accumulation term that grows with K (and
the inputs' sizes) and would catch less than the bf16 check; none is built, and FP8 matmul arithmetic stays
unchecked. What the plugin checks for FP8 layers is their input quantization (SDCGUARD_FP8=1, "An FP8 MoE bug from
detection to remedy").

### Calibrating an FP8 matmul check (written October 24, before any run)

The model of the error to fit, from published measurements of the tensor cores (DeepSeek-V3's report, section 3.5.2;
Khattak and Mikaitis, arXiv 2512.07004; MMA-Sim, arXiv 2511.10909) and vLLM 0.32's source: every sm90 configuration
of vLLM's `cutlass_scaled_mm` uses FP8 fast accumulation, and most sm89 ones do; the tensor core sums 32 products (16
on Ada) with the running accumulator at a time, aligned to the largest, keeps 13 fractional bits and truncates the
rest toward zero, over the whole of K; the epilogue is bf16(a_scale * (b_scale * acc)) in FP32. For a row i and a
block b of 128 output columns, the check would compare the sum of the outputs with the exact sum of the same FP8
products, E, allowing |O - E| <= z * sqrt(S^2 + alpha^2 A^2 + beta^2 B^2): S the outputs' own bf16 rounding, A = u *
sqrt(K * Q) the truncation's random part (Q the sum of the squared products, u = 2^-13), B = u * K * Y its bias
where products add coherently (Y the sum of |y|). On an emulator of that model (CPU, `tests/test_fp8mm_calibration.py`),
inputs with a common offset moved block sums at K = 8960 by up to 34.7 times S, the same order as the H100 run's 29.7
(in a unit within a factor of 2 of S); zero-mean inputs by 2.2. B is thousands of times S there, so if the bias
needs the B term with a beta of even 0.01, the check would allow tens of times the bf16 noise and catch only very
large errors.

`experiments/fp8mm_calibration.py` (`modal run scripts/modal_vllm.py --phase fp8mm_cal`: one H100 with
Qwen2.5-1.5B-Instruct, every linear layer, and Qwen2.5-7B-Instruct, every linear layer of blocks 0, 7, 14, 21 and 27;
one L4 with the 1.5B; about $1 to $2 in all) quantizes each layer as vLLM does (the weight per tensor, the
activations per token, and separately per tensor), multiplies with vLLM's `cutlass_scaled_mm` and, for comparison,
cuBLASLt without fast accumulation, on 64 prompts (questions, code, six other languages, numbers, repetition) and 4
long documents, and keeps r, S, A, B and E for 256 rows of every layer. And for 16 of those rows it recomputes the
products with the emulator above and the epilogue, and counts the outputs that come out bit for bit as the GPU's.

No pass line; what each outcome decides, written now:
- If the emulator gives the GPU's outputs bit for bit (or within one bf16 step) on at least 99.99% of outputs on a
  GPU, the FP8 check there can recompute sampled rows with the tensor core's own arithmetic and compare them exactly,
  which is far more sensitive than any tolerance; a run on a server would then decide it. If it doesn't, the model is
  wrong for that GPU in some detail, and the check uses a fitted tolerance.
- The tolerance is fitted per GPU, recipe and kernel on 52 of the 68 prompts and judged on the other 16: alpha and
  beta as small as the clean data allow, with the largest clean score on the held-out prompts at most 3 (half the
  alarm line of 6). Reported with it: how many times the bf16 noise it allows on each layer, so what errors it can
  see.
- Either way, no FP8 matmul check is called working until a run on a real server meets criteria of its own (no clean
  alert, the same answers as plain vLLM, injected faults caught).
Not claimed whatever the outcome: other models, block-scaled FP8, GPUs other than the H100 and L4, kernels other
than these.

Results (October 24, on 03e27ba; `results/fp8mm_cal_oct24/`; the per-layer arrays stay on the Modal volume; about
$0.18): 231 layers on the H100 (196 of the 1.5B, 35 of the 7B) and 196 on the L4.
- The emulator, against vLLM's CUTLASS kernel, on 16 rows of each layer (the first 16 of its 256 sampled rows: tokens
  of the short prompts, apart from 23 of the 3,696 rows on the H100 and 18 of the 3,136 on the L4, from the first long
  document): bit for bit on 99.45% of outputs (activations scaled per token) and 98.27% (per tensor) for the 1.5B on
  the H100, 98.89% and 97.33% for the 7B, 99.23% and 97.65% on the L4; within one bf16 step, 99.93%, 99.77%, 99.85%,
  99.61%, 99.90% and 99.68%. The largest differences were about 29,000 to 30,000 bf16 steps. Steps are counted
  through zero, so a difference that large means the two outputs had opposite signs, at least one of them within
  about 0.002 of zero; how far apart they were in value wasn't recorded. All are below 99.99%, so by the rule written
  before the run the model is wrong in some detail on both GPUs, and the check would use a fitted tolerance.
- What the data is. About 90% of the tokens, so of the sampled blocks (90.2% on the H100, 90.7% on the L4), come
  from the 4 long documents, and those are one paragraph's 42 words shuffled and repeated 40 times
  (`fp8mm_calibration.long_docs`). The 64 short prompts are the other 10%.
- The fit (`experiments/fp8mm_fit.py`, `FIT.json`). The rule written before the run names a 52/16 split and asks
  for a tolerance as small as the clean data allow; the rest was decided after the run. The 16 held-out prompts are
  drawn with a fixed seed (0), independent of the data: 14 short prompts and two of the long documents (65 and 66;
  64 and 67 are in training). So the held-out blocks (47.5% of all on the H100, 47.7% on the L4) are 95% long
  document and the training blocks 86%: the held-out test is mostly the same kind of text again. "As small as the
  data allow" was read as: for each beta on a grid (0 and 41 values from 1e-5 to 1), the smallest alpha keeping
  every training score at most 3, and of those pairs the one with the smallest median allowance. No block had a value
  that wasn't finite or a bf16 noise of 0 (S, A and B are kept in float16). For CUTLASS: alpha 3.9 and beta 0.0056 on
  the H100 (per token; 3.8 and 0.0056 per tensor), alpha 4.3 and beta 0 on the L4 (4.2 per tensor). The largest
  held-out clean score was 2.53 and 2.22 on the H100, 2.04 and 2.14 on the L4: at most 3, as asked. The short prompts
  set the fit (the training worst of 3 was on them), and on the 14 held-out short prompts alone the worst was 1.90
  and 2.01 on the H100, 1.60 and 2.01 on the L4. For `torch._scaled_mm` without fast accumulation (the comparison
  kernel, which vLLM doesn't run for these layers) the fit missed its target: held-out worst 3.02 to 3.13, all on the
  long documents (2.53 to 2.72 on the short prompts).
- What the tolerance allows, in units of a block's own bf16 noise: 12.5 for the median block on the H100 and 10.7 on
  the L4 (99th percentile 77 and 30). By layer, about 8 to 13 for most of the 1.5B's layers, 41 for its down
  projections on the H100 (K = 8960; 27 on the L4), about 17 to 21 for most of the 7B's and 78 for its down
  projections (K = 18944). The alarm line sits at 6 times those (about 50 to 80 times the bf16 noise for most of the
  1.5B's layers, 100 to 130 for most of the 7B's). A clean block can score up to 3 by itself, so an error in a
  block's sum is caught whatever the clean part only above about 9 times (from the per-layer allowances, about 70 to
  120 times the bf16 noise for most of the 1.5B's layers and 150 to 190 for most of the 7B's; for the median block of
  each GPU's run (both models on the H100, the 1.5B alone on the L4), `caught_for_sure_in_bf16_noise_units_median`
  in FIT.json: 113 on the H100 and 96 on the L4, per-token scaling); one between about 3 and 9 times may or may not be. That is far less than the bf16 check catches.
Not built. As written, no FP8 matmul check is called working until a run on a real server meets criteria of its own.

### Norm checks in serving (written October 24, before any run)

The vLLM plugin can now check RMSNorm layers too (`src/sdcguard/serve_norm.py`, SDCGUARD_NORMS=1, off by default
until this run says otherwise): after every call of an RMSNorm or
GemmaRMSNorm layer, each output is compared with the norm recomputed in fp32 from the input the layer used (the
returned sum, when the residual add is fused in) and the weights it had at load. vLLM rounds once (its CUDA kernel,
eager mode) or twice (its native PyTorch version, which torch.compile runs), and its native fused version normalizes
the sum before rounding it, so a correct output is within 4 bf16 roundings of the recompute (4.03 counting the
products of the rounding errors, and the recompute's own fp32 rounding adds about 0.0003). An output's score is its
distance from the recompute in roundings of its own size; the line is 6. The check runs as one opaque custom op (a
Triton kernel, compared with the PyTorch version at first use on each GPU), because traced into the compiled graph
it would be merged with the norm's own mean and rsqrt and compare the norm with itself.

What a flip of one bit of a norm weight w does to the outputs that use it: sign and exponent bits change w by a
factor of 2 or more; mantissa bit k (0 to 6) changes it by between 2^(k-8) and 2^(k-7) of its value. Caught for sure
when the change is over about 10 roundings (3.9%): sign, exponent, and mantissa bits 4 to 6 (at least 16 roundings).
Bit 0 (at most 2 roundings) can reach a score of 6.06 only if w is a power of two and every rounding falls the same
way, so it is practically never caught. Bits 1 to 3 are between the two.

`experiments/vllm_norm_check.py` (`modal run scripts/modal_vllm.py --phase norms`, one H100, vLLM 0.32.0, bf16,
prefix caching off on every server, about 35 to 50 minutes and $2.80 to $4): Qwen2.5-1.5B-Instruct plain, guarded
(SDCGUARD_REQUIRE=1, SDCGUARD_NORMS=1, compiled with CUDA graphs, vLLM's default), guarded without norm checks
(SDCGUARD_NORMS=0) and
guarded in eager mode; Qwen3-1.7B (per-head q/k norms) plain and guarded. The trials: 40 flips on the guarded
Qwen2.5 (bits 15, 14, 10, 7 three times each, 6 to 0 four times each, at norm layers and positions drawn with seed
0), each with 4 requests and a read with the flip in place, then the bit flipped back; 5 exact-naming trials on the
eager server (32 requests at once, one holding a marker token whose row of one norm's output is made wrong after the
norm, the design of A4 above); 4 flips of bit 14 in Qwen3's q/k norms.

Criteria:
- N1: every server starts; each guarded one checks every RMSNorm layer (57 in Qwen2.5-1.5B, 113 in Qwen3-1.7B)
  with the Triton kernel.
- N2: the guarded servers' 64 greedy completions are identical to plain vLLM's (both models).
- N3: no alert with no fault on any guarded server (identity requests, the speed load); the worst norm score read
  back stays at or below 4.05 (compiled and eager). A worst score above that fails the bound even with no alert.
- N4: every flip of bits 15, 14, 10, 7, 6, 5, 4 of a nonzero weight is caught by an alert naming the flipped layer;
  no alert with a finite score names another layer during a trial (a norm weight changes the next layers' inputs,
  not their arithmetic); every flip is undone.
- N5: in all 5 exact-naming trials, every alert with a kept row names exactly the request holding the marker.
- N6: the 4 Qwen3 q/k norm flips are caught, naming the layer, with no finite-score alert elsewhere.

Reported, not criteria: detection and changed answers by bit for bits 3 to 0, non-finite alerts on other layers,
and the cost: output tokens per second with and without norm checks at 16 and 256 requests at once. Decided now: if
norm checks pass and cost at most 5% at 16 at once, they become on by default; if they cost more, they stay
opt-in. Not claimed whatever the outcome: rates
for hardware faults in norm arithmetic (only weights are flipped here, and output corruption in eager mode), norms
fused into other kernels, or models with other norms (each run's log lists the norm layers it doesn't check).

Result (October 24, on 54c3a69): failed N2, met the rest. With norm checks on, 5 of Qwen2.5-1.5B's 64 greedy
completions differed from plain vLLM's in the compiled server (Qwen3-1.7B's 64 were identical). The likely cause, not
yet shown: the check's op after each norm changes what Inductor fuses around the norm, so the norm's reduction runs
in another order and a few near-tie tokens flip; the checks themselves were right. Of the 40 flips, every flip of
bits 2 to 15 of a nonzero norm weight was caught (32 of 32); bits 1 and 0 weren't (8 trials, no answer changed); no clean alert (worst
clean scores 2.65 compiled, 1.00 eager, 2.52 on Qwen3); the 5 exact-naming trials each named the marked request; the
4 Qwen3 q/k norm flips were caught. Cost: 7.2% fewer output tokens per second at 16 requests at once, 10.4% at 256.
Norm checks stay opt-in, for the cost and because they can change a compiled server's answers.

### Qualify on a real server (written October 24, before any run)

`sdcguard qualify` has only run against stand-in servers. Reading its defaults against decide's while preparing this
run found that they contradicted each other: qualify checked 1/8 of the layers by default, and decide's default
policy counts the plugin's silence only when it checks at least half the weights, so on a real server qualify with its
own defaults could not have accepted anything (the stand-in reported 97% coverage whatever the share). Qualify now
checks every layer by default, and the stand-in's coverage follows the share.

`experiments/vllm_qualify_check.py` (`modal run scripts/modal_vllm.py --phase qualify`, one L4, vLLM 0.32.0,
Qwen2.5-1.5B-Instruct in bf16, 7 server launches, about 35 to 50 minutes and $0.85 to $1.20): one `sdcguard
qualify` with its defaults (the plugin on every linear layer, decide's default policy, no policy file) and
--repeat-baseline, with three
candidates: `same` (the baseline's configuration), `triton` (--attention-backend TRITON_ATTN; the quality check said
PASS for it on an H100 on October 23) and `badeps` (rms_norm_eps set to 1.0 through --hf-overrides; FAIL on the H100,
mean KL 11.2).

Criteria:
- Q1: all five servers start; qualify writes REPORT.md and exits 1 (one REJECT expected).
- Q2: `same` is ACCEPT, with arithmetic NO FAULT OBSERVED (the plugin's coverage counts) and isolation NO LEAK SEEN.
- Q3: `badeps` is REJECT for quality (OUTSIDE), with arithmetic NO FAULT OBSERVED: the arithmetic is right, the
  configuration is wrong, and the decision keeps the two apart.
- Q4: no alert on any server (no fault is injected).
- Q5: the `same` configuration started again as a deployment (the plugin on, its metrics) is PROTECTED under
  `sdcguard ready --qualified decision_same.json`: the stack the plugin reads from inside the server matches the one
  recorded in the decision.
- Q6: the `triton` configuration under the same decision is NOT THE QUALIFIED STACK (exit 1), naming the attention
  backend as the difference.

Reported, not a criterion: `triton`'s decision (expected ACCEPT from the H100 result), each server's speed and the
isolation probes' details. Not claimed: anything about other models or GPUs, or about decisions under policies other
than the default.

Result of the first run (October 24, on 54c3a69): failed Q2, Q5 and Q6, from one bug of ours. `same`, `triton` and
the repeated baseline all came out REJECT on quality with mean KL about 5e-4 (well inside the pass line) but top-1
agreement 88%. vLLM applies a model's generation_config.json to requests that don't set a sampling parameter
(the server logged Qwen2.5-1.5B-Instruct's repetition_penalty of 1.1), so the "greedy" reference recorded from the
server wasn't greedy: its recorded token was its own most likely token at only 88.4% of the 2048 positions, and
top-1 agreement was counted against the recorded token. Q5 and Q6 failed only because `same` wasn't accepted. Fixed
before the next run: the reference is recorded with every sampling setting neutral in the request, keeps how often
its recorded token was its own most likely one, and top-1 agreement is counted against the reference's most likely
token (the consistency probe's requests are neutral too). The criteria stand as written for the rerun.

Result of the rerun (October 24, on 63f01f5): every criterion met. `same` ACCEPT (top-1 agreement 100%, mean KL
5.1e-4; arithmetic NO FAULT OBSERVED, isolation NO LEAK SEEN), `triton` ACCEPT, `badeps` REJECT on quality with
arithmetic NO FAULT OBSERVED; no alert on any of the 5 servers; the deployed `same` PROTECTED under
`ready --qualified`, `triton` NOT THE QUALIFIED STACK naming the attention backend. The reference's recorded token
was its own most likely one at every position. `sdcguard consistency` ran on every server: CONSISTENT on all 5,
prefix-cache hits confirmed on each, decode-vs-prefill largest gap 0.10 nats on the baseline's stack (0.15 on Triton).
Found in the results, a gap of ours: `badeps` had the trusted stack's fingerprint (the overridden rms_norm_eps
wasn't part of the stack), so `ready --qualified` would have taken a server started that way for the qualified
stack. Fixed since: the stack has the model's effective config (a hash), its overrides, its revision and its
generation config, and `ready` names a difference in any of them (CPU-tested).

### MoE checks in serving (written October 24, before any run)

The vLLM plugin can now check fused MoE layers (`src/sdcguard/serve_moe.py`, SDCGUARD_MOE=1, off by default until
this run says otherwise). A fused MoE kernel computes every routed expert of every token at once, so no single
matmul can be checked from outside. Instead, at each model step one MoE layer (they take turns by step number)
recomputes the routed output of 2 rows of its batch from the layer's own input, routing and expert weights, with the
rounding steps of vLLM's Triton MoE kernel (TritonExperts: the first matmul rounded to bf16, the activation rounded
before its multiply and the product rounded, as vLLM's act_and_mul kernel does, the second matmul times the routing
weight rounded, the experts summed in fp32 and rounded once, as moe_sum does). An output's score is its distance
from the recompute against a bound made of one rounding of the output, two of the largest expert's output, the
spread that intermediates rounded differently everywhere would cause, and one worst-placed flip of an
intermediate; the line is 6. Three early-exit Triton kernels do the work, so the inactive layers cost three
launches each per step.

On the CPU, against a kernel that rounds the same way but adds in another order, 99.97% of bf16 outputs came out
identical and the worst clean score was 0.25; a wrong expert, another row's output, an output 1.5 times too large,
a NaN and a flipped exponent bit scored from 20 to over 1,000. In bf16 the bound means an output must be off by
roughly a quarter of its value to be seen, so the check is for gross errors (a kernel's indexing, routing weights,
stale rows, non-finite values), not small ones. It does not see a weight changed in memory: the kernel and the
recompute read the same expert weights.

`experiments/vllm_moe_check.py` (`modal run scripts/modal_vllm.py --phase moe`, one H100, vLLM 0.32.0,
ibm-granite/granite-3.1-1b-a400m-instruct in bf16 (24 MoE layers, 32 experts, top 8), --moe-backend triton and
prefix caching off on every server, about 25 to 35 minutes and $2 to $2.80): plain, guarded (SDCGUARD_REQUIRE=1,
SDCGUARD_MOE=1, compiled with CUDA graphs), guarded without MoE checks (SDCGUARD_MOE=0) and guarded in eager mode.
Faults can't be put into a kernel inside captured CUDA graphs from Python, so the compiled server runs clean; the
eager one runs 6 trials, each making one MoE layer's kernel wrong after the fact (its output 1.5 times too large;
the kernel given each row's experts shifted by one; each row given another row's output), with up to 16 rounds of
4 requests at once until an alert names the layer (at once, because a layer checks the batch of the step its turn
comes in, and a batch of one row hides rows mixed up).

Criteria:
- M1: every server starts; the guarded and eager servers check all 24 MoE layers with the Triton kernels (passed
  their first-use comparison on the H100).
- M2: the guarded server's 64 greedy completions are identical to plain vLLM's.
- M3: no alert with no fault on the guarded and eager servers; the worst clean score read back is at most 2; the
  checks ran (active calls counted).
- M4: all 6 wrong kernels are caught by an alert naming the layer.

Reported, not criteria: rounds until each catch, the MoE backend vLLM uses, and the cost (output tokens per
second with and without MoE checks, at 16 and 256 requests at once). Decided now: if MoE checks pass and cost at most
3% at 16 at once, they become on by default; if they cost more, they stay opt-in. Not claimed whatever the outcome:
other MoE kernels or models, small errors, or faults in expert weights in memory.

Amendment (October 24, written after a run of this experiment on the earlier code had started, before any run with
it): SDCGUARD_MOE=1 now also turns on a routing check (`src/sdcguard/serve_route.py`), because the experts' recompute
takes the routing as given and agrees with a router that picked the wrong experts. At each step one router (they
take turns, as the MoE layers do) recomputes its policy in fp32 for 2 real rows from its own logits: softmax or
sigmoid scores, the correction bias for choosing, DeepSeek's grouped top-k, renormalization and the routed scaling
factor, for vLLM's top-k, grouped top-k and biased top-k routers (others are listed as not checked). An expert
counts as chosen correctly when its score reaches the k-th best within a tolerance (1e-6 of the score for fp32
logits, plus 4 unit roundoffs of a 16-bit logits dtype, because vLLM's grouped top-k kernel computes scores and adds
the bias in that dtype; where groups near the cut are that close, the bounds that hold whichever group was taken);
the weights are the chosen experts' scores renormalized and scaled, to 2e-5 plus 1e-4 of their size plus 6 unit
roundoffs of a 16-bit dtype. The first bad row is kept and judged again in float64 on the CPU. One early-exit Triton
kernel per router call does the work. On the CPU: 1,200 rows in each of 5 cases of routing rounded as vLLM's 16-bit
kernels round (bf16 and fp16, sigmoid with a bias and softmax, ties broken at random) gave no alarm; the Triton
kernel and the PyTorch version gave the same counts, records and kept rows in 84 cases. A run on the earlier code is
judged on M1 to M4 as written. From this amendment on: M1 also asks that the guarded and eager servers check the
routing of all 24 routers with the routing check's Triton kernel (passed its first-use comparison on the GPU); M3
also asks that routing checks ran (their alerts already count in M3); and M5: 2 routers made wrong in eager mode
(each row's last expert replaced by the next expert number; every weight 1.5 times too large) are both caught by an
alert naming the router with the matching finding, within 16 rounds of 4 requests. The cost reported is now that of
both checks.

Result of the run on the earlier code (October 24, on 54c3a69): guarded, nomoe and eager didn't start, so nothing
was measured. SDCGUARD_REQUIRE=1 counted the 24 fused MoE layers as left unchecked even with the MoE check on (a
bug of ours: the requirement didn't know about the MoE check), and on nomoe it refuses by design, since nothing
checks those layers there. Fixed before the rerun: fused MoE layers the MoE check registered aren't left over for
SDCGUARD_REQUIRE (their weights still don't count as checked by checksums in the weight share decide reads), and
nomoe runs without SDCGUARD_REQUIRE. The criteria stand.

Result of the rerun (October 24, on 63f01f5): failed M1 and M2, for two bugs of ours, both in the routing check.
Its Triton kernel didn't compile on the H100 (a compare-and-swap on the int64 kept-row flag with int32 literals,
which Triton's compiler rejects and its interpreter, where the CPU tests ran it, accepts), so it fell back to its
PyTorch version, which read a GPU value on the CPU; inside the compiled server's CUDA graph capture that is an error
("operation not permitted when stream is capturing"), the capture was invalidated and guarded didn't start, so M2
and the cost couldn't be measured. Everything else asked of the eager server was met: its 24 MoE layers checked with
the MoE Triton kernels (they passed their first-use comparison), no alert in clean traffic (worst clean score
0.26, under 2), all 6 wrong kernels caught by an alert naming the layer (in 1 or 2 rounds, each rechecked as
output_wrong), and both wrong routers caught by an alert naming the router with the matching finding (wrong_experts
in 1 round, wrong_weights in 2, each with an alert rechecked as output_wrong), with the routing check on its PyTorch
version. nomoe started. Fixed since: the flag's compare-and-swap is typed int64; every Triton kernel of the serving
checks is compiled for the A100, L4 and H100 in the CPU tests, with no GPU and specialized as Triton specializes a
real launch (`tests/test_triton_compiles.py`, with Triton 3.8.0; the old routing kernel fails it, and with Triton
3.7.1, the version in vLLM 0.32's image, the old kernel fails with the GPU's error at the same line and column and the
fixed one compiles); no check's PyTorch version reads a GPU value (see the FP8 result below). The criteria stand
for the rerun.

Amendment (October 24, after the reruns above were handed over, before any run with it): the rows a layer's check
takes were drawn from the step number, and the layers take turns by the same number; when the number of MoE layers
shares a factor with a steady batch's rows, each layer then only ever sees a few of them (24 layers and 16 rows: 2
rows a layer). The MoE and routing checks now draw rows from the layer's own turn count (the step number divided by
the number of layers), so each takes every row of a steady batch in turn (CPU-tested). The rerun on the earlier code
is judged as written; this changes which rows are sampled, not what a sample is judged against.

Amendment (October 25, before the next run): reported only, not criteria. The nomoe server answers the 64 prompts
too, and every server reports its cuBLAS reduced-precision setting (the plain server runs with PyTorch's default,
the others with the plugin's), so if M2 fails the run can say why: guarded against nomoe (both with the plugin's
setting) isolates the MoE checks; nomoe against plain is the plugin without its MoE checks (its cuBLAS setting and
the linear-layer checks together), and the reported setting shows the cuBLAS part on its own. M2 stays as written,
against plain vLLM at PyTorch's default.

Result of the third run (October 25, on 64c9d02): failed M3 on the compiled server, for a bug of ours; met M1, M2,
M4 and M5. With no fault, the guarded server raised 110 alerts, all from the experts' check, 109 with an infinite
score. Every one had kept a row whose 8 experts were all -1: vLLM 0.32 marks the padding rows of a CUDA graph's
batch that way (VLLM_MOE_SKIP_PADDING, on by default: the router gives them expert -1), and its Triton kernel then
computes nothing for them (moe_sum adds up workspace rows no matmul wrote for them), so the check compared leftover
values in rows no request holds against a recompute of zero. The eager server, whose batches aren't padded, had no
clean alert (worst score 0.30). Met: all four servers started and the guarded and eager servers checked all 24 MoE
layers and all 24 routers with the Triton kernels; the guarded server's 64 completions were identical to plain
vLLM's (and to nomoe's; the plain server ran with PyTorch's reduced-precision cuBLAS setting, the others with the
plugin's); on the eager server, 6 of 6 wrong kernels were caught by an alert naming the layer (in 1 or 2 rounds) and
2 of 2 wrong routers by an alert naming the router with the matching finding. The cost of both checks: 12.4% at 16
requests at once and 6.3% at 256, more than 3%, so they stay opt-in whatever a rerun shows. Fixed since: the
experts' check draws its rows from the step's real rows when the plugin knows them (the scheduled tokens, as the
routing check already did; packed above the step number in the layer's step buffer), and a row whose experts are all
-1 isn't judged or counted as checked (CPU-tested: a padded batch with leftover values in its padding rows gives no
alert, with the real rows known or not, and a wrong real row is still caught; the Triton and PyTorch versions agree
through Triton's interpreter, the kernels compile for the A100, L4 and H100 with Triton 3.8.0 and 3.7.1, and the
first-use comparison on the GPU now includes a padding row and leftover rows past the real ones). What this gives
up: a real row that a fault leaves with no experts (a router bug, a flipped bit in vLLM's padding mask) gets
leftover workspace as its output and isn't judged either; such rows among a step's known real rows are counted
(unrouted_real_rows, expected 0), not alarmed on.

Amendment (October 25, before the next run): the criteria stand as written and the next run is judged on all of
them. Also reported: over the clean traffic on the guarded and eager servers (from just after each was ready), the
rows the experts' check judged against the rows it sampled (2 for each active call; with the step's real rows known
they should match, apart from counts that a read still in flight at the start carried over from startup), and real
rows without experts (expected 0).

### An FP8 MoE bug from detection to remedy (written October 24, before any run)

Two new parts are tried on a real, open vLLM bug at once. The plugin's FP8 quantization check
(`src/sdcguard/serve_fp8.py`, SDCGUARD_FP8=1, off by default until this run says otherwise): right after vLLM
quantizes a layer's input to FP8 (an FP8 linear layer's input; each MoE layer's expert inputs, before both expert
matmuls, through vLLM's moe_kernel_quantize_input), it recomputes the recipe from the layer's input over the step's
real rows only (the plugin publishes their number before each step and clears it after, so warm-up and capture runs
aren't judged): a per-tensor dynamic scale against the scale the GEMM got (one that matches the maximum over every
row, padding included, is reported as taken from outside the batch), per-token scales and the FP8 values of 2
sampled real rows (two FP8 steps off is an alarm, one step is counted: vLLM's kernels use fast math). On the CPU
against vLLM's quantization as its kernels do it, correct scales pass and a per-tensor scale taken over padding rows
holding larger values is named with the request. And `sdcguard consistency` (cold vs prefix-cache-hit answers with
the hit confirmed from vLLM's counters; decode log-probabilities against the same text in one prefill), which
qualify now runs on every server.

vLLM issue #62124 (reported October 17, 2026): FP8 per-tensor MoE with CUDA graphs carries an earlier request's data
into a batch's activation scale through the graph's padding rows. `experiments/vllm_fp8_moe_check.py`
(`modal run scripts/modal_vllm.py --phase fp8_moe`, one L4, vLLM 0.32.0, about 60 to 85 minutes and $1.45 to $2):
one `sdcguard qualify` of ibm-granite/granite-3.1-1b-a400m-instruct with --moe-backend triton and the reporter's
compilation settings (CUDA graphs for batches 1, 2, 4 and 8), the plugin with SDCGUARD_FP8=1 on every server, a
policy that accepts a WARN (FP8 is an approximation) and requires the isolation check: the baseline in bf16 (twice),
and the candidates fp8pt (--quantization fp8_per_tensor, the reported configuration), eager (the same with
--enforce-eager: no padding rows) and fp8pc (--quantization fp8_per_channel: per-token activation scales, CUDA graphs
kept). Then the repro folder qualify writes for fp8pt is run once on its own.

Criteria:
- F1: the baseline, fp8pt and eager start.
- F2: fp8pt is REJECT, with its isolation check showing a LEAK (the defect is there with the checker on; the
  isolation run without the plugin, "Answers that depend on the request before", shows it without).
- F3: the FP8 check names at least one MoE layer's expert-input scale as taken from rows outside the batch on fp8pt.
- F4: eager has no FP8 alert and its isolation check shows NO LEAK SEEN.
- F5: the bf16 baseline has no FP8 alert.
- F6: fp8pt's repro folder, run on its own, shows the leak again.

Reported, not criteria: fp8pc (whether it starts, its FP8 alerts and isolation: a remedy we expect to work but
haven't seen), every candidate's decision, quality and consistency verdicts, the remedies' throughput against fp8pt
and the baseline (what each fix costs), and which requests the FP8 alerts name. Decided now: if F1 to F5 pass, the
FP8 quantization check becomes on by default for FP8 layers; if F3 fails while F2 passes, the check misses the
mechanism it was built for and stays off until it's understood. Not claimed whatever the outcome: FP8 recipes other
than these (block scales), the FP8 GEMMs' arithmetic (a separate check), other GPUs or MoE kernels.

Amendment (October 24, before the run): the rule above would turn the check on by default without its cost being
measured, and this run can't measure it (every FP8 server has the check on). So passing F1 to F5 makes the check
eligible; it becomes on by default only once a run measures it costing at most 3% of throughput at 16 requests at
once on an FP8 server, and stays opt-in until then.

Result of the first run (October 24, on 63f01f5): failed F1, F2, F3 and F6, for two bugs of ours. fp8pt and fp8pc
didn't start: vLLM's engine failed while capturing its CUDA graphs ("operation failed due to a previous error during
capture"). The FP8 check, run inside vLLM's MoE op at capture, indexed a tensor with a 0-d GPU tensor (`rows[i]`),
which PyTorch turns into `.item()`, a host sync, and a host sync invalidates a capture. The bf16 baseline and the
eager candidate started (no FP8 check under capture in either). Looking for the same mistake everywhere: the MoE,
norm and routing checks had it in their PyTorch versions, which run only when their Triton kernel isn't used (on
GPUs the kernels pass their first-use comparison: the norm run's servers used the Triton kernel and started). The run showed
a second bug: on real vLLM the FP8 check never took the FP8 linear layers (`sdcguard_fp8_layers_checked` was 24,
the MoE layers alone; there are 48 FP8 linear layers), because vLLM's scaled_mm package exports the
ScaledMMLinearKernel class under the name of the module the plugin imported, and our stand-in for vLLM had the
module there; the plugin's line saying so was at INFO, which nothing in a vLLM server prints. Reported from the
eager candidate: 347,040 MoE input quantizations judged, no FP8 alert, NO LEAK SEEN; REJECT on quality (mean KL
3.43e-2) and on consistency (with its prefix read from the cache, a token the cold answer was sure of changed at
position 13, largest gap 0.337 nats; the cache hits were confirmed); throughput -41.9% against the compiled bf16
baseline.

Fixed since: every 0-d index in the checks' PyTorch versions reads through index_select (no host read), and
`tests/test_serve_no_sync.py` runs each check's PyTorch version on fake tensors, where any host read raises (it fails
on the old code); the plugin imports the scaled-mm module itself, says so at WARNING when it can't, and the stand-in
exports the class as vLLM does (the old plugin fails it); sdcguard's own log lines now show in vLLM's output
(SDCGUARD_LOG_LEVEL), and the layer summary gives the number of FP8 layers checked. An independent review of these
fixes, before the rerun, found two more of ours, both fixed and CPU-tested: with the linear layers now registered,
the 72 FP8 layers outgrow the check's first 64-row state while they register, and the MoE layers registered before
that kept a copy of the old one (their checks would have counted and reported nothing, while the layer count read
72); and vLLM 0.32's GroupShape has no name, so a static per-tensor FP8 layer was listed as unsupported (our
stand-ins had a name; they now copy vLLM's). The criteria stand as written for the rerun. Reported in the rerun: the FP8 layers each server checks (72 expected on fp8pt: 48 linear, whose inputs
vLLM quantizes per token on CUTLASS, and 24 MoE layers, per tensor).

Amendment (October 25, before the rerun): limits only. Each server gets 15 minutes to start (qualify's default is
30), qualify 90 minutes and the repro folder 20, inside a function limit raised from 100 to 120 minutes, and a run
past either limit is still judged on what it saved, so a server that hangs costs at most 15 minutes and can't leave
the run without a verdict. The criteria are unchanged.

Result of the second run (October 25, on 64c9d02): not completed, for a mistake of ours in its limits and in
qualify. All five servers started; four were measured in full and the fifth, fp8pc, all but its speed. qualify would
have needed about 100 to 105 minutes (the fp8pt and eager servers took about 20 and 32 minutes each, their speed
loads alone about 15 and 18) and was stopped at its 90-minute limit during fp8pc's speed load. It wrote its record
only at the end, so there was no QUALIFY.json, and the run's judge, which reads only that file, reported every
server as not started (F1 asks about the baseline, fp8pt and eager) and failed F1, F2, F3, F4 and F6. That verdict
is wrong about the servers, and the run isn't counted as judged on these criteria; the amendment above ("still
judged on what it saved") didn't hold, because nothing summarizing was saved. What the files each server saved show,
read after the run (not the judge's verdict): on fp8pt, the isolation check found a LEAK (27 of 156 probe answers
changed with the request before them); the FP8 check raised 1,904 alerts there, every one on an MoE layer's expert
inputs (all 24 layers) with the finding that the per-tensor scale was taken from rows outside the batch, all in
steps whose batch had CUDA-graph padding (3 tokens padded to 4, 5 or 7 to 8). On eager: NO LEAK SEEN and no alert,
with all 72 FP8 layers checked (693,984 quantizations judged; 695,520 on fp8pt). The bf16 baseline had no alert.
fp8pc: NO LEAK SEEN. Quality and consistency failed on every FP8 server, the remedies too: quality FAIL against the
bf16 baseline (mean KL 3.88e-2 on fp8pt, 3.71e-2 on eager, 2.99e-2 on fp8pc, beyond the 5.4e-3 the policy allows)
and INCONSISTENT on all three, so these come with FP8 on this model, not with the leak, and both remedies would be
REJECT under this policy too. Neither fp8pt's decision nor its repro folder was produced. The requests the alerts
named are wrong: a per-tensor scale belongs to every row of the step, but each alert named one row, and for an MoE
layer's second quantization (8 rows per token) the check kept an expanded row, which the plugin compared with the
step's token count, so 1,523 alerts called a real row padding and 381 named one request instead of every request in
the step (239 of them by a token the row wasn't in). Fixed since: the record keeps the token (the row divided by the
rows per token), and a per-tensor scale names every request in its step (CPU-tested); qualify writes RUNS.json and
QUALIFY.json after each server (complete: false, with the decisions of the candidates measured so far; written
whole, so a stop mid-write can't leave half a file), so a qualify cut short keeps its record (CPU-tested); the run
gives qualify 150 minutes and the repro folder 25, inside a function limit of 190 minutes (about 110 to 130 minutes
and $2.60 to $3.10 expected, at most about $4.50 at requested resources), and its judge says which servers qualify
didn't finish.

Amendment (October 25, before the rerun): limits only, as above; the criteria are unchanged and the rerun is judged
on all six.

### Every check inside a captured CUDA graph (written October 24, before any run)

A check that is right on the CPU can still fail where vLLM runs it: inside a CUDA graph capture, where a host read
invalidates the capture, and in replays, where a value baked in at capture instead of read from a buffer would make
the check judge the wrong step. `tests/test_gpu_graphs.py` captures each serving check (linear, FP8 quantization,
MoE, routing, norms; their PyTorch versions and their Triton kernels) in a CUDA graph and replays it.
`modal run scripts/modal_vllm.py --phase graphs` runs it, the fake-tensor test and the two GPU tests of the linear
check already in `tests/test_serve.py`, on one L4 with vLLM 0.32's PyTorch and Triton (about 5 to 15 minutes, $0.15
to $0.40).

Criteria (all on the L4):
- G1: every capture succeeds.
- G2: each replay leaves exactly the state (and the kept-row flag) an eager call leaves, clean and wrong.
- G3: a wrong value is found inside the replayed graph for every check, and a clean one isn't.
- G4: a step marked between replays is read at replay (no real rows, nothing judged; FP8: every row real, the right
  scale passes; MoE: a step that is another layer's turn leaves the captured layer inactive; norms: the step recorded
  is the one marked before the replay).
- G5: no test in tests/test_gpu_graphs.py or tests/test_serve_no_sync.py skipped (the phase fails when pytest reports
  one; SDCGUARD_GPU_TESTS=1 makes a missing GPU a failure).
Where a check's state goes through cuBLAS (the MoE recompute and the linear check's sums), G2 is read as the same
counts and kept-row flags exactly and the same score to within 0.1% (2% for the linear check), on the same side of
the line, because a replay may run another cuBLAS algorithm than the eager call.
Not claimed: other GPUs, or vLLM's own capture of these checks, which the server runs show.

Amendment (October 24, written after the graphs run on b1a560d was handed over and before any run that includes it):
`tests/test_gpu_graphs.py` now also captures the attention check (below), its PyTorch version and its Triton
kernels, and G1 to G5 apply to it in the next graphs run. G4 there: the next step's batch written into the same
tensors (queries, outputs, KV cache, query_start_loc, seq_lens, block table, slots), as vLLM writes them before a
replay, is read at replay (the replay leaves the state an eager call on that batch leaves), and a step outside a
model step, or another layer's turn, counts nothing.

First run (October 24, on 582f91c, one L4, about $0.05): did not pass. 57 tests passed, 6 failed, 1 skipped, and 1
error: the compiled-model test of `tests/test_serve.py` that failed (failure 3 below) left its guard holding the
cuBLAS setting, which the suite's teardown check reports. Every
capture and replay that reached its check passed: the FP8, norm and attention checks (PyTorch and Triton), the routing
and linear checks' PyTorch versions, and the fake-tensor test. The failures were outside the graphs: (1) the MoE tests
never reached the check, because the test's stand-in for vLLM's MoE kernel built its output on the CPU (a bug in the
test); (2) the routing kernel's first-use comparison required the Triton kernel and the PyTorch version to keep the
same one of the wrong sampled rows, but on a GPU the kernel's programs race to be the one kept (Triton's interpreter
runs them in order, so the CPU tests couldn't show it); (3) called directly by a test, the linear kernel's first-use
comparison computed its probe layers with PyTorch's default cuBLAS settings (bf16 partial sums allowed), unlike served
layers, and scored a clean probe layer 9; it then kept the Triton check off for the rest of the process, which failed
the two linear GPU tests of `tests/test_serve.py` too. In serving on the same L4 and software (the qualify run) that
comparison passed (112 layers on the Triton check). The skip: PyTorch 2.13 has no CUDA graph replay hook, so the
compiled-model test in `tests/test_serve.py` can't prove replays (it isn't in G5's files). Fixed since: the stand-in
computes on its inputs' device; the routing comparison checks each kept row against the row it names (one of the
sampled rows, as given); the linear and MoE comparisons hold fp32 partial sums in cuBLAS while they run. The criteria
are unchanged; the next graphs run decides them.

Second run (October 24, on fbfbdaf, one L4, about $0.08): passed. 62 tests passed and 2 skipped. Every test in
`tests/test_gpu_graphs.py` and `tests/test_serve_no_sync.py` ran and passed, so G1 to G5 are met for every check, the
attention check included. Of the two linear GPU tests of `tests/test_serve.py`, the Triton check's comparison passed;
the compiled-model test (both variants) ran and then skipped at its replay check, the 2 skips: PyTorch 2.13 has no CUDA
graph replay hook, so it can't prove its replays there (it isn't in G5's files). After this run
the kernels changed how tied scores update their state (every program at the top score updates, so the state no
longer depends on the order programs run in), so the next bundle runs this phase again first.

Third run (October 24, on 03e27ba, the tied-score change included, one L4, about $0.09): passed, with the same 62
tests passing and the same 2 skips (never among the required tests; the runner now lists them in its STATUS.json as
skipped_other rather than leaving them out). The three runs are in `results/graphs_oct24/`, in the order they ran:
`graphs_before_20261025_002436` (582f91c, failed), `graphs_before_20261025_021449` (fbfbdaf) and `graphs` (03e27ba).

### Attention checks in serving (written October 24, before any run)

The vLLM plugin can now check attention (`src/sdcguard/serve_attn.py`, SDCGUARD_ATTN=1, off by default until this run
says otherwise), on vLLM's FlashAttention and Triton backends. At each model step one attention layer (they take turns
by step number) recomputes 4 of its outputs, each one query head of one token: two tokens of the step and the newest
tokens of two requests (a decode's only token, so long-context decodes are sampled in batches that are mostly prefill),
drawn from the layer's own turn count so that a steady batch has every (token, head) pair checked in turn. The recompute
reads the query row, and the request's keys and values from the paged KV cache through its block-table row, using the
same metadata tensors the backend reads (so CUDA-graph padding and replays are seen as the backend sees them), computes
softmax(scale q.K) V in fp32, split among 32 programs per sample and combined, and compares each of the head's outputs
against a bound written from how both backends round (vLLM 0.32's source, FlashAttention 2 and 3; FlashAttention 4,
vLLM's choice on Blackwell, is listed as not checked): P rounded to the KV dtype before the P V product, the sum of P in
fp32, the output rounded once, partial outputs rounded (slack: vLLM 0.32 has no cascade attention and keeps split-KV
partials in fp32), and the fp32 errors of the scores, the exponential, the accumulation and the split-KV combine on
either side. A correct output scores at most 1; the line is 4. The sampled token's own KV write is checked too: the slot
vLLM wrote it to (slot_mapping) must be the slot its position's block names, and that slot must hold the key and value
the layer computed for it, bit for bit, for every KV head. The first sample over the line in each layer is kept, with,
once per read across layers, the keys and values it attended (up to 65,536 of them), and its alert rechecks it in
float64. Two early-exit Triton kernels per layer call do the work: inside a CUDA graph every layer launches them and the
step's layer is chosen on the GPU; in eager calls during a step only that layer's call launches them. A model step is
one call of vLLM's execute_model with scheduled tokens; vLLM's dummy batches (memory profiling, warm-up, a data-parallel
rank with nothing scheduled), which its V2 runner also makes through execute_model, are not steps: they aren't counted
and the check stays off in them.

On the CPU (`tests/test_serve_attn.py`, `tests/_serve_attn_compare.py`): against a model of the backends' rounding
(tiles of 16 to 128 keys, 1 to 5 splits, partial outputs rounded or not, the sum of P taken before or after rounding;
sharp and diffuse attention, values of one sign, cancelling, large and tiny, up to 4,000 keys, fp16 P values in its
subnormal range), the worst score was 0.46 in bf16 and 0.63 in fp16; the Triton kernels and the PyTorch version gave the
same counts, scores and locations in 69 cases in Triton's interpreter, each keeping a sample when it alarmed, and on an
alarm the kept keys and values were the attended ones and the float64 recheck confirmed the wrong output; the kernels
compile for the A100, L4 and H100 with Triton 3.8.0 and 3.7.1 (head sizes 64, 80, 128 and 256); wrong outputs, attention
over another request's keys, attention a key short where that key carries the weight, NaN outputs, writes with the wrong
content, writes to another slot and block ids outside the cache were all caught and located; CUDA-graph padding was
never sampled. In bf16 the bound comes to about 1.6% of the attended values' size for ordinary logits (about 2.2% where
scale times the sum of |q k| is 200), so an output must be off by more than about 6% of it (9% there) to be caught for
sure: the check is for gross errors (a faulty unit, other keys than the metadata names, a length or schedule gone stale
inside the kernel, non-finite values), not small ones. Not seen: metadata that is itself wrong (the check reads the
same), and KV-cache contents that changed after their write (the kernel and the recompute read the same values).

`experiments/vllm_attn_check.py` (`modal run scripts/modal_vllm.py --phase attn`, vLLM 0.32.0, Qwen2.5-1.5B-Instruct
in bf16, 28 attention layers, max model length 4096). On one H100 (FlashAttention 3), one after another: plain
(SDCGUARD=0, FLASH_ATTN), guarded (SDCGUARD_REQUIRE=1, SDCGUARD_ATTN=1, compiled with CUDA graphs, prefix caching on as
by vLLM's default), noattn (guarded with SDCGUARD_ATTN=0), eager (guarded, --enforce-eager), tplain and tguarded (plain
and guarded on vLLM's Triton attention, TRITON_ATTN). On one L4 (FlashAttention 2), at the same time: plain and
guarded. Workload: 64 greedy completions of 32 tokens, and 8 long prompts (1,378 to 3,295 tokens, all starting with
the same 1,024 tokens, so the later ones hit the prefix cache) answered with 64 tokens, on each plain and guarded
server; the speed load (sdcguard.perf: 128 prompts, 128 tokens, 3 rounds, 16 and 256 requests at once) on plain,
guarded and noattn. Faults can't be put into the attention kernel inside captured CUDA graphs from Python, so the
compiled servers run clean and eager runs the trials: each makes one layer's attention wrong before the check sees
it, then sends rounds of 4 requests at once until an alert names the layer, or 16 rounds: every output at one
dimension made 2x + 1 ("value"); one query head's outputs with an exponent bit flipped ("head"); the backend computing
with the first two requests' block tables swapped ("other"); the backend computing with every seq_len one short
("short"); a bit of one key dimension flipped in the cache after every write of the layer ("kv"). About 45 to 75
minutes on the H100 and 20 to 35 on the L4, roughly $4 to $7 in all.

Criteria:
- A1: every server starts; guarded, eager and tguarded (and the L4's guarded) check all 28 attention layers with the
  Triton kernels (they passed their first-use comparison on the GPU), on the backend asked for (FlashAttentionImpl;
  TritonAttentionImpl for tguarded).
- A2: each guarded server's 64 completions and 8 long completions are identical to its plain server's (guarded to
  plain and tguarded to tplain on the H100, guarded to plain on the L4).
- A3: no alert with no fault on any guarded server (noattn included); on those checking attention, the worst clean
  score read back is at most 1 (the bound: a higher one means the bound's model of the backend is wrong, whatever the
  line); no wrong write, write to another slot, block id outside the cache or sample not judged; samples and writes
  were checked; no call left unchecked for a reason.
- A4: the "value", "head", "other" and "kv" faults are each caught by an alert naming the layer with the matching
  finding (output_wrong_or_misread; kv_write_wrong for "kv") within 16 rounds, and no alert with a finite score names
  another layer during the trials.
- A5: among the alerts of the "value", "head" and "other" trials, at least one each is rechecked in float64 as
  output_wrong from the keys and values the check kept.
- A6: on the compiled servers (guarded and tguarded on the H100, guarded on the L4), the calls checked are at least 98%
  of the model steps with scheduled tokens (one layer a step), which holds only if the check runs inside the CUDA
  graphs vLLM replays for decode steps.

Reported, not criteria: the "short" trial (how much a query's newest key moves its output depends on the attention,
not on the check), rounds until each catch, worst clean scores, samples and keys recomputed, and the cost (output
tokens per second with and without attention checks, at 16 and 256 requests at once). Decided now: if the attention
check passes and costs at most 3% at 16 at once, it becomes on by default; if it costs more, it stays opt-in. Not
claimed whatever the outcome: other models or backends (FlashInfer, MLA and the rest), FP8 KV caches, sliding windows,
attention sinks, soft caps, small errors, and the KV cache's state beyond the sampled tokens' own writes.

Amendment (October 24, after the first graphs run and before any attention run): the plain servers (plain and
tplain on the H100, plain on the L4) run with PyTorch's bf16 and fp16 reduced-precision cuBLAS reductions off, the
setting the plugin serves with (`vllm_plugin._exact_reductions`, which changes rounding slightly, as noted at the top),
set in vLLM's worker by the test's worker extension (SDCGUARD_TEST_EXACT=1). Each server reports its setting, and A2
also requires both servers of a pair to have it. Why: in the graphs run, the probe computed with PyTorch's default
setting on an L4 scored a clean layer 9, where served layers (the plugin's setting) are quiet, so a plain server on the
default could answer differently for a reason that has nothing to do with the checks. A2 now says that the checks
themselves change no answer. Whether the plugin's setting changes answers against default vLLM on an L4 isn't tested
here (on H100s earlier runs found no change: N2, M2).

Results (October 24, on 03e27ba; `results/attn_oct24/`; about $1.30 by the runner's estimate): every criterion met, on
the H100 and on the L4.
- A1: every server started. guarded, eager and tguarded on the H100, and guarded on the L4, registered all 28
  attention layers with the Triton kernels (which passed their first-use comparison on each GPU), on the backend asked
  for (FlashAttentionImpl; TritonAttentionImpl for tguarded). The compiled servers checked all 28 in their clean load;
  eager, which ran 8 clean steps before its trials (one layer a step), ran the check on 8 of them there, one call of
  which was read back before the trials began.
- A2: the 64 completions and the 8 long ones were identical between plain and guarded on both GPUs, and between
  tplain and tguarded; every server, plain ones included, reported cuBLAS partial sums kept in fp32 (the amendment
  above), so this compares answers with the checks and without them at the same cuBLAS setting, not with vLLM's
  default setting.
- A3: no alert on any server without a fault. Worst clean scores: 0.42 (guarded, H100), 0.38 (tguarded), 0.40
  (guarded, L4), against the bound of 1, and 0.23 on eager, from the one clean call (4 samples) read back there. No
  wrong write, write to another slot, block id outside the cache or sample not judged, and no call left unchecked for
  a reason (none listed as not checked).
- A4: the "value", "head", "other" and "kv" faults were each caught by an alert naming their layer with the matching
  finding within 2 rounds (the "short" fault too, in 1), and no alert named another layer.
- A5: the alerts of the "value", "head" and "other" trials (and "short") were rechecked in float64 as output_wrong.
- A6: calls checked were 8,276 of 8,283 model steps with tokens (guarded, H100), 2,565 of 2,572 (tguarded) and
  8,184 of 8,191 (guarded, L4): the check ran on the steps vLLM replayed as whole CUDA graphs (decode) as well as on
  those it runs with attention outside the graphs (prefill and mixed steps). The 7 missing each time are
  calls the GPU had counted but not yet handed back to be judged when the stats were read (the coverage report,
  which adds those, has exactly one call a step).
Reported: at 16 requests at once the attention check cost 2.72% of the throughput of a server already checking every
linear layer (noattn: the same plugin with SDCGUARD_ATTN=0), median against median of three rounds: 4,020.5 output
tokens per second with it, 4,132.8 without. Round by round it was 0.45%, 2.72% and 3.45%, so the result sits near
the 3% line. That baseline was itself 32% slower than plain vLLM (6,051.5, the plain server run with the plugin's
cuBLAS setting, as for A2). The time the attention check added, about 6.8 microseconds of the server's time per
output token, would be about 4.1% of that plain server's (165), so on a lighter setup of this model (one linear layer
in eight, say) its share is likely above 3%. That wasn't measured, nor other models (a larger one takes longer per
token, so the same added time would be a smaller share), nor any cost on the L4 (no noattn server there). At 256 at
once, -0.73% (within the noise). The rule written before the run: if the check passes and "costs at most 3% at 16 at
once" (with and without attention checks), it becomes on by default;
the runner written with it, before the run (`experiments/vllm_attn_check.py`, COST_LIMIT), compares the medians. By
that rule the attention check is on by default from this commit; SDCGUARD_ATTN=0 turns it off. Not claimed, as
written: other models or backends, FP8 KV caches, sliding windows, attention sinks, soft caps, small errors, and the
KV cache's state beyond the sampled tokens' own writes.

Amendment (October 25, after this run): with attention checks on by default, every other runner that starts servers
with the plugin (`vllm_check.py`, `vllm_serve_check.py`, `vllm_moe_check.py`, `vllm_norm_check.py`,
`vllm_ready_attrib_check.py`, `vllm_qualify_check.py`, `vllm_change_decisions.py`, `vllm_live_catch.py`,
`vllm_sensitivity.py`, `machete_diag.py`, `demo_vllm.py` and the gate output check in `scripts/modal_vllm.py`) sets
SDCGUARD_ATTN=0, and `sdcguard qualify` keeps attention checks off unless asked (`--env SDCGUARD_ATTN=1`). Their
criteria and results were written without attention checks, so a rerun of any of them runs what it describes.

### Exact operator tests for screening GPUs (written October 24, before any run)

`sdcguard screen` runs a checked transformer, but checks bf16 and fp16 tensor-core matmuls only, within a noise model.
Defective GPUs in the field also compute wrong FP64 and FP32 results on their CUDA cores (in ByteDance's study of silent
corruptions in the wild, OSDI '26, those units failed more often than the bf16 tensor cores), and wrong values in
memory, and synthetic tests missed more than 60% of the faulty GPUs there. `src/sdcguard/screen_ops.py` (`python -m
sdcguard.screen_ops`) adds tests whose every result is known exactly, so one wrong value is a finding with no noise
model: matmuls in FP64, FP32 with TF32 off, TF32, BF16, FP16, FP8 (e4m3, with and without fast accumulation) and INT8 on
integer-valued inputs small enough that every product and partial sum is exact in that arithmetic in any order of
additions (BF16 and FP16 rounded once to their output, with PyTorch's reduced-precision split-K reductions off), each
also scaled by powers of two across much of the exponent range and transposed, against float64 references computed on
the CPU; x * y + z on the CUDA cores in FP32 and FP64 against the same in int64; exp, sin, tanh, rsqrt and log of the
same values laid out in two orders (other SMs and lanes compute each), which must agree bit for bit; and a share of free
memory filled with an address-dependent pattern and its complement and read back after every chunk is written. A wrong
value is reported with its test, round, input set, scaling, element, value and expected value, with the GPU's identity
and telemetry (UUID, serial, PCI, driver, VBIOS, ECC mode and counters, remapped rows, clocks, temperature, power)
before and after. On the CPU (`tests/test_screen_ops.py`): the ranges keep every product and sum exact, every matmul set
comes out exact scaled and transposed, a clean run passes, and one value made wrong in software in each test is found,
located, and the only one found.

`experiments/screen_ops_check.py` (`modal run scripts/modal_vllm.py --phase screen_ops`, one H100 and one L4 at once,
about 10 minutes each, under $1.30 in all): a clean run of every test for 5 minutes, then a short run of each test
with one value made wrong in software.

Criteria (on each GPU):
- S1: every test runs (the H100 and the L4, Ada, have FP8, TF32 and INT8, so none is skipped), each checks values in at
  least one round, and the clean run finds no wrong value.
- S2: with one value made wrong in a test, that test finds exactly one wrong value, at the element made wrong
  (memory: at the changed address).

Reported: values checked, rounds, telemetry. What this is for and isn't: Modal screens its GPUs, and a defective one
is about one in a thousand (Meta, July 2025), so these runs check the tests themselves (no wrong value on healthy
GPUs, a wrong value always found); finding a defective GPU means screening many, which needs an account on a GPU
marketplace (consumer cards on Vast.ai or Salad cost cents an hour, so many can be screened), with criteria of its
own before it runs. Not claimed: that these tests find every defective GPU, or any particular fault rate.

Results (October 24, on 03e27ba; `results/screen_ops_oct24/`; about $0.56): both criteria met on both GPUs.
- S1: all eleven tests ran on the H100 and on the L4 (none skipped), and the clean runs found no wrong value:
  336,550,952,960 values checked on the H100 in 3,660 rounds, 100,696,850,432 on the L4 in 1,098. Telemetry: no ECC
  error before or after on either GPU; the L4 reported one memory row remapped at some earlier time after correctable
  errors (none pending, no remap failure), which is how its memory handles those, not a fault these tests found.
- S2: in each test, on each GPU, one value made wrong in software was found, the only one, where it was made wrong.

### An FP8 KV-cache decode bug through qualify (written October 24, before any run)

vLLM issue #54035, open on the current release: with an FP8 KV cache on Hopper, FlashAttention 3 tiles decode and
prefill differently (kBlockN 96 against 192 for head size 128), and from 97 tokens of KV on, a token's
log-probability when decoded differs from the same token's when the text is scored in one prefill (the reporter, with
VLLM_BATCH_INVARIANT=1 and eager mode: up to 0.5 nats on an H100 with Qwen3-4B, none with a bf16 cache; a proposed
fix is open in vLLM's flash-attention fork). `gate model-check` scores prefill only, so it can't see this, and a
bf16-vs-FP8 comparison differs anyway. `sdcguard consistency`'s decode-vs-prefill probe (4 chat prompts, 160 greedy
tokens each, so decode runs past 128 tokens of KV) compares one server's two paths; qualify judges a candidate's
largest gap against the baseline's own (INCONSISTENT above 4 times the baseline's and at least 0.1 nats).

`experiments/vllm_fp8kv_check.py` (`modal run scripts/modal_vllm.py --phase fp8kv`, one H100, vLLM 0.32.0, about
35 to 50 minutes and $2.80 to $4): one `sdcguard qualify` of Qwen/Qwen3-4B in bf16 with vLLM's defaults (CUDA
graphs, prefix caching on, no batch-invariant mode: the setting a team would run), the plugin on, a policy that
accepts a WARN: the baseline (bf16 KV cache) and the candidates fp8kv (--kv-cache-dtype fp8) and fp8kv_fi (the same
on --attention-backend FLASHINFER).

Criteria:
- K1: the baseline and fp8kv start.
- K2: the baseline's own consistency is CONSISTENT (its decode-vs-prefill gaps and its cached answers within the
  lines), so a candidate can be judged against it.
- K3: fp8kv's consistency is INCONSISTENT and its decision REJECT.

Reported, not criteria: every gap, the KV length where each first passes 0.5 nats and where it peaks, fp8kv_fi's
verdicts (whether FlashInfer's FP8 path shows the same), the prefix-cache probe on each, quality and throughput.
Without batch-invariant mode, the baseline's own decode-vs-prefill gaps include batch-shape effects; if they are
large enough that K2 fails, the probe can't judge this bug in vLLM's default setting, and that is the finding. Not
claimed whatever the outcome: other models, head sizes, GPUs or FP8 attention paths.

Result (October 24, on 63f01f5): K1, K2 and K3 met as written, but not by the probe they were written for. fp8kv was
REJECT, INCONSISTENT and OUTSIDE on quality (mean KL 1.01e-2, top-1 agreement 98.97%). The INCONSISTENT came from
the prefix-cache probe, narrowly: in one of its cases a token the cold answer was sure of changed when the prefix
came from the cache (the reference token's probability fell by 0.5145, the line is 0.5; the two cold answers were
identical and the cache hits confirmed). The decode-vs-prefill judgment written for this run, against the baseline's
own gaps, did not call it: the FP8 cache's largest gap per prompt was 0.297, 0.533, 0.399 and 0.278 nats (at 99,
168, 129 and 134 tokens of KV), the baseline's 0.118, 0.189, 0.102 and 0.117, so the largest was 2.8 times the
baseline's, under the 4-times line. (On its own, without a baseline, the probe calls any gap over 0.5 nats
INCONSISTENT, and the FP8 server's passed 0.5 at 168 tokens of KV; the line written for this run is the
baseline-relative one, because without batch-invariant mode the baseline has gaps of its own.)
Every FP8 peak came after 96 tokens of KV, as the issue describes, and its mean gaps were 2 to 4 times the
baseline's; that is consistent with #54035, not a demonstration of it. fp8kv_fi didn't start (FlashInfer isn't
installed in the image: reported only). Throughput -1.0%. What this run shows: qualify rejected an FP8 KV cache on
an H100 with reasons a reader can check; it doesn't show the decode-vs-prefill probe catching this bug in vLLM's
default setting. Since then the probe keeps every position's gap and reports gaps up to 96 tokens of KV and past
them separately, for a future run to judge on (CPU-tested).

Amendment (October 24, after this run, before any other run under these rules): three evidence rules of
`sdcguard consistency`, so that a verdict never rests on something not measured. The prefix-cache probe calls a case
INCONSISTENT, and the probe CONSISTENT, only where vLLM's counter confirmed the cache hit (otherwise INCONCLUSIVE); a
decode probe with any position it couldn't measure is INCONCLUSIVE; and in qualify, a baseline whose decode probe is
missing or incomplete sets no line, so the candidate's decode gaps are INCONCLUSIVE instead of judged by the
standalone 0.5-nat rule. None changes a verdict of the runs so far: in every one, every case's cache hit was
confirmed and no position was missing.

### Scheduled weight scan in serving (written October 25, before any run)

The bit-flip run above found that the arithmetic checks name a flipped weight bit only when it changes the weight
enough to move an output block past their line: all 5 flips of bit 14, 1 of 5 for each of bits 8, 9 and 10, none of
the others. The vLLM plugin can now also compare the weights themselves with a reference fingerprint
(`src/sdcguard/serve_weights.py`, SDCGUARD_WEIGHT_SCAN=1, off by default until this run says otherwise): at each read
of the results, the next layers in turn are fingerprinted on the GPU (two int64 sums of the raw bits: every value,
and each 4,096-value block's sum weighted by its position; a single bit flip always changes the first) until about
64 MB of their tensors have been compared, and the comparison is read back at a later read without waiting for the
GPU. The reference is taken when a layer is registered at load (or refreshed); until a layer's first comparison it
is provisional, and taken again if its tensors have moved (vLLM 0.32 processes reloaded weights in temporary tensors
and copies them back). After that, tensors that move, or that share their memory with another layer's (vLLM's CPU
offloading), aren't compared and are listed. It covers the checked layers (weight and bias; a quantized layer's
packed weights, scales, zero points and bias) and the bf16 and fp16 linear layers SDCGUARD_SHARE leaves out of the
arithmetic checks; quantized layers SDCGUARD_SHARE leaves out and layers the arithmetic check can't take are listed as
not scanned. A changed layer gets one alert (`serve_weights`, verdict `weights_changed`, which `sdcguard decide` reads
as a weight change) until refresh(). Tested so far on a CPU only (`tests/test_serve_weights.py`,
`tests/test_vllm_weight_scan.py`).

`experiments/vllm_weight_scan_check.py` (`modal run scripts/modal_vllm.py --phase weight_scan`, one H100, vLLM
0.32.0, Qwen2.5-1.5B-Instruct in bf16, 112 linear layers, SDCGUARD_SHARE=0.125 so 15 are checked and 97 are in the
scan alone, attention checks off; about 25 to 35 minutes and $2 to $2.80): two servers, noscan (the scan off) and
scan (on, 64 MB a read), each answering 64 prompts greedily (32 tokens). On scan, the speed load (sdcguard.perf: 128
prompts, 128 tokens) one round at a time with the scan on and paused in turn on the same server (one round left out
first, then on, off, off, on, on, off, off, on: 4 rounds each, at 16 and at 256 requests at once), so the cost isn't
a difference between two server starts; each pause and resume is confirmed by the worker, and the cost counts as
measured only if every one took and all 4 rounds each way came back. Then 48 trials: one bit of one weight flipped
through /collective_rpc (bits 0 to 15, 3 trials each, at a random layer, row and column, seed 0) and the prefix cache
emptied, rounds of 4 requests at once until a scan alert names the layer or 120 s have passed since just before the
flip, the bit flipped back, the layer's fingerprint taken again and the prefix cache emptied, so each trial starts
from a clean model. The trials stop after 3 misses in a row or when the run's 38 minutes are nearly up, what was
measured is written after each phase and each trial, and a server that fails partway is judged on what it measured
(the error is recorded and the run doesn't pass).

Criteria:
- W1: both servers start, and the scan compares all 112 linear layers: none listed as not scanned, and every one
  compared before the trials.
- W2: no alert on either server with no fault.
- W3: all 48 trials run, every flip happens and is named by a scan alert on its layer within 120 s (from just before
  the flip to the alert's time), no scan alert names a layer with no flip, every flip is undone, and every flipped
  layer's fingerprint is taken again.
- W4: the scan server's 64 completions are identical to noscan's.

Reported, not criteria: seconds from each flip to its alert, passes over the model and how long one took, which of
the flipped layers were in the scan alone, the arithmetic checks' alerts on the flipped layers (also counted up to
the next trial's flip, since the guard reads their results a read late), alerts that came after 120 s, how often the
prefix cache couldn't be emptied, and the cost (output tokens per second with the scan on against paused, medians of
the 4 rounds each way, at 16 and 256 at once). Decided now: if W1 to W4 are met and the scan's measured cost is at
most 1% at 16 at once, it becomes on by default; otherwise (including a cost not measured) it stays opt-in. Not
claimed whatever the outcome: other models or GPUs; a change undone before its layer's turn; changes that cancel
each other within one 4,096-value block (two opposite flips of one bit there); wrong arithmetic with the weights
right (the arithmetic checks' job); an idle server (the scan moves on only while the server runs model steps);
quantized layers SDCGUARD_SHARE leaves out, layers the arithmetic check can't take, embeddings, the LM head and the
KV cache (not scanned).

### The upgrade gate on cuBLAS issue 6862950

`sdcguard gate` keeps a matmul as a case (inputs, the exact call, the conditions it ran under) and runs it again in
fresh processes with cuBLASLt's API log on, so the kernel that ran is read from the library. Its verdicts are
written so a clean rerun is never called a fix (the module's docstring has the rules; `tests/test_gate.py` pins
them down on fakes). The run (`modal run scripts/modal_gate.py`, one H200 under MPS, PyTorch 2.14.1) is the first
time any of it runs on a GPU. Expected, from the October 19 and 20 runs and NVIDIA's notes:

1. Capturing a bf16 linear layer of 16 tokens x 2048 -> 1000 with bias at MPS 40% gives an incident (wrong in every
   run, algorithm 66 with a cluster of 4 or more CTAs in the log).
2. The same stack at 40% again: UNCHANGED, still wrong.
3. The same stack at 100%: INCONCLUSIVE, because the SMs visible and the MPS share differ, although every output is
   right.
4. The newest cuBLAS wheel (13.8.1.7) at 40%: UNCHANGED, either still wrong or avoided (another kernel ran). NVIDIA
   hasn't listed 6862950 as fixed, so FIXED here would be a finding to check, not a confirmation.
5. `gate pin` writes a TunableOp file forcing algorithm 66 with a 1x1x1 cluster, and `gate workaround` with it says
   WORKS: wrong in every run without it, right in 50 of 50 with it, and the log shows algorithm 66 with a 1x1x1
   cluster on every call.
6. The same TunableOp switches with a file that doesn't exist: DOESN'T WORK (the default kernel runs).

Any other outcome is reported as it came out. Steps 5 and 6 rest on reading PyTorch 2.14's TunableOp source, not on
a run; a pin PyTorch rejects shows up as INCONCLUSIVE in step 5, which would be a finding about the pin, not about
the bug.

**After the first run (October 22, written before the second).** The first run (code fingerprint 38c4e9be3126,
`results/gate_h200/h200_20261022_092329.json`) met 1, 2, 3, 4 and 6 as written: the capture was wrong in 20 of 20
runs with algorithm 66 and a 4x2x1 cluster in the log, and the newest cuBLAS wheel, 13.8.1.7, was still wrong in 20
of 20. It did not meet 5. `gate pin` found a configuration (algorithm 66, 1x1x1 cluster, right when run on its own)
and wrote it to pin.csv, but PyTorch puts the GPU's number in the file name (`TuningContext::SetFilename`), so it
looked for pin0.csv, found nothing, and the default kernel ran in all 50 runs, wrong as before. `gate workaround`
then said DOESN'T WORK where the paragraph above says INCONCLUSIVE, because it judged the outputs before the
kernel. Two changes, made after seeing that run: `gate pin` writes the file under the name PyTorch reads, and `gate
workaround` says INCONCLUSIVE whenever the workaround wasn't what ran (the TunableOp file PyTorch reads isn't there,
or the log shows another kernel than the expected one), and DOESN'T WORK only when it ran and the outputs are still
wrong. Expectations 1 to 5 for the second run stay as written; 6 now expects INCONCLUSIVE (the file isn't there),
and never WORKS.

**What the second run found (October 22): every expectation met.** Code fingerprint f36a0afc8410 (commit e17c55c),
`results/gate_h200/h200_20261022_110847.json`, one H200 under MPS, 45 seconds of GPU time.
1. The capture at 40% was wrong in 20 of 20 runs, every call on algorithm 66 with a 4x2x1 cluster.
2. The same stack at 40% again: UNCHANGED, wrong in 20 of 20 before and after.
3. At 100%: INCONCLUSIVE (52 against 132 SMs visible, MPS share 40 against 100), although all 20 runs were right,
   on the same kernel.
4. The newest cuBLAS wheel, 13.8.1.7: UNCHANGED, wrong in 20 of 20, on the same kernel. NVIDIA's newest release
   still has the bug.
5. `gate pin` wrote pin0.csv (the name PyTorch reads for pin.csv on GPU 0), and `gate workaround` with it said
   WORKS: wrong in 50 of 50 runs without it and 0 of 50 with it (a failure rate below 7.1% at 95%), and the log
   showed algorithm 66 with a 1x1x1 cluster on all 50 calls.
6. The same switches with a file that doesn't exist: INCONCLUSIVE, naming the file PyTorch looked for; the default
   kernel ran and was wrong in 50 of 50.
Every wrong output was one the kernel never wrote: the output is filled with NaN before each call, and in every
wrong run 7,808 of the 16,000 outputs (16 tokens x 1,000) were still NaN after it, the same number as were wrong.

### The gate's output check

`sdcguard gate model-ref` records greedy completions of 32 fixed prompts (64 tokens each) with the top 20
log-probabilities at every position; `sdcguard gate model-check` has another stack score those same tokens
(teacher forcing) and compares the distributions position by position: top-1 agreement, KL divergence over the union
of both top-20 lists, and the change in the recorded token's log-probability (`src/sdcguard/gate_model.py` has the
details and the thresholds, which are a starting point taken from published numbers, not calibrated per model). The
run (`modal run scripts/modal_vllm.py --phase outputs`, one H100, vLLM 0.31.0) records Qwen2.5-1.5B-Instruct in bf16,
then checks against it. Expected:

1. The same stack, with at most 4 sequences batched together (other batches, other kernels' shapes): PASS. Its
   numbers are the noise floor the next three are judged against.
2. The Triton attention backend instead of vLLM's default: PASS (a legitimate kernel change). A WARN here would
   mean the thresholds are too tight for this model, and is reported as such.
3. Qwen2.5-1.5B-Instruct-AWQ against the bf16 record: WARN or FAIL. Published KL for 4-bit quantization is about
   2.6e-3 to 1.2e-2 (Fireworks, Llama 3.1 8B), around the WARN line.
4. The bf16 model with `rms_norm_eps` set to 1.0 instead of 1e-6 (a configuration change that leaves every matmul
   correct): FAIL, with sdcguard's matmul checks on in the same run, checking every layer, and raising no alert.
   This is the case the output check exists for: the math is right, the model is wrong.

### The serving kernel

Variants of the Triton check (each a rewrite of the kernel's loop: one running key per element and a single
reduction after the loop, and with the four block sums in one reduction) are timed against the current one on every
linear-layer shape of Qwen2.5-1.5B and 7B, at 1 to 16,384 rows, with a few launch settings, on one H100. A variant
can replace the current kernel only if, on every shape and row count, the state it writes equals the current
kernel's bit for bit (scores, positions where the worst block is unique, counts) and its first-use comparison with
the plain check passes on that GPU in bf16 and fp16. Among those, the fastest by median time is ported. The ported
kernel then has to pass, on an H100 (`modal run scripts/modal_v050.py`), the comparison with the plain check over
`tests/_serve_fast_compare.py`'s cases (ALL AGREE) and the 0.4.0 serving capture criteria 6, 7 and 8 as written
above. Then the vLLM runs above are repeated on the released code, with the same criteria as the October 21 runs, and the cost of
checking every layer is reported as measured. The aim is about 10% of throughput on the 7B model with every layer
checked; it is an aim, not a criterion, and the number is published whatever it is.

**Changed before the timing run (October 22).** The variant timing above never ran. Looking first at what the
check's time is made of (a profile on one H100 earlier on October 22, of the kernel before its running-key
rewrite, on the same layer shapes) showed it isn't the loop's reductions: on Qwen2.5-7B's gate_up_proj at 256 rows
the kernel read 19 MB in 56 µs (about 0.35 TB/s of the H100's 3.35), the cuBLAS call for E cost 7 to 10 µs per
layer even at one row, and across a sweep of 48 launch settings the best still left the check at 57% of that
layer's own matmul. A loop rewrite can't close that gap, so a second design is timed against the current kernel
instead (`_serve_fast.check_v2`, two Triton kernels per layer): `_emm` computes E with Triton's dot (split over k
when there are few rows, so the parts matrix is read at full bandwidth) and the row norms from the same loads, and
`_pairs` scores one 128-wide block of y per tile row, 32 blocks per program, with one atomic per program and only
when it raises the layer's state. Its sums run in another order than cuBLAS's, so its scores can differ from the
current kernel's in the last bits and the bit-for-bit criterion above can't apply; it is replaced by the
criterion the current kernel itself had to meet, agreement with the plain PyTorch check. On one H100
(`modal run scripts/modal_serve_cost.py`), v2 becomes the default only if:
1. its first-use comparison passes in bf16 and fp16, and `tests/_serve_fast_compare.py` prints ALL AGREE with it;
2. on every linear-layer shape of Qwen2.5-1.5B and 7B at 1 to 16,384 rows, its scores agree with the plain check's
   on a clean output (within 0.05 + 2%), and on the same output with one value changed by 1% of the largest output
   and by 50 (the same verdict, and the same row and block when it alarms), with the same counts;
3. it costs less than the current kernel, summed over the four 7B layers, at every row count of the run.
Otherwise the current kernel stays as it is. The same run also times v2 with 16 and 64 blocks of y per `_pairs`
program instead of 32 and scores them the same way; if one of them also meets 2 and is cheaper than 32 summed over
the four 7B layers at most row counts of the run, it is the setting used. The script prints each criterion as met or
not met, from the saved numbers (`summarize()`, tested in `tests/test_serve_cost_runner.py`).

**What that run found (October 22): v2 met 1 and 2, not 3; the current kernel stays.** Code fingerprint
2414d31f36bb (commit 6bf498c), `results/serve_cost/h100_20261022_124331.json`, one H100, 3.1 minutes of GPU time.
Both kernels passed the first-use comparison in bf16 and fp16 and printed ALL AGREE over the comparison set, and
v2's scores agreed with the plain check on every shape and row count. But v2 cost about twice as much: summed over
the four 7B layers it added 105 µs at 1 row (v1: 58), 265 at 256 (v1: 133), 2,777 at 4,096 (v1: 1,604) and 10,359 at
16,384 (v1: 3,469). With 16 blocks per program it was a little cheaper than with 32, and with 64 dearer, which
points at the cause: each program reads its blocks of y and scores them in the same code, and the scoring's many
per-block values leave few programs resident on an SM, so few loads are in flight. One more design is timed the
same way, before any change to the default (`_serve_fast.check_v3`): cuBLAS for E as in v1, then `_sums`, which
only reads y (and, in the same launch, the rows of x) and writes each block's four sums, and `_score`, which scores
one (row, block) pair per lane from those sums. v2 is removed from the code. v3 launches one more kernel than v1,
which costs a few microseconds per layer, so at a handful of rows, where reading y takes almost nothing, it may not
be cheaper; it may then be used only from the row count on where it is. So v3 is held to criteria 1 and 2 as
written, and 3 becomes: there is a row count of 256 or fewer from which v3 is cheaper than v1, summed over the four
7B layers, at that and every larger row count of the run. If so, v3 is used from that row count on and v1 below it
(the batch size is known when the check is called, and each CUDA graph is captured for one batch size).
`modal run scripts/modal_serve_cost.py` measures this against v1, also reporting each kernel's own GPU time from
the profiler and the time a plain sum of y takes, so the cost can be placed, and tries 16 and 64 blocks per
`_sums` program as for v2. If v3 doesn't meet all three, v1 stays, and the release reports the cost of checking
every layer as measured, with checking a share of layers as the cheaper option it is. Either way, the ported kernel then goes through the H100 checks and
the vLLM runs described above.

**What the v3 run found (October 22): all three met; v3 is used from 256 rows on.** Code fingerprint 17d370a66db7
(commit 2592fa3), `results/serve_cost/h100_20261022_140222.json`, one H100, 2.7 minutes of GPU time, about $0.19.
Both kernels passed the first-use comparison in bf16 and fp16 and printed ALL AGREE over the comparison set (172
cases each), and v3's scores agreed with the plain check on every shape and row count. Summed over the four 7B
layers, v3 cost more than v1 below 256 rows (82 µs against 65 at 1 row, 83 against 77 at 64) and less from 256 on,
so it is used from 256 rows (`V3_FROM_ROWS`). With 64 blocks of y per `_sums` program it was cheaper than with 32 at
four of the seven row counts (16, 1,024, 4,096 and 16,384), which is "most" in the rule above, so 64 is the setting
(`SUMS_P`). The differences between 16, 32 and 64 are mostly a few percent, within what repeating a run moves (v1
itself measured 58 µs at 1 row in the v2 run and 65 in this one), except at 4,096 and 16,384 rows on 7B, where 64
was clearly cheaper; on the 1.5B model 64 was dearer than 32 from 1,024 rows on (236 µs against 224 at 4,096, 888
against 748 at 16,384). Time each check adds, summed over the four layers, with the settings now used:

| rows | 7B, v1 | 7B, v3 (64 blocks) | 1.5B, v1 | 1.5B, v3 (64 blocks) |
|---|---|---|---|---|
| 1 | 65 µs | 84 | 51 | 62 |
| 64 | 77 | 84 | 54 | 60 |
| 256 | 140 | 106 | 82 | 70 |
| 1,024 | 351 | 189 | 180 | 102 |
| 4,096 | 1,514 | 567 | 664 | 236 |
| 16,384 | 3,946 | 2,263 | 1,609 | 888 |

Against the four layers' own time this is still a lot: on 7B, v1 adds 38% at 1 row and 42% at 64, v3 adds 50% at
256, 29% at 1,024 and 23% at 4,096. The profiler shows where it goes. From about 1,024 rows on, `_sums` reads y at
close to the rate a plain sum of y does (7B gate_up_proj at 4,096 rows: 126 µs against 120), so that part has
little left to gain. Below 256 rows the time is mostly fixed costs per layer: on 7B, cuBLAS spends 7 to 8 µs of GPU time
on E at 1 row (4 to 7 on 1.5B), though the parts it reads (1.2 to 12.7 MB per layer) need 0.4 to 4 µs at the H100's
memory bandwidth, and each of the two or three kernels per layer takes at least 2 to 3 µs, plus the gap between
kernels. On down_proj at
1 to 64 rows, `_sums` takes about 11 µs because one program reads each row of x (18,944 values) alone. These timings
replay the same layer 20 times in a CUDA graph, so the parts and x can stay in the 50 MB L2 cache between calls; in
a model the other layers evict them, so E can only be slower there. The vLLM runs measure the cost in a model.

**One kernel per layer for few rows: v5 (written October 22, before any GPU run).** Below 256 rows, where serving
spends its decoding steps at modest load, the profile above leaves most of the check's time in fixed costs: cuBLAS's E
and two or three launches per layer. `_serve_fast.check_v5` does the whole check in one Triton kernel (`_fused`) and
calls no cuBLAS. Its programs split the layer by 8 blocks of y (16 columns of each part of E), by tiles of 16 or 64
rows, and the sum over k into parts (up to 32 ways for 16-row tiles, 8 for 64-row ones, about two programs per SM).
Each program computes its part of E with Triton's dot (fp32 accumulation, as cuBLAS), its part of the row norms of x
from the same loads, and the four sums of some rows of its blocks of y, and stores them; then it adds one to its
tile's counter, and the program that finds the count complete adds the parts up in a fixed order (so the result is
deterministic), resets the counter and scores the tile with the same code as `_score` (shared now as `_judge` and
`_report`). The counters are zero between launches. Each layer has its own, in one set per CUDA stream: launches on
a stream run one after another, and two streams never share a set. A graph keeps the set of the stream it was
captured on, so one layer's graphs captured on one stream must not be replayed on two streams at once (vLLM captures
and replays on one stream each); a count left over by such misuse would make that tile's later checks wrong. Counters
are never made while a CUDA graph is being captured (`warm()` reserves enough for every batch size v5 serves), and
if none are available that call runs v1 or v3 and is counted (`V5_FALLBACKS`). An independent review of the code
before the run found no fault in the kernel and three ways counters could have been shared or used before they were
zeroed: one set per stream for all layers, chosen by the capture stream for captured work; sets assigned without a
lock; a new buffer zeroed on one stream only. The last two were fixed before the run, and the first narrowed to the
case written above (each layer has its own counters now) and documented. On one CPU, in Triton's interpreter, v5 gave the plain check's answers on every case tried
(both tile heights, k split and not, partial tiles, a strided y), the same state on repeated runs, and left every
counter at zero. Compiled for an H100 (sm_90a, offline), its loads of x and the parts are pipelined and vectorized and
it uses 64 to 95 registers with no spills on the 7B shapes. Whether it is cheaper is for the GPU to say.

The run is `modal run scripts/modal_serve_cost.py` on one H100, on the code of this commit. It times v1, v3 (with the
settings now used) and v5 with each tile height on every linear-layer shape of Qwen2.5-1.5B and 7B at 1, 2, 4, 8, 16,
24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 1,024 and 4,096 rows, and reports each kernel's GPU time. One change in
how it times, made because v5 reads E's parts itself: the 20 calls in each CUDA graph go round several copies of the
layer, enough that its weights and parts (twice the 50 MB L2 cache) can't stay cached from one call to the next, as
in a model, where every other layer passes through between two calls of one. This applies to v1 and v3 in the same
run too, so the comparison is like for like, but the numbers are not comparable with the runs above. v5 is used only
if:
1. its first-use comparison passes in bf16 and fp16 and `tests/_serve_fast_compare.py` prints ALL AGREE with it;
2. with both tile heights, on every shape and row count of the run, its scores agree with the plain check's as v3's
   had to (clean within 0.05 + 2%; with one value changed by 1% of the largest output and by 50, the same verdict
   and the same row and block when it alarms; the same counts), no timed call fell back to v1 or v3, and every
   counter is back at zero at the end of the run;
3. with the tile heights chosen below, summed over the four 7B layers, at 1 row it costs at most 95% of what v1 costs.
Tile height: 64-row tiles from the smallest row count of the run from which they cost less than 16-row ones at that
and every larger row count, 16-row tiles below (`V5_T64_FROM_ROWS`). Then v5 is used from 1 row up to and including
the largest row count of the run up to which, at every row count, it costs at most 95% of the design used there now
(v1 below 256 rows, v3 from 256), summed over the four 7B layers; `V5_BELOW_ROWS` is one more than that row count.
Row counts between two measured ones keep the current design when v5 stops qualifying at the larger one. The 5% is
margin for the run to run variation seen above. If v5 is used, it runs by default only on the GPU generation it was
timed on (Hopper, compute capability 9: H100 and H200, `V5_MAJORS`); other GPUs keep v1 and v3. If 1, 2 or 3 is not met, v5 is not used and stays in the code only if
it can be fixed and timed again under these rules. The script prints each criterion and the choices, from the saved
numbers (`summarize()`, tested in `tests/test_serve_cost_runner.py`). The 1.5B model is reported, not judged. Either
way the vLLM runs described at the start of this section then measure the cost in a real model.

**What the v5 run found (October 22): 1 met, 2 and 3 not; v5 is removed.** Code fingerprint 43e38e8086ae (commit
faf352d), `results/serve_cost/h100_20261022_191009.json`, one H100, 4.9 minutes of GPU time, about $0.36.
1. Met. v1, v3 and v5 each passed the first-use comparison in bf16 and fp16 and printed ALL AGREE over the
   comparison set (172 cases each).
2. Not met as written, though not for v5's answers: its scores agreed with the plain check in every case (both tile
   heights, every shape and row count, clean and changed, 0 disagreements) and every counter was at zero at the end,
   but 125 timed calls fell back to v1 or v3. All were on layers timed with four or more copies (1.5B qkv_proj,
   o_proj and down_proj, 7B qkv_proj and o_proj) at 128 rows and up, mostly with 16-row tiles forced. The timing
   script's own fault: its warm-up called only the first three copies, so the later ones were first used inside a
   CUDA graph capture, needing more counters than the load-time reservation, which v5 then correctly declined to
   allocate.
3. Not met. At 1 row, summed over the four 7B layers, v5 added 56.2 µs and v1 54.9 (102%). It was dearer at
   every row count of the run: by 2% to 5% up to 16 rows, by 14% to 169% above.

By layer it was mixed. At 1 row on 7B v5 added 9.2 µs to qkv_proj (v1: 13.0), 8.1 to o_proj (13.0) and 14.0 to
down_proj (15.2), but 24.8 to gate_up_proj (13.6): there it read the 12.7 MB of parts at about 0.5 TB/s, where
cuBLAS's E reads the same in 6.7 µs, about 1.9 TB/s. On the 1.5B model it was dearer on every layer but down_proj
(qkv_proj 14.7 against 9.8, with 0.3 MB of parts), so each launch also carried a fixed time of about 8 to 14 µs, far
above the 2 to 3 µs of `_sums` or `_score` in the v3 run. The causes weren't measured further. v5 is removed from the
code (it is in commit faf352d); the scoring helpers it shared with v3 (`_judge`, `_report`) stay, and v1 and v3 passed
the GPU comparisons above with them.

The same run timed v1 and v3 at more row counts than the v3 run did, with the layer copies described above.
Summed over the four 7B layers, v3 cost less than v1 at 96 rows (78.4 µs against 80.6) and at every larger row
count of the run (128: 78.8 against 85.4; 192: 86.0 against 108.3; 224: 96.8 against 131.7; 256: 92.2 against
128.9), and more at 64 (75.2 against 70.5). The rule written for v3 (used from the smallest row count from which it
is cheaper at that and every larger one) gives 96 on these row counts, which the v3 run, with nothing between 64 and
256, could not show; `V3_FROM_ROWS` is now 96. On the 1.5B model v3 is slightly dearer at 96 and 128 rows (57.9
against 55.4, 58.3 against 57.5) and cheaper from 160. The numbers in this run were lower than in the v3 run, not
higher as the copies would suggest (v1 at 1 row on 7B: 54.9 µs against 64.6; v3 at 256: 92.2 against 105.7), so the
variation from run to run, 10 to 15%, is larger than the effect of the copies. Against the four 7B layers' own time
the check still adds a lot: 30% at 1 row, 38% at 64 (v1), 41% at 128, 43% at 256, 30% at 1,024 and 23% at 4,096
(v3). The vLLM runs measure what that comes to in a model, where the linear layers are only part of each step.

**Added to the vLLM runs before they run (October 22).** Each model's runs (`experiments/vllm_check.py`, every
role) also time 16 requests at a time, with the same 128 output tokens each, three timed runs and the median, right
after the 256-request timing and before the clean report, which therefore covers that traffic too. It is reported
with and without sdcguard, with whether its completions matched, and has no threshold: with 16 rows per decoding
step the check costs the most relative to the layers' own work (above: 30% to 34% of the 7B layers' time at 1 to 16
rows), so it is the number to give for light traffic next to the 256-request one.

**The serving checks of 0.5.0 on an H100 (October 22): ALL AGREE, and 6, 7 and 8 met.** Code fingerprint
78c664d5b914 (commit d9b8bad, v1 below 96 rows and v3 from there on), `results/v050/h100_20261022_203828.json`,
`modal run scripts/modal_v050.py`, 2.0 minutes, about $0.14. The comparison set printed ALL AGREE (172 cases), and
the 0.4.0 capture criteria were met as written: the one-time fault (replay 66 of 200, inside a CUDA graph) was kept
as it alarmed and rechecked as wrong, the checker fault (replay 133) was rechecked as right (not blamed on the
GPU); capturing cost the clean check 3.3% (76.8 µs to 79.3 per replay of the four layers); and under contention
every alert came with its row while slots were free (14 alerts, 12 rows kept, the 2 others counted as dropped once
the 4 slots were full, as designed).

**The vLLM runs on the 0.5.0 code (October 22): every criterion met on both models.** Code fingerprint 78c664d5b914
(commit d9b8bad), `results/vllm_oct22/`, `modal run scripts/modal_vllm.py --phase vllm`, 12.2 minutes, about $0.99.
In every guarded run (every layer, and 1 in 8) every registered layer executed checks after warm-up, no checksum
went unjudged, clean traffic raised no alert (highest score 2.82 on 7B, 2.15 on 1.5B; the alarm line is 6), the
64 compared completions and the 16 light-load ones were identical to plain vLLM's, and the flipped weight bit was
named by both the serving check and the weight checksums, with the alarming row kept and rechecked as "weights
changed". Throughput, vLLM 0.31's default compiled mode, median of three timed runs (each light-load run's three
times were within 1.3% of each other):

| model, layers checked | 256 requests: tokens/s, plain / with checks | each request longer by | 16 requests: tokens/s | each request longer by |
|---|---|---|---|---|
| Qwen2.5-7B, every layer (112) | 22,755 / 18,293 | 24.4% | 2,456 / 1,928 | 27.4% |
| Qwen2.5-7B, 1 in 8 (15) | 22,755 / 22,174 | 2.6% | 2,456 / 2,390 | 2.8% |
| Qwen2.5-1.5B, every layer (112) | 37,424 / 33,385 | 12.1% | 6,439 / 4,313 | 49.3% |
| Qwen2.5-1.5B, 1 in 8 (15) | 37,424 / 36,175 | 3.5% | 6,439 / 5,988 | 7.5% |

On 7B, checking every layer cost less than on October 20 (requests 29.7% longer then, 24.4% now; throughput 22.9%
lower then, 19.6% now), which is what v3 from 96 rows predicts. The timings above predict these numbers: 28 times the
four layers' added time per step at 16 rows (v1, 60.4 µs) is 1.69 ms, and decoding steps in the light-load run took
1.8 ms longer (8.3 against 6.5 ms per token); at 256 rows (v3, 92.2 µs) it is 2.58 ms, and steps took 2.74 ms longer.
With 16 requests on the 1.5B model, whose layers do little work per token, checking every layer is expensive (49%);
1 in 8 layers cost 2.6% to 7.5% on both models at both loads.

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
outputs. The three runs must match and GPUs of the same model in one machine must match (SUSPECT otherwise).
A result that differs from at least 3 agreeing GPUs of the identical setup (from at least 2 sources) on the
board gets a note; board data is posted by anyone, so since 0.4.0 it never changes a result. Datasheet peaks are dense numbers from nvidia.com, listed with their pages in `gpuinfo.py`.

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
