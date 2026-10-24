# Where each part of sdcguard stands

As of October 24, 2026. Four kinds of status, kept apart:

- **GPU-validated**: met criteria written before the run, on a real GPU. The run is in `results/` and its
  criteria and outcome are in [METHOD.md](METHOD.md).
- **Ran on a GPU, limits**: ran on real GPUs with mixed results, or with a scope narrower than the feature.
- **CPU-tested**: built, with tests that pass on CPU stand-ins, and never run on a GPU. A GPU run may be written
  in METHOD.md and not run yet.
- **Planned**: not built.

Faults are named for what they were: injected in software, injected in tensor-core instructions (NVBit), or a real
library bug (cuBLAS 6862950). No naturally defective GPU has been tested. Nobody uses sdcguard in production yet.

## Training

| Part | Status | Evidence |
|---|---|---|
| Checksums of every `nn.Linear` (forward, dX, dW), bf16 noise model, alarm at 6 | GPU-validated | 0 false alarms in 24,580,560 checks over 5.14 GPU-hours (H100, Oct 14-15); 4-GPU demo; cost 1.7% (GPT-2 small) and 1.3% (0.9B) checking every 20th step (H100, Oct 18) |
| Sensitivity of that check | GPU-validated, with known limits | one wrong value in an output: half flagged at about 29% of a typical output in its block, 0 of 594 at 10% (H100, A100, L4, Oct 23); harm study, faults moving a whole block: 99% caught at 10% or more, 70% at 1-10%, none under 1% (Oct 14-15); NVBit on an L4: an exponent flip and a 10% mantissa flip caught 30 of 30, a 1% flip 0 of 30 (Oct 18). Not measured for the vLLM plugin |
| fp32-accumulator check (`precise="fused"`) | GPU-validated for training | 4,568,751 clean checks on an H100 (worst 2.67) and 613,404 on an L4 (worst 3.32), no alarm (Oct 23); as sensitive as precise mode in the Oct 20 and 21 runs; precise mode (`precise=True`) flagged half of one-value errors at 0.71% to 0.92% of a typical output, the default check at about 29% (Oct 23; 50% points, not guarantees) |
| Recompute check (activation checkpointing) | Ran on a GPU, limits | 32 of 32 injected faults caught, 0 mismatches in 320 clean compares; the gradient criterion wasn't met, and a control run showed PyTorch's own nondeterminism is about that size (Oct 20) |
| Attention checks | Ran on a GPU, limits | calibration criteria met in bf16 on an L4, H100 and B200 (Oct 20); no fault study |
| `torch.compile` fullgraph, MoE grouped GEMMs, tensor parallel over NCCL | GPU-validated | the 0.4.0 runs, every criterion met in the last two (`results/v040/`, Oct 21) |
| DDP | GPU-validated | the 4-GPU demo (`results/demo_4gpu/`) |
| FSDP2, the Hugging Face Trainer callback | CPU-tested | gloo with 2 ranks (Oct 20) |

## Serving (vLLM)

| Part | Status | Evidence |
|---|---|---|
| The plugin in vLLM 0.31 and 0.32, bf16, one layer in eight or every layer | GPU-validated | 64 of 64 completions the same as plain vLLM, no clean alert (highest 2.82), a flipped weight bit caught and named (Qwen2.5 1.5B and 7B, H100, Oct 20 to 23); real `vllm serve` over HTTP on one and two H100s (Oct 23) |
| INT4 layers (Marlin, Machete) | GPU-validated | all criteria met on five INT4 runs, the 7B AWQ on Machete included (H100, Oct 23) and on Marlin on an L4 (Oct 23) |
| Fused MoE experts (SDCGUARD_MOE=1: one layer a step recomputes 2 rows) and their routing (one router a step) | CPU-tested | the first GPU run (Oct 24) didn't start its checked servers: SDCGUARD_REQUIRE didn't know the MoE check (fixed); rerun with 2 routing trials written ("MoE checks in serving", amendment) |
| FP8 layers' input quantization (SDCGUARD_FP8=1) | CPU-tested | GPU run on vLLM #62124 written Oct 24 ("An FP8 MoE bug from detection to remedy") |
| FP8 matmul arithmetic | Planned | measured, not built: Hopper's FP8 fast accumulation moves block sums up to 29.7 bf16-noise units on a K = 8960 layer, 6.4 without it (H100, Oct 24); a check needs a K-dependent term |
| RMSNorm checks (SDCGUARD_NORMS=1) | Ran on a GPU, limits | 32 of 32 weight flips of bits 2 to 15 caught and named, no clean alert, 5 of 5 exact namings, Qwen3 q/k norms caught; but 5 of 64 compiled Qwen2.5 completions differed from plain vLLM, and the cost is 7% to 10%, so opt-in (H100, Oct 24) |
| Cost | GPU-validated, as measured | throughput lower by: every layer, 11% (1.5B) and 20% (7B) at 256 requests at a time, 33% (1.5B) and 22% (7B) at 16; one layer in eight (the same layers every step), 2.5% to 7% (H100, Oct 22). Real `vllm serve`, 1.5B, every layer: 34% at 1 request at a time, 10% at 256 (Oct 23). INT4 7B: requests took 45% longer with every layer, 1% with one in eight (Oct 23) |
| The alert's row kept and rechecked in float64 | GPU-validated | captured inside CUDA graphs (Oct 21, 22) |
| `sdcguard ready` (protection status from outside) | GPU-validated | second run on real `vllm serve`, with IDLE and `--probe`: 30 of 30 verdicts as expected, never PROTECTED in the 21 cases where it shouldn't be (H100, vLLM 0.32, Oct 24) |
| The requests an alert concerns | GPU-validated, narrow | 5 of 5 single requests named exactly; with 32 requests at once, the request owning the corrupted row named exactly in 5 of 5 trials (eager mode, Oct 24); dual-batch overlap, sequence parallelism and speculative decoding not covered |
| The real cuBLAS bug caught inside the server | GPU-validated | cuBLAS 6862950 inside real `vllm serve` on an H200 as an MPS client at 40%: 14,707 alerts, every one of the 2,115 with a captured row rechecked as wrong in float64; replays at the alerts' real shapes wrong in 20 of 20 runs at 40% and right in 20 of 20 at 100%, on the server's kernel; no alert at 100% (Oct 24; judged from the saved files after the run hit its time limit in a reported-only step) |
| `sdcguard isolation` (one request's answer depending on another's) | GPU-validated | vLLM issue #62124 (FP8 per-tensor MoE with CUDA graphs) on an L4: LEAK on vLLM 0.31.0 (20 of 192 answers changed with the request before them) and 0.32.0 (55 of 180); the bf16 and eager controls 0 of 192 (Oct 24) |

## Deciding on a change

| Part | Status | Evidence |
|---|---|---|
| `sdcguard gate` (cases, compare, workaround, pin) on cuBLAS 6862950 | GPU-validated | all six expectations in the second H200 run: wrong in every run without the pin, right in 50 of 50 with it (Oct 22) |
| `gate model-check` (teacher-forced next-token distributions) | GPU-validated, offline | PASS for batching and Triton attention, FAIL for rms_norm_eps 1.0 with the matmul checks quiet, and FAIL for the AWQ model, which measures AWQ's intended approximation against the default thresholds (starting points from published numbers, not calibrated per model), not a defect (H100, Oct 23) |
| `gate model-check --url` (against a running server) | CPU-tested, fixed after a GPU run | the first server runs (Oct 24) found our bug: vLLM applies a model's generation_config (a repetition penalty) to requests that don't set it, so the reference wasn't greedy and identical servers agreed on only 88% to 95% of top tokens; the reference is now recorded with neutral sampling settings and top-1 is measured against its own top token; reruns pending |
| `sdcguard qualify` (one command: baseline and candidates, every check, a decision each, repro folders) | CPU-tested | first GPU run failed on the bug above (Oct 24); rerun pending |
| `sdcguard consistency` (prefix-cache and decode-vs-prefill probes) | CPU-tested | GPU run on vLLM #54035 written Oct 24 ("An FP8 KV-cache decode bug through qualify") |
| `gate model-check --trace-replay` (vLLM's Trace Replay, the decode path) | CPU-tested | used in R3 and R4, reported only |
| `sdcguard kernels` (inventory, coverage, diff, known-issue match) | CPU-tested | first real server inventory in R2 |
| `sdcguard kernels test` (rerun every inventoried call, judged) | CPU-tested | first GPU run in R2, reported only |
| `sdcguard perf` | CPU-tested | used in R3 |
| `sdcguard decide` (arithmetic, approximation, performance, isolation, consistency, FP8 quantization, scope) | CPU-tested | first GPU run (four changes on a 7B server, Oct 24) failed: the quality bug above, and one layer in eight checked the same 15 layers, not the faulted one; rerun on every layer pending |
| Telling a bad GPU from a library bug (`investigate`) | Planned | |

## Whole machines

| Part | Status | Evidence |
|---|---|---|
| `sdcguard check` | GPU-validated | 10 cloud GPUs, all PASS, 0 wrong in 1,683,363 checks (Oct 18); FAIL at 40%, 75% and 14% MPS on the H200 where DCGM and gpu-burn passed (Oct 20) |
| `sdcguard blas`, `sdcguard issues` | GPU-validated | the 0.4.0 runs: on the H200 at 40% every wrong configuration was algorithm 66 with 4+ CTA clusters (`results/v040/`, Oct 21) |
| Precise mode outside the fused check (`precise=True`) | Ran on a GPU, limits | held-out criteria met on 2 of 5 GPUs; costs 261% to 790% per checked step (Oct 19) |
| Fleet report, certificates, Kubernetes and Slurm checks, NVSentinel | CPU-tested | |

## Not established

- That sdcguard finds a naturally defective GPU.
- Anyone paying for it, or using it in production.
- Detection of errors under about 1% of a block with the default check (the fused check reaches below 1%, in
  training only).
