# Evidence ledger

Two kinds of claims appear in these docs, and they are kept apart here: what other people published (papers,
vendor documentation, source code we read), and what sdcguard's own runs measured. A published finding is not
an sdcguard result, and an sdcguard result on a few cloud GPUs is not a production finding. Negative results stay
listed. Every pre-registered criterion is listed with its outcome, including the ones that weren't met.

## 1. Published by others (not our measurements)

| Source | What it says | What it does not say |
|---|---|---|
| ByteDance, AEGIS ([OSDI '26, July 13-15, 2026](https://www.usenix.org/system/files/osdi26-lei.pdf)), section 7.2 | 18 SDC incidents on 13 faulty GPUs over 3.5 x 10^7 GPU-hours, at 0.86% overhead with sampling; 15 of the 18 were silent. Deterministic detection (fingerprints of an output compared with the recompute training already does) found 12; algorithmic detection and its outlier warnings found 6. | That checksums are useless: it calls algorithmic detection essential where recomputation is limited. |
| AEGIS, section 8 | "we did not successfully locate SDC incidents using checksums accumulated from bfloat16 outputs. Instead, all confirmed detections were achieved by accumulating checksums from float32 accumulators." The same section reports an SDC-like recompute mismatch caused by a training-framework bug (an out-of-bounds write). | The sentence is about checksum localization; it does not say all 18 incidents came from fp32 checksums (12 came from fingerprints). |
| ByteDance and SJTU, SDCHunter ("SDCs in the Wild", [OSDI '26](https://www.usenix.org/conference/osdi26/presentation/zheng)), Tables 2 and 3 | 23 SDC-defective GPUs from production. The first SDC was seen in GEMM kernels on 17.3% of them, in-place MoE 17.3%, FlashAttention 13.0%, RMSNorm 8.69% and others. On the four GPUs with GEMM SDCs, BF16 and FP16 tensor-core GEMMs were correct on all four; FP32 and FP64 CUDA-core GEMMs were wrong on three. | That bf16 matmul checks would have found those four GPUs. They would not have (sdcguard's training checks are of bf16/fp16 matmuls). |
| NVIDIA, CUDA 13.4 Update 2 release notes (October 12, 2026), cuBLAS known issue 6862950 | GEMMs "can produce incorrect results on Hopper GPUs for FP16, BF16, and FP8 matrices when the process runs as an MPS client with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE below 100. This affects only algorithms with CUBLASLT_ALGO_CONFIG_ID equal to 66 with cluster configurations that use 4 or more CTAs." Introduced in cuBLAS 12.6.3; workaround: 1 or 2 CTA clusters. | How common that configuration is in production. |
| NVIDIA DCGM 4.6.1 source, [compare.cu](https://github.com/NVIDIA/DCGM/blob/64df9f894541e426e416131a9820cae97aa4dd81/nvvs/plugin_src/diagnostic/compare.cu#L30-L136); [docs](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html) | The Diagnostic test verifies matrix results by repetition: the same Hgemm/Sgemm/Dgemm of the same inputs many times, each result compared with the pass's first (tolerance 0.001; 1e-7 in FP64), NaNs counted. Its memory tests compare reads with written patterns. | That it compares GEMMs with an independent reference, or runs BF16 or FP8. |
| NVIDIA Megatron Core [rerun state machine](https://github.com/NVIDIA/Megatron-LM/blob/ebab48fc40c2b6f0a8768f765b9364656a2dccb9/megatron/core/rerun_state_machine.py) (source read October 20, 2026) | Any rejection function; built in: NaN/Inf in the loss and gradient-bucket norms, opt-in spiky loss and large gradients (10x the largest seen); reruns on the same GPU, then another; outcomes correct, transient or persistent; a stats mode for nondeterminism. Self-described as alpha-level. | That it checks individual matmuls. |

More sources, and the claims we decided not to make, are in [WHY.md](WHY.md).

## 2. Measured by sdcguard (our runs)

| What | When, where | Result | Kind | Details |
|---|---|---|---|---|
| False alarms, clean training | Oct 14-15, H100 | 0 in 24,580,560 checks over 5.14 GPU-hours | positive | [METHOD.md](METHOD.md#passfail-criteria-written-before-the-first-gpu-runs) |
| Harm study (software-injected faults, GPT-2) | Oct 14-15 | 15 of 24 faulty runs crashed; checks caught 99% of faults moving a block by 10%+, 70% of 1-10%, none under 1% | mixed | [REPORT.md](../results/harm_study_gpt2/REPORT.md) |
| Catch before the crash | Oct 14-15 | every 10 steps: 14 of 15 harmed runs, but +10.9% cost then; every 40 steps: 8 of 15 | partly met | [METHOD.md](METHOD.md#passfail-criteria-written-before-the-first-gpu-runs) |
| NVBit bit flips on an L4 | Oct 18 | exponent flip and a mantissa flip of about 10% caught on 30 of 30 steps; a 1% mantissa flip not caught (0 of 30) | mixed | [METHOD.md](METHOD.md#hardware-fault-injection-nvbit) |
| `sdcguard check` first sweep | Oct 18, 10 cloud GPUs | all PASS, 0 wrong in 1,683,363 checks, fingerprints matched | positive (no faulty GPU in the set) | [METHOD.md](METHOD.md#sdcguard-check) |
| Math modes and the first precise-mode model | Oct 19, five GPUs | math modes clean after one fix; precise mode failed (70 to 78 false alarms in 2,970 checks) | negative for precise mode | [METHOD.md](METHOD.md#what-the-run-found-october-19-2026) |
| Precise mode, refit model, held-out | Oct 19 | 0 alarms in 17,424 checks, but worst scores 4.27 (A100) and 4.26 (L4) over the line of 4; T4 didn't run | criteria not met | [METHOD.md](METHOD.md#what-the-held-out-run-found-october-19-2026-evening-criteria-not-met) |
| Precise mode cost | Oct 19 | +261% to +790% per checked step | negative | same |
| cuBLAS known issue 6862950 reproduced | Oct 19, Modal H200 | wrong FP16/BF16/FP8 outputs at some MPS shares; sdcguard alerted; at 20% it alerted while outputs were right (consistent with its own checksum matmul being hit) | positive, with a caveat | [results/mps_bug](../results/mps_bug) |
| Attention checks, calibration | Oct 20, L4, H100, B200 | criteria met in bf16 (worst clean backward score 0.139) | positive, limited scope | [METHOD.md](METHOD.md#what-the-calibration-found-october-20-2026) |
| GPU test suite | Oct 19-20, L4, H100, B200 | passed at fingerprint 74b844ee7ca8 | positive | [results/gpu_tests](../results/gpu_tests) |
| H200 preflight for the head-to-head | Oct 20, Modal H200 | GO: under MPS below 100%, only cuBLASLt algorithm 66 with 4+ CTA clusters went wrong (20 of 32 runs at 40%, 10 of 32 at 14%); 1x1x1 workaround right; DCGM ran as an MPS client; its Hgemm at 2048 was right at every share | positive (the bug as published) | [result](../results/h2h/preflight_h200_20261020_055833.json) |
| Head-to-head on cuBLAS known issue 6862950 (one H200, MPS shares 100, 40, 75, 14) | Oct 20, Modal H200 | sdcguard check DETECTED at 40, 75 and 14, passed at 100; DCGM's diagnostic and gpu-burn passed everywhere and were NOT EXERCISED: their own GEMMs were right at every share (DCGM's FP16 GEMMs used algorithm 39, 1x2x1 cluster); one Xid 31 of unknown cause, no step affected | positive, with its limits (one GPU, one bug) | [summary](../results/h2h/h200_20261020_060928.md) |
| vLLM plugin in real vLLM, compiled mode, executed-count gate | Oct 20, Modal H100, vLLM 0.31.0 | gate passed in all four guarded runs (Qwen2.5-1.5B and 7B, every layer and 1 in 8): every layer's check count rose by 384 during measured traffic, none unjudged, 0 false alarms in 170,688 layer checks, 64 of 64 answers unchanged, the flipped weight bit named; cost on 7B: 4.2% of throughput for the checks at 1 in 8 (1.7% on October 18), 22.9% at every layer | positive, with its limits (one GPU, offline API, an easy injected fault; the 1-in-8 cost rose) | [METHOD.md](METHOD.md#vllm-again-october-20-2026-with-the-executed-count-gate) |
| A naturally defective GPU | never | none has been found or tested | gap | |
| Real users or production incidents | never | none yet | gap | |

## 3. Pre-registered and not yet run

| Experiment | Written | Where |
|---|---|---|
| Default check vs precise mode on the same matmuls | Oct 19 | [METHOD.md](METHOD.md#the-default-check-and-precise-mode-on-the-same-matmuls-written-october-19-2026-before-any-gpu-run) |

When a run finishes, its row moves to section 2 with its result, whatever it is.
