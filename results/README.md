# Results

Raw output of the GPU runs behind the numbers in the README, copied unchanged, so paths inside the
files refer to the original run folders (`calibration/ts-gpt2/CALIBRATION.md` is
`false_alarms_and_slowdown_gpt2/CALIBRATION.md` here). NVIDIA H100 80GB HBM3 on Modal, PyTorch 2.14.1
(CUDA 13.0), October 14-15, 2026, except `round3_oct18/` and the October 19-22 folders at the end of the table,
which record their own GPUs. Most runs since October 19 also record the code fingerprint they ran
(math_modes, precise_diag and the first two mps_bug files don't).

| Folder | What it is | Start with |
|---|---|---|
| `harm_study_gpt2/` | 24 runs of GPT-2 small (124M) with a simulated bad GPU, 4 clean runs; one folder per faulty run (verdict, summary, loss plot, text samples) and every run's raw JSON in `raw/` | `REPORT.md` |
| `false_alarms_and_slowdown_gpt2/` | 5.14 GPU-hours of clean training with every matmul checked (`soak/`), and the slowdown measurement | `CALIBRATION.md` |
| `h100_check_run/` | the last check run before the main run: tests, self-test, false-alarm diagnosis, a small end-to-end study, slowdown on GPT-2 small and on a 0.9B model | `selftest.txt`, `overhead_big.txt`, `runs/diag/ts-gpt2/DIAG.md` |
| `demo_4gpu/` | the 4-GPU demo: GPU 2 goes bad at step 205 | `STORY.txt`, `demo.png`, `demo_log.txt` |
| `round3_oct18/` | October 18, 2026: fast training mode, the screen, the serving check (Triton kernel) and its cost on 1.1B and 7B models, vLLM 0.31 with Qwen2.5 1.5B and 7B, NVBit fault injection on an L4. Has its own README | `round3_oct18/README.md` |
| `demo_4gpu_first_run/` | the first demo run, where GPU 2 went bad at step 200 (a checked step, so the instant catch was partly timing) | `STORY.txt` |
| `gpu_tests/` | the GPU test suite with the kernel comparisons, the installed wheel, `sdcguard check`, B200 kernel timings and the attention calibration, one log per GPU per run. Each file is the runner's summary: the tail of every step's output and its exit code (the full step logs stay on the `sdcguard-runs` Modal volume under `/check_sweep/`). File names give the time each GPU's result came back, on the launching Mac's clock (UTC-4 for these), so `*_20261019_23*.txt` is the October 20 (UTC) run at code fingerprint 74b844ee7ca8, which passed on all three GPUs. Earlier logs are kept as they came back, failures and empty runs included | the three `*_20261019_23*.txt` |
| `attention_calib/` | the attention check calibration from that run, one JSON per GPU (criteria met on all three) | `docs/METHOD.md`, attention section |
| `math_modes/` | October 19: matmuls in every math mode against a float64 reference, five GPUs | `SUMMARY.md` |
| `precise_diag/`, `precise_models/`, `precise_validate/` | October 19: precise mode check by check, its noise models on real training matmuls, and its held-out validation (criteria met on 2 of 5 GPUs) | each folder's `SUMMARY.md` |
| `blackwell_triton/` | October 19: the B200 Triton fault bisect | `docs/METHOD.md`, Blackwell section |
| `mps_bug/` | October 19: the H200 MPS reproduction, three runs | `docs/METHOD.md`, head-to-head section |
| `h2h/` | October 20: the head-to-head on cuBLAS known issue 6862950 on an H200 (sdcguard check, DCGM, gpu-burn at four MPS shares), with its preflight | `h200_20261020_060928.md` |
| `vllm_oct20/` | October 20: vLLM 0.31 with Qwen2.5 1.5B and 7B, every layer and 1 in 8 checked | each model's `SUMMARY.json` |
| `fused/`, `grad_control/` | October 20: the first fused precise check on an H100 and an L4, and the recompute check's gradient control run | each folder's `SUMMARY_*.md` |
| `v040/` | October 21: the 0.4.0 runs on one H100, two H100s and an H200 under MPS. Five runs, every criterion met in the last two; the last is the released code (fingerprint 8d34bf97fcf7) | `SUMMARY_20261021_194128.md`, and `docs/METHOD.md`, 0.4.0 section |
| `vllm_oct21/` | October 21: vLLM 0.31 with Qwen2.5 1.5B and 7B on the 0.4.0 code, including capture of the alerting row inside vLLM | each model's `SUMMARY.json` |
| `gate_h200/` | October 22: `sdcguard gate` on cuBLAS issue 6862950 on an H200 under MPS, two runs (the first found the pin wasn't applied; every expectation met in the second) | `docs/METHOD.md`, upgrade gate section |
| `serve_cost/` | October 22: the serving check's cost kernel by kernel on an H100, three runs: v2 against v1 (v1 stayed), v3 against v1 (v3 used from 256 rows), and v5 against both with more row counts (v5 no cheaper and removed; v3 cheaper from 96 rows) | `docs/METHOD.md`, serving kernel section |
| `v050/` | October 22: the 0.5.0 serving checks on an H100 (comparison set ALL AGREE, capture criteria 6, 7 and 8 met) | `docs/METHOD.md`, serving kernel section |
| `vllm_oct22/` | October 22: vLLM 0.31 with Qwen2.5 1.5B and 7B on the 0.5.0 code, every layer and 1 in 8 checked, at 256 and at 16 requests at a time; every criterion met | each model's `SUMMARY.json` |
| `machete_diag_oct23/` | October 23: which part of the INT4 Machete patch changed completions, eight variants against vLLM without sdcguard (the check as an op of its own did; loading, probing and adding the bias after Machete didn't) | `DIAG.json`, and `docs/METHOD.md`, INT4 section |
| `machete_diag2_oct23/` | October 23: the Machete diagnostic's second round: the GEMM and the check as one op, against vLLM without sdcguard, completions and Inductor's code compared (64 of 64 the same; the same kernels apart from their names) | `DIAG.json`, and `docs/METHOD.md`, INT4 section |
| `quant_oct23/` | October 23: the INT4 runs after it: every criterion met on the four 1.5B runs (Machete and Marlin); Qwen2.5-7B-Instruct-AWQ on Machete changed 5 of 64 completions (everything else met) | each model's `SUMMARY.json`, and `docs/METHOD.md`, INT4 section |
| `machete_diag3_oct23/` | October 23: the third round: the op named `machete_mm`, taking the layer's checksums (256 of 256 the same, code byte for byte vLLM's, nothing autotuned anew); one variant per difference; vLLM alone, compiled fresh, changed 28 of 256 through one autotuning choice | `DIAG.json`, and `docs/METHOD.md`, INT4 section |
| `quant3_oct23/` | October 23: the INT4 runs after round 3: every criterion met on all five, the 7B on Machete included (64 of 64 the same as untouched vLLM); every layer cost 45% on the 7B, 1 in 8 cost 1% | each model's `SUMMARY.json`, and `docs/METHOD.md`, INT4 section |
| `serve_oct23/` | October 23: real `vllm serve`, plain and guarded, over HTTP: vLLM 0.32 and 0.31 on one H100 met every criterion (64 of 64 the same, alerts reached a webhook, latency at 1, 16 and 256 requests at a time); the 2-GPU run died before serving, plain vLLM too (FlashInfer's all-reduce needs nvcc; kept as `serve032_tp2_before_20261023_201503`), and its rerun with vLLM's own all-reduce met every criterion | each run's `SUMMARY.json` and server logs, and `docs/METHOD.md` |
| `quant_l4_oct23/` | October 23: INT4 on Marlin on an L4: every criterion met on both 1.5B models | each model's `SUMMARY.json` |
| `outputs_oct23/` | October 23: the gate's output check: PASS for batching and Triton attention, FAIL for the AWQ model and for rms_norm_eps 1.0 (the matmul checks quiet), as expected | each check's JSON |
| `fused_soak/` | October 23: the fused precise check on 40 minutes of clean training on an H100 (4,568,751 checks, worst 2.67) and an L4 (613,404, worst 3.32): no alarm, every criterion met | each GPU's JSON |
| `mode_compare/` | October 23: the default check and precise mode on the same corrupted matmuls on an H100, an A100 and an L4: half of the errors flagged from 29% (default) and 0.71% to 0.92% (precise), with step times | `SUMMARY.md` |

In the demo folders, `baseline/` is the run without sdcguard and `sdcguard/` the run with it, and
`faults_rank2_attempt*.json` is the fault simulator's log for each attempt. In the first demo run the
`checked`/`caught` fields of that log weren't filled in (a demo script bug, fixed before the second run);
the alert in its `demo_log.txt` shows the catch.
