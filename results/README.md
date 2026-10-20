# Results

Raw output of the GPU runs behind the numbers in the README, copied unchanged, so paths inside the
files refer to the original run folders (`calibration/ts-gpt2/CALIBRATION.md` is
`false_alarms_and_slowdown_gpt2/CALIBRATION.md` here). NVIDIA H100 80GB HBM3 on Modal, PyTorch 2.14.1
(CUDA 13.0), October 14-15, 2026, except `round3_oct18/` and the October 19-20 folders at the end of the table,
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

In the demo folders, `baseline/` is the run without sdcguard and `sdcguard/` the run with it, and
`faults_rank2_attempt*.json` is the fault simulator's log for each attempt. In the first demo run the
`checked`/`caught` fields of that log weren't filled in (a demo script bug, fixed before the second run);
the alert in its `demo_log.txt` shows the catch.
