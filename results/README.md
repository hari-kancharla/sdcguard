# Results

Raw output of the GPU runs behind the numbers in the README, copied unchanged, so paths inside the
files refer to the original run folders (`calibration/ts-gpt2/CALIBRATION.md` is
`false_alarms_and_slowdown_gpt2/CALIBRATION.md` here). NVIDIA H100 80GB HBM3 on Modal, PyTorch 2.14.1
(CUDA 13.0), October 14-15, 2026, except `round3_oct18/`.

| Folder | What it is | Start with |
|---|---|---|
| `harm_study_gpt2/` | 24 runs of GPT-2 small (124M) with a simulated bad GPU, 4 clean runs; one folder per faulty run (verdict, summary, loss plot, text samples) and every run's raw JSON in `raw/` | `REPORT.md` |
| `false_alarms_and_slowdown_gpt2/` | 5.14 GPU-hours of clean training with every matmul checked (`soak/`), and the slowdown measurement | `CALIBRATION.md` |
| `h100_check_run/` | the last check run before the main run: tests, self-test, false-alarm diagnosis, a small end-to-end study, slowdown on GPT-2 small and on a 0.9B model | `selftest.txt`, `overhead_big.txt`, `runs/diag/ts-gpt2/DIAG.md` |
| `demo_4gpu/` | the 4-GPU demo: GPU 2 goes bad at step 205 | `STORY.txt`, `demo.png`, `demo_log.txt` |
| `round3_oct18/` | October 18, 2026: fast training mode, the screen, the serving check (Triton kernel) and its cost on 1.1B and 7B models, vLLM 0.31 with Qwen2.5 1.5B and 7B, NVBit fault injection on an L4. Has its own README | `round3_oct18/README.md` |
| `demo_4gpu_first_run/` | the first demo run, where GPU 2 went bad at step 200 (a checked step, so the instant catch was partly timing) | `STORY.txt` |

In the demo folders, `baseline/` is the run without sdcguard and `sdcguard/` the run with it, and
`faults_rank2_attempt*.json` is the fault simulator's log for each attempt. In the first demo run the
`checked`/`caught` fields of that log weren't filled in (a demo script bug, fixed before the second run);
the alert in its `demo_log.txt` shows the catch.
