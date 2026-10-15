# Harm study report: ts-gpu

## Kill criteria (this study)

- **#1 false alarms (in this study):** clean runs: **0** in 8,820 checks (worst clean z 2.75, threshold 6). Faulty runs: **0** unexplained alerts on outputs the injector did not corrupt (plus 108 alerts on NaN/Inf/overflowing outputs, which is the corruption spreading into later layers, not false alarms). The 5 GPU-hour soak and the slowdown are in calibration/<preset>/CALIBRATION.md.
- **#2 harm:** 2 of 2 faulty runs were harmed. A run counts as harmed if it crashed (NaN) or survived but ended worse than every clean run by more than the seed-to-seed spread.
- **#3 caught:** of the 2 harmed runs, caught (a check caught an injected fault before the crash, or during the run for survivors): checking every 10 steps: **2/2**, checking every 20 steps: **2/2**, checking every 50 steps: **2/2**.
- The detector checked every 10 steps during these runs. Results for every 20 and 50 steps come from the same runs, keeping only the checks on steps divisible by 20 or 50. (Checked steps keep matmul partial sums in fp32, so a real every-50 run could drift very slightly differently.) The GPU names come from the multi-GPU demo, not from these single-GPU runs.
- Code version(s): 27c88f80fb81

## Setup

- Hardware: NVIDIA H100 80GB HBM3, torch 2.14.1+cu130
- Model/training: {"dataset": "tinystories", "n_layer": 8, "n_head": 8, "n_embd": 512, "block": 256, "batch": 64, "steps": 300, "eval_every": 250, "eval_batches": 40, "lr": 0.0006, "ts_train_mb": 300}
- Clean runs: clean_s0 3.4844, clean_s1 3.6058
- Clean training time, including evaluation: 0.01 GPU-hours

| config | runs | crashed | loss looked normal before crash | caught before crash | median lead (steps) | survived | survived & harmed | harmed: caught @10/@20/@50 | caught / checked |
|---|---|---|---|---|---|---|---|---|---|
| ts-gpu qk nvidia tile rate 0.05 | 1 | 1 | 1 | 1 | 30 | 0 | 0 | 1/1/1 of 1 | 4 / 4 |
| ts-gpu qk nvidia warp rate 0.05 | 1 | 1 | 1 | 1 | 24 | 0 | 0 | 1/1/1 of 1 | 4 / 4 |

## Detection by fault size (all faulty runs)

Size = how much the corrupted values changed their 128-value output block (L2 norm ratio).

| fault size | checked | caught |
|---|---|---|
| 1-10% | 1 | 1 (100%) |
| 10-100% | 6 | 6 (100%) |
| NaN/Inf/huge | 1 | 1 (100%) |

## Every faulty run

```
=== faulty_qk_nvidia_tile_r0.05_f1 ===
Faulty run CRASHED (NaN) at step 30. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 0: 30 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 4/4, first at step 0; every 20: caught 2/2, first at step 0; every 50: caught 2/2, first at step 0.
Detector: caught 4 of 4 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 108 (108 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 8820 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 3/3, NaN/Inf/huge: 1/1.
```
```
=== faulty_qk_nvidia_warp_r0.05_f1 ===
Faulty run CRASHED (NaN) at step 24. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 0: 24 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 4/4, first at step 0; every 20: caught 3/3, first at step 0; every 50: caught 2/2, first at step 0.
Detector: caught 4 of 4 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 8820 checks.
Caught by fault size (change to the affected 128-value block): 1-10%: 1/1, 10-100%: 3/3.
```
