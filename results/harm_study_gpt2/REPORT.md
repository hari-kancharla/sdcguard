# Harm study report: ts-gpt2

## Kill criteria (this study)

- **#1 false alarms (in this study):** clean runs: **0** in 350,400 checks (worst clean z 3.07, threshold 6). Faulty runs: **0** unexplained alerts on outputs the injector did not corrupt (plus 234 alerts on NaN/Inf/overflowing outputs, which is the corruption spreading into later layers, not false alarms). The 5 GPU-hour soak and the slowdown are in calibration/<preset>/CALIBRATION.md.
- **#2 harm:** 15 of 24 faulty runs were harmed. A run counts as harmed if it crashed (NaN) or survived but ended worse than every clean run by more than the seed-to-seed spread.
- **#3 caught:** of the 15 harmed runs, caught (a check caught an injected fault before the crash, or during the run for survivors): checking every 10 steps: **14/15**, checking every 20 steps: **10/15**, checking every 50 steps: **6/15**.
- The detector checked every 10 steps during these runs. Results for every 20 and 50 steps come from the same runs, keeping only the checks on steps divisible by 20 or 50. (Checked steps keep matmul partial sums in fp32, so a real every-50 run could drift very slightly differently.) The GPU names come from the multi-GPU demo, not from these single-GPU runs.
- Code version(s): 944a231a5124

## Setup

- Hardware: NVIDIA H100 80GB HBM3, torch 2.14.1+cu130
- Model/training: {"dataset": "tinystories", "n_layer": 12, "n_head": 12, "n_embd": 768, "block": 512, "batch": 32, "steps": 4000, "eval_every": 250, "eval_batches": 40, "lr": 0.0006, "ts_train_mb": 300}
- Clean runs: clean_s0 1.4427, clean_s1 1.4421, clean_s2 1.4441; same seed again: 1.4429
- Clean training time, including evaluation: 0.45 GPU-hours

| config | runs | crashed | loss looked normal before crash | caught before crash | median lead (steps) | survived | survived & harmed | harmed: caught @10/@20/@50 | caught / checked |
|---|---|---|---|---|---|---|---|---|---|
| ts-gpt2 qk mantissa warp rate 0.01 | 3 | 0 | 0 | 0 | - | 3 | 0 | 0/0/0 of 0 | 162 / 312 |
| ts-gpt2 qk nvidia tile rate 0.001 | 3 | 3 | 3 | 3 | 195 | 0 | 0 | 3/2/1 of 3 | 6 / 6 |
| ts-gpt2 qk nvidia tile rate 0.01 | 3 | 3 | 3 | 3 | 23 | 0 | 0 | 3/1/1 of 3 | 5 / 5 |
| ts-gpt2 qk nvidia warp rate 0.001 | 3 | 3 | 3 | 3 | 1104 | 0 | 0 | 3/3/1 of 3 | 11 / 13 |
| ts-gpt2 qk nvidia warp rate 0.01 | 3 | 3 | 3 | 2 | 27.5 | 0 | 0 | 2/1/1 of 3 | 3 / 3 |
| ts-gpt2 qk zero tile rate 0.01 | 3 | 0 | 0 | 0 | - | 3 | 0 | 0/0/0 of 0 | 272 / 272 |
| ts-gpt2 qk zero warp rate 0.01 | 3 | 0 | 0 | 0 | - | 3 | 0 | 0/0/0 of 0 | 301 / 310 |
| ts-gpt2 qkvprojmlp nvidia warp rate 0.001 | 3 | 3 | 3 | 3 | 606 | 0 | 0 | 3/3/2 of 3 | 15 / 16 |

## Detection by fault size (all faulty runs)

Size = how much the corrupted values changed their 128-value output block (L2 norm ratio).

| fault size | checked | caught |
|---|---|---|
| <0.1% | 11 | 0 (0%) |
| 0.1-1% | 67 | 0 (0%) |
| 1-10% | 263 | 185 (70%) |
| 10-100% | 593 | 587 (99%) |
| NaN/Inf/huge | 3 | 3 (100%) |

## Every faulty run

```
=== faulty_qk_mantissa_warp_r0.01_f1 ===
Final val loss: faulty 1.4423 vs clean 1.4421-1.4441 (mean 1.4429, 3 seeds).
Fault effect (vs same-seed clean run): -0.0004; seed-to-seed spread: 0.0020; same-seed GPU noise: 0.0002.
Kill criterion 2 (harm): NOT MET - within normal run-to-run variation.
Loss curve: looks normal (max rise above the clean runs 0.0029, typical clean spread 0.0160).
sdcguard (checking every 10 steps): caught an injected fault at step 0.
Detection if checking less often (same run, keeping only those checks): every 10: caught 53/91, first at step 0; every 20: caught 25/45, first at step 0; every 50: caught 11/18, first at step 0.
Detector: caught 53 of 91 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): <0.1%: 0/2, 0.1-1%: 0/22, 1-10%: 51/65, 10-100%: 2/2.
```
```
=== faulty_qk_mantissa_warp_r0.01_f2 ===
Final val loss: faulty 1.4436 vs clean 1.4421-1.4441 (mean 1.4429, 3 seeds).
Fault effect (vs same-seed clean run): +0.0009; seed-to-seed spread: 0.0020; same-seed GPU noise: 0.0002.
Kill criterion 2 (harm): NOT MET - within normal run-to-run variation.
Loss curve: looks normal (max rise above the clean runs 0.0054, typical clean spread 0.0160).
sdcguard (checking every 10 steps): caught an injected fault at step 20.
Detection if checking less often (same run, keeping only those checks): every 10: caught 61/110, first at step 20; every 20: caught 30/54, first at step 20; every 50: caught 13/25, first at step 50.
Detector: caught 61 of 110 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): <0.1%: 0/5, 0.1-1%: 0/19, 1-10%: 57/82, 10-100%: 4/4.
```
```
=== faulty_qk_mantissa_warp_r0.01_f3 ===
Final val loss: faulty 1.4424 vs clean 1.4421-1.4441 (mean 1.4429, 3 seeds).
Fault effect (vs same-seed clean run): -0.0003; seed-to-seed spread: 0.0020; same-seed GPU noise: 0.0002.
Kill criterion 2 (harm): NOT MET - within normal run-to-run variation.
Loss curve: looks normal (max rise above the clean runs 0.0036, typical clean spread 0.0160).
sdcguard (checking every 10 steps): caught an injected fault at step 40.
Detection if checking less often (same run, keeping only those checks): every 10: caught 48/111, first at step 40; every 20: caught 21/54, first at step 40; every 50: caught 10/21, first at step 450.
Detector: caught 48 of 111 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): <0.1%: 0/4, 0.1-1%: 0/24, 1-10%: 45/80, 10-100%: 3/3.
```
```
=== faulty_qk_nvidia_tile_r0.001_f1 ===
Faulty run CRASHED (NaN) at step 215. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 20: 195 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 2/2, first at step 20; every 20: caught 2/2, first at step 20; every 50: caught 0/0, never caught.
Detector: caught 2 of 2 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 2/2.
```
```
=== faulty_qk_nvidia_tile_r0.001_f2 ===
Faulty run CRASHED (NaN) at step 746. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 360: 386 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 2/2, first at step 360; every 20: caught 2/2, first at step 360; every 50: caught 0/0, never caught.
Detector: caught 2 of 2 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 2/2.
```
```
=== faulty_qk_nvidia_tile_r0.001_f3 ===
Faulty run CRASHED (NaN) at step 254. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 210: 44 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 2/2, first at step 210; every 20: caught 0/0, never caught; every 50: caught 1/1, first at step 250.
Detector: caught 2 of 2 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 2/2.
```
```
=== faulty_qk_nvidia_tile_r0.01_f1 ===
Faulty run CRASHED (NaN) at step 34. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 0: 34 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 1/1, first at step 0; every 20: caught 1/1, first at step 0; every 50: caught 1/1, first at step 0.
Detector: caught 1 of 1 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 1/1.
```
```
=== faulty_qk_nvidia_tile_r0.01_f2 ===
Faulty run CRASHED (NaN) at step 40. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 30: 10 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 2/2, first at step 30; every 20: caught 1/1, first at step 40; every 50: caught 0/0, never caught.
Detector: caught 2 of 2 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 180 (180 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 1/1, NaN/Inf/huge: 1/1.
```
```
=== faulty_qk_nvidia_tile_r0.01_f3 ===
Faulty run CRASHED (NaN) at step 33. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 10: 23 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 2/2, first at step 10; every 20: caught 0/0, never caught; every 50: caught 0/0, never caught.
Detector: caught 2 of 2 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 2/2.
```
```
=== faulty_qk_nvidia_warp_r0.001_f1 ===
Faulty run CRASHED (NaN) at step 2244. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 20: 2224 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 5/6, first at step 20; every 20: caught 2/2, first at step 20; every 50: caught 1/1, first at step 150.
Detector: caught 5 of 6 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 0.1-1%: 0/1, 1-10%: 1/1, 10-100%: 4/4.
```
```
=== faulty_qk_nvidia_warp_r0.001_f2 ===
Faulty run CRASHED (NaN) at step 1224. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 120: 1104 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 4/4, first at step 120; every 20: caught 4/4, first at step 120; every 50: caught 0/0, never caught.
Detector: caught 4 of 4 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 1-10%: 1/1, 10-100%: 3/3.
```
```
=== faulty_qk_nvidia_warp_r0.001_f3 ===
Faulty run CRASHED (NaN) at step 571. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 180: 391 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 2/3, first at step 180; every 20: caught 1/2, first at step 180; every 50: caught 0/1, never caught.
Detector: caught 2 of 3 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 54 (54 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 0.1-1%: 0/1, 10-100%: 1/1, NaN/Inf/huge: 1/1.
```
```
=== faulty_qk_nvidia_warp_r0.01_f1 ===
Faulty run CRASHED (NaN) at step 27. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 0: 27 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 1/1, first at step 0; every 20: caught 1/1, first at step 0; every 50: caught 1/1, first at step 0.
Detector: caught 1 of 1 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 1/1.
```
```
=== faulty_qk_nvidia_warp_r0.01_f2 ===
Faulty run CRASHED (NaN) at step 39. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) did not catch an injected fault before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 0/0, never caught; every 20: caught 0/0, never caught; every 50: caught 0/0, never caught.
Detector: caught 0 of 0 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
```
```
=== faulty_qk_nvidia_warp_r0.01_f3 ===
Faulty run CRASHED (NaN) at step 38. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 10: 28 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 2/2, first at step 10; every 20: caught 0/0, never caught; every 50: caught 0/0, never caught.
Detector: caught 2 of 2 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 2/2.
```
```
=== faulty_qk_zero_tile_r0.01_f1 ===
Final val loss: faulty 1.4422 vs clean 1.4421-1.4441 (mean 1.4429, 3 seeds).
Fault effect (vs same-seed clean run): -0.0005; seed-to-seed spread: 0.0020; same-seed GPU noise: 0.0002.
Kill criterion 2 (harm): NOT MET - within normal run-to-run variation.
Loss curve: looks normal (max rise above the clean runs 0.0036, typical clean spread 0.0160).
sdcguard (checking every 10 steps): caught an injected fault at step 0.
Detection if checking less often (same run, keeping only those checks): every 10: caught 73/73, first at step 0; every 20: caught 42/42, first at step 0; every 50: caught 12/12, first at step 0.
Detector: caught 73 of 73 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 73/73.
```
```
=== faulty_qk_zero_tile_r0.01_f2 ===
Final val loss: faulty 1.4425 vs clean 1.4421-1.4441 (mean 1.4429, 3 seeds).
Fault effect (vs same-seed clean run): -0.0002; seed-to-seed spread: 0.0020; same-seed GPU noise: 0.0002.
Kill criterion 2 (harm): NOT MET - within normal run-to-run variation.
Loss curve: looks normal (max rise above the clean runs 0.0031, typical clean spread 0.0160).
sdcguard (checking every 10 steps): caught an injected fault at step 30.
Detection if checking less often (same run, keeping only those checks): every 10: caught 110/110, first at step 30; every 20: caught 59/59, first at step 40; every 50: caught 25/25, first at step 50.
Detector: caught 110 of 110 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 110/110.
```
```
=== faulty_qk_zero_tile_r0.01_f3 ===
Final val loss: faulty 1.4427 vs clean 1.4421-1.4441 (mean 1.4429, 3 seeds).
Fault effect (vs same-seed clean run): +0.0000; seed-to-seed spread: 0.0020; same-seed GPU noise: 0.0002.
Kill criterion 2 (harm): NOT MET - within normal run-to-run variation.
Loss curve: looks normal (max rise above the clean runs 0.0033, typical clean spread 0.0160).
sdcguard (checking every 10 steps): caught an injected fault at step 10.
Detection if checking less often (same run, keeping only those checks): every 10: caught 89/89, first at step 10; every 20: caught 33/33, first at step 100; every 50: caught 14/14, first at step 100.
Detector: caught 89 of 89 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 89/89.
```
```
=== faulty_qk_zero_warp_r0.01_f1 ===
Final val loss: faulty 1.4420 vs clean 1.4421-1.4441 (mean 1.4429, 3 seeds).
Fault effect (vs same-seed clean run): -0.0007; seed-to-seed spread: 0.0020; same-seed GPU noise: 0.0002.
Kill criterion 2 (harm): NOT MET - within normal run-to-run variation.
Loss curve: looks normal (max rise above the clean runs 0.0030, typical clean spread 0.0160).
sdcguard (checking every 10 steps): caught an injected fault at step 0.
Detection if checking less often (same run, keeping only those checks): every 10: caught 91/94, first at step 0; every 20: caught 46/47, first at step 0; every 50: caught 18/18, first at step 0.
Detector: caught 91 of 94 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 1-10%: 8/9, 10-100%: 83/85.
```
```
=== faulty_qk_zero_warp_r0.01_f2 ===
Final val loss: faulty 1.4442 vs clean 1.4421-1.4441 (mean 1.4429, 3 seeds).
Fault effect (vs same-seed clean run): +0.0015; seed-to-seed spread: 0.0020; same-seed GPU noise: 0.0002.
Kill criterion 2 (harm): NOT MET - within normal run-to-run variation.
Loss curve: looks normal (max rise above the clean runs 0.0038, typical clean spread 0.0160).
sdcguard (checking every 10 steps): caught an injected fault at step 30.
Detection if checking less often (same run, keeping only those checks): every 10: caught 90/94, first at step 30; every 20: caught 51/55, first at step 60; every 50: caught 15/17, first at step 50.
Detector: caught 90 of 94 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 1-10%: 10/11, 10-100%: 80/83.
```
```
=== faulty_qk_zero_warp_r0.01_f3 ===
Final val loss: faulty 1.4426 vs clean 1.4421-1.4441 (mean 1.4429, 3 seeds).
Fault effect (vs same-seed clean run): -0.0001; seed-to-seed spread: 0.0020; same-seed GPU noise: 0.0002.
Kill criterion 2 (harm): NOT MET - within normal run-to-run variation.
Loss curve: looks normal (max rise above the clean runs 0.0033, typical clean spread 0.0160).
sdcguard (checking every 10 steps): caught an injected fault at step 10.
Detection if checking less often (same run, keeping only those checks): every 10: caught 120/122, first at step 10; every 20: caught 59/59, first at step 80; every 50: caught 29/30, first at step 150.
Detector: caught 120 of 122 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 1-10%: 11/12, 10-100%: 109/110.
```
```
=== faulty_qkvprojmlp_nvidia_warp_r0.001_f1 ===
Faulty run CRASHED (NaN) at step 397. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 40: 357 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 6/6, first at step 40; every 20: caught 2/2, first at step 40; every 50: caught 1/1, first at step 50.
Detector: caught 6 of 6 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 1-10%: 1/1, 10-100%: 5/5.
```
```
=== faulty_qkvprojmlp_nvidia_warp_r0.001_f2 ===
Faulty run CRASHED (NaN) at step 646. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 40: 606 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 5/6, first at step 40; every 20: caught 3/4, first at step 40; every 50: caught 0/0, never caught.
Detector: caught 5 of 6 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 1-10%: 0/1, 10-100%: 5/5.
```
```
=== faulty_qkvprojmlp_nvidia_warp_r0.001_f3 ===
Faulty run CRASHED (NaN) at step 682. Counted as harmed (kill criterion 2).
Before the crash the loss curve looked normal (inside the clean runs' range).
sdcguard (checking every 10 steps) caught an injected fault at step 60: 622 steps before the crash.
Detection if checking less often (same run, keeping only those checks): every 10: caught 4/4, first at step 60; every 20: caught 2/2, first at step 60; every 50: caught 1/1, first at step 450.
Detector: caught 4 of 4 corrupted outputs it checked. Alerts on outputs the injector did not corrupt: 0 (0 had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; 0 unexplained, counted as false alarms). Clean runs: 0 false alarms in 350400 checks.
Caught by fault size (change to the affected 128-value block): 10-100%: 3/3, NaN/Inf/huge: 1/1.
```
