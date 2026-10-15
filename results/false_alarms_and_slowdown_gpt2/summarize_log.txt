# Calibration: ts-gpt2

## False alarms on clean training (detector checking every matmul, forward and backward)

- NVIDIA H100 80GB HBM3: 5.14 GPU-hours over 5 runs, 24,580,560 checks
- False alarms: **0**
- Worst clean residual: z = 3.54 (alert threshold 6)

| z range | checks |
|---|---|
| 0-0.5 | 0 |
| 0.5-1 | 7,390 |
| 1-1.5 | 7,679,161 |
| 1.5-2 | 14,899,659 |
| 2-2.5 | 1,931,323 |
| 2.5-3 | 62,717 |
| 3-3.5 | 309 |
| 3.5-4 | 1 |
| 4-5 | 0 |
| 5-6 | 0 |
| 6-inf | 0 |

## Slowdown (NVIDIA H100 80GB HBM3, base 83.18 ms/step)

| setting | share of matmuls checked | slowdown (estimated) | slowdown (measured) |
|---|---|---|---|
| 100% of matmuls every 1 steps | 100.0% | +112.15% |  |
| 100% of matmuls every 10 steps | 10.0% | +11.21% | +10.94% |
| 100% of matmuls every 20 steps | 5.0% | +5.61% | +5.58% |
| 100% of matmuls every 30 steps | 3.3% | +3.74% | +3.80% |
| 100% of matmuls every 40 steps | 2.5% | +2.80% | +2.82% |
| 100% of matmuls every 50 steps | 2.0% | +2.24% | +2.16% |
| 100% of matmuls every 100 steps | 1.0% | +1.12% | +1.15% |
| 10% of matmuls every 1 steps | 10.0% | +13.13% |  |

- 100% of matmuls checked: plain step 83.28 ms, checked step 176.66 ms (+93.39 ms per checked step)
- 10% of matmuls checked: plain step 82.96 ms, checked step 93.85 ms (+10.89 ms per checked step)
- False alarms during the slowdown runs (clean training): **0** in 18,265 checks, worst clean z 2.98

## False alarms in the harm study

- Clean runs (4, incl. the same-seed repeat): **0** in 350,400 checks
- Faulty runs (24): **0** unexplained alerts on outputs the injector did not corrupt (alerts on NaN/Inf/overflowing outputs are the corruption spreading and are not counted)

## Kill criterion #1

- Zero false alarms over ~5 clean GPU-hours: MET (0 in 5.14 GPU-hours of the soak; 0 during the slowdown runs; 0 in the harm study's clean runs; 0 unexplained in its faulty runs)
- Slowdown of 3% or less: MET at 100% of matmuls every 40 steps (2.5% of all matmuls checked, +2.82%, the higher of the estimated and measured numbers)
