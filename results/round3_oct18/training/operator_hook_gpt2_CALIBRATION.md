# Calibration: ts-gpt2

## Slowdown (NVIDIA H100 80GB HBM3, base 83.19 ms/step)

| setting | share of matmuls checked | slowdown (estimated) | slowdown (measured) |
|---|---|---|---|
| 100% of matmuls every 1 steps | 100.0% | +43.76% |  |
| 100% of matmuls every 10 steps | 10.0% | +4.38% | +4.47% |
| 100% of matmuls every 20 steps | 5.0% | +2.19% | +2.14% |
| 100% of matmuls every 50 steps | 2.0% | +0.88% | +0.62% |

- 100% of matmuls checked: plain step 83.19 ms, checked step 119.6 ms (+36.41 ms per checked step)
- False alarms during the slowdown runs (clean training): **0** in 11,169 checks, worst clean z 2.75

## Kill criterion #1

- False alarms so far (no soak yet): 0 (0 in 11,169 slowdown-run checks; 0 in harm-study clean runs; 0 unexplained in faulty runs)
- Slowdown of 3% or less: MET at 100% of matmuls every 20 steps (5.0% of all matmuls checked, +2.19%, the higher of the estimated and measured numbers)
