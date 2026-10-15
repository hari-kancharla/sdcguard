# Calibration: ts-gpt2

## Slowdown (NVIDIA H100 80GB HBM3, base 82.83 ms/step)

| setting | share of matmuls checked | slowdown (estimated) | slowdown (measured) |
|---|---|---|---|
| 100% of matmuls every 1 steps | 100.0% | +99.68% |  |
| 100% of matmuls every 10 steps | 10.0% | +9.97% | +10.47% |
| 100% of matmuls every 20 steps | 5.0% | +4.98% | +5.20% |
| 100% of matmuls every 50 steps | 2.0% | +1.99% | +2.03% |
| 100% of matmuls every 100 steps | 1.0% | +1.00% | +0.93% |
| 10% of matmuls every 1 steps | 10.0% | +10.76% |  |

- 100% of matmuls checked: plain step 82.87 ms, checked step 165.47 ms (+82.6 ms per checked step)
- 10% of matmuls checked: plain step 82.81 ms, checked step 91.72 ms (+8.91 ms per checked step)
- False alarms during the slowdown runs (clean training): **0** in 13,223 checks, worst clean z 2.70

## Kill criterion #1

- False alarms so far (no soak yet): 0 (0 in 13,223 slowdown-run checks; 0 in harm-study clean runs; 0 unexplained in faulty runs)
- Slowdown of 3% or less: MET at 100% of matmuls every 50 steps (2.0% of all matmuls checked, +2.03%, the higher of the estimated and measured numbers)
