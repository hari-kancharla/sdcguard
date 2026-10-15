# Calibration: ts-900m

## Slowdown (NVIDIA H100 80GB HBM3, base 163.06 ms/step)

| setting | share of matmuls checked | slowdown (estimated) | slowdown (measured) |
|---|---|---|---|
| 100% of matmuls every 1 steps | 100.0% | +54.28% |  |
| 100% of matmuls every 10 steps | 10.0% | +5.43% | +5.47% |
| 100% of matmuls every 20 steps | 5.0% | +2.71% | +2.60% |
| 100% of matmuls every 50 steps | 2.0% | +1.09% | +1.02% |

- 100% of matmuls checked: plain step 163.06 ms, checked step 251.57 ms (+88.51 ms per checked step)
- False alarms during the slowdown runs (clean training): **0** in 7,857 checks, worst clean z 2.72

## Kill criterion #1

- False alarms so far (no soak yet): 0 (0 in 7,857 slowdown-run checks; 0 in harm-study clean runs; 0 unexplained in faulty runs)
- Slowdown of 3% or less: MET at 100% of matmuls every 20 steps (5.0% of all matmuls checked, +2.71%, the higher of the estimated and measured numbers)
