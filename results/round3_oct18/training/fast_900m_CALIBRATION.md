# Calibration: ts-900m

## Slowdown (NVIDIA H100 80GB HBM3, base 163.73 ms/step)

| setting | share of matmuls checked | slowdown (estimated) | slowdown (measured) |
|---|---|---|---|
| 100% of matmuls every 1 steps | 100.0% | +25.77% |  |
| 100% of matmuls every 2 steps | 50.0% | +12.88% | +12.54% |
| 100% of matmuls every 5 steps | 20.0% | +5.15% | +5.01% |
| 100% of matmuls every 10 steps | 10.0% | +2.58% | +2.64% |
| 100% of matmuls every 20 steps | 5.0% | +1.29% | +1.26% |

- 100% of matmuls checked: plain step 163.73 ms, checked step 205.92 ms (+42.19 ms per checked step)
- False alarms during the slowdown runs (clean training): **0** in 17,169 checks, worst clean z 2.67

## Kill criterion #1

- False alarms so far (no soak yet): 0 (0 in 17,169 slowdown-run checks; 0 in harm-study clean runs; 0 unexplained in faulty runs)
- Slowdown of 3% or less: MET at 100% of matmuls every 10 steps (10.0% of all matmuls checked, +2.64%, the higher of the estimated and measured numbers)
