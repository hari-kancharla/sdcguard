# Calibration: ts-gpt2

## Slowdown (NVIDIA H100 80GB HBM3, base 83.24 ms/step)

| setting | share of matmuls checked | slowdown (estimated) | slowdown (measured) |
|---|---|---|---|
| 100% of matmuls every 1 steps | 100.0% | +37.84% |  |
| 100% of matmuls every 2 steps | 50.0% | +18.92% | +19.48% |
| 100% of matmuls every 5 steps | 20.0% | +7.57% | +7.64% |
| 100% of matmuls every 10 steps | 10.0% | +3.78% | +3.87% |
| 100% of matmuls every 20 steps | 5.0% | +1.89% | +1.74% |
| 100% of matmuls every 50 steps | 2.0% | +0.76% | +0.89% |

- 100% of matmuls checked: plain step 83.24 ms, checked step 114.75 ms (+31.5 ms per checked step)
- False alarms during the slowdown runs (clean training): **0** in 20,367 checks, worst clean z 2.86

## Kill criterion #1

- False alarms so far (no soak yet): 0 (0 in 20,367 slowdown-run checks; 0 in harm-study clean runs; 0 unexplained in faulty runs)
- Slowdown of 3% or less: MET at 100% of matmuls every 20 steps (5.0% of all matmuls checked, +1.89%, the higher of the estimated and measured numbers)
