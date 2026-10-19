# Precise mode, held-out validation

Criteria (pre-registered): 0 alarms and worst score at most 4 on every GPU. Met on 2 of 5: h100, b200.

## NVIDIA H100 80GB HBM3: 4356 held-out checks, 0 alarms, worst 3.476 (criteria MET: 0 alarms, worst at most 4)

### wide, 8,192 tokens, Zipf tokens: NVIDIA H100 80GB HBM3 (bfloat16), 12 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 612 | 0 | 3.476 | 2.679 | 0.557%, 1.91% |
| current | 612 | 10 | 23.852 | 16.053 | 0.249%, 0.279% |
| sqrt_steps | 612 | 0 | 4.232 | 3.653 | 0.303%, 0.94% |
| shipped | 612 | 0 | 3.972 | 3.438 | 0.557%, 1.89% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| qkv backward dW | 2.55 | 23.85 | 3.41 | 2.55 |
| down backward dW | 3.48 | 9.00 | 4.23 | 3.97 |
| up backward dW | 1.88 | 7.63 | 2.01 | 1.88 |
| proj backward dW | 1.49 | 4.18 | 2.02 | 1.92 |
| head forward | 2.09 | 4.01 | 3.24 | 2.55 |
| head backward dW | 1.11 | 3.53 | 1.27 | 1.11 |
| down backward dX | 1.21 | 2.32 | 1.95 | 1.63 |
| head backward dX | 0.25 | 2.22 | 0.51 | 0.25 |
| up forward | 0.87 | 1.91 | 1.23 | 0.87 |
| qkv backward dX | 0.43 | 1.30 | 0.66 | 0.43 |
| up backward dX | 0.46 | 1.11 | 0.60 | 0.46 |
| proj forward | 0.37 | 1.11 | 0.60 | 0.37 |
| down forward | 0.60 | 1.08 | 0.68 | 0.60 |
| proj backward dX | 0.35 | 0.86 | 0.52 | 0.35 |
| qkv forward | 0.32 | 0.79 | 0.45 | 0.32 |

### calibration shape, new seed, Zipf tokens: NVIDIA H100 80GB HBM3 (bfloat16), 20 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 1980 | 0 | 1.803 | 1.671 | 0.322%, 1.1% |
| current | 1980 | 76 | 22.889 | 18.403 | 0.135%, 0.159% |
| sqrt_steps | 1980 | 0 | 2.497 | 2.274 | 0.202%, 0.539% |
| shipped | 1980 | 0 | 1.843 | 1.82 | 0.323%, 1.11% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| qkv backward dW | 1.80 | 22.89 | 2.50 | 1.84 |
| down backward dW | 1.32 | 7.49 | 1.90 | 1.38 |
| head backward dW | 1.34 | 4.76 | 1.72 | 1.40 |
| up backward dW | 0.54 | 2.46 | 0.77 | 0.66 |
| proj backward dW | 0.54 | 2.42 | 0.65 | 0.54 |
| head backward dX | 0.24 | 2.15 | 0.50 | 0.24 |
| head forward | 0.93 | 1.76 | 1.36 | 1.00 |
| qkv backward dX | 0.18 | 1.13 | 0.36 | 0.18 |
| down backward dX | 0.29 | 1.08 | 0.52 | 0.29 |
| up backward dX | 0.18 | 1.04 | 0.32 | 0.18 |
| down forward | 0.30 | 0.94 | 0.44 | 0.30 |
| up forward | 0.23 | 0.77 | 0.40 | 0.23 |
| qkv forward | 0.12 | 0.64 | 0.25 | 0.12 |
| proj backward dX | 0.11 | 0.56 | 0.22 | 0.11 |
| proj forward | 0.14 | 0.50 | 0.24 | 0.14 |

### narrow and deep, 8,192 tokens: NVIDIA H100 80GB HBM3 (bfloat16), 12 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 1764 | 0 | 1.855 | 1.748 | 0.423%, 2.45% |
| current | 1764 | 85 | 19.453 | 16.097 | 0.0762%, 0.289% |
| sqrt_steps | 1764 | 0 | 2.202 | 1.975 | 0.213%, 1.17% |
| shipped | 1764 | 0 | 2.098 | 1.881 | 0.422%, 2.45% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| qkv backward dW | 1.22 | 19.45 | 1.80 | 1.22 |
| head backward dW | 1.86 | 9.99 | 2.20 | 2.10 |
| down backward dW | 0.69 | 7.18 | 1.22 | 0.69 |
| up backward dW | 0.30 | 3.59 | 0.52 | 0.30 |
| head backward dX | 0.22 | 1.72 | 0.46 | 0.22 |
| proj backward dW | 0.11 | 1.48 | 0.18 | 0.11 |
| down backward dX | 0.37 | 1.09 | 0.62 | 0.37 |
| qkv backward dX | 0.15 | 0.85 | 0.30 | 0.15 |
| up backward dX | 0.17 | 0.84 | 0.33 | 0.17 |
| down forward | 0.24 | 0.67 | 0.38 | 0.24 |
| qkv forward | 0.17 | 0.45 | 0.27 | 0.17 |
| up forward | 0.12 | 0.45 | 0.22 | 0.12 |
| head forward | 0.11 | 0.42 | 0.20 | 0.11 |
| proj forward | 0.15 | 0.40 | 0.24 | 0.15 |
| proj backward dX | 0.10 | 0.40 | 0.19 | 0.10 |

Overhead (median seconds per step, same model): {"seconds_per_step": {"unchecked": 0.0281, "default check, every step": 0.0345, "precise mode, every step": 0.1142}, "overhead_pct": {"default check, every step": 22.8, "precise mode, every step": 306.4}}

## NVIDIA B200: 4356 held-out checks, 0 alarms, worst 3.462 (criteria MET: 0 alarms, worst at most 4)

### wide, 8,192 tokens, Zipf tokens: NVIDIA B200 (bfloat16), 12 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 612 | 0 | 3.462 | 2.681 | 0.558%, 1.91% |
| current | 612 | 9 | 12.281 | 11.379 | 0.249%, 0.279% |
| sqrt_steps | 612 | 0 | 4.214 | 3.656 | 0.303%, 0.937% |
| shipped | 612 | 0 | 3.955 | 3.441 | 0.557%, 1.9% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| qkv backward dW | 2.17 | 12.28 | 2.42 | 2.18 |
| down backward dW | 3.46 | 10.35 | 4.21 | 3.95 |
| up backward dW | 1.23 | 6.26 | 1.66 | 1.23 |
| head forward | 2.10 | 4.01 | 3.25 | 2.56 |
| head backward dW | 0.74 | 3.30 | 0.84 | 0.74 |
| proj backward dW | 2.23 | 3.00 | 2.36 | 2.23 |
| down backward dX | 1.22 | 2.41 | 1.93 | 1.63 |
| up forward | 1.05 | 2.02 | 1.48 | 1.05 |
| qkv backward dX | 0.47 | 1.77 | 0.79 | 0.47 |
| head backward dX | 0.17 | 1.64 | 0.36 | 0.17 |
| down forward | 0.92 | 1.54 | 1.16 | 0.92 |
| up backward dX | 0.38 | 1.14 | 0.53 | 0.38 |
| proj forward | 0.36 | 0.93 | 0.54 | 0.36 |
| proj backward dX | 0.34 | 0.76 | 0.49 | 0.34 |
| qkv forward | 0.33 | 0.74 | 0.46 | 0.33 |

### calibration shape, new seed, Zipf tokens: NVIDIA B200 (bfloat16), 20 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 1980 | 0 | 2.139 | 1.743 | 0.322%, 1.1% |
| current | 1980 | 76 | 31.577 | 19.136 | 0.135%, 0.159% |
| sqrt_steps | 1980 | 0 | 3.495 | 2.674 | 0.202%, 0.539% |
| shipped | 1980 | 0 | 2.139 | 1.843 | 0.323%, 1.11% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| qkv backward dW | 2.14 | 31.58 | 3.50 | 2.14 |
| down backward dW | 1.32 | 7.49 | 1.90 | 1.38 |
| head backward dW | 1.66 | 5.11 | 1.77 | 1.66 |
| up backward dW | 0.56 | 3.42 | 0.77 | 0.66 |
| proj backward dW | 0.50 | 2.42 | 0.62 | 0.51 |
| head backward dX | 0.23 | 2.09 | 0.48 | 0.23 |
| head forward | 0.93 | 1.76 | 1.36 | 1.00 |
| qkv backward dX | 0.19 | 1.17 | 0.39 | 0.19 |
| down backward dX | 0.29 | 1.08 | 0.52 | 0.29 |
| up backward dX | 0.19 | 1.06 | 0.33 | 0.19 |
| down forward | 0.37 | 0.94 | 0.54 | 0.37 |
| up forward | 0.23 | 0.77 | 0.40 | 0.23 |
| qkv forward | 0.15 | 0.62 | 0.26 | 0.15 |
| proj forward | 0.17 | 0.54 | 0.29 | 0.17 |
| proj backward dX | 0.10 | 0.53 | 0.20 | 0.10 |

### narrow and deep, 8,192 tokens: NVIDIA B200 (bfloat16), 12 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 1764 | 0 | 1.744 | 1.476 | 0.423%, 2.45% |
| current | 1764 | 73 | 17.678 | 16.951 | 0.0762%, 0.289% |
| sqrt_steps | 1764 | 0 | 2.195 | 1.964 | 0.213%, 1.17% |
| shipped | 1764 | 0 | 2.091 | 1.871 | 0.422%, 2.45% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| qkv backward dW | 1.11 | 17.68 | 1.74 | 1.19 |
| head backward dW | 1.74 | 9.22 | 2.19 | 2.09 |
| down backward dW | 0.69 | 7.34 | 1.21 | 0.69 |
| up backward dW | 0.27 | 3.69 | 0.41 | 0.27 |
| proj backward dW | 0.27 | 3.47 | 0.45 | 0.27 |
| head backward dX | 0.20 | 1.61 | 0.43 | 0.20 |
| down backward dX | 0.36 | 1.13 | 0.62 | 0.36 |
| up backward dX | 0.18 | 0.85 | 0.33 | 0.18 |
| qkv backward dX | 0.15 | 0.82 | 0.30 | 0.15 |
| down forward | 0.25 | 0.70 | 0.38 | 0.25 |
| qkv forward | 0.14 | 0.45 | 0.23 | 0.14 |
| up forward | 0.12 | 0.44 | 0.22 | 0.12 |
| head forward | 0.11 | 0.42 | 0.21 | 0.11 |
| proj backward dX | 0.10 | 0.41 | 0.19 | 0.10 |
| proj forward | 0.15 | 0.38 | 0.24 | 0.15 |

Overhead (median seconds per step, same model): {"seconds_per_step": {"unchecked": 0.0167, "default check, every step": 0.1459, "precise mode, every step": 0.1487}, "overhead_pct": {"default check, every step": 773.7, "precise mode, every step": 790.4}}

## NVIDIA A100-SXM4-80GB: 4356 held-out checks, 0 alarms, worst 4.27 (criteria NOT MET: 0 alarms, worst at most 4)

### wide, 8,192 tokens, Zipf tokens: NVIDIA A100-SXM4-80GB (bfloat16), 12 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 612 | 0 | 4.27 | 3.448 | 0.641%, 2.11% |
| current | 612 | 17 | 23.01 | 22.37 | 0.456%, 0.486% |
| sqrt_steps | 612 | 0 | 5.214 | 4.725 | 0.526%, 1.43% |
| shipped | 612 | 0 | 5.083 | 4.615 | 0.642%, 2.09% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| head backward dW | 3.21 | 23.01 | 3.28 | 3.21 |
| qkv backward dW | 3.28 | 13.21 | 3.51 | 3.28 |
| up backward dW | 1.88 | 8.52 | 1.93 | 1.88 |
| down backward dW | 4.27 | 6.69 | 5.21 | 5.08 |
| head backward dX | 0.43 | 2.31 | 0.64 | 0.43 |
| proj backward dW | 0.43 | 1.92 | 0.55 | 0.53 |
| down backward dX | 0.99 | 1.83 | 1.57 | 1.43 |
| down forward | 1.10 | 1.51 | 1.15 | 1.10 |
| up forward | 0.85 | 1.51 | 1.15 | 0.98 |
| qkv backward dX | 0.50 | 1.17 | 0.64 | 0.50 |
| up backward dX | 0.44 | 0.85 | 0.53 | 0.44 |
| head forward | 0.42 | 0.80 | 0.67 | 0.60 |
| proj forward | 0.39 | 0.73 | 0.47 | 0.39 |
| qkv forward | 0.33 | 0.56 | 0.38 | 0.33 |
| proj backward dX | 0.28 | 0.54 | 0.34 | 0.28 |

### calibration shape, new seed, Zipf tokens: NVIDIA A100-SXM4-80GB (bfloat16), 20 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 1980 | 0 | 2.622 | 2.534 | 0.399%, 1.21% |
| current | 1980 | 88 | 40.703 | 29.011 | 0.238%, 0.263% |
| sqrt_steps | 1980 | 0 | 3.784 | 3.088 | 0.334%, 0.81% |
| shipped | 1980 | 0 | 2.622 | 2.534 | 0.404%, 1.21% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| qkv backward dW | 2.62 | 40.70 | 3.78 | 2.62 |
| head backward dW | 2.18 | 9.05 | 2.51 | 2.18 |
| down backward dW | 1.24 | 7.50 | 1.48 | 1.27 |
| up backward dW | 0.91 | 2.44 | 0.98 | 0.92 |
| head forward | 0.75 | 1.40 | 1.10 | 0.94 |
| qkv backward dX | 0.21 | 0.86 | 0.30 | 0.21 |
| up backward dX | 0.18 | 0.79 | 0.26 | 0.18 |
| down backward dX | 0.28 | 0.78 | 0.38 | 0.28 |
| proj backward dW | 0.12 | 0.73 | 0.14 | 0.12 |
| down forward | 0.31 | 0.67 | 0.36 | 0.31 |
| up forward | 0.22 | 0.56 | 0.30 | 0.22 |
| qkv forward | 0.13 | 0.43 | 0.18 | 0.13 |
| proj forward | 0.14 | 0.39 | 0.20 | 0.14 |
| proj backward dX | 0.11 | 0.38 | 0.15 | 0.11 |
| head backward dX | 0.05 | 0.28 | 0.08 | 0.05 |

### narrow and deep, 8,192 tokens: NVIDIA A100-SXM4-80GB (bfloat16), 12 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 1764 | 0 | 1.209 | 0.738 | 0.468%, 2.66% |
| current | 1764 | 29 | 17.735 | 13.939 | 0.115%, 0.496% |
| sqrt_steps | 1764 | 0 | 1.232 | 1.131 | 0.316%, 1.76% |
| shipped | 1764 | 0 | 1.209 | 0.973 | 0.467%, 2.66% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| head backward dW | 1.21 | 17.73 | 1.23 | 1.21 |
| qkv backward dW | 0.73 | 13.40 | 1.14 | 0.73 |
| down backward dW | 0.19 | 1.37 | 0.26 | 0.19 |
| up backward dW | 0.08 | 1.11 | 0.10 | 0.08 |
| down backward dX | 0.36 | 0.86 | 0.48 | 0.36 |
| qkv backward dX | 0.18 | 0.71 | 0.25 | 0.18 |
| up backward dX | 0.22 | 0.70 | 0.31 | 0.22 |
| head backward dX | 0.13 | 0.65 | 0.20 | 0.13 |
| down forward | 0.33 | 0.60 | 0.41 | 0.33 |
| proj backward dW | 0.04 | 0.57 | 0.05 | 0.04 |
| up forward | 0.12 | 0.37 | 0.17 | 0.12 |
| qkv forward | 0.14 | 0.34 | 0.18 | 0.14 |
| head forward | 0.11 | 0.34 | 0.16 | 0.11 |
| proj backward dX | 0.10 | 0.33 | 0.14 | 0.10 |
| proj forward | 0.14 | 0.33 | 0.19 | 0.14 |

Overhead (median seconds per step, same model): {"seconds_per_step": {"unchecked": 0.0558, "default check, every step": 0.0782, "precise mode, every step": 0.2881}, "overhead_pct": {"default check, every step": 40.1, "precise mode, every step": 416.3}}

## NVIDIA L4: 4356 held-out checks, 0 alarms, worst 4.26 (criteria NOT MET: 0 alarms, worst at most 4)

### wide, 8,192 tokens, Zipf tokens: NVIDIA L4 (bfloat16), 12 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 612 | 0 | 4.26 | 3.441 | 0.641%, 2.15% |
| current | 612 | 15 | 17.254 | 15.54 | 0.456%, 0.487% |
| sqrt_steps | 612 | 0 | 5.204 | 4.715 | 0.524%, 1.45% |
| shipped | 612 | 0 | 5.075 | 4.606 | 0.642%, 2.13% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| head backward dW | 3.07 | 17.25 | 3.13 | 3.07 |
| qkv backward dW | 0.94 | 9.62 | 1.25 | 0.94 |
| down backward dW | 4.26 | 9.45 | 5.20 | 5.08 |
| up backward dW | 1.74 | 8.03 | 1.78 | 1.74 |
| proj backward dW | 1.01 | 2.68 | 1.04 | 1.02 |
| down backward dX | 1.00 | 1.86 | 1.62 | 1.48 |
| head forward | 0.88 | 1.66 | 1.38 | 1.23 |
| qkv backward dX | 0.61 | 1.60 | 0.74 | 0.61 |
| up forward | 0.87 | 1.58 | 1.21 | 1.03 |
| down forward | 0.77 | 1.49 | 0.87 | 0.77 |
| head backward dX | 0.18 | 0.99 | 0.27 | 0.18 |
| up backward dX | 0.47 | 0.97 | 0.52 | 0.47 |
| proj forward | 0.41 | 0.74 | 0.51 | 0.41 |
| qkv forward | 0.31 | 0.55 | 0.37 | 0.31 |
| proj backward dX | 0.30 | 0.51 | 0.35 | 0.30 |

### calibration shape, new seed, Zipf tokens: NVIDIA L4 (bfloat16), 20 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 1980 | 0 | 1.112 | 0.973 | 0.399%, 1.21% |
| current | 1980 | 37 | 18.465 | 13.33 | 0.238%, 0.263% |
| sqrt_steps | 1980 | 0 | 1.778 | 1.567 | 0.334%, 0.81% |
| shipped | 1980 | 0 | 1.112 | 1.005 | 0.404%, 1.21% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| qkv backward dW | 1.11 | 18.46 | 1.78 | 1.11 |
| head backward dW | 0.92 | 8.96 | 1.25 | 1.01 |
| down backward dW | 0.54 | 3.93 | 0.76 | 0.65 |
| head forward | 0.74 | 1.38 | 1.08 | 0.93 |
| up backward dW | 0.26 | 1.27 | 0.34 | 0.32 |
| down backward dX | 0.28 | 0.79 | 0.39 | 0.28 |
| head backward dX | 0.14 | 0.73 | 0.20 | 0.14 |
| up forward | 0.22 | 0.57 | 0.30 | 0.22 |
| qkv forward | 0.14 | 0.45 | 0.19 | 0.14 |
| proj backward dW | 0.08 | 0.38 | 0.09 | 0.08 |
| up backward dX | 0.10 | 0.37 | 0.14 | 0.10 |
| qkv backward dX | 0.08 | 0.34 | 0.12 | 0.08 |
| down forward | 0.14 | 0.33 | 0.18 | 0.14 |
| proj forward | 0.07 | 0.18 | 0.09 | 0.07 |
| proj backward dX | 0.05 | 0.17 | 0.07 | 0.05 |

### narrow and deep, 8,192 tokens: NVIDIA L4 (bfloat16), 12 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 1764 | 0 | 0.429 | 0.345 | 0.468%, 2.66% |
| current | 1764 | 5 | 9.573 | 7.626 | 0.115%, 0.496% |
| sqrt_steps | 1764 | 0 | 0.665 | 0.505 | 0.316%, 1.76% |
| shipped | 1764 | 0 | 0.429 | 0.345 | 0.467%, 2.66% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | shipped |
|---|---|---|---|---|
| head backward dW | 0.43 | 9.57 | 0.67 | 0.43 |
| qkv backward dW | 0.26 | 4.98 | 0.40 | 0.26 |
| down backward dX | 0.35 | 0.87 | 0.46 | 0.35 |
| head backward dX | 0.10 | 0.48 | 0.14 | 0.10 |
| up backward dW | 0.04 | 0.43 | 0.05 | 0.04 |
| down backward dW | 0.08 | 0.43 | 0.10 | 0.08 |
| proj forward | 0.21 | 0.38 | 0.26 | 0.21 |
| up backward dX | 0.10 | 0.37 | 0.15 | 0.10 |
| up forward | 0.12 | 0.36 | 0.17 | 0.12 |
| qkv forward | 0.14 | 0.34 | 0.18 | 0.14 |
| head forward | 0.11 | 0.33 | 0.16 | 0.11 |
| proj backward dX | 0.09 | 0.32 | 0.14 | 0.09 |
| qkv backward dX | 0.08 | 0.30 | 0.11 | 0.08 |
| proj backward dW | 0.03 | 0.28 | 0.03 | 0.03 |
| down forward | 0.15 | 0.27 | 0.19 | 0.15 |

Overhead (median seconds per step, same model): {"seconds_per_step": {"unchecked": 0.2664, "default check, every step": 0.2896, "precise mode, every step": 0.9607}, "overhead_pct": {"default check, every step": 8.7, "precise mode, every step": 260.6}}

## t4: could not run: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.54 GiB. GPU 0 has a total capacity of 14.57 GiB of which 669.62 MiB is free. Process 1 has 13.91 GiB memory in use. Of the allocated memory 11.28 GiB is allocated by PyTorch, and 2.50 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memo
