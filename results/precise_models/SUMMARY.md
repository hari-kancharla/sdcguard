# Precise mode: noise models on real training matmuls

## NVIDIA H100 80GB HBM3 (bfloat16), 30 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 2970 | 74 | 17.892 | 16.142 | 0.135%, 0.163% |
| current | 2970 | 78 | 17.892 | 16.447 | 0.135%, 0.163% |
| sqrt_steps | 2970 | 0 | 3.501 | 3.02 | 0.421%, 0.592% |
| running_S16 | 2970 | 0 | 5.924 | 5.041 | 0.135%, 0.167% |
| running_S8 | 2970 | 0 | 5.93 | 5.046 | 0.135%, 0.167% |
| running_S4 | 2970 | 0 | 5.941 | 5.055 | 0.135%, 0.167% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | running_S16 | running_S8 | running_S4 |
|---|---|---|---|---|---|---|
| qkv backward dW | 17.89 | 17.89 | 3.50 | 2.91 | 2.87 | 3.10 |
| head backward dW | 6.02 | 6.02 | 2.14 | 5.92 | 5.93 | 5.94 |
| up backward dW | 3.50 | 3.50 | 0.67 | 1.55 | 1.56 | 1.58 |
| down backward dW | 3.27 | 3.27 | 0.96 | 2.55 | 2.54 | 2.54 |
| proj backward dW | 2.64 | 2.64 | 0.43 | 0.98 | 0.99 | 0.98 |
| head backward dX | 1.76 | 1.76 | 0.41 | 1.74 | 1.74 | 1.75 |
| qkv backward dX | 1.13 | 1.16 | 0.35 | 1.15 | 1.15 | 1.14 |
| up backward dX | 1.03 | 1.03 | 0.33 | 1.01 | 1.01 | 1.01 |
| down forward | 0.96 | 0.96 | 0.40 | 0.94 | 0.94 | 0.94 |
| down backward dX | 0.73 | 0.94 | 0.41 | 0.93 | 0.93 | 0.92 |
| head forward | 0.66 | 0.83 | 0.36 | 0.82 | 0.82 | 0.82 |
| up forward | 0.60 | 0.62 | 0.24 | 0.61 | 0.61 | 0.60 |
| qkv forward | 0.57 | 0.61 | 0.26 | 0.61 | 0.61 | 0.61 |
| proj backward dX | 0.55 | 0.55 | 0.22 | 0.54 | 0.54 | 0.54 |
| proj forward | 0.50 | 0.54 | 0.25 | 0.53 | 0.53 | 0.53 |

## NVIDIA B200 (bfloat16), 30 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 2970 | 72 | 17.892 | 15.573 | 0.135%, 0.163% |
| current | 2970 | 76 | 17.892 | 15.872 | 0.135%, 0.163% |
| sqrt_steps | 2970 | 0 | 3.501 | 2.92 | 0.421%, 0.592% |
| running_S16 | 2970 | 0 | 5.127 | 4.52 | 0.135%, 0.167% |
| running_S8 | 2970 | 0 | 5.131 | 4.446 | 0.135%, 0.167% |
| running_S4 | 2970 | 0 | 5.139 | 4.458 | 0.135%, 0.167% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | running_S16 | running_S8 | running_S4 |
|---|---|---|---|---|---|---|
| qkv backward dW | 17.89 | 17.89 | 3.50 | 2.28 | 2.28 | 2.18 |
| head backward dW | 4.53 | 5.19 | 1.62 | 5.13 | 5.13 | 5.14 |
| down backward dW | 3.27 | 3.27 | 0.96 | 2.55 | 2.54 | 2.54 |
| proj backward dW | 2.64 | 2.64 | 0.45 | 1.00 | 1.00 | 1.00 |
| up backward dW | 1.96 | 2.09 | 0.44 | 1.10 | 1.07 | 1.13 |
| head backward dX | 1.85 | 1.85 | 0.43 | 1.85 | 1.83 | 1.83 |
| qkv backward dX | 1.22 | 1.22 | 0.38 | 1.21 | 1.21 | 1.20 |
| up backward dX | 1.00 | 1.03 | 0.33 | 1.01 | 1.01 | 1.01 |
| down backward dX | 0.73 | 0.94 | 0.41 | 0.93 | 0.93 | 0.92 |
| down forward | 0.89 | 0.93 | 0.37 | 0.92 | 0.91 | 0.91 |
| head forward | 0.66 | 0.83 | 0.36 | 0.82 | 0.82 | 0.82 |
| qkv forward | 0.67 | 0.67 | 0.26 | 0.67 | 0.67 | 0.66 |
| proj backward dX | 0.63 | 0.63 | 0.24 | 0.62 | 0.61 | 0.61 |
| up forward | 0.60 | 0.62 | 0.24 | 0.61 | 0.61 | 0.60 |
| proj forward | 0.56 | 0.56 | 0.26 | 0.56 | 0.56 | 0.56 |

## NVIDIA A100-SXM4-80GB (bfloat16), 30 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 2970 | 84 | 23.834 | 18.636 | 0.238%, 0.266% |
| current | 2970 | 94 | 23.834 | 22.474 | 0.238%, 0.266% |
| sqrt_steps | 2970 | 0 | 5.227 | 3.886 | 0.631%, 0.885% |
| running_S16 | 2970 | 6 | 7.801 | 6.466 | 0.238%, 0.273% |
| running_S8 | 2970 | 6 | 7.807 | 6.308 | 0.238%, 0.274% |
| running_S4 | 2970 | 7 | 7.82 | 6.326 | 0.238%, 0.274% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | running_S16 | running_S8 | running_S4 |
|---|---|---|---|---|---|---|
| qkv backward dW | 23.83 | 23.83 | 5.23 | 3.05 | 3.05 | 3.05 |
| head backward dW | 7.90 | 7.90 | 2.81 | 7.80 | 7.81 | 7.82 |
| down backward dW | 2.39 | 2.39 | 0.78 | 1.87 | 1.86 | 1.86 |
| up backward dW | 2.33 | 2.33 | 0.48 | 1.00 | 1.00 | 0.99 |
| qkv backward dX | 0.90 | 0.90 | 0.30 | 0.89 | 0.88 | 0.88 |
| up backward dX | 0.78 | 0.80 | 0.28 | 0.79 | 0.78 | 0.78 |
| down forward | 0.72 | 0.72 | 0.31 | 0.72 | 0.72 | 0.72 |
| down backward dX | 0.48 | 0.65 | 0.28 | 0.64 | 0.64 | 0.64 |
| head forward | 0.47 | 0.64 | 0.28 | 0.64 | 0.64 | 0.64 |
| proj backward dW | 0.53 | 0.53 | 0.10 | 0.32 | 0.32 | 0.33 |
| up forward | 0.49 | 0.49 | 0.19 | 0.48 | 0.48 | 0.48 |
| qkv forward | 0.46 | 0.46 | 0.20 | 0.45 | 0.45 | 0.45 |
| proj forward | 0.42 | 0.42 | 0.22 | 0.41 | 0.41 | 0.41 |
| proj backward dX | 0.38 | 0.39 | 0.15 | 0.38 | 0.38 | 0.37 |
| head backward dX | 0.25 | 0.28 | 0.08 | 0.28 | 0.28 | 0.28 |

## NVIDIA L4 (bfloat16), 30 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 2970 | 6 | 9.262 | 6.6 | 0.238%, 0.266% |
| current | 2970 | 8 | 9.262 | 6.709 | 0.238%, 0.266% |
| sqrt_steps | 2970 | 0 | 1.962 | 1.589 | 0.631%, 0.885% |
| running_S16 | 2970 | 0 | 3.682 | 3.401 | 0.238%, 0.273% |
| running_S8 | 2970 | 0 | 3.682 | 3.401 | 0.238%, 0.274% |
| running_S4 | 2970 | 0 | 3.682 | 3.401 | 0.238%, 0.274% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | running_S16 | running_S8 | running_S4 |
|---|---|---|---|---|---|---|
| qkv backward dW | 9.26 | 9.26 | 1.96 | 1.97 | 1.93 | 2.48 |
| head backward dW | 3.68 | 3.68 | 1.25 | 3.68 | 3.68 | 3.68 |
| down backward dW | 0.93 | 1.06 | 0.36 | 0.90 | 0.90 | 0.90 |
| up backward dW | 0.94 | 0.94 | 0.21 | 0.76 | 0.76 | 0.76 |
| down backward dX | 0.49 | 0.66 | 0.29 | 0.66 | 0.66 | 0.65 |
| head backward dX | 0.63 | 0.65 | 0.18 | 0.65 | 0.65 | 0.65 |
| head forward | 0.46 | 0.60 | 0.27 | 0.60 | 0.60 | 0.60 |
| qkv forward | 0.43 | 0.49 | 0.21 | 0.48 | 0.48 | 0.48 |
| up forward | 0.42 | 0.46 | 0.17 | 0.45 | 0.45 | 0.45 |
| proj backward dW | 0.41 | 0.41 | 0.05 | 0.19 | 0.18 | 0.22 |
| up backward dX | 0.37 | 0.38 | 0.14 | 0.38 | 0.38 | 0.38 |
| qkv backward dX | 0.34 | 0.34 | 0.11 | 0.34 | 0.34 | 0.34 |
| down forward | 0.33 | 0.33 | 0.14 | 0.32 | 0.32 | 0.32 |
| proj forward | 0.19 | 0.19 | 0.09 | 0.18 | 0.18 | 0.18 |
| proj backward dX | 0.18 | 0.18 | 0.07 | 0.18 | 0.18 | 0.18 |

## Tesla T4 (float16), 30 clean training steps

| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |
|---|---|---|---|---|---|
| current_as_shipped | 2970 | 0 | 1.53 | 1.433 | 0.365%, 0.473% |
| current | 2970 | 0 | 2.194 | 1.968 | 0.364%, 0.473% |
| sqrt_steps | 2970 | 0 | 0.833 | 0.805 | 0.975%, 1.36% |
| running_S16 | 2970 | 0 | 2.153 | 1.898 | 0.365%, 0.486% |
| running_S8 | 2970 | 0 | 2.152 | 1.892 | 0.367%, 0.487% |
| running_S4 | 2970 | 0 | 2.14 | 1.898 | 0.367%, 0.488% |

Worst score by layer and pass:

| layer, pass | current_as_shipped | current | sqrt_steps | running_S16 | running_S8 | running_S4 |
|---|---|---|---|---|---|---|
| down backward dW | 1.53 | 2.19 | 0.83 | 2.15 | 2.15 | 2.14 |
| up backward dW | 1.40 | 1.40 | 0.51 | 1.36 | 1.35 | 1.35 |
| down backward dX | 1.24 | 1.35 | 0.54 | 1.29 | 1.29 | 1.29 |
| head forward | 0.66 | 1.10 | 0.77 | 1.10 | 1.10 | 1.10 |
| proj backward dX | 0.88 | 0.88 | 0.36 | 0.85 | 0.85 | 0.85 |
| qkv backward dX | 0.86 | 0.86 | 0.33 | 0.86 | 0.86 | 0.86 |
| up backward dX | 0.82 | 0.82 | 0.30 | 0.80 | 0.80 | 0.80 |
| qkv backward dW | 0.77 | 0.77 | 0.32 | 0.74 | 0.74 | 0.74 |
| up forward | 0.51 | 0.51 | 0.21 | 0.50 | 0.50 | 0.50 |
| head backward dW | 0.49 | 0.49 | 0.21 | 0.49 | 0.49 | 0.49 |
| qkv forward | 0.44 | 0.46 | 0.23 | 0.45 | 0.45 | 0.45 |
| proj forward | 0.44 | 0.44 | 0.22 | 0.42 | 0.43 | 0.42 |
| proj backward dW | 0.31 | 0.37 | 0.11 | 0.34 | 0.34 | 0.33 |
| down forward | 0.23 | 0.24 | 0.13 | 0.24 | 0.24 | 0.24 |
| head backward dX | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
