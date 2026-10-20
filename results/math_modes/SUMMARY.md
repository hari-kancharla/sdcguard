# Math modes on real GPUs

Clean results only. Worst score: the largest distance from the CPU's float64 answer, in units of the noise model (the alarm is at 6). Typical sensitivity: the size of an error in one output that the check typically catches (median over blocks), relative to a typical output, for the least sensitive input distribution. Not a guarantee for every position or every error pattern.

| GPU | mode | matmuls | wrong | worst score | typical sensitivity |
|---|---|---|---|---|---|
| Tesla T4 | bf16 | not run: needs compute capability 8.0 or newer | | | |
| Tesla T4 | fp16 | 672 | 0 | 0.51 | 0.13% |
| Tesla T4 | tf32 | not run: needs compute capability 8.0 or newer | | | |
| Tesla T4 | fp32 | 672 | 0 | 0.62 | 0.026% |
| Tesla T4 | fp64 | 672 | 0 | 0.65 | 9.67e-11% |
| Tesla T4 | int8 | not run: NotImplementedError: "baddbmm_cuda" not implemented for 'Long' | | | |
| Tesla T4 | fp8 | not run: needs compute capability 8.9 or newer | | | |
| NVIDIA L4 | bf16 | 396 | 0 | 0.31 | 0.0778% |
| NVIDIA L4 | fp16 | 396 | 0 | 0.43 | 0.0778% |
| NVIDIA L4 | tf32 | 396 | 0 | 0.95 | 0.13% |
| NVIDIA L4 | fp32 | 396 | 0 | 0.89 | 0.026% |
| NVIDIA L4 | fp64 | 396 | 0 | 0.66 | 9.67e-11% |
| NVIDIA L4 | int8 | not run: NotImplementedError: "baddbmm_cuda" not implemented for 'Long' | | | |
| NVIDIA L4 | fp8 | 396 | 0 | 0.27 | 20% |
| NVIDIA A100-SXM4-80GB | bf16 | 384 | 0 | 0.68 | 0.0778% |
| NVIDIA A100-SXM4-80GB | fp16 | 384 | 0 | 0.44 | 0.0778% |
| NVIDIA A100-SXM4-80GB | tf32 | 384 | 0 | 1.01 | 0.13% |
| NVIDIA A100-SXM4-80GB | fp32 | 384 | 0 | 0.87 | 0.026% |
| NVIDIA A100-SXM4-80GB | fp64 | 384 | 0 | 1.18 | 9.67e-11% |
| NVIDIA A100-SXM4-80GB | int8 | not run: NotImplementedError: "baddbmm_cuda" not implemented for 'Long' | | | |
| NVIDIA A100-SXM4-80GB | fp8 | not run: needs compute capability 8.9 or newer | | | |
| NVIDIA H100 80GB HBM3 | bf16 | 756 | 0 | 0.76 | 0.0519% |
| NVIDIA H100 80GB HBM3 | fp16 | 756 | 0 | 0.92 | 0.0519% |
| NVIDIA H100 80GB HBM3 | tf32 | 756 | 0 | 1.20 | 0.0778% |
| NVIDIA H100 80GB HBM3 | fp32 | 756 | 0 | 0.89 | 0.026% |
| NVIDIA H100 80GB HBM3 | fp64 | 756 | 0 | 0.66 | 9.67e-11% |
| NVIDIA H100 80GB HBM3 | int8 | not run: NotImplementedError: "baddbmm_cuda" not implemented for 'Long' | | | |
| NVIDIA H100 80GB HBM3 | fp8 | 756 | 0 | 0.30 | 16.6% |
| NVIDIA B200 | bf16 | 780 | 0 | 0.76 | 0.0519% |
| NVIDIA B200 | fp16 | 780 | 0 | 0.92 | 0.0519% |
| NVIDIA B200 | tf32 | 780 | 0 | 1.20 | 0.0778% |
| NVIDIA B200 | fp32 | 780 | 0 | 0.89 | 0.026% |
| NVIDIA B200 | fp64 | 780 | 0 | 0.66 | 9.67e-11% |
| NVIDIA B200 | int8 | not run: NotImplementedError: "baddbmm_cuda" not implemented for 'Long' | | | |
| NVIDIA B200 | fp8 | 780 | 0 | 0.06 | 0.0162% |


# Sensitivity: fp16 result vs fp32 accumulator (cuda)

Tesla T4, torch 2.14.1+cu130, matmuls of 2048 x 1024 by 1024 x 2048, 32 trials per error size (6 input distributions; 16 skipped because the fp16 result overflowed). An error is relative to a typical output of its block.

| error size | fp16 check flags | fp32 check flags | changed the fp16 output |
|---|---|---|---|
| 0.0001% | 0% | 0% | 0% |
| 0.0003% | 0% | 0% | 0% |
| 0.001% | 0% | 0% | 3% |
| 0.003% | 0% | 0% | 9% |
| 0.01% | 0% | 0% | 12% |
| 0.03% | 0% | 0% | 53% |
| 0.1% | 0% | 0% | 94% |
| 0.3% | 0% | 100% | 100% |
| 1% | 0% | 100% | 100% |
| 3% | 0% | 100% | 100% |
| 10% | 100% | 100% | 100% |
| 30% | 100% | 100% | 100% |
| 100% | 100% | 100% | 100% |

Half the errors are caught from about 5.5% (fp16 check) and 0.17% (fp32 check).
Clean trials: fp16 check 0 false alarms (worst score 1.37), fp32 check 0 (worst score 0.42); the alarm is at 6.


# Precise mode on Tesla T4 (float16)

Model: {'vocab': 32000, 'layers': 8, 'width': 1024, 'heads': 16, 'seq': 512, 'batch': 8}. torch 2.14.1+cu130.

Clean: 30 steps, 2970 fp32 checks, 0 false alarms, worst score 1.53 (99th percentile 1.12; the alarm is at 6).

Same step unguarded vs precise: logits identical False (largest difference 0.00195), gradients largest relative difference 0.815.

Time per step: unguarded 0.2603 s, bf16 checks 0.3033 s, precise checks 1.9552 s. At every=20: bf16 0.83%, precise 32.55%.

| injected faults | injected | checked | caught | largest missed change | smallest caught change |
|---|---|---|---|---|---|
| bf16 | 13 | 13 | 12 | 1.1451744285295717e-05 | 0.018940526992082596 |
| precise | 13 | 13 | 11 | 2.3534626052423846e-06 | 0.002213102765381336 |


# Sensitivity: bf16 result vs fp32 accumulator (cuda)

NVIDIA L4, torch 2.14.1+cu130, matmuls of 2048 x 1024 by 1024 x 2048, 48 trials per error size (6 input distributions). An error is relative to a typical output of its block.

| error size | bf16 check flags | fp32 check flags | changed the bf16 output |
|---|---|---|---|
| 0.0001% | 0% | 0% | 0% |
| 0.0003% | 0% | 0% | 0% |
| 0.001% | 0% | 0% | 0% |
| 0.003% | 0% | 0% | 0% |
| 0.01% | 0% | 0% | 4% |
| 0.03% | 0% | 0% | 15% |
| 0.1% | 0% | 100% | 38% |
| 0.3% | 0% | 100% | 75% |
| 1% | 0% | 100% | 98% |
| 3% | 0% | 100% | 100% |
| 10% | 0% | 100% | 100% |
| 30% | 81% | 100% | 100% |
| 100% | 100% | 100% | 100% |

Half the errors are caught from about 20% (bf16 check) and 0.055% (fp32 check).
Clean trials: bf16 check 0 false alarms (worst score 2.08), fp32 check 0 (worst score 0.30); the alarm is at 6.


# Precise mode on NVIDIA L4 (bfloat16)

Model: {'vocab': 32000, 'layers': 8, 'width': 1024, 'heads': 16, 'seq': 512, 'batch': 8}. torch 2.14.1+cu130.

Clean: 30 steps, 2970 fp32 checks, 3 false alarms, worst score 6.661 (99th percentile 3.862; the alarm is at 6).

Same step unguarded vs precise: logits identical False (largest difference 0.0156), gradients largest relative difference 0.0097.

Time per step: unguarded 0.2155 s, bf16 checks 0.2366 s, precise checks 0.9081 s. At every=20: bf16 0.49%, precise 16.07%.

| injected faults | injected | checked | caught | largest missed change | smallest caught change |
|---|---|---|---|---|---|
| bf16 | 13 | 13 | 11 | 0.13846690952777863 | 0.11583486199378967 |
| precise | 13 | 13 | 11 | 9.342168596049305e-06 | 0.000852597935590893 |


# Sensitivity: bf16 result vs fp32 accumulator (cuda)

NVIDIA A100-SXM4-80GB, torch 2.14.1+cu130, matmuls of 2048 x 1024 by 1024 x 2048, 48 trials per error size (6 input distributions). An error is relative to a typical output of its block.

| error size | bf16 check flags | fp32 check flags | changed the bf16 output |
|---|---|---|---|
| 0.0001% | 0% | 0% | 0% |
| 0.0003% | 0% | 0% | 0% |
| 0.001% | 0% | 0% | 0% |
| 0.003% | 0% | 0% | 0% |
| 0.01% | 0% | 0% | 4% |
| 0.03% | 0% | 0% | 15% |
| 0.1% | 0% | 100% | 38% |
| 0.3% | 0% | 100% | 75% |
| 1% | 0% | 100% | 98% |
| 3% | 0% | 100% | 100% |
| 10% | 0% | 100% | 100% |
| 30% | 79% | 100% | 100% |
| 100% | 100% | 100% | 100% |

Half the errors are caught from about 20% (bf16 check) and 0.055% (fp32 check).
Clean trials: bf16 check 0 false alarms (worst score 1.93), fp32 check 0 (worst score 0.58); the alarm is at 6.


# Precise mode on NVIDIA A100-SXM4-80GB (bfloat16)

Model: {'vocab': 32000, 'layers': 8, 'width': 1024, 'heads': 16, 'seq': 512, 'batch': 8}. torch 2.14.1+cu130.

Clean: 30 steps, 2970 fp32 checks, 78 false alarms, worst score 20.263 (99th percentile 9.449; the alarm is at 6).

Same step unguarded vs precise: logits identical True (largest difference 0), gradients largest relative difference 0.0406.

Time per step: unguarded 0.048 s, bf16 checks 0.0585 s, precise checks 0.1676 s. At every=20: bf16 1.1%, precise 12.45%.

| injected faults | injected | checked | caught | largest missed change | smallest caught change |
|---|---|---|---|---|---|
| bf16 | 13 | 13 | 12 | 0.0005466019501909614 | 0.11581313610076904 |
| precise | 13 | 13 | 11 | 9.272565876017325e-06 | 0.000852643046528101 |


# Sensitivity: bf16 result vs fp32 accumulator (cuda)

NVIDIA H100 80GB HBM3, torch 2.14.1+cu130, matmuls of 2048 x 1024 by 1024 x 2048, 48 trials per error size (6 input distributions). An error is relative to a typical output of its block.

| error size | bf16 check flags | fp32 check flags | changed the bf16 output |
|---|---|---|---|
| 0.0001% | 0% | 0% | 0% |
| 0.0003% | 0% | 0% | 0% |
| 0.001% | 0% | 0% | 0% |
| 0.003% | 0% | 0% | 0% |
| 0.01% | 0% | 0% | 4% |
| 0.03% | 0% | 0% | 15% |
| 0.1% | 0% | 100% | 38% |
| 0.3% | 0% | 100% | 75% |
| 1% | 0% | 100% | 98% |
| 3% | 0% | 100% | 100% |
| 10% | 0% | 100% | 100% |
| 30% | 77% | 100% | 100% |
| 100% | 100% | 100% | 100% |

Half the errors are caught from about 20% (bf16 check) and 0.055% (fp32 check).
Clean trials: bf16 check 0 false alarms (worst score 1.97), fp32 check 0 (worst score 0.70); the alarm is at 6.


# Precise mode on NVIDIA H100 80GB HBM3 (bfloat16)

Model: {'vocab': 32000, 'layers': 8, 'width': 1024, 'heads': 16, 'seq': 512, 'batch': 8}. torch 2.14.1+cu130.

Clean: 30 steps, 2970 fp32 checks, 76 false alarms, worst score 17.892 (99th percentile 8.941; the alarm is at 6).

Same step unguarded vs precise: logits identical True (largest difference 0), gradients largest relative difference 0.

Time per step: unguarded 0.0238 s, bf16 checks 0.0352 s, precise checks 0.1376 s. At every=20: bf16 2.4%, precise 23.92%.

| injected faults | injected | checked | caught | largest missed change | smallest caught change |
|---|---|---|---|---|---|
| bf16 | 13 | 13 | 12 | 0.0005463649285957217 | 0.11636568605899811 |
| precise | 13 | 13 | 11 | 9.289207810070366e-06 | 0.0008523077704012394 |


# Sensitivity: bf16 result vs fp32 accumulator (cuda)

NVIDIA B200, torch 2.14.1+cu130, matmuls of 2048 x 1024 by 1024 x 2048, 48 trials per error size (6 input distributions). An error is relative to a typical output of its block.

| error size | bf16 check flags | fp32 check flags | changed the bf16 output |
|---|---|---|---|
| 0.0001% | 0% | 0% | 0% |
| 0.0003% | 0% | 0% | 0% |
| 0.001% | 0% | 0% | 0% |
| 0.003% | 0% | 0% | 0% |
| 0.01% | 0% | 0% | 4% |
| 0.03% | 0% | 0% | 15% |
| 0.1% | 0% | 100% | 38% |
| 0.3% | 0% | 100% | 75% |
| 1% | 0% | 100% | 98% |
| 3% | 0% | 100% | 100% |
| 10% | 0% | 100% | 100% |
| 30% | 83% | 100% | 100% |
| 100% | 100% | 100% | 100% |

Half the errors are caught from about 19% (bf16 check) and 0.055% (fp32 check).
Clean trials: bf16 check 0 false alarms (worst score 2.13), fp32 check 0 (worst score 0.73); the alarm is at 6.


# Precise mode on NVIDIA B200 (bfloat16)

Model: {'vocab': 32000, 'layers': 8, 'width': 1024, 'heads': 16, 'seq': 512, 'batch': 8}. torch 2.14.1+cu130.

Clean: 30 steps, 2970 fp32 checks, 70 false alarms, worst score 17.892 (99th percentile 8.542; the alarm is at 6).

Same step unguarded vs precise: logits identical True (largest difference 0), gradients largest relative difference 0.

Time per step: unguarded 0.0147 s, bf16 checks 0.1206 s, precise checks 0.1336 s. At every=20: bf16 36.02%, precise 40.46%.

| injected faults | injected | checked | caught | largest missed change | smallest caught change |
|---|---|---|---|---|---|
| bf16 | 13 | 13 | 11 | 0.13920532166957855 | 0.11636568605899811 |
| precise | 13 | 13 | 11 | 9.289207810070366e-06 | 0.0008523077704012394 |

