# Precise mode, check by check

## NVIDIA H100 80GB HBM3 (bfloat16): 792 checks over 8 steps, 0 alarms

| layer, pass | m x k x n | tensor-core steps of | checks | over 6 | worst |
|---|---|---|---|---|---|
| qkv backward dW | 3072 x 4096 x 1024 | 16 | 64 | 0 | 4.92 |
| head backward dW | 32000 x 4096 x 1024 | 16 | 8 | 0 | 3.34 |
| down backward dW | 1024 x 4096 x 4096 | 16 | 64 | 0 | 2.45 |
| proj backward dW | 1024 x 4096 x 1024 | 16 | 64 | 0 | 2.05 |
| head backward dX | 4096 x 32000 x 1024 | 16 | 8 | 0 | 1.71 |
| qkv backward dX | 4096 x 3072 x 1024 | 16 | 64 | 0 | 1.11 |
| up backward dW | 4096 x 4096 x 1024 | 16 | 64 | 0 | 1.10 |
| up backward dX | 4096 x 4096 x 1024 | 16 | 64 | 0 | 1.09 |
| down forward | 4096 x 4096 x 1024 | 16 | 64 | 0 | 0.90 |
| head forward | 4096 x 1024 x 32000 | 16 | 8 | 0 | 0.66 |
| qkv forward | 4096 x 1024 x 3072 | 16 | 64 | 0 | 0.59 |
| proj backward dX | 4096 x 1024 x 1024 | 16 | 64 | 0 | 0.56 |
| up forward | 4096 x 1024 x 4096 | 16 | 64 | 0 | 0.54 |
| down backward dX | 4096 x 1024 x 4096 | 16 | 64 | 0 | 0.53 |
| proj forward | 4096 x 1024 x 1024 | 16 | 64 | 0 | 0.48 |

- blocks.7.qkv backward dW step 6: score 4.923 (plain 5.08, signed 4.92); coherence 1.84 (median 0.82); toward zero 0.836; error in u x |output| median 1065.8; same bits rerun True; plain fp32 FMA score 2.077; kernels ['nvjet_sm90_tss_128x192_64x5_2x1_v_bz_coopB_NTN']
- blocks.7.qkv backward dW step 7: score 4.83 (plain 4.74, signed 4.83); coherence 0.78 (median 0.85); toward zero 0.672; error in u x |output| median 1233.6; same bits rerun True; plain fp32 FMA score 2.785; kernels ['nvjet_sm90_tss_128x192_64x5_2x1_v_bz_coopB_NTN']
- blocks.5.qkv backward dW step 6: score 4.57 (plain 4.59, signed 4.57); coherence 1.55 (median 0.78); toward zero 0.672; error in u x |output| median 968.4; same bits rerun True; plain fp32 FMA score 1.746; kernels ['nvjet_sm90_tss_128x192_64x5_2x1_v_bz_coopB_NTN']
- blocks.6.qkv backward dW step 7: score 4.552 (plain 4.17, signed 4.55); coherence 0.98 (median 0.84); toward zero 0.438; error in u x |output| median 1260.7; same bits rerun True; plain fp32 FMA score 2.233; kernels ['nvjet_sm90_tss_128x192_64x5_2x1_v_bz_coopB_NTN']
- blocks.7.qkv backward dW step 4: score 4.324 (plain 4.89, signed 4.23); coherence 1.93 (median 0.85); toward zero 0.492; error in u x |output| median 1212.9; same bits rerun True; plain fp32 FMA score 1.879; kernels []
- blocks.6.qkv backward dW step 6: score 4.231 (plain 4.15, signed 4.23); coherence 0.49 (median 0.81); toward zero 0.211; error in u x |output| median 820.0; same bits rerun True; plain fp32 FMA score 2.248; kernels ['nvjet_sm90_tss_128x192_64x5_2x1_v_bz_coopB_NTN']

math modes: {'bf16': {'matmuls': 24, 'wrong': 0, 'worst_z': 0.711}, 'fp16': {'matmuls': 24, 'wrong': 0, 'worst_z': 0.791}, 'tf32': {'matmuls': 24, 'wrong': 0, 'worst_z': 1.079}, 'fp32': {'matmuls': 24, 'wrong': 0, 'worst_z': 0.743}, 'fp64': {'matmuls': 24, 'wrong': 0, 'worst_z': 0.534}, 'int8': {'matmuls': 24, 'wrong': 0, 'worst_z': None}, 'fp8': {'matmuls': 24, 'wrong': 0, 'worst_z': 0.259}}

## NVIDIA A100-SXM4-80GB (bfloat16): 792 checks over 8 steps, 2 alarms

| layer, pass | m x k x n | tensor-core steps of | checks | over 6 | worst |
|---|---|---|---|---|---|
| qkv backward dW | 3072 x 4096 x 1024 | 8 | 64 | 2 | 6.83 |
| head backward dW | 32000 x 4096 x 1024 | 8 | 8 | 0 | 5.03 |
| down backward dW | 1024 x 4096 x 4096 | 8 | 64 | 0 | 1.57 |
| up backward dW | 4096 x 4096 x 1024 | 8 | 64 | 0 | 1.02 |
| qkv backward dX | 4096 x 3072 x 1024 | 8 | 64 | 0 | 0.88 |
| up backward dX | 4096 x 4096 x 1024 | 8 | 64 | 0 | 0.74 |
| down forward | 4096 x 4096 x 1024 | 8 | 64 | 0 | 0.65 |
| proj backward dW | 1024 x 4096 x 1024 | 8 | 64 | 0 | 0.56 |
| head forward | 4096 x 1024 x 32000 | 8 | 8 | 0 | 0.47 |
| up forward | 4096 x 1024 x 4096 | 8 | 64 | 0 | 0.46 |
| qkv forward | 4096 x 1024 x 3072 | 8 | 64 | 0 | 0.42 |
| down backward dX | 4096 x 1024 x 4096 | 8 | 64 | 0 | 0.40 |
| proj backward dX | 4096 x 1024 x 1024 | 8 | 64 | 0 | 0.37 |
| proj forward | 4096 x 1024 x 1024 | 8 | 64 | 0 | 0.37 |
| head backward dX | 4096 x 32000 x 1024 | 8 | 8 | 0 | 0.26 |

- blocks.6.qkv backward dW step 5: score 6.83 (plain 5.49, signed 6.83); coherence 1.27 (median 0.78); toward zero 0.352; error in u x |output| median 1383.2; same bits rerun True; plain fp32 FMA score 1.605; kernels ['ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_nt']
- blocks.7.qkv backward dW step 5: score 6.582 (plain 6.18, signed 6.58); coherence 1.2 (median 0.81); toward zero 0.75; error in u x |output| median 1600.7; same bits rerun True; plain fp32 FMA score 1.63; kernels ['ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_nt']
- blocks.7.qkv backward dW step 3: score 5.852 (plain 4.78, signed 5.85); coherence 0.38 (median 0.73); toward zero 0.438; error in u x |output| median 1228.6; same bits rerun True; plain fp32 FMA score 1.564; kernels ['ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_nt']
- blocks.7.qkv backward dW step 7: score 5.732 (plain 5.69, signed 5.73); coherence 1.78 (median 0.84); toward zero 0.297; error in u x |output| median 1600.9; same bits rerun True; plain fp32 FMA score 2.158; kernels ['ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_nt']
- blocks.7.qkv backward dW step 6: score 5.488 (plain 4.94, signed 5.49); coherence 1.29 (median 0.82); toward zero 0.859; error in u x |output| median 1578.4; same bits rerun True; plain fp32 FMA score 2.179; kernels ['ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_nt']
- blocks.6.qkv backward dW step 3: score 5.092 (plain 3.74, signed 5.09); coherence 0.06 (median 0.71); toward zero 0.461; error in u x |output| median 893.5; same bits rerun True; plain fp32 FMA score 1.104; kernels ['ampere_s16816gemm_bf16_256x128_ldg8_stages_64x3_nt']

math modes: {'bf16': {'matmuls': 18, 'wrong': 0, 'worst_z': 0.605}, 'fp16': {'matmuls': 18, 'wrong': 0, 'worst_z': 0.376}, 'tf32': {'matmuls': 18, 'wrong': 0, 'worst_z': 0.915}, 'fp32': {'matmuls': 18, 'wrong': 0, 'worst_z': 0.789}, 'fp64': {'matmuls': 18, 'wrong': 0, 'worst_z': 1.096}, 'int8': {'matmuls': 18, 'wrong': 0, 'worst_z': None}, 'fp8': {'skipped': 'needs compute capability 8.9 or newer'}}

## NVIDIA L4 (bfloat16): 792 checks over 8 steps, 0 alarms

| layer, pass | m x k x n | tensor-core steps of | checks | over 6 | worst |
|---|---|---|---|---|---|
| head backward dW | 32000 x 4096 x 1024 | 8 | 8 | 0 | 2.94 |
| qkv backward dW | 3072 x 4096 x 1024 | 8 | 64 | 0 | 2.79 |
| down backward dW | 1024 x 4096 x 4096 | 8 | 64 | 0 | 0.67 |
| up backward dW | 4096 x 4096 x 1024 | 8 | 64 | 0 | 0.66 |
| head backward dX | 4096 x 32000 x 1024 | 8 | 8 | 0 | 0.63 |
| up forward | 4096 x 1024 x 4096 | 8 | 64 | 0 | 0.48 |
| head forward | 4096 x 1024 x 32000 | 8 | 8 | 0 | 0.46 |
| qkv forward | 4096 x 1024 x 3072 | 8 | 64 | 0 | 0.42 |
| down backward dX | 4096 x 1024 x 4096 | 8 | 64 | 0 | 0.38 |
| up backward dX | 4096 x 4096 x 1024 | 8 | 64 | 0 | 0.37 |
| down forward | 4096 x 4096 x 1024 | 8 | 64 | 0 | 0.33 |
| qkv backward dX | 4096 x 3072 x 1024 | 8 | 64 | 0 | 0.32 |
| proj backward dW | 1024 x 4096 x 1024 | 8 | 64 | 0 | 0.25 |
| proj forward | 4096 x 1024 x 1024 | 8 | 64 | 0 | 0.18 |
| proj backward dX | 4096 x 1024 x 1024 | 8 | 64 | 0 | 0.18 |

- blocks.6.qkv backward dW step 2: score 1.054 (plain 0.94, signed 1.05); coherence 1.49 (median 0.69); toward zero 0.758; error in u x |output| median 89.3; same bits rerun True; plain fp32 FMA score 1.295; kernels ['Memset (Device)', 'ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_nt']
- blocks.3.down backward dW step 2: score 0.571 (plain 0.78, signed 0.26); coherence 8.36 (median 3.06); toward zero 0.906; error in u x |output| median 52.9; same bits rerun True; plain fp32 FMA score 0.147; kernels ['Memset (Device)', 'ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_nt']
- blocks.7.up forward step 7: score 0.387 (plain 0.35, signed 0.39); coherence 0.89 (median 0.72); toward zero 0.836; error in u x |output| median 13.9; same bits rerun True; plain fp32 FMA score 0.255; kernels ['ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_tn']
- blocks.1.up forward step 4: score 0.38 (plain 0.38, signed 0.38); coherence 4.19 (median 0.68); toward zero 0.82; error in u x |output| median 14.6; same bits rerun True; plain fp32 FMA score 0.24; kernels ['ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_tn']
- blocks.7.down backward dX step 0: score 0.344 (plain 0.35, signed 0.34); coherence 4.13 (median 0.68); toward zero 0.852; error in u x |output| median 11.1; same bits rerun True; plain fp32 FMA score 0.224; kernels ['ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_nn']
- blocks.0.down backward dX step 5: score 0.328 (plain 0.33, signed 0.33); coherence 3.97 (median 0.71); toward zero 0.797; error in u x |output| median 11.6; same bits rerun True; plain fp32 FMA score 0.246; kernels ['ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_nn']

math modes: {'bf16': {'matmuls': 14, 'wrong': 0, 'worst_z': 0.272}, 'fp16': {'matmuls': 14, 'wrong': 0, 'worst_z': 0.372}, 'tf32': {'matmuls': 14, 'wrong': 0, 'worst_z': 0.846}, 'fp32': {'matmuls': 14, 'wrong': 0, 'worst_z': 0.743}, 'fp64': {'matmuls': 14, 'wrong': 0, 'worst_z': 0.534}, 'int8': {'matmuls': 14, 'wrong': 0, 'worst_z': None}, 'fp8': {'matmuls': 14, 'wrong': 0, 'worst_z': 0.228}}
