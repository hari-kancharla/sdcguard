# Default check vs precise mode, same matmuls, code fingerprint 9c9e05a04892

No pass line: the reporting rules were written before the run (docs/METHOD.md). Precise mode stays experimental whatever this shows.

# Default check vs precise mode, same matmuls (NVIDIA H100 80GB HBM3)

Code 17c41a10c789, torch 2.14.1+cu130. 198 linear-layer matmuls (forward, dX, dW) of 2 training steps, 3 random outputs each, so 594 corrupted matmuls per error size. An error is relative to a typical output of its 128-column block, added before rounding to bfloat16.

| error size | default check flags | precise mode flags | changed the bfloat16 output |
|---|---|---|---|
| 0.0001% | 0% | 0% | 0% |
| 0.0003% | 0% | 0% | 0% |
| 0.001% | 0% | 0% | 0% |
| 0.003% | 0% | 0% | 2% |
| 0.01% | 0% | 0% | 7% |
| 0.03% | 0% | 0% | 16% |
| 0.1% | 0% | 0% | 39% |
| 0.3% | 0% | 30% | 71% |
| 1% | 0% | 58% | 99% |
| 3% | 0% | 99% | 100% |
| 10% | 0% | 100% | 100% |
| 30% | 51% | 100% | 100% |
| 100% | 100% | 100% | 100% |

Nominal 95% Wilson intervals for sampled positions in these fixed matmuls. Matmuls have different detection probabilities; these are not uncertainty bounds for new training runs, models, hardware faults or the whole sensitivity curve.
Intervals across error sizes share the same injected positions and do not form a simultaneous band.
| error size | trials | changed trials | default 95% | precise 95% | default 95%, changed | precise 95%, changed |
|---|---|---|---|---|---|---|
| 0.0001% | 594 | 0 | [0%, 1%] | [0%, 1%] | [n/a, n/a] | [n/a, n/a] |
| 0.0003% | 594 | 1 | [0%, 1%] | [0%, 1%] | [0%, 79%] | [0%, 79%] |
| 0.001% | 594 | 2 | [0%, 1%] | [0%, 1%] | [0%, 66%] | [0%, 66%] |
| 0.003% | 594 | 13 | [0%, 1%] | [0%, 1%] | [0%, 23%] | [0%, 23%] |
| 0.01% | 594 | 43 | [0%, 1%] | [0%, 1%] | [0%, 8%] | [0%, 8%] |
| 0.03% | 594 | 98 | [0%, 1%] | [0%, 1%] | [0%, 4%] | [0%, 4%] |
| 0.1% | 594 | 232 | [0%, 1%] | [0%, 1%] | [0%, 2%] | [0%, 2%] |
| 0.3% | 594 | 424 | [0%, 1%] | [27%, 34%] | [0%, 1%] | [26%, 34%] |
| 1% | 594 | 586 | [0%, 1%] | [54%, 61%] | [0%, 1%] | [53%, 61%] |
| 3% | 594 | 594 | [0%, 1%] | [98%, 100%] | [0%, 1%] | [98%, 100%] |
| 10% | 594 | 594 | [0%, 1%] | [99%, 100%] | [0%, 1%] | [99%, 100%] |
| 30% | 594 | 594 | [47%, 55%] | [99%, 100%] | [47%, 55%] | [99%, 100%] |
| 100% | 594 | 594 | [99%, 100%] | [99%, 100%] | [99%, 100%] | [99%, 100%] |

Descriptive 50% crossings: 29% (default) and 0.71% (precise); 90% crossings: 78% and 2.4%.
Among errors that changed the bfloat16 output (the ones the model sees): 50% crossings: 29% (default) and 0.73% (precise); 90% crossings: 78% and 2.4%.
Crossings describe this sampled curve with log-linear interpolation. A crossing at the smallest tested size is an upper bound on the crossing size; detection need not increase monotonically.
Clean matmuls: default 0 false alarms in 198 (worst score 2.27), precise 0 in 198 (worst 0.52); the alarm is at 6.
Unjudged clean checks: default 0, precise 0. Unjudged injected checks by size: default [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], precise [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Positions with unavailable block scale: 0.

| per training step | seconds | vs unchecked |
|---|---|---|
| unchecked | 0.0282 |  |
| default check, every step | 0.0447 | +58.5% |
| precise mode, every step | 0.1765 | +525.9% |

# Default check vs precise mode, same matmuls (NVIDIA A100-SXM4-80GB)

Code 17c41a10c789, torch 2.14.1+cu130. 198 linear-layer matmuls (forward, dX, dW) of 2 training steps, 3 random outputs each, so 594 corrupted matmuls per error size. An error is relative to a typical output of its 128-column block, added before rounding to bfloat16.

| error size | default check flags | precise mode flags | changed the bfloat16 output |
|---|---|---|---|
| 0.0001% | 0% | 0% | 0% |
| 0.0003% | 0% | 0% | 0% |
| 0.001% | 0% | 0% | 1% |
| 0.003% | 0% | 0% | 3% |
| 0.01% | 0% | 0% | 8% |
| 0.03% | 0% | 0% | 18% |
| 0.1% | 0% | 0% | 41% |
| 0.3% | 0% | 4% | 74% |
| 1% | 0% | 53% | 98% |
| 3% | 0% | 99% | 100% |
| 10% | 0% | 100% | 100% |
| 30% | 52% | 100% | 100% |
| 100% | 100% | 100% | 100% |

Nominal 95% Wilson intervals for sampled positions in these fixed matmuls. Matmuls have different detection probabilities; these are not uncertainty bounds for new training runs, models, hardware faults or the whole sensitivity curve.
Intervals across error sizes share the same injected positions and do not form a simultaneous band.
| error size | trials | changed trials | default 95% | precise 95% | default 95%, changed | precise 95%, changed |
|---|---|---|---|---|---|---|
| 0.0001% | 594 | 2 | [0%, 1%] | [0%, 1%] | [0%, 66%] | [0%, 66%] |
| 0.0003% | 594 | 2 | [0%, 1%] | [0%, 1%] | [0%, 66%] | [0%, 66%] |
| 0.001% | 594 | 4 | [0%, 1%] | [0%, 1%] | [0%, 49%] | [0%, 49%] |
| 0.003% | 594 | 15 | [0%, 1%] | [0%, 1%] | [0%, 20%] | [0%, 20%] |
| 0.01% | 594 | 50 | [0%, 1%] | [0%, 1%] | [0%, 7%] | [0%, 7%] |
| 0.03% | 594 | 106 | [0%, 1%] | [0%, 1%] | [0%, 4%] | [0%, 4%] |
| 0.1% | 594 | 245 | [0%, 1%] | [0%, 1%] | [0%, 2%] | [0%, 2%] |
| 0.3% | 594 | 438 | [0%, 1%] | [3%, 6%] | [0%, 1%] | [2%, 6%] |
| 1% | 594 | 585 | [0%, 1%] | [49%, 57%] | [0%, 1%] | [49%, 57%] |
| 3% | 594 | 594 | [0%, 1%] | [98%, 100%] | [0%, 1%] | [98%, 100%] |
| 10% | 594 | 594 | [0%, 1%] | [99%, 100%] | [0%, 1%] | [99%, 100%] |
| 30% | 594 | 594 | [48%, 56%] | [99%, 100%] | [48%, 56%] | [99%, 100%] |
| 100% | 594 | 594 | [99%, 100%] | [99%, 100%] | [99%, 100%] | [99%, 100%] |

Descriptive 50% crossings: 29% (default) and 0.92% (precise); 90% crossings: 78% and 2.4%.
Among errors that changed the bfloat16 output (the ones the model sees): 50% crossings: 29% (default) and 0.93% (precise); 90% crossings: 78% and 2.4%.
Crossings describe this sampled curve with log-linear interpolation. A crossing at the smallest tested size is an upper bound on the crossing size; detection need not increase monotonically.
Clean matmuls: default 0 false alarms in 198 (worst score 2.11), precise 0 in 198 (worst 0.73); the alarm is at 6.
Unjudged clean checks: default 0, precise 0. Unjudged injected checks by size: default [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], precise [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Positions with unavailable block scale: 0.

| per training step | seconds | vs unchecked |
|---|---|---|
| unchecked | 0.0558 |  |
| default check, every step | 0.0733 | +31.4% |
| precise mode, every step | 0.2442 | +337.6% |

# Default check vs precise mode, same matmuls (NVIDIA L4)

Code 17c41a10c789, torch 2.14.1+cu130. 198 linear-layer matmuls (forward, dX, dW) of 2 training steps, 3 random outputs each, so 594 corrupted matmuls per error size. An error is relative to a typical output of its 128-column block, added before rounding to bfloat16.

| error size | default check flags | precise mode flags | changed the bfloat16 output |
|---|---|---|---|
| 0.0001% | 0% | 0% | 0% |
| 0.0003% | 0% | 0% | 0% |
| 0.001% | 0% | 0% | 1% |
| 0.003% | 0% | 0% | 2% |
| 0.01% | 0% | 0% | 7% |
| 0.03% | 0% | 0% | 19% |
| 0.1% | 0% | 0% | 43% |
| 0.3% | 0% | 5% | 74% |
| 1% | 0% | 53% | 98% |
| 3% | 0% | 99% | 100% |
| 10% | 0% | 100% | 100% |
| 30% | 52% | 100% | 100% |
| 100% | 100% | 100% | 100% |

Nominal 95% Wilson intervals for sampled positions in these fixed matmuls. Matmuls have different detection probabilities; these are not uncertainty bounds for new training runs, models, hardware faults or the whole sensitivity curve.
Intervals across error sizes share the same injected positions and do not form a simultaneous band.
| error size | trials | changed trials | default 95% | precise 95% | default 95%, changed | precise 95%, changed |
|---|---|---|---|---|---|---|
| 0.0001% | 594 | 1 | [0%, 1%] | [0%, 1%] | [0%, 79%] | [0%, 79%] |
| 0.0003% | 594 | 1 | [0%, 1%] | [0%, 1%] | [0%, 79%] | [0%, 79%] |
| 0.001% | 594 | 3 | [0%, 1%] | [0%, 1%] | [0%, 56%] | [0%, 56%] |
| 0.003% | 594 | 13 | [0%, 1%] | [0%, 1%] | [0%, 23%] | [0%, 23%] |
| 0.01% | 594 | 44 | [0%, 1%] | [0%, 1%] | [0%, 8%] | [0%, 8%] |
| 0.03% | 594 | 112 | [0%, 1%] | [0%, 1%] | [0%, 3%] | [0%, 3%] |
| 0.1% | 594 | 253 | [0%, 1%] | [0%, 1%] | [0%, 2%] | [0%, 2%] |
| 0.3% | 594 | 437 | [0%, 1%] | [3%, 7%] | [0%, 1%] | [2%, 6%] |
| 1% | 594 | 585 | [0%, 1%] | [49%, 57%] | [0%, 1%] | [49%, 57%] |
| 3% | 594 | 594 | [0%, 1%] | [98%, 100%] | [0%, 1%] | [98%, 100%] |
| 10% | 594 | 594 | [0%, 1%] | [99%, 100%] | [0%, 1%] | [99%, 100%] |
| 30% | 594 | 594 | [48%, 56%] | [99%, 100%] | [48%, 56%] | [99%, 100%] |
| 100% | 594 | 594 | [99%, 100%] | [99%, 100%] | [99%, 100%] | [99%, 100%] |

Descriptive 50% crossings: 29% (default) and 0.92% (precise); 90% crossings: 78% and 2.4%.
Among errors that changed the bfloat16 output (the ones the model sees): 50% crossings: 29% (default) and 0.92% (precise); 90% crossings: 78% and 2.4%.
Crossings describe this sampled curve with log-linear interpolation. A crossing at the smallest tested size is an upper bound on the crossing size; detection need not increase monotonically.
Clean matmuls: default 0 false alarms in 198 (worst score 2.13), precise 0 in 198 (worst 0.25); the alarm is at 6.
Unjudged clean checks: default 0, precise 0. Unjudged injected checks by size: default [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], precise [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Positions with unavailable block scale: 0.

| per training step | seconds | vs unchecked |
|---|---|---|
| unchecked | 0.268 |  |
| default check, every step | 0.2897 | +8.1% |
| precise mode, every step | 0.9604 | +258.4% |
