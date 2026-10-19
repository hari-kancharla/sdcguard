# The 60-second demo

One GPU (any NVIDIA GPU; an L4 is enough) plus a laptop for the last shot. Record the terminal; speed up the
waiting parts. Every fault here is labelled for what it is: simulated in software, a real software bug, or
(not yet available) a physically defective GPU.

| time | show | command | say |
|---|---|---|---|
| 0-8 s | the GPU and the install | `nvidia-smi -L && pip install sdcguard` | A rented GPU. Does it compute correctly? |
| 8-20 s | a real check, sped up | `sdcguard check --minutes 2` | Five minutes: a training and serving workload with every matmul checked, seven math modes against a float64 answer, memory, speed, repeatability. PASS. |
| 20-32 s | a labelled simulated fault | `sdcguard check --minutes 1 --simulate-faults 0.001 --out demo_fail` | Same check, but sdcguard corrupts 0.1% of matmul outputs in software with NVIDIA's published fault patterns. The card says SIMULATED in red. FAIL, and the failed matmuls are saved. |
| 32-42 s | the evidence | `sdcguard replay demo_fail/*/evidence_0.pt` | Recompute the saved matmul here (or on any GPU): the recorded output fails its check, every recomputation passes, and the CPU confirms the recorded output was wrong. That's the "failure follows the board" evidence NVIDIA's RMA process asks for. |
| 42-50 s | the operator view | `sdcguard report demo_fail --out fleet && open fleet/fleet_report.html` | One page for a fleet: which GPUs to pull (quarantine.txt) and an RMA bundle per failed GPU. |
| 50-60 s | in training | `python -m sdcguard.demo` (laptop CPU), or the 4-GPU chart in the README | A device goes bad at step 45; sdcguard names the layer at the next checked step, training rolls back to a clean checkpoint and finishes. |

Optional 10 seconds, the real bug: on an H200 with MPS limited to 14% of the SMs, a cuBLASLt bug NVIDIA
confirmed (forum thread 384873) leaves outputs unwritten, and sdcguard's training and serving checks alert on
the same step (results/mps_bug). Say it's a software bug, not a bad GPU.

Don't claim: that a PASS guarantees a healthy GPU, that sdcguard has caught a physically defective GPU in the
field (it hasn't been tried on one yet), or that the simulated FAIL is a real fault.
