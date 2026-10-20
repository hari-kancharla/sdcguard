# The 60-second demo

Use one supported NVIDIA GPU with enough free memory for the selected workload, plus a laptop for the last
shot. Record the terminal and speed up the waiting parts. Keep the command, revision, device and actual
verdict visible. A demonstration of a simulated fault is not evidence of a physically defective GPU.
Install the release from PyPI and keep its version visible.

| time | show | command | say |
|---|---|---|---|
| 0-8 s | the GPU and the install | `nvidia-smi -L`, then `python -m pip install "sdcguard[all]"` | A rented GPU. Does it compute correctly? |
| 8-20 s | a real check, sped up | `sdcguard check --offline --minutes 2 --out demo_check` | A two-minute target, plus setup: synthetic training and serving, supported math modes against a float64 reference, memory and repeatability. Show the actual verdict, completed modes and counts. |
| 20-32 s | a labelled simulated fault | `sdcguard check --offline --minutes 1 --simulate-faults 0.001 --out demo_fail` | Here the configured injection probability is 0.1% per eligible matmul result. This is software simulation. Show the SIMULATED label, actual failures and saved evidence. |
| 32-42 s | portable evidence | `sdcguard replay /path/to/copied/evidence_0.pt` | I copied one saved incident, with its neighboring JSON metadata, to another directory. Replay checks the recorded output and recomputes the operation. Show the actual mismatch and recomputation counts; the physical cause remains unproven. |
| 42-50 s | the operator view | `sdcguard report demo_fail --out fleet` then open `fleet/fleet_report.html` | The fleet view links the incident and support bundle. This run is simulated, so it is excluded from the quarantine candidates. Real incidents need operator investigation. |
| 50-60 s | in training | `python -m sdcguard.demo` (laptop CPU) | This example introduces simulated corruption at step 45. The example's recovery code uses a guard alert to roll back to a checkpoint; show the layer, detection step and final outcome from this run. |

Replace `/path/to/copied/evidence_0.pt` with one actual `.pt` file from the copied incident. Do not expand a
multi-GPU wildcard into several arguments to the single-file replay command. Keep the `.pt` and `.json`
together. Narrate the result you obtained, including an unresolved or unjudged result; do not prewrite a
successful CPU confirmation.

Optional extra clip: if you have a recorded MPS/cuBLASLt reproduction, show the specific revision, software
versions, device configuration and observed results from `results/mps_bug`. Link the primary report and
the actual reproduction log. Describe it as a software-bug reproduction; it is not a known defective GPU.

Use a reviewed `result.json` as the minimal card to invite others to share. It includes environment and
provider details and may include error text, so inspect it first. The fleet report is not anonymous:
it can expose hostnames, full GPU identifiers, serials and paths; incident bundles may contain tensors.
No uploads or issue submissions should happen without explicit opt-in. `--offline` avoids fetching the
board reference, and `--share` only prepares an issue link for the user to review and submit.

Don't claim: that a PASS guarantees a healthy GPU, that sdcguard has caught a physically defective GPU in the
field (it hasn't been tried on one yet), or that the simulated FAIL is a real fault.
