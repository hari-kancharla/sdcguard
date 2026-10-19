# Trying sdcguard on your GPUs

Start with sdcguard's synthetic workloads; no access to customer jobs or data is needed unless you choose
step 3. Allow time for installation and reviewing the results as well as the check itself.

## 1. Check every GPU on a few nodes (5 minutes, unattended)

```bash
pip install sdcguard
sdcguard check --offline --out sdcguard_check  # every visible GPU, in parallel; about 5 minutes plus setup
```

Each GPU gets a verdict (PASS, FAIL, SUSPECT or INCOMPLETE), a card and a `result.json`. Start with
`result.json` if you choose to share anything: it is the minimal result card. It contains the GPU model,
driver, software versions, provider and region when detected, counts, speed and health readings, notes, and
a short hash of the GPU UUID. It omits dedicated hostname, serial, PCI and full UUID fields. Review it first:
provider labels and error text in notes can still disclose operational details.

`--offline` disables the read-only request for the public board's reference data. The check does not upload
its results. Uploading any file requires your explicit opt-in. `--share` prepares a GitHub issue link for you
to review; the card is published only if you submit that issue.

To collect many nodes into one page:

```bash
sdcguard report node*/sdcguard_check --out fleet_report     # fleet_report.html, quarantine.txt, rma/
```

Keep this report private by default: it includes hostnames, full GPU identifiers, PCI addresses, serial
numbers and local paths when available. Support bundles under `rma/` can include saved tensor operands and
outputs. Inspect and redact these files before explicitly choosing to send them to anyone.

To review a copied incident bundle, extract it and keep each evidence `.pt` beside its `.json`. Select one
actual file from the extracted directory and run `sdcguard replay /path/to/evidence_0.pt`. Replay recomputes
the saved operation and reports what the CPU check and current device found. A passing recomputation does
not establish that the original cause was a physical GPU defect. A signature shown in the fleet page is
only present, not verified there; use `sdcguard verify /path/to/certificate.json --pubkey /path/to/trusted.pub`
with the complete bundle and a public key you trust when verification matters.

## 2. Optional: as a node health check

`sdcguard nhc` runs a short screen and prints one line (exit 0 OK, 1 wrong results, 2 couldn't tell), for
Kubernetes node-problem-detector or Slurm's HealthCheckProgram. Examples in [deploy/](../deploy/).

## 3. Optional: inside one training job

```python
from sdcguard import SDCGuard
guard = SDCGuard(every=50, log_path="sdcguard_events.jsonl").protect(model)
...
with guard.on_step(step):
    loss = model(x, y); loss.backward()
```

On selected steps, supported `nn.Linear` matmuls (forward, dX and dW) are checked. Review skipped or unjudged
counts and the coverage limits in the README. Alerts go to the JSON-lines file:
layer name, matrix shapes, the position in the output, the score and the GPU's identity, never tensor values.

## What to send back (review first; sharing is optional)

- The selected `result.json` files are enough to start. Do not substitute a fleet report or support bundle
  without inspecting its machine identifiers, paths and any tensor data and explicitly agreeing to share it.
- For step 3: the checked, skipped and unjudged counts and step time with and without the guard. An empty
  events file alone does not demonstrate coverage. Share the events file only after reviewing its layer
  names and GPU identifiers.
- Five answers, a sentence each:
  1. How did you find the last GPU that computed wrong results (or how would you)?
  2. What does one cost you: hours lost, jobs rerun, credits paid, an RMA you couldn't get?
  3. What do you run today for acceptance and between jobs (DCGM level, burn-in, NCCL tests, your own)?
  4. Did anything sdcguard reported surprise you, or look wrong to you?
  5. What would make this worth paying for: scheduled checks, private fleet history, RMA evidence, alerts into
     your own tooling, signed acceptance reports for customers, something else?

## What the verdicts mean, and don't

PASS means the implemented checks found no failure in the work they judged, and the required supported
parts completed. It does not guarantee a healthy GPU or exact output correctness. A short run can miss an
intermittent fault or an error outside the detector's coverage. FAIL records a failed numerical or memory
check, or a relevant reported hardware error; SUSPECT records a repeatability discrepancy; INCOMPLETE means
some required work could not be run or judged.

Available failing matmul inputs and outputs are saved for replay; a dead CUDA context can prevent saving
some evidence. Inspect the actual bundle. Investigate an unexpected FAIL before deciding what to do with
the node: possible causes include hardware, backend software, configuration and the checker itself.

`--simulate-faults` exists for demos only: its results are labelled SIMULATED everywhere and the board rejects
them. It demonstrates detection and evidence handling, not a defect in the GPU. Simulated fleet results
remain visible but do not enter the quarantine list.
