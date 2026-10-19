# Trying sdcguard on your GPUs

A trial takes about 30 minutes of your time and needs no access to your data or your customers' jobs.
Everything below runs sdcguard's own synthetic workloads unless you choose step 3.

## 1. Check every GPU on a few nodes (5 minutes, unattended)

```bash
pip install sdcguard
sdcguard check --out sdcguard_check            # every GPU on the node, in parallel, about 5 minutes
```

Each GPU gets a verdict (PASS, FAIL, SUSPECT or INCOMPLETE), a card and a `result.json`. `result.json` holds the
GPU model, driver, software versions, what was checked and found, speed and health readings, and a short hash of
the GPU's UUID (to count distinct GPUs). It has no hostname, serial number, PCI address, full UUID or IP
address. The full record, with those, stays in `check.json` on your machine.

To collect many nodes into one page:

```bash
sdcguard report node*/sdcguard_check --out fleet_report     # fleet_report.html, quarantine.txt, rma/
```

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

Every 50th step, every nn.Linear's matmuls (forward, dX and dW) are checked. Alerts go to the JSON-lines file:
layer name, matrix shapes, the position in the output, the score and the GPU's identity, never tensor values.

## What to send back (only what you're comfortable sharing)

- The `result.json` files, or `fleet_report.json` (counts per GPU, no tensor data).
- For step 3: the events file (empty is a useful answer) and the step time with and without the guard.
- Five answers, a sentence each:
  1. How did you find the last GPU that computed wrong results (or how would you)?
  2. What does one cost you: hours lost, jobs rerun, credits paid, an RMA you couldn't get?
  3. What do you run today for acceptance and between jobs (DCGM level, burn-in, NCCL tests, your own)?
  4. Did anything sdcguard reported surprise you, or look wrong to you?
  5. What would make this worth paying for: scheduled checks, private fleet history, RMA evidence, alerts into
     your own tooling, signed acceptance reports for customers, something else?

## What the verdicts mean, and don't

PASS says every checked result was correct and every part of the check ran. It is not a guarantee: a faulty
GPU can pass a 5-minute check, because many faults are intermittent (ByteDance found synthetic tests miss most
defective GPUs). FAIL means at least one result was wrong, with the inputs and output saved so `sdcguard
replay` can recompute it on another GPU; or the driver reported memory errors it couldn't correct. A FAIL on a
GPU you believe is healthy is the most valuable thing you can send us: it's either a bad GPU or a bug in
sdcguard, and we want to know which.

`--simulate-faults` exists for demos only: its results are labelled SIMULATED everywhere and the board rejects
them.
