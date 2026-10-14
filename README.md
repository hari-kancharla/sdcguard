# sdcguard

**Catch GPUs that silently corrupt AI training, from inside the training run.**

Some GPUs go bad without crashing: they keep running and return wrong numbers. In ByteDance's
35M GPU-hour deployment, 15 of 18 such incidents caused no visible training failure, and standard
offline GPU tests missed over 60% of defective GPUs. `sdcguard` checks the math while your model
trains and names the GPU that's lying.

> Status: pre-alpha (v0.0.1). Working name. Eager PyTorch, bf16/fp16, data parallel.

## Three detectors
| Detector | What it checks | Catches |
|---|---|---|
| `SDCGuard` | two checksums per 128-value block of every sampled matrix multiply, forward and backward | corrupted matmul outputs (zeroed chunks, bit flips, NaN/Inf) |
| `Canary` | the same batch on every GPU, layer-by-layer fingerprints, majority vote | faults in any operation; names the GPU and the first bad layer |
| `check_replicas` | data-parallel copies of the weights must stay bit-identical | corrupted optimizer updates or memory on one GPU |

Every alert carries the host, rank, GPU UUID, PCI bus id and physical GPU index, so you know
which chip to pull, and `rollback_checkpoint()` picks the last checkpoint saved before the
corruption could have started.

## Quick look
```python
from sdcguard import SDCGuard, Canary, check_replicas, rollback_checkpoint

guard = SDCGuard().watch(model)                # watch() names layers in alerts (optional)
for step, (x, y) in enumerate(loader):
    if step % 20 == 0:                          # check every 20th step to keep the slowdown small
        with guard.at_step(step):
            loss = model(x, y); loss.backward()
    else:
        loss = model(x, y); loss.backward()
    optimizer.step(); optimizer.zero_grad()
    if step % 100 == 0:                         # every rank calls these together
        suspects, _ = check_replicas(model.parameters(), step=step)

print(guard.stats, [e.to_json() for e in guard.log.events])
```
Checks are queued on the GPU and read back once, when the `with` block ends, so the detector
never stalls the GPU in the middle of a step.

## Setup
```bash
pip install -e ".[dev,experiments]"
python -m sdcguard.selftest
python experiments/harm_study.py --preset smoke
```
If you hit an Apple GPU (MPS) error, add `--dtype fp16` (or `--device cpu`) to the command.

**Self-test first, on every new machine:** `python -m sdcguard.selftest` must print `ALL PASS`.
It proves the detector really sees and checks this machine's matrix multiplies (Apple GPUs, for
example, run linear layers through different internal ops than NVIDIA GPUs). The harm study also
marks a run INVALID if no faults were injected or nothing was checked.

## On NVIDIA GPUs (rented through Modal, no machine to manage)
`scripts/modal_gpu.py` runs every GPU experiment on H100s and downloads the results into
`runs/gpu/`. Billing is per second and stops when each phase ends; every function has a hard
time limit. Setup and phases are described at the top of that file:
```bash
modal run --detach scripts/modal_gpu.py --phase check
modal run --detach scripts/modal_gpu.py --phase main
modal run --detach scripts/modal_gpu.py --phase demo
```

## Experiments
- `experiments/harm_study.py`: the same GPT trained several times: clean with 3 different seeds
  (plus seed 0 again, to measure the GPU's own run-to-run noise), and on a simulated bad GPU
  (faults injected into chosen layers, paired with the first clean seed). For each faulty run it
  writes `verdict.txt`, `summary.json`, `harm.png` and `samples.txt`, plus a `REPORT.md` with every
  run and a table of what was caught, by fault size. Runs can go one at a time (`--one`) so many
  GPUs work in parallel; `--analyze` rebuilds the report.
- `experiments/calibrate.py`: kill criterion #1. Slowdown for several detector settings, and a long
  clean "soak" with every matmul checked, counting false alarms.
- `experiments/ddp_demo.py`: 4 GPUs, one goes bad mid-run. Without sdcguard the run crashes and gets
  restarted on the same bad GPU; with it, the bad GPU is named, evicted, and training resumes from
  the last checkpoint saved before the corruption.

Fault patterns follow NVIDIA's published GPU error study (arXiv 2605.04213).

## Tests
```bash
python -m pytest -q
```
