# sdcguard

Hi, I'm Hari. sdcguard catches GPUs that give wrong answers without crashing.

A GPU can return wrong numbers and keep going. No error, no crash, and the loss curve looks fine. Sometimes the
chip is bad, sometimes it's a library bug. Meta sees roughly one faulty device per thousand, and ByteDance
found 18 incidents in 35 million GPU-hours, 15 of them silent
([AEGIS](https://www.usenix.org/system/files/osdi26-lei.pdf)).

The usual health checks test an idle GPU. sdcguard checks the math of your actual training run or vLLM server
while it runs.

## Try it

```bash
pip install "sdcguard[all]"
sdcguard check
```

That takes about five minutes per machine. It runs matmuls in every precision your GPU supports and checks them
against float64, tests memory, trains and serves a small model with the checks on, and makes sure a fixed
model gives the same output every time. Each GPU gets PASS, FAIL, SUSPECT or INCOMPLETE. A FAIL saves the
inputs and the wrong output so you can replay them on another GPU.

Add `--share` to put your result on the [public board](https://hari-kancharla.github.io/sdcguard/board/).

No GPU? `python -m sdcguard.demo` runs a simulated bad GPU on your CPU, or try it in
[Colab](https://colab.research.google.com/github/hari-kancharla/sdcguard/blob/main/examples/colab_demo.ipynb).

<img alt="An example sdcguard check result card" src="docs/img/example_card.png" width="560">

## What a bad GPU looks like

Here's a simulated bad GPU: 4 H100s training a 51M-parameter GPT model on TinyStories. From step 205,
GPU 2 corrupts 5% of its attention Q/K matmuls, using the fault patterns from NVIDIA's
[gate-level fault-injection study](https://arxiv.org/abs/2605.04213). sdcguard checked every 10th step,
flagged GPU 2 at step 210, rolled back to the step 200 checkpoint and finished all 1,000 steps on the other
three GPUs (the blue line after step 200 is that rerun). Without it, the loss looked normal until it turned
NaN at step 215, and it crashed the same way after both restarts.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/demo_zoom_dark.png">
  <img alt="Training loss on 4 H100s with and without sdcguard. GPU 2 starts returning wrong results at step 205. sdcguard flags it at step 210, rolls back to step 200 and finishes all 1,000 steps on GPUs 0, 1 and 3. Without sdcguard the loss looks normal until it turns NaN at step 215, and restarting crashes again." src="docs/img/demo_zoom.png">
</picture>

Logs and raw numbers are in [results/demo_4gpu](results/demo_4gpu).

## A real bug that GPU health checks can't see

cuBLAS has a bug (NVIDIA known issue 6862950) that gives wrong FP16, BF16 and FP8 matmuls on Hopper GPUs
running under MPS with less than 100% of the SMs. A matching report was
[posted on NVIDIA's forum](https://forums.developer.nvidia.com/t/cublaslt-returns-silently-wrong-fp16-bf16-fp8-gemm-results-on-hopper-under-cuda-mps-active-thread-percentage-algoid-66-cluster-kernels/384873)
by another engineer on October 2, 2026, and NVIDIA said on October 6 that it had reproduced it.

I ran sdcguard, DCGM and gpu-burn on one H200 at four MPS shares, with the rules written down before the run.
The second column is the ground truth: PyTorch's own matmuls at that share, checked against float64.

| MPS share | PyTorch's matmuls | sdcguard check | DCGM diagnostic | gpu-burn |
|---|---|---|---|---|
| 100% | right | pass | pass | pass |
| 40% | **wrong** | **FAIL** | pass | pass |
| 75% | **wrong** (weight gradients) | **FAIL** | pass | pass |
| 14% | **wrong** | **FAIL** | pass | pass |

DCGM and gpu-burn were right to pass: their own matmuls, in their own shapes and settings rather than the job's,
came out right every time, so the rules written before the run count them as not exercised, not as misses. That's the point: a test
that runs its own matmuls can't vouch for the ones your job runs. Full rules and logs are in
[METHOD.md](docs/METHOD.md#head-to-head-on-a-real-cublas-bug-written-october-20-2026-before-any-run).

## In training

```python
from sdcguard import SDCGuard

guard = SDCGuard(every=20).protect(model)    # checks every nn.Linear: forward, dX and dW
for step, (x, y) in enumerate(loader):
    with guard.on_step(step):                # every 20th step is checked
        loss = model(x, y); loss.backward()
    optimizer.step(); optimizer.zero_grad()
    if guard.log.events:
        print(guard.log.events[0].to_json()); break
```

In tests on an H100 (October 18), checking every 20th step cost 1.7% on GPT-2 small and 1.3% on a 0.9B model.
It works with autocast,
DDP, FSDP2, tensor parallel and `torch.compile`. With the Hugging Face Trainer it's one line:

```python
trainer = Trainer(..., callbacks=[SDCGuardCallback(every=20, stop_on_alert=True)])   # from sdcguard.hf
```

## In vLLM

```bash
SDCGUARD=1 SDCGUARD_SHARE=0.125 vllm serve Qwen/Qwen2.5-7B-Instruct
```

This checks one linear layer in eight: the same layers on every step, so the other seven in eight are never
checked on this server (give replicas different `SDCGUARD_OFFSET`s and between them they check every layer once).
On an H100 (October 22, Qwen2.5 1.5B and 7B, 16 and 256 requests at a time) it lowered throughput by 2.5% to 7%,
the answers were word for word the same as plain vLLM, and a weight bit I flipped on purpose was caught and the
layer named. Checking every layer lowered throughput by 20% to 22% on the 7B model and by 11% (256 at a time) to
33% (16 at a time) on the 1.5B. I'm working on making that cheaper.

When a check fires, sdcguard keeps the exact numbers that failed (copied on the GPU before vLLM reuses the
memory) and rechecks them in float64 on the CPU, so the alert tells you whether the output was really wrong.
Those evidence files hold real rows of your inputs and of the layer's weights: treat them like your data, and
don't attach them to a public issue unless you're fine sharing them.

Each alert also says which requests it concerns, in three parts kept apart: the request directly implicated (the
one that held the row the check alarmed on, in that model step), the requests potentially exposed (every request in
the steps that read covered, and every request since the layer's last clean read, which a persistent fault such as
a changed weight concerns), and what neither covers (prefix-cache blocks reused later, and what isn't checked).
Tokens already streamed to a client can't be taken back; the alert tells you which responses to distrust.

If the plugin fails to load, vLLM serves without it, and the plugin can't report its own absence. So check from
outside. Give each GPU worker a metrics port, then ask:

```bash
SDCGUARD=1 SDCGUARD_REQUIRE=1 SDCGUARD_METRICS_PORT=9400 vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2
sdcguard ready --port 9400 --workers 2 --vllm http://127.0.0.1:8000/metrics --probe http://127.0.0.1:8000
```

It says PROTECTED only if every worker answers, checks its layers, has read results back, and ran checks while
vLLM produced tokens. It works as a Kubernetes readiness probe (give the probe more time than its 5-second window).
On a real `vllm serve` (H100, vLLM 0.32, October 24) it gave the expected answer 21 times out of 21 and never said
PROTECTED when the plugin wasn't loaded, was turned off, had stopped reading its results while serving, or the
server was gone. An idle server can't show its checks running, so it says IDLE; add `--probe http://127.0.0.1:8000`
and it sends two short requests to find out. Request naming: one request alone was named exactly in 5 of 5
trials; with 32 requests sent 8 at a time, every alert named one of the 32 (which one held the row wasn't checked
independently).

For any other PyTorch model there's `ServeGuard().attach(model, share=0.125)`.

## Across a fleet

```bash
sdcguard screen --minutes 10                         # screen every GPU on a node
sdcguard replay screen_out/GPU-.../evidence_0.pt     # replay a failure on another GPU
sdcguard report node*/sdcguard_check --out fleet_report
sdcguard blas                                        # every cuBLAS algorithm on a few shapes, checked
sdcguard issues                                      # which known cuBLAS bugs apply to this machine
```

There are also Kubernetes and Slurm health checks ([deploy/](deploy/)), signed per-GPU certificates, and alerts
to Slack, Discord, Prometheus or NVIDIA NVSentinel.

## How it works

For `C = A @ B`, the sum of each block of 128 output columns has to equal `A` times the matching column sums of
`B`. Both sides are cheap to compute, so a wrong value in `C` shows up as a mismatch. This is algorithm-based
fault tolerance (Huang and Abraham, 1984). The hard part is bf16: normal rounding makes the two sides differ a
little, so sdcguard models each block's rounding noise and only alarms past six times that. In 24.6 million
checks of clean GPT-2 small training on H100s (every matmul on every step, five 62-minute runs on October 14-15,
with the check as it was then) it raised no false alarm; the highest score was 3.54. On NVIDIA GPUs the checks run
as Triton kernels.

## What it can't do yet

- Small errors. With one wrong value in a matmul's output, the default check flagged half of them only once the
  error reached about 29% of a typical output in its block, and none of 594 at 10% (H100, A100 and L4, October 23,
  training matmuls); precise mode (`precise=True`) flagged half at 0.71% to 0.92% on the same matmuls. These are
  detection rates for injected errors in training, not a measured curve for the vLLM plugin. ByteDance found its
  real incidents with fp32 checksums, not bf16 ones. The fused fp32 check (`precise="fused"`) matched precise
  mode's sensitivity on October 20 and 21 and passed 40 minutes of clean training on an H100 and an L4.
- Anything outside the checked matmuls: attention internals, norms, the optimizer. The experimental
  `recompute=True` check covers activation-checkpointed blocks in training.
- FP8 layers and fused MoE experts in vLLM. INT4 and INT8 weight-only layers (GPTQ, AWQ on vLLM's Marlin and
  Machete kernels) are checked.
- AMD GPUs.
- It hasn't caught a naturally bad GPU yet. So far the evidence is injected faults and the cuBLAS bug above.

## Project layout

```
src/sdcguard/     the library
experiments/      the GPU experiments behind every number here
scripts/          the runners that launch them on cloud GPUs
results/          raw results of every GPU run
docs/             METHOD.md (how each number was measured), EVIDENCE.md, WHY.md, the board
deploy/           Kubernetes, Slurm and Docker
tests/            pytest -q runs them all on a CPU
```

## Feedback

This is alpha (0.4.0), MIT licensed, and I built it on my own. If you try it, I'd love to hear how it went,
and most of all if you have a GPU you think is bad. Open an issue.
