# Release validation for the October 20 review of 12e23f5

This began as a validation plan. Its first gate, the updated GPU tests, ran on October 20 and is recorded below;
the other gates haven't run on this source yet. It was written by an outside review of the
b778d27 source (CPU only: Python 3.12.14, PyTorch 2.14.1+cpu, Triton 3.8.0, no CUDA device) and updated after
its fixes were merged onto the later local work. The October 20 review covers the supplied `12e23f5` snapshot
and the new patches against it. CPU tests and the Triton interpreter do not establish CUDA execution, graph
replay, numerical calibration or GPU overhead. Separate local compilation can establish compiler acceptance
for specific targets, but not device correctness. Results of each gate are
recorded in [METHOD.md](METHOD.md) and `results/`.

Keep the existing hold on GitHub/Pages/PyPI publication until the founder's release authorization. All existing
features remain in the source. Experimental paths retain that status until their own evidence is adequate.

## 1. Bind every result to the actual source

The earlier review was of `b778d27`, whose 77-Python-file fingerprint was:

`1265b3e0a38589060c14c399d072cb6fda5985849bac282bb54e8b88a494ccfb`

The supplied archive for this review identifies its source as `12e23f5`. Recomputing the runner's fingerprint
directly from that archive gives 92 Python files and:

`7508925675608f0b95a6753f7d86b45748e93abe6e9daa16c6cc81b21a84a279`

The October 20 patches are against this supplied snapshot and change its fingerprint. If the working Mac
checkout has advanced, preserve its work and port the fixes; do not overwrite it or assume an old test count applies.

The October 20 GPU tests ran on code whose Python fingerprint (108 files) is
`74b844ee7ca8d6f5bcf687a00e3ed61d72d817b3b64cd5460348f4e22df3b5fd`, the same as commit `77d5238`
(`3ec5bdd` changed no Python files). Each GPU printed that fingerprint from the code it actually ran. The
fingerprint covers only .py files, not pyproject.toml or the other uploaded files, and the runner recorded
neither `git rev-parse HEAD` nor the working-tree status.

Commit the integrated candidate locally, record `git rev-parse HEAD` and its working-tree status, and save the
source fingerprint printed by each runner. A version string alone is insufficient. The vLLM runner predates
that fingerprint field; retain the exact committed checkout, wheel hash and run logs together for that run.

Run tests against the integrated tree. A log is current only while its relevant source files are unchanged.
Do not replace old failure logs with a success narrative. Missing GPU artifacts are missing evidence, even
when a Markdown summary reports an earlier successful run.

## 2. Local checks

Install the candidate and declared test dependencies in the intended Python environment:

```bash
python -m pip install -e ".[dev,experiments]"
python -m pytest -q -rfEs
python -m sdcguard.selftest
```

Record every skip and its reason. CPU success does not include the CUDA-only cases. If testing optional
Accelerate integration, install that optional package and run its test explicitly; do not call its missing
package skip a GPU skip. Build/install the wheel outside the source checkout and confirm `sdcguard.__file__`
points into the installed package before running its selftest and a small offline check.

## 3. Planned GPU checks and recorded validation

Three runs were planned: the updated tests (passed October 20, recorded in the table), the vLLM integration,
then the paired default/precise comparison. The table also retains the frozen precise-mode run's history and the later public-sweep gate.

These commands allocate paid Modal resources. Confirm current pricing and the planned nominal allocation
under the founder's spending authorization before starting. Scalar CPU and memory settings are requests,
and actual usage can cost more; startup, idle retention and restarted attempts can add charges too
([resources](https://modal.com/docs/guide/resources), [pricing](https://modal.com/pricing), accessed October 20, 2026).
The function timeouts below limit execution per attempt, excluding scheduling time; they do not establish a
strict total bill cap ([timeouts](https://modal.com/docs/guide/timeouts), accessed October 20, 2026).
Do not launch duplicate watchers or use `--force` to work around an
active-run lock. A timeout, NOT STARTED message, missing report or nonzero exit is not a passed run.

| Gate | Existing command | Configured resources and per-attempt execution limit | What must be inspected |
|---|---|---|---|
| Updated training/serving kernels and wheel | `modal run scripts/modal_check.py --phase tests` | One L4, up to 45 minutes; one H100 and one B200, up to 30 minutes each | Full test logs and skip reasons; real training-kernel and serving-kernel comparisons (`tests/_fast_compare.py`, `tests/_serve_fast_compare.py cuda`); installed-wheel selftest/check; all subprocess return codes. The changed training tile buffer has an extra completeness plane. On the B200 the reviewed candidate needs full validation of the one-stage Triton path, with per-shape kernel timings. Last, with its own time limit and reported separately: the attention calibration's results, including partial or unjudged work. **Ran October 20 (fingerprint 74b844ee7ca8): every step exited 0 on all three GPUs.** Each: 569 tests passed, 1 skipped (Accelerate not installed; tests/test_fast.py's interpreter runs replaced by the comparison scripts on CUDA); both comparisons agree; wheel selftest ALL PASS; `sdcguard check` PASS, 0 wrong. B200 timings and the attention calibration (criteria met on all three) are in METHOD.md. The files in `results/gpu_tests` are the runner's tails of each step; the full step logs are on the `sdcguard-runs` Modal volume under `/check_sweep/`. |
| Experimental precise noise model | `modal run scripts/modal_precise_validate.py` | H100, B200, A100-80GB, L4 and T4, up to 30 minutes each | Ran October 19 on the frozen model (fingerprint 1265b3e0a385): 0 alarms in 17,424 checks, but worst 4.27 (A100) and 4.26 (L4) against 4, and the T4 didn't run. Criteria not met; precise mode stays experimental. Rerunning the same model adds nothing; a revised model needs new held-out workloads and the stricter count gate added here. |
| Default check vs precise mode, one revision | `modal run scripts/modal_mode_compare.py` | H100, A100-80GB and L4, up to 30 minutes each | Confirm every GPU completed; inspect injected-error detection per size for both checks with nominal pointwise 95% intervals, clean false alarms and step times. Report all missing or unjudged work (no pass line; October 19 rules and October 20 interpretation clarifications in METHOD.md precede the GPU results). |
| Current public check and cards | `modal run scripts/modal_check.py --phase sweep` | Ten GPUs across nine functions, up to 15 minutes per function | Supported modes actually completed; no unexpected INCOMPLETE or missing cards; all three fingerprint runs recorded; Blackwell running the Triton kernels (or an intentional opt-out) distinct from a broken kernel; board build succeeds. These are healthy-device controls, not measured defective-device recall. |
| Actual vLLM integration | `modal run scripts/modal_vllm.py --phase vllm` | One H100, up to 75 minutes | `STATUS.json`, both model reports, compared completions, clean alerts, registered layer checks, injected-layer identification, and per-share summaries. A fallback does not validate the default compiled/CUDA-graph behavior: since October 19 the experiment reports a failure when only a fallback mode passed, and keeps every attempt's files (`guarded_compiled.json` and so on); `SUMMARY.json` says which attempt passed. |

The precise runner evaluates the frozen model on held-out workloads. If it fails, retain the failure and
explain what it means before changing coefficients. A post-result threshold change starts a new calibration
round and needs another held-out comparison, including sensitivity. Zero alarms achieved by widening
thresholds is not sufficient evidence of a useful detector.

The vLLM gate snapshots per-layer GPU counts after warm-up and after measured clean traffic. Every registered layer must advance
its check count during measured clean traffic, worker/layer reports and aggregate counts must agree, and no
unjudged count may increase after that boundary. Clean alerts, changed-weight reports or failure to identify
the injected layer also fail the gate. Startup counts remain visible but cannot satisfy traffic coverage.
A zero residual is valid when its count advances; a positive residual is not an execution requirement.
On October 20 the counts passed on three GPUs, including a `torch.compile` CUDA-graph test that confirms replay
with PyTorch's graph-replay hook. The vLLM gate itself still needs its run on this source, with explicit evidence
of vLLM's CUDA graph capture and replay rather than eager fallback.

`experiments/demo_vllm.py` is a recording aid. Since October 19 it exits 1 unless the healthy answers raised
no alert and an alert named the flipped layer, but it checks less than `vllm_check.py` (no output comparison,
no per-layer counts): use the gated `vllm_check.py` results for validation and inspect the actual alert in
the demo.

## 4. Sensitivity and behavior, not just green tests

Reuse the existing fault fixtures and experiments on the release candidate. Report the injection location
(CPU-emulated output, post-GEMM output or NVBit instruction), size, dtype, shape, layer/pass, sign seed,
checked/injected/caught counts, unjudged cases and misses. Separate mathematically constructed examples from
naturally defective GPU evidence.

Measure default and precise paths separately, at the actual checked-step rate. Record warm-up, model,
dtype, GPU, software and whether throughput loss or latency increase is being reported. The old H100 1.7%
training and approximately 3% serving results are not measurements of the precise path or this candidate.
The previous precise model's failures and cost remain in the historical summaries.

A large legitimate neighboring output can mask a smaller error in either block-checking path. Fresh random
signs reduce some cancellation blind spots but do not remove the nullspace of compressed checksums. Keep
these as visible limitations until a measured alternative improves them.

## 5. Incident and launch checks

Use the updated `examples/demo_60s.md` and inspect each generated artifact. Demonstrate a real clean run,
a clearly labeled simulated failure, a portable copied evidence bundle and a replay. The simulation must not
enter the node health cache or the quarantine candidate list. Certificate verification must fail for a missing
or changed declared artifact; use an independently trusted public key when testing issuer trust.

The H200 MPS reproduction's raw output is in `results/mps_bug` (three runs on October 19). The earlier
`b778d27` review did not include it; the supplied `12e23f5` archive does. NVIDIA's own reports verify the external software bug, not this
project's reproduction or a physical defect. Keep the GPU identity, MPS share, CUDA/cuBLAS versions and known
algorithm/configuration together.

Before sharing, restore the release's referenced raw artifacts and images, verify the public installation path
and board, and review the precise wording of every number. Save private machine identifiers and tensors only
where intended. Use reviewed `result.json` cards for the public board, with explicit user submission.

The founder writes and submits the YC application and any launch posts. A passing test suite is engineering
evidence; outside adoption, diagnosis of a real incident and a purchase commitment are separate evidence.
