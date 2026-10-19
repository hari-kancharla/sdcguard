# Release validation for the October 19 review candidate

This is a validation plan, not a record of a completed GPU run. It was written by an outside review of the
b778d27 source (CPU only: Python 3.12.14, PyTorch 2.14.1+cpu, Triton 3.8.0, no CUDA device) and updated after
its fixes were merged onto the later local work. CPU tests and the Triton interpreter cannot establish actual
kernel compilation, CUDA graph behavior, numerical calibration or GPU overhead. Results of each gate are
recorded in [METHOD.md](METHOD.md) and `results/`.

Keep the existing hold on GitHub/Pages/PyPI publication until the founder's release authorization. All existing
features remain in the source. Experimental paths retain that status until their own evidence is adequate.

## 1. Bind every result to the actual source

The reviewed archive claims b778d27 and its original 77-Python-file fingerprint is:

`1265b3e0a38589060c14c399d072cb6fda5985849bac282bb54e8b88a494ccfb`

The review fixes change that fingerprint. The handoff contains a patch against that exact archive, an original
file manifest, a reviewed source snapshot and a final manifest. If the working Mac checkout has advanced,
preserve its work and port the fixes; do not overwrite it with the snapshot or assume the old test count applies.

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
python -m pytest -q -rs
python -m sdcguard.selftest
```

Record every skip and its reason. CPU success does not include the CUDA-only cases. If testing optional
Accelerate integration, install that optional package and run its test explicitly; do not call its missing
package skip a GPU skip. Build/install the wheel outside the source checkout and confirm `sdcguard.__file__`
points into the installed package before running its selftest and a small offline check.

## 3. Real GPU checks, in this order

These commands allocate paid Modal resources. Obtain current pricing and a concrete total cap under the
founder's existing spending authorization before starting. The hard function timeouts below are runtime
limits, not a guaranteed dollar ceiling. Do not launch duplicate watchers or use `--force` to work around an
active-run lock. A timeout, NOT STARTED message, missing report or nonzero exit is not a passed run.

| Gate | Existing command | Configured maximum allocation | What must be inspected |
|---|---|---|---|
| Updated training/serving kernels and wheel | `modal run scripts/modal_check.py --phase tests` | One L4, one H100 and one B200, up to 30 minutes each | Full test logs and skip reasons; real training-kernel and serving-kernel comparisons (`tests/_fast_compare.py`, `tests/_serve_fast_compare.py cuda`); installed-wheel selftest/check; all subprocess return codes. The changed training tile buffer has an extra completeness plane. On the B200 this is also the first full test of the Triton kernels without software pipelining, with per-shape kernel timings. Last, with its own time limit and reported separately: the attention check's calibration against its written criteria. |
| Experimental precise noise model | `modal run scripts/modal_precise_validate.py` | H100, B200, A100-80GB, L4 and T4, up to 30 minutes each | Ran October 19 on the frozen model (fingerprint 1265b3e0a385): 0 alarms in 17,424 checks, but worst 4.27 (A100) and 4.26 (L4) against 4, and the T4 didn't run. Criteria not met; precise mode stays experimental. Rerunning the same model adds nothing; a revised model needs new held-out workloads and the stricter count gate added here. |
| Current public check and cards | `modal run scripts/modal_check.py --phase sweep` | Ten GPUs across nine functions, up to 15 minutes per function | Supported modes actually completed; no unexpected INCOMPLETE or missing cards; all three fingerprint runs recorded; Blackwell running the Triton kernels (or an intentional opt-out) distinct from a broken kernel; board build succeeds. These are healthy-device controls, not measured defective-device recall. |
| Actual vLLM integration | `modal run scripts/modal_vllm.py --phase vllm` | One H100, up to 75 minutes | `STATUS.json`, both model reports, compared completions, clean alerts, registered layer checks, injected-layer identification, and per-share summaries. Verify the successful execution mode: an eager fallback does not validate default compiled/CUDA-graph behavior. |

The precise runner evaluates the frozen model on held-out workloads. If it fails, retain the failure and
explain what it means before changing coefficients. A post-result threshold change starts a new calibration
round and needs another held-out comparison, including sensitivity. Zero alarms achieved by widening
thresholds is not sufficient evidence of a useful detector.

The vLLM experiment currently uses positive residuals to demonstrate per-layer activity. This is more evidence
than an empty alerts list, but is not a complete executed/unjudged counter. Do not claim that limitation is
solved. `experiments/demo_vllm.py` is a recording aid and its zero exit alone is not a behavior validation gate;
use the gated `vllm_check.py` results and inspect the actual alert in the demo.

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

The H200 MPS reproduction's raw output is in `results/mps_bug` (three runs on October 19). The outside
review's snapshot didn't include it. NVIDIA's own reports verify the external software bug, not this
project's reproduction or a physical defect. Keep the GPU identity, MPS share, CUDA/cuBLAS versions and known
algorithm/configuration together.

Before sharing, restore the release's referenced raw artifacts and images, verify the public installation path
and board, and review the precise wording of every number. Save private machine identifiers and tensors only
where intended. Use reviewed `result.json` cards for the public board, with explicit user submission.

The founder writes and submits the YC application and any launch posts. A passing test suite is engineering
evidence; outside adoption, diagnosis of a real incident and a purchase commitment are separate evidence.
