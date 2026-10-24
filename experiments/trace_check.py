"""`gate model-check --trace-replay` for the experiments, with the option of the same weight fault the server runs
got (`--fault LAYER:K`: weight [0, 7] of the layer multiplied by 2**K through the test worker extension, before any
request). Runs in its own process, so the engine's GPU memory is freed when it exits.

    python experiments/trace_check.py ref.json --out fault_trace.json --model M --fault model.layers.10.mlp.down_proj:8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--vllm-arg", action="append", default=[])
    ap.add_argument("--fault", default=None)
    a = ap.parse_args(argv)
    os.environ["PYTHONPATH"] = HERE + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    sys.path.insert(0, HERE)
    from sdcguard import gate_model as gm
    extra = gm._extra(a.vllm_arg)
    extra["enable_trace_replay"] = True
    if a.fault:
        extra["worker_extension_cls"] = "sdcguard_test_ext.FaultInjection"
    llm = gm._llm(a.model, "bfloat16", a.max_len, extra)
    injected = None
    if a.fault:
        layer, k = a.fault.rsplit(":", 1)
        injected = llm.collective_rpc("sdcguard_test_scale", args=(layer, int(k)))
    run = gm.check_trace(a.ref, a.out, a.model, extra, a.noise, llm=llm)
    run["fault"] = {"spec": a.fault, "injected_in": injected}
    with open(a.out, "w") as f:
        json.dump(run, f, indent=1, default=str)
    print(json.dumps({"verdict": run["verdict"], "summary": run["summary"], "fault": run["fault"]}, indent=1,
                     default=str))
    return {"PASS": 0, "WARN": 1, "FAIL": 1}.get(run["verdict"]["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
