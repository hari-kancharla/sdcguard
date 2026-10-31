"""Compare matched runs, or prepare their local identity without starting a server.

Offline preparation: sdcguard overhead prepare CONFIG --model-snapshot SNAPSHOT
--gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --out NEW_IDENTITY.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

from .perf import compare, SCHEMA

IDENTITY = ("engine_files_sha256", "model_revision", "weights_sha256", "gpu_uuid", "deployment_config_sha256")
METRICS = ("output_tokens_per_s", "ttft_p99_ms", "itl_p99_ms", "latency_p99_ms")


def summarize(base: dict, checked: dict) -> dict:
    missing = []
    if not isinstance(base, dict) or not isinstance(checked, dict):
        return {"status": "inconclusive", "missing": ["two performance objects"], "metrics": {}}
    for run in (base, checked):
        if (not isinstance(run.get("identity"), dict) or not isinstance(run.get("median"), dict)
                or not isinstance(run.get("rounds"), list)
                or not all(isinstance(row, dict) for row in run["rounds"])):
            return {"status": "inconclusive", "missing": ["structured identity, medians and raw rounds"], "metrics": {}}
    receipts = any("measurement_receipt" in run for run in (base, checked))
    if receipts:
        from .performance_identity import check_measurement_receipt
        for side, run in (("baseline", base), ("checked", checked)):
            try:
                check_measurement_receipt(run)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                missing.append(f"valid {side} measurement receipt: {exc}")
        if not missing:
            left = base["measurement_receipt"]["prepared_identity"]["manifest_sha256"]
            right = checked["measurement_receipt"]["prepared_identity"]["manifest_sha256"]
            if left != right:
                missing.append("same prepared local identity, checker source and engine variant on both sides")
    for key in IDENTITY:
        a, b = base.get("identity", {}).get(key), checked.get("identity", {}).get(key)
        if not isinstance(a, str) or not a or a != b:
            missing.append(f"matching recorded {key}")
        elif key != "gpu_uuid" and re.fullmatch(r"[0-9a-f]{40}" if key == "model_revision" else r"[0-9a-f]{64}", a) is None:
            missing.append(f"valid {key} digest")
    if base.get("checks_enabled") is not False or checked.get("checks_enabled") is not True:
        missing.append("baseline checks off and candidate checks on")
    if base.get("model") != checked.get("model") or not base.get("model"):
        missing.append("same model")
    for side, run in (("baseline", base), ("checked", checked)):
        if run.get("schema") != SCHEMA or not isinstance(run.get("load"), dict):
            missing.append(f"supported {side} performance schema and load")
        load = run.get("load") if isinstance(run.get("load"), dict) else {}
        if (type(load.get("prompts")) is not int or load["prompts"] <= 0
                or type(load.get("rounds")) is not int or load["rounds"] != len(run["rounds"])
                or type(load.get("concurrency")) is not int or load["concurrency"] <= 0
                or type(load.get("max_tokens")) is not int or load["max_tokens"] <= 0
                or type(load.get("warmup")) is not int or load["warmup"] < 0
                or re.fullmatch(r"[0-9a-f]{64}", str(load.get("prompts_sha256", ""))) is None):
            missing.append(f"complete {side} workload identity")
        if len(run.get("rounds", [])) < 3:
            missing.append(f"at least three {side} rounds")
        for row in run.get("rounds", []):
            if (type(row.get("requests")) is not int or row["requests"] <= 0
                    or type(row.get("completed")) is not int or row["completed"] != row["requests"]
                    or type(row.get("errors")) is not int or row["errors"] != 0
                    or row.get("tokens_counted_by_server") is not True or row["requests"] != load.get("prompts")):
                missing.append(f"complete token-counted {side} rounds")
        for metric in METRICS:
            v = run.get("median", {}).get(metric)
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                missing.append(f"valid {side} {metric}")
            values = [r.get(metric) for r in run.get("rounds", [])]
            if (not values or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values)
                    or statistics.median(values) != v):
                missing.append(f"{side} {metric} median agrees with raw rounds")
    if missing:
        return {"status": "inconclusive", "missing": missing, "metrics": {}}
    comparison = compare(base, checked)
    if not comparison["comparable"]:
        return {"status": "inconclusive", "missing": comparison["why"], "metrics": {}}
    return {"status": "measured", "missing": [], "metrics": {k: comparison["metrics"][k] for k in METRICS},
            "metric_note": "itl_p99_ms is the existing tool's SSE text-chunk gap p99, not per-token device timing",
            "scope": "These matched runs only; round spread is descriptive, not a confidence interval",
            "identity_basis": ("content-bound local inspection receipts; GPU and checks state are operator-declared; "
                               "server labels are self-reported, not remote attestation" if receipts
                               else "recorded metadata; not remote attestation"),
            "comparison_note": "Observer overhead between matching engine-source pins, not a stock-versus-fix comparison"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepare":
        from .performance_identity import write_identity
        p = argparse.ArgumentParser(prog="sdcguard overhead prepare", description=__doc__)
        p.add_argument("config", type=Path)
        p.add_argument("--model-snapshot", type=Path, required=True)
        p.add_argument("--model-pin", type=Path, help="external model integrity pin; default: snapshot/sdcguard-model-pin.json")
        p.add_argument("--gpu-uuid", required=True, help="operator-declared physical GPU UUID; never queried here")
        p.add_argument("--out", type=Path, required=True)
        a = p.parse_args(argv[1:])
        try:
            identity = write_identity(a.config, a.model_snapshot, a.gpu_uuid, a.out, model_pin=a.model_pin)
            report = {"status": "prepared", "identity_file": str(a.out.resolve()),
                      "identity": identity["identity"], "engine_fix": identity["engine"]["fix"],
                      "scope": identity["scope"], "basis": identity["basis"]}
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report = {"status": "inconclusive", "missing": [str(exc)], "runtime_evidence": "not collected"}
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "prepared" else 2
    p = argparse.ArgumentParser(prog="sdcguard overhead", description=__doc__)
    p.add_argument("baseline")
    p.add_argument("checked")
    a = p.parse_args(argv)
    try:
        report = summarize(json.loads(Path(a.baseline).read_text()), json.loads(Path(a.checked).read_text()))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = {"status": "inconclusive", "missing": [str(exc)], "metrics": {}}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
