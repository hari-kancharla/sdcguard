"""Installed ordinary capture and verification. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .. import evidence_io as io


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard operator run",
        description="Capture native checksum coverage on the pinned same-host profile, then verify every planned request.")
    parser.add_argument("--out", required=True, help="new local evidence directory")
    parser.add_argument("--require", action="append", choices=("checksum", "lease"), required=True)
    parser.add_argument("--model-cache", help="existing pinned Qwen3-0.6B snapshot directory")
    parser.add_argument("--preflight-only", action="store_true", help="CPU checks only; no server or GPU action")
    args = parser.parse_args(argv)
    required = list(dict.fromkeys(args.require))
    if required != ["checksum"]:
        print("HOLD: this installed capture profile supports checksum only; combined lease and checksum is unsupported.")
        return 2
    out = Path(args.out).expanduser().absolute()
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        print("HOLD: output must be a new directory with an existing parent.")
        return 2
    if args.model_cache:
        try:
            snapshot = Path(args.model_cache).expanduser().resolve(strict=True)
        except OSError as error:
            print("HOLD: model cache is unavailable: " + str(error)[:512])
            return 2
        if not snapshot.is_dir():
            print("HOLD: model cache must be an existing directory.")
            return 2
        os.environ["SDCGUARD_OPERATOR_SNAPSHOT"] = str(snapshot)
    out.mkdir(mode=0o700)
    from .common import checkpoint, encoded
    try:
        from .preflight import execute as preflight
        receipt, plan, payloads = preflight()
        checkpoint(out / "CPU_PREFLIGHT.json", receipt)
        if receipt["status"] != "PASS":
            print("HOLD: CPU preflight did not pass: " + str(receipt.get("failure")))
            return 2
        if args.preflight_only:
            checkpoint(out / "REQUEST_PLAN.json", plan)
            print("CPU preflight passed. No server or GPU action was started.")
            return 0
        print("Starting the pinned same-host profile with prefix caching on and ordinary shared-prefix requests.", flush=True)
        from .runtime import execute
        result = execute(out, plan_raw=encoded(plan), payloads_raw=encoded(payloads), preflight=receipt, commit=lambda: None)
        archive = result.pop("archive", None)
        if type(archive) is bytes:
            io.write_file(out, "ordinary-evidence.zip", archive)
        checkpoint(out / "RESULT.json", result)
        from ..required_gate import inspect, render
        answer = inspect(out, required)
        checkpoint(out / "GATE.json", answer)
        print(render(answer), end="")
        return 0 if answer["decision"] == "PASS" else 2
    except (Exception, KeyboardInterrupt) as error:
        checkpoint(out / "OPERATOR_ERROR.json", dict(kind=type(error).__name__, detail=str(error)[:1024], decision="HOLD"))
        print("HOLD: operator execution is incomplete: " + str(error)[:1024])
        return 2
