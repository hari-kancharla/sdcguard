"""Export and judge only the already-saved #61229 case. No discovery API."""
from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
from importlib.metadata import PackageNotFoundError
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def historical_case():
    return json.loads(files("sdcguard").joinpath("case_data/61229.json").read_text())


def replay_model_pin():
    case = historical_case()
    data = files("sdcguard").joinpath("case_data/61229/MODEL_PIN.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != case["replay_model_pin_sha256"]:
        raise ValueError("packaged replay model pin differs from its reviewed digest")
    return json.loads(data)


def judge(result, variant, manifest=None):
    out = {"status": "inconclusive", "missing_or_failed": [],
           "scope": "Existing #61229 condition and declared local model pin; no revised-hook GPU guarantee"}
    why = out["missing_or_failed"]
    if not isinstance(result, dict) or variant not in ("unpatched", "fixed"):
        why.append("supported result object and variant")
        return out
    case = historical_case()
    label = "stock" if variant == "unpatched" else "fixed"
    try:
        result_digest = digest(result)
        saved = result_digest == case["canonical_result_sha256"][label]
    except (ValueError, TypeError):
        why.append("finite, canonical result data")
        return out
    if saved and result.get("variant") == variant:
        out.update(status="pass", evidence="exact preserved historical result",
                   model_pin="historical resolved model commit was not recorded",
                   scope="Historical experiment criterion only; not current evaluator validation")
        return out
    if (not isinstance(manifest, dict) or type(manifest.get("schema")) is not int or manifest["schema"] != 1
            or manifest.get("complete") is not True
            or manifest.get("variant") != variant or manifest.get("result_canonical_sha256") != result_digest
            or manifest.get("harness_sha256") != case["reviewed_harness_sha256"]
            or manifest.get("candidate") != case["candidate"]):
        why.append("complete matching exported-run manifest (historical artifacts must match exactly)")
        return out
    from .assessment import pins
    identity = manifest.get("engine_identity")
    model_pin = manifest.get("model_pin")
    if (manifest.get("nixl_version") != "1.4.1"
            or type(manifest.get("harness_exit_code")) is not int
            or manifest.get("harness_exit_code") != (1 if variant == "unpatched" else 0)
            or not isinstance(result.get("evaluator_validation"), dict)
            or result["evaluator_validation"].get("passed") is not True):
        why.append("supported NIXL, expected harness exit and successful revised evaluator validation")
    expected = pins()["variants"][label]
    if (not isinstance(identity, dict) or identity.get("file_sha256") != expected
            or identity.get("version") != "0.32.0" or identity.get("adapter_fits") is not True):
        why.append("exact pinned engine identity")
    if (not isinstance(model_pin, dict) or model_pin.get("model") != case["model"]
            or re.fullmatch(r"[0-9a-f]{40}", str(model_pin.get("revision", ""))) is None
            or not isinstance(model_pin.get("files"), dict) or not model_pin["files"]
            or any(not isinstance(k, str) or re.fullmatch(r"[0-9a-f]{64}", str(v)) is None
                   for k, v in model_pin["files"].items())):
        why.append("resolved model identity and complete snapshot integrity pin")
    elif model_pin != replay_model_pin():
        why.append("exact recovered replay snapshot file identity")
    if (result.get("model_revision") != case["replay_model_revision"]
            or result.get("tokenizer_revision") != case["replay_model_revision"]):
        why.append("recorded immutable model and tokenizer revision")
    if result.get("model") != manifest.get("model_path") or result.get("variant") != variant or result.get("mode") != "compare":
        why.append("matching model path, variant and compare mode")
    refs = result.get("baseline_x_refs")
    if (not isinstance(refs, list) or len(refs) < 3 or not all(isinstance(x, str) and x for x in refs)
            or len(set(refs)) != 1 or result.get("baseline_ok") is not True):
        why.append("three agreed nonempty references and clean baseline")
    controls, trials = result.get("controls"), result.get("tried")
    if (not isinstance(controls, list) or len(controls) != 2 or not all(isinstance(x, dict) for x in controls)
            or not isinstance(trials, list) or len(trials) != 5 or not all(isinstance(x, dict) for x in trials)):
        why.append("both controls and all five paired replays")
        return out
    if why:
        return out
    for trial in controls + trials:
        required = ("x", "x_error", "x_exc", "x_finished", "hung", "started_clean", "outcome")
        if (any(k not in trial for k in required) or trial["started_clean"] is not True
                or trial["hung"] is not False or trial["x_finished"] is not True
                or trial["x_exc"] is not None or not isinstance(trial["outcome"], dict)):
            why.append("complete consistent terminal observation from idle")
    if why:
        return out
    for control in controls:
        if control.get("x") != refs[0] or control.get("x_error") is not None or control.get("outcome", {}).get("terminal") != "clean":
            why.append("both controls return the exact clean reference without error")
    for trial in trials:
        producer = trial.get("x_prefill_id")
        expected_applied = (json.loads(json.dumps(case["candidate"]["spec"]).replace("$X_PREFILL_ID", producer))
                            if isinstance(producer, str) and re.fullmatch(r"[\w:.-]{1,256}", producer) else None)
        if (trial.get("spec") != case["candidate"]["spec"] or expected_applied is None
                or trial.get("spec_applied") != expected_applied or trial.get("hold_outcome") != "condition_reached"
                or not isinstance(trial.get("ordering"), dict) or trial["ordering"].get("trigger_demonstrated") is not True):
            why.append("actual saved condition exercised, with its producer identity")
    if why:
        out["missing_or_failed"] = list(dict.fromkeys(why))
        return out
    failed = []
    for trial in trials:
        oc = trial["outcome"]
        if variant == "unpatched":
            if (oc.get("terminal") != "harm_garbled" or oc.get("harm") is not True
                    or not isinstance(trial["x"], str) or trial["x"] == refs[0] or trial["x_error"] is not None):
                failed.append("historical stock failure class was not reproduced consistently")
        elif trial["x_error"] is not None:
            why.append("fixed recompute arm returned a terminal error")
        elif trial["x"] != refs[0]:
            if isinstance(trial["x"], str) and trial["x"]:
                failed.append("fixed arm returned a differing answer after the saved stale-read condition")
            else:
                why.append("fixed answer unavailable")
        else:
            decline, client = trial.get("decline_line"), trial.get("x_request_id")
            if (oc.get("terminal") != "recovered" or oc.get("caught_stale_lease") is not True
                    or trial.get("decline_attribution") != "request_id" or not isinstance(decline, dict)
                    or decline.get("cause") != "lease_mismatch" or decline.get("kind") != "post_read"
                    or not isinstance(client, str) or not client or not isinstance(decline.get("req"), str)
                    or not (decline["req"] == client or decline["req"].startswith(f"cmpl-{client}-"))):
                why.append("request-attributed normalized invalid-lease rejection")
    if why:
        return out
    out.update(status="fail" if failed else "pass", missing_or_failed=list(dict.fromkeys(failed)),
               evidence="exported-run integrity manifest; local observations, not remote attestation")
    return out


def export_case(destination: Path, checkout: Path | None = None):
    from .case_bundle import export_bundle
    return export_bundle(destination, checkout)


def _model_sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _model_files(snapshot):
    return {str(p.relative_to(snapshot)): _model_sha(p)
            for p in snapshot.rglob("*") if p.is_file() and p != snapshot / "sdcguard-model-pin.json"}


def run_export(a):
    """Owner-run only. Execute the original harness with the one saved spec."""
    from importlib.metadata import distribution
    import os
    from .assessment import inspect_engine
    case, root = historical_case(), a.case_dir.resolve()
    harness, candidate = root / "existing_case.py", root / "candidate.json"
    if (hashlib.sha256(harness.read_bytes()).hexdigest() != case["reviewed_harness_sha256"]
            or json.loads(candidate.read_text()) != case["candidate"]):
        raise ValueError("exported runner or saved condition differs from the reviewed pin")
    identity = inspect_engine(a.engine_root, "0.32.0")
    if identity.get("variant") != ("stock" if a.variant == "unpatched" else "fixed"):
        raise ValueError("engine source does not match the pinned variant")
    dist = distribution("vllm")
    if dist.version != "0.32.0" or inspect_engine(dist.locate_file(""), "0.32.0") != identity:
        raise ValueError("running interpreter does not contain the inspected engine")
    nixl_version = distribution("nixl").version
    if nixl_version != "1.4.1":
        raise ValueError("only the pinned NIXL 1.4.1 is supported")
    tool = distribution("sdcguard")
    if Path(tool.locate_file("sdcguard")).resolve() != Path(__file__).resolve().parent:
        raise ValueError("replay requires the reviewed clean-installed package, not a checkout override")
    bindir = str(Path(sys.executable).parent)
    console = shutil.which("vllm", path=bindir)
    if not console:
        raise ValueError("matching vllm console entry point unavailable")
    first = Path(console).read_text().splitlines()[0]
    if not first.startswith("#!") or Path(first[2:]).resolve() != Path(sys.executable).resolve():
        raise ValueError("vllm console entry point uses another interpreter")
    snapshot = a.model_snapshot.resolve()
    explicit_pin = getattr(a, "model_pin", None)
    default_pin = root / "MODEL_PIN.json"
    pin_path = (explicit_pin if explicit_pin is not None else default_pin if default_pin.is_file()
                else snapshot / "sdcguard-model-pin.json")
    pin = json.loads(pin_path.read_text())
    if (not isinstance(pin, dict) or pin.get("model") != case["model"]
            or re.fullmatch(r"[0-9a-f]{40}", str(pin.get("revision", ""))) is None
            or snapshot.name != pin["revision"] or pin != replay_model_pin()
            or not pin.get("files") or _model_files(snapshot) != pin["files"]):
        raise ValueError("resolved model snapshot and complete integrity pin required")
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "complete": False, "variant": a.variant, "engine_identity": identity,
                "nixl_version": nixl_version,
                "model_path": str(snapshot), "model_pin": pin, "model_pin_path": str(pin_path.resolve()),
                "candidate": case["candidate"],
                "harness_sha256": case["reviewed_harness_sha256"]}
    mp = a.out / "RUN_MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, PATH=bindir + os.pathsep + os.environ.get("PATH", ""), HF_HUB_OFFLINE="1",
               SDCGUARD_PACKAGED_REPLAY="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    cmd = [sys.executable, "-I", "-B", str(harness), "--out", str(a.out.resolve()), "--model", str(snapshot),
           "--revision", pin["revision"], "--variant", a.variant, "--mode", "compare", "--candidate", str(candidate),
           "--lease", "4", "--blocks", "48", "--others", "8", "--baseline-reps", "3", "--reps", "5"]
    manifest["harness_exit_code"] = subprocess.call(cmd, env=env)
    result = json.loads((a.out / "RESULT.json").read_text())
    manifest["complete"] = (_model_files(snapshot) == pin["files"]
                            and inspect_engine(a.engine_root, "0.32.0") == identity
                            and hashlib.sha256(harness.read_bytes()).hexdigest() == case["reviewed_harness_sha256"]
                            and json.loads(candidate.read_text()) == case["candidate"])
    manifest["result_canonical_sha256"] = digest(result)
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    return judge(result, a.variant, manifest)


def main(argv=None):
    p = argparse.ArgumentParser(prog="sdcguard regression", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("destination", type=Path)
    e.add_argument("--checkout", type=Path, help="optional matching source checkout; packaged case is the default")
    v = sub.add_parser("verify")
    v.add_argument("directory", type=Path)
    j = sub.add_parser("judge")
    j.add_argument("result", type=Path)
    j.add_argument("--variant", choices=("unpatched", "fixed"), required=True)
    j.add_argument("--manifest", type=Path)
    r = sub.add_parser("run")
    r.add_argument("--case-dir", type=Path, required=True)
    r.add_argument("--variant", choices=("unpatched", "fixed"), required=True)
    r.add_argument("--engine-root", type=Path, required=True)
    r.add_argument("--model-snapshot", type=Path, required=True)
    r.add_argument("--model-pin", type=Path, help="optional external pin; packet MODEL_PIN.json is the default")
    r.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "export":
            export_case(a.destination, a.checkout)
            print(f"Exported saved case to {a.destination}; minimality and GPU rerun pending")
            return 0
        if a.cmd == "verify":
            from .case_bundle import verify_bundle
            report = verify_bundle(a.directory)
            print(json.dumps(report, indent=2))
            return 0 if report["status"] == "verified" else 2
        report = run_export(a) if a.cmd == "run" else judge(json.loads(a.result.read_text()), a.variant,
            json.loads(a.manifest.read_text()) if a.manifest else None)
    except (OSError, ValueError, TypeError, KeyError, IndexError, PackageNotFoundError) as exc:
        report = {"status": "inconclusive", "missing_or_failed": [str(exc)]}
    print(json.dumps(report, indent=2))
    return {"pass": 0, "fail": 1, "inconclusive": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
