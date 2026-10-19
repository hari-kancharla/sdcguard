"""Signed GPU correctness certificates: what a screen found, per GPU, in a form anyone can check.

    sdcguard keygen --out fleet                  # fleet.key (keep secret) and fleet.pub (share)
    sdcguard certify --minutes 10 --key fleet.key
    sdcguard verify screen_out/GPU-.../certificate.json --pubkey fleet.pub

A certificate records which GPU was tested (UUID, serial number, PCI slot, VBIOS, driver), the exact
workload (model size, seed, sdcguard and PyTorch versions), how much of it was checked (matmuls, layer
checks, TFLOP), the result (PASS: every check passed; FAIL: corrupted results, with the SHA-256 of each
saved evidence file), when, and until when it is meant to be valid. It is signed with Ed25519.

What a signature proves: the certificate hasn't been changed since the holder of that key issued it.
Who to trust is up to the reader: a renter who screens a rented GPU and signs with their own key holds
evidence that doesn't depend on the provider; a provider or marketplace that signs with its key vouches
for the result. The test runs on the GPU being tested, so it is built for faulty hardware, not for a
provider that cheats on purpose.

Signing and verifying need the `cryptography` package (pip install "sdcguard[certify]").
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import os
from typing import Optional

FORMAT = "sdcguard-certificate/1"


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    except ImportError as e:  # pragma: no cover
        raise RuntimeError('signing needs the cryptography package: pip install "sdcguard[certify]"') from e
    return serialization, Ed25519PrivateKey, Ed25519PublicKey


def _raw_public(pub) -> bytes:
    serialization, _, _ = _crypto()
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id(pub) -> str:
    """First 16 hex digits of the SHA-256 of the raw public key."""
    return hashlib.sha256(_raw_public(pub)).hexdigest()[:16]


def keygen(prefix: str) -> tuple[str, str]:
    """Write <prefix>.key (private, PEM, readable by the owner only) and <prefix>.pub (public, PEM)."""
    serialization, Ed25519PrivateKey, _ = _crypto()
    priv = Ed25519PrivateKey.generate()
    key_path, pub_path = prefix + ".key", prefix + ".pub"
    pem = priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    with open(pub_path, "wb") as f:
        f.write(priv.public_key().public_bytes(serialization.Encoding.PEM,
                                               serialization.PublicFormat.SubjectPublicKeyInfo))
    return key_path, pub_path


def load_private(path: str):
    serialization, _, _ = _crypto()
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def load_public(path: str):
    serialization, _, _ = _crypto()
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def canonical(cert: dict) -> bytes:
    """The bytes that are signed: the certificate without its signature, keys sorted, no spaces."""
    body = {k: v for k, v in cert.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def _sha256(path: str) -> Optional[str]:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)


def make(summary: dict, issuer: str = "", valid_days: float = 30.0, now: Optional[_dt.datetime] = None,
         check: Optional[dict] = None, full: Optional[dict] = None) -> dict:
    """An unsigned certificate from a screen summary (screen.screen_one's return value). full: `sdcguard
    check`'s complete result (check.json), whose evidence and wrong-result counts cover every phase (the
    workload, the stress check and the math modes), not only the workload."""
    now = now or _now()
    g = summary.get("gpu", {})
    evidence = {}
    for e in (full or {}).get("evidence", summary.get("evidence", [])):
        for ext in (".json", ".pt"):
            if os.path.exists(e + ext):
                evidence[os.path.basename(e) + ext] = _sha256(e + ext)
    corrupted = {"training": summary.get("corrupted_results_training"),
                 "serving": summary.get("corrupted_results_serving")}
    if full and full.get("wrong"):
        w = full["wrong"]
        corrupted.update(stress=w.get("stress", 0), math_modes=w.get("math_modes", 0),
                         memory_words=w.get("memory_words", 0))
    simulated = bool(summary.get("SIMULATED_FAULTS")) or bool((full or {}).get("simulated_faults"))
    # the files a fleet report or a person reads the result from: a change to them after signing is detected
    files = {}
    if full and full.get("out_dir"):
        for name in ("check.json", "result.json"):
            digest = _sha256(os.path.join(full["out_dir"], name))
            if digest:
                files[name] = digest
    cert = {
        "format": FORMAT,
        "issued_at": now.isoformat(),
        "valid_until": (now + _dt.timedelta(days=valid_days)).isoformat(),
        "issuer": issuer,
        "gpu": {k: g.get(k) for k in ("gpu_uuid", "gpu_name", "serial", "pci_bus_id", "vbios", "driver", "host")},
        "software": {"sdcguard": summary.get("sdcguard"), "torch": summary.get("torch")},
        "workload": {"size": summary.get("size"), "seed": summary.get("seed"), "minutes": summary.get("minutes"),
                     "training_steps": summary.get("training_steps"), "serving_passes": summary.get("serving_passes"),
                     "matmuls_checked": summary.get("matmuls_checked"),
                     "linear_layer_checks_while_serving": summary.get("linear_layer_checks_while_serving"),
                     "approx_matmul_tflop": summary.get("approx_matmul_tflop"),
                     "unjudged_training_checks": summary.get("unjudged_training_checks", 0),
                     "serving_passes_with_nonfinite_output": summary.get("serving_passes_with_nonfinite_output", 0)},
        "result": summary.get("result"),
        "corrupted_results": corrupted,
        "worst_clean_score": summary.get("worst_clean_z"),
        "alarm_line": summary.get("alarm_line", 6.0),
        "evidence_sha256": evidence,
        "files_sha256": files,
        "hardware_errors": list((full or {}).get("hardware_errors") or []),
        "simulated_faults": simulated,
        "meaning": ("PASS: every matrix multiply checked on this GPU matched its checksums, and every check could "
                    "be judged (counts above). It says what was checked, not that the GPU has no defect. INCOMPLETE: "
                    "nothing wrong was found, but part of the check couldn't run or couldn't judge its results; this certificate doesn't vouch for the GPU. "
                    "FAIL: some results were wrong (the evidence files hold the failed matmuls' inputs and outputs, "
                    "and `sdcguard replay <evidence>.pt` recomputes them on any GPU), or the driver reported memory "
                    "errors it couldn't correct (hardware_errors)."),
    }
    if check:   # from `sdcguard check`: speed, memory test, repeatability, health, and its overall verdict
        cert["check"] = {k: check.get(k) for k in ("result", "checked", "wrong", "headroom", "repeatable",
                                                    "fingerprint", "speed", "health", "notes", "digest")}
        if check.get("result") in ("FAIL", "SUSPECT", "INCOMPLETE") and cert["result"] != "FAIL":
            cert["result"] = check["result"]
    return cert


def sign(cert: dict, key_path: str) -> dict:
    priv = load_private(key_path)
    pub = priv.public_key()
    cert = dict(cert)
    cert["signer"] = {"algorithm": "ed25519", "key_id": key_id(pub),
                      "public_key": base64.b64encode(_raw_public(pub)).decode()}
    cert["signature"] = base64.b64encode(priv.sign(canonical(cert))).decode()
    return cert


def write(summary: dict, out_dir: str, key_path: Optional[str] = None, issuer: str = "",
          valid_days: float = 30.0, check: Optional[dict] = None, full: Optional[dict] = None) -> str:
    """Certificate for one screened GPU, written next to its summary as certificate.json."""
    cert = make(summary, issuer=issuer, valid_days=valid_days, check=check, full=full)
    if key_path:
        cert = sign(cert, key_path)
    else:
        cert["signature"] = None
    path = os.path.join(out_dir, "certificate.json")
    with open(path, "w") as f:
        json.dump(cert, f, indent=2, default=str)
    return path


def verify(path: str, pubkey_path: Optional[str] = None, now: Optional[_dt.datetime] = None) -> dict:
    """Check a certificate. Returns {"ok", "signed", "signature_valid", "trusted_key", "expired",
    "evidence_files_match", "result", "gpu", "problems"}. ok means: signature valid, signed by the given
    public key (when one is given), not expired, evidence files and the check's result files (check.json,
    result.json, if present next to it) unchanged. ok says the certificate is authentic, not that the GPU
    passed: read "result" (or use `sdcguard verify --require-pass`)."""
    with open(path) as f:
        cert = json.load(f)
    problems = []
    out = {"path": path, "result": cert.get("result"), "gpu": cert.get("gpu"), "issued_at": cert.get("issued_at"),
           "valid_until": cert.get("valid_until"), "issuer": cert.get("issuer"),
           "simulated_faults": cert.get("simulated_faults")}
    if cert.get("format") != FORMAT:
        problems.append(f"unknown format {cert.get('format')!r}")
    sig = cert.get("signature")
    signer = cert.get("signer") or {}
    out["signed"] = bool(sig)
    out["signature_valid"] = False
    out["trusted_key"] = None
    if sig:
        try:
            serialization, _, Ed25519PublicKey = _crypto()
            pub = Ed25519PublicKey.from_public_bytes(base64.b64decode(signer.get("public_key", "")))
            pub.verify(base64.b64decode(sig), canonical(cert))
            out["signature_valid"] = True
            out["key_id"] = key_id(pub)
            if pubkey_path:
                trusted = load_public(pubkey_path)
                out["trusted_key"] = _raw_public(trusted) == _raw_public(pub)
                if not out["trusted_key"]:
                    problems.append("signed by a different key than the one you trust")
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001  (bad signature, bad key bytes)
            problems.append("signature does not match the certificate (changed after signing, or a bad key)")
    else:
        problems.append("not signed")
    now = now or _now()
    try:
        out["expired"] = _dt.datetime.fromisoformat(cert.get("valid_until")) < now
    except (TypeError, ValueError):
        out["expired"] = None
        problems.append("no valid_until date")
    if out["expired"]:
        problems.append(f"expired on {cert.get('valid_until')}")
    here = os.path.dirname(os.path.abspath(path))
    match = True
    for name, digest in {**(cert.get("evidence_sha256") or {}), **(cert.get("files_sha256") or {})}.items():
        p = os.path.join(here, os.path.basename(str(name)))
        if os.path.exists(p) and _sha256(p) != digest:
            match = False
            problems.append(f"{os.path.basename(str(name))} changed since the certificate was issued")
    out["evidence_files_match"] = match
    if cert.get("simulated_faults"):
        problems.append("this screen injected SIMULATED faults (a demo, not a real result)")
    out["problems"] = problems
    out["ok"] = (out["signature_valid"] and out["trusted_key"] is not False and out["expired"] is False and match
                 and not cert.get("simulated_faults") and cert.get("format") == FORMAT)
    return out
