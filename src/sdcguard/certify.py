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
    """Write <prefix>.key (private, PEM, readable by the owner only) and <prefix>.pub (public, PEM). Never
    replaces either file: a replaced key loses the fleet's signing key, and every certificate it signed would then
    read as signed by another key."""
    serialization, Ed25519PrivateKey, _ = _crypto()
    priv = Ed25519PrivateKey.generate()
    key_path, pub_path = prefix + ".key", prefix + ".pub"
    for p in (key_path, pub_path):
        if os.path.lexists(p):
            raise FileExistsError(f"{p} already exists; keygen never replaces a key (move it away first)")
    pem = priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())
    # O_EXCL: a file created in the meantime is never written into (it would keep its own, maybe looser, mode)
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    with open(pub_path, "xb") as f:
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


def load_trusted(path: str):
    """The public key certificates must be signed with. Raises when it can't be used (missing, unreadable, a
    private key given by mistake, not Ed25519): an unusable key is an error, never the same as no key."""
    _, _, Ed25519PublicKey = _crypto()
    try:
        pub = load_public(path)
    except ValueError as e:
        raise ValueError(f"{path} is not a PEM public key (a private key, or another file?)") from e
    if not isinstance(pub, Ed25519PublicKey):
        raise ValueError(f"{path} is not an Ed25519 public key (from sdcguard keygen)")
    return pub


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


def artifact_problems(cert: dict, out_dir: str) -> dict:
    """Check every artifact declared by a certificate, including absent files.

    Signature validity and artifact availability are different facts: a detached
    certificate can have a valid signature without supplying its evidence.
    """
    problems = {}
    manifest = {}
    for field in ("evidence_sha256", "files_sha256"):
        entries = cert.get(field) or {}
        if not isinstance(entries, dict):
            problems[field] = "malformed artifact manifest"
            continue
        for name, digest in entries.items():
            if not isinstance(name, str) or os.path.basename(name) != name or name in ("", ".", ".."):
                problems[str(name)] = "invalid artifact name"
                continue
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                problems[name] = "invalid SHA-256 digest"
                continue
            if name in manifest and manifest[name] != digest:
                problems[name] = "conflicting artifact digests"
            manifest[name] = digest
    for name, digest in manifest.items():
        actual = _sha256(os.path.join(out_dir, name))
        if actual is None:
            problems[name] = "missing or unreadable since the certificate was issued"
        elif actual != digest:
            problems[name] = "changed since the certificate was issued"
    return problems


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
            digest = _sha256(e + ext)
            if digest is None:
                raise FileNotFoundError(f"cannot certify missing evidence: {e + ext}")
            name = os.path.basename(e) + ext
            if name in evidence and evidence[name] != digest:
                raise ValueError(f"evidence files have the same name and different contents: {name}")
            evidence[name] = digest
    corrupted = {"training": summary.get("corrupted_results_training"),
                 "serving": summary.get("corrupted_results_serving")}
    if full and full.get("wrong"):
        w = full["wrong"]
        corrupted.update(stress=w.get("stress", 0), math_modes=w.get("math_modes", 0),
                         memory_words=w.get("memory_words", 0))
    simulated = bool(summary.get("SIMULATED_FAULTS")) or bool((full or {}).get("simulated_faults"))
    # the files a fleet report or a person reads the result from: a change to them after signing is detected
    files = {}
    out_dir = (full or {}).get("out_dir") or summary.get("out_dir")
    if out_dir:
        required = ("check.json", "result.json") if full else ("summary.json",)
        for name in ("summary.json", "check.json", "result.json"):
            digest = _sha256(os.path.join(out_dir, name))
            if digest is not None:
                files[name] = digest
            elif name in required:
                raise FileNotFoundError(f"cannot certify missing result: {os.path.join(out_dir, name)}")
    cert = {
        "format": FORMAT,
        "issued_at": now.isoformat(),
        "valid_until": (now + _dt.timedelta(days=valid_days)).isoformat(),
        "issuer": issuer,
        # the run it vouches for: a later run's files in the same folder are never judged by it
        "run_id": (full or {}).get("run_id") or summary.get("run_id"),
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
    cert = make({**summary, "out_dir": out_dir}, issuer=issuer, valid_days=valid_days, check=check, full=full)
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
    result.json) present and unchanged. ok says the certificate is authentic and its declared artifacts are
    available, not that the GPU
    passed: read "result" (or use `sdcguard verify --require-pass`). A public key that can't be used raises."""
    # loaded first, outside the signature check below: a key that can't be used raises (the CLI exits 2) instead
    # of reading as "the signature doesn't match" with no key to compare against
    trusted = load_trusted(pubkey_path) if pubkey_path else None
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
            if trusted is not None:
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
    artifacts = artifact_problems(cert, here)
    match = not artifacts
    problems.extend(f"{name} {why}" for name, why in artifacts.items())
    out["artifact_problems"] = artifacts
    out["evidence_files_match"] = match
    if cert.get("simulated_faults"):
        problems.append("this screen injected SIMULATED faults (a demo, not a real result)")
    out["problems"] = problems
    # with a trusted key given, only a signature made with that key counts
    key_ok = out["trusted_key"] is True if trusted is not None else True
    out["ok"] = (out["signature_valid"] and key_ok and out["expired"] is False and match
                 and not cert.get("simulated_faults") and cert.get("format") == FORMAT)
    return out
