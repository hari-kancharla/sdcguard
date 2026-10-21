"""Certificates and their keys (certify.py, `sdcguard keygen`, `sdcguard verify`): a trusted key that can't be used
is an error and never reads as trusted, keygen never replaces a key, a certificate vouches only for the run it was
issued for, and the fleet report names the GPU a trusted certificate names."""
import contextlib
import io
import json
import os
import stat
import time
from unittest.mock import patch

import pytest

from sdcguard import certify, check, cli, fleet, modes

pytest.importorskip("cryptography")


def _main(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(argv)
    return code, buf.getvalue()


def _check(out, *extra):
    return _main(check.main, ["--device", "cpu", "--minutes", "0.01", "--out", str(out), "--offline", *extra])


def _wrong_int8(mode, a, b, real=modes.gemm):
    c = real(mode, a, b)
    if mode == "int8":
        c[0, 0] += 1
    return c


def test_a_trusted_key_that_cant_be_used_is_an_error_never_valid(tmp_path):
    """verify() used to load --pubkey inside the try that guards the signature check: a missing file, the private
    key passed by mistake (fleet.key for fleet.pub) or a key of another kind read as "signature does not match",
    left trusted_key None, and None counted as trusted, so `sdcguard verify cert.json --pubkey <typo>
    --require-pass` said VALID and exited 0 for a certificate signed with anyone's key. Such a key now raises, the
    commands exit 2 (couldn't tell), and with a usable key only a signature made with it is ok."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    attacker_key, _ = certify.keygen(str(tmp_path / "attacker"))
    other = tmp_path / "ec.pub"
    other.write_bytes(ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    d = tmp_path / "g"
    d.mkdir()
    s = {"result": "PASS", "gpu": {"gpu_uuid": "GPU-1"}, "matmuls_checked": 10}
    (d / "summary.json").write_text(json.dumps(s))
    path = certify.write(s, str(d), key_path=attacker_key)
    for bad in (str(tmp_path / "missing.pub"), attacker_key, str(other)):
        with pytest.raises((OSError, ValueError)):
            certify.verify(path, bad)
        code, out = _main(cli.main, ["verify", path, "--pubkey", bad, "--require-pass"])
        assert code == 2 and not out.startswith("VALID"), out
        code, out = _main(cli.main, ["report", str(d), "--pubkey", bad, "--out", str(tmp_path / "rep")])
        assert code == 2, out
    _, trusted = certify.keygen(str(tmp_path / "fleet"))
    v = certify.verify(path, trusted)
    assert v["signature_valid"] and v["trusted_key"] is False and not v["ok"]


def test_keygen_never_overwrites_a_key(tmp_path):
    """`sdcguard keygen --out fleet` run a second time silently replaced fleet.key and fleet.pub: the fleet's signing
    key was gone, and every certificate it signed read as signed by another key. O_TRUNC also kept an existing
    file's permissions, so a world-readable fleet.key stayed world-readable with the new secret in it."""
    prefix = str(tmp_path / "fleet")
    assert _main(cli.main, ["keygen", "--out", prefix])[0] == 0
    first = open(prefix + ".key", "rb").read()
    assert stat.S_IMODE(os.stat(prefix + ".key").st_mode) & 0o077 == 0
    code, out = _main(cli.main, ["keygen", "--out", prefix])
    assert code == 2 and "already exists" in out and open(prefix + ".key", "rb").read() == first
    loose = str(tmp_path / "loose")
    with open(loose + ".key", "w") as f:
        f.write("placeholder")
    os.chmod(loose + ".key", 0o644)
    with pytest.raises(FileExistsError):
        certify.keygen(loose)
    assert open(loose + ".key").read() == "placeholder"
    pub_only = str(tmp_path / "pub_only")
    with open(pub_only + ".pub", "w") as f:
        f.write("someone's public key")
    with pytest.raises(FileExistsError):              # never half a pair
        certify.keygen(pub_only)
    assert not os.path.exists(pub_only + ".key")


def test_a_later_uncertified_check_isnt_judged_by_the_last_runs_certificate(tmp_path):
    """write_result wrote certificate.json only when asked and never removed an old one, and a certificate recorded
    no run: a clean `sdcguard check` into a folder holding last run's signed certificate left it beside the new
    files, and with --pubkey the fleet report found them 'changed since the certificate was issued' (it reads like
    tampering) and turned the new PASS into INCOMPLETE; a stale trusted FAIL would have kept the GPU failed the
    same way. A certificate now names its run, a check that writes none removes the old one, and one for another
    run never vouches for the files beside it."""
    key, pub = certify.keygen(str(tmp_path / "k"))
    out = tmp_path / "o"
    assert _check(out, "--key", key)[0] == 0                # last month: a certified PASS
    d = out / "cpu"
    cert = json.load(open(d / "certificate.json"))
    assert cert["run_id"] == json.load(open(d / "check.json"))["run_id"]
    assert fleet.collect([str(out)], pubkey=pub)[0]["certificate_trusted"]
    assert _check(out)[0] == 0                              # today: a clean check, no certificate asked for
    assert not (d / "certificate.json").exists()
    row = fleet.collect([str(out)], pubkey=pub)[0]
    assert row["result"] == "PASS", (row["result"], row["notes"][:3])
    (d / "certificate.json").write_text(json.dumps(cert))  # copied back in: it is for another run
    row = fleet.collect([str(out)], pubkey=pub)[0]
    assert not row["certificate_trusted"] and row["result"] == "INCOMPLETE"
    assert any("another run" in n for n in row["notes"]), row["notes"]


def test_a_trusted_certificate_fail_stays_on_the_gpu_it_names(tmp_path):
    """With a certificate signed by the trusted key, collect() took the verdict from the certificate but the row's
    identity (UUID, host, PCI slot, serial: what quarantine.txt lists) from the unsigned files, summary.json first.
    Editing summary.json's gpu (keeping its run_id) moved the signed FAIL onto another GPU in quarantine.txt."""
    key, pub = certify.keygen(str(tmp_path / "k"))
    out = tmp_path / "o"
    with patch.object(modes, "gemm", side_effect=_wrong_int8):
        assert _check(out, "--key", key)[0] == 1
    d = out / "cpu"
    cert = json.load(open(d / "certificate.json"))
    s = json.load(open(d / "summary.json"))
    s["gpu"] = dict(s["gpu"], gpu_uuid="GPU-innocent", serial="1650922000001", pci_bus_id="0000:9A:00.0")
    (d / "summary.json").write_text(json.dumps(s))
    rows = fleet.collect([str(out)], pubkey=pub)
    assert rows[0]["certificate_trusted"] and rows[0]["result"] == "FAIL"
    assert rows[0]["gpu_uuid"] == cert["gpu"]["gpu_uuid"] and rows[0]["serial"] == cert["gpu"]["serial"]
    assert any("another GPU than the certificate" in n for n in rows[0]["notes"]), rows[0]["notes"]
    fleet.write(rows, str(tmp_path / "rep"))
    q = (tmp_path / "rep" / "quarantine.txt").read_text()
    assert "GPU-innocent" not in q and "1650922000001" not in q, q


def test_a_pass_certified_later_doesnt_sort_after_a_later_fail(tmp_path):
    """A row's time came from a trusted certificate's issued_at, which is when the certificate was written, not when
    the GPU was checked: a PASS checked 40 days ago and certified today sorted after a FAIL checked 35 days ago, the
    FAIL fell out of the 30-day window, and the GPU read PASS. The time the run recorded decides."""
    key, pub = certify.keygen(str(tmp_path / "k"))
    now = time.time()

    def iso(t):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    a = tmp_path / "a" / "GPU-l"
    a.mkdir(parents=True)
    s = {"result": "PASS", "gpu": {"gpu_uuid": "GPU-l"}, "matmuls_checked": 10, "time": iso(now - 40 * 86400)}
    (a / "summary.json").write_text(json.dumps(s))
    certify.write(s, str(a), key_path=key)                  # certified today
    b = tmp_path / "b" / "GPU-l"
    b.mkdir(parents=True)
    (b / "summary.json").write_text(json.dumps({"result": "FAIL", "gpu": {"gpu_uuid": "GPU-l"},
                                                "corrupted_results_training": 2, "matmuls_checked": 10,
                                                "time": iso(now - 35 * 86400)}))
    rows = fleet.collect([str(tmp_path / "a"), str(tmp_path / "b")], pubkey=pub)
    assert len(rows) == 1 and rows[0]["result"] == "FAIL", (rows[0]["result"], rows[0]["history"])
    assert [h["result"] for h in rows[0]["history"]] == ["PASS", "FAIL"]
