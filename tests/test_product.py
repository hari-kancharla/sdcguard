"""The product layer: CLI, certificates, fleet report, alerts (webhook, Prometheus), node health check."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")


def _cli(args, cwd, timeout=900):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-m", "sdcguard", *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout)


def _screen(tmp_path, name="ok", faults=0.0):
    from sdcguard.screen import screen_one
    return screen_one(torch.device("cpu"), minutes=5, size="tiny", out_root=str(tmp_path / name), max_steps=3,
                      simulate_faults=faults)


def test_certificate_sign_verify_and_tamper(tmp_path):
    pytest.importorskip("cryptography")
    from sdcguard import certify
    key, pub = certify.keygen(str(tmp_path / "fleet"))
    assert oct(os.stat(key).st_mode & 0o777) == "0o600"
    s = _screen(tmp_path)
    d = os.path.dirname(os.path.join(str(tmp_path / "ok"), "cpu", "summary.json"))
    path = certify.write(s, d, key_path=key, issuer="test")
    r = certify.verify(path, pub)
    assert r["ok"] and r["signature_valid"] and r["trusted_key"] and r["result"] == "PASS"
    cert = json.load(open(path))
    assert cert["workload"]["matmuls_checked"] == s["matmuls_checked"] and cert["signer"]["algorithm"] == "ed25519"
    cert["workload"]["matmuls_checked"] += 1                      # any change breaks the signature
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(cert))
    assert not certify.verify(str(bad), pub)["ok"]
    _, other = certify.keygen(str(tmp_path / "other"))
    r2 = certify.verify(path, other)
    assert r2["signature_valid"] and r2["trusted_key"] is False and not r2["ok"]


def test_certificate_expiry_unsigned_and_evidence_hashes(tmp_path):
    pytest.importorskip("cryptography")
    import datetime as dt

    from sdcguard import certify
    key, pub = certify.keygen(str(tmp_path / "k"))
    s = _screen(tmp_path, "bad", faults=0.05)
    d = os.path.join(str(tmp_path / "bad"), "cpu")
    assert s["result"] == "FAIL" and s["evidence"]
    path = certify.write(s, d, key_path=key, valid_days=1)
    cert = json.load(open(path))
    assert cert["simulated_faults"] and len(cert["evidence_sha256"]) == 2 * len(s["evidence"])
    r = certify.verify(path, pub)
    assert r["signature_valid"] and not r["ok"]                    # simulated faults never verify as a real result
    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=2)
    assert certify.verify(path, pub, now=later)["expired"]
    with open(s["evidence"][0] + ".json", "a") as f:               # evidence changed after the fact
        f.write(" ")
    assert certify.verify(path, pub)["evidence_files_match"] is False
    unsigned = certify.write(s, d)
    assert certify.verify(unsigned)["signed"] is False


def test_fleet_report(tmp_path):
    from sdcguard import fleet
    _screen(tmp_path, "a")
    _screen(tmp_path, "b", faults=0.05)
    rows = fleet.collect([str(tmp_path / "a"), str(tmp_path / "b")])
    # both runs are "cpu" without a GPU UUID, so they stay separate rows (keyed by folder)
    assert len(rows) == 2 and rows[0]["result"] == "FAIL"
    rep = fleet.write(rows, str(tmp_path / "rep"))
    assert rep["stats"]["failed"] == 1 and rep["stats"]["passed"] == 1
    page = open(tmp_path / "rep" / "fleet_report.html").read()
    assert "GPU fleet report" in page and "FAIL" in page and "\u2014" not in page
    # The simulated FAIL remains in the report and has replay artifacts, but
    # must never become an operational quarantine instruction.
    assert open(tmp_path / "rep" / "quarantine.txt").read() == ""
    import zipfile
    z = zipfile.ZipFile(list(rep["rma"].values())[0])
    assert "README.txt" in z.namelist() and any(n.endswith(".pt") for n in z.namelist())


def test_cli_end_to_end(tmp_path):
    pytest.importorskip("cryptography")
    assert _cli(["keygen", "--out", "fleet"], tmp_path).returncode == 0
    r = _cli(["certify", "--minutes", "0.03", "--size", "tiny", "--key", "fleet.key", "--out", "shots"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "certificate: PASS" in r.stdout
    import glob                                     # shots/cpu/ here; shots/GPU-<uuid>/ on a GPU machine
    cert = os.path.relpath(glob.glob(str(tmp_path / "shots" / "*" / "certificate.json"))[0], tmp_path)
    v = _cli(["verify", cert, "--pubkey", "fleet.pub"], tmp_path)
    assert v.returncode == 0 and v.stdout.startswith("VALID")
    rep = _cli(["report", "shots", "--out", "rep"], tmp_path)
    assert rep.returncode == 0 and "1 PASS" in rep.stdout
    n = _cli(["nhc", "--minutes", "0.03", "--size", "tiny", "--out", "nhc"], tmp_path)
    assert n.returncode == 0 and n.stdout.startswith("OK:"), n.stdout + n.stderr
    nf = _cli(["nhc", "--minutes", "0.03", "--size", "tiny", "--out", "nhc2", "--simulate-faults", "0.05"], tmp_path)
    assert nf.returncode == 2 and nf.stdout.startswith("SIMULATED:")
    assert not (tmp_path / "nhc2" / "status.json").exists()


class _Hook(BaseHTTPRequestHandler):
    got: list = []

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        _Hook.got.append(json.loads(self.rfile.read(n)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_webhook_and_metrics_on_a_weight_flip():
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard import ServeGuard
    srv = HTTPServer(("127.0.0.1", 0), _Hook)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Hook.got.clear()
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64)).to(torch.bfloat16)
    from sdcguard.alerts import Metrics
    g = ServeGuard(webhook=f"http://127.0.0.1:{srv.server_port}/hook", metrics=Metrics()).attach(m)
    x = torch.randn(4, 64).to(torch.bfloat16)
    m(x)
    assert g.poll(wait=True) == []
    text = generate_latest(g.metrics.registry).decode()
    assert "sdcguard_layers_checked 2.0" in text and "sdcguard_reads_total 1.0" in text
    with torch.no_grad():
        m[2].weight.view(torch.int16)[3, 5] ^= (1 << 14)
    m(x)
    ev = g.poll(wait=True)
    assert len(ev) == 1
    import time
    for _ in range(50):
        if _Hook.got:
            break
        time.sleep(0.1)
    srv.shutdown()
    assert _Hook.got and "sdcguard ALERT: 2" in _Hook.got[0]["text"] and "weights changed" in _Hook.got[0]["text"]
    text = generate_latest(g.metrics.registry).decode()
    assert 'sdcguard_alerts_total{UUID="",layer="2"} 1.0' in text          # (no GPU here: no UUID)


def test_describe_and_chain():
    from sdcguard.alerts import chain, describe
    from sdcguard.events import SDCEvent
    ev = SDCEvent(detector="checksum", step=7, detail={"where": "blocks.3.attn.k", "max_z": 72.7},
                  identity={"gpu_uuid": "GPU-abc", "gpu_name": "NVIDIA H100", "pci_bus_id": "0000:0B:00.0", "host": "n1"})
    line = describe(ev)
    assert "blocks.3.attn.k on GPU-abc" in line and "72.7x" in line and "step 7" in line
    seen = []
    cb = chain(None, seen.append, lambda e: 1 / 0, seen.append)     # a failing callback doesn't stop the others
    cb(ev)
    assert len(seen) == 2


def test_a_later_pass_doesnt_clear_an_earlier_failure(tmp_path):
    """An intermittent GPU: FAIL on Monday, PASS on Tuesday. It stays in quarantine, with Monday's evidence."""
    import json
    import os
    import time
    from sdcguard import fleet
    gpu = {"gpu_uuid": "GPU-aaaa-bbbb", "gpu_name": "NVIDIA H100 80GB HBM3", "host": "node17"}
    for day, res, bad in (("mon", "FAIL", 3), ("tue", "PASS", 0)):
        d = tmp_path / day / "GPU-aaaa-bbbb"
        d.mkdir(parents=True)
        ev = [str(d / "evidence_0")] if bad else []
        if bad:
            (d / "evidence_0.json").write_text("{}")
        (d / "summary.json").write_text(json.dumps({"gpu": gpu, "result": res, "corrupted_results_training": bad,
                                                    "matmuls_checked": 1000, "evidence": ev}))
        if day == "mon":
            old = time.time() - 86400
            os.utime(d / "summary.json", (old, old))
    rows = fleet.collect([str(tmp_path / "mon"), str(tmp_path / "tue")])
    assert [r["result"] for r in rows] == ["SUSPECT"] and "doesn't clear" in rows[0]["notes"][0]
    assert [h["result"] for h in rows[0]["history"]] == ["FAIL", "PASS"]
    rep = fleet.write(rows, str(tmp_path / "out"))
    assert "GPU-aaaa-bbbb" in (tmp_path / "out" / "quarantine.txt").read_text() and rep["rma"]
    assert fleet.collect([str(tmp_path / "mon"), str(tmp_path / "tue")], window_days=0.5)[0]["result"] == "PASS"
