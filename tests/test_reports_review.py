"""The second independent review of Oct 19, 2026 (verdicts, reports, certificates, the board), each finding
turned into a test built from its reproduction: what a check finds is never lost when the GPU crashes
afterwards, a part that didn't run is never a PASS, every report agrees with check.json, and every command
exits 1 only for wrong results."""
import contextlib
import io
import json
import os
import zipfile
from unittest.mock import patch

import pytest
import torch

from sdcguard import board, certify, check, cli, fleet, gpuinfo, guard, modes, screen

HERE = os.path.dirname(os.path.abspath(__file__))


def _cuda_error(*a, **k):
    raise RuntimeError("CUDA error: an illegal memory access was encountered")


@contextlib.contextmanager
def _wrong_int8():
    """The int8 math mode returns a wrong result (a stand-in for a faulty GPU)."""
    real = modes.gemm

    def corrupt(mode, a, b):
        c = real(mode, a, b)
        if mode == "int8":
            c[0, 0] += 1
        return c

    with patch.object(modes, "gemm", side_effect=corrupt):
        yield


def _main(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(argv)
    return code, buf.getvalue()


def _check(tmp_path, *extra):
    return _main(check.main, ["--device", "cpu", "--minutes", "0.01", "--out", str(tmp_path), "--offline", *extra])


def _full(tmp_path):
    return json.load(open(os.path.join(tmp_path, "cpu", "check.json")))


# --- H1: wrong results, then a crash ----------------------------------------------------------------------

def test_wrong_results_then_a_crash_in_the_workload_is_a_fail_with_evidence(tmp_path):
    with _wrong_int8(), patch.object(screen, "_loop", side_effect=_cuda_error):
        code, out = _check(tmp_path)
    full = _full(tmp_path)
    assert code == 1 and full["result"] == "FAIL" and full["wrong"]["math_modes"] > 0
    assert full["stopped"]["phase"] == "workload" and "illegal memory access" in full["stopped"]["error"]
    assert all(os.path.exists(e + ".pt") for e in full["evidence"]) and full["evidence"]
    assert any("stopped" in t for _l, t in full["notes"])
    # the fleet report and its RMA zip see the same FAIL
    code, _ = _main(cli.main, ["report", str(tmp_path), "--out", str(tmp_path / "rep")])
    assert code == 1


def test_a_crash_after_the_workload_keeps_the_fail_and_check_json(tmp_path):
    real_run = check.Fingerprinter.run
    calls = {"n": 0}

    def third_crashes(self):
        calls["n"] += 1
        if calls["n"] == 3:
            _cuda_error()
        return real_run(self)

    with _wrong_int8(), patch.object(check.Fingerprinter, "run", third_crashes):
        code, _ = _check(tmp_path)
    full = _full(tmp_path)
    assert code == 1 and full["result"] == "FAIL" and full["stopped"]["phase"] == "fingerprint"
    # summary.json (the workload alone) says PASS; the report must still say FAIL
    rows = fleet.collect([str(tmp_path)])
    assert len(rows) == 1 and rows[0]["result"] == "FAIL"


def test_a_crash_with_nothing_wrong_is_incomplete_not_could_not_run(tmp_path):
    with patch.object(check, "stress_check", side_effect=_cuda_error):
        code, _ = _check(tmp_path)
    full = _full(tmp_path)
    assert code == 2 and full["result"] == "INCOMPLETE" and full["stopped"]["phase"] == "stress"
    # nothing ran on the GPU after the error
    assert full["math_modes"].get("not_run") and full["workload"]["training_steps"] is None


def test_a_screen_that_crashes_after_finding_corruption_reports_it(tmp_path):
    calls = {"n": 0}
    real_poll = screen.ServeGuard.poll

    def poll_then_crash(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            _cuda_error()
        return real_poll(self, *a, **k)

    with patch.object(screen.ServeGuard, "poll", poll_then_crash):
        s = screen.screen_one(torch.device("cpu"), minutes=1, size="tiny", out_root=str(tmp_path), max_steps=5,
                              simulate_faults=0.5)
    assert s["result"] == "FAIL" and s["error"] and s["training_steps"] >= 1
    assert os.path.exists(os.path.join(s["out_dir"], "summary.json")) and s["evidence"]


def test_nhc_keeps_a_fail_when_a_later_screen_cant_finish(tmp_path, monkeypatch):
    out = str(tmp_path / "nhc")
    cli._nhc_status(out, "FAIL: 1 of 1 GPUs computed wrong results", 1)

    class Done:
        returncode = 2

    monkeypatch.setattr(cli, "_run_screen", lambda a, capture=False: Done())
    code, line = _main(cli.main, ["nhc", "--out", out])
    assert code == 1 and line.startswith("FAIL") and "couldn't finish" in line
    assert _main(cli.main, ["nhc", "--cached", "--out", out])[0] == 1
    monkeypatch.setattr(cli, "_run_screen", lambda a, capture=False: (_ for _ in ()).throw(OSError("no python")))
    assert _main(cli.main, ["nhc", "--out", out])[0] == 1


# --- H2, M6: parts that didn't run --------------------------------------------------------------------------

def test_a_memory_test_that_didnt_run_is_incomplete_and_on_the_card(tmp_path):
    with patch.object(check, "memory_test", return_value={"skipped": "not enough free memory"}):
        code, out = _check(tmp_path)
    full = _full(tmp_path)
    assert code == 2 and full["result"] == "INCOMPLETE"
    pub = check.public(full)
    assert pub["checked"]["memory_skipped"] == "not enough free memory"
    assert dict(check.lines(pub))["memory test"].startswith("NOT RUN")
    card = dict(pub, gpu=dict(pub["gpu"], name="NVIDIA L4"), minutes=5, result="PASS")
    card.pop("digest")
    card["digest"] = board.digest(card)
    assert board.validate(card) is not None          # relabelled PASS: rejected


def test_a_memory_test_that_got_no_memory_says_so():
    """Out of memory at the first block used to come back as 0 GiB tested with no errors."""
    real_empty = torch.empty
    calls = {"n": 0}

    def oom_after_two(*a, **k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise torch.cuda.OutOfMemoryError("out of memory")
        return real_empty(*a, **k)

    with patch.object(torch, "empty", oom_after_two):
        r = check.memory_test(torch.device("cpu"), 0.01)
    assert r == {"skipped": "no memory block could be allocated"}


def test_checks_skipped_for_lack_of_memory_make_the_screen_incomplete(tmp_path):
    real = guard.matmul_zscores
    n = {"c": 0}

    def oom_mostly(*a, **k):
        n["c"] += 1
        if n["c"] % 10:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return real(*a, **k)

    with patch.object(guard, "matmul_zscores", oom_mostly):
        s = screen.screen_one(torch.device("cpu"), minutes=0, size="tiny", out_root=str(tmp_path), max_steps=2)
    assert s["result"] == "INCOMPLETE" and s["skipped_training_checks"] > 0


def test_evidence_pairs_with_its_own_check_when_others_were_skipped():
    """A check skipped at the read-back (out of memory) used to make the recorder drop the evidence of the whole
    flush. A queued Triton check whose read-back ran out of memory stands in for it here."""
    g = screen._Recorder(sample_rate=1.0, max_pending=64)
    torch.manual_seed(0)
    A, B = torch.randn(16, 64).bfloat16(), torch.randn(64, 256).bfloat16()
    good = (A.float() @ B.float()).bfloat16()
    bad = good.clone()
    bad[3, 7] += 50
    g._queue("aten::mm", "x", None, A, B, good, None, 1.0, 1.0)
    lost = ("g", object(), 0, None, {"shape_A": (16, 64)})      # its group never reads back: skipped
    g._pending.append(lost)
    g.stats["checked"] += 1
    g.kept.append((lost, (A, B, good, None, 1.0, 1.0, "aten::mm", "lost", 0)))
    g._queue("aten::mm", "y", None, A, B, bad, None, 1.0, 1.0)
    results = g.flush()
    assert len(results) == 2 and g.stats["skipped_oom"] == 1
    assert len(g.failed) == 1 and g.failed[0][1][2] is bad and g.failed[0][1][7] == "y"


# --- H3, M1, M4, L6: certificates ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def keys(tmp_path_factory):
    d = tmp_path_factory.mktemp("keys")
    return certify.keygen(str(d / "fleet"))


def test_a_bad_key_is_refused_before_the_check_runs(tmp_path):
    code, out = _check(tmp_path, "--key", str(tmp_path / "missing.key"))
    assert code == 2 and "can't sign" in out and not os.path.exists(os.path.join(tmp_path, "cpu"))
    bad = tmp_path / "bad.key"
    bad.write_text("not a key")
    assert _check(tmp_path, "--key", str(bad))[0] == 2


def test_a_certificate_covers_every_phase_and_the_result_files(tmp_path, keys):
    key, pub = keys
    with _wrong_int8():
        code, _ = _check(tmp_path, "--key", key)
    d = os.path.join(tmp_path, "cpu")
    cert = json.load(open(os.path.join(d, "certificate.json")))
    assert code == 1 and cert["result"] == "FAIL" and cert["corrupted_results"]["math_modes"] > 0
    assert set(cert["files_sha256"]) == {"check.json", "result.json"} and cert["evidence_sha256"]
    assert certify.verify(os.path.join(d, "certificate.json"), pub)["ok"]
    # a valid FAIL certificate is authentic, but doesn't pass --require-pass
    assert _main(cli.main, ["verify", os.path.join(d, "certificate.json"), "--pubkey", pub])[0] == 0
    assert _main(cli.main, ["verify", os.path.join(d, "certificate.json"), "--pubkey", pub, "--require-pass"])[0] == 1
    assert _main(cli.main, ["verify", os.path.join(d, "certificate.json"), "--require-pass"])[0] == 2
    # editing check.json to PASS breaks the certificate, and the fleet report keeps the signed FAIL
    full = json.load(open(os.path.join(d, "check.json")))
    full.update(result="PASS", wrong={k: 0 for k in full["wrong"]})
    json.dump(full, open(os.path.join(d, "check.json"), "w"))
    v = certify.verify(os.path.join(d, "certificate.json"), pub)
    assert not v["ok"] and any("check.json changed" in p for p in v["problems"])
    row = fleet.collect([str(tmp_path)])[0]
    assert row["result"] == "FAIL" and row["changed_since_certificate"] == ["check.json"]


def test_multi_gpu_certificates_cover_every_phase(tmp_path, keys, monkeypatch):
    """The multi-GPU path wrote certificates without the stress and math-mode evidence (only the single-GPU
    path had been fixed). Two children on the CPU stand in for two GPUs."""
    key, pub = keys
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import os\n"
        "if os.environ.get('CUDA_VISIBLE_DEVICES') == '0':\n"
        "    from sdcguard import modes\n"
        "    real = modes.gemm\n"
        "    def corrupt(mode, a, b):\n"
        "        c = real(mode, a, b)\n"
        "        if mode == 'int8':\n"
        "            c[0, 0] += 1\n"
        "        return c\n"
        "    modes.gemm = corrupt\n")
    src = os.path.join(HERE, "..", "src")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(site), src, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("SDCGUARD_CHILD_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    code, out = _main(check.main, ["--minutes", "0.01", "--out", str(tmp_path / "out"), "--offline", "--key", key])
    assert code == 1, out
    certs = [os.path.join(r, "certificate.json") for r, _d, fs in os.walk(tmp_path / "out") if "certificate.json" in fs]
    assert len(certs) == 2
    failed = [json.load(open(c)) for c in certs if json.load(open(c))["result"] == "FAIL"]
    assert len(failed) == 1 and failed[0]["corrupted_results"]["math_modes"] > 0 and failed[0]["evidence_sha256"]
    assert all(certify.verify(c, pub)["ok"] for c in certs)


# --- M2: hardware errors ---------------------------------------------------------------------------------

def test_new_uncorrectable_ecc_errors_or_a_failed_row_remap_fail_the_check(tmp_path):
    before = {"ecc.errors.uncorrected.volatile.total": "0", "remapped_rows.failure": "No"}
    after = {"ecc.errors.uncorrected.volatile.total": "3", "remapped_rows.failure": "Yes"}
    with patch.object(gpuinfo, "health", side_effect=[before, after]):
        code, out = _check(tmp_path)
    full = _full(tmp_path)
    assert code == 1 and full["result"] == "FAIL" and len(full["hardware_errors"]) == 2
    pub = check.public(full)
    assert check.headline(pub) == "the driver reported memory errors it couldn't correct"
    assert "3 new uncorrectable ECC" in out
    card = dict(pub, gpu=dict(pub["gpu"], name="NVIDIA A100-SXM4-80GB"), minutes=5)
    card.pop("digest")
    card["digest"] = board.digest(dict(card, result="PASS"))
    assert board.validate(dict(card, result="PASS")) is not None


# --- M3: the board -------------------------------------------------------------------------------------

def _seed():
    seed = os.path.join(HERE, "..", "docs", "board", "seed")
    with open(os.path.join(seed, sorted(os.listdir(seed))[0])) as f:
        return json.load(f)


def _signed(r):
    r = dict(r)
    r.pop("digest", None)
    r["digest"] = board.digest(r)
    return r


@pytest.mark.parametrize("change,why", [
    (lambda r: r.pop("wrong"), "incomplete"),
    (lambda r: r.update(wrong={}), "incomplete"),
    (lambda r: r.pop("repeatable"), "repeatab"),
    (lambda r: r.update(repeatable=0), "true or false"),
    (lambda r: r.update(board={"disagrees_with_majority": True}), "SUSPECT"),
    (lambda r: r.update(notes=[["bad", "the same model on the same input gave different outputs"]]), "problem"),
    (lambda r: r["checked"].update(unjudged=-5), "whole"),
    (lambda r: r["checked"].update(results=0.5), "whole"),
    (lambda r: r.update(result="SUSPECT"), "disagrees"),
    (lambda r: r.update(result="FAIL", wrong={**r["wrong"], "gpu_fan": 3}), "part"),
    (lambda r: r["health"].update(max_temp_c=float("nan")), "NaN"),
    (lambda r: r["gpu"].update(name="CPU (Xeon)"), "not a GPU"),
    (lambda r: r["provider"].update(region="x" * 100000), "KB"),
    (lambda r: r.update(extra=1), "fields"),
    (lambda r: r.update(math_modes={"ran": ["bf16"], "matmuls": 9, "wrong": 0, "not_checked": {"fp16": "x"},
                                    "errors": {}, "typical_error_caught": {}}, wrong={**r["wrong"], "math_modes": 0},
                        gpu={**r["gpu"], "compute_cap": "8.0"}), "didn't run"),
])
def test_the_board_rejects_inconsistent_hand_made_cards(change, why):
    r = json.loads(json.dumps(_seed()))
    assert board.validate(_signed(r)) is None
    change(r)
    reason = board.validate(_signed(r))
    assert reason is not None and why in reason, reason


def test_results_json_stays_standard_json(tmp_path):
    r = json.loads(json.dumps(_seed()))
    r["health"]["max_temp_c"] = float("nan")
    rep = board.build([_signed(_seed()), _signed(r)], str(tmp_path / "b"))
    assert rep["accepted"] == 1
    text = (tmp_path / "b" / "results.json").read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))


# --- M5, L1: exit codes ----------------------------------------------------------------------------------

def test_errors_exit_2_never_1(tmp_path):
    assert _main(cli.main, ["replay", str(tmp_path / "missing.pt")])[0] == 2
    (tmp_path / "garbage.pt").write_text("not a torch file")
    assert _main(cli.main, ["replay", str(tmp_path / "garbage.pt")])[0] == 2
    a, b = torch.randn(4, 8).bfloat16(), torch.randn(8, 128).bfloat16()
    c = (a.float() @ b.float()).bfloat16()
    c[0, 0] += 100
    torch.save({"A": a, "B": b, "C": c, "bias": None, "alpha": 1.0, "beta": 1.0, "op": "aten::mm", "where": "x"},
               tmp_path / "ev.pt")
    assert _main(cli.main, ["replay", str(tmp_path / "ev.pt"), "--times", "0"])[0] == 2
    assert _main(cli.main, ["replay", str(tmp_path / "ev.pt"), "--times", "-5"])[0] == 2
    (tmp_path / "rep" / "g").mkdir(parents=True)
    (tmp_path / "rep" / "g" / "summary.json").write_text('{"result":"PASS","gpu":{"gpu_uuid":"G"},"matmuls_checked":"12"}')
    assert _main(cli.main, ["report", str(tmp_path / "rep"), "--out", str(tmp_path / "o")])[0] == 0
    (tmp_path / "nhc").mkdir()
    (tmp_path / "nhc" / "status.json").write_text('{"time":"yesterday","code":0,"line":"OK"}')
    assert _main(cli.main, ["nhc", "--cached", "--out", str(tmp_path / "nhc")])[0] == 2
    (tmp_path / "nhc" / "status.json").write_text("[1, 2]")
    assert _main(cli.main, ["nhc", "--cached", "--out", str(tmp_path / "nhc")])[0] == 2


def test_a_math_mode_replay_on_a_device_without_that_mode_says_so(tmp_path):
    with _wrong_int8():
        full = check.check_one("cpu", minutes=0.01, out_root=str(tmp_path))
    ev = [e for e in full["evidence"] if "math_mode" in e][0]
    with patch.object(modes, "available", return_value="needs compute capability 8.9"):
        code, out = _main(cli.main, ["replay", ev + ".pt", "--times", "2"])
    assert code == 2 and "can't run" in out


# --- L2, L3, L4, L5 --------------------------------------------------------------------------------------

def test_rma_zips_are_unique_and_stay_inside_the_report(tmp_path):
    for host, uuid in (("a", None), ("b", None), ("c", "../../../escaped")):
        d = tmp_path / "in" / host / "g"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({"result": "FAIL", "corrupted_results_training": 1,
                                                    "gpu": {"gpu_uuid": uuid}}))
    rep = fleet.write(fleet.collect([str(tmp_path / "in")]), str(tmp_path / "rep"))
    zips = sorted(os.listdir(tmp_path / "rep" / "rma"))
    assert len(zips) == 3 and len(rep["rma"]) == 3
    assert all(os.path.dirname(os.path.abspath(p)) == str(tmp_path / "rep" / "rma") for p in rep["rma"].values())
    assert not os.path.exists(tmp_path / "escaped.zip")
    with zipfile.ZipFile(tmp_path / "rep" / "rma" / zips[0]) as z:
        assert "README.txt" in z.namelist()


def test_cards_escape_what_the_user_typed(tmp_path):
    with _wrong_int8():
        code, _ = _check(tmp_path, "--provider", "box $\\badcmd$ <script>x</script> [l](javascript:1)")
    d = os.path.join(tmp_path, "cpu")
    md = open(os.path.join(d, "card.md")).read()
    assert "<script>" not in md and "](javascript" not in md
    try:
        import matplotlib  # noqa: F401
        assert os.path.exists(os.path.join(d, "card.png"))
    except ImportError:
        pass


@pytest.mark.parametrize("ref", [{"fingerprints": ["x"]}, {"models": {"CPU": 3}}, [1, 2], {"fingerprints": {"k": 5}}])
def test_a_malformed_reference_file_doesnt_lose_the_result(tmp_path, ref):
    with _wrong_int8():
        full = check.check_one("cpu", minutes=0.01, out_root=str(tmp_path), reference=ref)
    assert full["result"] == "FAIL" and os.path.exists(os.path.join(full["out_dir"], "check.json"))


def test_suspect_outranks_incomplete_everywhere():
    def r(fp):
        return {"fingerprint_key": "k", "fingerprints": [{"output": fp}], "notes": [], "result": "INCOMPLETE"}
    rs = [r("a"), r("a"), r("a"), r("b")]
    check.cross_compare(rs)
    assert rs[3]["result"] == "SUSPECT" and [x["result"] for x in rs[:3]] == ["INCOMPLETE"] * 3
    full = {"fingerprint_key": "k", "fingerprints": [{"output": "b"}], "gpu": {"gpu_name": "G"}, "notes": [],
            "result": "INCOMPLETE"}
    check.apply_board(full, {"fingerprints": {"k": {"a": {"gpus": 40, "sources": 5}}}})
    assert full["result"] == "SUSPECT"


# --- the evidence itself ---------------------------------------------------------------------------------

def test_saved_evidence_holds_the_wrong_output_and_the_cpu_confirms_it(tmp_path):
    """The old stress-evidence test corrupted a copy inside the check, so the saved file held the correct
    output and couldn't tell a mix-up apart. Here the output the GPU 'computed' is wrong in place."""
    from sdcguard import checksum

    def wrong_mm(A, B):
        C = A @ B
        C[1, 5] += 100.0
        return C

    torch.manual_seed(0)
    A, B = torch.randn(64, 256).bfloat16(), torch.randn(256, 384).bfloat16()
    C = wrong_mm(A, B)
    r = checksum.check_matmul(A, B, C)
    assert not r.ok
    path = screen._save_evidence(str(tmp_path), 0, r, (A, B, C, None, 1.0, 1.0, "aten::mm", "stress check", 0),
                                 {}, {})
    info = json.load(open(path + ".json"))
    rc = info["cpu_recheck"]
    assert rc["output_wrong"] and rc["outputs_outside_rounding_envelope"] >= 1
    ev = torch.load(path + ".pt", weights_only=True)
    assert torch.equal(ev["C"], C) and ev["row"] == r.row and ev["col_block"] == r.col_block
    rep = screen.replay(path + ".pt", times=3, device="cpu")
    assert rep["recomputations_that_failed_the_check"] == 0 and rep["recorded_block_on_cpu"]["output_wrong"]


def test_an_alert_from_the_checks_own_arithmetic_is_told_apart(tmp_path):
    """The H200 MPS case at a 20% share: the output was right, the check's own matmul wasn't. The CPU recheck
    and the replay say so instead of blaming the output."""
    from sdcguard import checksum
    torch.manual_seed(1)
    A, B = torch.randn(32, 256).bfloat16(), torch.randn(256, 256).bfloat16()
    C = (A.float() @ B.float()).bfloat16()
    fake = checksum.CheckResult(ok=False, max_z=111.0, nonfinite=False, row=4, col_block=1)
    path = screen._save_evidence(str(tmp_path), 0, fake, (A, B, C, None, 1.0, 1.0, "aten::mm", "x", 0), {}, {})
    rc = json.load(open(path + ".json"))["cpu_recheck"]
    assert rc["judged"] and rc["output_wrong"] is False and "check's own GPU arithmetic" in rc["reading"]
    assert rc["outputs_outside_rounding_envelope"] == 0
    rep = screen.replay(path + ".pt", times=2, device="cpu")
    assert "check's own arithmetic" in rep["reading"]
