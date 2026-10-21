"""The fleet report (fleet.py, `sdcguard report`): which result is a GPU's row, what goes on quarantine.txt and in
the RMA zips, and what the command's exit code says. A run that died keeps what it found, runs are ordered by when
they ran (not by file times), a demo never takes a failed GPU out of quarantine, one bad folder never stops the
report, and the zip never takes a file from outside the results folder."""
import contextlib
import io
import json
import os
import time
import warnings
import zipfile
from unittest.mock import patch

import pytest

from sdcguard import check, cli, fleet, gpuinfo, modes


def _main(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(argv)
    return code, buf.getvalue()


def _check(out, *extra):
    return _main(check.main, ["--device", "cpu", "--minutes", "0.01", "--out", str(out), "--offline", *extra])


def _age(path, seconds_ago):
    t = time.time() - seconds_ago
    os.utime(path, (t, t))


def _iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _put_check(root, uuid, result, recorded, mtime):
    d = root / uuid
    d.mkdir(parents=True)
    (d / "check.json").write_text(json.dumps({"run_id": root.name, "result": result, "time": _iso(recorded),
                                              "gpu": {"gpu_uuid": uuid},
                                              "wrong": {"stress": 1 if result == "FAIL" else 0}}))
    os.utime(d / "check.json", (mtime, mtime))


# --- a run that died -----------------------------------------------------------------------------------------

def test_report_keeps_a_fail_from_a_run_that_died_before_writing_check_json(tmp_path):
    """sdcguard_check/<GPU UUID>/ is reused by every run. A run whose workload found corrupted results and wrote
    summary.json, and was then killed (a deadline, the OOM killer, a hung nvidia-smi) before its check.json, left
    the previous run's check.json (PASS) beside its own summary.json (FAIL). collect() dropped any summary whose
    run_id differed from check.json's, whichever was newer: the GPU came out PASS and stayed off quarantine.txt."""
    code, _ = _check(tmp_path)
    d = tmp_path / "cpu"
    assert code == 0 and json.load(open(d / "check.json"))["result"] == "PASS"
    for n in os.listdir(d):
        _age(d / n, 86400)                                  # yesterday's run
    s = json.load(open(d / "summary.json"))
    s.update(run_id="todays-run", result="FAIL", corrupted_results_training=2)
    (d / "summary.json").write_text(json.dumps(s))        # today's workload, then the process was killed
    rows = fleet.collect([str(tmp_path)])
    assert rows[0]["result"] == "FAIL", (rows[0]["result"], rows[0]["notes"])
    assert any("no check.json of its own" in n for n in rows[0]["notes"])


class _Killed(BaseException):
    """The process dying (SIGKILL, the OOM killer) between two lines of check_one."""


def test_a_check_killed_after_finding_wrong_results_leaves_a_fail_not_the_last_pass(tmp_path):
    """While a check runs, its GPU's check.json says INCOMPLETE under the new run's id (the previous run's PASS and
    certificate are gone), and the failures it saved before it was killed make the fleet report's row FAIL."""
    assert _check(tmp_path)[0] == 0                        # last week's PASS
    real = modes.gemm

    def wrong_int8(mode, a, b):
        c = real(mode, a, b)
        if mode == "int8":
            c[0, 0] += 1
        return c

    with patch.object(modes, "gemm", side_effect=wrong_int8), \
            patch.object(gpuinfo, "datasheet", side_effect=_Killed), pytest.raises(_Killed):
        check.check_one("cpu", minutes=0.01, out_root=str(tmp_path))
    full = json.load(open(tmp_path / "cpu" / "check.json"))
    assert full["result"] == "INCOMPLETE" and full["running"]
    rows = fleet.collect([str(tmp_path)])
    assert rows[0]["result"] == "FAIL" and rows[0]["wrong_by_phase"]["math_modes"] >= 1, rows[0]["notes"]
    assert any(os.path.basename(e).startswith("math_mode_evidence_") for e in rows[0]["evidence"])


# --- which result is the row ---------------------------------------------------------------------------------

def test_a_demo_run_doesnt_take_a_failed_gpu_out_of_quarantine(tmp_path):
    """The newest result was the row, and a simulated row never goes on quarantine.txt; the history rule ('a
    simulated-fault demo never keeps a real GPU in quarantine') only ran for a PASS or INCOMPLETE row. So a demo
    (`--simulate-faults`, a FAIL) run after a real FAIL hid the real FAIL: the GPU dropped off quarantine.txt."""
    gpu = {"gpu_uuid": "GPU-real-fail", "host": "n1", "pci_bus_id": "0000:01:00.0"}
    real = tmp_path / "mon" / "GPU-real-fail"
    real.mkdir(parents=True)
    (real / "summary.json").write_text(json.dumps({"gpu": gpu, "result": "FAIL", "corrupted_results_training": 3,
                                                   "matmuls_checked": 100}))
    _age(real / "summary.json", 86400)
    demo = tmp_path / "tue" / "GPU-real-fail"
    demo.mkdir(parents=True)
    (demo / "summary.json").write_text(json.dumps({"gpu": gpu, "result": "FAIL", "corrupted_results_training": 9,
                                                   "matmuls_checked": 100,
                                                   "SIMULATED_FAULTS": "yes: 0.3 of matmul outputs corrupted"}))
    rows = fleet.collect([str(tmp_path)])
    assert rows[0]["result"] == "FAIL" and not rows[0]["simulated"] and len(rows[0]["history"]) == 2
    fleet.write(rows, str(tmp_path / "rep"))
    assert "GPU-real-fail" in (tmp_path / "rep" / "quarantine.txt").read_text()


def test_a_copied_older_pass_doesnt_clear_a_newer_fail(tmp_path):
    """'Newest' and the window used the file's mtime, not the time check.json records. An older PASS copied into the
    report's folders later (cp -r, scp and object-store downloads all set mtime to now) became the newest result,
    and the real, later FAIL fell out of the window: the GPU was reported PASS."""
    now = time.time()
    _put_check(tmp_path / "early", "GPU-c", "PASS", now - 49 * 86400, now)             # copied today
    _put_check(tmp_path / "later", "GPU-c", "FAIL", now - 40 * 86400, now - 40 * 86400)  # its last run
    rows = fleet.collect([str(tmp_path / "early"), str(tmp_path / "later")])
    assert rows[0]["result"] == "FAIL", rows[0]["history"]


def test_a_host_clock_running_ahead_doesnt_clear_a_recent_fail(tmp_path):
    """The window was measured from the newest row's mtime. A PASS written on a host whose clock runs 40 days ahead
    pushed a FAIL from 3 days ago out of the 30-day window. A time later than now now counts as now, and of two
    results recorded at the same time the worse one is the row (equal mtimes used to go to whichever was read
    first)."""
    now = time.time()
    _put_check(tmp_path / "a", "GPU-d", "FAIL", now - 3 * 86400, now - 3 * 86400)
    _put_check(tmp_path / "b", "GPU-d", "PASS", now + 40 * 86400, now + 40 * 86400)
    row = fleet.collect([str(tmp_path / "a"), str(tmp_path / "b")])[0]
    assert row["result"] == "SUSPECT" and "doesn't clear" in row["notes"][0], row["history"]
    _put_check(tmp_path / "c", "GPU-e", "PASS", now - 3600, now - 3600)
    _put_check(tmp_path / "e", "GPU-e", "FAIL", now - 3600, now - 3600)
    assert fleet.collect([str(tmp_path / "c"), str(tmp_path / "e")])[0]["result"] == "FAIL"


def test_one_hand_edited_result_file_doesnt_stop_the_whole_fleet_report(tmp_path):
    """collect() guarded counts against hand-edited files (_num) but nothing else: one check.json anywhere under the
    given folders with a malformed notes entry (or a list for wrong, a text speed.tflops) raised out of collect or
    write, and `sdcguard report` wrote nothing, no quarantine.txt for any GPU of the fleet (the threat model in
    collect's docstring is anyone who can write to a results folder)."""
    good = tmp_path / "res" / "node1" / "GPU-a"
    good.mkdir(parents=True)
    (good / "summary.json").write_text(json.dumps({"gpu": {"gpu_uuid": "GPU-a"}, "result": "FAIL",
                                                   "corrupted_results_training": 2, "matmuls_checked": 10}))
    for node, uuid, extra in (("node2", "GPU-b", {"notes": [["bad"]]}),
                              ("node3", "GPU-c", {"speed": {"tflops": "fast"}}),
                              ("node4", "GPU-d", {"wrong": ["stress", 1]})):
        d = tmp_path / "res" / node / uuid
        d.mkdir(parents=True)
        (d / "check.json").write_text(json.dumps({"run_id": "r", "result": "PASS", "gpu": {"gpu_uuid": uuid},
                                                  **extra}))
    code, out = _main(cli.main, ["report", str(tmp_path / "res"), "--out", str(tmp_path / "rep")])
    q = tmp_path / "rep" / "quarantine.txt"
    assert code == 1 and q.exists() and "GPU-a" in q.read_text(), out
    rows = {r["gpu_uuid"]: r for r in json.load(open(tmp_path / "rep" / "fleet_report.json"))["gpus"]}
    assert rows["GPU-b"]["result"] == "PASS" and rows["GPU-c"]["matmul_tflops"] is None
    assert rows["GPU-d"]["result"] == "INCOMPLETE" and "couldn't be read" in rows["GPU-d"]["notes"][0]


# --- the RMA zip ---------------------------------------------------------------------------------------------

def test_the_rma_zip_doesnt_follow_symlinks_out_of_the_results_folder(tmp_path):
    """Anyone who can write to a results folder could make the RMA zip of whoever runs `sdcguard report` include
    any file that person can read: _rma_zip followed a symlinked evidence file (or result file) with z.write. The
    zip is what gets sent to NVIDIA."""
    secret = tmp_path / "home" / "id_ed25519"
    secret.parent.mkdir()
    secret.write_text("-----BEGIN OPENSSH PRIVATE KEY----- secret")
    d = tmp_path / "res" / "GPU-s"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"gpu": {"gpu_uuid": "GPU-s"}, "result": "FAIL",
                                                "corrupted_results_training": 1, "evidence": ["evidence_0"]}))
    (d / "evidence_0.json").write_text("{}")
    os.symlink(secret, d / "evidence_0.pt")
    os.symlink(secret, d / "result.json")
    rep = fleet.write(fleet.collect([str(tmp_path / "res")]), str(tmp_path / "rep"))
    with zipfile.ZipFile(list(rep["rma"].values())[0]) as z:
        names = z.namelist()
        contents = b"".join(z.read(n) for n in names)
    assert b"PRIVATE KEY" not in contents and "evidence_0.json" in names and "summary.json" in names, names


def test_rma_zip_names_and_replay_commands_match_for_repeated_failures(tmp_path):
    """Earlier failures' evidence was renamed <grandparent>_<name> on a clash. Runs kept as
    <date>/sdcguard_check/<UUID> share the grandparent, so a third evidence_0 got the same new name: two zip
    entries with one name (extraction keeps one). The README listed `sdcguard replay evidence_0.pt` for every one
    of them, never the renamed files."""
    for i, (day, res) in enumerate((("d1", "FAIL"), ("d2", "FAIL"), ("d3", "FAIL"), ("d4", "PASS"))):
        d = tmp_path / day / "sdcguard_check" / "GPU-r"
        d.mkdir(parents=True)
        ev = []
        if res == "FAIL":
            ev = [str(d / "evidence_0")]
            (d / "evidence_0.json").write_text(json.dumps({"day": day}))
            (d / "evidence_0.pt").write_text(day)
        (d / "summary.json").write_text(json.dumps({"gpu": {"gpu_uuid": "GPU-r"}, "result": res,
                                                    "corrupted_results_training": int(res == "FAIL"),
                                                    "matmuls_checked": 10, "evidence": ev}))
        _age(d / "summary.json", (4 - i) * 3600)
    rows = fleet.collect([str(tmp_path)])
    assert rows[0]["result"] == "SUSPECT" and len(rows[0]["evidence"]) == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")               # zipfile warns "Duplicate name"
        rep = fleet.write(rows, str(tmp_path / "rep"))
    with zipfile.ZipFile(list(rep["rma"].values())[0]) as z:
        names = z.namelist()
        readme = z.read("README.txt").decode()
        days = sorted(z.read(n).decode() for n in names if n.endswith(".pt"))
    assert len(names) == len(set(names)) and days == ["d1", "d2", "d3"], names
    for n in names:
        if n.endswith(".pt"):
            assert f"sdcguard replay {n}" in readme, (n, readme)


# --- sdcguard report -----------------------------------------------------------------------------------------

def test_report_exits_1_only_for_gpus_to_pull(tmp_path):
    """Exit 1 means a GPU is to be pulled. A report whose only failure was a simulated-fault demo exited 1 and
    printed 'pull: quarantine.txt' while quarantine.txt was empty (nhc exits 2 for demos)."""
    d = tmp_path / "res" / "GPU-demo"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"gpu": {"gpu_uuid": "GPU-demo"}, "result": "FAIL",
                                                "corrupted_results_training": 4, "matmuls_checked": 10,
                                                "SIMULATED_FAULTS": "yes"}))
    code, out = _main(cli.main, ["report", str(tmp_path / "res"), "--out", str(tmp_path / "rep")])
    assert (tmp_path / "rep" / "quarantine.txt").read_text() == ""
    assert code == 2 and "pull:" not in out and "simulated" in out, out


def test_report_says_when_a_folder_it_was_given_doesnt_exist(tmp_path):
    """A mistyped or unmounted results folder was skipped without a word: the report covered fewer GPUs than asked
    and exited 0. A window that isn't a number of days was taken as one."""
    d = tmp_path / "res" / "GPU-ok"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"gpu": {"gpu_uuid": "GPU-ok"}, "result": "PASS",
                                                "matmuls_checked": 10}))
    code, out = _main(cli.main, ["report", str(tmp_path / "res"), str(tmp_path / "node17_unmounted"),
                                 "--out", str(tmp_path / "rep")])
    assert code == 2 and "node17_unmounted" in out and "1 PASS" in out, out
    for window in ("nan", "-1"):
        code, out = _main(cli.main, ["report", str(tmp_path / "res"), "--window-days", window,
                                     "--out", str(tmp_path / "rep2")])
        assert code == 2 and "--window-days" in out, out
