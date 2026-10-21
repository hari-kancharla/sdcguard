"""`sdcguard nhc` (cli.py) with the multi-GPU check under it: a GPU that computed wrong results drains the node even
when its worker died before it reported, and nhc's deadline stops the per-GPU workers too (exit codes: 0 healthy,
1 drain, 2 unknown; deploy/ depends on them)."""
import contextlib
import io
import json
import os
import signal
import time

from sdcguard import cli

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))


def _main(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(argv)
    return code, buf.getvalue()


def _site(tmp_path, body: str) -> str:
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(body)
    return str(site)


def _alive(pid: int) -> bool:
    """Whether a process is still running (a killed one that nobody has reaped yet, a zombie, isn't)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def test_nhc_drains_when_a_gpu_computed_wrong_results_and_then_its_worker_died(tmp_path, monkeypatch):
    """`sdcguard check` exits 1 when a GPU's worker computed wrong results and then died before its check.json
    ('FAIL: it computed wrong results and then its check died'). nhc read only fresh check.json files, found no
    FAIL among them, saw exit code 1 != 0 and printed UNKNOWN with exit 2: node-problem-detector and the Slurm
    scripts didn't drain the node. The dead worker's GPU now gets a FAIL check.json of its own."""
    site = _site(tmp_path,
                 "import os\n"
                 "if os.environ.get('NHC_TEST_TWO_GPUS') and 'CUDA_VISIBLE_DEVICES' not in os.environ:\n"
                 "    import torch\n"
                 "    torch.cuda.is_available = lambda: True      # two GPUs (the workers run on the CPU)\n"
                 "    torch.cuda.device_count = lambda: 2\n"
                 "if os.environ.get('CUDA_VISIBLE_DEVICES') == '0':\n"
                 "    from sdcguard import modes, gpuinfo\n"
                 "    real = modes.gemm\n"
                 "    def corrupt(mode, a, b):\n"
                 "        c = real(mode, a, b)\n"
                 "        if mode == 'int8':\n"
                 "            c[0, 0] += 1\n"
                 "        return c\n"
                 "    modes.gemm = corrupt\n"
                 "    def die(*a, **k):\n"
                 "        os._exit(9)     # the worker dies (driver crash, OOM killer) before writing check.json\n"
                 "    gpuinfo.datasheet = die\n")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([site, SRC, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("SDCGUARD_CHILD_DEVICE", "cpu")
    monkeypatch.setenv("NHC_TEST_TWO_GPUS", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    out = str(tmp_path / "nhc")
    code, line = _main(cli.main, ["nhc", "--minutes", "0.01", "--out", out])
    log = open(os.path.join(out, "child_0.log")).read() if os.path.exists(os.path.join(out, "child_0.log")) else ""
    assert any(n.startswith("math_mode_evidence_") for _r, _d, fs in os.walk(out) for n in fs), log
    assert code == 1 and line.startswith("FAIL"), line
    dead = json.load(open(os.path.join(out, "cpu0", "check.json")))
    assert dead["result"] == "FAIL" and dead["wrong"]["math_modes"] >= 1 and dead["evidence"]
    assert json.load(open(os.path.join(out, "status.json")))["code"] == 1          # what --cached reports
    assert _main(cli.main, ["nhc", "--cached", "--out", out])[0] == 1


def test_the_nhc_deadline_stops_the_per_gpu_workers_too(tmp_path, monkeypatch):
    """_run_check's deadline was subprocess.run(timeout=...), which kills only `python -m sdcguard.check`; the
    per-GPU workers it started with Popen (check._children) kept running and kept their GPUs (a hung one forever).
    A sitecustomize stands in for the check process: it starts one worker the way _children does (output to a
    log file) and then hangs."""
    pidfile = tmp_path / "worker.pid"
    site = _site(tmp_path,
                 "import os, subprocess, sys, time\n"
                 "pf = os.environ.pop('NHC_TEST_HANG_PIDFILE', None)\n"
                 "if pf:\n"
                 "    log = open(pf + '.log', 'w')\n"
                 "    w = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], stdout=log,\n"
                 "                         stderr=subprocess.STDOUT)\n"
                 "    open(pf, 'w').write(str(w.pid))\n"
                 "    time.sleep(120)\n")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([site, SRC, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("NHC_TEST_HANG_PIDFILE", str(pidfile))
    monkeypatch.setattr(cli, "_deadline", lambda minutes: 4.0)
    pid = None
    try:
        code, line = _main(cli.main, ["nhc", "--minutes", "2", "--out", str(tmp_path / "nhc")])
        assert code == 2 and "didn't finish" in line, line
        pid = int(pidfile.read_text())
        time.sleep(0.5)
        assert not _alive(pid), "the GPU worker outlived the deadline"
    finally:
        if pid is None and pidfile.exists():
            pid = int(pidfile.read_text())
        if pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


def test_a_gpu_worker_always_stops_before_the_deadline_nhc_gives_the_check():
    """The check's own limit for a hung worker (check._children) has to come before nhc's and certify's deadline
    for the whole check, or the parent never gets to stop the worker and report the other GPUs. It came after it
    for nhc's default 2 minutes (900 s against 780 s) and certify's default 10."""
    from sdcguard import check
    for minutes in (0.5, 2, 5, 10, 20, 60):
        assert check.worker_limit(minutes) < cli._deadline(minutes)
