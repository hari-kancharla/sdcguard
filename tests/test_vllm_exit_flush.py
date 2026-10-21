"""An alert from a vLLM worker's last steps before it shuts down still reaches the log and the webhook.

Results are read every SDCGUARD_POLL_EVERY steps, so the plugin reads the rest when the worker exits. vLLM starts
its engine core and workers with multiprocessing 'fork' by default (VLLM_WORKER_MULTIPROC_METHOD), and on CPython
3.11 and 3.12 a forked multiprocessing child ends with os._exit() without running atexit; and a webhook post sent
from a daemon thread at exit is cut off when the process ends. Each test runs a stand-in worker in a subprocess."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# A self-contained vLLM worker (no pytest needed, so any interpreter with torch can run it).
_WORKER = r"""
import os, sys, time, types, urllib.request
import torch, torch.nn as nn
os.environ.update({"SDCGUARD": "1", "SDCGUARD_ALLOW_CPU": "1", "SDCGUARD_POLL_EVERY": "100000",
                   "SDCGUARD_POLL_SECONDS": "1e9", "SDCGUARD_LOG": LOG, "VLLM_CACHE_ROOT": TMP})
if WEBHOOK:
    os.environ.update({"SDCGUARD_WEBHOOK": "https://hooks.example.invalid/T0", "SDCGUARD_WEBHOOK_FORMAT": "json"})

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(req, timeout=None):     # a real endpoint: DNS + TLS + request take a few hundred ms
        time.sleep(0.3)
        open(MARKER, "w").write("delivered")
        return Resp()
    urllib.request.urlopen = urlopen

linear = types.ModuleType("vllm.model_executor.layers.linear")
class UnquantizedLinearMethod:
    def process_weights_after_loading(self, layer): pass
    def apply(self, layer, x, bias=None): return torch.nn.functional.linear(x, layer.weight, bias)
linear.UnquantizedLinearMethod = UnquantizedLinearMethod
runner = types.ModuleType("vllm.v1.worker.gpu_model_runner")
class GPUModelRunner:
    def __init__(self, model): self.model = model
    def execute_model(self, x): return self.model(x)
runner.GPUModelRunner = GPUModelRunner
for n in ["vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.v1", "vllm.v1.worker"]:
    sys.modules[n] = types.ModuleType(n)
sys.modules["vllm.model_executor.layers.linear"] = linear
sys.modules["vllm.v1.worker.gpu_model_runner"] = runner

class Lin(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(256, 256).to(torch.bfloat16) * 0.05, requires_grad=False)
        self.bias, self.prefix, self.quant_method = None, f"model.layers.{i}.proj", UnquantizedLinearMethod()
    def forward(self, x): return self.quant_method.apply(self, x, self.bias)

def worker():
    from sdcguard import vllm_plugin
    vllm_plugin.register()
    torch.manual_seed(0)
    model = nn.Sequential(Lin(0), nn.ReLU(), Lin(1))
    for m in model.modules():
        if hasattr(m, "quant_method"): m.quant_method.process_weights_after_loading(m)
    with torch.no_grad():
        model[0].weight.view(torch.int16)[5, 9] ^= (1 << 14)   # a flipped weight bit: an alert to report
    GPUModelRunner(model).execute_model(torch.randn(3, 256).to(torch.bfloat16))
    # then the worker shuts down: vLLM's WorkerProc.worker_main returns (or raises SystemExit) after shutdown()

if FORK:
    import multiprocessing
    p = multiprocessing.get_context("fork").Process(target=worker)   # vLLM's default VLLM_WORKER_MULTIPROC_METHOD
    p.start(); p.join(120)
    sys.exit(p.exitcode)
worker()                                 # plain interpreter exit: vllm_plugin._flush_at_exit runs from atexit
"""


def _run_worker(python, tmp_path, fork, webhook):
    log, marker = str(tmp_path / "alerts.jsonl"), str(tmp_path / "posted")
    head = f"LOG, TMP, MARKER, FORK, WEBHOOK = {log!r}, {str(tmp_path)!r}, {marker!r}, {fork!r}, {webhook!r}\n"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}
    r = subprocess.run([python, "-c", head + _WORKER], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return (open(log).read() if os.path.exists(log) else ""), os.path.exists(marker)


def _fork_python():
    """An interpreter whose forked multiprocessing children end with os._exit() without running atexit: CPython
    3.11 and 3.12 (vLLM's Docker image is 3.12). CPython 3.13.16 runs atexit in fork children (popen_fork calls
    atexit._run_exitfuncs()), so this needs SDCGUARD_FORK_PYTHON=<python3.12 with torch> there."""
    py = os.environ.get("SDCGUARD_FORK_PYTHON") or sys.executable
    probe = ("import atexit, multiprocessing, os, sys\n"
             "def c(): atexit.register(lambda: os.write(1, b'ran'))\n"
             "p = multiprocessing.get_context('fork').Process(target=c); p.start(); p.join()\n")
    out = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=120).stdout
    if "ran" in out:
        pytest.skip(f"{py} runs atexit in forked children; set SDCGUARD_FORK_PYTHON to a CPython 3.11/3.12 with torch")
    return py


def test_an_alert_from_the_last_steps_is_reported_when_a_forked_worker_exits(tmp_path):
    """The module docstring: "a server that goes idle right after a bad step reports it at its next step, or when it
    exits". An atexit handler alone never runs in a forked worker on CPython 3.11/3.12: the exit flush is also a
    multiprocessing exit finalizer, which _bootstrap runs before os._exit()."""
    alerts, _ = _run_worker(_fork_python(), tmp_path, fork=True, webhook=False)
    assert "serve_checksum" in alerts


def test_a_webhook_alert_found_by_the_exit_flush_is_delivered(tmp_path):
    """The exit flush's alert is posted from a daemon thread started during interpreter shutdown; the flush waits
    for it (up to the webhook's timeout), so the post isn't cut off when the process ends."""
    alerts, posted = _run_worker(sys.executable, tmp_path, fork=False, webhook=True)
    assert "serve_checksum" in alerts                           # the exit flush did find the alert
    assert posted, "the webhook post was cut off at exit"
