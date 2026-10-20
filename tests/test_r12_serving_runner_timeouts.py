"""Local runner regression utilities; no service calls are made."""
import importlib.util
import json
import sys
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from _r12_runner_helpers import ROOT, Clock, load_runner


class RunnerDeadlineReview(unittest.TestCase):
    def test_each_model_has_a_finite_budget_and_a_timeout_preserves_status(self):
        module = load_runner("modal_vllm")
        clock, budgets = Clock(), []
        with TemporaryDirectory() as td:
            def tee(cmd, log_path, **kwargs):
                if "entry_points" in str(log_path):
                    return 0
                allowance = kwargs.get("timeout")
                budgets.append(allowance)
                clock.sleep(allowance if allowance is not None else 4500)
                Path(log_path).write_text("partial model output\n")
                return 124

            with patch.object(module, "RUNS", td), patch.object(module, "time", clock.api()), \
                    patch.object(module, "_locked", lambda *args: None), \
                    patch.object(module, "_tee", tee), \
                    patch.object(module, "vol", types.SimpleNamespace(commit=lambda: None)):
                with self.assertRaises(RuntimeError):
                    module.vllm()
            self.assertEqual(len(budgets), 2)
            self.assertTrue(all(value is not None and value > 0 for value in budgets), budgets)
            self.assertLess(clock.now, 75 * 60)
            state = json.loads((Path(td) / "vllm" / "STATUS.json").read_text())
            self.assertFalse(state["passed"])
            self.assertEqual(state["exit_codes"]["qwen1.5b"], 124)
            self.assertEqual(state["exit_codes"]["qwen7b"], 124)

    def test_timed_streaming_step_keeps_partial_lines_and_stops_its_group(self):
        module = load_runner("modal_vllm")
        with TemporaryDirectory() as td:
            pidfile, log = Path(td) / "child.pid", Path(td) / "step.txt"
            heartbeat = Path(td) / "heartbeat.txt"
            child = (f"import time\nwhile True:\n open({str(heartbeat)!r}, 'a').write('x')\n time.sleep(0.02)")
            code = ("import subprocess, sys, time; "
                    f"p=subprocess.Popen([sys.executable, '-c', {child!r}]); "
                    f"open({str(pidfile)!r}, 'w').write(str(p.pid)); "
                    "print('partial without newline', end='', flush=True); time.sleep(60)")
            start = time.monotonic()
            result = module._tee([sys.executable, "-c", code], str(log), cwd=td, timeout=0.5)
            self.assertEqual(result, 124)
            self.assertLess(time.monotonic() - start, 5)
            self.assertIn("partial without newline", log.read_text())
            self.assertGreater(int(pidfile.read_text()), 0)
            self.assertTrue(heartbeat.read_text(), "the child must actually run before timeout")
            before = heartbeat.read_bytes()
            time.sleep(0.1)
            self.assertEqual(heartbeat.read_bytes(), before, "the timed-out step's child kept running")

    def test_role_timeout_terminates_its_children_before_fallback(self):
        spec = importlib.util.spec_from_file_location("_r12_role", ROOT / "experiments" / "vllm_check.py")
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        with TemporaryDirectory() as td:
            heartbeat = Path(td) / "role-heartbeat.txt"
            child = f"import time\nwhile True:\n open({str(heartbeat)!r}, 'a').write('x')\n time.sleep(0.02)"
            code = ("import subprocess, sys, time; "
                    f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)")
            started = time.monotonic()
            result = experiment._run_role_process([sys.executable, "-c", code], env=None, timeout=0.5)
            self.assertEqual(result.returncode, 124)
            self.assertLess(time.monotonic() - started, 5)
            self.assertTrue(heartbeat.read_text())
            before = heartbeat.read_bytes()
            time.sleep(0.1)
            self.assertEqual(before, heartbeat.read_bytes())
            skipped = experiment._run_role_process(["must-not-execute"], env=None, timeout=0)
            self.assertEqual(skipped.returncode, 125)


if __name__ == "__main__":
    unittest.main()
