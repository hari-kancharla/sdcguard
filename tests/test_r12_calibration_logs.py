"""Local runner regression utilities; no service calls are made."""
import json
import subprocess
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from _r12_runner_helpers import load_runner


class RunnerDeadlineReview(unittest.TestCase):
    def test_attention_keeps_full_timeout_log_and_completed_incomplete_report(self):
        for code in (2, 124):
            with self.subTest(code=code), TemporaryDirectory() as td:
                module = load_runner("modal_check")
                repo, out = Path(td) / "repo", Path(td) / "out"
                for part in ("src", "tests", "scripts", "experiments"):
                    (repo / part).mkdir(parents=True)
                output = "first diagnostic line\n" + "x" * 7000

                def step(cmd, budget, **kwargs):
                    if isinstance(cmd, list) and "experiments/attention_calib.py" in cmd:
                        target = out / "attention_h100"
                        target.mkdir(parents=True, exist_ok=True)
                        (target / "attention_calib.json").write_text('{"complete":false}')
                        return subprocess.CompletedProcess(cmd, code, output, "calibration status")
                    return subprocess.CompletedProcess(cmd, 0, "ok", "")

                with patch.object(module, "R", str(repo)), patch.object(module, "OUT", str(out)), \
                        patch.object(module, "_step", step), \
                        patch.object(module, "vol", types.SimpleNamespace(commit=lambda: None)):
                    result = module._tests("h100")
                log = (out / "attention_output_h100.txt").read_text()
                self.assertIn(output, log)
                self.assertTrue(log.startswith("code fingerprint "))
                if code == 2:
                    self.assertEqual(json.loads(result["attention_json"]), {"complete": False})
                else:
                    self.assertIsNone(result["attention_json"])


if __name__ == "__main__":
    unittest.main()
