"""Local runner regression utilities; no service calls are made."""
import subprocess
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from _r12_runner_helpers import Clock, load_runner


class RunnerDeadlineReview(unittest.TestCase):
    def test_tests_budget_does_not_restart_after_time_consumed_by_setup(self):
        module = load_runner("modal_check")
        clock, launches = Clock(), []
        with TemporaryDirectory() as td:
            repo, out = Path(td) / "repo", Path(td) / "out"
            for part in ("src", "tests", "scripts", "experiments"):
                (repo / part).mkdir(parents=True)
            commits = [0]

            def commit():
                # A delayed initial volume commit is outside every subprocess's time limit.
                clock.sleep(850 if commits[0] == 0 else 5)
                commits[0] += 1

            def step(cmd, budget, floor=30.0, **kwargs):
                allowance = max(floor, budget)
                launches.append((clock.now, allowance, cmd))
                clock.sleep(allowance)
                return subprocess.CompletedProcess(cmd, 124, "partial output", "step timed out")

            with patch.object(module, "R", str(repo)), patch.object(module, "OUT", str(out)), \
                    patch.object(module, "time", clock.api()), \
                    patch.object(module, "vol", types.SimpleNamespace(commit=commit)), \
                    patch.object(module, "_step", step):
                result = module._tests("h100")
            self.assertTrue(launches)
            self.assertTrue(all(start + allowance <= module.TESTS_LIMIT["h100"] - 120
                                for start, allowance, _ in launches), launches)
            self.assertLess(clock.now, module.TESTS_LIMIT["h100"])
            self.assertNotEqual(result["wheel"], 0)
            self.assertNotEqual(result["check"], 0)
            self.assertIn("partial output", (out / "pytest_h100.txt").read_text())


if __name__ == "__main__":
    unittest.main()
