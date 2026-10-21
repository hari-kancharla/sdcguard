"""Local runner regression utilities; no service calls are made."""
import unittest
from unittest.mock import patch

from _runner_helpers import Clock, load_runner


class RunnerDeadlineReview(unittest.TestCase):
    def test_connection_retries_do_not_extend_the_overall_wait_cap(self):
        module = load_runner("modal_check")
        clock = Clock()

        class Disconnected:
            def get(self, timeout=0):
                raise ConnectionError("transport unavailable")

        with patch.object(module, "time", clock.api()):
            result = list(module._wait_all([("h100", Disconnected())], cap=40, every=10,
                                           retry_for=600, say=lambda *args, **kwargs: None))
        self.assertEqual(result[0][1]["run"], 2)
        self.assertLessEqual(clock.now, 40, "transport retries must obey the overall wait limit")


if __name__ == "__main__":
    unittest.main()
