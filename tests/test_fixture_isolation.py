"""Local runner regression utilities; no service calls are made."""
import importlib.util
import unittest
from unittest.mock import patch

from _runner_helpers import ROOT


class RunnerDeadlineReview(unittest.TestCase):
    def test_leaked_flag_hold_does_not_poison_the_following_test(self):
        import pytest
        import torch
        from sdcguard import guard

        spec = importlib.util.spec_from_file_location("_loaded_fixture", ROOT / "tests" / "conftest.py")
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        original_state = guard._FLAGS.copy()
        matmul = torch.backends.cuda.matmul
        original_flags = guard._reduced_precision_flags()
        try:
            guard._FLAGS.update(holds=0, changed=None)
            matmul.allow_bf16_reduced_precision_reduction = True
            matmul.allow_fp16_reduced_precision_reduction = True
            with patch.object(torch.cuda, "is_available", lambda: True):
                fixture = fixture_module._cublas_settings_released.__wrapped__()
                next(fixture)
                leaked = guard._exact_reductions_on()
                self.assertEqual(guard._FLAGS["holds"], 1)
                with self.assertRaises(pytest.fail.Exception):
                    next(fixture)
                self.assertEqual(guard._reduced_precision_flags(), (True, True))
                self.assertEqual(guard._FLAGS, {"holds": 0, "changed": None})
                next_hold = guard._exact_reductions_on()
                guard._exact_reductions_off(leaked)
                self.assertEqual(guard._FLAGS["holds"], 1, "a stale finalizer must not release a later hold")
                self.assertEqual(guard._reduced_precision_flags(), (False, False))
                guard._exact_reductions_off(next_hold)
        finally:
            guard._FLAGS.clear()
            guard._FLAGS.update(original_state)
            matmul.allow_bf16_reduced_precision_reduction = original_flags[0]
            matmul.allow_fp16_reduced_precision_reduction = original_flags[1]


if __name__ == "__main__":
    unittest.main()
