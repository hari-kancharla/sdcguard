"""Tests use one fixed sign pattern for the second checksum, so a case near the alarm line can't pass on one
run and fail on the next. sdcguard itself draws a new pattern in every process (test_edge_cases checks that).

On a GPU every guard changes two process-wide cuBLAS settings while it's active (guard._exact_reductions_on).
A test that leaves a guard alive would change them for every test after it: on Oct 19 two tests failed on
the L4 and the H100 that way, after earlier tests dropped ServeGuards without detaching them. So after each
test, guards nobody refers to are collected (a collected ServeGuard lets go), and a test that still leaves the
settings held fails, by name."""
import gc
import os

import pytest

os.environ.setdefault("SDCGUARD_SIGN_SEED", "1")


@pytest.fixture(autouse=True)
def _cublas_settings_released():
    from sdcguard import guard
    import torch
    saved = guard._FLAGS.copy()
    before = saved["holds"]
    flags = guard._reduced_precision_flags()
    existing = ({id(item) for item in gc.get_objects() if type(item) is guard._FlagHold and item.live}
                if before else set())
    yield
    if guard._FLAGS["holds"] != before:
        gc.collect()
    left = guard._FLAGS["holds"] - before
    if left:
        # A counter reset alone leaves the backend flags changed. Retire leaked tokens too: a later
        # finalizer for a failed test must not release a hold acquired by the following test.
        with guard._FLAGS_LOCK:
            for item in gc.get_objects():
                if type(item) is guard._FlagHold and id(item) not in existing:
                    item.live = False
            guard._FLAGS.update(saved)
            matmul = torch.backends.cuda.matmul
            matmul.allow_bf16_reduced_precision_reduction = flags[0]
            matmul.allow_fp16_reduced_precision_reduction = flags[1]
        pytest.fail(f"this test left {left} guard(s) holding the cuBLAS settings (detach or exit them)",
                    pytrace=False)
