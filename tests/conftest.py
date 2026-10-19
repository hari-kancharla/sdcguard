"""Tests use one fixed sign pattern for the second checksum, so a case near the alarm line can't pass on one
run and fail on the next. sdcguard itself draws a new pattern in every process (test_edge_cases checks that)."""
import os

os.environ.setdefault("SDCGUARD_SIGN_SEED", "1")
