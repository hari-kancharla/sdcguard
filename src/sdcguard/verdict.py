"""What a run's counts add up to. A PASS needs evidence: checks that ran, could be judged, and were right.

    PASS        no wrong result, and the checks that ran could judge their results
    FAIL        at least one result was wrong
    SUSPECT     no checksum failed, but outputs that should be identical weren't (sdcguard check)
    INCOMPLETE  no wrong result was found, but part of the check couldn't run or couldn't judge its
                results (non-finite values in the workload, a math mode that errored): no verdict either way

A check can't judge a result whose inputs aren't finite, or whose values legitimately overflow. In
sdcguard's own workloads (random data, a freshly initialized model) neither should happen on a healthy
GPU, so more than a sliver of such checks makes the run INCOMPLETE rather than a PASS.
"""
from __future__ import annotations

RESULTS = ("PASS", "FAIL", "SUSPECT", "INCOMPLETE")
UNJUDGED_SHARE = 0.01      # at or above this share of checks that couldn't be judged, a clean run isn't a PASS


def from_counts(wrong: int, checked: int, unjudged: int = 0) -> str:
    """FAIL, INCOMPLETE or PASS from one set of check counts."""
    if wrong:
        return "FAIL"
    if checked <= 0 or unjudged >= max(1, UNJUDGED_SHARE * checked):
        return "INCOMPLETE"
    return "PASS"


def unjudged_note(unjudged: int, checked: int, where: str) -> list:
    """A card note for checks that couldn't be judged (empty when there were none)."""
    if not unjudged:
        return []
    level = "bad" if unjudged >= max(1, UNJUDGED_SHARE * checked) else "warn"
    return [[level, f"{unjudged:,} of {checked:,} {where} checks couldn't be judged (values that weren't finite); "
                    "a healthy GPU running this workload shouldn't produce any"]]
