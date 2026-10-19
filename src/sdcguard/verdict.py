"""What a run's counts add up to. A PASS needs evidence: checks that ran, could be judged, and were right.

    PASS        no wrong result, and every check that ran could judge its result
    FAIL        at least one result was wrong
    SUSPECT     no checksum failed, but outputs that should be identical weren't (sdcguard check)
    INCOMPLETE  no wrong result was found, but part of the check couldn't run or couldn't judge its
                results (non-finite values in the workload, a math mode that errored): no verdict either way

A check can't judge a result whose inputs aren't finite, or whose values legitimately overflow. sdcguard's own
workloads (seeded random data, a freshly initialized model) shouldn't produce either on a healthy GPU, so even
one such check makes the run INCOMPLETE. (Until Oct 19, 2026 up to 1% were let through with a warning; an
outside review pointed out that the README promised more than that.)
"""
from __future__ import annotations

RESULTS = ("PASS", "FAIL", "SUSPECT", "INCOMPLETE")
EXIT_CODES = {"PASS": 0, "FAIL": 1, "SUSPECT": 1, "INCOMPLETE": 2}


def from_counts(wrong: int, checked: int, unjudged: int = 0) -> str:
    """FAIL, INCOMPLETE or PASS from one set of check counts."""
    if wrong:
        return "FAIL"
    if checked <= 0 or unjudged > 0:
        return "INCOMPLETE"
    return "PASS"


def unjudged_note(unjudged: int, checked: int, where: str) -> list:
    """A card note for checks that couldn't be judged (empty when there were none)."""
    if not unjudged:
        return []
    return [["bad", f"{unjudged:,} of {checked:,} {where} checks couldn't be judged (values that weren't finite); "
                    "a healthy GPU running this workload shouldn't produce any"]]


def replay_exit(result: dict) -> int:
    """Exit code of a replay: 1 if a recomputation came out wrong, 2 if none could be judged, else 0."""
    if result.get("recomputations_that_failed_the_check"):
        return 1
    n = result.get("recomputations") or 0
    if n and result.get("recomputations_that_couldnt_be_judged", 0) >= n:
        return 2
    return 0
