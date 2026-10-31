"""Client-observed terminal outcomes, separate from protocol-rule verdicts."""
from __future__ import annotations

import hashlib
import re


def terminal_outcome(finding: dict, trial: dict, references: list[str]) -> dict:
    """An attributed response can show output recovery, not internal recomputation.

    Only the existing case's assigned request id and normalized lease rejection
    are accepted. Missing hook evidence cannot be backfilled from answer text.
    """
    result = {"outcome": "unknown", "basis": "insufficient terminal evidence",
              "not_observed": ["completion of internal local recomputation"]}
    required = ("x_request_id", "x_finished", "hung", "x_exc", "x_error", "x", "decline_attribution", "decline_line")
    if (not isinstance(finding, dict) or not isinstance(trial, dict)
            or any(k not in trial for k in required) or not isinstance(references, list)
            or not isinstance(trial["decline_line"], dict)):
        return result
    rid = finding.get("request")
    client = trial.get("x_request_id")
    if (finding.get("verdict") not in ("correctly_rejected", "recovered", "refused")
            or not isinstance(client, str) or not client
            or not isinstance(rid, str)
            or not (rid == client or rid.startswith(f"cmpl-{client}-"))
            or trial.get("x_finished") is not True or trial.get("hung") is not False
            or trial.get("x_exc") is not None or trial.get("decline_attribution") != "request_id"):
        return result
    decline = trial["decline_line"]
    if (decline.get("cause") != "lease_mismatch" or decline.get("kind") != "post_read"
            or decline.get("req") != rid):
        return result
    error = trial.get("x_error")
    if isinstance(error, str) and re.fullmatch(r"HTTP [45][0-9]{2}|finish_reason=error", error):
        if trial.get("x") is None:
            result.update(outcome="refused", basis="attributed invalid-lease rejection and terminal client error")
        return result
    if finding.get("verdict") == "refused":
        return result
    if (error is not None or len(references) < 3 or not all(isinstance(x, str) and x for x in references)
            or len(set(references)) != 1 or trial.get("x") != references[0]):
        return result
    result.update(outcome="recovered", basis="output exactly matches three agreed clean references after rejection",
                  response_sha256=hashlib.sha256(trial["x"].encode()).hexdigest())
    return result
