"""Explicit checksum source profiles. Owner: Hari Krishna Kancharla.

Offline selection depends only on recorded source identity. Runtime selection
is passed to each observer; no import-time environment selects a verifier.
"""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

BASELINE = "29d2f4e03740acb14439ce22d24b06d8095d17d3"
CANDIDATE = "1a0c6954a9d881c81f85fc1a6cbf51c76f3fcba8"
REVISIONS = (BASELINE, CANDIDATE)


def supported(revision):
    return type(revision) is str and revision in REVISIONS


def get(revision=BASELINE):
    if not supported(revision):
        raise ValueError("unsupported checksum engine revision")
    value = json.loads(files("sdcguard").joinpath("case_data/checksum-builds.json").read_bytes())[revision]
    if value["revision"] != revision:
        raise ValueError("checksum profile identity differs")
    return value


def directory(revision=BASELINE):
    get(revision)
    root = Path(__file__).resolve().parent / "native_operator/profile_data"
    return root if revision == BASELINE else root / "builds" / revision


def startup_supported(value, revision):
    """The new scheduler branch must be explicitly observed as unused."""
    if not supported(revision) or type(value) is not dict:
        return False
    if revision == BASELINE:
        return True
    return ("hisparse_host_num_blocks" in value and value["hisparse_host_num_blocks"] is None
            and value.get("defer_async_loads") is False)
