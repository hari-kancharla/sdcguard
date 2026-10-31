"""Reconcile a declared client batch with recorded transfer attempts, offline.

Owner: Hari Krishna Kancharla.
An exact join establishes recorded coverage of the supplied batch. It does not
establish all server traffic, authentic identities, or request isolation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import stat

SCHEMA = "sdcguard.client-inventory.v1"
MAX_REQUESTS = 10000
MAX_BLOCKS = 65536
MAX_EVENTS = 1000000
MAX_TEXT = 1024
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 64
TRANSFER_FIELDS = frozenset(("remote_request_id", "remote_engine_id", "remote_lease_slot",
                             "remote_lease_nonce", "remote_block_ids"))
SCOPE = "One-to-one recorded-attempt coverage of the declared single-group client batch"
LIMITS = [
    "The supplied client inventory is a completeness declaration, not an authenticated traffic census.",
    "Matching transfer descriptors does not establish lifecycle correctness, ownership, physical reads or answer correctness.",
    "Only unique single-group transfer descriptors are supported; ambiguous associations remain unknown.",
    "Client and engine IDs are not joined by prefixes, timestamps or response text.",
]


def _text(value, name):
    if type(value) is not str or not value or len(value) > MAX_TEXT:
        raise ValueError(name + " must be a nonempty bounded string")
    return value


def _uint(value, name):
    if type(value) is not int or value < 0 or value > (1 << 64) - 1:
        raise ValueError(name + " must be an unsigned 64-bit integer")
    return value


def _transfer(value):
    if type(value) is not dict or set(value) != TRANSFER_FIELDS:
        raise ValueError("transfer must contain exactly the supported descriptor fields")
    _text(value["remote_request_id"], "remote request ID")
    _text(value["remote_engine_id"], "remote engine ID")
    _uint(value["remote_lease_slot"], "lease slot")
    _uint(value["remote_lease_nonce"], "lease nonce")
    groups = value["remote_block_ids"]
    if (type(groups) is not list or len(groups) != 1 or type(groups[0]) is not list
            or not 1 <= len(groups[0]) <= MAX_BLOCKS):
        raise ValueError("exactly one nonempty bounded block group is required")
    for block in groups[0]:
        _uint(block, "block ID")
    if len(set(groups[0])) != len(groups[0]):
        raise ValueError("duplicate source blocks are unsupported")
    return (value["remote_request_id"], value["remote_engine_id"],
            value["remote_lease_slot"], value["remote_lease_nonce"], tuple(groups[0]))


def _inventory(value):
    if (type(value) is not dict or set(value) != {"schema", "workload", "complete", "requests"}
            or value.get("schema") != SCHEMA):
        raise ValueError("unsupported client inventory schema or fields")
    _text(value["workload"], "workload")
    if value["complete"] is not True:
        raise ValueError("client inventory must explicitly declare a complete batch")
    rows = value["requests"]
    if type(rows) is not list or not 1 <= len(rows) <= MAX_REQUESTS:
        raise ValueError("a nonempty bounded client request inventory is required")
    for row in rows:
        if type(row) is not dict or set(row) != {"client_request_id", "transfer"}:
            raise ValueError("invalid client request fields")
        _text(row["client_request_id"], "client request ID")
        _transfer(row["transfer"])
    return deepcopy(value)


def validate_inventory(value):
    """Validate normalized input without converting or discarding any rows."""
    result = _inventory(value)
    rows = result["requests"]
    if len({row["client_request_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate client request IDs are ambiguous")
    if len({_transfer(row["transfer"]) for row in rows}) != len(rows):
        raise ValueError("duplicate client transfer descriptors are ambiguous")
    return result


load_inventory = validate_inventory


def _attempt(event):
    args = event["args"]
    _text(args.get("request"), "recorded request ID")
    if args.get("mapping_complete") is not True or args.get("lease_fields") is not True:
        raise ValueError("recorded single-group mapping and lease identity are not established")
    source = args.get("source_regions")
    if type(source) is not list or not 1 <= len(source) <= MAX_BLOCKS:
        raise ValueError("recorded source regions are missing or unsupported")
    engine = _text(args.get("remote_engine"), "recorded remote engine ID")
    blocks = []
    for region in source:
        if type(region) not in (list, tuple) or len(region) != 2 or region[0] != engine:
            raise ValueError("recorded source regions disagree with the remote engine")
        blocks.append(_uint(region[1], "recorded source block"))
    transfer = {"remote_request_id": args.get("remote_request"), "remote_engine_id": engine,
                "remote_lease_slot": args.get("lease_slot"), "remote_lease_nonce": args.get("lease_nonce"),
                "remote_block_ids": [blocks]}
    return args["request"], _transfer(transfer)


def reconcile(inventory, events):
    """Report unknown for any absent, malformed, extra or ambiguous attempt.

    Lifecycle findings must be judged separately. This function emits no
    isolation violation and never converts a matched descriptor into one.
    """
    result = {"schema": "sdcguard.request-coverage.v1", "status": "unknown", "scope": SCOPE,
              "declared_requests": None, "matched_requests": 0, "observed_attempts": 0,
              "requests": [], "unexpected_attempts": [], "issues": [], "limits": list(LIMITS)}
    try:
        normalized = _inventory(inventory)
    except (ValueError, TypeError, KeyError) as exc:
        result["issues"].append(str(exc))
        return result
    rows = normalized["requests"]
    result.update(workload=normalized["workload"], declared_requests=len(rows))
    if type(events) not in (list, tuple) or len(events) > MAX_EVENTS:
        result["issues"].append("recorded events must be a bounded list")
        return result
    client_ids = Counter(row["client_request_id"] for row in rows)
    client_keys = Counter(_transfer(row["transfer"]) for row in rows)
    by_key, by_id = defaultdict(list), Counter()
    for index, event in enumerate(events):
        if type(event) is not dict:
            result["issues"].append("malformed recorded event at index " + str(index))
            continue
        args = event.get("args")
        if event.get("kind") != "evidence":
            continue
        if type(args) is not dict:
            result["issues"].append("malformed evidence arguments at index " + str(index))
            continue
        if args.get("what") != "attempt":
            continue
        result["observed_attempts"] += 1
        request = args.get("request")
        if type(request) is str:
            by_id[request] += 1
        try:
            request, key = _attempt(event)
        except (ValueError, TypeError, KeyError) as exc:
            result["unexpected_attempts"].append({"event_index": index, "request": request
                if type(request) is str else None, "status": "unknown", "reason": str(exc)})
            continue
        by_key[key].append((index, request))
    used = set()
    for row in rows:
        key = _transfer(row["transfer"])
        matches = by_key.get(key, [])
        reason = None
        if client_ids[row["client_request_id"]] != 1:
            reason = "duplicate client request ID"
        elif client_keys[key] != 1:
            reason = "duplicate client transfer descriptor"
        elif not matches:
            reason = "no recorded attempt with this exact transfer descriptor"
        elif len(matches) != 1:
            reason = "multiple recorded attempts have this transfer descriptor"
        elif by_id[matches[0][1]] != 1:
            reason = "recorded request ID is not unique"
        entry = {"client_request_id": row["client_request_id"], "status": "unknown" if reason else "matched",
                 "engine_request": matches[0][1] if len(matches) == 1 else None,
                 "reason": reason or "unique exact transfer descriptor match"}
        result["requests"].append(entry)
        used.update(index for index, _ in matches)
        if reason is None:
            result["matched_requests"] += 1
    for key, matches in by_key.items():
        for index, request in matches:
            if index not in used:
                result["unexpected_attempts"].append({"event_index": index, "request": request,
                    "status": "unknown", "reason": "recorded attempt is outside the declared client inventory"})
    if any(row["status"] != "matched" for row in result["requests"]):
        result["issues"].append("not every declared client request has one unambiguous recorded attempt")
    if result["unexpected_attempts"]:
        result["issues"].append("recorded attempts include unsupported or undeclared entries")
    result["issues"] = list(dict.fromkeys(result["issues"]))
    if not result["issues"] and result["matched_requests"] == len(rows):
        result["status"] = "complete"
    return result


def from_performance(value, *, workload="saved-owner-batch"):
    """Convert all retained owner-batch client rows, excluding response text.

    This accepts completed ordinary batches only. It starts no live work and
    refuses incomplete input instead of silently removing its rows.
    """
    if type(value) is not dict or value.get("schema") != "sdcguard.owner-pd-arm/1":
        raise ValueError("unsupported saved performance schema")
    references, rounds = value.get("references"), value.get("rounds")
    if type(references) is not list or not references or type(rounds) is not list or not rounds:
        raise ValueError("saved references and rounds are required")
    load = value.get("load")
    if type(load) is not dict or any(type(load.get(key)) is not int or not 1 <= load[key] <= MAX_REQUESTS
                                   for key in ("references", "rounds", "requests_per_round")):
        raise ValueError("explicit bounded saved reference and round counts are required")
    if load["references"] != len(references) or load["rounds"] != len(rounds):
        raise ValueError("saved reference or round count does not match the declared load")
    if load["references"] + load["rounds"] * load["requests_per_round"] > MAX_REQUESTS:
        raise ValueError("saved client batch exceeds the request bound")
    if len(references) > MAX_REQUESTS:
        raise ValueError("saved client batch exceeds the request bound")
    rows = list(references)
    for expected_round, batch in enumerate(rounds):
        if type(batch) is not dict or batch.get("complete") is not True:
            raise ValueError("every saved round must be complete")
        if type(batch.get("round")) is not int or batch["round"] != expected_round:
            raise ValueError("saved round identities must be contiguous from zero")
        planned, requests = batch.get("planned_requests"), batch.get("requests")
        if (type(planned) is not int or planned < 1 or type(requests) is not list
                or len(requests) != planned or planned != load["requests_per_round"]):
            raise ValueError("saved round request count does not match its plan")
        rows.extend(requests)
        if len(rows) > MAX_REQUESTS:
            raise ValueError("saved client batch exceeds the request bound")
    result = {"schema": SCHEMA, "workload": workload, "complete": True, "requests": []}
    for row in rows:
        if type(row) is not dict or row.get("complete") is not True:
            raise ValueError("every saved client request must be complete")
        stages = row.get("stages")
        if type(stages) is not list or len(stages) != 2 or any(type(s) is not dict for s in stages):
            raise ValueError("exactly one prefill and one decode stage are required")
        if [s.get("phase") for s in stages] != ["prefill", "decode"]:
            raise ValueError("saved stages must be prefill then decode")
        if any(s.get("complete") is not True for s in stages):
            raise ValueError("saved request stages must be complete")
        response, body = stages[0].get("response"), stages[1].get("body")
        if type(response) is not dict or type(body) is not dict:
            raise ValueError("saved transfer handoff receipts are missing")
        prefill, decode = response.get("kv_transfer_params"), body.get("kv_transfer_params")
        if type(prefill) is not dict or type(decode) is not dict or prefill != decode:
            raise ValueError("prefill and decode transfer parameters do not match exactly")
        if not TRANSFER_FIELDS.issubset(prefill):
            raise ValueError("saved client transfer identity is incomplete")
        transfer = {key: deepcopy(prefill[key]) for key in TRANSFER_FIELDS}
        _transfer(transfer)
        result["requests"].append({"client_request_id": _text(row.get("decode_id"), "client request ID"),
                                   "transfer": transfer})
    return validate_inventory(result)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite JSON value is unsupported")


def read_json(path):
    """Read one stable bounded regular JSON file without following its link."""
    path = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("stable no-follow regular-file access is unsupported")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("input must be a regular file, not a symlink")
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            raise ValueError("input exceeds the supported regular-file byte bound")
        raw = stream.read(MAX_FILE_BYTES + 1)
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (len(raw) > MAX_FILE_BYTES or signature(before) != signature(after)
            or path.is_symlink() or signature(after) != signature(path.lstat())):
        raise ValueError("input changed during reading")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the supported parser depth") from exc
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if type(item) not in (dict, list):
            continue
        if depth > MAX_JSON_DEPTH:
            raise ValueError("JSON nesting exceeds the supported parser depth")
        children = item.values() if type(item) is dict else item
        pending.extend((child, depth + 1) for child in children)
    return value


def read_inventory(path):
    return validate_inventory(read_json(path))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard inventory", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("import-performance", help="normalize a saved complete client batch offline")
    convert.add_argument("input", type=Path)
    convert.add_argument("--out", type=Path, required=True)
    convert.add_argument("--workload", default="saved-owner-batch")
    args = parser.parse_args(argv)
    try:
        normalized = from_performance(read_json(args.input), workload=args.workload)
        raw = (json.dumps(normalized, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("normalized inventory exceeds the byte bound")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(args.out, flags, 0o600), "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"status": "unknown", "errors": [str(exc)]}))
        return 2
    print(json.dumps({"status": "normalized", "requests": len(normalized["requests"]),
                      "output": str(args.out), "scope": SCOPE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
