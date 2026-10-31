"""Check bounded eager attention metadata without claiming device reads.

Owner: Hari Krishna Kancharla.
The input is a recording, not authenticated runtime evidence. Physical reads,
ownership, content correctness and GPU completion are deliberately not inferred.
"""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import re


SCHEMA = 1
RESULT_SCHEMA = "sdcguard.forward-inputs.v1"
MAX_RECORDS = 8192
MAX_REQUESTS = 256
MAX_LAYERS = 128
MAX_TOKENS = 32768
MAX_CONTEXT = 1048576
MAX_BLOCKS_PER_ROW = 65536
MAX_SNAPSHOT_CELLS = 262144
MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_RECORD_NODES = 1000000
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_TOTAL_NODES = 2000000
MAX_TOTAL_CELLS = 1000000
MAX_TEXT = 512
MAX_REASONS = 64
SUPPORTED_PROFILE = {
    "runner": "V2", "eager": True, "backend": "FLASH_ATTN",
    "model": "Qwen/Qwen3-0.6B", "architecture": "Qwen3ForCausalLM",
    "attention": "dense_causal_full", "cache_groups": 1,
    "prefix_caching": False, "speculative": False,
    "tp": 1, "pp": 1, "dcp": 1, "pcp": 1, "dp": 1,
    "lora": False, "microbatching": False, "kv_sharing": False,
    "kv_dtype": "auto", "layout": "LBHNC",
}
_COMMON = {"schema", "session", "seq", "kind"}
_HEADER = _COMMON | {"process_incarnation", "engine", "profile"}
_SNAPSHOT = _COMMON | {
    "forward_id", "snapshot_id", "pool_incarnation", "requests", "num_tokens",
    "snapshot_copy_complete", "forward_device_complete", "observation_point", "layers",
}
_LAYER = {
    "name", "block_size", "pool_num_blocks", "block_table", "seq_lens",
    "query_start_loc", "write_slots", "causal", "window", "layout", "backend",
    "backend_version",
}
_GAP = _COMMON | {"forward_id", "reason"}
_FOOTER = _COMMON | {"snapshots", "dropped", "errors", "healthy"}
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def source_pins() -> dict:
    """Return a fresh relevant-source hash map, without importing an engine."""
    value = json.loads(files("sdcguard").joinpath("case_data/forward-pins.json").read_text())
    return {entry["path"]: entry["sha256"] for entry in value["files"]}


def _base() -> dict:
    return {"schema": RESULT_SCHEMA, "scope_verdict": "inconclusive",
            "observation_status": "unknown", "evidence_depth": "approximate",
            "evidence_basis": "unavailable", "physical_read_verdict": "unknown",
            "ownership_verdict": "unknown", "forward_device_completion": "unknown",
            "authenticity": "not established", "reasons": []}


def _integer(value, low=0, high=2**63 - 1):
    return type(value) is int and low <= value <= high


def _text(value, maximum=MAX_TEXT):
    return (type(value) is str and 0 < len(value) <= maximum
            and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value))


def _uuid(value):
    return type(value) is str and _HEX32.fullmatch(value) is not None


def _shape(value, keys):
    return type(value) is dict and value.keys() == keys


def _bounded(value):
    """Check only concrete JSON values and bound work before serialization."""
    stack = [(value, 0)]
    nodes = 0
    text_bytes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_RECORD_NODES or depth > 32:
            raise ValueError("record structure exceeds supported bounds")
        typ = type(item)
        if typ is str:
            if len(item) > MAX_RECORD_BYTES:
                raise ValueError("record text exceeds supported bounds")
            try:
                text_bytes += len(item.encode("utf-8"))
            except UnicodeError:
                raise ValueError("record contains invalid Unicode") from None
            if text_bytes > MAX_RECORD_BYTES:
                raise ValueError("record text exceeds supported bounds")
        elif typ is dict:
            if len(item) > MAX_RECORD_NODES or len(stack) + 2 * len(item) > MAX_RECORD_NODES:
                raise ValueError("record structure exceeds supported bounds")
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("record keys must be strings")
                stack.append((key, depth + 1))
                stack.append((child, depth + 1))
        elif typ is list:
            if len(item) > MAX_RECORD_NODES or len(stack) + len(item) > MAX_RECORD_NODES:
                raise ValueError("record structure exceeds supported bounds")
            stack.extend((child, depth + 1) for child in item)
        elif typ is int:
            if item.bit_length() > 63:
                raise ValueError("record integer exceeds supported bounds")
        elif item is not None and typ is not bool:
            raise ValueError("record contains unsupported values")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("record exceeds byte limit")
    return raw, nodes


def _envelope(record, kind=None):
    return (type(record) is dict and type(record.get("schema")) is int
            and record["schema"] == SCHEMA and _uuid(record.get("session"))
            and _integer(record.get("seq"), high=MAX_RECORDS - 1)
            and _text(record.get("kind"), 32)
            and (kind is None or record.get("kind") == kind))


def validate_header(header) -> dict:
    result = _base()
    try:
        _bounded(header)
        if not _shape(header, _HEADER) or not _envelope(header, "header") or header["seq"] != 0:
            raise ValueError("missing or malformed session header")
        if header["process_incarnation"] != header["session"]:
            raise ValueError("process incarnation is not bound to this session")
        engine = header["engine"]
        if not _shape(engine, {"name", "version", "source_sha256", "dependencies"}):
            raise ValueError("engine identity fields are incomplete or unsupported")
        if engine["name"] != "vllm" or engine["version"] != "0.32.0":
            raise ValueError("engine version is unsupported")
        if type(engine["source_sha256"]) is not dict or engine["source_sha256"] != source_pins():
            raise ValueError("relevant engine source does not match the supported pins")
        dependencies = engine["dependencies"]
        if not _shape(dependencies, {"python", "torch", "flash_attn"}):
            raise ValueError("backend dependency identities are incomplete")
        if not all(_text(value, 256) for value in dependencies.values()):
            raise ValueError("backend dependency identities are invalid")
        if dependencies["flash_attn"] != "3":
            raise ValueError("FlashAttention selector is unsupported")
        profile = header["profile"]
        old_fields = set(SUPPORTED_PROFILE) | {"model_revision", "layer_names"}
        local_reference = type(profile) is dict and "model_reference" in profile
        if not _shape(profile, old_fields | ({"model_reference"} if local_reference else set())):
            raise ValueError("effective profile fields are incomplete or unsupported")
        if local_reference:
            from .. import model_reference
            model_reference.validate_observation(profile["model"], profile["model_revision"], profile["model_reference"])
        for name, expected in SUPPORTED_PROFILE.items():
            if local_reference and name == "model":
                continue
            if type(profile[name]) is not type(expected) or profile[name] != expected:
                raise ValueError("effective profile is unsupported: " + name)
        revision = profile["model_revision"]
        if type(revision) is not str or _HEX40.fullmatch(revision) is None:
            raise ValueError("model revision is not a full immutable identifier")
        layers = profile["layer_names"]
        if (type(layers) is not list or not 1 <= len(layers) <= MAX_LAYERS
                or not all(_text(name) for name in layers) or len(set(layers)) != len(layers)):
            raise ValueError("expected layer inventory is missing or invalid")
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        result["reasons"] = [str(exc)]
        return result
    result.update(observation_status="observed", evidence_basis="recorded_identity")
    return result


def validate_snapshot(header, record) -> dict:
    """Derive bounded per-block candidate intervals from a complete snapshot.

    A valid result describes the submitted metadata. It never infers ownership
    from a block number, reference count, admission event or transfer completion.
    """
    result = _base()
    checked_header = validate_header(header)
    if checked_header["observation_status"] != "observed":
        result["reasons"] = checked_header["reasons"]
        return result
    try:
        raw, _ = _bounded(record)
        if not _shape(record, _SNAPSHOT) or not _envelope(record, "snapshot"):
            raise ValueError("snapshot fields or envelope are invalid")
        if record["session"] != header["session"]:
            raise ValueError("snapshot belongs to another session")
        if not _integer(record["forward_id"], 1) or not _uuid(record["snapshot_id"]) or not _uuid(record["pool_incarnation"]):
            raise ValueError("snapshot identity is incomplete")
        if record["snapshot_copy_complete"] is not True:
            raise ValueError("immutable host snapshot completion is missing")
        if record["observation_point"] != "model_entry":
            raise ValueError("snapshot observation point is unsupported")
        if record["forward_device_complete"] is not None:
            raise ValueError("this format cannot establish forward device completion")
        requests = record["requests"]
        if (type(requests) is not list or not 1 <= len(requests) <= MAX_REQUESTS
                or not all(_text(rid) for rid in requests) or len(set(requests)) != len(requests)):
            raise ValueError("request row identities are missing or ambiguous")
        count = record["num_tokens"]
        if not _integer(count, 1, MAX_TOKENS):
            raise ValueError("actual query token count is invalid")
        layers = record["layers"]
        names = header["profile"]["layer_names"]
        if type(layers) is not list or len(layers) != len(names):
            raise ValueError("snapshot does not cover the expected layer inventory")
        seen = set()
        cells = 0
        candidate_count = 0
        plan = []
        reference_queries = None
        reference_lengths = None
        reference_pool = None
        for layer in layers:
            if not _shape(layer, _LAYER) or not _text(layer["name"]):
                raise ValueError("layer fields are incomplete or unsupported")
            name = layer["name"]
            if name not in names or name in seen:
                raise ValueError("layer identity is missing, unexpected or repeated")
            seen.add(name)
            if (layer["causal"] is not True or layer["window"] is not None
                    or layer["layout"] != "LBHNC" or layer["backend"] != "FLASH_ATTN"
                    or layer["backend_version"] != header["engine"]["dependencies"]["flash_attn"]):
                raise ValueError("layer attention semantics are unsupported")
            size, capacity = layer["block_size"], layer["pool_num_blocks"]
            if not _integer(size, 1, MAX_CONTEXT) or not _integer(capacity, 1, 2**31 - 1):
                raise ValueError("physical cache dimensions are invalid")
            table, lengths = layer["block_table"], layer["seq_lens"]
            queries, writes = layer["query_start_loc"], layer["write_slots"]
            if (type(table) is not list or len(table) != len(requests)
                    or type(lengths) is not list or len(lengths) != len(requests)
                    or type(queries) is not list or len(queries) != len(requests) + 1
                    or type(writes) is not list or len(writes) != count):
                raise ValueError("layer tensor shapes do not match actual requests and tokens")
            if (not all(_integer(q, 0, count) for q in queries)
                    or queries[0] != 0 or queries[-1] != count
                    or any(b <= a for a, b in zip(queries, queries[1:]))):
                raise ValueError("actual query boundaries are invalid or include padding")
            if not all(_integer(n, 1, MAX_CONTEXT) for n in lengths):
                raise ValueError("exact sequence lengths are invalid")
            if reference_queries is not None and (queries != reference_queries or lengths != reference_lengths):
                raise ValueError("dense layer row lengths or query boundaries disagree")
            reference_queries, reference_lengths = queries, lengths
            # The supported profile has one dense cache group. Its layers use
            # the same table and write map (attn_utils.py:446-454, 503-546).
            pool = (size, capacity, table, writes)
            if reference_pool is not None and pool != reference_pool:
                raise ValueError("single-group layer cache dimensions or mappings disagree")
            reference_pool = pool
            cells += len(queries) + len(lengths) + len(writes)
            rows = []
            for row_index, rid in enumerate(requests):
                row = table[row_index]
                seq_len = lengths[row_index]
                begin, end = queries[row_index:row_index + 2]
                qlen = end - begin
                expected_blocks = (seq_len + size - 1) // size
                if qlen > seq_len:
                    raise ValueError("query length exceeds exact context length")
                if (type(row) is not list or len(row) != expected_blocks
                        or len(row) > MAX_BLOCKS_PER_ROW
                        or not all(_integer(block, 0, capacity - 1) for block in row)):
                    raise ValueError("block table is not the exact clipped context mapping")
                cells += len(row)
                if cells > MAX_SNAPSHOT_CELLS:
                    raise ValueError("snapshot tensor cells exceed supported bounds")
                if len(set(row)) != len(row):
                    raise ValueError("one dense context repeats a physical block")
                for token_index in range(begin, end):
                    pos = seq_len - qlen + token_index - begin
                    expected_slot = row[pos // size] * size + pos % size
                    if not _integer(writes[token_index], 0, capacity * size - 1) or writes[token_index] != expected_slot:
                        raise ValueError("write slots disagree with the actual query placement")
                ranges = [{"logical_start": i * size, "logical_end": min((i + 1) * size, seq_len),
                           "physical_block": block, "slot_start": 0,
                           "slot_end": min(size, seq_len - i * size)} for i, block in enumerate(row)]
                candidate_count += len(ranges)
                rows.append({"request": rid, "row": row_index, "query_start": begin, "query_end": end,
                             "context_tokens": seq_len, "candidate_ranges": ranges,
                             "write_slots": list(writes[begin:end])})
            plan.append({"layer": name, "block_size": size, "pool_num_blocks": capacity, "requests": rows})
    except (ValueError, KeyError, TypeError, RecursionError) as exc:
        result["reasons"] = [str(exc)]
        return result
    result.update(observation_status="observed", evidence_basis="forward_metadata",
                  observation_point="model_entry",
                  forward_id=record["forward_id"], snapshot_id=record["snapshot_id"],
                  pool_incarnation=record["pool_incarnation"], requests=list(requests),
                  query_tokens=count, candidate_block_ranges=candidate_count, tensor_cells=cells,
                  pool_dimensions={"block_size": size, "pool_num_blocks": capacity},
                  metadata_sha256=hashlib.sha256(raw).hexdigest(), candidate_plan=plan)
    return result


def audit_records(records) -> dict:
    """Audit exactly one closed process session with bounded, rederived summaries."""
    result = _base()
    result.update(record_count=0, snapshots=[], snapshot_count=0, valid_snapshot_count=0,
                  gap_count=0, recorded_profile=None, session=None,
                  candidate_block_ranges=0, query_tokens=0,
                  completion_scope="metadata copy only; GPU execution completion is unknown")
    if type(records) is not list or not 2 <= len(records) <= MAX_RECORDS:
        result["reasons"] = ["record list is empty, incomplete or exceeds supported bounds"]
        return result
    result["record_count"] = len(records)
    total_bytes = total_nodes = 0
    try:
        for record in records:
            raw, nodes = _bounded(record)
            total_bytes += len(raw)
            total_nodes += nodes
            if total_bytes > MAX_TOTAL_BYTES or total_nodes > MAX_TOTAL_NODES:
                raise ValueError("session exceeds supported byte or node bounds")
    except (ValueError, TypeError, RecursionError) as exc:
        result["reasons"] = [str(exc)]
        return result
    header = records[0]
    checked = validate_header(header)
    if checked["observation_status"] != "observed":
        result["reasons"] = checked["reasons"]
        return result
    result["session"] = header["session"]
    result["recorded_profile"] = json.loads(json.dumps(header["profile"]))
    reasons = []
    seen_snapshots = set()
    pools = {}
    previous_forward = 0
    cells = 0
    snapshots = 0
    for index, record in enumerate(records):
        if not _envelope(record) or record["session"] != header["session"] or record["seq"] != index:
            reasons.append("record sequence or session identity is inconsistent")
            break
        if index == 0:
            continue
        kind = record["kind"]
        if index == len(records) - 1:
            if not _shape(record, _FOOTER) or kind != "footer":
                reasons.append("closed session footer is missing or malformed")
            elif (not _integer(record["snapshots"], high=MAX_RECORDS)
                  or record["snapshots"] != snapshots
                  or not _integer(record["dropped"]) or not _integer(record["errors"])
                  or type(record["healthy"]) is not bool):
                reasons.append("footer accounting does not match the recorded session")
            elif record["dropped"] or record["errors"] or record["healthy"] is not True:
                reasons.append("recorder reports dropped or unhealthy evidence")
            continue
        if kind == "gap":
            result["gap_count"] += 1
            if (not _shape(record, _GAP) or not _text(record["reason"])
                    or not (record["forward_id"] is None or _integer(record["forward_id"], 1))):
                reasons.append("recording gap has malformed identity or reason")
            else:
                reasons.append("recording contains an explicit evidence gap")
            continue
        if kind != "snapshot":
            reasons.append("session contains an unsupported record or premature footer")
            continue
        snapshots += 1
        checked = validate_snapshot(header, record)
        if checked["observation_status"] == "observed":
            if checked["forward_id"] <= previous_forward or checked["snapshot_id"] in seen_snapshots:
                checked = _base()
                checked["reasons"] = ["forward or snapshot identity is reused or out of order"]
            elif (checked["pool_incarnation"] in pools
                  and pools[checked["pool_incarnation"]] != checked["pool_dimensions"]):
                checked = _base()
                checked["reasons"] = ["cache dimensions changed without a new pool incarnation"]
            else:
                previous_forward = checked["forward_id"]
                seen_snapshots.add(checked["snapshot_id"])
                pools[checked["pool_incarnation"]] = checked["pool_dimensions"]
                cells += checked["tensor_cells"]
                if cells > MAX_TOTAL_CELLS:
                    reasons.append("session tensor cells exceed supported bounds")
                    break
        summary = {key: value for key, value in checked.items() if key != "candidate_plan"}
        summary["record_index"] = index
        result["snapshots"].append(summary)
        if checked["observation_status"] == "observed":
            result["valid_snapshot_count"] += 1
            result["candidate_block_ranges"] += checked["candidate_block_ranges"]
            result["query_tokens"] += checked["query_tokens"]
        else:
            reasons.append("one or more snapshots are incomplete, unsupported or inconsistent")
    result["snapshot_count"] = snapshots
    if not snapshots:
        reasons.append("no forward metadata snapshots were recorded")
    result["reasons"] = list(dict.fromkeys(reasons))[:MAX_REASONS]
    if not result["reasons"]:
        result.update(observation_status="observed", evidence_basis="forward_metadata")
    return result
