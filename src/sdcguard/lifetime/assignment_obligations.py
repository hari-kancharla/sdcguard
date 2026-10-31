"""Re-derive a recorded allocator-to-model-entry mapping relation.

Owner: Hari Krishna Kancharla.
This is an offline evidence contract. No native adapter or runtime protection is
established by accepting a structural record. Read rights and physical reads are
not inferred from assignments, allocation ordinals or reference counts.
"""
from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
import json
import re


PROPERTY = "allocator_to_model_entry_mapping_consistency"
SCHEMA = 1
DEFAULT_ADAPTER = "vllm.0.32.uniproc.eager.v1"
MAX_RECORDS = 16384
MAX_RECORD_NODES = 1_000_000
MAX_TOTAL_NODES = 2_000_000
MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_REQUESTS = 256
MAX_LAYERS = 128
MAX_FRAMES = 512
MAX_CONTEXT = 1_048_576
MAX_BLOCKS = 262144
MAX_CELLS = 1_000_000
MAX_REASONS = 64
MAX_FINDINGS = 512
MAX_WITNESSES = 64
_COMMON = {"schema", "session", "process", "seq", "kind"}
_UUID = re.compile(r"[0-9a-f]{32}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_CONFIG_DIMENSIONS = {"capacity", "block_size", "scheduler_block_size", "hash_block_size",
                      "groups", "layout", "layer_names"}


class EvidenceError(ValueError):
    pass


def adapter_contract(adapter=None):
    """Return a fresh installed description. This does not inspect an engine."""
    value = json.loads(files("sdcguard").joinpath("case_data/assignment-pins.json").read_text())
    try:
        return deepcopy(value["adapters"][adapter or DEFAULT_ADAPTER])
    except (KeyError, TypeError):
        raise EvidenceError("unsupported assignment adapter") from None


def source_pins(adapter=None):
    return {entry["path"]: entry["sha256"] for entry in adapter_contract(adapter)["files"]}


def _base():
    return {"schema": "sdcguard.assignment-report.v1", "property": PROPERTY,
            "verdict": "unknown", "scope_verdict": "inconclusive", "findings": [],
            "ownership_verdict": "unknown", "physical_read_verdict": "unknown",
            "content_provenance": "unknown", "runtime_isolation": "not established",
            "authenticity": "not established", "missing": [], "session": None,
            "allocation_incarnations": 0, "closed_frames": 0, "complete_frames": 0,
            "summary": {"satisfied": 0, "violated": 0, "unknown": 0,
                        "compared_layer_positions": 0, "contradictions": 0},
            "findings_omitted": 0, "findings_complete": True,
            "witnesses_omitted": 0, "witnesses_complete": True,
            "basis": "recorded same-process initialization, dispatch and stable mapping observations",
            "limits": ["Mapping agreement does not prove the allocator assignment was valid.",
                       "Allocation ordinals label acquisitions, not content changes or native lease nonces.",
                       "Missing assignment and zero references do not prove absent read rights.",
                       "No native recorder or GPU validation is supplied by this CPU contract."]}


def _require(condition, reason):
    if not condition:
        raise EvidenceError(reason)


def _integer(value, low=0, high=2**63 - 1):
    return type(value) is int and low <= value <= high


def _id(value):
    return type(value) is str and _UUID.fullmatch(value) is not None


def _text(value, maximum=512):
    return (type(value) is str and 0 < len(value) <= maximum
            and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value))


def _shape(value, keys, reason="record fields are incomplete or unsupported"):
    _require(type(value) is dict and value.keys() == keys, reason)


def _bounded(value):
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _require(nodes <= MAX_RECORD_NODES and depth <= 32, "record structure exceeds supported bounds")
        typ = type(item)
        if typ is str:
            _require(len(item) <= MAX_RECORD_BYTES, "record string exceeds supported bounds")
            _require(not any(0xD800 <= ord(c) <= 0xDFFF for c in item), "record has invalid Unicode")
        elif typ is dict:
            _require(len(item) <= MAX_RECORD_NODES and len(pending) + 2 * len(item) <= MAX_RECORD_NODES,
                     "record structure exceeds supported bounds")
            for key, child in item.items():
                _require(type(key) is str, "record keys must be strings")
                pending.extend(((key, depth + 1), (child, depth + 1)))
        elif typ is list:
            _require(len(item) <= MAX_RECORD_NODES and len(pending) + len(item) <= MAX_RECORD_NODES,
                     "record structure exceeds supported bounds")
            pending.extend((child, depth + 1) for child in item)
        elif typ is int:
            _require(item.bit_length() <= 63, "record integer exceeds supported bounds")
        else:
            _require(item is None or typ is bool, "record contains unsupported JSON values")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    _require(len(raw) <= MAX_RECORD_BYTES, "record exceeds supported byte bound")
    return raw, nodes


class _Trace:
    def __init__(self, header):
        _shape(header, _COMMON | {"pid", "adapter", "source_sha256", "profile", "coverage", "dependencies"})
        _require(_integer(header["pid"], 1) and _text(header["adapter"]), "process or adapter identity is invalid")
        self.contract = adapter_contract(header["adapter"])
        _shape(header["dependencies"], {"python", "torch", "flash_attn"}, "dependency observations are incomplete")
        _require(all(_text(value, 256) for value in header["dependencies"].values())
                 and header["dependencies"]["flash_attn"] == "3", "dependency observations are unsupported")
        _require(type(header["source_sha256"]) is dict
                 and header["source_sha256"] == source_pins(header["adapter"]),
                 "exact relevant source pins are missing or differ")
        self.profile = header["profile"]
        local_reference = type(self.profile) is dict and "model_reference" in self.profile
        _shape(self.profile, set(self.contract["profile"]) | {"model_revision", "layer_names"}
               | ({"model_reference"} if local_reference else set()),
               "effective profile is incomplete or unsupported")
        if local_reference:
            from .. import model_reference
            model_reference.validate_observation(self.profile["model"], self.profile["model_revision"],
                                                  self.profile["model_reference"])
        for name, expected in self.contract["profile"].items():
            if local_reference and name == "model":
                continue
            _require(type(self.profile[name]) is type(expected) and self.profile[name] == expected,
                     "effective profile is unsupported")
        _require(type(self.profile["model_revision"]) is str
                 and _REVISION.fullmatch(self.profile["model_revision"]) is not None,
                 "immutable model revision is missing")
        names = self.profile["layer_names"]
        _require(type(names) is list and 1 <= len(names) <= MAX_LAYERS
                 and all(_text(name) for name in names) and len(set(names)) == len(names),
                 "full layer inventory is missing or ambiguous")
        self.methods = {**self.contract["mutation_methods"], **self.contract["guard_methods"]}
        self.coverage = set(self.methods)
        self.check_coverage(header["coverage"])
        self.header = header
        self.objects = {}
        self.fresh_objects = set()
        self.inits = {}
        self.core_active = {}
        self.schedules = {}
        self.scheduler_steps = {}
        self.request_ids = {}
        self.request_names = {}
        self.pending_dispatches = {}
        self.mutations = {}
        self.used_mutations = set()
        self.frames = {}
        self.findings = []
        self.findings_omitted = 0
        self.witnesses = 0
        self.summary = {"satisfied": 0, "violated": 0, "unknown": 0,
                        "compared_layer_positions": 0, "contradictions": 0}
        self.acquisition_count = 0
        self.block_acquisitions = 0
        self.frame_count = 0
        self.complete_count = 0
        self.cells = 0
        self.gaps = []

    def fields(self, record, extra):
        _shape(record, _COMMON | set(extra))

    def check_coverage(self, value):
        _require(type(value) is list and len(value) == len(self.coverage)
                 and all(type(x) is str for x in value) and set(value) == self.coverage,
                 "complete installed mutation-hook coverage is missing")

    def object(self, value, role):
        _require(_id(value), "object lifetime token is invalid")
        previous = self.objects.get(value)
        _require(previous is None or previous == role, "object token changes native role")
        self.objects[value] = role
        return value

    def fresh(self, value):
        _require(value not in self.fresh_objects, "a new cache or allocator lifetime reuses an old object token")
        self.fresh_objects.add(value)

    def participants(self, record, roles):
        _shape(record["objects"], set(roles), "native object chain is incomplete")
        _shape(record["classes"], set(roles), "native class observations are incomplete")
        result = {}
        for role in roles:
            _require(record["classes"][role] in self.contract["classes"][role]
                     and type(record["classes"][role]) is str, "native class is unsupported")
            result[role] = self.object(record["objects"][role], role)
        return result

    def dimensions(self, value):
        _shape(value, _CONFIG_DIMENSIONS, "config-derived cache dimensions are incomplete or premature")
        _require(_integer(value["capacity"], 2, 2**31 - 1), "cache capacity is invalid")
        size = value["block_size"]
        _require(_integer(size, 1, MAX_CONTEXT), "cache block size is invalid")
        _require(all(type(value[k]) is int and value[k] == size for k in
                     ("scheduler_block_size", "hash_block_size")),
                 "scheduler, group and hash block mapping is unsupported")
        _require(type(value["groups"]) is int and value["groups"] == 1,
                 "cache group count is unsupported")
        _require(value["layout"] == self.profile["layout"]
                 and value["layer_names"] == self.profile["layer_names"],
                 "cache layout or layer association differs")
        return deepcopy(value)

    def init(self, record, phase=None):
        _require(_id(record["init"]) and record["init"] in self.inits, "parent initialization is missing")
        value = self.inits[record["init"]]
        if phase is not None:
            _require(value["phase"] == phase, "native initialization order is incomplete or inconsistent")
        return value

    def association(self, token):
        _require(_id(token) and token in self.inits, "cache association is missing")
        item = self.inits[token]
        _require(item["phase"] == "ready", "cache association has no complete initialization proof")
        return item

    def init_begin(self, r):
        self.fields(r, {"init", "objects", "classes"})
        _require(_id(r["init"]) and r["init"] not in self.inits, "initialization lifetime token is reused")
        objects = self.participants(r, ("core", "executor"))
        previous = self.core_active.get(objects["core"])
        if previous is not None:
            _require(self.inits[previous]["phase"] == "ready", "overlapping initializations share a native core")
            self.retire_item(self.inits[previous])
        self.inits[r["init"]] = {"phase": "begin", "objects": objects, "active": False,
                                 "retired": False, "epoch": 0, "open": set(), "poison": [], "allocations": {}}
        self.core_active[objects["core"]] = r["init"]

    def configs_derived(self, r):
        self.fields(r, {"init", "worker_config", "scheduler_config", "dimensions"})
        item = self.init(r, "begin")
        item["worker_config"] = self.object(r["worker_config"], "worker_config")
        item["scheduler_config"] = self.object(r["scheduler_config"], "scheduler_config")
        self.fresh(item["worker_config"])
        self.fresh(item["scheduler_config"])
        _require(r["worker_config"] != r["scheduler_config"], "configuration copy edge is not recorded")
        item["dimensions"] = self.dimensions(r["dimensions"])
        item["phase"] = "derived"

    def worker_init(self, r):
        self.fields(r, {"init", "executor", "selected_config", "rank", "objects", "classes"})
        item = self.init(r, "derived")
        _require(r["executor"] == item["objects"]["executor"]
                 and r["selected_config"] == item["worker_config"]
                 and type(r["rank"]) is int and r["rank"] == 0,
                 "selected worker is not linked to the parent initialization")
        item["objects"].update(self.participants(r, ("wrapper", "worker", "runner")))
        for other in self.inits.values():
            if other is not item and any(other["objects"].get(role) == item["objects"][role]
                                         for role in ("worker", "runner")):
                _require(other["phase"] == "ready", "overlapping initializations share a native worker or runner")
                self.retire_item(other)
        self.pending_dispatches.pop(item["objects"]["runner"], None)
        item["phase"] = "worker"

    def runner_ready(self, r):
        self.fields(r, {"init", "runner", "input_config", "runner_config", "worker_pool", "dimensions",
                        "kernel_block_size", "ratio"})
        item = self.init(r, "worker")
        _require(not item["retired"], "retired initialization cannot become ready")
        _require(r["runner"] == item["objects"]["runner"]
                 and r["input_config"] == item["worker_config"], "runner initialization input is not causally bound")
        item["runner_config"] = self.object(r["runner_config"], "runner_config")
        self.fresh(item["runner_config"])
        item["worker_pool"] = self.object(r["worker_pool"], "worker_pool")
        self.fresh(item["worker_pool"])
        _require(r["runner_config"] not in (item["worker_config"], item["scheduler_config"]),
                 "runner configuration copy is missing")
        _require(self.dimensions(r["dimensions"]) == item["dimensions"], "worker cache dimensions differ")
        kernel = r["kernel_block_size"]
        size = item["dimensions"]["block_size"]
        _require(_integer(kernel, 1, MAX_CONTEXT) and size % kernel == 0
                 and type(r["ratio"]) is int and r["ratio"] == size // kernel and r["ratio"] == 1,
                 "observed runner kernel size or block subdivision is unsupported")
        item["kernel_block_size"] = kernel
        item["phase"] = "runner"

    def init_return(self, r):
        self.fields(r, {"init", "scheduler_config"})
        item = self.init(r, "runner")
        _require(not item["retired"], "retired initialization cannot return a ready cache")
        _require(r["scheduler_config"] == item["scheduler_config"], "returned configuration is not the derived object")
        item["phase"] = "returned"

    def scheduler_ready(self, r):
        self.fields(r, {"init", "passed_config", "objects", "classes", "dimensions", "null_block"})
        item = self.init(r, "returned")
        _require(not item["retired"], "retired initialization cannot activate its scheduler")
        _require(r["passed_config"] == item["scheduler_config"], "scheduler did not receive the retained returned configuration")
        _require(self.dimensions(r["dimensions"]) == item["dimensions"], "allocator and worker cache mapping differ")
        item["objects"].update(self.participants(r, ("scheduler", "coordinator", "manager", "group_manager", "pool")))
        for role in ("scheduler", "coordinator", "manager", "group_manager", "pool"):
            self.fresh(item["objects"][role])
        _require(type(r["null_block"]) is int and r["null_block"] == self.contract["null_block"],
                 "null block identity differs from the pinned source constructor")
        item.update(phase="ready", active=True, null_block=r["null_block"])

    def retire(self, r):
        self.fields(r, {"association", "reason"})
        item = self.association(r["association"])
        _require(r["reason"] in ("shutdown", "cache_replacement", "reset"), "retirement reason is unsupported")
        self.retire_item(item)

    def retire_item(self, item):
        item.update(active=False, retired=True)

    def mutation_begin(self, r):
        self.fields(r, {"association", "mutation", "method", "receiver", "parent"})
        item = self.association(r["association"])
        _require(_id(r["mutation"]) and r["mutation"] not in self.used_mutations,
                 "mutation call identity is missing or reused")
        _require(type(r["method"]) is str and r["method"] in self.methods, "mutation method has no source coverage")
        role = self.methods[r["method"]]
        _require(r["receiver"] == item["objects"][role], "mutation receiver belongs to another object or pool")
        if r["parent"] is not None:
            _require(r["parent"] in item["open"], "mutation parent call is missing")
        self.used_mutations.add(r["mutation"])
        self.mutations[r["mutation"]] = {"association": r["association"], "method": r["method"],
                                        "parent": r["parent"], "acquired": False}
        item["open"].add(r["mutation"])
        item["epoch"] += 1

    def acquire(self, r):
        self.fields(r, {"association", "mutation", "blocks"})
        item = self.association(r["association"])
        _require(r["mutation"] in item["open"], "acquisition has no active native call")
        call = self.mutations[r["mutation"]]
        _require(call["method"] == self.contract["acquisition_method"] and not call["acquired"],
                 "acquisition is not the unique return of a native allocation call")
        _require(type(r["blocks"]) is list and len(r["blocks"]) <= MAX_BLOCKS,
                 "acquisition block list exceeds bounds")
        seen = set()
        for block in r["blocks"]:
            _shape(block, {"block", "incarnation"}, "allocation fact is malformed")
            index, ordinal = block["block"], block["incarnation"]
            _require(_integer(index, 0, item["dimensions"]["capacity"] - 1)
                     and index != item["null_block"] and index not in seen,
                     "acquisition block is null, repeated or outside the pool")
            _require(_integer(ordinal, 1) and ordinal == item["allocations"].get(index, 0) + 1,
                     "allocation incarnation history is missing or inconsistent")
            seen.add(index)
            item["allocations"][index] = ordinal
            self.block_acquisitions += 1
            _require(self.block_acquisitions <= MAX_BLOCKS, "allocation history exceeds bounds")
        call["acquired"] = True
        self.acquisition_count += 1

    def mutation_end(self, r):
        self.fields(r, {"association", "mutation", "success"})
        item = self.association(r["association"])
        _require(r["mutation"] in item["open"] and type(r["success"]) is bool,
                 "mutation closure is missing or malformed")
        _require(not any(m["parent"] == r["mutation"] for token, m in self.mutations.items()
                         if token in item["open"]), "mutation closed before its child")
        call = self.mutations[r["mutation"]]
        if not r["success"]:
            item["poison"].append("a native mutation failed")
        elif call["method"] == self.contract["acquisition_method"] and not call["acquired"]:
            item["poison"].append("successful native acquisition has no recorded return")
        item["open"].remove(r["mutation"])
        item["epoch"] += 1

    def unsupported_branch(self, r):
        self.fields(r, {"association", "mutation", "method", "reason"})
        item = self.association(r["association"])
        _require(type(r["method"]) is str and r["method"] in self.methods and _text(r["reason"]),
                 "unsupported-operation observation is malformed")
        _require(r["mutation"] in item["open"]
                 and self.mutations[r["mutation"]]["method"] == r["method"],
                 "unsupported operation lacks its active native call")
        item["poison"].append("an actual unsupported operation was observed")

    def scheduled_requests(self, value, association):
        _require(type(value) is list and len(value) <= MAX_REQUESTS, "scheduled request set is oversized")
        seen = set()
        result = {}
        for request in value:
            _shape(request, {"request", "incarnation", "query_tokens"}, "request lifetime observation is malformed")
            name, token = request["request"], request["incarnation"]
            _require(_text(name) and _id(token) and name not in seen
                     and _integer(request["query_tokens"], 1, MAX_CONTEXT), "scheduled request identity or count is invalid")
            self.object(token, "request")
            _require(self.request_names.get(token, name) == name, "native request object changes its request identity")
            self.request_names[token] = name
            key = (association, name)
            previous = self.request_ids.get(key)
            _require(previous is None or previous == token, "reused request lifetime is outside this contract")
            self.request_ids[key] = token
            seen.add(name)
            result[name] = (token, request["query_tokens"])
        return result

    def schedule(self, r):
        self.fields(r, {"association", "core", "core_scheduler", "scheduler", "dispatch", "step", "requests"})
        item = self.association(r["association"])
        _require(r["core"] == item["objects"]["core"]
                 and r["core_scheduler"] == item["objects"]["scheduler"]
                 and self.core_active[item["objects"]["core"]] == r["association"],
                 "actual core.scheduler attachment is not bound to this initialization")
        _require(_id(r["dispatch"]) and r["dispatch"] not in self.schedules,
                 "schedule object lifetime is missing or reused")
        self.object(r["dispatch"], "schedule_output")
        _require(r["scheduler"] == item["objects"]["scheduler"] and _integer(r["step"], 1)
                 and r["step"] > self.scheduler_steps.get(r["scheduler"], 0),
                 "native scheduler step or receiver is inconsistent")
        self.scheduler_steps[r["scheduler"]] = r["step"]
        self.schedules[r["dispatch"]] = {"association": r["association"], "step": r["step"],
                                         "requests": self.scheduled_requests(r["requests"], r["association"]),
                                         "entered": False, "framed": False}

    def worker_entry(self, r):
        self.fields(r, {"association", "dispatch", "runner", "requests"})
        item = self.association(r["association"])
        call = self.schedules.get(r["dispatch"])
        _require(call is not None and call["association"] == r["association"] and not call["entered"],
                 "worker invocation has no unique matching schedule object")
        _require(r["runner"] == item["objects"]["runner"]
                 and self.scheduled_requests(r["requests"], r["association"]) == call["requests"],
                 "worker invocation or immutable schedule facts differ")
        call["entered"] = True
        self.pending_dispatches[r["runner"]] = r["dispatch"]

    def frame_begin(self, r):
        self.fields(r, {"association", "dispatch", "runner", "frame", "epoch", "coverage"})
        item = self.association(r["association"])
        call = self.schedules.get(r["dispatch"])
        _require(call is not None and call["association"] == r["association"]
                 and call["entered"] and call["requests"] and not call["framed"],
                 "frame lacks a unique positive-work direct worker invocation")
        _require(r["runner"] == item["objects"]["runner"] and _id(r["frame"])
                 and r["frame"] not in self.frames, "frame lifetime or runner identity is invalid")
        _require(self.pending_dispatches.get(r["runner"]) == r["dispatch"],
                 "frame does not belong to the runner's current pending direct dispatch")
        self.check_coverage(r["coverage"])
        _require(_integer(r["epoch"]) and r["epoch"] == item["epoch"], "declared epoch does not match recorded mutations")
        call["framed"] = True
        self.frames[r["frame"]] = {"association": r["association"], "dispatch": r["dispatch"],
                                   "epoch": item["epoch"], "assignments": None, "metadata": None,
                                   "reasons": list(item["poison"]), "ended": False}
        if not item["active"] or item["open"]:
            self.frames[r["frame"]]["reasons"].append("cache association is retired or a mutation is open")
        for token, other in self.frames.items():
            if token != r["frame"] and other["association"] == r["association"] and not other["ended"]:
                reason = "model-entry observation intervals overlap"
                other["reasons"].append(reason)
                self.frames[r["frame"]]["reasons"].append(reason)
        _require(len(self.frames) <= MAX_FRAMES, "frame count exceeds bounds")

    def frame(self, token):
        _require(_id(token) and token in self.frames and not self.frames[token]["ended"],
                 "observation has no open frame")
        return self.frames[token]

    def assignments(self, r):
        self.fields(r, {"frame", "manager", "pool", "rows"})
        frame = self.frame(r["frame"])
        item = self.association(frame["association"])
        _require(r["manager"] == item["objects"]["manager"] and r["pool"] == item["objects"]["pool"],
                 "allocator sample belongs to another manager or pool")
        _require(frame["assignments"] is None and type(r["rows"]) is list
                 and len(r["rows"]) <= MAX_REQUESTS, "assignment rows are missing, repeated or oversized")
        rows = {}
        call = self.schedules[frame["dispatch"]]
        for row in r["rows"]:
            _shape(row, {"request", "incarnation", "blocks"}, "assignment row is malformed")
            name = row["request"]
            _require(name in call["requests"] and name not in rows
                     and row["incarnation"] == call["requests"][name][0], "assignment request lifetime differs")
            blocks = row["blocks"]
            if blocks is not None:
                _require(type(blocks) is list and len(blocks) <= MAX_BLOCKS, "assignment blocks exceed bounds")
                for block in blocks:
                    _shape(block, {"block", "incarnation"}, "assignment allocation fact is malformed")
                    _require(_integer(block["block"], 0, item["dimensions"]["capacity"] - 1)
                             and _integer(block["incarnation"], 1), "assignment allocation dimensions are invalid")
                self.add_cells(len(blocks))
            rows[name] = deepcopy(blocks)
        frame["assignments"] = rows

    def add_cells(self, count):
        self.cells += count
        _require(self.cells <= MAX_CELLS, "aggregate block observations exceed bounds")

    def model_entry(self, r):
        self.fields(r, {"frame", "runner", "worker_pool", "requests", "seq_lens", "query_lengths",
                        "layers", "semantics", "copy_complete"})
        frame = self.frame(r["frame"])
        item = self.association(frame["association"])
        _require(frame["metadata"] is None, "model-entry observation is repeated")
        _require(r["runner"] == item["objects"]["runner"] and r["worker_pool"] == item["worker_pool"],
                 "model-entry cache belongs to another runner or pool lifetime")
        _require(r["copy_complete"] is True, "immutable model-entry copy completion is missing")
        expected_semantics = {"observation_point": "model_entry", "causal": True, "window": None,
                              "layout": self.profile["layout"], "backend": self.profile["backend"]}
        _shape(r["semantics"], set(expected_semantics), "model-entry semantics are incomplete")
        for key, value in expected_semantics.items():
            _require(type(r["semantics"][key]) is type(value) and r["semantics"][key] == value,
                     "model-entry attention semantics are unsupported")
        names, lengths, queries = r["requests"], r["seq_lens"], r["query_lengths"]
        scheduled = self.schedules[frame["dispatch"]]["requests"]
        _require(type(names) is list and len(names) == len(scheduled)
                 and all(type(name) is str and name in scheduled for name in names)
                 and len(set(names)) == len(names), "actual request row identity is incomplete")
        _require(type(lengths) is list and type(queries) is list and len(lengths) == len(names) == len(queries),
                 "actual sequence/query dimensions differ")
        _require(all(_integer(n, 1, MAX_CONTEXT) for n in lengths)
                 and all(_integer(q, 1, n) and q == scheduled[name][1]
                         for name, q, n in zip(names, queries, lengths)), "exact context or query lengths are invalid")
        layers = r["layers"]
        _require(type(layers) is list and len(layers) == len(self.profile["layer_names"]),
                 "model-entry layer coverage is incomplete")
        seen = set()
        width = item["dimensions"]["block_size"]
        for layer in layers:
            _shape(layer, {"name", "rows"}, "model-entry layer data is malformed")
            _require(type(layer["name"]) is str and layer["name"] in self.profile["layer_names"]
                     and layer["name"] not in seen, "model-entry layer identity is repeated or unsupported")
            seen.add(layer["name"])
            _require(type(layer["rows"]) is list and len(layer["rows"]) == len(names), "model-entry table row count differs")
            for row, length in zip(layer["rows"], lengths):
                _require(type(row) is list and len(row) == (length + width - 1) // width
                         and len(row) <= MAX_BLOCKS
                         and all(_integer(block, 0, item["dimensions"]["capacity"] - 1) for block in row),
                         "model-entry table is not a bounded clipped context")
                self.add_cells(len(row))
        frame["metadata"] = deepcopy({k: r[k] for k in ("requests", "seq_lens", "query_lengths", "layers")})

    def finding(self, value):
        self.summary[value["verdict"]] += 1
        if len(self.findings) < MAX_FINDINGS:
            self.findings.append(value)
        else:
            self.findings_omitted += 1

    def frame_end(self, r):
        self.fields(r, {"frame", "epoch", "coverage"})
        frame = self.frame(r["frame"])
        item = self.association(frame["association"])
        self.check_coverage(r["coverage"])
        _require(_integer(r["epoch"]) and r["epoch"] == item["epoch"], "ending epoch does not match recorded mutations")
        frame["ended"] = True
        self.frame_count += 1
        if (frame["epoch"] != item["epoch"] or item["open"] or not item["active"]):
            frame["reasons"].append("allocator mutation or cache retirement overlapped the observation")
        if self.pending_dispatches.get(item["objects"]["runner"]) != frame["dispatch"]:
            frame["reasons"].append("the runner entered another direct dispatch during the observation")
        else:
            self.pending_dispatches.pop(item["objects"]["runner"])
        frame["reasons"].extend(item["poison"])
        if frame["assignments"] is None or frame["metadata"] is None:
            frame["reasons"].append("allocator or model-entry observation is missing")
        if frame["reasons"]:
            self.finding({"frame": r["frame"], "verdict": "unknown",
                          "missing": list(dict.fromkeys(frame["reasons"]))[:MAX_REASONS]})
            return
        metadata = frame["metadata"]
        size = item["dimensions"]["block_size"]
        complete = True
        for row_index, name in enumerate(metadata["requests"]):
            required = (metadata["seq_lens"][row_index] + size - 1) // size
            assigned = frame["assignments"].get(name)
            missing, mismatches = set(), []
            compared = contradictions = 0
            for position in range(required):
                if assigned is None or position >= len(assigned):
                    missing.add("allocator assignment is absent or shorter than the required context")
                    continue
                expected = assigned[position]
                if item["allocations"].get(expected["block"]) != expected["incarnation"]:
                    missing.add("assigned allocation incarnation has no complete native acquisition history")
                    continue
                for layer in metadata["layers"]:
                    actual = layer["rows"][row_index][position]
                    compared += 1
                    if actual != expected["block"]:
                        contradictions += 1
                        if len(self.findings) < MAX_FINDINGS and self.witnesses < MAX_WITNESSES:
                            mismatches.append({"layer": layer["name"], "logical_block": position,
                                               "allocator_block": expected["block"], "model_entry_block": actual,
                                               "allocation_incarnation": expected["incarnation"]})
                            self.witnesses += 1
            self.summary["compared_layer_positions"] += compared
            self.summary["contradictions"] += contradictions
            verdict = "violated" if contradictions else "unknown" if missing else "satisfied"
            complete = complete and not missing
            self.finding({"frame": r["frame"], "request": name,
                                  "request_incarnation": self.schedules[frame["dispatch"]]["requests"][name][0],
                                  "verdict": verdict, "checked_context_blocks": required,
                                  "compared_layer_positions": compared, "contradiction_count": contradictions,
                                  "contradictions": mismatches, "witnesses_omitted": contradictions - len(mismatches),
                                  "missing": sorted(missing)})
        if complete:
            self.complete_count += 1

    def gap(self, r):
        self.fields(r, {"reason"})
        _require(_text(r["reason"]), "recording gap is malformed")
        self.gaps.append("recording contains an explicit evidence gap")

    def footer(self, r, record_count):
        self.fields(r, {"records", "samples", "acquisitions", "dropped", "errors", "healthy"})
        _require(_integer(r["records"]) and r["records"] == record_count
                 and _integer(r["samples"]) and r["samples"] == self.frame_count
                 and _integer(r["acquisitions"]) and r["acquisitions"] == self.acquisition_count,
                 "footer accounting does not match recorded events")
        _require(_integer(r["dropped"]) and _integer(r["errors"]) and type(r["healthy"]) is bool,
                 "footer health accounting is malformed")
        _require(not r["dropped"] and not r["errors"] and r["healthy"] is True and not self.gaps,
                 "recording has lost or unhealthy evidence")
        _require(not any(item["open"] for item in self.inits.values()), "native mutation has no terminal observation")
        _require(all(frame["ended"] for frame in self.frames.values()), "observation interval is not closed")
        _require(all(item["phase"] == "ready" for item in self.inits.values()), "native initialization chain is unfinished")
        _require(all(not call["requests"] or (call["entered"] and call["framed"])
                     for call in self.schedules.values()), "positive-work dispatch has missing worker or frame evidence")


def evaluate(records):
    """Judge the named mapping relation from one bounded closed session."""
    result = _base()
    try:
        _require(type(records) is list and 2 <= len(records) <= MAX_RECORDS,
                 "record list is missing, incomplete or exceeds supported bounds")
        total_bytes = total_nodes = 0
        for index, record in enumerate(records):
            raw, nodes = _bounded(record)
            total_bytes += len(raw)
            total_nodes += nodes
            _require(total_bytes <= MAX_TOTAL_BYTES and total_nodes <= MAX_TOTAL_NODES,
                     "session exceeds aggregate byte or node bounds")
            _require(type(record) is dict and type(record.get("schema")) is int
                     and record["schema"] == SCHEMA and _id(record.get("session"))
                     and _id(record.get("process")) and type(record.get("seq")) is int
                     and record["seq"] == index and type(record.get("kind")) is str,
                     "record sequence, process or schema is malformed")
        header = records[0]
        _require(header["kind"] == "header" and records[-1]["kind"] == "footer", "closed session header or footer is missing")
        trace = _Trace(header)
        result["session"] = header["session"]
        handlers = {name: getattr(trace, name) for name in (
            "init_begin", "configs_derived", "worker_init", "runner_ready", "init_return", "scheduler_ready",
            "retire", "mutation_begin", "mutation_end", "acquire", "unsupported_branch", "schedule",
            "worker_entry", "frame_begin", "assignments", "model_entry", "frame_end", "gap")}
        for record in records[1:-1]:
            _require(record["session"] == header["session"] and record["process"] == header["process"],
                     "records mix session or process incarnations")
            _require(record["kind"] in handlers, "unsupported record kind or premature session marker")
            handlers[record["kind"]](record)
        footer = records[-1]
        _require(footer["session"] == header["session"] and footer["process"] == header["process"],
                 "footer belongs to another process session")
        trace.footer(footer, len(records))
        result.update(findings=trace.findings, allocation_incarnations=trace.block_acquisitions,
                      closed_frames=trace.frame_count, complete_frames=trace.complete_count, summary=trace.summary,
                      findings_omitted=trace.findings_omitted,
                      findings_complete=trace.findings_omitted == 0,
                      witnesses_omitted=trace.summary["contradictions"] - trace.witnesses,
                      witnesses_complete=trace.summary["contradictions"] == trace.witnesses)
        if trace.summary["violated"]:
            result.update(verdict="violated", scope_verdict="not fine")
        elif trace.summary["satisfied"] and not trace.summary["unknown"]:
            result.update(verdict="satisfied", scope_verdict="fine")
        else:
            result["missing"] = ["one or more mapping observations are absent or incomplete"]
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result = _base()
        result["missing"] = [str(exc)[:512]]
    return result
