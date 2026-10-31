"""Replay completed positive local allocation-recipient observations.

Owner: Hari Krishna Kancharla.
This pure contract checks local captured metadata. It establishes no read
rights, exclusive ownership, lease validity, content provenance or device load.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.resources import files
import json

from .. import evidence_io as io
from . import assignment_obligations as mapping
from . import mapping_closure


SCHEMA = 1
DEFAULT_ADAPTER = "vllm.0.32.uniproc.eager.allocation-recipient.v1"
PROPERTY = "causal_local_allocation_recipient_facts"
MAX_RECORDS = 16384
MAX_RECORD_BYTES = 262144
MAX_TOTAL_BYTES = 16777216
MAX_RECORD_NODES = 32768
MAX_TOTAL_NODES = 262144
MAX_OUTER_CALLS = 1024
MAX_PHASE_CALLS = 4096
MAX_BLOCKS_PER_RETURN = 512
MAX_TOTAL_BLOCK_FACTS = 8192
MAX_NESTING = 8
_COMMON = {"schema", "session", "process", "seq", "kind"}
OUTER = "KVCacheManager.allocate_slots"
COMPUTED = "KVCacheCoordinator.allocate_new_computed_blocks"
ORDINARY = "KVCacheCoordinator.allocate_new_blocks"
LOCAL = "SingleTypeKVCacheManager.add_local_computed_blocks"
EXTERNAL = "SingleTypeKVCacheManager.allocate_external_computed_blocks"
NEW = "SingleTypeKVCacheManager.allocate_new_blocks"
_LOWER = {LOCAL, EXTERNAL, NEW}
_COORDINATOR = {COMPUTED, ORDINARY}
_OBJECTS = {"scheduler", "manager", "coordinator", "group_manager", "pool"}
_ARGUMENTS = {"num_new_tokens", "num_new_computed_tokens", "new_computed_blocks",
              "num_lookahead_tokens", "num_external_computed_tokens", "delay_cache_blocks",
              "num_encoder_tokens", "full_sequence_must_fit", "reserved_blocks",
              "has_scheduled_reqs", "skip_zeroing_group_ids", "computed_blocks_native_empty",
              "retained_hit_group_ids"}
_REQUEST_POLICY = {"n": 1, "resumable": False, "streaming": False, "preemptions": 0,
                   "replay_start": 0, "shared_prefix_boundary": 0, "in_flight_tokens": 0,
                   "pooling_params_absent": True, "lora_request_absent": True}
_SUMMARY = ("outer_calls", "phase_calls", "pool_returns", "block_facts",
            "empty_pool_returns", "completed_grants")


class EvidenceError(ValueError):
    pass


def _require(value, reason):
    if not value:
        raise EvidenceError(reason)


def _shape(value, fields, reason="recipient record fields are incomplete or unsupported"):
    _require(type(value) is dict and set(value) == set(fields), reason)


def _integer(value, low=0, high=2**63 - 1):
    return type(value) is int and low <= value <= high


def _claim_token(roles, token, role):
    """Keep the shared registry's object and generated-token roles disjoint."""
    _require(mapping._id(token), "identity token is invalid")
    _require(roles.get(token, role) == role, "identity token changes its recorded role or namespace")
    roles[token] = role


def _same(value, expected):
    """JSON equality with strict scalar types, including nested booleans."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return value.keys() == expected.keys() and all(_same(value[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected))
    return value == expected


def adapter_contract(adapter=None):
    """Return detached installed source metadata. No engine is imported."""
    value = json.loads(files("sdcguard").joinpath("case_data/allocation-recipient-pins.json").read_text())
    _require(adapter is None or type(adapter) is str, "unsupported allocation-recipient adapter")
    try:
        result = value["adapters"][DEFAULT_ADAPTER if adapter is None else adapter]
    except (KeyError, TypeError):
        raise EvidenceError("unsupported allocation-recipient adapter") from None
    return deepcopy(result)


def source_pins(adapter=None):
    return {row["path"]: row["sha256"] for row in adapter_contract(adapter)["files"]}


def _bounded(value):
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _require(nodes <= MAX_RECORD_NODES and depth <= 32, "recipient record structure exceeds bounds")
        kind = type(item)
        if kind is dict:
            _require(len(item) <= MAX_RECORD_NODES and len(pending) + 2 * len(item) <= MAX_RECORD_NODES,
                     "recipient record structure exceeds bounds")
            for key, child in item.items():
                _require(type(key) is str, "recipient record keys must be strings")
                pending.extend(((key, depth + 1), (child, depth + 1)))
        elif kind is list:
            _require(len(item) <= MAX_RECORD_NODES and len(pending) + len(item) <= MAX_RECORD_NODES,
                     "recipient record structure exceeds bounds")
            pending.extend((child, depth + 1) for child in item)
        elif kind is str:
            _require(len(item) <= MAX_RECORD_BYTES
                     and not any(0xD800 <= ord(c) <= 0xDFFF for c in item), "recipient string is unsupported")
        elif kind is int:
            _require(item.bit_length() <= 63, "recipient integer exceeds bounds")
        else:
            _require(item is None or kind is bool, "recipient value is not supported JSON")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    _require(len(raw) <= MAX_RECORD_BYTES, "recipient record exceeds byte bound")
    return raw, nodes


def measure(records_without_footer):
    """Canonical nonfooter accounting, including header and dictionary keys."""
    _require(type(records_without_footer) is list and len(records_without_footer) < MAX_RECORDS,
             "recipient body count exceeds bounds")
    size = nodes = 0
    for record in records_without_footer:
        raw, count = _bounded(record)
        size += len(raw)
        nodes += count
        _require(size <= MAX_TOTAL_BYTES and nodes <= MAX_TOTAL_NODES,
                 "recipient body exceeds aggregate bounds")
    return {"body_bytes": size, "body_nodes": nodes}


def _base():
    return {"schema": "sdcguard.allocation-recipient-report.v1", "property": PROPERTY,
            "verdict": "unknown", "status": "unknown", "scope_verdict": "inconclusive",
            "metadata_completeness": "unknown", "receipts": [], "missing": [],
            "session": None, "mapping_context": None,
            "summary": {key: None for key in _SUMMARY},
            "ownership_verdict": "unknown", "still_live_exclusive_grant": "not established",
            "local_attention_permission": "not established", "read_rights_verdict": "unknown",
            "lease_join_verdict": "unknown", "physical_read_verdict": "unknown",
            "content_provenance": "unknown", "kv_load_admission": "unknown",
            "tenant_authorization": "unknown", "runtime_isolation": "not established",
            "removal_and_retention_coverage": "not established", "authenticity": "not established",
            "limits": ["Only completed positive local capacity-recipient observations are checked.",
                       "Allocation ordinals are observations, not native lease nonces or byte generations.",
                       "Missing grants do not establish foreign, unowned or forbidden memory.",
                       "Settled recorder closure does not establish process shutdown or physical GPU reads."]}


class _Mapping:
    def __init__(self, captured):
        _require(len(captured["payload"]) == 1, "one exact paired mapping session is required")
        self.name, self.raw = next(iter(captured["payload"].items()))
        self.receipt_name, self.receipt_raw = next(iter(captured["receipts"].items()))
        closure = io.parse_json(self.receipt_raw, max_nodes=8192)
        _require(type(closure["journal"].get("late_calls")) is int and closure["journal"]["late_calls"] == 0,
                 "paired mapping contains late or unbounded calls")
        self.records = [io.parse_json(line, max_nodes=mapping.MAX_RECORD_NODES)
                        for line in self.raw.splitlines()]
        header = self.records[0]
        _require(header["adapter"] == "vllm.0.32.uniproc.eager.native.v1",
                 "historical or nonnative mapping context is unsupported")
        trace = mapping._Trace(header)
        handlers = {name: getattr(trace, name) for name in (
            "init_begin", "configs_derived", "worker_init", "runner_ready", "init_return", "scheduler_ready",
            "retire", "mutation_begin", "mutation_end", "acquire", "unsupported_branch", "schedule",
            "worker_entry", "frame_begin", "assignments", "model_entry", "frame_end", "gap")}
        self.begins, self.ends, self.acquisitions = {}, {}, {}
        self.frame_ranges, frames, stack = [], {}, []
        for record in self.records[1:-1]:
            name = record["kind"]
            _require(name in handlers, "mapping body kind is unsupported")
            if name == "mutation_begin":
                item = trace.association(record["association"])
                _require(item["active"] and not item["retired"], "mapping mutation belongs to a retired association")
                _require(record["method"] not in ("BlockPool.unpin_blocks", "SingleTypeKVCacheManager._apply_cow"),
                         "paired mapping mutation is outside the recipient no-unpin/no-CoW support profile")
                _require(record["parent"] == (stack[-1] if stack else None),
                         "mapping mutations lack a strict same-thread parent stack")
                stack.append(record["mutation"])
                self.begins[record["mutation"]] = record
            elif name == "mutation_end":
                _require(stack and stack[-1] == record["mutation"], "mapping mutation order is incomplete")
                stack.pop()
                self.ends[record["mutation"]] = record
            elif name == "acquire":
                self.acquisitions[record["mutation"]] = record
            elif name == "frame_begin":
                frames[record["frame"]] = record["seq"]
            elif name == "frame_end":
                _require(record["frame"] in frames, "mapping frame entry is missing")
                self.frame_ranges.append((frames.pop(record["frame"]), record["seq"]))
            handlers[name](record)
        trace.footer(self.records[-1], len(self.records))
        _require(not stack and not frames and len(trace.inits) == 1
                 and not trace.gaps and all(not item["poison"] for item in trace.inits.values()),
                 "mapping initialization or mutation coverage is incomplete")
        _require(trace.summary["unknown"] == 0 and trace.complete_count == trace.frame_count,
                 "paired mapping observation interval is incomplete")
        self.header, self.trace = header, trace
        self.association, self.item = next(iter(trace.inits.items()))
        # Use every observed object role, including config copies, worker cache,
        # Requests and schedule outputs. Generated call tokens come from the same
        # registry and cannot alias those objects or one another's namespaces.
        self.token_roles = dict(trace.objects)
        for role, tokens in (("mapping_initialization", trace.inits),
                             ("mapping_mutation", trace.used_mutations),
                             ("mapping_frame", trace.frames)):
            for token in tokens:
                _claim_token(self.token_roles, token, role)
        _claim_token(self.token_roles, header["session"], "mapping_session")
        _claim_token(self.token_roles, header["process"], "process")
        ready = next(r for r in self.records if r["kind"] == "scheduler_ready")
        _require(ready["classes"]["scheduler"] == "vllm.v1.core.sched.scheduler.Scheduler",
                 "actual synchronous Scheduler class is required")

    def binding(self):
        return {"journal": {"name": self.name, "bytes": len(self.raw),
                            "sha256": hashlib.sha256(self.raw).hexdigest()},
                "receipt": {"name": self.receipt_name, "bytes": len(self.receipt_raw),
                            "sha256": hashlib.sha256(self.receipt_raw).hexdigest()}}


class _Recipient:
    def __init__(self, header, context):
        self.context = context
        _shape(header, _COMMON | {"adapter", "source_sha256", "profile", "dependencies", "mapping_session",
                                  "association", "objects", "policy", "coverage", "callables"})
        _require(header["adapter"] == DEFAULT_ADAPTER, "recipient adapter is unsupported")
        self.contract = adapter_contract(header["adapter"])
        _require(source_pins() == mapping.source_pins(self.contract["mapping_adapter"]),
                 "installed recipient and native mapping source contracts differ")
        _require(_same(header["source_sha256"], source_pins())
                 and all(_same(header[key], context.header[key]) for key in ("source_sha256", "profile", "dependencies")),
                 "exact source, observed model profile or dependencies differ from mapping context")
        _require(header["mapping_session"] == context.header["session"]
                 and header["session"] != header["mapping_session"]
                 and header["process"] == context.header["process"]
                 and header["association"] == context.association,
                 "recipient session, process or initialization association differs")
        expected_objects = {key: context.item["objects"][key] for key in _OBJECTS}
        _require(_same(header["objects"], expected_objects), "recipient native object graph differs")
        _require(_same(header["policy"], self.contract["policy"]), "observed synchronous policy is missing or unsupported")
        methods = set(self.contract["methods"])
        _require(type(header["coverage"]) is list and len(header["coverage"]) == len(methods)
                 and all(type(name) is str for name in header["coverage"]) and set(header["coverage"]) == methods,
                 "recipient callable coverage is incomplete")
        _shape(header["callables"], methods, "recipient callable identities are missing")
        _require(all(mapping._id(token) for token in header["callables"].values())
                 and len(set(header["callables"].values())) == len(methods), "recipient callable identities are invalid")
        self.token_roles = dict(context.token_roles)
        _claim_token(self.token_roles, header["session"], "recipient_session")
        for token in header["callables"].values():
            _claim_token(self.token_roles, token, "recipient_callable")
        self.header, self.objects = header, expected_objects
        self.stack, self.calls, self.used_calls, self.requests = [], {}, set(), {}
        self.used_group_mutations, self.used_acquisitions = set(), set()
        self.last_group_end = -1
        self.counts = {key: 0 for key in _SUMMARY}
        self.receipts = []

    def fields(self, record, fields):
        _shape(record, _COMMON | set(fields))

    def block_refs(self, value):
        _require(type(value) is list and len(value) <= MAX_BLOCKS_PER_RETURN, "block references exceed bounds")
        seen = set()
        for item in value:
            _shape(item, {"block", "incarnation"}, "block reference is malformed")
            _require(_integer(item["block"], 1, self.context.item["dimensions"]["capacity"] - 1)
                     and _integer(item["incarnation"], 1) and item["block"] not in seen,
                     "block is null, repeated or lacks an observed allocation ordinal")
            seen.add(item["block"])
        return value

    def push(self, call, value):
        _require(mapping._id(call) and call not in self.used_calls, "recipient call identity is missing or repeated")
        _claim_token(self.token_roles, call, "recipient_call")
        _require(len(self.stack) < MAX_NESTING, "recipient nesting exceeds bound")
        self.used_calls.add(call)
        self.calls[call] = value
        self.stack.append(call)

    def allocation_begin(self, r):
        self.fields(r, {"call", "request", "request_incarnation", "request_policy", "request_state", "arguments"})
        _require(not self.stack, "recipient outer calls overlap")
        name, incarnation = r["request"], r["request_incarnation"]
        _require(type(name) is str and 0 < len(name) <= 128 and all(32 <= ord(c) <= 126 for c in name)
                 and mapping._id(incarnation),
                 "actual Request lifetime or name is unsupported")
        _claim_token(self.token_roles, incarnation, "request")
        _require(self.requests.get(name, incarnation) == incarnation, "request lifetime reuse is outside this contract")
        _require(all(previous_name == name or previous_incarnation != incarnation
                     for previous_name, previous_incarnation in self.requests.items()), "Request lifetime changes its name")
        self.requests[name] = incarnation
        recorded = self.context.trace.request_ids.get((self.context.association, name))
        _require(recorded is None or recorded == incarnation, "Request lifetime differs from native mapping dispatch")
        _require(self.context.trace.request_names.get(incarnation, name) == name,
                 "Request lifetime changes its name recorded by mapping")
        _require(_same(r["request_policy"], _REQUEST_POLICY), "actual Request policy is missing or unsupported")
        state, args = r["request_state"], r["arguments"]
        _shape(state, {"num_computed_tokens", "num_tokens", "num_prompt_tokens", "max_model_len"})
        _require(all(_integer(value, 0, mapping.MAX_CONTEXT) for value in state.values())
                 and state["max_model_len"] > 0
                 and state["num_computed_tokens"] <= state["num_tokens"]
                 and state["num_prompt_tokens"] <= state["num_tokens"], "Request source counters are invalid")
        _shape(args, _ARGUMENTS, "effective allocate_slots arguments are incomplete")
        for key in ("num_new_tokens", "num_new_computed_tokens", "num_lookahead_tokens",
                    "num_external_computed_tokens", "num_encoder_tokens", "reserved_blocks"):
            _require(_integer(args[key], 0, mapping.MAX_CONTEXT), "effective allocation scalar is invalid")
        _require(all(type(args[key]) is bool for key in ("delay_cache_blocks", "full_sequence_must_fit",
                                                       "has_scheduled_reqs", "computed_blocks_native_empty")),
                 "effective allocation boolean is invalid")
        _require(args["num_new_computed_tokens"] == args["num_lookahead_tokens"] == args["num_encoder_tokens"] == 0,
                 "prefix hits, lookahead or encoder allocation are unsupported")
        _require(args["new_computed_blocks"] is None or _same(args["new_computed_blocks"], [[]]),
                 "effective computed blocks must be None or a concrete empty group")
        _require(args["new_computed_blocks"] is not None or args["computed_blocks_native_empty"] is True,
                 "None does not resolve to the observed native-empty tuple")
        _require(_same(args["retained_hit_group_ids"], []) and
                 (_same(args["skip_zeroing_group_ids"], []) or _same(args["skip_zeroing_group_ids"], [0])),
                 "retained hits or skip-zeroing groups are unsupported")
        self.push(r["call"], {"kind": "outer", "record": r, "children": [], "pools": []})
        self.counts["outer_calls"] += 1
        _require(self.counts["outer_calls"] <= MAX_OUTER_CALLS, "outer call count exceeds bound")

    def expected_arguments(self, method, outer, actual):
        state, args = outer["record"]["request_state"], outer["record"]["arguments"]
        local = state["num_computed_tokens"]
        external = args["num_external_computed_tokens"]
        computed = min(local + external, state["max_model_len"])
        main = computed + args["num_new_tokens"]
        total = min(main, state["max_model_len"])
        if method == COMPUTED:
            _require(type(actual) is dict and type(actual.get("already_tracked")) is bool,
                     "computed-coordinator branch observation is missing")
            return {"new_computed_blocks": [[]], "num_local_computed_tokens": local,
                    "num_external_computed_tokens": external,
                    "skip_zeroing_group_ids": args["skip_zeroing_group_ids"],
                    "already_tracked": actual["already_tracked"]}
        if method == ORDINARY:
            return {"num_tokens": total, "num_tokens_main_model": main, "num_encoder_tokens": 0}
        if method == NEW:
            return {"num_tokens": total, "num_tokens_main_model": main}
        value = {"num_local_computed_tokens": local, "num_external_computed_tokens": external}
        if method == LOCAL:
            value["new_computed_blocks"] = []
        else:
            value["record_for_zeroing"] = 0 not in args["skip_zeroing_group_ids"]
        return value

    def phase_begin(self, r):
        self.fields(r, {"call", "outer", "parent", "method", "receiver", "request", "arguments", "mapping_mutation"})
        _require(self.stack and r["parent"] == self.stack[-1] and r["outer"] == self.stack[0],
                 "recipient phase parent is missing or not the immediate call")
        outer, parent = self.calls[r["outer"]], self.calls[r["parent"]]
        method = r["method"]
        _require(type(method) is str and method in _LOWER | _COORDINATOR, "recipient phase method is unsupported")
        _require((method in _COORDINATOR and parent["kind"] == "outer")
                 or (method in _LOWER and parent.get("method") in _COORDINATOR), "recipient phase nesting differs from source")
        _require(r["request"] == outer["record"]["request"]
                 and r["receiver"] == self.objects[self.contract["methods"][method]],
                 "lower recipient or native receiver differs from outer Request graph")
        _require(_same(r["arguments"], self.expected_arguments(method, outer, r["arguments"])),
                 "effective lower arguments differ from the source call")
        mutation = r["mapping_mutation"]
        if method in _COORDINATOR:
            _require(mutation is None, "coordinator phase invents a mapping mutation")
        else:
            begin, end = self.context.begins.get(mutation), self.context.ends.get(mutation)
            _require(mapping._id(mutation) and mutation not in self.used_group_mutations
                     and begin is not None and end is not None
                     and begin["association"] == self.context.association
                     and begin["method"] == method and begin["receiver"] == r["receiver"]
                     and begin["parent"] is None and end["success"] is True,
                     "lower phase has no exact successful paired mapping mutation")
            _require(begin["seq"] > self.last_group_end, "lower phase order differs from paired mapping")
            _require(not any(start < end["seq"] and stop > begin["seq"] for start, stop in self.context.frame_ranges),
                     "allocation phase overlaps a model-entry observation interval")
            self.last_group_end = end["seq"]
            self.used_group_mutations.add(mutation)
        parent["children"].append(r["call"])
        self.push(r["call"], {"kind": "phase", "method": method, "record": r, "children": [], "pools": []})
        self.counts["phase_calls"] += 1
        _require(self.counts["phase_calls"] <= MAX_PHASE_CALLS, "phase call count exceeds bound")

    def pool_return(self, r):
        self.fields(r, {"outer", "phase", "mutation", "pool", "requested_blocks", "blocks"})
        _require(self.stack and r["outer"] == self.stack[0] and r["phase"] == self.stack[-1],
                 "pool return has no exact active lower phase")
        phase = self.calls[r["phase"]]
        _require(phase.get("method") in (NEW, EXTERNAL) and not phase["pools"],
                 "pool return is outside the ordinary one-acquisition source path")
        _require(r["pool"] == self.objects["pool"]
                 and _integer(r["requested_blocks"], 0, MAX_BLOCKS_PER_RETURN), "pool identity or requested count differs")
        self.block_refs(r["blocks"])
        _require(len(r["blocks"]) == r["requested_blocks"], "pool requested and returned block counts differ")
        _require(phase["method"] != NEW or bool(r["blocks"]), "ordinary no-demand path cannot invoke an empty pool call")
        mutation = r["mutation"]
        begin, acquire, end = (group.get(mutation) for group in
                               (self.context.begins, self.context.acquisitions, self.context.ends))
        _require(mapping._id(mutation) and mutation not in self.used_acquisitions
                 and begin is not None and acquire is not None and end is not None
                 and begin["method"] == "BlockPool.get_new_blocks"
                 and begin["association"] == self.context.association and begin["receiver"] == r["pool"]
                 and begin["parent"] == phase["record"]["mapping_mutation"] and end["success"] is True
                 and _same(acquire["blocks"], r["blocks"]), "pool return lacks its exact paired acquisition and incarnation")
        parent_begin = self.context.begins[begin["parent"]]
        parent_end = self.context.ends[begin["parent"]]
        _require(parent_begin["seq"] < begin["seq"] < acquire["seq"] < end["seq"] < parent_end["seq"],
                 "paired pool return is outside its lower mutation interval")
        self.used_acquisitions.add(mutation)
        phase["pools"].append(r)
        self.calls[r["outer"]]["pools"].append(r)
        self.counts["pool_returns"] += 1
        self.counts["block_facts"] += len(r["blocks"])
        self.counts["empty_pool_returns"] += not bool(r["blocks"])
        _require(self.counts["block_facts"] <= MAX_TOTAL_BLOCK_FACTS, "total pool block facts exceed bound")

    def phase_end(self, r):
        self.fields(r, {"call", "outer", "outcome", "return_type", "returned", "exception_type"})
        _require(self.stack and r["call"] == self.stack[-1] and r["outer"] == self.stack[0],
                 "phase closes without its exact current parent stack")
        phase = self.calls[r["call"]]
        _require(phase["kind"] == "phase" and r["outcome"] == "returned" and r["exception_type"] is None,
                 "phase has an exceptional or incomplete completion; no rollback is inferred")
        method = phase["method"]
        children = [self.calls[token] for token in phase["children"]]
        if method == COMPUTED:
            args = phase["record"]["arguments"]
            expected = [] if args["already_tracked"] else [LOCAL] + ([EXTERNAL] if args["num_external_computed_tokens"] > 0 else [])
            _require([child["method"] for child in children] == expected, "computed-coordinator source branch is incomplete")
            return_type, returned = "NoneType", None
        elif method == ORDINARY:
            _require([child["method"] for child in children] == [NEW], "ordinary coordinator has no exact group result")
            return_type, returned = "tuple", [children[0]["returned"]]
        elif method == NEW:
            _require(not children, "ordinary lower phase contains an unsupported nested phase")
            return_type = "list"
            returned = phase["pools"][0]["blocks"] if phase["pools"] else []
        else:
            _require(not children and (len(phase["pools"]) == 1 if method == EXTERNAL else not phase["pools"]),
                     "local or external lower source path is incomplete")
            if method == EXTERNAL:
                # Successful first-time empty local adoption starts with an
                # empty FullAttention row. There is no skipped/prefix/CoW path.
                args = phase["record"]["arguments"]
                size = self.context.item["dimensions"]["block_size"]
                expected = (args["num_local_computed_tokens"] + args["num_external_computed_tokens"] + size - 1) // size
                _require(expected > 0 and phase["pools"][0]["requested_blocks"] == expected,
                         "external first-time capacity demand differs from the pinned source")
            return_type, returned = "NoneType", None
        _require(r["return_type"] == return_type and _same(r["returned"], returned),
                 "phase native return representation or concrete object order differs")
        phase["returned"] = deepcopy(returned)
        self.stack.pop()

    def allocation_end(self, r):
        self.fields(r, {"call", "outcome", "return_type", "returned", "exception_type",
                        "request_revalidated", "graph_revalidated"})
        _require(len(self.stack) == 1 and r["call"] == self.stack[-1], "outer completion precedes a required phase closure")
        outer = self.calls[r["call"]]
        _require(outer["kind"] == "outer" and r["exception_type"] is None
                 and r["request_revalidated"] is True and r["graph_revalidated"] is True,
                 "outer completion or native identity revalidation is incomplete")
        children = [self.calls[token] for token in outer["children"]]
        args = outer["record"]["arguments"]
        _require(args["num_new_tokens"] > 0 or args["num_external_computed_tokens"] > 0
                 or args["num_new_computed_tokens"] > 0,
                 "all-zero source allocation cannot complete normally")
        if r["outcome"] == "not_allocated":
            _require(not children and r["return_type"] == "NoneType" and r["returned"] is None,
                     "normal None is not a completed allocation and cannot hide completed phases")
        else:
            _require(r["outcome"] == "returned" and r["return_type"] == "KVCacheBlocks",
                     "outer call has an exceptional or unsupported native return")
            expected = ([] if args["computed_blocks_native_empty"] and args["num_external_computed_tokens"] == 0
                        else [COMPUTED]) + [ORDINARY]
            _require([child["method"] for child in children] == expected
                     and _same(r["returned"], children[-1]["returned"]),
                     "outer ordinary return or complete phase order differs from source")
            for pool in outer["pools"]:
                if not pool["blocks"]:
                    continue
                phase = self.calls[pool["phase"]]
                self.receipts.append({"verdict": "observed", "call": r["call"],
                    "request": outer["record"]["request"],
                    "request_incarnation": outer["record"]["request_incarnation"],
                    "phase": pool["phase"], "method": phase["method"], "pool": pool["pool"],
                    "blocks": deepcopy(pool["blocks"]), "mapping_acquisition": pool["mutation"],
                    "basis": "completed_local_capacity_allocation", "still_live_exclusive_grant": "not established"})
                self.counts["completed_grants"] += 1
        self.stack.pop()

    def gap(self, r):
        self.fields(r, {"call", "reason"})
        _require(False, "recipient capture contains an explicit gap")

    def footer(self, r, records):
        self.fields(r, set(_SUMMARY) | {"records", "errors", "dropped", "healthy", "closed", "open_calls",
                                       "source_stable", "coverage_stable", "body_bytes", "body_nodes", "mapping_binding"})
        _require(not self.stack and type(r["open_calls"]) is int and r["open_calls"] == 0,
                 "recipient calls are unclosed")
        _require(all(type(r[key]) is int and r[key] == value for key, value in self.counts.items())
                 and type(r["records"]) is int and r["records"] == len(records), "recipient footer counts differ")
        _require(all(type(r[key]) is int and r[key] == 0 for key in ("errors", "dropped"))
                 and all(r[key] is True for key in ("healthy", "closed", "source_stable", "coverage_stable")),
                 "recipient evidence is incomplete, unhealthy or changed")
        _require(all(type(r[key]) is int and r[key] == value for key, value in measure(records[:-1]).items()),
                 "recipient measured body bytes or nodes differ")
        _require(_same(r["mapping_binding"], self.context.binding()), "exact paired mapping raw-byte binding differs")
        _require(self.used_acquisitions == set(self.context.acquisitions), "paired mapping acquisition lacks recipient attribution")
        expected_groups = {token for token, begin in self.context.begins.items() if begin["method"] in _LOWER}
        _require(self.used_group_mutations == expected_groups, "paired lower allocation call lacks recipient phase evidence")


def evaluate(records, mapping_payload, mapping_receipts):
    """Recheck actual settled byte pairs first, then the separate receipt chain."""
    report = _base()
    try:
        captured = mapping_closure.validate_captured(mapping_payload, mapping_receipts)
        context = _Mapping(captured)
        _require(type(records) is list and 2 <= len(records) <= MAX_RECORDS,
                 "recipient closed record list is missing or exceeds bounds")
        frozen, total_bytes, total_nodes = [], 0, 0
        for index, record in enumerate(records):
            raw, nodes = _bounded(record)
            total_bytes += len(raw)
            total_nodes += nodes
            _require(total_bytes <= MAX_TOTAL_BYTES and total_nodes <= MAX_TOTAL_NODES,
                     "recipient session exceeds aggregate bounds")
            record = json.loads(raw)
            _require(type(record) is dict and type(record.get("schema")) is int and record["schema"] == SCHEMA
                     and mapping._id(record.get("session")) and mapping._id(record.get("process"))
                     and type(record.get("seq")) is int and record["seq"] == index
                     and type(record.get("kind")) is str, "recipient record sequence or schema is malformed")
            frozen.append(record)
        header, footer = frozen[0], frozen[-1]
        _require(header["kind"] == "header" and footer["kind"] == "footer", "recipient header or final closure is missing")
        trace = _Recipient(header, context)
        for record in frozen[1:]:
            _require(record["session"] == header["session"] and record["process"] == header["process"],
                     "recipient records mix sessions or processes")
            if record is footer:
                trace.footer(record, frozen)
            else:
                _require(record["kind"] in ("allocation_begin", "phase_begin", "pool_return", "phase_end", "allocation_end", "gap"),
                         "recipient body kind is unsupported")
                getattr(trace, record["kind"])(record)
        report.update(session=header["session"], metadata_completeness="complete",
                      status="observed" if trace.receipts else "not_exercised",
                      verdict="observed" if trace.receipts else "unknown", receipts=trace.receipts,
                      summary=dict(trace.counts), mapping_context={"session": context.header["session"],
                          "process": context.header["process"], "association": context.association,
                          "binding": context.binding(), "semantic_replay": "complete",
                          "bookkeeping": "locally consistent"})
        if not trace.receipts:
            report["missing"] = ["no completed positive local capacity-recipient receipt was exercised"]
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError, OSError) as exc:
        report = _base()
        report["missing"] = [str(exc)[:512]]
    return report
