"""An ordinary finite prompt mix. Owner: Hari Krishna Kancharla.

Tokenization is supplied by the caller. Importing this file performs no model,
SDK, HTTP or device work.
"""
from __future__ import annotations

import re

from .common import digest, encoded

REVISION = "29d2f4e03740acb14439ce22d24b06d8095d17d3"
MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PREFIX_TARGETS = (128, 512, 1408)
CONCURRENCIES = (1, 4, 16)
ARMS = (("observer-on", True),)
MAX_TOKENS = 16
BLOCK_SIZE = 16
LATER_REQUESTS = 16
WORKLOAD = "ordinary grouped plain completions"

PARAGRAPHS = (
    "A public library is changing its opening hours. The morning team sorts returned books, "
    "checks room bookings and updates the notice board. The afternoon team runs workshops "
    "and helps visitors use the catalogue. Summarize the practical changes for a new volunteer.",
    "The planning team keeps a small list of tasks. Design has finished the first sketch, "
    "engineering is reviewing the interface, and operations is checking the delivery dates. "
    "Write a short status note that separates completed work from the next action.",
    "A garden group records observations each week. The beds near the wall receive less rain, "
    "the raised beds dry out earlier and the shaded path stays cool. List two sensible questions "
    "the group could answer with another month of observations.",
    "An example service accepts a document and returns a concise outline. A request has a title, "
    "a language and a list of paragraphs. An empty list should produce a clear explanation. "
    "Describe a small input validation routine without using an external dependency.",
    "A museum guide is preparing a walking route. The entrance is beside the ticket desk, "
    "the sculpture room is upstairs and the photography room closes early. Explain a route "
    "that gives a visitor time to see the temporary exhibition first.",
    "A workshop ships labelled sample boxes. Each box has a catalogue number, a packing date "
    "and a count of parts. The daily note lists missing labels separately from late deliveries. "
    "Extract the fields an operator should check before the boxes leave.",
    "A study group is comparing two summaries of an article. One summary keeps the main claim "
    "but leaves out a limitation; the other includes the limitation but loses a concrete example. "
    "Suggest a short revision that preserves the claim and the limitation.",
)


def _identity(value, pattern, name):
    if type(value) is not str or re.fullmatch(pattern, value) is None:
        raise ValueError(name + " has an unsupported shape")
    return value


def _tokens(encode, text):
    values = encode(text, add_special_tokens=False)
    if (type(values) is not list or not 1 <= len(values) <= 2000
            or any(type(token) is not int or not 0 <= token < 1000000 for token in values)):
        raise ValueError("the pinned tokenizer did not return bounded ordinary token IDs")
    return values


def _common_prefix(values):
    return next((i for i, column in enumerate(zip(*values)) if len(set(column)) != 1),
                min(map(len, values)))


def prompt_group(encode, group, target):
    """Tokenize complete plain-text notes and endings without templates or slicing."""
    prefix = "Group " + str(group) + " library notes. Read the following ordinary working notes."
    for section in range(40):
        prefix += "\n\n" + PARAGRAPHS[(group + section) % len(PARAGRAPHS)]
        if len(_tokens(encode, prefix)) >= target:
            break
    else:
        raise ValueError("ordinary prefix did not reach its target")
    values = []
    for ending in range(LATER_REQUESTS + 1):
        text = prefix + "\n\nQuestion " + str(ending + 1) + ": Give a short practical next action for this working group.\nAnswer:"
        values.append(_tokens(encode, text))
    if _common_prefix(values) < 4 * BLOCK_SIZE or len({tuple(v) for v in values}) != len(values):
        raise ValueError("ordinary group lacks a shared full-block prefix or distinct endings")
    return values


def build_plan(encode, *, run_id, packet_sha256, engine_sessions, revision=REVISION, gpu_blocks=None):
    """Build an independent client inventory; cache hits are measured later."""
    from ..checksum_profiles import get
    get(revision)
    if gpu_blocks is not None and (type(gpu_blocks) is not int or gpu_blocks != 8192):
        raise ValueError("only the 8192-block paired capacity is supported")
    _identity(run_id, r"[0-9a-f]{32}", "run_id")
    _identity(packet_sha256, r"[0-9a-f]{64}", "packet_sha256")
    if type(engine_sessions) is not dict or set(engine_sessions) != {x[0] for x in ARMS}:
        raise ValueError("one independently generated session is required per observer arm")
    for value in engine_sessions.values():
        _identity(value, r"[0-9a-f]{32}", "engine_session_id")
    arms, payloads = [], {}
    for arm_index, (arm_id, enabled) in enumerate(ARMS):
        requests = []
        for phase_index, concurrency in enumerate(CONCURRENCIES):
            phase_id = "c" + str(concurrency)
            for target_index, target in enumerate(PREFIX_TARGETS):
                group = phase_index * len(PREFIX_TARGETS) + target_index + 1
                group_id = "group-" + str(group)
                values = prompt_group(encode, group, target)
                common_tokens = _common_prefix(values)
                for position, tokens in enumerate(values):
                    slot = len(requests)
                    base = "sgc-" + run_id + "-" + str(arm_index) + "-" + str(concurrency) + "-" + str(slot)
                    client_id, prefill_id = base + "-d", base + "-p"
                    requests.append(dict(slot=slot, phase_id=phase_id,
                        kind="warmup" if position == 0 else "measured",
                        concurrency=1 if position == 0 else concurrency,
                        cache_group=group_id, group_position=position,
                        shared_prefix_tokens=common_tokens, prefix_target_tokens=target,
                        prefill_request_id=prefill_id, client_request_id=client_id,
                        prompt_tokens=len(tokens), prompt_sha256=digest(encoded(tokens)), max_tokens=MAX_TOKENS))
                    payloads[client_id] = list(tokens)
        arms.append(dict(arm_id=arm_id, engine_session_id=engine_sessions[arm_id],
            observer_enabled=enabled, requests=requests))
    plan = dict(schema="sdcguard.checksum-request-plan.v1", run_id=run_id,
        packet_sha256=packet_sha256, source_revision=revision, model=MODEL,
        prefix_caching=True, workload=WORKLOAD, chat_template=False,
        cache_state_basis="recorded per-request cache events, never group position",
        phases=[dict(phase_id="c" + str(value), concurrency=value) for value in CONCURRENCIES], arms=arms)
    if gpu_blocks is not None:
        plan["gpu_blocks_override"] = gpu_blocks
    validate_payloads(plan, payloads)
    return plan, payloads


def summarize_recipe():
    count = len(PREFIX_TARGETS) * len(CONCURRENCIES) * (1 + LATER_REQUESTS)
    return dict(owner="Hari Krishna Kancharla", source_revision=REVISION, model=MODEL,
        model_revision=MODEL_REVISION, prefix_targets=list(PREFIX_TARGETS), concurrency_levels=list(CONCURRENCIES),
        groups_per_phase=len(PREFIX_TARGETS), first_requests_per_group=1, later_requests_per_group=LATER_REQUESTS,
        phases=len(CONCURRENCIES), logical_requests=count, planned_http_stages=2 * count,
        max_decode_tokens=MAX_TOKENS, observer_arms=[arm for arm, _ in ARMS],
        ordinary_traffic_only=True, retry_count=0, chat_template=False,
        token_count_basis="complete plain text tokenized by the pinned tokenizer",
        first_request="sent alone and completed before later requests; cold status requires recorded evidence",
        performance_scope="descriptive only; no observer-off overhead comparison",
        runtime_validation="pending owner execution")


def validate_payloads(plan, payloads):
    from ..checksum_client import _planned
    rows, _, _ = _planned(plan)
    if "gpu_blocks_override" in plan:
        # Bound the entire finite plan without assuming any shared cache hit.
        # Reserve one extra block per request and one pool sentinel block.
        bound = 1 + sum((row['prompt_tokens'] + row['max_tokens'] + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
                        for row in rows)
        if bound > plan['gpu_blocks_override']:
            raise ValueError('ordinary plan exceeds the pinned no-sharing cache capacity')
    if (type(payloads) is not dict or set(payloads) != {r['client_request_id'] for r in rows}
            or plan.get('prefix_caching') is not True or plan.get('model') != MODEL
            or plan.get('workload') != WORKLOAD or plan.get('chat_template') is not False
            or [(a['arm_id'], a['observer_enabled']) for a in plan['arms']] != list(ARMS)
            or plan['phases'] != [dict(phase_id='c'+str(c), concurrency=c) for c in CONCURRENCIES]
            or len(rows) != len(PREFIX_TARGETS) * len(CONCURRENCIES) * (1 + LATER_REQUESTS)):
        raise ValueError('ordinary client payload inventory or profile differs')
    first_blocks = set()
    for group_index in range(len(PREFIX_TARGETS) * len(CONCURRENCIES)):
        group = rows[group_index * (1 + LATER_REQUESTS):(group_index + 1) * (1 + LATER_REQUESTS)]
        values = []
        for position, row in enumerate(group):
            tokens = payloads[row['client_request_id']]
            if (type(tokens) is not list or len(tokens) != row['prompt_tokens'] or not 1 <= len(tokens) <= 2000
                    or any(type(t) is not int or not 0 <= t < 1000000 for t in tokens)
                    or digest(encoded(tokens)) != row['prompt_sha256'] or row['max_tokens'] != MAX_TOKENS
                    or row.get('cache_group') != 'group-'+str(group_index+1)
                    or type(row.get('group_position')) is not int or row['group_position'] != position
                    or row['phase_id'] != 'c'+str(CONCURRENCIES[group_index // len(PREFIX_TARGETS)])
                    or row['kind'] != ('warmup' if position == 0 else 'measured')):
                raise ValueError('ordinary client payload or group differs from its recorded prompt')
            values.append(tokens)
        common = _common_prefix(values)
        if (common < 4 * BLOCK_SIZE or len({tuple(v) for v in values}) != len(values)
                or any(type(r.get('shared_prefix_tokens')) is not int or r['shared_prefix_tokens'] != common for r in group)
                or any(type(r.get('prefix_target_tokens')) is not int or r['prefix_target_tokens'] != PREFIX_TARGETS[group_index % len(PREFIX_TARGETS)] for r in group)):
            raise ValueError('ordinary group shared prefix or endings differ')
        first = tuple(values[0][:BLOCK_SIZE])
        if first in first_blocks:
            raise ValueError('ordinary groups share a first block; cold grouping is unverified')
        first_blocks.add(first)
    if not any(row['prompt_tokens'] % BLOCK_SIZE for row in rows):
        raise ValueError('ordinary tokenizer workload lacks a partial final block')
