"""Join native cache accounting to both client stages. Owner: Hari Krishna Kancharla."""
from . import checksum_client as client, prefix_cache_observer as observer
from .kv_checksum_observer import CACHED_PREFIX_REASON


def capture_facts(capture, plan, arm):
    context = {k: plan[k] if k in ("run_id", "packet_sha256") else arm[k] for k in client.CONTEXT_KEYS}
    result = {}
    for role in ("prefill", "decode"):
        envelope = capture.get("prefill_capture") if role == "prefill" else capture
        issues, records, bindings = [], [], []
        envelope = envelope if type(envelope) is dict else {}
        startup = envelope.get("startup", {})
        startup = startup if type(startup) is dict else {}
        if (type(startup) is not dict or startup.get("role") != role
                or any(startup.get(k) != v for k, v in context.items())
                or startup.get("prefix_caching") is not True or startup.get("checksum_enabled") is not True
                or startup.get("source_revision") != plan["source_revision"]
                or not client.profiles.startup_supported(startup, plan["source_revision"])):
            issues.append("native_cache_startup_unverified:" + role)
        snapshots = envelope.get("cache_snapshots")
        if type(snapshots) is not list or len(snapshots) != 1:
            issues.append("native_cache_snapshot_missing_or_ambiguous:" + role)
        else:
            snapshot = snapshots[0]
            records, errors = observer.evaluate_snapshot(snapshot, context, role, plan["source_revision"])
            issues.extend(errors)
            if type(snapshot) is not dict or snapshot.get("process_id") != startup.get("process_id"):
                issues.append("native_cache_process_binding_unverified:" + role)
        binding_snapshots = envelope.get("binding_snapshots")
        if type(binding_snapshots) is not list or len(binding_snapshots) > client.MAX_SNAPSHOTS:
            issues.append("cache_request_binding_snapshots_missing:" + role)
        else:
            active = 0
            for snapshot in binding_snapshots:
                rows, errors, domain = client._bindings(snapshot, context, plan["source_revision"])
                issues.extend(errors)
                active += bool(rows)
                bindings.extend(rows)
                if snapshots and type(snapshots) is list and len(snapshots) == 1 and type(snapshots[0]) is dict:
                    if domain != snapshots[0].get("qualification"):
                        issues.append("native_cache_qualification_disagrees:" + role)
                if len(bindings) > client.MAX_REQUESTS:
                    raise client.io.EvidenceError("native cache bindings exceed their bound")
            if active != 1:
                issues.append("cache_request_binding_ambiguous:" + role)
        result[role] = dict(records=records, bindings=bindings, issues=list(dict.fromkeys(issues)), joined=set())
    return result


def join(facts, raw, row):
    usage = dict(prompt_tokens=row["prompt_tokens"], transferred_tokens_checked=None,
                 decode_cached_tokens_unchecked=None, decode_locally_computed_tokens=None,
                 prefill_cached_tokens=None, prefill_reused_cached_blocks="unknown")
    errors, stats = [], {}
    for role in ("prefill", "decode"):
        fact = facts[role]
        errors.extend(fact["issues"])
        stage = raw.get(role, {}) if type(raw) is dict else {}
        expected = row["prefill_request_id"] if role == "prefill" else row["client_request_id"]
        matches = [b for b in fact["bindings"] if b.get("external_request_id") == "cmpl-" + expected + "-0"]
        if len(matches) != 1:
            errors.append("native_cache_request_join_unverified:" + role)
            continue
        binding = matches[0]
        internal = binding.get("internal_request_id")
        if (sum(b.get("internal_request_id") == internal for b in fact["bindings"]) != 1
                or binding.get("prompt_sha256") != row["prompt_sha256"]
                or binding.get("prompt_tokens") != row["prompt_tokens"]
                or role == "decode" and internal != row.get("internal_request_id")):
            errors.append("native_cache_request_join_unverified:" + role)
            continue
        fact["joined"].add(internal)
        response = stage.get("response_body", {}) if type(stage) is dict else {}
        response = response if type(response) is dict else {}
        counts = response.get("usage", {}) if type(response) is dict else {}
        counts = counts if type(counts) is dict else {}
        value, issues = observer.request_usage(fact["records"], internal, row["prompt_tokens"],
                                               row["prompt_sha256"], counts.get("completion_tokens"))
        errors.extend(reason + ":" + role for reason in issues)
        if not fact["issues"] and not issues and type(value) is dict:
            stats[role] = value
        if role == "prefill":
            descriptor = response.get("kv_transfer_params", {})
            if type(descriptor) is not dict or descriptor.get("remote_request_id") != internal:
                errors.append("producer_cache_transfer_request_binding_unverified")
            if role in stats:
                cached = descriptor.get("remote_prefill_cached_tokens") if type(descriptor) is dict else None
                if (stats[role]["num_external_cached_tokens"] != 0 or type(cached) is not int
                        or cached != stats[role]["num_cached_tokens"]):
                    errors.append("producer_cache_descriptor_counts_unverified")
                    del stats[role]
    if "prefill" in stats:
        count = stats["prefill"]["num_local_cached_tokens"]
        usage.update(prefill_cached_tokens=count, prefill_reused_cached_blocks=count > 0)
    if "decode" in stats:
        value = stats["decode"]
        local, remote = value["num_local_cached_tokens"], value["num_external_cached_tokens"]
        usage.update(decode_cached_tokens_unchecked=local,
                     decode_locally_computed_tokens=value["num_computed_tokens"])
        if local:
            errors.append(CACHED_PREFIX_REASON)
        if value["num_computed_tokens"]:
            errors.append("locally_computed_prompt_not_covered_by_checksum")
        calls = row.get("checksum_calls", [])
        scope = calls[0].get("conditional_scope", {}).get("token_coverage", {}) if len(calls) == 1 else {}
        if (len(calls) == 1 and not calls[0].get("recording_issues")
                and scope.get("prompt_tokens") == row["prompt_tokens"]
                and scope.get("decode_cached_tokens_unchecked") == local
                and scope.get("transferred_tokens_checked") == remote
                and remote == row["prompt_tokens"] - local):
            usage["transferred_tokens_checked"] = remote
        elif not calls and remote == 0:
            usage["transferred_tokens_checked"] = 0
            errors.append("no_transfer_checksum_comparison_recorded")
        else:
            errors.append("native_cache_to_checksum_span_unverified")
    return usage, list(dict.fromkeys(errors))


def unexpected(facts):
    reasons = []
    for role, fact in facts.items():
        observed = {r.get("request_id") for r in fact["records"] if type(r) is dict and type(r.get("request_id")) is str}
        bound = {r.get("internal_request_id") for r in fact["bindings"] if type(r.get("internal_request_id")) is str}
        if observed - fact["joined"] or bound - fact["joined"]:
            reasons.append("unexpected_native_cache_requests:" + role)
    return reasons
