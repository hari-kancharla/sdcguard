"""Static deployment contracts. No engine import, server or request is needed."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

from . import config_capacity, config_inputs, config_protections

RULE_SET = "sdcguard.static-pd-contracts.v1"
SCOPE = ("Configuration and saved startup facts only. A satisfied rule covers its named "
         "contract. This report does not establish request isolation or current runtime state.")


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _rule(rule_id, message, facts=None):
    return dict(id=rule_id, verdict="unknown", message=message, basis="missing evidence",
                facts=facts or {}, sources=[])


_PROTECTION_RULES = {
    "vllm": ("protection.retirement",),
    "sglang": ("protection.retirement", "protection.deferred_release", "protection.failed_send_drain",
               "protection.checksum_configured", "protection.canary_configured", "detection.architecture_coverage"),
    "vllm-ascend": ("protection.retirement",),
    "vllm-lmcache": ("compatibility.pairing",),
}


def _complete_rules(worker, rules):
    expected = _PROTECTION_RULES.get(worker.get("engine"))
    if expected is None:
        expected = tuple(dict.fromkeys(name for names in _PROTECTION_RULES.values() for name in names))
    found = {rule["id"] for rule in rules}
    for name in expected:
        if name in found:
            continue
        rule = _rule(name, "This contract could not be mapped from the supplied source and configuration facts.")
        # The later NIXL/MORI follow-up is a backend-specific policy, not a version cutoff.
        if (name == "protection.failed_send_drain" and config_capacity.supported_pin(worker)
                and worker.get("connector", {}).get("name") == "mooncake"):
            rule.update(verdict="not_applicable", basis="backend-specific follow-up policy",
                        message="The named NIXL/MORI follow-up does not apply to the declared Mooncake backend.",
                        sources=["https://github.com/sgl-project/sglang/pull/45973"])
        rules.append(rule)
    return rules


def _peer_rules(document):
    """Only source-backed peer constraints; a difference is not generic failure."""
    workers = document["workers"]
    identities = {(w.get("engine"), w.get("version"), w.get("build_commit"),
                   json.dumps(w.get("components", {}), sort_keys=True),
                   json.dumps(w.get("connector", {}), sort_keys=True)) for w in workers}
    rule = _rule("deployment.peer_profile", "Mixed or unreviewed peer source/protocol profiles need a separate mapping.")
    if len(identities) == 1 and all(config_capacity.supported_pin(w) for w in workers):
        rule.update(verdict="satisfied", basis="declared uniform reviewed peer profile",
                    message="Peer source and connector declarations name the same reviewed profile; runtime negotiation is unverified.")
    result = [rule]
    if all(w.get("engine") == "sglang" and w.get("version") == "0.5.22"
           and w.get("build_commit") == config_capacity.SGLANG for w in workers):
        source = "https://github.com/sgl-project/sglang/blob/" + config_capacity.SGLANG + "/python/sglang/srt/arg_groups/fields/disagg.py"
        for key, default, choices in (("disaggregation_enable_kv_checksum", False, (False, True)),
                                      ("disaggregation_decode_allocation_policy", "early", ("early", "prefill_complete"))):
            values = []
            for w in workers:
                value = w.get("parameters", {}).get(key)
                if key not in w.get("parameters", {}) and w.get("layout", {}).get("parameters_complete") is True:
                    value = default
                values.append(value)
            mapped = all(config_capacity.supported_pin(w) and
                         w.get("parameters", {}).get("disaggregation_mode") == w.get("role")
                         and w.get("role") in ("prefill", "decode") for w in workers)
            known = mapped and all(type(value) is type(default) and value in choices for value in values)
            pair = _rule("deployment." + key + "_agreement", "Peer setting agreement was not established.", {"values": values})
            pair["sources"] = [source + ("#L72-L77" if type(default) is bool else "#L58-L63")]
            if known:
                pair.update(verdict="satisfied" if len(set(values)) == 1 else "violated",
                            basis="declared peer settings under pinned source",
                            message="Both roles declare the same setting." if len(set(values)) == 1
                            else "Peer settings differ where the pinned source requires agreement.")
            result.append(pair)
    return result


def assess(document, provenance, base):
    """Judge supplied facts under pinned mappings, retaining the evidence basis."""
    workers = []
    model = document.get("model", {})
    context = document.get("advertised_context_tokens")
    for worker in document["workers"]:
        log = provenance.get("logs", {}).get(worker["id"])
        if log is None:
            log = config_inputs.read_startup(worker, base, document)
        text = log.get("text") if "error" not in log else None
        rules = config_capacity.evaluate(worker, context, text)
        if model.get("architecture") not in ("LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"):
            for rule in rules:
                rule.update(verdict="unknown", basis="unreviewed model geometry",
                            message="Capacity mapping is limited to explicitly declared ordinary Llama/Qwen2/Qwen3 full-attention geometry.")
        if "error" in log:
            for rule in rules:
                if rule["id"] == "capacity.startup":
                    rule.update(verdict="unknown", basis="startup evidence unavailable",
                                message=log["error"])
        rules += config_protections.evaluate(worker, model)
        rules = _complete_rules(worker, rules)
        identity_missing = [key for key in ("engine", "version", "build_commit", "role")
                            if not worker.get(key)]
        identity_missing += ["model." + key for key in ("id", "revision", "architecture")
                             if not model.get(key)]
        if identity_missing:
            # No positive or negative engine claim survives an unidentified stack.
            for rule in rules:
                rule.update(verdict="unknown", basis="missing stack identity",
                            message="Missing identity: " + ", ".join(identity_missing))
        if worker.get("engine") == "sglang" and worker.get("parameters", {}).get("disaggregation_mode") != worker.get("role"):
            for rule in rules:
                rule.update(verdict="unknown", basis="PD role not established",
                            message="Explicit SGLang disaggregation_mode must match the supplied worker role.")
        if worker.get("role") not in ("prefill", "decode"):
            for rule in rules:
                rule.update(verdict="unknown", basis="unsupported role",
                            message="Only explicitly identified prefill/decode workers are mapped.")
        workers.append({"id": worker["id"], "role": worker.get("role"),
                        "identity": {k: worker.get(k) for k in
                                     ("engine", "version", "build_commit", "components", "connector")},
                        "config_sha256": config_inputs.binding_digest(worker, document),
                        "startup": {k: v for k, v in log.items() if k != "text"},
                        "rules": rules})
    roles = {worker.get("role") for worker in document["workers"]}
    completeness = _peer_rules(document)
    if not {"prefill", "decode"} <= roles:
        completeness.append(_rule("deployment.roles", "Both prefill and decode facts are required."))
    if context is None:
        completeness.append(_rule("deployment.context", "Advertised endpoint context was not supplied."))
    # No invented P/D equality requirement. Capacity is assessed on each worker.
    for worker in workers:
        startup = next((r for r in worker["rules"] if r["id"] == "capacity.startup"), {})
        for rule in worker["rules"]:
            superseded = (rule["id"] == "capacity.planned" and rule["verdict"] == "unknown"
                          and rule.get("facts", {}).get("state") in ("upper_bound_only", "automatic_pool")
                          and startup.get("verdict") == "satisfied")
            rule["required_for_status"] = not superseded
            if superseded:
                rule["resolved_by"] = "capacity.startup"
    verdicts = [r["verdict"] for w in workers for r in w["rules"] if r["required_for_status"]]
    verdicts += [r["verdict"] for r in completeness]
    status = "not fine" if "violated" in verdicts else "inconclusive" if "unknown" in verdicts else "fine"
    modules = ("config_safety.py", "config_inputs.py", "config_capacity.py", "config_protections.py")
    evaluator = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                 for name in modules}
    report = {"schema": "sdcguard.config-safety-report.v1", "rule_set": RULE_SET,
              "scope": SCOPE, "status": status, "deployment_id": document.get("deployment_id"),
              "advertised_context_tokens": context, "model": model,
              "identity_basis": "operator-supplied identities, not authenticated build attestation",
              "input": document, "provenance": provenance, "evaluator_sha256": evaluator,
              "workers": workers, "deployment_rules": completeness}
    report["report_sha256"] = _sha(report)
    return report


def render(report):
    """Compact printable report; detailed facts remain in the adjacent JSON."""
    esc = lambda x: html.escape(str(x), quote=True)
    rows = []
    identities = []
    grouped = {}
    for w in report["workers"]:
        i = w["identity"]
        identities.append(f'<li><b>{esc(w["id"])} ({esc(w["role"])})</b>: '
                          f'{esc(i.get("engine"))} {esc(i.get("version"))}; '
                          f'build {esc(i.get("build_commit"))}<br>'
                          f'config {esc(w["config_sha256"])}; '
                          f'components {esc(json.dumps(i.get("components"), sort_keys=True))}; '
                          f'connector {esc(json.dumps(i.get("connector"), sort_keys=True))}</li>')
        for r in w["rules"]:
            grouped.setdefault(r["id"], []).append((w["id"], r))
    for rule_id, group in grouped.items():
        verdicts, findings, urls = [], [], []
        for worker_id, r in group:
            verdicts.append(esc(worker_id) + ": " + esc(r["verdict"]))
            facts = r.get("facts", {})
            values = {key: facts[key] for key in ("advertised_context_tokens", "usable_capacity_tokens",
                      "startup_pool_tokens", "configured_pool_upper_bound_tokens", "state", "mode", "coverage")
                      if key in facts}
            note = r["message"]
            if values:
                note += " " + "; ".join(str(k) + "=" + str(v) for k, v in values.items())
            if r.get("resolved_by"):
                note += " Resolved by startup capacity."
            findings.append((worker_id, note))
            for url in r["sources"]:
                if url.startswith("https://") and url not in urls:
                    urls.append(url)
        if len({note for _, note in findings}) == 1:
            detail = esc(findings[0][1])
        else:
            detail = "<br>".join(esc(worker_id) + ": " + esc(note) for worker_id, note in findings)
        links = " ".join(f'<a href="{esc(url)}">source</a>' for url in urls[:2])
        rows.append(f'<tr><td>{esc(rule_id)}</td><td>{"<br>".join(verdicts)}</td><td>{detail} {links}</td></tr>')
    for r in report["deployment_rules"]:
        rows.append(f'<tr><td>{esc(r["id"])}</td><td>{esc(r["verdict"])}</td><td>{esc(r["message"])}</td></tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Isolation configuration report</title><style>'
            'body{font:14px/1.4 system-ui;max-width:1100px;margin:24px auto;padding:0 16px;color:#15232c}'
            'h1{font-size:23px}table{width:100%;border-collapse:collapse}td,th{text-align:left;'
            'vertical-align:top;padding:6px;border-bottom:1px solid #ccd4da}small{display:block;color:#465762}'
            'li{overflow-wrap:anywhere}a{color:#004c89}@media print{body{font-size:9px;margin:0}'
            'h1{font-size:16px}td,th{padding:3px}a{color:inherit}}@page{size:A4;margin:10mm}</style><h1>Isolation configuration report: '
            + esc(report["status"]) + '</h1><p>' + esc(SCOPE) + '</p><p>'
            + esc(report["rule_set"]) + '; advertised context ' + esc(report["advertised_context_tokens"])
            + '</p><p>Model ' + esc(json.dumps(report["model"], sort_keys=True)) + '</p><ul>'
            + ''.join(identities) + '</ul><table><thead><tr><th>Worker / contract</th><th>Verdict</th>'
            '<th>Finding and evidence</th></tr></thead><tbody>' + ''.join(rows)
            + '</tbody></table><p>Identities and log bindings are supplied by the operator. '
            'A matching hash checks consistency, not authenticity. '
            '<a href="report.json">Full facts, source links and hashes</a>. '
            'No server was started.</p></html>')


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard config-safety", description=SCOPE)
    parser.add_argument("config", type=Path, help="normalized deployment JSON; see docs/CONFIG_SAFETY.md")
    parser.add_argument("--out", type=Path, help="new directory for report.json and index.html")
    parser.add_argument("--json", action="store_true", help="print full report JSON")
    args = parser.parse_args(argv)
    try:
        document, provenance = config_inputs.load_deployment(args.config)
        report = assess(document, provenance, args.config.resolve().parent)
        if args.out:
            # Never replace an existing report or write through an output symlink.
            args.out.mkdir(parents=True, exist_ok=False)
            (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (args.out / "index.html").write_text(render(report), encoding="utf-8")
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"Static contracts: {report['status']}. {SCOPE}")
            for w in report["workers"]:
                for r in w["rules"]:
                    values = {k: r.get("facts", {}).get(k) for k in
                              ("advertised_context_tokens", "usable_capacity_tokens", "startup_pool_tokens",
                               "configured_pool_upper_bound_tokens") if k in r.get("facts", {})}
                    suffix = "; " + json.dumps(values, sort_keys=True) if values else ""
                    print(f"{w['id']} ({w['identity']['engine']} {w['identity']['version']}) {r['id']}: {r['verdict']}: {r['message']}{suffix}")
            for r in report["deployment_rules"]:
                print(f"{r['id']}: {r['verdict']}: {r['message']}")
            if args.out:
                print("Report: " + str(args.out.resolve() / "index.html"))
        return {"fine": 0, "not fine": 1, "inconclusive": 2}[report["status"]]
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(f"sdcguard config-safety: inconclusive: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
