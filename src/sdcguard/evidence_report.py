"""One envelope for distinct, existing evidence contracts.

Owner: Hari Krishna Kancharla.
Combining evidence sections never extends one section's scope into another.
"""
from __future__ import annotations

from copy import deepcopy
import html
import json

from . import evidence_catalog as catalog
from . import evidence_io as io

SCHEMA = "sdcguard.evidence-report.v1"
SCOPE = "Named static, recorded-event and source-fragment contracts only"
LIMITS = [
    "This packet does not establish runtime request isolation or whole-model correctness.",
    "Local hashes check consistency, not authenticity or remote deployment state.",
    "Static inputs are normalized operator declarations and supplied startup facts.",
    "CPU component observations use reviewed functions with controlled mocks, not live engines.",
    "A protocol-held auxiliary slot is not a physical KV allocation generation.",
    "Actual model reads, GPU/DMA behavior and unobserved kernel identities remain outside scope.",
]


def catalog_hash():
    return io.digest(io.json_bytes(catalog.catalog()))


def static_findings(native):
    findings = []
    for worker in native["workers"]:
        for row in worker["rules"]:
            metadata = catalog.describe(row["id"], "static_config")
            facts = deepcopy(row["facts"])
            facts.update(runtime_execution="not established", native_basis=row["basis"],
                         native_message=row["message"])
            findings.append(dict(rule_id=row["id"], semantic_revision=metadata["semantic_revision"],
                subject={"worker": worker["id"]}, scenario="declared-deployment-contract",
                evidence_kind="static_config", scope=metadata["scope"], resource_kind=metadata["resource_kind"],
                verdict=row["verdict"], facts=facts,
                missing=[row["message"]] if row["verdict"] == "unknown" else [],
                required=row["required_for_status"], resolved_by=row.get("resolved_by"),
                evidence={"native_section": "static", "worker": worker["id"], "sources": row["sources"]}))
    for row in native["deployment_rules"]:
        metadata = catalog.describe(row["id"], "static_config")
        facts = deepcopy(row["facts"])
        facts.update(runtime_execution="not established", native_basis=row["basis"],
                     native_message=row["message"])
        findings.append(dict(rule_id=row["id"], semantic_revision=metadata["semantic_revision"],
            subject={"deployment": True}, scenario="declared-deployment-contract",
            evidence_kind="static_config", scope=metadata["scope"], resource_kind=metadata["resource_kind"],
            verdict=row["verdict"], facts=facts,
            missing=[row["message"]] if row["verdict"] == "unknown" else [], required=True,
            evidence={"native_section": "static", "sources": row["sources"]}))
    return findings


def finding_key(finding):
    return (finding["rule_id"], finding["semantic_revision"], finding["evidence_kind"],
            finding["scope"], finding["resource_kind"],
            json.dumps(finding["subject"], sort_keys=True, separators=(",", ":")),
            json.dumps(finding["scenario"], sort_keys=True, separators=(",", ":")))


def build(*, profile, static=None, components=None, recorded=None):
    """Canonicalize already recomputed native results; never trust a report flag."""
    if (not isinstance(profile, dict) or set(profile) != {"static", "recorded", "components"}
            or type(profile["static"]) is not bool or type(profile["recorded"]) is not bool
            or profile["components"] not in ("none", "all")
            or not (profile["static"] or profile["recorded"] or profile["components"] == "all")):
        raise io.EvidenceError("unsupported or empty evidence profile")
    if ((static is not None) != profile["static"] or (recorded is not None) != profile["recorded"]
            or (components is not None) != (profile["components"] == "all")):
        raise io.EvidenceError("evidence sections do not match the selected profile")
    findings, identities, native, gaps = [], {}, {}, []
    if static is not None:
        findings.extend(static_findings(static))
        identities["static"] = {"model": static["model"],
            "advertised_context_tokens": static["advertised_context_tokens"],
            "workers": [{"id": w["id"], "role": w["role"], "identity": w["identity"],
                         "config_sha256": w["config_sha256"]} for w in static["workers"]],
            "basis": static["identity_basis"], "kernel": "not observed in static inputs"}
        native["static"] = static
        if static["status"] == "inconclusive":
            gaps.append("The static contract section contains required unknown facts.")
    if components is not None:
        from . import evidence_components
        findings.extend(evidence_components.normalize(components))
        identities["components"] = {engine: {"source": item["source"],
            "engine_version": "not observed", "model": "not exercised", "kernel": "not exercised"}
            for engine, item in components["engines"].items()}
        native["components"] = components
        if components["status"] != "satisfied":
            gaps.append("The source-fragment component section contains unknown proof facts.")
    if recorded is not None:
        findings.extend(deepcopy(recorded["findings"]))
        identities["recorded"] = deepcopy(recorded["identity"])
        native["recorded"] = deepcopy(recorded)
        if recorded["legacy_scope_verdict"] == "inconclusive":
            gaps.append("The recorded-event section contains required identity or observation gaps.")
    seen = set()
    for finding in findings:
        finding.setdefault("required", True)
        if type(finding["required"]) is not bool:
            raise io.EvidenceError("required finding state is malformed")
        errors = catalog.validate_finding(finding, finding.get("evidence_kind"))
        if errors:
            raise io.EvidenceError("catalog contract mismatch: " + "; ".join(errors))
        key = finding_key(finding)
        if key in seen:
            raise io.EvidenceError("duplicate finding identity")
        seen.add(key)
    required = [finding for finding in findings if finding["required"]]
    if not required:
        raise io.EvidenceError("no required catalogued finding was produced")
    verdicts = {finding["verdict"] for finding in required}
    status = ("not fine" if "violated" in verdicts else "inconclusive"
              if gaps or verdicts & {"unknown", "not_exercised"} else "fine")
    return {"schema": SCHEMA, "catalog_schema": catalog.SCHEMA,
            "catalog_semantic_revision": catalog.SEMANTIC_REVISION, "catalog_sha256": catalog_hash(),
            "scope": SCOPE, "status": status, "profile": deepcopy(profile),
            "runtime_isolation": "not established", "authenticity": "not established",
            "identities": identities, "findings": findings, "missing": gaps,
            "limits": list(LIMITS), "native_sections": native}


def render(report):
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for finding in report["findings"]:
        note = finding["facts"].get("native_message", "")
        if not note:
            note = "; ".join(finding["missing"]) or "See the captured observations and original assessment."
        if not finding["required"]:
            note += " This row is resolved by the recorded startup-capacity contract."
        rows.append("<tr><td>" + esc(finding["rule_id"]) + "<br>" + esc(finding["subject"])
                    + "</td><td>" + esc(finding["verdict"]) + "</td><td>"
                    + esc(finding["evidence_kind"]) + "<br>" + esc(finding["resource_kind"])
                    + "</td><td>" + esc(note) + "</td></tr>")
    coverage = report.get("native_sections", {}).get("recorded", {}).get("request_coverage")
    coverage_html = ("<h2>Declared request coverage</h2><pre>"
                     + esc(json.dumps(coverage, indent=2, sort_keys=True)) + "</pre>") if coverage is not None else ""
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Scoped evidence report</title><style>body{font:15px/1.4 system-ui;max-width:1200px;"
        "margin:24px auto;padding:0 16px;color:#15232c}table{border-collapse:collapse;width:100%}"
        "td,th{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #ccd4da}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#004c89}</style>"
        "<h1>Within the named scopes: " + esc(report["status"]) + "</h1><p>"
        + esc(SCOPE) + ". Runtime request isolation is not established.</p><p>"
        "Each section keeps its own source, model and resource identity. Combining rows does not extend coverage. "
        "Local hashes establish consistency, not authenticity.</p><p>"
        "<a href='report.json'>Full findings and captured facts</a> | "
        "<a href='manifest.json'>File manifest</a></p><table><thead><tr><th>Rule and subject</th>"
        "<th>Verdict</th><th>Evidence and resource</th><th>Finding</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table><h2>Scoped identities</h2><pre>"
        + esc(json.dumps(report["identities"], indent=2, sort_keys=True))
        + "</pre>" + coverage_html + "<h2>Limits</h2><ul>" + "".join("<li>" + esc(note) + "</li>" for note in LIMITS)
        + "</ul></html>\n").encode("utf-8")
