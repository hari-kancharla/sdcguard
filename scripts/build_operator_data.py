"""Copy reviewed setup resources into the installed package.

Owner: Hari Krishna Kancharla.
This development helper reads local source files and starts no runtime work.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sdcguard import config_capacity as capacity
from sdcguard import config_protections as protection
from sdcguard.operator_setup import EXPORT_FORMAT, _RESOURCE_NAMES


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def build():
    payload, sources = {}, {}
    for name in _RESOURCE_NAMES:
        source = None
        if name.startswith("guides/"):
            source = ROOT / "docs" / name.split("/", 1)[1]
        elif name.startswith("profiles/"):
            source = ROOT / "examples/config-safety" / name.split("/", 1)[1]
        elif name == "recording-deployment.example.json":
            source = ROOT / "examples/deployment-61229.json"
        if source is not None:
            payload[name] = source.read_bytes()
            sources[name] = {"source": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(payload[name]).hexdigest()}
    payload["static-deployment.example.json"] = encoded({
        "schema": "sdcguard.config-safety.v1", "deployment_id": "fill-from-your-saved-deployment",
        "workers": [{"id": "prefill", "role": "prefill"}, {"id": "decode", "role": "decode"}]})
    payload["SUPPORT.json"] = encoded({"schema": "sdcguard.operator-support.v1", "owner": "Hari Krishna Kancharla",
        "checked_on": "2026-10-27", "identity_basis": "supplied source declarations; no remote attestation",
        "static_profiles": [
            {"engine": "vllm", "version": "0.32.0", "build_commit": capacity.VLLM,
             "runtime_patch_sha256": protection.VLLM_PATCH, "connector": "NixlConnector",
             "scope": "declared ordinary capacity and first-direction pull verifier configuration"},
            {"engine": "sglang", "version": "0.5.22", "build_commit": capacity.SGLANG,
             "scope": "declared ordinary dense capacity, source capability and configured safeguard inventory"},
            {"engine": "vllm-ascend", "version": "source-snapshot", "build_commit": capacity.ASCEND,
             "vllm_commit": capacity.ASCEND_BASE, "scope": "pinned plugin/base ordinary capacity and proposed hold-verifier inventory"},
            {"engine": "vllm-lmcache", "version": "0.26.0", "build_commit": capacity.VLLM_LMCACHE,
             "lmcache_version": "0.5.2", "lmcache_commit": capacity.LMCACHE,
             "scope": "ordinary device capacity and named connector/layout compatibility conditions"}],
        "recorded_profile": {"engine": "vllm", "version": "0.32.0", "nixl_version": "1.4.1",
            "model": "Qwen/Qwen3-0.6B", "model_revision": "required from supplied recording; never filled by preflight",
            "scope": "recorded source lease lifetime and scheduler admission only"},
        "fix_status": "exact proposed PR62065 backport; not a released upstream fix",
        "upgrade_policy": "preserve-named-contracts.v1",
        "native_evidence": {"format": "sdcguard.evidence-envelope.v1",
            "rule": "allocator_to_model_entry_mapping_consistency",
            "required_slots": "explicit named slots; missing evidence remains unknown",
            "upgrade_policy": "preserve-retained-native-facts.v1",
            "current_runtime_upgrade_gate": "inconclusive without an independent runtime witness",
            "ownership": "unknown", "physical_reads": "unknown"},
        "allocation_recipient": {"engine": "vllm", "version": "0.32.0",
            "build_commit": capacity.VLLM, "activation": "explicit internal programmatic factory only",
            "scope": "recorded positive local capacity attribution only",
            "native_originals_checked": 22, "runtime_validation": "pending",
            "common_evidence_ingest": "not implemented", "ownership": "unknown",
            "read_rights": "unknown", "physical_reads": "unknown"},
        "native_check_coverage": {"format": "sdcguard.native-check-coverage.v1",
            "profile": "vllm032-fixed-nixl-pull-single-rank.v1", "engine": "vllm", "version": "0.32.0",
            "scope": "explicit native lease-check execution for the declared ordinary capture client batch",
            "connector": "NixlConnector", "mode": "pull", "tp": 1, "pp": 1, "dp": 1, "dcp": 1, "pcp": 1,
            "cache_groups": 1, "lease_unit": "one remote request lease by rank",
            "required_checks_default": ["lease", "checksum"], "checksum_runtime_observation": "not implemented",
            "sglang_runtime_adapter": "not implemented", "router_forwarding": "unknown",
            "saved_owner_run": "incomplete; all release policies hold", "authenticity": "not established"},
        "checksum_observer_candidate": {"format": "sdcguard.checksum-observer.v1",
            "source_revision": "29d2f4e03740acb14439ce22d24b06d8095d17d3", "checked_on": "2026-10-29",
            "activation": "explicit programmatic installation only", "runtime_qualified": False,
            "release_qualifying": False, "scope": "conditional native checksum chunk comparisons",
            "client_inventory": "unknown", "lease_pairing": "unqualified separate source build"},
        "checksum_client_candidate": {"format": "sdcguard.checksum-client.packet.v1",
            "source_revision": "29d2f4e03740acb14439ce22d24b06d8095d17d3",
            "scope": "planned ordinary client requests, actual request-ID assignments and native checksum calls",
            "runtime_qualified": False, "release_qualifying": False,
            "client_request_ids": "actual passive frontend assignment; native randomization unchanged",
            "inputs": ["REQUEST_PLAN.json", "CLIENT_INVENTORY.json", "CAPTURES.json"],
            "lease_pairing": "unqualified separate source revision"},
        "engine_release_transitions": "unreviewed releases remain inconclusive",
        "runtime_isolation": "not established", "physical_write_safety": "unknown"})
    payload["README.md"] = b"""# Installed operator setup

Owner: Hari Krishna Kancharla.

These files are input shapes and current scope notes. They are not deployment recipes or evidence that a server is safe. The root static template deliberately omits unresolved identities and settings. The recording template keeps its unresolved revision null. The four profile examples are synthetic controls, not measurements. Do not label a setting complete unless it was captured from the relevant deployment.

1. Copy static-deployment.example.json to a new file and supply the actual normalized source, model, settings and bound saved startup facts. See guides/CONFIG_SAFETY.md. Do not relabel an old log to fit a new build.
2. Run `sdcguard operator preflight --static /path/static-deployment.json --out /new/static-preflight`. It starts no engine or request. Run `sdcguard operator verify /new/static-preflight` to recompute its retained facts offline.
3. For the existing supported vLLM recording path, copy recording-deployment.example.json to a different new file and supply real local source, revision and event paths. Run `sdcguard record-plan /path/recording-deployment.json --producer-engine-id P --consumer-engine-id D --out /new/observation`. Use actual connector engine IDs. The plan prepares passive instructions only; it does not start or exercise a deployment.
4. After ordinary recording has been collected, run `sdcguard operator preflight --recording /new/observation/deployment.json --out /new/recording-preflight`. Read report.json and index.html, then run `sdcguard operator verify /new/recording-preflight`. Configured readiness fetches stay disabled and missing readiness facts remain explicit.
5. Each successful retained section is a common evidence packet under sections/static or sections/recording. Compare the same named section across builds using `sdcguard upgrade compare BASELINE_SECTION CANDIDATE_SECTION --out /new/comparison`. See guides/UPGRADE_GATE.md. Unknown engine changes stay inconclusive. Setup/preflight does not re-run a GPU case or a serving workload.
6. The installed package also carries `sdcguard regression export /new/saved-case` and `sdcguard regression verify /new/saved-case`. Those commands export and inspect the reviewed saved case without a checkout. Only the owner runs any documented GPU replay.

You may select both --static and --recording in one preflight. Each selected section remains visible. Only fully retained and independently recomputable section packets can support fine or not fine. Missing, unsupported, oversized or incomplete sections stay unknown. A verified failure takes priority over unrelated gaps. The aggregate output is bounded to 512 files, 512 MiB total and 64 MiB per file. If selected data cannot fit, its standalone command remains available separately; it is never silently dropped from the preflight verdict.

Forward and mapping observations have separate offline commands. See guides/FORWARD_INPUTS.md. For native mapping journals and their settled-close receipts, use `sdcguard assignment capture MAPPING_DIR --out NEW_PACKET`, then `sdcguard assignment verify NEW_PACKET`. See guides/ASSIGNMENT.md and guides/ASSIGNMENT_NATIVE.md. This path is implemented but has no owner-run GPU validation yet. It checks the recorded allocator-to-model-entry relation; ownership, content provenance and physical reads remain unknown. Raw `assignment inspect` does not check recorder closure. These packets are not silently included in static or source-lifetime preflight or its upgrade policy.

To select native mapping evidence explicitly, run `sdcguard evidence collect --native-packet decode=MAPPING_PACKET --out NEW_EVIDENCE`, then `sdcguard evidence verify NEW_EVIDENCE`. Required slots remain visible when their inputs are missing. See guides/NATIVE_EVIDENCE.md. Native upgrade comparisons retain historical evidence but do not establish a candidate runtime execution. Their separate policy does not extend the older V1 policy.

The separate passive allocation-recipient component is documented in guides/ALLOCATION_RECIPIENT.md. It requires explicit internal activation and reports local capacity attribution only. It is not enabled by this setup, the common evidence command or the current owner packet. Read rights, ownership, lease validity and physical reads remain unknown.

Engine pass vs actually covered: `sdcguard native-checks capture --owner-result /path/H100-owner-result --out /new/check-coverage` reads an existing ordinary client recording and explicit native lease-check events. `sdcguard native-checks verify /new/check-coverage` recomputes the portable report. See guides/NATIVE_CHECK_COVERAGE.md. A missing checksum observation stays unverified; native admission and counters do not supply a passing comparison. An incomplete owner run holds every policy. The supported runtime adapter is the narrow vLLM 0.32.0 NIXL pull lease backport. This command is separate from setup preflight and starts no engine or requests.

The separate passive checksum candidate is documented in guides/CHECKSUM_OBSERVER.md. `sdcguard checksum-observer capture --snapshot /path/snapshot.json --out /new/checksum-packet` retains conditional native comparison observations; `sdcguard checksum-observer verify /new/checksum-packet` recomputes them. This candidate always holds release. Runtime qualification, the independent client inventory join and a combined lease/checksum build remain missing. Setup does not activate the observer or start an engine.

The separate client reconciliation candidate uses `sdcguard checksum-client capture --plan REQUEST_PLAN.json --inventory CLIENT_INVENTORY.json --captures CAPTURES.json --out /new/client-checksum-packet` and `sdcguard checksum-client verify /new/client-checksum-packet`. See guides/CHECKSUM_CLIENT.md. It preserves every planned request, including warmups, missing receipts and off-arm requests. It uses the actual native request-ID assignment rather than guessing the engine suffix. Runtime qualification and the combined lease/checksum profile remain pending; every candidate report holds release.

All output directories must be new and their parents must exist. `sdcguard operator verify SETUP_DIR` checks this exported setup against the installed resource profile; exit zero here concerns files only. Preflight and preflight verification exit zero for fine within selected named scopes, one for an established contract failure, and two for inconclusive or invalid evidence. Runtime isolation, authenticity, actual model reads, loaded weights and physical write safety are not established. Keep captured local data private until reviewed for sharing.

SUPPORT.json lists exact inspected profiles. guides/OWNER_ACTIONS.md records project-owner evidence tasks, not a generic operator deployment recipe. guides/OVERLAP.md records material prior art. guides/H100_OBSERVER_RESULT.md and its JSON preserve the separately reviewed owner-run scope and measurements. These guides and the manifest identify the installed files; local hashes do not authenticate a remote process.
"""
    assert set(payload) == set(_RESOURCE_NAMES)
    destination = ROOT / "src/sdcguard/operator_data"
    destination.mkdir(exist_ok=True)
    for name, raw in sorted(payload.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = {"schema": EXPORT_FORMAT, "owner": "Hari Krishna Kancharla",
                "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                          for name, raw in sorted(payload.items())}, "source_inputs": sources,
                "limits": ["local installed resource consistency only", "templates contain unresolved or synthetic declarations",
                           "no runtime isolation or remote authenticity; raw GPU recordings are not bundled",
                           "result notes grant no new deployment verdict"]}
    (destination / "manifest.json").write_bytes(encoded(manifest))
    print(json.dumps({"files": len(payload), "manifest_sha256": hashlib.sha256(encoded(manifest)).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    build()
