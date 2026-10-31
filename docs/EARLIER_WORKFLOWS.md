# Earlier workflows and evidence scopes

These interfaces retain their original evidence scopes. Start with the [current checksum quickstart](../README.md).

## Evidence for inference-state correctness

Declared workload coverage: config schema `2` requires a `client_inventory` path in addition to the existing engine and recording inputs. The inventory is a separate client declaration. `sdcguard inventory import-performance saved-PERFORMANCE.json --out new-client-inventory.json` extracts the complete ordinary batch from the supported saved client format, including its reference requests. It starts no server. Then `sdcguard assess workload-deployment.json --out new-assessment` captures the inventory, inspected source and events together. `sdcguard report-verify new-assessment` recomputes the result offline.

Every declared request must match exactly one recorded attempt by its full transfer descriptor. Missing, duplicate, ambiguous and unexpected attempts make the workload result inconclusive. A matched descriptor does not prove the lifecycle requirement held. The HTML report keeps coverage, lifecycle findings and unknown client outcomes separate. Config schema `1` remains a narrower event-only assessment; its result does not qualify a declared workload. Local hashes prove consistency, not authenticity or a complete census of server traffic.

Installed setup: `sdcguard operator export new-setup` writes distinct static and recording templates, current guides and exact support pins. `sdcguard operator preflight --static static-deployment.json --recording recording-deployment.json --out new-preflight` inspects supplied local inputs and preserves separately verifiable sections. Select either input or both. It starts no server and makes no readiness request. Missing facts remain inconclusive. `sdcguard operator verify new-preflight` recomputes the saved summary from its captured sections.

Static deployment preflight: `sdcguard config-safety static-deployment.json --out new-report` reads local config and saved startup logs for pinned vLLM, SGLang, vLLM-Ascend and vLLM + LMCache profiles. It starts no server. The [contract guide](CONFIG_SAFETY.md) distinguishes capacity plans, observed startup facts, protection inventory and unknown coverage. A fine result is scoped to those contracts.

Portable evidence: `sdcguard evidence collect --config static-deployment.json --components --out new-packet` combines the named checks with the existing ordinary CPU component results. `sdcguard evidence verify new-packet` re-judges the captured facts offline. The [report guide](EVIDENCE_REPORT.md) explains each evidence domain, native outcomes, source identity and missing facts. Runtime request isolation remains outside these scoped results.

Upgrade gate: `sdcguard upgrade compare baseline-packet candidate-packet --out new-comparison` verifies both packets and checks the same named contracts under a fixed installed policy. It blocks known failures, lost coverage and unreviewed changes. The [comparison guide](UPGRADE_GATE.md) explains permitted capacity changes, distinct outcomes, attribution and limits. A scoped pass does not establish that a candidate deployment executed an upgrade test.

Forward inputs: `sdcguard forward-inputs inspect /path/forward-recording --out new-forward-packet` captures and checks recorded attention metadata offline. `sdcguard forward-inputs verify new-forward-packet` recomputes the report. This experimental recorder has not had an owner-run GPU validation. The [forward-input guide](FORWARD_INPUTS.md) distinguishes the observed input plan from physical reads and ownership, which remain unknown. Saved admission traces cannot acquire forward observations through replay.

**Engine pass vs actually covered.** `sdcguard native-checks capture --owner-result /path/H100-owner-result --out /new/check-coverage` reconciles the saved ordinary client batch with explicitly recorded native checks. `sdcguard native-checks verify /new/check-coverage` recomputes the retained JSON, text and HTML reports. The first adapter covers the narrow vLLM 0.32.0 lease-backport recording. Missing checksum observations remain unverified, and an incomplete owner run holds every release policy. See the [native-check coverage guide](NATIVE_CHECK_COVERAGE.md).

Passive checksum candidate: the separate, explicitly installed observer watches the pinned checksum scheduler's original callbacks and retains their native results and conditional chunk scope. `sdcguard checksum-observer capture --snapshot /path/snapshot.json --out /new/checksum-packet` retains an offline report; `sdcguard checksum-observer verify /new/checksum-packet` recomputes it. This candidate always holds release. It has no runtime qualification or independent client completeness claim and cannot borrow the older build's lease positives. See the [checksum observer guide](CHECKSUM_OBSERVER.md).

Client reconciliation candidate: `sdcguard checksum-client capture --plan REQUEST_PLAN.json --inventory CLIENT_INVENTORY.json --captures CAPTURES.json --out /new/client-checksum-packet` joins each planned ordinary request to its client receipts, actual engine request-ID assignment and recorded checksum calls. `sdcguard checksum-client verify /new/client-checksum-packet` recomputes the portable result. Every planned slot remains visible. Native success, comparison coverage and runtime qualification stay separate. This candidate always holds release. See the [client checksum guide](CHECKSUM_CLIENT.md).


Mapping relation: `sdcguard assignment capture /path/completed-native-mapping --out new-mapping-packet` checks native journals together with their settled-close receipts and retains both. `sdcguard assignment verify new-mapping-packet` checks those receipts again and recomputes allocator-to-model-entry mapping consistency. The passive native collector is implemented for one pinned vLLM profile; its GPU validation remains pending. The [native collector guide](ASSIGNMENT_NATIVE.md) covers activation and closure. The [mapping guide](ASSIGNMENT.md) distinguishes native capture from raw-format inspection and explains the ownership, content and runtime limits.

Named native evidence: `sdcguard evidence collect --native-packet decode=new-mapping-packet --out new-evidence` retains and rechecks a closed mapping packet in the report workflow. Add `--require-native prefill` when that slot is required but its input is not available. Missing slots stay unknown. The [native evidence guide](NATIVE_EVIDENCE.md) explains aliases, separate contracts and upgrade comparison. Retained native evidence does not approve a candidate runtime execution.

Allocation-recipient facts: the separate [capacity attribution component](ALLOCATION_RECIPIENT.md) checks actual Request objects, native allocation returns and the paired mapping journal. It is off by default and requires explicit internal activation. It does not yet establish read rights, ownership, lease validity or physical reads, and is not enabled by the current owner packet.

An inference engine can compute every matrix multiply correctly while using KV state that belongs to another request. sdcguard is building a vendor-independent acceptance and regression workflow for that serving-state boundary: inspect the exact deployment, collect passive observations, evaluate bounded rules, and preserve an independently inspectable packet.

Deployment support for these workflows is pinned to the saved public vLLM #61229 configuration. The installed CPU conformance command applies one destination-retirement rule to pinned original vLLM and SGLang component functions. That evidence is separate from deployed multi-engine protection. Legacy arithmetic, training and fleet tools keep their separate historical evidence in the full repository's `docs/STATUS.md`. That historical archive requires the repository and is outside this installed workflow.

## Why now, and what is actually demonstrated

The [vLLM issue](https://github.com/vllm-project/vllm/issues/61229) is open. [PR #62065](https://github.com/vllm-project/vllm/pull/62065) is an open, unmerged proposal as checked October 27, 2026. The saved fixed arm is an exact backport of that proposal, not a released official fix. The tested stock engine was vLLM 0.32.0; open issue status alone does not establish the status of every later release.

[Cloudflare's incident report](https://blog.cloudflare.com/kv-cache-exposure-remediation/) describes GLM-5.3 on Workers AI returning 155 affected responses between August 27 and September 9, 2026, with disaggregated SGLang/Mooncake and HTTP 200 responses. This is a real-world reason to check serving-state correctness. It is not evidence that this package detected that incident.

Our [historical result](result-61229.html) preserves the original JSONs: the stock arm returned wrong answers in 5/5 replays; the backport recovered in 5/5, with clean controls. The saved case did not demonstrate cross-request text disclosure. A [fresh owner-run H100 validation](H100_OBSERVER_RESULT.md) independently confirmed the revised source-lifetime hooks on five fixed-arm replays and clean ordinary traffic. Actual attention consumption, terminal refusal and verified minimality remain unproved.

## Install and inspect the packet without a source checkout

Use Python 3.10 or 3.13 and the reviewed local wheel. A similarly numbered public package does not establish that it contains this source.

```sh
python -m venv /new/environment
/new/environment/bin/python -m pip install /path/to/reviewed/sdcguard-0.4.0-py3-none-any.whl
/new/environment/bin/sdcguard regression export /new/case
/new/environment/bin/sdcguard regression verify /new/case
```

The wheel carries the reviewed harness, exact saved artifacts, model pin, result page, criteria and support notes. Export and verification execute no GPU work. Open `/new/case/RESULT.html`. Packet integrity is checked against this installed package's pins; it is not a signature or remote attestation.

The replay model/tokenizer is pinned to `c1899de289a04d12100db370d81485cdf75e47ca`. A read-only CPU audit recovered the sole snapshot from the persistent Modal cache and verified all ten files against immutable upstream identities, including actual weight bytes. `MODEL_PIN.json` and `MODEL_PROVENANCE.json` preserve that evidence. The original run artifacts remain unchanged and record `main`; the subsequent cache recovery corroborates their model association without inventing an original run manifest.

The GPU replay is owner-run. The packet's README gives the exact command. It checks the local snapshot against the external packet pin, so the original cache can stay read-only. Pass/fail/inconclusive criteria are explicit in `CRITERIA.md`. No new failure ordering, retry/reused-ID case or transfer mechanism is added.

## Run the shared-rule CPU demonstration

```sh
sdcguard lifetime conformance --json
```

This installed, offline command verifies packaged excerpt/source pins and upstream licenses, executes the original ordinary vLLM allocation/READ/completion path and SGLang matching-generation ACK/deferred-release path with controlled CPU mocks, and emits both receipts and the same rule assessment. Use `--engine vllm` or `--engine sglang` to inspect one side. It starts no server, sends no requests and allocates no GPU memory.

The shared obligation is concrete: every writer bound to an observed destination incarnation must have an attributable terminal fence before that incarnation is released. vLLM allocation generations and SGLang metadata-slot protocol holds retain their different meanings. The source pins are vLLM `5df63a7203cdb9328d824a27b682e77c7b9928be` and SGLang `41fb719cf25966e6155f3e3d91f1157bc1a15123`. The rule contains no engine or case exceptions. Missing operations, handles, mappings, ACK attribution, coverage or closure produce `unknown`.

Exit 0 means these ordinary component observations satisfy the bounded rule; exit 2 means inconclusive. Scope is always `source_fragment_cpu`, with `physical_write_safety: unknown`. Mocked transport and counters do not establish physical writes, deployed engine isolation or the SGLang incident's reproduction. The existing vLLM source-lease rule and deployment support remain separate.

## Prepare ordinary observation, then assess it

Supply the supported deployment declaration by copying `deployment.example.json` from the exported packet to a new file outside that packet and filling in the local paths. The template pins the recovered model revision; its unresolved engine/event paths intentionally remain inconclusive until real values are supplied. The [source template](../examples/deployment-61229.json) also remains available.

```sh
sdcguard record-plan /path/deployment.json --producer-engine-id P --consumer-engine-id D --out /new/observation
```

Use the actual existing connector engine IDs, which are P and D in the saved configuration; role labels are not substituted for them. This offline command inspects supported source/configuration pins and writes a fresh shared recording directory, deployment file and sourceable producer/consumer environment snippets. It starts no server and sends no traffic. The snippets refuse active holds, transport overrides, numerical-check activation/require settings and incompatible plugin filtering. Prepared means instructions exist, not that hooks ran or loaded weights were attested. They are then applied to the existing supported deployment and ordinary workload.

```sh
sdcguard assess /new/observation/deployment.json --out /new/assessment --json
sdcguard report-verify /new/assessment
```

Assessment reports within-scope FINE, NOT FINE or INCONCLUSIVE. The packet includes frozen configuration/events, the nine inspected engine files, JSON, a readable HTML report and a content manifest. Verification is offline, recomputes local facts and never refreshes readiness HTTP. Applicability, proposed-backport source presence, observed check execution, lifecycle result, client outcome, regression evidence and overhead are separate sections. Missing client/performance evidence remains unknown or not measured.

FINE covers the complete recorded source-lifetime/admission obligation. It does not certify later traffic, every operation, live model weights, actual attention consumption, every answer or whole-deployment safety. A source patch is distinct from evidence that its check executed. Optional configured readiness checks read existing worker metrics only. Unknown source layouts and incomplete recordings remain inconclusive.

## Bind performance measurements without editing result JSON

Preparation is offline and reads existing local files; the measurements below are owner-run against an existing supported endpoint.

```sh
sdcguard overhead prepare /path/deployment.json --model-snapshot /path/to/COMMIT --model-pin /new/case/MODEL_PIN.json --gpu-uuid "$DEPLOYMENT_GPU_UUID" --out /new/performance-identity.json
sdcguard perf --url http://127.0.0.1:8000 --identity /new/performance-identity.json --checks off --out /new/baseline.json
sdcguard perf --url http://127.0.0.1:8000 --identity /new/performance-identity.json --checks on --out /new/checked.json
sdcguard overhead /new/baseline.json /new/checked.json
```

The same engine source variant, model and configuration must be used on both sides. GPU UUID and checks state are operator declarations; the tool does not toggle the server. Local pins are validated before and after measurement, outside timed rounds. Failed/interrupted or mismatched measurements are retained and inconclusive. Do not compare stock versus backported engines as observer cost.

Existing `perf` sends ordinary OpenAI completion traffic. It does not create a prefill/decode router or a new transfer path. Use the existing supported front door/load and establish that it exercises the intended NIXL checks. `itl_p99_ms` is SSE text-chunk gap p99, not device per-token timing; grouped round spread is descriptive, not a confidence interval or proof of alternating trials.

The separate short [H100 measurement](H100_OBSERVER_RESULT.md) used the same fixed engine with lease verification enabled in both arms. Its observed lifetime-observer/recording delta was -4.025577679806001% in median round decode-token rate and +5.426599345229883% in median round end-to-end p99. The grouped serial run excludes offline evaluation time and does not measure saturated throughput, TTFT, inter-token latency or the lease fix's own cost. Raw values, logs and provenance limits remain explicit.

## Evidence and product direction

- [Pinned support and evidence matrix](SUPPORT_61229.md)
- [Fresh H100 observer result and measurement limits](H100_OBSERVER_RESULT.md)
- [Product assessment, evidence gates and next work](PRODUCT_REVIEW.md)
- [Direct and adjacent prior-art overlap](OVERLAP.md)
- [Rule semantics](LIFETIME_RULES.md)
- [Owner-only validation actions](OWNER_ACTIONS.md)

GRIEF is direct serving-fuzzer/forensics overlap. ResidentClaim already checks shared obligations and anchored cross-engine evidence. Metria implements qualified runtime-upgrade verification, and KVSentry implements independent static cross-engine auditing. AEGIS, OpGuard and Vosti have adjacent integrity/determinism scopes. The intended focus here is maintained KV transfer/config-contract mappings for a declared stack. Shared rules, unknowns, replay, reports and CI gates alone do not establish novelty. See the [source-read overlap table](OVERLAP.md); deployed generality and comparative benefit remain unproven.

Full CPU validation runs on both Python versions. CPU tests and synthetic handler fixtures do not establish GPU runtime behavior or universal absence of false alarms.
