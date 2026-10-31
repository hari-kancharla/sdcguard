# Native mapping evidence in one report

Owner: Hari Krishna Kancharla.

This workflow reads local configuration, saved startup facts and completed packets. It starts no engine or request. A native mapping result concerns the allocator-to-model-entry relation recorded in that packet. Ownership, read rights, content identity and physical GPU reads remain unknown.

## Collect and verify

First obtain a closed mapping packet using the separately documented `sdcguard assignment capture` command. That command needs the raw mapping journal and its settled-close receipt. Raw `assignment inspect` output is a different format and cannot replace native closure evidence.

```sh
sdcguard evidence collect --native-packet decode=/path/mapping-packet --out /new/evidence
sdcguard evidence verify /new/evidence
```

Use a stable lowercase slot name for each required recording. Names start with a letter and may contain lowercase letters, digits, underscores and hyphens. Each supplied slot is required. Assigning two paths to one name refuses before capture. Repeated `--require-native NAME` selects the same requirement once.

Add `--config /path/static-deployment.json`, `--assessment-packet /path/assessment-packet` or `--components` to include the existing evidence contracts. Alternatively use `--legacy-packet /path/existing-evidence` to retain a completed V1 evidence packet. A retained V1 packet cannot be mixed with newly selected V1 inputs. Each section keeps its own source, model and scope. A native-only report says the existing evidence section was not selected.

An absent recording stays explicit:

```sh
sdcguard evidence collect --require-native decode --out /new/missing-evidence
```

This creates a completed collection with an inconclusive scoped result. It creates no fictitious native session. Collection completion means the selected inputs and their gaps were retained, not that every check passed.

Open `index.html` and retain `report.json`, the input packets and `manifest.json`. Verification recomputes the nested V1 and native reports using the installed evaluators. Changed derived findings, evaluator identities or outer inventories cannot establish a verified report. When selected, current ordinary CPU component checks must also complete. They do not execute a candidate engine.

Each native slot shows the recorded adapter, model revision, configuration digest and source-file digest. The complete recorded fields remain in the native report. Those fields do not authenticate the deployed process or loaded weights.

Exit codes are 0 for fine within the selected named contracts, 1 for an independently established contract violation, and 2 for inconclusive or invalid evidence. Missing or malformed native inputs produce gaps. A verified violation from another selected section remains visible.

## Copies and conflicting recorded identity

Identical raw journals are retained aliases, even if packet paths or packaging differ. The report counts raw streams separately from executions. It does not establish any independent runtime execution count.

Conflicting raw journals that claim one recorded session stay an ambiguous group. Their effective mapping findings become unknown. Original packets and recomputed native results remain available for audit. Other independently verified findings keep their own verdicts. Slot names, session identifiers and filenames are locators, not authenticated deployment identities or stable workload bindings.

## Compare across builds

```sh
sdcguard upgrade compare /path/baseline-evidence /path/candidate-evidence --out /new/comparison
sdcguard upgrade verify /new/comparison
sdcguard upgrade policy --native
```

The native comparison retains both input packets. Removing a required slot is a gap. Argument order does not change slot identity. Existing V1 contracts use their unchanged preservation policy; native mapping has its own versioned policy.

Unchanged journals may preserve a historical mapping result. A different capture remains incomparable without a reusable workload binding and a reviewed source transition. Matching recorded model or build fields does not supply that evidence.

The current required-native upgrade execution gate remains inconclusive when it has only retained recordings. A verified candidate violation still blocks the named-scope gate. It is labelled a candidate violation, with attribution to the upgrade unestablished. The saved comparison grants no current approval. Run verification to refresh any selected ordinary CPU checks and obtain the current bounded decision.

## Bounds and compatibility

All output paths must be new and outside every external packet input tree. Original packet members and bytes are rechecked. This is local consistency during the observed reads, not an atomic snapshot or remote authentication.

The combined artifact shares the existing 512-file, 512-MiB aggregate and 64-MiB per-file bounds, with at most 512 directories including the root and 32 directory levels. Nested prefixes consume that depth. Collection reserves room for the report, readable page, manifest and exact metadata before holding the next leaf. A standalone packet can be too large or deep for the combined format. Unsupported input stays unknown. Oversized derived output refuses completion.

Retained `.json` and `.jsonl` members also share a one-million-node admission allowance. A byte scan charges containers, strings including keys, and primitive values before reconstruction or decoding. It creates no JSON tree and supplies no validity or security verdict. Existing verifiers still judge syntax and facts. Structured input must use NUL-free UTF8 without a leading BOM; unsupported encodings remain explicit gaps. Other text and source files retain their separate byte bounds.

The new external packet path uses streamed directory inventories and bounded exact-byte rereads. Older standalone IO contracts remain separate. Existing V1 evaluator hashes remain strict. A future checker change needs a reviewed migration or a separately labelled rejudgment that preserves the original inputs and evaluator provenance.

This workflow adds no new engine support, failure condition, request scheduling, retry case or live-server experiment. The retained ordinary CPU fixture used during development is not GPU validation. Allocation-recipient, permission and lease joins remain separate work.
