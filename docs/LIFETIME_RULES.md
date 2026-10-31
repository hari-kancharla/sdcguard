# Recorded lifetime rules

The rule evaluator judges recorded protocol facts. Generated answers, English engine logs, and a configured recovery policy are not proof of a safety rule.

` sdcguard lifetime rules EVENTS_DIRECTORY --json report.json ` reads a recording once and reports each recorded request attempt separately. Exit 1 means an observed rule violation, 2 means missing/ambiguous evidence or no attempts, and 0 means no violation or unknown among the evaluated obligations. Exit 0 is not a deployment safety guarantee. Retry/reused-id evaluation is excluded from this package.

## Verdicts

- `satisfied`: the recorded facts establish this obligation for this attempt within the stated scope.
- `violated`: the recorded facts establish the broken relationship. A changed answer is not required or sufficient.
- `correctly_rejected`: a normalized lease mismatch was observed, and the transferred state was not admitted.
- `recovered`: rejection was followed by an independently recorded, correctly attributed recomputation completion. Current hooks do not emit this completion, so recompute initiation remains `correctly_rejected`. Client-observed output recovery is separately labeled with its exact-reference basis.
- `unknown`: required identity, mapping, ordering, lifecycle, check, or recording-integrity evidence is absent or ambiguous.
- `refused`: an invalid-lease rejection was followed by an observed failed-load terminal closure. Client outcomes are separately reported from exact attributed response evidence.

## Current scope

The adapter observes vLLM 0.32.0 NIXL pull transfers and the supplied PR #62065 backport. Exact block correspondence is limited to one cache group/source rank with equal block layouts, supported single-rank topology, and no partial local prefix hit. Unsupported layouts are recorded as unknown. Scheduler admission is observed; attention's actual use of the transferred KV is not.

Related process events require a common host-boot monotonic clock. Cross-host wall-clock comparisons cannot prove order. Each relevant recording stream must have a contiguous sequence prefix and a healthy receipt covering the terminal observation. Recording loss invalidates confidence instead of silently passing the attempt. Duplicate publications or attempt identity reuse that cannot be disambiguated are unknown.

A producer completion notification can precede the decoder's completion poll. A source free/reuse during that interval is not automatically a read violation. A satisfied verdict needs an attributable transport-completion witness, or another complete lifetime/check proof. Reallocation before the confirmed READ submission is distinct from this harmless polling interval. Free-only evidence is conservative because reference release alone does not establish another allocation or physical overwrite.

## Experiment reporting

The paired experiment keeps its previous answer/log outcome separately from evidence-only findings. `evaluator_validation` requires both controls to be satisfied and the held attempts to be violated on stock, or correctly rejected/recovered on the fixed build. The comparison includes unmapped outcomes and is not an independent accuracy estimate: the historical oracle also uses the ledger.

Raw ledger records are retained. A source-only R2 overlap may be classified as a completion-polling artifact only when the evaluator establishes the exact attempt's attributable completion notification. Destination-region violations, other operations, other rules and unknown evidence remain actionable.

## Shared destination-retirement component rule

`sdcguard lifetime conformance --json` runs packaged, hash-pinned original component functions on CPU. `generation_obligations.py` judges one sufficient rule: each bound operation/resource/incarnation requires a complete writer set and an attributable terminal fence from every writer before observed release. The rule has no engine branches. It returns satisfied or unknown within `source_fragment_cpu`; physical write safety stays unknown. Missing evidence is never labelled a defect.

`generation_observations.py` translates two distinct source contracts. vLLM allocation/free and local READ destinations provide observed allocation incarnations; actual registered handle sets and DONE poll results are required because done-request receipts can follow failure cleanup. SGLang observes a metadata-slot protocol hold, its armed wire generation, zero outstanding counters, matching-rank drain ACKs and release. That auxiliary-write destination is not relabelled as a physical KV-page allocation. Every mapped operation needs individual complete evidence, and stream coverage must span the fresh start through close. Reused IDs, ambiguous layouts, missing operations, timeout release and cross-process pool mapping are unknown.

The command verifies its exact source/excerpt pins and upstream license files before executing the bundled ordinary functions. It accepts no external source path, starts no serving process and uses controlled transport, memory and coordinator mocks. The current source pins are vLLM 5df63a7203cdb9328d824a27b682e77c7b9928be and SGLang 41fb719cf25966e6155f3e3d91f1157bc1a15123. This component proof does not replace the source-lease evaluator, attest a live deployment, validate physical DMA or reproduce the SGLang production incident.

## Evidence status

The evaluator and revised hooks are under independent review and local CPU validation. Historical GPU results from the preserved 576ac8b snapshot do not validate these revised hooks. A new paired GPU recording is required before calling this evaluator GPU-validated. CPU protocol fixtures cannot establish physical GPU memory behavior or whole-deployment isolation.
