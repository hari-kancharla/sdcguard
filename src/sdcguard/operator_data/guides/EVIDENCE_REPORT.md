# Portable evidence report

Owner: Hari Krishna Kancharla.

The evidence packet puts the existing named rules into one inspectable format. It preserves each rule's source, subject, scope, resource kind and native verdict. It does not combine separate evidence into a stronger isolation claim.

## Collect and verify

Use an existing normalized deployment config to collect the static section:

```sh
sdcguard evidence collect --config deployment.json --out new-packet
sdcguard evidence verify new-packet
```

Add both existing source-fragment CPU component checks:

```sh
sdcguard evidence collect --config deployment.json --components --out new-packet
```

An existing portable recording assessment can also be included:

```sh
sdcguard evidence collect --assessment-packet saved-assessment --components --out new-packet
```

The three inputs can be selected together. At least one is required. The output's parent directory must exist, and the output directory must be new. Existing files are never replaced. A failed capture retains an incomplete manifest when possible. It cannot verify as a completed packet.

Collection starts no serving engine, makes no serving or readiness request, and runs no owner replay. The optional component selection runs only the already reviewed ordinary component functions with CPU mocks. It accepts no source path or caller code. Saved engine source files in recording packets are read as data, not imported or executed.

Both collection and verification exit 0 for fine within the selected named scopes, 1 for an established required contract violation, and 2 for inconclusive or invalid evidence. A valid unknown packet can have verified local integrity while still exiting 2. A fresh component execution gap cannot erase a known violation in another section.

## Evidence domains

| Evidence kind | Named contract | What it does not establish |
| --- | --- | --- |
| Static config and saved startup text | Capacity plans, available startup facts, protection inventory and specific source/layout policies | Actual deployed settings, authenticated binaries, completed allocation from vLLM planning logs, or runtime isolation |
| Recorded engine events | The existing source-lifetime rule at scheduler admission, with native rejection, recovery and refusal outcomes | Actual attention reads, answer correctness or unobserved activity |
| Source-fragment CPU | The same destination-incarnation rule fed by reviewed vLLM and SGLang ordinary code fragments | A live engine, GPU writes or physical DMA safety |

The SGLang component resource is protocol-held auxiliary metadata. Its generation is a protocol hold. The vLLM component resource is an observed destination KV allocation. The report retains that distinction. Neither row is relabelled as a proof about the other's resource.

Each subject has its own identity. Static and recorded model/config declarations do not become the model used by a CPU component. CPU component engine versions, model weights and kernels were not exercised; the exact source commits and fragment hashes are recorded instead. A missing fact needed for a rule remains unknown.

## Catalog and native outcomes

`sdcguard evidence catalog` prints the immutable catalog metadata for the existing rule IDs. Each entry defines its semantic revision, evidence domain, resource kind, allowed verdicts, required facts, assumptions and state fields.

Catalog validation checks shape and scope. It does not prove a verdict. The original evaluators remain authoritative. The packet verifier re-runs them from captured inputs and rejects a saved finding that disagrees.

The source-lifetime verdicts stay distinct: satisfied, violated, correctly rejected, recovered, refused, unknown and not exercised. Recovery is not inferred from rejection or a matching model answer. The current destination-incarnation evaluator has only satisfied and unknown outcomes. It does not invent a violation from a missing fence.

Optional detection inventory can be known on or known off. Both are known inventory facts, not equivalent protection states. Their state fields are preserved so the later upgrade gate can detect a setting change even if the inventory verdict remains satisfied. This report does not yet implement that upgrade comparison.

## Packet contents and verification

The packet contains the original static config bytes, a portable config, captured startup bytes or explicit missing-capture receipts, the selected existing recording packet, and any CPU component observations. Unavailable startup bytes remain missing in the portable config. The original reason is kept separately; it cannot become a successful startup observation by changing a path.

The packet also contains the installed catalog and relevant evaluator hashes, JSON findings and a static HTML report. The completed manifest is written last. It binds member names, file lengths, hashes and directory inventory. Absolute capture paths are not used to read data during verification.

Verification checks the complete inventory and takes a private temporary snapshot. It re-judges the static inputs, normalizes recorded facts through the existing offline assessment verifier, and re-evaluates component observations through the shared rule. It compares the resulting JSON and readable report with the captured derived files. It rejects edited-and-rehashed findings, caller catalog replacements, unsupported evaluator identities, missing evidence, extra members and scope promotion.

When components were selected, verification also runs the existing trusted CPU components again. This fresh execution is a separate witness. Process, session and resource IDs can differ, so it is not a byte-for-byte replay of the historical observations. Saved observations still do not authenticate who ran them.

Local hashes establish consistency, not authenticity or remote attestation. Someone able to fabricate an entirely self-consistent input packet has not thereby proved a real engine produced it. The report states that limit. It does not claim to authenticate model weights or an installed deployment.

The packet can be copied to another folder and verified without the original source directories or an engine installation. An unreviewed catalog or evaluator revision returns inconclusive; its fields cannot be forced into a pass by changing the manifest. Native config parsing remains the reviewed normalized wrapper described in [CONFIG_SAFETY.md](CONFIG_SAFETY.md), not arbitrary YAML or launch arguments.

The packet bounds are 512 files, 64 MiB per member, 512 MiB total and 32 directory levels. Evidence JSON has at most 1,000,000 structural nodes per document and 64 nesting levels; deployment configurations and individual catalog findings retain their 100,000-node bound. Oversized evidence stays unsupported rather than being truncated. Regular local files are required. Symlinks, special files, duplicate JSON keys, non-finite numbers and unsafe paths are refused. Verification does not modify the supplied packet. It may create and remove its own private temporary snapshot.

## Current limits and next step

This is the portable report foundation for the [upgrade gate](UPGRADE_GATE.md). The comparison command preserves required findings, exact comparison scope and detection states; missing evidence blocks as inconclusive. Recovery and refusal have no automatic severity ranking.

The existing static support pins and architecture limits remain unchanged. Actual GLM/DSA and DeepSeek V4 capacity are still unknown. The historical fixed recording lacks structured attempt facts and an immutable model revision in its original metadata. It remains inconclusive even when the captured fix source is present and both CPU components satisfy their own rules.

The [fresh owner-run H100 result](H100_OBSERVER_RESULT.md) is separate evidence with revised recorded identities. Its ordinary observer-on records can be evaluated for the bounded source-lifetime rule. Reusing those records does not create a new runtime or upgrade witness, and separate client/performance receipts are not inferred from lifecycle findings. Source files reconstructed locally to match recorded runtime hashes must be described as corroborated local source association, not authenticated remote source capture.

No new failure schedule, retry case, reused ID, request mutation or live-server condition is introduced by this command. The completed H100 owner run retains its separate scope and results.
