# Static input examples

Owner: Hari Krishna Kancharla.

These files are synthetic input shapes and arithmetic examples. The placeholder model revision is not a measured stack. Do not run them as serving configurations. The checker reads them only.

- `vllm-capacity-shortfall.json`: 64 blocks include one null block, so 16-token blocks leave 1008 usable tokens for a declared 1024-token context. It violates that named capacity contract. Stock NIXL pull also lacks the proposed post-read lease guard.
- `sglang-unobserved-capacity.json`: a configured 4096-token ceiling does not establish that a 1024-token context fits the profiled pool. Without saved startup capacity, that rule is unknown.
- `ascend-declared-capacity.json`: an exact source pair and ordinary declared geometry permit planned-capacity arithmetic. Startup allocation remains unobserved. The inspected Mooncake V2 baseline lacks the proposed hold verifier.
- `lmcache-unverified-layout.json`: exact versions do not establish whether the reported packed-layout branch is used. Its compatibility rule stays unknown.

No file contains a serving command, request, failure schedule or live-condition driver. Replace every identity and setting with collected local facts before assessing a deployment. See [the contract guide](../../docs/CONFIG_SAFETY.md).
