Pinned original function excerpts for bounded CPU component conformance.

Each .py.txt contains the exact original method source, including its body and
indentation. The harness dedents it and enables postponed annotations, then
supplies controlled CPU mocks for imports/dependencies. MANIFEST.json records
the full original source file SHA256, Git blob SHA, commit, line range, excerpt
SHA256 and canonical source link. No upstream method body is modified.

SGLang: copyright the SGLang project contributors; Apache License 2.0, reproduced
in sglang-LICENSE.txt. vLLM: copyright the vLLM project contributors; Apache
License 2.0, reproduced in vllm-LICENSE.txt. Canonical license sources and hashes
are in MANIFEST.json. These excerpts are redistributed under those licenses.
Original vLLM notices: SPDX-License-Identifier: Apache-2.0;
SPDX-FileCopyrightText: Copyright contributors to the vLLM project.

The SGLang sequence follows the upstream ordinary holds-until-drained CPU test:
one room, an already-drained sender per required rank, two matching generation
ACKs, one release. It never exercises stale ACKs, room reuse or timeout release.
Its generation identifies a protocol hold of a metadata slot, not GPU pages.

The vLLM sequence uses original allocation/free, one homogeneous READ, PROC then
DONE, and release. CPU mocks stand in for cache coordinator, free queue, block
objects, descriptor computation and transport. There is one request, one pool,
one observer process and no sharing, retry, reused ID, failed transfer, mixed
memory path or second request. This is source-fragment conformance, not a full
upstream test suite, engine installation, live deployment or GPU evidence.

Negative evaluator cases corrupt already-recorded inert receipts only. No new
unsafe ordering, cross-request mechanism, server, network or GPU is executed.
