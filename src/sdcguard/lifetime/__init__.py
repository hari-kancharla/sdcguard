"""Lifetime checks for inference state: does every request read and write only the KV memory it currently owns?

A KV-cache block is a physical region (a worker's block index) that holds one request's state for a while and then
another's. Each allocation of a region gets a new generation, so "block 42 holding request A" and "block 42 later
holding request B" are different handles. Asynchronous operations (a prefill worker writing KV into a decode worker's
blocks, a decode worker pulling KV from a prefill worker, a host-to-GPU load from an offload tier) refer to handles,
and finish later. The bugs this finds are the ones where an operation outlives the allocation it referred to: a write
that lands after the block was given to another request, a read of a block the producer already reused, a completion
that marks the wrong allocation ready, a request that runs on state whose transfer has not finished. The arithmetic can
be perfectly correct while computing on the wrong data, and correct cache keys at lookup do not prevent it.

Four parts:
  ledger    an independent record of allocations, generations, permitted sharing and outstanding operations, with the
            four lifetime rules checked as events arrive (ledger.Ledger).
  explore   a controller for the order of events in a model of a serving protocol (or, through an adapter, in a real
            server): it searches the legal orderings of conflicting events, stops at the first broken rule, and
            reduces the trace to the events that caused it (explore.explore).
  models    small protocol models of real failure families (prefill/decode abort with writes in flight, a producer
            lease that expires under a delayed consumer read, completions matched to the wrong allocation, a load that
            is consumed before it lands), each with a fixed variant (models.MODELS).

  rules     explicit safety rules for KV handed between engines (a pulled block's lease must hold through the read; a
            read notification releases only its own lease), judged per request attempt from facts the adapters record,
            with unknown when a fact was not observed (obligations.evaluate; `sdcguard lifetime rules DIR`).

A finding in a model is a candidate: it says the protocol as modeled allows the ordering. It becomes a finding about a
server only when the same ordering is driven in the real implementation, which is what the adapter interface is for
(adapter.py)."""

from .ledger import Handle, Ledger, Violation, RULES  # noqa: F401
from .explore import Finding, Result, explore  # noqa: F401
