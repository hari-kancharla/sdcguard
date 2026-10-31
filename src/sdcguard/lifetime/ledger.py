"""The lifetime ledger: allocations, generations, outstanding operations, and the four rules.

Vocabulary.
  region      a physical piece of KV memory, named (pool, index): ("decode", 3) is block 3 of the decode worker's pool.
  generation  a counter per region that goes up at every allocation. A handle (region, generation) names one
              allocation; it stays the name of that allocation after the region is freed and given to someone else.
  owner       the request an allocation was made for. Permitted sharing (a prefix block reused by another request with
              the same tokens and keys) is declared with share(); a request may consume a region it owns or shares.
  operation   an asynchronous access that refers to handles: a write into a destination handle, a read from a source
              handle into a destination handle, a load (an offload tier into GPU memory). It is outstanding from
              submit until it completes or until the transport confirms it has stopped. A cancel request alone does
              not stop it: "cancellation requested" is not "writes stopped".
  content     what a region holds: a tag (owner, generation, label) set by local writes and by landed operations. A
              consume checks it, so a corruption is seen even when no rule caught its cause.

The rules (RULES), checked as each event is recorded:
  R1 stale_handle          an operation is submitted, or a descriptor accepted, for an allocation that is no longer
                           current (the region was freed or reallocated since the handle was issued).
  R2 reuse_outstanding     a region is reallocated while an older operation that refers to it can still access it (a
                           late write would corrupt the new owner; a late read would hand the new owner's data to the
                           old consumer).
  R3 consume_unready       a request consumes state that is not current, not its own or shared, still has a transfer
                           writing into it, or holds content that belongs to someone else.
  R4 stale_completion      a completion of an operation is applied to an allocation the operation did not refer to
                           (it marks a newer allocation ready).
Allowed outcomes are not violations: a request that fails or is aborted after losing its lease, a rejected transfer, a
free of a region with nothing outstanding.

The ledger never decides what a server should do; it records what it did and says which rule an event broke. Events
carry the step number given by the caller (the explorer's step, or a real server's event sequence)."""

from __future__ import annotations

from dataclasses import dataclass, field

RULES = {
    "R1": "stale_handle: an operation or descriptor used an allocation that is no longer current",
    "R2": "reuse_outstanding: a region was reallocated while an older operation could still access it",
    "R3": "consume_unready: state was consumed before it was current, ready and the consumer's own",
    "R4": "stale_completion: a completion was applied to an allocation its operation did not refer to",
}


@dataclass(frozen=True)
class Handle:
    region: tuple
    gen: int

    def __str__(self) -> str:
        return f"{self.region[0]} block {self.region[1]} gen {self.gen}"


@dataclass
class Region:
    gen: int = 0
    owner: str | None = None          # None: free
    sharers: set = field(default_factory=set)
    content: tuple | None = None      # (owner, gen, label) of the data it holds
    ready: bool = False


@dataclass
class Op:
    op_id: str
    kind: str                          # "write" (into dst), "read" (src -> dst), "load" (tier -> dst)
    src: Handle | None
    dst: Handle | None
    by: str
    payload: tuple | None = None       # for a write or load: the content it carries
    state: str = "outstanding"         # outstanding | completed | stopped
    cancel_requested: bool = False
    step: int = 0

    def touches(self, region) -> list:
        """The handles of this operation on region, with the access each makes ("write" or "read")."""
        out = []
        if self.dst is not None and self.dst.region == region:
            out.append((self.dst, "write"))
        if self.src is not None and self.src.region == region:
            out.append((self.src, "read"))
        return out


@dataclass
class Violation:
    rule: str
    step: int
    message: str
    regions: tuple = ()
    ops: tuple = ()
    requests: tuple = ()

    def key(self) -> tuple:
        """What makes two violations the same finding: the rule and the shape, not the step or names."""
        return (self.rule, self.message.split(":")[0])


@dataclass
class Event:
    step: int
    kind: str
    text: str
    regions: tuple = ()
    ops: tuple = ()
    requests: tuple = ()


class Ledger:
    """Record lifetime events and check the rules as they arrive. Violations are collected, not raised."""

    def __init__(self, track_content: bool = True):
        self.track_content = track_content   # False for a real server, whose block contents are not recorded
        self.regions: dict = {}
        self.ops: dict = {}
        self.descriptors: dict = {}
        self.events: list = []
        self.violations: list = []

    # -- helpers ---------------------------------------------------------------------------------------------
    def _r(self, region) -> Region:
        if region not in self.regions:
            self.regions[region] = Region()
        return self.regions[region]

    def current(self, region) -> Handle:
        return Handle(region, self._r(region).gen)

    def _log(self, step, kind, text, regions=(), ops=(), requests=()):
        self.events.append(Event(step, kind, text, tuple(regions), tuple(ops), tuple(requests)))

    def _violate(self, rule, step, message, regions=(), ops=(), requests=()):
        v = Violation(rule, step, message, tuple(regions), tuple(ops), tuple(requests))
        self.violations.append(v)
        self._log(step, "violation", f"{rule} {message}", regions, ops, requests)
        return v

    def _is_current(self, h: Handle) -> bool:
        r = self._r(h.region)
        return r.owner is not None and r.gen == h.gen

    def _stale_note(self, h: Handle) -> str:
        r = self._r(h.region)
        if r.owner is None:
            return f"{h} was freed (region now free, gen {r.gen})"
        return f"{h} was retired; the region is now gen {r.gen} owned by {r.owner}"

    def outstanding(self, region=None) -> list:
        ops = [o for o in self.ops.values() if o.state == "outstanding"]
        if region is None:
            return ops
        return [o for o in ops if o.touches(region)]

    # -- allocation ------------------------------------------------------------------------------------------
    def alloc(self, region, owner, step=0) -> Handle:
        """Give region to owner: a new generation. R2: no older operation may still access the region."""
        r = self._r(region)
        for o in self.outstanding(region):
            for h, access in o.touches(region):
                if h.gen <= r.gen:
                    what = "write into" if access == "write" else "read from"
                    self._violate("R2", step,
                                  f"reuse while {o.op_id} can still {what} it: {region[0]} block {region[1]} "
                                  f"reallocated from gen {r.gen} ({r.owner or 'free'}) to gen {r.gen + 1} ({owner}) "
                                  f"while {o.op_id} ({o.kind}, by {o.by}, cancel requested: {o.cancel_requested}) "
                                  f"still refers to gen {h.gen}",
                                  regions=[region], ops=[o.op_id], requests=[owner, o.by])
        r.gen += 1
        r.owner = owner
        r.sharers = {owner}
        r.content = None
        r.ready = False
        h = Handle(region, r.gen)
        self._log(step, "alloc", f"{h} allocated to {owner}", [region], requests=[owner])
        return h

    def free(self, region, by, step=0):
        r = self._r(region)
        self._log(step, "free", f"{region[0]} block {region[1]} gen {r.gen} freed by {by}", [region], requests=[by])
        r.owner = None
        r.sharers = set()
        r.ready = False

    def share(self, region, request, step=0):
        """Permitted sharing: request may consume region's current allocation (a prefix-cache hit)."""
        r = self._r(region)
        r.sharers.add(request)
        self._log(step, "share", f"{self.current(region)} shared with {request}", [region], requests=[request])

    def write_local(self, region, label, step=0):
        """A local computation writes the region (a prefill forward pass filling its own blocks)."""
        r = self._r(region)
        r.content = (r.owner, r.gen, label)
        self._log(step, "write", f"{self.current(region)} written locally ({label})", [region], requests=[r.owner])

    # -- descriptors and operations --------------------------------------------------------------------------
    def publish(self, desc_id, handles, by, step=0):
        """Descriptor issuance: by makes handles available to another worker."""
        self.descriptors[desc_id] = tuple(handles)
        self._log(step, "publish", f"descriptor {desc_id} by {by}: " + ", ".join(map(str, handles)),
                  [h.region for h in handles], requests=[by])

    def accept(self, desc_id, by, step=0) -> bool:
        """A consumer accepts a descriptor as current state. R1 if any of its handles is retired."""
        ok = True
        for h in self.descriptors.get(desc_id, ()):
            if not self._is_current(h):
                ok = False
                self._violate("R1", step, f"stale descriptor accepted: {by} accepted {desc_id}; "
                              f"{self._stale_note(h)}", [h.region], requests=[by])
        self._log(step, "accept", f"descriptor {desc_id} accepted by {by}", requests=[by])
        return ok

    def submit(self, op_id, kind, by, src=None, dst=None, payload=None, step=0) -> Op:
        """An asynchronous operation starts. R1 if it refers to an allocation that is no longer current."""
        o = Op(op_id, kind, src, dst, by, payload, step=step)
        self.ops[op_id] = o
        for h, role in ((src, "source"), (dst, "destination")):
            if h is not None and not self._is_current(h):
                self._violate("R1", step, f"stale handle submitted: {op_id} ({kind} by {by}) uses {role} "
                              f"{self._stale_note(h)}", [h.region], ops=[op_id], requests=[by])
        parts = [f"{op_id} submitted by {by}: {kind}"]
        if src is not None:
            parts.append(f"from {src}")
        if dst is not None:
            parts.append(f"into {dst}")
        self._log(step, "submit", " ".join(parts), [h.region for h in (src, dst) if h is not None], [op_id], [by])
        return o

    def cancel(self, op_id, by, step=0):
        """Cancellation requested. The operation stays outstanding until stopped() or complete()."""
        o = self.ops[op_id]
        o.cancel_requested = True
        self._log(step, "cancel", f"{op_id} cancel requested by {by} (still outstanding)", ops=[op_id], requests=[by])

    def stopped(self, op_id, step=0):
        """The transport confirms the operation will not touch memory any more."""
        o = self.ops[op_id]
        if o.state == "outstanding":
            o.state = "stopped"
        self._log(step, "stopped", f"{op_id} confirmed stopped", ops=[op_id])

    def complete(self, op_id, step=0, marks_ready=None):
        """The operation's data lands and its completion is delivered. A write or load sets the destination's
        content; a read copies the source's content as it is now (another owner's data if the source was reused).
        marks_ready: the region the server marks ready because of this completion; R4 if that region's current
        allocation is not the one the operation referred to."""
        o = self.ops[op_id]
        if o.state != "outstanding":
            self._log(step, "complete", f"{op_id} completion after {o.state} (ignored)", ops=[op_id])
            return
        o.state = "completed"
        regions = []
        if o.dst is not None:
            d = self._r(o.dst.region)
            regions.append(o.dst.region)
            if o.kind == "read":
                s = self._r(o.src.region)
                data = s.content
                if s.gen != o.src.gen:
                    self._log(step, "effect", f"{op_id} read {o.src.region[0]} block {o.src.region[1]} after it moved "
                              f"to gen {s.gen} ({s.owner or 'free'}): it carried "
                              f"{_content_owner(data)}'s data", [o.src.region], [op_id], [o.by])
            else:
                data = o.payload
            if d.gen == o.dst.gen:
                d.content = data
            else:
                d.content = data           # the bytes land in whatever the region holds now
                self._log(step, "effect", f"{op_id} landed in {o.dst.region[0]} block {o.dst.region[1]} gen {d.gen} "
                          f"({d.owner or 'free'}), meant for gen {o.dst.gen}", [o.dst.region], [op_id],
                          [d.owner or "", o.by])
        self._log(step, "complete", f"{op_id} completed", regions, [op_id], [o.by])
        if marks_ready is not None:
            self.apply_completion(op_id, marks_ready, step)

    def land_only(self, op_id, step=0):
        """The operation's data lands and the transport retires it, but nothing is marked ready yet (the server
        learns of it later, for example through a notification on another channel: apply_completion)."""
        self.complete(op_id, step)

    def apply_completion(self, op_id, region, step=0):
        """The server applies op_id's completion to region (marks it ready). R4 if region's current allocation is
        not the one the operation referred to."""
        o = self.ops[op_id]
        r = self._r(region)
        meant = o.dst.region if o.dst is not None else None
        if meant != region or (o.dst is not None and r.gen != o.dst.gen):
            self._violate("R4", step, f"stale completion applied: {op_id} (for "
                          f"{o.dst if o.dst is not None else 'no destination'}) marked {region[0]} block "
                          f"{region[1]} gen {r.gen} ({r.owner or 'free'}) ready",
                          [region], [op_id], [r.owner or "", o.by])
        r.ready = True
        self._log(step, "ready", f"{self.current(region)} marked ready by {op_id}'s completion", [region], [op_id])

    def mark_ready(self, region, by, step=0):
        r = self._r(region)
        r.ready = True
        self._log(step, "ready", f"{self.current(region)} marked ready by {by}", [region], requests=[by])

    # -- consumption -----------------------------------------------------------------------------------------
    def consume(self, request, handles, step=0) -> bool:
        """The model runs request on these handles. R3 if any is stale, not the request's own or shared, still
        being written, or holds another owner's content."""
        ok = True
        for h in handles:
            r = self._r(h.region)
            if not self._is_current(h):
                ok = False
                self._violate("R3", step, f"consumed a retired allocation: {request} ran on {self._stale_note(h)}",
                              [h.region], requests=[request])
                continue
            if request not in r.sharers:
                ok = False
                self._violate("R3", step, f"consumed state it does not own or share: {request} ran on {h} "
                              f"(owner {r.owner})", [h.region], requests=[request, r.owner])
            writers = [o for o in self.outstanding(h.region)
                       if any(acc == "write" and hh.gen == h.gen for hh, acc in o.touches(h.region))]
            if writers:
                ok = False
                self._violate("R3", step, f"consumed while a transfer is still writing it: {request} ran on {h} "
                              f"before {', '.join(o.op_id for o in writers)} completed",
                              [h.region], [o.op_id for o in writers], [request])
            if not self.track_content:
                continue
            owner = _content_owner(r.content)
            if r.content is not None and owner not in r.sharers:
                ok = False
                self._violate("R3", step, f"consumed another request's content: {request} ran on {h}, which holds "
                              f"{owner}'s data ({r.content[2]})", [h.region], requests=[request, owner])
            elif r.content is None and not writers:
                ok = False
                self._violate("R3", step, f"consumed state that was never written: {request} ran on {h}",
                              [h.region], requests=[request])
        self._log(step, "consume", f"{request} consumed " + ", ".join(map(str, handles)),
                  [h.region for h in handles], requests=[request])
        return ok


def _content_owner(content):
    return None if content is None else content[0]
