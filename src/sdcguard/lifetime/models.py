"""Protocol models of KV-state lifetime, one per failure family seen in real serving stacks, each in a buggy and a
fixed variant. The explorer finds the buggy variants' orderings without any bug-specific script: the model states the
protocol (who allocates, publishes, transfers, times out, frees and runs), and the explorer chooses the order.

pd-abort     prefill/decode disaggregation, the prefill side pushes KV into blocks the decode side allocated. The
             decode side aborts a request whose prefill is late. Buggy: it frees the blocks at once, so a later
             request gets them while the old write is queued or in flight (the race Z.ai described in GLM-5 serving,
             April 2026). Fixed: abort is sent to the prefill side, and the blocks are reclaimed only after it answers
             that no write was issued or that every issued write has completed or stopped. release-timeout:
             the fixed handshake, but the decode side gives up waiting after a drain timeout and frees the blocks
             anyway (a liveness fallback; SGLang v0.5.22's decode-side deferred release does this for device
             destinations, logging a warning, and notes for host destinations that "A timeout cannot prove that a
             remote write has stopped"). The explorer shows the ordering that fallback permits.
pd-lease     the decode side pulls KV from blocks the prefill side holds for it under a lease (vLLM's NIXL read
             path). Buggy: the lease expires and the producer reuses its blocks while the consumer's read is delayed
             or in flight (the sequence in vLLM issue 61229). Fixed: the consumer pins the blocks before reading; the
             producer never expires a pinned lease, and answers "expired" to a late pin, so the request fails
             instead of reading someone else's KV (an allowed outcome).
pd-notify    transfer-done notifications and control messages travel on separate channels with no order between
             them, and the decode side matches a notification to a request by its id. Buggy: ids can be reused (a
             client-supplied id, or a retry), so a late notification for an aborted request marks the new request's
             blocks ready before its own transfer lands. Fixed: notifications carry the transfer's id and the
             allocation generation, and stale ones are dropped.
offload-load a prefix-cache hit loads KV from a host tier into a GPU block on a load stream while the forward stream
             runs. Buggy: the forward pass may start once the load is issued (the HiCache read-before-ready fix in
             SGLang PR 22811). Fixed: it waits for the load's completion.

Each model lists its assumptions (what the transport and memory are taken to guarantee) and what it does not cover.
A model finding is a candidate ordering; a server finding needs the same ordering driven in the real implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .explore import Action
from .ledger import Handle, Ledger


def _ledger_sig(led: Ledger) -> tuple:
    regs = tuple(sorted((k, r.gen, r.owner, tuple(sorted(r.sharers)), r.content, r.ready)
                        for k, r in led.regions.items()))
    ops = tuple(sorted((k, o.state, o.cancel_requested) for k, o in led.ops.items()))
    return regs, ops, len(led.violations)


class _Base:
    name = ""
    variant = "buggy"
    VARIANTS: tuple = ("buggy", "fixed")
    assumptions: tuple = ()
    not_covered: tuple = ()
    ORDER: tuple = ()

    def __init__(self, variant: str = "buggy"):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {', '.join(self.VARIANTS)}")
        self.variant = variant
        self.name = f"{type(self).key}:{variant}"

    def _sorted(self, acts):
        rank = {a: i for i, a in enumerate(self.ORDER)}
        return [a for _, _, a in sorted(((rank.get(a.actor, 99), i, a) for i, a in enumerate(acts)),
                                        key=lambda t: (t[0], t[1]))]

    @staticmethod
    def _act(label, actor, footprint, fn):
        return Action(label, actor, frozenset(footprint), fn)


# ----------------------------------------------------------------------------------------------------------------
@dataclass
class _PDState:
    ledger: Ledger = field(default_factory=Ledger)
    time: int = 0
    req: dict = field(default_factory=dict)      # request -> dict of its progress
    to_p: list = field(default_factory=list)     # messages decode -> prefill
    to_d: list = field(default_factory=list)     # messages prefill -> decode (control)
    free: list = field(default_factory=list)     # free decode blocks


class PDAbort(_Base):
    """Push model. One decode block, two requests: A (arrives at 0, decode-side timeout 1 after admission) and B
    (arrives at 2). The prefill side receives A's descriptor, runs its forward pass, then issues an RDMA write into
    the descriptor's block."""
    key = "pd-abort"
    VARIANTS = ("buggy", "fixed", "release-timeout")
    ORDER = ("decode", "prefill", "transport", "clock")
    assumptions = ("a write's bytes land when it completes; nothing lands after the transport confirms it stopped",
                   "messages between the workers are delivered in order on one channel",
                   "the decode side frees a request's blocks only on abort (buggy) or on the prefill side's answer "
                   "(fixed)")
    not_covered = ("partial landing of one write across several blocks", "multiple prefill workers",
                   "GPU stream ordering on the decode side after completion")

    ARRIVE = {"A": 0, "B": 2}
    TIMEOUT = 1
    RELEASE_TIMEOUT = 1      # release-timeout variant: give up waiting for the prefill side's answer after this

    def initial(self):
        s = _PDState()
        s.free = [("decode", 0)]
        for r in self.ARRIVE:
            s.req[r] = {"phase": "waiting", "block": None, "deadline": None, "op": None, "aborting": False}
        return s

    def done(self, s):
        return all(v["phase"] in ("finished", "aborted") for v in s.req.values()) and not s.to_p and not s.to_d \
            and not s.ledger.outstanding()

    def signature(self, s):
        return (s.time, tuple(sorted((k, tuple(sorted((kk, str(vv)) for kk, vv in v.items())))
                                     for k, v in s.req.items())),
                tuple(map(str, s.to_p)), tuple(map(str, s.to_d)), tuple(s.free), _ledger_sig(s.ledger))

    def enabled(self, s):
        acts = []
        led = s.ledger
        for r, v in s.req.items():
            ph = v["phase"]
            if ph == "waiting" and s.time >= self.ARRIVE[r] and s.free:
                def admit(st, step, r=r):
                    st_v = st.req[r]
                    blk = st.free.pop(0)
                    h = st.ledger.alloc(blk, r, step)
                    st_v.update(phase="admitted", block=h, deadline=st.time + self.TIMEOUT)
                    st.ledger.publish(f"desc-{r}", [h], r, step)
                    st.to_p.append(("work", r, h))
                acts.append(self._act(f"decode admits {r}", "decode", {r, ("decode", 0)}, admit))
            if ph == "ready":
                def run(st, step, r=r):
                    st_v = st.req[r]
                    st.ledger.consume(r, [st_v["block"]], step)
                    st.ledger.free(st_v["block"].region, r, step)
                    st.free.append(st_v["block"].region)
                    st_v["phase"] = "finished"
                acts.append(self._act(f"decode runs {r}", "decode", {r, ("decode", 0)}, run))
            if ph == "admitted" and not v["aborting"] and v["deadline"] is not None and s.time >= v["deadline"]:
                def timeout(st, step, r=r):
                    st_v = st.req[r]
                    if self.variant == "buggy":
                        st.ledger.free(st_v["block"].region, r, step)
                        st.free.append(st_v["block"].region)
                        st_v["phase"] = "aborted"
                        st.to_p.append(("abort", r, None))       # told, but nothing waits for an answer
                    else:
                        st_v["aborting"] = True
                        st_v["release_at"] = st.time + self.RELEASE_TIMEOUT
                        st.to_p.append(("abort", r, None))
                acts.append(self._act(f"decode times out {r}", "decode", {r, ("decode", 0)}, timeout))
        for r, v in s.req.items():
            if self.variant == "release-timeout" and v["aborting"] and v["phase"] == "admitted" \
                    and s.time >= v.get("release_at", 1 << 30):
                def give_up(st, step, r=r):
                    st_v = st.req[r]
                    st.ledger.free(st_v["block"].region, r, step)
                    st.free.append(st_v["block"].region)
                    st_v["phase"] = "aborted"
                acts.append(self._act(f"decode releases {r} after its drain timeout (no answer yet)", "decode",
                                      {r, ("decode", 0)}, give_up))
        if s.to_d:
            msg = s.to_d[0]
            if msg[0] == "safe" and s.req[msg[1]]["phase"] != "aborted":
                r = msg[1]

                def reclaim(st, step, r=r):
                    st.to_d.pop(0)
                    st_v = st.req[r]
                    st.ledger.free(st_v["block"].region, r, step)
                    st.free.append(st_v["block"].region)
                    st_v["phase"] = "aborted"
                acts.append(self._act(f"decode reclaims {r} (prefill said safe)", "decode", {r, ("decode", 0)},
                                      reclaim))
        # prefill side
        if s.to_p:
            msg = s.to_p[0]
            kind, r, h = msg
            if kind == "work":
                def forward(st, step, r=r, h=h):
                    st.to_p.pop(0)
                    st.req[r]["queued"] = True
                    st.req[r]["dst"] = h
                acts.append(self._act(f"prefill receives {r}'s work", "prefill", {r}, forward))
            else:
                def got_abort(st, step, r=r):
                    st.to_p.pop(0)
                    st_v = st.req[r]
                    st_v["queued"] = False
                    if self.variant == "fixed":
                        op = st_v.get("op")
                        if op is not None and st.ledger.ops[op].state == "outstanding":
                            st.ledger.cancel(op, "prefill", step)
                            st_v["safe_pending"] = True
                        else:
                            st.to_d.append(("safe", r, None))
                    else:
                        op = st_v.get("op")
                        if op is not None and st.ledger.ops[op].state == "outstanding":
                            st.ledger.cancel(op, "prefill", step)
                acts.append(self._act(f"prefill receives {r}'s abort", "prefill", {r}, got_abort))
        for r, v in s.req.items():
            if v.get("queued") and v.get("op") is None:
                def issue(st, step, r=r):
                    st_v = st.req[r]
                    st_v["queued"] = False
                    op = f"write-{r}"
                    st_v["op"] = op
                    st.ledger.submit(op, "write", "prefill", dst=st_v["dst"], payload=(r, 0, f"{r}'s KV"), step=step)
                acts.append(self._act(f"prefill computes {r} and issues its write", "prefill",
                                      {r, ("decode", 0)}, issue))
        # transport
        for r, v in s.req.items():
            op = v.get("op")
            if op is not None and s.ledger.ops[op].state == "outstanding":
                def land(st, step, r=r, op=op):
                    st_v = st.req[r]
                    aborted = st_v["phase"] == "aborted" or st_v["aborting"]
                    st.ledger.complete(op, step)
                    if not aborted and st_v["phase"] == "admitted":
                        st.ledger.mark_ready(st_v["block"].region, "decode", step)
                        st_v["phase"] = "ready"
                    if st_v.get("safe_pending"):
                        st_v["safe_pending"] = False
                        st.to_d.append(("safe", r, None))
                acts.append(self._act(f"{op} lands", "transport", {r, ("decode", 0), op}, land))
        # clock
        pending = any(v["phase"] == "waiting" and self.ARRIVE[k] > s.time for k, v in s.req.items()) or \
            any(v["phase"] == "admitted" and v["deadline"] is not None and v["deadline"] > s.time
                for v in s.req.values()) or \
            any(v["aborting"] and v["phase"] == "admitted" and v.get("release_at", -1) > s.time
                for v in s.req.values())
        if pending:
            def tick(st, step):
                st.time += 1
            acts.append(self._act("time passes", "clock", {"*"}, tick))
        return self._sorted(acts)


# ----------------------------------------------------------------------------------------------------------------
class PDLease(_Base):
    """Pull model. One prefill block, two requests: X (prefilled at 0) and Y (arrives at the prefill side at 2). The
    prefill side holds X's block under a lease of 1 time unit; the decode side allocates its own block and reads
    X's KV from the descriptor's handle."""
    key = "pd-lease"
    ORDER = ("decode", "prefill", "transport", "clock")
    assumptions = ("a read copies the source block's bytes as they are when it completes",
                   "the producer cannot observe a one-sided read, so it learns about the consumer only from messages",
                   "messages are delivered in order on one channel per direction")
    not_covered = ("reads that copy part of a block before the reuse and part after", "several consumers per block")

    ARRIVE_Y = 2
    LEASE = 1

    def initial(self):
        s = _PDState()
        s.free = [("prefill", 0)]
        s.req = {"X": {"phase": "new", "pblock": None, "dblock": None, "lease": None, "op": None, "pinned": False},
                 "Y": {"phase": "waiting"}}
        return s

    def done(self, s):
        x, y = s.req["X"], s.req["Y"]
        return x["phase"] in ("finished", "failed") and y["phase"] == "prefilled" and not s.to_p and not s.to_d \
            and not s.ledger.outstanding()

    def signature(self, s):
        return (s.time, tuple(sorted((k, tuple(sorted((kk, str(vv)) for kk, vv in v.items())))
                                     for k, v in s.req.items())),
                tuple(map(str, s.to_p)), tuple(map(str, s.to_d)), tuple(s.free), _ledger_sig(s.ledger))

    def enabled(self, s):
        acts = []
        x, y = s.req["X"], s.req["Y"]
        P = ("prefill", 0)
        # prefill side
        if x["phase"] == "new" and s.free:
            def prefill_x(st, step):
                v = st.req["X"]
                st.free.pop(0)
                h = st.ledger.alloc(P, "X", step)
                st.ledger.write_local(P, "X's KV", step)
                st.ledger.publish("desc-X", [h], "X", step)
                v.update(phase="published", pblock=h, lease=st.time + self.LEASE)
                st.to_d.append(("desc", "X", h))
            acts.append(self._act("prefill runs X and publishes its block", "prefill", {"X", P}, prefill_x))
        if y["phase"] == "waiting" and s.time >= self.ARRIVE_Y and s.free:
            def prefill_y(st, step):
                st.free.pop(0)
                st.ledger.alloc(P, "Y", step)
                st.ledger.write_local(P, "Y's KV", step)
                st.req["Y"]["phase"] = "prefilled"
            acts.append(self._act("prefill runs Y in the free block", "prefill", {"Y", P}, prefill_y))
        if x["phase"] in ("published",) and x["lease"] is not None and s.time >= x["lease"] and not x["pinned"]:
            def expire(st, step):
                v = st.req["X"]
                st.ledger.free(P, "X", step)
                st.free.append(P)
                v["lease"] = None
                v["phase"] = "expired" if self.variant == "fixed" else "published-expired"
            acts.append(self._act("prefill's lease on X's block expires", "prefill", {"X", P}, expire))
        if s.to_p:
            kind, r, _ = s.to_p[0]
            if kind == "done":
                def got_done(st, step):
                    st.to_p.pop(0)
                    v = st.req["X"]
                    if v["phase"] in ("published", "pinned-wait") or v["pinned"]:
                        st.ledger.free(P, "X", step)
                        st.free.append(P)
                        v["pinned"] = False
                        v["lease"] = None
                acts.append(self._act("prefill gets X's done and frees its block", "prefill", {"X", P}, got_done))
            elif kind == "pin":
                def got_pin(st, step):
                    st.to_p.pop(0)
                    v = st.req["X"]
                    if v["phase"] == "published":
                        v["pinned"] = True
                        st.to_d.append(("pinned", "X", None))
                    else:
                        st.to_d.append(("expired", "X", None))
                acts.append(self._act("prefill answers X's pin", "prefill", {"X", P}, got_pin))
        # decode side
        D = ("decode", 0)
        if s.to_d:
            kind, r, h = s.to_d[0]
            if kind == "desc":
                def got_desc(st, step, h=h):
                    st.to_d.pop(0)
                    v = st.req["X"]
                    v["dblock"] = st.ledger.alloc(D, "X", step)
                    v["src"] = h
                    if self.variant == "fixed":
                        st.to_p.append(("pin", "X", None))
                        v["dphase"] = "pinning"
                    else:
                        v["dphase"] = "can_read"
                acts.append(self._act("decode gets X's descriptor and allocates its block", "decode", {"X", D},
                                      got_desc))
            elif kind == "pinned":
                def got_pinned(st, step):
                    st.to_d.pop(0)
                    st.req["X"]["dphase"] = "can_read"
                acts.append(self._act("decode learns X's block is pinned", "decode", {"X"}, got_pinned))
            elif kind == "expired":
                def got_expired(st, step):
                    st.to_d.pop(0)
                    v = st.req["X"]
                    st.ledger.free(D, "X", step)
                    v["dphase"] = "done"
                    v["phase"] = "failed"          # an allowed outcome: the request fails, nobody reads stale KV
                acts.append(self._act("decode fails X (lease lost before pin)", "decode", {"X", D}, got_expired))
        if x.get("dphase") == "can_read" and x["op"] is None:
            def read(st, step):
                v = st.req["X"]
                v["op"] = "read-X"
                st.ledger.accept("desc-X", "decode", step)
                st.ledger.submit("read-X", "read", "decode", src=v["src"], dst=v["dblock"], step=step)
            acts.append(self._act("decode reads X's KV from the descriptor", "decode", {"X", P, D}, read))
        if x.get("dphase") == "ready":
            def run_x(st, step):
                v = st.req["X"]
                st.ledger.consume("X", [v["dblock"]], step)
                st.ledger.free(D, "X", step)
                st.to_p.append(("done", "X", None))
                v["dphase"] = "done"
                v["phase"] = "finished"
            acts.append(self._act("decode runs X", "decode", {"X", D}, run_x))
        # transport
        if x["op"] is not None and s.ledger.ops["read-X"].state == "outstanding":
            def land(st, step):
                v = st.req["X"]
                st.ledger.complete("read-X", step, marks_ready=D)
                v["dphase"] = "ready"
            acts.append(self._act("read-X completes", "transport", {"X", P, D, "read-X"}, land))
        # clock
        pending = (y["phase"] == "waiting" and s.time < self.ARRIVE_Y) or \
            (x["phase"] == "published" and x["lease"] is not None and s.time < x["lease"])
        if pending:
            def tick(st, step):
                st.time += 1
            acts.append(self._act("time passes", "clock", {"*"}, tick))
        return self._sorted(acts)


# ----------------------------------------------------------------------------------------------------------------
class PDNotify(_Base):
    """Two decode blocks; request A (id r1) is admitted, its write lands, then it times out; the client reuses id r1
    for request B. Transfer-done notifications (notif channel) and the prefill side's answers (control channel) are
    two FIFO channels with no order between them."""
    key = "pd-notify"
    ORDER = ("decode", "prefill", "transport", "notif", "control", "clock")
    assumptions = ("each channel is FIFO; the two channels are not ordered with each other",
                   "a write's completion is reported to the decode side only by a notification on the notif channel")
    not_covered = ("lost or duplicated notifications",)

    TIMEOUT = 1

    def initial(self):
        s = _PDState()
        s.free = [("decode", 0), ("decode", 1)]
        s.req = {"A": {"phase": "waiting", "rid": "r1"}, "B": {"phase": "later", "rid": "r1"}}
        s.notif = []
        s.by_rid = {}
        return s

    def done(self, s):
        return s.req["A"]["phase"] in ("aborted", "finished") and s.req["B"]["phase"] == "finished" and not s.notif \
            and not s.to_d and not s.to_p and not s.ledger.outstanding()

    def signature(self, s):
        return (s.time, tuple(sorted((k, tuple(sorted((kk, str(vv)) for kk, vv in v.items())))
                                     for k, v in s.req.items())),
                tuple(map(str, s.to_p)), tuple(map(str, s.to_d)), tuple(map(str, s.notif)), tuple(s.free),
                tuple(sorted(s.by_rid.items())), _ledger_sig(s.ledger))

    def enabled(self, s):
        acts = []
        for r in ("A", "B"):
            v = s.req[r]
            if (v["phase"] == "waiting" or (r == "B" and v["phase"] == "later"
                                            and s.req["A"]["phase"] in ("aborted", "finished"))) and s.free:
                def admit(st, step, r=r):
                    vv = st.req[r]
                    blk = st.free.pop(0)
                    h = st.ledger.alloc(blk, r, step)
                    gen = h.gen
                    vv.update(phase="admitted", block=h, deadline=st.time + self.TIMEOUT if r == "A" else None)
                    st.by_rid[vv["rid"]] = r
                    st.to_p.append(("work", r, h, gen))
                acts.append(self._act(f"decode admits {r} as {s.req[r]['rid']}", "decode", {r, "rid"}, admit))
            if v["phase"] == "ready":
                def run(st, step, r=r):
                    vv = st.req[r]
                    st.ledger.consume(r, [vv["block"]], step)
                    st.ledger.free(vv["block"].region, r, step)
                    st.free.append(vv["block"].region)
                    vv["phase"] = "finished"
                acts.append(self._act(f"decode runs {r}", "decode", {r}, run))
            if v["phase"] == "admitted" and v.get("deadline") is not None and s.time >= v["deadline"] \
                    and not v.get("aborting"):
                def timeout(st, step, r=r):
                    st.req[r]["aborting"] = True
                    st.to_p.append(("abort", r, None, None))
                acts.append(self._act(f"decode times out {r}", "decode", {r}, timeout))
        if s.to_d:
            kind, r, _, _ = s.to_d[0]

            def reclaim(st, step, r=r):
                st.to_d.pop(0)
                vv = st.req[r]
                st.ledger.free(vv["block"].region, r, step)
                st.free.append(vv["block"].region)
                vv["phase"] = "aborted"
                if st.by_rid.get(vv["rid"]) == r:
                    del st.by_rid[vv["rid"]]
            acts.append(self._act(f"control: decode reclaims {r} (prefill said safe)", "control", {r, "rid"},
                                  reclaim))
        if s.notif:
            op, rid, gen = s.notif[0]

            def deliver(st, step, op=op, rid=rid, gen=gen):
                st.notif.pop(0)
                target = st.by_rid.get(rid)
                if target is None:
                    return
                tv = st.req[target]
                if self.variant == "fixed" and (op != tv.get("op") or gen != tv["block"].gen):
                    return                         # a stale notification: dropped
                st.ledger.apply_completion(op, tv["block"].region, step)
                if tv["phase"] == "admitted" and not tv.get("aborting"):
                    tv["phase"] = "ready"
            acts.append(self._act(f"notif: deliver {op}'s done for {rid}", "notif", {"rid", op}, deliver))
        if s.to_p:
            kind, r, h, gen = s.to_p[0]

            def got(st, step, kind=kind, r=r, h=h, gen=gen):
                st.to_p.pop(0)
                vv = st.req[r]
                if kind == "work":
                    op = f"write-{r}"
                    vv["op"] = op
                    st.ledger.submit(op, "write", "prefill", dst=h, payload=(r, 0, f"{r}'s KV"), step=step)
                    vv["wgen"] = gen
                else:
                    op = vv.get("op")
                    if op is not None and st.ledger.ops[op].state == "outstanding":
                        st.ledger.cancel(op, "prefill", step)
                        vv["safe_after"] = True
                    else:
                        st.to_d.append(("safe", r, None, None))
            acts.append(self._act(f"prefill handles {kind} for {r}", "prefill", {r}, got))
        for r in ("A", "B"):
            v = s.req[r]
            op = v.get("op")
            if op is not None and s.ledger.ops[op].state == "outstanding":
                def land(st, step, r=r, op=op):
                    vv = st.req[r]
                    st.ledger.land_only(op, step)
                    st.notif.append((op, vv["rid"], vv["wgen"]))
                    if vv.get("safe_after"):
                        vv["safe_after"] = False
                        st.to_d.append(("safe", r, None, None))
                acts.append(self._act(f"{op} lands", "transport", {r, op}, land))
        if any(v["phase"] == "admitted" and v.get("deadline") is not None and v["deadline"] > s.time
               for v in s.req.values()):
            def tick(st, step):
                st.time += 1
            acts.append(self._act("time passes", "clock", {"*"}, tick))
        return self._sorted(acts)


# ----------------------------------------------------------------------------------------------------------------
class OffloadLoad(_Base):
    """One GPU block, one host-tier block holding a cached prefix for request X. The scheduler allocates the GPU
    block and issues a load on the load stream; the forward stream runs X."""
    key = "offload-load"
    ORDER = ("scheduler", "load", "forward")
    assumptions = ("a load's bytes are visible to the forward stream only after the load completes",)
    not_covered = ("CUDA graph replay of the forward pass", "partial visibility within one block")

    def initial(self):
        s = _PDState()
        s.ledger.alloc(("host", 0), "X", 0)
        s.ledger.write_local(("host", 0), "X's cached prefix", 0)
        s.req = {"X": {"phase": "hit"}}
        return s

    def done(self, s):
        return s.req["X"]["phase"] == "finished" and not s.ledger.outstanding()

    def signature(self, s):
        return (tuple(sorted(s.req["X"].items(), key=str)), _ledger_sig(s.ledger))

    def enabled(self, s):
        acts = []
        v = s.req["X"]
        G, H = ("gpu", 0), ("host", 0)
        if v["phase"] == "hit":
            def sched(st, step):
                vv = st.req["X"]
                g = st.ledger.alloc(G, "X", step)
                st.ledger.share(G, "X", step)
                vv["g"] = g
                st.ledger.submit("load-X", "read", "load stream", src=st.ledger.current(H), dst=g, step=step)
                vv["phase"] = "loading"
            acts.append(self._act("scheduler allocates a GPU block and issues X's load", "scheduler", {"X", G, H},
                                  sched))
        if v["phase"] == "loading" and s.ledger.ops["load-X"].state == "outstanding":
            def land(st, step):
                st.ledger.complete("load-X", step, marks_ready=G)
                st.req["X"]["loaded"] = True
            acts.append(self._act("load-X completes", "load", {"X", G, "load-X"}, land))
        runnable = v["phase"] == "loading" and (self.variant == "buggy" or v.get("loaded"))
        if runnable:
            def run(st, step):
                vv = st.req["X"]
                st.ledger.consume("X", [vv["g"]], step)
                vv["phase"] = "ran"
            acts.append(self._act("forward stream runs X", "forward", {"X", G}, run))
        if v["phase"] == "ran" and v.get("loaded"):
            def fin(st, step):
                st.req["X"]["phase"] = "finished"
            acts.append(self._act("X finishes", "forward", {"X"}, fin))
        return self._sorted(acts)


MODELS = {m.key: m for m in (PDAbort, PDLease, PDNotify, OffloadLoad)}


def model(key: str, variant: str = "buggy"):
    if key not in MODELS:
        raise ValueError(f"unknown model {key!r}; known: {', '.join(MODELS)}")
    return MODELS[key](variant)


_ = Handle  # re-exported for adapters building handles
