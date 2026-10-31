"""The adapter: real concurrent code (threads, a real timer-driven lease) recorded through a Probe, and a Hold that
drives the ordering the explorer found in the pd-lease model."""

import queue
import threading
import time

from sdcguard.lifetime.adapter import (Hold, JsonlSink, LocalSink, Probe, desc_ref, merge, until_event)

P0 = ("prefill", 0)
D0 = ("decode", 0)


class ToyPull:
    """A tiny pull-transfer implementation: a producer with one block and a timer lease, a consumer that reads the
    producer's block bytes. Not a model: real threads, a real timer, a real copy of bytes at completion time."""

    def __init__(self, probe, lease_s=0.05, fixed=False, consumer_probe=None):
        self.pp = probe                     # the producer's process
        self.cp = consumer_probe or probe   # the consumer's process
        self.lease_s = lease_s
        self.fixed = fixed
        self.mem = {}                       # region -> bytes
        self.owner = {}
        self.q = queue.Queue()
        self.pinned = set()
        self.lock = threading.Lock()
        self.result = None
        self.failed = False

    # producer side
    def producer_x(self):
        with self.lock:
            self.owner[P0] = "X"
            self.pp.alloc(P0, "X")
            self.mem[P0] = b"X-kv"
            self.pp.write_local(P0, "X's KV")
            self.pp.publish("dX", [P0], "X")
        threading.Timer(self.lease_s, self.expire).start()
        self.q.put(("desc", "dX"))

    def expire(self):
        with self.lock:
            if self.fixed and "X" in self.pinned:
                return
            if self.owner.get(P0) == "X":
                self.pp.free(P0, "X")
                self.owner[P0] = None
        threading.Thread(target=self.producer_y).start()

    def producer_y(self):
        with self.lock:
            if self.owner.get(P0) is None:
                self.owner[P0] = "Y"
                self.pp.alloc(P0, "Y")
                self.mem[P0] = b"Y-kv"
                self.pp.write_local(P0, "Y's KV")

    def pin(self):
        with self.lock:
            if self.owner.get(P0) == "X":
                self.pinned.add("X")
                return True
            return False

    # consumer side
    def consumer(self):
        _, desc = self.q.get(timeout=5)
        self.cp.alloc(D0, "X")
        if self.fixed and not self.pin():
            self.cp.free(D0, "X")
            self.failed = True                   # allowed: the request fails, nothing stale is read
            return
        self.cp.checkpoint("read_submit", "X")
        self.cp.accept(desc, "decode")
        self.cp.submit("read-X", "read", "decode", src=desc_ref(desc, 0), dst=D0)
        data = self.mem[P0]                      # the one-sided read copies whatever the block holds now
        self.mem[D0] = data
        self.cp.complete("read-X", marks_ready=D0)
        self.cp.consume("X", [D0])
        self.result = data


def run(probe, fixed=False, lease_s=0.05):
    t = ToyPull(probe, lease_s=lease_s, fixed=fixed)
    c = threading.Thread(target=t.consumer)
    c.start()
    t.producer_x()
    c.join(5)
    time.sleep(0.15)                             # let the lease timer and Y finish
    return t


def test_without_a_hold_the_fast_consumer_reads_its_own_kv():
    sink = LocalSink()
    t = run(Probe(sink), lease_s=2.0)           # a lease far longer than a fast read: no timing luck needed
    assert t.result == b"X-kv"
    assert sink.ledger.violations == []


# Hold until Y has written its KV into the reused block (the record of Y's allocation comes a moment before Y's
# bytes land, so holding only until the allocation could let the read copy the old bytes in between).
UNTIL_Y_WROTE = dict(region=P0, label="Y's KV")


def test_a_hold_drives_the_lease_race_and_the_real_bytes_are_wrong():
    sink = LocalSink()
    hold = Hold("read_submit", "X", until_event("write_local", **UNTIL_Y_WROTE), timeout=3)
    probe = Probe(sink, holds=[hold])
    t = run(probe)
    assert probe.held and probe.held[0][2] is True               # the condition was reached, not timed out
    assert t.result == b"Y-kv"                                   # the real copy carried Y's KV into X
    rules = [v.rule for v in sink.ledger.violations]
    assert rules[0] == "R1" and "R3" in rules
    assert any("Y's data" in v.message for v in sink.ledger.violations if v.rule == "R3")


def test_the_fixed_protocol_under_the_same_hold_fails_the_request_instead():
    sink = LocalSink()
    hold = Hold("read_submit", "X", until_event("alloc", region=P0, owner="Y"), timeout=0.3)
    probe = Probe(sink, holds=[hold])
    t = run(probe, fixed=True)
    assert t.result == b"X-kv"                                   # pinned: the lease never expired under the read
    assert sink.ledger.violations == []
    assert probe.held and probe.held[0][2] is False              # Y never got the block: the hold expired


def test_two_processes_jsonl_files_merge_into_one_history(tmp_path):
    pf, df = str(tmp_path / "prefill.jsonl"), str(tmp_path / "decode.jsonl")
    prod = Probe(JsonlSink(pf, peers=(df,)), pid=1)
    cons = Probe(JsonlSink(df, peers=(pf,)), pid=2,
                 holds=[Hold("read_submit", "X", until_event("write_local", **UNTIL_Y_WROTE), timeout=3)])

    t = ToyPull(prod, consumer_probe=cons)
    c = threading.Thread(target=t.consumer)
    c.start()
    t.producer_x()
    c.join(5)
    time.sleep(0.15)
    led = merge([pf, df])
    rules = [v.rule for v in led.violations]
    assert rules and rules[0] == "R1"
    assert t.result == b"Y-kv"
