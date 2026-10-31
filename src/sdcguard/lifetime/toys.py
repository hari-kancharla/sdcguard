"""Two small real implementations of KV transfer protocols, for demonstrating and testing the model-free search.

They are real code, not models: threads, real timers, queues and a byte store the transfers really copy. Each
calls the Probe's recording methods and has checkpoints where a thread could legally be slow. The search
(realsearch.search) is given only make() and run(); it is not told where the bug is.

ToyPull  the decode side pulls KV from a block the prefill side holds under a lease (vLLM NIXL pull shape).
         fixed=True: the consumer pins the block before reading; a pin after the lease expired fails the request.
ToyPush  the prefill side writes KV into a block the decode side allocated; the decode side times out slow
         requests (the Z.ai shape). fixed=True: on timeout the decode side asks the prefill side to stop and
         frees the block only after it answers that no write is outstanding.
ToyWakeup a lost wakeup (a liveness bug): a completion that lands between "done?" and "wait" is missed."""

from __future__ import annotations

import queue
import threading

from .adapter import desc_ref

P0 = ("prefill", 0)
D0 = ("decode", 0)


class ToyPull:
    def __init__(self, probe, fixed: bool = False, lease_s: float = 0.25):
        self.p = probe
        self.fixed = fixed
        self.lease_s = lease_s
        self.mem: dict = {}
        self.owner: dict = {}
        self.pinned: set = set()
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.result = None
        self.failed = False
        self.y_done = threading.Event()

    def _expire(self):
        with self.lock:
            expired = not (self.fixed and "X" in self.pinned) and self.owner.get(P0) == "X"
            if expired:
                self.p.free(P0, "X")
                self.owner[P0] = None
        if expired:
            self._prefill_y()
        else:
            self.y_done.set()

    def _prefill_y(self):
        with self.lock:
            if self.owner.get(P0) is None:
                self.owner[P0] = "Y"
                self.p.alloc(P0, "Y")
                self.mem[P0] = b"Y-kv"
                self.p.write_local(P0, "Y's KV")
        self.y_done.set()

    def _release_after_done(self):
        with self.lock:
            if self.owner.get(P0) == "X":
                self.p.free(P0, "X")
                self.owner[P0] = None
        self._prefill_y()

    def producer(self):
        with self.lock:
            self.owner[P0] = "X"
            self.p.alloc(P0, "X")
            self.mem[P0] = b"X-kv"
            self.p.write_local(P0, "X's KV")
            self.p.publish("dX", [P0], "X")
        self.timer = threading.Timer(self.lease_s, self._expire)
        self.timer.daemon = True
        self.timer.start()
        self.q.put("dX")

    def _pin(self) -> bool:
        with self.lock:
            if self.owner.get(P0) == "X":
                self.pinned.add("X")
                return True
            return False

    def consumer(self):
        desc = self.q.get(timeout=5)
        self.p.alloc(D0, "X")
        if self.fixed:
            self.p.checkpoint("pin", "X")
            if not self._pin():
                self.p.free(D0, "X")
                self.failed = True               # allowed: the request fails; nothing stale is read
                return
        self.p.checkpoint("read_submit", "X", desc=desc)
        self.p.accept(desc, "decode")
        self.p.submit("read-X", "read", "decode", src=desc_ref(desc, 0), dst=D0)
        t = threading.Thread(target=self._land, daemon=True)
        t.start()
        t.join(10)

    def _land(self):
        self.p.checkpoint("read_land", "read-X")
        self.mem[D0] = self.mem.get(P0)          # a one-sided read copies whatever the block holds now
        self.p.complete("read-X", marks_ready=D0)
        self.p.checkpoint("decode_run", "X")
        self.p.consume("X", [D0])
        self.result = self.mem[D0]
        self.p.free(D0, "X")
        if self.fixed:
            with self.lock:
                self.pinned.discard("X")
            self.timer.cancel()
            self._release_after_done()

    def run(self):
        c = threading.Thread(target=self.consumer, daemon=True)
        c.start()
        self.producer()
        c.join(10)
        if not self.y_done.wait(self.lease_s * 8):
            pass
        self.timer.cancel()


class ToyPush:
    def __init__(self, probe, fixed: bool = False, timeout_s: float = 0.25):
        self.p = probe
        self.fixed = fixed
        self.timeout_s = timeout_s
        self.mem: dict = {}
        self.lock = threading.Lock()
        self.block_free = threading.Condition(self.lock)
        self.free_blocks = [D0]
        self.state: dict = {}
        self.work: queue.Queue = queue.Queue()
        self.results: dict = {}
        self.outstanding: dict = {}           # request -> landing thread

    # decode side
    def _admit(self, r):
        with self.block_free:
            if not self.free_blocks:
                self.block_free.wait_for(lambda: self.free_blocks, 10)
            if not self.free_blocks:
                raise TimeoutError(f"no free block for {r}")
            self.free_blocks.pop(0)
            self.p.alloc(D0, r)
            self.p.publish(f"d-{r}", [D0], r)
            self.state[r] = {"phase": "admitted", "done": threading.Event()}
        self.work.put(("work", r))
        t = threading.Timer(self.timeout_s, self._timeout, args=(r,))
        t.daemon = True
        self.state[r]["timer"] = t
        t.start()

    def _release(self, r):
        with self.block_free:
            self.p.free(D0, r)
            self.free_blocks.append(D0)
            self.block_free.notify_all()

    def _timeout(self, r):
        with self.lock:
            st = self.state[r]
            if st["phase"] != "admitted":
                return
            st["phase"] = "aborting" if self.fixed else "aborted"
        self.work.put(("abort", r))
        if not self.fixed:
            self._release(r)                  # buggy: the blocks go back at once
            st["done"].set()

    def _on_safe(self, r):
        st = self.state[r]
        st["phase"] = "aborted"
        self._release(r)
        st["done"].set()

    def _on_landed(self, r):
        with self.lock:
            st = self.state[r]
            ready = st["phase"] == "admitted"
            if ready:
                st["phase"] = "ready"
                st["timer"].cancel()
        if ready:
            self.p.checkpoint("decode_run", r)
            self.p.consume(r, [D0])
            self.results[r] = self.mem.get(D0)
            self._release(r)
            st["phase"] = "finished"
            st["done"].set()

    # prefill side
    def _prefill(self):
        aborted = set()
        while True:
            kind, r = self.work.get()
            if kind == "stop":
                return
            if kind == "abort":
                aborted.add(r)
                if self.fixed:
                    t = self.outstanding.get(r)
                    if t is not None and t.is_alive():
                        self.p.cancel(f"write-{r}", "prefill")
                        t.join(10)
                    self._on_safe(r)
                continue
            if r in aborted:
                continue
            self.p.checkpoint("prefill_forward", r, desc=f"d-{r}")
            self.p.submit(f"write-{r}", "write", "prefill", dst=desc_ref(f"d-{r}", 0), payload=(r, 0, f"{r}'s KV"))
            t = threading.Thread(target=self._land, args=(r,), daemon=True)
            t.start()
            self.outstanding[r] = t

    def _land(self, r):
        self.p.checkpoint("write_land", f"write-{r}", desc=f"d-{r}")
        self.mem[D0] = f"{r}-kv".encode()
        self.p.complete(f"write-{r}")
        self._on_landed(r)

    def run(self):
        pf = threading.Thread(target=self._prefill, daemon=True)
        pf.start()
        try:
            for r in ("A", "B"):
                self._admit(r)
                self.state[r]["done"].wait(10)
            for t in list(self.outstanding.values()):
                t.join(10)
        finally:
            self.work.put(("stop", None))
            pf.join(10)
            for st in self.state.values():
                if st.get("timer") is not None:
                    st["timer"].cancel()


class ToyWakeup:
    """A liveness bug, not a lifetime one: the decode side waits for a transfer's completion signal, and checks
    "already done?" and then waits as two steps. A completion that lands between them is missed and the request
    waits forever (a lost wakeup). fixed=True waits on the condition itself (Condition.wait_for)."""

    def __init__(self, probe, fixed: bool = False, work_s: float = 0.1):
        self.p = probe
        self.fixed = fixed
        self.work_s = work_s
        self.cv = threading.Condition()
        self.done = False

    def waiter(self):
        with self.cv:
            done = self.done
        if not done:
            self.p.checkpoint("before_wait", "X")
            with self.cv:
                if self.fixed:
                    self.cv.wait_for(lambda: self.done)
                else:
                    self.cv.wait()             # buggy: a notify that came before this wait is lost
        self.p.consume("X", [D0])
        self.p.free(D0, "X")

    def worker(self):
        self.p.checkpoint("transfer", "load-X")
        threading.Event().wait(self.work_s)
        self.p.submit("load-X", "write", "transfer", dst=D0, payload=("X", 0, "X's KV"))
        self.p.complete("load-X", marks_ready=D0)
        with self.cv:
            self.done = True
            self.cv.notify_all()

    def run(self):
        self.p.alloc(D0, "X")
        w = threading.Thread(target=self.waiter, daemon=True)
        k = threading.Thread(target=self.worker, daemon=True)
        w.start()
        k.start()
        k.join()
        w.join()                               # no timeout: a lost wakeup hangs here, as the real server would


TOYS = {"pull": ToyPull, "push": ToyPush, "wakeup": ToyWakeup}
