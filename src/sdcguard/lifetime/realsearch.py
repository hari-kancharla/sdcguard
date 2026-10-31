"""Search a real implementation for lifetime-breaking orderings, with no protocol model.

The input is code instrumented with a Probe (the same hooks the adapters use): its recording calls and its
checkpoints, the points where work could legally be delayed. The search knows nothing about the protocol.

1. Baseline: run the system in its normal timing (baseline_runs times), recording every checkpoint and event.
2. Footprints: for each checkpoint, the blocks the same thread touches right after it (until its next checkpoint):
   the regions of its submits, completions, consumes and allocations, with descriptor references resolved to the
   regions they name.
3. Candidate holds. Two sources, neither told where a bug is:
   rule    from the lifetime rules: hold the checkpoint until one of its blocks is freed; reallocated to another
           owner; consumed by another request; or until an operation landing in it completes.
   flip    from the baseline itself: an event of another thread on one of those blocks (an allocation, a free, a
           write, a completion, a consume, a submit) that came after the checkpoint's work; hold the checkpoint
           until that event, so the observed order flips. Most real timing bugs need one or two such flips.
   Each candidate is a single delay that a slow thread, a stalled process or a late network message could cause;
   none fabricates an event. Candidates whose flipped order never appeared in any baseline go first (rarest
   first), and max_runs bounds the work.
4. Run the system afresh under each candidate (a hold whose condition never comes expires and the run goes on), and
   check the rules on what was recorded. A new finding is confirmed by replaying the same holds (confirm runs in
   all); it counts only if it reproduces in at least two thirds of them, otherwise it is listed as unconfirmed.
   Findings are grouped by rule and shape; each keeps the first holds that produced it (as a replayable holds
   spec), the ledger events that caused it, every other hold that also produced it, and whether each hold's
   condition was actually reached.
5. Liveness: once the holds lift, the run must still finish. A run that raises, or does not finish within
   run_timeout seconds, is a liveness finding (confirmed the same way).
depth=2 also tries every pair of candidates on different checkpoints. random_runs adds runs with short random
pauses at every checkpoint (no holds), as a check on what the targeted holds miss.

A finding here is about the implementation that ran (its code, its timing, its transport as exercised). It is a
server finding when the implementation is the server, on the configuration and hardware that ran.

Limitation: a candidate names blocks and keys as they appeared in the baseline. That suits code whose block ids and
request ids repeat from run to run (these toys, a seeded test harness); a real server that numbers them differently
each run needs its holds resolved per run (experiments/lifetime_pd_check.py does that for vLLM)."""

from __future__ import annotations

import itertools
import json
import threading
from dataclasses import dataclass, field

from .adapter import Hold, LocalSink, Probe, until_event

RULE_KINDS = ("realloc", "free", "consume", "complete")
FLIP_KINDS = ("alloc", "free", "write_local", "complete", "consume", "submit", "apply_completion", "cancel",
              "stopped")
_PRIORITY = {"realloc": 0, "alloc": 0, "free": 1, "write_local": 2, "complete": 3, "apply_completion": 3,
             "consume": 4, "submit": 5, "cancel": 6, "stopped": 6}


@dataclass
class Candidate:
    point: str
    key: object
    until: str                 # human description of the condition
    kind: str
    region: tuple
    other: object = None       # owner or request the condition is about
    spec: dict = None          # the condition as plain JSON: {"kind": ..., "args": {...}} (see adapter.until_event)
    source: str = "rule"       # "rule" or "flip"
    seen_flipped: int = 0      # baselines in which the condition already came before the checkpoint's work

    def hold(self, timeout: float) -> Hold:
        from .adapter import holds_from_spec
        return holds_from_spec([self.hold_spec(timeout)])[0]

    def hold_spec(self, timeout: float) -> dict:
        return {"point": self.point, "key": self.key, "until": self.spec, "timeout": timeout}

    def describe(self) -> str:
        return f"hold {self.point} ({self.key}) until {self.until}"

    def ident(self) -> str:
        return json.dumps([self.point, self.key, self.spec], sort_keys=True, default=list)


@dataclass
class RealFinding:
    rule: str
    message: str
    holds: list
    reached: list
    trace: list = field(default_factory=list)
    then: list = field(default_factory=list)     # later violations in the same run (consequences)
    spec: list = field(default_factory=list)     # the holds as a replayable holds file (adapter.holds_from_file)
    repro: tuple = (1, 1)                        # (runs that reproduced it, runs tried)
    also: list = field(default_factory=list)     # other holds that produced the same finding

    @property
    def confirmed(self) -> bool:
        k, n = self.repro
        return 3 * k >= 2 * n


@dataclass
class RealResult:
    checkpoints: list
    candidates: int
    runs: int
    findings: list
    baseline_violations: list
    liveness: list = field(default_factory=list)
    unconfirmed: list = field(default_factory=list)
    baseline_problem: str = ""
    budget_hit: bool = False

    @property
    def verdict(self) -> str:
        if self.baseline_violations or self.baseline_problem:
            return "VIOLATION_IN_BASELINE"
        if self.findings:
            return "VIOLATION"
        return "LIVENESS" if self.liveness else "NONE_FOUND"

    def as_dict(self) -> dict:
        def f(x):
            return {"rule": x.rule, "message": x.message, "holds": x.holds, "reached": x.reached, "trace": x.trace,
                    "then": x.then, "spec": x.spec, "repro": list(x.repro), "also": x.also}
        return {"verdict": self.verdict, "checkpoints": [list(c) for c in self.checkpoints],
                "candidates": self.candidates, "runs": self.runs, "budget_hit": self.budget_hit,
                "findings": [f(x) for x in self.findings], "liveness": [f(x) for x in self.liveness],
                "unconfirmed": [f(x) for x in self.unconfirmed],
                "baseline_violations": [f"{v.rule} {v.message}" for v in self.baseline_violations],
                "baseline_problem": self.baseline_problem}


def _descs(events) -> dict:
    return {e["args"]["desc"]: [tuple(r) for r in e["args"]["regions"]] for e in events if e["kind"] == "publish"}


def _regions_of(ev, descs) -> list:
    a = ev.get("args", {})
    out = []
    for k in ("region", "src", "dst", "marks_ready"):
        v = a.get(k)
        if isinstance(v, dict) and "desc" in v:
            rs = descs.get(v["desc"], [])
            i = v.get("i", 0)
            if i < len(rs):
                out.append(tuple(rs[i]))
        elif isinstance(v, list):
            out.append(tuple(v))
    for r in a.get("regions", []) or []:
        out.append(tuple(r))
    return out


def _owner_of(ev):
    a = ev.get("args", {})
    return a.get("owner") or a.get("request") or a.get("by")


def _thread(e):
    return (e.get("pid"), e.get("tid"))


def _spans(events):
    """For each checkpoint: (index, event, indices of the same thread's events up to its next checkpoint)."""
    out = []
    for i, e in enumerate(events):
        if e["kind"] != "checkpoint":
            continue
        span = []
        for j in range(i + 1, len(events)):
            f = events[j]
            if _thread(f) != _thread(e):
                continue
            if f["kind"] == "checkpoint":
                break
            span.append(j)
        out.append((i, e, span))
    return out


def footprints(events) -> list:
    """(checkpoint event, regions touched after it by the same thread, the actor) for each recorded checkpoint."""
    descs = _descs(events)
    out = []
    for _, e, span in _spans(events):
        regs, actor = [], None
        for j in span:
            f = events[j]
            if f["kind"] in ("submit", "complete", "consume", "alloc", "accept", "write_local", "free"):
                regs += [r for r in _regions_of(f, descs) if r not in regs]
                actor = actor or _owner_of(f)
            if f["kind"] == "complete":
                op = f["args"].get("op")
                sub = next((x for x in events if x["kind"] == "submit" and x["args"].get("op") == op), None)
                if sub is not None:
                    regs += [r for r in _regions_of(sub, descs) if r not in regs]
        out.append((e, regs, actor))
    return out


def _rule_spec(kind, r, other):
    r = list(r)
    if kind == "free":
        return {"kind": "free", "args": {"region": r}}
    if kind == "realloc":
        return {"kind": "alloc", "args": {"region": r, "owner": {"not": other}}}
    if kind == "consume":
        return {"kind": "consume", "args": {"regions": {"has": r}, "request": {"not": other}}}
    return {"kind": "complete", "args": {"marks_ready": r}}


_RULE_TEXT = {"free": "{p} block {b} is freed", "realloc": "{p} block {b} is reallocated to another owner",
              "consume": "another request runs on {p} block {b}",
              "complete": "an operation landing in {p} block {b} completes"}


def _flip_spec(ev):
    a = {k: v for k, v in ev.get("args", {}).items() if k not in ("label", "payload") and v is not None}
    return {"kind": ev["kind"], "args": a}


def _flip_text(ev, r):
    a = ev.get("args", {})
    who = a.get("owner") or a.get("request") or a.get("by") or a.get("op") or ""
    verb = {"alloc": "is allocated to", "free": "is freed by", "write_local": "is written",
            "complete": "gets the completion of", "apply_completion": "gets the completion of",
            "consume": "is consumed by", "submit": "gets a transfer", "cancel": "gets a cancel",
            "stopped": "gets a stop"}.get(ev["kind"], ev["kind"])
    return f"{r[0]} block {r[1]} {verb} {who}".rstrip() + " (seen later in the normal run)"


_IDS = ("owner", "request", "by", "op", "desc")


def _symbol(r, cp, descs, events, max_index):
    """The checkpoint-relative name of region r (see adapter.resolve), or None when it has none."""
    d = cp["args"].get("desc")
    if d is not None and r in descs.get(d, []):
        i = descs[d].index(r)
        return {"desc_block": i} if i < max_index else None
    key = cp["args"].get("key")
    owned = []
    for e in events:
        if e["kind"] == "alloc" and e["args"].get("owner") == key and tuple(e["args"]["region"]) not in owned:
            owned.append(tuple(e["args"]["region"]))
    if r in owned:
        i = owned.index(r)
        return {"key_block": i} if i < max_index else None
    return None


def _symbolize_flip(spec, r, sym):
    a = {}
    for k, v in spec["args"].items():
        if k in ("region", "marks_ready") and isinstance(v, list) and tuple(v) == r:
            a[k] = sym
        elif k == "regions" and isinstance(v, list):
            a[k] = {"has": sym}
        elif k in _IDS or k in ("src", "dst"):
            a[k] = "*"
        else:
            a[k] = v
    return {"kind": spec["kind"], "args": a}


def candidates(events, symbolic: bool = False, max_index: int = 2) -> list:
    """Candidate holds from one recorded baseline (rule-derived first, then observed flips).

    symbolic=True names blocks relative to the checkpoint (the i-th block of the descriptor it is about, or of the
    request it holds; only the first max_index of each) and holds the next arrival at the checkpoint whatever its
    key, so the candidates apply to a fresh run of a server that numbers blocks and requests differently. Request,
    owner and operation ids in observed flips become wildcards. Regions with no relative name are skipped."""
    descs = _descs(events)
    out, seen = [], set()

    def add(c):
        k = c.ident()
        if k not in seen:
            seen.add(k)
            out.append(c)

    def name(r, cp):
        return _symbol(r, cp, descs, events, max_index) if symbolic else r

    def label(r, sym):
        if not symbolic:
            return r[0], r[1]
        if "desc_block" in sym:
            return "published", f"{sym['desc_block']} ({r[0]} block {r[1]} in the baseline)"
        return "held request's", f"{sym['key_block']} ({r[0]} block {r[1]} in the baseline)"

    fps = footprints(events)
    for cp, regs, actor in fps:
        point, key = cp["args"]["point"], None if symbolic else cp["args"].get("key")
        for r in regs:
            sym = name(r, cp)
            if sym is None:
                continue
            p_, b_ = label(r, sym)
            for kind in RULE_KINDS:
                spec = _rule_spec(kind, r, actor)
                if symbolic:            # conditions only match events after the hold began, so no owner test
                    spec = {"free": {"kind": "free", "args": {"region": sym}},
                            "realloc": {"kind": "alloc", "args": {"region": sym}},
                            "consume": {"kind": "consume", "args": {"regions": {"has": sym}}},
                            "complete": {"kind": "complete", "args": {"marks_ready": sym}}}[kind]
                add(Candidate(point, key, _RULE_TEXT[kind].format(p=p_, b=b_), kind, r, actor, spec, "rule"))
    for (i, cp, span), (_, regs, _) in zip(_spans(events), fps):
        point, key = cp["args"]["point"], None if symbolic else cp["args"].get("key")
        if not regs:
            continue
        after = span[-1] if span else i
        for j in range(after + 1, len(events)):
            ev = events[j]
            if _thread(ev) == _thread(cp) or ev["kind"] not in FLIP_KINDS:
                continue
            hit = [r for r in _regions_of(ev, descs) if r in regs]
            if not hit:
                continue
            spec = _flip_spec(ev)
            if symbolic:
                sym = name(hit[0], cp)
                if sym is None:
                    continue
                spec = _symbolize_flip(spec, hit[0], sym)
            add(Candidate(point, key, _flip_text(ev, hit[0]), ev["kind"], hit[0], _owner_of(ev), spec, "flip"))
    return out


def _already_flipped(events, c: Candidate, spans=None) -> bool:
    """In this baseline, did an event matching c's condition come between the checkpoint and the end of its work?"""
    from .adapter import resolve
    for i, cp, span in (spans if spans is not None else _spans(events)):
        if cp["args"]["point"] != c.point or (c.key is not None and cp["args"].get("key") != c.key):
            continue
        spec = resolve(c.spec, cp["args"].get("key"), cp["args"], events[:i + 1])
        if spec is None:
            continue
        pred = until_event(spec["kind"], **spec.get("args", {}))
        end = span[-1] if span else i
        if any(pred(events[j]) and _thread(events[j]) != _thread(cp) for j in range(i + 1, end + 1)):
            return True
    return False


def rank(cands, baselines) -> list:
    """Rarest first: candidates whose flipped order never showed up in a baseline, then by kind (reuse and frees
    before writes, completions and consumes), rule-derived before observed flips, then baseline order."""
    spans = [_spans(b) for b in baselines]
    for c in cands:
        c.seen_flipped = sum(_already_flipped(b, c, sp) for b, sp in zip(baselines, spans))
    order = {c.ident(): n for n, c in enumerate(cands)}
    return sorted(cands, key=lambda c: (c.seen_flipped, _PRIORITY.get(c.kind, 9), c.source != "rule",
                                        order[c.ident()]))


class _Outcome:
    def __init__(self, sink, probe, status, error=""):
        self.sink, self.probe, self.status, self.error = sink, probe, status, error


def _run(make, run, holds, record=False, run_timeout=30.0, pauses=None) -> _Outcome:
    sink = LocalSink()
    probe = Probe(sink, holds=holds, record_checkpoints=record, pauses=pauses)
    box = {}

    def go():
        try:
            run(make(probe))
        except BaseException as e:  # noqa: BLE001  (a crash under a hold is a finding, not a search failure)
            box["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(run_timeout)
    if t.is_alive():
        return _Outcome(sink, probe, "hang", f"the run did not finish within {run_timeout:g}s")
    if "error" in box:
        return _Outcome(sink, probe, "crash", box["error"])
    return _Outcome(sink, probe, "ok")


def _key(o: _Outcome):
    if o.status != "ok":
        return ("LIVENESS", o.status, o.error.split(":")[0])
    vs = o.sink.ledger.violations
    return vs[0].key() if vs else None


def search(make, run, depth: int = 1, hold_timeout: float = 1.0, max_runs: int = 400, confirm: int = 3,
           run_timeout: float = 30.0, baseline_runs: int = 2, random_runs: int = 0, pause_prob: float = 0.2,
           pause_max: float | None = None, seed: int = 0, symbolic: bool = False) -> RealResult:
    """make(probe) builds a fresh system wired to probe; run(system) runs it to completion (blocking)."""
    bases = [_run(make, run, [], record=True, run_timeout=run_timeout) for _ in range(max(1, baseline_runs))]
    problem = next((f"{b.status}: {b.error}" for b in bases if b.status != "ok"), "")
    base = bases[0]
    cands, seen = [], set()
    for b in bases:
        for c in candidates(b.sink.events, symbolic=symbolic):
            if c.ident() not in seen:
                seen.add(c.ident())
                cands.append(c)
    cands = rank(cands, [b.sink.events for b in bases])
    points = sorted({(c.point, str(c.key)) for c in cands})
    combos = [(c,) for c in cands]
    if depth >= 2:
        combos += [(a, b) for a, b in itertools.combinations(cands, 2) if (a.point, a.key) != (b.point, b.key)]
    found: dict = {}
    runs = 0
    budget_hit = False

    def record(key, o, label_holds, spec, reached):
        old = found.get(key)
        if old is not None and not (spec and reached and all(reached) and not all(old.reached or [False])):
            old.also.append(" + ".join(label_holds))
            return None
        if o.status != "ok":
            f = RealFinding("LIVENESS", f"{o.status}: {o.error}", label_holds, reached, [], [], spec)
        else:
            vs = o.sink.ledger.violations
            f = RealFinding(vs[0].rule, vs[0].message, label_holds, reached, _trace(o.sink, vs[0]),
                            [f"{v.rule} {v.message}" for v in vs[1:]], spec)
        if old is not None:              # prefer, as the shown cause, holds whose conditions were all reached
            f.also = old.also + [" + ".join(old.holds)]
        found[key] = f
        return f

    def confirmed_by_replay(key, f, prev, again):
        nonlocal runs
        k = 1
        for _ in range(confirm - 1):
            runs += 1
            k += _key(again()) == key
        f.repro = (k, confirm)
        if prev is not None and prev.confirmed and not f.confirmed:     # keep the cause that replays
            prev.also.append(" + ".join(f.holds))
            found[key] = prev

    for combo in combos:
        if runs >= max_runs:
            budget_hit = True
            break
        o = _run(make, run, [c.hold(hold_timeout) for c in combo], run_timeout=run_timeout)
        runs += 1
        key = _key(o)
        if key is None:
            continue
        prev = found.get(key)
        f = record(key, o, [c.describe() for c in combo], [c.hold_spec(hold_timeout) for c in combo],
                   [h[2] for h in o.probe.held])
        if f is not None:
            confirmed_by_replay(key, f, prev, lambda combo=combo: _run(
                make, run, [c.hold(hold_timeout) for c in combo], run_timeout=run_timeout))
    for s in range(random_runs):            # asked for explicitly, so not bounded by max_runs
        pz = (pause_prob, pause_max if pause_max is not None else hold_timeout, seed + s)
        o = _run(make, run, [], run_timeout=run_timeout, pauses=pz)
        runs += 1
        key = _key(o)
        if key is None:
            continue
        prev = found.get(key)
        f = record(key, o, [f"random pauses at checkpoints (seed {seed + s})"], [], [])
        if f is not None:                   # the same seed gives the same pauses; thread timing still varies
            confirmed_by_replay(key, f, prev, lambda pz=pz: _run(make, run, [], run_timeout=run_timeout, pauses=pz))
    allf = list(found.values())
    safety = [f for f in allf if f.rule != "LIVENESS" and f.confirmed]
    live = [f for f in allf if f.rule == "LIVENESS" and f.confirmed]
    unconf = [f for f in allf if not f.confirmed]
    return RealResult(points, len(cands), runs, safety, list(base.sink.ledger.violations), live, unconf, problem,
                      budget_hit)


def replay(make, run, spec, times: int = 3, run_timeout: float = 30.0) -> list:
    """Run the system under a holds spec (as written by the search) times times; the first finding key of each run
    (None for a clean run). A pinned regression test asserts every entry is None on the fixed code."""
    from .adapter import holds_from_spec
    out = []
    for _ in range(times):
        o = _run(make, run, holds_from_spec(spec), run_timeout=run_timeout)
        out.append(_key(o))
    return out


def _trace(sink, v) -> list:
    led = sink.ledger
    regs = set(v.regions)
    for o in v.ops:
        op = led.ops.get(o)
        if op is not None:
            regs |= {h.region for h in (op.src, op.dst) if h is not None}
    return [f"{e.step:>3}  {e.kind:<9} {e.text}" for e in led.events
            if (set(e.regions) & regs or set(e.ops) & set(v.ops) or e.kind in ("held", "released"))
            and e.step <= v.step]


def text(res: RealResult, title: str = "") -> str:
    lines = [f"sdcguard lifetime search{': ' + title if title else ''}"]
    lines.append(f"{len(res.checkpoints)} checkpoints seen in the baseline run, {res.candidates} candidate holds, "
                 f"{res.runs} runs{' (run budget reached; not every candidate was tried)' if res.budget_hit else ''}.")
    if res.baseline_problem:
        lines.append(f"The baseline run itself did not finish cleanly: {res.baseline_problem}")
    if res.baseline_violations:
        lines.append("The baseline run itself broke a rule:")
        lines.extend(f"  {v.rule} {v.message}" for v in res.baseline_violations)
    if not res.findings and not res.liveness:
        lines.append("NONE FOUND: no tried delay at a checkpoint broke a lifetime rule or stopped the run finishing.")

    def show(f):
        lines.append("")
        lines.append(f"{f.rule} {f.message}   (reproduced {f.repro[0]} of {f.repro[1]} runs)")
        for h, ok in zip(f.holds, f.reached):
            lines.append(f"  by: {h} ({'condition reached' if ok else 'condition never came; hold expired'})")
        lines.extend("    " + t for t in f.trace)
        if f.then:
            lines.append("  then, in the same run:")
            lines.extend("    " + t for t in f.then)
        if f.also:
            lines.append(f"  also produced by {len(f.also)} other hold(s), e.g. {f.also[0]}")
    for f in res.findings:
        show(f)
    for f in res.liveness:
        show(f)
    if res.unconfirmed:
        lines.append("")
        lines.append(f"{len(res.unconfirmed)} finding(s) did not reproduce in two thirds of the replays (not counted):")
        lines.extend(f"  {f.rule} {f.message} ({f.repro[0]} of {f.repro[1]})" for f in res.unconfirmed)
    return "\n".join(lines)
