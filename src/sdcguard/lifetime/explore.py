"""The explorer: search the orderings of a serving protocol's events for one that breaks a lifetime rule.

A model (models.py, or an adapter over a real server) exposes, in each state, the actions that may happen next: a
worker issuing a transfer, a transfer completing, a timeout firing, the clock advancing, a request arriving. Each
action names the resources it touches (regions, requests, operations). The default schedule takes the first enabled
action every time, which is the model's normal, unhurried order (time only passes when nothing else can run). The
explorer departs from it on purpose: choosing the k-th enabled action instead of the first costs k "delays" (delay
bounding), so a budget of 2 explores every schedule that holds back the normal order twice (a consumer that stalls,
then a clock that runs on). Iterative deepening over the budget means the first schedule found for a rule is one with
the fewest delays, which is usually the short, readable one.

Only conflicting choices branch: when every enabled action is independent of the default one (disjoint resources),
taking them in another order cannot change what the ledger sees, so the explorer does not branch there (a sleep-set
style reduction; the tests check it finds the same violations with and without it). States already seen with at least
as much budget left are not explored again.

A Finding is a broken rule with the schedule that broke it, reduced: the explorer replays the schedule with each
departure from the default undone in turn and keeps only the departures the violation needs, then keeps only the
ledger events that touch the violating regions, operations and requests (the causal trace). Result.coverage says how
much was explored and what was not (the model's assumptions, the bounds that were hit)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .ledger import RULES


@dataclass(frozen=True)
class Action:
    """One thing that may happen next. footprint: the resources it reads or writes (regions, request ids, op ids,
    'clock'); two actions conflict when their footprints intersect."""
    label: str
    actor: str
    footprint: frozenset           # "*" means it may change what anything else can do (the clock)
    run: object = field(compare=False, repr=False, default=None)


@dataclass
class Finding:
    rule: str
    message: str
    schedule: list                 # the labels of the reduced schedule, in order
    departures: list               # (step, chosen label, default label) where the schedule left the default order
    trace: list                    # ledger event lines that touch the violation
    delays: int
    then: list = field(default_factory=list)   # what happens next if nothing stops it (effects, later violations)

    def as_dict(self) -> dict:
        return {"rule": self.rule, "rule_text": RULES[self.rule], "message": self.message, "delays": self.delays,
                "schedule": self.schedule, "departures": [list(d) for d in self.departures], "trace": self.trace,
                "then": self.then}


@dataclass
class Result:
    model: str
    findings: list
    coverage: dict

    @property
    def verdict(self) -> str:
        return "VIOLATION" if self.findings else "NONE_FOUND"

    def as_dict(self) -> dict:
        return {"model": self.model, "verdict": self.verdict, "findings": [f.as_dict() for f in self.findings],
                "coverage": self.coverage}


def _conflicts(a: Action, b: Action) -> bool:
    """Whether the order of a and b can matter. "*" in a footprint (the clock: time enables timeouts and arrivals
    anywhere) conflicts with everything."""
    return "*" in a.footprint or "*" in b.footprint or bool(a.footprint & b.footprint)


def explore(model, max_delays: int = 3, max_steps: int = 80, reduce_por: bool = True,
            max_paths: int = 200000) -> Result:
    """Search model's schedules with up to max_delays departures from the default order. Returns every distinct
    violation shape found (by rule and kind), each with its fewest-delay schedule, reduced."""
    found: dict = {}
    stats = {"paths": 0, "states": 0, "pruned_seen": 0, "pruned_independent": 0, "step_bound_hit": 0,
             "path_bound_hit": False}
    for budget in range(max_delays + 1):
        seen: dict = {}
        _dfs(model, model.initial(), [], [], budget, budget, max_steps, reduce_por, seen, found, stats, max_paths)
        if stats["path_bound_hit"]:
            break
    findings = []
    for key in sorted(found, key=lambda k: (found[k][2], k)):
        v, labels, delays = found[key]
        findings.append(_reduce(model, v, labels, delays))
    cov = {"max_delays": max_delays, "max_steps": max_steps, "partial_order_reduction": reduce_por, **stats,
           "exhaustive_within_bounds": not stats["path_bound_hit"] and stats["step_bound_hit"] == 0,
           "assumptions": list(getattr(model, "assumptions", ())),
           "not_covered": list(getattr(model, "not_covered", ()))}
    return Result(getattr(model, "name", type(model).__name__), findings, cov)


def _dfs(model, state, labels, choices, budget, total, max_steps, por, seen, found, stats, max_paths):
    if stats["paths"] >= max_paths:
        stats["path_bound_hit"] = True
        return
    led = state.ledger
    if led.violations:
        v = led.violations[0]
        key = v.key()
        delays = total - budget
        if key not in found or len(labels) < len(found[key][1]) and delays <= found[key][2]:
            found[key] = (v, list(labels), delays)
        stats["paths"] += 1
        return
    acts = model.enabled(state)
    if not acts or model.done(state):
        stats["paths"] += 1
        return
    if len(labels) >= max_steps:
        stats["step_bound_hit"] += 1
        stats["paths"] += 1
        return
    sig = model.signature(state)
    if seen.get(sig, -1) >= budget:
        stats["pruned_seen"] += 1
        return
    seen[sig] = budget
    stats["states"] += 1
    default = acts[0]
    for k, a in enumerate(acts):
        if k > budget:
            break
        if por and k > 0 and not _conflicts(a, default):
            stats["pruned_independent"] += 1
            continue
        nxt = copy.deepcopy(state)
        a.run(nxt, len(labels))
        _dfs(model, nxt, labels + [a.label], choices + [(len(labels), a.label, default.label) if k else None],
             budget - k, total, max_steps, por, seen, found, stats, max_paths)
        if stats["path_bound_hit"]:
            return


def replay(model, labels, strict: bool = False, stop: bool = True):
    """Run model following labels as a wish list: at each step take the head label if it is enabled; otherwise take
    the default action (and cross it off the list if it appears there), or stop when strict. Then finish in the
    default order. Returns (state, labels taken, departures), a departure being (step, chosen, default, index)."""
    state = model.initial()
    taken, departures = [], []
    queue = list(labels)
    step = 0
    while not (stop and state.ledger.violations) and not model.done(state) and step < 400:
        acts = model.enabled(state)
        if not acts:
            break
        choice, idx = acts[0], 0
        if queue:
            hit = [i for i, a in enumerate(acts) if a.label == queue[0]]
            if hit:
                idx = hit[0]
                choice = acts[idx]
                queue.pop(0)
            elif strict:
                break
            elif acts[0].label in queue:
                queue.remove(acts[0].label)
        if idx:
            departures.append((step, choice.label, acts[0].label, idx))
        choice.run(state, step)
        taken.append(choice.label)
        step += 1
    return state, taken, departures


def _cost(departures) -> tuple:
    return (len(departures), sum(d[3] for d in departures))


def _reduce(model, v, labels, delays) -> Finding:
    """Move each departure from the default order back toward the default while the violation persists, so only
    the departures it needs remain; then keep the ledger events that touch it."""
    key = v.key()
    best = replay(model, labels)
    if not _has(best[0], key):
        best = replay(model, labels, strict=True)
    for _ in range(200):
        improved = False
        for d in best[2]:
            step_i, chosen, default = d[0], d[1], d[2]
            rest = list(best[1][step_i:])
            if default in rest:
                rest.remove(default)
            trial = best[1][:step_i] + [default] + rest
            cand = replay(model, trial)
            if _has(cand[0], key) and _cost(cand[2]) < _cost(best[2]):
                best, improved = cand, True
                break
        if not improved:
            break
    state, taken, departures = best
    viol = next(x for x in state.ledger.violations if x.key() == key)
    trace = _causal(state.ledger, viol)
    after, _, _ = replay(model, taken, stop=False)
    then = [f"step {e.step:>3}  {e.kind:<9} {e.text}" for e in after.ledger.events
            if e.step > viol.step and e.kind in ("effect", "violation")]
    return Finding(viol.rule, viol.message, taken, [d[:3] for d in departures], trace, len(departures), then)


def _has(state, key) -> bool:
    return any(x.key() == key for x in state.ledger.violations)


def _causal(led, viol) -> list:
    """The ledger events that share a region, operation or request with the violation, transitively through
    operations (an operation pulls in the regions it touched)."""
    regions, ops, reqs = set(viol.regions), set(viol.ops), set(r for r in viol.requests if r)
    for o in list(ops):
        op = led.ops.get(o)
        if op is not None:
            regions |= {h.region for h in (op.src, op.dst) if h is not None}
    out = []
    events = sorted(led.events, key=lambda e: (e.step, e.kind == "violation"))
    for e in events:
        if e.step > viol.step:
            break
        if set(e.regions) & regions or set(e.ops) & ops or (set(e.requests) & reqs and e.kind in
                                                             ("alloc", "free", "publish", "accept", "cancel",
                                                              "submit", "consume", "violation", "effect")):
            out.append(f"step {e.step:>3}  {e.kind:<9} {e.text}")
    return out
