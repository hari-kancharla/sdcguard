"""Plain-text and JSON reports of an exploration: what broke, the shortest ordering that broke it, the events that
caused it, what would follow, and how much was explored."""

from __future__ import annotations

from .ledger import RULES


def text(result) -> str:
    lines = [f"sdcguard lifetime: {result.model}"]
    if not result.findings:
        lines.append("NONE FOUND: no ordering within the bounds below broke a lifetime rule.")
    else:
        n = len(result.findings)
        lines.append(f"VIOLATION: {n} way{'s' if n > 1 else ''} to break a lifetime rule.")
    for f in result.findings:
        lines.append("")
        lines.append(f"{f.rule} {RULES[f.rule]}")
        lines.append(f"  {f.message}")
        if f.departures:
            lines.append(f"  Ordering ({len(f.departures)} departure{'s' if len(f.departures) != 1 else ''} "
                         f"from the normal order):")
        else:
            lines.append("  Ordering (the normal order breaks it):")
        dep = {d[0]: d for d in f.departures}
        for i, label in enumerate(f.schedule):
            note = f"   (instead of: {dep[i][2]})" if i in dep else ""
            lines.append(f"    {i:>3}  {label}{note}")
        lines.append("  Events that caused it:")
        lines.extend("    " + t for t in f.trace)
        if f.then:
            lines.append("  If nothing stops it:")
            lines.extend("    " + t for t in f.then)
    c = result.coverage
    lines.append("")
    bound = "exhaustive within those bounds" if c.get("exhaustive_within_bounds") else \
        "NOT exhaustive (a step or path bound was hit)"
    lines.append(f"Explored {c['paths']} schedules ({c['states']} states), up to {c['max_delays']} departures from "
                 f"the normal order and {c['max_steps']} steps: {bound}.")
    for a in c.get("assumptions", ()):
        lines.append(f"Assumed: {a}")
    for a in c.get("not_covered", ()):
        lines.append(f"Not covered: {a}")
    lines.append("A model finding is a candidate ordering for the protocol as modeled; it is a finding about a server "
                 "only when the same ordering is driven in the real implementation.")
    return "\n".join(lines)
