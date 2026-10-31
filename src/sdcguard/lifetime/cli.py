"""sdcguard lifetime: search a serving protocol's event orderings for KV-state lifetime bugs.

  sdcguard lifetime list                         the models and what each one is
  sdcguard lifetime explore pd-abort             search the buggy variant (exit 1 if a rule broke)
  sdcguard lifetime explore pd-lease --variant fixed --max-delays 5 --json out.json
  sdcguard lifetime selfcheck                    every model: the buggy variant must break a rule, the fixed one not
  sdcguard lifetime merge DIR                    check the event files recorded by a real server's processes
  sdcguard lifetime rules DIR                    judge the safety rules (obligations.py) for every recorded attempt
  sdcguard lifetime conformance --json           run pinned vLLM/SGLang component functions on CPU
  sdcguard lifetime search-toy pull [--fixed]    the model-free search on a small real implementation
  sdcguard lifetime search-toy push --json out.json   ... and write each finding's holds as a replayable spec
  sdcguard lifetime replay-toy push holds.json --fixed  rerun a finding's holds (a pinned regression check)

Exit codes: 0 nothing broke (or selfcheck passed), 1 a rule broke (or selfcheck failed), 2 could not tell (for rules:
no violation, but a rule could not be judged for some attempt)."""

from __future__ import annotations

import argparse
import json

from . import models as M
from .explore import explore
from .report import text


def main(argv=None) -> int:
    # Preserve the component command's own parser and lazy loading. It accepts
    # only packaged, pinned ordinary functions, never a caller's source path.
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["conformance"]:
        from .component_conformance import main as component_main
        return component_main(arguments[1:])
    p = argparse.ArgumentParser(prog="sdcguard lifetime", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("conformance", help="pinned original vLLM/SGLang functions with CPU mocks; no deployed-engine proof")
    sub.add_parser("list", help="the protocol models")
    e = sub.add_parser("explore", help="search one model's orderings")
    e.add_argument("model", choices=sorted(M.MODELS))
    e.add_argument("--variant", default="buggy", help="buggy (default), fixed, or a model's own (see list)")
    e.add_argument("--max-delays", type=int, default=3, help="departures from the normal order to try (default 3)")
    e.add_argument("--max-steps", type=int, default=80)
    e.add_argument("--json", metavar="PATH", help="also write the result as JSON")
    g = sub.add_parser("merge", help="merge recorded event files (one per process) and check the rules")
    g.add_argument("dir")
    g.add_argument("--json", metavar="PATH")
    o = sub.add_parser("rules", help="judge the safety rules for every recorded attempt (never from the answers)")
    o.add_argument("dir")
    o.add_argument("--json", metavar="PATH")
    t = sub.add_parser("search-toy", help="the model-free search on a small real implementation (pull or push)")
    t.add_argument("toy", choices=("pull", "push", "wakeup"))
    t.add_argument("--fixed", action="store_true")
    t.add_argument("--depth", type=int, default=1, choices=(1, 2))
    t.add_argument("--random-runs", type=int, default=0, help="extra runs with random short pauses at checkpoints")
    t.add_argument("--json", metavar="PATH", help="also write the result, with each finding's holds spec")
    r = sub.add_parser("replay-toy", help="rerun a holds spec (a finding's 'spec', or a list of them) on a toy")
    r.add_argument("toy", choices=("pull", "push", "wakeup"))
    r.add_argument("spec", help="JSON file: a holds list, or a search result written with --json (first finding)")
    r.add_argument("--fixed", action="store_true")
    r.add_argument("--times", type=int, default=3)
    s = sub.add_parser("selfcheck", help="every buggy variant must break a rule and every fixed one must not")
    s.add_argument("--max-delays", type=int, default=4)
    a = p.parse_args(arguments)
    if a.cmd == "list":
        for k, cls in M.MODELS.items():
            print(f"{k:<14} variants: {', '.join(cls.VARIANTS)}")
            print(f"{'':<14} {' '.join((cls.__doc__ or '').split())}")
        return 0
    if a.cmd == "explore":
        r = explore(M.model(a.model, a.variant), max_delays=a.max_delays, max_steps=a.max_steps)
        print(text(r))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(r.as_dict(), f, indent=2)
        if not r.findings and not r.coverage["exhaustive_within_bounds"]:
            return 2
        return 1 if r.findings else 0
    if a.cmd == "merge":
        from .adapter import merge
        led = merge(a.dir, track_content=False)
        print(f"sdcguard lifetime merge: {len(led.events)} events from {a.dir}")
        if not led.violations:
            print("NONE FOUND: no recorded event broke a lifetime rule.")
        for v in led.violations:
            print(f"{v.rule} at event {v.step}: {v.message}")
        if a.json:
            with open(a.json, "w") as f:
                json.dump({"events": len(led.events), "violations": [vars(v) for v in led.violations]}, f, indent=2,
                          default=list)
        return 1 if led.violations else 0
    if a.cmd == "rules":
        from .adapter import read_events, merge_events
        from .obligations import evaluate
        evs = read_events(a.dir)
        try:
            led = merge_events(evs, track_content=False)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            evs.append({"kind": "evidence", "args": {"what": "recording_error",
                        "reason": f"legacy ledger could not reconstruct trace: {type(exc).__name__}"}, "t": 0})
            led = None
        found = evaluate(evs, led.violations if led else [])
        print(f"sdcguard lifetime rules: {len(found)} findings over {len({f.request for f in found})} attempts "
              f"from {a.dir}")
        for f in found:
            print(f"{f.verdict.upper():<19} {f.rule} for {f.request}: {f.relationship}"
                  + (f" [missing: {'; '.join(f.missing)}]" if f.missing else "")
                  + (f" ({f.detail})" if f.detail else ""))
        if a.json:
            with open(a.json, "w") as fh:
                json.dump([f.as_dict() for f in found], fh, indent=2, default=list)
        if any(f.verdict == "violated" for f in found):
            return 1
        return 2 if any(f.verdict == "unknown" for f in found) or not found else 0
    if a.cmd == "search-toy":
        from .realsearch import search, text as rtext
        from .toys import TOYS
        cls = TOYS[a.toy]
        res = search(lambda p: cls(p, fixed=a.fixed), lambda s: s.run(), depth=a.depth, hold_timeout=0.8,
                     run_timeout=5.0, random_runs=a.random_runs)
        print(rtext(res, f"toy {a.toy} ({'fixed' if a.fixed else 'buggy'})"))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res.as_dict(), f, indent=2, default=list)
        return 0 if res.verdict == "NONE_FOUND" else 1
    if a.cmd == "replay-toy":
        from .realsearch import replay
        from .toys import TOYS
        with open(a.spec) as f:
            spec = json.load(f)
        if isinstance(spec, dict):
            fs = spec.get("findings") or spec.get("liveness") or []
            if not fs:
                print("the result file has no finding to replay")
                return 2
            spec = fs[0]["spec"]
        cls = TOYS[a.toy]
        keys = replay(lambda p: cls(p, fixed=a.fixed), lambda s: s.run(), spec, times=a.times, run_timeout=5.0)
        for i, k in enumerate(keys, 1):
            print(f"run {i}: {'clean' if k is None else ' '.join(map(str, k))}")
        bad = sum(k is not None for k in keys)
        print(f"{bad} of {len(keys)} runs broke a rule or did not finish")
        return 1 if bad else 0
    if a.cmd == "selfcheck":
        ok = True
        for k in M.MODELS:
            for variant, want in (("buggy", True), ("fixed", False)):
                r = explore(M.model(k, variant), max_delays=a.max_delays)
                good = bool(r.findings) == want and r.coverage["exhaustive_within_bounds"]
                ok &= good
                rules = ",".join(sorted({f.rule for f in r.findings})) or "none"
                print(f"{'ok  ' if good else 'FAIL'} {k}:{variant:<6} rules broken: {rules:<8} "
                      f"({r.coverage['paths']} schedules, {r.coverage['states']} states)")
        print("selfcheck passed" if ok else "selfcheck FAILED")
        return 0 if ok else 1
    p.print_help()
    return 0
