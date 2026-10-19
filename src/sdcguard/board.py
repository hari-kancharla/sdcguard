"""The public results board: every shared `sdcguard check` card, by GPU model and by provider.

    sdcguard board --dir results/ --out docs/board            # from result.json files
    python src/sdcguard/board.py --issues OWNER/REPO --out docs/board    # from GitHub issues (the Action)

People share a result with `sdcguard check --share`, which opens a GitHub issue holding the card's JSON.
A GitHub Action runs this file on every new result issue and on a schedule, and commits:

  index.html       the board: totals, a table per GPU model and per provider, recent results
  results.json     every accepted result, with a link to where it came from
  reference.json   what `sdcguard check` compares against: how many runs of each exact setup gave each
                   output fingerprint, and median speeds per GPU model

Results are self-reported. Each card carries a digest of its contents, hardware identifiers are hashed,
and a FAIL or SUSPECT links to its issue, where the evidence lives. Simulated-fault demos, CPU runs and
runs shorter than a minute are not counted.

Standard library only (no PyTorch), so the Action needs nothing installed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import json
import math
import os
import subprocess
import sys
from typing import Optional

if __package__:
    from ._capabilities import expected
else:                                       # GitHub Actions runs this file without installing torch
    from _capabilities import expected

SCHEMA = "sdcguard-check/1"
RESULTS = ("PASS", "FAIL", "SUSPECT")
MIN_MINUTES = 1.0


def digest(pub: dict) -> str:
    body = {k: v for k, v in pub.items() if k != "digest" and not k.startswith("_")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


def extract(text: str) -> Optional[dict]:
    """The first JSON object in a block of text (an issue body)."""
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except (ValueError, RecursionError):
            pass
        i = text.find("{", i + 1)
    return None


def _num_ok(v, lo=0.0, hi=1e15) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi)


def _shape_problem(r: dict) -> Optional[str]:
    """Anyone can write a card by hand (the digest only catches edits), so check every field's type."""
    for k in ("gpu", "software", "provider", "checked", "wrong", "headroom", "fingerprint", "speed", "health", "board"):
        if r.get(k) is not None and not isinstance(r.get(k), dict):
            return f"{k} is not an object"
    g, f, p = r.get("gpu") or {}, r.get("fingerprint") or {}, r.get("provider") or {}
    for v in (g.get("name"), g.get("id_hash"), f.get("key"), f.get("output"), p.get("name"), r.get("date")):
        if v is not None and not isinstance(v, str):
            return "a text field holds something else"
    if not _num_ok(r.get("minutes"), 0, 1e6):
        return "minutes is not a number"
    for k, v in (r.get("speed") or {}).items():
        if k != "dtype" and not _num_ok(v, -1e9, 1e9):
            return f"speed.{k} is not a number"
    for part in ("checked", "wrong", "headroom"):
        for k, v in (r.get(part) or {}).items():
            if part == "checked" and k == "memory_skipped":       # (why the memory test didn't run, or None)
                if v is not None and not isinstance(v, str):
                    return "checked.memory_skipped is malformed"
                continue
            if not _num_ok(v, -1e18, 1e18):
                return f"{part}.{k} is not a number"
    if r.get("stopped") is not None and not isinstance(r.get("stopped"), dict):
        return "stopped is malformed"
    mm = r.get("math_modes")
    if mm is not None:
        if not isinstance(mm, dict):
            return "math_modes is not an object"
        if not all(_num_ok(mm.get(k), 0, 1e18) for k in ("matmuls", "wrong")):
            return "math_modes counts are not numbers"
        if not isinstance(mm.get("ran"), list) or not all(isinstance(x, str) for x in mm["ran"]):
            return "math_modes.ran is malformed"
        for k in ("not_checked", "errors", "typical_error_caught"):
            d = mm.get(k, {})
            if not isinstance(d, dict) or not all(isinstance(x, str) for x in d):
                return f"math_modes.{k} is malformed"
        if not all(isinstance(v, str) for v in list(mm.get("not_checked", {}).values()) + list(mm.get("errors", {}).values())) or \
                not all(v == "exact sums" or _num_ok(v, 0, 1e6) for v in mm.get("typical_error_caught", {}).values()):
            return "math_modes values are malformed"
    notes = r.get("notes") or []
    if not isinstance(notes, list) or not all(isinstance(n, list) and len(n) == 2 and all(isinstance(x, str) for x in n)
                                              for n in notes):
        return "notes are malformed"
    return None


_KEYS = {"schema", "result", "date", "minutes", "gpu", "software", "provider", "checked", "wrong", "hardware_errors",
         "stopped", "headroom", "repeatable", "fingerprint", "speed", "health", "notes", "board", "simulated_faults",
         "math_modes", "digest"}
_WRONG = {"workload_training", "workload_serving", "stress", "math_modes", "memory_words"}
MAX_CARD_BYTES = 64 * 1024


def _all_finite(v, depth: int = 0) -> bool:
    """No NaN or Infinity anywhere (they make results.json invalid JSON), no huge strings, no deep nesting."""
    if depth > 8:
        return False
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, str):
        return len(v) <= 2000
    if isinstance(v, dict):
        return all(isinstance(k, str) and len(k) <= 200 and _all_finite(x, depth + 1) for k, x in v.items())
    if isinstance(v, list):
        return all(_all_finite(x, depth + 1) for x in v)
    return v is None or isinstance(v, (bool, int))


def _count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 10 ** 15


def _consistency_problem(r: dict) -> Optional[str]:
    """A card's parts have to agree with its result. (The board can check that a card is consistent; it can't
    prove a hand-made card describes a real run. The digest only shows it wasn't edited by accident.)"""
    res, wrong, checked = r.get("result"), r.get("wrong") or {}, r.get("checked") or {}
    if set(wrong) - _WRONG:
        return "it counts wrong results in a part the check doesn't have"
    if not {"workload_training", "workload_serving", "stress", "memory_words"} <= set(wrong):
        return "its wrong-result counts are incomplete"
    if not all(_count(v) for v in wrong.values()) or not _count(checked.get("results")) or \
            not _count(checked.get("unjudged") or 0):
        return "its counts aren't whole numbers"
    if r.get("repeatable") is not None and not isinstance(r.get("repeatable"), bool):
        return "repeatable is not true or false"
    if r.get("hardware_errors") is not None and (not isinstance(r["hardware_errors"], list) or
                                                 not all(isinstance(x, str) for x in r["hardware_errors"])):
        return "hardware_errors is malformed"
    name = str((r.get("gpu") or {}).get("name") or "").strip().lower()
    if not name or name in ("cuda", "none") or name.startswith("cpu"):
        return "not a GPU result"
    mm = r.get("math_modes") or {}
    if mm and "math_modes" not in wrong:
        return "its math-mode results aren't in its wrong-result counts"
    b = r.get("board") or {}
    odd = any(lvl == "bad" and ("odd one out" in t or "differs from" in t) for lvl, t in r.get("notes") or [])
    if res == "PASS":
        if any(lvl == "bad" for lvl, _t in r.get("notes") or []):
            return "says PASS, but its notes report a problem"
        if r.get("repeatable") is not True:
            return "says PASS, but its repeatability wasn't confirmed"
        if b.get("disagrees_with_majority") or odd:
            return "says PASS, but its fingerprint disagrees with other GPUs (that's SUSPECT)"
        if r.get("hardware_errors") or r.get("stopped") or checked.get("memory_skipped") or \
                (checked.get("serving_nonfinite") or 0) > 0:
            return "says PASS, but part of the check didn't run or reported errors (that's INCOMPLETE or FAIL)"
        cc = str((r.get("gpu") or {}).get("compute_cap") or "")
        try:
            cap = tuple(int(x) for x in cc.split(".")[:2]) if cc else None
        except ValueError:
            cap = None
        for m in (mm.get("not_checked") or {}):
            if cap is None or expected(m, cap):
                return f"says PASS, but the {m} math mode didn't run on a GPU that supports it"
    if res == "FAIL" and not sum(wrong.values()) and not r.get("hardware_errors"):
        return "says FAIL, but counts no wrong results"
    if res == "SUSPECT" and r.get("repeatable") is not False and not b.get("disagrees_with_majority") and not odd:
        return "says SUSPECT, but nothing on it disagrees"
    return None


def validate(r: dict) -> Optional[str]:
    """None if the card can go on the board, else the reason it can't."""
    if not isinstance(r, dict) or r.get("schema") != SCHEMA:
        return f"not a {SCHEMA} result card"
    try:
        if len(json.dumps(r, default=str)) > MAX_CARD_BYTES:
            return f"larger than {MAX_CARD_BYTES // 1024} KB"
    except (TypeError, ValueError, RecursionError):
        return "malformed card"
    # _source is attached by from_dirs/from_issues after ingestion; it is not part of the card's digest.
    # Reject unknown card fields while retaining the board's own provenance metadata.
    extra = set(r) - _KEYS - {"_source"}
    if extra:
        return "malformed card (fields the check doesn't write: " + ", ".join(sorted(map(str, extra)))[:120] + ")"
    if not _all_finite(r):
        return "malformed card (NaN, Infinity, an oversized text or deep nesting)"
    shape = _shape_problem(r)
    if shape:
        return f"malformed card ({shape})"
    if r.get("digest") != digest(r):
        return "the digest doesn't match the card (it was changed after `sdcguard check` wrote it)"
    if r.get("result") == "INCOMPLETE":
        return "an incomplete check (part of it couldn't run or couldn't judge its results): rerun it"
    if r.get("result") not in RESULTS:
        return "unknown result"
    if r.get("simulated_faults"):
        return "a simulated-fault demo, not a real result"
    name = str((r.get("gpu") or {}).get("name") or "")
    if not name or name.lower() in ("cpu", "cuda", "none"):
        return "not a GPU result"
    try:
        if float(r.get("minutes") or 0) < MIN_MINUTES:
            return f"shorter than {MIN_MINUTES:g} minute (run the default `sdcguard check`)"
        tf = (r.get("speed") or {}).get("matmul_tflops")
        if tf is not None and not 0 <= float(tf) < 50000:
            return "implausible speed"
        wrong = [float(v) for v in (r.get("wrong") or {}).values()]
        checked = float((r.get("checked") or {}).get("results") or 0)
        if any(v < 0 for v in wrong) or checked < 0:
            return "negative counts"
        mm = r.get("math_modes") or {}
        if mm and "math_modes" in (r.get("wrong") or {}) and float(mm.get("wrong") or 0) != float(r["wrong"]["math_modes"]):
            return "its two math-mode counts disagree"
        unjudged = float((r.get("checked") or {}).get("unjudged") or 0)
        if r["result"] == "PASS" and (checked == 0 or sum(wrong) > 0 or r.get("repeatable") is False
                                      or float(mm.get("wrong") or 0) > 0 or mm.get("errors")
                                      or unjudged > 0):
            return "says PASS, but its counts don't (nothing checked, wrong or unjudged results, or not repeatable)"
        if r["result"] == "SUSPECT" and sum(wrong) > 0:
            return "says SUSPECT, but counts wrong results (that's a FAIL)"
    except (TypeError, ValueError):
        return "malformed numbers"
    return _consistency_problem(r)


def _s(v, n: int = 80) -> str:
    return str(v)[:n] if v is not None else ""


def _median(v):
    v = sorted(float(x) for x in v if isinstance(x, (int, float)))
    return v[len(v) // 2] if v else None


def _provider(r) -> str:
    p = r.get("provider") or {}
    name = _s(p.get("name"), 40) or "not stated"
    return name


def _slow(r) -> bool:
    return any(level in ("warn", "bad") and "slowed itself down" in str(text) for level, text in (r.get("notes") or []))


def aggregate(results: list) -> dict:
    models: dict = {}
    providers: dict = {}
    fps: dict = {}            # key -> output -> {"gpus": set, "sources": set}
    for r in results:
        g = r["gpu"]
        name = _s(g.get("name"), 60)
        m = models.setdefault(name, {"runs": 0, "gpus": set(), "PASS": 0, "FAIL": 0, "SUSPECT": 0, "tflops": [],
                                     "pct": [], "bw": [], "slow": 0, "fingerprints": {}})
        m["runs"] += 1
        m["gpus"].add(g.get("id_hash") or r["digest"])
        m[r["result"]] += 1
        sp = r.get("speed") or {}
        m["tflops"].append(sp.get("matmul_tflops"))
        m["pct"].append(sp.get("pct_of_datasheet"))
        m["bw"].append(sp.get("bandwidth_gbs"))
        m["slow"] += _slow(r)
        f = r.get("fingerprint") or {}
        if f.get("key") and f.get("output"):
            src = r.get("_source") or {}
            who = src.get("author") or src.get("root") or "files"
            slot = fps.setdefault(f["key"], {}).setdefault(f["output"], {"gpus": set(), "sources": set()})
            slot["gpus"].add(g.get("id_hash") or r["digest"])
            slot["sources"].add(who)
            m["fingerprints"].setdefault(f["key"], {}).setdefault(f["output"], 0)
            m["fingerprints"][f["key"]][f["output"]] += 1
        p = providers.setdefault(_provider(r), {"runs": 0, "models": set(), "PASS": 0, "FAIL": 0, "SUSPECT": 0,
                                                "pct": [], "slow": 0})
        p["runs"] += 1
        p["models"].add(name)
        p[r["result"]] += 1
        p["pct"].append(sp.get("pct_of_datasheet"))
        p["slow"] += _slow(r)
    for m in models.values():
        agree = [max(v.values()) for v in m["fingerprints"].values()]
        total = [sum(v.values()) for v in m["fingerprints"].values()]
        m["fingerprint_agreement"] = (sum(agree), sum(total))
    totals = {"results": len(results), "gpus": len({r["gpu"].get("id_hash") or r["digest"] for r in results}),
              "models": len(models), "providers": len([p for p in providers if p != "not stated"]),
              "checked_results": sum(int((r.get("checked") or {}).get("results") or 0) for r in results),
              "workload_tflop": round(sum(float((r.get("checked") or {}).get("workload_tflop") or 0) for r in results), 1),
              "memory_gib": round(sum(float((r.get("checked") or {}).get("memory_gib") or 0) for r in results), 1),
              **{k: sum(r["result"] == k for r in results) for k in RESULTS}}
    return {"totals": totals, "models": models, "providers": providers, "fingerprints": fps}


def reference(agg: dict) -> dict:
    return {"schema": "sdcguard-board-reference/1",
            "generated": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
            "fingerprints": {k: {o: {"gpus": len(v["gpus"]), "sources": len(v["sources"])} for o, v in outs.items()}
                             for k, outs in agg["fingerprints"].items()},
            "models": {n: {"runs": m["runs"], "median_tflops": _median(m["tflops"]),
                           "median_bandwidth_gbs": _median(m["bw"])} for n, m in agg["models"].items()}}


_CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--line:#e7e5e4;--card:#fff;--ok:#15803d;--bad:#b91c1c;--warn:#b45309}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;
--line:#292524;--card:#1c1917;--ok:#4ade80;--bad:#f87171;--warn:#fbbf24}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 16px 48px}h1{font-size:26px;margin:0 0 6px}
h2{font-size:18px;margin:32px 0 10px}p{margin:0 0 12px}.sub{color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.stat b{display:block;font-size:24px}.stat span{color:var(--muted);font-size:13px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:9px 11px;border-bottom:1px solid var(--line);
text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600}tr:last-child td{border-bottom:0}
.PASS{color:var(--ok);font-weight:700}.FAIL{color:var(--bad);font-weight:700}.SUSPECT{color:var(--warn);font-weight:700}
code{font-size:13px;background:var(--line);padding:1px 6px;border-radius:6px}a{color:inherit}
.how{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-top:20px}
"""


def _num(x, digits=0) -> str:
    if x is None:
        return ""
    return f"{x:,.{digits}f}"


def html_page(agg: dict, results: list, repo: Optional[str]) -> str:
    e = html.escape
    t = agg["totals"]
    mrows = []
    for name, m in sorted(agg["models"].items(), key=lambda kv: -kv[1]["runs"]):
        a, n = m["fingerprint_agreement"]
        mrows.append("<tr>" + "".join(f"<td>{c}</td>" for c in (
            e(name), m["runs"], len(m["gpus"]), f"<span class='PASS'>{m['PASS']}</span>" if m["PASS"] else "0",
            f"<span class='FAIL'>{m['FAIL']}</span>" if m["FAIL"] else "0",
            f"<span class='SUSPECT'>{m['SUSPECT']}</span>" if m["SUSPECT"] else "0",
            _num(_median(m["tflops"]), 0), (_num(_median(m["pct"]), 0) + "%") if _median(m["pct"]) else "",
            _num(_median(m["bw"]), 0), f"{m['slow']} of {m['runs']}", f"{a} of {n}" if n else "")) + "</tr>")
    prows = []
    for name, p in sorted(agg["providers"].items(), key=lambda kv: -kv[1]["runs"]):
        prows.append("<tr>" + "".join(f"<td>{c}</td>" for c in (
            e(name), p["runs"], e(", ".join(sorted(p["models"]))[:120]),
            f"<span class='PASS'>{p['PASS']}</span>" if p["PASS"] else "0",
            f"<span class='FAIL'>{p['FAIL']}</span>" if p["FAIL"] else "0",
            f"<span class='SUSPECT'>{p['SUSPECT']}</span>" if p["SUSPECT"] else "0",
            (_num(_median(p["pct"]), 0) + "%") if _median(p["pct"]) else "", f"{p['slow']} of {p['runs']}")) + "</tr>")
    recent = []
    for r in sorted(results, key=lambda r: r.get("_source", {}).get("time") or r.get("date") or "", reverse=True)[:50]:
        src = r.get("_source") or {}
        link = f"<a href='{e(src['url'])}'>#{e(str(src.get('number')))}</a>" if src.get("url") else e(src.get("label", ""))
        sp = r.get("speed") or {}
        recent.append("<tr>" + "".join(f"<td>{c}</td>" for c in (
            e(_s(r.get("date"), 10)), f"<span class='{r['result']}'>{r['result']}</span>", e(_s(r["gpu"].get("name"), 60)),
            e(_provider(r)), _num(sp.get("matmul_tflops"), 0), _num((r.get("checked") or {}).get("results"), 0),
            e(_s((r.get("fingerprint") or {}).get("output"), 12)), link)) + "</tr>")
    share = (f"<a href='https://github.com/{e(repo)}/issues?q=label%3Aresult'>GitHub issues</a>" if repo else "GitHub issues")
    home = f" <a href='https://github.com/{e(repo)}'>Code and docs on GitHub</a>." if repo else ""

    def table(head, rows):
        return ("<div class='wrap'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
                + ("".join(rows) or f"<tr><td colspan='{len(head)}'>No results yet.</td></tr>") + "</tbody></table></div>")

    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU Correctness Board</title>
<meta name="description" content="Do GPUs compute correctly? Shared sdcguard check results by GPU model and provider.">
<style>{_CSS}</style></head><body><main>
<h1>GPU correctness board</h1>
<p class="sub">Every GPU here ran <code>sdcguard check</code>: a real transformer workload with every matrix multiply
checked against checksums, a memory pattern test, speed tests, and the same fixed model run three times to see if its
output stays bit-identical. Updated {e(generated)}.</p>
<div class="stats">
<div class="stat"><b>{t['gpus']:,}</b><span>GPUs checked</span></div>
<div class="stat"><b>{t['models']:,}</b><span>GPU models</span></div>
<div class="stat"><b>{t['checked_results']:,}</b><span>results checked</span></div>
<div class="stat"><b>{t['workload_tflop']:,.0f}</b><span>TFLOP of checked work</span></div>
<div class="stat"><b class="FAIL">{t['FAIL']:,}</b><span>FAIL (wrong results)</span></div>
<div class="stat"><b class="SUSPECT">{t['SUSPECT']:,}</b><span>SUSPECT (not repeatable)</span></div>
</div>
<h2>By GPU model</h2>
{table(("GPU", "runs", "distinct GPUs", "PASS", "FAIL", "SUSPECT", "median matmul TFLOPS", "of datasheet peak",
        "median GB/s", "slowed down", "same fingerprint"), mrows)}
<p class="sub" style="margin-top:8px">"Same fingerprint": runs of the identical setup (GPU model, software versions) whose
fixed-model output matched the most common one. A healthy GPU should always match.</p>
<h2>By provider (as reported)</h2>
{table(("provider", "runs", "GPU models", "PASS", "FAIL", "SUSPECT", "median of datasheet peak", "slowed down"), prows)}
<h2>Recent results</h2>
{table(("date", "result", "GPU", "provider", "TFLOPS", "results checked", "fingerprint", "source"), recent)}
<div class="how"><p><b>Add your GPU:</b> <code>pip install sdcguard &amp;&amp; sdcguard check --share</code>, then press
Submit on the issue it opens. Results are self-reported and come from {share}; each card carries a digest of its contents,
hardware IDs are hashed, and every FAIL or SUSPECT links to its issue. Simulated-fault demos, CPU runs and runs under a
minute are left out. Raw data: <a href="results.json">results.json</a>, <a href="reference.json">reference.json</a>.{home}</p></div>
</main></body></html>
"""


def from_dirs(dirs: list) -> list:
    out = []
    for root in dirs:
        for d, sub, files in os.walk(root):
            sub.sort()                      # same order on every machine, so the board only changes with its data
            for name in sorted(files):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(d, name)
                try:
                    with open(path) as f:
                        r = json.load(f)
                except (OSError, ValueError):
                    continue
                if isinstance(r, dict) and r.get("schema") == SCHEMA:
                    # folder name only: the board is published, local paths must not end up in it
                    r["_source"] = {"label": os.path.relpath(path, root),
                                    "root": os.path.basename(os.path.normpath(os.path.abspath(root))),
                                    "time": r.get("date")}
                    out.append(r)
    return out


def from_issues(repo: str) -> list:
    raw = subprocess.run(["gh", "issue", "list", "-R", repo, "--label", "result", "--state", "all", "--limit", "5000",
                          "--json", "number,title,body,author,createdAt,url,labels"],
                         capture_output=True, text=True, check=True).stdout
    out = []
    for it in json.loads(raw):
        r = extract(it.get("body") or "") or {}
        r["_source"] = {"number": it["number"], "url": it["url"], "author": (it.get("author") or {}).get("login"),
                        "time": it.get("createdAt"), "labels": [lb.get("name") for lb in it.get("labels") or []]}
        out.append(r)
    return out


def _gh(*args) -> bool:
    p = subprocess.run(["gh", *args], capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  gh {' '.join(args[:3])}: {p.stderr.strip()[:200]}")
    return p.returncode == 0


def build(results: list, out: str, repo: Optional[str] = None) -> dict:
    accepted, rejected, seen = [], [], set()
    for r in results:
        try:
            why = validate(r)
            if why is None:
                key = (r["gpu"].get("id_hash"), r["digest"])
                if key in seen:
                    why = "the same card was already shared"
                else:
                    seen.add(key)
        except Exception as e:  # noqa: BLE001  (a hand-made card must never stop the board)
            why = f"malformed card ({type(e).__name__})"
        (rejected if why else accepted).append((r, why))
    good = []
    for r, _ in accepted:
        try:
            aggregate([r])
            html_page(aggregate([r]), [r], repo)
            good.append(r)
        except Exception as e:  # noqa: BLE001
            rejected.append((r, f"malformed card ({type(e).__name__})"))
    agg = aggregate(good)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(html_page(agg, good, repo))
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(good, f, indent=1, default=str, allow_nan=False)
    with open(os.path.join(out, "reference.json"), "w") as f:
        json.dump(reference(agg), f, indent=1)
    return {"totals": agg["totals"], "accepted": len(good),
            "rejected": [{"source": r.get("_source"), "why": w} for r, w in rejected],
            "reasons": {id(r): w for r, w in rejected}}


def acknowledge(repo: str, results: list, reasons: dict, board_url: str) -> int:
    """Label and comment on every result issue whose state changed: on the board, or what's wrong with it.
    One comment per change, so edits and reruns don't repeat themselves."""
    n = 0
    for r in results:
        src = r.get("_source") or {}
        num = src.get("number")
        if num is None:
            continue
        why = reasons.get(id(r))
        labels = set(src.get("labels") or [])
        if why is None and "on-board" not in labels:
            if _gh("issue", "edit", str(num), "-R", repo, "--add-label", "on-board"):
                if "needs-fix" in labels:
                    _gh("issue", "edit", str(num), "-R", repo, "--remove-label", "needs-fix")
                _gh("issue", "comment", str(num), "-R", repo, "--body",
                    f"Added to the board: {board_url} (card digest `{r.get('digest')}`). Thanks for checking your GPU.")
                n += 1
        elif why is not None and "needs-fix" not in labels:
            if _gh("issue", "edit", str(num), "-R", repo, "--add-label", "needs-fix"):
                if "on-board" in labels:
                    _gh("issue", "edit", str(num), "-R", repo, "--remove-label", "on-board")
                _gh("issue", "comment", str(num), "-R", repo, "--body",
                    f"This result can't go on the board yet: {why}. Paste the card exactly as `sdcguard check --share` "
                    "filled it in (or edit this issue), and it will be picked up automatically.")
                n += 1
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sdcguard board", description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", nargs="*", default=[], help="folders with result.json files")
    p.add_argument("--issues", default=None, help="OWNER/REPO: read result issues with the gh CLI")
    p.add_argument("--out", default="docs/board")
    p.add_argument("--ack", action="store_true", help="with --issues: label and comment on result issues whose state changed")
    p.add_argument("--board-url", default=None)
    a = p.parse_args(argv)
    results = from_dirs(a.dir) if a.dir else []
    if a.issues:
        results += from_issues(a.issues)
    rep = build(results, a.out, a.issues)
    t = rep["totals"]
    print(f"board: {rep['accepted']} results ({t['gpus']} GPUs, {t['models']} models), {t['FAIL']} FAIL, "
          f"{t['SUSPECT']} SUSPECT, {len(rep['rejected'])} not accepted -> {a.out}/index.html")
    for x in rep["rejected"][:20]:
        print(f"  not accepted: {x['source']}: {x['why']}")
    if a.issues and a.ack:
        url = a.board_url or f"https://{a.issues.split('/')[0]}.github.io/{a.issues.split('/')[1]}/board/"
        print(f"board: {acknowledge(a.issues, results, rep['reasons'], url)} issues labelled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
