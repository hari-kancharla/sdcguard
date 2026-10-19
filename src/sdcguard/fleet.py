"""Fleet report: many screened GPUs (one machine or a whole fleet) in one place.

    sdcguard report screen_out/ other_host_out/ --out fleet_report/

Reads every summary.json (and certificate.json, if present) under the given folders and writes:

  fleet_report.html   one page: how many GPUs passed, which failed, per-GPU details
  fleet_report.json   the same, for scripts and dashboards
  quarantine.txt      the GPUs to pull: UUID, host, PCI slot, serial (empty when all passed)
  rma/<GPU UUID>.zip  for each failed GPU: its summary, full check result (check.json), certificate,
                      evidence (inputs and wrong outputs of the failed matmuls, from every phase) and a
                      README with the replay commands. NVIDIA's RMA process
                      asks for data "illustrating that the failure follows the particular" board; a
                      replay on a healthy GPU giving the right answer, and on this one the wrong answer,
                      is that.
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import os
import zipfile
from typing import Iterable


def collect(roots: Iterable[str]) -> list[dict]:
    """One row per screened GPU (the newest summary for each GPU UUID wins)."""
    rows: dict = {}
    for root in roots:
        for d, _dirs, files in os.walk(root):
            if "summary.json" not in files:
                continue
            try:
                with open(os.path.join(d, "summary.json")) as f:
                    s = json.load(f)
            except (OSError, ValueError):
                continue
            cert = None
            if "certificate.json" in files:
                try:
                    with open(os.path.join(d, "certificate.json")) as f:
                        cert = json.load(f)
                except (OSError, ValueError):
                    cert = None
            chk = None
            if "check.json" in files:
                try:
                    with open(os.path.join(d, "check.json")) as f:
                        chk = json.load(f)
                except (OSError, ValueError):
                    chk = None
            g = s.get("gpu", {})
            key = g.get("gpu_uuid") or os.path.abspath(d)
            # `sdcguard check` writes check.json with every phase (workload, stress check, math modes); a bare
            # `sdcguard screen` has only the workload summary
            wrong = dict((chk or {}).get("wrong") or {"workload_training": s.get("corrupted_results_training") or 0,
                                                     "workload_serving": s.get("corrupted_results_serving") or 0})
            evidence = [_local(e, d) for e in ((chk or {}).get("evidence") or s.get("evidence") or [])]
            row = {"dir": os.path.abspath(d), "gpu_uuid": g.get("gpu_uuid"), "gpu_name": g.get("gpu_name"),
                   "host": g.get("host"), "pci_bus_id": g.get("pci_bus_id"), "serial": g.get("serial"),
                   "driver": g.get("driver"), "result": s.get("result"), "minutes": s.get("minutes"),
                   "matmuls_checked": s.get("matmuls_checked") or 0,
                   "serving_layer_checks": s.get("linear_layer_checks_while_serving") or 0,
                   "tflop": s.get("approx_matmul_tflop") or 0.0,
                   "corrupted": sum(v or 0 for k, v in wrong.items() if k != "memory_words"),
                   "wrong_by_phase": wrong,
                   "worst_clean_score": s.get("worst_clean_z"), "evidence": evidence,
                   "simulated": bool(s.get("SIMULATED_FAULTS")), "certificate": bool(cert),
                   "signed": bool(cert and cert.get("signature")), "issued_at": (cert or {}).get("issued_at"),
                   "mtime": os.path.getmtime(os.path.join(d, "summary.json")),
                   "matmul_tflops": ((chk or {}).get("speed") or {}).get("tflops"),
                   "repeatable": (chk or {}).get("repeatable"),
                   "notes": [t for lvl, t in (chk or {}).get("notes") or [] if lvl in ("bad", "warn")]}
            if chk and chk.get("result") in ("FAIL", "SUSPECT", "INCOMPLETE") and row["result"] != "FAIL":
                row["result"] = chk["result"]
            if key not in rows or row["mtime"] > rows[key]["mtime"]:
                rows[key] = row
    return sorted(rows.values(), key=lambda r: (r["result"] != "FAIL", r["host"] or "", r["pci_bus_id"] or ""))


def _local(e: str, d: str) -> str:
    """An evidence path as recorded, or the same file next to the summary (results moved or copied)."""
    if os.path.exists(e + ".pt") or os.path.exists(e + ".json"):
        return e
    return os.path.join(d, os.path.basename(e))


_PHASES = {"workload_training": "in the training workload", "workload_serving": "in the serving workload",
           "stress": "in the stress check", "math_modes": "in the math-mode check",
           "memory_words": "memory words read back wrong in the memory test"}


def _rma_zip(row: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{row['gpu_uuid'] or 'gpu'}.zip")
    phases = "".join(f"    {v:,} {_PHASES.get(k, k)}\n" for k, v in (row.get("wrong_by_phase") or {}).items() if v)
    readme = (f"sdcguard evidence for GPU {row['gpu_uuid']} ({row['gpu_name']}), serial {row['serial']}, "
              f"host {row['host']}, PCI {row['pci_bus_id']}\n\n"
              f"Result: {row['result']}. {row['corrupted']:,} matrix multiply results failed their checks "
              f"({row['matmuls_checked']:,} workload matmuls and {row['serving_layer_checks']:,} layer outputs "
              f"checked, plus the stress and math-mode checks):\n" + (phases or "    none\n") + "\n"
              "Each .pt file holds the inputs and the output this GPU produced for one failed matmul, and the .json "
              "next to it describes it; check.json is the whole result. To show the failure follows the board, run "
              "on this GPU and on a healthy one:\n\n" +
              ("".join(f"    sdcguard replay {os.path.basename(e)}.pt\n" for e in row["evidence"]) or
               "    (no evidence files: the failure wasn't in a saved matmul)\n") +
              "\nA healthy GPU recomputes the right answer every time (and the recorded output fails the check); "
              "a faulty one tends to repeat the error. A replay that fails again points at the GPU but doesn't by "
              "itself rule out a software or configuration cause.\n")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", readme)
        for name in ("summary.json", "check.json", "certificate.json"):
            p = os.path.join(row["dir"], name)
            if os.path.exists(p):
                z.write(p, name)
        for e in row["evidence"]:
            for ext in (".json", ".pt"):
                if os.path.exists(e + ext):
                    z.write(e + ext, os.path.basename(e) + ext)
    return path


_CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--line:#e7e5e4;--ok:#15803d;--bad:#b91c1c;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#f5f5f4;--muted:#a8a29e;--line:#292524;--ok:#4ade80;
--bad:#f87171;--card:#1c1917}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 20px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.stat b{display:block;font-size:22px}.stat span{color:var(--muted);font-size:13px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);
text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:600}
.PASS{color:var(--ok);font-weight:700}.FAIL,.SUSPECT{color:var(--bad);font-weight:700}.INCOMPLETE{color:var(--muted);font-weight:700}
code{font-size:12px}
"""


def write(rows: list[dict], out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    failed = [r for r in rows if r["result"] in ("FAIL", "SUSPECT")]
    stats = {"gpus": len(rows), "passed": sum(r["result"] == "PASS" for r in rows), "failed": len(failed),
             "suspect": sum(r["result"] == "SUSPECT" for r in rows),
             "incomplete": sum(r["result"] == "INCOMPLETE" for r in rows),
             "matmuls_checked": sum(r["matmuls_checked"] for r in rows),
             "serving_layer_checks": sum(r["serving_layer_checks"] for r in rows),
             "tflop_checked": round(sum(r["tflop"] for r in rows), 1),
             "simulated_runs": sum(r["simulated"] for r in rows),
             "generated_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()}
    rma = {r["gpu_uuid"]: _rma_zip(r, os.path.join(out_dir, "rma")) for r in failed}
    with open(os.path.join(out_dir, "quarantine.txt"), "w") as f:
        for r in failed:
            f.write(f"{r['gpu_uuid']}\thost={r['host']}\tpci={r['pci_bus_id']}\tserial={r['serial']}"
                    f"{'  (SIMULATED faults: demo)' if r['simulated'] else ''}\n")
    report = {"stats": stats, "gpus": [{k: v for k, v in r.items() if k != "mtime"} for r in rows], "rma": rma}
    with open(os.path.join(out_dir, "fleet_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    e = html.escape
    trs = []
    for r in rows:
        trs.append("<tr>" + "".join(f"<td>{c}</td>" for c in (
            f"<span class='{e(str(r['result']))}'>{e(str(r['result']))}</span>" + (" (simulated)" if r["simulated"] else ""),
            e(str(r["host"] or "")), f"<code>{e(str(r['gpu_uuid'] or ''))}</code>", e(str(r["gpu_name"] or "")),
            e(str(r["pci_bus_id"] or "")), e(str(r["serial"] or "")), f"{r['matmuls_checked']:,}",
            f"{r['serving_layer_checks']:,}", f"{r['tflop']:,}", str(r["corrupted"]),
            e(str(r["worst_clean_score"])), "" if r.get("matmul_tflops") is None else f"{r['matmul_tflops']:,.0f}",
            e("; ".join(r.get("notes") or [])[:160]),
            "signed" if r["signed"] else ("unsigned" if r["certificate"] else "none"))) +
            "</tr>")
    head = ("result", "host", "GPU UUID", "GPU", "PCI", "serial", "matmuls checked", "layer checks (serving)",
            "TFLOP checked", "corrupted results", "worst clean score", "matmul TFLOPS", "notes", "certificate")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU Fleet Report</title>
<style>{_CSS}</style></head><body><main>
<h1>GPU fleet report</h1>
<p class="sub">sdcguard screen results, generated {e(stats['generated_at'])}. PASS: every checked matrix multiply
matched its checksums. FAIL: some didn't; pull the GPU (quarantine.txt) and send the evidence (rma/).</p>
<div class="stats">
<div class="stat"><b>{stats['gpus']}</b><span>GPUs screened</span></div>
<div class="stat"><b class="PASS">{stats['passed']}</b><span>passed</span></div>
<div class="stat"><b class="FAIL">{stats['failed']}</b><span>failed or suspect</span></div>
<div class="stat"><b>{stats['matmuls_checked'] + stats['serving_layer_checks']:,}</b><span>results checked</span></div>
<div class="stat"><b>{stats['tflop_checked']:,}</b><span>TFLOP of work checked</span></div>
</div>
<div class="wrap"><table><thead><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr></thead>
<tbody>{''.join(trs)}</tbody></table></div>
</main></body></html>"""
    with open(os.path.join(out_dir, "fleet_report.html"), "w") as f:
        f.write(page)
    return report
