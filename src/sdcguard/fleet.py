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
import shutil
import stat
import time
import zipfile
from typing import Iterable, Optional


_SEVERITY = {"FAIL": 3, "SUSPECT": 2, "INCOMPLETE": 1, "PASS": 0}


def _worse(a, b):
    """The more serious of two results (an unknown one counts as INCOMPLETE)."""
    sa, sb = _SEVERITY.get(a, 1), _SEVERITY.get(b, 1)
    return a if sa >= sb else b


def _num(v) -> float:
    """A count from a file that may have been edited by hand: a number, or 0."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf") else 0


def _load(path: str):
    try:
        with open(path) as f:
            v = json.load(f)
        return v if isinstance(v, dict) else None
    except (OSError, ValueError):
        return None


def _certificate_status(path: str, pubkey: Optional[str]) -> dict:
    """Whether a certificate can be believed: its signature verifies, and it was made with the key you trust."""
    try:
        from .certify import verify
        v = verify(path, pubkey)
    except RuntimeError as e:                 # the cryptography package isn't installed
        return {"signature_valid": False, "trusted": False, "why": f"it couldn't be checked ({e})"}
    except Exception as e:  # noqa: BLE001  (unreadable or malformed)
        return {"signature_valid": False, "trusted": False, "why": f"it couldn't be read ({type(e).__name__})"}
    valid = bool(v.get("signature_valid"))
    if not v.get("signed"):
        why = "it isn't signed"
    elif not valid:
        why = "its signature doesn't verify"
    elif pubkey is None:
        why = "no trusted public key was given (sdcguard report --pubkey)"
    elif not v.get("trusted_key"):
        why = "it was signed by another key than the one you trust"
    else:
        why = None
    return {"signature_valid": valid, "trusted": why is None, "why": why}


_IDENTITY = ("gpu_uuid", "gpu_name", "serial", "pci_bus_id", "vbios", "driver", "host")


def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _text(v):
    """An identity field from a file that may have been edited by hand: None, or text (never a list or an object,
    which can't name a GPU in quarantine.txt or key a row)."""
    return v if v is None or isinstance(v, str) else str(v)


def _time_of(v) -> Optional[float]:
    """A time a result file records (ISO 8601; UTC when it has no zone), in seconds, or None."""
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        t = _dt.datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return (t if t.tzinfo else t.replace(tzinfo=_dt.timezone.utc)).timestamp()


def _iso(t: float) -> str:
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _summary_is_newer(s: dict, chk: dict, d: str) -> bool:
    """Whether summary.json is from a later run than check.json: by the times they record when both do (a copy
    changes file times), else by file times. A tie keeps check.json, the finished result."""
    ts, tc = _time_of(s.get("time") or s.get("started")), _time_of(chk.get("time"))
    if ts is None or tc is None:
        ts, tc = os.path.getmtime(os.path.join(d, "summary.json")), os.path.getmtime(os.path.join(d, "check.json"))
    return ts > tc


def _row(d: str, files: list, pubkey: Optional[str], now: float) -> Optional[dict]:
    """One folder's result as a report row (None when it holds no result)."""
    s = _load(os.path.join(d, "summary.json")) if "summary.json" in files else None
    chk = _load(os.path.join(d, "check.json")) if "check.json" in files else None
    if s is None and chk is None:
        return None
    s = s or {}
    early = []
    unfinished = False
    if chk and chk.get("run_id") and s.get("run_id") != chk["run_id"]:
        if s and _summary_is_newer(s, chk, d):
            # every run reuses the GPU's folder: a run that wrote its workload summary and was then killed (a
            # deadline, the OOM killer, a hung driver call) left the previous run's check.json, which doesn't
            # speak for it. What it found counts, and it didn't finish.
            unfinished = True
            early.append("the newest run here wrote summary.json but no check.json of its own (its check didn't "
                         "finish, or it was a bare `sdcguard screen`); the check.json here is from an earlier run")
            chk = None
        else:
            s = {}                      # a workload summary left by an older run is not this run's evidence
    run_id = (chk or s).get("run_id")
    cert = _load(os.path.join(d, "certificate.json")) if "certificate.json" in files else None
    cstat = {"signature_valid": False, "trusted": False}
    if cert:
        cstat = _certificate_status(os.path.join(d, "certificate.json"), pubkey)
        if cert.get("run_id") and cert["run_id"] != run_id:
            # a certificate vouches for the run it was issued for, never for another run's files in its folder
            cstat = {**cstat, "trusted": False, "why": "it was issued for another run than the files here"}
    g = {**_dict((chk or {}).get("gpu")), **_dict(s.get("gpu"))} if chk else _dict(s.get("gpu"))
    g = {k: _text(g.get(k)) for k in _IDENTITY}
    if cert and cstat["trusted"]:
        # the verdict comes from the signed certificate, so the GPU it names does too: the unsigned files could
        # otherwise move a signed FAIL onto another GPU in quarantine.txt
        signed = {k: _text(_dict(cert.get("gpu")).get(k)) for k in _IDENTITY}
        moved = [k for k in _IDENTITY if g[k] is not None and g[k] != signed[k]]
        if moved:
            early.append("the files here name another GPU than the certificate (" + ", ".join(moved) +
                         "): the certificate's is used")
        g = signed
    # `sdcguard check` writes check.json with every phase (workload, stress check, math modes); a bare
    # `sdcguard screen` has only the workload summary
    wrong = (chk or {}).get("wrong") or {"workload_training": s.get("corrupted_results_training") or 0,
                                         "workload_serving": s.get("corrupted_results_serving") or 0}
    wrong = {str(k): _num(v) for k, v in dict(wrong).items()}
    evidence = [_local(e, d) for e in ((chk or {}).get("evidence") or s.get("evidence") or [])]
    saved = 0
    if chk and chk.get("running") and chk.get("run_id"):
        # a check still running, or killed before it reported: the failures it saved on the way count
        from .check import saved_failures
        counts, paths = saved_failures(d, chk["run_id"])
        saved = sum(counts.values())
        wrong = {k: max(wrong.get(k, 0), counts.get(k, 0)) for k in {**wrong, **counts}}
        evidence += [e for e in paths if e not in evidence]
        if saved:
            early.append(f"this run stopped before it reported, after saving {saved} failed result(s)")
    wl = _dict((chk or {}).get("workload"))
    raw_notes = (chk or {}).get("notes") or []
    notes = early + [str(n[1]) for n in (raw_notes if isinstance(raw_notes, list) else [])
                     if isinstance(n, (list, tuple)) and len(n) == 2 and n[0] in ("bad", "warn")]
    if chk is None and s.get("error"):
        notes.insert(len(early), f"the screen stopped with an error ({s['error']})")
    result = (chk or {}).get("result") or s.get("result") or "INCOMPLETE"
    if (chk and s.get("result") == "FAIL") or saved:
        result = "FAIL"
    if not isinstance(result, str) or result not in _SEVERITY:
        notes.insert(0, f"unknown result {str(result)[:40]!r}")
        result = "INCOMPLETE"
    if unfinished:
        result = _worse(result, "INCOMPLETE")
    changed = []
    if cert:
        from .certify import artifact_problems
        artifact_errors = artifact_problems(cert, d)
        changed = [name for name, why in artifact_errors.items() if why.startswith("changed")]
        if cstat["trusted"]:
            result = _worse(result, cert.get("result") or "INCOMPLETE")
            if artifact_errors:
                notes[:0] = [f"{name} {why}" for name, why in artifact_errors.items()]
                result = _worse(result, "INCOMPLETE")
        else:
            disagrees = (cert.get("result") or "INCOMPLETE") != result or bool(artifact_errors)
            notes.insert(0, f"certificate not counted: {cstat['why']}"
                         + ("; it disagrees with the files here, so this GPU needs a look" if disagrees
                            else ""))
            if disagrees:
                notes[1:1] = [f"{name} {why}" for name, why in artifact_errors.items()]
                result = _worse(result, "INCOMPLETE")
    mtime = os.path.getmtime(os.path.join(d, "check.json" if chk else "summary.json"))
    # when it ran: the time the run recorded (for a trusted certificate, in a file whose hash it signs), not the
    # file's time, which a copy (cp -r, scp, an object-store download) sets to now, and not the certificate's
    # issued_at, which is when `sdcguard certify` ran (maybe days later); never later than now (a host clock
    # running ahead)
    t = _time_of(chk.get("time")) if chk else _time_of(s.get("time") or s.get("started"))
    if t is None and cert and cstat["trusted"]:
        t = _time_of(cert.get("issued_at"))
    t = min(mtime if t is None else t, now)
    tf = _dict((chk or {}).get("speed")).get("tflops")
    hw = (chk or {}).get("hardware_errors") or []
    return {"dir": os.path.abspath(d), "gpu_uuid": g["gpu_uuid"], "gpu_name": g["gpu_name"],
            "host": g["host"], "pci_bus_id": g["pci_bus_id"], "serial": g["serial"],
            "driver": g["driver"], "result": result,
            "minutes": (chk or {}).get("minutes") or s.get("minutes"),
            "matmuls_checked": _num(s.get("matmuls_checked") or wl.get("matmuls_checked")),
            "serving_layer_checks": _num(s.get("linear_layer_checks_while_serving")
                                         or wl.get("linear_layer_checks_while_serving")),
            "tflop": _num(s.get("approx_matmul_tflop") or wl.get("approx_matmul_tflop")),
            "corrupted": sum(v for k, v in wrong.items() if k != "memory_words"),
            "wrong_by_phase": wrong, "hardware_errors": [str(x) for x in hw] if isinstance(hw, list) else [str(hw)],
            "worst_clean_score": (chk or {}).get("worst_clean_score", s.get("worst_clean_z")),
            "evidence": evidence,
            "simulated": bool(s.get("SIMULATED_FAULTS") or (chk or {}).get("simulated_faults")),
            "certificate": bool(cert), "signed": bool(cert and cstat["signature_valid"]),
            "certificate_trusted": bool(cert and cstat["trusted"]),
            "changed_since_certificate": changed, "issued_at": (cert or {}).get("issued_at"),
            "mtime": mtime, "_t": t, "time": _iso(t), "matmul_tflops": None if tf is None or _num(tf) != tf else tf,
            "repeatable": (chk or {}).get("repeatable"), "notes": notes}


def _unreadable(d: str, files: list, e: Exception, now: float) -> dict:
    """The row for a folder whose result files couldn't be read (hand-edited, truncated): never a PASS, and never
    a reason to drop the rest of the fleet's report. A FAIL it states is kept."""
    found = [x for x in (_load(os.path.join(d, n)) for n in ("check.json", "summary.json") if n in files) if x]
    result = "INCOMPLETE"
    for x in found:
        r = x.get("result")
        result = _worse(result, r if isinstance(r, str) and r in _SEVERITY else "INCOMPLETE")
    g = next((x["gpu"] for x in found if isinstance(x.get("gpu"), dict)), {})
    uuid = g.get("gpu_uuid") if isinstance(g.get("gpu_uuid"), str) else None
    name = "check.json" if "check.json" in files else "summary.json"
    mtime = os.path.getmtime(os.path.join(d, name)) if os.path.exists(os.path.join(d, name)) else now
    t = min(mtime, now)
    return {"dir": os.path.abspath(d), "gpu_uuid": uuid, "gpu_name": None, "host": None, "pci_bus_id": None,
            "serial": None, "driver": None, "result": result, "minutes": None, "matmuls_checked": 0,
            "serving_layer_checks": 0, "tflop": 0, "corrupted": 0, "wrong_by_phase": {}, "hardware_errors": [],
            "worst_clean_score": None, "evidence": [],
            "simulated": any(bool(x.get("SIMULATED_FAULTS") or x.get("simulated_faults")) for x in found),
            "certificate": "certificate.json" in files, "signed": False, "certificate_trusted": False,
            "changed_since_certificate": [], "issued_at": None, "mtime": mtime, "_t": t, "time": _iso(t),
            "matmul_tflops": None, "repeatable": None,
            "notes": [f"its result files couldn't be read ({type(e).__name__}: {str(e)[:80]}): this GPU needs a look"]}


def _rank(row: dict) -> tuple:
    """Newest first, and of two results recorded at the same time, the worse one."""
    return row["_t"], _SEVERITY.get(row["result"], 1)


def collect(roots: Iterable[str], window_days: float = 30.0, pubkey: Optional[str] = None) -> list[dict]:
    """One row per screened GPU: its newest real result. A folder counts when it has a summary.json (`sdcguard
    screen`) or a check.json (`sdcguard check`, which also covers a check that stopped before its workload ran).
    "Newest" goes by the time each run recorded, not file times (a copy changes those), with a time later than now
    taken as now; on a tie the worse result is the row. A simulated-fault demo is the row only for a GPU with no
    real result. When a certificate is there, the row's result is the more serious of the
    certificate's and the files', and files changed since the certificate was issued are flagged, but only for a
    certificate signed with the key you trust (`pubkey`) for the run whose files are there; the row then names the
    GPU the certificate names. Anyone who can write to a results folder can write a
    certificate, so one that can't be verified never sets the verdict: when it disagrees with the files, or the
    files changed since it, the row is INCOMPLETE (to look at), never FAIL (to quarantine). A folder whose files
    can't be read is an INCOMPLETE row of its own (or FAIL, when it says so), never the end of the report.

    A GPU that failed within `window_days` of its newest result stays out of service: a later PASS doesn't
    clear it (faulty GPUs are often intermittent), so its row is SUSPECT, with the earlier failure in its notes
    and every result in `history`."""
    if pubkey is not None:
        from .certify import load_trusted
        load_trusted(pubkey)            # a key that can't be used is an error, not a reason to trust no certificate
    now = time.time()
    found: dict = {}
    for root in roots:
        for d, _dirs, files in os.walk(root):
            if "summary.json" not in files and "check.json" not in files:
                continue
            try:
                row = _row(d, files, pubkey, now)
            except Exception as e:  # noqa: BLE001  (one hand-edited file must not stop the whole fleet's report)
                row = _unreadable(d, files, e, now)
            if row is not None:
                found.setdefault(row["gpu_uuid"] or row["dir"], []).append(row)
    rows = []
    for runs in found.values():
        real = [r for r in runs if not r["simulated"]]
        row = max(real or runs, key=_rank)
        past = sorted(runs, key=_rank)
        row["history"] = [{"result": old["result"], "dir": old["dir"],
                           "time": time.strftime("%Y-%m-%d %H:%M", time.gmtime(old["_t"]))}
                          for old in past]
        later_demo = [old for old in runs if old["simulated"] and old["_t"] > row["_t"]] if real else []
        if later_demo:
            row["notes"].append("a simulated-fault demo ran on this GPU later; a demo never changes its result")
        # a simulated-fault demo stays in the history but never keeps a real GPU in quarantine
        bad = [old for old in past if old is not row and old["result"] in ("FAIL", "SUSPECT") and not old["simulated"]
               and old["_t"] <= row["_t"] and row["_t"] - old["_t"] <= window_days * 86400]
        if bad and row["result"] in ("PASS", "INCOMPLETE"):
            for old in bad:                           # the failure's evidence goes in the RMA bundle
                row["evidence"] += [e for e in old["evidence"] if e not in row["evidence"]]
            when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(bad[-1]["_t"]))
            row["notes"].insert(0, f"{bad[-1]['result']} on {when}, {len(bad)} failed result(s) in the last "
                                   f"{window_days:g} days: a later PASS doesn't clear an intermittent GPU")
            row["result"] = "SUSPECT"
        rows.append(row)
    return sorted(rows, key=lambda r: (r["result"] != "FAIL", r["host"] or "", r["pci_bus_id"] or ""))


def _local(e: str, d: str) -> str:
    """Evidence belongs to the result directory being reviewed, including moved/copied bundles.

    Never follow an old absolute path into a different run or another person's files.
    """
    return os.path.join(d, os.path.basename(e))


_PHASES = {"workload_training": "in the training workload", "workload_serving": "in the serving workload",
           "stress": "in the stress check", "math_modes": "in the math-mode check",
           "memory_words": "memory words read back wrong in the memory test"}


def _zip_name(row: dict, taken: set) -> str:
    """A file name for a GPU's RMA zip: its UUID (or its folder's name) made safe for a path, unique."""
    from .screen import _safe_name
    base = _safe_name(str(row.get("gpu_uuid") or os.path.basename(row["dir"]) or "gpu"))
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    taken.add(name)
    return name + ".zip"


def _own_file(p: str) -> bool:
    """A regular file in its own result folder: never a symlink, and never one that resolves outside that folder.
    Anyone who can write to a results folder could otherwise put any file the person running the report can read
    (a private key, say) into the zip that gets sent away."""
    if os.path.islink(p) or not os.path.isfile(p):
        return False
    return os.path.dirname(os.path.realpath(p)) == os.path.realpath(os.path.dirname(p))


def _add(z: zipfile.ZipFile, p: str, arcname: str) -> None:
    """Copy one checked file into the zip, opened without following a symlink (one swapped in after the check)."""
    try:
        fd = os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            return
        info = zipfile.ZipInfo(arcname, max(time.localtime(st.st_mtime)[:6], (1980, 1, 1, 0, 0, 0)))
        info.external_attr = (st.st_mode & 0xFFFF) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        with z.open(info, "w") as out:
            shutil.copyfileobj(f, out, 1 << 20)


def _rma_zip(row: dict, out_dir: str, name: str = "") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name or _zip_name(row, set()))
    # each evidence file's name in the zip: an earlier run's evidence_0 next to this run's gets its run folder's
    # name, and a number when that clashes too (runs kept as <date>/sdcguard_check/<UUID> share that folder)
    stored, used = [], set()
    for e in row["evidence"]:
        exts = [ext for ext in (".json", ".pt") if _own_file(e + ext)]
        if not exts:
            continue
        base = os.path.basename(e)
        if base in used:
            base = f"{os.path.basename(os.path.dirname(os.path.dirname(e))) or 'earlier'}_{base}"
        stem, i = base, 2
        while stem in used:
            stem, i = f"{base}_{i}", i + 1
        used.add(stem)
        stored.append((e, stem, exts))
    phases = "".join(f"    {v:,} {_PHASES.get(k, k)}\n" for k, v in (row.get("wrong_by_phase") or {}).items() if v)
    readme = (f"sdcguard evidence for GPU {row['gpu_uuid']} ({row['gpu_name']}), serial {row['serial']}, "
              f"host {row['host']}, PCI {row['pci_bus_id']}\n\n"
              f"Result: {row['result']}. {row['corrupted']:,} matrix multiply results failed their checks "
              f"({row['matmuls_checked']:,} workload matmuls and {row['serving_layer_checks']:,} layer outputs "
              f"checked, plus the stress and math-mode checks):\n" + (phases or "    none\n") + "\n" +
              "".join(f"Reported by the driver: {t}\n" for t in row.get("hardware_errors") or []) +
              ("\n" if row.get("hardware_errors") else "") +
              "Each .pt file holds the inputs and the output this GPU produced for one failed matmul, and the .json "
              "next to it describes it; check.json is the whole result. To show the failure follows the board, run "
              "on this GPU and on a healthy one:\n\n" +
              ("".join(f"    sdcguard replay {stem}.pt\n" for _e, stem, exts in stored if ".pt" in exts) or
               "    (no evidence files: the failure wasn't in a saved matmul)\n") +
              "\nA healthy GPU recomputes the right answer every time (and the recorded output fails the check); "
              "a faulty one tends to repeat the error. A replay that fails again points at the GPU but doesn't by "
              "itself rule out a software or configuration cause.\n")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", readme)
        for result_file in ("summary.json", "check.json", "result.json", "certificate.json"):
            p = os.path.join(row["dir"], result_file)
            if _own_file(p):
                _add(z, p, result_file)
        for e, stem, exts in stored:
            for ext in exts:
                _add(z, e + ext, stem + ext)
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
    candidates = [r for r in failed if not r["simulated"]]
    stats = {"gpus": len(rows), "passed": sum(r["result"] == "PASS" for r in rows), "failed": len(failed),
             "suspect": sum(r["result"] == "SUSPECT" for r in rows),
             "incomplete": sum(r["result"] == "INCOMPLETE" for r in rows),
             "matmuls_checked": sum(r["matmuls_checked"] for r in rows),
             "serving_layer_checks": sum(r["serving_layer_checks"] for r in rows),
             "tflop_checked": round(sum(r["tflop"] for r in rows), 1),
             "simulated_runs": sum(r["simulated"] for r in rows),
             "simulated_failed": len(failed) - len(candidates),
             "quarantine_candidates": len(candidates),
             "generated_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()}
    taken: set = set()
    rma = {r["gpu_uuid"] or r["dir"]: _rma_zip(r, os.path.join(out_dir, "rma"), _zip_name(r, taken)) for r in failed}
    with open(os.path.join(out_dir, "quarantine.txt"), "w") as f:
        for r in candidates:
            f.write(f"{r['gpu_uuid']}\thost={r['host']}\tpci={r['pci_bus_id']}\tserial={r['serial']}"
                    "\n")
    report = {"stats": stats, "gpus": [{k: v for k, v in r.items() if k not in ("mtime", "_t")} for r in rows],
              "rma": rma}
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
            e(str(r["worst_clean_score"])),
            "" if r.get("matmul_tflops") is None else f"{_num(r['matmul_tflops']):,.0f}",
            e("; ".join(r.get("notes") or [])[:160]),
            ("signed with your key" if r.get("certificate_trusted") else
             "signature valid, key not trusted" if r["signed"] else
             "not verified" if r["certificate"] else "none"))) +
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
