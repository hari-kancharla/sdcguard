"""The sdcguard command.

    sdcguard check                        is this GPU computing correctly, and at full speed? (5 min, share card)
    sdcguard demo                         a device goes bad mid-training; watch it get caught (CPU is fine)
    sdcguard selftest                     does sdcguard see and check this machine's matmuls?
    sdcguard screen --minutes 10          screen every GPU here with a real workload (evidence on failure)
    sdcguard certify --minutes 10 --key fleet.key   check, then write a signed certificate per GPU
    sdcguard verify <certificate.json> --pubkey fleet.pub
    sdcguard keygen --out fleet           make a signing key pair (fleet.key, fleet.pub)
    sdcguard report <screen dirs> --out fleet_report     fleet page, quarantine list, RMA evidence zips
    sdcguard replay <evidence.pt>         recompute a failed matmul here and check every result
    sdcguard casefile <evidence.pt> --replay-here 200   what the evidence and replays support (and don't)
    sdcguard blas                         every cuBLASLt algorithm for some matmul shapes, each checked in float64
    sdcguard issues                       which of NVIDIA's listed cuBLAS wrong-result issues apply to this machine
    sdcguard board --dir results/ --out site/   build the public results board from shared result cards
    sdcguard nhc --minutes 2              node health check (sdcguard check): one line, exit 0 OK / 1 problem / 2 unknown
    sdcguard nhc --cached                 the last check's result, instantly (for Slurm's 60-second limit)
                                          (Kubernetes node-problem-detector, Slurm NHC: see deploy/)
Serving and training are libraries: SDCGUARD=1 vllm serve ..., ServeGuard, SDCGuard (see the README).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import signal
import subprocess
import sys
import time


def _screen_args(p: argparse.ArgumentParser, minutes: float) -> None:
    p.add_argument("--minutes", type=float, default=minutes)
    p.add_argument("--device", default=None, help="e.g. cuda:0 (default: every visible GPU)")
    p.add_argument("--size", default=None, help="tiny, small, 1b or 3b (default: by GPU memory)")
    p.add_argument("--out", default="screen_out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--simulate-faults", type=float, default=0.0, help="demo only: corrupt this share in software")


def _deadline(minutes: float) -> float:
    """How long a check may run before it counts as hung: its budget plus room for loading and the slowest
    phase (a GPU that hangs must not hold the node, or every GPU on it, indefinitely)."""
    return 60.0 * float(minutes) * 1.5 + 600.0


def _stop_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _run_check(a, capture: bool = False, extra: tuple = ()):
    """`sdcguard check` (math modes in every precision, memory, stress, the workload, cross-GPU comparison, the
    driver's health counters) in its own process, with a deadline. nhc and certify are built on it: a screen of
    the bf16 workload alone misses defects in the other precisions (an fp32-only GEMM fault passes it).

    The check runs in a process group of its own, and at the deadline (or when this process is interrupted) the
    whole group is stopped: the per-GPU workers it started too, which would otherwise keep their GPUs (a hung one
    forever). Raises subprocess.TimeoutExpired at the deadline, like subprocess.run."""
    cmd = [sys.executable, "-m", "sdcguard.check", "--minutes", str(a.minutes), "--out", a.out, "--seed",
           str(a.seed), "--simulate-faults", str(a.simulate_faults), "--offline", *extra]
    if a.device:
        cmd += ["--device", a.device]
    if a.size:
        cmd += ["--size", a.size]
    pipe = subprocess.PIPE if capture else None
    p = subprocess.Popen(cmd, stdout=pipe, stderr=pipe, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=_deadline(a.minutes))
    except BaseException:                # the deadline, or an interrupt: no worker outlives the check
        _stop_group(p)
        try:
            # bounded: a process stuck in a driver call (uninterruptible) dies only when the call returns, and a
            # health check must report instead of hanging with it
            p.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        raise
    _stop_group(p)                       # (a worker left behind by a check that crashed)
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def _fresh_checks(out: str, since: float) -> list[tuple[str, dict]]:
    found = []
    for p in sorted(glob.glob(os.path.join(out, "*", "check.json"))):
        if os.path.getmtime(p) >= since - 1:
            try:
                with open(p) as f:
                    c = json.load(f)
                if isinstance(c, dict):
                    found.append((os.path.dirname(p), c))
            except (OSError, ValueError):
                pass
    return found


def cmd_certify(a) -> int:
    """`sdcguard check` on every GPU, then a certificate per GPU (signed with --key)."""
    t0 = time.time()
    extra = ["--certificate", "--valid-days", str(a.valid_days), "--issuer", a.issuer] + (["--key", a.key] if a.key
                                                                                         else [])
    try:
        code = _run_check(a, extra=tuple(extra)).returncode
    except subprocess.TimeoutExpired:
        print(f"sdcguard certify: the check didn't finish within {_deadline(a.minutes):.0f} s (a hung GPU?); "
              "no certificate")
        return 2
    rows = _fresh_checks(a.out, t0)
    certs = [(d, c) for d, c in rows if os.path.exists(os.path.join(d, "certificate.json"))]
    if not certs:
        print("sdcguard certify: no check results to certify")
        return 2
    for d, c in certs:
        g = c.get("gpu", {})
        print(f"certificate: {c.get('result')}  {g.get('gpu_uuid') or c.get('device')}  "
              f"{'signed' if a.key else 'UNSIGNED (no --key)'}  {os.path.join(d, 'certificate.json')}")
    return code


def cmd_verify(a) -> int:
    """Exit 0 when every certificate is authentic (and, with --require-pass, says PASS); 1 otherwise."""
    from . import certify
    if a.require_pass and not a.pubkey:
        print("sdcguard verify: --require-pass needs --pubkey (anyone can sign a certificate with their own key)")
        return 2
    worst = 0
    for path in a.certificates:
        r = certify.verify(path, a.pubkey)
        if a.json:
            print(json.dumps(r, indent=2, default=str))
        else:
            g = r.get("gpu") or {}
            verdict = "VALID" if r["ok"] else "NOT VALID"
            print(f"{verdict}: {r.get('result')} for {g.get('gpu_uuid')} ({g.get('gpu_name')}, serial {g.get('serial')}, "
                  f"host {g.get('host')}), issued {r.get('issued_at')}, valid until {r.get('valid_until')}"
                  + (f", key {r.get('key_id')}" if r.get("key_id") else ""))
            for pr in r["problems"]:
                print(f"  - {pr}")
            if r["ok"] and r["trusted_key"] is None:
                print("  (signature is valid; pass --pubkey to check it was signed by a key you trust)")
        bad = not r["ok"] or (a.require_pass and r.get("result") != "PASS")
        if a.require_pass and r["ok"] and r.get("result") != "PASS" and not a.json:
            print(f"  --require-pass: the certificate is authentic but says {r.get('result')}")
        worst = max(worst, 1 if bad else 0)
    return worst


def cmd_keygen(a) -> int:
    from . import certify
    try:
        key, pub = certify.keygen(a.out)
    except FileExistsError as e:
        print(f"sdcguard keygen: {e}")
        return 2
    print(f"private key: {key} (keep it secret)\npublic key:  {pub} (give this to whoever verifies)")
    return 0


def cmd_report(a) -> int:
    """Exit 1 when a GPU is to be pulled (quarantine.txt isn't empty), 2 when the report couldn't cover
    everything (a folder that doesn't exist, an incomplete result, a simulated-fault demo), else 0."""
    from . import fleet
    if not (math.isfinite(a.window_days) and a.window_days >= 0):
        print(f"sdcguard report: --window-days must be a number of days, 0 or more (got {a.window_days:g})")
        return 2
    missing = [d for d in a.dirs if not os.path.isdir(d)]
    if missing:          # a mistyped or unmounted folder: its GPUs aren't in the report, and that is said
        print("sdcguard report: no such folder (its GPUs are not in this report): " + ", ".join(missing))
    rows = fleet.collect([d for d in a.dirs if d not in missing], window_days=a.window_days, pubkey=a.pubkey)
    if not rows:
        print("sdcguard report: no summary.json found under " + ", ".join(a.dirs))
        return 2
    rep = fleet.write(rows, a.out)
    s = rep["stats"]
    demos = s.get("simulated_failed", 0)
    print(f"sdcguard report: {s['gpus']} GPUs, {s['passed']} PASS, {s['failed'] - s['suspect']} FAIL, "
          f"{s['suspect']} SUSPECT, {s.get('incomplete', 0)} INCOMPLETE, "
          f"{s['matmuls_checked'] + s['serving_layer_checks']:,} results checked. "
          f"{os.path.join(a.out, 'fleet_report.html')}")
    if demos:
        print(f"  {demos} of the failed results are simulated-fault demos: not in quarantine.txt")
    if s["quarantine_candidates"]:
        print(f"  pull: {os.path.join(a.out, 'quarantine.txt')}   evidence: {os.path.join(a.out, 'rma')}/")
        return 1
    return 2 if missing or demos or s.get("incomplete") else 0


def cmd_replay(a) -> int:
    from .screen import replay
    if a.times < 1:
        print("sdcguard replay: --times must be at least 1")
        return 2
    r = replay(a.evidence, a.times, a.device)          # records the software and settings it ran with
    print(json.dumps(r, indent=2, default=str))
    if getattr(a, "save", None):
        with open(a.save, "w") as f:
            json.dump(r, f, indent=2, default=str)
    from .verdict import replay_exit
    return replay_exit(r)


def _nhc_status(out: str, line: str, code: int) -> None:
    try:
        os.makedirs(out, exist_ok=True)
        tmp = os.path.join(out, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"time": time.time(), "code": code, "line": line}, f)
        os.replace(tmp, os.path.join(out, "status.json"))
    except OSError:
        pass


def _nhc_previous(out: str):
    """The last status.json, or None (missing or not a JSON object)."""
    try:
        with open(os.path.join(out, "status.json")) as f:
            st = json.load(f)
        return st if isinstance(st, dict) else None
    except (OSError, ValueError):
        return None


def _nhc_cached(a) -> int:
    """Report the last screen's result without running anything (instant): for health checks with a short
    time limit, such as Slurm's HealthCheckProgram (killed after 60 seconds). The screen itself runs on its
    own schedule (deploy/slurm/sdcguard-screen.sbatch, or a timer) and writes status.json."""
    st = _nhc_previous(a.out)
    if st is None:
        print(f"UNKNOWN: no readable sdcguard screen result in {a.out} yet")
        return 2
    try:
        age_h = (time.time() - float(st.get("time", 0))) / 3600
        if not math.isfinite(age_h) or age_h < 0:
            raise ValueError("invalid or future timestamp")
    except (TypeError, ValueError):
        print(f"UNKNOWN: unreadable sdcguard screen result in {a.out}")
        return 2
    if st.get("code") == 1:                       # a failure stays a failure until someone screens again
        print(st.get("line", "FAIL"))
        return 1
    if age_h > a.max_age_hours:
        print(f"UNKNOWN: last sdcguard screen was {age_h:.0f} hours ago (limit {a.max_age_hours:g})")
        return 2
    print(st.get("line", "OK"))
    return int(st.get("code", 2)) if st.get("code") in (0, 2) else 2


def cmd_nhc(a) -> int:
    """Node health check. One line on stdout; exit 0 = OK, 1 = a GPU computed wrong results,
    2 = couldn't tell (the conventions of Kubernetes node-problem-detector custom plugins)."""
    if a.cached:
        return _nhc_cached(a)
    if a.simulate_faults > 0:
        # A demonstration must never turn into a production drain signal or
        # overwrite a real cached FAIL/PASS. Keep its workload files separate too.
        demo_args = argparse.Namespace(**{**vars(a), "out": os.path.join(a.out, "simulated")})
        try:
            code = _run_check(demo_args, capture=True).returncode
            detail = f"demo screen exited {code}; evidence in {os.path.abspath(demo_args.out)}"
        except Exception as e:  # noqa: BLE001
            detail = f"demo could not run ({type(e).__name__})"
        print(f"SIMULATED: {detail}; not a node health result, cached status unchanged")
        return 2
    t0 = time.time()
    try:
        done = _run_check(a, capture=True)
        code = done.returncode
    except Exception as e:  # noqa: BLE001  (TimeoutExpired: a hung GPU; the process is killed)
        why = (f"didn't finish within {_deadline(a.minutes):.0f} s" if isinstance(e, subprocess.TimeoutExpired)
               else f"could not run ({type(e).__name__})")
        line, rc = f"UNKNOWN: sdcguard {why}", 2
        prev = _nhc_previous(a.out)
        if prev is not None and prev.get("code") == 1:          # (see below: a FAIL stays until a clean check)
            line, rc = f"{prev.get('line', 'FAIL')} (a later check {why})", 1
        print(line)
        _nhc_status(a.out, line, rc)
        return rc
    rows = _fresh_checks(a.out, t0)
    if any(c.get("simulated_faults") for _d, c in rows):
        print("UNKNOWN: simulated check results cannot update the node health status")
        return 2
    bad = [c for _d, c in rows if c.get("result") in ("FAIL", "SUSPECT")]
    partial = [c for _d, c in rows if c.get("result") == "INCOMPLETE"]
    if bad:
        g = bad[0].get("gpu", {})
        what = "computed wrong results" if any(c.get("result") == "FAIL" for c in bad) else \
            "disagree with each other (at least one computed wrong results)"
        line = (f"FAIL: {len(bad)} of {len(rows)} GPUs {what}; first {g.get('gpu_uuid')} PCI "
                f"{g.get('pci_bus_id')} serial {g.get('serial')}; evidence in {os.path.abspath(a.out)}")
        rc = 1
    elif code == 1:
        # `sdcguard check` exits 1 only for wrong results, also when a GPU computed them and its worker then died
        # before its check.json: that drains the node like any other FAIL, whatever the other GPUs' files say
        said = [ln.strip() for ln in (getattr(done, "stdout", None) or "").splitlines() if "FAIL:" in ln]
        line = ("FAIL: sdcguard check exited 1 (a GPU computed wrong results)"
                + (f"; {said[0][:200]}" if said else "") + f"; evidence in {os.path.abspath(a.out)}")
        rc = 1
    elif partial:
        line, rc = (f"UNKNOWN: {len(partial)} of {len(rows)} GPUs couldn't be fully checked; see "
                    f"{os.path.abspath(a.out)}"), 2
    elif code != 0 or not rows:
        line, rc = f"UNKNOWN: sdcguard check exited {code} with {len(rows)} results", 2
    else:
        checked = sum((c.get("workload") or {}).get("matmuls_checked") or 0 for _d, c in rows)
        line, rc = (f"OK: {len(rows)} GPUs computed correctly (math in every precision, memory, stress and "
                    f"{checked:,} workload results checked)"), 0
    prev = _nhc_previous(a.out)
    if rc == 2 and prev is not None and prev.get("code") == 1:
        # a later check that couldn't finish (often the same faulty GPU crashing) doesn't clear a FAIL:
        # only a screen that completes with every GPU correct does
        line, rc = f"{prev.get('line', 'FAIL')} (a later check couldn't finish: {line})", 1
    print(line)
    _nhc_status(a.out, line, rc)
    return rc


def main(argv=None) -> int:
    from . import __version__
    p = argparse.ArgumentParser(prog="sdcguard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"sdcguard {__version__}")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("check", help="is this GPU computing correctly, and at full speed?", add_help=False)
    sub.add_parser("board", help="build the public results board", add_help=False)
    sub.add_parser("demo", help="a device goes bad mid-training; watch it get caught", add_help=False)
    sub.add_parser("selftest", help="does sdcguard see and check this machine's matmuls?", add_help=False)
    sub.add_parser("screen", help="screen GPUs with a real workload", add_help=False)
    sub.add_parser("blas", help="run every cuBLASLt kernel for your matmul shapes and say which compute wrong",
                   add_help=False)
    sub.add_parser("casefile", help="what saved evidence of a failure supports (experimental)", add_help=False)
    sub.add_parser("nvsentinel", help="send check results and job alerts to NVIDIA NVSentinel (Kubernetes)",
                   add_help=False)
    c = sub.add_parser("certify", help="check every GPU, then write a signed certificate per GPU")
    _screen_args(c, 10.0)
    c.add_argument("--key", default=None, help="private key from `sdcguard keygen` (omit: unsigned)")
    c.add_argument("--issuer", default="", help="who issues the certificate (a name or URL)")
    c.add_argument("--valid-days", type=float, default=30.0)
    v = sub.add_parser("verify", help="check certificates")
    v.add_argument("certificates", nargs="+")
    v.add_argument("--pubkey", default=None, help="the public key you trust")
    v.add_argument("--json", action="store_true")
    v.add_argument("--require-pass", action="store_true",
                   help="also exit 1 unless every certificate says PASS (needs --pubkey); for scripts that gate on it")
    k = sub.add_parser("keygen", help="make a signing key pair")
    k.add_argument("--out", default="sdcguard")
    r = sub.add_parser("report", help="fleet report from screen results")
    r.add_argument("dirs", nargs="+")
    r.add_argument("--out", default="fleet_report")
    r.add_argument("--pubkey", default=None,
                   help="the public key certificates must be signed with to count (from sdcguard keygen)")
    r.add_argument("--window-days", type=float, default=30.0,
                   help="a GPU that failed within this many days of its newest result stays SUSPECT (default 30)")
    rp = sub.add_parser("replay", help="recompute a failed matmul from its evidence file")
    rp.add_argument("evidence")
    rp.add_argument("--times", type=int, default=200)
    rp.add_argument("--device", default=None)
    rp.add_argument("--save", default=None, help="also write the result (with this machine's software) to a file, "
                    "for sdcguard casefile --replays")
    n = sub.add_parser("nhc", help="node health check for Kubernetes / Slurm")
    _screen_args(n, 2.0)
    n.add_argument("--cached", action="store_true", help="report the last screen's result instantly (Slurm)")
    n.add_argument("--max-age-hours", type=float, default=36.0, help="with --cached: older results count as unknown")
    n.set_defaults(out="/tmp/sdcguard_nhc")
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in _DISPATCHED:
        cmd, run = argv[0], lambda: _dispatch(argv[0], argv[1:])
    else:
        a = p.parse_args(argv)
        handlers = {"certify": cmd_certify, "verify": cmd_verify, "keygen": cmd_keygen, "report": cmd_report,
                    "replay": cmd_replay, "nhc": cmd_nhc}
        if a.cmd not in handlers:
            p.print_help()
            return 0
        cmd, run = a.cmd, lambda: handlers[a.cmd](a)
    try:
        return run()
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001  (exit 1 means wrong results: an error here is "couldn't tell", 2)
        print(f"sdcguard {cmd}: error: {type(e).__name__}: {e}")
        return 2


_DISPATCHED = ("demo", "selftest", "screen", "check", "board", "casefile", "blas", "issues", "nvsentinel")


def _dispatch(cmd: str, rest: list) -> int:
    """The subcommands with parsers of their own."""
    if cmd == "issues":
        from .known_issues import main as m
        try:                       # (known_issues.environment() reads the GPU only once CUDA is in use)
            import torch
            if torch.cuda.is_available():
                torch.cuda.init()
        except Exception:  # noqa: BLE001  (no usable GPU: the cuBLAS version is still reported)
            pass
    elif cmd == "blas":
        from .blas import main as m
    elif cmd == "casefile":
        from .casefile import main as m
    elif cmd == "nvsentinel":
        from .nvsentinel import main as m
    elif cmd == "check":
        from .check import main as m
    elif cmd == "board":
        from .board import main as m
    elif cmd == "demo":
        from .demo import main as m
    elif cmd == "screen":
        from .screen import main as m
    else:
        from .selftest import main as m
        return m()
    return m(rest)


if __name__ == "__main__":
    sys.exit(main())
