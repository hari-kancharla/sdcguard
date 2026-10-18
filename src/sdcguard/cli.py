"""The sdcguard command.

    sdcguard check                        is this GPU computing correctly, and at full speed? (5 min, share card)
    sdcguard demo                         a device goes bad mid-training; watch it get caught (CPU is fine)
    sdcguard selftest                     does sdcguard see and check this machine's matmuls?
    sdcguard screen --minutes 10          screen every GPU here with a real workload (evidence on failure)
    sdcguard certify --minutes 10 --key fleet.key   screen, then write a signed certificate per GPU
    sdcguard verify <certificate.json> --pubkey fleet.pub
    sdcguard keygen --out fleet           make a signing key pair (fleet.key, fleet.pub)
    sdcguard report <screen dirs> --out fleet_report     fleet page, quarantine list, RMA evidence zips
    sdcguard replay <evidence.pt>         recompute a failed matmul here and check every result
    sdcguard board --dir results/ --out site/   build the public results board from shared result cards
    sdcguard nhc --minutes 2              node health check: one line, exit 0 OK / 1 problem / 2 unknown
    sdcguard nhc --cached                 the last screen's result, instantly (for Slurm's 60-second limit)
                                          (Kubernetes node-problem-detector, Slurm NHC: see deploy/)
Serving and training are libraries: SDCGUARD=1 vllm serve ..., ServeGuard, SDCGuard (see the README).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
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


def _run_screen(a, capture: bool = False):
    cmd = [sys.executable, "-m", "sdcguard.screen", "--minutes", str(a.minutes), "--out", a.out, "--seed",
           str(a.seed), "--simulate-faults", str(a.simulate_faults)]
    if a.device:
        cmd += ["--device", a.device]
    if a.size:
        cmd += ["--size", a.size]
    return subprocess.run(cmd, capture_output=capture, text=True)


def _fresh_summaries(out: str, since: float) -> list[tuple[str, dict]]:
    found = []
    for p in sorted(glob.glob(os.path.join(out, "*", "summary.json"))):
        if os.path.getmtime(p) >= since - 1:
            try:
                with open(p) as f:
                    found.append((os.path.dirname(p), json.load(f)))
            except (OSError, ValueError):
                pass
    return found


def cmd_certify(a) -> int:
    from . import certify
    t0 = time.time()
    code = _run_screen(a).returncode
    rows = _fresh_summaries(a.out, t0)
    if not rows:
        print("sdcguard certify: no screen results to certify")
        return 2
    for d, s in rows:
        path = certify.write(s, d, key_path=a.key, issuer=a.issuer, valid_days=a.valid_days)
        g = s.get("gpu", {})
        print(f"certificate: {s.get('result')}  {g.get('gpu_uuid') or s.get('device')}  "
              f"{'signed' if a.key else 'UNSIGNED (no --key)'}  {path}")
    return code


def cmd_verify(a) -> int:
    from . import certify
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
        worst = max(worst, 0 if r["ok"] else 1)
    return worst


def cmd_keygen(a) -> int:
    from . import certify
    key, pub = certify.keygen(a.out)
    print(f"private key: {key} (keep it secret)\npublic key:  {pub} (give this to whoever verifies)")
    return 0


def cmd_report(a) -> int:
    from . import fleet
    rows = fleet.collect(a.dirs)
    if not rows:
        print("sdcguard report: no summary.json found under " + ", ".join(a.dirs))
        return 2
    rep = fleet.write(rows, a.out)
    s = rep["stats"]
    print(f"sdcguard report: {s['gpus']} GPUs, {s['passed']} PASS, {s['failed'] - s['suspect']} FAIL, "
          f"{s['suspect']} SUSPECT, "
          f"{s['matmuls_checked'] + s['serving_layer_checks']:,} results checked. "
          f"{os.path.join(a.out, 'fleet_report.html')}")
    if s["failed"]:
        print(f"  pull: {os.path.join(a.out, 'quarantine.txt')}   evidence: {os.path.join(a.out, 'rma')}/")
    return 1 if s["failed"] else 0


def cmd_replay(a) -> int:
    from .screen import replay
    r = replay(a.evidence, a.times, a.device)
    print(json.dumps(r, indent=2, default=str))
    return 1 if r["recomputations_that_failed_the_check"] else 0


def _nhc_status(out: str, line: str, code: int) -> None:
    try:
        os.makedirs(out, exist_ok=True)
        tmp = os.path.join(out, "status.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"time": time.time(), "code": code, "line": line}, f)
        os.replace(tmp, os.path.join(out, "status.json"))
    except OSError:
        pass


def _nhc_cached(a) -> int:
    """Report the last screen's result without running anything (instant): for health checks with a short
    time limit, such as Slurm's HealthCheckProgram (killed after 60 seconds). The screen itself runs on its
    own schedule (deploy/slurm/sdcguard-screen.sbatch, or a timer) and writes status.json."""
    try:
        with open(os.path.join(a.out, "status.json")) as f:
            st = json.load(f)
    except (OSError, ValueError):
        print(f"UNKNOWN: no sdcguard screen result in {a.out} yet")
        return 2
    age_h = (time.time() - float(st.get("time", 0))) / 3600
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
    t0 = time.time()
    try:
        code = _run_screen(a, capture=True).returncode
    except Exception as e:  # noqa: BLE001
        line = f"UNKNOWN: sdcguard could not run ({type(e).__name__})"
        print(line)
        _nhc_status(a.out, line, 2)
        return 2
    rows = _fresh_summaries(a.out, t0)
    bad = [s for _d, s in rows if s.get("result") == "FAIL"]
    if bad:
        g = bad[0].get("gpu", {})
        line = (f"FAIL: {len(bad)} of {len(rows)} GPUs computed wrong results; first {g.get('gpu_uuid')} PCI "
                f"{g.get('pci_bus_id')} serial {g.get('serial')}; evidence in {os.path.abspath(a.out)}")
        rc = 1
    elif code != 0 or not rows:
        line, rc = f"UNKNOWN: sdcguard screen exited {code} with {len(rows)} results", 2
    else:
        checked = sum((s.get("matmuls_checked") or 0) + (s.get("linear_layer_checks_while_serving") or 0)
                      for _d, s in rows)
        line, rc = f"OK: {len(rows)} GPUs computed correctly ({checked:,} results checked)", 0
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
    c = sub.add_parser("certify", help="screen, then write a signed certificate per GPU")
    _screen_args(c, 10.0)
    c.add_argument("--key", default=None, help="private key from `sdcguard keygen` (omit: unsigned)")
    c.add_argument("--issuer", default="", help="who issues the certificate (a name or URL)")
    c.add_argument("--valid-days", type=float, default=30.0)
    v = sub.add_parser("verify", help="check certificates")
    v.add_argument("certificates", nargs="+")
    v.add_argument("--pubkey", default=None, help="the public key you trust")
    v.add_argument("--json", action="store_true")
    k = sub.add_parser("keygen", help="make a signing key pair")
    k.add_argument("--out", default="sdcguard")
    r = sub.add_parser("report", help="fleet report from screen results")
    r.add_argument("dirs", nargs="+")
    r.add_argument("--out", default="fleet_report")
    rp = sub.add_parser("replay", help="recompute a failed matmul from its evidence file")
    rp.add_argument("evidence")
    rp.add_argument("--times", type=int, default=200)
    rp.add_argument("--device", default=None)
    n = sub.add_parser("nhc", help="node health check for Kubernetes / Slurm")
    _screen_args(n, 2.0)
    n.add_argument("--cached", action="store_true", help="report the last screen's result instantly (Slurm)")
    n.add_argument("--max-age-hours", type=float, default=36.0, help="with --cached: older results count as unknown")
    n.set_defaults(out="/tmp/sdcguard_nhc")
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("demo", "selftest", "screen", "check", "board"):
        rest = argv[1:]
        if argv[0] == "check":
            from .check import main as m
            return m(rest)
        if argv[0] == "board":
            from .board import main as m
            return m(rest)
        if argv[0] == "demo":
            from .demo import main as m
            return m(rest)
        if argv[0] == "screen":
            from .screen import main as m
            return m(rest)
        from .selftest import main as m
        return m()
    a = p.parse_args(argv)
    handlers = {"certify": cmd_certify, "verify": cmd_verify, "keygen": cmd_keygen, "report": cmd_report,
                "replay": cmd_replay, "nhc": cmd_nhc}
    if a.cmd not in handlers:
        p.print_help()
        return 0
    return handlers[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
