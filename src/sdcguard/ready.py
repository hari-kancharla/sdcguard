"""sdcguard ready: is this vLLM server's math being checked right now? Asked from outside the server.

The plugin's own SDCGUARD_REQUIRE refuses to serve when it can't check, but it can only do that once it runs: a
plugin that fails to import is logged and skipped by vLLM's loader, and the server serves unchecked. This command
doesn't trust the plugin to report its own absence. It reads every GPU worker's sdcguard metrics
(SDCGUARD_METRICS_PORT, one port per worker) twice, a few seconds apart, and says PROTECTED only if, on every
worker it was told to expect:

  - the metrics are there (no answer: the plugin didn't load, the port isn't set, or the exporter failed);
  - linear layers are checked, none whose check never ran, the same number on every worker (and as many as
    --layers, a share of the weights of at least --min-weights, when given);
  - the alarm line is at most --max-z (default 6: a server set to alert at z=1000 never alerts);
  - checks have run and their results have been read back at least once;
  - each worker is on its own GPU (with --gpus, exactly the GPUs given: the answer is then bound to this
    server's workers, not to whatever process exports metrics on those ports), and none restarted between the
    two reads;
  - every worker's executed checks went up between the two reads. If vLLM's own metrics (--vllm, the API
    server's /metrics) show it produced tokens and a worker's checks didn't go up, that worker is serving
    unchecked. If they show no tokens in between, the server is idle and the answer is IDLE: everything else
    holds, but nothing shows the checks would run for the next request (stopped checks look the same on an idle
    server); --probe sends two short requests and reads again, which turns IDLE into PROTECTED or NOT PROTECTED.
    Without --vllm, checks that didn't go up can mean an idle server or stopped checks, so the answer is
    INCONCLUSIVE: pass --vllm (and --probe) for a definite one.

Exit codes: 0 PROTECTED or IDLE (a readiness probe on an idle server must pass, or no traffic ever reaches it;
use --probe to have IDLE checked), 1 NOT PROTECTED, 2 INCONCLUSIVE (couldn't tell: no checks read back yet, a worker
restarted between the reads, checks that didn't go up with no vLLM metrics to say the server was idle). For a
Kubernetes readiness probe any nonzero code keeps traffic away, so the probe's timeoutSeconds must be longer than
--window. Only the Python standard library is used, so the check runs in a small sidecar as well as in the
server's own image. PROTECTED means the configured checks are running and their results are being read, on every
expected worker; it doesn't mean every answer is right (see what the checks cover, and SDCGUARD_SHARE: a share
checks the same subset of layers every step).

    sdcguard ready --port 9400 --workers 2 --vllm http://127.0.0.1:8000/metrics --probe http://127.0.0.1:8000
    sdcguard ready --url http://10.0.0.5:9400/metrics --url http://10.0.0.5:9401/metrics --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
INFO = "sdcguard_protection_info"
VLLM_TOKENS = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")


def parse(text: str) -> tuple:
    """(values, info): every series' value summed over its labels (series named *_created are left out), and
    the labels of sdcguard_protection_info (the last one seen)."""
    values: dict = {}
    info: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        if name.endswith("_created"):
            continue
        try:
            v = float(value)
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + v
        if name == INFO and labels:
            info = dict(_LABEL.findall(labels))
    return values, info


def scrape(url: str, timeout: float = 2.0) -> tuple:
    """(values, info, error): error is None when the page was read."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {}, {}, f"{type(e).__name__}: {getattr(e, 'reason', e)}"
    values, info = parse(text)
    return values, info, None


def _num(text) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def judge(first: list, second: list, vllm_first: Optional[dict], vllm_second: Optional[dict], *,
          max_z: float = 6.0, layers: Optional[int] = None, min_weights: Optional[float] = None,
          gpus: Optional[list] = None) -> dict:
    """first, second: per worker (url, values, info, error) from the two reads; vllm_*: vLLM's values or None.
    Returns {"verdict", "problems", "unsure", "notes", "workers"}."""
    problems, unsure, notes, workers = [], [], [], []
    idle = False
    for (url, a, ia, ea), (_u, b, ib, eb) in zip(first, second):
        w = {"url": url, "gpu": ia.get("UUID") or None, "pid": ia.get("pid") or None,
             "layers_checked": a.get("sdcguard_layers_checked"),
             "checks_executed": b.get("sdcguard_checks_executed_total", a.get("sdcguard_checks_executed_total"))}
        workers.append(w)
        for err in (ea, eb):
            if err:
                problems.append(f"{url}: no answer ({err}): the plugin didn't load, SDCGUARD_METRICS_PORT isn't "
                                "set for this worker, or its exporter failed")
                break
        else:
            if "sdcguard_layers_checked" not in a or "sdcguard_layers_checked" not in b:
                problems.append(f"{url}: answers, but has no sdcguard metrics")
                continue
            if not ia:
                problems.append(f"{url}: no {INFO} (sdcguard older than this check, or not the vLLM plugin)")
                continue
            z = _num(ia.get("z"))
            if z is None or not z <= max_z:
                problems.append(f"{url}: alarm line z={ia.get('z')}, above --max-z {max_z:g}")
            n = a.get("sdcguard_layers_checked", 0.0)
            if not n > 0:
                problems.append(f"{url}: no linear layer is checked")
            elif layers is not None and n != layers:
                problems.append(f"{url}: {n:g} layers checked, expected {layers}")
            if a.get("sdcguard_layers_never_checked", 0.0) > 0 or b.get("sdcguard_layers_never_checked", 0.0) > 0:
                problems.append(f"{url}: {max(a.get('sdcguard_layers_never_checked', 0.0), b.get('sdcguard_layers_never_checked', 0.0)):g} "
                                "registered layers whose check never ran")
            ws = a.get("sdcguard_weights_checked_ratio")
            if min_weights is not None and not (ws is not None and ws >= min_weights):
                problems.append(f"{url}: share of weights checked {ws}, below --min-weights {min_weights:g}")
            if ia.get("pid") != ib.get("pid") or ia.get("UUID") != ib.get("UUID"):
                unsure.append(f"{url}: the worker changed between the two reads (pid {ia.get('pid')} -> "
                              f"{ib.get('pid')}): restarted")
            if not (b.get("sdcguard_reads_total", 0.0) > 0 and b.get("sdcguard_checks_executed_total", 0.0) > 0):
                unsure.append(f"{url}: no check results read back yet (still warming up?)")
    reported = [w["gpu"] for w in workers if w["gpu"]]
    if len(reported) != len(set(reported)):
        twice = sorted({g for g in reported if reported.count(g) > 1})
        problems.append(f"two workers report the same GPU ({', '.join(twice)}): "
                        "a worker is missing or answering twice")
    if gpus is not None:
        seen = set(reported)
        if seen != set(gpus):
            problems.append(f"the workers report GPUs {sorted(seen)}, not the server's {sorted(set(gpus))}: these "
                            "metrics may come from another process")
    counts = {w["layers_checked"] for w in workers if w["layers_checked"] is not None}
    if len(counts) > 1:
        problems.append(f"workers check different numbers of layers: {sorted(counts)}")
    grew = [b.get("sdcguard_checks_executed_total", 0.0) > a.get("sdcguard_checks_executed_total", 0.0)
            for (_u, a, _i, _e), (_v, b, _j, _f) in zip(first, second)]
    if vllm_first is not None and vllm_second is not None:
        t1 = sum(vllm_first.get(k, 0.0) for k in VLLM_TOKENS)
        t2 = sum(vllm_second.get(k, 0.0) for k in VLLM_TOKENS)
        if not any(k in vllm_second for k in VLLM_TOKENS):
            unsure.append("vLLM's metrics have no token counters (" + ", ".join(VLLM_TOKENS) + ")")
        elif t2 > t1:
            for w, g in zip(workers, grew):
                if not g:
                    problems.append(f"{w['url']}: vLLM produced {t2 - t1:g} tokens between the reads but this "
                                    "worker's executed checks didn't go up: it is serving unchecked")
        else:
            idle = True
            notes.append("idle: vLLM produced no tokens between the reads, so the checks weren't shown running now "
                         "(--probe sends a short request to show it)")
    elif not all(grew):
        unsure.append("executed checks didn't go up between the reads on every worker: the server is idle, or its "
                      "checks stopped, and without --vllm these can't be told apart")
    verdict = "NOT PROTECTED" if problems else ("INCONCLUSIVE" if unsure else ("IDLE" if idle else "PROTECTED"))
    return {"verdict": verdict, "problems": problems, "unsure": unsure, "notes": notes, "workers": workers}


def probe(base: str, model: Optional[str] = None, timeout: float = 60.0) -> Optional[str]:
    """Two short greedy completions 2.5 s apart (the plugin reads its results back at the first step after 2 s
    idle), so the checks have to run, and be read, for this request. None when they were answered, else why not."""
    import http.client
    from urllib.parse import urlparse
    u = urlparse(base if "://" in base else "http://" + base)
    try:
        if model is None:
            with urllib.request.urlopen(f"{u.scheme}://{u.netloc}/v1/models", timeout=timeout) as r:
                model = json.loads(r.read())["data"][0]["id"]
        for i in range(2):
            c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
            try:
                c.request("POST", "/v1/completions", headers={"Content-Type": "application/json"},
                          body=json.dumps({"model": model, "prompt": "sdcguard readiness probe", "max_tokens": 8,
                                           "temperature": 0.0, "ignore_eos": True}))
                r = c.getresponse()
                r.read()
                if r.status != 200:
                    return f"the probe request got HTTP {r.status}"
            finally:
                c.close()
            if i == 0:
                time.sleep(2.5)
    except (OSError, ValueError, KeyError, IndexError) as e:
        return f"the probe request failed: {type(e).__name__}: {e}"
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sdcguard ready", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, help="SDCGUARD_METRICS_PORT of the server (its workers use port, port+1, ...)")
    p.add_argument("--workers", type=int, default=1, help="GPU workers to expect (tensor x pipeline parallel size)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--url", action="append", default=[], help="a worker's metrics URL (instead of --port; repeat)")
    p.add_argument("--vllm", help="vLLM's own /metrics URL, to tell an idle server from one whose checks stopped")
    p.add_argument("--window", type=float, default=5.0, help="seconds between the two reads (default 5)")
    p.add_argument("--max-z", type=float, default=6.0, help="highest alarm line accepted (default 6)")
    p.add_argument("--layers", type=int, help="layers each worker must check")
    p.add_argument("--min-weights", type=float, help="lowest share of the model's linear-layer weights checked")
    p.add_argument("--gpus", help="the GPUs this server must run on (UUIDs, comma-separated, as nvidia-smi -L "
                   "prints them): binds the answer to this server's workers, not any process exporting metrics")
    p.add_argument("--probe", metavar="URL", help="vLLM's API base (http://host:8000): when the server is idle, send "
                   "two short requests and read again, so the checks are shown running now (PROTECTED) or not "
                   "(NOT PROTECTED) instead of IDLE")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    urls = list(a.url) or ([f"http://{a.host}:{a.port + i}/metrics" for i in range(a.workers)]
                           if a.port is not None else [])
    if not urls:
        p.error("give --port (with --workers) or --url")
    if not a.window >= 0:
        p.error("--window must be 0 or more")
    first = [(u, *scrape(u)) for u in urls]
    v1 = scrape(a.vllm) if a.vllm else None
    time.sleep(a.window)
    second = [(u, *scrape(u)) for u in urls]
    v2 = scrape(a.vllm) if a.vllm else None
    vllm_first = vllm_second = None
    if a.vllm:
        if v1[2] or v2[2]:
            print(f"sdcguard ready: vLLM metrics at {a.vllm} not read ({v1[2] or v2[2]})", file=sys.stderr)
        else:
            vllm_first, vllm_second = v1[0], v2[0]
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()] if a.gpus else None
    r = judge(first, second, vllm_first, vllm_second, max_z=a.max_z, layers=a.layers, min_weights=a.min_weights,
              gpus=gpus)
    if a.vllm and vllm_first is None:
        r["unsure"].append(f"vLLM's metrics at {a.vllm} couldn't be read")
        if r["verdict"] in ("PROTECTED", "IDLE"):
            r["verdict"] = "INCONCLUSIVE"
    if r["verdict"] == "IDLE" and a.probe:
        why = probe(a.probe)
        if why:
            r["unsure"].append(why)
            r["verdict"] = "INCONCLUSIVE"
        else:
            time.sleep(1.0)
            third = [(u, *scrape(u)) for u in urls]
            v3 = scrape(a.vllm)
            if v3[2]:
                r["unsure"].append(f"vLLM's metrics at {a.vllm} couldn't be read after the probe")
                r["verdict"] = "INCONCLUSIVE"
            else:
                r2 = judge(second, third, vllm_second, v3[0], max_z=a.max_z, layers=a.layers,
                           min_weights=a.min_weights, gpus=gpus)
                r2["notes"].insert(0, "probed: two short requests were sent because the server was idle")
                if r2["verdict"] == "IDLE":   # vLLM counted no tokens for the probe: it can't show anything
                    r2["verdict"] = "INCONCLUSIVE"
                    r2["unsure"].append("vLLM counted no tokens for the probe requests")
                r = r2
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"{r['verdict']}: {len(urls)} worker{'s' if len(urls) != 1 else ''}")
        for line in r["problems"]:
            print(f"  problem: {line}")
        for line in r["unsure"]:
            print(f"  unsure: {line}")
        for line in r["notes"]:
            print(f"  note: {line}")
    return {"PROTECTED": 0, "IDLE": 0, "NOT PROTECTED": 1}.get(r["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
