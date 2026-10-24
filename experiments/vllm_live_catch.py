"""A real library bug caught live inside `vllm serve`, kept, replayed at its real shape, and worked around
(docs/METHOD.md, "The cuBLAS bug inside the real server", written before the run).

NVIDIA cuBLAS known issue 6862950: on Hopper, FP16/BF16/FP8 GEMMs can give wrong results when the process is an MPS
client below 100% of the SMs (cuBLASLt algorithm 66 with clusters of 4 or more CTAs). sdcguard found it with its
offline check and the gate (October 19 to 22); its vLLM plugin has never met it. This runs the guarded server
(Qwen2.5-1.5B-Instruct, bf16, vLLM's default compiled mode with CUDA graphs) as an MPS client on an H200, with
cuBLASLt's log on (one file per process), in roles one after another:

  ref100   MPS at 100%: the reference completions, and the kernel inventory there
  mps40    MPS at 40%: the same requests; the inventory; the alerts (SDCGUARD_LOG) and their evidence
  pinned40 MPS at 40% with a TunableOp file pinning every GEMM shape the 40% inventory flagged for issue 6862950 to
           algorithm 66 with a 1x1x1 cluster (`sdcguard gate pin`, NVIDIA's workaround), built after mps40; run
           only if mps40 flagged shapes

Each role: 64 prompts one at a time (greedy, 128 tokens), then 64 requests 16 at a time. After mps40, for up to
three alerts with a captured row: a replay at the alert's real shape (its layer's output and input features, the
step's rows with CUDA-graph padding, its bias), with `sdcguard gate capture` in a fresh process at 40% and the same
case again at 100%, both 20 runs, each with the kernel cuBLASLt's log shows; and, beside it, the replay of the kept
row and block alone (`sdcguard gate add` on the evidence file), today's way. After each of ref100 and mps40,
`sdcguard kernels test` on its inventory at its share: every logged linear-layer call run again on its own.
With pins, each replayed case is also run at 40% with the pin file (`gate workaround`), and reproduce/ keeps the
cases, the pin file, the conditions and the commands.

The MPS daemon must already be running (scripts/modal_vllm.py starts it), with CUDA_MPS_PIPE_DIRECTORY set.

    python experiments/vllm_live_catch.py --out runs/live_catch
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_serve_check as vs  # noqa: E402

MODEL = vs.MODEL
REPLAYS = 3


def _env(role: str, out: str, share: int, sink, metrics_port: int, pin: str | None) -> dict:
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_WEBHOOK=f"http://127.0.0.1:{sink.port}/hook",
               SDCGUARD_WEBHOOK_FORMAT="json", SDCGUARD_METRICS_PORT=str(metrics_port),
               SDCGUARD_LOG=os.path.join(out, f"{role}_alerts.jsonl"),
               SDCGUARD_EVIDENCE_DIR=os.path.join(out, f"{role}_evidence"),
               CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(share),
               CUBLASLT_LOG_LEVEL="5", CUBLASLT_LOG_FILE=os.path.join(out, f"{role}_lt", "cublaslt.%i.log"))
    os.makedirs(os.path.join(out, f"{role}_lt"), exist_ok=True)
    if pin:
        env.update(PYTORCH_TUNABLEOP_ENABLED="1", PYTORCH_TUNABLEOP_TUNING="0", PYTORCH_TUNABLEOP_FILENAME=pin)
    return env


def _alerts(path: str) -> list:
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def run_role(role: str, out: str, share: int, port: int, metrics_port: int, pin: str | None = None) -> dict:
    from sdcguard import kernels
    sink = vs.Sink()
    env = _env(role, out, share, sink, metrics_port, pin)
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    res: dict = {"role": role, "mps_share": share, "pinned": pin}
    try:
        res["ready"] = vs.wait_ready(port, proc, timeout=1500)
        if not res["ready"]:
            res["exit_code"] = proc.poll()
            return res
        ps = vs.prompts(256)
        res["identity"] = [vs.complete(port, MODEL, p, stream=False)["text"] for p in ps[:vs.IDENTITY_PROMPTS]]
        res["load"] = vs.load(port, MODEL, ps[64:128], at_once=16)
        time.sleep(5)
        vs.complete(port, MODEL, ps[0], stream=False)        # a step after the wait: the last results are read
        time.sleep(3)
        res["metrics"] = vs.scrape(metrics_port, 1)
    finally:
        res["stop"] = vs.stop(proc)
        log.close()
        sink.server.shutdown()
    res["alerts"] = _alerts(env["SDCGUARD_LOG"])
    logs = sorted(glob.glob(os.path.join(out, f"{role}_lt", "cublaslt.*.log")))
    inv = kernels.inventory(kernels.read_logs(logs), {"role": role, "mps_share": share, "logs": len(logs)})
    with open(os.path.join(out, f"{role}_kernels.json"), "w") as f:
        json.dump(inv, f, indent=1)
    res["inventory"] = {"calls": inv["calls"], "kernels": len(inv["kernels"]),
                        "flagged_6862950": flagged(inv)}
    return res


def flagged(inv: dict) -> list:
    """The GEMM calls the inventory shows on a kernel matching issue 6862950's conditions: shape (out, in, rows),
    the kernel, and whether it had a bias epilogue."""
    out = []
    for k in inv["kernels"]:
        if not any(f["issue"] == "6862950" for f in k["issues"]):
            continue
        for s in k["shapes"]:
            out.append({"out": s["m"], "in": s["k"], "rows": s["n"], "transa": s["transa"],
                        "bias": "BIAS" in (k.get("epilogue") or ""), "types": k["types"], "algo": k["algo"],
                        "calls": s["calls"], "sm_count_target": s.get("sm_count_target")})
    return out


def _gate(args: list, share: int, timeout: int = 900) -> dict:
    env = dict(os.environ, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(share))
    p = subprocess.run([sys.executable, "-m", "sdcguard", "gate", *args], env=env, capture_output=True, text=True,
                       timeout=timeout)
    return {"exit_code": p.returncode, "output": (p.stdout + p.stderr)[-2000:]}


def _read(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def replays(out: str, alerts: list) -> list:
    """For up to REPLAYS alerts with a captured row: the replay at the alert's real shape at 40% and 100%, and the
    replay of the kept slice alone at 40%."""
    done, seen = [], set()
    for ev in alerts:
        d = ev.get("detail") or {}
        mm, rows = d.get("matmul") or {}, (d.get("step_rows") or {}).get("with_padding") or \
            (d.get("step_rows") or {}).get("tokens")
        if not d.get("captured") or not mm or not rows:
            continue
        shape = (mm["out_features"], mm["in_features"], int(rows), bool(mm["bias"]))
        if shape in seen:
            continue
        seen.add(shape)
        name = f"live-{d.get('where')}-{shape[0]}x{shape[1]}x{shape[2]}".replace(".", "_")
        r: dict = {"pack": os.path.join(out, "replay_" + name), "layer": d.get("where"), "shape": {"out": shape[0], "in": shape[1], "rows": shape[2],
                                                       "bias": shape[3]},
                   "directly_implicated": d.get("directly_implicated"), "recheck": (d.get("recheck") or {}).get("verdict")}
        pack = os.path.join(out, "replay_" + name)          # one case per pack: `gate run` runs a whole pack
        cap = ["capture", "--shape", f"{shape[0]},{shape[1]},{shape[2]}", "--dtype", "bfloat16", "--pack", pack,
               "--repeats", "20", "--name", name] + (["--bias"] if shape[3] else [])
        r["capture40"] = _gate(cap, 40)
        r["case"] = _read(os.path.join(pack, name, "case.json"))
        for share in (40, 100):
            path = os.path.join(out, f"{name}_{share}.json")
            r[f"run{share}"] = _gate(["run", pack, "--out", path, "--repeats", "20"], share)
            r[f"result{share}"] = ((_read(path) or {}).get("cases") or {}).get(name)
        ev_file = d.get("evidence")
        if isinstance(ev_file, str) and os.path.exists(ev_file + ".pt"):
            sl = os.path.join(out, "slice_" + name)
            r["slice_add"] = _gate(["add", ev_file + ".pt", "--pack", sl, "--name", name + "-slice"], 40)
            path = os.path.join(out, f"{name}_slice40.json")
            r["slice_run40"] = _gate(["run", sl, "--out", path, "--repeats", "20"], 40)
            r["slice_result40"] = ((_read(path) or {}).get("cases") or {}).get(name + "-slice")
        done.append(r)
        if len(done) >= REPLAYS:
            break
    return done


def pins(out: str, flagged_calls: list) -> dict:
    """A TunableOp file pinning every flagged shape to algorithm 66 with a 1x1x1 cluster (gate capture + gate pin per
    shape, then merged), or why there is none."""
    from sdcguard import gate
    pack, texts, per = os.path.join(out, "pin_pack"), [], []
    shapes = sorted({(c["out"], c["in"], c["rows"], c["bias"]) for c in flagged_calls if c["transa"] == "OP_T"})
    for o, i, n, b in shapes:
        name = f"pin-{o}x{i}x{n}{'-bias' if b else ''}"
        cap = _gate(["capture", "--shape", f"{o},{i},{n}", "--dtype", "bfloat16", "--pack", pack, "--repeats", "3",
                     "--name", name] + (["--bias"] if b else []), 40)
        res = _gate(["pin", os.path.join(pack, name), "--out", os.path.join(out, "pins", name + ".csv")], 40)
        files = sorted(glob.glob(os.path.join(out, "pins", name + "*.csv")))
        per.append({"shape": [o, i, n, b], "capture": cap["exit_code"], "pin": res["exit_code"], "files": files})
        for f in files:
            texts.append(open(f).read())
    if not texts:
        return {"shapes": per, "file": None, "why": "no pin was made"}
    text, rows = gate.merge_pins(texts)
    path = os.path.join(out, "pins", "merged.csv")
    with open(gate.tunableop_file(path, 0), "w") as f:      # PyTorch reads merged<GPU>.csv (GPU 0 here)
        f.write(text)
    return {"shapes": per, "file": path, "rows": rows}


def test_role(out: str, role: str, share: int) -> dict:
    """`sdcguard kernels test` on the role's inventory at the role's MPS share: every logged linear-layer call run
    again on its own (random inputs, 3 runs), judged against float64, with the kernel cuBLASLt picks for it."""
    path = os.path.join(out, f"{role}_kernel_test.json")
    env = dict(os.environ, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(share))
    p = subprocess.run([sys.executable, "-m", "sdcguard.kernels", "test", os.path.join(out, f"{role}_kernels.json"),
                        "--out", path, "--repeats", "3", "--logs", os.path.join(out, f"{role}_kernel_test_lt")],
                       env=env, capture_output=True, text=True, timeout=2400)
    r = _read(path) or {}
    flagged_keys = {c["call"] for c in r.get("calls", []) if any(i.get("issue") == "6862950" for i in c["issues"])}
    return {"exit_code": p.returncode, "output": (p.stdout + p.stderr)[-1500:], "counts": r.get("counts"),
            "flagged_calls": len(flagged_keys),
            "flagged_wrong": sum(1 for c in r.get("calls", []) if c["call"] in flagged_keys and c["verdict"] == "WRONG"),
            "unflagged_wrong": sum(1 for c in r.get("calls", []) if c["call"] not in flagged_keys
                                   and c["verdict"] == "WRONG")}


def workarounds(out: str, reps: list, pin_file: str) -> None:
    """Each replay's case at 40% with the merged pin file (`sdcguard gate workaround`, 20 runs, expecting algorithm
    66 with a 1x1x1 cluster in the log): the fix checked on the very call the server got wrong."""
    env = ["--env", "PYTORCH_TUNABLEOP_ENABLED=1", "--env", "PYTORCH_TUNABLEOP_TUNING=0",
           "--env", f"PYTORCH_TUNABLEOP_FILENAME={pin_file}"]
    for r in reps:
        name = os.path.basename(r["pack"])[len("replay_"):]
        path = os.path.join(out, f"{name}_workaround40.json")
        r["workaround40"] = _gate(["workaround", os.path.join(r["pack"], name), *env, "--repeats", "20",
                                   "--expect-kernel", "algoId=66 cluster=1x1x1", "--json", path], 40)
        r["workaround40_result"] = _read(path)


def bundle(out: str, res: dict) -> str:
    """reproduce/: what someone else needs to see the server's wrong call and its fix without this script: the
    case packs, the pin file, the conditions, and the commands."""
    d = os.path.join(out, "reproduce")
    os.makedirs(d, exist_ok=True)
    reps, pin = res.get("replays") or [], (res.get("pins") or {}).get("file")
    lines = ["#!/bin/sh", "# Needs: a Hopper GPU, the MPS daemon running, CUDA_MPS_PIPE_DIRECTORY set, sdcguard and",
             "# the PyTorch build in conditions.json. Run from this folder.", "set -x"]
    cond = []
    for r in reps:
        name = os.path.basename(r["pack"])[len("replay_"):]
        dst = os.path.join(d, os.path.basename(r["pack"]))
        if os.path.isdir(r["pack"]) and not os.path.exists(dst):
            shutil.copytree(r["pack"], dst)
        cond.append({"case": name, "layer": r.get("layer"), "shape": r["shape"],
                     "conditions": (r.get("case") or {}).get("conditions")})
        for share in (40, 100):
            lines.append(f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={share} sdcguard gate run {os.path.basename(dst)} "
                         f"--out {name}_{share}.json --repeats 20")
        if pin:
            lines.append(f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 sdcguard gate workaround {os.path.basename(dst)}/{name} "
                         "--env PYTORCH_TUNABLEOP_ENABLED=1 --env PYTORCH_TUNABLEOP_TUNING=0 "
                         "--env PYTORCH_TUNABLEOP_FILENAME=$PWD/pins/merged.csv --repeats 20 "
                         "--expect-kernel 'algoId=66 cluster=1x1x1'")
    if pin and os.path.isdir(os.path.dirname(pin)):
        dst = os.path.join(d, "pins")
        if not os.path.exists(dst):
            shutil.copytree(os.path.dirname(pin), dst)
    with open(os.path.join(d, "reproduce.sh"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "conditions.json"), "w") as f:
        json.dump({"model": res.get("model"), "verdict": (res.get("summary") or {}).get("verdict"), "cases": cond},
                  f, indent=1, default=str)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--metrics-port", type=int, default=9400)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res: dict = {"model": MODEL}
    res["ref100"] = run_role("ref100", a.out, 100, a.port, a.metrics_port)
    res["mps40"] = run_role("mps40", a.out, 40, a.port, a.metrics_port)
    for role, share in (("ref100", 100), ("mps40", 40)):
        if res[role].get("ready"):
            res[role]["kernel_test"] = test_role(a.out, role, share)
    res["replays"] = replays(a.out, res["mps40"].get("alerts") or [])
    fl = (res["mps40"].get("inventory") or {}).get("flagged_6862950") or []
    if fl:
        res["pins"] = pins(a.out, fl)
        if res["pins"].get("file"):
            workarounds(a.out, res["replays"], res["pins"]["file"])
            res["pinned40"] = run_role("pinned40", a.out, 40, a.port, a.metrics_port, pin=res["pins"]["file"])
    ok, fails, summary = judge(res)
    res.update(passed=ok, failures=fails, summary=summary)
    if res["replays"]:
        res["reproduce"] = bundle(a.out, res)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"verdict": summary.get("verdict"), "failures": fails, "summary": summary}, indent=1,
                     default=str))
    return {"CAUGHT": 0, "NOT EXERCISED": 3}.get(summary.get("verdict"), 1)


def _wrong_runs(result) -> tuple:
    if not isinstance(result, dict):
        return None, None
    runs = result.get("runs") or []
    bad = sum(1 for r in runs if (r.get("wrong") or 0) > 0 or (r.get("unwritten") or 0) > 0)
    return bad, len(runs)


def _kernels(result) -> dict:
    return ((result or {}).get("kernel_log") or {}).get("kernels") or {}


def _same_kernel(result, flagged_calls: list, shape: dict) -> bool:
    """The replay's logged kernel is the one the server's inventory shows for that shape at 40%."""
    from sdcguard.gate import _algo_cluster
    want = {(c["algo"].get("algoId"), c["algo"].get("clusterShape")) for c in flagged_calls
            if (c["out"], c["in"], c["rows"]) == (shape["out"], shape["in"], shape["rows"])}
    got = {_algo_cluster(t) for t in _kernels(result)}
    return bool(want) and bool(got) and got <= want


def judge(res: dict) -> tuple:
    """(caught, failures, summary): docs/METHOD.md, "The cuBLAS bug inside the real server". summary["verdict"] is
    CAUGHT (criteria 1 to 4 met), NOT EXERCISED (the server never ran the affected kernel at 40%: no catch, and
    none claimed), NOT CAUGHT (it ran, and a criterion failed) or DIDN'T START. Only CAUGHT counts as a catch."""
    ok, fails, s = _judge(res)
    if fails and fails[0].startswith("1:"):
        s["verdict"] = "DIDN'T START"
    elif "finding" in s:
        s["verdict"] = "NOT EXERCISED"
    else:
        s["verdict"] = "CAUGHT" if ok else "NOT CAUGHT"
    return s["verdict"] == "CAUGHT", fails, s


def _judge(res: dict) -> tuple:
    fails, s = [], {}
    ref, m40 = res.get("ref100") or {}, res.get("mps40") or {}
    fl = (m40.get("inventory") or {}).get("flagged_6862950") or []
    s["flagged_calls_40"] = len(fl)
    s["flagged_calls_100"] = len((ref.get("inventory") or {}).get("flagged_6862950") or [])
    if not (ref.get("ready") and m40.get("ready")):
        fails.append("1: a server didn't start")
        return False, fails, s
    s["alerts_100"] = len(ref.get("alerts") or [])
    s["alerts_40"] = len(m40.get("alerts") or [])
    if s["alerts_100"]:
        fails.append(f"2: {s['alerts_100']} alerts at 100%, where the bug doesn't bite")
    if not fl:
        s["finding"] = ("this model's GEMMs did not reach the affected kernel at 40% on this H200: no call in the "
                        "inventory ran algorithm 66 with 4 or more CTAs per cluster")
        return not fails, fails, s
    diff = sum(a != b for a, b in zip(ref.get("identity") or [], m40.get("identity") or []))
    s["completions_differing_40_vs_100"] = diff
    if not s["alerts_40"]:
        fails.append("3: the flagged kernel ran at 40%, and no alert was raised")
    caught = [a for a in (m40.get("alerts") or []) if (a.get("detail") or {}).get("captured")]
    wrong = [a for a in caught if ((a.get("detail") or {}).get("recheck") or {}).get("verdict") == "output_wrong"]
    s["alerts_with_captured_row"] = len(caught)
    s["alerts_rechecked_wrong"] = len(wrong)
    if s["alerts_40"] and not wrong:
        fails.append("3: no alert's captured row was rechecked as wrong in float64")
    reps = res.get("replays") or []
    good = 0
    s["replays"] = []
    for r in reps:
        b40, n40 = _wrong_runs(r.get("result40"))
        b100, n100 = _wrong_runs(r.get("result100"))
        sb, sn = _wrong_runs(r.get("slice_result40"))
        same = _same_kernel(r.get("result40"), fl, r["shape"])
        row = {"shape": r["shape"], "wrong_40": f"{b40}/{n40}", "wrong_100": f"{b100}/{n100}",
               "same_kernel_as_server": same, "slice_wrong_40": None if sb is None else f"{sb}/{sn}"}
        s["replays"].append(row)
        good += bool(b40 is not None and n40 == 20 and b40 >= 19 and b100 == 0 and n100 == 20 and same)
    if not reps or good == 0:
        fails.append("4: no replay at the alert's real shape was wrong in at least 19 of 20 runs at 40%, clean in 20 "
                     "of 20 at 100%, on the server's kernel")
    for role in ("ref100", "mps40"):
        c = (res.get(role) or {}).get("kernel_test")
        if c is not None:
            s[f"kernel_test_{role}"] = {k: c.get(k) for k in ("counts", "flagged_calls", "flagged_wrong", "unflagged_wrong")}
    p40 = res.get("pinned40")
    if p40 is not None:
        pdiff = sum(a != b for a, b in zip(ref.get("identity") or [], p40.get("identity") or []))
        s["workaround40"] = [{"shape": r["shape"], "exit_code": (r.get("workaround40") or {}).get("exit_code"),
                              "verdict": (r.get("workaround40_result") or {}).get("verdict")} for r in reps
                             if "workaround40" in r]
        s["pinned40"] = {"ready": p40.get("ready"), "alerts": len(p40.get("alerts") or []),
                         "completions_differing_vs_100": pdiff,
                         "flagged_calls": len((p40.get("inventory") or {}).get("flagged_6862950") or [])}
    return not fails, fails, s


if __name__ == "__main__":
    sys.exit(main())
