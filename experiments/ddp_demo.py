"""Multi-GPU demo: one GPU goes bad in the middle of a data-parallel training run.

Two stories, same faulty GPU, same seeds:
  without sdcguard   the run trains normally for a while, then crashes (NaN). The team restarts
                     from the last checkpoint on the same machine, and it crashes again.
  with sdcguard      sdcguard names the bad GPU (UUID, PCI bus, index) before the crash,
                     training stops, and restarts on the remaining healthy GPUs from the last
                     checkpoint saved before corruption could have started (one detector
                     interval before the first alert). It finishes cleanly.

The "bad GPU" is simulated: the process on that GPU corrupts its matmul outputs with NVIDIA's
published fault patterns from --bad-from on (and, with --memory-flip-at, flips a bit in its
copy of the weights). Every fault is logged per attempt in faults_rank*_attempt*.json, with
whether a sdcguard check looked at it and caught it.

Run everything (launches torchrun itself):
  python experiments/ddp_demo.py --gpus 4 --bad-gpu 2
On a machine without GPUs (tests): add --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ============================================================================ worker (one per GPU)
def worker(a):
    from torch.nn.parallel import DistributedDataParallel as DDP

    from harm_study import GPT, PRESETS, autocast_ctx, get_batch, load_dataset, resolve_dtype
    from sdcguard import Canary, FaultInjector, SDCGuard, check_replicas
    from sdcguard.events import device_identity
    from sdcguard.inject import fault_model

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    use_cuda = a.device == "cuda"
    if use_cuda:
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)
    dist.init_process_group("nccl" if use_cuda else "gloo")
    # physical GPU behind this process (what an operator would pull from the machine)
    phys = (os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("SDC_VISIBLE") or "").split(",")
    phys_id = phys[local] if local < len(phys) and phys[local] != "" else str(local)
    is_bad = phys_id == str(a.bad_gpu)

    cfg = dict(PRESETS[a.preset])
    dtype = resolve_dtype(device, torch.bfloat16)
    ds = load_dataset(cfg["dataset"], a.data_dir, cfg.get("ts_train_mb", 200))
    torch.manual_seed(a.seed)
    model = GPT(ds.vocab, cfg["block"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"] + 1
    ddp = DDP(model, device_ids=[local] if use_cuda else None)

    # ---- the simulated bad GPU
    inj = None
    if is_bad:
        targets = [m for b in model.blocks for m in (b.attn.q, b.attn.k)]
        inj = FaultInjector(rate=a.rate, fault_model=fault_model("nvidia", "warp"), seed=a.fault_seed,
                            start_step=a.bad_from).arm_on(targets)

    # ---- sdcguard (only in the "with sdcguard" story)
    guard = None
    if a.mode == "detect":
        if inj is not None:
            # tell the fault log which injected faults a check looked at, and which it caught
            def on_check(_op, token, res):
                if token is not None:
                    inj.resolve(token, res.ok, res.max_z)

            guard = SDCGuard(every=a.guard_every, on_check=on_check,
                             on_check_start=lambda _op, C: inj.match(C)).watch(model)
        else:
            guard = SDCGuard(every=a.guard_every).watch(model)
        if a.dense_steps > 0:
            guard.after_restart(start, a.dense_steps)     # resumed after a crash: check every step for a while
    canary_gen = torch.Generator().manual_seed(1234)
    canary_x, _ = get_batch(ds.val, cfg["block"], 8, canary_gen, device)

    def fwd(m, b):
        with autocast_ctx(device, dtype):
            return m(b)

    canary = Canary(model, forward_fn=fwd) if a.mode == "detect" else None
    me = device_identity(device)
    me["gpu_index"] = phys_id

    gen = torch.Generator().manual_seed(a.seed * 1000 + rank + 17 * start)
    losses, status, alert = [], "finished", None
    t0 = time.time()
    lr_steps = dict(cfg, steps=a.steps)
    from harm_study import lr_at
    for step in range(start, a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, lr_steps)
        x, y = get_batch(ds.train, cfg["block"], a.batch, gen, device)
        if inj is not None:
            inj.step = step
        guarded = guard is not None and guard.should_check(step)
        with (inj if inj is not None else _null()):
            with (guard.at_step(step) if guarded else _null()):
                with autocast_ctx(device, dtype):
                    loss = ddp(x, y)
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if is_bad and a.memory_flip_at is not None and step == a.memory_flip_at:
            with torch.no_grad():                       # the bad GPU also corrupts its weight memory
                w = model.blocks[-1].mlp[0].weight.view(-1)
                if w.element_size() == 4:
                    w.view(torch.int32)[1234] ^= 1 << 20
                else:
                    w.view(torch.int16)[1234] ^= 1 << 4

        # every rank agrees on the loss (NaN anywhere = NaN everywhere: that's the crash)
        lt = loss.detach().float().reshape(1)
        dist.all_reduce(lt)
        lv = float(lt) / world
        losses.append([step, lv])

        found = []
        if guard is not None and guarded and guard.log.events:
            ev = guard.log.events[0]
            found.append({"detector": "checksum", "step": step, "rank": rank, "identity": me,
                          "detail": {k: v for k, v in ev.detail.items() if k in
                                     ("op", "where", "pass", "max_z", "row", "col_block", "nonfinite")}})
        if canary is not None and step % a.canary_every == 0 and step > start:
            with (inj if inj is not None else _null()):
                suspects, layer, _ = canary.check(canary_x, step=step)
            if rank in suspects and len(suspects) < world:
                found.append({"detector": "canary", "step": step, "rank": rank, "identity": me,
                              "detail": {"first_bad_layer": layer, "suspect_ranks": suspects}})
        if a.mode == "detect" and step % a.replica_every == 0 and step > start:
            suspects, _ = check_replicas(model.parameters(), step=step)
            if rank in suspects and len(suspects) < world:
                found.append({"detector": "replica", "step": step, "rank": rank, "identity": me,
                              "detail": {"suspect_ranks": suspects}})
        # share alerts with every rank (cheap: a few bytes, only checked on detector steps)
        flag = torch.tensor([1.0 if found else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if float(flag) > 0:
            gathered = [None] * world
            dist.all_gather_object(gathered, found)
            alerts = [f for fs in gathered for f in fs]
            if rank == 0:
                for f in alerts:
                    who = f["identity"]
                    print(f"\n*** sdcguard ALERT at step {f['step']} ({f['detector']}): rank {f['rank']} = GPU "
                          f"{who.get('gpu_index')} {who.get('gpu_name', '')} {who.get('gpu_uuid') or ''} "
                          f"{who.get('pci_bus_id') or ''} on {who.get('host')}  {f['detail']}", flush=True)
            status, alert = "alert", alerts
            break
        if not math.isfinite(lv):
            status = "crashed"
            if rank == 0:
                print(f"\n*** step {step}: loss is {lv}. Training crashed.", flush=True)
            break
        if rank == 0 and (step % a.log_every == 0 or step == a.steps - 1):
            print(f"  step {step:5d}  loss {lv:.4f}  ({time.time() - t0:.0f}s, {world} GPUs)", flush=True)
        if rank == 0 and step > start and step % a.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step},
                       os.path.join(a.run_dir, f"ckpt_{step:06d}.pt"))
            for old in sorted(glob.glob(os.path.join(a.run_dir, "ckpt_*.pt")))[:-8]:
                os.remove(old)                 # keep the 8 newest (rollback needs at most 2 back)

    val = None
    if status == "finished":
        model.eval()
        with torch.no_grad():
            vg = torch.Generator().manual_seed(999)
            vs = []
            for _ in range(10):
                xv, yv = get_batch(ds.val, cfg["block"], a.batch, vg, device)
                with autocast_ctx(device, dtype):
                    vs.append(float(model(xv, yv)))
            val = sum(vs) / len(vs)
    if rank == 0:
        with open(os.path.join(a.run_dir, "status.json"), "w") as f:
            json.dump({"status": status, "start": start, "last_step": losses[-1][0] if losses else start,
                       "world": world, "losses": losses, "alert": alert, "val_loss": val}, f)
    if inj is not None:
        with open(os.path.join(a.run_dir, f"faults_rank{rank}_attempt{a.attempt}.json"), "w") as f:
            json.dump(inj.events, f, default=str)
    dist.barrier()
    dist.destroy_process_group()


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ============================================================================ launcher
def checkpoints(run_dir):
    return sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(os.path.join(run_dir, "ckpt_*.pt")))


def fault_summary(path):
    """What the bad GPU's fault log says about one attempt."""
    with open(path) as f:
        ev = json.load(f)
    caught = [e for e in ev if e.get("caught")]
    return {"injected": len(ev), "first_step": min((e["step"] for e in ev), default=None),
            "checked": sum(bool(e.get("checked")) for e in ev), "caught": len(caught),
            "first_caught": ({k: caught[0].get(k) for k in ("step", "kind", "row", "col", "rel_change", "z")}
                             if caught else None)}


def torchrun(a, gpus, mode, run_dir, resume=None, attempt=1, dense_steps=0):
    os.makedirs(run_dir, exist_ok=True)
    for old in glob.glob(os.path.join(run_dir, f"faults_rank*_attempt{attempt}.json")):
        os.remove(old)                         # never read a previous run's fault log
    env = dict(os.environ)
    key = "CUDA_VISIBLE_DEVICES" if a.device == "cuda" else "SDC_VISIBLE"
    env[key] = ",".join(str(g) for g in gpus)
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(gpus)}",
           os.path.abspath(__file__), "--worker", "--mode", mode, "--run-dir", run_dir, "--attempt", str(attempt),
           "--dense-steps", str(dense_steps)]
    for k in ("preset", "steps", "batch", "bad_gpu", "bad_from", "rate", "fault_seed", "guard_every",
              "canary_every", "replica_every", "ckpt_every", "log_every", "seed", "device", "data_dir"):
        cmd += [f"--{k.replace('_', '-')}", str(getattr(a, k))]
    if a.memory_flip_at is not None:
        cmd += ["--memory-flip-at", str(a.memory_flip_at)]
    if resume:
        cmd += ["--resume", resume]
    print(f"\n$ {key}={env[key]} torchrun ... --mode {mode}" + (f" --resume {os.path.basename(resume)}" if resume else ""),
          flush=True)
    status_path = os.path.join(run_dir, "status.json")
    if os.path.exists(status_path):
        os.remove(status_path)                 # never read a previous attempt's result
    t = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    if os.path.exists(status_path):
        with open(status_path) as f:
            st = json.load(f)
    else:
        st = {"status": "error", "start": 0, "last_step": 0, "world": len(gpus), "losses": [], "alert": None,
              "val_loss": None}
        print(f"!!! torchrun exited with code {rc} and wrote no status: see the log above", flush=True)
    st["seconds"], st["gpus"], st["returncode"] = round(time.time() - t, 1), list(gpus), rc
    st["dense_steps"] = dense_steps
    logs = sorted(glob.glob(os.path.join(run_dir, f"faults_rank*_attempt{attempt}.json")))
    st["faults"] = fault_summary(logs[0]) if logs else None      # None: the bad GPU wasn't in this run
    return st


def launch(a):
    from sdcguard.events import rollback_checkpoint

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    stories = {}

    # ---- story 1: no sdcguard. Crash, restart from the latest checkpoint on the same GPUs.
    print("=" * 78 + "\nWITHOUT sdcguard\n" + "=" * 78)
    gpus, attempts, resume = list(range(a.gpus)), [], None
    for i in range(a.max_restarts + 1):
        st = torchrun(a, gpus, "baseline", os.path.join(out, "baseline"), resume, attempt=i + 1)
        attempts.append(st)
        if st["status"] != "crashed":
            break
        ck = [c for c in checkpoints(os.path.join(out, "baseline")) if c < st["last_step"]]
        if not ck:
            break
        resume = os.path.join(out, "baseline", f"ckpt_{ck[-1]:06d}.pt")
        print(f"--> crashed at step {st['last_step']}; restarting from checkpoint {ck[-1]} on the same GPUs")
    stories["without_sdcguard"] = attempts

    # ---- story 2: sdcguard names the GPU; restart without it from a checkpoint before the corruption.
    print("\n" + "=" * 78 + "\nWITH sdcguard\n" + "=" * 78)
    gpus, attempts, resume, dense, crash = list(range(a.gpus)), [], None, 0, None
    for i in range(a.max_restarts + 1):
        st = torchrun(a, gpus, "detect", os.path.join(out, "sdcguard"), resume, attempt=i + 1, dense_steps=dense)
        attempts.append(st)
        dense = 0
        if st["status"] == "crashed" and a.triage_steps > 0 and crash is None:
            # A fault crashed training between two scheduled checks. Restart on the same GPUs (we don't
            # know the bad one yet) from the newest checkpoint, checking every step for a while.
            ck = [c for c in checkpoints(os.path.join(out, "sdcguard")) if c < st["last_step"]]
            if not ck:
                break
            crash, dense = st["last_step"], a.triage_steps
            resume = os.path.join(out, "sdcguard", f"ckpt_{ck[-1]:06d}.pt")
            st["triage_restart_from"] = ck[-1]
            print(f"--> crashed at step {crash} between scheduled checks; restarting from checkpoint {ck[-1]} on "
                  f"the same GPUs and checking every step for the next {a.triage_steps} steps")
            continue
        if st["status"] != "alert":
            break
        first = min(x["step"] for x in st["alert"])
        bad = sorted({str(x["identity"]["gpu_index"]) for x in st["alert"]})
        ck = checkpoints(os.path.join(out, "sdcguard"))
        if crash is not None:
            # Named on the restart after a crash: go back to the newest checkpoint saved no later than
            # the last scheduled check that passed before the crash.
            last_pass = ((crash - 1) // a.guard_every) * a.guard_every
            safe = max((c for c in ck if c <= last_pass), default=None)
            st["after_crash_at"] = crash
            crash = None
        else:
            # corruption may have started up to one detector interval before the first alert
            window = max({"checksum": a.guard_every, "canary": a.canary_every,
                          "replica": a.replica_every}[x["detector"]] for x in st["alert"])
            safe = rollback_checkpoint(first, [c for c in ck if c <= first], check_every=window)
        st["evicted"], st["rollback_to"] = bad, safe
        gpus = [g for g in gpus if str(g) not in bad]
        resume = os.path.join(out, "sdcguard", f"ckpt_{safe:06d}.pt") if safe is not None else None
        if safe is not None and safe > a.bad_from:
            st["note"] = (f"rolled back to step {safe}, which is after the GPU went bad at step {a.bad_from}: no "
                          "fault landed on a checked step before then, so that checkpoint may hold corrupted weights")
        print(f"--> sdcguard flagged GPU {', '.join(bad)} at step {first}. Evicting it and restarting on "
              f"GPUs {gpus} from checkpoint {safe if safe is not None else '(start)'}")
    stories["with_sdcguard"] = attempts

    summary = summarize(a, stories)
    with open(os.path.join(out, "demo.json"), "w") as f:
        json.dump({"args": vars(a), "stories": stories, "summary": summary}, f, indent=1)
    with open(os.path.join(out, "STORY.txt"), "w") as f:
        f.write(summary["text"] + "\n")
    plot(a, stories, summary, os.path.join(out, "demo.png"))
    print("\n" + summary["text"])


def summarize(a, stories):
    base, det = stories["without_sdcguard"], stories["with_sdcguard"]
    crashes = [s["last_step"] for s in base if s["status"] == "crashed"]
    alerts = [s for s in det if s["status"] == "alert"]
    first_alert = min(x["step"] for x in alerts[0]["alert"]) if alerts else None
    ident = alerts[0]["alert"][0]["identity"] if alerts else {}
    detectors = sorted({x["detector"] for s in alerts for x in s["alert"]})
    finished = det[-1]["status"] == "finished"
    steps_base = sum(s["last_step"] - s["start"] + 1 for s in base)
    steps_det = sum(s["last_step"] - s["start"] + 1 for s in det)
    lines = [f"Detectors: checksums every {a.guard_every} steps, canary every {a.canary_every}, replica check "
             f"every {a.replica_every}; checkpoints every {a.ckpt_every}; fault pattern seed {a.fault_seed}.",
             f"Simulated bad GPU: GPU {a.bad_gpu} of {a.gpus}, starts corrupting at step {a.bad_from} "
             f"(NVIDIA fault patterns, {a.rate:.0%} of its attention Q/K matmuls)"
             + (f"; flips a bit in its weight memory at step {a.memory_flip_at}" if a.memory_flip_at is not None else "")
             + ".", "",
             "WITHOUT sdcguard:"]
    def num(x, spec):
        return format(x, spec) if isinstance(x, (int, float)) and math.isfinite(x) else str(x)

    def faults_note(s):
        f = s.get("faults")
        if not f or not f["injected"]:
            return ""
        note = (f"\n      fault log: the bad GPU corrupted {f['injected']} matmul outputs from step {f['first_step']}; "
                f"a check looked at {f['checked']}, caught {f['caught']}")
        c = f["first_caught"]
        if c:
            note += (f"; first catch: step {c['step']}, {c['kind']} fault at row {c['row']}, col {c['col']}, "
                     f"{num(c['rel_change'], '.1%')} change, z {num(c['z'], '.1f')}")
        return note

    for s in base:
        lines.append(f"  run on GPUs {s['gpus']} from step {s['start']}: "
                     + (f"crashed (NaN) at step {s['last_step']}" if s["status"] == "crashed"
                        else f"{s['status']} at step {s['last_step']}") + faults_note(s))
    lines += [f"  {f'Gave up after {len(base)} attempts' if base[-1]['status'] != 'finished' else 'Finished'}; "
              f"{steps_base} steps computed, nobody knew which GPU was bad.", "", "WITH sdcguard:"]
    for s in det:
        if s["status"] == "alert":
            lines.append(f"  run on GPUs {s['gpus']} from step {s['start']}"
                         + (f" (every step checked for {s['dense_steps']} steps after the restart)"
                            if s.get("dense_steps") else "")
                         + f": alert at step {min(x['step'] for x in s['alert'])} by "
                         f"{', '.join(sorted({x['detector'] for x in s['alert']}))}"
                         f" -> GPU {', '.join(s['evicted'])} evicted, rolled back to "
                         + (f"step {s['rollback_to']}" if s["rollback_to"] is not None else "the start")
                         + faults_note(s))
            if s.get("note"):
                lines.append(f"  WARNING: {s['note']}.")
        else:
            lines.append(f"  run on GPUs {s['gpus']} from step {s['start']}: "
                         + (f"crashed (NaN) at step {s['last_step']}, between two scheduled checks"
                            if s["status"] == "crashed" else f"{s['status']} at step {s['last_step']}")
                         + (f", final val loss {s['val_loss']:.4f}" if s.get("val_loss") else "")
                         + faults_note(s))
            if s.get("triage_restart_from") is not None:
                lines.append(f"  -> restarted from checkpoint {s['triage_restart_from']} on the same GPUs, checking "
                             f"every step for {a.triage_steps} steps (sdcguard.after_restart)")
    if alerts:
        lines.append(f"  Named the GPU: index {ident.get('gpu_index')}, {ident.get('gpu_name', '')} "
                     f"{ident.get('gpu_uuid') or ''} {ident.get('pci_bus_id') or ''} on host {ident.get('host')}")
    det_crash = next((s["last_step"] for s in det if s["status"] == "crashed"), None)
    if det_crash is not None and first_alert is not None:
        lines.append(f"  The bad GPU crashed the first run at step {det_crash} before a scheduled check saw a fault; "
                     f"on the restart sdcguard named it at step {first_alert} ({first_alert - a.bad_from} steps after "
                     "it went bad), so the crash happened once instead of on every restart.")
    elif crashes and first_alert is not None:
        lines.append(f"  First alert came {crashes[0] - first_alert} steps before the crash it prevented "
                     f"({first_alert - a.bad_from} steps after the GPU went bad).")
    lines.append(f"  {'Finished' if finished else 'Did not finish'}; {steps_det} steps computed.")
    return {"text": "\n".join(lines), "first_alert": first_alert, "crashes": crashes, "detectors": detectors,
            "finished_with_sdcguard": finished, "finished_without": base[-1]["status"] == "finished",
            "identity": ident}


def plot(a, stories, summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, (key, title) in zip(axes, [("without_sdcguard", "Without sdcguard"), ("with_sdcguard", "With sdcguard")]):
        labelled = set()
        for i, s in enumerate(stories[key]):
            xs = [p[0] for p in s["losses"] if math.isfinite(p[1])]
            ys = [p[1] for p in s["losses"] if math.isfinite(p[1])]
            ax.plot(xs, ys, linewidth=1.2, label=f"attempt {i + 1}: GPUs {s['gpus']}")
            if s["status"] == "crashed":
                ax.axvline(s["last_step"], color="black", linestyle=":", linewidth=1.2)
                if "crash" not in labelled:            # restarts crash at nearly the same step: label once
                    top = max(p[1] for st in stories[key] for p in st["losses"] if math.isfinite(p[1]))
                    n = sum(x["status"] == "crashed" for x in stories[key])
                    ax.text(s["last_step"], top, " crash" + (f" ({n} times)" if n > 1 else ""), fontsize=8, va="top")
                    labelled.add("crash")
            if s["status"] == "alert":
                st = min(x["step"] for x in s["alert"])
                ax.axvline(st, color="red", linestyle="--", linewidth=1.2)
                ax.text(st, max(ys) if ys else 0, f" sdcguard: GPU {', '.join(s['evicted'])} is bad",
                        color="red", fontsize=8, va="top")
        ax.axvspan(a.bad_from, a.steps, color="orange", alpha=0.08, label=f"GPU {a.bad_gpu} faulty")
        ax.set_title(title)
        ax.set_ylabel("training loss")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", default="detect", choices=["detect", "baseline"], help=argparse.SUPPRESS)
    p.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--resume", default=None, help=argparse.SUPPRESS)
    p.add_argument("--attempt", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--dense-steps", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--triage-steps", type=int, default=0,
                   help="if training crashes between checks, restart and check every step for this many steps")
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--bad-gpu", type=int, default=2)
    p.add_argument("--bad-from", type=int, default=150, help="step at which the GPU goes bad")
    p.add_argument("--rate", type=float, default=0.05, help="share of the bad GPU's Q/K matmuls it corrupts")
    p.add_argument("--fault-seed", type=int, default=3)
    p.add_argument("--memory-flip-at", type=int, default=None, help="step at which it flips a bit in its weights")
    p.add_argument("--preset", default="ts-gpu")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch", type=int, default=32, help="per GPU")
    p.add_argument("--guard-every", type=int, default=10)
    p.add_argument("--canary-every", type=int, default=50)
    p.add_argument("--replica-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-restarts", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="runs/demo")
    a = p.parse_args()
    if a.worker:
        worker(a)
    else:
        launch(a)


if __name__ == "__main__":
    main()
