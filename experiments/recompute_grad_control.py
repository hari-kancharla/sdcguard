"""Follow-up to criterion 3 of the recompute check (docs/METHOD.md, "Follow-up: gradients with a control run"),
written after the October 20 GPU run found gradients not bit-identical with and without the check, and before
this ran. That run had no control, so it couldn't tell a change made by the check from the GPU's own run-to-run
differences.

Per model (the same GPT-2 and Llama configurations as experiments/recompute_check.py) and per setting, one
warm-up backward pass that is discarded, then four backward passes of one training step with the same weights,
data and seeds, in this order:

  A  without the check
  B  without the check again (the control)
  C  with the check
  D  without the check, after it was uninstalled (did it leave anything behind?)

Settings, each in its own process:

  default        PyTorch's defaults, as in the first run
  deterministic  torch.use_deterministic_algorithms(True, warn_only=True), CUBLAS_WORKSPACE_CONFIG=:4096:8 set
                 before CUDA starts, cuDNN deterministic, attention on SDPA's math backend; warnings about
                 nondeterministic operations that still ran are recorded

Decision per model, from the deterministic setting only:

  C didn't compare every block, or mismatched   INCONCLUSIVE: the check didn't run as intended
  A and B differ                                INCONCLUSIVE: the setting isn't deterministic here
  A = B, and C and D both equal A               UNCHANGED: the check left the gradients bit-identical
  A = B, and C or D differs from A              CHANGED: the check changed the gradients (a bug)

Criterion 3 (follow-up) is met when both models are UNCHANGED on the GPU.

The default setting is reported, not judged: whether two plain steps there match bit for bit, and how far apart
they and the checked step are.

    python experiments/recompute_grad_control.py --device cuda
    python experiments/recompute_grad_control.py --device cpu
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import torch  # noqa: E402

import recompute_check as rc_run  # noqa: E402
from sdcguard.recompute import RecomputeCheck  # noqa: E402

SETTINGS = ("default", "deterministic")
MODELS = ("gpt2", "llama")


def _attention_context(setting: str):
    if setting != "deterministic":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def _grads(m, dev, sz, setting: str, check: bool) -> dict:
    torch.manual_seed(7)
    ids = torch.randint(0, sz["vocab"], (sz["batch"], sz["seq"]), device=dev)
    rc = RecomputeCheck().install(m) if check else None
    stats = None
    try:
        torch.manual_seed(8)
        with _attention_context(setting), torch.autocast(dev.type, dtype=torch.bfloat16,
                                                         enabled=dev.type == "cuda"):
            loss = m(input_ids=ids, labels=ids).loss
            loss.backward()
        if rc:
            rc.flush()
            stats = {k: rc.stats[k] for k in ("recomputes_compared", "mismatches")}
    finally:
        if rc:
            rc.uninstall()
    g = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    return {"grads": g, "loss": float(loss.detach()), "check_stats": stats}


def compare(a: dict, b: dict) -> dict:
    """Bit-level and size comparison of two gradient sets: how many parameters differ in any bit, and the
    largest difference relative to that parameter's largest gradient."""
    ga, gb = a["grads"], b["grads"]
    same_names = sorted(ga) == sorted(gb)
    names = sorted(set(ga) & set(gb))
    differ, worst = [], []
    for n in names:
        x, y = ga[n], gb[n]
        if x.shape != y.shape or not torch.equal(x, y):
            differ.append(n)
            if x.shape == y.shape:
                scale = float(x.float().abs().max()) or 1.0
                worst.append((float((x.float() - y.float()).abs().max()) / scale, n))
    worst.sort(reverse=True)
    return {"bit_identical": same_names and not differ, "parameters": len(names),
            "parameters_that_differ": len(differ), "same_parameter_names": same_names,
            "max_relative_difference": worst[0][0] if worst else 0.0,
            "worst_parameters": {n: v for v, n in worst[:5]},
            "losses": [a["loss"], b["loss"]], "losses_equal": a["loss"] == b["loss"]}


def decide(det: dict, layers: int) -> str:
    st = det.get("check_stats") or {}
    if st.get("recomputes_compared") != layers or st.get("mismatches") != 0:
        return "INCONCLUSIVE"          # the check didn't compare every block cleanly, so C tested nothing
    if not det["A vs B"]["bit_identical"]:
        return "INCONCLUSIVE"
    if det["A vs C"]["bit_identical"] and det["A vs D"]["bit_identical"]:
        return "UNCHANGED"
    return "CHANGED"


def one_setting(setting: str, dev, sz) -> dict:
    """Runs in the process the setting needs (see run_all)."""
    caught = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if setting == "deterministic":
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        out = {"setting": setting, "models": {}}
        try:
            for kind in MODELS:
                m = rc_run.make(kind, dev, sz)
                _grads(m, dev, sz, setting, False)              # warm-up, discarded (lazy initialization)
                runs = {"A": _grads(m, dev, sz, setting, False), "B": _grads(m, dev, sz, setting, False),
                        "C": _grads(m, dev, sz, setting, True), "D": _grads(m, dev, sz, setting, False)}
                res = {"attention": getattr(m.config, "_attn_implementation", None),
                       "check_stats": runs["C"]["check_stats"],
                       "A vs B": compare(runs["A"], runs["B"]), "A vs C": compare(runs["A"], runs["C"]),
                       "A vs D": compare(runs["A"], runs["D"]), "B vs C": compare(runs["B"], runs["C"])}
                if setting == "deterministic":
                    res["decision"] = decide(res, sz["layers"])
                out["models"][kind] = res
                del m, runs
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            torch.use_deterministic_algorithms(False)
        caught = sorted({str(x.message)[:240] for x in w if "nondetermin" in str(x.message).lower()})
    out["nondeterminism_warnings"] = caught[:10]
    if dev.type == "cuda":
        out["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return out


def _child(setting: str, device: str, quick: bool) -> dict:
    env = dict(os.environ)
    if setting == "deterministic":
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # before CUDA starts, so only in a fresh process
    else:
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    cmd = [sys.executable, os.path.abspath(__file__), "--device", device, "--setting-only", setting]
    if quick:
        cmd.append("--quick")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"setting": setting, "error": f"exit {p.returncode}", "output": (p.stdout + p.stderr)[-3000:]}


def criteria(r: dict) -> dict:
    det = (r.get("settings") or {}).get("deterministic") or {}
    models = det.get("models") or {}
    decisions = {k: v.get("decision") for k, v in models.items()}
    complete = sorted(decisions) == sorted(MODELS) and "error" not in det
    return {"decisions": decisions,
            "3 (follow-up) gradients unchanged by the check, against a deterministic control":
                complete and all(d == "UNCHANGED" for d in decisions.values()),
            "complete": complete}


def run_all(dev, quick=False, say=print) -> dict:
    sz = dict(rc_run.SIZES["cuda" if dev.type == "cuda" else "cpu"])
    if quick:
        sz.update(layers=2)
    r = {"sizes": sz, "settings": {}}
    if dev.type == "cuda":
        r.update(gpu=torch.cuda.get_device_name(dev), torch=torch.__version__)
    for s in SETTINGS:
        r["settings"][s] = _child(s, dev.type, quick)
        res = r["settings"][s]
        if "error" in res:
            say(f"{s}: could not run: {res['error']}")
            continue
        for kind, m in res["models"].items():
            say(f"{s} {kind}: A vs B identical {m['A vs B']['bit_identical']} "
                f"({m['A vs B']['parameters_that_differ']} of {m['A vs B']['parameters']} differ, max "
                f"{m['A vs B']['max_relative_difference']:.2e}); A vs C identical {m['A vs C']['bit_identical']} "
                f"({m['A vs C']['parameters_that_differ']} differ, max {m['A vs C']['max_relative_difference']:.2e});"
                f" A vs D identical {m['A vs D']['bit_identical']}"
                + (f"; decision {m['decision']}" if "decision" in m else ""))
        if res.get("nondeterminism_warnings"):
            say(f"{s}: nondeterminism warnings: {res['nondeterminism_warnings']}")
    r["criteria"] = criteria(r)
    return r


def markdown(r: dict) -> str:
    lines = [f"## recompute gradients with a control, {r.get('gpu', 'cpu')}", ""]
    for s, res in (r.get("settings") or {}).items():
        if "error" in res:
            lines.append(f"- {s}: could not run ({res['error']})")
            continue
        for kind, m in res["models"].items():
            parts = [f"{k} {'identical' if m[k]['bit_identical'] else 'differ'} "
                     f"({m[k]['parameters_that_differ']} of {m[k]['parameters']}, max relative "
                     f"{m[k]['max_relative_difference']:.2e})" for k in ("A vs B", "A vs C", "A vs D")]
            lines.append(f"- {s}, {kind}: " + "; ".join(parts) + (f"; **{m['decision']}**" if "decision" in m else ""))
        if res.get("nondeterminism_warnings"):
            lines.append(f"- {s}: nondeterministic operations still warned: {len(res['nondeterminism_warnings'])}")
    c = r.get("criteria", {})
    for k, v in c.items():
        if k != "decisions":
            lines.append(f"- {k}: {'met' if v else 'not met'}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--setting-only", choices=SETTINGS, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    if a.setting_only:
        sz = dict(rc_run.SIZES["cuda" if dev.type == "cuda" else "cpu"])
        if a.quick:
            sz.update(layers=2)
        try:
            res = one_setting(a.setting_only, dev, sz)
        except Exception as e:  # noqa: BLE001
            import traceback
            res = {"setting": a.setting_only, "error": f"{type(e).__name__}: {str(e)[:300]}",
                   "trace": traceback.format_exc()[-1500:]}

        print(json.dumps(res, default=str))
        return 0
    r = run_all(dev, a.quick)
    print(markdown(r))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"grad_control_{a.device}.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
    return 0 if r["criteria"]["3 (follow-up) gradients unchanged by the check, against a deterministic control"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
