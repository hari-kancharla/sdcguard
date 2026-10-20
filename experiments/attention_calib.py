"""Does the attention check (attention.py) stay quiet on clean training, and what does it catch?

Trains small GPT models whose attention runs in PyTorch's fused scaled_dot_product_attention, with every step
checked by SDCGuard(attention=True), on each fused kernel the GPU offers (flash, memory-efficient, cuDNN; on
the CPU its flash kernel). Records the attention backward score of every clean call, and every forward
alarm. Then, without the optimizer step, injects one error per call below the guard (so the guard sees the
corrupted tensor, as it would see a faulty kernel's output): a value added to one random dV entry, in
multiples of dV's typical size, and one random output multiplied by a power of 2 (an exponent-bit change).

Criteria, written before the first GPU run (Oct 19, 2026): on every GPU and kernel, 0 alarms on clean steps
and a worst clean backward score of at most 4 (a 1.5x margin under the alarm line at 6). The detection rates
are reported with no pass line. If the criteria fail, attention checking stays off by default.

    python experiments/attention_calib.py                 # CPU, a quick run
    python experiments/attention_calib.py --device cuda   # every fused kernel this GPU has
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sdcguard import SDCGuard  # noqa: E402
from sdcguard import attention as att  # noqa: E402

ALARM = 6.0
CONFIGS = {
    "cuda": [dict(name="GPT, 8 layers, width 1024, 16 heads, 2,048 tokens", layers=8, width=1024, heads=16, kv=16,
                  seq=2048, batch=4, vocab=32000, steps=12, seed=1),
             dict(name="GQA 16:4, head size 128, 4,096 tokens", layers=4, width=2048, heads=16, kv=4, seq=4096,
                  batch=2, vocab=32000, steps=12, seed=2),
             dict(name="narrow, 32 heads of 32, 1,024 tokens", layers=6, width=1024, heads=32, kv=32, seq=1024,
                  batch=8, vocab=8192, steps=12, seed=3)],
    "cpu": [dict(name="tiny, GQA 4:2", layers=2, width=128, heads=4, kv=2, seq=128, batch=2, vocab=512, steps=3,
                 seed=1)],
}
MULTS = (1, 4, 16, 64, 256)        # injected dV error, in units of dV's root-mean-square value
POW2 = (2, 8, 64, 2**16)           # injected forward error: one output times this


class Block(nn.Module):
    def __init__(self, d, h, kv):
        super().__init__()
        self.h, self.kv, self.hd = h, kv, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False)
        self.kv_proj = nn.Linear(d, 2 * kv * self.hd, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        y = self.ln1(x)
        q = self.q(y).view(B, T, self.h, self.hd).transpose(1, 2)
        k, v = self.kv_proj(y).view(B, T, 2, self.kv, self.hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.h != self.kv)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, C))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.tok, self.pos = nn.Embedding(c["vocab"], c["width"]), nn.Embedding(c["seq"], c["width"])
        self.blocks = nn.ModuleList([Block(c["width"], c["heads"], c["kv"]) for _ in range(c["layers"])])
        self.ln, self.head = nn.LayerNorm(c["width"]), nn.Linear(c["width"], c["vocab"], bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def _tokens(vocab, shape, gen):
    w = 1.0 / torch.arange(1, vocab + 1, dtype=torch.float64)          # word-like (Zipf) frequencies
    return torch.multinomial(w, int(math.prod(shape)), replacement=True, generator=gen).view(*shape)


class Inject(TorchDispatchMode):
    """Below the guard: corrupts one value of each attention call's output (see the module docstring)."""

    def __init__(self, gen, mult=None, pow2=None):
        super().__init__()
        self.gen, self.mult, self.pow2, self.n = gen, mult, pow2, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func._schema.name
        if self.mult is not None and name in att.BACKWARD_OPS and att.why_not(name, args, kwargs or {}) is None:
            dv = out[2]
            rms = dv.float().pow(2).mean().sqrt()
            i = tuple(int(torch.randint(0, s, (1,), generator=self.gen)) for s in dv.shape)
            sign = 1.0 if torch.rand(1, generator=self.gen).item() < 0.5 else -1.0
            dv[i] = (dv[i].float() + sign * self.mult * rms).to(dv.dtype)
            self.n += 1
        if self.pow2 is not None and name in att.FORWARD_OPS and att.why_not(name, args, kwargs or {}) is None:
            o = out[0]
            i = tuple(int(torch.randint(0, s, (1,), generator=self.gen)) for s in o.shape)
            o[i] = (o[i].float() * self.pow2).to(o.dtype)
            self.n += 1
        return out


def backends(dev):
    if dev.type != "cuda":
        return [("cpu flash", None)]
    from torch.nn.attention import SDPBackend
    return [("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("cudnn", SDPBackend.CUDNN_ATTENTION)]


def _ctx(backend):
    if backend is None:
        import contextlib
        return contextlib.nullcontext()
    from torch.nn.attention import sdpa_kernel
    return sdpa_kernel([backend])


def run(dev, cfg, backend, low, say=print) -> dict:
    torch.manual_seed(cfg["seed"])
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(cfg["seed"] + 100)
    scores, fwd_alarms, ops = [], 0, {}
    forward_checks, unjudged = 0, 0

    def on_result(detail, res):
        nonlocal fwd_alarms, forward_checks, unjudged
        kind = detail.get("kind")
        if kind in ("attention_backward", "attention_forward"):
            if not res.judged:
                unjudged += 1
            elif kind == "attention_backward":
                scores.append(res.max_z)
            else:
                forward_checks += 1
                fwd_alarms += not res.ok
        if kind:
            ops[detail["op"]] = ops.get(detail["op"], 0) + 1

    g = SDCGuard(attention=True, on_result=on_result)

    def step(idx, inject=None):
        with _ctx(backend):
            if inject is not None:
                with inject, g:
                    with torch.autocast(dev.type, dtype=low):
                        logits = model(idx[:, :-1])
                        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
                    loss.backward()
            else:
                with g:
                    with torch.autocast(dev.type, dtype=low):
                        logits = model(idx[:, :-1])
                        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
                    loss.backward()
        return float(loss.detach())

    for n in range(cfg["steps"]):
        loss = step(_tokens(cfg["vocab"], (cfg["batch"], cfg["seq"] + 1), gen).to(dev))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if n % 4 == 3 or n == cfg["steps"] - 1:
            say(f"step {n}: loss {loss:.3f}, attention checks {len(scores)}, worst {max(scores, default=0):.2f}, "
                f"forward alarms {fwd_alarms}")
    clean = {"checks": len(scores), "alarms": sum(z > ALARM for z in scores) + fwd_alarms,
             "backward_alarms": sum(z > ALARM for z in scores), "forward_alarms": fwd_alarms,
             "worst": max(scores, default=0.0), "ops": dict(ops),
             "forward_checks": forward_checks, "unjudged": unjudged,
             "skipped": dict(g.attention_skipped)}
    # injected errors (no optimizer step): the share of corrupted calls that alarm
    caught = {}
    for label, kw in [(f"dV + {m} x rms", dict(mult=m)) for m in MULTS] + \
                     [(f"output x {p}", dict(pow2=p)) for p in POW2]:
        before = len(g.log.events)
        inj = Inject(torch.Generator().manual_seed(7), **kw)
        trials = 0
        for _ in range(2):
            n0 = inj.n
            step(_tokens(cfg["vocab"], (cfg["batch"], cfg["seq"] + 1), gen).to(dev), inject=inj)
            opt.zero_grad(set_to_none=True)
            trials += inj.n - n0
        caught[label] = {"corrupted_calls": trials,
                         "alarms": sum(str(e.detail.get("kind", "")).startswith("attention")
                                       for e in g.log.events[before:])}
    return {"config": cfg, "backend": label_of(backend), "dtype": str(low).replace("torch.", ""), "clean": clean,
            "injected": caught}


def label_of(backend):
    return "cpu flash" if backend is None else str(backend).split(".")[-1].lower()


def calibrate(dev, say=print) -> dict:
    low = torch.bfloat16 if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    runs = []
    for cfg in CONFIGS["cuda" if dev.type == "cuda" else "cpu"]:
        for name, be in backends(dev):
            say(f"{cfg['name']}, {name}")
            try:
                r = run(dev, cfg, be, low, say=lambda s, a=name: say(f"  {a}: {s}"))
            except torch.cuda.OutOfMemoryError as e:
                r = {"config": cfg, "backend": name, "error": f"out of memory: {str(e).splitlines()[0][:150]}"}
            except RuntimeError as e:          # e.g. "No available kernel": this GPU doesn't have that backend
                r = {"config": cfg, "backend": name, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            runs.append(r)
    ran = [r for r in runs if "clean" in r and r["clean"]["checks"] > 0]
    incomplete = []
    for r in runs:
        c = r.get("clean", {})
        expected = r["config"]["layers"] * r["config"]["steps"]
        reasons = []
        if r.get("error"):
            reasons.append(r["error"])
        if expected <= 0 or c.get("checks") != expected or c.get("forward_checks") != expected:
            reasons.append(f"expected {expected} judged forward and backward checks")
        if c.get("unjudged", 0) or c.get("skipped"):
            reasons.append("unjudged or skipped attention checks")
        z = c.get("worst", float("nan"))
        if not math.isfinite(z) or z < 0:
            reasons.append("missing or non-finite clean score")
        if reasons:
            incomplete.append({"config": r["config"]["name"], "backend": r["backend"], "reasons": reasons})
    alarms = sum(r["clean"]["alarms"] for r in ran)
    worst = max((r["clean"]["worst"] for r in ran), default=0.0)
    out = {"runs": runs, "clean_checks": sum(r["clean"]["checks"] for r in ran), "alarms": alarms, "worst": worst,
           "criteria_met": bool(ran) and not incomplete and alarms == 0 and worst <= 4.0,
           "nothing_ran": not ran, "incomplete": incomplete}
    if dev.type == "cuda":
        out.update(gpu=torch.cuda.get_device_name(dev), cc=list(torch.cuda.get_device_capability(dev)),
                   torch=torch.__version__)
    return out


def markdown(res: dict) -> str:
    lines = [f"## {res.get('gpu', 'CPU')}: {res['clean_checks']} clean attention backward checks, {res['alarms']} "
             f"alarms, worst {res['worst']} (criteria "
             f"{'INCOMPLETE' if res.get('incomplete') or res.get('nothing_ran') else 'MET' if res['criteria_met'] else 'NOT MET'}"
             f": 0 alarms, worst at most 4)", "",
             "| workload | kernel | clean checks | alarms | worst | " + " | ".join(
                 f"{k}" for k in [f"dV + {m} x rms" for m in MULTS] + [f"output x {p}" for p in POW2]) + " |",
             "|---|---|---|---|---|" + "---|" * (len(MULTS) + len(POW2))]
    for r in res["runs"]:
        if "error" in r:
            lines.append(f"| {r['config']['name']} | {r['backend']} | not run: {r['error']} |")
            continue
        c = r["clean"]
        cells = [f"{v['alarms']}/{v['corrupted_calls']}" for v in r["injected"].values()]
        lines.append(f"| {r['config']['name']} | {r['backend']} | {c['checks']} | {c['alarms']} | {c['worst']} | "
                     + " | ".join(cells) + " |")
    for r in res.get("incomplete", []):
        lines.append(f"\nIncomplete: {r['config']}, {r['backend']}: {'; '.join(r['reasons'])}.")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    res = calibrate(torch.device(a.device))
    res["seconds"] = round(time.time() - t0, 1)
    md = markdown(res)
    print(md)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "attention_calib.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        with open(os.path.join(a.out, "attention_calib.md"), "w") as f:
            f.write(md)
    # 0 criteria met, 1 not met, 2 incomplete (including partial execution).
    return 2 if res.get("nothing_ran") or res.get("incomplete") else 0 if res["criteria_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
