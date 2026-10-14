"""Harm study: do silent GPU faults damage training, does the loss curve show it, and does
sdcguard catch them first?

Trains the same GPT several times:
  clean_s0, clean_s1, ...   no faults, different seeds  -> how much runs differ by chance
  clean_s0_repeat           (optional) seed 0 again     -> how much the GPU itself varies run to run
  faulty_*                  seed 0 (paired with clean_s0) on a simulated bad GPU: faults injected
                            into the attention Q/K projections (or other layers, --target)

clean_s0 and the faulty runs see the same initial weights and the same batches, so any
difference between them comes from the injected faults. Kill criterion #2 asks: does the
faulty run end worse than EVERY clean run, by more than the seed-to-seed spread?

Presets:
  smoke / mac / gpu             character-level Shakespeare (tiny, fast)
  ts-smoke / ts-gpu / ts-gpt2   GPT-2 tokenizer on TinyStories (real text, readable samples)
  synth                         offline synthetic text (tests only)

Three ways to run it (paste one line at a time):
  python experiments/harm_study.py --preset mac --rates 0.005,0.02 --fault-scale warp,tile --fault-seeds 1,2,3
      everything in one go (clean runs, then each faulty run, then the report)
  python experiments/harm_study.py --preset ts-gpt2 --one faulty --rate 0.01 --fault-seed 1
      a single run, saved to runs/<preset>/raw/ (the GPU runner starts many of these in parallel)
  python experiments/harm_study.py --preset ts-gpt2 --analyze
      rebuild every verdict, plot and the report from the saved runs

Faults follow NVIDIA's published GPU error patterns (sdcguard.inject). --fault-mix picks
nvidia (their mix), zero (zeroing only) or mantissa (small, plausible-looking bit flips).
NaN/Inf faults are off unless --specials, but big bit flips can still produce them.
"""
from __future__ import annotations

import argparse
import glob
import copy
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sdcguard import FaultInjector, SDCGuard
from sdcguard.inject import fault_model

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TS_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TS_FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}

PRESETS = {
    # character-level Shakespeare
    "smoke": dict(dataset="shakespeare", n_layer=2, n_head=2, n_embd=64, block=64, batch=16,
                  steps=150, eval_every=50, eval_batches=8, lr=1e-3),
    "mac": dict(dataset="shakespeare", n_layer=4, n_head=4, n_embd=128, block=128, batch=32,
                steps=2000, eval_every=200, eval_batches=20, lr=1e-3),
    "gpu": dict(dataset="shakespeare", n_layer=6, n_head=6, n_embd=384, block=256, batch=64,
                steps=5000, eval_every=250, eval_batches=40, lr=1e-3),
    # GPT-2 tokenizer on TinyStories
    "ts-smoke": dict(dataset="tinystories", n_layer=2, n_head=2, n_embd=64, block=64, batch=8,
                     steps=100, eval_every=50, eval_batches=4, lr=1e-3, ts_train_mb=20),
    "ts-gpu": dict(dataset="tinystories", n_layer=8, n_head=8, n_embd=512, block=256, batch=64,
                   steps=3000, eval_every=250, eval_batches=40, lr=6e-4, ts_train_mb=300),
    "ts-gpt2": dict(dataset="tinystories", n_layer=12, n_head=12, n_embd=768, block=512, batch=32,
                    steps=4000, eval_every=250, eval_batches=40, lr=6e-4, ts_train_mb=300),
    # about 0.9B parameters; used to measure how the detector's slowdown changes with model size
    "ts-900m": dict(dataset="tinystories", n_layer=16, n_head=16, n_embd=2048, block=512, batch=16,
                    steps=4000, eval_every=250, eval_batches=20, lr=3e-4, ts_train_mb=300),
    # offline synthetic text, for tests in environments without internet
    "synth": dict(dataset="synthetic", n_layer=2, n_head=2, n_embd=64, block=64, batch=16,
                  steps=120, eval_every=40, eval_batches=4, lr=2e-3),
}


# ----------------------------------------------------------------------------- model
class Attention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.q = nn.Linear(d, d, bias=False)   # separate Q/K/V so faults can target Q/K only
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        h = self.n_head
        q = self.q(x).view(B, T, h, C // h).transpose(1, 2)
        k = self.k(x).view(B, T, h, C // h).transpose(1, 2)
        v = self.v(x).view(B, T, h, C // h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d, n_head)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, block, n_layer, n_head, n_embd):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok.weight          # weight tying, as in GPT-2
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))


# ----------------------------------------------------------------------------- data
@dataclass
class Dataset:
    train: np.ndarray
    val: np.ndarray
    vocab: int
    encode: callable
    decode: callable
    prompt: str


def _download(url, path, max_bytes=None, tries=4):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    headers = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {}
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r, \
                    open(path + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(path + ".part", path)
            return
        except OSError as e:
            if attempt == tries - 1:
                raise
            print(f"download failed ({e}); retrying in {5 * (attempt + 1)}s")
            time.sleep(5 * (attempt + 1))


def load_dataset(name, data_dir, ts_train_mb=200) -> Dataset:
    if name == "shakespeare":
        path = os.path.join(data_dir, "tinyshakespeare.txt")
        if not os.path.exists(path):
            print(f"downloading tiny shakespeare -> {path}")
            _download(SHAKESPEARE_URL, path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        data = np.array([stoi[c] for c in text], dtype=np.uint16)
        n = int(0.9 * len(data))
        return Dataset(data[:n], data[n:], len(chars),
                       lambda s: [stoi[c] for c in s if c in stoi],
                       lambda ids: "".join(chars[i] for i in ids), "ROMEO:")

    if name == "tinystories":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token
        arrays = {}
        for split, fname in TS_FILES.items():
            mb = ts_train_mb if split == "train" else None
            tag = f"{mb}mb" if mb else "all"
            bin_path = os.path.join(data_dir, f"tinystories_{split}_{tag}.bin")
            if not os.path.exists(bin_path):
                txt_path = bin_path[:-4] + ".txt"
                if not os.path.exists(txt_path):
                    print(f"downloading TinyStories {split} ({tag}) -> {txt_path}")
                    _download(TS_BASE + fname, txt_path, mb * 1_000_000 if mb else None)
                with open(txt_path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                stories = text.split("<|endoftext|>")
                if mb:
                    stories = stories[:-1]            # last story was cut off by the byte limit
                print(f"tokenizing {len(stories):,} stories ({split})...")
                with open(bin_path + ".part", "wb") as out:
                    for i in range(0, len(stories), 20000):
                        chunk = [s.strip() for s in stories[i:i + 20000] if s.strip()]
                        for seq in enc.encode_ordinary_batch(chunk):
                            np.asarray(seq + [eot], dtype=np.uint16).tofile(out)
                os.replace(bin_path + ".part", bin_path)
            arrays[split] = np.memmap(bin_path, dtype=np.uint16, mode="r")
        print(f"TinyStories tokens: train {len(arrays['train']):,}  val {len(arrays['val']):,}")
        return Dataset(arrays["train"], arrays["val"], enc.n_vocab,
                       enc.encode_ordinary, enc.decode, "Once upon a time")

    if name == "synthetic":
        # A fixed random bigram language over 64 symbols: learnable, offline, deterministic.
        g = np.random.default_rng(0)
        V = 64
        trans = g.dirichlet(np.full(V, 0.3), size=V)
        seq = np.empty(400_000, dtype=np.uint16)
        seq[0] = 0
        u = g.random(len(seq))
        cdf = np.cumsum(trans, axis=1)
        for i in range(1, len(seq)):
            seq[i] = min(int(np.searchsorted(cdf[seq[i - 1]], u[i])), V - 1)
        n = int(0.9 * len(seq))
        return Dataset(seq[:n], seq[n:], V, lambda s: [ord(c) % V for c in s],
                       lambda ids: " ".join(str(i) for i in ids), "AB")

    raise ValueError(f"unknown dataset {name}")


def get_batch(data, block, batch, gen, device):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=gen).tolist()
    x = torch.from_numpy(np.stack([np.asarray(data[i:i + block], dtype=np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([np.asarray(data[i + 1:i + 1 + block], dtype=np.int64) for i in ix]))
    return x.to(device), y.to(device)


# ----------------------------------------------------------------------------- helpers
def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype)


def resolve_dtype(device, dtype):
    """Fall back to fp16 if this device can't autocast to bf16 (older Macs)."""
    try:
        with autocast_ctx(device, dtype):
            out = F.linear(torch.randn(4, 8, device=device), torch.randn(8, 8, device=device))
        if out.dtype == dtype:
            return dtype
    except Exception as e:  # noqa: BLE001
        print(f"autocast to {dtype} failed on {device} ({e}); using float16")
    return torch.float16


def lr_at(step, cfg):
    """Linear warmup over 5% of steps, then cosine decay to 10% of the peak."""
    peak, steps = cfg["lr"], cfg["steps"]
    warm = max(1, int(0.05 * steps))
    if step < warm:
        return peak * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, steps - warm))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def smooth(xs, k=25):
    out, acc = [], []
    for v in xs:
        acc.append(v)
        if len(acc) > k:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


# ----------------------------------------------------------------------------- training
@dataclass
class RunResult:
    name: str
    seed: int
    train_loss: list
    val_steps: list
    val_loss: list
    final_val: float
    nan_step: int | None
    injected: int
    injected_kinds: dict
    faults_on_checked_steps: int
    faults_caught: int
    guard_checked: int
    guard_alerts: int
    first_alert_step: int | None
    seconds: float
    guard_ops: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    spec: dict = field(default_factory=dict)            # how this run was configured
    fault_events: list = field(default_factory=list)    # [step, kind, rel_change, checked, caught, z]
    alert_steps: list = field(default_factory=list)     # steps where the detector raised alerts
    alerts_off_fault: int = 0                            # alerts not on an injected fault's own output
    off_fault_steps: list = field(default_factory=list)  # steps of those alerts
    off_fault_nonfinite: int = 0                         # of those: NaN/Inf/overflowing outputs (corruption spreading)
    injector_saw: int | None = None                      # matmuls the injector could have corrupted
    max_ok_z: float = 0.0                                # worst residual among checks that passed
    ms_per_step: float | None = None
    env: dict = field(default_factory=dict)
    # After a crash: restart from the newest snapshot with every matmul checked on every step
    # (the bad GPU is still in the machine) and record whether a check names it. None = not run.
    triage: dict | None = None


def code_version(extra=()):
    """Short hash of the code that shapes results (library + this file + `extra` files)."""
    import hashlib
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(root, "src", "sdcguard", "*.py"))) + [os.path.abspath(__file__)]
    files += [os.path.join(os.path.dirname(os.path.abspath(__file__)), x) for x in extra]
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def env_info(device):
    info = {"torch": torch.__version__, "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["bf16_reduced_precision_reduction"] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    return info


@torch.no_grad()
def evaluate(model, ds, cfg, device, dtype):
    model.eval()
    gen = torch.Generator().manual_seed(999)        # same val batches for every run
    losses = []
    for _ in range(cfg["eval_batches"]):
        x, y = get_batch(ds.val, cfg["block"], cfg["batch"], gen, device)
        with autocast_ctx(device, dtype):
            losses.append(model(x, y).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(model, ds, device, dtype, n_tokens=160, seed=0, temperature=0.8, top_k=50):
    """Sample from the trained weights on healthy hardware (no injector)."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([ds.encode(ds.prompt)], dtype=torch.long, device=device)
    for _ in range(n_tokens):
        with autocast_ctx(device, dtype):
            logits = model(idx[:, -model.block:])[:, -1, :].float()
        logits = logits / temperature
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=g).to(device)
        idx = torch.cat([idx, nxt], 1)
    model.train()
    return ds.decode(idx[0].tolist())


def target_modules(model, target):
    mods = []
    for b in model.blocks:
        if "q" in target: mods.append(b.attn.q)
        if "k" in target: mods.append(b.attn.k)
        if "v" in target: mods.append(b.attn.v)
        if "mlp" in target: mods += [b.mlp[0], b.mlp[2]]
        if "proj" in target: mods.append(b.attn.proj)
    if not mods:
        raise SystemExit(f"--target {target!r} matches no layers (use any of q,k,v,mlp,proj)")
    return mods


def train_run(name, seed, ds, cfg, device, dtype, fault_rate=0.0, target="qk", guard_every=10,
              specials=False, n_samples=2, fault_scale="warp", fault_seed=None, fault_mix="nvidia",
              spec=None, triage_steps=0, snapshot_every=50):
    torch.manual_seed(seed)
    model = GPT(ds.vocab, cfg["block"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed)

    inj = None
    if fault_rate > 0:
        inj = FaultInjector(rate=fault_rate, fault_model=fault_model(fault_mix, fault_scale, specials),
                            seed=seed + 7 if fault_seed is None else fault_seed
                            ).arm_on(target_modules(model, target))
    off_fault, off_steps, off_nonfinite, cur = [0], [], [0], [0]

    def on_check(_op, token, res):
        if token is not None:
            inj.resolve(token, res.ok, res.max_z)
        elif not res.ok:
            off_fault[0] += 1
            off_steps.append(cur[0])
            off_nonfinite[0] += int(res.nonfinite)

    guard = SDCGuard(sample_rate=1.0, seed=seed, on_check=on_check,
                     on_check_start=(lambda _op, C: inj.match(C)) if inj is not None else None)

    def snapshot(step):
        # what a checkpoint holds: weights, optimizer state and the data position
        return (step, {k: v.detach().clone() for k, v in model.state_dict().items()},
                copy.deepcopy(opt.state_dict()), gen.get_state())

    triage_on = triage_steps > 0 and inj is not None
    snap = snapshot(-1) if triage_on else None

    train_loss, val_steps, val_loss, nan_step, alert_steps = [], [], [], None, []
    t0, train_secs = time.time(), 0.0
    for step in range(cfg["steps"]):
        ts = time.time()
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        x, y = get_batch(ds.train, cfg["block"], cfg["batch"], gen, device)
        use_guard = guard_every > 0 and step % guard_every == 0
        cur[0] = step
        if inj is not None:
            inj.step = step
        alerts_before = guard.stats["alerts"]
        with (inj if inj is not None else _null()):
            with (guard.at_step(step) if use_guard else _null()):
                with autocast_ctx(device, dtype):
                    loss = model(x, y)
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        lv = loss.item()
        train_secs += time.time() - ts
        train_loss.append(lv)
        if guard.stats["alerts"] > alerts_before:
            alert_steps.append(step)
        if not math.isfinite(lv):
            nan_step = step
            print(f"  [{name}] loss became {lv} at step {step}: visible failure, stopping run")
            break
        if triage_on and (step + 1) % snapshot_every == 0:
            snap = snapshot(step)
        if (step + 1) % cfg["eval_every"] == 0 or step == cfg["steps"] - 1:
            # evaluated outside the injector: we measure the trained weights on a healthy GPU
            vl = evaluate(model, ds, cfg, device, dtype)
            val_steps.append(step + 1)
            val_loss.append(vl)
            print(f"  [{name}] step {step + 1:5d}  train {lv:.4f}  val {vl:.4f}  "
                  f"alerts {guard.stats['alerts']}  faults {len(inj.events) if inj else 0}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    n_main = len(inj.events) if inj else 0
    saw_main = inj.eligible if inj else None
    triage = None
    if triage_on and nan_step is not None:
        triage = _triage(name, model, opt, gen, ds, cfg, device, dtype, inj, snap, triage_steps, seed)
    if inj is not None:
        inj.remove_hooks()
    samples = [] if nan_step is not None else [generate(model, ds, device, dtype, seed=i) for i in range(n_samples)]
    kinds, on_checked, caught, events = {}, 0, 0, []
    for e in (inj.events[:n_main] if inj else []):
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        on_checked += bool(e.get("checked"))
        caught += bool(e.get("caught"))
        rc = e.get("rel_change")
        events.append([e["step"], e["kind"], rc if rc is None or math.isfinite(rc) else "inf",
                       bool(e.get("checked")), bool(e.get("caught")), e.get("z")])
    return RunResult(
        name=name, seed=seed, train_loss=train_loss, val_steps=val_steps, val_loss=val_loss,
        final_val=val_loss[-1] if val_loss and nan_step is None else float("nan"),
        nan_step=nan_step, injected=n_main, injected_kinds=kinds,
        faults_on_checked_steps=on_checked, faults_caught=caught,
        guard_checked=guard.stats["checked"], guard_alerts=guard.stats["alerts"],
        first_alert_step=guard.log.first_step(), seconds=round(time.time() - t0, 1),
        guard_ops=dict(guard.ops_seen), samples=samples, spec=dict(spec or {}),
        fault_events=events, alert_steps=alert_steps, alerts_off_fault=off_fault[0], off_fault_steps=off_steps,
        off_fault_nonfinite=off_nonfinite[0],
        injector_saw=saw_main,
        max_ok_z=guard.max_z_seen, ms_per_step=round(1000 * train_secs / max(1, len(train_loss)), 2),
        env=dict(env_info(device), code=code_version(), guard_every=guard_every),
        triage=triage,
    )


def _triage(name, model, opt, gen, ds, cfg, device, dtype, inj, snap, max_steps, seed):
    """The run crashed. Restart from the newest snapshot on the same (still bad) device, checking
    every matmul on every step, and stop as soon as a check catches an injected fault, the run
    crashes again, or max_steps pass. Uses its own detector so the main run's numbers don't change."""
    s0, msd, osd, gst = snap                    # s0 = -1: crashed before the first snapshot, start over
    model.load_state_dict(msd)
    opt.load_state_dict(osd)
    gen.set_state(gst)
    first = len(inj.events)
    off = {"alerts": 0, "nonfinite": 0}

    def on_check(_op, token, res):
        if token is not None:
            inj.resolve(token, res.ok, res.max_z)
        elif not res.ok:
            off["alerts"] += 1
            off["nonfinite"] += int(res.nonfinite)

    tg = SDCGuard(sample_rate=1.0, seed=seed + 1, on_check=on_check, on_check_start=lambda _op, C: inj.match(C))
    caught_step = recrash = None
    steps_run = 0
    for step in range(s0 + 1, min(cfg["steps"], s0 + 1 + max_steps)):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        x, y = get_batch(ds.train, cfg["block"], cfg["batch"], gen, device)
        inj.step = step
        with inj:
            with tg.at_step(step):
                with autocast_ctx(device, dtype):
                    loss = model(x, y)
                loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps_run += 1
        hit = [e["step"] for e in inj.events[first:] if e.get("caught")]
        if hit:
            caught_step = min(hit)
            break
        if not math.isfinite(loss.item()):
            recrash = step
            break
    ev = inj.events[first:]
    out = {"restart_after_step": s0, "max_steps": max_steps, "steps_run": steps_run,
           "caught_step": caught_step, "steps_to_catch": (caught_step - s0) if caught_step is not None else None,
           "recrash_step": recrash, "faults": len(ev), "faults_checked": sum(bool(e.get("checked")) for e in ev),
           "faults_caught": sum(bool(e.get("caught")) for e in ev), "checks": tg.stats["checked"],
           "alerts_off_fault": off["alerts"], "alerts_off_fault_nonfinite": off["nonfinite"]}
    print(f"  [{name}] restart triage from step {s0 + 1}: "
          + (f"named the bad device at step {caught_step} ({caught_step - s0} step"
             f"{'' if caught_step - s0 == 1 else 's'} after the restart point)"
             if caught_step is not None else
             (f"crashed again at step {recrash} before a check caught a fault" if recrash is not None
              else f"no fault caught in {steps_run} steps")), flush=True)
    return out


# ----------------------------------------------------------------------------- saving runs
def run_name(kind, seed=0, rate=None, target="qk", mix="nvidia", scale="warp", fault_seed=None, repeat=False):
    if kind == "clean":
        return f"clean_s{seed}" + ("_repeat" if repeat else "")
    return f"faulty_{target}_{mix}_{scale}_r{rate}" + (f"_f{fault_seed}" if fault_seed is not None else "")


def save_run(r: RunResult, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".part", "w") as f:
        json.dump(r.__dict__, f)
    os.replace(path + ".part", path)


def load_run(path: str) -> RunResult:
    with open(path) as f:
        d = json.load(f)
    known = RunResult.__dataclass_fields__
    return RunResult(**{k: v for k, v in d.items() if k in known})


def run_from_spec(spec: dict, data_dir="data", out_root="runs", skip_existing=True, device="auto", dtype="bf16"):
    """Train one run described by `spec` and save it to <out_root>/<preset>/raw/<name>.json.

    spec keys: preset, kind ('clean'|'faulty'), seed, repeat, rate, target, mix, scale,
    fault_seed, guard_every, specials, steps, exact_reductions. Used by the GPU runner.
    """
    preset = spec["preset"]
    cfg = dict(PRESETS[preset])
    if spec.get("steps"):
        cfg["steps"] = spec["steps"]
    kind = spec.get("kind", "clean")
    seed = spec.get("seed", 0)
    name = run_name(kind, seed, spec.get("rate"), spec.get("target", "qk"), spec.get("mix", "nvidia"),
                    spec.get("scale", "warp"), spec.get("fault_seed"), spec.get("repeat", False))
    path = os.path.join(out_root, preset, "raw", name + ".json")
    if skip_existing and os.path.exists(path):
        try:
            saved = load_run(path).env.get("code")
        except (OSError, ValueError, TypeError, KeyError):
            saved = None
        now = code_version()
        if saved == now:
            print(f"[{name}] already done with this code, skipping")
            return path
        print(f"[{name}] saved with other code ({saved}, now {now}): running it again")
    if spec.get("exact_reductions") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dev = pick_device(device)
    dt = resolve_dtype(dev, torch.bfloat16 if dtype == "bf16" else torch.float16)
    ds = load_dataset(cfg["dataset"], data_dir, cfg.get("ts_train_mb", 200))
    print(f"[{name}] device={dev} dtype={dt} preset={preset} cfg={cfg}", flush=True)
    r = train_run(name, seed, ds, cfg, dev, dt,
                  fault_rate=spec.get("rate", 0.0) if kind == "faulty" else 0.0,
                  target=spec.get("target", "qk"), guard_every=spec.get("guard_every", 10),
                  specials=spec.get("specials", False), fault_scale=spec.get("scale", "warp"),
                  fault_seed=spec.get("fault_seed"), fault_mix=spec.get("mix", "nvidia"),
                  spec={**spec, "cfg": cfg, "name": name},
                  triage_steps=spec.get("triage_steps", 0), snapshot_every=spec.get("snapshot_every", 50))
    save_run(r, path)
    print(f"[{name}] saved -> {path} ({r.seconds:.0f}s, {r.ms_per_step} ms/step)", flush=True)
    return path


# ----------------------------------------------------------------------------- analysis
SIZE_BUCKETS = [(0.0, 1e-3, "<0.1%"), (1e-3, 1e-2, "0.1-1%"), (1e-2, 1e-1, "1-10%"),
                (1e-1, 1.0, "10-100%"), (1.0, float("inf"), ">100%"), (float("inf"), None, "NaN/Inf/huge")]


def size_bucket(rel):
    if rel is None:
        return None
    if rel == "inf" or not math.isfinite(float(rel)):
        return SIZE_BUCKETS[-1][2]
    rel = float(rel)
    for lo, hi, label in SIZE_BUCKETS[:-1]:
        if lo <= rel < hi:
            return label
    return SIZE_BUCKETS[-2][2]


def detection_by_size(events):
    """{bucket: [checked, caught]} for faults the detector looked at."""
    out = {label: [0, 0] for _, _, label in SIZE_BUCKETS}
    for _step, _kind, rel, checked, caught, _z in events:
        b = size_bucket(rel)
        if b is not None and checked:
            out[b][0] += 1
            out[b][1] += int(caught)
    return out


DETECT_EVERY = (10, 20, 50)


def detection_at(r, k, crash):
    """Detection if the detector had checked only every k-th step (k a multiple of the run's
    guard_every): keep only checks on steps divisible by k. Counts injected faults the detector
    looked at and caught, the first step a check caught one, and alerts not on a fault's output."""
    checked = [e for e in r.fault_events if e[3] and e[0] % k == 0]
    caught = [e for e in checked if e[4]]
    first = min((e[0] for e in caught), default=None)
    before = crash is not None and first is not None and first < crash
    return {"every": k, "faults_checked": len(checked), "faults_caught": len(caught), "first_caught_step": first,
            "flagged_before_crash": before, "lead_steps": (crash - first) if before else None,
            "off_fault_alerts": sum(1 for st in r.off_fault_steps if st % k == 0)}


def summarize(cleans, faulty, repeat=None):
    cf = [r.final_val for r in cleans if math.isfinite(r.final_val)]
    if not cf:
        raise SystemExit("no finished clean runs to compare against")
    mean = sum(cf) / len(cf)
    std = math.sqrt(sum((x - mean) ** 2 for x in cf) / (len(cf) - 1)) if len(cf) > 1 else 0.0
    fv = faulty.final_val
    spread = max(cf) - min(cf)                       # how much changing the seed moves the result
    base = next((c for c in cleans if c.seed == faulty.seed), None)
    paired = fv - base.final_val if (base is not None and math.isfinite(fv)
                                     and math.isfinite(base.final_val)) else None   # same seed, only faults differ
    same_seed_noise = (abs(repeat.final_val - base.final_val)
                       if repeat is not None and base is not None and math.isfinite(repeat.final_val) else None)
    # Harm must beat chance: worse than every clean run AND the fault effect bigger than the seed effect
    # (and bigger than the GPU's own run-to-run noise, when we measured it).
    harm = bool(len(cf) >= 3 and paired is not None and fv > max(cf) and paired > spread
                and (same_seed_noise is None or paired > same_seed_noise))

    # does the faulty training curve leave the band the clean runs span?
    runs = cleans + [faulty]
    n = min(len([v for v in r.train_loss if math.isfinite(v)]) for r in runs)
    sm = {r.name: smooth(r.train_loss[:n]) for r in runs}
    lo = [min(sm[c.name][i] for c in cleans) for i in range(n)]
    hi = [max(sm[c.name][i] for c in cleans) for i in range(n)]
    f = sm[faulty.name]
    widths = sorted(hi[i] - lo[i] for i in range(n))
    typical_width = widths[len(widths) // 2] if widths else 0.0
    max_excess = max((f[i] - hi[i] for i in range(n)), default=0.0)
    visible = bool(max_excess > 2 * typical_width)   # a bump clearly above normal run-to-run spread

    ev = faulty.fault_events
    big = [e for e in ev if e[2] is not None and (e[2] == "inf" or float(e[2]) >= 1e-2)]
    problems = []
    if faulty.injected == 0:
        if faulty.injector_saw:      # the injector worked; no fault happened to be drawn
            problems.append(f"LEFT OUT: no fault happened in this run by chance ({faulty.injector_saw:,} eligible "
                            f"matmuls at rate {faulty.spec.get('rate')}); not a device problem")
        else:
            problems.append("no faults were injected (the injector never saw a matmul on this device)")
    if sum(c.guard_checked for c in cleans) == 0:
        problems.append("the detector checked 0 matmuls on the clean runs")
    if base is None:
        problems.append(f"no clean run with the same seed ({faulty.seed}) to pair with")
    crash = faulty.nan_step
    if crash is None and not math.isfinite(fv):
        crash = len(faulty.train_loss)       # weights went bad at the very end: still a crash
    first_alert = faulty.first_alert_step
    ge = int(faulty.env.get("guard_every") or faulty.spec.get("guard_every") or 10)
    at = {str(k): detection_at(faulty, k, crash) for k in DETECT_EVERY if ge > 0 and k % ge == 0}
    harmed = crash is not None or harm
    all_clean = cleans + ([repeat] if repeat is not None else [])
    return {
        "name": faulty.name, "spec": {k: v for k, v in faulty.spec.items() if k != "cfg"},
        "valid": not problems, "problems": problems,
        "clean_final_val": {c.name: c.final_val for c in cleans},
        "clean_mean": mean, "clean_std": std, "clean_min": min(cf), "clean_max": max(cf),
        "faulty_final_val": fv,
        "clean_seed_spread": spread,
        "same_seed_gpu_noise": same_seed_noise,
        "paired_harm_vs_same_seed": paired,
        "harm_vs_spread_ratio": (paired / spread) if paired is not None and spread > 0 else None,
        "clean_seeds": len(cf),
        "kill2_harm": harm,
        "harmed": harmed,                    # crashed (NaN), or survived but worse beyond seed spread
        "guard_every": ge,
        "detection_at": at,
        "visible_failure_nan_step": crash,
        "train_curve_max_excess_over_clean_band": max_excess,
        "clean_band_typical_width": typical_width,
        "loss_curve_shows_anomaly": visible,
        "faults_injected": faulty.injected, "fault_kinds": faulty.injected_kinds,
        "first_fault_step": ev[0][0] if ev else None,
        "first_big_fault_step": big[0][0] if big else None,
        "triage": faulty.triage,
        "detector": {
            "false_alarms_clean": sum(c.guard_alerts for c in all_clean),
            "checks_clean": sum(c.guard_checked for c in all_clean),
            "worst_clean_z": max((c.max_ok_z for c in all_clean), default=None),
            "faults_while_detector_on": faulty.faults_on_checked_steps,
            "faults_caught": faulty.faults_caught,
            "alerts_faulty": faulty.guard_alerts,
            "alerts_off_fault": faulty.alerts_off_fault,
            "alerts_off_fault_spread": faulty.off_fault_nonfinite,
            "alerts_off_fault_unexplained": faulty.alerts_off_fault - faulty.off_fault_nonfinite,
            "first_alert_step": first_alert,
            "alerted_before_crash": bool(crash is not None and first_alert is not None and first_alert < crash),
            "lead_steps": (crash - first_alert) if crash is not None and first_alert is not None
            and first_alert < crash else None,
            "by_size": detection_by_size(ev),
        },
    }


def _at_line(at):
    parts = []
    for k, d in at.items():
        first = d["first_caught_step"]
        parts.append(f"every {k}: caught {d['faults_caught']}/{d['faults_checked']}"
                     + (f", first at step {first}" if first is not None else ", never caught"))
    return "; ".join(parts)


def verdict_lines(name, s):
    d = s["detector"]
    lines = [f"=== {name} ==="]
    if not s["valid"]:
        if all(p.startswith("LEFT OUT") for p in s["problems"]):
            lines.append("; ".join(s["problems"]) + ".")
        else:
            lines.append("INVALID RUN - " + "; ".join(s["problems"]) + ". Run: python -m sdcguard.selftest")
        return "\n".join(lines)
    ge = s.get("guard_every", 10)
    at = s.get("detection_at", {})
    own = at.get(str(ge)) or {}
    crash = s["visible_failure_nan_step"]
    if crash is not None:
        first = own.get("first_caught_step")
        lines.append(f"Faulty run CRASHED (NaN) at step {crash}. Counted as harmed (kill criterion 2).")
        lines.append("Before the crash the loss curve " + ("showed a visible bump" if s["loss_curve_shows_anomaly"]
                     else "looked normal (inside the clean runs' range)") + ".")
        if first is not None and first < crash:
            lines.append(f"sdcguard (checking every {ge} steps) caught an injected fault at step {first}: "
                         f"{crash - first} steps before the crash.")
        else:
            lines.append(f"sdcguard (checking every {ge} steps) did not catch an injected fault before the crash.")
    else:
        lines.append(f"Final val loss: faulty {s['faulty_final_val']:.4f} vs clean "
                     f"{s['clean_min']:.4f}-{s['clean_max']:.4f} (mean {s['clean_mean']:.4f}, {s['clean_seeds']} seeds).")
        noise = s.get("same_seed_gpu_noise")
        paired = s["paired_harm_vs_same_seed"]
        lines.append(f"Fault effect (vs same-seed clean run): "
                     + (f"{paired:+.4f}" if paired is not None else "n/a")
                     + f"; seed-to-seed spread: {s['clean_seed_spread']:.4f}"
                     + (f"; same-seed GPU noise: {noise:.4f}" if noise is not None else "") + ".")
        if s["clean_seeds"] < 3:
            lines.append("Kill criterion 2 (harm): NO VERDICT - run at least 3 clean seeds.")
        else:
            lines.append("Kill criterion 2 (harm): " + (
                "MET - worse than every clean run, and the fault effect is bigger than the seed effect."
                if s["kill2_harm"] else "NOT MET - within normal run-to-run variation."))
        lines.append("Loss curve: " + ("shows a visible bump" if s["loss_curve_shows_anomaly"] else "looks normal")
                     + f" (max rise above the clean runs {s['train_curve_max_excess_over_clean_band']:.4f}, "
                     f"typical clean spread {s['clean_band_typical_width']:.4f}).")
        first = own.get("first_caught_step")
        lines.append(f"sdcguard (checking every {ge} steps): "
                     + (f"caught an injected fault at step {first}." if first is not None else "never caught a fault."))
    tr = s.get("triage")
    if crash is not None and tr:
        if tr["caught_step"] is not None:
            lines.append(f"Restart triage: restarted from the snapshot after step {tr['restart_after_step']} on the "
                         f"same bad device with every step checked; a check caught an injected fault at step "
                         f"{tr['caught_step']} ({tr['steps_to_catch']} step{'' if tr['steps_to_catch'] == 1 else 's'} "
                         f"in), naming the device.")
        elif tr["recrash_step"] is not None:
            lines.append(f"Restart triage: restarted from the snapshot after step {tr['restart_after_step']}; "
                         f"crashed again at step {tr['recrash_step']} before a check caught a fault.")
        else:
            lines.append(f"Restart triage: no fault caught in {tr['steps_run']} checked steps after the restart.")
    if at:
        lines.append(f"Detection if checking less often (same run, keeping only those checks): {_at_line(at)}.")
    spread = d.get("alerts_off_fault_spread", 0)
    unexplained = d.get("alerts_off_fault_unexplained", d["alerts_off_fault"])
    lines.append(f"Detector: caught {d['faults_caught']} of {d['faults_while_detector_on']} corrupted outputs it "
                 f"checked. Alerts on outputs the injector did not corrupt: {d['alerts_off_fault']} "
                 f"({spread} had NaN/Inf/overflowing values, i.e. the corruption spreading into later layers; "
                 f"{unexplained} unexplained, counted as false alarms). Clean runs: {d['false_alarms_clean']} "
                 f"false alarms in {d['checks_clean']} checks.")
    sizes = ", ".join(f"{k}: {v[1]}/{v[0]}" for k, v in d["by_size"].items() if v[0])
    if sizes:
        lines.append(f"Caught by fault size (change to the affected 128-value block): {sizes}.")
    return "\n".join(lines)


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _flagged(s, k):
    """Did a check every k steps catch an injected fault before the harm? Crashed runs: before the
    crash. Survivors that were harmed: at any point in the run."""
    d = s.get("detection_at", {}).get(str(k))
    if d is None:
        return None
    if s["visible_failure_nan_step"] is not None:
        return d["flagged_before_crash"]
    return d["first_caught_step"] is not None


def _named(s, k):
    """Caught before the harm at every k steps, or (for crashes) named by the restart triage."""
    f = _flagged(s, k)
    tr = s.get("triage")
    if s["visible_failure_nan_step"] is not None and tr and tr.get("caught_step") is not None:
        return True
    return f


def aggregate(label, summaries):
    """Roll up several faulty runs (different fault patterns) of one configuration."""
    n = len(summaries)
    valid = [s for s in summaries if s["valid"]]
    crashed = [s for s in valid if s["visible_failure_nan_step"] is not None]
    survived = [s for s in valid if s["visible_failure_nan_step"] is None]
    ge = valid[0]["guard_every"] if valid else 10
    own = str(ge)
    leads = [s["detection_at"][own]["lead_steps"] for s in crashed
             if own in s.get("detection_at", {}) and s["detection_at"][own]["lead_steps"] is not None]
    silent_crashes = [s for s in crashed if not s["loss_curve_shows_anomaly"]]
    faults_on = sum(s["detector"]["faults_while_detector_on"] for s in valid)
    caught = sum(s["detector"]["faults_caught"] for s in valid)
    harmed_survivors = [s for s in survived if s["kill2_harm"]]
    harmed = [s for s in valid if s["harmed"]]
    ks = sorted({int(k) for s in valid for k in s.get("detection_at", {})})
    flagged_at = {str(k): sum(1 for s in harmed if _flagged(s, k)) for k in ks}
    triaged = [s for s in crashed if s.get("triage")]
    named_at = {str(k): sum(1 for s in harmed if _named(s, k)) for k in ks} if triaged else {}
    caught_at = {str(k): [sum(s["detection_at"].get(str(k), {}).get("faults_caught", 0) for s in valid),
                          sum(s["detection_at"].get(str(k), {}).get("faults_checked", 0) for s in valid)]
                 for k in ks}
    by_size = {}
    for s in valid:
        for k, (a, b) in s["detector"]["by_size"].items():
            by_size.setdefault(k, [0, 0])
            by_size[k][0] += a
            by_size[k][1] += b
    agg = {
        "label": label, "runs": n, "invalid_runs": n - len(valid),
        "spec": summaries[0]["spec"] if summaries else {}, "guard_every": ge,
        "crashed": len(crashed), "crash_steps": [s["visible_failure_nan_step"] for s in crashed],
        "crashes_with_normal_loss_curve_before": len(silent_crashes),
        "crashes_flagged_before": len(leads),
        "detection_lead_steps": leads,
        "survived": len(survived),
        "survived_and_worse_than_every_clean_run": len(harmed_survivors),
        "survived_fault_effects": [s["paired_harm_vs_same_seed"] for s in survived],
        "clean_seed_spread": summaries[0]["clean_seed_spread"] if summaries else None,
        "survived_flagged": sum(1 for s in survived if _flagged(s, ge)),
        "harmed": len(harmed), "harmed_flagged_at": flagged_at, "caught_at": caught_at,
        "harmed_named_at": named_at,
        "triage_runs": len(triaged),
        "triage_named": sum(1 for s in triaged if s["triage"]["caught_step"] is not None),
        "triage_steps_to_catch": [s["triage"]["steps_to_catch"] for s in triaged
                                  if s["triage"]["caught_step"] is not None],
        "triage_recrashed": sum(1 for s in triaged if s["triage"]["recrash_step"] is not None),
        "triage_unexplained_alerts": sum(s["triage"]["alerts_off_fault"] - s["triage"]["alerts_off_fault_nonfinite"]
                                         for s in triaged),
        "faults_caught": caught, "faults_while_detector_on": faults_on, "by_size": by_size,
        "first_alert_steps": [s["detector"]["first_alert_step"] for s in summaries],
        "alerts_off_fault": sum(s["detector"]["alerts_off_fault"] for s in valid),
        "alerts_off_fault_spread": sum(s["detector"].get("alerts_off_fault_spread", 0) for s in valid),
        "alerts_off_fault_unexplained": sum(s["detector"].get("alerts_off_fault_unexplained",
                                                              s["detector"]["alerts_off_fault"]) for s in valid),
        "false_alarms_clean": summaries[0]["detector"]["false_alarms_clean"] if summaries else None,
    }
    lines = [f"##### {label}: {n} runs with different fault patterns (detector every {ge} steps) #####"]
    if n - len(valid):
        lines.append(f"Left out (no faults happened, or invalid): {n - len(valid)}. See their verdict.txt.")
    lines += [f"Crashed: {len(crashed)} of {len(valid)}"
              + (f" (steps {agg['crash_steps']}); loss curve looked normal before {len(silent_crashes)} of them"
                 if crashed else "") + ".",
              (f"Caught an injected fault before the crash: {len(leads)} of {len(crashed)}; lead {sorted(leads)} steps."
               if crashed else "Caught before the crash: n/a (no crashes)."),
              f"Survived: {len(survived)}; worse than every clean run AND beyond seed spread: {len(harmed_survivors)}."
              + (f" Fault effects: {[round(x, 4) if x is not None else None for x in agg['survived_fault_effects']]}"
                 f" vs seed spread {agg['clean_seed_spread']:.4f}. Caught by sdcguard: {agg['survived_flagged']} of "
                 f"{len(survived)}." if survived else ""),
              f"Harmed (crashed, or survived but worse beyond seed spread): {len(harmed)} of {len(valid)}. "
              "Of those, caught before the harm: "
              + ", ".join(f"every {k} steps {v}/{len(harmed)}" for k, v in flagged_at.items()) + ".",
              (f"Restart triage after the crash (every step checked from the newest snapshot): named the device in "
               f"{agg['triage_named']} of {len(triaged)} crashed runs, {sorted(agg['triage_steps_to_catch'])} steps "
               f"after the restart point. Named before the crash or on the restart: "
               + ", ".join(f"every {k} steps {v}/{len(harmed)}" for k, v in named_at.items()) + "."
               if triaged else "Restart triage: not run."),
              f"Detector caught {caught} of {faults_on} corrupted outputs it checked. Alerts on outputs the injector "
              f"did not corrupt: {agg['alerts_off_fault_spread']} from the corruption spreading (NaN/Inf/overflow), "
              f"{agg['alerts_off_fault_unexplained']} unexplained (false alarms); "
              f"{agg['false_alarms_clean']} false alarms on clean runs.",
              "By fault size: " + ", ".join(f"{k}: {v[1]}/{v[0]}" for k, v in by_size.items() if v[0]) + "."]
    return agg, "\n".join(lines)


def plot(cleans, faulty, summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, r in enumerate(cleans):
        lbl = "clean runs (different seeds)" if i == 0 else None
        ax1.plot(smooth([v for v in r.train_loss if math.isfinite(v)]), color="0.6", linewidth=1, label=lbl)
        ax2.plot(r.val_steps, r.val_loss, color="0.6", marker="o", markersize=3, linewidth=1, label=lbl)
    finite = [v for v in faulty.train_loss if math.isfinite(v)]
    ax1.plot(smooth(finite), color="tab:orange", linewidth=1.6, label=faulty.name)
    ax2.plot(faulty.val_steps, faulty.val_loss, color="tab:orange", marker="o", markersize=3, label=faulty.name)
    own = summary.get("detection_at", {}).get(str(summary.get("guard_every", 10)))
    fa = own["first_caught_step"] if own else summary["detector"]["first_alert_step"]
    if fa is not None:
        for ax in (ax1, ax2):
            ax.axvline(fa, color="red", linestyle="--", linewidth=1, label="sdcguard catches a fault")
    crash = summary.get("visible_failure_nan_step")
    if crash is not None:
        for ax in (ax1, ax2):
            ax.axvline(crash, color="black", linestyle=":", linewidth=1.2, label="faulty run crashed (NaN)")
    ax1.set_title("Training loss (smoothed)")
    ax2.set_title("Validation loss")
    for ax in (ax1, ax2):
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _group_key(spec):
    return (spec.get("target", "qk"), spec.get("mix", "nvidia"), spec.get("scale", "warp"), spec.get("rate"))


def analyze(out_root, preset, write_plots=True):
    """Rebuild every verdict, plot, aggregate and the report from the saved runs."""
    base = os.path.join(out_root, preset)
    runs = {}
    for p in sorted(glob.glob(os.path.join(base, "raw", "*.json"))):
        r = load_run(p)
        runs[r.name] = r
    cleans = [r for n, r in sorted(runs.items()) if n.startswith("clean_s") and not n.endswith("_repeat")]
    repeat = runs.get("clean_s0_repeat")
    faulty = [r for n, r in sorted(runs.items()) if n.startswith("faulty_")]
    if not cleans:
        print(f"no clean runs in {base}/raw yet")
        return None
    report, summaries, groups = [], {}, {}
    for f in faulty:
        s = summarize(cleans, f, repeat)
        summaries[f.name] = s
        groups.setdefault(_group_key(f.spec), []).append(s)
        out_dir = os.path.join(base, f.name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump({"summary": s, "config": f.spec.get("cfg"), "env": f.env}, fh, indent=1)
        with open(os.path.join(out_dir, "samples.txt"), "w") as fh:
            for r in cleans + [f]:
                fh.write(f"===== {r.name} (final val {r.final_val:.4f}) =====\n")
                for smp in r.samples:
                    fh.write(smp.strip() + "\n-----\n")
        if write_plots:
            plot(cleans, f, s, os.path.join(out_dir, "harm.png"))
        v = verdict_lines(f.name, s)
        with open(os.path.join(out_dir, "verdict.txt"), "w") as fh:
            fh.write(v + "\n")
        report.append(v)
    aggregates, agg_text = [], []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        target, mix, scale, rate = key
        agg, text = aggregate(f"{preset} {target} {mix} {scale} rate {rate}", groups[key])
        aggregates.append(agg)
        agg_text.append(text)
    with open(os.path.join(base, "aggregate.json"), "w") as fh:
        json.dump(aggregates, fh, indent=1)
    with open(os.path.join(base, "aggregate.txt"), "w") as fh:
        fh.write("\n\n".join(agg_text) + "\n")
    write_report(base, preset, cleans, repeat, summaries, aggregates, faulty)
    print("\n" + "\n\n".join(report + agg_text))
    print(f"\nreport -> {os.path.join(base, 'REPORT.md')}")
    return aggregates


def write_report(base, preset, cleans, repeat, summaries, aggregates, faulty_runs=()):
    env = cleans[0].env
    all_clean = cleans + ([repeat] if repeat is not None else [])
    checks = sum(c.guard_checked for c in all_clean)
    alarms = sum(c.guard_alerts for c in all_clean)
    hours = sum(c.seconds for c in all_clean) / 3600
    off_fault = sum(a["alerts_off_fault_unexplained"] for a in aggregates)
    spread = sum(a["alerts_off_fault_spread"] for a in aggregates)
    ge = aggregates[0]["guard_every"] if aggregates else env.get("guard_every", "?")
    versions = sorted({str(r.env.get("code")) for r in list(all_clean) + list(faulty_runs)})
    harmed = sum(a["harmed"] for a in aggregates)
    valid = sum(a["runs"] - a["invalid_runs"] for a in aggregates)
    ks = sorted({int(k) for a in aggregates for k in a["harmed_flagged_at"]})
    L = [f"# Harm study report: {preset}", ""]
    if len(versions) > 1:
        L += [f"**WARNING: runs were made with {len(versions)} different code versions {versions}. "
              "Results may mix old and new code.**", ""]
    L += ["## Kill criteria (this study)", "",
          f"- **#1 false alarms (in this study):** clean runs: **{alarms}** in {checks:,} checks "
          f"(worst clean z {max(c.max_ok_z for c in all_clean):.2f}, threshold 6). Faulty runs: **{off_fault}** "
          f"unexplained alerts on outputs the injector did not corrupt (plus {spread} alerts on NaN/Inf/overflowing "
          "outputs, which is the corruption spreading into later layers, not false alarms). The 5 GPU-hour soak and "
          "the slowdown are in calibration/<preset>/CALIBRATION.md.",
          f"- **#2 harm:** {harmed} of {valid} faulty runs were harmed. A run counts as harmed if it crashed (NaN) "
          "or survived but ended worse than every clean run by more than the seed-to-seed spread.",
          f"- **#3 caught:** of the {harmed} harmed runs, caught (a check caught an injected fault before the crash, "
          "or during the run for survivors): "
          + ", ".join(f"checking every {k} steps: **{sum(a['harmed_flagged_at'].get(str(k), 0) for a in aggregates)}"
                      f"/{harmed}**" for k in ks) + "."]
    tr_runs = sum(a.get("triage_runs", 0) for a in aggregates)
    if tr_runs:
        tr_named = sum(a.get("triage_named", 0) for a in aggregates)
        steps = sorted(x for a in aggregates for x in a.get("triage_steps_to_catch", []))
        tr_unexpl = sum(a.get("triage_unexplained_alerts", 0) for a in aggregates)
        L += [f"- **#3b named on the restart:** after each crash the run restarted from its newest snapshot on the "
              f"same bad device with every matmul checked on every step (up to "
              f"{aggregates[0]['spec'].get('triage_steps', '?')} steps). A check caught an injected fault in "
              f"**{tr_named} of {tr_runs}** crashed runs, {steps} steps after the restart point "
              f"({tr_unexpl} unexplained alerts). Harmed runs named before the crash or on the restart: "
              + ", ".join(f"checking every {k} steps: **{sum(a.get('harmed_named_at', {}).get(str(k), 0) for a in aggregates)}"
                          f"/{harmed}**" for k in ks) + "."]
    L += [
          f"- The detector checked every {ge} steps during these runs. Results for longer intervals come from "
          "the same runs, keeping only the checks on steps divisible by them. (Checked steps keep matmul "
          "partial sums in fp32, so a real run checking less often could drift very slightly differently.) The GPU names "
          "come from the multi-GPU demo, not from these single-GPU runs.",
          f"- Code version(s): {', '.join(versions)}", "",
          "## Setup", "",
          f"- Hardware: {env.get('gpu') or env.get('device')}, torch {env.get('torch')}",
          f"- Model/training: {json.dumps(cleans[0].spec.get('cfg', {}))}",
          f"- Clean runs: {', '.join(f'{c.name} {c.final_val:.4f}' for c in cleans)}"
          + (f"; same seed again: {repeat.final_val:.4f}" if repeat else ""),
          f"- Clean training time, including evaluation: {hours:.2f} GPU-hours", "",
          "| config | runs | crashed | loss looked normal before crash | caught before crash | "
          "median lead (steps) | survived | survived & harmed | harmed: caught "
          + "/".join(f"@{k}" for k in ks) + " | caught / checked |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in aggregates:
        lead = _median(a["detection_lead_steps"])
        L.append(f"| {a['label']} | {a['runs']} | {a['crashed']} | {a['crashes_with_normal_loss_curve_before']} | "
                 f"{a['crashes_flagged_before']} | {lead if lead is not None else '-'} | {a['survived']} | "
                 f"{a['survived_and_worse_than_every_clean_run']} | "
                 + "/".join(str(a["harmed_flagged_at"].get(str(k), "-")) for k in ks) + f" of {a['harmed']} | "
                 f"{a['faults_caught']} / {a['faults_while_detector_on']} |")
    tot = {}
    for a in aggregates:
        for k, (c, h) in a["by_size"].items():
            tot.setdefault(k, [0, 0])
            tot[k][0] += c
            tot[k][1] += h
    L += ["", "## Detection by fault size (all faulty runs)", "",
          "Size = how much the corrupted values changed their 128-value output block (L2 norm ratio).", "",
          "| fault size | checked | caught |", "|---|---|---|"]
    for _, _, label in SIZE_BUCKETS:
        c, h = tot.get(label, [0, 0])
        if c:
            L.append(f"| {label} | {c} | {h} ({100 * h / c:.0f}%) |")
    L += ["", "## Every faulty run", ""]
    for name, s in summaries.items():
        L.append("```")
        L.append(verdict_lines(name, s))
        L.append("```")
    with open(os.path.join(base, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="smoke")
    p.add_argument("--rate", type=float, default=0.02, help="chance each targeted matmul is corrupted")
    p.add_argument("--rates", type=str, default=None, help="comma-separated sweep, e.g. 0.001,0.01,0.05")
    p.add_argument("--target", default="qk", help="any of q,k,v,mlp,proj, e.g. qk or v or mlp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean-seeds", type=int, default=3, help="number of clean runs (different seeds)")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--guard-every", type=int, default=10, help="run the detector every N steps (0 = off)")
    p.add_argument("--specials", action="store_true", help="also inject NaN/Inf (loud) faults")
    p.add_argument("--fault-seeds", default=None,
                   help="comma-separated seeds for different fault patterns, e.g. 1,2,3 (default: one pattern)")
    p.add_argument("--fault-seed", type=int, default=None, help="with --one faulty: the fault pattern seed")
    p.add_argument("--fault-scale", default="warp",
                   help="warp, tile or mixed (comma-separated for several). warp: a few values per event "
                        "(ALU/tensor-core style). tile: 20-75%% of a 128x128 output tile "
                        "(control-logic style, per NVIDIA). mixed: 30%% tile events")
    p.add_argument("--fault-mix", default="nvidia",
                   help="nvidia, zero or mantissa (comma-separated for several)")
    p.add_argument("--exact-reductions", action="store_true",
                   help="CUDA only: disable reduced-precision bf16/fp16 reductions inside matmuls")
    p.add_argument("--one", choices=["clean", "faulty"], default=None,
                   help="train a single run and save it under runs/<preset>/raw/")
    p.add_argument("--repeat", action="store_true", help="with --one clean: save as clean_s<seed>_repeat")
    p.add_argument("--analyze", action="store_true", help="only rebuild verdicts/plots/report from saved runs")
    p.add_argument("--resume", action="store_true", help="skip runs that are already saved")
    p.add_argument("--out", default="runs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--triage-steps", type=int, default=0,
                   help="after a crash, restart from the newest snapshot and check every step for up to N steps")
    p.add_argument("--snapshot-every", type=int, default=50, help="snapshot interval for --triage-steps")
    a = p.parse_args()

    if a.analyze:
        analyze(a.out, a.preset)
        return
    common = dict(preset=a.preset, guard_every=a.guard_every, specials=a.specials, steps=a.steps,
                  exact_reductions=a.exact_reductions, triage_steps=a.triage_steps, snapshot_every=a.snapshot_every)
    run = lambda spec: run_from_spec(spec, a.data_dir, a.out, skip_existing=a.resume or a.one is not None,  # noqa: E731
                                     device=a.device, dtype=a.dtype)
    if a.one == "clean":
        run({**common, "kind": "clean", "seed": a.seed, "repeat": a.repeat})
        return
    if a.one == "faulty":
        run({**common, "kind": "faulty", "seed": a.seed, "rate": a.rate, "target": a.target,
             "mix": a.fault_mix.split(",")[0], "scale": a.fault_scale.split(",")[0], "fault_seed": a.fault_seed})
        return

    rates = [float(r) for r in a.rates.split(",")] if a.rates else [a.rate]
    scales = [x.strip() for x in a.fault_scale.split(",")]
    mixes = [x.strip() for x in a.fault_mix.split(",")]
    for sc in scales:
        if sc not in ("warp", "tile", "mixed"):
            raise SystemExit(f"unknown --fault-scale {sc}")
    for mx in mixes:
        if mx not in ("nvidia", "zero", "mantissa"):
            raise SystemExit(f"unknown --fault-mix {mx}")
    fault_seeds = [int(x) for x in a.fault_seeds.split(",")] if a.fault_seeds else [None]
    for i in range(max(2, a.clean_seeds)):
        run({**common, "kind": "clean", "seed": a.seed + i})
    for mix in mixes:
        for scale in scales:
            for rate in rates:
                for fseed in fault_seeds:
                    run({**common, "kind": "faulty", "seed": a.seed, "rate": rate, "target": a.target,
                         "mix": mix, "scale": scale, "fault_seed": fseed})
    analyze(a.out, a.preset)


if __name__ == "__main__":
    main()
