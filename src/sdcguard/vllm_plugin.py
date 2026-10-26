"""vLLM plugin: check the math of every bf16/fp16 linear layer, and of INT4/INT8 weight-only quantized ones (GPTQ,
AWQ, compressed-tensors W4A16/W8A16 on vLLM's Marlin and Machete kernels), while vLLM serves.

    pip install sdcguard
    SDCGUARD=1 vllm serve <model>

Off unless SDCGUARD=1 (vLLM loads every installed plugin, so installing sdcguard alone changes
nothing). What it does, in every vLLM process:

  - after vLLM loads a linear layer's weights, computes that layer's checksums once (serve.py); for a quantized
    layer, from the matrix its kernel multiplies by, read before the kernel repacks it and confirmed bit for bit by
    running the kernel on unit vectors (_quant.py);
  - after every matmul of a checked linear layer, queues the check (one custom op, so
    torch.compile keeps a single graph and CUDA graphs capture it);
  - with SDCGUARD_NORMS=1 (off by default for now), after every call of an RMSNorm layer, queues its check
    (serve_norm.py: every output against the norm recomputed in fp32 from its input and its weights at load; one
    opaque custom op, so torch.compile can't merge the recompute with the norm it checks; SDCGUARD_NORM_Z sets its
    line in roundoffs, default 6);
  - with SDCGUARD_MOE=1 (off by default for now), at each model step one fused MoE layer (they take turns)
    recomputes the routed experts' output of 2 rows of its batch, with the rounding of vLLM's Triton MoE kernel,
    and compares (serve_moe.py: wrong experts, rows, scaling and non-finite values; vLLM's Triton MoE backend only),
    and one router (they take turns too) recomputes its routing policy for 2 rows from its logits (serve_route.py:
    the experts chosen and their weights; vLLM's top-k, grouped top-k and biased top-k routers);
  - unless SDCGUARD_ATTN=0 (on by default since its GPU run met its criteria, October 24, 2026, and cost 2.7% at 16
    requests at once on top of a server checking every linear layer; docs/METHOD.md), at each model step one
    attention layer (they take turns)
    recomputes 4 of its outputs (one query head of one token each) from the KV cache through the block table, and
    checks that those tokens' own KV-cache writes went where attention reads them (serve_attn.py: vLLM's
    FlashAttention and Triton backends, decoder attention with a 16-bit KV cache; SDCGUARD_ATTN_Z sets its line,
    default 4);
  - with SDCGUARD_SPOT=1 (off by default until its GPU run), at each model step one of the checked linear layers (they
    take turns) recomputes 8 output columns for up to 32 of the step's real rows exactly (fp32 products, compensated
    fp32 sums) and compares each served output with it, to about twice its own rounding (serve_spot.py): small errors
    the checksum check can't see, in the outputs it samples;
  - with SDCGUARD_WEIGHT_SCAN=1 (off by default until its GPU run), at each read of the results below, compares a
    few layers' weights with their fingerprint at load, in turn, so a changed weight bit is named within a pass over
    the model however small its effect (serve_weights.py; the checked layers, quantized ones included, the bf16
    linear layers SDCGUARD_SHARE leaves out of the arithmetic checks, and every other parameter of the model:
    embeddings, the LM head, norm weights, MoE experts' weights; SDCGUARD_WEIGHT_SCAN_MB, default 64, is how many
    megabytes of weights each read compares; SDCGUARD_WEIGHT_SCAN_AGE, in seconds, raises that as needed so a
    pass over the weights is scheduled to take at most 2/3 of it at the measured time between reads, up to
    SDCGUARD_WEIGHT_SCAN_MAX_MB a read, default 1024: a schedule, not a guarantee; the scan's coverage reports the
    actual oldest verification age and whether it is within the target);
  - once the model has loaded, reads what the server runs (sdcguard.stack, published as sdcguard_stack_info for
    `sdcguard qualify` and `sdcguard ready --qualified`): among it the commit its model revision resolved to and one
    digest of every parameter (the weight scan's fingerprint, one pass over the weights at load, scan on or off);
  - every SDCGUARD_POLL_EVERY model steps (default 16), reads the per-layer results back without
    stalling the GPU and logs an alert for any layer whose output didn't match its checksums (that read
    reports the previous one, so an alert shows up one read later); and after the first model step that ends
    SDCGUARD_POLL_SECONDS (default 2) or more after the last read, reads them right away, waiting for the GPU
    once, so a lightly loaded server reports within one step of that time. There's no background timer (it
    would get in the way of vLLM's CUDA graph capture): a server that goes idle right after a bad step
    reports it at its next step, or when it exits:

    sdcguard ALERT: model.layers.17.mlp.down_proj on GPU-1a2b... (NVIDIA H100 80GB HBM3, PCI ...)
      z=inf (row 3, block 12)

Settings (environment variables): SDCGUARD_Z (alert threshold, default 6), SDCGUARD_LOG (also append
alerts as JSON lines to this file), SDCGUARD_POLL_EVERY, SDCGUARD_SHARE (check only this share of the
linear layers, e.g. 0.125, 1/8 or 12.5%, for a lower cost) and SDCGUARD_OFFSET (which ones: with share 1/R,
replicas with offsets 0 to R-1 check every layer between them, each exactly once),
SDCGUARD_WEBHOOK (post alerts to Slack, Discord or any JSON endpoint; SDCGUARD_WEBHOOK_FORMAT=slack,
discord or json), SDCGUARD_EVIDENCE_DIR (save the numbers that alarmed: when a layer's check fires, the check
itself copies that row of the layer's input and that block of its output before vLLM reuses the memory, also
inside CUDA graphs; each alert carries a float64 recheck of them that says whether the output was really wrong,
and with this variable they are also saved as evidence files for `sdcguard replay` and `sdcguard casefile`),
SDCGUARD_METRICS_PORT (Prometheus metrics; each GPU worker adds its LOCAL_RANK, or its device index) and
SDCGUARD_REQUIRE (refuse to serve when fewer layers are checked than asked: 1 for every
layer SDCGUARD_SHARE selected, with no quantized layers or fused MoE experts left over, or a share such as 0.9
for that share of them; _require says exactly what counts; anything that keeps the plugin from checking or counting,
such as SDCGUARD not 1, a vLLM it doesn't know or no model runner to hook, refuses too, since unknown coverage is
never enough). The first model step logs the count either way,
by reason (quantization method, LM head, not selected, ...) and with the share of the linear-layer weights
checked, so a quantized model that nothing checks says so; the first read of results names any registered
layer whose check never ran.

vLLM caches compiled graphs on disk, keyed by its own settings only, and a graph traced with sdcguard has its
checks (which layers, their slots and z) built in. So the plugin points VLLM_CACHE_ROOT (yours, or vLLM's
default) at a subfolder named after those sdcguard settings, sdcguard-<hash>, in every vLLM process;
SDCGUARD_SHARED_CACHE=1 leaves it alone. (A cache_dir set in vLLM's compilation config is used as it is.)

Updating weights in place while serving (an RL weight sync)? Call sdcguard.vllm_plugin.refresh() in each
worker afterwards; otherwise every updated layer is reported (with weights_changed_since_load = true). With
vLLM's CPU offloading, an offloaded layer's weights can't be read back as its own: its alerts leave
weights_changed_since_load unknown, and refresh() turns its check off and lists it in guard().unsupported.

Quantized layers (SDCGUARD_QUANT=0 leaves them alone): a quantized layer is checked without its bias, which is then
added as its kernel adds it (Marlin's rounded once in the activation dtype; Machete's by vLLM's own code, with the GEMM
and the check as one op in the compiled graph), so the output is unchanged, and a
CPU copy of its quantized weights is kept, about the size of the layer, so an alert can be rechecked
(SDCGUARD_QUANT_RECHECK=0 keeps none: alerts then say they can't be rechecked). refresh() can't recompute a quantized
layer's checksums after its kernel repacked the weights: it turns that layer's check off and says so.

Not checked: FP8 and other quantized layers outside Marlin's and Machete's INT4/INT8 weight-only kernels (and on
Marlin, layers with one scale per output, which it applies after accumulating), fused MoE experts, router gates that
vLLM computes with its own GEMM (GateLinear with a bf16 weight and fp32 output), attention outside SDCGUARD_ATTN's
sampled recompute (and MLA's decode path, whose matmuls use copies of kv_b_proj's weight, so kv_b_proj is checked in
prefill only), the KV cache's contents beyond the sampled tokens' own writes, the LM head, norms
other than RMSNorm and GemmaRMSNorm (each run's log lists them), the residual adds fused into norms.
Tested against vLLM 0.31 (V1 engine); vLLM's internal APIs change, so the plugin checks what it
patches and turns itself off (with a log line) if something is missing.
"""
from __future__ import annotations

import atexit
import bisect
import collections
import hashlib
import itertools
import logging
import math
import os
import re
import threading
import time
from typing import Optional

import torch

log = logging.getLogger("sdcguard")


def _setup_logging() -> None:
    """Make sdcguard's own lines (what it checks, which kernels, what it can't check) show in vLLM's output. vLLM
    configures only its own "vllm" logger, so without a handler of ours Python drops sdcguard's INFO lines and prints
    its warnings bare. Added only when neither sdcguard's logger nor the root logger has a handler (a logging setup of
    the user's is left alone, and then SDCGUARD_LOG_LEVEL doesn't apply). Lines go to the stream vLLM logs to
    (VLLM_LOGGING_STREAM, standard output by default), so they come in order with vLLM's own. SDCGUARD_LOG_LEVEL:
    DEBUG, INFO (the default), WARNING or ERROR."""
    import sys
    root = logging.getLogger()
    if log.handlers or root.handlers:
        return
    h = logging.StreamHandler(sys.stderr if os.environ.get("VLLM_LOGGING_STREAM", "").endswith("stderr")
                              else sys.stdout)
    h.setFormatter(logging.Formatter("%(levelname)s %(asctime)s %(message)s", "%m-%d %H:%M:%S"))
    log.addHandler(h)
    level = os.environ.get("SDCGUARD_LOG_LEVEL", "INFO").upper()
    log.setLevel(level if level in ("DEBUG", "INFO", "WARNING", "ERROR") else "INFO")
    log.propagate = False


_STATE = {"guard": None, "steps": 0, "op": None, "patched": False, "reported": False, "layers": None,
          "rechecked": None, "exit_hooks": None, "flushed": None, "hooked": False, "z": None, "step_log": None,
          "norm_classes": [], "norms_stale": False, "moe_classes": [], "fp8_classes": [], "route_classes": [],
          "attn_classes": [], "attn_why": None, "token_steps": 0, "dummy_runs": 0, "dummy_after": None,
          "in_dummy": 0}

# Why a linear layer isn't checked (layer_report). The quantized methods are reported by their own names.
NOT_SELECTED = "not selected by SDCGUARD_SHARE"
FAILED = "selected, but registering it failed (see the warning logged then)"
NEVER_RAN = "registered, but its check never ran (its forward doesn't call vLLM's linear method)"
ROUTER = "router gate computed by GateLinear's own GEMM"
LM_HEAD = "LM head"
UNSEEN = "unquantized, but its weights were processed without the plugin"
NOT_A_MATRIX = "-D weight (a convolution kept in a linear layer)"
# Quantized layers the plugin can't check (yet): these reasons start with QUANT, and count as quantized layers left
# over (SDCGUARD_REQUIRE). Layers of a quantized method whose kernel isn't hooked at all are reported by the method.
QUANT = "quantized, not checked: "
QUANT_KERNELS = ("MarlinLinearKernel", "MacheteLinearKernel")
# How Machete's checked calls appear in vLLM's traced graph (_graph_key): 3, one op named machete_mm for the GEMM
# and the check, taking the layer's checksums (2: an op named machete_checked taking the check's arguments)
MACHETE_GRAPH = 3

# The expert weights of vLLM's fused MoE layers (RoutedExperts, FusedMoE in older releases): bf16/fp16/fp8 as
# w13_weight/w2_weight, packed INT4 and the like as w13_qweight or w13_weight_packed. No `weight` attribute.
_MOE_WEIGHTS = ("w13_weight", "w13_qweight", "w13_weight_packed", "w2_weight", "w2_qweight", "w2_weight_packed")


class RequirementNotMet(RuntimeError):
    """SDCGUARD_REQUIRE: fewer layers are checked than it asks for, so vLLM doesn't serve."""


def _metrics_port():
    """SDCGUARD_METRICS_PORT plus this worker's LOCAL_RANK when it has one (torchrun, vLLM's external launcher),
    else its CUDA device index: one exporter per GPU worker. vLLM's multiprocessing and Ray workers leave
    CUDA_VISIBLE_DEVICES as it is and each select their GPU by its index on the node (gpu_worker.init_device),
    so the index tells a node's workers apart."""
    base = os.environ.get("SDCGUARD_METRICS_PORT")
    if not base:
        return None
    idx = os.environ.get("LOCAL_RANK")
    if idx is None or not idx.strip().isdigit():
        try:
            idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        except Exception:  # noqa: BLE001
            idx = 0
    return int(base) + int(idx)


def _parse_share(text: str) -> float:
    """A share as 0.125, 1/8 or 12.5%: ValueError unless it's a number from 0 up (NaN and negatives aren't)."""
    t = text.strip()
    try:
        if t.endswith("%"):
            v = float(t[:-1]) / 100
        elif "/" in t:
            a, b = t.split("/", 1)
            v = float(a) / float(b)
        else:
            v = float(t)
    except ZeroDivisionError as e:
        raise ValueError(str(e)) from e
    if not v >= 0:                               # NaN fails this too
        raise ValueError(f"{text!r} isn't a share")
    return min(1.0, v)


def _share(text: str) -> float:
    """SDCGUARD_SHARE: a share of layers, as 0.125, 1/8 or 12.5%. One that can't be read (NaN, negative, text)
    checks every layer, the safe reading, with a warning; 0 checks none (SDCGUARD_REQUIRE then refuses)."""
    try:
        return _parse_share(text)
    except ValueError:
        log.warning("sdcguard: SDCGUARD_SHARE=%r isn't a share (use 0.125, 1/8 or 12.5%%); checking every layer",
                    text)
        return 1.0


def _z() -> float:
    """SDCGUARD_Z, the alert line in noise units (default 6). One that isn't a positive number (NaN would mean no
    alert could ever fire) refuses under SDCGUARD_REQUIRE, and otherwise uses 6 with a warning."""
    text = os.environ.get("SDCGUARD_Z", "6")
    z = _z_from(text)
    if z is not None:
        return z
    _refuse_if_required(f"SDCGUARD_Z={text!r} isn't a positive number; using 6")
    return 6.0


def _z_from(text: str) -> Optional[float]:
    try:
        z = float(text)
    except (TypeError, ValueError):
        return None
    return z if 0 < z < math.inf else None


def _guard(device: Optional[torch.device] = None):
    if _STATE["guard"] is None:
        from .serve import ServeGuard
        port = _metrics_port()
        metrics = None
        if port is not None:                  # its own try: a taken port mustn't cost the webhook too
            try:
                from .alerts import Metrics
                metrics = Metrics(port=port)
                log.info("sdcguard: Prometheus metrics on port %s", port)
            except Exception as e:  # noqa: BLE001  (prometheus-client missing, or the port taken)
                log.warning("sdcguard: Prometheus metrics on port %s failed (%s); alerts still go to the log and "
                            "webhook", port, e)
        kw = dict(z=_STATE.get("z") or _z(), log_path=os.environ.get("SDCGUARD_LOG") or None,
                  on_alert=_on_alert, metrics=metrics, evidence_dir=os.environ.get("SDCGUARD_EVIDENCE_DIR") or None)
        try:
            g = ServeGuard(webhook=os.environ.get("SDCGUARD_WEBHOOK") or None,
                           webhook_format=os.environ.get("SDCGUARD_WEBHOOK_FORMAT", "slack"), **kw)
        except Exception as e:  # noqa: BLE001  (a bad webhook format: keep checking)
            log.warning("sdcguard: webhook setup failed (%s); alerts still go to the log", e)
            g = ServeGuard(**kw)
        _STATE["guard"] = g
        g.attribute = _attribute
        if _weight_scan_on():
            g.weight_scan(_weight_scan_mb(), target_age=_weight_scan_age(), max_mb=_weight_scan_max_mb())
        if metrics is not None:               # who and how, for `sdcguard ready` (checks from outside)
            try:
                from .events import device_identity
                uuid = device_identity(device).get("gpu_uuid") if device is not None and device.type == "cuda" \
                    else None
                metrics.identify(uuid, kw["z"], _share(os.environ.get("SDCGUARD_SHARE", "1")))
            except Exception as e:  # noqa: BLE001
                log.warning("sdcguard: couldn't publish sdcguard_protection_info (%s)", e)
            if _STATE.get("stack"):           # read at load_model, before this guard existed
                try:
                    metrics.stack(_STATE["stack"])
                except Exception as e:  # noqa: BLE001
                    log.warning("sdcguard: couldn't publish sdcguard_stack_info (%s)", e)
        _exit_hooks()
        # NVIDIA's listed cuBLAS issues that apply to this GPU: logged here, with the first layer's weights on the
        # worker's own GPU, not in register(). register() also runs in vLLM's front-end process, where initializing
        # CUDA makes vLLM start its processes with spawn, and in each worker before it has selected its GPU (the
        # current device is still 0 then).
        if device is not None and device.type == "cuda":
            try:
                from . import known_issues
                known_issues.log_once(device=device, logger=log)
            except Exception:  # noqa: BLE001
                pass
    return _STATE["guard"]


def _exit_hooks() -> None:
    """Read the last results when the process ends: from atexit, and from multiprocessing's exit finalizers,
    because a forked multiprocessing child (vLLM's default for its engine and workers) ends with os._exit() and,
    on CPython 3.11 and 3.12, never runs atexit. Once per process: a forked child starts with its parent's atexit
    list but an empty finalizer list."""
    if _STATE.get("exit_hooks") == os.getpid():
        return
    _STATE["exit_hooks"] = os.getpid()
    atexit.register(_flush_at_exit)
    try:
        import multiprocessing.util
        multiprocessing.util.Finalize(None, _flush_at_exit, exitpriority=100)
    except Exception:  # noqa: BLE001  (atexit still covers a plain exit)
        pass


def _weights(mod) -> int:
    """How many weights a linear layer (or a fused MoE layer) multiplies by. Packed quantized weights hold several
    per stored element, so for them the sizes vLLM's layers record are used where they exist."""
    moe = [getattr(mod, a) for a in _MOE_WEIGHTS if isinstance(getattr(mod, a, None), torch.Tensor)]
    if moe:
        e, h, i = (getattr(mod, a, None) for a in ("local_num_experts", "hidden_size",
                                                     "intermediate_size_per_partition"))
        if all(isinstance(v, int) for v in (e, h, i)):
            gated = getattr(getattr(mod, "moe_config", None), "is_act_and_mul", True)
            return e * h * i * (3 if gated else 2)
        return sum(t.numel() for t in moe)
    w = next((getattr(mod, a) for a in ("weight", "weight_packed", "qweight")
              if isinstance(getattr(mod, a, None), torch.Tensor)), None)
    if w is not None and w.is_floating_point():
        return w.numel()
    for a, b in (("input_size_per_partition", "output_size_per_partition"), ("input_size", "output_size")):
        n, m = getattr(mod, a, None), getattr(mod, b, None)
        if isinstance(n, int) and isinstance(m, int):
            return n * m
    return w.numel() if w is not None else 0


def _registered_counts() -> Optional[dict]:
    """guard.counts() (waits for the GPU) once any check has run in this process, else None (nothing known yet)."""
    g = _STATE["guard"]
    if g is None or not g.layers:
        return None
    try:
        c = g.counts()
    except Exception:  # noqa: BLE001
        return None
    return c if any(v["checks"] for v in c.values()) else None


def layer_report(model, counts: Optional[dict] = None) -> dict:
    """Which of a loaded vLLM model's linear layers are checked, and why the others aren't. Logged when serving
    starts (and again at the first read of results when a registered layer's check never ran); exported as
    sdcguard_layers_unchecked{method}, sdcguard_layers_never_checked and sdcguard_weights_checked_ratio.

    counts: checks executed per registered layer name, as ServeGuard.counts() gives them. By default they are read
    from this process's guard (waiting for the GPU) once any check has run (vLLM's warm-up runs the model before
    the first step); a registered layer whose check never ran then isn't counted as checked. Before any check has
    run, a registered layer counts as checked.

    checked: layers checked. selected: layers SDCGUARD_SHARE selected out of the eligible ones (unquantized 2-D
    bf16/fp16 weights on a GPU, computed through vLLM's linear method). cant_check: layers of linear-layer work the
    plugin can't check yet (quantized methods, fused MoE experts, router gates computed by their own GEMM).
    unchecked_by_method: every layer not checked, by reason (also the LM head, other dtypes and the layers not
    selected). weight_share_checked: the checked share of all those layers' weights. moe_checked: fused MoE layers
    the MoE check (SDCGUARD_MOE=1) covers by its sampled recompute: not left over for SDCGUARD_REQUIRE, and their
    weights stay out of weight_share_checked's checked part (a sampled check isn't every output's checksum)."""
    if counts is None:
        counts = _registered_counts()
    g = _STATE["guard"]
    checked = selected = eligible = cant = quant = moe_checked = 0
    unchecked: dict = {}
    never: list = []
    w_all = w_checked = 0
    for mod in model.modules() if model is not None else ():
        qm = getattr(mod, "quant_method", None)
        if qm is None:
            continue
        method = type(qm).__name__
        if any(isinstance(getattr(mod, a, None), torch.Tensor) for a in _MOE_WEIGHTS):
            if getattr(mod, "_sdcguard_moe", None) is not None:      # registered with the MoE check
                moe_checked += 1
                w_all += _weights(mod)
                continue
            why, kind = f"fused MoE experts ({method})", "quant"
        elif not hasattr(mod, "weight") and not hasattr(mod, "weight_packed") and not hasattr(mod, "qweight"):
            continue
        elif "LMHead" in type(mod).__name__:
            why, kind = LM_HEAD, "out"
        elif "Embedding" in method:
            continue                          # the input embedding is a lookup, not a matmul
        elif getattr(mod, "_sdcguard_slot", 0):
            name = getattr(mod, "_sdcguard_name", None) or getattr(mod, "prefix", "") or type(mod).__name__
            if getattr(mod, "_sdcguard_args", None) is None:     # turned off since (refresh to another dtype, say)
                why = (g.unsupported.get(name) if g is not None else None) or "registered, its check turned off"
                kind = "sel"
            elif counts is not None and not (counts.get(name) or {}).get("checks"):
                why, kind = NEVER_RAN, "sel"
                never.append(name)
            else:
                why, kind = None, "checked"
        elif getattr(mod, "_sdcguard_why", None):
            why = mod._sdcguard_why
            kind = "quant" if why.startswith(QUANT) else {FAILED: "sel", ROUTER: "cant",
                                                           NOT_SELECTED: "unselected"}.get(why, "out")
        elif method.lower().startswith("unquantized"):
            why, kind = UNSEEN, "cant"
        else:
            why, kind = method, "quant"       # a quantization method: not checked yet
        size = 0 if why and why.endswith(NOT_A_MATRIX) else _weights(mod)
        w_all += size
        if kind == "checked":
            checked += 1
            selected += 1
            eligible += 1
            w_checked += size
            continue
        unchecked[why] = unchecked.get(why, 0) + 1
        if kind == "sel":
            selected += 1
            eligible += 1
        elif kind == "unselected":
            eligible += 1
        elif kind in ("cant", "quant"):
            cant += 1
            quant += kind == "quant"
    total = checked + sum(unchecked.values())
    return {"checked": checked, "total": total, "share_checked": checked / total if total else 0.0,
            "unchecked_by_method": unchecked, "selected": selected, "eligible": eligible, "cant_check": cant,
            "quantized_or_moe": quant, "never_checked": sorted(never), "moe_checked": moe_checked,
            "weight_share_checked": w_checked / w_all if w_all else 0.0}


def _model_of(runner):
    model = getattr(runner, "model", None)
    if model is None and callable(getattr(runner, "get_model", None)):
        try:
            model = runner.get_model()
        except Exception:  # noqa: BLE001
            model = None
    return model


def _summary(r: dict) -> str:
    msg = (f"sdcguard: checking {r['checked']} of {r['total']} linear layers "
           f"({r['weight_share_checked']:.0%} of their weights)")
    if r.get("moe_checked"):
        msg += f"; {r['moe_checked']} fused MoE layers checked by the MoE check's sampled recompute"
    if r["unchecked_by_method"]:
        msg += "; not checked: " + ", ".join(f"{n} {why}" for why, n in sorted(r["unchecked_by_method"].items()))
    if r["quantized_or_moe"]:
        msg += (" (the arithmetic of fused MoE experts, FP8 layers and quantized layers outside Marlin's and Machete's "
                "INT4/INT8 kernels isn't checked yet)")
    if _STATE.get("fp8_classes"):
        f = _STATE.get("fp8") or {}
        msg += f"; FP8 input quantization checked on {f.get('checked', 0)} layers (SDCGUARD_FP8=1)"
        if f.get("not_checked"):
            msg += f", not on {len(f['not_checked'])}"
        if _STATE.get("fp8_why"):
            msg += f"; FP8 linear layers' quantization not checked: {_STATE['fp8_why']}"
    if _STATE.get("attn_classes"):
        a = _STATE.get("attn") or {}
        if a.get("checked"):
            ok, why = _attn_kernels()
            where = "" if ok else f" possibly outside CUDA graphs only ({why})" if ok is None else \
                f" outside CUDA graphs only (its Triton kernels aren't in use: {why})"
            msg += (f"; attention checked by a sampled recompute on {a['checked']} layers{where} (SDCGUARD_ATTN=0 "
                    "turns it off)")
            if a.get("not_checked"):
                msg += f", not on {len(a['not_checked'])}"
        elif a.get("not_checked"):
            msg += f"; attention not checked on any of its {len(a['not_checked'])} layers"
        else:
            why = _STATE.get("attn_why") or "its layers weren't registered"
            msg += f"; attention not checked: {why}"
    elif _STATE.get("attn_why"):
        msg += f"; attention not checked: {_STATE['attn_why']}"
    ws = getattr(_STATE.get("guard"), "wscan", None)
    if ws is not None:
        try:
            ws.sync()
            other = sum(1 for e in ws.entries if e["kind"] == "other")
            what = f"{len(ws.entries) - other} layers" + (f" and {other} parts of the model's other parameters"
                                                          if other else "")
            per = f"{ws.budget / 2 ** 20:g} MiB a read" + (f", more if needed to verify every weight within "
                                                           f"{ws.target_age:g}s" if ws.target_age else "")
            msg += (f"; weights compared with their load-time fingerprint on {what} in turn "
                    f"({ws.stats['bytes'] / 2 ** 30:.2f} GiB, {per}; SDCGUARD_WEIGHT_SCAN=1)")
            if ws.not_scanned:
                msg += f", not on {len(ws.not_scanned)}"
        except Exception as e:  # noqa: BLE001  (the summary is information)
            msg += f"; weight scan on, but its coverage couldn't be read ({type(e).__name__})"
    if r["never_checked"]:
        msg += "; never ran their check: " + ", ".join(r["never_checked"][:4]) + (
            " ..." if len(r["never_checked"]) > 4 else "")
    return msg


def _require(r: dict, msg: str) -> None:
    """SDCGUARD_REQUIRE=<floor>: raise RequirementNotMet (vLLM then doesn't serve) unless the checked layers are at
    least floor x (the layers SDCGUARD_SHARE selected + that share of the layers the plugin can't check yet:
    quantized layers, fused MoE experts, router gates computed by their own GEMM). So REQUIRE=1 asks for every
    selected layer and refuses any model with such layers left over, and any floor refuses a model with nothing
    the plugin can check (an fp32 model, say). Not counted: the LM head, input embeddings, convolutions kept in
    linear layers, layers of other dtypes or off the GPU, and the layers SDCGUARD_SHARE didn't select."""
    floor = _require_floor()
    if floor is None:
        return
    want = r["selected"] + _share(os.environ.get("SDCGUARD_SHARE", "1")) * r["cant_check"]
    # a model with linear layers and none of them checked meets no requirement, whatever the share selected
    # (SDCGUARD_SHARE=0 selects nothing, so "every selected layer" would be met by none)
    nothing = r["total"] > 0 and (r["eligible"] == 0 or r["checked"] == 0)
    if r["checked"] + 1e-9 < floor * want or nothing:
        raise RequirementNotMet(f"{msg}; SDCGUARD_REQUIRE={os.environ['SDCGUARD_REQUIRE']} asks for more. Unset it to "
                                "serve anyway.")


def _require_floor() -> Optional[float]:
    """SDCGUARD_REQUIRE's floor, or None when nothing is required (unset or 0). A value that can't be read asks for
    every layer (1), the strict reading."""
    need = os.environ.get("SDCGUARD_REQUIRE")
    if not need or not need.strip():
        return None
    try:
        floor = _parse_share(need)
    except ValueError:                           # NaN, negative, text: unreadable, so every layer
        return 1.0
    return floor if floor > 0 else None


def _refuse_if_required(why: str) -> None:
    """Something keeps the plugin from checking or counting the layers. With SDCGUARD_REQUIRE that refuses to serve:
    coverage that can't be shown is never enough. Without it, a warning."""
    if _require_floor() is not None:
        raise RequirementNotMet(f"sdcguard: {why}, so SDCGUARD_REQUIRE={os.environ['SDCGUARD_REQUIRE']} can't be "
                                "shown to be met. Unset it to serve anyway.")
    log.warning("sdcguard: %s", why)


def _require_runner() -> None:
    """At model load, in the process that loads the weights: with SDCGUARD_REQUIRE and no model runner hooked, the
    layers would never be counted and results never read, so it refuses. Not at register(), which also runs in
    processes that never run the model (vLLM's API server). Without SDCGUARD_REQUIRE, register() has warned once."""
    if not _STATE.get("hooked") and _require_floor() is not None:
        _refuse_if_required("vLLM's model runner isn't hooked, so the checked layers can't be counted or results read")


def _report_layers(runner, counts: Optional[dict] = None, again: bool = False) -> Optional[dict]:
    """Count the model's checked layers, log the count, export it and apply SDCGUARD_REQUIRE. again: the recount at
    the first read of results, which only speaks up when a registered layer's check never ran."""
    model = _model_of(runner)
    if model is None:
        if not again:
            _refuse_if_required("couldn't find the model to count its checked layers")
        return None
    r = layer_report(model, counts)
    _STATE["layers"] = r
    g = _STATE["guard"]
    if g is not None:
        g.stats["layers_unchecked"] = sum(r["unchecked_by_method"].values())
        g.stats["layers_never_checked"] = len(r["never_checked"])
        if g.metrics is not None:
            try:
                g.metrics.layer_report(r)
            except Exception:  # noqa: BLE001  (metrics never break serving)
                pass
    if again and not r["never_checked"]:
        return r
    msg = _summary(r)
    if r["total"] and r["checked"] == 0:
        log.warning("%s. Nothing in this model is being checked.", msg)
    elif r["cant_check"] or r["checked"] < r["selected"]:
        log.warning(msg)
    else:
        log.info(msg)
    _require(r, msg)
    return r


def _flush_at_exit() -> None:
    """Results are read every SDCGUARD_POLL_EVERY steps; read whatever the last steps left when the
    worker shuts down, so an alert from the final requests isn't lost, and wait (up to the webhook's own
    timeout) for its post to go out. Once per process (it's both an atexit handler and an exit finalizer)."""
    if _STATE.get("flushed") == os.getpid():
        return
    _STATE["flushed"] = os.getpid()
    try:
        poll(wait=True)
    except Exception:  # noqa: BLE001  (shutting down: the CUDA context may already be gone)
        pass
    hook = getattr(_STATE["guard"], "webhook", None)
    if hook is not None:
        try:
            hook.flush(hook.timeout)
        except Exception:  # noqa: BLE001
            pass


# Which requests an alert concerns. Each model step, the plugin keeps which request held which rows (tokens) of the
# batch, from the scheduler's output and the runner's order of requests: (step, request ids in batch order, the
# end row of each), the last SDCGUARD_STEP_LOG steps (default 4096). A captured row carries its step
# (ServeGuard.mark_step), so the request is named exactly; without a captured row, every request in the steps the
# read covered is named. vLLM's internal ids are the client's id plus "-" and 8 characters: alerts give the client's.
_INTERNAL_SUFFIX = re.compile(r"-[0-9a-f]{8}$")


def _external(rid: str) -> str:
    return _INTERNAL_SUFFIX.sub("", str(rid))


def _batch_rows(runner, args, kwargs) -> Optional[tuple]:
    """(request ids in batch order, the end row of each) of the step the runner just ran, or None when they can't
    be read (then alerts say attribution is unavailable). Rows past the last end are CUDA-graph padding."""
    so = args[0] if args else kwargs.get("scheduler_output")
    counts = getattr(so, "num_scheduled_tokens", None)
    if not isinstance(counts, dict) or not counts:
        return None
    ems = getattr(runner, "execute_model_state", None)                  # the V2 runner: this step's batch
    ids = getattr(getattr(ems, "input_batch", None), "req_ids", None)
    if not ids:
        ids = getattr(getattr(runner, "input_batch", None), "req_ids", None)   # the V1 runner (reused list)
    if not ids:
        return None
    ids = [r for r in ids if r is not None]
    if len(ids) != len(counts) or set(ids) != set(counts):   # the runner's order doesn't cover this step's requests
        return None
    try:
        ends = list(itertools.accumulate(int(counts[r]) for r in ids))
    except (TypeError, ValueError):
        return None
    padded = getattr(getattr(ems, "input_batch", None), "num_tokens_after_padding", None)
    return tuple(ids), ends, int(padded) if isinstance(padded, int) else None


def _ubatching(runner) -> bool:
    """Whether vLLM may split steps into microbatches: dual-batch overlap, or --ubatch-size above 1 (vLLM 0.32's
    ParallelConfig.use_ubatching; it splits a step only under some conditions, such as data parallelism and enough
    tokens)."""
    pc = getattr(runner, "parallel_config", None)
    if pc is None:
        return False
    try:
        return bool(getattr(pc, "use_ubatching", False) or getattr(pc, "enable_dbo", False) or
                    int(getattr(pc, "ubatch_size", 0) or 0) > 1)
    except (TypeError, ValueError):
        return bool(getattr(pc, "enable_dbo", False))


def _rows_split(runner) -> Optional[str]:
    """Why a step's rows don't map to its requests one to one (microbatching, sequence parallelism), or None."""
    if _ubatching(runner):
        return "rows can be split into microbatches (dual-batch overlap or --ubatch-size)"
    pc = getattr(getattr(runner, "compilation_config", None), "pass_config", None)
    if getattr(pc, "enable_sequence_parallelism", False):
        return "rows are split across tensor-parallel ranks (sequence parallelism)"
    par = getattr(runner, "parallel_config", None) or getattr(getattr(runner, "vllm_config", None),
                                                              "parallel_config", None)
    if int(getattr(par, "prefill_context_parallel_size", 1) or 1) > 1:
        return "rows are split across prefill context-parallel ranks"
    spec = getattr(runner, "speculative_config", None) or getattr(getattr(runner, "vllm_config", None),
                                                                  "speculative_config", None)
    if getattr(spec, "enable_adaptive_verification", False):
        return "adaptive verification runs fewer rows than the scheduler counted"
    return None


def _record_batch(runner, step: int, args, kwargs) -> None:
    if _STATE.get("step_log") is None:
        _STATE["step_log"] = collections.deque(maxlen=max(16, int(os.environ.get("SDCGUARD_STEP_LOG", "4096"))))
    split = _rows_split(runner)
    rows = None if split else _batch_rows(runner, args, kwargs)
    _STATE["step_log"].append((step, rows[0], rows[1], rows[2]) if rows else
                              (step, None, split or "the step's batch couldn't be read", None))


def _attribute(ev) -> None:
    """Add to an alert which requests it concerns (ServeGuard.attribute), in three parts kept apart:
      directly_implicated   the request that held the row the check alarmed on, in the step it alarmed (from the
                            captured row): [id], [] for a CUDA-graph padding row, None when it can't be told; for a
                            finding about every row of the step (an FP8 per-tensor scale), every request in it
      potentially_exposed   every request in the steps the read covered ("read_window"), and every request since
                            the layer's last clean read ("since_last_clean_read": a fault that persists, like a
                            changed weight, concerns them all); a conservative window, not a proof of harm
      unknown_impact        what neither covers (state that outlives the window, layers not checked)
    Ids are the client's (vLLM's internal suffix removed); "attribution" says how they were found."""
    d = ev.detail
    log_ = _STATE.get("step_log") or ()
    steps = {e[0]: (e[1], e[2]) for e in log_}
    padded = {e[0]: (e[3] if len(e) > 3 else None) for e in log_}
    cap = d.get("captured") or {}
    st, row = cap.get("step"), cap.get("row")
    exposed: dict = {}
    attn = bool((_STATE.get("attn") or {}).get("checked"))          # layers registered, not just backends patched
    graphs = _attn_kernels()[0] if attn else False     # True: inside CUDA graphs too; None: not known
    where = {True: "", False: " outside CUDA graphs", None: " (perhaps only outside CUDA graphs)"}[graphs]
    also = {True: "", False: " or run inside CUDA graphs", None: " or perhaps run inside CUDA graphs"}[graphs]
    parts = ["linear layers"] + (["RMSNorm layers"] if _STATE.get("norm_classes") else []) + \
        (["sampled rows of MoE experts and their routing"] if _STATE.get("moe_classes") else []) + \
        (["FP8 input quantization"] if _STATE.get("fp8_classes") else []) + \
        ([f"sampled attention outputs{where} and those tokens' KV writes"] if attn else []) + \
        (["sampled outputs of the linear layers against an exact recompute (spot checks)"]
         if getattr(getattr(_STATE.get("guard"), "spot", None), "layers", None) else []) + \
        (["the scanned layers' weights against their load-time fingerprints"]
         if getattr(_STATE.get("guard"), "wscan", None) is not None else [])
    rest = [p for p, on in (("norms", _STATE.get("norm_classes")), ("MoE experts", _STATE.get("moe_classes"))) if not on]
    unchecked = ([f"attention outputs not sampled{also}", *rest, "the rest of the KV cache", "sampling"] if attn else
                 ["attention", *rest, "the KV cache", "sampling"])
    d["unknown_impact"] = [
        "requests that reuse prefix-cache (KV) blocks written during these steps can carry a wrong value further",
        f"the plugin checks {' and '.join(parts) if len(parts) < 3 else ', '.join(parts[:-1]) + ' and ' + parts[-1]}"
        + ("" if len(parts) > 1 else " only") + "; " + ", ".join(unchecked) +
        " and communication between GPUs (all-reduce, all-to-all) aren't checked: a wrong collective result is an "
        "input the checked layers accept"]
    if _share(os.environ.get("SDCGUARD_SHARE", "1")) < 1:
        d["unknown_impact"].append("SDCGUARD_SHARE checks the same subset of layers every step; the others aren't seen")
    d["potentially_exposed"] = exposed
    if not steps:
        d["directly_implicated"], d["attribution"] = None, "unavailable: no model step was recorded"
        return
    always_on = getattr(ev, "detector", None) not in ("serve_attn", "serve_route", "serve_fp8")  # (step-only)
    if st is not None and row is not None and always_on and st in set(_STATE.get("dummy_after") or ()):
        d["directly_implicated"] = None
        d["attribution"] = (f"unavailable: row {row} carries step {st}, but vLLM ran a dummy batch (no request's "
                            f"tokens: warm-up, CUDA-graph capture or a data-parallel rank with nothing scheduled) "
                            f"after step {st}, before the next, and the row may be from it")
    elif st is not None and cap.get("every_row") and st in steps:
        ids, ends = steps[st]
        if ids is None:
            d["directly_implicated"], d["attribution"] = None, f"unavailable for step {st}: {ends}"
        else:
            d["step_rows"] = {"tokens": ends[-1], "with_padding": padded.get(st)}
            d["directly_implicated"] = [_external(i) for i in ids]
            d["attribution"] = (f"exact: every row of step {st} was quantized with the scale the check found wrong, "
                                f"so every request in that step")
    elif st is not None and row is not None and st in steps:
        ids, ends = steps[st]
        if ids is None:
            d["directly_implicated"], d["attribution"] = None, f"unavailable for step {st}: {ends}"
        elif row < ends[-1]:
            d["step_rows"] = {"tokens": ends[-1], "with_padding": padded.get(st)}
            d["directly_implicated"] = [_external(ids[bisect.bisect_right(ends, row)])]
            d["attribution"] = f"exact: row {row} of step {st}, where the check alarmed, belonged to this request"
        else:
            d["step_rows"] = {"tokens": ends[-1], "with_padding": padded.get(st)}
            d["directly_implicated"] = []
            d["attribution"] = f"row {row} of step {st} is CUDA-graph padding (the step had {ends[-1]} tokens)"
    elif d.get("read_window_steps"):
        d["directly_implicated"] = None
        d["attribution"] = "no row was captured for this alert, so only the read's window can be named"
    else:
        d["directly_implicated"], d["attribution"] = None, "unavailable: the alert has no step to look up"
    if d.get("read_window_steps"):
        a, b = d["read_window_steps"]
        named = [steps[s][0] for s in range(a, b + 1) if s in steps and steps[s][0] is not None]
        exposed["read_window"] = {"steps": [a, b], "requests": sorted({_external(r) for ids in named for r in ids}),
                                  "complete": len(named) == b - a + 1}
    clean = d.get("last_clean_read_steps")
    end = (d.get("read_window_steps") or [None, None])[1]
    if clean and end is not None:
        since = sorted({_external(r) for s, (ids, _) in steps.items() if clean[1] < s <= end and ids for r in ids})
        exposed["since_last_clean_read"] = {"steps": [clean[1] + 1, end], "count": len(since),
                                            "requests": since[:1000], "complete": min(steps) <= clean[1] + 1}


def _from(d: dict) -> str:
    """Where an alert's place comes from, for its log line."""
    return "kept and rechecked" if str(d.get("place") or "").startswith("the kept") else "from the score maxima"


def _on_alert(ev) -> None:
    d, i = ev.detail, ev.identity
    if ev.detector == "serve_weights":
        log.warning("sdcguard ALERT: the weights of %s on %s (%s, PCI %s, host %s) no longer match their reference "
                    "fingerprint (%s; %s s since its last clean comparison)", d.get("where"), i.get("gpu_uuid"),
                    i.get("gpu_name"), i.get("pci_bus_id"), i.get("host"), d.get("kind"),
                    d.get("seconds_since_last_clean_comparison"))
        return
    if ev.detector == "serve_norm":
        log.warning("sdcguard ALERT: norm %s on %s (%s, PCI %s, host %s) max z=%s (row %s, column %s, %s: %s, expected "
                    "%s; %s)", d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"), i.get("pci_bus_id"),
                    i.get("host"), d.get("max_z"), d.get("row"), d.get("col"), _from(d), d.get("output"),
                    d.get("expected"), (d.get("recheck") or {}).get("verdict"))
        return
    if ev.detector == "serve_moe" and d.get("kind") == "moe_routing":
        log.warning("sdcguard ALERT: MoE routing into %s on %s (%s, PCI %s, host %s): %s (%s real rows without experts, "
                    "%s missing some; last row %s; %s)", d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"),
                    i.get("pci_bus_id"), i.get("host"), d.get("finding"), d.get("real_rows_without_experts"),
                    d.get("real_rows_missing_some_experts"), d.get("row"), (d.get("recheck") or {}).get("verdict"))
        return
    if ev.detector == "serve_spot":
        log.warning("sdcguard ALERT: spot check of %s on %s (%s, PCI %s, host %s) max z=%s, line %s (row %s, column %s, "
                    "%s: %s, expected %s, relative error %s; %s)", d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"),
                    i.get("pci_bus_id"), i.get("host"), d.get("max_z"), d.get("alarm_line"), d.get("row"), d.get("col"),
                    _from(d), d.get("output"), d.get("expected"), d.get("relative_error"),
                    (d.get("recheck") or {}).get("verdict"))
        return
    if ev.detector == "serve_moe":
        log.warning("sdcguard ALERT: MoE experts %s on %s (%s, PCI %s, host %s) max z=%s (row %s, column %s, %s: %s, "
                    "expected %s; %s)", d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"), i.get("pci_bus_id"),
                    i.get("host"), d.get("max_z"), d.get("row"), d.get("col"), _from(d), d.get("output"),
                    d.get("expected"), (d.get("recheck") or {}).get("verdict"))
        return
    if ev.detector == "serve_fp8":
        log.warning("sdcguard ALERT: FP8 quantization of %s on %s (%s, PCI %s, host %s): %s (%s of %s calls with a "
                    "wrong scale, %s from rows outside the batch; %s sampled rows with wrong values; scale ratio %s)",
                    d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"), i.get("pci_bus_id"), i.get("host"),
                    d.get("finding"), d.get("calls_with_wrong_scale"), d.get("calls_judged"),
                    d.get("calls_scale_from_outside_rows"), d.get("sampled_rows_with_wrong_values"),
                    d.get("scale_ratio"))
        return
    if ev.detector == "serve_attn":
        c = d.get("captured") or {}
        if d.get("finding") == "kv_write_wrong":
            log.warning("sdcguard ALERT: KV-cache write of %s on %s (%s, PCI %s, host %s): %s of %s sampled writes "
                        "with wrong content, %s to another slot than their block names (latest: step %s row %s)",
                        d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"), i.get("pci_bus_id"), i.get("host"),
                        d.get("writes_with_wrong_content"), d.get("writes_checked"), d.get("writes_to_another_slot"),
                        c.get("step"), c.get("row"))
        elif d.get("finding") == "block_table_out_of_range":
            log.warning("sdcguard ALERT: attention %s on %s (%s, PCI %s, host %s): %s sampled requests' block tables "
                        "point outside the KV cache", d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"),
                        i.get("pci_bus_id"), i.get("host"), d.get("samples"))
        else:
            log.warning("sdcguard ALERT: attention %s on %s (%s, PCI %s, host %s) max z=%s (row %s, head %s, dim %s, "
                        "position %s, %s: %s, expected %s; %s)", d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"),
                        i.get("pci_bus_id"), i.get("host"), d.get("max_z"), d.get("row"), d.get("head"),
                        d.get("dim"), d.get("position"), _from(d), d.get("output"), d.get("expected"),
                        (d.get("recheck") or {}).get("verdict"))
        return
    if ev.detector == "serve_route":
        c = d.get("captured") or {}
        log.warning("sdcguard ALERT: MoE routing of %s on %s (%s, PCI %s, host %s): %s (%s of %s sampled rows with "
                    "wrong experts, %s with wrong weights; worst weight difference %s; step %s row %s)",
                    d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"), i.get("pci_bus_id"), i.get("host"),
                    d.get("finding"), d.get("rows_with_wrong_experts"), d.get("rows_judged"),
                    d.get("rows_with_wrong_weights"), d.get("worst_weight_difference"), c.get("step"), c.get("row"))
        return
    log.warning("sdcguard ALERT: %s on %s (%s, PCI %s, host %s) z=%s (row %s, block %s, %s checksum)",
                d.get("where"), i.get("gpu_uuid"), i.get("gpu_name"), i.get("pci_bus_id"), i.get("host"),
                d.get("max_z"), d.get("row"), d.get("col_block"), d.get("checksum"))


def poll(wait: bool = False):
    """Read results back now (also called automatically every SDCGUARD_POLL_EVERY steps)."""
    g = _STATE["guard"]
    return g.poll(wait=wait) if g is not None else []


def guard():
    """The ServeGuard of this process (None until a layer was registered)."""
    return _STATE["guard"]


def refresh(*_args) -> int:
    """Recompute every checked layer's checksums from its current weights. Call this after updating
    weights in place (RL weight sync), e.g. llm.collective_rpc(lambda worker: sdcguard.vllm_plugin.refresh()).
    Returns the number of layers refreshed."""
    g = _STATE["guard"]
    if g is None:
        return 0
    try:
        return g.refresh()
    finally:
        _rearm()


def _rearm() -> None:
    """After weights changed (refresh(), a vLLM weight reload): count the checked layers again at the next model
    step and at the read after it, so SDCGUARD_REQUIRE applies to what is checked now (a refresh can turn a
    layer's check off: quantized layers, weights cast to fp32), not only to what was checked at start-up."""
    _STATE["reported"] = False
    _STATE["rechecked"] = None
    _STATE["norms_stale"] = True          # norm and MoE layers take their current weights before the next batch
    _identity_stale()


def _publish_info(info: dict) -> None:
    from .stack import fingerprint
    info = {k: v for k, v in info.items() if k != "fingerprint" and v not in (None, "")}
    info["fingerprint"] = fingerprint(info)
    _STATE["stack"] = info
    g = _STATE.get("guard")
    if g is not None and g.metrics is not None and hasattr(g.metrics, "stack"):
        g.metrics.stack(info)


def _identity_stale() -> None:
    """The weights changed (or are changing: a reload calls this once per layer): the published stack stops naming
    a weights digest at once, so `sdcguard ready --qualified` can't match the old identity against new weights, and
    the next model step publishes the new one (_take_new_weights)."""
    info = _STATE.get("stack")
    if not info or _STATE.get("identity_stale"):
        return
    _STATE["identity_stale"] = True
    try:
        _publish_info({k: v for k, v in info.items() if k != "weights_digest"})
    except Exception as e:  # noqa: BLE001  (the stack is information; never break a weight update)
        log.warning("sdcguard: couldn't withdraw the weights digest after a weight change (%s: %s)", type(e).__name__, e)


def _identity_fresh() -> None:
    """After a weight change has been taken (outside any CUDA-graph capture): the new weights' digest, published."""
    if not _STATE.get("identity_stale"):
        return
    _STATE["identity_stale"] = False
    runner, info = _STATE.get("stack_runner"), _STATE.get("stack")
    if runner is None or not info:
        return
    try:
        from .serve_weights import model_digest
        try:
            model = runner.get_model()
        except Exception:  # noqa: BLE001
            model = _model_of(runner)
        digest = model_digest(model) if model is not None else None
        if digest:
            _publish_info({**info, "weights_digest": digest})
            log.info("sdcguard: weights changed: the stack now names their new digest %s", digest)
    except Exception as e:  # noqa: BLE001
        log.warning("sdcguard: couldn't take the new weights' digest (%s: %s); the stack names none until the next "
                    "change", type(e).__name__, e)


def _take_new_weights(g) -> None:
    """After weights changed (_rearm): norm and MoE layers (and the weight scan's other parameters) compare with the
    current weights from here on. Called
    before each model step and each dummy batch, so neither is judged against the old weights; never inside a
    CUDA-graph capture (it waits for the next call outside one)."""
    if g is None or not _STATE.get("norms_stale"):
        return
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
    except Exception:  # noqa: BLE001
        pass
    _STATE["norms_stale"] = False
    for part in (g.norms, g.moe, getattr(g, "spot", None)):
        if part is not None:
            try:
                part.refresh()
            except Exception as e:  # noqa: BLE001
                log.warning("sdcguard: couldn't take the new weights of norm or MoE layers: %s", e)
    if getattr(g, "wscan", None) is not None:
        try:
            g.wscan.refresh(kind="other")       # the scan's other parameters: their new values are the reference
        except Exception as e:  # noqa: BLE001
            log.warning("sdcguard: the weight scan couldn't take the new weights of the other parameters: %s", e)
    _identity_fresh()


def _dummy_seen() -> None:
    """vLLM is about to run a dummy batch: one with no request's tokens (memory profiling, compile warm-up, the
    sampler's warm-up, a data-parallel rank with nothing scheduled, and the V1 runner's CUDA-graph captures). The V1
    runner makes these without execute_model, the V2 runner through it (dummy_run=True; its CUDA-graph captures call
    the model directly, outside both). Not a model step: no step is marked or counted
    and nothing is read, so the checks that judge model steps (attention, routing, FP8 quantization) stay off, as the
    last end_step left them. The checks that always run (linear, norm and MoE layers) check it too, and a row they
    capture there carries the last model step's number, so that step is noted: an alert on such a row doesn't name
    the step's requests as the ones it concerns."""
    _STATE["dummy_runs"] = _STATE.get("dummy_runs", 0) + 1
    d = _STATE.get("dummy_after")
    if d is None:
        d = _STATE["dummy_after"] = collections.deque(maxlen=max(16, int(os.environ.get("SDCGUARD_STEP_LOG",
                                                                                          "4096"))))
    if not d or d[-1] != _STATE["steps"]:
        d.append(_STATE["steps"])
    _take_new_weights(_STATE["guard"])


def _note_dummy() -> None:
    try:
        _dummy_seen()
    except Exception as e:  # noqa: BLE001  (bookkeeping only: never break vLLM's warm-up or a dummy batch)
        log.debug("sdcguard: couldn't note a dummy batch: %s", e)


def _exact_reductions() -> None:
    """Keep bf16/fp16 split-K partial sums in fp32 (cuBLAS), as the training guard does on checked
    steps: the noise model assumes one rounding per output. Changes rounding slightly; costs little.
    SDCGUARD_EXACT=0 leaves PyTorch's default. Only process-wide flags are set: nothing here may touch the GPU
    (not even torch.cuda.is_available(), which can initialize CUDA's driver in vLLM's front-end process before it
    forks; PyTorch's docs say it poisons fork unless PYTORCH_NVML_BASED_CUDA_CHECK=1)."""
    if os.environ.get("SDCGUARD_EXACT", "1") == "0" or not (torch.version.cuda or getattr(torch.version, "hip", None)):
        return
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    except (AttributeError, RuntimeError, TypeError) as e:
        log.warning("sdcguard: couldn't keep matmul partial sums in fp32 (%s)", e)


_CACHE_DIR = re.compile(r"^sdcguard-[0-9a-f]{12}$")


def _graph_key(share: float, offset: int, inline: bool) -> str:
    """A hash of what sdcguard builds into vLLM's traced graph: which layers have a check (SDCGUARD_SHARE,
    SDCGUARD_OFFSET, SDCGUARD_ALLOW_CPU), the constants in each check's call (its slot, and z), whether the check
    is traced in or called as the custom op (SDCGUARD_OPAQUE, SDCGUARD_FAST, SDCGUARD_SERVE_FAST), whether quantized
    layers are checked (SDCGUARD_QUANT), the sdcguard version, the check op's arguments (serve.OP_SCHEMA: a graph
    traced with other arguments can't run), whether norms are checked, with what line and op arguments
    (SDCGUARD_NORMS, SDCGUARD_NORM_Z, serve_norm.NORM_SCHEMA), whether linear layers are spot-checked, with what
    line and op arguments (SDCGUARD_SPOT, serve_spot.SPOT_Z and SPOT_SCHEMA: its op is called in the linear layers'
    apply, so it is in the traced graph), and whether FP8 inputs, MoE experts and attention are checked
    (SDCGUARD_FP8: its op is called in the FP8 linear layers' apply, traced too; SDCGUARD_MOE and SDCGUARD_ATTN run
    inside vLLM's own opaque ops on NVIDIA GPUs, which leave them out of the graph, but are traced where vLLM calls
    those layers directly, as on the CPU)."""
    from . import __version__
    from .serve import OP_SCHEMA
    env = os.environ
    z = _z_from(env.get("SDCGUARD_Z", "6")) or 6.0
    norms = spot = None
    if env.get("SDCGUARD_NORMS", "0") == "1":        # (serve_norm imports Triton: only when norms are checked)
        from .serve_norm import NORM_SCHEMA
        norms = (NORM_SCHEMA, _norm_line())
    if _spot_on():                                   # (serve_spot imports Triton: only when spot checks are on)
        from .serve_spot import SPOT_SCHEMA, SPOT_Z
        spot = (SPOT_SCHEMA, SPOT_Z)
    others = (env.get("SDCGUARD_FP8", "0") == "1", env.get("SDCGUARD_MOE", "0") == "1", _attn_on())
    return hashlib.sha256(repr((__version__, OP_SCHEMA, MACHETE_GRAPH, share, offset, z,
                                inline,
                                env.get("SDCGUARD_FAST", "1") != "0", env.get("SDCGUARD_SERVE_FAST", "1") != "0",
                                env.get("SDCGUARD_ALLOW_CPU") == "1", env.get("SDCGUARD_QUANT", "1") != "0", norms,
                                spot, others)).encode()).hexdigest()[:12]


def _cache_root(key: str) -> str:
    """VLLM_CACHE_ROOT for sdcguard's compiled graphs: (the user's VLLM_CACHE_ROOT, or vLLM's default
    $XDG_CACHE_HOME/vllm or ~/.cache/vllm)/sdcguard-<key>. vLLM keeps its compile caches under VLLM_CACHE_ROOT
    (torch_compile_cache: the AOT-compiled graphs, and Inductor's and Triton's caches; envs.py,
    compilation/backends.py and decorators.py in v0.32.0 and on main) and keys them by its own settings, so a graph
    traced without sdcguard, or with other sdcguard settings, would be loaded as it is. Every vLLM process runs
    register() again with the environment its parent set (Ray copies the VLLM_ variables to its workers), so a
    folder that already is an sdcguard one is replaced, not nested."""
    root = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "vllm")
    root = os.path.normpath(os.path.expanduser(root))
    if _CACHE_DIR.match(os.path.basename(root)):
        root = os.path.dirname(root)
    return os.path.join(root, f"sdcguard-{key}")


def _cant_check(layer, w) -> Optional[str]:
    """Why a layer of vLLM's UnquantizedLinearMethod can't be checked, or None."""
    if not isinstance(w, torch.Tensor):
        return "no weight tensor"
    # GateLinear (MoE routers, vllm/model_executor/layers/fused_moe/router/gate_linear.py) with a bf16 weight and
    # fp32 output computes on NVIDIA GPUs with torch.mm(out_dtype=fp32) or ll_bf16_gemm, never through apply()
    if getattr(layer, "allow_cublas_router_gemm", False) or getattr(layer, "allow_ll_bf16_gemm", False):
        return ROUTER
    if w.dim() != 2:                          # Mamba's and ShortConv's conv1d is a ColumnParallelLinear
        return f"{w.dim()}{NOT_A_MATRIX}"
    if w.dtype not in (torch.bfloat16, torch.float16):
        return f"{str(w.dtype).replace('torch.', '')} weights (the check takes bf16 and fp16)"
    if not (w.is_cuda or os.environ.get("SDCGUARD_ALLOW_CPU") == "1"):
        return "weights not on a GPU"
    return None


def _quant_cant_check(kernel, layer) -> Optional[str]:
    """Why a layer run by Marlin or Machete can't be checked, or None. Checked: INT4 and INT8 codes with bf16 or fp16
    activations and output, as GPTQ (uint4b8, uint8b128), AWQ and compressed-tensors (uint4 with zero points) store
    them; on Marlin only with group scales (_quant explains why)."""
    c = kernel.config
    name = type(kernel).__name__
    if c.act_type not in (torch.bfloat16, torch.float16):
        return f"{QUANT}{name} with {str(c.act_type).replace('torch.', '')} activations"
    if c.out_type is not None and c.out_type != c.act_type:
        return f"{QUANT}{name} with {str(c.out_type).replace('torch.', '')} output"
    wt = c.weight_type
    bits, bias = int(getattr(wt, "size_bits", 0)), int(getattr(wt, "bias", 0))
    known = (bits, bias, bool(c.zero_points)) in ((4, 8, False), (8, 128, False), (4, 0, True), (8, 0, True))
    if not known:
        return f"{QUANT}{name} with weight type {wt}"
    if name == "MarlinLinearKernel":
        if c.group_size in (-1, 0):
            return (f"{QUANT}one scale per output on Marlin (applied after accumulating: a second rounding of every "
                    "output)")
        if c.zero_points and bits != 4:
            return f"{QUANT}8-bit codes with zero points on Marlin"
        if os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0") == "1":
            return f"{QUANT}VLLM_MARLIN_USE_ATOMIC_ADD=1 (Marlin then adds partial sums in {c.act_type})"
    if os.environ.get("VLLM_MARLIN_INPUT_DTYPE"):
        return f"{QUANT}VLLM_MARLIN_INPUT_DTYPE (activations quantized inside the kernel)"
    return None


# Machete under torch.compile: its GEMM and the check run as one opaque op (_machete_op), set up by _machete_ready.
_MACHETE = {"op": None, "mm": None, "ops": None, "tags": None, "slots": {}}
_LAYOUT_TAGS = ("needs_exact_strides", "needs_contiguous_strides", "needs_fixed_stride_order", "flexible_layout")
_MACHETE_OP = [None]
_machete_lock = threading.Lock()


def _layout_tags(op) -> tuple:
    """An op's layout tags (how Inductor lays out its inputs; an op without one gets
    torch._functorch.config.custom_op_default_layout_constraint, the same for every op outside aten)."""
    known = [t for t in (getattr(torch._C.Tag, n, None) for n in _LAYOUT_TAGS) if t is not None]
    return tuple(t for t in getattr(getattr(op, "default", op), "tags", ()) if t in known)


def _slot_of(g, checksums: torch.Tensor) -> Optional[int]:
    """The slot of the layer whose checksums (LayerChecksums.parts) this tensor is, or None. By its address, kept in
    a table rebuilt when it misses (a layer registered later, or checksums recomputed by a weight reload)."""
    key = checksums.data_ptr()
    slots = _MACHETE["slots"]
    s = slots.get(key)
    if s is None or s >= len(g.layers) or g.layers[s]["cs"].parts.data_ptr() != key:
        slots.clear()
        slots.update({lay["cs"].parts.data_ptr(): i for i, lay in enumerate(g.layers)})
        s = slots.get(key)
    return s


def _machete_op(tags: tuple = ()):
    """sdcguard::machete_mm: Machete's GEMM (torch.ops._C.machete_mm, called as vLLM's Python wrapper calls it) and
    then the check of its output, as one opaque op returning that output. Under torch.compile vLLM's graph then has
    one opaque call where it had machete_mm, with the same inputs and output plus one tensor of the layer's
    checksums, and nothing else added for Inductor to fuse around.
    That tensor says which layer it is: the op finds the layer's slot from it, and the rest of what the check takes
    and writes from the guard (ServeGuard.op_args), when it runs, as vLLM's attention ops find the KV cache through
    vLLM's forward context. So no constant in vLLM's graph differs between layers, and Inductor compiles each piece
    of it (between attention ops) once for all the layers, as it does without sdcguard; and the op takes no tensor it
    writes, so it is declared as mutating nothing, which is true of its arguments. (Declared as mutating buffers
    passed in, it compiled to other code: on the CPU with PyTorch 2.8, and on an H100 with 2.13, where it changed
    completions.) A layer whose check refresh() turned off isn't checked here either.
    Its name is machete_mm too (in sdcguard's namespace): Inductor names the kernels it generates after the ops in
    them, namespace left out (triton_red_fused_fused_add_rms_norm_machete_mm_0), writes the name into the code it
    hashes, and Triton's autotuning results are cached by that hash. Under another name the kernels around it got
    vLLM's code but other names, and couldn't reuse the launch configs vLLM's own run picked.
    Its layout tags are machete_mm's (_layout_tags), so Inductor lays out its inputs as it does machete_mm's.
    A check as an op of its own after Machete (the plugin until October 23, 2026) changed 2 to 3 of 64 greedy
    completions on an H100; experiments/machete_diag.py and docs/METHOD.md have that and what came after it.
    tests/test_quant.py compares Inductor's code for this op with vLLM's own on the CPU. Registered once per
    process, with the tags of the first call. None on a PyTorch without torch.library.custom_op."""
    with _machete_lock:
        if _MACHETE_OP[0] is None:
            from .serve import check_static_any
            try:
                @torch.library.custom_op("sdcguard::machete_mm", mutates_args=(), tags=tuple(tags))
                def machete_mm(a: torch.Tensor, b_q: torch.Tensor, b_type_id: int,
                               b_group_scales: Optional[torch.Tensor], b_group_zeros: Optional[torch.Tensor],
                               b_group_size: Optional[int], checksums: torch.Tensor) -> torch.Tensor:
                    mm, g = _MACHETE["mm"], _STATE["guard"]
                    if mm is None or g is None:
                        raise RuntimeError("sdcguard::machete_mm called before Machete was set up")
                    # vllm._custom_ops.machete_mm(a, b_q, b_type, b_group_zeros=..., b_group_scales=...,
                    # b_group_size=...): out_type, channel and token scales and schedule left at None
                    y = mm(a, b_q, b_type_id, None, b_group_scales, b_group_zeros, b_group_size, None, None, None)
                    slot = _slot_of(g, checksums)
                    if slot is not None and getattr(g.layers[slot]["module"], "_sdcguard_args", None) is not None:
                        check_static_any(a, y, *g.op_args(slot, with_bias=False))
                    return y

                @machete_mm.register_fake
                def _(a, b_q, b_type_id, b_group_scales, b_group_zeros, b_group_size, checksums):
                    return torch.empty((a.size(0), b_q.size(1)), device=a.device, dtype=a.dtype)   # as vLLM's fake

                _MACHETE_OP[0] = torch.ops.sdcguard.machete_mm
            except (AttributeError, RuntimeError):      # very old PyTorch, or already registered
                _MACHETE_OP[0] = getattr(getattr(torch.ops, "sdcguard", None), "machete_mm", None) or False
        return _MACHETE_OP[0] or None


def _machete_ready(orig_apply) -> Optional[str]:
    """Set up the checked Machete call, or say why Machete's layers can't be checked. The plugin's Machete
    apply_weights is a copy of vLLM's with its machete_mm call replaced, and the op calls torch.ops._C.machete_mm as
    vLLM's wrapper does: both are right only for the code they copy, so that code must be what _quant's
    fingerprints say (vLLM 0.31.0 and 0.32.0)."""
    from . import _quant
    try:
        import importlib
        ops = importlib.import_module("vllm._custom_ops")
        fa, fm = _quant.code_fingerprint(orig_apply), _quant.code_fingerprint(ops.machete_mm)
        if fa not in _quant.MACHETE_APPLY_KNOWN or fm not in _quant.MACHETE_MM_KNOWN:
            log.warning("sdcguard: not checking Machete's layers: this vLLM's MacheteLinearKernel.apply_weights or "
                        "_custom_ops.machete_mm isn't the code sdcguard's check copies (fingerprints %s and %s)",
                        fa, fm)
            return f"{QUANT}MacheteLinearKernel code other than vLLM 0.31.0's and 0.32.0's, which the check copies"
        mm = ops.machete_mm.__globals__["torch"].ops._C.machete_mm   # what the wrapper calls
        tags = _layout_tags(mm)
        op = _machete_op(tags)
    except Exception as e:  # noqa: BLE001  (never break model loading)
        log.warning("sdcguard: not checking Machete's layers: %s: %s", type(e).__name__, e)
        return f"{QUANT}MacheteLinearKernel: setting up its check failed ({type(e).__name__})"
    if op is None:
        return f"{QUANT}MacheteLinearKernel: this PyTorch can't register sdcguard's Machete op"
    _MACHETE.update(op=op, mm=mm, ops=ops, tags=tags)
    return None


def _patch_quant(op, select, inline: bool) -> list:
    """Check the layers vLLM runs with Marlin's and Machete's weight-only INT kernels (GPTQ, AWQ, compressed-tensors
    W4A16 and W8A16): before the kernel repacks a layer's weights, read the matrix it will multiply by (_quant), let
    it repack, run it on unit vectors to confirm that reading bit for bit (for Machete with zero points, whose
    rounding the compiler decides, with both readings _quant knows), then register the layer.
    Marlin adds the bias inside its kernel, rounded to the activation dtype: every call runs the kernel without the
    bias, checks that output (the check op), and adds the bias by one opaque call (_quant.add_bias_op, so
    torch.compile can't fuse it into the next operations and keep it unrounded).
    Machete adds it after its GEMM with add_, which torch.compile fuses into what follows unrounded (qkv's rotary
    embedding then sees q and k plus bias in fp32). Its calls run vLLM's own apply_weights code with the GEMM and
    the check as one opaque op (_machete_op), then add_(bias) as vLLM does, so the compiled graph is vLLM's with one
    op swapped; eagerly, vLLM's machete_mm, the check, then add_ (the check runs on the same stream first).
    select(layer) -> True to check it (SDCGUARD_SHARE), else sets the layer's reason. Returns the kernel classes
    patched."""
    import importlib
    from . import _quant
    from .serve import check_static_any, check_static_impl
    keep = os.environ.get("SDCGUARD_QUANT_RECHECK", "1") != "0"
    add_bias = _quant.add_bias_op()
    patched = []
    for mod_name, cls_name, rule in (
            ("vllm.model_executor.kernels.linear.mixed_precision.marlin", "MarlinLinearKernel", _quant.MARLIN),
            ("vllm.model_executor.kernels.linear.mixed_precision.machete", "MacheteLinearKernel", _quant.MACHETE)):
        try:
            K = getattr(importlib.import_module(mod_name), cls_name)
            orig_pwal, orig_apply = K.process_weights_after_loading, K.apply_weights
        except Exception as e:  # noqa: BLE001  (a vLLM without this kernel: its layers are reported as quantized)
            log.info("sdcguard: %s not found (%s); its layers aren't checked", cls_name, e)
            continue
        if getattr(K, "_sdcguard", False):
            patched.append(cls_name)
            continue
        kernel_why = _machete_ready(orig_apply) if rule == _quant.MACHETE else None

        def process_weights_after_loading(self, layer, _orig=orig_pwal, _apply=orig_apply, _rule=rule,
                                          _kernel_why=kernel_why):
            _require_runner()
            w = bias = qw = None
            again = bool(getattr(layer, "_sdcguard_slot", 0))
            if again:                     # a weight reload: count again, SDCGUARD_REQUIRE on what's checked now
                _rearm()
            try:
                why = _quant_cant_check(self, layer) or _kernel_why
                if why is None and not again and not select(layer):
                    why = layer._sdcguard_why
                    _scan_skips(layer, "a quantized layer SDCGUARD_SHARE leaves out (not scanned yet)")
                if why is None:
                    c = self.config
                    # vLLM's layout of the standard (GPTQ-like) weights, before this kernel changes it
                    qw = _quant.from_layer(layer, self.w_q_name, self.w_s_name, self.w_zp_name if c.zero_points
                                           else None, c.weight_type, c.group_size, _rule, c.act_type)
                    w = qw.weight()                                   # (n, k) on the layer's device
                    b = getattr(layer, "bias", None)
                    bias = b.detach().clone() if isinstance(b, torch.Tensor) else None   # Marlin permutes its own
                    if not (w.is_cuda or os.environ.get("SDCGUARD_ALLOW_CPU") == "1"):
                        why, w = "weights not on a GPU", None
                if why is not None and not again:
                    layer._sdcguard_why = why
            except Exception as e:  # noqa: BLE001  (never break model loading)
                log.warning("sdcguard: not checking %s: %s", getattr(layer, "prefix", "a layer"), e)
                w = None
                if not again:
                    layer._sdcguard_why = FAILED
            _orig(self, layer)
            if w is None:
                if again:                                         # reloaded into something it can't check now
                    layer._sdcguard_args = None
                return
            try:
                name = getattr(layer, "prefix", "") or type(layer).__name__
                rows = _quant.probe_rows(qw.k, qw.group)
                pr = _quant.probe(lambda x: _apply(self, layer, x, None), w, rows)
                if pr["mismatches"] and qw.rounding_varies:
                    # Machete's zero-point rounding is the compiler's choice (_quant): try the other reading
                    alt = qw.other_rounding()
                    w2 = alt.weight()
                    pr2 = _quant.probe(lambda x: _apply(self, layer, x, None), w2, rows)
                    if not pr2["mismatches"]:
                        qw, w, pr = alt, w2, pr2
                    del w2
                if pr["mismatches"]:
                    log.warning("sdcguard: not checking %s: %s on unit vectors differed from the weights read before "
                                "its repack in %d of %d values (first at input %s, output %s); either this reading "
                                "of its weights is wrong for this vLLM, or the GPU computed it wrong",
                                name, type(self).__name__, pr["mismatches"], pr["values"], *pr["first"])
                    if not again:
                        layer._sdcguard_why = (f"{QUANT}{type(self).__name__} on unit vectors didn't match the weights "
                                               "read before its repack")
                    layer._sdcguard_args = None
                    return
                names = [self.w_q_name, self.w_s_name] + ([self.w_zp_name] if self.w_zp_name else []) + ["bias"]
                label = f"{type(self).__name__}, {self.config.weight_type}"
                if qw.describe():
                    label += f", {qw.describe()}"
                src = _quant.QuantSource(layer, names, qw.to("cpu") if keep else None, label)
                layer._sdcguard_qbias = bias
                g = _guard(w.device)
                if again:
                    g.refresh_layer(layer._sdcguard_slot - 1, weight=w, source=src)
                    _rearm()
                    return
                layer._sdcguard_why = FAILED
                slot = g.add_layer(layer, name, weight=w, source=src)
                layer._sdcguard_args = (g.op_args(slot, with_bias=False), g.op_args(slot, with_bias=False))
                layer._sdcguard_trace = inline and not g.uses_fast(slot)
                del layer._sdcguard_why
            except Exception as e:  # noqa: BLE001  (never break model loading)
                log.warning("sdcguard: not checking %s: %s", getattr(layer, "prefix", "a layer"), e)
                layer._sdcguard_args = None
                if not again:
                    layer._sdcguard_why = FAILED

        def apply_weights(self, layer, x, bias=None, _orig=orig_apply):
            """Marlin: the kernel without the bias, the check, then the bias rounded once (as Marlin adds it)."""
            a = getattr(layer, "_sdcguard_args", None)
            if a is None:
                return _orig(self, layer, x, bias)
            y = _orig(self, layer, x, None)
            if torch.compiler.is_compiling():
                fn = check_static_impl if getattr(layer, "_sdcguard_trace", False) else op
            else:
                fn = check_static_any
            fn(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), *a[0])
            if bias is not None:
                y = add_bias(y, layer._sdcguard_qbias)            # Marlin's bias (its own copy is permuted)
            return y

        def machete_apply_weights(self, layer, x, bias=None, _orig=orig_apply):
            """vLLM 0.31's MacheteLinearKernel.apply_weights (Apache-2.0; _machete_ready makes sure it's the code
            running), its machete_mm call replaced by sdcguard::machete_mm under torch.compile, and by
            machete_mm followed by the check eagerly. Always the op under torch.compile, even where the PyTorch
            check would otherwise be traced in (SDCGUARD_OPAQUE=0): that would add its operations to vLLM's graph."""
            a = getattr(layer, "_sdcguard_args", None)
            if a is None:
                return _orig(self, layer, x, bias)
            c = self.config
            w_q, w_s, w_zp = self._get_weight_params(layer)
            x_2d = x.reshape(-1, x.shape[-1])
            out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)
            if c.zero_points:
                assert w_zp is not None
            else:
                w_zp = None
            if torch.compiler.is_compiling():
                # the layer's checksums (op_args' first tensor) tell the op which layer this is
                output = _MACHETE["op"](x_2d, w_q, c.weight_type.id, w_s, w_zp, c.group_size, a[0][0])
            else:
                output = _MACHETE["ops"].machete_mm(a=x_2d, b_q=w_q, b_type=c.weight_type, b_group_zeros=w_zp,
                                                    b_group_scales=w_s, b_group_size=c.group_size)
                check_static_any(x_2d, output, *a[0])
            if bias is not None:
                output.add_(bias)  # In-place add
            return output.reshape(out_shape)

        K.process_weights_after_loading = process_weights_after_loading
        K.apply_weights = machete_apply_weights if rule == _quant.MACHETE else apply_weights
        K._sdcguard = True
        patched.append(cls_name)
    return patched


def _norm_line() -> float:
    """SDCGUARD_NORM_Z, the norm check's line in roundoffs of the output dtype (default 6). A correct norm can be 4
    off, so a line below 4.5 (or one that isn't a number) uses 6, with a warning."""
    from .serve_norm import MIN_LINE, NORM_Z
    text = os.environ.get("SDCGUARD_NORM_Z", "")
    if not text:
        return NORM_Z
    v = _z_from(text)
    if v is None or v < MIN_LINE:
        log.warning("sdcguard: SDCGUARD_NORM_Z=%r isn't a line of at least %s roundoffs; using %s", text, MIN_LINE,
                    NORM_Z)
        return NORM_Z
    return v


def _patch_norms() -> list:
    """vLLM's RMSNorm and GemmaRMSNorm: after the norm, the check of a registered layer (serve_norm.check_call). At
    register(), before any model is built: vLLM binds each norm's forward when it constructs the layer, and its
    forward_cuda calls forward_native, so patching forward_native covers both. Returns the classes patched."""
    try:
        from vllm.model_executor.layers import layernorm as LN
    except Exception as e:  # noqa: BLE001
        log.warning("sdcguard: vLLM's norm layers not found (%s: %s); norms aren't checked", type(e).__name__, e)
        return []
    from .serve_norm import check_call
    done = []
    for cname in ("RMSNorm", "GemmaRMSNorm"):
        cls = getattr(LN, cname, None)
        orig = vars(cls).get("forward_native") if cls is not None else None
        if orig is None:
            continue
        if getattr(cls, "_sdcguard_norm_orig", None) is not None:
            done.append(cname)
            continue

        def forward_native(self, x, residual=None, _orig=orig):
            out = _orig(self, x, residual)
            a = getattr(self, "_sdcguard_norm", None)
            if a is not None:
                if torch.compiler.is_compiling():
                    check_call(a, x, residual, out)
                else:
                    try:
                        check_call(a, x, residual, out)
                    except Exception as e:  # noqa: BLE001  (never break serving: this layer goes unchecked)
                        _norm_failed(self, e)
            return out

        forward_native.__doc__ = orig.__doc__
        cls._sdcguard_norm_orig = orig
        cls.forward_native = forward_native
        done.append(cname)
    return done


def _norm_failed(mod, e: Exception) -> None:
    mod.__dict__.pop("_sdcguard_norm", None)
    g = _STATE["guard"]
    if g is not None:
        g.stats["norm_check_errors"] = g.stats.get("norm_check_errors", 0) + 1
    log.warning("sdcguard: a norm check failed (%s: %s); that layer isn't checked from now on", type(e).__name__, e)


def _attn_line() -> float:
    """SDCGUARD_ATTN_Z, the attention check's line in multiples of its rounding bound (default 4). A correct output
    scores at most 1, so a line below 1.5 (or one that isn't a number) uses 4, with a warning."""
    from .serve_attn import ATTN_Z, MIN_LINE
    text = os.environ.get("SDCGUARD_ATTN_Z", "")
    if not text:
        return ATTN_Z
    v = _z_from(text)
    if v is None or v < MIN_LINE:
        log.warning("sdcguard: SDCGUARD_ATTN_Z=%r isn't a line of at least %s; using %s", text, MIN_LINE, ATTN_Z)
        return ATTN_Z
    return v


ATTN_BACKENDS = (("vllm.v1.attention.backends.flash_attn", "FlashAttentionImpl"),
                 ("vllm.v1.attention.backends.triton_attn", "TritonAttentionImpl"))


def _attn_on() -> bool:
    """SDCGUARD_ATTN: on unless 0 (or false, off, no); an unknown value keeps it on, with a warning."""
    v = os.environ.get("SDCGUARD_ATTN", "1").strip().lower()
    if v in ("0", "false", "off", "no"):
        return False
    if v not in ("1", "true", "on", "yes", ""):
        log.warning("sdcguard: SDCGUARD_ATTN=%r isn't 0 or 1; attention checks stay on (SDCGUARD_ATTN=0 turns them "
                    "off)", os.environ.get("SDCGUARD_ATTN"))
    return True


def _weight_scan_on() -> bool:
    return os.environ.get("SDCGUARD_WEIGHT_SCAN", "0").strip().lower() in ("1", "true", "on", "yes")


def _weight_scan_mb() -> float:
    """SDCGUARD_WEIGHT_SCAN_MB: megabytes of weights each read compares (default 64; at least 1)."""
    from .serve_weights import DEFAULT_BUDGET_MB
    text = os.environ.get("SDCGUARD_WEIGHT_SCAN_MB", "")
    if not text.strip():
        return DEFAULT_BUDGET_MB
    try:
        v = float(text)
        if v >= 1:
            return v
    except ValueError:
        pass
    log.warning("sdcguard: SDCGUARD_WEIGHT_SCAN_MB=%r isn't a number of megabytes of at least 1; using %g", text,
                DEFAULT_BUDGET_MB)
    return DEFAULT_BUDGET_MB


def _positive_env(name: str, default: Optional[float], least: float, what: str) -> Optional[float]:
    text = os.environ.get(name, "")
    if not text.strip():
        return default
    try:
        v = float(text)
        if v >= least:
            return v
    except ValueError:
        pass
    log.warning("sdcguard: %s=%r isn't %s of at least %g; using %s", name, text, what, least, default)
    return default


def _weight_scan_age() -> Optional[float]:
    """SDCGUARD_WEIGHT_SCAN_AGE: seconds; the scan compares enough each read that a pass is scheduled to verify every
    weight at least that often (a schedule from the measured time between reads, not a guarantee: the coverage
    report gives the actual oldest verification age; unset: the byte budget alone)."""
    return _positive_env("SDCGUARD_WEIGHT_SCAN_AGE", None, 1.0, "a number of seconds")


def _weight_scan_max_mb() -> float:
    """SDCGUARD_WEIGHT_SCAN_MAX_MB: the most a read compares to meet SDCGUARD_WEIGHT_SCAN_AGE (default 1024)."""
    from .serve_weights import DEFAULT_MAX_MB
    return _positive_env("SDCGUARD_WEIGHT_SCAN_MAX_MB", DEFAULT_MAX_MB, 1.0, "a number of megabytes")


SET_WHILE_SERVING = ("k_scale", "v_scale", "q_scale", "prob_scale")   # attention scales vLLM may set at first use


def _param_tensors(name: str, p):
    """A parameter's tensors (serve_weights.param_tensors, which the weights' digest at load uses too)."""
    from .serve_weights import param_tensors
    return param_tensors(name, p)


def _module_tensors(mod) -> list:
    """(name, tensor) for a module's own parameters (not its children's), read now."""
    out = []
    for pname, p in mod.named_parameters(recurse=False):
        for tname, t in _param_tensors(pname, p):
            if isinstance(t, torch.Tensor) and t.numel():
                out.append((tname, t))
    return out


def _register_scan_rest(runner) -> Optional[dict]:
    """With SDCGUARD_WEIGHT_SCAN=1, after the model loaded: every parameter no scan entry holds yet (embeddings, the
    LM head, norm weights, MoE experts' weights, linear layers the arithmetic checks can't take or SDCGUARD_SHARE
    leaves out when quantized) is scanned too, as kind "other": one entry per module, or per tensor when the module
    is larger than a read's budget, and per slice of at most that budget when a contiguous tensor is (so no read
    takes a whole MoE layer at once). A tensor two modules hold (tied embeddings) is scanned once. Not scanned:
    attention scales vLLM may set at first use, tensors outside the GPU (offloaded), memory that distinct tensors
    share (vLLM's prefetch offloading), layers the checks found unreadable as their own, and MoE experts' weights
    when vLLM's expert load balancer moves them (--enable-eplb)."""
    if not _weight_scan_on() or _STATE.get("scan_rest_done"):
        return None
    g = _STATE.get("guard")
    ws = getattr(g, "wscan", None) if g is not None else None
    if ws is None:
        return None
    _STATE["scan_rest_done"] = True
    try:
        model = runner.get_model()
    except Exception:  # noqa: BLE001
        model = _model_of(runner)
    if model is None or not hasattr(model, "named_modules"):
        return None
    pc = getattr(runner, "parallel_config", None) or getattr(getattr(runner, "vllm_config", None),
                                                               "parallel_config", None)
    eplb = bool(getattr(pc, "enable_eplb", False))     # vLLM's expert load balancer moves experts' weights
    ws.sync()
    held = set()
    for e in ws.entries:
        for t in e["get"]():
            if isinstance(t, torch.Tensor) and t.numel():
                held.add((str(t.device), t.data_ptr()))
    # memory that more than one distinct tensor points at (vLLM's prefetch offloader points layers N, N + its step,
    # ... at one GPU buffer and copies each layer's weights in before it runs; a tied tensor is one object): not
    # scanned, since what is there belongs to whichever layer ran last
    owners: dict = {}
    for _, mod in model.named_modules():
        for _, t in _module_tensors(mod):
            owners.setdefault((str(t.device), t.data_ptr()), set()).add(id(t))
    shared = {k for k, ids in owners.items() if len(ids) > 1}
    # layers the checks' registration found unreadable as their own, or sharing memory, stay out
    blocked = {n for n, why in ws.not_scanned.items() if "can't be read back" in why or "shares its memory" in why}
    added, skipped = 0, 0
    for mname, mod in model.named_modules():
        if mname in blocked or getattr(mod, "prefix", None) in blocked:
            continue
        mine = []
        for tname, t in _module_tensors(mod):
            full = f"{mname}.{tname}" if mname else tname
            key = (str(t.device), t.data_ptr())
            if key in shared:
                ws.not_scanned[full] = "shares its memory with another layer's tensors (vLLM's prefetch offloading)"
                skipped += 1
                continue
            if key in held:
                continue
            held.add(key)
            if tname.split(".")[-1] in SET_WHILE_SERVING:
                ws.not_scanned[full] = "an attention scale vLLM may set at its first use"
                skipped += 1
                continue
            if eplb and type(mod).__name__ in ("RoutedExperts", "FusedMoE"):
                ws.not_scanned[full] = "an MoE expert's weights, which vLLM's expert load balancer (--enable-eplb) " \
                                       "moves while serving"
                skipped += 1
                continue
            if t.device.type != "cuda" and torch.cuda.is_available():
                ws.not_scanned[full] = f"on the {t.device.type}, not the GPU (offloaded)"
                skipped += 1
                continue
            mine.append((tname, t))
        if not mine:
            continue
        for alias in {mname, getattr(mod, "prefix", None)}:
            if alias:
                ws.not_scanned.pop(alias, None)        # (a layer listed as not scanned is scanned now)

        def current(m=mod, n=None):
            return dict(_module_tensors(m)).get(n)

        size = sum(t.numel() * t.element_size() for _, t in mine)
        base = mname or type(mod).__name__
        if size <= ws.budget:
            names = tuple(n for n, _ in mine)
            if ws.add(base, (lambda m=mod, ns=names: [dict(_module_tensors(m)).get(n) for n in ns]), "other",
                      mod) is not None:
                added += 1
            continue
        for tname, t in mine:
            nbytes = t.numel() * t.element_size()
            if nbytes <= ws.budget or not t.is_contiguous():
                if ws.add(f"{base}.{tname}", (lambda c=current, n=tname: [c(n=n)]), "other", mod) is not None:
                    added += 1
                continue
            step = max(1, ws.budget // t.element_size())
            for a in range(0, t.numel(), step):
                b = min(a + step, t.numel())

                def piece(c=current, n=tname, lo=a, hi=b):
                    x = c(n=n)
                    return [x.view(-1)[lo:hi] if isinstance(x, torch.Tensor) and x.is_contiguous() and
                            x.numel() >= hi else x]

                if ws.add(f"{base}.{tname}[{a}:{b}]", piece, "other", mod) is not None:
                    added += 1
    log.info("sdcguard: the weight scan also compares %d entries of the model's other parameters (%d tensors not "
             "scanned, listed in the coverage report); %.1f GB in all", added, skipped, ws.stats["bytes"] / 1e9)
    return {"other_entries": added, "not_scanned": skipped}


def _scan_skips(layer, why: str) -> None:
    """With SDCGUARD_WEIGHT_SCAN=1, list a linear layer the scan doesn't take, with why (so the coverage it reports
    isn't read as every layer)."""
    g = _STATE.get("guard")
    ws = getattr(g, "wscan", None) if g is not None else None
    if ws is not None and _weight_scan_on():
        ws.not_scanned[getattr(layer, "prefix", "") or type(layer).__name__] = why


def _scan_unchecked(layer, w) -> None:
    """A linear layer SDCGUARD_SHARE left out of the arithmetic checks: with SDCGUARD_WEIGHT_SCAN=1 its weights are
    still compared with their load-time fingerprint. A reload of the layer takes a new fingerprint."""
    if not _weight_scan_on() or not isinstance(w, torch.Tensor):
        return
    try:
        g = _guard(w.device)
        if g.wscan is None:
            return
        if getattr(layer, "_sdcguard_scanned", False):
            g.wscan.refresh(layer)
            return
        if g.scan_weights_of(layer, getattr(layer, "prefix", "") or type(layer).__name__) is not None:
            layer._sdcguard_scanned = True
    except Exception as e:  # noqa: BLE001  (never break model loading)
        log.warning("sdcguard: not scanning the weights of %s: %s", getattr(layer, "prefix", "a layer"), e)


def _attn_kernels() -> tuple:
    """(ok, why): whether the attention check's Triton kernels are in use on every device it checks, so it runs inside
    CUDA graphs too (True). Without them (they failed their first-use comparison or a launch, or this isn't a CUDA GPU)
    the PyTorch check runs outside CUDA graphs only, and a compiled server's decode steps go unchecked (False). None
    when that can't be read."""
    g = _STATE.get("guard")
    ac = getattr(g, "attn", None) if g is not None else None
    try:
        status = ac.kernel_status() if ac is not None and getattr(ac, "layers", None) else \
            (_STATE.get("attn") or {}).get("kernel")
    except Exception as e:  # noqa: BLE001  (a report never breaks serving)
        return None, f"its kernel status couldn't be read ({type(e).__name__})"
    if not status:
        return None, "no kernel status"
    bad = [f"{k}: {v[1]}" for k, v in status.items() if not v[0]]
    return (not bad), "; ".join(bad)


def _patch_attn() -> list:
    """vLLM's FlashAttention and Triton attention backends: after a registered layer's attention ran, its check
    (serve_attn.check_call). vLLM calls the backend's forward inside an opaque op of its own (eagerly, and while CUDA
    graphs are captured). Returns the classes patched."""
    import importlib
    from .serve_attn import check_call
    done = []
    for mod_name, cname in ATTN_BACKENDS:
        try:
            cls = getattr(importlib.import_module(mod_name), cname)
            orig = vars(cls)["forward"]
        except ImportError as e:      # (this vLLM build hasn't that backend: nothing of it to check)
            log.info("sdcguard: vLLM's %s not found (%s); its attention isn't checked", cname, e)
            continue
        except Exception as e:  # noqa: BLE001  (there, but not as the check knows it)
            log.warning("sdcguard: vLLM's %s not found (%s: %s); its attention isn't checked", cname,
                        type(e).__name__, e)
            continue
        if getattr(cls, "_sdcguard_attn_orig", None) is not None:
            done.append(cname)
            continue

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, *args, _orig=orig, **kwargs):
            out = _orig(self, layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
            a = getattr(layer, "_sdcguard_attn", None)
            if a is not None and attn_metadata is not None:
                output = kwargs.get("output", args[0] if args else None)
                scale = kwargs.get("output_scale", args[1] if len(args) > 1 else None)
                try:
                    check_call(a, self, query, key, value, kv_cache, attn_metadata, output, scale)
                except Exception as e:  # noqa: BLE001  (never break serving: this layer goes unchecked)
                    _attn_failed(layer, e)
            return out

        forward.__doc__ = orig.__doc__
        cls._sdcguard_attn_orig = orig
        cls.forward = forward
        done.append(cname)
    return done


def _attn_failed(layer, e: Exception) -> None:
    layer.__dict__.pop("_sdcguard_attn", None)
    g = _STATE["guard"]
    if g is not None:
        g.stats["attn_check_errors"] = g.stats.get("attn_check_errors", 0) + 1
    log.warning("sdcguard: an attention check failed (%s: %s); that layer isn't checked from now on",
                type(e).__name__, e)


def _register_attn(runner) -> Optional[dict]:
    """After vLLM loaded the model and before it captures CUDA graphs: register its attention layers with the guard
    (once per process), and log how many are checked and why the others aren't."""
    if not _STATE.get("attn_classes") or _STATE.get("attn_registered"):
        return None
    _STATE["attn_registered"] = True
    try:
        model = runner.get_model()
    except Exception:  # noqa: BLE001
        model = _model_of(runner)
    if model is None or not hasattr(model, "named_modules"):
        log.warning("sdcguard: couldn't find the model to register its attention layers; attention isn't checked")
        _STATE["attn_why"] = "the model to register its attention layers wasn't found"
        return None
    found = [(n, m) for n, m in model.named_modules() if hasattr(m, "impl") and hasattr(m, "layer_name")]
    if not found:
        log.info("sdcguard: no attention layers found in this model")
        _STATE["attn"] = {"checked": 0, "not_checked": {}}
        _STATE["attn_why"] = "no attention layers found in this model"
        return None
    if _ubatching(runner):
        # steps can run as microbatches (each layer called once per microbatch), which the attention check hasn't
        # been run with: left off and listed rather than assumed right
        why = "microbatching (dual-batch overlap or --ubatch-size), which the attention check hasn't been run with"
        log.warning("sdcguard: attention isn't checked: %s", why)
        _STATE["attn"] = {"checked": 0, "not_checked": {str(getattr(m, "layer_name", n)): why for n, m in found}}
        return None
    mc = getattr(runner, "model_config", None)
    dt = getattr(mc, "dtype", None)
    dev = getattr(runner, "device", None)
    if not isinstance(dev, torch.device):
        dev = next((p.device for p in model.parameters()), None) or torch.device("cuda")
    g = _guard(dev)
    ac = g.attn_checks(_attn_line())
    for name, m in found:
        ac.add(m, m.impl, str(getattr(m, "layer_name", "") or name or type(m).__name__), dev, dt)
    max_len = int(getattr(mc, "max_model_len", 0) or 0) or 32768
    ac.finalize(max_len=max_len, dtypes=(dt,) if dt in (torch.bfloat16, torch.float16) else (torch.bfloat16,))
    # the layers are checked from here on, so say so before anything else can fail
    _STATE["attn"] = {"checked": len(ac.layers), "not_checked": dict(ac.unsupported), "kernel": None}
    try:
        status = ac.kernel_status()
        _STATE["attn"]["kernel"] = status
        how = "the Triton kernels" if status and all(v[0] for v in status.values()) else \
            "the PyTorch check outside CUDA graphs only (" + "; ".join(f"{k}: {v[1]}" for k, v in status.items()
                                                                    if not v[0]) + ")"
        rest: dict = {}
        for why in ac.unsupported.values():
            rest[why] = rest.get(why, 0) + 1
        from .serve_attn import SAMPLES
        log.info("sdcguard: checking %d of %d attention layers (one layer a step, %d outputs each) with %s, line "
                 "%g%s", len(ac.layers), len(found), SAMPLES, how, ac.line,
                 "; not checked: " + ", ".join(f"{n} {w}" for w, n in sorted(rest.items())) if rest else "")
    except Exception as e:  # noqa: BLE001  (the layers are registered and checked; only the report failed)
        log.warning("sdcguard: %d attention layers are checked, but reading how failed (%s: %s)", len(ac.layers),
                    type(e).__name__, e)
    return _STATE["attn"]


def _patch_moe() -> list:
    """vLLM's RoutedExperts (fused MoE): after its experts ran (forward_modular: routing chosen, output finalized),
    the sampled recompute of a registered layer (serve_moe.check_call). vLLM runs MoE inside an opaque op of its own,
    so this runs eagerly, also while CUDA graphs are captured. Returns the classes patched."""
    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    except Exception as e:  # noqa: BLE001
        log.warning("sdcguard: vLLM's RoutedExperts not found (%s: %s); MoE layers aren't checked", type(e).__name__, e)
        return []
    if getattr(RoutedExperts, "_sdcguard_moe_orig", None) is not None:
        return ["RoutedExperts"]
    orig = vars(RoutedExperts).get("forward_modular")
    if orig is None:
        return []
    from .serve_moe import check_call

    def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None, _orig=orig):
        fp8 = getattr(self, "_sdcguard_fp8_moe", None) is not None and isinstance(x, torch.Tensor) and \
            isinstance(topk_ids, torch.Tensor)
        if fp8:                       # the FP8 check of its expert inputs' quantization needs their rows
            _FP8_CTX.moe = (self, x.reshape(-1, x.shape[-1]).shape[0], int(topk_ids.shape[-1]))
        try:
            out = _orig(self, x=x, topk_weights=topk_weights, topk_ids=topk_ids, shared_experts=shared_experts,
                        shared_experts_input=shared_experts_input)
        finally:
            if fp8:
                _FP8_CTX.moe = None
        a = getattr(self, "_sdcguard_moe", None)
        if a is not None:
            try:
                check_call(a, x, topk_weights, topk_ids, out)
            except Exception as e:  # noqa: BLE001  (never break serving: this layer goes unchecked)
                self.__dict__.pop("_sdcguard_moe", None)
                g = _STATE["guard"]
                if g is not None:
                    g.stats["moe_check_errors"] = g.stats.get("moe_check_errors", 0) + 1
                log.warning("sdcguard: an MoE check failed (%s: %s); that layer isn't checked from now on",
                            type(e).__name__, e)
        return out

    forward_modular.__doc__ = orig.__doc__
    RoutedExperts._sdcguard_moe_orig = orig
    RoutedExperts.forward_modular = forward_modular
    return ["RoutedExperts"]


def _patch_route() -> list:
    """vLLM's MoE routers (FusedMoERouter.select_experts): after a router picks each token's experts and weights,
    serve_route's check of that against the router's policy recomputed from its logits. Returns what was patched."""
    try:
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
    except Exception as e:  # noqa: BLE001
        log.info("sdcguard: vLLM's MoE router not found (%s); routing isn't checked", e)
        return []
    if getattr(FusedMoERouter, "_sdcguard_route_orig", None) is not None:
        return ["FusedMoERouter"]
    orig = vars(FusedMoERouter).get("select_experts")
    if orig is None:
        return []
    from .serve_route import check_call

    def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None, _orig=orig):
        w, ids = _orig(self, hidden_states, router_logits, topk_indices_dtype=topk_indices_dtype, input_ids=input_ids)
        a = getattr(self, "_sdcguard_route", None)
        if a is not None:
            try:
                check_call(a, router_logits, w, ids)
            except Exception as e:  # noqa: BLE001  (never break serving: this router goes unchecked)
                self.__dict__.pop("_sdcguard_route", None)
                log.warning("sdcguard: a routing check failed (%s: %s); that router isn't checked from now on",
                            type(e).__name__, e)
        return w, ids

    select_experts.__doc__ = orig.__doc__
    FusedMoERouter._sdcguard_route_orig = orig
    FusedMoERouter.select_experts = select_experts
    return ["FusedMoERouter"]


def _register_moe(runner) -> Optional[dict]:
    """After vLLM loaded the model, before CUDA graphs are captured: register its fused MoE layers (once)."""
    if not _STATE.get("moe_classes") or _STATE.get("moe_registered"):
        return None
    _STATE["moe_registered"] = True
    try:
        model = runner.get_model()
    except Exception:  # noqa: BLE001
        model = _model_of(runner)
    if model is None or not hasattr(model, "named_modules"):
        return None
    found = [(n, m) for n, m in model.named_modules() if type(m).__name__ in ("RoutedExperts", "FusedMoE")]
    if not found:
        return None
    dev = found[0][1].w13_weight.device if isinstance(getattr(found[0][1], "w13_weight", None), torch.Tensor) else None
    g = _guard(dev)
    mc = g.moe_checks()
    for name, mod in found:
        mc.add(mod, name or type(mod).__name__)
    mc.finalize()
    routers = None
    if _STATE.get("route_classes"):               # their routers too: the experts' recompute takes routing as given
        rc = g.route_checks()
        seen = 0
        for name, mod in model.named_modules():
            r = getattr(mod, "router", None)
            if r is not None and hasattr(r, "_select_experts") and getattr(r, "_sdcguard_route", None) is None:
                seen += 1
                rc.add(r, f"{name or type(mod).__name__}.router", dev if dev is not None else torch.device("cpu"))
        rc.finalize()
        why: dict = {}
        for w in rc.unsupported.values():
            why[w] = why.get(w, 0) + 1
        log.info("sdcguard: checking the routing of %d of %d MoE routers (one router a step, %d rows each)%s",
                 len(rc.layers), seen, rc_samples(), "; not checked: " + ", ".join(f"{n} {w}" for w, n in
                                                                                  sorted(why.items())) if why else "")
        routers = {"checked": len(rc.layers), "not_checked": dict(rc.unsupported)}
    from .serve_moe import SAMPLES, STATUS as MS
    kern = {f"{k[0]} {str(k[1]).replace('torch.', '')} act {k[2]}": v for k, v in MS.items()}
    rest: dict = {}
    for why in mc.unsupported.values():
        rest[why] = rest.get(why, 0) + 1
    log.info("sdcguard: checking %d of %d MoE layers (one layer a step, %s rows each; kernels %s)%s", len(mc.layers),
             len(found), SAMPLES, kern or "none",
             "; not checked: " + ", ".join(f"{n} {w}" for w, n in sorted(rest.items())) if rest else "")
    return {"checked": len(mc.layers), "not_checked": dict(mc.unsupported), "kernel": kern, "routers": routers}


def rc_samples() -> int:
    from .serve_route import SAMPLES
    return SAMPLES


_FP8_CTX = threading.local()          # the MoE layer whose experts run now (its input rows), for the FP8 check
# FP8ScaledMMLinearKernel.apply_weights in vLLM 0.31.0, unchanged in 0.32.0 and on main (October 24, 2026), as
# _quant.code_fingerprint reads it: _patch_fp8 runs a copy of that code, right only for it.
FP8_APPLY_FINGERPRINTS = frozenset({"f998fdda14eb8c7d"})


def _fp8_apply_ready(fn) -> Optional[str]:
    """None when vLLM's FP8ScaledMMLinearKernel.apply_weights is the code _patch_fp8 runs in its place (vLLM 0.31.0
    and 0.32.0), else why its layers' quantization isn't checked."""
    from ._quant import code_fingerprint
    fp = code_fingerprint(fn)
    if fp is None:
        return "FP8 apply_weights: its source couldn't be read"
    return None if fp in FP8_APPLY_FINGERPRINTS else f"FP8 apply_weights of an unknown version (fingerprint {fp})"


def _patch_fp8() -> list:
    """vLLM's FP8 W8A8 linear layers (FP8ScaledMMLinearKernel: the CUTLASS, FlashInfer, torch._scaled_mm and Triton
    scaled-mm kernels) and MoE layers with FP8 experts: right after vLLM quantizes a layer's input to FP8, its input,
    the FP8 input and the scale go to serve_fp8's check. The linear layers' apply_weights runs vLLM's own code (the
    same in vLLM 0.31 and 0.32, Apache-2.0; _fp8_apply_ready makes sure it is the code installed) with that call
    added; the MoE layers' quantization is vLLM's moe_kernel_quantize_input with the check after it. Returns what was
    patched."""
    import importlib
    import sys as _sys
    from .serve_fp8 import check_call
    patched = []
    try:
        # the module: vLLM's scaled_mm package exports the ScaledMMLinearKernel class under the module's name, so
        # `from ...scaled_mm import ScaledMMLinearKernel` gives the class (on vLLM 0.32 that left every FP8 linear
        # layer unchecked, silently)
        smk = importlib.import_module("vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel")
        K = smk.FP8ScaledMMLinearKernel
        orig = vars(K)["apply_weights"]
        if not callable(getattr(smk, "as_quantized_activation", None)):
            raise AttributeError("as_quantized_activation not found next to FP8ScaledMMLinearKernel")
    except Exception as e:  # noqa: BLE001
        why = f"vLLM's FP8 scaled-mm kernel not found ({type(e).__name__}: {e})"
        log.warning("sdcguard: not checking the FP8 quantization of linear layers: %s", why)
        _STATE["fp8_why"] = why
        K = orig = None
    if K is not None and not getattr(K, "_sdcguard_fp8", False):
        why = _fp8_apply_ready(orig)
        if why:
            log.warning("sdcguard: not checking FP8 quantization of linear layers: %s", why)
            _STATE["fp8_why"] = why
        else:
            def apply_weights(self, layer, x, bias=None, _orig=orig, _smk=smk):
                a = getattr(layer, "_sdcguard_fp8", None)
                if a is None:
                    return _orig(self, layer, x, bias)
                fp8_dtype = self.fp8_dtype  # noqa: F841  (as vLLM's code)
                maybe_out_dtype = self.config.out_dtype
                w, w_s, x_s, x_s_ub = self._get_layer_params(layer)
                if _smk.as_quantized_activation(x, self.input_quant_key()) is not None:
                    return _orig(self, layer, x, bias)    # quantized before this layer: no input to compare with
                x_2d = x.view(-1, x.shape[-1])
                output_shape = [*x.shape[:-1], w.shape[1]]
                out_dtype = x.dtype if maybe_out_dtype is None else maybe_out_dtype
                x_2d_q, x_s = self.quant_fp8(x_2d, x_s, x_s_ub)
                check_call(a, x_2d, x_2d_q, x_s, x_s_ub)
                return self.apply_scaled_mm(A=x_2d_q, B=w, out_dtype=out_dtype, As=x_s, Bs=w_s, bias=bias,
                                            output_shape=output_shape)

            apply_weights.__doc__ = orig.__doc__
            K.apply_weights = apply_weights
            K._sdcguard_fp8 = True
            patched.append("FP8ScaledMMLinearKernel")
    elif K is not None:
        patched.append("FP8ScaledMMLinearKernel")
    try:
        from vllm.model_executor.layers.fused_moe import utils as mu
        qorig = getattr(mu, "_sdcguard_fp8_orig", None) or mu.moe_kernel_quantize_input
    except Exception as e:  # noqa: BLE001
        log.warning("sdcguard: vLLM's MoE input quantization not found (%s); FP8 MoE layers aren't checked", e)
        return patched

    def moe_kernel_quantize_input(A, A_scale, quant_dtype, per_act_token_quant, block_shape=None, *args,
                                  _orig=qorig, **kwargs):
        out = _orig(A, A_scale, quant_dtype, per_act_token_quant, block_shape, *args, **kwargs)
        ctx = getattr(_FP8_CTX, "moe", None)
        try:
            if ctx is not None and isinstance(out, tuple) and len(out) == 2 and block_shape is None and \
                    not kwargs.get("quantization_emulation") and isinstance(out[0], torch.Tensor) and \
                    out[0].dtype in (torch.float8_e4m3fn,) and isinstance(A, torch.Tensor) and A.dim() == 2:
                layer, rows, top_k = ctx
                # the layer's live arguments (Fp8Checks.add re-points them when its state grows), when the recipe
                # registered for its expert inputs is this one
                kind = getattr(layer, "_sdcguard_fp8_moe", None)
                a = getattr(layer, "_sdcguard_fp8", None) if kind == ("per_token" if per_act_token_quant
                                                                      else "per_tensor") else None
                if a is not None and A_scale is None:
                    mult = 1 if A.shape[0] == rows else (top_k if A.shape[0] == rows * top_k else 0)
                    if mult:
                        check_call(a, A, out[0], out[1], None, mult)
        except Exception as e:  # noqa: BLE001  (never break serving)
            log.debug("sdcguard: FP8 MoE quantization check skipped: %s", e)
        return out

    n = 0
    for name, mod in list(_sys.modules.items()):
        # vLLM's own modules only, read through their __dict__: a lazy module elsewhere (transformers') imports on
        # attribute access and can raise for a missing optional dependency
        if mod is None or not (name == "vllm" or name.startswith("vllm.")):
            continue
        try:
            held = vars(mod).get("moe_kernel_quantize_input")
        except TypeError:
            continue
        if held is qorig:
            try:
                setattr(mod, "moe_kernel_quantize_input", moe_kernel_quantize_input)
                n += 1
            except Exception:  # noqa: BLE001
                pass
    mu._sdcguard_fp8_orig = qorig
    if n:
        patched.append(f"moe_kernel_quantize_input ({n} modules)")
    return patched


def _register_fp8(runner) -> Optional[dict]:
    """After vLLM loaded the model, before it compiles it or captures CUDA graphs: register its FP8 linear layers and
    FP8 MoE layers with the FP8 quantization check (once)."""
    if not _STATE.get("fp8_classes") or _STATE.get("fp8_registered"):
        return None
    _STATE["fp8_registered"] = True
    try:
        model = runner.get_model()
    except Exception:  # noqa: BLE001
        model = _model_of(runner)
    if model is None or not hasattr(model, "named_modules"):
        return None
    from .serve_fp8 import RECIPES
    fc, checked = None, 0
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        kern = getattr(qm, "fp8_linear", None)
        w = getattr(mod, "weight", None)
        if kern is not None and isinstance(w, torch.Tensor) and getattr(type(kern), "_sdcguard_fp8", False):
            if fc is None:
                fc = _guard(w.device).fp8_checks()
            if "apply_weights" in vars(type(kern)):
                fc.unsupported[name] = f"{type(kern).__name__}: its own apply_weights"
                continue
            xs = getattr(mod, "input_scale", None)
            if fc.add(mod, getattr(mod, "prefix", "") or name, getattr(kern, "quant_fp8", None),
                      xs if isinstance(xs, torch.Tensor) else None, w.device) is not None:
                checked += 1
        elif type(mod).__name__ in ("RoutedExperts", "FusedMoE") and \
                getattr(getattr(qm, "moe_quant_config", None), "use_fp8_w8a8", False):
            # (vLLM's FusedMoEQuantConfig, set on the layer's method when its weights were processed)
            w13 = getattr(mod, "w13_weight", None)
            if not isinstance(w13, torch.Tensor):
                continue
            if fc is None:
                fc = _guard(w13.device).fp8_checks()
            q = qm.moe_quant_config
            if getattr(q, "block_shape", None) is not None:
                fc.unsupported[name] = "FP8 MoE with block scales"
                continue
            per_token = bool(getattr(q, "per_act_token_quant", False))
            if getattr(q, "a1_scale", None) is not None:
                fc.unsupported[name] = "FP8 MoE with static input scales"
                continue
            kind = "dynamic_token" if per_token else "dynamic_tensor"
            quant = type("Q", (), {"static": False, "use_per_token_if_dynamic": per_token, "is_group_quant": False,
                                   "group_shape": kind})()
            a = fc.add(mod, name + " (expert inputs)", quant, None, w13.device)
            if a is not None:
                # which recipe its expert inputs have; the arguments themselves are read from mod._sdcguard_fp8 at
                # each call, which Fp8Checks keeps current when its state grows past 64 layers
                mod._sdcguard_fp8_moe = "per_token" if per_token else "per_tensor"
                checked += 1
    if fc is None:
        _STATE["fp8"] = {"checked": 0, "not_checked": {}, "recipes": sorted(RECIPES)}
        return None
    log.info("sdcguard: checking the FP8 quantization of %d layers%s", checked,
             "; not checked: " + ", ".join(f"{n} ({w})" for n, w in sorted(fc.unsupported.items())[:8])
             if fc.unsupported else "")
    _STATE["fp8"] = {"checked": checked, "not_checked": dict(fc.unsupported), "recipes": sorted(RECIPES)}
    return _STATE["fp8"]


def _step_rows(args, kwargs) -> Optional[int]:
    """The rows this model step's input holds before padding (its scheduled tokens), or None when unknown."""
    so = args[0] if args else kwargs.get("scheduler_output")
    n = getattr(so, "total_num_scheduled_tokens", None)
    if isinstance(n, int) and n > 0:
        return n
    counts = getattr(so, "num_scheduled_tokens", None)
    if isinstance(counts, dict) and counts:
        try:
            return int(sum(int(v) for v in counts.values()))
        except (TypeError, ValueError):
            return None
    return None


def _publish_stack(runner) -> Optional[dict]:
    """After the model loaded: what this server runs (sdcguard.stack), logged and published as sdcguard_stack_info."""
    from .stack import collect
    try:
        model = runner.get_model()
    except Exception:  # noqa: BLE001
        model = _model_of(runner)
    checks = {"share": f"{_share(os.environ.get('SDCGUARD_SHARE', '1')):g}", "z": f"{_STATE.get('z') or 6:g}",
              "norms": "on" if _STATE.get("norm_classes") else "off",
              "moe": "on" if _STATE.get("moe_classes") else "off",
              "spot": "on" if getattr(getattr(_STATE.get("guard"), "spot", None), "layers", None) else "off",
              "fp8": "on" if _STATE.get("fp8_classes") else "off",
              "attention": {True: "on", False: "outside_graphs", None: "on_unknown_where"}[_attn_kernels()[0]]
              if (_STATE.get("attn") or {}).get("checked") else "off",
              "weights": "on" if getattr(_STATE.get("guard"), "wscan", None) is not None else "off"}
    # the lines the checks that are on alert at (and the weight scan's verification-age target), so that
    # `sdcguard ready --qualified` can hold a server to the ones its qualification ran with
    g = _STATE.get("guard")
    for key, detector, v in (("norm_z", "norms", getattr(getattr(g, "norms", None), "line", None)),
                             ("attn_z", "attention", getattr(getattr(g, "attn", None), "line", None)),
                             ("weight_scan_age", "weights", getattr(getattr(g, "wscan", None), "target_age", None))):
        if checks[detector] != "off" and isinstance(v, (int, float)):
            checks[key] = f"{v:g}"
    info = collect(runner, model, checks)
    _STATE["stack"] = info
    _STATE["stack_runner"] = runner          # (to take a new weights digest after a weight change)
    _STATE["identity_stale"] = False
    g = _STATE["guard"]
    if g is not None and g.metrics is not None and hasattr(g.metrics, "stack"):
        g.metrics.stack(info)
    log.info("sdcguard: stack %s (%s)", info.get("fingerprint"),
             ", ".join(f"{k} {info[k]}" for k in ("model", "model_commit", "weights_digest", "dtype", "quantization",
                                                  "lora", "attention", "moe", "compilation", "cudagraph", "vllm",
                                                  "gpu") if info.get(k)))
    if (info.get("lora") or "off") != "off":
        log.info("sdcguard: LoRA adapters are enabled: they load per request, so the weights digest (%s) is of the "
                 "base model only", info.get("weights_digest"))
    return info


def _spot_on() -> bool:
    return os.environ.get("SDCGUARD_SPOT", "0").strip().lower() in ("1", "true", "on", "yes")


def _register_spot(runner) -> Optional[dict]:
    """With SDCGUARD_SPOT=1, after vLLM loaded the model and before it compiles it or captures CUDA graphs: the linear
    layers the checksum check takes are spot-checked too (serve_spot: one layer a step recomputes a few outputs
    exactly), once per process."""
    if not _spot_on() or _STATE.get("spot_registered"):
        return None
    _STATE["spot_registered"] = True
    model = _model_of(runner)
    if model is None or not hasattr(model, "named_modules"):
        log.warning("sdcguard: couldn't find the model to register its spot-checked layers; nothing is spot-checked")
        return None
    found = [(n, m) for n, m in model.named_modules() if getattr(m, "_sdcguard_args", None) is not None]
    if not found:
        return None
    w = getattr(found[0][1], "weight", None)
    g = _guard(w.device if isinstance(w, torch.Tensor) else None)
    sc = g.spot_checks()
    for name, mod in found:
        sc.add(mod, getattr(mod, "prefix", "") or name)
    sc.finalize()
    from .serve_spot import SPOT_COLS, SPOT_ROWS, STATUS as SS
    why: dict = {}
    for w_ in sc.unsupported.values():
        why[w_] = why.get(w_, 0) + 1
    log.info("sdcguard: spot-checking %d of %d checked linear layers (one layer a step, %d columns of up to %d rows; "
             "kernels %s)%s", len(sc.layers), len(found), SPOT_COLS, SPOT_ROWS,
             {f"{d} {str(t).replace('torch.', '')}": ("Triton" if ok else f"not usable: {kw}")
              for (d, t), (ok, kw) in SS.items()} or "none",
             "; not spot-checked: " + ", ".join(f"{n} {w_}" for w_, n in sorted(why.items())) if why else "")
    return {"checked": len(sc.layers), "not_checked": dict(sc.unsupported)}


def _register_norms(runner) -> Optional[dict]:
    """After vLLM loaded the model and before it compiles it or captures CUDA graphs: register its RMSNorm layers
    with the guard (once per process), and log how many are checked and what other norms the model has."""
    if not _STATE.get("norm_classes") or _STATE.get("norms_registered"):
        return None
    _STATE["norms_registered"] = True
    model = None
    try:
        model = runner.get_model()
    except Exception:  # noqa: BLE001
        model = _model_of(runner)
    if model is None or not hasattr(model, "named_modules"):
        log.warning("sdcguard: couldn't find the model to register its norm layers; norms aren't checked")
        return None
    from .serve_norm import kind_of
    found = []
    others = {}
    for name, mod in model.named_modules():
        kind = kind_of(mod)
        if kind is not None:
            found.append((name, mod, kind))
        elif "norm" in type(mod).__name__.lower() and not any(True for _ in mod.children()):
            others[name] = type(mod).__name__
    if not found:
        log.info("sdcguard: no RMSNorm layers in this model%s", f" (other norms: {len(others)})" if others else "")
        return None
    dev = found[0][1].weight.device
    g = _guard(dev)
    nc = g.norm_checks(_norm_line())
    for name, mod, kind in found:
        nc.add(mod, name or type(mod).__name__, kind)
    for name, cname in others.items():
        nc.unsupported[name] = f"{cname}: not an RMSNorm the check knows"
    dt = getattr(getattr(runner, "model_config", None), "dtype", None)
    dtypes = {d for d in (dt, *(m.weight.dtype for _, m, _ in found)) if d in (torch.bfloat16, torch.float16)}
    nc.finalize(dtypes=tuple(dtypes) or (torch.bfloat16,))
    status = nc.kernel_status()
    how = "the Triton kernel" if status and all(v[0] for v in status.values()) else \
        "the PyTorch check (" + "; ".join(f"{k}: {v[1]}" for k, v in status.items() if not v[0]) + ")"
    rest = {}
    for why in nc.unsupported.values():
        rest[why] = rest.get(why, 0) + 1
    log.info("sdcguard: checking %d of %d norm layers with %s, line %g%s", len(nc.layers), len(nc.layers) +
             len(nc.unsupported), how, nc.line,
             "; not checked: " + ", ".join(f"{n} {w}" for w, n in sorted(rest.items())) if rest else "")
    return {"checked": len(nc.layers), "not_checked": dict(nc.unsupported), "kernel": status}


def register() -> None:
    """vLLM general-plugin entry point. Runs in every vLLM process; safe to call more than once. With
    SDCGUARD_REQUIRE, anything that keeps it from checking refuses to serve (an error here stops vLLM's startup):
    SDCGUARD not 1, a vLLM layout it doesn't know, no model runner to hook (raised at model load, see
    _require_runner). Not covered: a plugin that fails to import at all, which vLLM's loader logs and skips."""
    if os.environ.get("SDCGUARD", "0") != "1":
        if _require_floor() is not None:
            raise RequirementNotMet("sdcguard: SDCGUARD_REQUIRE is set but SDCGUARD isn't 1, so nothing would be "
                                    "checked. Set SDCGUARD=1, or unset SDCGUARD_REQUIRE to serve unchecked.")
        return
    if _STATE["patched"]:
        return
    _setup_logging()
    try:
        from vllm.model_executor.layers import linear as L
        M = L.UnquantizedLinearMethod
        orig_pwal, orig_apply = M.process_weights_after_loading, M.apply
    except Exception as e:  # noqa: BLE001
        _refuse_if_required(f"vLLM layout not recognized ({type(e).__name__}: {e}); not checking anything")
        return
    if getattr(M, "_sdcguard", False):
        _STATE["patched"] = True
        return
    _STATE["z"] = _z()                    # a bad SDCGUARD_Z refuses here under SDCGUARD_REQUIRE, with its reason
    inline = os.environ.get("SDCGUARD_OPAQUE", "0") != "1"
    share = _share(os.environ.get("SDCGUARD_SHARE", "1"))
    share_offset = int(os.environ.get("SDCGUARD_OFFSET", "0"))
    # vLLM caches compiled graphs on disk, keyed by vLLM's own settings. A graph compiled without sdcguard has no
    # checks in it, and one compiled with other sdcguard settings checks other layers under other slots: keep
    # sdcguard's graphs in a folder of their own for each setting.
    if os.environ.get("SDCGUARD_SHARED_CACHE", "0") != "1":
        os.environ["VLLM_CACHE_ROOT"] = _cache_root(_graph_key(share, share_offset, inline))
    _exact_reductions()
    from .serve import _op, check_static_any, check_static_impl, select_layer
    op = _op()
    seen = [0]
    spot_call = None
    if _spot_on():
        from .serve_spot import check_call as spot_call

    def select(layer) -> bool:
        """SDCGUARD_SHARE: is this eligible layer (unquantized or quantized, in load order) one of those checked?
        Its number among the eligible ones is kept from its first load: a weight reload (vLLM calls
        process_weights_after_loading again for every layer) must select the same layers."""
        i = getattr(layer, "_sdcguard_index", None)
        if i is None:
            i = layer._sdcguard_index = seen[0]
            seen[0] += 1
        if select_layer(i, share, share_offset):
            return True
        layer._sdcguard_why = NOT_SELECTED
        return False

    def process_weights_after_loading(self, layer):
        _require_runner()
        orig_pwal(self, layer)
        try:
            w = getattr(layer, "weight", None)
            b = getattr(layer, "bias", None)
            b = b if isinstance(b, torch.Tensor) else None
            if getattr(layer, "_sdcguard_slot", 0):              # weights reloaded: same slot, same tensors
                _STATE["guard"].refresh_layer(layer._sdcguard_slot - 1, weight=w, bias=b)
                _rearm()
                return
            why = _cant_check(layer, w)
            if why is not None:
                layer._sdcguard_why = why
                _scan_skips(layer, f"the arithmetic check can't take it: {why}")
                return
            if not select(layer):
                _scan_unchecked(layer, w)                        # its weights still watched (SDCGUARD_WEIGHT_SCAN=1)
                return
            layer._sdcguard_why = FAILED                         # until it's registered
            g = _guard(w.device)
            slot = g.add_layer(layer, getattr(layer, "prefix", "") or type(layer).__name__, weight=w, bias=b)
            layer._sdcguard_args = (g.op_args(slot, with_bias=False), g.op_args(slot, with_bias=True))
            # trace the PyTorch check into vLLM's compiled graph only where the Triton check can't run
            layer._sdcguard_trace = inline and not g.uses_fast(slot)
            del layer._sdcguard_why
        except Exception as e:  # noqa: BLE001  (never break model loading)
            log.warning("sdcguard: not checking %s: %s", getattr(layer, "prefix", "a layer"), e)

    def apply(self, layer, x, bias=None):
        y = orig_apply(self, layer, x, bias)
        a = getattr(layer, "_sdcguard_args", None)
        if a is not None and isinstance(y, torch.Tensor):
            # one custom-op call (on NVIDIA GPUs: one cuBLAS call + one Triton kernel; CUDA graphs capture
            # them). Without the Triton check, under torch.compile the PyTorch check is traced into vLLM's
            # graph instead, so Inductor fuses its operations.
            if torch.compiler.is_compiling():
                fn = check_static_impl if getattr(layer, "_sdcguard_trace", False) else op
            else:
                fn = check_static_any              # eager (enforce_eager): no custom-op dispatch
            fn(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), *(a[1] if bias is not None else a[0]))
            sa = getattr(layer, "_sdcguard_spot", None)
            if sa is not None and spot_call is not None:   # SDCGUARD_SPOT=1: one Triton kernel more (it returns
                if torch.compiler.is_compiling():          # at once unless this layer's turn has come)
                    spot_call(sa, x, y, bias)
                else:
                    try:
                        spot_call(sa, x, y, bias)
                    except Exception as e:  # noqa: BLE001  (never break serving: this layer isn't spot-checked)
                        layer.__dict__.pop("_sdcguard_spot", None)
                        log.warning("sdcguard: a spot check failed (%s: %s); that layer isn't spot-checked from now "
                                    "on", type(e).__name__, e)
        return y

    M.process_weights_after_loading = process_weights_after_loading
    M.apply = apply
    M._sdcguard = True
    _STATE["patched"] = True
    if _spot_on():
        from .serve_spot import spot_op
        spot_op()                             # registered here, in every process, before any graph is traced
    if os.environ.get("SDCGUARD_NORMS", "0") == "1":
        from .serve_norm import norm_op
        norm_op()                             # registered here, in every process, before any graph is traced
        _STATE["norm_classes"] = _patch_norms()
    if os.environ.get("SDCGUARD_MOE", "0") == "1" or os.environ.get("SDCGUARD_FP8", "0") == "1":
        moe = _patch_moe()                    # (also the FP8 check's view of which MoE layer runs)
        _STATE["moe_classes"] = moe if os.environ.get("SDCGUARD_MOE", "0") == "1" else []
        if _STATE["moe_classes"]:
            _STATE["route_classes"] = _patch_route()
    if os.environ.get("SDCGUARD_FP8", "0") == "1":
        _STATE["fp8_classes"] = _patch_fp8()
    if _attn_on():
        _STATE["attn_classes"] = _patch_attn()
        if not _STATE["attn_classes"]:
            _STATE["attn_why"] = "neither of vLLM's FlashAttention and Triton attention backends was found"
    if os.environ.get("SDCGUARD_QUANT", "1") != "0":
        _STATE["quant_kernels"] = _patch_quant(op, select, inline)

    poll_every = max(1, int(os.environ.get("SDCGUARD_POLL_EVERY", "16")))
    poll_seconds = float(os.environ.get("SDCGUARD_POLL_SECONDS", "2"))
    last_poll = [time.monotonic()]

    def hook(module_name: str) -> bool:
        try:
            import importlib
            GM = importlib.import_module(module_name).GPUModelRunner
        except Exception:  # noqa: BLE001
            return False
        if getattr(GM, "_sdcguard", False):
            return True
        orig_exec = GM.execute_model
        orig_load = getattr(GM, "load_model", None)

        if orig_load is not None:
            def load_model(self, *args, **kwargs):
                out = orig_load(self, *args, **kwargs)
                for what, fn in (("norm", _register_norms), ("MoE", _register_moe), ("FP8", _register_fp8),
                                 ("attention", _register_attn), ("other weights to scan", _register_scan_rest),
                                 ("spot-checked", _register_spot)):
                    try:
                        fn(self)
                    except Exception as e:  # noqa: BLE001  (never break model loading)
                        log.warning("sdcguard: registering %s layers failed (%s: %s); they aren't checked", what,
                                    type(e).__name__, e)
                        if what == "attention" and not (_STATE.get("attn") or {}).get("checked"):
                            _STATE["attn_why"] = f"registering its layers failed ({type(e).__name__})"
                try:
                    _publish_stack(self)
                except Exception as e:  # noqa: BLE001  (the stack is information; never break model loading)
                    log.warning("sdcguard: couldn't read what this server runs (%s: %s)", type(e).__name__, e)
                return out

            GM.load_model = load_model

        def execute_model(self, *args, **kwargs):
            if not _STATE.get("reported"):
                _STATE["reported"] = True
                try:
                    _report_layers(self)
                except RequirementNotMet:
                    raise                              # SDCGUARD_REQUIRE: refuse to serve unchecked
                except Exception as e:  # noqa: BLE001  (never break serving, unless SDCGUARD_REQUIRE asks)
                    _refuse_if_required(f"couldn't count the checked layers ({type(e).__name__}: {e})")
            g0 = _STATE["guard"]
            _take_new_weights(g0)
            if kwargs.get("dummy_run"):
                # the V2 runner's dummy batches come through here, from _dummy_run (hooked below, which noted it)
                if not _STATE.get("in_dummy"):
                    _note_dummy()
                return orig_exec(self, *args, **kwargs)
            step = _STATE["steps"] + 1
            if g0 is not None:
                try:
                    rows = None if _rows_split(self) else _step_rows(args, kwargs)
                    if rows:                     # (steps with tokens: what each checks one attention layer in)
                        _STATE["token_steps"] = _STATE.get("token_steps", 0) + 1
                    g0.mark_step(step, rows)     # before the step's work is queued, outside its CUDA graph
                except Exception as e:  # noqa: BLE001  (attribution only; never break serving)
                    log.debug("sdcguard: mark_step failed: %s", e)
            try:
                out = orig_exec(self, *args, **kwargs)
            finally:
                if g0 is not None:
                    try:
                        g0.end_step()            # warm-up and capture runs after this aren't model steps
                    except Exception as e:  # noqa: BLE001
                        log.debug("sdcguard: end_step failed: %s", e)
            try:
                _record_batch(self, step, args, kwargs)
            except Exception as e:  # noqa: BLE001
                log.debug("sdcguard: couldn't record the step's batch: %s", e)
            _STATE["steps"] += 1
            now = time.monotonic()
            # read results every poll_every model steps without waiting (that read reports the previous one),
            # or, once poll_seconds have passed since the last read, right away (one wait for the GPU every few
            # seconds at most); at exit everything is read
            late = now - last_poll[0] >= poll_seconds
            g = _STATE["guard"]
            if (late or _STATE["steps"] % poll_every == 0) and g is not None:
                last_poll[0] = now
                try:
                    g.poll(wait=late)
                except Exception as e:  # noqa: BLE001  (never break serving)
                    log.warning("sdcguard: poll failed: %s", e)
                # the first read with results in it: a registered layer that hasn't run its check by now (warm-up
                # and the first steps) has a forward that doesn't reach it. Counted from what was read back, so
                # this doesn't wait for the GPU. Except with SDCGUARD_REQUIRE=1 at the first read: no results at
                # all may mean no check ever runs (it waited for one before, so a model none of whose layers reached
                # its check served unchecked), so it waits for the GPU once and counts what came back.
                strict = _require_floor() is not None
                if _STATE.get("rechecked") is not g and g.layers and (g.stats.get("checks_executed", 0) or strict):
                    if strict:
                        try:
                            g.poll(wait=True)
                        except Exception as e:  # noqa: BLE001  (results that can't be read can't show coverage)
                            raise RequirementNotMet(f"SDCGUARD_REQUIRE=1: the checks' results couldn't be read to "
                                                    f"count what ran ({type(e).__name__}: {e})") from e
                    _STATE["rechecked"] = g
                    try:
                        _report_layers(self, {lay["name"]: {"checks": g.checks_seen.get(lay["name"], 0)}
                                              for lay in g.layers}, again=True)
                    except RequirementNotMet:
                        raise
                    except Exception as e:  # noqa: BLE001
                        _refuse_if_required(f"couldn't count the checked layers ({type(e).__name__}: {e})")
            return out

        GM.execute_model = execute_model
        orig_dummy = getattr(GM, "_dummy_run", None)
        if orig_dummy is not None:
            def _dummy_run(self, *args, **kwargs):
                _STATE["in_dummy"] = _STATE.get("in_dummy", 0) + 1
                try:
                    _note_dummy()
                    return orig_dummy(self, *args, **kwargs)
                finally:
                    _STATE["in_dummy"] -= 1

            GM._dummy_run = _dummy_run
        GM._sdcguard = True
        return True

    # vLLM 0.31 has two model runners (the V2 one is the default); hook whichever exist
    hooked = [m for m in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner") if hook(m)]
    _STATE["hooked"] = bool(hooked)
    if not hooked:
        log.warning("sdcguard: couldn't hook the model runner; call sdcguard.vllm_plugin.poll() yourself")
    log.info("sdcguard: checking bf16/fp16 linear layers%s (alert threshold z=%s, share %s); fused MoE experts and "
             "other quantized layers aren't checked yet, and the first model step logs how many layers that leaves "
             "out", " and INT weight-only layers on " + " and ".join(_STATE.get("quant_kernels") or []) if
             _STATE.get("quant_kernels") else "", os.environ.get("SDCGUARD_Z", "6"), share)
