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
vLLM computes with its own GEMM (GateLinear with a bf16 weight and fp32 output), attention (including MLA's decode
path, whose matmuls use copies of kv_b_proj's weight, so kv_b_proj is checked in prefill only), the LM head.
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

_STATE = {"guard": None, "steps": 0, "op": None, "patched": False, "reported": False, "layers": None,
          "rechecked": None, "exit_hooks": None, "flushed": None, "hooked": False, "z": None, "step_log": None}

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
        if metrics is not None:               # who and how, for `sdcguard ready` (checks from outside)
            try:
                from .events import device_identity
                uuid = device_identity(device).get("gpu_uuid") if device is not None and device.type == "cuda" \
                    else None
                metrics.identify(uuid, kw["z"], _share(os.environ.get("SDCGUARD_SHARE", "1")))
            except Exception as e:  # noqa: BLE001
                log.warning("sdcguard: couldn't publish sdcguard_protection_info (%s)", e)
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
    selected). weight_share_checked: the checked share of all those layers' weights."""
    if counts is None:
        counts = _registered_counts()
    g = _STATE["guard"]
    checked = selected = eligible = cant = quant = 0
    unchecked: dict = {}
    never: list = []
    w_all = w_checked = 0
    for mod in model.modules() if model is not None else ():
        qm = getattr(mod, "quant_method", None)
        if qm is None:
            continue
        method = type(qm).__name__
        if any(isinstance(getattr(mod, a, None), torch.Tensor) for a in _MOE_WEIGHTS):
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
            "quantized_or_moe": quant, "never_checked": sorted(never),
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
    if r["unchecked_by_method"]:
        msg += "; not checked: " + ", ".join(f"{n} {why}" for why, n in sorted(r["unchecked_by_method"].items()))
    if r["quantized_or_moe"]:
        msg += (" (fused MoE experts, FP8 layers and quantized layers outside Marlin's and Machete's INT4/INT8 kernels "
                "aren't checked yet)")
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
    return tuple(ids), ends


def _rows_split(runner) -> Optional[str]:
    """Why a step's rows don't map to its requests one to one (microbatching, sequence parallelism), or None."""
    if getattr(getattr(runner, "parallel_config", None), "enable_dbo", False):
        return "rows are split into microbatches (dual-batch overlap)"
    pc = getattr(getattr(runner, "compilation_config", None), "pass_config", None)
    if getattr(pc, "enable_sequence_parallelism", False):
        return "rows are split across tensor-parallel ranks (sequence parallelism)"
    return None


def _record_batch(runner, step: int, args, kwargs) -> None:
    if _STATE.get("step_log") is None:
        _STATE["step_log"] = collections.deque(maxlen=max(16, int(os.environ.get("SDCGUARD_STEP_LOG", "4096"))))
    split = _rows_split(runner)
    rows = None if split else _batch_rows(runner, args, kwargs)
    _STATE["step_log"].append((step, rows[0], rows[1]) if rows else
                              (step, None, split or "the step's batch couldn't be read"))


def _attribute(ev) -> None:
    """Add to an alert which requests it concerns (ServeGuard.attribute), in three parts kept apart:
      directly_implicated   the request that held the row the check alarmed on, in the step it alarmed (from the
                            captured row): [id], [] for a CUDA-graph padding row, None when it can't be told
      potentially_exposed   every request in the steps the read covered ("read_window"), and every request since
                            the layer's last clean read ("since_last_clean_read": a fault that persists, like a
                            changed weight, concerns them all); a conservative window, not a proof of harm
      unknown_impact        what neither covers (state that outlives the window, layers not checked)
    Ids are the client's (vLLM's internal suffix removed); "attribution" says how they were found."""
    d = ev.detail
    steps = {s: (ids, ends) for s, ids, ends in (_STATE.get("step_log") or ())}
    cap = d.get("captured") or {}
    st, row = cap.get("step"), cap.get("row")
    exposed: dict = {}
    d["unknown_impact"] = [
        "requests that reuse prefix-cache (KV) blocks written during these steps can carry a wrong value further",
        "the plugin checks linear layers only; attention, norms, MoE experts, the KV cache and sampling aren't checked"]
    if _share(os.environ.get("SDCGUARD_SHARE", "1")) < 1:
        d["unknown_impact"].append("SDCGUARD_SHARE checks the same subset of layers every step; the others aren't seen")
    d["potentially_exposed"] = exposed
    if not steps:
        d["directly_implicated"], d["attribution"] = None, "unavailable: no model step was recorded"
        return
    if st is not None and row is not None and st in steps:
        ids, ends = steps[st]
        if ids is None:
            d["directly_implicated"], d["attribution"] = None, f"unavailable for step {st}: {ends}"
        elif row < ends[-1]:
            d["directly_implicated"] = [_external(ids[bisect.bisect_right(ends, row)])]
            d["attribution"] = f"exact: row {row} of step {st}, where the check alarmed, belonged to this request"
        else:
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


def _on_alert(ev) -> None:
    d, i = ev.detail, ev.identity
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
    layers are checked (SDCGUARD_QUANT), the sdcguard version, and the check op's arguments (serve.OP_SCHEMA: a graph
    traced with other arguments can't run)."""
    from . import __version__
    from .serve import OP_SCHEMA
    env = os.environ
    z = _z_from(env.get("SDCGUARD_Z", "6")) or 6.0
    return hashlib.sha256(repr((__version__, OP_SCHEMA, MACHETE_GRAPH, share, offset, z,
                                inline,
                                env.get("SDCGUARD_FAST", "1") != "0", env.get("SDCGUARD_SERVE_FAST", "1") != "0",
                                env.get("SDCGUARD_ALLOW_CPU") == "1", env.get("SDCGUARD_QUANT", "1") != "0")).encode()
                          ).hexdigest()[:12]


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
                return
            if not select(layer):
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
        return y

    M.process_weights_after_loading = process_weights_after_loading
    M.apply = apply
    M._sdcguard = True
    _STATE["patched"] = True
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

        def execute_model(self, *args, **kwargs):
            if not _STATE.get("reported"):
                _STATE["reported"] = True
                try:
                    _report_layers(self)
                except RequirementNotMet:
                    raise                              # SDCGUARD_REQUIRE: refuse to serve unchecked
                except Exception as e:  # noqa: BLE001  (never break serving, unless SDCGUARD_REQUIRE asks)
                    _refuse_if_required(f"couldn't count the checked layers ({type(e).__name__}: {e})")
            step = _STATE["steps"] + 1
            g0 = _STATE["guard"]
            if g0 is not None:
                try:
                    g0.mark_step(step)           # before the step's work is queued, outside its CUDA graph
                except Exception as e:  # noqa: BLE001  (attribution only; never break serving)
                    log.debug("sdcguard: mark_step failed: %s", e)
            out = orig_exec(self, *args, **kwargs)
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
