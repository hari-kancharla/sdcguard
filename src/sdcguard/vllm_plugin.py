"""vLLM plugin: check the math of every unquantized (bf16/fp16) linear layer while vLLM serves.

    pip install sdcguard
    SDCGUARD=1 vllm serve <model>

Off unless SDCGUARD=1 (vLLM loads every installed plugin, so installing sdcguard alone changes
nothing). What it does, in every vLLM process:

  - after vLLM loads a linear layer's weights, computes that layer's checksums once (serve.py);
  - after every matmul of an unquantized linear layer, queues the check (one custom op, so
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
for that share of them; _require says exactly what counts). The first model step logs the count either way,
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

Not checked: quantized layers (FP8, INT4, ...), fused MoE experts, router gates that vLLM computes with its own
GEMM (GateLinear with a bf16 weight and fp32 output), attention (including MLA's decode path, whose matmuls
use copies of kv_b_proj's weight, so kv_b_proj is checked in prefill only), the LM head.
Tested against vLLM 0.31 (V1 engine); vLLM's internal APIs change, so the plugin checks what it
patches and turns itself off (with a log line) if something is missing.
"""
from __future__ import annotations

import atexit
import hashlib
import logging
import os
import re
import time
from typing import Optional

import torch

log = logging.getLogger("sdcguard")

_STATE = {"guard": None, "steps": 0, "op": None, "patched": False, "reported": False, "layers": None,
          "rechecked": None, "exit_hooks": None, "flushed": None}

# Why a linear layer isn't checked (layer_report). The quantized methods are reported by their own names.
NOT_SELECTED = "not selected by SDCGUARD_SHARE"
FAILED = "selected, but registering it failed (see the warning logged then)"
NEVER_RAN = "registered, but its check never ran (its forward doesn't call vLLM's linear method)"
ROUTER = "router gate computed by GateLinear's own GEMM"
LM_HEAD = "LM head"
UNSEEN = "unquantized, but its weights were processed without the plugin"
NOT_A_MATRIX = "-D weight (a convolution kept in a linear layer)"

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


def _share(text: str) -> float:
    """SDCGUARD_SHARE: a share of layers, as 0.125, 1/8 or 12.5%."""
    t = text.strip()
    try:
        if t.endswith("%"):
            v = float(t[:-1]) / 100
        elif "/" in t:
            a, b = t.split("/", 1)
            v = float(a) / float(b)
        else:
            v = float(t)
    except (ValueError, ZeroDivisionError):
        log.warning("sdcguard: SDCGUARD_SHARE=%r isn't a share (use 0.125, 1/8 or 12.5%%); checking every layer",
                    text)
        return 1.0
    return min(1.0, max(0.0, v))


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
        kw = dict(z=float(os.environ.get("SDCGUARD_Z", "6")), log_path=os.environ.get("SDCGUARD_LOG") or None,
                  on_alert=_on_alert, metrics=metrics, evidence_dir=os.environ.get("SDCGUARD_EVIDENCE_DIR") or None)
        try:
            g = ServeGuard(webhook=os.environ.get("SDCGUARD_WEBHOOK") or None,
                           webhook_format=os.environ.get("SDCGUARD_WEBHOOK_FORMAT", "slack"), **kw)
        except Exception as e:  # noqa: BLE001  (a bad webhook format: keep checking)
            log.warning("sdcguard: webhook setup failed (%s); alerts still go to the log", e)
            g = ServeGuard(**kw)
        _STATE["guard"] = g
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
            name = getattr(mod, "prefix", "") or type(mod).__name__
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
            kind = {FAILED: "sel", ROUTER: "cant", NOT_SELECTED: "unselected"}.get(why, "out")
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
        msg += " (quantized layers and fused MoE experts aren't checked yet)"
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
    need = os.environ.get("SDCGUARD_REQUIRE")
    if not need:
        return
    try:
        floor = 1.0 if need.strip() == "1" else _share(need)
    except Exception:  # noqa: BLE001
        floor = 1.0
    want = r["selected"] + _share(os.environ.get("SDCGUARD_SHARE", "1")) * r["cant_check"]
    nothing = r["total"] > 0 and r["eligible"] == 0
    if floor > 0 and (r["checked"] + 1e-9 < floor * want or nothing):
        raise RequirementNotMet(f"{msg}; SDCGUARD_REQUIRE={need} asks for more. Unset it to serve anyway.")


def _report_layers(runner, counts: Optional[dict] = None, again: bool = False) -> Optional[dict]:
    """Count the model's checked layers, log the count, export it and apply SDCGUARD_REQUIRE. again: the recount at
    the first read of results, which only speaks up when a registered layer's check never ran."""
    model = _model_of(runner)
    if model is None:
        if not again:
            log.warning("sdcguard: couldn't find the model to count its checked layers")
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
    return g.refresh()


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
    is traced in or called as the custom op (SDCGUARD_OPAQUE, SDCGUARD_FAST, SDCGUARD_SERVE_FAST), the sdcguard
    version, and the check op's arguments (serve.OP_SCHEMA: a graph traced with other arguments can't run)."""
    from . import __version__
    from .serve import OP_SCHEMA
    env = os.environ
    return hashlib.sha256(repr((__version__, OP_SCHEMA, share, offset, float(env.get("SDCGUARD_Z", "6")), inline,
                                env.get("SDCGUARD_FAST", "1") != "0", env.get("SDCGUARD_SERVE_FAST", "1") != "0",
                                env.get("SDCGUARD_ALLOW_CPU") == "1")).encode()).hexdigest()[:12]


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


def register() -> None:
    """vLLM general-plugin entry point. Runs in every vLLM process; safe to call more than once."""
    if os.environ.get("SDCGUARD", "0") != "1" or _STATE["patched"]:
        return
    try:
        from vllm.model_executor.layers import linear as L
        M = L.UnquantizedLinearMethod
        orig_pwal, orig_apply = M.process_weights_after_loading, M.apply
    except Exception as e:  # noqa: BLE001
        log.warning("sdcguard: vLLM layout not recognized (%s); not checking anything", e)
        return
    if getattr(M, "_sdcguard", False):
        _STATE["patched"] = True
        return
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

    def process_weights_after_loading(self, layer):
        orig_pwal(self, layer)
        try:
            w = getattr(layer, "weight", None)
            b = getattr(layer, "bias", None)
            b = b if isinstance(b, torch.Tensor) else None
            if getattr(layer, "_sdcguard_slot", 0):              # weights reloaded: same slot, same tensors
                _STATE["guard"].refresh_layer(layer._sdcguard_slot - 1, weight=w, bias=b)
                return
            why = _cant_check(layer, w)
            if why is not None:
                layer._sdcguard_why = why
                return
            # the layer's number among the eligible ones, from its first load: a weight reload (vLLM calls this
            # again for every layer) must select the same layers
            i = getattr(layer, "_sdcguard_index", None)
            if i is None:
                i = layer._sdcguard_index = seen[0]
                seen[0] += 1
            if not select_layer(i, share, share_offset):
                layer._sdcguard_why = NOT_SELECTED
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
                except Exception as e:  # noqa: BLE001  (never break serving)
                    log.warning("sdcguard: couldn't count the checked layers: %s", e)
            out = orig_exec(self, *args, **kwargs)
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
                strict = os.environ.get("SDCGUARD_REQUIRE") == "1"
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
                        log.warning("sdcguard: couldn't count the checked layers: %s", e)
            return out

        GM.execute_model = execute_model
        GM._sdcguard = True
        return True

    # vLLM 0.31 has two model runners (the V2 one is the default); hook whichever exist
    hooked = [m for m in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner") if hook(m)]
    if not hooked:
        log.warning("sdcguard: couldn't hook the model runner; call sdcguard.vllm_plugin.poll() yourself")
    log.info("sdcguard: checking unquantized linear layers (alert threshold z=%s, share %s); quantized layers "
             "and fused MoE experts aren't checked yet, and the first model step logs how many layers that leaves "
             "out", os.environ.get("SDCGUARD_Z", "6"), share)
