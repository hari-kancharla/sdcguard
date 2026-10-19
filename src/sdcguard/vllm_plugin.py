"""vLLM plugin: check the math of every unquantized (bf16/fp16) linear layer while vLLM serves.

    pip install sdcguard
    SDCGUARD=1 vllm serve <model>

Off unless SDCGUARD=1 (vLLM loads every installed plugin, so installing sdcguard alone changes
nothing). What it does, in every vLLM process:

  - after vLLM loads a linear layer's weights, computes that layer's checksums once (serve.py);
  - after every matmul of an unquantized linear layer, queues the check (one custom op, so
    torch.compile keeps a single graph and CUDA graphs capture it);
  - every SDCGUARD_POLL_EVERY model steps (default 16), reads the per-layer results back without
    stalling the GPU and logs an alert for any layer whose output didn't match its checksums:

    sdcguard ALERT: model.layers.17.mlp.down_proj on GPU-1a2b... (NVIDIA H100 80GB HBM3, PCI ...)
      z=inf (row 3, block 12)

Settings (environment variables): SDCGUARD_Z (alert threshold, default 6), SDCGUARD_LOG (also append
alerts as JSON lines to this file), SDCGUARD_POLL_EVERY, SDCGUARD_SHARE (check only this share of the
linear layers, e.g. 0.125, for a lower cost) and SDCGUARD_OFFSET (which ones; vary it across replicas),
SDCGUARD_WEBHOOK (post alerts to Slack, Discord or any JSON endpoint; SDCGUARD_WEBHOOK_FORMAT=slack,
discord or json) and SDCGUARD_METRICS_PORT (Prometheus metrics; each GPU worker adds its local index).

Updating weights in place while serving (an RL weight sync)? Call sdcguard.vllm_plugin.refresh() in each
worker afterwards; otherwise every updated layer is reported (with weights_changed_since_load = true).

Not checked: quantized layers (FP8, INT4, ...), fused MoE experts, attention, the LM head.
Tested against vLLM 0.31 (V1 engine); vLLM's internal APIs change, so the plugin checks what it
patches and turns itself off (with a log line) if something is missing.
"""
from __future__ import annotations

import atexit
import logging
import os

import torch

log = logging.getLogger("sdcguard")

_STATE = {"guard": None, "steps": 0, "op": None, "patched": False}


def _metrics_port():
    """SDCGUARD_METRICS_PORT plus this worker's local GPU index (one exporter per GPU worker)."""
    base = os.environ.get("SDCGUARD_METRICS_PORT")
    if not base:
        return None
    try:
        idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        idx = 0
    return int(base) + int(idx)


def _guard():
    if _STATE["guard"] is None:
        from .serve import ServeGuard
        port = _metrics_port()
        try:
            g = ServeGuard(z=float(os.environ.get("SDCGUARD_Z", "6")), log_path=os.environ.get("SDCGUARD_LOG") or None,
                           on_alert=_on_alert, webhook=os.environ.get("SDCGUARD_WEBHOOK") or None,
                           webhook_format=os.environ.get("SDCGUARD_WEBHOOK_FORMAT", "slack"), metrics_port=port)
        except Exception as e:  # noqa: BLE001  (e.g. prometheus-client missing or the port taken: keep checking)
            log.warning("sdcguard: alerts/metrics setup failed (%s); checking without them", e)
            g = ServeGuard(z=float(os.environ.get("SDCGUARD_Z", "6")), log_path=os.environ.get("SDCGUARD_LOG") or None,
                           on_alert=_on_alert)
        if port is not None and g.metrics is not None:
            log.info("sdcguard: Prometheus metrics on port %s", port)
        _STATE["guard"] = g
        atexit.register(_flush_at_exit)
    return _STATE["guard"]


def _flush_at_exit() -> None:
    """Results are read every SDCGUARD_POLL_EVERY steps; read whatever the last steps left when the
    worker shuts down, so an alert from the final requests isn't lost."""
    try:
        poll(wait=True)
    except Exception:  # noqa: BLE001  (shutting down: the CUDA context may already be gone)
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
    g.refresh()
    return len(g.layers)


def _exact_reductions() -> None:
    """Keep bf16/fp16 split-K partial sums in fp32 (cuBLAS), as the training guard does on checked
    steps: the noise model assumes one rounding per output. Changes rounding slightly; costs little.
    SDCGUARD_EXACT=0 leaves PyTorch's default."""
    if os.environ.get("SDCGUARD_EXACT", "1") == "0" or not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    except (AttributeError, RuntimeError, TypeError) as e:
        log.warning("sdcguard: couldn't keep matmul partial sums in fp32 (%s)", e)


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
    # vLLM caches compiled graphs on disk, keyed by vLLM's own settings. A graph compiled without
    # sdcguard has no checks in it, so keep sdcguard's compiled graphs in their own cache folder.
    if "VLLM_CACHE_ROOT" not in os.environ and os.environ.get("SDCGUARD_SHARED_CACHE", "0") != "1":
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(os.path.expanduser("~/.cache/vllm"), "sdcguard")
    _exact_reductions()
    from .serve import _op, check_static_any, check_static_impl, select_layer
    op = _op()
    inline = os.environ.get("SDCGUARD_OPAQUE", "0") != "1"

    share = float(os.environ.get("SDCGUARD_SHARE", "1"))
    share_offset = int(os.environ.get("SDCGUARD_OFFSET", "0"))
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
            on_gpu = isinstance(w, torch.Tensor) and (w.is_cuda or os.environ.get("SDCGUARD_ALLOW_CPU") == "1")
            if not (on_gpu and w.dim() == 2 and w.dtype in (torch.bfloat16, torch.float16)):
                return
            seen[0] += 1
            if not select_layer(seen[0] - 1, share, share_offset):
                return
            g = _guard()
            slot = g.add_layer(layer, getattr(layer, "prefix", "") or type(layer).__name__, weight=w, bias=b)
            layer._sdcguard_args = (g.op_args(slot, with_bias=False), g.op_args(slot, with_bias=True))
            # trace the PyTorch check into vLLM's compiled graph only where the Triton check can't run
            layer._sdcguard_trace = inline and not g.uses_fast(slot)
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
            out = orig_exec(self, *args, **kwargs)
            _STATE["steps"] += 1
            if _STATE["steps"] % poll_every == 0 and _STATE["guard"] is not None:
                try:
                    _STATE["guard"].poll()
                except Exception as e:  # noqa: BLE001  (never break serving)
                    log.warning("sdcguard: poll failed: %s", e)
            return out

        GM.execute_model = execute_model
        GM._sdcguard = True
        return True

    # vLLM 0.31 has two model runners (the V2 one is the default); hook whichever exist
    hooked = [m for m in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner") if hook(m)]
    if not hooked:
        log.warning("sdcguard: couldn't hook the model runner; call sdcguard.vllm_plugin.poll() yourself")
    log.info("sdcguard: checking unquantized linear layers (alert threshold z=%s, share %s)",
             os.environ.get("SDCGUARD_Z", "6"), share)
