"""What a vLLM server actually runs, read from inside it: the model and how it is served (its revision, the commit
that revision resolved to on the Hugging Face hub, a digest of its weights as loaded, its effective configuration
after any --hf-overrides (model_config: a hash of the whole config, so a changed rms_norm_eps counts; hf_overrides:
the overrides themselves), the generation config that sets default sampling, dtype, quantization, whether LoRA
adapters are enabled, the attention and MoE backends its layers ended up with, the norm implementation, compilation
and CUDA graph modes, parallelism, KV cache dtype), the software (vLLM, PyTorch, CUDA, Triton), the GPU, the
environment variables that steer kernels, and how sdcguard checks it. The vLLM plugin publishes it as
sdcguard_stack_info (one label per field, plus a fingerprint of them all), `sdcguard qualify` records it with each
decision (every worker's), and `sdcguard ready --qualified` compares a running server with the stack a decision was
made on.

What binds a decision to the model: model_commit, the commit vLLM resolved --revision to (vLLM 0.32 keeps the
revision as asked, "main" say, and the commit in it; a local checkpoint has none), and weights_digest, one digest of
every parameter as loaded (serve_weights.model_digest: each tensor's raw bits fingerprinted as the weight scan does,
once at load, never per request; with tensor parallelism each worker's digest is of its own shard). Neither sees a
weight changed after load (that is the weight scan's job), nor LoRA adapters, which load per request.

Only what can be read is recorded; a field that can't be read is left out (and an empty value never matches a
recorded one). A record made before model_commit, weights_digest and lora were fields doesn't have them, and a
server can't be shown to be the stack it describes: compare() says unknown, not the same. Most of these aren't
visible from outside the server: its flags aren't in its API.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

# The fields, in the order they're shown; FINGERPRINT_KEYS is what the fingerprint covers (not the sdcguard settings:
# a decision is about the serving stack, and the checks a server runs are compared on their own by `ready`).
FIELDS = ("model", "revision", "model_commit", "weights_digest", "model_config", "hf_overrides", "generation_config",
          "dtype", "quantization", "lora", "kv_cache_dtype", "max_model_len", "tensor_parallel", "pipeline_parallel",
          "data_parallel", "attention", "moe", "rms_norm", "compilation", "cudagraph", "eager", "vllm", "torch",
          "cuda", "triton", "gpu", "env", "sdcguard", "checks")
FINGERPRINT_KEYS = FIELDS[:-2]
# Fields added after records were first kept: a record without one is older, not a stack without it. They join the
# fingerprint only when present, so a stack without them keeps the fingerprint it had before them.
IDENTITY = ("model_commit", "weights_digest", "lora")
PER_WORKER = ("weights_digest",)               # each worker's own (its shard of the weights)
_WHAT = {"model_commit": "its model's resolved revision (the Hugging Face commit its files were read at)",
         "weights_digest": "its weights' identity (a digest of every parameter as loaded)",
         "lora": "whether LoRA adapters are enabled"}


def _get(obj, *path, default=None):
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return default
    return obj


def _s(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (list, tuple, set)):
        return ",".join(sorted(str(x) for x in v)) or None
    s = str(v)
    for prefix in ("torch.", "CompilationMode.", "CUDAGraphMode.", "AttentionBackendEnum."):
        s = s.replace(prefix, "")
    return s


def _config_hash(mc) -> Optional[str]:
    """A hash of the model's effective Hugging Face config (vLLM applies --hf-overrides to it), without the keys
    that only say where it was read from."""
    hf = _get(mc, "hf_config")
    if hf is None:
        return None
    try:
        d = hf.to_dict() if callable(getattr(hf, "to_dict", None)) else dict(vars(hf))
    except Exception:  # noqa: BLE001
        return None
    d = {k: v for k, v in d.items() if not str(k).startswith("_")}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _commit(mc) -> Optional[str]:
    """The commit the model's files were read at: vLLM 0.32 resolves --revision once (model_config.revision is then
    huggingface_hub's ResolvedRevision, a str equal to the revision asked for whose .resolved is the commit hash);
    older transformers kept it in the config as _commit_hash. None for a local checkpoint."""
    rev = _get(mc, "revision")
    hf = _get(mc, "hf_config")
    for c in (getattr(rev, "resolved", None), getattr(hf, "_commit_hash", None)):
        if isinstance(c, str) and c:
            return c
    if isinstance(rev, str) and re.fullmatch(r"[0-9a-f]{40}", rev):
        return rev
    return None


def _lora(cfg) -> Optional[str]:
    """Whether vLLM serves LoRA adapters (--enable-lora) and how many at once. The adapters themselves load per request
    (or from the API server's --lora-modules, after the workers loaded the model), so their identity can't be read at
    load and they aren't in weights_digest; default_mm_loras, set in the config, are named."""
    if cfg is None:
        return None
    lc = getattr(cfg, "lora_config", None)
    if lc is None:
        return "off"
    parts = [f"{k}={getattr(lc, k)}" for k in ("max_loras", "max_lora_rank") if getattr(lc, k, None) is not None]
    mm = _pairs(getattr(lc, "default_mm_loras", None))
    if mm:
        parts.append(f"default_mm_loras {mm}")
    return "on: " + ", ".join(parts) + "; adapters load per request, not in the weights digest"


def _pairs(d) -> Optional[str]:
    if callable(d):
        return f"callable:{getattr(d, '__qualname__', type(d).__name__)}"
    if not isinstance(d, dict) or not d:
        return None
    return ",".join(f"{k}={json.dumps(v, sort_keys=True, default=str)}" for k, v in sorted(d.items()))


def collect(runner, model=None, checks: Optional[dict] = None) -> dict:
    """The stack of the vLLM model runner `runner` (and its loaded model)."""
    cfg = getattr(runner, "vllm_config", None)
    mc = _get(cfg, "model_config") or getattr(runner, "model_config", None)
    pc = _get(cfg, "parallel_config") or getattr(runner, "parallel_config", None)
    cc = _get(cfg, "compilation_config") or getattr(runner, "compilation_config", None)
    out: dict = {
        "model": _get(mc, "model"), "revision": _get(mc, "revision"), "model_commit": _commit(mc),
        "model_config": _config_hash(mc), "lora": _lora(cfg),
        "hf_overrides": _pairs(_get(mc, "hf_overrides")),
        "generation_config": "; ".join(x for x in (_s(_get(mc, "generation_config")),
                                                   _pairs(_get(mc, "override_generation_config"))) if x) or None,
        "dtype": _get(mc, "dtype"), "quantization": _get(mc, "quantization"),
        "kv_cache_dtype": _get(cfg, "cache_config", "cache_dtype"), "max_model_len": _get(mc, "max_model_len"),
        "tensor_parallel": _get(pc, "tensor_parallel_size"), "pipeline_parallel": _get(pc, "pipeline_parallel_size"),
        "data_parallel": _get(pc, "data_parallel_size"), "compilation": _get(cc, "mode"),
        "cudagraph": _get(cc, "cudagraph_mode"), "eager": _get(mc, "enforce_eager"),
        "rms_norm": _get(cfg, "kernel_config", "ir_op_priority", "rms_norm"),
    }
    if model is not None:
        attn, moe = set(), set()
        try:
            for _, m in model.named_modules():
                b = getattr(m, "attn_backend", None)
                if b is not None:
                    name = getattr(b, "get_name", None)
                    try:
                        attn.add(name() if callable(name) else getattr(b, "__name__", type(b).__name__))
                    except Exception:  # noqa: BLE001
                        attn.add(getattr(b, "__name__", type(b).__name__))
                qm = getattr(m, "quant_method", None)
                if type(m).__name__ in ("RoutedExperts", "FusedMoE") and qm is not None:
                    be = getattr(getattr(qm, "unquantized_backend", None), "name", None)
                    moe.add(be or type(qm).__name__)
        except Exception:  # noqa: BLE001  (best effort: a field left out isn't a wrong one)
            pass
        out["attention"], out["moe"] = sorted(attn) or None, sorted(moe) or None
        try:                                  # one pass over the weights, here at load only
            from . import serve_weights
            out["weights_digest"] = serve_weights.model_digest(model)
        except Exception:  # noqa: BLE001  (best effort: a digest that can't be taken is left out)
            pass
    try:
        import vllm
        out["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        out["torch"], out["cuda"] = torch.__version__, torch.version.cuda
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton
        out["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        from .gate import steering_env
        env = steering_env()
        out["env"] = ",".join(f"{k}={v}" for k, v in sorted(env.items())) if env else None
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import __version__
        out["sdcguard"] = __version__
    except Exception:  # noqa: BLE001
        pass
    if checks:
        out["checks"] = ",".join(f"{k}={v}" for k, v in sorted(checks.items()))
    clean = {k: _s(v) for k, v in out.items() if _s(v) not in (None, "")}
    clean["fingerprint"] = fingerprint(clean)
    return clean


def fingerprint(stack: dict) -> str:
    blob = json.dumps({k: stack.get(k) for k in FINGERPRINT_KEYS if k not in IDENTITY or stack.get(k)},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def shared(stack: Optional[dict]) -> Optional[str]:
    """What every worker of one server should agree on: the stack without its per-worker fields."""
    if not stack:
        return None
    return json.dumps({k: v for k, v in stack.items() if k not in PER_WORKER and k != "fingerprint"},
                      sort_keys=True, default=str)


_LINE = re.compile(r'^sdcguard_stack_info\{(.*)\}\s+\S+\s*$')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def from_metrics(text: str) -> Optional[dict]:
    """The sdcguard_stack_info labels on a worker's metrics page, or None."""
    for line in (text or "").splitlines():
        m = _LINE.match(line.strip())
        if m:
            return {k: v.replace('\\"', '"').replace("\\\\", "\\") for k, v in _LABEL.findall(m.group(1))}
    return None


def compare(qualified: dict, live: dict) -> dict:
    """{"same": True/False/None, "differences": [...]}: the fields each records, then the fingerprint. None (unknown,
    never the same) when an IDENTITY field is recorded on one side only (a record older than the field, or a value
    that couldn't be read), or when neither records the weights' digest: nothing then ties the stack to its
    weights."""
    if not qualified or not live:
        return {"same": None, "differences": [], "why": "no stack recorded" if not qualified else
                "the server publishes no stack (sdcguard_stack_info: the plugin isn't loaded, or is older)"}
    q, n = (lambda k: qualified.get(k) or None), (lambda k: live.get(k) or None)
    one_sided = [k for k in IDENTITY if (q(k) is None) != (n(k) is None)]
    # a field either side records (a metrics label left empty is a field not read)
    diffs = [f"{k}: {q(k)!r} qualified, {n(k)!r} now" for k in FINGERPRINT_KEYS
             if k not in one_sided and q(k) != n(k)]
    if diffs:
        return {"same": False, "differences": diffs, "why": "not the stack qualified: " + "; ".join(diffs)}
    unknown = [f"the record doesn't have {_WHAT[k]} (made by an older sdcguard, or it couldn't be read then); this "
               f"server's is {n(k)!r}" if q(k) is None else
               f"the server doesn't publish {_WHAT[k]} (an older plugin, or it couldn't be read at load); the "
               f"record's is {q(k)!r}" for k in one_sided]
    if q("weights_digest") is None and n("weights_digest") is None:
        unknown.append(f"neither the record nor the server has {_WHAT['weights_digest']}: nothing shows these are "
                       "the weights qualified")
    if unknown:
        return {"same": None, "differences": [], "unknown": unknown,
                "why": "can't be shown to be the stack qualified: " + "; ".join(unknown)}
    same_fp = qualified.get("fingerprint") is not None and qualified.get("fingerprint") == live.get("fingerprint")
    if same_fp and not diffs:
        return {"same": True, "differences": [], "why": f"the stack qualified (fingerprint {live['fingerprint']})"}
    if not diffs:
        diffs = [f"fingerprint: {qualified.get('fingerprint')!r} qualified, {live.get('fingerprint')!r} now "
                 "(a field the qualified record doesn't show)"]
    return {"same": False, "differences": diffs, "why": "not the stack qualified: " + "; ".join(diffs)}
