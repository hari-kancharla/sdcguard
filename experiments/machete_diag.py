"""Does sdcguard's Machete patch give vLLM's completions, and compile to vLLM's own kernels? (a diagnostic, round 2)

Round 1 (October 23, 2026; commit a56db68, results/machete_diag_oct23/) ran Qwen2.5-1.5B-Instruct-GPTQ-Int4 on
Machete with the patch taken apart: loading, probing and registering every layer, and calling Machete without its
bias then adding it in place or out of place, all gave vLLM's 64 completions exactly; every variant with the check as
an op of its own next to Machete changed 2 or 3 of them, on layers with a bias and on layers without one; rounding
the bias sum to bf16 changed them too (vLLM doesn't round it), and once it was rounded the checks changed nothing
more. The plugin now runs Machete's GEMM and the check as one op (vllm_plugin._machete_op), so vLLM's compiled graph
has one opaque call where it had machete_mm and nothing added around it; the op takes no tensor it writes (it finds
the guard's buffers by the layer's slot), so it is declared as mutating nothing.

Round 2 runs the same model in fresh processes, in this order:
  baseline        no sdcguard
  separate        the patch as it was in round 1: Machete, then the check as an op of its own, then y + bias
  fused           the patch as it is now
  fused_declared  the same, but with the buffers the check writes passed to the op and declared as mutated (an
                  alternative if fused doesn't match; on the CPU with PyTorch 2.8 it compiled to other code)
  baseline_fresh  no sdcguard, with an Inductor cache of its own: does vLLM compiling anew give its own completions?
  fused_fresh     fused, with an Inductor cache of its own: what a server with sdcguard compiles (sdcguard keeps
                  its own compile cache)
The first four share one Inductor cache (a kernel whose code is the same as one already compiled reuses the
launch config Triton's autotuning picked for it), as experiments/vllm_check.py's runs do. Every run records its
completions, Inductor's code for every graph it compiled (torch._inductor.config.trace: the graph after Inductor's
passes, the IR after fusion and the generated code), and which kernels it autotuned anew. compare() reports, for
each run against baseline: how many completions are the same, whether the same Triton kernels were generated
(their code with kernel names and comments taken out), whether each graph launches them and its opaque ops in the
same order (sdcguard::machete_checked counted as machete_mm), and the launch configs autotuning picked.

    python experiments/machete_diag.py --out DIR [--model M]          # every run, then the comparison
    python experiments/machete_diag.py --variant V --out DIR           # one run
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
RUNS = ("baseline", "separate", "fused", "fused_declared", "baseline_fresh", "fused_fresh")
SHARED = ("baseline", "separate", "fused", "fused_declared")   # one Inductor cache, in this order
PROMPTS = 64
TOKENS = 128
LOCAL = "/tmp/sdcguard_machete_diag"                            # Inductor caches and traces (local disk)
MACHETE_OPS = ("_C.machete_mm", "sdcguard.machete_checked", "sdcguard_diag.machete_checked_declared",
               "sdcguard_test.machete_mm")


# ---------------------------------------------------------------- Inductor's generated code
_KERNEL = re.compile(r"^(?P<name>\w+) = async_compile\.(?P<kind>\w+)\((?P<pre>[^\n]*?)'''(?P<src>.*?)'''",
                     re.M | re.S)
# torch.ops.<ns>.<op>[.<overload>]( and aten.<op>.<overload>( (Inductor's alias), extern_kernels.<name>(
_EXTERN = re.compile(r"\b(?:torch\.ops\.(\w+)|(aten))\.(\w+)(?:\.\w+)?\(|\bextern_kernels\.(\w+)\(")


def _normalize_kernel(name: str, pre: str, src: str) -> str:
    """A kernel's code without what changes when nothing it computes does: its own name (it numbers kernels in
    graph order), comment lines, the backend hash, trailing spaces."""
    text = (pre + "\n" + src).replace(name, "KERNEL")
    text = re.sub(r"'backend_hash': '[0-9A-Fa-f]+',?\s*", "", text)
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return "\n".join(lines)


def kernels_and_calls(code: str) -> dict:
    """The kernels one output_code.py defines ({id: normalized code}) and the order its call() runs kernels and
    opaque ops in: "K:<id>" for a kernel, "X:<namespace>.<op>" for a torch.ops call (any Machete call as
    X:MACHETE), "X:extern.<name>" for Inductor's extern kernels (mm, addmm, ...). Buffer names, allocations and
    asserts are left out."""
    kernels, ids, last = {}, {}, 0
    for m in _KERNEL.finditer(code):
        norm = _normalize_kernel(m.group("name"), m.group("pre"), m.group("src"))
        kid = hashlib.sha256(norm.encode()).hexdigest()[:12]
        kernels[kid] = norm
        ids[m.group("name")] = kid
        last = m.end()
    calls = []
    names = sorted(ids, key=len, reverse=True)
    launch = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")(?:\.run)?\(") if names else None
    for line in code[last:].splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        for m in _EXTERN.finditer(s):
            if m.group(4):
                calls.append(f"X:extern.{m.group(4)}")
            else:
                op = f"{m.group(1) or m.group(2)}.{m.group(3)}"
                calls.append("X:MACHETE" if op in MACHETE_OPS else f"X:{op}")
        if launch is not None:
            for m in launch.finditer(s):
                if not s.startswith(f"{m.group(1)} = "):
                    calls.append(f"K:{ids[m.group(1)]}")
    return {"kernels": kernels, "calls": calls}


def code_summary(trace_dir: str) -> dict:
    """Every graph Inductor compiled in one run (trace_dir/torchinductor/<graph>.<n>/), in compile order."""
    root = os.path.join(trace_dir, "torchinductor")
    graphs = []
    if os.path.isdir(root):
        dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        dirs.sort(key=lambda d: int(d.rsplit(".", 1)[-1]) if d.rsplit(".", 1)[-1].isdigit() else -1)
        for d in dirs:
            p = os.path.join(root, d, "output_code.py")
            if os.path.exists(p):
                with open(p, errors="replace") as f:
                    graphs.append({"dir": d, **kernels_and_calls(f.read())})
            else:
                graphs.append({"dir": d, "kernels": {}, "calls": None})     # an FX graph cache hit: no new code
    kernels = {}
    for g in graphs:
        kernels.update(g["kernels"])
    return {"graphs": len(graphs), "with_code": sum(g["calls"] is not None for g in graphs), "kernels": kernels,
            "sequences": sorted({tuple(g["calls"]) for g in graphs if g["calls"] is not None})}


def compare_code(base: dict, other: dict) -> dict:
    """other's generated code against base's: kernels only one of them generated, and call sequences (one per
    distinct compiled graph) only one of them has. same: code was captured for both, and nothing differs."""
    kb, ko = set(base["kernels"]), set(other["kernels"])
    sb, so = set(map(tuple, base["sequences"])), set(map(tuple, other["sequences"]))
    captured = base["with_code"] > 0 and other["with_code"] > 0
    return {"captured": captured, "kernels": len(ko), "kernels_base": len(kb),
            "kernels_only_here": sorted(ko - kb), "kernels_only_in_base": sorted(kb - ko),
            "sequences": len(so), "sequences_base": len(sb),
            "sequences_only_here": len(so - sb), "sequences_only_in_base": len(sb - so),
            "same": captured and kb == ko and sb == so}


def _write_code_diff(path: str, base: dict, other: dict, cmp: dict) -> None:
    """The kernels and graph sequences that differ, for reading."""
    with open(path, "w") as f:
        for title, ids, src in (("only in this run", cmp["kernels_only_here"], other["kernels"]),
                                ("only in baseline", cmp["kernels_only_in_base"], base["kernels"])):
            f.write(f"===== {len(ids)} kernels {title}\n")
            for kid in ids[:20]:
                f.write(f"----- {kid}\n{src[kid]}\n")
        sb, so = set(map(tuple, base["sequences"])), set(map(tuple, other["sequences"]))
        for title, seqs in (("only in this run", so - sb), ("only in baseline", sb - so)):
            f.write(f"===== {len(seqs)} graph sequences {title}\n")
            for s in sorted(seqs)[:10]:
                f.write(" ".join(s) + "\n")


# ---------------------------------------------------------------- the variants
_DECLARED = []


def _declared_op():
    """sdcguard::machete_checked's alternative for fused_declared: the buffers the check writes passed in and
    declared as mutated. Registered once per process. (torch.library reads the op's types from its annotations, so
    this module doesn't postpone their evaluation: torch is imported in here, after TORCHINDUCTOR_CACHE_DIR is set.)"""
    if _DECLARED:
        return _DECLARED[0]
    import torch
    from typing import Optional
    from sdcguard import vllm_plugin
    from sdcguard.serve import check_static_any

    try:
        @torch.library.custom_op("sdcguard_diag::machete_checked_declared",
                                 mutates_args=("state", "cap_meta", "cap_data"),
                                 tags=tuple(vllm_plugin._MACHETE["tags"] or ()))       # machete_mm's, as the plugin's
        def declared(a: torch.Tensor, b_q: torch.Tensor, b_type_id: int, b_group_scales: Optional[torch.Tensor],
                     b_group_zeros: Optional[torch.Tensor], b_group_size: Optional[int], parts: torch.Tensor,
                     pair_full: torch.Tensor, pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor,
                     b2max: torch.Tensor, bmax: torch.Tensor, bmax_bias: torch.Tensor, bsum: torch.Tensor,
                     wsq: torch.Tensor, width: torch.Tensor, state: torch.Tensor, slot: int, z: float, abs_floor: float,
                     cap_meta: torch.Tensor, cap_data: torch.Tensor, cap_w: int, cap_tag: int) -> torch.Tensor:
            y = vllm_plugin._MACHETE["mm"](a, b_q, b_type_id, None, b_group_scales, b_group_zeros, b_group_size, None,
                                           None, None)
            check_static_any(a, y, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum, wsq, width,
                             state, slot, z, abs_floor, cap_meta, cap_data, cap_w, cap_tag)
            return y

        @declared.register_fake
        def _(a, b_q, b_type_id, b_group_scales, b_group_zeros, b_group_size, parts, pair_full, pair_rem, scale, b2n,
              b2max, bmax, bmax_bias, bsum, wsq, width, state, slot, z, abs_floor, cap_meta, cap_data, cap_w, cap_tag):
            return torch.empty((a.size(0), b_q.size(1)), device=a.device, dtype=a.dtype)
    except RuntimeError:                  # already registered in this process (the module loaded twice)
        pass
    _DECLARED.append(torch.ops.sdcguard_diag.machete_checked_declared)
    return _DECLARED[0]


def _install(variant: str) -> None:
    """The plugin, then the variant's Machete apply_weights (fused: the plugin's own)."""
    import inspect
    import torch
    from vllm.model_executor.kernels.linear.mixed_precision import machete
    from sdcguard import vllm_plugin
    from sdcguard.serve import _op, check_static_any, check_static_impl
    vllm_plugin.register()
    K = machete.MacheteLinearKernel
    if vllm_plugin._MACHETE["op"] is None:
        raise SystemExit("sdcguard didn't set up its Machete op on this vLLM; see the warning above")
    orig = inspect.signature(K.apply_weights).parameters["_orig"].default         # vLLM's own apply_weights
    if variant == "fused_declared":
        declared = _declared_op()

        def apply_weights(self, layer, x, bias=None):          # the plugin's, with the declared op
            a = getattr(layer, "_sdcguard_args", None)
            if a is None:
                return orig(self, layer, x, bias)
            c = self.config
            w_q, w_s, w_zp = self._get_weight_params(layer)
            x_2d = x.reshape(-1, x.shape[-1])
            out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)
            if not c.zero_points:
                w_zp = None
            if torch.compiler.is_compiling():
                output = declared(x_2d, w_q, c.weight_type.id, w_s, w_zp, c.group_size, *a[0])
            else:
                output = vllm_plugin._MACHETE["ops"].machete_mm(a=x_2d, b_q=w_q, b_type=c.weight_type,
                                                                b_group_zeros=w_zp, b_group_scales=w_s,
                                                                b_group_size=c.group_size)
                check_static_any(x_2d, output, *a[0])
            if bias is not None:
                output.add_(bias)
            return output.reshape(out_shape)

        K.apply_weights = apply_weights
    elif variant == "separate":
        op = _op()

        def apply_weights(self, layer, x, bias=None):
            a = getattr(layer, "_sdcguard_args", None)
            if a is None:
                return orig(self, layer, x, bias)
            y = orig(self, layer, x, None)
            if torch.compiler.is_compiling():
                fn = check_static_impl if getattr(layer, "_sdcguard_trace", False) else op
            else:
                fn = check_static_any
            fn(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), *a[0])
            return y + bias if bias is not None else y

        K.apply_weights = apply_weights


def _trace(directory: str) -> None:
    """Inductor's debug trace for every graph it compiles: the graph after its passes, the IR after fusion and the
    generated code (not the input graph or the IR before fusion)."""
    import torch._inductor.config as ic
    t = ic.trace
    t.enabled, t.debug_dir = True, directory
    t.fx_graph, t.fx_graph_transformed, t.ir_pre_fusion, t.ir_post_fusion, t.output_code = (False, True, False,
                                                                                             True, True)
    t.graph_diagram = t.draw_orig_fx_graph = False


# what PyTorch 2.13 saves in a .best_config besides the launch config (autotune_cache.AutotuneCache.save): the
# time the benchmarking took and the Triton cache hash differ between runs that picked the same config
_NOT_THE_CONFIG = ("time_taken_ms", "triton_cache_hash", "found_by_coordesc", "configs_hash")


def _launch_config(saved):
    """The launch config autotuning picked (block sizes, num_warps, num_stages, ...), from a .best_config."""
    return {k: v for k, v in saved.items() if k not in _NOT_THE_CONFIG} if isinstance(saved, dict) else saved


def _autotuned(cache: str, since: float) -> dict:
    """{best_config file: launch config} for every kernel whose config Triton's autotuning saved in this cache since
    `since`. Inductor writes <key>.best_config next to the kernel's code, the key made from the kernel file's name
    (a hash of its code) and the torch version, so the same kernel has the same name in every cache."""
    found = {}
    for root, _dirs, files in os.walk(cache):
        for name in files:
            if name.endswith(".best_config"):
                p = os.path.join(root, name)
                if os.path.getmtime(p) >= since:
                    try:
                        with open(p) as f:
                            cfg = _launch_config(json.load(f))
                    except (OSError, ValueError):
                        cfg = None
                    found[name[:-len(".best_config")]] = cfg
    return found


def run(variant: str, out: str, model: str) -> int:
    started = time.time()                               # what this process autotunes is newer than this
    cache = os.path.join(LOCAL, "inductor_shared" if variant in SHARED else f"inductor_{variant}")
    trace = os.path.join(LOCAL, "trace", variant)
    shutil.rmtree(trace, ignore_errors=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache       # before torch is imported; vLLM leaves it alone when its
    # own compile cache is off (VLLM_DISABLE_COMPILE_CACHE=1 below: no AOT compile, standalone_compile)
    sys.path.insert(0, HERE)
    import vllm_check
    os.environ["SDCGUARD"] = "0" if variant.startswith("baseline") else "1"
    os.environ["SDCGUARD_POLL_EVERY"] = "1000000000"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False      # (as vllm_check's baseline)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    _trace(trace)
    if not variant.startswith("baseline"):
        _install(variant)
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=TOKENS, ignore_eos=True)
    prompts = vllm_check._prompts(PROMPTS)
    llm.generate(prompts[:16], sp, use_tqdm=False)                              # warm-up, as vllm_check
    outs = llm.generate(prompts, sp, use_tqdm=False)
    light = llm.generate(prompts[:16], sp, use_tqdm=False)
    import vllm
    res = {"variant": variant, "model": model, "seconds": round(time.time() - t0, 1),
           "torch": torch.__version__, "vllm": getattr(vllm, "__version__", None),
           "inductor_cache": "shared" if variant in SHARED else "its own",
           "outputs": [o.outputs[0].text for o in outs], "light_outputs": [o.outputs[0].text for o in light],
           "autotuned": _autotuned(cache, started)}
    if not variant.startswith("baseline"):
        from sdcguard import vllm_plugin
        res["layout_tags"] = {"machete_mm": [str(t) for t in vllm_plugin._layout_tags(vllm_plugin._MACHETE["mm"])],
                              "op": [str(t) for t in vllm_plugin._layout_tags(vllm_plugin._MACHETE["op"])]}

        def counts(_m):
            g = vllm_plugin._STATE.get("guard")
            if g is None:
                return None
            c = g.counts()
            return {"layers": len(g.layers), "layers_with_checks": sum(1 for v in c.values() if v["checks"] > 0)}
        res["checks"] = llm.apply_model(counts)
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, "code", variant)
    shutil.rmtree(dest, ignore_errors=True)
    if os.path.isdir(trace):
        shutil.copytree(trace, dest)
    with open(os.path.join(out, f"{variant}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: (len(v) if k == "autotuned" else v) for k, v in res.items() if "outputs" not in k}),
          flush=True)
    return 0


# ---------------------------------------------------------------- the comparison
def _ok(v: dict) -> bool:
    """The decision rule written in docs/METHOD.md: every completion the same as baseline's, and, where code was
    captured, the same kernels in the same order; and every registered layer ran checks."""
    code = v.get("code") or {}
    checks = [c for c in (v.get("checks") or []) if c]
    checked = bool(checks) and all(c["layers"] > 0 and c["layers_with_checks"] == c["layers"] for c in checks)
    return bool(v.get("all_same")) and (not code.get("captured") or bool(code.get("same"))) and checked


def compare(out: str) -> dict:
    """Each run against baseline: completions, generated code, autotuned launch configs; and the fix it points to
    (fused if it meets the rule, else fused_declared if that does, else none)."""
    base = json.load(open(os.path.join(out, "baseline.json")))
    base_code = code_summary(os.path.join(out, "code", "baseline"))
    base_tuned = base.get("autotuned") or {}
    summary = {"model": base["model"], "baseline_code": {k: base_code[k] for k in ("graphs", "with_code")},
               "baseline_kernels": len(base_code["kernels"]), "runs": {}}
    for v in RUNS[1:]:
        p = os.path.join(out, f"{v}.json")
        if not os.path.exists(p):
            summary["runs"][v] = {"error": "no result"}
            continue
        r = json.load(open(p))
        same = sum(a == b for a, b in zip(base["outputs"], r["outputs"]))
        lsame = sum(a == b for a, b in zip(base["light_outputs"], r["light_outputs"]))
        code = code_summary(os.path.join(out, "code", v))
        cmp = compare_code(base_code, code)
        _write_code_diff(os.path.join(out, f"code_diff_{v}.txt"), base_code, code, cmp)
        tuned = r.get("autotuned") or {}
        entry = {"same": f"{same}/{len(base['outputs'])}", "light_same": f"{lsame}/{len(base['light_outputs'])}",
                 "all_same": same == len(base["outputs"]) and lsame == len(base["light_outputs"]),
                 "differing": [i for i, (a, b) in enumerate(zip(base["outputs"], r["outputs"])) if a != b],
                 "inductor_cache": r.get("inductor_cache"), "checks": r.get("checks"),
                 "code": {"graphs": code["graphs"], "with_code": code["with_code"],
                          **{k: (len(x) if isinstance(x, list) else x) for k, x in cmp.items()}},
                 "autotuned_here": len(tuned)}
        if v not in SHARED:   # compiled anew: the configs autotuning picked for kernels baseline also autotuned
            both = set(tuned) & set(base_tuned)
            entry["autotuned_like_baseline"] = sum(tuned[k] == base_tuned[k] for k in both)
            entry["autotuned_unlike_baseline"] = sorted(k for k in both if tuned[k] != base_tuned[k])
        summary["runs"][v] = entry
    runs = summary["runs"]
    summary["fix"] = next((v for v in ("fused", "fused_declared") if v in runs and "error" not in runs[v] and
                           _ok(runs[v])), None)
    with open(os.path.join(out, "DIAG.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--variant", choices=RUNS)
    a = p.parse_args()
    if a.variant:
        return run(a.variant, a.out, a.model)
    shutil.rmtree(LOCAL, ignore_errors=True)
    codes = {}
    for v in RUNS:
        cmd = [sys.executable, os.path.abspath(__file__), "--variant", v, "--out", a.out, "--model", a.model]
        print("$", " ".join(cmd), flush=True)
        try:
            codes[v] = subprocess.run(cmd, timeout=15 * 60).returncode
        except subprocess.TimeoutExpired:
            codes[v] = 124
        print(f"{v}: exit code {codes[v]}", flush=True)
        if v == "baseline" and codes[v] != 0:
            break                                   # nothing to compare against
    s = compare(a.out) if codes.get("baseline") == 0 else {"fix": None}
    s["exit_codes"] = codes
    with open(os.path.join(a.out, "DIAG.json"), "w") as f:
        json.dump(s, f, indent=1)
    # the decision needs baseline and fused (the others are reported: their failures are in exit_codes)
    return 0 if codes.get("baseline") == 0 and codes.get("fused") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
