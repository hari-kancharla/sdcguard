"""A vLLM worker extension for experiments/vllm_serve_check.py, vllm_ready_attrib_check.py and the other server
experiments (vllm_norm_check.py, vllm_moe_check.py, vllm_attn_check.py), test use
only: `vllm serve ... --worker-extension-cls sdcguard_test_ext.FaultInjection` adds these methods to every GPU
worker, and with VLLM_SERVER_DEV_MODE=1 the server's /collective_rpc endpoint calls them in each one.

With SDCGUARD_TEST_EXACT=1 (a plain server, SDCGUARD=0, compared with a guarded one), importing this module in the
worker keeps PyTorch's bf16/fp16 cuBLAS partial sums in fp32, as the plugin does in every process it runs in
(vllm_plugin._exact_reductions): the two servers then differ only by sdcguard's checks."""
import os as _os

if _os.environ.get("SDCGUARD_TEST_EXACT") == "1":
    import torch as _torch
    _torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    _torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def _num(v: float):
    """A float as JSON can carry it: inf and NaN as their names (vLLM's /collective_rpc encodes its response with
    allow_nan=False, so a non-finite number would turn the whole call into an HTTP 500)."""
    import math
    v = float(v)
    return v if math.isfinite(v) else repr(v)


class FaultInjection:
    def sdcguard_test_flip(self, suffix: str) -> str:
        """Flip bit 14 of weight [0, 7] of the first linear layer whose name ends with `suffix` (with tensor
        parallelism, in this worker's shard), what a memory error would do; a second call flips it back. Returns
        the layer's name, or "" when there is none."""
        import torch
        for name, mod in self.get_model().named_modules():
            w = getattr(mod, "weight", None)
            if name.endswith(suffix) and isinstance(w, torch.Tensor) and w.dim() == 2 and w.element_size() == 2:
                with torch.no_grad():
                    w.data.view(torch.int16)[0, 7] ^= (1 << 14)
                torch.cuda.synchronize()
                return name
        return ""

    def sdcguard_test_scale(self, suffix: str, k: int) -> str:
        """Multiply weight [0, 7] of the first linear layer whose name ends with `suffix` by 2**k: exact in bf16 and
        fp16 (the exponent changes, the mantissa doesn't), and what flipping an exponent bit that was 0 does (k = 8
        is bit 10 of a bf16 weight). k = -8 afterwards restores it. Returns the layer's name, or ""."""
        import torch
        for name, mod in self.get_model().named_modules():
            w = getattr(mod, "weight", None)
            if name.endswith(suffix) and isinstance(w, torch.Tensor) and w.dim() == 2 and w.element_size() == 2:
                with torch.no_grad():
                    w.data[0, 7] *= 2.0 ** k
                torch.cuda.synchronize()
                return name
        return ""

    def sdcguard_test_flip_bit(self, name: str, row_frac: float, col_frac: float, bit: int) -> dict:
        """Flip one bit of one weight of the linear layer named `name` (exactly), at row int(row_frac * rows) and
        column int(col_frac * cols) of this worker's shard; a second call with the same arguments flips it back.
        Returns where, and the weight's value before and after."""
        import torch
        for n, mod in self.get_model().named_modules():
            w = getattr(mod, "weight", None)
            if n == name and isinstance(w, torch.Tensor) and w.dim() == 2 and w.element_size() == 2:
                r, c = int(row_frac * w.shape[0]) % w.shape[0], int(col_frac * w.shape[1]) % w.shape[1]
                with torch.no_grad():
                    before = float(w.data[r, c])
                    w.data.view(torch.int16)[r, c] ^= (1 << bit) if bit < 15 else -(1 << 15)
                    after = float(w.data[r, c])
                torch.cuda.synchronize()
                return {"layer": n, "row": r, "col": c, "bit": bit, "before": _num(before), "after": _num(after)}
        return {}

    def sdcguard_test_mark_rows(self, name: str, marker: int, delta: float = 64.0) -> str:
        """In eager mode (--enforce-eager): from now on, add `delta` to the first 4 outputs of the linear layer named
        `name` in every row whose input token is `marker`, after its GEMM and before sdcguard's check (a wrong
        result in exactly those rows, so the request that owns the row is known without the plugin's mapping).
        The embedding's input ids give the rows. sdcguard_test_unmark_rows undoes it. Returns the layer's name."""
        import torch
        model = self.get_model()
        mods = dict(model.named_modules())
        target = mods.get(name)
        emb = next((m for n, m in model.named_modules() if n.endswith("embed_tokens")), None)
        if target is None or emb is None or not hasattr(getattr(target, "quant_method", None), "_gemm_impl"):
            return ""
        state = {"rows": None}

        def pre(m, args):
            ids = args[0]
            state["rows"] = (ids.reshape(-1) == marker).nonzero().reshape(-1)
        handle = emb.register_forward_pre_hook(pre)
        qm = target.quant_method
        orig = qm._gemm_impl

        def gemm(layer, x, weight, bias=None):
            y = orig(layer, x, weight, bias)
            rows = state["rows"]
            if layer is target and rows is not None and rows.numel():
                y2 = y.view(-1, y.shape[-1])
                rows = rows[rows < y2.shape[0]]
                y2[rows, :4] += delta
            return y
        qm._gemm_impl = gemm
        self._sdcguard_marks = getattr(self, "_sdcguard_marks", []) + [(qm, orig, handle)]
        torch.cuda.synchronize()
        return name

    def sdcguard_test_unmark_rows(self) -> int:
        n = 0
        for qm, orig, handle in getattr(self, "_sdcguard_marks", []):
            qm._gemm_impl = orig
            handle.remove()
            n += 1
        self._sdcguard_marks = []
        return n

    def sdcguard_test_flip_norm_bit(self, name: str, col_frac: float, bit: int) -> dict:
        """Flip one bit of one weight of the norm layer named `name` (exactly), at int(col_frac * width); a second
        call with the same arguments flips it back. Returns where, and the weight's value before and after."""
        import torch
        for n, mod in self.get_model().named_modules():
            w = getattr(mod, "weight", None)
            if n == name and isinstance(w, torch.Tensor) and w.dim() == 1 and w.element_size() == 2:
                c = int(col_frac * w.shape[0]) % w.shape[0]
                with torch.no_grad():
                    before = float(w.data[c])
                    w.data.view(torch.int16)[c] ^= (1 << bit) if bit < 15 else -(1 << 15)
                    after = float(w.data[c])
                torch.cuda.synchronize()
                return {"layer": n, "col": c, "bit": bit, "before": _num(before), "after": _num(after)}
        return {}

    def sdcguard_test_norm_stats(self) -> dict:
        """The plugin's norm checks in this worker: layers checked and not, which check runs them (the Triton kernel
        after its first-use comparison, or the PyTorch one), calls checked so far, and the worst score read back per
        layer (a correct norm stays at or below 4; the line is 6)."""
        from sdcguard import vllm_plugin
        g = vllm_plugin._STATE.get("guard")
        nc = getattr(g, "norms", None)
        if nc is None:
            return {"norms": 0}
        worst = sorted(nc.worst_seen.items(), key=lambda kv: -kv[1])
        return {"norms": len(nc.layers), "line": nc.line, "kernel": {k: list(v) for k, v in nc.kernel_status().items()},
                "not_checked": dict(nc.unsupported), "stats": dict(nc.stats), "coverage": nc.coverage(),
                "worst": _num(worst[0][1]) if worst else None, "worst_layers": [[k, _num(v)] for k, v in worst[:5]],
                "names": [lay["name"] for lay in nc.layers]}

    def sdcguard_test_mark_norm_rows(self, name: str, marker: int) -> str:
        """In eager mode (--enforce-eager): from now on, make the first 4 outputs of the norm layer named `name`
        wrong (x 1.25 + 1) in every row whose input token is `marker`, after the norm and before sdcguard's check,
        so the request that owns the row is known without the plugin's mapping. sdcguard_test_unmark_norm_rows undoes
        it. Returns the layer's name, or "" when there is none or it isn't checked."""
        import torch
        from sdcguard.serve_norm import check_call
        model = self.get_model()
        mod = dict(model.named_modules()).get(name)
        emb = next((m for n, m in model.named_modules() if n.endswith("embed_tokens")), None)
        orig_cls = getattr(type(mod), "_sdcguard_norm_orig", None) if mod is not None else None
        if mod is None or emb is None or orig_cls is None or getattr(mod, "_sdcguard_norm", None) is None:
            return ""
        state = {"rows": None}

        def pre(m, args):
            state["rows"] = (args[0].reshape(-1) == marker).nonzero().reshape(-1)
        handle = emb.register_forward_pre_hook(pre)
        bound = mod._forward_method

        def forward(x, residual=None):
            out = orig_cls(mod, x, residual)
            y = out if residual is None else out[0]
            rows = state["rows"]
            if rows is not None and rows.numel():
                y2 = y.view(-1, y.shape[-1])
                rows = rows[rows < y2.shape[0]]
                y2[rows, :4] = y2[rows, :4] * 1.25 + 1
            args = getattr(mod, "_sdcguard_norm", None)
            if args is not None:
                check_call(args, x, residual, out)
            return out
        mod._forward_method = forward
        self._sdcguard_norm_marks = getattr(self, "_sdcguard_norm_marks", []) + [(mod, bound, handle)]
        torch.cuda.synchronize()
        return name

    def sdcguard_test_unmark_norm_rows(self) -> int:
        n = 0
        for mod, bound, handle in getattr(self, "_sdcguard_norm_marks", []):
            mod._forward_method = bound
            handle.remove()
            n += 1
        self._sdcguard_norm_marks = []
        return n

    def sdcguard_test_flip_expert_bit(self, name: str, which: str, expert: int, row_frac: float, col_frac: float,
                                      bit: int) -> dict:
        """Flip one bit of one weight of one expert of the MoE layer named `name` (which: "w13" or "w2"), at
        row int(row_frac * rows) and column int(col_frac * cols) of this worker's shard; a second call flips it
        back. Returns where, and the weight's value before and after."""
        import torch
        for n, mod in self.get_model().named_modules():
            w = getattr(mod, f"{which}_weight", None)
            if n == name and isinstance(w, torch.Tensor) and w.dim() == 3 and w.element_size() == 2:
                e = int(expert) % w.shape[0]
                r, c = int(row_frac * w.shape[1]) % w.shape[1], int(col_frac * w.shape[2]) % w.shape[2]
                with torch.no_grad():
                    before = float(w.data[e, r, c])
                    w.data.view(torch.int16)[e, r, c] ^= (1 << bit) if bit < 15 else -(1 << 15)
                    after = float(w.data[e, r, c])
                torch.cuda.synchronize()
                return {"layer": n, "which": which, "expert": e, "row": r, "col": c, "bit": bit,
                        "before": _num(before), "after": _num(after)}
        return {}

    def sdcguard_test_moe_stats(self) -> dict:
        """The plugin's MoE checks in this worker: layers checked and not, kernels, active calls, worst scores."""
        from sdcguard import serve_moe, vllm_plugin
        g = vllm_plugin._STATE.get("guard")
        mc = getattr(g, "moe", None)
        if mc is None:
            return {"layers": 0}
        worst = sorted(mc.worst_seen.items(), key=lambda kv: -kv[1])
        backends = sorted({str(getattr(getattr(getattr(m, "quant_method", None), "unquantized_backend", None), "name",
                                       type(getattr(m, "quant_method", None)).__name__))
                           for _, m in self.get_model().named_modules() if type(m).__name__ == "RoutedExperts"})
        return {"layers": len(mc.layers), "names": [lay["name"] for lay in mc.layers], "line": mc.line,
                "kernel": {f"{k[0]} {str(k[1]).replace('torch.', '')} act {k[2]}": list(v)
                           for k, v in serve_moe.STATUS.items()},
                "not_checked": dict(mc.unsupported), "stats": dict(mc.stats), "coverage": mc.coverage(),
                "worst": _num(worst[0][1]) if worst else None, "worst_layers": [[k, _num(v)] for k, v in worst[:5]],
                "backends": backends, "experts": int(mc.layers[0]["module"].w13_weight.shape[0]) if mc.layers else 0,
                "routing": self.sdcguard_test_route_stats()}

    def sdcguard_test_route_stats(self) -> dict:
        """The plugin's routing checks in this worker: routers checked and not, kernel, calls, rows, alerts."""
        from sdcguard import vllm_plugin
        g = vllm_plugin._STATE.get("guard")
        rc = getattr(g, "route", None)
        if rc is None:
            return {"routers": 0}
        return {"routers": len(rc.layers), "names": [lay["name"] for lay in rc.layers], "coverage": rc.coverage(),
                "stats": dict(rc.stats), "not_checked": dict(rc.unsupported)}

    def sdcguard_test_corrupt_route(self, name: str, mode: str) -> str:
        """In eager mode (--enforce-eager): from now on, make the MoE router named `name` (as the routing check names
        it: its runner's module name and ".router") wrong before sdcguard's check sees its result: "experts" (each
        row's last expert replaced by the next expert number) or "weights" (every routing weight 1.5 times what it
        should be). sdcguard_test_uncorrupt_route undoes it. Returns the name, or ""."""
        import torch
        mod = dict(self.get_model().named_modules()).get(name[:-len(".router")] if name.endswith(".router") else name)
        r = getattr(mod, "router", None)
        if r is None or mode not in ("experts", "weights") or "_select_experts" in vars(r):
            return ""
        orig = r._select_experts
        experts = int(getattr(r, "global_num_experts", 0) or 0)

        def _select_experts(hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
            w, ids = orig(hidden_states, router_logits, topk_indices_dtype=topk_indices_dtype, input_ids=input_ids)
            if not (isinstance(w, torch.Tensor) and isinstance(ids, torch.Tensor)):
                return w, ids
            if mode == "experts" and experts:
                ids = ids.clone()
                ids[:, -1] = (ids[:, -1] + 1) % experts
            elif mode == "weights":
                w = w * 1.5
            return w, ids

        r._select_experts = _select_experts
        self._sdcguard_route_faults = getattr(self, "_sdcguard_route_faults", []) + [r]
        return name

    def sdcguard_test_uncorrupt_route(self) -> int:
        n = 0
        for r in getattr(self, "_sdcguard_route_faults", []):
            vars(r).pop("_select_experts", None)
            n += 1
        self._sdcguard_route_faults = []
        return n

    def sdcguard_test_corrupt_moe(self, name: str, mode: str) -> str:
        """In eager mode (--enforce-eager): from now on, make the kernel of the MoE layer named `name` wrong in one of
        three ways, before sdcguard's check sees its output: "scale" (the routed output 1.5 times what it should be),
        "expert" (the kernel given each row's experts shifted by one, so it computes with the wrong ones while the
        check is given the right ones) or "row" (each row given the next row's output, as stale or mixed-up rows
        would be); or "drop": the first row of each batch reaches the experts with all its expert ids -1, after the
        router (and its routing check) produced them right, as a corrupted id buffer or a padding mask marking a real
        row would. sdcguard_test_uncorrupt_moe undoes it. Returns the layer's name, or ""."""
        import torch
        mod = dict(self.get_model().named_modules()).get(name)
        if mode == "drop" and callable(getattr(type(mod), "forward_modular", None)):
            cls_fm = type(mod).forward_modular

            def forward_modular(**kwargs):
                ids = kwargs.get("topk_ids")
                if isinstance(ids, torch.Tensor) and ids.shape[0]:
                    ids = ids.clone()
                    ids[0] = -1
                    kwargs = {**kwargs, "topk_ids": ids}
                return cls_fm(mod, **kwargs)
            mod.forward_modular = forward_modular                  # (vLLM 0.32 calls it with keyword arguments)
            self._sdcguard_moe_drops = getattr(self, "_sdcguard_moe_drops", []) + [mod]
            return name
        qm = getattr(mod, "quant_method", None)
        if qm is None or not hasattr(qm, "apply") or mode not in ("scale", "expert", "row"):
            return ""
        orig = qm.apply
        experts = int(mod.w13_weight.shape[0])

        def apply(*args, **kwargs):
            if mode == "expert" and isinstance(kwargs.get("topk_ids"), torch.Tensor):
                kwargs = {**kwargs, "topk_ids": (kwargs["topk_ids"] + 1) % experts}
            out = orig(*args, **kwargs)
            if not isinstance(out, torch.Tensor):
                return out
            if mode == "scale":
                return out * 1.5
            if mode == "row":
                return torch.roll(out, 1, 0)
            return out
        qm.apply = apply
        self._sdcguard_moe_faults = getattr(self, "_sdcguard_moe_faults", []) + [(qm, orig)]
        return name

    def sdcguard_test_uncorrupt_moe(self) -> int:
        n = 0
        for qm, orig in getattr(self, "_sdcguard_moe_faults", []):
            qm.apply = orig
            n += 1
        self._sdcguard_moe_faults = []
        for mod in getattr(self, "_sdcguard_moe_drops", []):
            mod.__dict__.pop("forward_modular", None)
            n += 1
        self._sdcguard_moe_drops = []
        return n

    def sdcguard_test_attn_stats(self) -> dict:
        """The plugin's attention checks in this worker: layers checked and not, backends, kernels, calls, samples,
        worst scores, writes checked, calls not checked by why."""
        from sdcguard import vllm_plugin
        g = vllm_plugin._STATE.get("guard")
        ac = getattr(g, "attn", None)
        if ac is None:
            return {"layers": 0}
        worst = sorted(ac.worst_seen.items(), key=lambda kv: -kv[1])
        return {"layers": len(ac.layers), "names": [lay["name"] for lay in ac.layers], "line": ac.line,
                "backends": sorted({lay["backend"] for lay in ac.layers}), "kernel": ac.kernel_status(),
                "not_checked": dict(ac.unsupported), "calls_not_checked": dict(ac.not_checked_calls),
                "stats": dict(ac.stats), "coverage": ac.coverage(), "cap_rows": ac.cap_rows,
                "worst": _num(worst[0][1]) if worst else None, "worst_layers": [[k, _num(v)] for k, v in worst[:5]]}

    def sdcguard_test_scan_stats(self) -> dict:
        """The plugin's weight scan in this worker: its coverage and counters ({"entries": 0} when it's off)."""
        from sdcguard import vllm_plugin
        ws = getattr(vllm_plugin._STATE.get("guard"), "wscan", None)
        if ws is None:
            return {"entries": 0}
        ws.sync()
        return {"entries": len(ws.entries), "coverage": ws.coverage(), "stats": dict(ws.stats)}

    def sdcguard_test_scan_pause(self, paused: bool) -> dict:
        """Pause (True) or resume the plugin's weight scan in this worker, to measure its cost on one server.
        {"scan": whether there is one, "paused": its state now} (a dict: vLLM's /collective_rpc returns other values
        as strings)."""
        from sdcguard import vllm_plugin
        ws = getattr(vllm_plugin._STATE.get("guard"), "wscan", None)
        if ws is None:
            return {"scan": False, "paused": None}
        ws.paused = bool(paused)
        return {"scan": True, "paused": bool(ws.paused)}

    def sdcguard_test_scan_refresh(self, name: str) -> dict:
        """A new fingerprint for the layer named `name` (after a trial flipped a bit back): the scan compares it again
        from a clean start. {"refreshed": how many scan entries were refreshed, -1 if the scan is off} (a dict:
        vLLM's /collective_rpc returns other values as strings)."""
        from sdcguard import vllm_plugin
        ws = getattr(vllm_plugin._STATE.get("guard"), "wscan", None)
        if ws is None:
            return {"refreshed": -1}
        for n, mod in self.get_model().named_modules():
            if n == name:
                return {"refreshed": int(ws.refresh(mod))}
        return {"refreshed": 0}

    def sdcguard_test_change_param(self, prefix: str, suffix: str, frac: float, op: str, bit: int = 0,
                                   index: int = -1) -> dict:
        """Change one parameter of the model in place, what a memory fault would do: the first parameter whose name
        starts with `prefix` and ends with `suffix` (a tied one under any of its names), at
        flat index int(frac * numel) of its raw integer values, or at `index` when it is 0 or more. op "flip": flip
        `bit` there; "swap": exchange the value there with the next one, starting from the first index from there
        whose next value differs (two values exchanged in one 4,096-value block, which the old block-sum
        fingerprint couldn't see). The same call with index= the returned index undoes it. Returns the parameter's
        name, the index, the scan entry that holds that value ("" if none), and the raw values before and after."""
        import torch
        from sdcguard import vllm_plugin
        for name, p in self.get_model().named_parameters(remove_duplicate=False):
            if not (name.startswith(prefix) and name.endswith(suffix)) or not p.numel():
                continue
            es = p.element_size()
            raw = p.data.view(-1).view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[es])
            n = raw.numel()
            i = index if index >= 0 else int(frac * n) % n
            with torch.no_grad():
                if op == "swap":
                    if index < 0:
                        end = min(n - 1, i + (1 << 20))
                        diff = (raw[i:end] != raw[i + 1:end + 1]).nonzero()
                        if not diff.numel():
                            return {"name": name, "error": "no two neighbouring values differ there"}
                        i += int(diff[0])
                    before = [int(raw[i]), int(raw[i + 1])]
                    raw[i], raw[i + 1] = before[1], before[0]
                    after = [int(raw[i]), int(raw[i + 1])]
                elif op == "flip":
                    before = [int(raw[i])]
                    raw[i] ^= (1 << bit) if bit < es * 8 - 1 or es == 1 else -(1 << bit)
                    after = [int(raw[i])]
                else:
                    return {"name": name, "error": f"op {op!r}"}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            addr = p.data.data_ptr() + i * es
            entry = ""
            ws = getattr(vllm_plugin._STATE.get("guard"), "wscan", None)
            if ws is not None:
                for e in ws.entries:
                    for t in e["get"]():
                        if isinstance(t, torch.Tensor) and t.numel() and \
                                t.data_ptr() <= addr < t.data_ptr() + t.numel() * t.element_size():
                            entry = e["name"]
                            break
                    if entry:
                        break
            return {"name": name, "index": i, "op": op, "bit": bit, "entry": entry, "before": before,
                    "after": after, "numel": n}
        return {}

    def sdcguard_test_scan_refresh_entry(self, name: str) -> dict:
        """A new fingerprint for the scan entry named `name` (after a trial undid its change). {"refreshed": how many
        entries were refreshed, -1 if the scan is off}."""
        from sdcguard import vllm_plugin
        ws = getattr(vllm_plugin._STATE.get("guard"), "wscan", None)
        if ws is None:
            return {"refreshed": -1}
        return {"refreshed": int(ws.refresh(name=name))}

    def sdcguard_test_scan_uncovered(self) -> dict:
        """The model's parameters (their tensors, as the plugin takes them) that no scan entry holds, and what the
        scan lists as not scanned: {"uncovered": [names], "not_scanned": {name: why}, "parameters": how many}."""
        import torch
        from sdcguard import vllm_plugin
        ws = getattr(vllm_plugin._STATE.get("guard"), "wscan", None)
        if ws is None:
            return {"uncovered": None, "not_scanned": None, "parameters": None}
        ws.sync()
        held = []
        for e in ws.entries:
            if e["excluded"]:
                continue
            for t in e["get"]():
                if isinstance(t, torch.Tensor) and t.numel():
                    held.append((str(t.device), t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
        out, count = [], 0
        for mname, mod in self.get_model().named_modules():
            for tname, t in vllm_plugin._module_tensors(mod):
                count += 1
                lo, hi = t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()
                # held whole, or in slices that together cover it
                parts = sorted((a, b) for d, a, b in held if d == str(t.device) and a < hi and b > lo)
                at = lo
                for a, b in parts:
                    if a > at:
                        break
                    at = max(at, b)
                if at < hi:
                    out.append(f"{mname}.{tname}" if mname else tname)
        return {"uncovered": out, "not_scanned": dict(ws.not_scanned), "parameters": count}

    def sdcguard_test_reductions(self) -> dict:
        """PyTorch's cuBLAS reduced-precision settings in this worker (False: bf16/fp16 partial sums kept in fp32)."""
        import torch
        m = torch.backends.cuda.matmul
        return {"bf16": bool(m.allow_bf16_reduced_precision_reduction),
                "fp16": bool(m.allow_fp16_reduced_precision_reduction)}

    def sdcguard_test_steps(self) -> dict:
        """The model steps the plugin has marked in this worker, and those with scheduled tokens."""
        from sdcguard import vllm_plugin
        return {"steps": int(vllm_plugin._STATE.get("steps") or 0),
                "token_steps": int(vllm_plugin._STATE.get("token_steps") or 0)}

    def sdcguard_test_corrupt_attn(self, name: str, mode: str) -> str:
        """In eager mode (--enforce-eager): from now on, make the attention of the layer named `name` (vLLM's
        layer_name) wrong before sdcguard's check sees it, in one of these ways:
          "value"    every output at dimension 5 of every head 2 * x + 1 (a faulty unit on every row)
          "head"     every output of query head 1 with bit 13 flipped (an exponent bit: 2^64 or 2^-64 times)
          "other"    the backend computing with the block tables of the batch's first two requests swapped (the
                     wrong request's keys and values, as a misplaced replay or table would give)
          "short"    the backend computing with every seq_len one short (each query leaves out its newest key, as a
                     stale length would)
          "kv"       after the layer writes the step's keys and values, bit 12 of dimension 9 of KV head 0's key
                     flipped at every slot written (a corrupted write)
        The backend's real forward runs, then the fault, then the check, as the plugin runs them.
        sdcguard_test_uncorrupt_attn undoes it. Returns the layer's name, or ""."""
        import copy
        import torch
        from sdcguard.serve_attn import check_call
        layer = next((m for _, m in self.get_model().named_modules() if getattr(m, "layer_name", None) == name
                      and hasattr(m, "impl")), None)
        if layer is None or mode not in ("value", "head", "other", "short", "kv"):
            return ""
        impl = layer.impl
        cls = type(impl)
        real = getattr(cls, "_sdcguard_attn_orig", None)
        if real is None or "forward" in vars(impl) or "do_kv_cache_update" in vars(impl):
            return ""

        def forward(layer_, query, key, value, kv_cache, md, *args, **kwargs):
            if md is None or mode == "kv":
                out = real(impl, layer_, query, key, value, kv_cache, md, *args, **kwargs)
            else:
                md2 = md
                if mode == "other" and md.block_table.shape[0] >= 2:
                    md2 = copy.copy(md)
                    md2.block_table = md.block_table.clone()
                    md2.block_table[[0, 1]] = md.block_table[[1, 0]]
                elif mode == "short":
                    md2 = copy.copy(md)
                    md2.seq_lens = (md.seq_lens - 1).clamp(min=0)
                out = real(impl, layer_, query, key, value, kv_cache, md2, *args, **kwargs)
                output = kwargs.get("output", args[0] if args else None)
                if mode == "value":
                    output[:, :, 5] = output[:, :, 5].float() * 2 + 1
                elif mode == "head":
                    output[:, 1].view(torch.int16).bitwise_xor_(1 << 13)
            a = getattr(layer_, "_sdcguard_attn", None)
            if a is not None and md is not None:
                check_call(a, impl, query, key, value, kv_cache, md, kwargs.get("output", args[0] if args else None),
                           kwargs.get("output_scale", args[1] if len(args) > 1 else None))
            return out

        impl.forward = forward
        if mode == "kv":
            write = cls.do_kv_cache_update

            def do_kv_cache_update(layer_, key, value, kv_cache, slot_mapping):
                write(impl, layer_, key, value, kv_cache, slot_mapping)
                n = kv_cache.shape[2]
                ok = slot_mapping >= 0
                s = torch.where(ok, slot_mapping, torch.zeros_like(slot_mapping))
                blk, off = s // n, s % n
                cur = kv_cache[blk, 0, off, 9]
                flipped = (cur.view(torch.int16) ^ (1 << 12)).view(cur.dtype)
                kv_cache[blk, 0, off, 9] = torch.where(ok, flipped, cur)

            impl.do_kv_cache_update = do_kv_cache_update
        self._sdcguard_attn_faults = getattr(self, "_sdcguard_attn_faults", []) + [impl]
        return name

    def sdcguard_test_uncorrupt_attn(self) -> int:
        n = 0
        for impl in getattr(self, "_sdcguard_attn_faults", []):
            vars(impl).pop("forward", None)
            vars(impl).pop("do_kv_cache_update", None)
            n += 1
        self._sdcguard_attn_faults = []
        return n

    def sdcguard_test_stop_reads(self) -> bool:
        """Make the plugin stop reading results back (its checks keep running on the GPU, nobody reads them), the
        way a broken read path would: what `sdcguard ready` must call NOT PROTECTED while vLLM serves."""
        from sdcguard import vllm_plugin
        g = vllm_plugin._STATE.get("guard")
        if g is None:
            return False
        g.poll = lambda *a, **k: []
        return True
