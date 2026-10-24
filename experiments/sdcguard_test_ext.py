"""A vLLM worker extension for experiments/vllm_serve_check.py and experiments/vllm_ready_attrib_check.py, test use
only: `vllm serve ... --worker-extension-cls sdcguard_test_ext.FaultInjection` adds these methods to every GPU
worker, and with VLLM_SERVER_DEV_MODE=1 the server's /collective_rpc endpoint calls them in each one."""


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

    def sdcguard_test_stop_reads(self) -> bool:
        """Make the plugin stop reading results back (its checks keep running on the GPU, nobody reads them), the
        way a broken read path would: what `sdcguard ready` must call NOT PROTECTED while vLLM serves."""
        from sdcguard import vllm_plugin
        g = vllm_plugin._STATE.get("guard")
        if g is None:
            return False
        g.poll = lambda *a, **k: []
        return True
