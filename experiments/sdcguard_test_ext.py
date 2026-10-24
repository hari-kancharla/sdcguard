"""A vLLM worker extension for experiments/vllm_serve_check.py and experiments/vllm_ready_attrib_check.py, test use
only: `vllm serve ... --worker-extension-cls sdcguard_test_ext.FaultInjection` adds these methods to every GPU
worker, and with VLLM_SERVER_DEV_MODE=1 the server's /collective_rpc endpoint calls them in each one."""


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

    def sdcguard_test_stop_reads(self) -> bool:
        """Make the plugin stop reading results back (its checks keep running on the GPU, nobody reads them), the
        way a broken read path would: what `sdcguard ready` must call NOT PROTECTED while vLLM serves."""
        from sdcguard import vllm_plugin
        g = vllm_plugin._STATE.get("guard")
        if g is None:
            return False
        g.poll = lambda *a, **k: []
        return True
