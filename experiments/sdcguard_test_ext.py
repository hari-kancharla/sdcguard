"""A vLLM worker extension for experiments/vllm_serve_check.py, test use only: `vllm serve ... --worker-extension-cls
sdcguard_test_ext.FaultInjection` adds this method to every GPU worker, and with VLLM_SERVER_DEV_MODE=1 the server's
/collective_rpc endpoint calls it in each one. It flips one high exponent bit of one weight of a layer, what a
memory error would do, in a server whose engine runs in other processes than the one sending requests."""


class FaultInjection:
    def sdcguard_test_flip(self, suffix: str) -> str:
        """Flip bit 14 of weight [0, 7] of the first linear layer whose name ends with `suffix` (with tensor
        parallelism, in this worker's shard). Returns the layer's name, or "" when there is none."""
        import torch
        for name, mod in self.get_model().named_modules():
            w = getattr(mod, "weight", None)
            if name.endswith(suffix) and isinstance(w, torch.Tensor) and w.dim() == 2 and w.element_size() == 2:
                with torch.no_grad():
                    w.data.view(torch.int16)[0, 7] ^= (1 << 14)
                torch.cuda.synchronize()
                return name
        return ""
