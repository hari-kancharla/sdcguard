"""Which Triton training-check kernel faults on Blackwell, and does aligning its buffers fix it?

On Oct 18, 2026 two B200s hit "CUDA error: misaligned address" in sdcguard's Triton training-check kernels
(_group.py; the serving kernel, cuBLAS and the plain checks were fine), so on compute capability 10 and up
sdcguard runs the plain PyTorch checks, which cost more (36% at every 20th step on a small model on Oct 19,
against 2.4% on an H100). This finds the faulting kernel: every launch is logged with the alignment of its
pointer arguments and followed by a synchronize, and each variant runs in its own process (the fault ends
the CUDA context):

  as_is        the kernels as they are, on the comparison cases of tests/_fast_compare.py
  aligned      the same, with every tile-result offset rounded up to 16 floats (64 bytes)

    python experiments/triton_blackwell.py                  # on a Blackwell GPU
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def child(variant: str) -> int:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import torch
    from sdcguard import _group

    def logged(name, kernel):
        class Launcher:
            def __getitem__(self, grid):
                def run(*args, **kw):
                    info = {"kernel": name, "grid": list(grid) if isinstance(grid, tuple) else grid,
                            "ptr_mod64": [a.data_ptr() % 64 for a in args if isinstance(a, torch.Tensor)],
                            "ints_mod16": [a % 16 for a in args if isinstance(a, int) and not isinstance(a, bool)]}
                    print("LAUNCH " + json.dumps(info), flush=True)
                    out = kernel[grid](*args, **kw)
                    torch.cuda.synchronize()
                    print("OK " + name, flush=True)
                    return out
                return run
        return Launcher()

    for name in ("_stage", "_compare", "_compare_part", "_finish", "_worst"):
        if hasattr(_group, name):
            setattr(_group, name, logged(name, getattr(_group, name)))
    if variant == "aligned":
        real_space = _group.Group._space

        def space(self, n):
            if self._buf is not None:
                self._used = (self._used + 15) // 16 * 16
            return real_space(self, (n + 15) // 16 * 16)
        _group.Group._space = space
    import _fast_compare
    return _fast_compare.main("cuda")


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return child(sys.argv[2])
    import torch
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "cc": torch.cuda.get_device_capability(0),
                      "torch": torch.__version__}), flush=True)
    results = {}
    for variant in ("as_is", "aligned"):
        env = dict(os.environ, SDCGUARD_TRITON_NEW_ARCH="1", CUDA_LAUNCH_BLOCKING="1", SDCGUARD_SIGN_SEED="1",
                   PYTHONPATH=os.path.join(ROOT, "src"))
        p = subprocess.run([sys.executable, __file__, "--child", variant], env=env, capture_output=True, text=True,
                           timeout=900)
        lines = p.stdout.splitlines()
        last = next((ln for ln in reversed(lines) if ln.startswith("LAUNCH ")), None)
        faulted = last is not None and not any(ln.startswith("OK ") for ln in lines[lines.index(last) + 1:])
        results[variant] = {"exit": p.returncode, "all_agree": "ALL AGREE" in p.stdout,
                            "faulting_launch": json.loads(last[7:]) if faulted else None,
                            "mismatches": [ln[:300] for ln in lines if '"agree": false' in ln][:10],
                            "error_tail": p.stderr[-1500:] if p.returncode else ""}
        print(variant, json.dumps(results[variant], default=str), flush=True)
    print("RESULT " + json.dumps(results, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
