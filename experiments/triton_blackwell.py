"""Which Triton training-check kernel faults on Blackwell, and which change makes it run correctly?

On Oct 18, 2026 two B200s hit "CUDA error: misaligned address" in sdcguard's Triton training-check kernels
(_group.py; the serving kernel, cuBLAS and the plain checks were fine), so on compute capability 10 and up
sdcguard runs the plain PyTorch checks, which cost more (36% at every 20th step on a small model on Oct 19,
against 2.4% on an H100). The first bisect (Oct 19, results/blackwell_triton/b200_20261019_101117.json) found
the faulting kernel: _compare, on the first comparison case, with every pointer 64-byte aligned; rounding the
tile offsets up to 64 bytes changed nothing. This run tries one change at a time, each in its own process
(the fault ends the CUDA context), on every comparison case of tests/_fast_compare.py:

  as_is      the kernels as they are (expected to fault, the control)
  stages1    no software pipelining (num_stages=1): no asynchronous copies into shared memory
  warps4     4 warps for every tile shape
  fp32_b2    B's block sums as fp32 with TF32x3 dots instead of three bf16 parts (SDCGUARD_TRITON_NATIVE=0)
  ieee       plain fp32 dots, no tensor cores at all
  pad_a      A copied with its row length rounded up to 16 elements (32 bytes in bf16) when it isn't

For the as_is run it also saves the compiled _compare kernel (Triton IR and PTX), to read offline.

    python experiments/triton_blackwell.py                  # on a Blackwell GPU
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
VARIANTS = ("as_is", "stages1", "warps4", "fp32_b2", "ieee", "pad_a")


def child(variant: str, dump: str) -> int:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import torch
    from sdcguard import _group
    dumped = []

    def logged(name, kernel):
        class Launcher:
            def __getitem__(self, grid):
                def run(*args, **kw):
                    info = {"kernel": name, "grid": list(grid) if isinstance(grid, tuple) else grid,
                            "ptr_mod64": [a.data_ptr() % 64 for a in args if isinstance(a, torch.Tensor)],
                            "shapes": [list(a.shape) for a in args if isinstance(a, torch.Tensor)],
                            "ints": [a for a in args if isinstance(a, int) and not isinstance(a, bool)],
                            "const": {k: v for k, v in kw.items() if isinstance(v, (int, str, bool))}}
                    print("LAUNCH " + json.dumps(info), flush=True)
                    out = kernel[grid](*args, **kw)
                    if dump and name == "_compare" and not dumped:
                        dumped.append(1)
                        try:
                            for key in ("ttgir", "llir", "ptx"):
                                with open(os.path.join(dump, f"_compare.{key}"), "w") as f:
                                    f.write(str(out.asm[key]))
                        except Exception as e:  # noqa: BLE001
                            print("DUMP FAILED " + repr(e)[:200], flush=True)
                    torch.cuda.synchronize()
                    print("OK " + name, flush=True)
                    return out
                return run
        return Launcher()

    for name in ("_stage", "_compare", "_compare_part", "_finish", "_worst"):
        if hasattr(_group, name):
            setattr(_group, name, logged(name, getattr(_group, name)))
    if variant == "stages1":
        real = _group._cfg
        _group._cfg = lambda bj: (real(bj)[0], 1, real(bj)[2])
    elif variant == "warps4":
        real = _group._cfg
        _group._cfg = lambda bj: (real(bj)[0], real(bj)[1], 4)
    elif variant == "ieee":
        _group._precision = lambda dev: "ieee"
    elif variant == "pad_a":
        real_add = _group.Group.add

        def add(self, A, B, C, bias, alpha, beta):
            if A.stride(-1) == 1 and A.shape[-1] % 16:
                k = A.shape[-1]
                buf = torch.zeros(*A.shape[:-1], (k + 15) // 16 * 16, device=A.device, dtype=A.dtype)
                buf[..., :k] = A
                A = buf[..., :k]
            return real_add(self, A, B, C, bias, alpha, beta)
        _group.Group.add = add
    import _fast_compare
    return _fast_compare.main("cuda")


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return child(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    import torch
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "cc": torch.cuda.get_device_capability(0),
                      "torch": torch.__version__}), flush=True)
    try:
        import triton
        print(json.dumps({"triton": triton.__version__}), flush=True)
    except Exception:  # noqa: BLE001
        pass
    dump = os.path.join(ROOT, "blackwell_dump")
    os.makedirs(dump, exist_ok=True)
    results = {}
    for variant in VARIANTS:
        env = dict(os.environ, SDCGUARD_TRITON_NEW_ARCH="1", CUDA_LAUNCH_BLOCKING="1", SDCGUARD_SIGN_SEED="1",
                   PYTHONPATH=os.path.join(ROOT, "src"))
        if variant == "fp32_b2":
            env["SDCGUARD_TRITON_NATIVE"] = "0"
        p = subprocess.run([sys.executable, __file__, "--child", variant, dump if variant == "as_is" else ""],
                           env=env, capture_output=True, text=True, timeout=600)
        lines = p.stdout.splitlines()
        last = next((ln for ln in reversed(lines) if ln.startswith("LAUNCH ")), None)
        faulted = last is not None and not any(ln.startswith("OK ") for ln in lines[lines.index(last) + 1:])
        launches = [json.loads(ln[7:]) for ln in lines if ln.startswith("LAUNCH ")]
        results[variant] = {"exit": p.returncode, "all_agree": "ALL AGREE" in p.stdout,
                            "launches_ok": sum(ln.startswith("OK ") for ln in lines), "launches": len(launches),
                            "faulting_launch": json.loads(last[7:]) if faulted else None,
                            "mismatches": [ln[:300] for ln in lines if '"agree": false' in ln][:10],
                            "error_tail": p.stderr[-1200:] if p.returncode else ""}
        print(variant, json.dumps({k: v for k, v in results[variant].items() if k != "error_tail"}, default=str)[:1500],
              flush=True)
    files = {}
    for name in sorted(os.listdir(dump)):
        with open(os.path.join(dump, name)) as f:
            files[name] = f.read()[:400_000]
    print("RESULT " + json.dumps({"variants": results, "compiled_compare": files}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
