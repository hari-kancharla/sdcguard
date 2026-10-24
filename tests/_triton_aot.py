"""Compile sdcguard's Triton kernels for NVIDIA GPUs on a machine without one.

Triton's interpreter (TRITON_INTERPRET=1, what the CPU tests run the kernels in) doesn't type-check a kernel the way
the compiler does: the routing kernel ran there for days and then failed to compile on the first GPU it met
(October 24, 2026: `tl.atomic_cas` on an int64 pointer with an int32 literal). Triton's compiler needs no GPU to go
from the kernel to PTX and a cubin (its NVIDIA backend and ptxas ship with it), so `compiling()` replaces the
launch of every Triton kernel with a compile for the given GPU generations: the launch code runs as it would, with
the arguments it would pass, and a kernel that wouldn't compile raises here.

The arguments are bound and specialized by Triton's own binder for each target, as a real launch does (an int equal
to 1 becomes a constant, ints and pointers divisible by 16 get that attribute, `do_not_specialize` is honored). That
binder is internal to Triton (3.7 and 3.8 have it); without it, a plain typing of the arguments is used and
`FAITHFUL` is False."""
import contextlib
import inspect

import torch

TT_TYPES = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16", torch.float64: "fp64",
            torch.int64: "i64", torch.int32: "i32", torch.int16: "i16", torch.int8: "i8", torch.uint8: "u8",
            torch.bool: "i1", torch.float8_e4m3fn: "fp8e4nv"}
TARGETS = (80, 89, 90)          # A100, L4, H100/H200

try:
    from triton.compiler import make_backend
    from triton.runtime.jit import JITFunction as _JF
    from triton.runtime.jit import create_function_from_signature
    FAITHFUL = hasattr(_JF, "_pack_args")
except ImportError:             # no Triton, or one without these internals
    FAITHFUL = False


def _plain_signature(fn, args, kwargs):
    params = list(inspect.signature(fn.fn).parameters)
    known = {k: v for k, v in kwargs.items() if k in params}
    options = {k: v for k, v in kwargs.items() if k not in params and k in ("num_warps", "num_stages", "num_ctas")}
    bound = inspect.signature(fn.fn).bind(*args, **known)
    bound.apply_defaults()
    sig, consts = {}, {}
    for i, name in enumerate(params):
        v = bound.arguments[name]
        if i in fn.constexprs or v is None:
            sig[name] = "constexpr"
            consts[(i,)] = v
        elif isinstance(v, torch.Tensor):
            sig[name] = "*" + TT_TYPES[v.dtype]
        elif isinstance(v, bool):
            sig[name] = "i1"
        elif isinstance(v, int):
            sig[name] = "i32" if -2 ** 31 <= v < 2 ** 31 else "i64"
        elif isinstance(v, float):
            sig[name] = "fp32"
        else:
            raise TypeError(f"{fn.__name__}: argument {name} of type {type(v).__name__}")
    return sig, consts, None, options


_COMPILED: set = set()          # specializations already compiled in this process


@contextlib.contextmanager
def compiling(targets=TARGETS):
    """Inside: every Triton kernel launch compiles the kernel for each target instead (nothing runs). Yields the
    list of (kernel name, target) compiled."""
    import triton
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.runtime.jit import JITFunction

    done = []
    real = JITFunction.run
    seen = _COMPILED

    def run(self, *args, grid=None, warmup=False, **kwargs):
        for cc in targets:
            target = GPUTarget("cuda", cc, 32)
            if FAITHFUL:
                kw = dict(kwargs)
                kw["debug"] = kw.get("debug", self.debug) or knobs.runtime.debug
                kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
                backend = make_backend(target)
                binder = create_function_from_signature(self.signature, self.params, backend)
                bound_args, specialization, options = binder(*args, **kw)
                options, sig, consts, attrs = self._pack_args(backend, kw, bound_args, specialization, options)
                key = (id(self), cc, repr(sig), repr(consts), repr(attrs), repr(sorted(options.__dict__.items())))
                if key not in seen:          # the same specialization compiles to the same code
                    triton.compile(ASTSource(self, sig, consts, attrs), target=target, options=options.__dict__)
                    seen.add(key)
            else:
                sig, consts, attrs, options = _plain_signature(self, args, kwargs)
                key = (id(self), cc, repr(sig), repr(consts), repr(options))
                if key not in seen:
                    triton.compile(ASTSource(fn=self, signature=sig, constexprs=consts), target=target,
                                   options=options or None)
                    seen.add(key)
            done.append((self.__name__, cc))
        return None

    JITFunction.run = run
    try:
        yield done
    finally:
        JITFunction.run = real
