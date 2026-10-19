"""How sdcguard takes over a linear layer's forward: by giving the layer a subclass of its own class.

The subclass's forward calls the function the guard stored on the layer (`_sdcguard_forward(layer, input)`),
or the original forward when there is none. Because the forward is a method, it always runs with the weights
of the layer it's called on, including layers the guard never saw:

- copy.deepcopy and pickling (torch.save(model), torch.optim.swa_utils.AveragedModel) make plain layers of the
  original class with their own weights, unchecked: a copy is a separate model.
- nn.DataParallel replicas (a shallow copy of the layer's attributes) keep the subclass and the guard, and
  compute with their own weights.

Until Oct 19, 2026 the forward was a function stored on the layer instance, closed over that layer: copies
ran the original layer's weights on every call (found by an independent review). PyTorch's parametrizations
(torch.nn.utils.parametrize) swap a module's class the same way. The subclass has the original's name, so a
model prints the same; `type(layer) is nn.Linear` is False while it is checked (isinstance still holds).
"""
from __future__ import annotations

import copyreg
import threading

import torch

KEY = "_sdcguard"
_CLASSES: dict = {}
_LOCK = threading.Lock()


def guarded_class(base: type) -> type:
    """The subclass of `base` whose forward goes through `_sdcguard_forward` (one per base class)."""
    with _LOCK:
        cls = _CLASSES.get(base)
        if cls is not None:
            return cls

        def forward(self, input):  # noqa: A002  (nn.Linear's argument name, so layer(input=x) works)
            fn = getattr(self, "_sdcguard_forward", None)
            if fn is None:
                return base.forward(self, input)
            return fn(self, input)

        def __reduce_ex__(self, protocol):
            # copies and pickles are plain layers of the original class, without the guard's attributes
            # (copyreg._reconstructor: object.__new__(target), then __setstate__; standard library only, so a
            # model saved while checked loads without sdcguard)
            target = base if type(self) is cls else type(self)
            state = {k: v for k, v in self.__dict__.items() if not k.startswith(KEY)}
            return copyreg._reconstructor, (target, object, None), state

        cls = type(base.__name__, (base,), {"forward": forward, "__reduce_ex__": __reduce_ex__,
                                            "_sdcguard_base": base, "__module__": base.__module__,
                                            "__qualname__": base.__qualname__, "__doc__": base.__doc__})
        _CLASSES[base] = cls
        return cls


def forwards() -> set:
    """The forward functions of every subclass made so far."""
    return {c.forward for c in _CLASSES.values()}


def is_guarded(mod: torch.nn.Module) -> bool:
    return any(c in _CLASSES.values() for c in type(mod).__mro__)


def take_over(mod: torch.nn.Module, fn) -> None:
    """Route mod's forward through fn(mod, input)."""
    if type(mod) not in _CLASSES.values():
        mod.__class__ = guarded_class(type(mod))
    mod.__dict__["_sdcguard_forward"] = fn


def give_back(mod: torch.nn.Module) -> None:
    """Undo take_over: the original class, and none of the guard's attributes. (A function that still holds
    the subclass's forward, such as a device hook added meanwhile, then runs the original forward.)"""
    cls = type(mod)
    base = getattr(cls, "_sdcguard_base", None)
    if base is not None and cls in _CLASSES.values():
        mod.__class__ = base
    for k in [k for k in mod.__dict__ if k.startswith(KEY)]:
        del mod.__dict__[k]


def multihead_out_projs(model: torch.nn.Module) -> dict:
    """{id(layer): why} for linear layers whose forward never runs: nn.MultiheadAttention computes with its
    out_proj's weight directly (F.multi_head_attention_forward), so a check on out_proj's forward would be
    listed as attached and never run."""
    out = {}
    for mod in model.modules():
        if isinstance(mod, torch.nn.MultiheadAttention) and isinstance(getattr(mod, "out_proj", None), torch.nn.Linear):
            out[id(mod.out_proj)] = ("inside nn.MultiheadAttention, which uses this weight directly (the layer's "
                                     "forward never runs)")
    return out


def fused_encoder_linears(model: torch.nn.Module) -> dict:
    """{id(layer): why} for the linear layers of nn.TransformerEncoderLayer, whose inference fast path (eval
    mode, no gradients) runs the whole layer as one fused PyTorch op without calling them."""
    out = {}
    for mod in model.modules():
        if isinstance(mod, torch.nn.TransformerEncoderLayer):
            for name in ("linear1", "linear2"):
                lin = getattr(mod, name, None)
                if isinstance(lin, torch.nn.Linear):
                    out[id(lin)] = ("inside nn.TransformerEncoderLayer, whose fused inference path doesn't call "
                                    "this layer's forward")
    return out
