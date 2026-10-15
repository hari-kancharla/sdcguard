"""sdcguard: catch GPUs that silently corrupt AI training, from inside the training run.

  SDCGuard        checksums on matrix multiplies (forward and backward)
  Canary          same batch on every GPU, layer-by-layer fingerprints, majority vote
  check_replicas  data-parallel copies of the weights must stay bit-identical
"""
__version__ = "0.1.0"

_LAZY = {
    "Canary": "canary", "CheckResult": "checksum", "check_linear": "checksum", "check_matmul": "checksum",
    "EventLog": "events", "SDCEvent": "events", "rollback_checkpoint": "events", "SDCError": "guard",
    "SDCGuard": "guard", "FaultInjector": "inject", "FaultModel": "inject", "corrupt_": "inject",
    "check_replicas": "replica", "weight_fingerprint": "replica",
}
__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'sdcguard' has no attribute {name!r}")
