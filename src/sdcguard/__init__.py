"""sdcguard: catch GPUs that silently compute wrong answers, in training, in serving and before use.

Training:
  SDCGuard        checksums on matrix multiplies (forward and backward); protect(model) = fast mode
  Canary          same batch on every GPU, layer-by-layer fingerprints, majority vote
  check_replicas  data-parallel copies of the weights must stay bit-identical
Serving:
  ServeGuard      checksums on every linear layer while a model serves (sdcguard.vllm_plugin for vLLM)
Checking and screening (command line):
  sdcguard check    is this GPU computing correctly, and at full speed? (five minutes, a shareable card)
  sdcguard screen   check GPUs with a real workload and keep replayable evidence
"""
__version__ = "0.4.0"

# Imported on first use, so commands that don't need PyTorch (`sdcguard nhc --cached`, `sdcguard board`)
# start instantly.
_LAZY = {
    "Canary": "canary", "CheckResult": "checksum", "check_linear": "checksum", "check_matmul": "checksum",
    "EventLog": "events", "SDCEvent": "events", "rollback_checkpoint": "events", "SDCError": "guard",
    "SDCGuard": "guard", "FaultInjector": "inject", "FaultModel": "inject", "corrupt_": "inject",
    "check_replicas": "replica", "weight_fingerprint": "replica", "ServeGuard": "serve",
}
__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        value = getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'sdcguard' has no attribute {name!r}")
