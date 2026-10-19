"""Device capability rules shared by GPU probes and the standard-library-only board."""


def expected(mode: str, cc: tuple) -> bool:
    """Whether the supported PyTorch path should run a mode at this compute capability."""
    need = {"bf16": (8, 0), "tf32": (8, 0), "fp8": (8, 9), "int8": (7, 5)}.get(mode, (0, 0))
    return tuple(cc) >= need
