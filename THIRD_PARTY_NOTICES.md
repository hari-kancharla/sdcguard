# Third-party notices

sdcguard is MIT licensed (LICENSE). It contains code from these projects, under their licenses:

## vLLM

`src/sdcguard/vllm_plugin.py`, function `machete_apply_weights`, is a modified copy of
`MacheteLinearKernel.apply_weights` from vLLM 0.31.0 and 0.32.0
(`vllm/model_executor/kernels/linear/mixed_precision/machete.py`), Copyright contributors to the vLLM project,
licensed under the Apache License, Version 2.0. The change: its call to `machete_mm` is replaced with sdcguard's
op, which runs the same GEMM and sdcguard's check together.

`src/sdcguard/vllm_plugin.py`, the `apply_weights` function in `_patch_fp8`, is a modified copy of
`FP8ScaledMMLinearKernel.apply_weights` from vLLM 0.31.0 and 0.32.0
(`vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py`), Copyright contributors to the vLLM project,
licensed under the Apache License, Version 2.0. The changes: an input vLLM quantized before the layer goes to
vLLM's own code unchanged, and after the input is quantized to FP8 sdcguard's quantization check is called.

The license text is in `licenses/vLLM-Apache-2.0.txt`.

The exact vLLM component method excerpts in `src/sdcguard/case_data/engine_generation/`
and `tests/fixtures/engine_generation/` come from commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`, Copyright contributors to the vLLM project,
under Apache License 2.0. Each directory includes `vllm-LICENSE.txt` and a manifest
with upstream paths, complete-source hashes and excerpt hashes. These unchanged
methods are executed with controlled CPU mocks for the bounded conformance command.

## SGLang

The exact SGLang component method excerpts in `src/sdcguard/case_data/engine_generation/`
and `tests/fixtures/engine_generation/` come from commit
`41fb719cf25966e6155f3e3d91f1157bc1a15123`, Copyright contributors to the SGLang project,
under Apache License 2.0. Each directory includes `sglang-LICENSE.txt` and a manifest
with upstream paths, complete-source hashes and excerpt hashes. These unchanged
methods are executed with controlled CPU mocks for the bounded conformance command.
