# Round 3 results (October 18, 2026)

Raw outputs, copied from the runs. GPUs rented on Modal. PyTorch 2.14.1 (CUDA 13.0) and Triton 3.8.0,
except vLLM 0.31.0 (its own PyTorch 2.13).

| Folder | GPU | What |
|---|---|---|
| training/ | H100 80GB | fast training mode: slowdown on GPT-2 small and a 0.9B model, the old operator-level hook for comparison, bit-identity check, GPU self-test, Triton vs plain comparison |
| screen/ | H100 80GB | 3-minute screen (PASS), 30-second screen with simulated faults (FAIL, evidence saved) and a replay of that evidence |
| serving/ | H100 80GB | serve.json: the serving check (Triton kernel) vs the training check, its cost on 1.1B and 7B Llama-shaped models, false alarms, catches; pytest.txt: 136 tests passed; serve_first_version_pytorch.json: the first (PyTorch) version, which the Triton kernel replaced |
| vllm/ | H100 80GB | vLLM 0.31 with Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct: plain vLLM (vanilla), with the fp32 reduction setting (baseline), sdcguard on every layer (guarded) and on 1/8 of the layers; SUMMARY.json per model |
| nvbit/ | L4 | hardware-level fault injection with NVBit (TU Berlin's injector): bit 14, bit 4 and bit 1 of an input register of every tensor-core instruction on lane 0 of SM 0. first_try/: the first attempt (in mask_4096 the injector never switched on; in that try the training run also stepped into NaN weights) |

Note: in the 1.5B "every layer" vLLM run, Triton compiled kernels during the timed runs (since fixed: they
now compile when the model loads), so that run's speed isn't a steady-state number. Method: docs/METHOD.md.
