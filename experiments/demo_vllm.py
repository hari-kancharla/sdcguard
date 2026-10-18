"""A 1-minute terminal demo: vLLM serving with sdcguard, then one bit of one weight flips in GPU memory.

    SDCGUARD=1 python experiments/demo_vllm.py          (scripts/modal_vllm.py --phase demo runs it on an H100)

What you see:
  1. vLLM loads Qwen2.5-1.5B-Instruct with the sdcguard plugin; it answers a few questions.
  2. "Flipping one bit of one weight in GPU memory" (what a memory error does; done here on purpose).
  3. The next answers come out wrong, and sdcguard's alert names the layer, the GPU (UUID, PCI slot)
     and whether the weights changed in memory.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDCGUARD", "1")
os.environ["SDCGUARD_POLL_EVERY"] = "1"                  # report as soon as possible in a demo
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # (its warm-up needs nvcc; see vllm_check.py)

QUESTIONS = ["What is the capital of France? Answer in one word.",
             "What is 12 times 12? Answer with the number only.",
             "Name the largest planet in the solar system. One word."]
LAYER = "model.layers.12.mlp.down_proj"


def say(msg):
    print(f"\n\033[1m>>> {msg}\033[0m", flush=True)
    time.sleep(1.0)


def flip(model):
    import torch
    for name, mod in model.named_modules():
        if name.endswith(LAYER):
            with torch.no_grad():
                mod.weight.data.view(torch.int16)[200, 300] ^= (1 << 14)
            torch.cuda.synchronize()
            return name


def report(_model):
    from sdcguard import vllm_plugin
    vllm_plugin.poll(wait=True)
    g = vllm_plugin.guard()
    return [(e.detail, e.identity) for e in g.log.events] if g else []


def main():
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    from sdcguard import vllm_plugin
    vllm_plugin.register()
    from vllm import LLM, SamplingParams
    say("Starting vLLM with sdcguard (every linear layer checked while it serves)")
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=1024)
    sp = SamplingParams(temperature=0.0, max_tokens=12)

    def ask():
        msgs = [[{"role": "user", "content": q}] for q in QUESTIONS]
        for q, o in zip(QUESTIONS, llm.chat(msgs, sp, use_tqdm=False)):
            print(f"  Q: {q}\n  A: {o.outputs[0].text.strip()!r}", flush=True)

    say("A healthy GPU:")
    ask()
    ev = llm.apply_model(report)[0]
    print(f"  sdcguard: {len(ev)} alerts", flush=True)
    say(f"Flipping ONE bit of ONE weight in GPU memory ({LAYER}), like a memory error would")
    llm.apply_model(flip)
    ask()
    ev = llm.apply_model(report)[0]
    say(f"sdcguard: {len(ev)} alert(s)")
    for d, i in ev[:3]:
        print(f"  ALERT {d['where']}: score {d['max_z']} (alarm line 6), row {d['row']}, block {d['col_block']}\n"
              f"        GPU {i.get('gpu_name')} {i.get('gpu_uuid')} PCI {i.get('pci_bus_id')}\n"
              f"        {d.get('reading')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
