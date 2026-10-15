"""Add sdcguard to a training loop: check every matmul on every 20th step.

    python examples/quickstart.py        (uses your NVIDIA GPU if PyTorch sees one, else CPU)
"""
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from sdcguard import SDCGuard

device = "cuda" if torch.cuda.is_available() else "cpu"
# sdcguard checks bf16/fp16 matmuls. GPUs without native bf16 (like Colab's free T4) use fp16.
bf16_ok = device == "cpu" or torch.cuda.is_bf16_supported(including_emulation=False)
dtype = torch.bfloat16 if bf16_ok else torch.float16
model = nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 512)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
guard = SDCGuard().watch(model)          # watch() is optional: it names layers in alerts

for step in range(200):
    x = torch.randn(64, 512, device=device)
    checked = step % 20 == 0             # every 20th step: about +5.6% time on GPT-2 small (H100)
    with guard.at_step(step) if checked else nullcontext():
        with torch.autocast(device, dtype=dtype):
            loss = F.mse_loss(model(x).float(), x)
        loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if guard.log.events:                 # an alert: stop, find the GPU, roll back
        print(guard.log.events[0].to_json())
        break

print(f"checked {guard.stats['checked']} matmuls, {guard.stats['alerts']} alerts")
