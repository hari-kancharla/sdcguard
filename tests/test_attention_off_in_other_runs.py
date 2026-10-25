"""Attention checks are on by default since their GPU run (docs/METHOD.md); every other runner that starts vLLM with
the plugin sets SDCGUARD_ATTN to 0, since its criteria and results were written without attention checks."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TURNS_ON = re.compile(r'SDCGUARD="1"|"SDCGUARD": "1"|os\.environ(\.setdefault\(|\[)"SDCGUARD"')
# env["SDCGUARD_ATTN"] = "0", os.environ.setdefault("SDCGUARD_ATTN", "0"), {"SDCGUARD_ATTN": "0"}
SETS_OFF = re.compile(r'"SDCGUARD_ATTN"\]?\s*[=:,]\s*"0"')


def test_every_runner_that_turns_the_plugin_on_sets_sdcguard_attn():
    seen = []
    for d in ("experiments", "scripts"):
        for name in sorted(os.listdir(os.path.join(ROOT, d))):
            if not name.endswith(".py") or name == "vllm_attn_check.py":
                continue
            src = open(os.path.join(ROOT, d, name)).read()
            if TURNS_ON.search(src):
                seen.append(name)
                assert SETS_OFF.search(src), f"{d}/{name} turns the plugin on without setting SDCGUARD_ATTN to 0"
    assert {"vllm_moe_check.py", "vllm_norm_check.py", "vllm_serve_check.py", "vllm_check.py",
            "modal_vllm.py", "demo_vllm.py"} <= set(seen)
