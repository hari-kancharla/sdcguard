"""Edge cases from an outside code review: rounding that lines up with the sign pattern, layers with their
own forward, empty batches, certificate dates, result cards whose verdict contradicts their counts."""
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from sdcguard import SDCGuard
from sdcguard import checksum as cs
from sdcguard.serve import ServeGuard

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")


@pytest.fixture
def plain_check():
    old = cs.USE_FAST
    cs.USE_FAST = False
    yield
    cs.USE_FAST = old


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_outputs_aligned_with_a_fixed_sign_pattern_dont_alarm(plain_check, dtype):
    """B's columns proportional to a fixed +-1 pattern make every output round the same way relative
    to that pattern. The signs are drawn per process, so data can't line up with them."""
    g = torch.Generator().manual_seed(20261014 + 128)
    fixed = (torch.randint(0, 2, (128,), generator=g) * 2 - 1).float()
    a = torch.tensor([[-0.39453125]], dtype=dtype)
    b = (fixed[None, :] * 0.1923828125).to(dtype)
    c = (a.float() @ b.float()).to(dtype)
    r = cs.check_matmul(a, b, c)
    assert r.ok, r


def _signs_in_new_process(env_extra=None):
    env = dict(os.environ, PYTHONPATH=SRC)
    env.pop("SDCGUARD_SIGN_SEED", None)
    env.update(env_extra or {})
    code = "from sdcguard import checksum as cs; print(cs._signs(16, 'cpu').int().tolist())"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          check=True).stdout.strip()


def test_sign_pattern_is_drawn_per_process_unless_pinned():
    assert _signs_in_new_process({"SDCGUARD_SIGN_SEED": "7"}) == _signs_in_new_process({"SDCGUARD_SIGN_SEED": "7"})
    draws = {_signs_in_new_process() for _ in range(3)}
    assert len(draws) > 1


class PlusOne(nn.Linear):
    def forward(self, x):
        return super().forward(x) + 1


def _ones(layer):
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()
    return layer


def test_fast_mode_leaves_layers_with_their_own_forward_alone():
    model = nn.Sequential(_ones(PlusOne(4, 4)), _ones(nn.Linear(4, 4))).to(torch.bfloat16)
    x = torch.ones(1, 4, dtype=torch.bfloat16)
    before = model(x)
    g = SDCGuard(every=1).protect(model)
    with g.on_step(0):
        after = model(x)
    assert torch.equal(before, after)
    assert list(g.skipped) == ["0"] and "own forward" in g.skipped["0"]
    assert model[1] in g._protected


def test_fast_mode_keeps_a_forward_set_on_the_instance():
    lin = _ones(nn.Linear(2, 2))
    orig = lin.forward
    lin.forward = lambda inp: orig(inp) * 10
    model = nn.Sequential(lin)
    x = torch.ones(1, 2)
    before = model(x)
    g = SDCGuard(every=1).protect(model)
    with g.on_step(0):
        assert torch.equal(model(x), before)
    g.unprotect()
    assert torch.equal(model(x), before)
    assert "instance" in g.skipped["0"]


def test_serving_skips_layers_with_their_own_forward():
    model = nn.Sequential(_ones(PlusOne(256, 256)), nn.Linear(256, 256)).to(torch.bfloat16)
    x = torch.ones(3, 256, dtype=torch.bfloat16)
    before = model(x)
    sg = ServeGuard().attach(model)
    assert torch.equal(model(x), before)
    assert list(sg.unsupported) == ["0"] and sg.stats["layers"] == 1
    assert sg.poll(wait=True) == []
    sg.detach()


def _accelerate_style_hook(lin, calls):
    """What accelerate's add_hook_to_module does: keep the real forward as _old_forward, set a wrapper on the
    instance that calls the hook's pre_forward, _old_forward and post_forward."""
    import functools

    class Hook:
        no_grad = False

        def pre_forward(self, module, *args, **kwargs):
            calls.append("pre")
            return args, kwargs

        def post_forward(self, module, output):
            calls.append("post")
            return output

    lin._hf_hook = Hook()
    lin._old_forward = lin.forward

    def new_forward(module, *args, **kwargs):
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
        return module._hf_hook.post_forward(module, module._old_forward(*args, **kwargs))
    lin.forward = functools.partial(new_forward, lin)
    return lin


def test_serving_checks_underneath_an_accelerate_device_hook_and_restores_it():
    torch.manual_seed(0)
    calls = []
    lin = _accelerate_style_hook(nn.Linear(256, 256).to(torch.bfloat16), calls)
    hooked, old = lin.forward, lin._old_forward
    model = nn.Sequential(lin)
    x = torch.randn(4, 256).to(torch.bfloat16)
    before = model(x)
    sg = ServeGuard().attach(model)
    assert sg.stats["layers"] == 1 and sg.unsupported == {}
    calls.clear()
    with torch.no_grad():
        assert torch.equal(model(x), before)
    assert calls == ["pre", "post"] and sg.worst_z()["0"] > 0 and sg.poll(wait=True) == []
    with torch.no_grad():
        lin.weight.view(torch.int16)[5, 9] ^= 1 << 14          # a bit flip in the weights after attach
        model(x)
    assert len(sg.poll(wait=True)) == 1
    sg.detach()
    assert lin.forward is hooked and lin._old_forward == old


def test_serving_compiled_right_after_attach_still_checks_a_hooked_layer():
    """No first-call probe to miss under torch.compile (outside review, Oct 19, 2026)."""
    torch.manual_seed(0)
    lin = _accelerate_style_hook(nn.Linear(128, 128, bias=False).to(torch.bfloat16), [])
    sg = ServeGuard().attach(lin)
    compiled = torch.compile(lin, backend="eager")
    x = torch.randn(2, 128).to(torch.bfloat16)
    with torch.no_grad():
        compiled(x)
        assert sg.worst_z()["Linear"] > 0 and sg.poll(wait=True) == []
        lin.weight[0, 0] = 1000
        compiled(x)
    assert len(sg.poll(wait=True)) == 1
    sg.detach()


@pytest.mark.parametrize("case", ["plain wrapper", "batch dependent", "fp16 scale"])
def test_serving_lists_any_other_instance_forward_and_never_alarms_on_it(case):
    """A check of one sample can't prove an instance forward is a plain linear layer for every input (an outside
    review found a wrapper that was linear for one row and doubled larger batches, and an fp16 scale of 1.0059
    inside bf16's tolerance): such layers are left alone and listed."""
    dt = torch.float16 if case == "fp16 scale" else torch.bfloat16
    lin = nn.Linear(128, 128, bias=False).to(dt)
    with torch.no_grad():
        lin.weight.copy_(torch.eye(128).to(dt))
    plain = lin.forward
    lin.forward = {"plain wrapper": lambda x: plain(x),
                   "batch dependent": lambda x: plain(x) if x.shape[0] == 1 else plain(x) * 2,
                   "fp16 scale": lambda x: plain(x) * 1.005859375}[case]
    own = lin.forward
    sg = ServeGuard().attach(lin)
    assert sg.stats["layers"] == 0 and "not checked" in sg.unsupported["Linear"]
    with torch.no_grad():
        for rows in (1, 2, 5):
            lin(torch.ones(rows, 128).to(dt))
    assert sg.poll(wait=True) == []
    sg.detach()
    assert lin.forward is own


def test_serving_with_the_real_accelerate_hook():
    accelerate = pytest.importorskip("accelerate")
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module
    torch.manual_seed(0)
    lin = nn.Linear(256, 256).to(torch.bfloat16)
    add_hook_to_module(lin, AlignDevicesHook(execution_device="cpu", io_same_device=True))
    model = nn.Sequential(lin)
    x = torch.randn(4, 256).to(torch.bfloat16)
    before = model(x)
    sg = ServeGuard().attach(model)
    assert sg.stats["layers"] == 1 and sg.unsupported == {}, accelerate.__version__
    with torch.no_grad():
        assert torch.equal(model(x), before)
        assert sg.worst_z()["0"] > 0 and sg.poll(wait=True) == []
        lin.weight.view(torch.int16)[5, 9] ^= 1 << 14
        model(x)
    assert len(sg.poll(wait=True)) == 1
    sg.detach()
    remove_hook_from_module(lin)                     # leaves the bound nn.Linear.forward on the instance
    sg = ServeGuard().attach(model)
    assert sg.stats["layers"] == 1 and sg.unsupported == {}
    sg.detach()


def test_serving_lists_offloaded_layers():
    lin = nn.Linear(64, 64, device="meta", dtype=torch.bfloat16)
    sg = ServeGuard().attach(nn.Sequential(lin))
    assert sg.stats["layers"] == 0 and "offloaded" in sg.unsupported["0"]


def test_serving_handles_an_empty_batch():
    lin = nn.Linear(256, 256).to(torch.bfloat16)
    sg = ServeGuard().attach(nn.Sequential(lin))
    out = lin(torch.empty(0, 256, dtype=torch.bfloat16))
    assert out.shape == (0, 256)
    sg.detach()


def test_certificate_with_an_unreadable_expiry_is_not_ok(tmp_path):
    pytest.importorskip("cryptography")
    from sdcguard import certify
    key, pub = certify.keygen(str(tmp_path / "k"))
    cert = certify.make({"result": "PASS"}, issuer="test")
    cert["valid_until"] = "not a date"
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(certify.sign(cert, key)))
    v = certify.verify(str(path), pub)
    assert v["ok"] is False and "no valid_until date" in v["problems"]


def _card(**changes):
    from sdcguard import board
    seed = os.path.join(HERE, "..", "docs", "board", "seed")
    with open(os.path.join(seed, sorted(os.listdir(seed))[0])) as f:
        r = json.load(f)
    for k, v in changes.items():
        r[k] = v
    r.pop("digest", None)
    r["digest"] = board.digest(r)
    return r


def test_board_rejects_cards_whose_verdict_contradicts_their_counts():
    from sdcguard import board
    assert board.validate(_card()) is None
    assert "PASS" in board.validate(_card(checked={"results": 0}))
    assert "PASS" in board.validate(_card(wrong={"stress": 2}))
    assert "FAIL" in board.validate(_card(result="FAIL"))
    every = {"workload_training": 0, "workload_serving": 0, "stress": 0, "memory_words": 0}
    assert board.validate(_card(result="FAIL", wrong={**every, "stress": 2})) is None
    assert "incomplete" in board.validate(_card(result="FAIL", wrong={"stress": 2}))     # parts left out
    assert "negative" in board.validate(_card(wrong={**every, "stress": -1}))


def test_a_check_that_cant_judge_anything_says_so(plain_check):
    """NaN inputs make every block unjudgeable: that's "no evidence", not a clean result."""
    a = torch.full((4, 8), float("nan"), dtype=torch.bfloat16)
    b = torch.randn(8, 128).bfloat16()
    r = cs.check_matmul(a, b, (a.float() @ b.float()).bfloat16())
    assert r.ok and not r.judged and r.max_z < 0
    good = cs.check_matmul(torch.randn(4, 8).bfloat16(), b, (torch.randn(4, 8).bfloat16().float() @ b.float()).bfloat16())
    assert good.judged


def test_training_guard_counts_unjudged_checks_apart_from_clean_ones():
    model = nn.Sequential(nn.Linear(8, 128)).to(torch.bfloat16)
    g = SDCGuard(every=1).protect(model)
    with g.on_step(0):
        model(torch.full((4, 8), float("nan"), dtype=torch.bfloat16))
    with g.on_step(1):
        model(torch.randn(4, 8).bfloat16())
    assert g.stats["unjudged"] == 1 and not g.log.events
    assert g.max_z_seen >= 0
