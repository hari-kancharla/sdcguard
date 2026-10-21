"""The Hugging Face Trainer callback (sdcguard/hf.py) on a tiny GPT-2 trained on the CPU in bf16."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from sdcguard import FaultInjector  # noqa: E402
from sdcguard.hf import SDCGuardCallback  # noqa: E402


class _Tokens(torch.utils.data.Dataset):
    def __init__(self, n=64, seq=32, vocab=128):
        g = torch.Generator().manual_seed(0)
        self.ids = torch.randint(0, vocab, (n, seq), generator=g)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "labels": self.ids[i]}


def _trainer(tmp_path, cb, steps=6, accum=1, checkpointing=False, before=()):
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=4, vocab_size=128, n_positions=32)
    model = transformers.GPT2LMHeadModel(cfg)
    args = transformers.TrainingArguments(output_dir=str(tmp_path), max_steps=steps, per_device_train_batch_size=4,
                                          gradient_accumulation_steps=accum, bf16=True, use_cpu=True,
                                          logging_steps=1, report_to=[], save_strategy="no",
                                          gradient_checkpointing=checkpointing, disable_tqdm=True)
    return transformers.Trainer(model=model, args=args, train_dataset=_Tokens(), callbacks=[*before, cb])


def test_clean_training_is_checked_on_schedule_and_logged(tmp_path):
    cb = SDCGuardCallback(every=2)
    tr = _trainer(tmp_path, cb, steps=6, accum=2)
    tr.train()
    st = cb.guard.stats
    assert st["checked"] > 0 and st["alerts"] == 0 and st["unjudged"] == 0
    logs = {}
    cb.on_log(tr.args, tr.state, tr.control, logs=logs)
    assert logs == {"sdcguard/checked": st["checked"], "sdcguard/alerts": 0}
    assert not cb.guard._active                       # every checked step was closed


def test_a_fault_is_caught_and_training_stops(tmp_path):
    cb = SDCGuardCallback(every=1, stop_on_alert=True)
    tr = _trainer(tmp_path, cb, steps=6)
    inj = FaultInjector(rate=1.0, seed=3).arm_on([tr.model.transformer.h[1].mlp.c_fc])   # a GPT-2 Conv1D
    with inj:
        tr.train()
    inj.remove_hooks()
    assert cb.guard.stats["alerts"] > 0
    assert tr.state.global_step < 6                   # stopped after the first failed step


def test_recompute_check_through_the_callback(tmp_path):
    cb = SDCGuardCallback(every=1, recompute=True)
    tr = _trainer(tmp_path, cb, steps=3, checkpointing=True)
    tr.train()
    st = cb.guard.stats
    assert st["recompute_compared"] > 0 and st["recompute_mismatches"] == 0


def test_training_end_puts_the_model_and_checkpoint_back(tmp_path):
    import torch.utils.checkpoint as tuc
    orig = tuc.checkpoint
    cb = SDCGuardCallback(every=1, recompute=True)
    tr = _trainer(tmp_path, cb, steps=2, checkpointing=True)
    tr.train()
    assert tuc.checkpoint is orig and not cb.guard._protected
    assert all(not hasattr(type(m), "_sdcguard_base") for m in tr.model.modules())


def test_a_logged_spike_checks_the_very_next_step():
    """The Trainer counts a step (global_step += 1) before it logs it, and on_step_begin decides with that same
    count, so a spike logged with global_step == N belongs to step N - 1 in the guard's numbering: the window
    observe() opens must start at step N, the next on_step_begin, not one step later."""
    from types import SimpleNamespace
    cb = SDCGuardCallback(every=1000)
    ctl = SimpleNamespace(should_training_stop=False)
    for gs in range(1, 11):                         # logged after steps 0..9
        cb.on_log(None, SimpleNamespace(global_step=gs), ctl, logs={"loss": 1.0})
    cb.on_log(None, SimpleNamespace(global_step=11), ctl, logs={"loss": 1.0, "grad_norm": float("inf")})
    assert cb.guard.anomalies[-1]["step"] == 10
    cb.on_step_begin(None, SimpleNamespace(global_step=11), ctl)
    checked = cb._ctx is not None
    cb._close(ctl)
    assert checked, "the step after the spike wasn't checked"


def test_the_trainer_checks_the_step_after_a_logged_spike(tmp_path):
    """The same with a real Trainer: a callback before SDCGuardCallback makes the gradient norm logged at
    global_step 3 infinite, and step 3 (the next optimizer step) is checked."""
    class Spike(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs is not None and state.global_step == 3 and "loss" in logs:
                logs["grad_norm"] = float("inf")

    cb = SDCGuardCallback(every=1000)
    steps = []
    real = cb.guard.at_step

    def at_step(step):
        steps.append(step)
        return real(step)
    cb.guard.at_step = at_step
    _trainer(tmp_path, cb, steps=6, before=[Spike()]).train()
    assert steps == [0, 3, 4, 5], steps
