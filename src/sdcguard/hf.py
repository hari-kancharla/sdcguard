"""Hugging Face Trainer integration: one callback.

    from sdcguard.hf import SDCGuardCallback

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      callbacks=[SDCGuardCallback(every=20)])

On every `every`-th optimizer step the callback checks that step's forward and backward matmuls (all of its
gradient-accumulation micro-batches) and reads the results back before the optimizer runs. Alerts go to the
"sdcguard" logger and to `callback.guard.log`; the counts are in `callback.guard.stats`, and each log line the
Trainer prints carries `sdcguard/checked` and `sdcguard/alerts` (reporting integrations such as W&B run before
user callbacks, so log those counts yourself if you want them there). A logged loss or gradient norm that
spikes, or isn't finite, makes every following step a checked one for a while (SDCGuard.observe). With stop_on_alert=True a failed check
stops training after that step, before a corrupted model is saved over a good checkpoint. When training
ends, the layers the callback protected go back to their own forward. Any other SDCGuard option passes
through: recompute=True (with gradient_checkpointing=True), replicas=True (data-parallel weight check),
evidence_dir="sdc_evidence", precise=True.

Needs transformers (the callback events on_step_begin and on_pre_optimizer_step, transformers 4.46 or newer).
"""
from __future__ import annotations

import logging
from typing import Optional

from transformers import TrainerCallback

from .guard import SDCGuard

log = logging.getLogger("sdcguard")


class SDCGuardCallback(TrainerCallback):
    """Checks every `every`-th training step of a Hugging Face Trainer with SDCGuard."""

    def __init__(self, every: int = 20, guard: Optional[SDCGuard] = None, stop_on_alert: bool = False,
                 **guard_kwargs):
        self.guard = guard if guard is not None else SDCGuard(every=every, **guard_kwargs)
        self.stop_on_alert = stop_on_alert
        self._ctx = None
        self._alerts_seen = 0
        self._protected = guard is not None and bool(getattr(guard, "_protected", None))
        self._we_protected = False

    # ------------------------------------------------------------------ lifecycle
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._protected and model is not None:
            self.guard.protect(model)
            self._protected = self._we_protected = True
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._close(control)                         # a step that ended without on_pre_optimizer_step
        step = state.global_step
        if self.guard.should_check(step):
            ctx = self.guard.at_step(step)
            ctx.__enter__()
            self._ctx = ctx
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._close(control)                         # results are read back before the weights change
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._close(control)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._close(control)
        if self._we_protected:            # the model goes back to its own layers (and checkpoint to torch's)
            self.guard.unprotect()
            self._protected = self._we_protected = False
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # a loss or gradient-norm spike starts a window of checked steps (SDCGuard.observe). The Trainer logs
            # after it has counted the step (global_step += 1), so the logged step is global_step - 1 in the
            # numbering on_step_begin checks by, and the window starts at the very next step
            self.guard.observe(state.global_step - 1, loss=logs.get("loss"), grad_norm=logs.get("grad_norm"))
            logs["sdcguard/checked"] = self.guard.stats.get("checked", 0)
            logs["sdcguard/alerts"] = self.guard.stats.get("alerts", 0)
        return control

    # ------------------------------------------------------------------ internals
    def _close(self, control) -> None:
        ctx, self._ctx = self._ctx, None
        if ctx is not None:
            ctx.__exit__(None, None, None)
        events = self.guard.log.events
        new = events[self._alerts_seen:]
        self._alerts_seen = len(events)
        for ev in new:
            log.warning("sdcguard alert at step %s: %s", ev.step, ev.to_json())
        if new and self.stop_on_alert:
            control.should_training_stop = True
