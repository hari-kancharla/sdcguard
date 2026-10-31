"""Explicit passive mapping activation through the existing forward bridge.

Owner: Hari Krishna Kancharla.
Importing this module starts no engine, device work or recording interval.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import threading
import weakref


_ACTIVATION = None
_PID = os.getpid()
_LOCK = threading.Lock()
_INSTALLING = False


def _unknown(reason):
    return dict(status="unknown", reason=reason, verdict="unknown",
                runtime_validation="pending")


class _Activation:
    def __init__(self, forward_directory, assignment_directory):
        self.pid = os.getpid()
        self.forward_directory = forward_directory
        self.assignment_directory = assignment_directory
        self.observer = None
        self.installation = None
        self.reason = None
        self.result = None

        def factory(registration, health):
            from .mapping_observer import mapping_factory
            from .mapping_closure import write_receipt
            package = importlib.import_module("vllm")
            root = Path(package.__file__).absolute().parent.parent
            observer = mapping_factory(self.assignment_directory, root,
                                       closure_callback=write_receipt)(registration, health)
            self.observer = observer
            return observer
        self.factory = factory

    def fail(self, reason):
        self.reason = reason
        if self.observer is not None and self.pid == os.getpid():
            self.observer.health.error(reason)
        return _unknown(reason)

    def status(self):
        if self.pid != os.getpid():
            return _unknown("mapping initialization was not observed in this process")
        if self.reason:
            result = _unknown(self.reason)
        elif self.observer is None:
            result = _unknown("mapping observer registration is incomplete")
        else:
            observed = self.observer.status
            result = dict(status="closed" if observed["closed"] else "registered",
                          healthy=observed["healthy"], verdict="unknown", runtime_validation="pending",
                          observer=observed)
        result["directories"] = dict(forward=str(self.forward_directory), assignment=str(self.assignment_directory))
        return result


def _install():
    """Register only when both explicit recording directories are supplied."""
    global _ACTIVATION
    assignment = os.environ.get("SDCGUARD_ASSIGNMENT_DIR")
    if not assignment:
        return dict(status="disabled", verdict="unknown", runtime_validation="pending")
    forward = os.environ.get("SDCGUARD_FORWARD_DIR")
    if not forward:
        return _unknown("SDCGUARD_ASSIGNMENT_DIR also requires SDCGUARD_FORWARD_DIR")
    try:
        forward_directory, assignment_directory = Path(forward).absolute(), Path(assignment).absolute()
        if forward_directory == assignment_directory:
            return _unknown("forward and assignment recordings need distinct directories")
    except (TypeError, ValueError, OSError):
        return _unknown("recording directories are unavailable")
    current = _ACTIVATION
    if current is not None:
        if current.pid != os.getpid():
            return _unknown("mapping initialization was not observed in this process")
        if (current.forward_directory != forward_directory
                or current.assignment_directory != assignment_directory):
            return current.fail("mapping recording directories changed after registration")
    else:
        current = _ACTIVATION = _Activation(forward_directory, assignment_directory)
    if current.reason:
        return current.status()
    from . import vllm_forward
    active = vllm_forward._INSTALLATION
    if active is not None and (active.pid != os.getpid() or active.assignment_factory is not current.factory):
        return current.fail("mapping requires a fresh forward installation in this process")
    try:
        # Validate and create only the explicitly selected host output path.
        vllm_forward._directory(assignment_directory)
        result = vllm_forward.install(assignment_factory=current.factory)
        current.result = result
        installed = vllm_forward._INSTALLATION
        owned = (installed is not None and installed.pid == os.getpid()
                 and installed.assignment_factory is current.factory)
        if owned:
            current.installation = weakref.ref(installed)
        if (result.get("status") != "installed" or installed is None
                or installed.pid != os.getpid() or installed.assignment_factory is not current.factory
                or installed.bridge is None or installed.bridge_error
                or current.observer is None):
            reason = (result.get("reason") or getattr(installed, "bridge_error", None)
                      or "mapping observer registration is incomplete")
            if owned:
                vllm_forward.uninstall()
            return current.fail(reason)
        return current.status()
    except Exception as exc:
        return current.fail("mapping registration failed: " + type(exc).__name__)


def install():
    """Admit one setup attempt without holding a state lock during setup."""
    global _INSTALLING
    if not os.environ.get("SDCGUARD_ASSIGNMENT_DIR"):
        return dict(status="disabled", verdict="unknown", runtime_validation="pending")
    if _PID != os.getpid():
        return _unknown("mapping initialization was not observed in this process")
    with _LOCK:
        if _INSTALLING:
            return _unknown("mapping registration is already in progress")
        _INSTALLING = True
    try:
        return _install()
    finally:
        with _LOCK:
            _INSTALLING = False


def status():
    """Return independent host status, including the journal's own health."""
    if _ACTIVATION is None:
        if os.environ.get("SDCGUARD_ASSIGNMENT_DIR"):
            return _unknown("mapping observer has not been registered")
        return dict(status="disabled", verdict="unknown", runtime_validation="pending")
    return _ACTIVATION.status()


def uninstall():
    """Close only the forward installation owned by this activation."""
    current = _ACTIVATION
    if current is None:
        return dict(status="disabled", verdict="unknown", runtime_validation="pending")
    if current.pid != os.getpid():
        return _unknown("mapping initialization was not observed in this process")
    from . import vllm_forward
    owned = None if current.installation is None else current.installation()
    if owned is not None and vllm_forward._INSTALLATION is owned:
        vllm_forward.uninstall()
        current.installation = None
    elif (owned is not None and vllm_forward._INSTALLATION is None
          and not owned.wrappers and current.observer is not None
          and current.observer.status["closed"]):
        current.installation = None
    elif owned is not None:
        return current.fail("the registered forward installation was replaced")
    return current.status()
