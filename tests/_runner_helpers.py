"""Local runner regression utilities; no service calls are made."""
import importlib.util
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    class Image:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: self

    class App:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, **kwargs):
            return lambda fn: fn

        def local_entrypoint(self):
            return lambda fn: fn

    stub = types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **kwargs: Image()), App=App,
        Volume=types.SimpleNamespace(from_name=lambda *args, **kwargs: Image()))
    spec = importlib.util.spec_from_file_location("_loaded_" + name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modal=stub):
        spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def api(self):
        return types.SimpleNamespace(time=self.time, monotonic=self.time, sleep=self.sleep,
                                     strftime=time.strftime)
