"""The paired experiment must preserve missing trials and completed measurements."""
import importlib.util
import json
from pathlib import Path

import torch


def _module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "mode_compare.py"
    spec = importlib.util.spec_from_file_location("_r12_mode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg():
    return dict(vocab=8, layers=1, width=16, heads=2, seq=4, batch=1, seed=11, steps=1, trials=1)


def test_zero_block_scale_is_counted_as_an_unavailable_trial():
    m = _module()
    a = torch.zeros(2, 8, dtype=torch.bfloat16)
    b = torch.zeros(8, 16, dtype=torch.bfloat16)
    queued = m._inject_and_check(a, b, a.float() @ b.float(), None, 1, 0,
                                 torch.bfloat16, torch.Generator().manual_seed(1), 3)
    assert sum(q[0] == "skipped_scale" for q in queued) == 3


def test_missing_checks_do_not_make_an_apparently_complete_experiment(monkeypatch):
    m = _module()
    monkeypatch.setattr(m, "_inject_and_check", lambda *a: [("clean", None, None, None, None)])
    r = m.run(torch.device("cpu"), _cfg(), say=lambda _: None, max_matmuls=1)
    assert r["clean"]["default"]["unjudged"] == 1
    assert r["clean"]["precise"]["unjudged"] == 1
    assert not r["complete"]
    assert r["incomplete_reasons"]
    assert "INCOMPLETE" in m.markdown(r)


def test_completed_sensitivity_survives_overhead_failure(monkeypatch, tmp_path):
    m = _module()
    measured = m.run(torch.device("cpu"), _cfg(), say=lambda _: None, max_matmuls=1)
    monkeypatch.setattr(m, "run", lambda *a, **kw: measured)
    import precise_models

    def broken(*_):
        saved = json.loads((tmp_path / "cpu.json").read_text())
        assert saved["matmuls"] == 1, "Persist the completed experiment before starting timing."
        assert saved["overhead_status"] == "pending"
        raise RuntimeError("timing failed")

    monkeypatch.setattr(precise_models, "overhead", broken)
    assert m.main(["--out", str(tmp_path)]) == 1
    saved = json.loads((tmp_path / "cpu.json").read_text())
    assert saved["matmuls"] == 1
    assert saved["overhead_status"] == "failed"
    assert "timing failed" in saved["error"]
    assert len(saved["default"]["flagged"]) == len(m.SIZES)


def test_fingerprint_covers_timing_implementation(monkeypatch, tmp_path):
    m = _module()
    experiments = tmp_path / "experiments"
    package = tmp_path / "src" / "sdcguard"
    experiments.mkdir()
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# fixture\n")
    (experiments / "mode_compare.py").write_text("# fixture\n")
    helper = experiments / "precise_models.py"
    helper.write_text("# timing version one\n")
    monkeypatch.setattr(m, "HERE", str(experiments))
    before = m.code_digest()
    helper.write_text("# timing version two\n")
    assert before != m.code_digest()


def test_initial_crossing_remains_left_censored_if_the_curve_later_dips():
    m = _module()
    for level in (0.5, 0.9):
        rates = [level + 0.05, level - 0.05] + [level + 0.05] * (len(m.SIZES) - 2)
        assert m._crossing(rates, level) == m.SIZES[0]
