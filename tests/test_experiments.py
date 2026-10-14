"""End-to-end smoke test of the experiment pipeline on tiny offline data (CPU, ~20 s)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

import harm_study  # noqa: E402


def test_runs_save_load_and_analyze(tmp_path):
    out, data = str(tmp_path / "runs"), str(tmp_path / "data")
    base = {"preset": "synth", "steps": 40, "guard_every": 5}
    for seed in (0, 1, 2):
        harm_study.run_from_spec({**base, "kind": "clean", "seed": seed}, data, out)
    harm_study.run_from_spec({**base, "kind": "clean", "seed": 0, "repeat": True}, data, out)
    path = harm_study.run_from_spec({**base, "kind": "faulty", "seed": 0, "rate": 0.5, "target": "qk",
                                     "mix": "zero", "scale": "warp", "fault_seed": 1}, data, out)
    r = harm_study.load_run(path)
    assert r.injected > 0 and r.faults_on_checked_steps > 0
    assert r.faults_caught == r.faults_on_checked_steps          # zeroing big chunks is always caught
    assert all(len(e) == 6 for e in r.fault_events)
    # resumable: a second call skips the finished run...
    assert harm_study.run_from_spec({**base, "kind": "clean", "seed": 0}, data, out).endswith("clean_s0.json")
    # ...unless it was made by other code, which is run again
    p0 = os.path.join(out, "synth", "raw", "clean_s0.json")
    d0 = json.load(open(p0))
    d0["env"]["code"] = "old"
    json.dump(d0, open(p0, "w"))
    harm_study.run_from_spec({**base, "kind": "clean", "seed": 0}, data, out)
    assert json.load(open(p0))["env"]["code"] == harm_study.code_version()

    aggs = harm_study.analyze(out, "synth", write_plots=False)
    assert len(aggs) == 1 and aggs[0]["runs"] == 1
    rep = open(os.path.join(out, "synth", "REPORT.md")).read()
    assert "false alarms" in rep and "Detection by fault size" in rep
    s = json.load(open(os.path.join(out, "synth", "faulty_qk_zero_warp_r0.5_f1", "summary.json")))["summary"]
    assert s["valid"] and s["detector"]["false_alarms_clean"] == 0
    assert s["same_seed_gpu_noise"] is not None


def test_calibration_soak_and_summary(tmp_path):
    import calibrate
    import torch
    out, data = str(tmp_path / "runs"), str(tmp_path / "data")
    cfg = dict(harm_study.PRESETS["synth"], name="synth")
    ds = harm_study.load_dataset("synthetic", data)
    model, opt = calibrate.make(cfg, ds.vocab, torch.device("cpu"))
    path = os.path.join(out, "calibration", "synth", "soak", "w0.json")
    saves = []
    rec = calibrate.soak(model, opt, ds.train, cfg, torch.device("cpu"), torch.bfloat16,
                         torch.Generator().manual_seed(0), 0.05, path, save_every_s=1, on_save=lambda: saves.append(1))
    assert rec["checks"] > 0 and rec["false_alarms"] == 0 and saves
    summary = calibrate.summarize(os.path.join(out, "calibration", "synth"), "synth")
    assert summary["soak"]["false_alarms"] == 0


def test_fp16_training_with_loss_scaling_has_no_false_alarms():
    """fp16 training with GradScaler: overflowing steps (skipped by the scaler) and tiny gradients
    must not raise alarms. GradScaler runs on the CPU here, as the CUDA one would on a GPU."""
    from types import SimpleNamespace

    import calibrate
    import torch
    from sdcguard import SDCGuard
    cfg = dict(harm_study.PRESETS["ts-gpt2"])
    cfg.update(n_layer=1, n_head=2, n_embd=64, block=32, batch=4, name="tiny", steps=200)
    ds = SimpleNamespace(train=torch.randint(0, 1000, (20000,)), vocab=1000)
    for init in (2.0**40, 2.0**4):          # too high (overflows), and low (tiny gradients)
        model, opt = calibrate.make(cfg, ds.vocab, torch.device("cpu"), 0)
        opt._sdc_scaler = torch.amp.GradScaler("cpu", init_scale=init)
        opt._sdc_overflows = 0
        g = SDCGuard()
        orig = calibrate._scaler
        calibrate._scaler = lambda o, d, t: o._sdc_scaler
        try:
            calibrate.train_steps(model, opt, ds.train, cfg, torch.device("cpu"), torch.float16, 30,
                                  torch.Generator().manual_seed(0), g, 1, schedule=True)
        finally:
            calibrate._scaler = orig
        assert g.stats["checked"] > 100 and g.stats["alerts"] == 0, (init, g.stats, g.log.events[:2])
        if init > 1e10:
            assert opt._sdc_overflows > 0
