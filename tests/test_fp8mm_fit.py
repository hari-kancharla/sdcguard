"""experiments/fp8mm_fit.py on made-up calibration data: the held-out prompts are fixed, the fit finds the smallest
tolerance that keeps every training score at most 3, and it reports the held-out verdict and the allowance."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import fp8mm_fit as ff  # noqa: E402


def _layer(path, rng, rows=256, nb=12, bias=0.0, noise_a=1.5, a_scale=1.0, outlier=False):
    """Blocks whose error is bf16 noise S plus a truncation part 1.5 A (and a bias part, if asked; and, if asked, one
    block of the first held-out row 1,000 times off: the draws don't change, so neither does any other block)."""
    prompt = rng.integers(0, ff.N_PROMPTS, rows).astype(np.int16)
    S = rng.uniform(0.5, 1.5, (rows, nb))
    A = rng.uniform(0.5, 2.0, (rows, nb)) * a_scale
    B = rng.uniform(5.0, 50.0, (rows, nb))
    r = rng.normal(0, 1, (rows, nb)) * S + rng.normal(0, 1, (rows, nb)) * noise_a * A + bias * B
    if outlier:
        i = int(np.flatnonzero(np.isin(prompt, np.asarray(ff.HELD_OUT)))[0])
        r[i, 0] = 1000.0 * S[i, 0]
    arrays = {"prompt": prompt}
    for key in ("per_token_cutlass", "per_tensor_cutlass"):
        arrays[f"{key}_r"] = r.astype(np.float32)
        for nm, t in (("S", S), ("A", A), ("B", B), ("E", np.zeros_like(S))):
            arrays[f"{key}_{nm}"] = t.astype(np.float16 if nm != "E" else np.float32)
    np.savez_compressed(path, **arrays)


def test_the_held_out_prompts_are_fixed_and_16_of_68():
    # the 16 recorded in results/fp8mm_cal_oct24/FIT.json (two of them, 65 and 66, are long documents)
    assert ff.HELD_OUT == (0, 2, 4, 10, 15, 17, 28, 32, 34, 40, 45, 49, 50, 58, 65, 66)


def test_the_fit_is_the_smallest_tolerance_the_training_prompts_allow(tmp_path):
    rng = np.random.default_rng(1)
    for i in range(3):
        _layer(tmp_path / f"1.5b__model.layers.{i}.mlp.down_proj.npz", rng)
    (tmp_path / "SUMMARY.json").write_text(json.dumps({"gpu": "test GPU", "torch": "x"}))
    out = tmp_path / "fit.json"
    assert ff.main([str(tmp_path), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    ft = res["runs"][tmp_path.name]["fits"]["per_token_cutlass"]
    assert ft["fitted"] and ft["train_worst"] <= ff.TARGET + 1e-9
    # tight: a little less alpha (at the fitted beta) breaks the training target
    d = ff.load(str(tmp_path), "per_token_cutlass")
    train = ~np.isin(d["prompt"], np.asarray(ff.HELD_OUT))
    if ft["alpha"] > 0:
        assert np.max(ff.scores(d, ft["alpha"] * 0.98, ft["beta"], train)) > ff.TARGET
    # and of every beta on the grid with its smallest alpha, the pair with the smallest median training allowance
    mine = float(np.median(ff.allowance(d, ft["alpha"], ft["beta"], train)))
    for beta in ff.BETAS:
        a = ff.smallest_alpha(d, beta, train)
        if a is not None:
            assert mine <= float(np.median(ff.allowance(d, a, beta, train))) + 1e-12, beta
    assert set(ft["layers"]) == {f"1.5b__model.layers.{i}.mlp.down_proj" for i in range(3)}
    assert ft["allowance_median"] >= 1.0 and ft["held_out_passed"] == (ft["held_out_worst"] <= ff.TARGET)
    # an error of (line + 3) tolerances scores over the line against any clean part scoring up to 3, either sign
    assert ft["line_in_bf16_noise_units_median"] == ff.LINE * ft["allowance_median"]
    assert abs(ft["caught_for_sure_in_bf16_noise_units_median"] - 9 * ft["allowance_median"]) < 1e-9


def test_a_bias_the_training_data_needs_is_fitted_with_beta(tmp_path):
    """An error that follows B (a bias) where A is tiny: covering it with alpha would allow far more noise, so the
    fit takes a beta."""
    rng = np.random.default_rng(2)
    for i in range(2):
        _layer(tmp_path / f"7b__model.layers.{i}.mlp.up_proj.npz", rng, bias=0.02, noise_a=0.0, a_scale=0.001)
    ft = ff.run([str(tmp_path)])["runs"][tmp_path.name]["fits"]["per_tensor_cutlass"]
    assert ft["fitted"] and ft["train_worst"] <= ff.TARGET + 1e-9 and ft["beta"] > 0
    d = ff.load(str(tmp_path), "per_tensor_cutlass")
    assert np.max(ff.scores(d, 0.0, 0.0)) > ff.TARGET          # bf16 noise alone doesn't cover it


def test_held_out_prompts_dont_shape_the_fit_and_an_outlier_there_fails_it(tmp_path):
    rng = np.random.default_rng(3)
    _layer(tmp_path / "1.5b__model.layers.0.mlp.gate_proj.npz", rng)
    clean = ff.run([str(tmp_path)])["runs"][tmp_path.name]["fits"]["per_token_cutlass"]
    rng = np.random.default_rng(3)
    _layer(tmp_path / "1.5b__model.layers.0.mlp.gate_proj.npz", rng, outlier=True)
    out = ff.run([str(tmp_path)])["runs"][tmp_path.name]["fits"]["per_token_cutlass"]
    assert (out["alpha"], out["beta"]) == (clean["alpha"], clean["beta"])       # the fit didn't see it
    assert out["held_out_worst"] > ff.TARGET and out["held_out_passed"] is False
    assert clean["blocks_held_out"] == out["blocks_held_out"]                    # the same blocks on each side


def test_blocks_that_arent_finite_are_left_out_and_counted(tmp_path):
    rng = np.random.default_rng(4)
    _layer(tmp_path / "7b__model.layers.0.mlp.down_proj.npz", rng)
    z = dict(np.load(tmp_path / "7b__model.layers.0.mlp.down_proj.npz"))
    z["per_token_cutlass_B"][0, 0] = np.inf                    # float16 B past 65,504
    np.savez_compressed(tmp_path / "7b__model.layers.0.mlp.down_proj.npz", **z)
    d = ff.load(str(tmp_path), "per_token_cutlass")
    assert d["blocks_not_finite"] == 1 and d["blocks_total"] == 256 * 12 and d["r"].size == 256 * 12 - 1
    ft = ff.fit(d)
    assert ft["blocks_not_finite"] == 1 and np.isfinite(ft["train_worst"])


def test_short_prompts_and_long_documents_are_reported_apart(tmp_path):
    rng = np.random.default_rng(5)
    _layer(tmp_path / "1.5b__model.layers.0.mlp.up_proj.npz", rng)
    ft = ff.fit(ff.load(str(tmp_path), "per_token_cutlass"))
    d = ff.load(str(tmp_path), "per_token_cutlass")
    test = np.isin(d["prompt"], np.asarray(ff.HELD_OUT))
    longd = np.isin(d["prompt"], np.asarray(ff.LONG_DOCS))
    z = ff.scores(d, ft["alpha"], ft["beta"])
    assert ft["held_out_worst_long_documents"] == float(np.max(z[test & longd]))
    assert ft["held_out_worst_short_prompts"] == float(np.max(z[test & ~longd]))
    assert max(ft["held_out_worst_short_prompts"], ft["held_out_worst_long_documents"]) == ft["held_out_worst"]
    assert ft["long_documents_share_held_out"] == float(np.mean(longd[test]))
    assert ff.LONG_DOCS == tuple(range(64, 68)) and set(ff.LONG_DOCS) & set(ff.HELD_OUT) == {65, 66}


def test_zero_noise_blocks_are_judged_without_dividing_by_zero():
    d = {"r": np.array([0.0, 1.0]), "S": np.zeros(2), "A": np.zeros(2), "B": np.zeros(2)}
    z = ff.scores(d, 1.0, 0.0)
    assert z[0] == 0.0 and z[1] == np.inf and ff.allowance(d, 1.0, 0.0).size == 0
