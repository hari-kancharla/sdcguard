"""sdcguard check (one GPU, or the CPU when there's none), its result card, the public board, nhc --cached."""
import copy
import json
import os
import subprocess
import sys
import time

import pytest

from sdcguard import board, check, cli, gpuinfo

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")


@pytest.fixture(scope="module")
def cpu_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("check")
    full = check.check_one("cpu", minutes=0.1, out_root=str(out), reference=None)
    return full, out


def test_check_on_cpu_writes_card_and_result(cpu_run):
    full, out = cpu_run
    assert full["result"] == "PASS" and full["repeatable"] and len(full["fingerprints"]) == 3
    assert full["stress"]["matmuls_checked"] >= 2 and full["memory"]["wrong_words"] == 0
    d = full["out_dir"]
    for name in ("check.json", "result.json", "card.svg", "card.md", "summary.json"):
        assert os.path.exists(os.path.join(d, name)), name
    pub = json.load(open(os.path.join(d, "result.json")))
    text = json.dumps(pub)
    host = (full["gpu"] or {}).get("host")
    assert host and host not in text                      # nothing that identifies the machine
    assert "serial" not in text and "pci" not in text.lower().replace("pcie", "")
    assert pub["digest"] == board.digest(pub)
    assert board.validate(pub) == "not a GPU result"      # CPU runs never go on the board
    card = open(os.path.join(d, "card.svg")).read()
    assert "PASS" in card and "\u2014" not in card and "\u2014" not in check.markdown(pub)
    url = check.share_url(pub)
    assert url.startswith(f"https://github.com/{check.BOARD_REPO}/issues/new?") and "template=result.yml" in url
    assert len(url) < 8000


def test_terminal_card_lines(cpu_run):
    full, _ = cpu_run
    pub = check.public(full)
    t = check.terminal(pub)
    assert "PASS" in t and "repeatable" in t and "headroom" in t and "speed" in t


def test_simulated_faults_fail_and_are_kept_off_the_board(tmp_path):
    full = check.check_one("cpu", minutes=0.08, out_root=str(tmp_path), simulate_faults=0.05)
    assert full["result"] == "FAIL" and full["wrong"]["workload_training"] > 0 and full["simulated_faults"]
    pub = check.public(full)
    assert "SIMULATED" in check.svg(pub) and "SIMULATED" in check.markdown(pub)
    assert board.validate(pub) == "a simulated-fault demo, not a real result"


def _fake(out, key="k", name="NVIDIA H100 80GB HBM3"):
    return {"fingerprint_key": key, "fingerprints": [{"output": out, "layers": []}], "result": "PASS",
            "notes": [], "gpu": {"gpu_name": name}}


def test_cross_compare_finds_the_odd_gpu():
    rs = [_fake("a"), _fake("a"), _fake("b"), _fake("a")]
    check.cross_compare(rs)
    assert [r["result"] for r in rs] == ["PASS", "PASS", "SUSPECT", "PASS"]
    assert "odd one out" in rs[2]["notes"][0][1] and "matches 2 other" in rs[0]["cross_gpu"]
    two = [_fake("a"), _fake("b")]
    check.cross_compare(two)
    assert all(r["result"] == "PASS" and r["notes"][0][0] == "warn" for r in two)   # can't tell which one
    same = [_fake("a"), _fake("a")]
    check.cross_compare(same)
    assert "identical output on all 2" in same[0]["cross_gpu"]


def test_board_reference_makes_a_lone_mismatch_suspect():
    ref = {"fingerprints": {"k": {"b": {"gpus": 3, "sources": 2}}},
           "models": {"NVIDIA H100 80GB HBM3": {"runs": 3, "median_tflops": 700}}}
    r = _fake("a")
    check.apply_board(r, ref)
    assert r["result"] == "SUSPECT" and r["board"]["same_setup_runs"] == 3 and r["board"]["same_fingerprint"] == 0
    ok = _fake("b")
    check.apply_board(ok, ref)
    assert ok["result"] == "PASS" and ok["board"]["same_fingerprint"] == 3 and ok["board"]["model_median_tflops"] == 700
    # one person's three cards (or the maintainer's seed runs alone) can't make anyone SUSPECT: a note only
    weak = {"fingerprints": {"k": {"b": {"gpus": 3, "sources": 1}}}}
    w = _fake("a")
    check.apply_board(w, weak)
    assert w["result"] == "PASS" and w["notes"][0][0] == "warn"


def test_board_survives_hand_made_cards(cpu_run, tmp_path):
    full, _ = cpu_run
    base = check.public(full)
    bad = []
    for path, value in ((("speed", "matmul_tflops"), "700"), (("checked", "results"), "lots"), (("gpu", "id_hash"), [1]),
                        (("fingerprint", "output"), [1]), ((None, "gpu"), [1]), ((None, "provider"), "Modal"),
                        ((None, "notes"), [1]), ((None, "minutes"), float("nan"))):
        r = copy.deepcopy(base)
        r["gpu"]["name"] = "NVIDIA L4"
        r["minutes"] = 5
        if path[0]:
            r[path[0]][path[1]] = value
        else:
            r[path[1]] = value
        r["digest"] = board.digest(r)
        bad.append(r)
    nested = board.extract('{"a":' + "[" * 60000)
    assert nested is None
    rep = board.build(bad, str(tmp_path / "b"))
    assert rep["accepted"] == 0 and len(rep["rejected"]) == len(bad)
    assert all("malformed" in x["why"] or "not a" in x["why"] for x in rep["rejected"])


def test_screen_puts_cublas_settings_back(monkeypatch, tmp_path):
    """The serving check turns off reduced-precision reductions; the screen must restore them, or the check's
    later fingerprint run would use different cuBLAS settings than the earlier ones."""
    import torch
    from sdcguard import guard
    from sdcguard.screen import screen_one
    monkeypatch.setattr(guard.torch.cuda, "is_available", lambda: True)
    mm = torch.backends.cuda.matmul
    before = (mm.allow_bf16_reduced_precision_reduction, mm.allow_fp16_reduced_precision_reduction)
    screen_one(torch.device("cpu"), minutes=0.01, size="tiny", out_root=str(tmp_path), max_steps=1)
    assert (mm.allow_bf16_reduced_precision_reduction, mm.allow_fp16_reduced_precision_reduction) == before


def test_board_build_validates_and_aggregates(cpu_run, tmp_path):
    full, _ = cpu_run
    base = check.public(full)

    def mk(name, res, tf, logits, prov, idh, minutes=5.0):
        r = copy.deepcopy(base)
        r["gpu"].update(name=name, id_hash=idh)
        r.update(result=res, minutes=minutes, provider={"name": prov})
        if res == "SUSPECT":            # (a real SUSPECT card says what disagreed: here, its own reruns)
            r["repeatable"] = False
        r["speed"]["matmul_tflops"] = tf
        r["fingerprint"] = {"key": "k1", "output": logits}
        r["digest"] = board.digest(r)
        return r

    good = [mk("NVIDIA H100 80GB HBM3", "PASS", 700, "aa", "Modal", "g1"),
            mk("NVIDIA H100 80GB HBM3", "PASS", 690, "aa", "<b>RunPod</b>", "g2"),
            mk("NVIDIA H100 80GB HBM3", "SUSPECT", 650, "bb", None, "g3")]
    tampered = copy.deepcopy(good[0])
    tampered["speed"]["matmul_tflops"] = 9999
    short = mk("NVIDIA L4", "PASS", 95, "cc", None, "g4", minutes=0.3)
    body = "### Result card (JSON)\n\n```json\n" + json.dumps(good[0]) + "\n```\n\n### Anything else?\n\n_No response_"
    assert board.extract(body) == good[0] and board.validate(board.extract(body)) is None
    rep = board.build(good + [tampered, short, copy.deepcopy(good[0])], str(tmp_path / "b"), "me/sdcguard")
    assert rep["accepted"] == 3 and rep["totals"]["SUSPECT"] == 1 and rep["totals"]["gpus"] == 3
    whys = sorted(x["why"] for x in rep["rejected"])
    assert any("digest" in w for w in whys) and any("minute" in w for w in whys) and any("already" in w for w in whys)
    ref = json.load(open(tmp_path / "b" / "reference.json"))
    assert ref["fingerprints"]["k1"] == {"aa": {"gpus": 2, "sources": 1}, "bb": {"gpus": 1, "sources": 1}}
    assert ref["models"]["NVIDIA H100 80GB HBM3"]["median_tflops"] == 690
    page = open(tmp_path / "b" / "index.html").read()
    assert "GPU correctness board" in page and "&lt;b&gt;RunPod" in page and "<b>RunPod" not in page
    assert "\u2014" not in page


def test_board_from_dirs(cpu_run, tmp_path):
    full, out = cpu_run
    rows = board.from_dirs([str(out)])
    assert len(rows) == 1 and rows[0]["schema"] == board.SCHEMA


def test_nhc_cached(tmp_path, capsys):
    d = str(tmp_path)
    assert cli.main(["nhc", "--cached", "--out", d]) == 2
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"time": time.time(), "code": 0, "line": "OK: 8 GPUs computed correctly"}, f)
    assert cli.main(["nhc", "--cached", "--out", d]) == 0
    assert "OK: 8 GPUs" in capsys.readouterr().out
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"time": time.time() - 48 * 3600, "code": 0, "line": "OK"}, f)
    assert cli.main(["nhc", "--cached", "--out", d]) == 2
    with open(os.path.join(d, "status.json"), "w") as f:
        json.dump({"time": time.time() - 48 * 3600, "code": 1, "line": "FAIL: 1 of 8 GPUs"}, f)
    assert cli.main(["nhc", "--cached", "--out", d]) == 1        # a failure doesn't age out


def test_check_with_signed_certificate(tmp_path):
    pytest.importorskip("cryptography")
    from sdcguard import certify
    key, pub = certify.keygen(str(tmp_path / "k"))
    full = check.check_one("cpu", minutes=0.06, out_root=str(tmp_path / "o"), key_path=key, issuer="me")
    path = os.path.join(full["out_dir"], "certificate.json")
    r = certify.verify(path, pub)
    cert = json.load(open(path))
    assert r["ok"] and cert["check"]["repeatable"] is True and cert["check"]["speed"]["matmul_tflops"] > 0


def test_cli_check_json(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, "-m", "sdcguard", "check", "--minutes", "0.06", "--offline", "--json",
                        "--out", "o"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout + p.stderr
    pub = json.loads(p.stdout[p.stdout.index("{"):p.stdout.rindex("}") + 1])
    assert pub["schema"] == board.SCHEMA and pub["result"] == "PASS"


def test_datasheet_names():
    assert gpuinfo.datasheet("NVIDIA H100 80GB HBM3")["matmul_tflops"] == 989.5
    assert gpuinfo.datasheet("NVIDIA H100 NVL")["memory_tbs"] == 3.9
    assert gpuinfo.datasheet("NVIDIA GH200 480GB") == {}               # not the H200
    assert gpuinfo.datasheet("NVIDIA L40") == {} and gpuinfo.datasheet("NVIDIA L40S")["matmul_tflops"] == 362.05
    assert gpuinfo.datasheet("NVIDIA A10G") == {} and gpuinfo.datasheet("NVIDIA A10")["matmul_tflops"] == 125.0
    assert gpuinfo.datasheet("Tesla T4")["matmul_tflops"] == 65.0 and gpuinfo.datasheet("NVIDIA GeForce RTX 4090") == {}
    assert gpuinfo.throttle_reasons("0x0000000000000044") == ["sw_power_cap", "hw_thermal_slowdown"]


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs a CUDA GPU")
def test_check_on_gpu(tmp_path):
    full = check.check_one("cuda", minutes=0.5, out_root=str(tmp_path))
    assert full["result"] == "PASS", full["notes"]
    assert full["repeatable"] and full["speed"]["tflops"] > 1 and full["bandwidth"]["gbs"] > 10
    assert full["memory"]["wrong_words"] == 0 and full["memory"]["gib_tested"] > 0.5
    assert full["stress"]["wrong"] == 0 and full["stress"]["matmuls_checked"] > 5      # (1.8 s after compiling)
    pub = check.public(full)
    assert board.validate(dict(pub, minutes=5, digest=board.digest(dict(pub, minutes=5)))) is None


def test_many_gpus_path_runs_every_child_and_compares(monkeypatch, tmp_path):
    """The multi-GPU path (one process per GPU, then the cross-GPU comparison), with the children on the CPU."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("SDCGUARD_CHILD_DEVICE", "cpu")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    out = str(tmp_path / "o")
    code = check.main(["--minutes", "0.05", "--out", out, "--offline"])
    dirs = sorted(d for d in os.listdir(out) if os.path.exists(os.path.join(out, d, "result.json")))
    assert code == 0 and dirs == ["cpu0", "cpu1"]
    full = json.load(open(os.path.join(out, "cpu0", "check.json")))
    assert "identical output on all 2" in full["cross_gpu"]


def test_many_gpus_path_reports_a_child_that_crashed(monkeypatch, tmp_path, capsys):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setenv("SDCGUARD_CHILD_DEVICE", "cuda:99")      # no such GPU anywhere: both children fail
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([SRC, os.environ.get("PYTHONPATH", "")]))
    code = check.main(["--minutes", "0.05", "--out", str(tmp_path / "o"), "--offline"])
    text = capsys.readouterr().out
    assert code == 2 and "could not be checked" in text and "child_0.log" in text


def test_triton_kernels_wait_for_validation_on_new_gpu_generations(monkeypatch):
    """A fault in an unvalidated kernel ends the process, so generations newer than the validated ones run the
    plain checks unless SDCGUARD_TRITON_NEW_ARCH=1. Blackwell (capability 10) runs the kernels without
    software pipelining since Oct 19 (a B200 faulted with it; test_arch_config.py), =0 opts out."""
    import torch
    from sdcguard import _group, _serve_fast
    cuda = torch.device("cuda", 0)
    monkeypatch.delenv("SDCGUARD_TRITON_NEW_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    assert _group.arch_ok(cuda) is False
    monkeypatch.setenv("SDCGUARD_TRITON_NEW_ARCH", "1")
    assert _group.arch_ok(cuda) is True
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    monkeypatch.setenv("SDCGUARD_TRITON_NEW_ARCH", "0")
    assert _group.arch_ok(cuda) is False
    monkeypatch.delenv("SDCGUARD_TRITON_NEW_ARCH")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    assert _group.arch_ok(cuda) is True
    monkeypatch.setattr(_group, "arch_ok", lambda d: False)
    monkeypatch.setattr(_serve_fast, "STATUS", {})
    monkeypatch.setattr(_serve_fast, "_probe", lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    assert _serve_fast.verified(torch.device("cuda", 7), torch.bfloat16) is False


def test_cloud_names():
    assert [gpuinfo.cloud_name(x) for x in ("CLOUD_PROVIDER_UNSPECIFIED", "AZURE", "CLOUD_PROVIDER_GCP", None)] == \
        [None, "Azure", "GCP", None]


def test_check_runs_every_math_mode(cpu_run):
    from sdcguard import modes
    full, _ = cpu_run
    mm = full["math_modes"]
    assert mm["ran"] == list(modes.MODES) and mm["wrong"] == 0 and full["wrong"]["math_modes"] == 0
    pub = check.public(full)
    assert pub["math_modes"]["matmuls"] == mm["matmuls"] and pub["math_modes"]["typical_error_caught"]["int8"] == "exact sums"
    assert "math modes" in check.terminal(pub) and board._shape_problem(pub) is None
    bad = copy.deepcopy(pub)
    bad["math_modes"]["ran"] = "all of them"
    assert "math_modes" in board._shape_problem(bad)


def test_a_wrong_math_mode_fails_the_gpu_and_problems_stay_on_the_card(monkeypatch, tmp_path):
    from sdcguard import modes
    real = modes.screen

    def one_wrong(*a, **k):
        r = real(*a, **k)
        r["modes"]["fp32"]["wrong"] = 1
        r["wrong"] = 1
        return r
    monkeypatch.setattr(modes, "screen", one_wrong)
    full = check.check_one("cpu", minutes=0.06, out_root=str(tmp_path))
    assert full["result"] == "FAIL" and full["wrong"]["math_modes"] == 1
    pub = check.public(full)
    assert board.validate(dict(pub, gpu=dict(pub["gpu"], name="NVIDIA L4"), minutes=5)) is not None   # digest
    full["notes"] = [["info", "x"], ["bad", "something broke"], ["warn", "something else"]]
    rows = check.lines(check.public(full))
    labels = [k for k, _ in rows[:check.CARD_ROWS]]
    assert labels[:5] == ["checked", "math modes", "memory test", "bad", "warn"] and rows[-1] == ("note", "x")
    assert "1 WRONG" in dict(rows)["math modes"] and "math modes 1" in dict(rows)["checked"]
