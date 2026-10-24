"""sdcguard kernels: reading cuBLASLt's Trace lines into an inventory, coverage of a workload by tests, the
kernel conditions of NVIDIA's listed issues, diffs between stacks, and recording a command. The log lines are
written here in the format cuBLASLt prints (CUBLASLT_LOG_LEVEL=5)."""
import json
import os
import sys
import textwrap

from sdcguard import gate, kernels


def _line(fn="cublasLtHSHMatmul", t="R_16F", m=1000, n=16, k=2048, transa="OP_T", sms=None,
          algo="algoId=66 tile=MATMUL_TILE_64x16 stages=MATMUL_STAGES_64xAUTO customOption=1 "
               "clusterShape=CLUSTER_SHAPE_4x1x1", epi=None):
    comp = f"computeType=COMPUTE_32F scaleType=R_32F transa={transa}"
    if sms:
        comp += f" smCountTarget={sms}"
    if epi:
        comp += f" epilogue={epi}"
    a_rows, a_cols = (k, m) if transa == "OP_T" else (m, k)
    return (f"[2026-10-24 10:00:00][cublasLt][7][Trace][{fn}] A=0X7C59 Adesc=[type={t} rows={a_rows} cols={a_cols} "
            f"ld={a_rows}] B=0X7C57 Bdesc=[type={t} rows={k} cols={n} ld={k}] C=0X7C58 Cdesc=[type={t} rows={m} "
            f"cols={n} ld={m}] D=0X7C58 Ddesc=[type={t} rows={m} cols={n} ld={m}] computeDesc=[{comp}] "
            f"algo=[{algo}] workSpace=0X0 workSpaceSizeInBytes=0 beta=0 outOfPlace=0 stream=0X0")


API = ("[2026-10-24 10:00:00][cublasLt][7][Api][cublasLtHSHMatmul] A=0X7C59 Adesc=0X7FFE B=0X7C57 Bdesc=0X7FFE "
       "computeDesc=0X7FFE algo=0X7FFE workSpace=0X0 workSpaceSizeInBytes=0 stream=0X0")
SPLITK = "algoId=55 tile=MATMUL_TILE_64x32 stages=MATMUL_STAGES_8x3 reductionScheme=REDUCTION_SCHEME_COMPUTE_TYPE numSplitsK=8"


def test_a_trace_line_reads_as_a_call_and_names_its_kernel():
    c = kernels.parse_line(_line(sms=18))
    assert (c["m"], c["n"], c["k"]) == (1000, 16, 2048) and c["types"] == ["fp16"] * 4
    assert c["sm_count_target"] == "18" and c["algo"]["clusterShape"] == "4x1x1" and c["algo"]["tile"] == "64x16"
    key = kernels.kernel_key(c)
    assert "algoId=66 tile=64x16 stages=64xAUTO customOption=1 clusterShape=4x1x1" in key and "fp16xfp16->fp16" in key
    assert "sms=18" in kernels.call_key(c) and "m=1000 n=16 k=2048" in kernels.call_key(c)
    assert kernels.parse_line(API) is None and kernels.parse_line("nothing here") is None
    plain = kernels.parse_line(_line(transa="OP_N", m=64, n=8, k=32))
    assert (plain["m"], plain["n"], plain["k"]) == (64, 8, 32)


def test_the_inventory_counts_calls_kernels_shapes_and_work_and_keeps_unreadable_lines_out():
    lines = [_line(), _line(), _line(n=32), _line(t="R_32F", algo=SPLITK), API,
             "[2026-10-24][cublasLt][7][Trace][cublasLtMatmul] A=0X1 broken"]
    inv = kernels.inventory(lines, {"command": "x"})
    assert inv["calls"] == 4 and inv["unreadable_lines"] == 1 and len(inv["kernels"]) == 2
    top = inv["kernels"][0]
    assert top["calls"] == 3 and len(top["shapes"]) == 2
    assert top["work"] == 2 * 1000 * 2048 * (16 + 16 + 32) and inv["work"] == sum(k["work"] for k in inv["kernels"])
    json.dumps(inv)                                        # saveable as it is


def test_issue_6862950_is_matched_by_algorithm_cluster_and_precision_only():
    def flags(**kw):
        return [f["issue"] for f in kernels.inventory([_line(**kw)])["kernels"][0]["issues"]]
    assert flags() == ["6862950"] and flags(t="R_16BF", sms=52) == ["6862950"]
    assert flags(algo="algoId=66 tile=MATMUL_TILE_64x16 clusterShape=CLUSTER_SHAPE_2x1x1") == []
    assert flags(t="R_32F") == [] and flags(algo="algoId=21 clusterShape=CLUSTER_SHAPE_4x1x1") == []
    why = kernels.inventory([_line(sms=18)])["kernels"][0]["issues"][0]["why"]
    assert "SM count target of 18" in why
    split = "algoId=66 tile=MATMUL_TILE_64x16 numSplitsK=3 clusterShape=CLUSTER_SHAPE_1x1x1"
    assert flags(algo=split, k=2048) == ["6580689"] and flags(algo=split, k=2049 * 3) == []
    grouped = kernels.inventory([_line(fn="cublasLtGroupedMatmul", algo="algoId=7")])["kernels"][0]["issues"]
    assert grouped and "grouped" in grouped[0]["why"]


def test_coverage_counts_kernels_and_exact_calls_by_work_and_lists_what_no_test_ran():
    work = kernels.inventory([_line(), _line(n=32), _line(t="R_32F", algo=SPLITK)])
    burn = kernels.inventory([_line(t="R_32F", algo=SPLITK, m=8192, n=8192, k=8192)], {"command": "gpu_burn"})
    r = kernels.coverage(work, [burn])
    assert r["kernel"] == {"workload": 2, "exercised": 1, "work_share_exercised": r["kernel"]["work_share_exercised"]}
    assert 0 < r["kernel"]["work_share_exercised"] < 1
    assert r["call"]["exercised"] == 0                      # same kernel, other shape: not the same call
    assert [n["issues"][0]["issue"] for n in r["not_exercised"]] == ["6862950"]
    same = kernels.coverage(work, [work])
    assert same["kernel"]["work_share_exercised"] == 1.0 and same["call"]["exercised"] == same["call"]["workload"]
    empty = kernels.coverage(work, [kernels.inventory([], {"command": "dcgmi diag -r 3"})])
    assert empty["kernel"]["exercised"] == 0 and any("nv-hostengine" in n for n in empty["notes"])


def test_diff_names_the_kernels_a_change_added():
    before = kernels.inventory([_line(t="R_32F", algo=SPLITK)])
    after = kernels.inventory([_line(t="R_32F", algo=SPLITK), _line()])
    d = kernels.diff(before, after)
    assert d["same"] == 1 and len(d["new"]) == 1 and d["new"][0]["issues"][0]["issue"] == "6862950"
    assert 0 < d["work_share_on_new_kernels"] < 1 and d["gone"] == []


def test_record_runs_the_command_with_the_log_on_and_reads_every_process_log(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import os, sys
        path = os.environ["CUBLASLT_LOG_FILE"].replace("%i", str(os.getpid()))
        assert os.environ["CUBLASLT_LOG_LEVEL"] == "5"
        with open(path, "w") as f:
            f.write({_line()!r} + "\\n" + {_line(n=32)!r} + "\\n")
    """))
    inv = kernels.record([sys.executable, str(child)], str(tmp_path / "logs"))
    assert inv["calls"] == 2 and inv["source"]["exit_code"] == 0 and len(inv["source"]["logs"]) == 1
    sleeper = tmp_path / "sleep.py"
    sleeper.write_text("import time\ntime.sleep(60)\n")
    inv = kernels.record([sys.executable, str(sleeper)], str(tmp_path / "logs2"), seconds=0.5)
    assert inv["source"]["stopped_by"] == "time" and inv["calls"] == 0 and inv["source"]["seconds"] < 30


def test_the_command_line_saves_shows_compares_and_diffs(tmp_path, capsys):
    log = tmp_path / "w.log"
    log.write_text("\n".join([_line(), _line(t="R_32F", algo=SPLITK)]) + "\n")
    tlog = tmp_path / "t.log"
    tlog.write_text(_line(t="R_32F", algo=SPLITK) + "\n")
    w, t = str(tmp_path / "w.json"), str(tmp_path / "t.json")
    assert kernels.main(["parse", str(log), "--out", w]) == 0 and kernels.main(["parse", str(tlog), "--out", t]) == 0
    capsys.readouterr()
    assert kernels.main(["coverage", w, "--test", t]) == 0
    out = capsys.readouterr().out
    assert "ran 1 of the workload's 2 kernels" in out and "matches issue 6862950" in out
    assert kernels.main(["diff", t, w]) == 0 and "1 new" in capsys.readouterr().out
    assert kernels.main(["show", w]) == 0
    empty = tmp_path / "e.log"
    empty.write_text(API + "\n")
    assert kernels.main(["parse", str(empty)]) == 2


def test_the_gate_reads_the_typed_entry_points_legacy_cublas_goes_through():
    r = gate.parse_lt_log("\n".join([_line(), API, _line(fn="cublasLtMatmul")]))
    assert r["calls"] == 2 and r["kernels"] == {"algoId=66 tile=64x16 stages=64xAUTO cluster=4x1x1": 2}
