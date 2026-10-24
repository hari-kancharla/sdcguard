"""sdcguard consistency against the stand-in server (tests/_fake_vllm.py): a consistent server passes both probes
with its prefix-cache hits confirmed; a server whose cached prefixes are read wrong, and one whose decode drifts from
prefill past 97 tokens of KV (vLLM #54035's shape), are INCONSISTENT for the right probe; with prefix caching off
the cache probe can't judge; against a baseline, decode gaps are judged relative to the baseline's own."""
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

from sdcguard import consistency as cs

HERE = os.path.dirname(os.path.abspath(__file__))


def _free():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def server():
    procs = []

    def start(*flags):
        port = _free()
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "_fake_vllm.py"), "fake/m", "--port", str(port),
                              "--slow", "0", *flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        url = f"http://127.0.0.1:{port}"
        for _ in range(200):
            try:
                urllib.request.urlopen(url + "/health", timeout=1)
                return url
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("the stand-in server didn't start")

    yield start
    for p in procs:
        p.terminate()
        p.wait(timeout=10)


def test_a_consistent_server_passes_with_its_cache_hits_confirmed(server):
    r = cs.run(server())
    assert r["verdict"] == "CONSISTENT", r["why"]
    pc, dp = r["prefix_cache"], r["decode_prefill"]
    assert pc["verdict"] == "CONSISTENT" and pc["cache_hits_confirmed"] is True and len(pc["cases"]) == len(cs.DOCS)
    assert all(c["cached"]["max_gap"] == 0 and c["floor"]["max_gap"] == 0 for c in pc["cases"])
    assert dp["verdict"] == "CONSISTENT" and dp["largest_gap"] == 0
    assert all(c["tokens"] == 160 and c["prompt_tokens"] + c["tokens"] > 128 for c in dp["cases"])


def test_wrong_cached_prefixes_are_inconsistent(server):
    r = cs.run(server("--prefix-bug"))
    assert r["verdict"] == "INCONSISTENT" and r["prefix_cache"]["verdict"] == "INCONSISTENT"
    assert r["decode_prefill"]["verdict"] == "CONSISTENT"
    c = r["prefix_cache"]["cases"][0]
    assert c["cache_hit"] is True and c["cached"]["significant"] and not c["floor"]["significant"]
    assert "prefix cache" in r["why"][0]


def test_decode_drifting_from_prefill_past_97_tokens_is_inconsistent(server):
    r = cs.run(server("--decode-gap", "0.6"))
    dp = r["decode_prefill"]
    assert dp["verdict"] == "INCONSISTENT" and r["prefix_cache"]["verdict"] == "CONSISTENT"
    first = [c["first_over_min_gap"] for c in dp["cases"]]
    assert all(f is not None and f["kv_length"] >= 97 for f in first)
    assert abs(dp["largest_gap"] - 0.6) < 1e-6


def test_without_prefix_caching_the_cache_probe_cant_judge(server):
    r = cs.run(server("--no-enable-prefix-caching"), decode=False)
    assert r["prefix_cache"]["verdict"] == "INCONCLUSIVE" and r["prefix_cache"]["cache_hits_confirmed"] is False


def test_against_a_baseline_decode_gaps_are_relative(server):
    base = cs.run(server("--decode-gap", "0.05"))
    small = cs.run(server("--decode-gap", "0.08"))
    big = cs.run(server("--decode-gap", "0.4"))
    assert cs.judge_against(small, base)["decode_prefill"] == "CONSISTENT"     # within 4x the baseline's
    j = cs.judge_against(big, base)
    assert j["decode_prefill"] == "INCONSISTENT" and j["verdict"] == "INCONSISTENT" and "decode vs prefill" in j["why"][0]


def test_the_command_line(server, tmp_path):
    url = server("--prefix-bug")
    out = tmp_path / "c.json"
    p = subprocess.run([sys.executable, "-m", "sdcguard", "consistency", "--url", url, "--no-decode", "--out", str(out)],
                       capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(os.path.dirname(HERE), "src"),
                                                                        os.environ.get("PYTHONPATH", "")])))
    assert p.returncode == 1 and p.stdout.startswith("INCONSISTENT")
    assert out.exists()


def test_without_the_cache_counter_the_cache_probe_claims_nothing_either_way(server):
    """vLLM's prefix-cache counter can't be read: the cached run may not have hit the cache, so neither a pass nor
    a fail is about the cached path."""
    r = cs.run(server("--no-vllm-metrics"), decode=False)
    assert r["prefix_cache"]["verdict"] == "INCONCLUSIVE" and r["prefix_cache"]["cache_hits_confirmed"] is None
    r = cs.run(server("--no-vllm-metrics", "--prefix-bug"), decode=False)
    assert r["prefix_cache"]["verdict"] == "INCONCLUSIVE"


def test_a_partial_decode_measurement_decides_nothing():
    full = {"verdict": "CONSISTENT", "largest_gap": 0.05, "cases": []}
    partial = {"verdict": "INCONCLUSIVE", "largest_gap": 0.06, "cases": [], "partial_cases": 1}
    ok = {"verdict": "CONSISTENT", "cases": [], "cache_hits_confirmed": True}
    j = cs.judge_against({"prefix_cache": ok, "decode_prefill": partial}, {"decode_prefill": full})
    assert j["decode_prefill"] == "INCONCLUSIVE" and j["verdict"] == "INCONCLUSIVE"
    # a baseline that couldn't be measured sets no line: the candidate stands on its own
    j = cs.judge_against({"prefix_cache": ok, "decode_prefill": full}, {"decode_prefill": partial})
    assert j["decode_prefill"] == "CONSISTENT" and "decode_line" not in j


def test_the_decode_probe_keeps_every_positions_gap_and_splits_them_by_kv_length(server):
    r = cs.run(server("--decode-gap", "0.4"), prefix=False)
    c = r["decode_prefill"]["cases"][0]
    assert len(c["gaps"]) == c["tokens"] and c["missing"] == 0 and r["decode_prefill"]["kv_split"] == 96
    assert c["mean_gap_kv_past_split"] > c["mean_gap_kv_upto_split"]       # the stand-in drifts past 97 tokens
