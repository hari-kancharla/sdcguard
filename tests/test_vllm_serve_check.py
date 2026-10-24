"""experiments/vllm_serve_check.py without a GPU: the request client against a stand-in completions server, the metrics
parser, and the criteria (docs/METHOD.md) as the run judges them."""
import http.server
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import vllm_serve_check as vs  # noqa: E402


class _Fake:
    """A stand-in for vLLM's /v1/completions: a fixed text, streamed as server-sent events or not."""

    def __init__(self, text="one two three"):
        words = text.split(" ")

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if not body.get("stream"):
                    out = json.dumps({"choices": [{"text": text}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i, w in enumerate(words):
                    piece = w if i == 0 else " " + w
                    self.wfile.write(f"data: {json.dumps({'choices': [{'text': piece}]})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *a):
                pass
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_streamed_and_whole_completions_read_the_same_text_and_load_counts_them():
    f = _Fake()
    try:
        whole = vs.complete(f.port, "m", "p", stream=False)
        streamed = vs.complete(f.port, "m", "p", stream=True)
        assert whole["text"] == streamed["text"] == "one two three" and streamed["ttft"] is not None
        r = vs.load(f.port, "m", ["a"] * 10, at_once=4)
        assert r["completed"] == 10 and not r["errors"] and r["ttft_p99_ms"] is not None
    finally:
        f.server.shutdown()


def test_the_sink_keeps_what_is_posted():
    s = vs.Sink()
    try:
        assert vs._request(s.port, "POST", "/hook", {"detail": {"where": vs.FLIP}})[0] == 200
        assert s.bodies == [{"detail": {"where": vs.FLIP}}]
    finally:
        s.server.shutdown()


def test_metrics_are_summed_over_labels_and_counters_keep_their_total_name():
    text = ("# HELP sdcguard_alerts Alerts\n# TYPE sdcguard_alerts counter\n"
            'sdcguard_alerts_total{layer="a",UUID="g"} 2.0\nsdcguard_alerts_total{layer="b",UUID="g"} 1.0\n'
            'sdcguard_alerts_created{layer="a",UUID="g"} 1.7e9\nsdcguard_layers_checked 112.0\n'
            "sdcguard_reads_total 40.0\n")
    m = vs.parse_metrics(text)
    assert m == {"sdcguard_alerts_total": 3.0, "sdcguard_layers_checked": 112.0, "sdcguard_reads_total": 40.0}


def test_percentiles_are_nearest_rank():
    xs = list(range(1, 101))
    assert vs.percentile(xs, 50) == 50 and vs.percentile(xs, 99) == 99 and vs.percentile([], 50) is None
    assert vs.percentile([3.0], 99) == 3.0


def _role(role, texts, **kw):
    r = {"role": role, "model": vs.MODEL, "tp": 1, "ready": True, "identity": texts,
         "load": [{"at_once": 1, "requests": 4, "completed": 4, "errors": [], "output_tokens_per_s": 100.0,
                   "latency_p50_ms": 10.0, "latency_p99_ms": 20.0, "ttft_p50_ms": 2.0, "ttft_p99_ms": 4.0}],
         "stop": {"exited_in_s": 3.0, "killed": False, "code": 0}}
    if role == "guarded":
        clean = {"sdcguard_layers_checked": 112.0, "sdcguard_layers_never_checked": 0.0, "sdcguard_reads_total": 9.0}
        r.update(metrics_clean=[clean], alerts_clean=[], alerts_after_flip=[{"detail": {"where": vs.FLIP}}],
                 alert_seconds=4.0, metrics_after_flip=[{**clean, "sdcguard_all_alerts_total": 1.0}],
                 flip={"status": 200, "results": [vs.FLIP]})
    r.update(kw)
    return r


TEXTS = [f"t{i}" for i in range(vs.IDENTITY_PROMPTS)]


def test_a_run_meeting_every_criterion_passes():
    ok, fails, s = vs.judge(_role("plain", TEXTS), _role("guarded", TEXTS))
    assert ok and not fails and s["same_one_at_a_time"] == f"{vs.IDENTITY_PROMPTS}/{vs.IDENTITY_PROMPTS}"


@pytest.mark.parametrize("change,criterion", [
    ({"identity": TEXTS[:-1] + ["other"]}, "2:"),
    ({"metrics_clean": [{"sdcguard_layers_checked": 56.0, "sdcguard_reads_total": 1.0}]}, "3:"),
    ({"metrics_clean": [{"error": "ConnectionRefusedError"}]}, "3:"),
    ({"metrics_clean": [{"sdcguard_layers_checked": 112.0, "sdcguard_layers_never_checked": 2.0,
                         "sdcguard_reads_total": 1.0}]}, "3:"),
    ({"metrics_clean": [{"sdcguard_layers_checked": 112.0, "sdcguard_reads_total": 0.0}]}, "3:"),
    ({"alerts_clean": [{"detail": {"where": "x"}}]}, "4:"),
    ({"alerts_after_flip": []}, "5:"),
    ({"metrics_after_flip": [{"sdcguard_layers_checked": 112.0}]}, "5:"),
    ({"stop": {"exited_in_s": None, "killed": True, "code": -9}}, "6:"),
    ({"ready": False, "identity": None}, "1:"),
])
def test_each_criterion_can_fail(change, criterion):
    ok, fails, _ = vs.judge(_role("plain", TEXTS), _role("guarded", TEXTS, **change))
    assert not ok and any(f.startswith(criterion) for f in fails), fails


def test_failed_requests_under_load_fail_the_run():
    bad = _role("guarded", TEXTS)
    bad["load"][0] = {**bad["load"][0], "completed": 3, "errors": ["HTTP 500"]}
    ok, fails, _ = vs.judge(_role("plain", TEXTS), bad)
    assert not ok and any("requests failed" in f for f in fails)
