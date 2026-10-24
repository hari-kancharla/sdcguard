"""sdcguard perf against a stand-in server that streams completions the way vLLM does."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sdcguard import perf


def _server(delay=0.001, tokens=8, model="m"):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": model}]} if self.path == "/v1/models" else {"version": "0.32.0"})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert req["ignore_eos"] and req["temperature"] == 0.0 and req["stream"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(tokens):
                time.sleep(delay)
                self.wfile.write(f"data: {json.dumps({'choices': [{'text': 'x'}]})}\n\n".encode())
            self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': tokens}})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}"


def test_a_run_counts_the_servers_tokens_and_times_every_request():
    s, url = _server()
    try:
        r = perf.measure(url, texts=perf.default_prompts(12), concurrency=4, max_tokens=8, rounds=2, warmup=1)
    finally:
        s.shutdown()
    assert r["model"] == "m" and r["server_version"] == "0.32.0" and r["complete"]
    assert all(x["output_tokens"] == 12 * 8 and x["tokens_counted_by_server"] for x in r["rounds"])
    assert r["median"]["output_tokens_per_s"] > 0 and r["median"]["itl_p50_ms"] is not None


def test_a_comparison_needs_the_same_load_and_calls_small_changes_noise():
    s, url = _server(delay=0.002)
    s2, url2 = _server(delay=0.0005)
    try:
        texts = perf.default_prompts(8)
        a = perf.measure(url, texts=texts, concurrency=4, max_tokens=8, rounds=3, warmup=0)
        b = perf.measure(url2, texts=texts, concurrency=4, max_tokens=8, rounds=3, warmup=0)
        c = perf.measure(url2, texts=texts[:4], concurrency=4, max_tokens=8, rounds=3, warmup=0)
    finally:
        s.shutdown()
        s2.shutdown()
    r = perf.compare(a, b)
    assert r["comparable"] and r["metrics"]["output_tokens_per_s"]["direction"] == "better"
    assert not perf.compare(a, c)["comparable"]
    same = perf.compare(a, a)
    assert same["metrics"]["output_tokens_per_s"]["change"] == 0 and same["metrics"]["output_tokens_per_s"]["noise"]


def test_the_default_prompts_are_the_same_every_time():
    assert perf.default_prompts() == perf.default_prompts() and len(set(perf.default_prompts())) > 100
