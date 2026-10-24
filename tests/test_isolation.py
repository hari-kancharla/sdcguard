"""sdcguard isolation against stand-in servers: an isolated one, one whose answers depend on the request before
(the way vLLM #62124 reports), one that isn't deterministic, and one that leaks another request's context."""
import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sdcguard import isolation as iso


def _server(mode="clean"):
    state = {"last_primer": 0, "last_code": None, "n": 0}
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"data": [{"id": "m"}]} if self.path == "/v1/models" else {"version": "0.32.0"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                return self._send({"tokens": [1] + [ord(c) % 997 for c in req["prompt"]], "count": 1})
            assert req.get("temperature") == 0.0
            p = req["prompt"]
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices": [{"text": "x"}]}\n\n')
                return
            if isinstance(p, str):                       # a canary
                code = re.search(r"Access code: ([A-Z0-9-]+)\.", p).group(1)
                with lock:
                    prev, state["last_code"] = state["last_code"], code
                    state["n"] += 1
                    leak = mode == "canary_leak" and prev is not None and state["n"] % 7 == 0
                return self._send({"choices": [{"text": " " + (prev if leak else code)}]})
            if req["max_tokens"] == 2:                   # a primer
                with lock:
                    state["last_primer"] = sum(p)
                return self._send({"choices": [{"text": "", "token_ids": [5, 6], "logprobs": None}]})
            with lock:
                salt = state["last_primer"] if mode == "primer_leak" else 0
            seed = sum(p) + (salt % 3 if mode == "primer_leak" else 0)
            if mode == "noisy":
                seed += random.randint(0, 1)
            toks = [(seed * 31 + i) % 1000 for i in range(req["max_tokens"])]
            return self._send({"choices": [{"text": f"t{seed}", "token_ids": toks,
                                            "logprobs": {"tokens": [f"token_id:{t}" for t in toks],
                                                         "token_logprobs": [-0.1] * len(toks)}}]})
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}"


def _run(mode, **kw):
    s, url = _server(mode)
    try:
        return iso.run(url, canaries=kw.get("canaries", 24), concurrency=kw.get("concurrency", 4))
    finally:
        s.shutdown()


def test_an_isolated_server_shows_no_leak_with_both_probes_judged():
    r = _run("clean")
    assert r["verdict"] == "NO LEAK SEEN", r["why"]
    assert r["primer"]["judged"] == len(iso.PROBES) * len(iso.PRIMER_TOKENS) and r["primer"]["pairs_differing"] == 0
    assert r["canary"]["answered"] + r["canary"]["cancelled"] == 24 and r["canary"]["echoed_own_code"] > 0


def test_answers_that_change_with_the_request_before_are_a_leak():
    r = _run("primer_leak")
    assert r["verdict"] == "LEAK" and r["primer"]["verdict"] == "LEAK" and "primer dependence" in r["why"][0]
    assert r["canary"]["verdict"] == "NO LEAK SEEN"


def test_a_server_that_isnt_deterministic_can_not_be_judged_by_the_primer_probe():
    random.seed(3)
    r = _run("noisy")
    assert r["primer"]["verdict"] in ("INCONCLUSIVE", "LEAK")
    # a control that disagrees takes its case out of the judgment: never a leak from noise alone
    judged = [c for c in r["primer"]["cases"] if c["control_agrees"]]
    assert all(c["control_agrees"] for c in judged)
    assert r["primer"]["judged"] < len(r["primer"]["cases"])


def test_another_requests_code_in_an_answer_is_a_leak():
    r = _run("canary_leak", canaries=40, concurrency=1)
    assert r["verdict"] == "LEAK" and r["canary"]["leaks"] and "canaries" in r["why"][-1]


def test_a_dead_server_is_never_no_leak():
    import pytest
    with pytest.raises(Exception):
        iso.run("http://127.0.0.1:9", model="m")
