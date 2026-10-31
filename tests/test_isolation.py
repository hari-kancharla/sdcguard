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
            if mode == "rare_noise":                     # a token flipped in about 3% of answers, nothing else
                seed += random.random() < 0.03
            if mode == "leak_and_noise":                 # the #62124 kind of dependence, plus rare noise
                with lock:
                    salt = state["last_primer"]
                seed += salt % 3 + (random.random() < 0.01)
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


def test_a_server_that_isnt_deterministic_is_never_called_a_leak():
    """An outside review's case: one differing token after any primer was a LEAK, so a server that flips a token in
    2% of requests got LEAK in 9 of 10 runs, and this stand-in got LEAK on 6 of 6 seeds. Noise alone is never a leak:
    a change has to repeat with its primer on a server whose repeats otherwise agree."""
    for mode in ("noisy", "rare_noise"):
        for seed in range(8):
            random.seed(seed)
            r = _run(mode)
            assert r["primer"]["verdict"] == "INCONCLUSIVE" and r["verdict"] != "LEAK", (mode, seed, r["why"])
            if r["primer"]["unstable_cases"]:
                assert "isn't deterministic at batch size 1" in " ".join(r["why"])


def test_a_dependence_on_the_request_before_is_still_caught_on_a_slightly_noisy_server():
    caught = 0
    for seed in range(6):
        random.seed(seed)
        r = _run("leak_and_noise")
        caught += r["primer"]["verdict"] == "LEAK"
        assert r["verdict"] in ("LEAK", "INCONCLUSIVE")
    assert caught >= 4


def test_another_requests_code_in_an_answer_is_a_leak():
    r = _run("canary_leak", canaries=40, concurrency=1)
    assert r["verdict"] == "LEAK" and r["canary"]["leaks"] and "canaries" in r["why"][-1]


def test_a_dead_server_is_never_no_leak():
    import pytest
    with pytest.raises(Exception):
        iso.run("http://127.0.0.1:9", model="m")
