"""gate model-ref and model-check against a running server, with a stand-in that answers /tokenize and
/v1/completions the way vLLM 0.32 does (token ids as "token_id:N" with return_tokens_as_token_ids, prompt_logprobs as
{token id: {"logprob", "rank", "decoded_token"}})."""
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sdcguard import gate_model as gm

V = 50


def _dist(prefix, shift):
    """A deterministic next-token distribution over V tokens given the prefix (shift perturbs it)."""
    h = sum((i + 1) * t for i, t in enumerate(prefix[-4:])) % V
    logits = [-(((t - h) % V) * 0.7) + (shift if t == (h + 1) % V else 0.0) for t in range(V)]
    z = math.log(sum(math.exp(x) for x in logits))
    return {t: x - z for t, x in enumerate(logits)}


def _server(shift=0.0):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"data": [{"id": "m"}]} if self.path == "/v1/models" else {"version": "0.32.0"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                text = req["messages"][0]["content"]
                return self._send({"tokens": [1] + [ord(c) % V for c in text[:12]] + [2], "count": 14})
            ids = list(req["prompt"])
            if req.get("prompt_logprobs"):
                plp = [None]
                for j in range(1, len(ids)):
                    d = _dist(ids[:j], shift)
                    top = sorted(d, key=lambda t: -d[t])[:req["prompt_logprobs"]]
                    keep = set(top) | {ids[j]}
                    plp.append({str(t): {"logprob": d[t], "rank": 1, "decoded_token": "x"} for t in keep})
                return self._send({"choices": [{"index": 0, "text": "", "prompt_logprobs": plp}]})
            toks, tops, seq = [], [], list(ids)
            for _ in range(req["max_tokens"]):
                d = _dist(seq, shift)
                best = max(d, key=lambda t: d[t])
                top = sorted(d, key=lambda t: -d[t])[:req["logprobs"]]
                tops.append({f"token_id:{t}": d[t] for t in top})
                toks.append(best)
                seq.append(best)
            return self._send({"choices": [{"index": 0, "text": "", "token_ids": toks,
                                            "logprobs": {"tokens": [f"token_id:{t}" for t in toks],
                                                         "top_logprobs": tops}}]})
    s = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}"


def test_a_reference_from_a_server_checks_clean_on_the_same_server_and_fails_on_a_changed_one(tmp_path):
    s, url = _server()
    bad, burl = _server(shift=4.0)
    try:
        ref = tmp_path / "ref.json"
        info = gm.make_ref_url(url, str(ref), tokens=8, extra_stack={"gpu": "fake"})
        assert info["prompts"] == len(gm.PROMPTS) and info["positions"] == 8 * len(gm.PROMPTS)
        r = json.loads(ref.read_text())
        assert r["via"] == "server" and r["stack"]["vllm"] == "0.32.0" and r["stack"]["gpu"] == "fake"
        same = gm.check_url(str(ref), str(tmp_path / "same.json"), url)
        assert same["verdict"]["verdict"] == "PASS" and same["summary"]["mean_kl"] < 1e-9
        assert same["summary"]["top1_agreement"] == 1.0 and same["summary"]["missing_positions"] == 0
        changed = gm.check_url(str(ref), str(tmp_path / "new.json"), burl, noise_path=str(tmp_path / "same.json"))
        assert changed["verdict"]["verdict"] == "FAIL" and changed["stack"]["url"] == burl
    finally:
        s.shutdown()
        bad.shutdown()


def test_a_server_that_stops_answering_leaves_the_check_inconclusive(tmp_path):
    s, url = _server()
    try:
        ref = tmp_path / "ref.json"
        gm.make_ref_url(url, str(ref), tokens=4)
    finally:
        s.shutdown()
    run = gm.check_url(str(ref), str(tmp_path / "x.json"), url.rsplit(":", 1)[0] + ":9", model="m")
    assert run["verdict"]["verdict"] == "INCONCLUSIVE"
