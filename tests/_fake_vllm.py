"""A stand-in for `vllm serve MODEL [flags] --port P`, for testing `sdcguard qualify` on a CPU: the endpoints
qualify's checks use (/health, /v1/models, /version, /tokenize, /v1/completions with logprobs, prompt_logprobs,
streaming and cache_salt), and, with SDCGUARD_METRICS_PORT set, an sdcguard metrics page whose checks go up with
each request. Flags of its own change what it does:

  --shift X   next-token distributions moved by X (a quality change)
  --leak      a short prompt's answer depends on the request before it (the way vLLM #62124 reports)
  --slow S    S seconds per generated token
  --fault     log an alert with changed weights to SDCGUARD_LOG at the first request
  --fail      exit at once (a server that doesn't start)
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

V = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--leak", action="store_true")
    ap.add_argument("--slow", type=float, default=0.0005)
    ap.add_argument("--fault", action="store_true")
    ap.add_argument("--fail", action="store_true")
    ap.add_argument("--attention-backend", default="FLASH_ATTN")
    ap.add_argument("--quantization", default=None)
    a, _ = ap.parse_known_args()
    if a.fail:
        sys.exit(3)
    state = {"last": 0, "checks": 0, "reads": 0, "faulted": False}
    lock = threading.Lock()

    def dist(prefix):
        h = sum((i + 1) * t for i, t in enumerate(prefix[-4:])) % V
        if a.leak and len(prefix) < 16:
            h = (h + state["last"] % 3) % V
        logits = [-(((t - h) % V) * 0.7) + (a.shift if t == (h + 1) % V else 0.0) for t in range(V)]
        z = math.log(sum(math.exp(x) for x in logits))
        return {t: x - z for t, x in enumerate(logits)}

    def tok(text):
        return [1] + [ord(c) % V for c in text[:40]]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send({})
            if self.path == "/v1/models":
                return self._send({"data": [{"id": a.model}]})
            if self.path == "/version":
                return self._send({"version": "0.0.fake"})
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                text = req.get("prompt") or req["messages"][0]["content"]
                return self._send({"tokens": tok(text), "count": len(tok(text))})
            with lock:
                state["checks"] += 112
                state["reads"] += 1
                if a.fault and not state["faulted"] and os.environ.get("SDCGUARD_LOG"):
                    state["faulted"] = True
                    with open(os.environ["SDCGUARD_LOG"], "a") as f:
                        f.write(json.dumps({"detector": "serve_checksum", "step": None, "time": time.time(),
                                            "detail": {"where": "model.layers.3.mlp.down_proj",
                                                       "weights_changed_since_load": True}}) + "\n")
            p = req["prompt"]
            ids = tok(p) if isinstance(p, str) else list(p)
            if req.get("prompt_logprobs"):
                plp = [None]
                for j in range(1, len(ids)):
                    d = dist(ids[:j])
                    keep = set(sorted(d, key=lambda t: -d[t])[:req["prompt_logprobs"]]) | {ids[j]}
                    plp.append({str(t): {"logprob": d[t], "rank": 1, "decoded_token": "x"} for t in keep})
                return self._send({"choices": [{"index": 0, "text": "", "prompt_logprobs": plp}]})
            n = req.get("max_tokens", 16)
            seq, toks, tops = list(ids), [], []
            for _ in range(n):
                d = dist(seq)
                best = max(d, key=lambda t: d[t])
                tops.append({f"token_id:{t}": d[t] for t in sorted(d, key=lambda t: -d[t])[:req.get("logprobs") or 1]})
                toks.append(best)
                seq.append(best)
            if n <= 2:
                with lock:
                    state["last"] = sum(ids)
            text = " " + "".join(chr(97 + t % 26) for t in toks)
            if isinstance(p, str) and "Access code:" in p:
                text = " " + p.split("Access code: ")[1].split(".")[0]
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for t in toks:
                    time.sleep(a.slow)
                    self.wfile.write(f"data: {json.dumps({'choices': [{'text': chr(97 + t % 26)}]})}\n\n".encode())
                self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': n}})}\n\n"
                                 .encode())
                self.wfile.write(b"data: [DONE]\n\n")
                return
            time.sleep(a.slow * n)
            lp = {"tokens": [f"token_id:{t}" for t in toks], "top_logprobs": tops,
                  "token_logprobs": [tops[i].get(f"token_id:{t}", -1.0) for i, t in enumerate(toks)]}
            return self._send({"choices": [{"index": 0, "text": text, "token_ids": toks, "logprobs": lp}]})

    mp = os.environ.get("SDCGUARD_METRICS_PORT")
    if mp and os.environ.get("SDCGUARD") == "1":
        class M(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                text = os.environ.get("SDCGUARD_SHARE", "1").strip()   # the weights checked follow the share
                try:
                    num, _, den = text.rstrip("%").partition("/")
                    share = float(num) / (float(den) if den else 100.0 if text.endswith("%") else 1.0)
                except ValueError:
                    share = 1.0
                ratio = round(min(1.0, share) * 0.97, 4)
                from sdcguard.stack import fingerprint
                st = {"model": a.model, "dtype": "bfloat16", "attention": a.attention_backend, "vllm": "0.0.fake",
                      **({"quantization": a.quantization} if a.quantization else {})}
                st["fingerprint"] = fingerprint(st)
                stack_line = "sdcguard_stack_info{" + ",".join(f'{k}="{v}"' for k, v in st.items()) + "} 1.0\n"
                with lock:
                    body = (f'sdcguard_protection_info{{UUID="GPU-fake",pid="{os.getpid()}",version="x",z="6",'
                            f'share="{os.environ.get("SDCGUARD_SHARE", "1")}"}} 1.0\n'
                            f"sdcguard_layers_checked 112\nsdcguard_layers_never_checked 0\n"
                            f"sdcguard_weights_checked_ratio {ratio}\n"
                            f"sdcguard_checks_executed_total {state['checks']}\n"
                            f"sdcguard_reads_total {state['reads']}\n" + stack_line).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        ms = ThreadingHTTPServer(("127.0.0.1", int(mp)), M)
        threading.Thread(target=ms.serve_forever, daemon=True).start()
    ThreadingHTTPServer.request_queue_size = 256       # vLLM's server takes far more than socketserver's 5 at once
    ThreadingHTTPServer.daemon_threads = True
    s = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
