"""sdcguard ready against stand-in metrics pages: what it calls PROTECTED, NOT PROTECTED and INCONCLUSIVE."""
import http.server
import threading

import pytest

from sdcguard import ready


def _page(uuid="GPU-a", pid=11, z=6, layers=112, never=0, executed=100, reads=5, weights=1.0):
    return (f'sdcguard_protection_info{{UUID="{uuid}",pid="{pid}",version="0.4.0",z="{z}",share="1"}} 1.0\n'
            f"sdcguard_layers_checked {layers}\nsdcguard_layers_never_checked {never}\n"
            f"sdcguard_weights_checked_ratio {weights}\nsdcguard_checks_executed_total {executed}\n"
            f"sdcguard_checks_executed_created 1.7e9\nsdcguard_reads_total {reads}\n")


class _Server:
    """Serves pages[0] to the first request, then pages[1] (or pages[0] again)."""

    def __init__(self, *pages):
        self.pages, self.n = pages, 0
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = outer.pages[min(outer.n, len(outer.pages) - 1)].encode()
                outer.n += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/metrics"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _run(servers, vllm=None, extra=()):
    args = [a for s in servers for a in ("--url", s.url)] + ["--window", "0", "--json", *extra]
    if vllm is not None:
        args += ["--vllm", vllm.url]
    return ready.main(args)


@pytest.fixture
def close():
    made = []
    yield made.append
    for s in made:
        s.close()


def _srv(close, *pages):
    s = _Server(*pages)
    close(s)
    return s


def test_two_workers_checking_while_vllm_serves_are_protected(close):
    a = _srv(close, _page(), _page(executed=150))
    b = _srv(close, _page(uuid="GPU-b", pid=12), _page(uuid="GPU-b", pid=12, executed=160))
    v = _srv(close, "vllm:generation_tokens_total 10\n", "vllm:generation_tokens_total 90\n")
    assert _run([a, b], v) == 0


def test_an_idle_server_is_idle_not_protected_when_vllm_says_it_is_idle(close, capsys):
    v = _srv(close, "vllm:generation_tokens_total 10\n")          # no tokens between the reads
    assert _run([_srv(close, _page())], v) == 0
    out = capsys.readouterr().out
    assert '"verdict": "IDLE"' in out and "--probe" in out
    # without vLLM's metrics, checks that didn't move can't be told from stopped checks
    assert _run([_srv(close, _page())]) == 2 and "can't be told apart" in capsys.readouterr().out


@pytest.mark.parametrize("change,needle", [
    (dict(never=2), "never ran"),
    (dict(z=1000), "alarm line"),
    (dict(layers=0), "no linear layer"),
])
def test_a_worker_that_doesnt_really_check_is_not_protected(close, capsys, change, needle):
    a = _srv(close, _page(**change), _page(executed=150, **change))
    assert _run([a]) == 1 and needle in capsys.readouterr().out


def test_a_missing_worker_or_plugin_is_not_protected(close, capsys):
    a = _srv(close, _page())
    dead = _Server(_page())
    dead.close()                                       # nothing listens there now
    assert _run([a, dead]) == 1 and "no answer" in capsys.readouterr().out
    plain = _srv(close, "vllm:generation_tokens_total 5\n")    # a port that answers without sdcguard
    assert _run([plain]) == 1


def test_serving_tokens_without_checks_is_not_protected(close, capsys):
    a = _srv(close, _page(), _page())                  # executed stays at 100
    v = _srv(close, "vllm:generation_tokens_total 10\n", "vllm:generation_tokens_total 90\n")
    assert _run([a], v) == 1 and "serving unchecked" in capsys.readouterr().out


def test_two_workers_on_one_gpu_or_with_other_layer_counts_are_not_protected(close):
    assert _run([_srv(close, _page()), _srv(close, _page(pid=12))]) == 1
    assert _run([_srv(close, _page()), _srv(close, _page(uuid="GPU-b", layers=56))]) == 1


def test_expected_layers_and_weights_are_enforced(close):
    a = _srv(close, _page(weights=0.5), _page(weights=0.5, executed=150))
    assert _run([a], extra=("--layers", "112")) == 0
    assert _run([_srv(close, _page())], extra=("--layers", "56")) == 1
    assert _run([_srv(close, _page(weights=0.5))], extra=("--min-weights", "0.9")) == 1


def test_no_results_yet_or_a_restart_is_inconclusive(close):
    assert _run([_srv(close, _page(reads=0, executed=0))]) == 2
    assert _run([_srv(close, _page(), _page(pid=99, executed=150))]) == 2


def test_parse_sums_labels_and_keeps_the_info_labels():
    values, info = ready.parse('sdcguard_protection_info{UUID="GPU-x",pid="7",version="v",z="6",share="1"} 1\n'
                               'sdcguard_alerts_total{layer="a",UUID="g"} 2\nsdcguard_alerts_total{layer="b"} 1\n'
                               "sdcguard_alerts_created 9\n# HELP x y\n")
    assert values == {"sdcguard_protection_info": 1.0, "sdcguard_alerts_total": 3.0}
    assert info == {"UUID": "GPU-x", "pid": "7", "version": "v", "z": "6", "share": "1"}


def test_the_metrics_page_has_the_info_series():
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard.alerts import Metrics
    m = Metrics()
    m.identify("GPU-q", 6.0, 0.125)
    values, info = ready.parse(generate_latest(m.registry).decode())
    assert info["UUID"] == "GPU-q" and info["z"] == "6" and info["share"] == "0.125" and info["pid"].isdigit()


def test_gpus_bind_the_answer_to_the_servers_workers(close):
    a = _srv(close, _page(), _page(executed=150))
    assert _run([a], extra=("--gpus", "GPU-a")) == 0
    b = _srv(close, _page(), _page(executed=150))
    assert _run([b], extra=("--gpus", "GPU-other")) == 1


class _Api:
    """A stand-in vLLM API: /v1/models and /v1/completions, counting the completions."""

    def __init__(self, status=200):
        self.n, outer = 0, self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'{"data": [{"id": "m"}]}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                outer.n += 1
                body = b'{"choices": [{"text": "ok"}]}'
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_probe_turns_idle_into_protected_when_the_checks_run_for_it(close, monkeypatch):
    monkeypatch.setattr(ready.time, "sleep", lambda s: None)
    api = _Api()
    close(api)
    w = _srv(close, _page(), _page(), _page(executed=180))        # idle, idle, then the probe's checks
    v = _srv(close, "vllm:generation_tokens_total 10\n", "vllm:generation_tokens_total 10\n",
             "vllm:generation_tokens_total 26\n")
    assert _run([w], v, ["--probe", api.url]) == 0 and api.n == 2


def test_probe_finds_checks_stopped_on_an_idle_server(close, monkeypatch, capsys):
    monkeypatch.setattr(ready.time, "sleep", lambda s: None)
    api = _Api()
    close(api)
    w = _srv(close, _page())                                      # the counters never move: the checks stopped
    v = _srv(close, "vllm:generation_tokens_total 10\n", "vllm:generation_tokens_total 10\n",
             "vllm:generation_tokens_total 26\n")
    assert _run([w], v, ["--probe", api.url]) == 1
    out = capsys.readouterr().out
    assert '"verdict": "NOT PROTECTED"' in out and "serving unchecked" in out and "probed" in out


def test_a_probe_that_fails_or_counts_no_tokens_is_inconclusive(close, monkeypatch):
    monkeypatch.setattr(ready.time, "sleep", lambda s: None)
    bad = _Api(status=500)
    close(bad)
    v = _srv(close, "vllm:generation_tokens_total 10\n")
    assert _run([_srv(close, _page())], v, ["--probe", bad.url]) == 2
    api = _Api()
    close(api)
    v = _srv(close, "vllm:generation_tokens_total 10\n")          # vLLM never counts the probe's tokens
    assert _run([_srv(close, _page())], v, ["--probe", api.url]) == 2
