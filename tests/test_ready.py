"""sdcguard ready against stand-in metrics pages: what it calls PROTECTED, NOT PROTECTED and INCONCLUSIVE."""
import http.server
import threading

import pytest

from sdcguard import ready


def _page(uuid="GPU-a", pid=11, z=6, layers=112, never=0, executed=100, reads=5, weights=1.0, alerts=None,
          confirmed=None):
    return (f'sdcguard_protection_info{{UUID="{uuid}",pid="{pid}",version="0.4.0",z="{z}",share="1"}} 1.0\n'
            f"sdcguard_layers_checked {layers}\nsdcguard_layers_never_checked {never}\n"
            f"sdcguard_weights_checked_ratio {weights}\nsdcguard_checks_executed_total {executed}\n"
            f"sdcguard_checks_executed_created 1.7e9\nsdcguard_reads_total {reads}\n"
            + (f"sdcguard_all_alerts_total {alerts}\n" if alerts is not None else "")
            + (f"sdcguard_confirmed_alerts_total {confirmed}\n" if confirmed is not None else ""))


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


def test_probe_lets_a_server_that_hasnt_served_yet_become_ready(close, monkeypatch, capsys):
    """As a readiness probe sees every new server: no request yet, so no checks have run. Without --probe that's
    INCONCLUSIVE (and no traffic would ever reach it); with --probe the two short requests show the checks running,
    or not."""
    monkeypatch.setattr(ready.time, "sleep", lambda s: None)
    fresh = _page(reads=0, executed=0)
    v = lambda: _srv(close, "vllm:generation_tokens_total 0\n", "vllm:generation_tokens_total 0\n",  # noqa: E731
                     "vllm:generation_tokens_total 16\n")
    assert _run([_srv(close, fresh)], v()) == 2
    api = _Api()
    close(api)
    assert _run([_srv(close, fresh, fresh, _page(reads=1, executed=40))], v(), ["--probe", api.url]) == 0
    out = capsys.readouterr().out
    assert api.n == 2 and '"verdict": "PROTECTED"' in out and "hadn't served a request yet" in out
    api2 = _Api()
    close(api2)
    assert _run([_srv(close, fresh)], v(), ["--probe", api2.url]) == 1           # the probe's tokens, no checks
    assert "serving unchecked" in capsys.readouterr().out


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


# ---------------------------------------------------------------- --qualified: the checks the qualification required
CHECKS = "attention=off,fp8=off,moe=on,norms=off,share=1,spot=off,weights=off,z=6"


def _stack(**over):
    from sdcguard import stack
    st = {"model": "m", "vllm": "0.32.0", "weights_digest": "0123456789abcdef", "lora": "off", "sdcguard": "0.5.0",
          "checks": CHECKS, **over}
    st = {k: v for k, v in st.items() if v is not None}
    st["fingerprint"] = stack.fingerprint(st)
    return st


def _stack_line(st):
    return "sdcguard_stack_info{" + ",".join(f'{k}="{v}"' for k, v in st.items()) + "} 1.0\n"


def _decision(tmp_path, st, name="d.json", policy=None, created="now"):
    import json
    import time
    d = {"decision": "ACCEPT", "scope": {"stacks": {"candidate": {"deployed": st}}},
         "policy": policy if policy is not None else {"arithmetic": {"min_weights_checked": 0.5}}}
    if created == "now":
        d["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    elif created is not None:
        d["created"] = created
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def _live(close, st, **page):
    line = _stack_line(st)
    return _srv(close, _page(**page) + line, _page(**{**page, "executed": 150}) + line)


def _verdict(capsys):
    import json
    return json.loads(capsys.readouterr().out)


def test_ready_holds_a_server_to_the_checks_its_qualification_required(close, tmp_path, capsys):
    """An external review cut the share of weights checked to 12.5% while the decision's policy required 50%: the
    stack still matched, and ready said PROTECTED with its own defaults. ready now takes the stricter of its command
    line and the decision on the share of weights checked, the alarm line and the detectors run."""
    st = _stack()
    dec = _decision(tmp_path, st)
    assert _run([_live(close, st, weights=0.125)], extra=("--qualified", dec)) == 1
    r = _verdict(capsys)
    assert r["verdict"] == "NOT PROTECTED" and any("0.125" in p and "min_weights_checked" in p for p in r["problems"])
    assert _run([_live(close, st)], extra=("--qualified", dec)) == 0
    assert any("held to the qualification's requirements" in n for n in _verdict(capsys)["notes"])
    # the command line's value when it is the stricter one
    assert _run([_live(close, st, weights=0.6)], extra=("--qualified", dec, "--min-weights", "0.9")) == 1
    capsys.readouterr()
    # a qualification run at a stricter alarm line holds the server to it
    strict = _stack(checks=CHECKS.replace("z=6", "z=4"))
    assert _run([_live(close, strict)], extra=("--qualified", _decision(tmp_path, strict, "z.json"))) == 1
    assert any("above the qualification's line 4" in p for p in _verdict(capsys)["problems"])
    # a detector the qualification ran with, off now
    weaker = _stack(checks=CHECKS.replace("moe=on", "moe=off"))
    assert _run([_live(close, weaker)], extra=("--qualified", dec)) == 1
    r = _verdict(capsys)
    assert r["verdict"] == "NOT PROTECTED" and any("moe" in p and "weaker" in p for p in r["problems"])
    # running more checks than the qualification did is fine
    more = _stack(checks=CHECKS.replace("norms=off", "norms=on,norm_z=6"))
    assert _run([_live(close, more)], extra=("--qualified", dec)) == 0
    capsys.readouterr()
    # another checker version: its checks can't be shown to be the ones qualified
    other = _stack(sdcguard="0.4.9")
    assert _run([_live(close, other)], extra=("--qualified", dec)) == 2
    assert any("0.4.9" in u for u in _verdict(capsys)["unsure"])


def test_a_decision_that_doesnt_record_a_requirement_leaves_it_to_the_command_line(close, tmp_path, capsys):
    st = _stack(checks=None, sdcguard=None)
    dec = _decision(tmp_path, st, policy={})
    assert _run([_live(close, st, weights=0.125)], extra=("--qualified", dec)) == 0
    notes = " ".join(_verdict(capsys)["notes"])
    assert "left to the command line" in notes and "detectors" in notes
    # a server that doesn't say which checks it runs: unknown only when the qualification ran a detector
    quiet = _stack(checks=None)
    off = _stack(checks=CHECKS.replace("moe=on", "moe=off"))
    assert _run([_live(close, quiet)], extra=("--qualified", _decision(tmp_path, off, "off.json"))) == 0
    capsys.readouterr()
    assert _run([_live(close, quiet)], extra=("--qualified", _decision(tmp_path, _stack(), "on.json"))) == 2
    assert any("doesn't publish which checks" in u for u in _verdict(capsys)["unsure"])


def test_an_old_decision_is_inconclusive_past_max_age_days(close, tmp_path, capsys):
    import time
    st = _stack()
    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30 * 86400))
    dec = _decision(tmp_path, st, created=old)
    assert _run([_live(close, st)], extra=("--qualified", dec)) == 0
    assert any("30 days ago" in n for n in _verdict(capsys)["notes"])
    assert _run([_live(close, st)], extra=("--qualified", dec, "--max-age-days", "7")) == 2
    assert any("--max-age-days 7" in u for u in _verdict(capsys)["unsure"])
    by_policy = _decision(tmp_path, st, "p.json", policy={"ready": {"max_age_days": 7}}, created=old)
    assert _run([_live(close, st)], extra=("--qualified", by_policy)) == 2
    assert any("ready.max_age_days" in u for u in _verdict(capsys)["unsure"])
    undated = _decision(tmp_path, st, "u.json", created=None)
    assert _run([_live(close, st)], extra=("--qualified", undated, "--max-age-days", "7")) == 2
    assert any("when it was made" in u for u in _verdict(capsys)["unsure"])


def _tokens(close):
    return _srv(close, "vllm:generation_tokens_total 10\n", "vllm:generation_tokens_total 90\n")


def test_a_worker_that_confirmed_a_fault_is_taken_out_of_rotation(close, capsys):
    """A confirmed fault (an output rechecked as wrong, or weights changed) is FAULT FOUND, exit 1, though the checks
    run: a readiness probe then keeps traffic away from that GPU."""
    a = _srv(close, _page(alerts=0, confirmed=0), _page(executed=150, alerts=0, confirmed=0))
    b = _srv(close, _page(uuid="GPU-b", pid=12, alerts=3, confirmed=2),
             _page(uuid="GPU-b", pid=12, executed=160, alerts=3, confirmed=2))
    assert _run([a, b], _tokens(close)) == 1
    out = capsys.readouterr().out
    assert '"verdict": "FAULT FOUND"' in out and "2 confirmed faults" in out and "take it out of service" in out
    assert "1 alert not confirmed" in out                      # the third alert: a note
    # it wins over everything else: an idle server, or one that is also not protected
    assert _run([_srv(close, _page(alerts=1, confirmed=1))], _srv(close, "vllm:generation_tokens_total 10\n")) == 1
    assert '"verdict": "FAULT FOUND"' in capsys.readouterr().out
    assert _run([_srv(close, _page(z=1000, alerts=1, confirmed=1))], _tokens(close)) == 1
    out = capsys.readouterr().out
    assert '"verdict": "FAULT FOUND"' in out and "alarm line" in out


def test_alerts_not_confirmed_are_notes_unless_strict(close, capsys):
    pages = (_page(alerts=2, confirmed=0), _page(executed=150, alerts=2, confirmed=0))
    assert _run([_srv(close, *pages)], _tokens(close)) == 0
    out = capsys.readouterr().out
    assert '"verdict": "PROTECTED"' in out and "2 alerts not confirmed" in out and "--strict-alerts" in out
    assert _run([_srv(close, *pages)], _tokens(close), extra=("--strict-alerts",)) == 1
    assert '"verdict": "FAULT FOUND"' in capsys.readouterr().out
    # a plugin from before the confirmed count: its alerts are a note (any alert a fault with --strict-alerts)
    old = (_page(alerts=1), _page(executed=150, alerts=1))
    assert _run([_srv(close, *old)], _tokens(close)) == 0 and "too old" in capsys.readouterr().out
    assert _run([_srv(close, *old)], _tokens(close), extra=("--strict-alerts",)) == 1


def test_the_metrics_count_only_alerts_their_own_evidence_confirms():
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard.alerts import Metrics, confirmed
    from sdcguard.events import SDCEvent
    ev = lambda verdict: SDCEvent(detector="serve_spot", step=1, identity={"gpu_uuid": "GPU-q"},   # noqa: E731
                                  detail={"where": "l0", **({"recheck": {"verdict": verdict}} if verdict else {})})
    changed = SDCEvent(detector="serve_checksum", step=1, identity={}, detail={"where": "l1",
                                                                         "weights_changed_since_load": True})
    events = [ev("output_wrong"), ev("weights_changed"), ev("output_agrees"), ev("output_off"), ev("not_rechecked"),
              ev(None)]
    assert [confirmed(e) for e in events] == [True, True, False, False, False, False] and confirmed(changed)
    m = Metrics()

    class Spot:
        layers, stats = [1], {"outputs_judged": 4}
    m.observe_spot(Spot(), events)
    values, _info = ready.parse(generate_latest(m.registry).decode())
    assert values["sdcguard_all_alerts_total"] == 6 and values["sdcguard_confirmed_alerts_total"] == 2
    fresh, _ = ready.parse(generate_latest(Metrics().registry).decode())
    assert fresh["sdcguard_confirmed_alerts_total"] == 0                 # exported at 0 before any alert
