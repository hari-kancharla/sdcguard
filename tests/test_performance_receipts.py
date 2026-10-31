"""The receipt workflow on tiny local files and a fake HTTP server; no engine or GPU."""
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import pytest

from sdcguard import assessment, overhead, perf
from sdcguard.performance_identity import (canonical_digest, check_identity_document,
    prepare_identity, read_json, write_identity)


GPU = "GPU-11111111-2222-3333-4444-555555555555"
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "a" * 40


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    """Declare tiny stand-in source pins, distribution metadata and model files."""
    original = assessment.pins()
    fixture_pins = deepcopy(original)
    contents = {}
    for variant, expected in fixture_pins["variants"].items():
        for name, digest in list(expected.items()):
            if digest is not None:
                contents[(variant, name)] = f"{variant} fixture file {name}\n".encode()
                expected[name] = hashlib.sha256(contents[(variant, name)]).hexdigest()
    monkeypatch.setattr(assessment, "pins", lambda: deepcopy(fixture_pins))

    def make(name="input", variant="fixed"):
        root = tmp_path / name
        root.mkdir()
        engine = root / "engine"
        engine.mkdir()
        for filename, digest in fixture_pins["variants"][variant].items():
            if digest is not None:
                path = engine / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[(variant, filename)])
        for package, version in (("vllm", "0.32.0"), ("nixl", "1.4.1")):
            info = engine / f"{package}-{version}.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n")
        snapshot = root / REVISION
        snapshot.mkdir()
        (snapshot / "config.json").write_text('{"model_type":"fixture"}\n')
        (snapshot / "model.safetensors").write_bytes(b"tiny fake weights; never loaded")
        files = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in snapshot.iterdir()}
        _json(snapshot / "sdcguard-model-pin.json", {"model": MODEL, "revision": REVISION, "files": files})
        config = {"schema": 1, "engine": "vllm", "engine_version": "0.32.0", "nixl_version": "1.4.1",
                  "engine_root": "engine", "model": MODEL, "model_revision": REVISION,
                  "topology": dict(assessment.TOPOLOGY), "events": "events"}
        path = root / "deployment.json"
        _json(path, config)
        return {"root": root, "config": path, "snapshot": snapshot, "engine": engine,
                "identity": root / "identity.json", "config_data": config, "pins": fixture_pins}
    return make


class FakeServer:
    def __init__(self, *, model=MODEL, version="0.32.0", usage=True, on_post=None, fail_at=None,
                 version_after_posts=None, stream_lines=None, terminal=True, extra_sse=""):
        self.model, self.version, self.usage = model, version, usage
        self.on_post, self.fail_at, self.version_after_posts = on_post, fail_at, version_after_posts
        self.stream_lines, self.terminal, self.extra_sse = stream_lines, terminal, extra_sse
        self.gets, self.posts = [], []
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.gets.append(self.path)
                version = outer.version_after_posts if outer.posts and outer.version_after_posts else outer.version
                body = {"data": [{"id": outer.model}]} if self.path == "/v1/models" else {"version": version}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer.lock:
                    outer.posts.append(request)
                    post_index = len(outer.posts)
                    callback, outer.on_post = outer.on_post, None
                if callback is not None:
                    callback()
                if post_index == outer.fail_at:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"fixture request failure")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for text in ("one", "two"):
                    time.sleep(0.003)
                    self.wfile.write(("data: " + json.dumps({"choices": [{"text": text}]}) + "\n\n").encode())
                    self.wfile.flush()
                lines = outer.stream_lines(post_index) if callable(outer.stream_lines) else outer.stream_lines
                if lines is None:
                    lines = [json.dumps({"choices": [], "usage": {"completion_tokens": 2}})] if outer.usage else []
                for line in lines:
                    self.wfile.write(("data: " + line + "\n\n").encode())
                self.wfile.write(outer.extra_sse.encode())
                if outer.terminal:
                    self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()


def _prepare(inputs):
    assert overhead.main(["prepare", str(inputs["config"]), "--model-snapshot", str(inputs["snapshot"]),
                          "--gpu-uuid", GPU, "--out", str(inputs["identity"])]) == 0
    return read_json(inputs["identity"])


def _measure(inputs, server, name, mode="on", extra=()):
    prompts = inputs["root"] / "prompts.txt"
    prompts.write_text("fixed fixture prompt one\nfixed fixture prompt two\n")
    output = inputs["root"] / name
    code = perf.main(["--url", server.url, "--identity", str(inputs["identity"]), "--checks", mode,
                      "--out", str(output), "--prompts-file", str(prompts), "--concurrency", "2",
                      "--max-tokens", "2", "--rounds", "3", "--warmup", "0", *extra])
    return code, read_json(output)


def test_prepare_perf_overhead_public_workflow_without_result_json_editing(local_inputs, capsys):
    inputs = local_inputs()
    prepared = _prepare(inputs)
    with FakeServer() as server:
        assert _measure(inputs, server, "baseline.json", "off")[0] == 0
        assert _measure(inputs, server, "checked.json", "on")[0] == 0
        assert len(server.posts) == 12
        assert all(request["model"] == MODEL and request["ignore_eos"] for request in server.posts)
    capsys.readouterr()
    assert overhead.main([str(inputs["root"] / "baseline.json"), str(inputs["root"] / "checked.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "measured" and set(report["metrics"]) == set(overhead.METRICS)
    assert "operator-declared" in report["identity_basis"] and "not remote attestation" in report["identity_basis"]
    assert prepared["engine"]["fix"] == "present"  # On/off is not stock/fixed engine selection.
    assert prepared["basis"]["remote_attestation"] is False


def test_preparation_is_offline_and_preserves_inputs_and_existing_output(local_inputs, monkeypatch):
    inputs = local_inputs()
    before = {path: path.read_bytes() for path in inputs["root"].rglob("*") if path.is_file()}
    monkeypatch.setattr(perf, "_get", lambda *args: pytest.fail("offline preparation made an HTTP request"))
    manifest = _prepare(inputs)
    assert all(path.read_bytes() == value for path, value in before.items())
    assert manifest["identity"]["weights_sha256"] == canonical_digest(manifest["local_inspection"]["model_files_sha256"])
    assert manifest["identity"]["engine_files_sha256"] == canonical_digest(manifest["engine"]["file_sha256"])
    original = inputs["identity"].read_bytes()
    assert overhead.main(["prepare", str(inputs["config"]), "--model-snapshot", str(inputs["snapshot"]),
                          "--gpu-uuid", GPU, "--out", str(inputs["identity"])]) == 2
    assert inputs["identity"].read_bytes() == original


@pytest.mark.parametrize("change", ["config_revision", "topology", "engine", "nixl", "model_pin", "missing_file", "gpu"])
def test_preparation_refuses_unsupported_or_incomplete_local_pins(local_inputs, change):
    inputs = local_inputs()
    gpu = GPU
    if change in ("config_revision", "topology"):
        config = inputs["config_data"]
        if change == "config_revision":
            config["model_revision"] = "main"
        else:
            config["topology"]["tp"] = 2
        _json(inputs["config"], config)
    elif change == "engine":
        (inputs["engine"] / next(iter(inputs["pins"]["variants"]["fixed"]))).write_text("changed source")
    elif change == "nixl":
        (inputs["engine"] / "nixl-1.4.1.dist-info/METADATA").write_text("Name: nixl\nVersion: 1.4.2\n")
    elif change == "model_pin":
        pin_path = inputs["snapshot"] / "sdcguard-model-pin.json"
        pin = read_json(pin_path)
        del pin["files"]["model.safetensors"]
        _json(pin_path, pin)
    elif change == "missing_file":
        (inputs["snapshot"] / "model.safetensors").unlink()
    else:
        gpu = "not-a-gpu-uuid"
    assert overhead.main(["prepare", str(inputs["config"]), "--model-snapshot", str(inputs["snapshot"]),
                          "--gpu-uuid", gpu, "--out", str(inputs["identity"])]) == 2
    assert not inputs["identity"].exists()


@pytest.mark.parametrize("change", ["weights", "config", "engine", "nixl", "identity"])
def test_stale_receipts_are_rejected_before_any_http_request(local_inputs, change):
    inputs = local_inputs()
    _prepare(inputs)
    if change == "weights":
        (inputs["snapshot"] / "model.safetensors").write_bytes(b"changed weights")
    elif change == "config":
        inputs["config"].write_text(inputs["config"].read_text() + " ")
    elif change == "engine":
        (inputs["engine"] / next(iter(inputs["pins"]["variants"]["fixed"]))).write_text("changed source")
    elif change == "nixl":
        (inputs["engine"] / "nixl-1.4.1.dist-info/METADATA").write_text("Name: nixl\nVersion: 1.4.2\n")
    else:
        receipt = read_json(inputs["identity"])
        receipt["identity"]["gpu_uuid"] = "GPU-99999999-2222-3333-4444-555555555555"
        _json(inputs["identity"], receipt)
    with FakeServer() as server:
        code, result = _measure(inputs, server, "failed.json")
        assert code == 2 and server.gets == [] and server.posts == []
    assert result["rounds"] == [] and not result["complete"]
    assert result["measurement_receipt"]["status"] == "inconclusive"


@pytest.mark.parametrize("model,version", [("another-model", "0.32.0"), (MODEL, "0.32.1"), (MODEL, None)])
def test_remote_metadata_contradictions_do_not_send_inference_requests(local_inputs, model, version):
    inputs = local_inputs()
    _prepare(inputs)
    with FakeServer(model=model, version=version) as server:
        code, result = _measure(inputs, server, "failed.json")
        assert code == 2 and server.posts == [] and server.gets
    assert not result["complete"] and result["measurement_receipt"]["missing"]


def test_explicit_model_contradiction_and_overwrite_are_rejected_before_http(local_inputs):
    inputs = local_inputs()
    _prepare(inputs)
    with FakeServer() as server:
        code, result = _measure(inputs, server, "failed.json", extra=("--model", "other"))
        assert code == 2 and not server.gets and not server.posts
        original = (inputs["root"] / "failed.json").read_bytes()
        assert perf.main(["--url", server.url, "--identity", str(inputs["identity"]), "--checks", "on",
                          "--out", str(inputs["root"] / "failed.json")]) == 2
        assert (inputs["root"] / "failed.json").read_bytes() == original
        assert not server.gets and not server.posts
    assert not result["complete"]


@pytest.mark.parametrize("change", ["weights", "config", "engine", "identity"])
def test_mutation_during_measurement_retains_rounds_and_fails_post_validation(local_inputs, change):
    inputs = local_inputs()
    _prepare(inputs)
    def mutate():
        if change == "weights":
            (inputs["snapshot"] / "model.safetensors").write_bytes(b"changed")
        elif change == "config":
            inputs["config"].write_text(inputs["config"].read_text() + " ")
        elif change == "engine":
            (inputs["engine"] / next(iter(inputs["pins"]["variants"]["fixed"]))).write_text("changed source")
        else:
            inputs["identity"].write_text("{}")
    with FakeServer(on_post=mutate) as server:
        code, result = _measure(inputs, server, "incomplete.json")
    assert code == 2 and len(result["rounds"]) == 3
    assert result["measurement_receipt"]["local_validation"] == {"before": True, "after": False}
    assert not result["complete"] and any("post-measurement" in s for s in result["measurement_receipt"]["missing"])
    assert overhead.summarize(result, result)["status"] == "inconclusive"


def test_external_pin_keeps_snapshot_read_only_and_is_revalidated(local_inputs):
    inputs = local_inputs()
    external = inputs["root"] / "MODEL_PIN.json"
    original = inputs["snapshot"] / "sdcguard-model-pin.json"
    external.write_bytes(original.read_bytes())
    original.unlink()  # Tiny test fixture only; the production utility never modifies a snapshot.
    files = {path: path.read_bytes() for path in inputs["snapshot"].iterdir()}
    assert overhead.main(["prepare", str(inputs["config"]), "--model-snapshot", str(inputs["snapshot"]),
                          "--model-pin", str(external), "--gpu-uuid", GPU, "--out", str(inputs["identity"])]) == 0
    prepared = read_json(inputs["identity"])
    assert prepared["local_inputs"]["model_pin"] == str(external.resolve())
    assert all(path.read_bytes() == value for path, value in files.items()) and not original.exists()
    with FakeServer(on_post=lambda: external.write_text("{}")) as server:
        code, result = _measure(inputs, server, "incomplete.json")
    assert code == 2 and len(result["rounds"]) == 3
    assert result["measurement_receipt"]["local_validation"]["after"] is False
    assert not original.exists()


def test_interruption_retains_completed_rounds_without_more_queries_or_hashing(local_inputs, monkeypatch):
    inputs = local_inputs()
    _prepare(inputs)
    original = perf.run_round
    calls = []

    def interrupted(*args):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return original(*args)

    monkeypatch.setattr(perf, "run_round", interrupted)
    with FakeServer() as server:
        code, result = _measure(inputs, server, "interrupted.json")
        assert server.gets == ["/v1/models", "/version"]
    assert code == 130 and len(result["rounds"]) == 1 and not result["complete"]
    assert result["measurement_receipt"]["interrupted"] is True
    assert result["measurement_receipt"]["local_validation"]["after"] is False


def test_interruption_during_final_hashing_keeps_all_measured_rounds(local_inputs, monkeypatch):
    from sdcguard import performance_identity
    inputs = local_inputs()
    _prepare(inputs)
    original = performance_identity.validate_identity
    calls = []

    def interrupted(value):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return original(value)

    monkeypatch.setattr(performance_identity, "validate_identity", interrupted)
    with FakeServer() as server:
        code, result = _measure(inputs, server, "interrupted.json")
    assert code == 130 and len(result["rounds"]) == 3 and not result["complete"]
    assert result["measurement_receipt"]["interrupted"] is True


@pytest.mark.parametrize("kind", ["http_error", "missing_usage", "metadata_drift"])
def test_failed_or_incomplete_measurements_are_retained_and_never_measured(local_inputs, kind):
    inputs = local_inputs()
    _prepare(inputs)
    kwargs = {"fail_at": 2} if kind == "http_error" else {"usage": False} if kind == "missing_usage" else {
        "version_after_posts": "0.32.1"}
    with FakeServer(**kwargs) as server:
        code, result = _measure(inputs, server, "incomplete.json")
    assert code == 2 and len(result["rounds"]) == 3 and not result["complete"]
    assert overhead.summarize(result, result)["status"] == "inconclusive"
    if kind == "http_error":
        assert sum(row["errors"] for row in result["rounds"]) == 1
    if kind == "missing_usage":
        assert all(row["tokens_counted_by_server"] is False for row in result["rounds"])


def test_measurement_tampering_and_stock_fixed_comparison_are_inconclusive(local_inputs):
    fixed = local_inputs("fixed", "fixed")
    stock = local_inputs("stock", "stock")
    _prepare(fixed)
    _prepare(stock)
    with FakeServer() as server:
        _, baseline = _measure(fixed, server, "baseline.json", "off")
        _, checked = _measure(fixed, server, "checked.json", "on")
        _, changed_engine = _measure(stock, server, "stock.json", "on")
    assert overhead.summarize(baseline, checked)["status"] == "measured"
    report = overhead.summarize(baseline, changed_engine)
    assert report["status"] == "inconclusive" and any("engine" in m for m in report["missing"])
    altered = deepcopy(checked)
    altered["rounds"][0]["ttft_p99_ms"] += 1
    report = overhead.summarize(baseline, altered)
    assert report["status"] == "inconclusive" and any("receipt" in m for m in report["missing"])


def test_prepared_document_is_canonical_but_not_an_attestation(local_inputs):
    inputs = local_inputs()
    original = prepare_identity(inputs["config"], inputs["snapshot"], GPU)
    check_identity_document(original)
    config = read_json(inputs["config"])
    inputs["config"].write_text(json.dumps(dict(reversed(list(config.items()))), separators=(",", ":")))
    formatted = prepare_identity(inputs["config"], inputs["snapshot"], GPU)
    assert original["identity"] == formatted["identity"]
    assert original["local_inspection"] != formatted["local_inspection"]
    assert "not proof of live loaded weights" in original["basis"]["model"]
    assert original["basis"]["remote_attestation"] is False


def test_new_flags_are_paired_and_legacy_measurement_still_works(local_inputs):
    inputs = local_inputs()
    with pytest.raises(SystemExit):
        perf.main(["--checks", "on"])
    with pytest.raises(SystemExit):
        perf.main(["--identity", str(inputs["identity"])])
    with FakeServer(model="legacy") as server:
        result = perf.measure(server.url, texts=["legacy prompt"], concurrency=1, rounds=1, max_tokens=2, warmup=0)
    assert result["complete"] and result["model"] == "legacy" and "measurement_receipt" not in result


@pytest.mark.parametrize("kind,kwargs", [
    ("EOF without terminal", {"terminal": False}),
    ("SSE error event", {"extra_sse": "event: error\ndata: {}\n\n"}),
    ("SSE error payload", {"stream_lines": ['{"error":{"message":"fixture error"}}']}),
    ("SSE error finish", {"stream_lines": ['{"choices":[{"finish_reason":"error"}]}']}),
    ("duplicate token count", {"stream_lines": ['{"usage":{"completion_tokens":true,"completion_tokens":2}}']}),
    ("missing usage", {"usage": False}),
    ("null usage", {"stream_lines": ['{"usage":null}']}),
    ("missing token count", {"stream_lines": ['{"usage":{}}']}),
    ("malformed usage", {"stream_lines": ['{"usage":[]}']}),
    ("negative count", {"stream_lines": ['{"usage":{"completion_tokens":-1}}']}),
    ("boolean count", {"stream_lines": ['{"usage":{"completion_tokens":true}}']}),
    ("fractional count", {"stream_lines": ['{"usage":{"completion_tokens":1.5}}']}),
    ("integer-looking float", {"stream_lines": ['{"usage":{"completion_tokens":2.0}}']}),
    ("string count", {"stream_lines": ['{"usage":{"completion_tokens":"2"}}']}),
    ("NaN count", {"stream_lines": ['{"usage":{"completion_tokens":NaN}}']}),
    ("infinite count", {"stream_lines": ['{"usage":{"completion_tokens":Infinity}}']}),
    ("unrepresentable integer", {"stream_lines": ['{"usage":{"completion_tokens":' + "9" * 350 + '}}']}),
    ("malformed JSON", {"stream_lines": ['{"usage":']}),
])
def test_invalid_stream_evidence_preserves_requests_and_never_measures(local_inputs, kind, kwargs):
    inputs = local_inputs()
    _prepare(inputs)
    with FakeServer() as server:
        code, baseline = _measure(inputs, server, "baseline.json", "off")
    assert code == 0
    with FakeServer(**kwargs) as server:
        code, result = _measure(inputs, server, "incomplete.json")
        assert len(server.posts) == 6  # No new load, replay or retry was added.
    assert code == 2 and not result["complete"], kind
    assert len(result["rounds"]) == 3
    assert not any("preflight" in reason for reason in result["measurement_receipt"]["missing"])
    for row in result["rounds"]:
        assert row["completed"] == 0 and row["errors"] == 2
        assert row["output_tokens"] is None and row["output_tokens_per_s"] is None
        assert row["tokens_counted_by_server"] is False
        assert len(row["request_results"]) == 2
        for request in row["request_results"]:
            assert request["complete"] is False and request["error"]
            assert request["chunks"] == 2 and request["latency"] > 0
            if kind == "EOF without terminal":
                assert request["terminal_observed"] is False and request["tokens"] == 2
            if kind == "NaN count":
                assert "NaN" in request["last_event"] and request["tokens"] is None
    assert overhead.summarize(baseline, result)["status"] == "inconclusive"
    # Strict JSON serialization succeeds even when the server sent NaN/Infinity.
    json.dumps(result, allow_nan=False)


def test_one_failed_request_keeps_all_other_request_and_round_observations(local_inputs):
    inputs = local_inputs()
    _prepare(inputs)
    with FakeServer(fail_at=3) as server:
        code, result = _measure(inputs, server, "partial.json", extra=("--concurrency", "1"))
    assert code == 2 and len(result["rounds"]) == 3
    assert [row["completed"] for row in result["rounds"]] == [2, 1, 2]
    failed = result["rounds"][1]["request_results"][0]
    assert failed["http_status"] == 503 and failed["complete"] is False
    assert failed["terminal_observed"] is False and failed["tokens"] is None
    assert sum(len(row["request_results"]) for row in result["rounds"]) == 6
    assert overhead.summarize(result, result)["status"] == "inconclusive"


def test_request_parse_interruption_preserves_partial_round_and_prior_round(local_inputs, monkeypatch):
    inputs = local_inputs()
    _prepare(inputs)
    original = json.loads
    marker = '{"interrupt_fixture":true}'
    def interrupted(value, *args, **kwargs):
        if value == marker:
            raise KeyboardInterrupt()
        return original(value, *args, **kwargs)
    monkeypatch.setattr(perf.json, "loads", interrupted)
    with FakeServer(stream_lines=lambda index: [marker] if index == 3 else None) as server:
        code, result = _measure(inputs, server, "partial.json", extra=("--concurrency", "1"))
        assert len(server.posts) == 3 and server.gets == ["/v1/models", "/version"]
    assert code == 130 and not result["complete"]
    assert len(result["rounds"]) == 2 and result["rounds"][0]["completed"] == 2
    partial = result["rounds"][1]
    assert partial["interrupted"] is True and partial["started"] == 1 and partial["unstarted"] == 1
    request = partial["request_results"][0]
    assert request["complete"] is False and request["chunks"] == 2
    assert request["last_event"] == marker and request["terminal_observed"] is False
    assert result["measurement_receipt"]["interrupted"] is True
    assert overhead.summarize(result, result)["status"] == "inconclusive"


@pytest.mark.parametrize("failure", [ValueError("fixture digest failure"), KeyboardInterrupt()])
def test_late_digest_failures_retain_all_rounds_instead_of_becoming_preflight(local_inputs, monkeypatch, failure):
    from sdcguard import performance_identity
    inputs = local_inputs()
    _prepare(inputs)
    def failed_digest(value):
        raise failure
    monkeypatch.setattr(performance_identity, "measurement_digest", failed_digest)
    with FakeServer() as server:
        code, result = _measure(inputs, server, "digest-failed.json")
        assert len(server.posts) == 6
    assert code == (130 if isinstance(failure, KeyboardInterrupt) else 2)
    assert not result["complete"] and len(result["rounds"]) == 3
    assert all(row["completed"] == 2 and len(row["request_results"]) == 2 for row in result["rounds"])
    receipt = result["measurement_receipt"]
    assert receipt["measurement_sha256"] is None and any("binding failed" in item for item in receipt["missing"])
    assert not any("preflight" in item for item in receipt["missing"])
    assert overhead.summarize(result, result)["status"] == "inconclusive"


def test_nonfinite_late_metric_is_explicit_evidence_and_never_zero_or_measured(local_inputs, monkeypatch):
    inputs = local_inputs()
    _prepare(inputs)
    original = perf.run_round
    def invalid_metric(*args):
        result = original(*args)
        result["output_tokens_per_s"] = float("nan")
        return result
    monkeypatch.setattr(perf, "run_round", invalid_metric)
    with FakeServer() as server:
        code, result = _measure(inputs, server, "invalid-metric.json")
    assert code == 2 and len(result["rounds"]) == 3
    assert all(row["output_tokens_per_s"] == {"invalid_numeric_evidence": "nan"} for row in result["rounds"])
    assert not result["complete"] and any("nonfinite evidence" in item for item in result["measurement_receipt"]["missing"])
    assert overhead.summarize(result, result)["status"] == "inconclusive"


def test_bound_receipt_without_explicit_request_terminal_evidence_is_inconclusive(local_inputs):
    from sdcguard.performance_identity import measurement_digest
    inputs = local_inputs()
    _prepare(inputs)
    with FakeServer() as server:
        _, baseline = _measure(inputs, server, "baseline.json", "off")
        _, checked = _measure(inputs, server, "checked.json", "on")
    assert overhead.summarize(baseline, checked)["status"] == "measured"
    for row in checked["rounds"]:
        del row["request_results"]
    checked["measurement_receipt"]["measurement_sha256"] = measurement_digest(checked)
    report = overhead.summarize(baseline, checked)
    assert report["status"] == "inconclusive"
    assert any("terminal evidence" in item for item in report["missing"])


def test_failed_warmup_retains_partial_request_without_starting_rounds(local_inputs):
    inputs = local_inputs()
    _prepare(inputs)
    with FakeServer(terminal=False) as server:
        code, result = _measure(inputs, server, "warmup-failed.json", extra=("--warmup", "1"))
        assert len(server.posts) == 1
    assert code == 2 and result["rounds"] == [] and not result["complete"]
    assert len(result["warmup_results"]) == 1
    request = result["warmup_results"][0]
    assert request["complete"] is False and request["chunks"] == 2 and request["tokens"] == 2
    assert request["terminal_observed"] is False and request["error"]
    assert overhead.summarize(result, result)["status"] == "inconclusive"


def test_numeric_aggregation_failure_keeps_valid_request_observations(local_inputs):
    inputs = local_inputs()
    _prepare(inputs)
    huge = json.dumps({"usage": {"completion_tokens": 10 ** 308}})
    with FakeServer(stream_lines=[huge]) as server:
        code, result = _measure(inputs, server, "overflow.json")
    assert code == 2 and len(result["rounds"]) == 3 and not result["complete"]
    for row in result["rounds"]:
        assert row["completed"] == 2 and row["output_tokens"] == 2 * 10 ** 308
        assert row["output_tokens_per_s"] is None and row["aggregation_error"]
        assert len(row["request_results"]) == 2
    assert overhead.summarize(result, result)["status"] == "inconclusive"


@pytest.mark.parametrize("kwargs,posts", [
    ({"model": float("nan")}, 0),
    ({"version": float("nan")}, 0),
    ({"version_after_posts": float("nan")}, 6),
])
def test_nonfinite_metadata_stays_unknown_without_losing_rounds(local_inputs, kwargs, posts):
    inputs = local_inputs()
    _prepare(inputs)
    with FakeServer(**kwargs) as server:
        code, result = _measure(inputs, server, "invalid-metadata.json")
        assert len(server.posts) == posts
    assert code == 2 and not result["complete"]
    assert len(result["rounds"]) == (3 if posts else 0)
    assert result["measurement_receipt"]["status"] == "inconclusive"
    json.dumps(result, allow_nan=False)
    assert overhead.summarize(result, result)["status"] == "inconclusive"
