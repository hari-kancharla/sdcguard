"""NVIDIA NVSentinel health events (nvsentinel.py): the bytes NVSentinel's platform connector parses, what each kind
of finding is sent as (a job's alert never asks for action; a check FAIL asks only with --remediate), which GPU
it names, and the gRPC call itself (token, retries, refusals) against a local server."""
from __future__ import annotations

import json
import os
import threading
import time

import pytest

from sdcguard import nvsentinel as nv

# Built from NVIDIA's data-models/protobufs/health_event.proto (NVSentinel v1.28.0, commit 0d7125f, Oct 19, 2026)
# with grpcio-tools and serialized by protobuf 7.36 with deterministic=True: the encoder must write these bytes.
EVENT = {"version": 1, "agent": "sdcguard", "componentClass": "GPU", "checkName": "GpuComputeCorrectness",
         "isFatal": True, "isHealthy": False, "message": "sdcguard check FAIL on NVIDIA H100 GPU-1234",
         "recommendedAction": "RUN_FIELDDIAG", "errorCode": ["SDCGUARD_CHECK_FAIL", "X2"],
         "entitiesImpacted": [{"entityType": "PCI", "entityValue": "0000:1b:00.0"},
                              {"entityType": "GPU_UUID", "entityValue": "GPU-1234"}],
         "metadata": {"zeta": "1", "alpha": "two", "source": "sdcguard check"},
         "generatedTimestamp": 1761055200.25, "nodeName": "node-a", "processingStrategy": "EXECUTE_REMEDIATION",
         "quarantineOverrides": {"skip": True}}
GOLDEN = ("080112ec010801120873646367756172641a034750552215477075436f6d70757465436f72726563746e65737328013a2b73646367"
          "7561726420636865636b204641494c206f6e204e56494449412048313030204750552d3132333440064a1353444347554152445f43"
          "4845434b5f4641494c4a02583252130a03504349120c303030303a31623a30302e3052140a084750555f5555494412084750552d31"
          "3233345a0c0a05616c706861120374776f5a180a06736f75726365120e736463677561726420636865636b5a090a047a6574611201"
          "3162 0b08e0a3dec7061080e59a776a066e6f64652d6172021001800101").replace(" ", "")
MINIMAL = {"version": 1, "agent": "sdcguard", "checkName": "GpuComputeCorrectness", "isHealthy": True,
           "nodeName": "n", "processingStrategy": "STORE_AND_ANALYSE"}
MINIMAL_GOLDEN = "0801122b0801120873646367756172642215477075436f6d70757465436f72726563746e65737330016a016e800103"

GPU = {"gpu_uuid": "GPU-1234", "gpu_name": "NVIDIA H100 80GB HBM3", "pci_bus_id": "00000000:1B:00.0",
       "host": "node-a", "serial": "1650123456789"}


def test_the_bytes_are_what_nvidias_own_protobuf_writes():
    assert nv.encode_events([EVENT]).hex() == GOLDEN
    assert nv.encode_events([MINIMAL]).hex() == MINIMAL_GOLDEN


def test_pci_addresses_are_written_the_way_nvsentinel_writes_them():
    assert nv.pci_address("00000000:1B:00.0") == "0000:1b:00.0"
    assert nv.pci_address("0001:AF:00.0") == "0001:af:00.0"
    assert nv.pci_address(None) is None


def _check(result="FAIL", **kw):
    return {"result": result, "gpu": dict(GPU), "run_id": "r1", "notes": [["bad", "2 math-mode results wrong"]],
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}


def test_a_check_fail_is_stored_for_analysis_unless_the_operator_asks_for_action():
    ev, why = nv.from_check(_check(), node="node-a")
    assert why is None and ev["processingStrategy"] == "STORE_AND_ANALYSE"
    assert not ev["isFatal"] and not ev["isHealthy"] and ev["recommendedAction"] == "NONE"
    assert ev["errorCode"] == ["SDCGUARD_CHECK_FAIL"] and ev["agent"] == "sdcguard"
    assert ev["entitiesImpacted"] == [{"entityType": "PCI", "entityValue": "0000:1b:00.0"},
                                      {"entityType": "GPU_UUID", "entityValue": "GPU-1234"}]
    assert "math-mode" in ev["message"] and ev["nodeName"] == "node-a"
    ev, _ = nv.from_check(_check(), node="node-a", remediate=True)
    assert ev["processingStrategy"] == "EXECUTE_REMEDIATION" and ev["isFatal"]
    assert ev["recommendedAction"] == "RUN_FIELDDIAG"


def test_a_pass_clears_with_the_same_check_and_entities():
    bad, _ = nv.from_check(_check("FAIL"), node="n", remediate=True)
    good, _ = nv.from_check(_check("PASS"), node="n", remediate=True)
    assert good["isHealthy"] and not good["isFatal"] and good["errorCode"] == []
    assert good["checkName"] == bad["checkName"] and good["entitiesImpacted"] == bad["entitiesImpacted"]
    assert good["processingStrategy"] == "EXECUTE_REMEDIATION"            # (only then does it clear a condition)


def test_a_suspect_result_never_asks_for_action():
    ev, _ = nv.from_check(_check("SUSPECT"), node="n", remediate=True)
    assert ev["processingStrategy"] == "STORE_AND_ANALYSE" and not ev["isFatal"]
    assert ev["errorCode"] == ["SDCGUARD_CHECK_SUSPECT"]


@pytest.mark.parametrize("result,extra", [("INCOMPLETE", {}), ("FAIL", {"simulated_faults": True}), ("", {})])
def test_incomplete_results_and_demos_are_never_sent(result, extra):
    ev, why = nv.from_check(_check(result, **extra), node="n", remediate=True)
    assert ev is None and why


def test_a_result_without_a_gpu_uuid_is_never_sent():
    r = _check()
    r["gpu"] = {"gpu_name": "x"}
    assert nv.from_check(r, node="n")[0] is None


def test_an_untrusted_certificate_never_asks_for_action():
    ev, _ = nv.from_check(_check(), node="n", remediate=True, trusted=False)
    assert ev["processingStrategy"] == "STORE_AND_ANALYSE" and not ev["isFatal"]
    assert ev["metadata"]["certificate"] == "not verified with the operator's key"


def _alert(det="checksum", uuid="GPU-1234", host="node-a", z=40.0, t=1761055200.0):
    return {"detector": det, "step": 3, "time": t, "detail": {"where": "blocks.0.up", "max_z": z},
            "identity": {"gpu_uuid": uuid, "host": host, "gpu_name": "NVIDIA H100", "pci_bus_id": "00000000:1B:00.0"}}


def test_job_alerts_are_leads_one_event_per_gpu_and_detector():
    evs, skipped = nv.from_alerts([_alert(z=9), _alert(z=40.5, t=1761055300.0), _alert("serve_checksum"),
                                   _alert(uuid=None), _alert(host="node-b")], node="node-a", host="node-a")
    assert len(evs) == 2 and len(skipped) == 2
    train = next(e for e in evs if e["errorCode"] == ["SDCGUARD_TRAINING_CHECKSUM"])
    assert train["processingStrategy"] == "STORE_AND_ANALYSE" and not train["isFatal"]
    assert train["recommendedAction"] == "NONE" and train["metadata"]["alerts"] == "2"
    assert train["metadata"]["max_z"] == "40.5" and train["metadata"]["verified"] == "false"
    assert train["generatedTimestamp"] == 1761055300.0 and "not a verdict" in train["message"]
    assert any("node-b" in s for s in skipped)


def test_metadata_values_are_short_strings():
    ev, _ = nv.from_check(_check(notes=[["bad", "x" * 5000]]), node="n")
    assert all(isinstance(v, str) and len(v) <= 256 for v in ev["metadata"].values())
    assert len(ev["message"]) <= 1024
    nv.encode_events([ev])


def test_the_node_name_comes_from_the_downward_api_inside_a_pod(monkeypatch):
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.delenv("NODE_NAME", raising=False)
    assert nv.node_name() is None                              # a pod's hostname isn't the node's
    monkeypatch.setenv("NODE_NAME", "gpu-node-7")
    assert nv.node_name() == "gpu-node-7" and nv.node_name("given") == "given"


# ------------------------------------------------------------------------------------- the call, on a local server

grpc = pytest.importorskip("grpc")


class _Server:
    """A stand-in platform connector on a unix socket: keeps each request's bytes and metadata; `answers` is
    the status to return for each call in turn (OK when exhausted)."""

    def __init__(self, path, answers=()):
        from concurrent import futures
        self.path, self.answers, self.calls = path, list(answers), []

        def handle(request, context):
            self.calls.append((request, dict(context.invocation_metadata())))
            if self.answers:
                code = self.answers.pop(0)
                context.abort(code, f"{code.name} for the test")
            return b""

        handler = grpc.method_handlers_generic_handler("datamodels.PlatformConnector", {
            "HealthEventOccurredV1": grpc.unary_unary_rpc_method_handler(handle, request_deserializer=None,
                                                                         response_serializer=None)})
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        self.server.add_generic_rpc_handlers((handler,))
        self.server.add_insecure_port(f"unix://{path}")

    def __enter__(self):
        self.server.start()
        return self

    def __exit__(self, *a):
        self.server.stop(0)


@pytest.fixture
def sock(tmp_path):
    return str(tmp_path / "nvsentinel.sock")


def test_events_reach_the_platform_connector_with_the_token(sock, tmp_path):
    tok = tmp_path / "token"
    tok.write_text("first\n")
    with _Server(sock) as s:
        nv.send([EVENT], socket_path=sock, token_file=str(tok))
        tok.write_text("rotated")                              # the kubelet rewrote it: the next call reads it
        nv.send([MINIMAL], socket_path=sock, token_file=str(tok))
    assert s.calls[0][0].hex() == GOLDEN and s.calls[0][1]["authorization"] == "Bearer first"
    assert s.calls[1][1]["authorization"] == "Bearer rotated"


def test_unavailable_is_retried_and_a_refusal_is_reported(sock):
    with _Server(sock, [grpc.StatusCode.UNAVAILABLE]) as s:
        nv.send([MINIMAL], socket_path=sock, backoff=0.01)
    assert len(s.calls) == 2
    with _Server(sock, [grpc.StatusCode.INVALID_ARGUMENT]) as s:
        with pytest.raises(nv.SendError, match="refused"):
            nv.send([MINIMAL], socket_path=sock, backoff=0.01)
    assert len(s.calls) == 1                                   # NVSentinel's answer: not tried again


def test_no_socket_and_an_empty_token_are_clear_errors(sock, tmp_path):
    with pytest.raises(nv.SendError, match="no NVSentinel socket"):
        nv.send([MINIMAL], socket_path=sock)
    (tmp_path / "empty").write_text("")
    with _Server(sock):
        with pytest.raises(nv.SendError, match="empty"):
            nv.send([MINIMAL], socket_path=sock, token_file=str(tmp_path / "empty"))


def test_the_command_sends_a_nodes_check_results(sock, tmp_path, capsys):
    d = tmp_path / "sdcguard_check" / "GPU-1234"
    d.mkdir(parents=True)
    (d / "check.json").write_text(json.dumps(_check("PASS")))
    other = tmp_path / "sdcguard_check" / "GPU-9"
    other.mkdir()
    (other / "check.json").write_text(json.dumps({**_check("INCOMPLETE"), "gpu": {"gpu_uuid": "GPU-9"}}))
    with _Server(sock) as s:
        code = nv.main([str(tmp_path / "sdcguard_check"), "--socket", sock, "--node-name", "node-a"])
    out = capsys.readouterr().out
    assert code == 0 and len(s.calls) == 1 and "sent 1 event" in out and "INCOMPLETE" in out
    code = nv.main([str(tmp_path / "sdcguard_check"), "--dry-run", "--node-name", "node-a"])
    printed = capsys.readouterr().out
    assert code == 0 and '"isHealthy": true' in printed


def test_old_results_are_not_sent(tmp_path, capsys):
    d = tmp_path / "c"
    d.mkdir()
    (d / "check.json").write_text(json.dumps(_check(time="2026-09-01T00:00:00Z")))
    assert nv.main([str(d), "--dry-run", "--node-name", "n"]) == 0
    out = capsys.readouterr().out
    assert "hours old" in out and out.rstrip().endswith("[]")


def test_the_command_says_when_nvsentinel_isnt_there(tmp_path, capsys):
    d = tmp_path / "c"
    d.mkdir()
    (d / "check.json").write_text(json.dumps(_check()))
    # 2, never 1: exit 1 means a GPU computed wrong results, which a job running nhc first reports itself
    assert nv.main([str(d), "--socket", str(tmp_path / "none.sock"), "--node-name", "n"]) == 2
    assert "no NVSentinel socket" in capsys.readouterr().out
    assert nv.main([str(tmp_path / "missing"), "--node-name", "n"]) == 2


def test_a_job_publisher_never_raises_and_sends_once_per_gpu_and_detector(sock, monkeypatch):
    from sdcguard.events import SDCEvent
    monkeypatch.setattr(nv.socket, "gethostname", lambda: "node-a")
    with _Server(sock) as s:
        pub = nv.Publisher(socket_path=sock, node="node-a", min_interval=60)
        a = _alert()
        for _ in range(3):
            pub(SDCEvent(detector="checksum", step=1, detail=a["detail"], identity=a["identity"]))
        pub.flush(10)
    assert pub.sent == 1 and len(s.calls) == 1 and pub.failed == 0
    down = nv.Publisher(socket_path=sock + ".gone", node="node-a")
    down(SDCEvent(detector="checksum", step=1, detail=a["detail"], identity=a["identity"]))
    down.flush(10)
    assert down.failed == 1                                    # counted and logged, never raised


def test_a_guard_hands_its_alerts_to_the_publisher(sock, monkeypatch):
    import torch
    from sdcguard import SDCGuard
    from torch.utils._python_dispatch import TorchDispatchMode
    monkeypatch.setattr(nv.socket, "gethostname", lambda: "node-a")

    class Corrupt(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if func is torch.ops.aten.mm.default:
                out[0, 0] += 100.0
            return out

    with _Server(sock) as s:
        pub = nv.Publisher(socket_path=sock, node="node-a")
        g = SDCGuard(on_alert=pub)
        A, B = torch.randn(8, 256).bfloat16(), torch.randn(256, 128).bfloat16()
        with Corrupt():
            with g.at_step(0):
                A @ B
        pub.flush(10)
    # a CPU run has no GPU UUID: nothing to name, so nothing is sent (and nothing breaks)
    assert g.stats["alerts"] == 1 and pub.skipped == 1 and not s.calls


def test_sdcguard_nvsentinel_is_a_command(tmp_path, capsys):
    """The Kubernetes job runs `sdcguard nvsentinel`, but the command line didn't dispatch it (it printed the
    help and exited 0, sending nothing)."""
    from sdcguard import cli
    d = tmp_path / "c"
    d.mkdir()
    (d / "check.json").write_text(json.dumps(_check()))
    assert cli.main(["nvsentinel", str(d), "--dry-run", "--node-name", "n"]) == 0
    assert '"checkName"' in capsys.readouterr().out


def _job_script():
    """The shell the NVSentinel Kubernetes job runs (its container's `args: - |` block), read without a YAML parser."""
    from pathlib import Path
    from textwrap import dedent
    lines = (Path(__file__).resolve().parents[1] / "deploy/kubernetes/nvsentinel-screen-job.yaml").read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "- |" and lines[i - 1].strip() == "args:") + 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = []
    for line in lines[start:]:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        block.append(line)
    return dedent("\n".join(block))


@pytest.mark.parametrize("nhc,sent", [(a, b) for a in (0, 1, 2, 137) for b in (0, 2, 137)])
def test_the_kubernetes_job_keeps_nhcs_fail_whatever_the_delivery_does(tmp_path, nhc, sent):
    """deploy/kubernetes/nvsentinel-screen-job.yaml: nhc's FAIL (1) is the job's exit code even when the delivery
    fails; otherwise a failed delivery is the job's exit code (never 1, which means wrong results); both results
    folders are the same. sdcguard is a stand-in that exits as told and records how it was called."""
    import subprocess
    b = tmp_path / "bin"
    b.mkdir()
    (b / "sdcguard").write_text('#!/bin/sh\necho "$@" >> "$FAKE_DIR/calls"\n'
                                '[ "$1" = nhc ] && exit "$NHC_CODE"\nexit "$SEND_CODE"\n')
    (b / "sdcguard").chmod(0o755)
    env = {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "FAKE_DIR": str(tmp_path),
           "NHC_CODE": str(nhc), "SEND_CODE": str(sent)}
    env.pop("SDCGUARD_MINUTES", None)
    r = subprocess.run(["/bin/sh", "-c", _job_script()], env=env, timeout=60)
    assert r.returncode == (1 if nhc == 1 else nhc if sent == 0 else sent)
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls == ["nhc --minutes 10 --out /var/lib/sdcguard/screen",
                     "nvsentinel /var/lib/sdcguard/screen --socket /var/run/nvsentinel/nvsentinel.sock"], calls
