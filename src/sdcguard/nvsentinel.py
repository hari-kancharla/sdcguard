"""NVIDIA NVSentinel: sdcguard's findings in the GPU fault pipeline a Kubernetes cluster already runs.

NVSentinel (github.com/NVIDIA/NVSentinel) takes health events from monitors over a gRPC socket on each node and
decides what to do with them: store and export them, set a node condition, cordon, drain, remediate. sdcguard
becomes one more monitor, agent "sdcguard", check "GpuComputeCorrectness":

  - an alert from a job (a training or serving checksum mismatch, a recompute or replica mismatch) is a lead, not
    a verdict: it goes as STORE_AND_ANALYSE, not fatal, no recommended action. NVSentinel stores and exports it
    beside the GPU's XID and ECC history and its analyzer can correlate it, but it never sets a node condition
    or cordons anything (NVSentinel's own default analyzer rules match XID codes, not these).
  - a `sdcguard check` run on the node: FAIL or SUSPECT is sent as an unhealthy event, PASS as a healthy one with
    the same check and entities (which clears it). Also STORE_AND_ANALYSE by default. Only with remediate=True
    (--remediate) does a FAIL go as EXECUTE_REMEDIATION, fatal, recommending RUN_FIELDDIAG (NVIDIA's field
    diagnostic, which an RMA needs), and a PASS as EXECUTE_REMEDIATION healthy, which clears the condition. Even
    then NVSentinel cordons only if the operator wrote a quarantine rule for agent "sdcguard": its defaults
    match its own monitors only. With a trusted public key, a FAIL asks for action only when its certificate
    verifies against that key.
  - a demo run (--simulate-faults) and an INCOMPLETE check are never sent: neither says anything about the GPU.

Each event names the GPU by UUID and PCI address (NVSentinel's own format, 0000:1b:00.0), never by an index
(CUDA's and NVIDIA's numbering can differ). Events go to unix:///var/run/nvsentinel.sock, method
/datamodels.PlatformConnector/HealthEventOccurredV1, with the projected service-account token when one is
mounted (read again for every call, since the kubelet rotates it); UNAVAILABLE and DEADLINE_EXCEEDED are
retried, anything else is NVSentinel's answer and is reported. Sending needs grpcio
(pip install "sdcguard[nvsentinel]"). The protobuf messages are encoded here from NVSentinel's
data-models/protobufs/health_event.proto (v1.28.0, Oct 19, 2026), so no generated code or protobuf package is
needed.

    sdcguard nvsentinel sdcguard_check                    # this node's check results
    sdcguard nvsentinel sdcguard_check --remediate        # a FAIL asks NVSentinel for action
    sdcguard nvsentinel alerts.jsonl                      # alerts from a job's log (SDCGuard(log_path=...))
    sdcguard nvsentinel sdcguard_check --dry-run          # print the events, send nothing

In a job: SDCGuard(on_alert=sdcguard.nvsentinel.Publisher()) sends each alert from a background thread (one per
GPU and detector every few minutes). A tenant pod usually can't reach the socket (it is a hostPath on the node);
then write alerts to a log the node's operator-run sdcguard reads.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Iterable, Optional

SOCKET = "/var/run/nvsentinel.sock"
TOKEN = "/var/run/secrets/nvsentinel/platform-connector/token"
METHOD = "/datamodels.PlatformConnector/HealthEventOccurredV1"
AGENT = "sdcguard"
CHECK = "GpuComputeCorrectness"

# health_event.proto enums
STRATEGIES = {"UNSPECIFIED": 0, "EXECUTE_REMEDIATION": 1, "STORE_ONLY": 2, "STORE_AND_ANALYSE": 3}
ACTIONS = {"NONE": 0, "COMPONENT_RESET": 2, "CONTACT_SUPPORT": 5, "RUN_FIELDDIAG": 6, "RESTART_VM": 15,
           "RESTART_BM": 24, "REPLACE_VM": 25, "RUN_DCGMEUD": 26, "CUSTOM": 27, "UNKNOWN": 99}
ERROR_CODES = {"checksum": "SDCGUARD_TRAINING_CHECKSUM", "serve_checksum": "SDCGUARD_SERVING_CHECKSUM",
               "recompute": "SDCGUARD_RECOMPUTE_MISMATCH", "replica": "SDCGUARD_REPLICA_DIVERGENCE",
               "canary": "SDCGUARD_CANARY_MISMATCH"}


# ---------------------------------------------------------------------------------------------- protobuf wire format

def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64                       # proto3 int64/enum: two's complement, 10 bytes
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _len(field: int, data: bytes) -> bytes:
    return _key(field, 2) + _varint(len(data)) + data


def _str(field: int, s: Optional[str]) -> bytes:
    return _len(field, s.encode()) if s else b""            # proto3: an empty string isn't written


def _int(field: int, n: Optional[int]) -> bytes:
    return _key(field, 0) + _varint(int(n)) if n else b""


def _bool(field: int, b: bool) -> bytes:
    return _key(field, 0) + b"\x01" if b else b""


def encode_event(ev: dict) -> bytes:
    """One HealthEvent message (fields in field-number order, proto3 defaults left out, map keys sorted: the bytes
    protobuf's deterministic serializer writes)."""
    out = [_int(1, ev.get("version", 1)), _str(2, ev.get("agent")), _str(3, ev.get("componentClass")),
           _str(4, ev.get("checkName")), _bool(5, bool(ev.get("isFatal"))), _bool(6, bool(ev.get("isHealthy"))),
           _str(7, ev.get("message")), _int(8, ACTIONS[ev.get("recommendedAction", "NONE")])]
    out += [_len(9, str(c).encode()) for c in ev.get("errorCode") or []]
    for ent in ev.get("entitiesImpacted") or []:
        out.append(_len(10, _str(1, ent.get("entityType")) + _str(2, ent.get("entityValue"))))
    for k, v in sorted((ev.get("metadata") or {}).items()):
        out.append(_len(11, _len(1, str(k).encode()) + _len(2, str(v).encode())))
    ts = ev.get("generatedTimestamp")
    if ts is not None:
        sec = int(ts)
        nanos = int(round((ts - sec) * 1e9))
        if nanos >= 1_000_000_000:
            sec, nanos = sec + 1, nanos - 1_000_000_000
        out.append(_len(12, _int(1, sec) + _int(2, nanos)))
    out.append(_str(13, ev.get("nodeName")))
    for field, name in ((14, "quarantineOverrides"), (15, "drainOverrides")):
        o = ev.get(name)
        if o:
            out.append(_len(field, _bool(1, bool(o.get("force"))) + _bool(2, bool(o.get("skip")))))
    out.append(_int(16, STRATEGIES[ev.get("processingStrategy", "STORE_AND_ANALYSE")]))
    out += [_str(17, ev.get("id")), _str(18, ev.get("customRecommendedAction"))]
    return b"".join(out)


def encode_events(events: Iterable[dict]) -> bytes:
    """The HealthEvents request: version 1 and the events."""
    return _int(1, 1) + b"".join(_len(2, encode_event(e)) for e in events)


# ----------------------------------------------------------------------------------------------------- the events

def pci_address(pci: Optional[str]) -> Optional[str]:
    """nvidia-smi's 00000000:1B:00.0 as NVSentinel writes PCI entities: 0000:1b:00.0 (its metadata collector keeps
    the last four digits of the domain, lower case)."""
    if not pci or not isinstance(pci, str):
        return None
    parts = pci.strip().split(":")
    if len(parts) != 3:
        return pci.strip().lower()
    return f"{parts[0][-4:].lower()}:{parts[1].lower()}:{parts[2].lower()}"


def _entities(gpu: dict) -> list:
    ents = []
    pci = pci_address(gpu.get("pci_bus_id") or gpu.get("pci"))
    if pci:
        ents.append({"entityType": "PCI", "entityValue": pci})
    if gpu.get("gpu_uuid"):
        ents.append({"entityType": "GPU_UUID", "entityValue": str(gpu["gpu_uuid"])})
    return ents


def node_name(given: Optional[str] = None) -> Optional[str]:
    """The Kubernetes node: given, else NODE_NAME (the downward API's spec.nodeName, as NVSentinel's own monitors
    use), else, outside a pod, this host's name (a pod's hostname is the pod's, not the node's)."""
    if given:
        return given
    if os.environ.get("NODE_NAME"):
        return os.environ["NODE_NAME"]
    if not os.environ.get("KUBERNETES_SERVICE_HOST"):
        return socket.gethostname()
    return None


def _clip(s, n: int = 256) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n - 3] + "..."


def from_check(result: dict, *, node: Optional[str] = None, remediate: bool = False,
               trusted: Optional[bool] = None) -> tuple:
    """(event or None, why not) for one `sdcguard check` result (check.json). trusted: whether its certificate
    verified against the operator's key (None: no key was given)."""
    from . import __version__
    r = str(result.get("result") or "")
    if result.get("simulated_faults") or result.get("SIMULATED_FAULTS"):
        return None, "a demo run (--simulate-faults): not a health result"
    if r not in ("PASS", "FAIL", "SUSPECT"):
        return None, f"{r or 'no result'}: says nothing about the GPU's health"
    gpu = result.get("gpu") or {}
    ents = _entities(gpu)
    if not any(e["entityType"] == "GPU_UUID" for e in ents):
        return None, "the result doesn't name its GPU's UUID"
    act = remediate and (trusted is not False)
    healthy = r == "PASS"
    fatal = r == "FAIL" and act
    why = "; ".join(n[1] for n in (result.get("notes") or [])[:2] if isinstance(n, (list, tuple)) and len(n) > 1)
    msg = (f"sdcguard check {r} on {gpu.get('gpu_name') or 'GPU'} {gpu.get('gpu_uuid')}"
           + (f": {why}" if why and not healthy else ""))
    meta = {"sdcguard_version": __version__, "source": "sdcguard check", "result": r,
            "gpu_name": gpu.get("gpu_name"), "run_id": result.get("run_id"), "serial": gpu.get("serial"),
            "certificate": {True: "verified with the operator's key", False: "not verified with the operator's key",
                            None: "not checked"}[trusted]}
    ev = {"version": 1, "agent": AGENT, "componentClass": "GPU", "checkName": CHECK, "isFatal": fatal,
          "isHealthy": healthy, "message": _clip(msg, 1024),
          "recommendedAction": "RUN_FIELDDIAG" if fatal else "NONE",
          "errorCode": [] if healthy else [f"SDCGUARD_CHECK_{r}"], "entitiesImpacted": ents,
          "metadata": {k: _clip(v) for k, v in meta.items() if v not in (None, "")},
          "generatedTimestamp": time.time(), "nodeName": node or "",
          "processingStrategy": "EXECUTE_REMEDIATION" if (act and r in ("PASS", "FAIL")) else "STORE_AND_ANALYSE"}
    return ev, None


def from_alerts(events: Iterable[dict], *, node: Optional[str] = None, host: Optional[str] = None) -> tuple:
    """(health events, skipped) for alerts from a job (SDCEvent dicts, as in its JSONL log): one event per GPU and
    detector, with the count, the first and last time, and the worst score. Always STORE_AND_ANALYSE, never
    fatal. Only this host's GPUs (host: default this machine's name): a node may report only its own GPUs."""
    from . import __version__
    host = host or socket.gethostname()
    groups: dict = {}
    skipped = []
    for e in events:
        ident = e.get("identity") or {}
        det = str(e.get("detector") or "checksum")
        if not ident.get("gpu_uuid"):
            skipped.append(f"{det} alert without a GPU UUID")
            continue
        if ident.get("host") and ident["host"] != host:
            skipped.append(f"{det} alert on host {ident['host']} (send it from that node)")
            continue
        g = groups.setdefault((ident["gpu_uuid"], det), {"ident": ident, "n": 0, "first": None, "last": None,
                                                          "max_z": None, "where": None, "detail": e.get("detail")})
        g["n"] += 1
        t = e.get("time")
        if isinstance(t, (int, float)):
            g["first"] = t if g["first"] is None else min(g["first"], t)
            g["last"] = t if g["last"] is None else max(g["last"], t)
        z = (e.get("detail") or {}).get("max_z")
        if isinstance(z, (int, float)):
            g["max_z"] = z if g["max_z"] is None else max(g["max_z"], z)
        g["where"] = g["where"] or (e.get("detail") or {}).get("where")
    out = []
    for (uuid, det), g in sorted(groups.items()):
        ident = g["ident"]
        what = {"checksum": "a training matmul failed its checksum", "serve_checksum": "a serving matmul failed "
                "its checksum", "recompute": "a recomputed activation differed from the original",
                "replica": "this GPU's weights drifted from the other replicas'",
                "canary": "this GPU's canary output differed from the other ranks'"}.get(det, f"{det} alert")
        meta = {"sdcguard_version": __version__, "source": "in-job alert", "detector": det, "alerts": g["n"],
                "first_alert": _iso(g["first"]), "last_alert": _iso(g["last"]), "max_z": g["max_z"],
                "where": g["where"], "gpu_name": ident.get("gpu_name"),
                "evidence": (g["detail"] or {}).get("evidence") if isinstance(g["detail"], dict) else None,
                "verified": "false"}
        out.append({"version": 1, "agent": AGENT, "componentClass": "GPU", "checkName": CHECK, "isFatal": False,
                    "isHealthy": False,
                    "message": _clip(f"sdcguard: {what} on {ident.get('gpu_name') or 'GPU'} {uuid} ({g['n']} "
                                     f"alert{'s' if g['n'] != 1 else ''}); a lead to confirm with sdcguard check "
                                     "or casefile, not a verdict", 1024),
                    "recommendedAction": "NONE", "errorCode": [ERROR_CODES.get(det, f"SDCGUARD_{det.upper()}")],
                    "entitiesImpacted": _entities(ident),
                    "metadata": {k: _clip(v) for k, v in meta.items() if v not in (None, "")},
                    "generatedTimestamp": g["last"] if g["last"] is not None else time.time(),
                    "nodeName": node or "", "processingStrategy": "STORE_AND_ANALYSE"})
    return out, skipped


def _iso(t) -> Optional[str]:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if isinstance(t, (int, float)) else None


# ------------------------------------------------------------------------------------------------------- sending

class SendError(RuntimeError):
    pass


def send(events: list, *, socket_path: str = SOCKET, token_file: Optional[str] = None, timeout: float = 10.0,
         retries: int = 4, backoff: float = 1.0) -> None:
    """Send one batch (NVSentinel accepts or rejects a batch whole). Raises SendError with the reason."""
    if not events:
        return
    if not os.path.exists(socket_path):
        raise SendError(f"no NVSentinel socket at {socket_path} (is NVSentinel's platform connector running on "
                        "this node, and is its hostPath mounted here?)")
    try:
        import grpc
    except ImportError as e:
        raise SendError('sending to NVSentinel needs grpcio: pip install "sdcguard[nvsentinel]"') from e
    token_file = token_file if token_file is not None else (TOKEN if os.path.exists(TOKEN) else None)
    body = encode_events(events)
    retry = {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED}
    last = None
    with grpc.insecure_channel(f"unix://{socket_path}") as ch:
        call = ch.unary_unary(METHOD, request_serializer=None, response_deserializer=None)
        for attempt in range(max(1, retries)):
            md = None
            if token_file:
                with open(token_file) as f:              # read again every time: the kubelet rotates it
                    token = f.read().strip()
                if not token:
                    raise SendError(f"the token file {token_file} is empty")
                md = (("authorization", f"Bearer {token}"),)
            try:
                call(body, timeout=timeout, metadata=md)
                return
            except grpc.RpcError as e:
                code = e.code() if hasattr(e, "code") else None
                last = f"{getattr(code, 'name', code)}: {e.details() if hasattr(e, 'details') else e}"
                if code not in retry:
                    raise SendError(f"NVSentinel refused the events ({last})") from e
            time.sleep(backoff * (2 ** attempt))
    raise SendError(f"NVSentinel didn't answer ({last})")


class Publisher:
    """on_alert callback that sends each alert to NVSentinel (STORE_AND_ANALYSE, never fatal), from a background
    thread, at most once per GPU and detector every `min_interval` seconds (NVSentinel folds repeats within its
    own 3-minute window anyway). Failures are counted and logged, never raised: an alert must not stop training.
    Only alerts about this host's GPUs are sent."""

    def __init__(self, socket_path: str = SOCKET, token_file: Optional[str] = None, node: Optional[str] = None,
                 min_interval: float = 300.0, timeout: float = 5.0):
        self.socket_path, self.token_file, self.timeout = socket_path, token_file, timeout
        self.node = node_name(node)
        self.min_interval = min_interval
        self._last: dict = {}
        self._lock = threading.Lock()
        self._threads: list = []
        self.sent = self.failed = self.skipped = 0

    def __call__(self, event) -> None:
        try:
            d = json.loads(event.to_json()) if hasattr(event, "to_json") else dict(event)
            evs, skipped = from_alerts([d], node=self.node)
            if not evs:
                self.skipped += 1
                return
            key = (d["identity"].get("gpu_uuid"), d.get("detector"))
            now = time.time()
            with self._lock:
                if now - self._last.get(key, -1e18) < self.min_interval:
                    return
                self._last[key] = now
            t = threading.Thread(target=self._send, args=(evs, key), daemon=True)
            t.start()
            with self._lock:
                self._threads = [x for x in self._threads if x.is_alive()] + [t]
        except Exception as e:  # noqa: BLE001
            self._fail(None, e)

    def _send(self, evs, key) -> None:
        try:
            send(evs, socket_path=self.socket_path, token_file=self.token_file, timeout=self.timeout)
            with self._lock:
                self.sent += 1
        except Exception as e:  # noqa: BLE001
            self._fail(key, e)

    def _fail(self, key, e) -> None:
        with self._lock:
            self.failed += 1
            if key is not None:
                self._last.pop(key, None)            # the next alert for this GPU tries again
        import logging
        logging.getLogger("sdcguard").warning("sdcguard: couldn't send an alert to NVSentinel: %s", e)

    def flush(self, timeout: float = 10.0) -> None:
        """Wait for the alerts being sent (call before the process exits)."""
        end = time.time() + timeout
        with self._lock:
            threads = list(self._threads)
        for t in threads:
            t.join(max(0.0, end - time.time()))


# ------------------------------------------------------------------------------------------------- the command

def _results(paths: list) -> tuple:
    """(check results with their folders, alert events, problems) from folders and files."""
    checks, alerts, problems = [], [], []
    for p in paths:
        if os.path.isdir(p):
            for d, _sub, files in os.walk(p):
                if "check.json" in files:
                    try:
                        with open(os.path.join(d, "check.json")) as f:
                            checks.append((d, json.load(f)))
                    except (OSError, ValueError) as e:
                        problems.append(f"{os.path.join(d, 'check.json')}: {e}")
        elif p.endswith(".jsonl"):
            try:
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                alerts.append(json.loads(line))
                            except ValueError:
                                problems.append(f"{p}: a line that isn't JSON")
            except OSError as e:
                problems.append(f"{p}: {e}")
        elif os.path.isfile(p):
            try:
                with open(p) as f:
                    checks.append((os.path.dirname(os.path.abspath(p)), json.load(f)))
            except (OSError, ValueError) as e:
                problems.append(f"{p}: {e}")
        else:
            problems.append(f"{p}: not found")
    return checks, alerts, problems


def _age_hours(result: dict) -> Optional[float]:
    import calendar
    try:
        t = calendar.timegm(time.strptime(str(result["time"]), "%Y-%m-%dT%H:%M:%SZ"))
    except (KeyError, ValueError):
        return None
    return max(0.0, (time.time() - t) / 3600)


def _trusted(folder: str, pubkey: Optional[str]) -> Optional[bool]:
    if not pubkey:
        return None
    cert = os.path.join(folder, "certificate.json")
    if not os.path.exists(cert):
        return False
    try:
        from .fleet import _certificate_status
        return bool(_certificate_status(cert, pubkey).get("trusted"))
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    """sdcguard nvsentinel: send check results and job alerts to NVIDIA NVSentinel. Exit 0 when everything that
    should be sent was sent, 2 when it couldn't be (NVSentinel refused or couldn't be reached, or bad input).
    Never 1: across sdcguard's commands that means a GPU computed wrong results, and this command computes none."""
    import argparse
    p = argparse.ArgumentParser(prog="sdcguard nvsentinel", description=__doc__.split("\n\n")[0])
    p.add_argument("paths", nargs="*", default=["sdcguard_check"],
                   help="check folders, check.json files, or alert logs (.jsonl); default sdcguard_check")
    p.add_argument("--socket", default=SOCKET)
    p.add_argument("--token-file", default=None, help=f"default {TOKEN} when it exists")
    p.add_argument("--node-name", default=None, help="the Kubernetes node (default NODE_NAME)")
    p.add_argument("--remediate", action="store_true",
                   help="a check FAIL asks for action (EXECUTE_REMEDIATION, fatal, RUN_FIELDDIAG) and a PASS clears it")
    p.add_argument("--pubkey", default=None, help="with --remediate: only a FAIL whose certificate verifies "
                                                  "against this key asks for action")
    p.add_argument("--max-age-hours", type=float, default=36.0,
                   help="check results older than this aren't sent (a GPU that left the node, a stale folder)")
    p.add_argument("--dry-run", action="store_true", help="print the events, send nothing")
    a = p.parse_args(argv)
    node = node_name(a.node_name)
    if not node and not a.dry_run:
        print("sdcguard nvsentinel: which Kubernetes node is this? Set NODE_NAME (the downward API's "
              "spec.nodeName) or --node-name")
        return 2
    checks, alerts, problems = _results(a.paths)
    for msg in problems:
        print(f"sdcguard nvsentinel: {msg}")
    events, notes = [], []
    for folder, r in checks:
        age = _age_hours(r)
        if age is None or age > a.max_age_hours:
            notes.append(f"{folder}: not sent (" + ("its time isn't recorded" if age is None else
                                                    f"{age:.0f} hours old") + ")")
            continue
        ev, why = from_check(r, node=node, remediate=a.remediate, trusted=_trusted(folder, a.pubkey))
        if ev is None:
            notes.append(f"{folder}: not sent ({why})")
        else:
            events.append(ev)
    evs, skipped = from_alerts(alerts, node=node)
    events += evs
    notes += [f"not sent: {s}" for s in sorted(set(skipped))]
    for n in notes:
        print(f"sdcguard nvsentinel: {n}")
    if a.dry_run:
        print(json.dumps(events, indent=2, default=str))
        return 2 if problems and not events else 0
    if not events:
        print("sdcguard nvsentinel: nothing to send")
        return 2 if problems else 0
    try:
        send(events, socket_path=a.socket, token_file=a.token_file)
    except SendError as e:
        print(f"sdcguard nvsentinel: {e}")
        return 2
    print(f"sdcguard nvsentinel: sent {len(events)} event(s) to {a.socket} as node {node}")
    return 2 if problems else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
