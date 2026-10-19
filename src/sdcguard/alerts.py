"""Where alerts go: chat/incident webhooks and Prometheus metrics. Both are optional and neither ever
makes the GPU wait (webhooks are sent from a background thread; metrics are updated when results are
read back anyway).

    from sdcguard import SDCGuard, ServeGuard
    from sdcguard.alerts import Webhook
    guard = SDCGuard(on_alert=Webhook("https://hooks.slack.com/services/..."))
    serve = ServeGuard(webhook="https://hooks.slack.com/services/...", metrics_port=9464)

vLLM plugin: SDCGUARD_WEBHOOK=<url> and SDCGUARD_METRICS_PORT=<port> (each worker adds its local GPU
index to the port). SDCGUARD_WEBHOOK_FORMAT: slack (default, {"text": ...}), discord ({"content":
...}) or json (the full event).
"""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
from typing import Callable, Iterable, Optional


def describe(event) -> str:
    """One line a person can act on: which GPU, which layer, how far off, what it means."""
    d, g = event.detail, event.identity
    z = d.get("max_z")
    how = ("produced Inf/NaN the inputs can't explain" if (z is None or not math.isfinite(z) or d.get("nonfinite"))
           else f"was off by {z:.1f}x its rounding noise (alarm line 6)")
    where = d.get("where") or d.get("op") or "a matmul"
    gpu = g.get("gpu_uuid") or g.get("device") or "unknown GPU"
    parts = [f"sdcguard ALERT: {where} on {gpu}"]
    extra = ", ".join(x for x in (g.get("gpu_name"), f"PCI {g['pci_bus_id']}" if g.get("pci_bus_id") else None,
                                  f"host {g['host']}" if g.get("host") else None,
                                  f"step {event.step}" if event.step is not None else None) if x)
    if extra:
        parts.append(f"({extra})")
    parts.append(how)
    if d.get("reading"):
        parts.append(f"- {d['reading']}")
    return " ".join(parts)


class Webhook:
    """on_alert callback that posts each alert to a webhook (Slack, Discord, PagerDuty/Opsgenie via
    their generic webhooks, or anything that takes JSON). At most one post per GPU and layer every
    `min_interval` seconds; failures are ignored (alerts are also in the guard's log)."""

    def __init__(self, url: str, fmt: str = "slack", min_interval: float = 600.0, timeout: float = 5.0):
        if fmt not in ("slack", "discord", "json"):
            raise ValueError("fmt must be slack, discord or json")
        self.url, self.fmt, self.min_interval, self.timeout = url, fmt, min_interval, timeout
        self._last: dict = {}
        self.sent = 0
        self.failed = 0

    def payload(self, event) -> dict:
        if self.fmt == "json":
            return json.loads(event.to_json())
        key = "content" if self.fmt == "discord" else "text"
        return {key: describe(event)}

    def __call__(self, event) -> None:
        key = (event.detail.get("where"), event.identity.get("gpu_uuid"))
        now = time.time()
        if now - self._last.get(key, -1e18) < self.min_interval:
            return
        self._last[key] = now
        body = json.dumps(self.payload(event), default=str).encode()
        threading.Thread(target=self._post, args=(body,), daemon=True).start()

    def _post(self, body: bytes) -> None:
        try:
            req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json",
                                                                         "User-Agent": "sdcguard"})
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
            self.sent += 1
        except Exception:  # noqa: BLE001  (never let alert delivery break serving or training)
            self.failed += 1


def chain(*callbacks: Optional[Callable]) -> Optional[Callable]:
    """One on_alert callback that calls each of the given ones (None entries are skipped)."""
    cbs = [c for c in callbacks if c is not None]
    if not cbs:
        return None
    if len(cbs) == 1:
        return cbs[0]

    def call(event):
        for c in cbs:
            try:
                c(event)
            except Exception:  # noqa: BLE001
                pass
    return call


class Metrics:
    """Prometheus metrics for a ServeGuard (needs `pip install prometheus-client`).

    sdcguard_layers_checked                 linear layers this process checks
    sdcguard_fast_layers                    of those, checked by the Triton kernel
    sdcguard_worst_score                    worst score over all layers at the last read (alarm line 6)
    sdcguard_layer_worst_score{layer}       per layer, at the last read
    sdcguard_alerts_total{layer,gpu}        alerts so far
    sdcguard_checks_executed_total          layer checks that ran, counted on the GPU
    sdcguard_unjudged_checksums_total       checksums that couldn't be judged (not evidence of a clean result)
    sdcguard_reads_total                    times results were read back
    sdcguard_last_read_timestamp_seconds    when (a stale value means checks stopped being read)
    """

    def __init__(self, port: Optional[int] = None, registry=None, addr: str = "0.0.0.0"):
        from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        self.layers = Gauge("sdcguard_layers_checked", "Linear layers this process checks", registry=r)
        self.fast = Gauge("sdcguard_fast_layers", "Layers checked by the Triton kernel", registry=r)
        self.worst = Gauge("sdcguard_worst_score", "Worst score over all layers at the last read", registry=r)
        self.layer_worst = Gauge("sdcguard_layer_worst_score", "Worst score per layer at the last read", ["layer"],
                                 registry=r)
        self.alerts = Counter("sdcguard_alerts", "Alerts", ["layer", "gpu"], registry=r)
        self.reads = Counter("sdcguard_reads", "Times results were read back", registry=r)
        self.executed = Counter("sdcguard_checks_executed", "Layer checks that ran, counted on the GPU", registry=r)
        self.unjudged = Counter("sdcguard_unjudged_checksums", "Checksums that couldn't be judged", registry=r)
        self._seen: dict = {}                     # id(guard) -> (checks executed, unjudged) already exported
        self.last = Gauge("sdcguard_last_read_timestamp_seconds", "When results were last read back", registry=r)
        self.port = None
        if port is not None:
            start_http_server(int(port), addr=addr, registry=r)
            self.port = int(port)

    def observe(self, guard, scores: Iterable[tuple], events: list) -> None:
        """scores: (layer name, score) for every layer read back."""
        self.layers.set(len(guard.layers))
        try:
            self.fast.set(sum(guard.uses_fast(s) for s in range(len(guard.layers))))
        except Exception:  # noqa: BLE001
            pass
        worst = 0.0
        for name, z in scores:
            z = float(z) if z == z else 0.0
            self.layer_worst.labels(layer=name).set(z)
            worst = max(worst, z)
        self.worst.set(worst)
        for e in events:
            self.alerts.labels(layer=str(e.detail.get("where")), gpu=str(e.identity.get("gpu_uuid"))).inc()
        stats = getattr(guard, "stats", {})
        ex, un = int(stats.get("checks_executed", 0)), int(stats.get("unjudged_checksums", 0))
        pex, pun = self._seen.get(id(guard), (0, 0))
        if ex > pex:
            self.executed.inc(ex - pex)
        if un > pun:
            self.unjudged.inc(un - pun)
        self._seen[id(guard)] = (max(ex, pex), max(un, pun))
        self.reads.inc()
        self.last.set(time.time())
