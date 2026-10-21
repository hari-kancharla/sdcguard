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


def _how(event) -> str:
    """What went wrong, in the words of the detector that saw it."""
    d, kind = event.detail, event.detector
    ranks = ", ".join(str(r) for r in d.get("suspect_ranks") or [])
    # suspect_ranks are global ranks; with a replica group, `world` is the size of that group
    of = f"in a group of {d.get('world')}" if "suspect_group_ranks" in d else f"of {d.get('world')}"
    if kind == "replica":
        return (f"holds data-parallel weights that drifted from the other ranks (rank {ranks} {of}; "
                "replicas must match bit for bit)" if d.get("clear_majority") else
                f"is in a data-parallel group whose weights disagree with no clear majority (ranks {ranks})")
    if kind == "canary":
        return (f"gave a different output for the same canary batch (rank {ranks} {of}, first at "
                f"{d.get('first_bad_layer')})")
    if kind == "recompute":
        n = d.get("outputs_that_differ")
        n = len(n) if isinstance(n, (list, tuple)) else n
        return ("gave different bits when training recomputed this checkpointed block from the same inputs"
                + (f" ({n} of its outputs differ)" if n else ""))
    if str(d.get("kind", "")).startswith("attention"):
        return f"failed the fused attention check ({d.get('check') or d.get('kind')})"
    z = d.get("max_z")
    line = d.get("alarm_line") or 6
    if z is None or not isinstance(z, (int, float)) or not math.isfinite(z) or d.get("nonfinite"):
        # the serving check tells Inf/NaN from finite values too large for the inputs (ServeGuard._report_rows);
        # the training checks' nonfinite covers both
        if d.get("nonfinite") is False and isinstance(z, (int, float)):
            return "produced finite values far larger than its inputs allow"
        if d.get("nonfinite") and kind == "serve_checksum":
            return "produced Inf/NaN the inputs can't explain"
        return "produced an output its inputs can't produce (Inf/NaN, or values too large for them)"
    return f"was off by {z:.1f}x its rounding noise (alarm line {line:g})"


def describe(event) -> str:
    """One line a person can act on: which GPU, which layer, how far off, what it means."""
    d, g = event.detail, event.identity
    how = _how(event)
    where = d.get("where") or d.get("op") or ("a checkpointed block" if event.detector == "recompute" else
                                              "a matmul")
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
    ev = d.get("evidence")
    if isinstance(ev, str) and not ev.startswith("not saved"):
        parts.append(f"(the numbers that alarmed: {ev}.pt)")
    return " ".join(parts)


class Webhook:
    """on_alert callback that posts each alert to a webhook (Slack, Discord, PagerDuty/Opsgenie via
    their generic webhooks, or anything that takes JSON). At most one delivered post per GPU and layer every
    `min_interval` seconds. A failed post is retried (`retries` times, with a growing pause) and logged to the
    "sdcguard" logger; only a delivered post starts the quiet interval, so a webhook that was down doesn't
    silence the next alert. Alerts are also in the guard's log either way."""

    def __init__(self, url: str, fmt: str = "slack", min_interval: float = 600.0, timeout: float = 5.0,
                 retries: int = 2, backoff: float = 1.0):
        if fmt not in ("slack", "discord", "json"):
            raise ValueError("fmt must be slack, discord or json")
        self.url, self.fmt, self.min_interval, self.timeout = url, fmt, min_interval, timeout
        self.retries, self.backoff = max(0, int(retries)), float(backoff)
        self._last: dict = {}
        self._sending: set = set()
        self._threads: list = []              # posts being sent (flush() waits for them)
        self._lock = threading.Lock()
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
        with self._lock:
            if now - self._last.get(key, -1e18) < self.min_interval or key in self._sending:
                return
            self._sending.add(key)
        try:
            body = json.dumps(self.payload(event), default=str).encode()
            t = threading.Thread(target=self._post, args=(body, key), daemon=True)
            t.start()
            with self._lock:
                self._threads = [x for x in self._threads if x.is_alive()] + [t]
        except Exception as e:  # noqa: BLE001  (no thread, or an event that won't serialize)
            # the reservation is released, so the next alert for this GPU and layer is tried again
            with self._lock:
                self._sending.discard(key)
                self.failed += 1
            import logging
            logging.getLogger("sdcguard").warning("sdcguard: webhook post couldn't start: %s", e)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait up to `timeout` seconds in all for the posts still being sent; True when none is left. A process
        that is exiting ends its daemon threads wherever they are, so call this before it does."""
        end = time.monotonic() + timeout
        with self._lock:
            threads = list(self._threads)
        for t in threads:
            t.join(max(0.0, end - time.monotonic()))
        with self._lock:
            self._threads = [x for x in self._threads if x.is_alive()]
            return not self._threads

    def _post(self, body: bytes, key=None) -> None:
        err = None
        try:
            for attempt in range(self.retries + 1):
                try:
                    req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json",
                                                                                 "User-Agent": "sdcguard"})
                    with urllib.request.urlopen(req, timeout=self.timeout):
                        pass
                    with self._lock:
                        self.sent += 1
                        self._last[key] = time.time()
                    return
                except Exception as e:  # noqa: BLE001  (never let alert delivery break serving or training)
                    err = e
                    if attempt < self.retries:
                        time.sleep(self.backoff * (2 ** attempt))
            with self._lock:
                self.failed += 1
            import logging
            logging.getLogger("sdcguard").warning("sdcguard: webhook post failed after %d attempts: %s",
                                                  self.retries + 1, err)
        finally:
            with self._lock:
                self._sending.discard(key)


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

    sdcguard_layers_checked                 linear layers this process checks (less those whose check never ran)
    sdcguard_fast_layers                    of those, checked by the Triton kernel
    sdcguard_worst_score                    worst score over all layers at the last read (alarm line 6)
    sdcguard_layer_worst_score{layer}       per layer, at the last read
    sdcguard_alerts_total{layer,UUID}       alerts so far; UUID is the GPU's (GPU-...), as dcgm-exporter labels its
                                            GPU series, so the two join on it (empty when unknown)
    sdcguard_all_alerts_total               alerts on any layer (exported at 0, for increase() rules)
    sdcguard_layers_unchecked{method}       vLLM: linear layers not checked, by reason (quantization method, ...)
    sdcguard_layers_never_checked           vLLM: registered layers whose check never ran (their forward bypasses it)
    sdcguard_weights_checked_ratio          vLLM: share of the model's linear-layer weights whose matmuls are checked
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
        # UUID, not gpu: dcgm-exporter's `gpu` is NVIDIA's device index, which a process's CUDA index often isn't
        self.alerts = Counter("sdcguard_alerts", "Alerts", ["layer", "UUID"], registry=r)
        # unlabelled, so it is exported at 0 before the first alert (increase() over a labelled counter that
        # appears at 1 misses that first alert)
        self.alerts_all = Counter("sdcguard_all_alerts", "Alerts on any layer", registry=r)
        self.unchecked = Gauge("sdcguard_layers_unchecked", "Linear layers this process doesn't check, by reason",
                               ["method"], registry=r)
        self.never_checked = Gauge("sdcguard_layers_never_checked", "Registered layers whose check never ran",
                                   registry=r)
        self.weight_share = Gauge("sdcguard_weights_checked_ratio",
                                  "Share of the model's linear-layer weights whose matmuls are checked", registry=r)
        self.reads = Counter("sdcguard_reads", "Times results were read back", registry=r)
        self.executed = Counter("sdcguard_checks_executed", "Layer checks that ran, counted on the GPU", registry=r)
        self.unjudged = Counter("sdcguard_unjudged_checksums", "Checksums that couldn't be judged", registry=r)
        self._seen: dict = {}                     # id(guard) -> (checks executed, unjudged) already exported
        self.last = Gauge("sdcguard_last_read_timestamp_seconds", "When results were last read back", registry=r)
        self.port = None
        if port is not None:
            start_http_server(int(port), addr=addr, registry=r)
            self.port = int(port)

    def layer_report(self, report: dict) -> None:
        """The vLLM plugin's count of its model's linear layers (vllm_plugin.layer_report)."""
        try:
            self.unchecked.clear()            # (a recount can drop a reason)
        except AttributeError:
            pass
        for why, n in (report.get("unchecked_by_method") or {}).items():
            self.unchecked.labels(method=why).set(n)
        self.never_checked.set(len(report.get("never_checked") or ()))
        self.weight_share.set(float(report.get("weight_share_checked") or 0.0))

    def observe(self, guard, scores: Iterable[tuple], events: list) -> None:
        """scores: (layer name, score) for every layer read back."""
        stats = getattr(guard, "stats", {})
        self.layers.set(len(guard.layers) - int(stats.get("layers_never_checked", 0)))
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
            self.alerts.labels(layer=str(e.detail.get("where")), UUID=str(e.identity.get("gpu_uuid") or "")).inc()
            self.alerts_all.inc()
        ex, un = int(stats.get("checks_executed", 0)), int(stats.get("unjudged_checksums", 0))
        pex, pun = self._seen.get(id(guard), (0, 0))
        if ex > pex:
            self.executed.inc(ex - pex)
        if un > pun:
            self.unjudged.inc(un - pun)
        self._seen[id(guard)] = (max(ex, pex), max(un, pun))
        self.reads.inc()
        self.last.set(time.time())
