"""Finite ordinary client receipts. Owner: Hari Krishna Kancharla.

Only the explicitly called transport sends HTTP. No retry, replacement request,
server, SDK or device action occurs on import.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import http.client
import json
import math
from pathlib import Path
import threading
import time

from .common import checkpoint, encoded

PATH = "/v1/completions"
PORTS = {"prefill": 8100, "decode": 8200}
RESPONSE_LIMIT = 512 * 1024


def post(role, body, headers, *, timeout, guard=None):
    """One nonstream completion on an owned numeric-loopback destination."""
    if role not in PORTS or type(body) is not dict or type(headers) is not dict:
        raise ValueError("one supported ordinary stage is required")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("HTTP stage needs a finite timeout of at most 60 seconds")
    raw = encoded(body)
    if len(raw) > 256 * 1024:
        raise ValueError("ordinary request exceeds its body bound")
    connection = http.client.HTTPConnection("127.0.0.1", PORTS[role], timeout=timeout)
    started = time.monotonic()
    try:
        if guard is not None:
            guard(role, None)
        connection.connect()
        if guard is not None:
            guard(role, connection.sock)
        connection.request("POST", PATH, raw, headers={**headers, "Content-Type": "application/json",
                           "Accept-Encoding": "identity", "Connection": "close"})
        response = connection.getresponse()
        response_raw = response.read(RESPONSE_LIMIT + 1)
        if len(response_raw) > RESPONSE_LIMIT:
            raise ValueError("ordinary response exceeds its body bound")
        parsed = json.loads(response_raw)
        if type(parsed) is not dict:
            raise ValueError("ordinary response is not an object")
        response_headers = response.getheaders()
        if len(response_headers) > 64 or any(len(k) + len(v) > 4096 for k, v in response_headers):
            raise ValueError("ordinary response headers exceed their bound")
        return dict(http_status=response.status, response_body=parsed,
            response_headers=response_headers, response_sha256=hashlib.sha256(response_raw).hexdigest(),
            http_seconds=time.monotonic() - started)
    finally:
        connection.close()


def initial_inventory(plan, plan_raw):
    return dict(schema="sdcguard.checksum-client-inventory.v1", run_id=plan["run_id"],
        packet_sha256=plan["packet_sha256"], plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
        completed=False, requests=[dict(arm_id=arm["arm_id"], slot=row["slot"],
            client_request_id=row["client_request_id"], prefill=None, decode=None)
            for arm in plan["arms"] for row in arm["requests"]], phases=[])


class ClientCapture:
    """All planned slots exist before traffic; journal loss stops new sends."""
    def __init__(self, plan, payloads, out, *, transport=post, clock=time.monotonic):
        from .workload import validate_payloads
        validate_payloads(plan, payloads)
        self.plan, self.payloads, self.out = deepcopy(plan), deepcopy(payloads), Path(out)
        self.plan_raw = encoded(plan)
        self.transport, self.clock = transport, clock
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.active = 0
        self.peak = 0
        self.http_active = {role: 0 for role in PORTS}
        self.http_peak = {role: 0 for role in PORTS}
        self.http_peak_total = 0
        self.sequence = 0
        self.journal_bytes = 0
        self.journal = self.out / "CLIENT_RECEIPTS.jsonl"
        self.inventory = initial_inventory(plan, self.plan_raw)
        self.index = {(r["arm_id"], r["slot"]): r for r in self.inventory["requests"]}
        self.out.mkdir(exist_ok=False)
        checkpoint(self.out / "REQUEST_PLAN.json", plan)
        self.journal.open("xb").close()
        self.flush()

    def flush(self):
        with self.lock:
            checkpoint(self.out / "CLIENT_INVENTORY.json", self.inventory)

    def emit(self, kind, value):
        with self.lock:
            self.sequence += 1
            raw = json.dumps(dict(sequence=self.sequence, kind=kind, **value),
                             sort_keys=True, allow_nan=False).encode() + b"\n"
            if len(raw) > 1024 * 1024 or self.journal_bytes + len(raw) > 16 * 1024 * 1024:
                self.stop.set()
                raise ValueError("client receipt journal exceeds its retained bound")
            with self.journal.open("ab", buffering=0) as stream:
                if stream.write(raw) != len(raw):
                    self.stop.set()
                    raise OSError("client receipt journal write was incomplete")
                import os
                os.fsync(stream.fileno())
            self.journal_bytes += len(raw)

    def _stage(self, arm_id, row, role, body, identifier, deadline):
        receipt = dict(completed=False, method="POST", path=PATH, http_status=None,
            request_headers={"X-Request-Id": identifier}, request_body=deepcopy(body),
            response_body=None, error=None)
        with self.lock:
            self.index[(arm_id, row["slot"])][role] = receipt
        self.emit("stage_started", dict(arm_id=arm_id, slot=row["slot"], role=role, receipt=receipt))
        try:
            if self.stop.is_set():
                raise ValueError("new sends stopped after incomplete evidence or lost server ownership")
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError("ordinary client deadline reached")
            with self.lock:
                self.http_active[role] += 1
                self.http_peak[role] = max(self.http_peak[role], self.http_active[role])
                self.http_peak_total = max(self.http_peak_total, sum(self.http_active.values()))
            try:
                observed = self.transport(role, body, receipt["request_headers"], timeout=min(60, remaining))
            finally:
                with self.lock:
                    self.http_active[role] -= 1
            if type(observed) is not dict:
                raise ValueError("HTTP transport receipt is not an object")
            receipt.update(observed)
            if receipt["http_status"] != 200:
                raise ValueError("ordinary HTTP stage did not succeed")
            response = receipt["response_body"]
            if type(response) is not dict or response.get("id") != "cmpl-" + identifier:
                raise ValueError("ordinary response request ID is not the pinned API form")
            usage = response.get("usage")
            if (type(usage) is not dict or type(usage.get("prompt_tokens")) is not int
                    or usage["prompt_tokens"] != row["prompt_tokens"]):
                raise ValueError("actual prompt usage differs from the pinned tokenizer count")
            choices = response.get("choices")
            if (type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict
                    or choices[0].get("index") != 0 or type(choices[0].get("text")) is not str
                    or choices[0].get("finish_reason") not in ("length", "stop")
                    or type(usage.get("completion_tokens")) is not int
                    or not 1 <= usage["completion_tokens"] <= body["max_tokens"]):
                raise ValueError("ordinary response has no successful bounded completion")
            receipt["completed"] = True
        except Exception as error:
            receipt["error"] = dict(kind=type(error).__name__, detail=str(error)[:512])
            self.stop.set()
        with self.lock:
            self.index[(arm_id, row["slot"])][role] = deepcopy(receipt)
        self.emit("stage_finished", dict(arm_id=arm_id, slot=row["slot"], role=role, receipt=receipt))
        return receipt

    def _request(self, arm, row, deadline):
        if self.stop.is_set():
            return
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        started = self.clock()
        try:
            prompt = self.payloads[row["client_request_id"]]
            common = dict(model=self.plan["model"], prompt=prompt, n=1, stream=False,
                          add_special_tokens=False, temperature=0.0)
            prefill_body = dict(common, request_id=row["prefill_request_id"], max_tokens=1,
                                kv_transfer_params=dict(do_remote_decode=True, do_remote_prefill=False))
            first = self._stage(arm["arm_id"], row, "prefill", prefill_body,
                                row["prefill_request_id"], deadline)
            if not first["completed"]:
                return
            descriptor = first["response_body"].get("kv_transfer_params")
            if (type(descriptor) is not dict or descriptor.get("do_remote_prefill") is not True
                    or descriptor.get("do_remote_decode") is not False
                    or type(descriptor.get("remote_request_id")) is not str
                    or type(descriptor.get("remote_engine_id")) is not str):
                self.stop.set()
                with self.lock:
                    self.index[(arm["arm_id"], row["slot"])]["handoff_error"] = "ordinary producer descriptor is missing"
                return
            decode_body = dict(common, request_id=row["client_request_id"], max_tokens=row["max_tokens"],
                               kv_transfer_params=deepcopy(descriptor))
            self._stage(arm["arm_id"], row, "decode", decode_body, row["client_request_id"], deadline)
        except Exception as error:
            self.stop.set()
            with self.lock:
                self.index[(arm["arm_id"], row["slot"])]["client_error"] = dict(
                    kind=type(error).__name__, detail=str(error)[:512])
        finally:
            with self.lock:
                self.active -= 1
                self.index[(arm["arm_id"], row["slot"])]["pair_elapsed_seconds"] = self.clock() - started

    def run_arm(self, arm_id, deadline):
        arms = [arm for arm in self.plan["arms"] if arm["arm_id"] == arm_id]
        if len(arms) != 1 or self.stop.is_set():
            raise ValueError("a fresh supported arm is required")
        arm = arms[0]
        for phase in self.plan["phases"]:
            if self.stop.is_set():
                break
            rows = [row for row in arm["requests"] if row["phase_id"] == phase["phase_id"]]
            measured = [row for row in rows if row["kind"] == "measured"]
            self.peak = 0
            self.http_peak = {role: 0 for role in PORTS}
            self.http_peak_total = 0
            started = self.clock()
            measured_seconds = 0.0
            groups = list(dict.fromkeys(row["cache_group"] for row in rows))
            for group_id in groups:
                group = [row for row in rows if row["cache_group"] == group_id]
                if self.stop.is_set():
                    break
                # The first ordinary completion seeds only its own group.
                # Its cache state still comes from the recorder, not this order.
                self._request(arm, group[0], deadline)
                if self.stop.is_set():
                    break
                batch_started = self.clock()
                with ThreadPoolExecutor(max_workers=phase["concurrency"]) as executor:
                    futures = [executor.submit(self._request, arm, row, deadline) for row in group[1:]]
                    for future in futures:
                        future.result()
                measured_seconds += self.clock() - batch_started
            with self.lock:
                observed = [self.index[(arm_id, row["slot"])] for row in measured]
                complete = sum(type(row.get("decode")) is dict and row["decode"].get("completed") is True
                               for row in observed)
                self.inventory["phases"].append(dict(arm_id=arm_id, phase_id=phase["phase_id"],
                    requested_concurrency=phase["concurrency"], actual_peak_client_requests=self.peak,
                    actual_peak_http_stages=self.http_peak_total, actual_peak_http_by_role=dict(self.http_peak),
                    measured_planned=len(measured), measured_complete=complete,
                    elapsed_scope="sum of later-request batches, excluding each first request",
                    all_requests_elapsed_seconds=self.clock() - started,
                    elapsed_seconds=measured_seconds))
            self.flush()
        return not self.stop.is_set()

    def finish(self):
        with self.lock:
            self.inventory["completed"] = (not self.stop.is_set() and self.active == 0 and
                all(type(row.get("prefill")) is dict and row["prefill"].get("completed") is True
                    and type(row.get("decode")) is dict and row["decode"].get("completed") is True
                    for row in self.inventory["requests"]))
        self.flush()
        return deepcopy(self.inventory)
