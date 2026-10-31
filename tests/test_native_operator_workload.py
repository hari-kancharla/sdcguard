"""Ordinary grouped traffic controls. Owner: Hari Krishna Kancharla."""
from copy import deepcopy
import threading
import time

import pytest

from sdcguard.native_operator import client, workload
from sdcguard.native_operator.common import digest, encoded


def plan(encode=None):
    return workload.build_plan(encode or (lambda text, **kw: list(text.encode())),
        run_id="a" * 32, packet_sha256="b" * 64, engine_sessions={"observer-on": "c" * 32})


def test_complete_text_is_tokenized_without_a_chat_template_or_slicing():
    calls = []
    def encode(text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        calls.append(text)
        return list(text.encode())
    value, payloads = plan(encode)
    assert value["chat_template"] is False
    assert all(bytes(tokens).decode() in calls for tokens in payloads.values())
    assert all(bytes(tokens).decode().endswith("\nAnswer:") for tokens in payloads.values())
    rows = value["arms"][0]["requests"]
    first = [r for r in rows if r["group_position"] == 0]
    assert len({tuple(payloads[r["client_request_id"]][:16]) for r in first}) == 9
    assert any(r["prompt_tokens"] % 16 for r in rows)
    assert all("cold" not in r and "covered" not in r for r in rows)


@pytest.mark.parametrize("tokens", [{"input_ids": [1]}, [True], [1.0], [], [1] * 2001, [-1], [1000000]])
def test_token_contract_stays_strict(tokens):
    with pytest.raises(ValueError, match="bounded ordinary token IDs"):
        plan(lambda *args, **kwargs: deepcopy(tokens))


@pytest.mark.parametrize("change", ["group", "order", "shared", "template", "duplicate_ending", "group_first_block"])
def test_group_plan_cannot_assume_cache_reuse_from_malformed_inputs(change):
    value, payloads = plan()
    rows = value["arms"][0]["requests"]
    if change == "group":
        rows[0]["cache_group"] = "other"
    elif change == "order":
        rows[0]["group_position"] = True
    elif change == "shared":
        rows[0]["shared_prefix_tokens"] += 1
    elif change == "template":
        value["chat_template"] = True
    elif change == "duplicate_ending":
        row = rows[1]
        tokens = payloads[rows[0]["client_request_id"]]
        payloads[row["client_request_id"]] = list(tokens)
        row.update(prompt_tokens=len(tokens), prompt_sha256=digest(encoded(tokens)))
    else:
        for index, row in enumerate(rows[17:34]):
            tokens = payloads[row["client_request_id"]]
            tokens[:16] = payloads[rows[index]["client_request_id"]][:16]
            row["prompt_sha256"] = digest(encoded(tokens))
    with pytest.raises(ValueError):
        workload.validate_payloads(value, payloads)


def test_each_first_completion_precedes_its_later_requests_and_all_slots_stay_in_scope(tmp_path):
    value, payloads = plan()
    rows = value["arms"][0]["requests"]
    ids = {r[k]: r for r in rows for k in ("prefill_request_id", "client_request_id")}
    finished = set()
    lock = threading.Lock()
    calls = []
    def transport(role, body, headers, *, timeout):
        row = ids[headers["X-Request-Id"]]
        with lock:
            group_index = (row["slot"] // 17) * 17
            if role == "prefill":
                if row["group_position"]:
                    assert group_index in finished
                else:
                    assert set(range(group_index)) <= finished
            calls.append((role, row["slot"]))
        response = dict(id="cmpl-" + headers["X-Request-Id"],
            usage=dict(prompt_tokens=len(body["prompt"]), completion_tokens=body["max_tokens"]),
            choices=[dict(index=0, text="ordinary CPU control", finish_reason="length")])
        if role == "prefill":
            response["kv_transfer_params"] = dict(do_remote_prefill=True, do_remote_decode=False,
                remote_request_id="recorded-" + str(row["slot"]), remote_engine_id="producer")
        else:
            with lock:
                finished.add(row["slot"])
        return dict(http_status=200, response_body=response)
    capture = client.ClientCapture(value, payloads, tmp_path / "capture", transport=transport)
    assert capture.run_arm("observer-on", time.monotonic() + 30)
    inventory = capture.finish()
    assert inventory["completed"] is True
    assert len(inventory["requests"]) == len(finished) == 153 and len(calls) == 306
    assert all(p["measured_complete"] == p["measured_planned"] == 48 for p in inventory["phases"])
    assert all(p["all_requests_elapsed_seconds"] >= p["elapsed_seconds"] for p in inventory["phases"])


def test_first_request_failure_stops_later_sends_without_erasing_planned_slots(tmp_path):
    value, payloads = plan()
    calls = []
    def transport(role, body, headers, *, timeout):
        calls.append(role)
        raise OSError("synthetic CPU transport unavailable")
    capture = client.ClientCapture(value, payloads, tmp_path / "capture", transport=transport)
    assert not capture.run_arm("observer-on", time.monotonic() + 30)
    inventory = capture.finish()
    assert calls == ["prefill"] and inventory["completed"] is False
    assert len(inventory["requests"]) == 153
    assert all(row["decode"] is None for row in inventory["requests"])
