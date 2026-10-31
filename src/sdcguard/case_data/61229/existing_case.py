"""Real vLLM prefill/decode pair with sdcguard's KV lifetime recording, searched with no model of the protocol:
from a recorded normal run, which single delay at a recorded checkpoint makes the decode side break a lifetime
rule, and (on a patched vLLM) does the open fix catch that same stall and recover?

Setup (vLLM 0.32.0, Qwen/Qwen3-0.6B, NixlConnector pull mode, both instances on one GPU like vLLM's own NIXL
integration test): a prefill instance (port 8100, engine id P, kv_producer) and a decode instance (port 8200,
engine id D, kv_consumer), each with SDCGUARD_LIFETIME_DIR so the plugin records allocations, frees, published
descriptors, reads, completions, admissions and checkpoints to one event file per process. The prefill side's KV
lease is shortened to --lease seconds, its KV pool to --blocks blocks (--max-model-len is 512: vLLM refuses to
start when one longest request cannot fit). Prefix caching is off on both sides.

What this experiment chooses, not the search: the lease length, the load after the lease, the canary prompts, and
a hold timeout longer than the lease (so every held read outlives the lease). The search chooses where to hold and
when the hold ends, and checks the rules. The experiment also adds one condition of its own, tried first: release
X's read when X's leased source block is freed by another request. On the prefill side a plain completion's blocks
are freed only after it finishes, so that free shows the block's next owner has already written its KV there; the
search's own conditions stop at reallocation, which comes before the write (so a read released there may still
see the old bytes, a race). Reproducing the known issue this way is "an automatic search over
schedules derived from a clean recording reproduces a known defect", not discovery. On one GPU a NIXL read
completes in microseconds, so this exercises a read issued after the blocks were reclaimed (R1), not one in flight
during the reclamation (R2); the ledger tells them apart.

Where signals live (verified against vLLM 0.32.0 + PR #62065 source). NIXL pull mode: the DECODE/consumer issues
the block READs and the lease verification; the PREFILL/producer holds the send leases and receives completion
notifications. So:
  prefill :8100  vllm:nixl_num_kv_expired_reqs, vllm:nixl_num_notifications_after_expiry, and the "Potentially
                 invalid KV blocks for unrecognized request" log line  (producer-side)
  decode  :8200  vllm:nixl_num_lease_check_failures (PR #62065's counter), vllm:nixl_num_failed_transfers, and the
                 per-request WARNING "Declining KV pulled for request <X>: ... released the KV blocks before the
                 read completed. The KV load fails."  (consumer-side)
A failed scrape, or a counter this build does not export, is recorded as null (unknown), never 0; a counter that
first appears during a run counts from 0. The counter is per-engine, not per-request, and
conflates a pre-read "cannot verify" decline with the real post-read nonce-mismatch catch, so the authoritative
proof that the fix rejected THIS request's stale pull is the decode WARNING line carrying X's request id; the
counter is corroboration.

What one run establishes is reported as independent facts, never one inferred from another:
  exposure    the ledger saw X's read refer to blocks whose allocation was retired or reassigned (R1 after the
              reclamation, R2 in flight); the descriptor is built before any check, so this fires even where a
              later verifier then rejects the read. This is a stale-REFERENCE attempt; the ledger alone does not
              prove inference consumed the rejected bytes (see limitations).
  lease       the prefill side expired X's lease (its counter, or a lease note naming X)
  rejection   the decode side declined X's pull: its WARNING line, matched to X by the request id this experiment
              assigns, with the decline's CAUSE kept apart. Only "released the KV blocks before the read completed"
              (lease_mismatch) is the fix catching a stale read; "the lease slot ... could not be read"
              (verification_unavailable) is a conservative refusal because the check could not run, and a pre-read
              decline (cannot_check) means nothing was verified at all. The counter is corroboration only.
  answer      X's greedy answer is correct (byte-for-byte equal to the clean prefill/decode reference, which must
              agree across its repetitions), correct_recompute (equal to the agreed local reference computed on the
              decode engine with no KV transfer, the recompute path), differs (anything else, even if it contains
              X's word: "PLUM ::::" is not correct), foreign (contains another request's distinctive canary: proof of
              a cross-request leak), failed_closed (the request errored, e.g. HTTP 500 under
              kv_load_failure_policy=fail), or unresolved (the request did not finish: a hang; inconclusive)
A foreign answer is always harm. A differing answer is harm when the ledger recorded a stale access and no decline
was recorded for X (the stale bytes were admitted). A differing answer AFTER a decline is never counted as success
or as recovery: it is reported as bad_answer_after_decline and makes the run inconclusive. A differing answer with
no recorded stale access is reported apart (nondeterminism or a missed event), not harm.

Variants. --variant unpatched: plain vLLM 0.32.0. --variant fixed: the same base plus a backport of PR #62065's
vLLM-only diff (lease-slot verification, on by default there). On the fixed build the read still refers to freed
blocks (exposure fires, the descriptor is built before the check), but the verifier declines it, its decode-side
counter rises, and the request recomputes (kv_load_failure_policy=recompute) to the correct answer, or fails closed
(fail). The result is the CONTRAST between the two on the SAME pinned candidate, not either run alone.

Paired replay. Candidates are derived from the clean baseline. --mode compare (default): unless --candidate is
given, this run DISCOVERS the schedule to compare: it tries candidates in ranked order (up to --max-candidates)
until one actually corrupts X's answer, saves it to OUT/candidate.json, runs a short-stall control at the same hold
point, then replays the schedule --reps times. With --candidate <file> (the other image, e.g. the fixed build
replaying the unpatched run's discovery) it skips discovery and replays that exact saved spec. Every run records
what its hold actually did: released because its condition was reached, or because its timeout expired, and how
long it stalled the decode's read. A failure after an expired hold is reported as such, not as the targeted
ordering. --mode search tries up to --max-candidates once each, for exploration. This is the same controlled
schedule on both images, not a deterministic replay of every GPU event. The hold at read_submit delays the decode
side posting its KV read (host side, before _read_blocks_for_req); it does not slow the transfer itself.

Exit codes: 0 no confirmed harm on a demonstrated trigger (for the fixed variant: every planned replay finished,
at least --confirm of them exercised the trigger with the hold released by its condition (not its timeout),
every exercised replay ended exactly as the configured policy
promises, recovered under recompute or failed closed under fail, matched to X by request id with the lease_mismatch
cause, and every other replay was clean), 1 confirmed harm (a verified differing or foreign answer), 2 the hooks
recorded nothing, 3 a control broke a rule or did not return the exact reference answer, 4 the harness failed (a
server did not start, say), 5 inconclusive (invalid baseline; no candidate that corrupts the answer was found;
harm seen once in discovery but not reproduced in the replays; the trigger was never demonstrated; a run did not
finish or did not start from idle servers; a decline could not be attributed; or the fixed build's outcomes were
mixed, unverifiable or too few). The `verdict` field states the outcome in words. NIXL's UCX backend runs with
ucx_error_handling_mode=none (see UCX_PARAMS).

Limitations stated honestly: same-machine CUDA-IPC/shm transport, not network RDMA; a read issued after
reclamation (R1), not one in flight during it; the trigger is induced by a deliberate hold, not observed
organically; no coverage of tokens already streamed to a client before detection; one model, one config, prefix
caching off; the fixed build applies PR #62065's fix commit 42b8f230 onto the 0.32.0 release (its runtime lines
unchanged), not the PR's latest head, which later moved two NIXL descriptor calls to a newer NIXL API;
and exposure is a stale-reference attempt recorded by the ledger, with the decode WARNING line (not the ledger) as
the proof that the fix refused to admit the bytes to inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("SDCGUARD_PACKAGED_REPLAY") != "1":
    sys.path.insert(0, os.path.join(ROOT, "src"))

from sdcguard.lifetime.adapter import merge, merge_events, read_events  # noqa: E402
from sdcguard.lifetime.obligations import Finding, evaluate as evaluate_rules  # noqa: E402
from sdcguard.lifetime.outcomes import terminal_outcome  # noqa: E402
from sdcguard.lifetime.realsearch import candidates, rank  # noqa: E402

# Distinctive single-token canaries the 0.6B model echoes reliably at temperature 0. X's is PLUM; each background
# request draws a DISTINCT word from the pool, so a foreign word in X's answer names the exact request it leaked
# from (swap-canary causality). Foreign detection is word-boundary, not substring, to avoid chance matches.
CANARY_X = "PLUM"
CANARIES = ["MANGO", "CEDAR", "TULIP", "AMBER", "COBALT", "SAFFRON", "JUNIPER", "QUARTZ", "ZEPHYR", "OBSIDIAN",
            "MARIGOLD", "VELVET"]


def canary_prompt(word, repeat=1):
    return " ".join([f"The secret word is {word}. Repeat the secret word: {word}."] * repeat) + " Again:"


PROMPT_X = canary_prompt(CANARY_X)
OTHER = [canary_prompt(w) for w in CANARIES]

# Modal's sandbox hides /sys/class/net, so UCX has no TCP transport, and under NIXL's default peer error handling
# the shared-memory transports are not allowed either (NIXL fails to start, "no active messages transport"). Both
# servers share one machine, so this turns peer error handling off (the setting vLLM PR #62905 proposes to expose);
# UCX then uses shared memory and CUDA IPC. It changes how a failed peer is detected, not when blocks are freed.
UCX_PARAMS = {"ucx_error_handling_mode": "none"}

# producer/prefill :8100 counters; consumer/decode :8200 counters (the lease-check one exists only on the fixed
# build). Verified sidedness against the 0.32.0 + PR #62065 source.
PRODUCER_SIGNALS = ("vllm:nixl_num_kv_expired_reqs", "vllm:nixl_num_notifications_after_expiry")
CONSUMER_SIGNALS = ("vllm:nixl_num_lease_check_failures", "vllm:nixl_num_failed_transfers")
VLLM_LOG_LINE = "Potentially invalid KV blocks for unrecognized request"       # producer-side, prefill log

# decode-side per-request decline lines (PR #62065). The post-read form is the #61229 catch; the pre-read form is a
# cannot-verify decline (missing lease params), which must NOT be read as a caught stale read.
POST_DECLINE_RE = re.compile(r"Declining KV pulled for request (\S+): (.+?)\. The KV load fails\.")
PRE_DECLINE_RE = re.compile(r"Declining KV pull for request (\S+) from engine \S+: (.+)")
# The fixed build prints the SAME post-read line for two different reasons (lease_slots.py at fix commit 42b8f230):
# the producer released the blocks before the read completed (the #61229 catch), or the lease slot could not be read
# at all (the check could not run, so declining is a conservative refusal, not a detection). Only the first is the
# fix catching a stale read.
LEASE_MISMATCH_TEXT = "released the KV blocks before the read completed"
VERIFY_UNAVAILABLE_TEXT = "could not be read"


def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def pd_complete(model, prompt, max_tokens=24, prefill_only=False, params=None):
    """vLLM's toy proxy flow: prefill on P with do_remote_decode, then decode on D with P's kv_transfer_params."""
    body = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
            "kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False, "remote_engine_id": None,
                                   "remote_block_ids": None, "remote_host": None, "remote_port": None}}
    p = post("http://localhost:8100/v1/completions", body)
    kv = p.get("kv_transfer_params") or {}
    if prefill_only:
        return kv
    text, _err, _id = decode(model, prompt, kv, max_tokens)
    return text


def local_complete(model, prompt, max_tokens=24):
    """X computed entirely on the decode engine, with no KV transfer: the path a recompute after a declined pull
    takes. Its agreed answer is the second clean reference (an answer equal to it is correct_recompute)."""
    d = post("http://localhost:8200/v1/completions",
             {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}, timeout=60)
    return d["choices"][0]["text"]


def decode(model, prompt, kv, max_tokens=24, request_id=None):
    """The decode request. Returns (text, None, request_id) on HTTP 200, or (None, error, request_id|None) when the
    engine fails the request closed (e.g. HTTP 500 under kv_load_failure_policy=fail when a lease verifier rejects
    the pull): a failed request is not a wrong answer, which the caller tells apart from a hang. request_id is the
    completion id vLLM assigns, used to match the decode-side decline log line to THIS request. Passing request_id
    sets it in advance (vLLM 0.32.0's completions API takes it in the body and names the request cmpl-<id>), so
    the decline line can be matched even when the request fails and its error body carries no id."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "kv_transfer_params": kv}
    if request_id:
        body["request_id"] = request_id
    try:
        d = post("http://localhost:8200/v1/completions", body, timeout=300)
    except urllib.error.HTTPError as e:            # the engine answered with an error status
        rid = None
        try:
            rid = json.loads(e.read()).get("id")
        except Exception:  # noqa: BLE001
            pass
        return None, f"HTTP {e.code}", rid
    choice = d["choices"][0]
    if choice.get("finish_reason") == "error":     # an engine-side failure in a 200 body is not an answer
        return None, "finish_reason=error", d.get("id")
    return choice["text"], None, d.get("id")


def counter(metrics_text, name):
    """The sum of a Prometheus counter over its label sets (name or name_total); 0.0 when absent."""
    total = 0.0
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        head = line.split("{", 1)[0].split(" ", 1)[0]
        if head in (name, name + "_total"):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                pass
    return total


def counter_or_none(metrics_text, name):
    """Like counter, but None when no sample of the counter is exported at all (this build does not have it, or
    it has not been created yet): absence is unknown, not 0."""
    found, total = False, 0.0
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        head = line.split("{", 1)[0].split(" ", 1)[0]
        if head in (name, name + "_total"):
            try:
                total += float(line.rsplit(" ", 1)[1])
                found = True
            except (IndexError, ValueError):
                pass
    return total if found else None


def scrape(port, names):
    """Scrape the given Prometheus counters from one server. Each value is a float, or None (unknown) when the
    scrape fails or the server exports no such counter: neither is ever silently 0."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as e:  # noqa: BLE001
        return {n: None for n in names} | {"metrics_error": str(e)}
    return {n: counter_or_none(text, n) for n in names}


def log_count(path, needle):
    try:
        with open(path) as f:
            return f.read().count(needle)
    except OSError:
        return 0


def producer_signals(logP):
    """Producer/prefill :8100 counters plus its error-log line count."""
    out = scrape(8100, PRODUCER_SIGNALS)
    out["log_lines"] = log_count(logP, VLLM_LOG_LINE)
    return out


def consumer_signals():
    """Consumer/decode :8200 counters, including PR #62065's lease-check-failure counter."""
    return scrape(8200, CONSUMER_SIGNALS)


def _rose(sig, name):
    """True only when a counter is known and above zero (a delta, or an absolute count since start)."""
    v = (sig or {}).get(name)
    return isinstance(v, (int, float)) and v > 0


def num_active(port):
    """running + waiting request counts on a server, or None when the scrape fails."""
    s = scrape(port, ("vllm:num_requests_running", "vllm:num_requests_waiting"))
    r, w = s.get("vllm:num_requests_running"), s.get("vllm:num_requests_waiting")
    return None if r is None or w is None else r + w


def wait_idle(timeout=15):
    """Wait until both servers report no running or waiting requests, so each run starts from a demonstrated clean
    state rather than mid-flight work from the one before it. True once idle is observed; False if the servers
    reported work for the whole timeout (the run would not start clean); None if the counts could never be read
    (unknown, recorded as such, never taken as idle)."""
    t0 = time.time()
    seen = False
    while time.time() - t0 < timeout:
        p, d = num_active(8100), num_active(8200)
        if p is not None and d is not None:
            seen = True
            if p == 0 and d == 0:
                return True
        time.sleep(0.5)
    return False if seen else None


def delta(before, after):
    """after - before per key. A failed scrape on either side makes every metric unknown (None); the log-line
    count, read from a file, is unaffected. A counter absent before and present after was created during the
    window and counts from 0. Anything else unknown stays None."""
    failed = "metrics_error" in before or "metrics_error" in after
    out = {}
    for k, v in before.items():
        if k == "metrics_error":
            continue
        a = after.get(k)
        if failed and k != "log_lines":
            out[k] = None
        elif isinstance(v, (int, float)) and isinstance(a, (int, float)):
            out[k] = a - v
        elif v is None and isinstance(a, (int, float)):
            out[k] = a
        else:
            out[k] = None
    return out


def loglines(path):
    try:
        with open(path) as f:
            return f.read().splitlines()
    except OSError:
        return []


def _id_core(rid):
    """The hex core of a completion id: cmpl-9a78...-0-8562 -> 9a78... . Used to match a decode decline line (whose
    request id carries extra engine/sequence suffixes) to the id returned by the completions API."""
    if not rid:
        return ""
    return rid.split("cmpl-")[-1].split("-")[0]


def decline_cause(kind, reason):
    """Why the decode side declined: lease_mismatch (the producer released the blocks before the read completed:
    the #61229 catch), verification_unavailable (the lease slot could not be read: nothing was verified),
    other_post_read (a post-read decline with a reason this experiment does not know), or cannot_check (a pre-read
    decline: the lease could not be checked, so nothing was read)."""
    if kind == "pre_read":
        return "cannot_check"
    if LEASE_MISMATCH_TEXT in (reason or ""):
        return "lease_mismatch"
    if VERIFY_UNAVAILABLE_TEXT in (reason or ""):
        return "verification_unavailable"
    return "other_post_read"


def declines(lines):
    """Parse decode-worker decline WARNING lines into {req, reason, kind, cause}. kind post_read / pre_read is where
    the decline happened; cause says why (see decline_cause). Only cause lease_mismatch is the #61229 catch."""
    out = []
    for ln in lines:
        m = POST_DECLINE_RE.search(ln)
        if m:
            out.append({"req": m.group(1), "reason": m.group(2), "kind": "post_read",
                        "cause": decline_cause("post_read", m.group(2))})
            continue
        m = PRE_DECLINE_RE.search(ln)
        if m:
            reason = m.group(2).rstrip(".")
            out.append({"req": m.group(1), "reason": reason, "kind": "pre_read",
                        "cause": decline_cause("pre_read", reason)})
    return out


def match_decline(dec_list, ids):
    """The decline whose request id carries any of these ids' hex cores, or None."""
    cores = [c for c in (_id_core(i) for i in ids) if c]
    for d in dec_list:
        if any(c in d["req"] for c in cores):
            return d
    return None


def foreign_hits(x, pool):
    """Background canaries that appear as whole words in X's answer (case-insensitive). Each names the one request
    that used it, so a hit is a traceable cross-request leak."""
    if not x:
        return []
    return [w for w in pool if re.search(rf"\b{re.escape(w)}\b", x, re.IGNORECASE)]


# what a hold at each recorded checkpoint actually delays (every intervention must say what it delayed)
INTERVENTION = {
    "read_submit": "delays the decode side posting its KV read (host side, before "
                   "NixlPullConnectorWorker._read_blocks_for_req); it does not slow the transfer itself",
}
_AFTER_RE = re.compile(r"after ([0-9.]+)s")


def hold_outcome(rel):
    """What the hold actually did in one run: released because its condition was reached, or because its timeout
    expired (a failure then does not show that the targeted ordering happened), or there was no hold."""
    if any("condition reached" in r for r in rel):
        return "condition_reached"
    if any("hold expired" in r for r in rel):
        return "expired"
    return "not_held"


def stall_seconds(rel):
    """How long the held read waited, from the plugin's release line ("... after 14.006s"), or None."""
    for r in rel:
        m = _AFTER_RE.search(r) if "released" in r else None
        if m:
            return float(m.group(1))
    return None


def _spec_key(spec):
    return json.dumps(spec, sort_keys=True, default=str)


def load_candidate(path):
    """A saved candidate (another run's OUT/candidate.json). Raises ValueError saying what to do when unusable."""
    try:
        with open(path) as f:
            rec = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read candidate {path} ({e}); run the unpatched variant first: it discovers the "
                         "schedule that corrupts the answer and saves it there") from None
    spec = rec.get("spec") if isinstance(rec, dict) else None
    if not isinstance(spec, dict) or "point" not in spec or "until" not in spec:
        raise ValueError(f"candidate {path} has no usable hold spec")
    return rec


def save_candidate(path, rec):
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)


class SavedCandidate:
    """A hold schedule fixed for replay: discovered in this run, or loaded from another run's candidate.json. It
    replays the saved spec exactly (its symbolic blocks resolve per run), so both images run the same schedule."""

    def __init__(self, rec):
        self.rec = dict(rec)
        self.spec = dict(rec["spec"])
        self.timeout = self.spec.get("timeout")
        self.source = rec.get("source")
        self.kind = rec.get("kind")

    @classmethod
    def from_candidate(cls, c, timeout, discovered=None):
        rec = {"label": c.describe(), "spec": c.hold_spec(timeout), "source": c.source, "kind": c.kind}
        if discovered is not None:              # what actually happened when it was found, stated plainly
            rec["found_by"] = {"hold_outcome": discovered.get("hold_outcome"), "stall_s": discovered.get("stall_s"),
                               "x": discovered.get("x"), "terminal": discovered["outcome"]["terminal"]}
        return cls(rec)

    def hold_spec(self, timeout=None):
        return dict(self.spec, timeout=self.timeout if timeout is None else timeout)

    def describe(self):
        return self.rec.get("label", "saved hold")

    def record(self):
        return dict(self.rec, intervention=INTERVENTION.get(self.spec.get("point"), "not described"))


X_PREFILL = "$X_PREFILL_ID"       # replaced in each run with X's prefill-side request id (its kv_transfer_params)


def rewritten_candidate(i=0):
    """The experiment's own condition, tried first: hold X's read until X's leased source block i is freed by a
    request other than X's prefill. A prefill (producer) server does not defer frees, and a plain completion frees
    its blocks only after it finishes, so that free shows the block's next owner has already written its KV there:
    the read is released right after the overwrite, not on a long timeout. The saved spec names X through a
    placeholder, so the same schedule replays on another image."""
    spec = {"point": "read_submit", "key": None,
            "until": {"kind": "free", "args": {"region": {"desc_block": i}, "by": {"not": X_PREFILL}}}}
    return SavedCandidate({"label": f"hold read_submit until X's source block {i} is freed by its next owner, "
                                    "which has written it", "spec": spec, "source": "experiment", "kind": "rewritten"})


def classify(violations, x, base_answer, x_prefill_id, producer_sig, consumer_sig, lease_notes, pool=CANARIES,
             x_error=None, x_finished=True, rejected=False, reject_reason=None, reject_kind=None,
             reject_cause=None, reject_unknown=False, local_answer=None):
    """Independent facts for one run; see the module docstring. Harm never depends on a counter or on the fix: a
    foreign answer is always harm, and a differing answer is harm when a stale access was recorded and the stale
    bytes were admitted (no decline recorded for X). Correctness is byte-for-byte equality with an agreed clean
    reference (the prefill/decode one, or the local recompute one), never "contains the word". Rejection is the
    decode decline line matched to X; only cause lease_mismatch counts as the fix catching the stale read.
    reject_unknown: a decline appeared in this run's window but could not be attributed to X."""
    stale = [v for v in violations if "read-" in v]
    hits = foreign_hits(x, pool)
    says_own = bool(x) and CANARY_X.lower() in x.lower()
    if not x_finished:
        answer = "unresolved"
    elif x is None and x_error is not None:
        answer = "failed_closed"
    elif x is None:
        answer = "unresolved"
    elif hits:
        answer = "foreign"                   # another request's canary: a cross-request leak, whatever else is there
    elif base_answer is not None and x == base_answer:
        answer = "correct"
    elif local_answer is not None and x == local_answer:
        answer = "correct_recompute"         # equals the agreed answer of the recompute path (no KV transfer)
    elif base_answer is None:
        answer = "unverified"
    else:
        answer = "differs"                   # anything else, even with X's word in it ("PLUM ::::" is not correct)
    said_ok = answer in ("correct", "correct_recompute")
    if rejected and reject_cause is None:    # derive the cause from the line itself; unknown reasons never count
        reject_cause = decline_cause(reject_kind, reject_reason) if (reject_reason or reject_kind == "pre_read") \
            else None
    caught = bool(rejected) and reject_cause == "lease_mismatch"

    exp = producer_sig.get("vllm:nixl_num_kv_expired_reqs")
    note_hit = any(x_prefill_id and x_prefill_id in n for n in lease_notes)
    lease_expired = bool((isinstance(exp, (int, float)) and exp > 0) or note_hit)
    lease_unknown = exp is None and not note_hit

    lcf = consumer_sig.get("vllm:nixl_num_lease_check_failures")
    counter_rose = isinstance(lcf, (int, float)) and lcf > 0
    counter_unknown = lcf is None

    admitted = not rejected and not reject_unknown          # no decline recorded or suspected for X
    harm = (answer == "foreign") or (answer == "differs" and bool(stale) and admitted)
    bad_after_decline = answer == "differs" and bool(rejected)
    changed_without_stale = answer == "differs" and not stale and admitted

    out = {"exposure": bool(stale), "access": None,
           "lease_expired": lease_expired, "lease_expired_unknown": lease_unknown,
           "rejected": bool(rejected), "reject_reason": reject_reason, "reject_kind": reject_kind,
           "reject_cause": reject_cause if rejected else None, "reject_unknown": bool(reject_unknown),
           "caught_stale_lease": caught,
           "reject_counter_rose": bool(counter_rose), "reject_counter_unknown": counter_unknown,
           "answer": answer, "says_own_word": says_own, "foreign_words": hits, "harm": harm,
           "bad_answer_after_decline": bad_after_decline,
           "changed_without_stale_access": changed_without_stale}
    if stale:
        first = stale[0]
        out["access"] = ("read issued after the blocks were reclaimed" if first.startswith("R1") and "submitted"
                         in first else "read in flight when the blocks were reclaimed" if first.startswith("R2")
                         else first.split()[0])
    # terminal outcome in words, and a level for continuity with earlier runs
    if answer == "unresolved":
        terminal = "inconclusive_unresolved"
    elif answer == "foreign":
        terminal = "harm_foreign"
    elif harm:
        terminal = "harm_garbled"
    elif reject_unknown:
        terminal = "inconclusive_decline_unattributed"     # a decline happened, but not provably X's
    elif rejected and reject_cause == "cannot_check":
        terminal = "declined_before_read"       # the lease could not be checked, so nothing was read: not #61229
    elif rejected and reject_cause == "verification_unavailable":
        terminal = "declined_unverifiable"      # the lease slot could not be read: a refusal, not a detection
    elif rejected and not caught:
        terminal = "declined_other_reason"
    elif answer == "failed_closed" and caught:
        terminal = "failed_closed"
    elif answer == "failed_closed":
        terminal = "failed_without_decline"     # an error with no matched decline: not the fix at work
    elif bad_after_decline:
        terminal = "declined_answer_differs"    # the fix fired, but the answer is not a clean reference: not success
    elif caught and said_ok and stale:
        terminal = "recovered"
    elif caught and said_ok:
        terminal = "rejected_without_recorded_stale_read"
    elif stale and said_ok:
        terminal = "exposed_answer_unchanged"   # a stale read the answer did not show: the hazard, invisible
    elif changed_without_stale:
        terminal = "changed_nondeterministic"
    elif said_ok:
        terminal = "clean"
    else:
        terminal = "inconclusive"
    out["terminal"] = terminal
    out["level"] = (4 if answer == "foreign" else 3 if harm else 2 if stale
                    else 1 if lease_expired else 0)
    return out


def wait_ready(port, proc, log, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server on {port} exited; see {log}")
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise TimeoutError(f"server on {port} not ready")


def events(dirpath):
    return read_events(dirpath)


def rules_for_request(evs, violations, client_id):
    """Map the experiment's assigned API id to one recorded internal attempt.

    vLLM prefixes completion ids with cmpl- and appends a sequence/scheduler suffix.
    This mapping belongs to this adapter, not to the engine-independent evaluator.
    Ambiguous/missing matches are unknown. Full history retains hook-installation,
    publication, allocation and recording-health evidence before the trial window.
    """
    ids = {e.get("args", {}).get("request") for e in evs
           if e.get("kind") == "evidence" and e.get("args", {}).get("what") == "attempt"}
    matches = sorted(r for r in ids if isinstance(r, str)
                     and (r == client_id or r.startswith(f"cmpl-{client_id}-")))
    if len(matches) != 1:
        return [Finding("lease_valid_at_use", client_id,
                        missing=["one unambiguous recorded internal request attempt"],
                        detail=f"API id maps to {len(matches)} recorded attempt ids").as_dict()]
    return [f.as_dict() for f in evaluate_rules(evs, violations, requests=matches,
                                               rules=["lease_valid_at_use"])]


def evaluator_summary(trials, controls, variant):
    """Keep rule validation separate from the historical answer/ledger oracle.

    That oracle itself uses ledger exposure, so agreement is not independent proof
    of detection accuracy. Unmapped and inconclusive outcomes remain in this table.
    """
    expected = {"harm_garbled": {"violated"}, "harm_foreign": {"violated"},
                "exposed_answer_unchanged": {"violated"}, "clean": {"satisfied"},
                "recovered": {"correctly_rejected", "recovered"},
                "failed_closed": {"correctly_rejected", "refused"}}
    comparison = []
    for t in controls + trials:
        terminal = t["outcome"]["terminal"]
        accepts = expected.get(terminal)
        comparison.append({"label": t["label"], "terminal": terminal,
                           "lease_rule": t["lease_rule"],
                           "expected_from_outcome": sorted(accepts) if accepts else None,
                           "agrees": t["lease_rule"] in accepts if accepts else None})
    held_allowed = {"violated"} if variant == "unpatched" else {"correctly_rejected", "recovered", "refused"}
    controls_ok = len(controls) == 2 and all(t["lease_rule"] == "satisfied" for t in controls)
    held_ok = bool(trials) and all(t["lease_rule"] in held_allowed for t in trials)
    return {"controls_satisfied": controls_ok, "held_rules_match_required_verdicts": held_ok,
            "passed": controls_ok and held_ok,
            "hooks_vs_log_and_outcome": comparison,
            "comparison_limit": "Historical outcomes also use the ledger; agreement is not an independent accuracy estimate.",
            "scope": "Observed source lifetime and admission, not model use or answer correctness."}


def dispose_notification_overlaps(evs, violations, rules):
    """Retain raw R2, but distinguish an explicitly witnessed completed READ.

    Only this attempt's source-only R2 overlaps after its attributable completion
    notification qualify. Other operations, destination regions and other rules
    remain actionable. An unknown evaluator finding never suppresses anything.
    """
    certificates = []
    for f in rules:
        if f.get("rule") != "lease_valid_at_use" or f.get("verdict") != "satisfied":
            continue
        refs = {(r.get("session"), r.get("pid"), r.get("seq")) for r in f.get("evidence", [])
                if r.get("what") == "released_by_read_notification"}
        witnesses = [e for e in evs if (e.get("session"), e.get("pid"), e.get("seq")) in refs]
        attempts = [e for e in evs if e.get("kind") == "evidence"
                    and e.get("args", {}).get("what") == "attempt"
                    and e["args"].get("request") == f.get("request")]
        if len(witnesses) == 1 and len(attempts) == 1:
            a = attempts[0]["args"]
            certificates.append((set(a["op_ids"]), {tuple(r) for r in a["source_regions"]}, witnesses[0]))
    actionable, dispositions = [], []
    for v in violations:
        matched = False
        if (v.rule == "R2" and getattr(v, "ops", ()) and getattr(v, "regions", ())
                and type(v.step) is int and 0 <= v.step < len(evs)):
            event = evs[v.step]
            for ops, regions, witness in certificates:
                if (set(v.ops) <= ops and {tuple(r) for r in v.regions} <= regions
                        and event.get("clock_id") == witness.get("clock_id")
                        and isinstance(event.get("mono"), (int, float))
                        and isinstance(witness.get("mono"), (int, float))
                        and event["mono"] >= witness["mono"]):
                    dispositions.append({"rule": v.rule, "step": v.step, "message": v.message,
                                         "disposition": "source retired after attributable READ completion notification",
                                         "witness": {k: witness.get(k) for k in ("session", "pid", "seq")}})
                    matched = True
                    break
        if not matched:
            actionable.append(v)
    return actionable, dispositions


def tail(path, n=40):
    try:
        with open(path) as f:
            return f.read().splitlines()[-n:]
    except OSError:
        return []


PINNED_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def model_revision(value):
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("a resolved 40-character lowercase model commit is required")
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", type=model_revision, default=PINNED_MODEL_REVISION,
                    help="resolved model and tokenizer commit; main is not accepted")
    ap.add_argument("--lease", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--baseline-reps", type=int, default=3, help="clean X references; must agree to be the oracle")
    ap.add_argument("--mode", default="compare", choices=("compare", "search"),
                    help="compare: pin the lead candidate and replay it --reps times (paired across variants); "
                         "search: try up to --max-candidates (discovery)")
    ap.add_argument("--reps", type=int, default=5, help="replays of the pinned candidate (compare mode)")
    ap.add_argument("--confirm", type=int, default=3,
                    help="discovery: a candidate that garbles the answer is pinned only if it does so in 2/3 of this "
                         "many runs (one garbled answer could be a race the read sometimes wins)")
    ap.add_argument("--max-candidates", type=int, default=8, help="search mode only")
    ap.add_argument("--candidate", default=None, help="path to a saved candidate spec to pin (byte-faithful replay)")
    ap.add_argument("--others", type=int, default=8, help="other requests sent to P while X is held")
    ap.add_argument("--variant", default="unpatched", choices=("unpatched", "fixed"),
                    help="fixed: the running vLLM carries a backport of PR #62065 (lease-slot verification)")
    ap.add_argument("--kv-load-failure-policy", default="recompute", choices=("recompute", "fail"),
                    help="what the engine does when a load is declined (fixed variant): recompute locally, or fail")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    evdir, holds = os.path.join(a.out, "events"), os.path.join(a.out, "holds.json")
    os.makedirs(evdir, exist_ok=True)
    with open(holds, "w") as f:
        json.dump([], f)
    result = {"model": a.model, "model_revision": a.revision, "tokenizer_revision": a.revision,
              "model_revision_basis": "requested immutable model/tokenizer revision only; local snapshot verification, when performed, is recorded separately in RUN_MANIFEST.json",
              "lease_s": a.lease, "blocks": a.blocks, "nixl_ucx_params": UCX_PARAMS,
              "variant": a.variant, "mode": a.mode, "reps": a.reps,
              "kv_load_failure_policy": a.kv_load_failure_policy,
              "metric_sources": {"producer_8100": list(PRODUCER_SIGNALS), "consumer_8200": list(CONSUMER_SIGNALS)}}
    procs, logs = {}, {}
    try:
        return _run(a, evdir, holds, result, procs, logs)
    except Exception as e:  # noqa: BLE001  (a harness failure must not look like a finding)
        result["error"] = f"{type(e).__name__}: {e}"
        result["verdict"] = "harness_failed"
        return 4
    finally:
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            try:
                p.wait(30)
            except Exception:  # noqa: BLE001
                p.kill()
        result["log_tail"] = {r: tail(path) for r, path in logs.items()}
        with open(os.path.join(a.out, "RESULT.json"), "w") as f:
            json.dump(result, f, indent=2)


def _run(a, evdir, holds, result, procs, logs) -> int:
    if a.candidate:                          # an unusable saved schedule fails here, before two server startups
        load_candidate(a.candidate)
    for role, port, side in (("P", 8100, 5559), ("D", 8200, 5659)):
        extra = {"kv_lease_duration": a.lease}
        if a.variant == "fixed":
            extra["kv_lease_verification"] = True      # PR #62065's verifier (on by default there; explicit here)
        kv = {"kv_connector": "NixlConnector", "kv_role": "kv_producer" if role == "P" else "kv_consumer",
              "engine_id": role, "kv_load_failure_policy": a.kv_load_failure_policy,
              "kv_connector_extra_config": extra}
        env = dict(os.environ, SDCGUARD_LIFETIME_DIR=evdir, SDCGUARD_LIFETIME_ENGINE=role,
                   SDCGUARD_LIFETIME_HOLDS=holds, SDCGUARD_LIFETIME_RECORD="1",
                   SDCGUARD_NIXL_UCX_PARAMS=json.dumps(UCX_PARAMS),
                   VLLM_NIXL_SIDE_CHANNEL_PORT=str(side),
                   UCX_NET_DEVICES="all", VLLM_PORT=str(port + 100))
        cmd = ["vllm", "serve", a.model, "--revision", a.revision, "--tokenizer-revision", a.revision,
               "--port", str(port), "--enforce-eager", "--block-size", "16",
               "--gpu-memory-utilization", "0.3", "--max-model-len", "512",
               "--kv-transfer-config", json.dumps(kv)]
        cmd += ["--no-enable-prefix-caching"]      # no partial local hits: reads pair block for block
        if role == "P":
            cmd += ["--num-gpu-blocks-override", str(a.blocks)]
        logs[role] = os.path.join(a.out, f"vllm_{role}.log")
        procs[role] = subprocess.Popen(cmd, env=env, stdout=open(logs[role], "w"), stderr=subprocess.STDOUT)
        result[f"ready_{role}_s"] = wait_ready(port, procs[role], logs[role])
    installed = [e for e in events(evdir) if e["kind"] == "note"]
    result["recording"] = len(events(evdir)) > 0
    result["hooks"] = {str(e.get("pid")): e["args"].get("hooks") for e in installed if "hooks" in e.get("args", {})}

    # baseline: repeated clean X references (must agree to be a trustworthy oracle), then the other canaries so the
    # search has a normal recording to derive holds from.
    refs = [pd_complete(a.model, PROMPT_X) for _ in range(max(1, a.baseline_reps))]
    base_x = refs[0]
    result["baseline_x"] = base_x
    result["baseline_x_refs"] = refs
    result["baseline_refs_agree"] = all(r == base_x for r in refs)
    result["baseline_x_says_its_word"] = CANARY_X.lower() in (base_x or "").lower()
    result["baseline_refs_all_say_word"] = all(bool(r) and CANARY_X.lower() in r.lower()
                                               and not foreign_hits(r, CANARIES) for r in refs)
    # the oracle is usable only when the clean references AGREE byte for byte, say X's word and carry no foreign
    # word: correctness is equality with that agreed answer, never "contains the word"
    result["baseline_oracle_ok"] = result["baseline_refs_agree"] and result["baseline_refs_all_say_word"]
    # the second reference: X computed on the decode engine alone (the recompute path), also repeated and agreed.
    # Optional: if it cannot be obtained, only the prefill/decode reference is used, and the reason is recorded.
    local_x = None
    try:
        lrefs = [local_complete(a.model, PROMPT_X) for _ in range(max(1, a.baseline_reps))]
        result["local_x_refs"] = lrefs
        result["local_refs_agree"] = all(r == lrefs[0] for r in lrefs)
        if result["local_refs_agree"] and lrefs[0] and not foreign_hits(lrefs[0], CANARIES):
            local_x = lrefs[0]
    except Exception as e:  # noqa: BLE001
        result["local_ref_error"] = f"{type(e).__name__}: {e}"
    result["local_x"] = local_x
    result["local_matches_baseline"] = (local_x == base_x) if local_x is not None else None
    for i in range(a.num - 1):
        pd_complete(a.model, OTHER[i % len(OTHER)])
    result["baseline_producer_signals"] = producer_signals(logs["P"])
    result["baseline_consumer_signals"] = consumer_signals()
    led = merge(evdir, track_content=False)
    result["baseline_events"] = len(led.events)
    result["baseline_violations"] = [f"{v.rule} {v.message}" for v in led.violations]
    kinds = {e["kind"] for e in events(evdir)}
    missing = [k for k in ("alloc", "publish", "checkpoint", "submit", "complete", "consume") if k not in kinds]
    if missing:                              # the hooks did not see the transfer: searching would mean nothing
        result["error"] = f"the baseline recorded no {', '.join(missing)} events; hooks: {result['hooks']}"
        result["verdict"] = "hooks_recorded_nothing"
        return 2
    # the clean pulls themselves must pass: on the fixed build a decline, a lease-check failure or a failed transfer
    # during the baseline would mean the verifier refuses clean pulls too, so every later "recovery" would only be
    # the recompute path agreeing with a reference that was itself recomputed. Not a usable oracle.
    bc = result["baseline_consumer_signals"]
    result["baseline_declines"] = len(declines(loglines(logs["D"])))
    result["baseline_pulls_clean"] = (result["baseline_declines"] == 0
                                      and not _rose(bc, "vllm:nixl_num_lease_check_failures")
                                      and not _rose(bc, "vllm:nixl_num_failed_transfers"))
    result["baseline_ok"] = result["baseline_oracle_ok"] and result["baseline_pulls_clean"]
    if not result["baseline_ok"]:           # stop before the paid search: nothing after this could be judged
        result["verdict"] = ("inconclusive_bad_baseline" if result["baseline_pulls_clean"]
                             else "inconclusive_clean_pulls_declined")
        return 5

    base_evs = events(evdir)
    cands = rank(candidates(base_evs, symbolic=True), [base_evs])

    def _lead(c):
        # the on-the-nose #61229 trigger first: hold the read until the PUBLISHED source block (desc_block in the
        # spec, the one the prefill side leased to this read) is reused by another owner or freed.
        src = "desc_block" in json.dumps(c.spec, default=str)
        return (0 if src and c.kind in ("realloc", "free") else 1 if c.kind in ("realloc", "free") else 2)
    cands = [rewritten_candidate(0)] + sorted(cands, key=_lead)    # the experiment's precise condition first
    result["checkpoints_in_baseline"] = sorted({e["args"]["point"] for e in base_evs if e["kind"] == "checkpoint"})
    result["candidates_total"] = len(cands)
    hold_s = a.lease + 10
    result["design"] = ("chosen for this experiment, not by the search: the lease length, the load on the prefill "
                        "side after the lease, the canary prompts, and a hold timeout longer than the lease. The "
                        "search chooses where to hold and when the hold ends. Controls: the same load with no hold, "
                        "and the compared hold with a stall under the lease.")

    def load_p():
        """After the lease: one tiny request, then ordinary requests with DISTINCT canaries. The prefill engine keeps
        stepping on its own and frees X's blocks when the lease expires; freed blocks are reused newest first, so
        these requests take them over and write their own KV there. Returns (errors, {canary: request label}) so a
        foreign word in X's answer names its source."""
        errs, used = [], {}
        try:
            post("http://localhost:8100/v1/completions",
                 {"model": a.model, "prompt": "Hi", "max_tokens": 1, "temperature": 0})
        except Exception as e:  # noqa: BLE001
            errs.append(str(e))
        time.sleep(0.3)
        for i in range(a.others):
            w = CANARIES[i % len(CANARIES)]
            used[w] = f"load request {i}"
            try:
                post("http://localhost:8100/v1/completions",
                     {"model": a.model, "prompt": canary_prompt(w, 3), "max_tokens": 32, "temperature": 0})
            except Exception as e:  # noqa: BLE001
                errs.append(str(e))
        return errs, used

    def one_run(c, timeout=None, label=None):
        started_clean = wait_idle()          # True idle, False busy throughout (run not clean), None unknown
        xrid = uuid.uuid4().hex             # X's decode request id, set in advance so its decline line is X's
        n0 = len(events(evdir))
        d0 = len(loglines(logs["D"]))
        prod0, cons0 = producer_signals(logs["P"]), consumer_signals()
        kvx = pd_complete(a.model, PROMPT_X, prefill_only=True)
        xp = kvx.get("remote_request_id")
        spec = c.hold_spec(timeout or hold_s) if c is not None else None
        if spec is not None and xp:         # name X's prefill in the condition (a placeholder in the saved spec)
            spec = json.loads(json.dumps(spec).replace(X_PREFILL, xp))
        with open(holds, "w") as f:         # written before the decode request, so its read is the one held
            json.dump([spec] if spec is not None else [], f)
        out = {}

        def run_decode():
            try:
                out["x"], out["x_error"], out["x_id"] = decode(a.model, PROMPT_X, kvx, request_id=xrid)
            except Exception as e:  # noqa: BLE001  (a thread error is not a wrong answer; surface it, don't drop it)
                out["x_exc"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=run_decode, daemon=True)
        t.start()
        time.sleep(a.lease + 1)
        errs, used = load_p()
        t.join(hold_s + 120)
        hung = t.is_alive()                 # the request never finished: do not proceed as if it had
        with open(holds, "w") as f:
            json.dump([], f)
        time.sleep(0.5)
        evs = events(evdir)
        led = merge_events(evs, track_content=False)
        raw = [v for v in led.violations if v.step >= n0]
        rules = rules_for_request(evs, led.violations, xrid)
        actionable, dispositions = dispose_notification_overlaps(evs, raw, rules)
        new = [f"{v.rule} {v.message}" for v in actionable]
        rel = [e.get("text", "") for e in evs[n0:] if e["kind"] in ("released", "note")
               and ("released" in e.get("text", "") or "hold condition" in e.get("text", ""))]
        notes = [e.get("text", "") for e in evs[n0:]
                 if e["kind"] == "note" and "lease expired" in e.get("text", "")]
        prod = delta(prod0, producer_signals(logs["P"]))
        cons = delta(cons0, consumer_signals())
        new_dlines = loglines(logs["D"])[d0:]
        xids = [xrid, out.get("x_id"), kvx.get("remote_request_id"), kvx.get("request_id")]
        parsed = declines(new_dlines)
        # Attribution: by X's request id, which this experiment assigns in advance and vLLM carries into the
        # decline line (cmpl-<id>-0-<suffix>), so it holds even when X fails and its error body carries no id. A
        # decline that does not carry X's id belongs to some other request: it is reported as unattributed and
        # makes the run unknown, never credited to the fix.
        dec = match_decline(parsed, xids)
        attribution = "request_id" if dec is not None else ("unattributed" if parsed else None)
        x_finished = ("x" in out) and not hung and ("x_exc" not in out)
        oc = classify(new, out.get("x"), base_x, kvx.get("remote_request_id"), prod, cons, notes, pool=CANARIES,
                      x_error=out.get("x_error"), x_finished=x_finished,
                      rejected=dec is not None, reject_reason=(dec or {}).get("reason"),
                      reject_kind=(dec or {}).get("kind"), reject_cause=(dec or {}).get("cause"),
                      reject_unknown=(attribution == "unattributed"), local_answer=local_x)
        # ordering evidence for the #61229 scenario, assembled from this run's own events + decode log
        order = {"x_published": any(e["kind"] == "publish" for e in evs[n0:]),
                 "held": any("hold condition" in r for r in rel),
                 "lease_expired": oc["lease_expired"],
                 "source_block_reused_or_freed": bool([v for v in new if v.startswith(("R1", "R2")) and "read-" in v]),
                 "decode_resumed": any(e["kind"] == "submit" for e in evs[n0:]),
                 "rejection_or_output": (dec["kind"] if dec else oc["answer"])}
        order["trigger_demonstrated"] = bool(oc["exposure"] and order["lease_expired"]
                                             and order["source_block_reused_or_freed"])
        # the safety rules, judged from the recorded facts alone (never from X's answer), for X's decode attempt
        lease_rule = next((f for f in rules if f["rule"] == "lease_valid_at_use"), None)
        trial = {"label": label or (c.describe() if c is not None else "no hold"),
                "spec": c.hold_spec(timeout or hold_s) if c is not None else None,
                "spec_applied": spec, "x_prefill_id": xp,
                "source": c.source if c is not None else None,
                "producer_signals": prod, "consumer_signals": cons, "canaries_in_load": used,
                "decline_line": dec, "decline_attribution": attribution, "declines_in_window": len(parsed),
                "x_request_id": xrid,
                "reached": any("condition reached" in r for r in rel), "hold_log": rel,
                "hold_outcome": hold_outcome(rel) if c is not None else "not_held", "stall_s": stall_seconds(rel),
                "intervention": (INTERVENTION.get(c.hold_spec(timeout or hold_s).get("point"))
                                 if c is not None else None),
                "violations": [f"{v.rule} {v.message}" for v in raw], "actionable_violations": new,
                "ledger_overlap_dispositions": dispositions,
                "x": out.get("x"), "x_error": out.get("x_error"), "x_exc": out.get("x_exc"),
                "x_finished": x_finished, "hung": hung, "started_clean": started_clean,
                "other_errors": errs, "ordering": order, "outcome": oc,
                "rules": rules, "lease_rule": lease_rule["verdict"] if lease_rule else "unknown"}
        trial["terminal_observation"] = terminal_outcome(lease_rule or {}, trial, refs)
        return trial

    def save():
        """Write what is known so far (atomically), so a run cut off by a timeout still leaves its evidence."""
        tmp = os.path.join(a.out, "RESULT.json.tmp")
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, os.path.join(a.out, "RESULT.json"))

    controls, discovery, tried = [], [], []
    result.update(controls=controls, discovery=discovery, tried=tried)
    stop = False

    def run(into, c, **kw):
        nonlocal stop
        r = one_run(c, **kw)
        into.append(r)
        save()
        if r["hung"]:                        # a request still in flight would contaminate every later run
            stop = True
            result["stopped_after_hung_run"] = r["label"]
        return r

    # control A: the same load with no hold. It must finish, say the right answer and break no rule.
    run(controls, None, label="control: the same load, no hold")

    pinned = None
    if stop:
        pass
    elif a.mode == "search":                 # exploration: every candidate once, no paired comparison
        for c in cands[:a.max_candidates]:
            if stop:
                break
            run(tried, c)
    else:
        if a.candidate:                      # replay the exact schedule another run (the other image) saved
            pinned = SavedCandidate(load_candidate(a.candidate))
            result["pinned_candidate_source"] = a.candidate
            result["pinned_candidate_in_this_baseline"] = any(      # audit: this baseline yields the same schedule
                _spec_key(c.hold_spec(pinned.timeout)) == _spec_key(pinned.spec) for c in cands)
        else:                                # discover it: ranked candidates until one actually corrupts X's answer
            for c in cands[:a.max_candidates]:
                if stop:
                    break
                r = run(discovery, c)
                if not r["outcome"]["harm"]:
                    continue
                # one garbled answer could be a race the read sometimes wins: confirm it before pinning
                conf = [r]
                while len(conf) < a.confirm and not stop:
                    conf.append(run(discovery, c))
                k = sum(x["outcome"]["harm"] for x in conf)
                result.setdefault("discovery_confirmations", []).append(
                    {"label": c.describe(), "harm": k, "of": len(conf)})
                if 3 * k >= 2 * len(conf):
                    pinned = SavedCandidate.from_candidate(c, hold_s, discovered=r)
                    break
        if pinned is not None and not stop:
            result["candidate"] = pinned.record()
            # control B: the same hold point with a stall shorter than the lease; it must be clean on either image
            run(controls, pinned, timeout=max(1, a.lease // 2),
                label=f"control: {pinned.describe()}, stall under the lease")
            for _ in range(a.reps):          # the paired arm: the same schedule, the same number of times
                if stop:
                    break
                run(tried, pinned, timeout=pinned.timeout)

    # aggregates over the paired arm (the replays; every run in search mode), with unknowns kept as unknown
    trig = [t for t in tried if t["ordering"]["trigger_demonstrated"]]
    harm_trials = [t for t in tried if t["outcome"]["harm"]]
    reject_trials = [t for t in tried if t["outcome"]["rejected"]]
    recovered = [t for t in tried if t["outcome"]["terminal"] == "recovered"]
    unresolved = [t for t in controls + discovery + tried if t["hung"] or t["x_exc"] or not t["x_finished"]]
    # a replay exercised the targeted ordering only if the trigger was demonstrated AND its hold was released by
    # its condition (the block's next owner had written it), not by the timeout
    exercised = [t for t in trig if t["hold_outcome"] == "condition_reached"]
    every = controls + discovery + tried
    unclean = [t for t in every if t["started_clean"] is False]

    def n_term(k):
        return sum(t["outcome"]["terminal"] == k for t in tried)

    result["runs_total"] = len(tried)
    result["runs_planned"] = a.reps if (a.mode == "compare" and pinned is not None) else len(tried)
    result["runs_trigger_demonstrated"] = len(trig)
    result["runs_with_exposure"] = sum(t["outcome"]["exposure"] for t in tried)
    result["runs_with_harm"] = len(harm_trials)
    result["runs_rejected_by_fix"] = len(reject_trials)       # a decline line matched to X, of any cause
    result["runs_caught_stale_lease"] = sum(t["outcome"]["caught_stale_lease"] for t in tried)
    result["runs_recovered"] = len(recovered)
    result["runs_failed_closed"] = n_term("failed_closed")
    result["runs_declined_before_read"] = n_term("declined_before_read")
    result["runs_declined_unverifiable"] = n_term("declined_unverifiable")
    result["runs_declined_other_reason"] = n_term("declined_other_reason")
    result["runs_bad_answer_after_decline"] = sum(t["outcome"]["bad_answer_after_decline"] for t in tried)
    result["runs_decline_unattributed"] = sum(t["outcome"]["reject_unknown"] for t in tried)
    result["runs_answer_correct"] = sum(t["outcome"]["answer"] in ("correct", "correct_recompute") for t in tried)
    result["runs_answer_exact"] = sum(t["outcome"]["answer"] == "correct" for t in tried)
    result["runs_answer_recompute_reference"] = sum(t["outcome"]["answer"] == "correct_recompute" for t in tried)
    result["runs_answer_differs"] = sum(t["outcome"]["answer"] == "differs" for t in tried)
    result["runs_answer_foreign"] = sum(t["outcome"]["answer"] == "foreign" for t in tried)
    result["runs_unresolved"] = len(unresolved)
    result["runs_unclean_start"] = len(unclean)
    result["runs_idle_unknown"] = sum(t["started_clean"] is None for t in every)
    # the fixed build's per-replay accounting: which replays exercised the trigger, and how each one ended
    result["runs_exercised"] = len(exercised)
    result["exercised_terminals"] = {}
    for t in exercised:
        k = t["outcome"]["terminal"]
        result["exercised_terminals"][k] = result["exercised_terminals"].get(k, 0) + 1
    result["unexercised_terminals"] = {}
    ex_ids = {id(t) for t in exercised}
    for t in tried:
        if id(t) in ex_ids:
            continue
        k = t["outcome"]["terminal"]
        result["unexercised_terminals"][k] = result["unexercised_terminals"].get(k, 0) + 1
    # a verified bad answer: any foreign leak (a distinctive canary is incontrovertible), or a garbled answer that
    # reproduces in at least two replays (one alone could be nondeterminism).
    result["harm_present"] = result["runs_answer_foreign"] >= 1 or len(harm_trials) >= 2
    result["hold_outcomes"] = {k: sum(t["hold_outcome"] == k for t in tried)
                               for k in ("condition_reached", "expired", "not_held")}
    result["runs_harm_after_expired_hold"] = sum(t["outcome"]["harm"] and t["hold_outcome"] == "expired"
                                                for t in tried)
    # what the ledger saw against what the answers alone show, over every held run here
    held = [t for t in discovery + tried if t["hold_outcome"] != "not_held"]
    result["held_runs"] = len(held)
    result["held_runs_ledger_flagged_stale_read"] = sum(t["outcome"]["exposure"] for t in held)
    result["held_runs_answer_visibly_wrong"] = sum(t["outcome"]["harm"] for t in held)
    result["discovered_harm_once"] = any(d["outcome"]["harm"] for d in discovery)
    # Evidence-only verdicts have their own validation bar; preserve the legacy oracle.
    result["lease_rule_verdicts"] = {}
    for t in every:
        result["lease_rule_verdicts"][t["lease_rule"]] = result["lease_rule_verdicts"].get(t["lease_rule"], 0) + 1
    result["evaluator_validation"] = evaluator_summary(discovery + tried, controls, a.variant)
    result["lease_notes"] = [e.get("text") for e in events(evdir) if e["kind"] == "note"
                             and "lease expired" in e.get("text", "")][-20:]

    # controls must be clean, finished, and return EXACTLY the prefill/decode reference; the oracle must be valid
    # controls must be clean, finished, return EXACTLY the prefill/decode reference, and show no decline, no
    # lease-check failure and no failed transfer (a fix that refuses clean pulls must not pass as a clean control)
    controls_ok = all(c["x_finished"] and not c["hung"] and not c["actionable_violations"]
                      and c["outcome"]["answer"] == "correct" and c["declines_in_window"] == 0
                      and not _rose(c["consumer_signals"], "vllm:nixl_num_lease_check_failures")
                      and not _rose(c["consumer_signals"], "vllm:nixl_num_failed_transfers") for c in controls)
    result["controls_clean"] = controls_ok
    baseline_ok = result["baseline_ok"]

    code = _verdict(result, a, controls_ok, baseline_ok, bool(trig), bool(unresolved),
                    no_candidate=(a.mode == "compare" and pinned is None), any_unclean=bool(unclean))
    # the fixed runs replay this schedule only if the stock run confirmed that it breaks the answer
    if pinned is not None and not a.candidate and result.get("verdict") == "harm_reproduced":
        save_candidate(os.path.join(a.out, "candidate.json"), result["candidate"])
        result["candidate_saved"] = True
    return code


def _verdict(result, a, controls_ok, baseline_ok, trigger, any_unresolved, no_candidate=False,
             any_unclean=False) -> int:
    """Translate the facts into an exit code and a plain-words verdict. Exit 0 is reserved for a demonstrated
    trigger with no harm; for the fixed variant it also needs every planned replay finished, at least --confirm of
    them exercising the trigger, every exercised replay ending exactly as the configured kv_load_failure_policy
    promises (recovered under recompute, failed_closed under fail), and every other replay clean. Anything mixed is
    inconclusive, with the counts kept in the result."""
    if not baseline_ok:
        result["verdict"] = "inconclusive_bad_baseline"
        return 5
    if any_unresolved:                      # a run (a control included) that did not finish is unknown, not broken
        result["verdict"] = "inconclusive_unresolved"
        return 5
    if not controls_ok:
        result["verdict"] = "control_broke"
        return 3
    if any_unclean:
        result["verdict"] = "inconclusive_unclean_start"       # a run began while the servers still had work
        return 5
    if no_candidate:
        result["verdict"] = "inconclusive_no_reproducing_candidate"
        return 5
    if result["harm_present"]:
        result["verdict"] = ("harm_reproduced" if a.variant == "unpatched"
                             else "fix_failed_harm_present")
        return 1
    if not trigger:
        result["verdict"] = "inconclusive_no_trigger"
        return 5
    # trigger demonstrated, no confirmed harm
    if a.variant == "fixed":
        policy = getattr(a, "kv_load_failure_policy", "recompute")
        expected = "failed_closed" if policy == "fail" else "recovered"
        result["fixed_expected_terminal"] = expected
        ex = result.get("exercised_terminals", {})
        n_ex = sum(ex.values())
        need = max(1, min(getattr(a, "confirm", 3), result.get("runs_planned", 0) or 1))
        others = result.get("unexercised_terminals", {})
        others_ok = all(k in ("clean", expected) for k in others)
        if result.get("runs_total", 0) < result.get("runs_planned", 0):
            result["verdict"] = "inconclusive_incomplete_replays"
        elif ex.get("declined_before_read"):
            result["verdict"] = "inconclusive_declined_before_read"    # the lease was never checked after a read
        elif ex.get("declined_unverifiable"):
            result["verdict"] = "inconclusive_lease_check_could_not_run"
        elif ex.get("inconclusive_decline_unattributed"):
            result["verdict"] = "inconclusive_decline_unattributed"
        elif n_ex < need:
            result["verdict"] = "inconclusive_too_few_replays_exercised"
        elif ex.get(expected, 0) != n_ex or not others_ok:
            result["verdict"] = "inconclusive_mixed_outcomes"
        else:
            result["verdict"] = "recovery_demonstrated" if expected == "recovered" else "fail_closed_as_configured"
            return 0
        return 5
    elif result.get("discovered_harm_once"):
        result["verdict"] = "harm_found_in_discovery_not_reproduced_in_replays"
        return 5
    else:
        result["verdict"] = "no_corruption_despite_trigger"
    return 0


if __name__ == "__main__":
    sys.exit(main())
