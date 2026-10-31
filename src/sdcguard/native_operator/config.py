"""One checksum source profile. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .workload import MODEL, MODEL_REVISION, REVISION

BUNDLE = str(Path(__file__).resolve().parent / "profile_data")
ENGINE_ROOT = "/opt/vllm-checksum"
SNAPSHOT = os.environ.get("SDCGUARD_OPERATOR_SNAPSHOT",
    "/cache/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/" + MODEL_REVISION)
PORTS = {"prefill": 8100, "decode": 8200}
JOB_SECONDS = 1800
CLIENT_SECONDS = 1000
STARTUP_SECONDS = 600
ENGINE_SHUTDOWN_SECONDS = 30
CLEANUP_SECONDS = 45
BLOCK_SIZE = 16
MAX_MODEL_LEN = 2048
MAX_SEQS = 16
MIN_GPU_BLOCKS = 1568


def server_command(role, *, gpu_blocks=None):
    if role not in PORTS:
        raise ValueError("unsupported ordinary engine role")
    if gpu_blocks is not None and (type(gpu_blocks) is not int or gpu_blocks != 8192):
        raise ValueError("only the 8192-block paired capacity is supported")
    transfer = dict(kv_connector="NixlConnector", engine_id=role,
        kv_role="kv_producer" if role == "prefill" else "kv_consumer",
        enable_kv_checksum=True, kv_load_failure_policy="fail",
        kv_connector_extra_config=dict(bidirectional_kv_xfer=False, backends=["UCX"]))
    command = ["vllm", "serve", SNAPSHOT, "--tokenizer", SNAPSHOT,
        "--served-model-name", MODEL, "--revision", MODEL_REVISION,
        "--tokenizer-revision", MODEL_REVISION, "--host", "127.0.0.1",
        "--port", str(PORTS[role]), "--dtype", "bfloat16", "--enforce-eager",
        "--shutdown-timeout", str(ENGINE_SHUTDOWN_SECONDS), "--api-server-count", "1",
        "--distributed-executor-backend", "uni", "--tensor-parallel-size", "1",
        "--pipeline-parallel-size", "1", "--data-parallel-size", "1",
        "--decode-context-parallel-size", "1", "--prefill-context-parallel-size", "1",
        "--block-size", str(BLOCK_SIZE), "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", str(MAX_SEQS), "--max-num-batched-tokens", "8192",
        "--gpu-memory-utilization", "0.2", "--kv-cache-dtype", "auto",
        "--enable-prefix-caching", "--enable-chunked-prefill", "--no-async-scheduling",
        "--kv-transfer-config", json.dumps(transfer, sort_keys=True, separators=(",", ":"))]
    if gpu_blocks is not None:
        command.extend(["--num-gpu-blocks-override", str(gpu_blocks)])
    return command


def environment(parent, *, out, run_id, packet_sha256, arm_id, engine_session_id, role, enabled, revision=REVISION):
    if role not in PORTS or type(enabled) is not bool:
        raise ValueError("unsupported observer role or mode")
    from .transport import clean_environment
    from ..checksum_profiles import get
    get(revision)
    clean = {name: value for name, value in clean_environment(parent).items()
        if not name.startswith(("VLLM_", "SDCGUARD", "OWNER_", "HF_", "HUGGINGFACE_", "TRANSFORMERS_"))
        and name not in ("PYTHONPATH", "PYTHONHOME")}
    clean.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONSAFEPATH="1",
        VLLM_PLUGINS="sdcguard_coverage", VLLM_USE_V2_MODEL_RUNNER="1", VLLM_USE_RUST_FRONTEND="0",
        VLLM_USE_FLASHINFER_SAMPLER="0", UCX_RCACHE_MAX_UNRELEASED="1024",
        VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="0",
        HF_HOME="/cache/hf", HF_HUB_CACHE="/cache/hf/hub", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", SDCGUARD="0",
        OWNER_CHECKSUM_DIR=str(Path(out)), OWNER_CHECKSUM_RUN_ID=run_id,
        OWNER_CHECKSUM_PACKET_SHA256=packet_sha256, OWNER_CHECKSUM_ARM=arm_id,
        OWNER_CHECKSUM_ENGINE_SESSION=engine_session_id, OWNER_CHECKSUM_ROLE=role,
        SDCGUARD_OPERATOR_SNAPSHOT=SNAPSHOT,
        SDCGUARD_OPERATOR_REVISION=revision,
        OWNER_CHECKSUM_ENABLED="1" if enabled else "0",
        VLLM_NIXL_SIDE_CHANNEL_PORT="5559" if role == "prefill" else "5659",
        VLLM_PORT="8300" if role == "prefill" else "8400")
    return clean
