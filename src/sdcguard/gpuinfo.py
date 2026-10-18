"""What the GPU says about itself: identity, health counters, clocks, throttling, and datasheet peaks.

Everything here is best effort and read-only: nvidia-smi when it exists, nothing otherwise. A field
the driver doesn't know (older drivers, consumer cards) comes back as None instead of failing.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Optional

# Datasheet peaks, dense (no sparsity), checked on nvidia.com on Oct 18, 2026. Matmul TFLOPS are for the
# dtype the check uses (bf16; fp16 on T4, which has no bf16 tensor cores). A healthy GPU running big
# matmuls usually lands well below the datasheet peak (clocks drop to stay inside the power limit), so
# the useful comparison is with other GPUs of the same model (the public board), not with 100%.
#   (substrings that must all be in the lowercased name, substrings that must not be, tflops, TB/s, source)
SPECS = [
    (("h100", "nvl"), (), 835.5, 3.9, "nvidia.com/en-us/data-center/h100 (1,671 sparse / 2)"),
    (("h100", "hbm3"), (), 989.5, 3.35, "nvidia.com/en-us/data-center/h100 (H100 SXM, 1,979 sparse / 2)"),
    (("h200", "nvl"), ("gh200",), 835.5, 4.8, "nvidia.com/en-us/data-center/h200 (1,671 sparse / 2)"),
    (("h200",), ("nvl", "gh200"), 989.5, 4.8, "nvidia.com/en-us/data-center/h200 (H200 SXM, 1,979 sparse / 2)"),
    (("b200",), (), 2250.0, 8.0, "nvidia.com/en-us/data-center/hgx (36 PF sparse per 8 GPUs / 2) and dgx-b200 "
                                 "(64 TB/s per 8 GPUs)"),
    (("a100", "80gb", "pcie"), (), 312.0, 1.935, "nvidia.com/en-us/data-center/a100"),
    (("a100", "80gb"), ("pcie",), 312.0, 2.039, "nvidia.com/en-us/data-center/a100 (SXM bandwidth)"),
    (("a100",), (), 312.0, None, "nvidia.com/en-us/data-center/a100 (bf16; bandwidth listed for 80GB only)"),
    (("l40s",), (), 362.05, 0.864, "nvidia.com/en-us/data-center/l40s"),
    (("nvidia l4",), (), 121.0, 0.300, "nvidia.com/en-us/data-center/l4 (242 sparse / 2)"),
    (("nvidia a10",), ("a10g", "a100"), 125.0, 0.600, "nvidia.com/en-us/data-center/products/a10-gpu"),
    (("tesla t4",), (), 65.0, 0.320, "nvidia.com/en-us/data-center/tesla-t4 (FP16 mixed precision)"),
]


def datasheet(name: Optional[str]) -> dict:
    """{'matmul_tflops', 'memory_tbs', 'source'} for a GPU name as PyTorch or nvidia-smi prints it, or {}."""
    n = " ".join((name or "").lower().split())
    for need, avoid, tf, bw, src in SPECS:
        if need in (("nvidia l4",), ("nvidia a10",)):       # short names: whole-name match only
            if n != need[0]:
                continue
        elif not all(s in n for s in need) or any(s in n for s in avoid):
            continue
        return {"matmul_tflops": tf, "memory_tbs": bw, "source": src}
    return {}


# --- nvidia-smi ---------------------------------------------------------------------------------------------

STATIC = ["name", "uuid", "serial", "vbios_version", "driver_version", "pci.bus_id", "memory.total",
          "power.limit", "clocks.max.sm", "clocks.max.memory", "pcie.link.gen.max", "pcie.link.width.max",
          "ecc.mode.current", "compute_cap"]
HEALTH = ["ecc.errors.uncorrected.volatile.total", "ecc.errors.corrected.volatile.total",
          "ecc.errors.uncorrected.aggregate.total", "retired_pages.pending", "remapped_rows.pending",
          "remapped_rows.failure", "remapped_rows.uncorrectable", "temperature.gpu"]
LIVE = ["temperature.gpu", "temperature.memory", "power.draw", "clocks.sm", "clocks.mem", "utilization.gpu",
        "pcie.link.gen.current", "pcie.link.width.current", "clocks_event_reasons.active"]
_OLD_NAMES = {"clocks_event_reasons.active": "clocks_throttle_reasons.active"}
_valid: dict = {}


def _run(args: list, timeout: float = 15) -> Optional[str]:
    try:
        p = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def _clean(v: str):
    v = v.strip()
    if not v or v.startswith("[") or v in ("N/A", "Not Supported"):
        return None
    return v


def _fields_ok(index, fields: list) -> list:
    """The subset of fields this driver accepts (an unknown field fails the whole query), cached."""
    key = tuple(fields)
    if key in _valid:
        return _valid[key]
    if _run([f"--id={index}", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]) is not None:
        _valid[key] = list(fields)
        return _valid[key]
    ok = []
    for f in fields:
        for name in (f, _OLD_NAMES.get(f)):
            if name and _run([f"--id={index}", f"--query-gpu={name}", "--format=csv,noheader,nounits"]) is not None:
                ok.append(name)
                break
    _valid[key] = ok
    return ok


def query(index, fields: list) -> dict:
    """{field: value or None} for one GPU: `index` is its UUID (preferred: nvidia-smi numbers GPUs in PCI
    order, CUDA fastest first) or nvidia-smi index. {} when nvidia-smi isn't there."""
    if index is None:
        return {}
    ok = _fields_ok(index, fields)
    if not ok:
        return {}
    out = _run([f"--id={index}", "--query-gpu=" + ",".join(ok), "--format=csv,noheader,nounits"])
    if out is None:
        return {}
    rows = [ln for ln in out.splitlines() if ln.strip()]
    if not rows:
        return {}
    vals = [_clean(v) for v in rows[-1].split(",")]            # (a warning line can come first)
    res = {f: None for f in fields}
    for name, v in zip(ok, vals):
        canon = next((k for k, old in _OLD_NAMES.items() if old == name), name)
        res[canon] = v
    return res


REMAP = ["remapped_rows.pending", "remapped_rows.failure", "remapped_rows.uncorrectable"]


def health(index) -> dict:
    """ECC, retired-page and row-remapping counters. Row remapping is also asked for through
    --query-remapped-rows (by UUID) when --query-gpu doesn't know those fields."""
    h = query(index, HEALTH)
    if h and all(h.get(k) is None for k in REMAP) and isinstance(index, str) and index.startswith("GPU-"):
        out = _run(["--query-remapped-rows=gpu_uuid," + ",".join(REMAP), "--format=csv,noheader,nounits"])
        for line in (out or "").splitlines():
            vals = [_clean(v) for v in line.split(",")]
            if vals and vals[0] == index and len(vals) == len(REMAP) + 1:
                h.update(dict(zip(REMAP, vals[1:])))
    return h


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


THROTTLE_BITS = {   # NVML clocks event reasons
    0x1: "gpu_idle", 0x2: "applications_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x10: "sync_boost",
    0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake", 0x100: "display_clock",
}
BAD_THROTTLE = {"hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake"}


def throttle_reasons(v) -> list:
    try:
        bits = int(str(v), 16) if str(v).lower().startswith("0x") else int(v)
    except (TypeError, ValueError):
        return []
    return [name for bit, name in THROTTLE_BITS.items() if bits & bit]


class Sampler:
    """Samples temperature, power, clocks, PCIe link and throttle reasons every `every` seconds in a thread."""

    def __init__(self, index, every: float = 1.0):
        self.index, self.every = index, every
        self.samples: list = []
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self) -> "Sampler":
        if self.index is not None and _run(["-L"]) is not None:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            q = query(self.index, LIVE)
            if q:
                q["t"] = time.time()
                self.samples.append(q)
            self._stop.wait(self.every)

    def stop(self) -> "Sampler":
        self._stop.set()
        if self._t:
            self._t.join(timeout=20)
        return self

    def summary(self) -> dict:
        s = self.samples
        if not s:
            return {}
        busy = [x for x in s if (_num(x.get("utilization.gpu")) or 0) >= 50] or s

        def mx(k):
            v = [_num(x.get(k)) for x in s if _num(x.get(k)) is not None]
            return max(v) if v else None

        def med(k, rows):
            v = sorted(_num(x.get(k)) for x in rows if _num(x.get(k)) is not None)
            return v[len(v) // 2] if v else None

        reasons: dict = {}
        for x in busy:
            for r in throttle_reasons(x.get("clocks_event_reasons.active")):
                reasons[r] = reasons.get(r, 0) + 1
        gens = [int(_num(x.get("pcie.link.gen.current"))) for x in busy if _num(x.get("pcie.link.gen.current"))]
        widths = [int(_num(x.get("pcie.link.width.current"))) for x in busy if _num(x.get("pcie.link.width.current"))]
        return {"samples": len(s), "busy_samples": len(busy), "max_temp_c": mx("temperature.gpu"),
                "max_memory_temp_c": mx("temperature.memory"), "max_power_w": mx("power.draw"),
                "median_sm_clock_mhz_busy": med("clocks.sm", busy), "median_mem_clock_mhz_busy": med("clocks.mem", busy),
                "throttle_share_busy": {k: round(v / len(busy), 3) for k, v in sorted(reasons.items())},
                "pcie_gen_busy": max(gens) if gens else None, "pcie_width_busy": min(widths) if widths else None}


# --- where is this running? ---------------------------------------------------------------------------------

def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


_CLOUDS = {"AWS": "AWS", "GCP": "GCP", "AZURE": "Azure", "OCI": "OCI"}


def cloud_name(v) -> Optional[str]:
    """Modal's MODAL_CLOUD_PROVIDER ("CLOUD_PROVIDER_AZURE") as a name ("Azure"); None when unspecified."""
    key = str(v or "").upper().replace("CLOUD_PROVIDER_", "").strip()
    return _CLOUDS.get(key, key.title() if key and key != "UNSPECIFIED" else None)


def detect_provider() -> dict:
    """Best guess at the GPU provider, from environment variables and DMI strings the platforms set.
    Conservative: returns {'name': None} when unsure. `sdcguard check --provider NAME` overrides it."""
    env = os.environ
    found = None
    if any(k.startswith("MODAL_") for k in env) and env.get("MODAL_TASK_ID"):
        found = ("Modal", "MODAL_TASK_ID")
    elif env.get("RUNPOD_POD_ID"):
        found = ("RunPod", "RUNPOD_POD_ID")
    elif env.get("VAST_CONTAINERLABEL") or env.get("CONTAINER_ID", "").startswith("C."):
        found = ("Vast.ai", "VAST_CONTAINERLABEL")
    elif env.get("COLAB_RELEASE_TAG") or env.get("COLAB_GPU"):
        found = ("Google Colab", "COLAB_RELEASE_TAG")
    elif env.get("KAGGLE_KERNEL_RUN_TYPE"):
        found = ("Kaggle", "KAGGLE_KERNEL_RUN_TYPE")
    elif env.get("LIGHTNING_CLOUD_PROJECT_ID"):
        found = ("Lightning AI", "LIGHTNING_CLOUD_PROJECT_ID")
    elif env.get("PAPERSPACE_METRIC_WORKLOAD_ID") or env.get("PAPERSPACE_CLUSTER_ID"):
        found = ("Paperspace", "PAPERSPACE_*")
    if found is None:
        vendor = _read("/sys/class/dmi/id/sys_vendor")
        product = _read("/sys/class/dmi/id/product_name")
        asset = _read("/sys/class/dmi/id/chassis_asset_tag")
        if vendor == "Amazon EC2" or product.startswith("Amazon EC2"):
            found = ("AWS", "DMI sys_vendor")
        elif product == "Google Compute Engine":
            found = ("Google Cloud", "DMI product_name")
        elif asset == "7783-7084-3265-9085-8269-3286-77":
            found = ("Azure", "DMI chassis_asset_tag")
        elif asset == "OracleCloud.com":
            found = ("Oracle Cloud", "DMI chassis_asset_tag")
    out = {"name": found[0] if found else None, "detected_from": found[1] if found else None}
    if found and found[0] == "Modal":
        out["underlying_cloud"] = cloud_name(env.get("MODAL_CLOUD_PROVIDER"))
        out["region"] = env.get("MODAL_REGION") or None
    return out
