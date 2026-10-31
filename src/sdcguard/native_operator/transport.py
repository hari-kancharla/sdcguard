"""Explicit same-host native transport profile. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import ctypes
import functools
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import types

API_SHA256 = "79ac70eed826672f09a21df7e71c8646654e761961c9783ecd8e145c0a7149a7"
UCP_NAME = "libucp-b39b3ed9.so.0.0.0"
UCP_SHA256 = "0770acf18bb7617b5084ae29186dba097dce51f04a774bba0a2e75a5586a37bf"
PARAMETERS = {"ucx_error_handling_mode": "none"}
SCHEMA = "sdcguard.operator-native-transport-readiness.v1"
RECEIPT_SCHEMA = "sdcguard.owner-checksum-transport.v1"
CASE_SCHEMA = "sdcguard.owner-checksum-transport-case.v1"
SCOPE = ("Native CPU endpoint initialization only, with no model, engine, memory registration, "
         "transfer or GPU allocation. CUDA IPC, peer-failure recovery and cross-host correctness remain unknown.")
_INSTALLATIONS = {}


def profile():
    return dict(name="same-host-ucx-no-peer-handler", backend="UCX", parameters=dict(PARAMETERS),
        nixl_version="1.5.0", module="nixl_cu13._api", api_sha256=API_SHA256,
        ucx_version=[1, 23, 0], ucx_revision="65e342a82dd2609a28c8c2b8c45ed7fe2cb33fa0",
        peer_failure_detection="disabled by explicit configuration",
        native_checksum_settings="unchanged", native_lease_settings="unchanged",
        unknown=["CUDA IPC", "peer-failure recovery", "cross-host correctness", "ownership", "physical reads"])


def clean_environment(parent):
    """Use one explicit transport profile instead of inherited overrides."""
    return {name: value for name, value in parent.items()
            if not name.startswith(("UCX_", "NIXL_", "SDCGUARD_NIXL_"))}


def _shape(code):
    return (code.co_code, tuple(_shape(x) if isinstance(x, types.CodeType) else x for x in code.co_consts),
        code.co_names, code.co_varnames, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_freevars, code.co_cellvars, code.co_flags,
        getattr(code, "co_exceptiontable", b""))


def _find(code, qualname, prefix=""):
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            name = prefix + item.co_name
            if name == qualname:
                return item
            found = _find(item, qualname, name + ".")
            if found is not None:
                return found
    return None


def _guard_api(api):
    if api.__name__ != "nixl_cu13._api":
        raise ValueError("the transport profile requires the exact CUDA 13 NIXL namespace")
    path = Path(api.__file__)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("the NIXL API source is unavailable or unsupported")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != API_SHA256:
        raise ValueError("the NIXL API source differs from the pinned CUDA 13 wheel")
    original = api.nixl_agent.__dict__.get("create_backend")
    if (not inspect.isfunction(original) or original.__closure__
            or original.__qualname__ != "nixl_agent.create_backend"
            or original.__globals__ is not vars(api)
            or Path(original.__code__.co_filename).resolve() != path.resolve()):
        raise ValueError("the native backend constructor is not the pinned original method")
    expected = _find(compile(raw, str(path), "exec", dont_inherit=True), original.__qualname__)
    if (expected is None or _shape(expected) != _shape(original.__code__)
            or original.__defaults__ != ({},) or original.__kwdefaults__ is not None):
        raise ValueError("the loaded native backend method differs from the pinned source")
    return original


def _params(value):
    if (type(value) is not dict or len(value) > 64
            or any(type(k) is not str or type(v) is not str or len(k) > 256 or len(v) > 4096
                   for k, v in value.items())):
        raise ValueError("the native backend parameter map is unsupported")
    return dict(value)


def install_policy(record_callback, *, api=None):
    """Apply the reviewed native option and report every UCX constructor call."""
    if not callable(record_callback):
        raise ValueError("a transport evidence callback is required")
    if api is None:
        api = importlib.import_module("nixl_cu13._api")
    cls = api.nixl_agent
    installed = _INSTALLATIONS.get(cls)
    if installed is not None:
        if (installed["pid"] != os.getpid() or cls.create_backend is not installed["wrapper"]
                or record_callback != installed["callback"]):
            raise ValueError("the transport wrapper differs or belongs to another recorder")
        return profile()
    original = _guard_api(api)
    lock, calls = threading.RLock(), [0]

    @functools.wraps(original)
    def create_backend(self, backend, initParams={}):
        if backend != "UCX":
            return original(self, backend, initParams)
        with lock:
            calls[0] += 1
            call_id = calls[0]
            requested = None
            try:
                requested = _params(initParams)
                record_callback(dict(kind="backend_attempt", call_id=call_id, backend="UCX",
                                     requested_parameters=dict(requested)))
                supplied = dict(requested)
                if supplied.get("ucx_error_handling_mode", "none") != "none":
                    raise ValueError("conflicting UCX peer error mode for the reviewed same-host profile")
                supplied.update(PARAMETERS)
                result = original(self, backend, supplied)
                accepted = _params(self.get_backend_params("UCX"))
                if any(accepted.get(key) != value for key, value in supplied.items()):
                    raise ValueError("the native backend did not retain the supplied transport parameters")
                record_callback(dict(kind="backend_return", call_id=call_id, backend="UCX",
                    supplied_parameters=supplied, accepted_parameters=accepted,
                    native_returned=True, api_sha256=API_SHA256))
                return result
            except BaseException as error:
                try:
                    if requested is None:
                        record_callback(dict(kind="backend_attempt", call_id=call_id, backend="UCX",
                                             requested_parameters=None))
                    record_callback(dict(kind="backend_failure", call_id=call_id, backend="UCX",
                        failure=dict(kind=type(error).__name__, detail=str(error)[:1024])))
                except Exception:
                    pass
                raise

    create_backend._owner_transport_original = original
    cls.create_backend = create_backend
    _INSTALLATIONS[cls] = dict(pid=os.getpid(), wrapper=create_backend, callback=record_callback)
    return profile()


def validate_events(events):
    if type(events) is not list or not 2 <= len(events) <= 128 or len(events) % 2:
        return False
    for index in range(0, len(events), 2):
        attempt, returned = events[index:index + 2]
        if not all(type(x) is dict for x in (attempt, returned)):
            return False
        if (attempt.get("kind") != "backend_attempt" or returned.get("kind") != "backend_return"
                or attempt.get("backend") != "UCX" or returned.get("backend") != "UCX"
                or type(attempt.get("call_id")) is not int or attempt["call_id"] != index // 2 + 1
                or type(returned.get("call_id")) is not int or returned.get("call_id") != attempt["call_id"]
                or returned.get("native_returned") is not True
                or returned.get("api_sha256") != API_SHA256):
            return False
        try:
            requested = _params(attempt["requested_parameters"])
            supplied = _params(returned["supplied_parameters"])
            accepted = _params(returned["accepted_parameters"])
        except (KeyError, ValueError):
            return False
        if (requested.get("ucx_error_handling_mode", "none") != "none"
                or supplied != dict(requested, **PARAMETERS)
                or any(accepted.get(k) != v for k, v in supplied.items())):
            return False
    return True


def validate_receipt(value, run_id, packet_sha256, arm_id, engine_session_id, role):
    context = dict(run_id=run_id, packet_sha256=packet_sha256, arm_id=arm_id,
                   engine_session_id=engine_session_id)
    actual = value.get("context") if type(value) is dict else None
    identity = value.get("process_identity") if type(value) is dict else None
    return bool(type(value) is dict and value.get("schema") == RECEIPT_SCHEMA
        and value.get("status") == "PASS" and value.get("role") == role
        and role in ("prefill", "decode") and value.get("profile") == profile()
        and type(actual) is dict and set(actual) == set(context) | {"process_id"}
        and all(actual.get(k) == v for k, v in context.items())
        and type(actual.get("process_id")) is int and actual["process_id"] > 0
        and type(identity) is dict and set(identity) == {"pid", "start", "group", "session"}
        and all(type(identity.get(k)) is int and identity[k] > 0 for k in identity)
        and identity["pid"] == actual["process_id"]
        and validate_events(value.get("events")))


def _ucx_identity(api):
    path = Path(api.__file__).resolve().parent.parent / "nixl_cu13.libs" / UCP_NAME
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != UCP_SHA256:
        raise ValueError("the native UCX library differs from the pinned wheel")
    loaded = {line.rsplit(None, 1)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
              if "/libucp" in line}
    if loaded != {str(path)}:
        raise ValueError("the actual loaded UCX library path is missing or ambiguous")
    library = ctypes.CDLL(str(path))
    version = [ctypes.c_uint() for _ in range(3)]
    query = library.ucp_get_version
    query.argtypes = [ctypes.POINTER(ctypes.c_uint)] * 3
    query.restype = None
    query(*(ctypes.byref(item) for item in version))
    observed = [item.value for item in version]
    if observed != [1, 23, 0]:
        raise ValueError("the loaded UCX version differs")
    return dict(path=str(path), sha256=UCP_SHA256, version=observed)


def _native_case(mode):
    if mode != "selected":
        raise ValueError("only the selected ordinary transport profile is supported")
    result = dict(schema=CASE_SCHEMA, mode=mode, status="UNKNOWN", failure=None,
        explicit_gpu_allocation=False, explicit_cuda_workload=False, events=[],
        process_id=os.getpid(), python_executable=sys.executable)
    try:
        api = importlib.import_module("nixl_cu13._api")
        _guard_api(api)
        if (importlib.metadata.version("nixl-cu13") != "1.5.0"
                or api.__name__ != "nixl_cu13._api"):
            raise ValueError("the native probe selected a different NIXL package")
        result.update(api_sha256=API_SHA256, api_path=str(Path(api.__file__).resolve()),
                      source_verified=True)
        if mode == "selected":
            install_policy(lambda event: result["events"].append(event), api=api)
        result["stage"] = "native_backend_initialization"
        agent = api.nixl_agent("sdcguard-native-readiness-" + mode,
            api.nixl_agent_config(num_threads=4, capture_telemetry=True))
        result["accepted_parameters"] = _params(agent.get_backend_params("UCX"))
        result["ucx_identity"] = _ucx_identity(api)
        if "UCX" not in agent.backends:
            raise ValueError("the native UCX backend was not created")
        if mode == "selected" and not validate_events(result["events"]):
            raise ValueError("the selected native transport policy was not observed")
        del agent
        gc.collect()
        result.update(status="PASS", stage="complete", agent_released=True)
    except Exception as error:
        result.update(status="FAIL" if result.get("source_verified") else "UNKNOWN",
                      failure=dict(kind=type(error).__name__, detail=str(error)[:1024]))
    return result


def probe(run_id, packet_sha256):
    result = dict(schema=SCHEMA, owner="Hari Krishna Kancharla", run_id=run_id,
        packet_sha256=packet_sha256, status="UNKNOWN", failure=None, scope=SCOPE,
        profile=profile(), cases={}, gpu_submitted=False,
        parent_process_id=os.getpid(),
        interface_inventory=None,
        network_sysfs_present=Path("/sys/class/net").is_dir(),
        pci_sysfs_present=Path("/sys/bus/pci/devices").is_dir(),
        explicit_transport_environment=dict(UCX_RCACHE_MAX_UNRELEASED="1024"))
    environment = clean_environment(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1", UCX_RCACHE_MAX_UNRELEASED="1024")
    try:
        result["interface_inventory"] = [list(pair) for pair in socket.if_nameindex()]
        with tempfile.TemporaryDirectory(prefix="sdcguard-transport-") as temporary:
            for mode in ("selected",):
                directory = Path(temporary)
                output, stdout, stderr = (directory / (mode + suffix) for suffix in (".json", ".stdout", ".stderr"))
                with stdout.open("wb") as out, stderr.open("wb") as err:
                    from .common import isolated_module
                    process = subprocess.run(isolated_module("transport", "--native-case", mode, str(output)), env=environment, stdin=subprocess.DEVNULL,
                        stdout=out, stderr=err, timeout=60, check=False)
                if not output.is_file() or output.stat().st_size > 256 * 1024:
                    raise ValueError("the native transport probe did not retain a bounded result")
                case = json.loads(output.read_bytes())
                if stdout.stat().st_size + stderr.stat().st_size > 512 * 1024:
                    raise ValueError("native transport diagnostic output exceeded its bound")
                case.update(returncode=process.returncode, child_exit_observed=True,
                    stdout=stdout.read_text(errors="replace"), stderr=stderr.read_text(errors="replace"))
                result["cases"][mode] = case
        result["status"] = "PASS"
        if not validate(result, run_id, packet_sha256):
            raise ValueError("the selected native transport initialization did not pass")
    except Exception as error:
        result.update(status="UNKNOWN", failure=dict(kind=type(error).__name__, detail=str(error)[:1024]))
    return result


def validate(value, run_id, packet_sha256):
    if (type(value) is not dict or value.get("schema") != SCHEMA
            or value.get("owner") != "Hari Krishna Kancharla" or value.get("run_id") != run_id
            or value.get("packet_sha256") != packet_sha256 or value.get("status") != "PASS"
            or value.get("failure") is not None or value.get("gpu_submitted") is not False
            or value.get("profile") != profile() or value.get("scope") != SCOPE):
        return False
    cases = value.get("cases")
    if (type(cases) is not dict or set(cases) != {"selected"}
            or type(value.get("parent_process_id")) is not int or value["parent_process_id"] <= 0):
        return False
    pids = {value["parent_process_id"]}
    for name, case in cases.items():
        if (type(case) is not dict or case.get("schema") != CASE_SCHEMA or case.get("mode") != name
                or case.get("source_verified") is not True or case.get("api_sha256") != API_SHA256
                or case.get("explicit_gpu_allocation") is not False
                or case.get("explicit_cuda_workload") is not False
                or case.get("child_exit_observed") is not True or case.get("returncode") != 0
                or type(case.get("returncode")) is not int or case.get("status") not in ("PASS", "FAIL")
                or type(case.get("process_id")) is not int or case["process_id"] <= 0
                or case["process_id"] in pids or type(case.get("python_executable")) is not str
                or not case["python_executable"]):
            return False
        pids.add(case["process_id"])
        if case["status"] == "PASS":
            ucx = case.get("ucx_identity")
            if (case.get("agent_released") is not True or case.get("failure") is not None
                    or type(ucx) is not dict or ucx.get("sha256") != UCP_SHA256
                    or ucx.get("version") != [1, 23, 0] or not str(ucx.get("path", "")).endswith("/" + UCP_NAME)):
                return False
        else:
            return False
    selected = cases["selected"]
    return bool(selected.get("status") == "PASS" and validate_events(selected.get("events"))
        and type(selected.get("accepted_parameters")) is dict
        and selected["accepted_parameters"].get("ucx_error_handling_mode") == "none")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--native-case" or sys.argv[2] != "selected":
        raise SystemExit("unsupported native initialization probe arguments")
    target = Path(sys.argv[3])
    if target.exists():
        raise SystemExit("native initialization result already exists")
    target.write_text(json.dumps(_native_case(sys.argv[2]), sort_keys=True, allow_nan=False) + "\n")
