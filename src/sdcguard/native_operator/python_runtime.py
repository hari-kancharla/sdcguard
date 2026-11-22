"""Exact interpreter and host header checks. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile

from .common import digest, regular_bytes, encoded, file_receipt

SCHEMA = "sdcguard.owner-python-runtime.v1"
PYTHON_ROOT = Path("/opt/sdcguard-python")
HEADER_SOURCE = r'''#include <Python.h>
static PyObject *version(PyObject *self, PyObject *args) {
    return Py_BuildValue("(ii)", PY_MAJOR_VERSION, PY_MINOR_VERSION);
}
static PyMethodDef methods[] = {{"version", version, METH_NOARGS, "Header version."}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_owner_header_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit__owner_header_probe(void) { return PyModule_Create(&module); }
'''
LOAD_SOURCE = '''import importlib.util,json,sys
spec=importlib.util.spec_from_file_location("_owner_header_probe",sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
print(json.dumps({"compiled":list(module.version()),"running":list(sys.version_info[:2])}))
'''


def cleanup_api_availability():
    return {
        "os.pidfd_open": callable(getattr(os, "pidfd_open", None)),
        "signal.pidfd_send_signal": callable(getattr(signal, "pidfd_send_signal", None)),
        "os.waitid": callable(getattr(os, "waitid", None)),
        "os.P_PIDFD": type(getattr(os, "P_PIDFD", None)) is int and os.P_PIDFD > 0,
        "select.poll": callable(getattr(select, "poll", None)),
        "select.POLLIN": type(getattr(select, "POLLIN", None)) is int and select.POLLIN > 0,
        "select.POLLHUP": type(getattr(select, "POLLHUP", None)) is int and select.POLLHUP > 0,
    }


def require_interpreter(*, check_launcher=True):
    """Read identity only. No engine import or subprocess is performed."""
    available = cleanup_api_availability()
    missing = [name for name, present in available.items() if not present]
    if missing:
        raise ValueError("cleanup APIs missing from this interpreter: " + ", ".join(missing))
    root = PYTHON_ROOT.resolve(strict=True)
    executable = Path(sys.executable)
    if (sys.platform != "linux" or sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12)
            or not executable.is_absolute() or executable.parent != PYTHON_ROOT / "bin"
            or Path(sys.prefix).resolve(strict=True) != root or sys.prefix == sys.base_prefix):
        raise ValueError("CPython 3.12 in the /opt/sdcguard-python virtual environment is required")
    config = regular_bytes(PYTHON_ROOT / "pyvenv.cfg", 64 * 1024)
    if "include-system-site-packages = false" not in config.decode("utf-8").lower():
        raise ValueError("the isolated virtual environment configuration differs")
    value = dict(schema="sdcguard.owner-python-identity.v1", version=list(sys.version_info[:3]),
        version_text=sys.version, implementation=sys.implementation.name, platform=sys.platform,
        executable=str(executable), executable_realpath=str(executable.resolve(strict=True)),
        prefix=str(PYTHON_ROOT), base_prefix=sys.base_prefix,
        pyvenv_cfg_sha256=digest(config), soabi=sysconfig.get_config_var("SOABI"),
        cleanup_api_availability=available)
    if check_launcher:
        launcher = shutil.which("vllm")
        path = Path(launcher) if launcher else None
        if path != PYTHON_ROOT / "bin/vllm" or path.is_symlink():
            raise ValueError("the vllm launcher is not the installed virtual environment script")
        raw = regular_bytes(path, 64 * 1024)
        first = raw.split(b"\n", 1)[0].decode("utf-8")
        interpreter = Path(first[2:]) if first.startswith("#!/") else None
        if (interpreter is None or interpreter.parent != PYTHON_ROOT / "bin"
                or interpreter.resolve(strict=True) != executable.resolve(strict=True)):
            raise ValueError("the vllm launcher shebang names a different interpreter")
        value["vllm_launcher"] = dict(path=str(path), shebang=first, sha256=digest(raw),
            interpreter_realpath=str(interpreter.resolve(strict=True)))
    return value


def _command(argv, *, timeout=60):
    """Retain bounded output from a known local CPU command."""
    with tempfile.TemporaryFile() as stream:
        completed = subprocess.run([str(x) for x in argv], stdout=stream, stderr=subprocess.STDOUT,
            timeout=timeout, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        size = stream.seek(0, os.SEEK_END)
        stream.seek(0)
        raw = stream.read(64 * 1024 + 1)
    if size > 64 * 1024 or len(raw) > 64 * 1024:
        raise ValueError("host Python probe output exceeded its bound")
    return dict(command=[str(x) for x in argv], returncode=completed.returncode,
        output=raw.decode("utf-8", "replace"), output_bytes=len(raw), output_sha256=digest(raw))


def header_probe():
    result = dict(status="FAIL", source_sha256=digest(HEADER_SOURCE.encode()), failure=None,
        scope="fresh host Python.h extension compile and load only; CUDA and Triton driver startup remain unobserved")
    try:
        compiler = shutil.which("gcc")
        if not compiler:
            raise ValueError("gcc is unavailable for the Python.h check")
        includes = list(dict.fromkeys(sysconfig.get_path(name) for name in ("include", "platinclude")))
        if not includes or any(not name or not (Path(name) / "Python.h").is_file() for name in includes):
            raise ValueError("the running interpreter's Python.h include path is unavailable")
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if type(suffix) is not str or not suffix.endswith(".so") or "/" in suffix:
            raise ValueError("the running interpreter extension suffix is unsupported")
        result["include_paths"] = includes
        with tempfile.TemporaryDirectory(prefix="sdcguard-python-headers-") as directory:
            root = Path(directory)
            source = root / "probe.c"
            source.write_text(HEADER_SOURCE)
            extension = root / ("_owner_header_probe" + suffix)
            result["compile"] = _command([compiler, "-O3", "-shared", "-fPIC",
                *["-I" + name for name in includes], source, "-o", extension])
            if result["compile"]["returncode"] != 0:
                raise ValueError("fresh Python.h extension compilation failed")
            raw = regular_bytes(extension, 4 * 1024 * 1024)
            result["extension"] = dict(bytes=len(raw), sha256=digest(raw))
            result["load"] = _command([sys.executable, "-I", "-c", LOAD_SOURCE, extension])
            if result["load"]["returncode"] != 0:
                raise ValueError("fresh Python.h extension loading failed")
            observation = json.loads(result["load"]["output"])
            if observation != {"compiled": [3, 12], "running": [3, 12]}:
                raise ValueError("Python.h compiled and running interpreter versions differ")
            result.update(status="PASS", observation=observation)
    except Exception as error:
        result["failure"] = dict(kind=type(error).__name__, detail=str(error)[:1024])
    return result


def pip_check():
    result = _command([sys.executable, "-m", "pip", "check"], timeout=120)
    result["status"] = "PASS" if result["returncode"] == 0 else "FAIL"
    return result


def distro_packages():
    names = ["python3.12", "python3.12-dev", "python3.12-venv", "libpython3.12-dev"]
    result = _command(["dpkg-query", "-W", "-f=${Package}\t${Version}\n", *names])
    rows = [line.split("\t") for line in result["output"].splitlines()]
    versions = {row[0]: row[1] for row in rows if len(row) == 2}
    agrees = (len(rows) == len(versions) == len(names) and set(versions) == set(names)
              and len(set(versions.values())) == 1)
    result.update(status="PASS" if result["returncode"] == 0 and agrees else "FAIL", versions=versions)
    return result


def probe(run_id, packet_sha256):
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None or re.fullmatch(r"[0-9a-f]{64}", packet_sha256) is None:
        raise ValueError("Python runtime invocation identity differs")
    result = dict(schema=SCHEMA, owner="Hari Krishna Kancharla", run_id=run_id, packet_sha256=packet_sha256,
        status="FAIL", identity=None, cleanup_api_availability=cleanup_api_availability(),
        header_probe=None, pip_check=None, distro_packages=None, failure=None, cpu_only=True, gpu_submitted=False,
        runtime_qualified=False)
    stage = "interpreter"
    try:
        result["identity"] = require_interpreter()
        stage = "distro_packages"
        result["distro_packages"] = distro_packages()
        if result["distro_packages"].get("status") != "PASS":
            raise ValueError("the installed Python, venv and development package versions differ")
        stage = "python_headers"
        result["header_probe"] = header_probe()
        if result["header_probe"].get("status") != "PASS":
            raise ValueError("fresh Python.h compile/load check did not pass")
        stage = "pip_check"
        result["pip_check"] = pip_check()
        if result["pip_check"].get("status") != "PASS":
            raise ValueError("the exact interpreter's pip check did not pass")
        result["status"] = "PASS"
    except Exception as error:
        result["failure"] = dict(stage=stage, kind=type(error).__name__, detail=str(error)[:1024])
    return result


def valid_command(value):
    return (type(value) is dict and type(value.get("returncode")) is int and value["returncode"] == 0
            and type(value.get("command")) is list and bool(value["command"])
            and all(type(part) is str for part in value["command"])
            and type(value.get("output")) is str and type(value.get("output_bytes")) is int
            and value["output_bytes"] == len(value["output"].encode()) <= 64 * 1024
            and value.get("output_sha256") == digest(value["output"].encode()))


def validate(value, run_id, packet_sha256):
    if not (type(value) is dict and value.get("schema") == SCHEMA
            and value.get("owner") == "Hari Krishna Kancharla" and value.get("run_id") == run_id
            and value.get("packet_sha256") == packet_sha256 and value.get("status") == "PASS"
            and value.get("failure") is None and value.get("cpu_only") is True
            and value.get("gpu_submitted") is False and value.get("runtime_qualified") is False):
        return False
    identity, header, pip = value.get("identity"), value.get("header_probe"), value.get("pip_check")
    packages = value.get("distro_packages")
    names = set(cleanup_api_availability())
    availability = value.get("cleanup_api_availability")
    executable = identity.get("executable") if type(identity) is dict else None
    launcher = identity.get("vllm_launcher") if type(identity) is dict else None
    shebang = launcher.get("shebang") if type(launcher) is dict else None
    version = identity.get("version") if type(identity) is dict else None
    return (type(availability) is dict and set(availability) == names and all(x is True for x in availability.values())
        and type(identity) is dict and identity.get("schema") == "sdcguard.owner-python-identity.v1"
        and type(version) is list and len(version) == 3 and version[:2] == [3, 12]
        and all(type(part) is int and part >= 0 for part in version)
        and type(executable) is str and Path(executable).parent == PYTHON_ROOT / "bin"
        and type(identity.get("executable_realpath")) is str
        and identity.get("platform") == "linux"
        and identity.get("implementation") == "cpython" and identity.get("prefix") == str(PYTHON_ROOT)
        and identity.get("base_prefix") != str(PYTHON_ROOT)
        and encoded(identity.get("cleanup_api_availability")) == encoded(availability)
        and type(launcher) is dict and launcher.get("path") == str(PYTHON_ROOT / "bin/vllm")
        and type(shebang) is str and shebang.startswith("#!/")
        and Path(shebang[2:]).parent == PYTHON_ROOT / "bin"
        and launcher.get("interpreter_realpath") == identity.get("executable_realpath")
        and re.fullmatch(r"[0-9a-f]{64}", str(launcher.get("sha256"))) is not None
        and type(header) is dict and header.get("status") == "PASS" and header.get("failure") is None
        and header.get("source_sha256") == digest(HEADER_SOURCE.encode())
        and encoded(header.get("observation")) == encoded({"compiled": [3, 12], "running": [3, 12]})
        and file_receipt(header.get("extension"))
        and valid_command(header.get("compile")) and valid_command(header.get("load"))
        and valid_command(pip) and pip.get("status") == "PASS"
        and valid_command(packages) and packages.get("status") == "PASS"
        and type(packages.get("versions")) is dict
        and set(packages["versions"]) == {"python3.12", "python3.12-dev", "python3.12-venv", "libpython3.12-dev"}
        and all(type(version) is str and version for version in packages["versions"].values())
        and len(set(packages["versions"].values())) == 1
        and pip.get("command") == [identity.get("executable"), "-m", "pip", "check"])
