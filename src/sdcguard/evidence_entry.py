"""Route additive native evidence options. Owner: Hari Krishna Kancharla."""
import argparse
import json
from pathlib import Path
import sys

from . import config_inputs
from . import evidence_bundle as legacy
from . import evidence_combined as combined
from . import evidence_io as io
from . import evidence_json_budget as json_budget


def packet_format(path):
    raw = config_inputs._read_regular(Path(path).absolute(), ["manifest.json"], io.MAX_FILE_BYTES)
    json_budget.count_members({"manifest.json": raw})
    manifest = io.parse_json(raw)
    if type(manifest) is not dict:
        raise io.EvidenceError("packet manifest must be an object")
    return manifest.get("schema")


def _packet(value):
    name, separator, path = value.partition("=")
    if not separator or not path:
        raise argparse.ArgumentTypeError("use NAME=PATH for each native packet")
    try:
        combined.slot_name(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return name, Path(path)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv == ["catalog", "--native"]:
            print(json.dumps(combined.contract(), sort_keys=True, indent=2))
            return 0
        if len(argv) == 2 and argv[0] == "verify" and packet_format(argv[1]) == combined.FORMAT:
            result = combined.verify_bundle(Path(argv[1]))
        elif argv and argv[0] == "collect" and any(
                value.split("=", 1)[0] in ("--native-packet", "--require-native", "--legacy-packet")
                or value in ("--help", "-h") for value in argv[1:]):
            parser = argparse.ArgumentParser(prog="sdcguard evidence collect")
            parser.add_argument("--native-packet", action="append", type=_packet, default=[])
            parser.add_argument("--require-native", action="append", default=[])
            parser.add_argument("--legacy-packet", type=Path)
            parser.add_argument("--config", type=Path)
            parser.add_argument("--assessment-packet", type=Path)
            parser.add_argument("--components", action="store_true")
            parser.add_argument("--out", type=Path, required=True)
            args = parser.parse_args(argv[1:])
            packet = combined.create_bundle(args.out, native_packets=args.native_packet,
                require_native=args.require_native, legacy_packet=args.legacy_packet,
                config=args.config, components=args.components, recorded_packet=args.assessment_packet)
            checked = combined.verify_bundle(packet["directory"])
            result = {"directory": packet["directory"], "capture_complete": True,
                      "integrity": checked["integrity"], "scope_verdict": checked["scope_verdict"],
                      "errors": checked["errors"], "component_execution": checked.get("component_execution", "not selected"),
                      "runtime_isolation": "not established"}
        else:
            return legacy.main(argv)
        print(json.dumps(result, sort_keys=True, indent=2))
        if result.get("integrity", "verified") != "verified":
            return 2
        return {"fine": 0, "not fine": 1, "inconclusive": 2}[result["scope_verdict"]]
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"scope_verdict": "inconclusive", "errors": [str(exc)]}))
        return 2
