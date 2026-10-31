"""Route additive native upgrade envelopes. Owner: Hari Krishna Kancharla."""
import argparse
import json
from pathlib import Path
import sys

from . import evidence_bundle as legacy
from . import evidence_entry
from . import upgrade
from . import upgrade_combined as combined


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv == ["policy", "--native"]:
            return combined.main(["policy"])
        if argv and argv[0] == "compare" and not any(value in ("-h", "--help") for value in argv):
            parser = argparse.ArgumentParser(prog="sdcguard upgrade compare")
            parser.add_argument("baseline", type=Path)
            parser.add_argument("candidate", type=Path)
            parser.add_argument("--out", type=Path, required=True)
            args = parser.parse_args(argv[1:])
            formats = []
            for path in (args.baseline, args.candidate):
                try:
                    formats.append(evidence_entry.packet_format(path))
                except (OSError, ValueError, TypeError, KeyError, RecursionError):
                    formats.append(None)
            if formats != [legacy.FORMAT, legacy.FORMAT]:
                return combined.main(argv)
        elif len(argv) == 2 and argv[0] == "verify":
            if evidence_entry.packet_format(argv[1]) != upgrade.FORMAT:
                return combined.main(argv)
        return upgrade.main(argv)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"scope_verdict": "inconclusive", "errors": [str(exc)]}))
        return 2
