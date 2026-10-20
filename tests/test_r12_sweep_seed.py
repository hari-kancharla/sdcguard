"""Local runner regression utilities; no service calls are made."""
import io
import json
import subprocess
import tarfile
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from _r12_runner_helpers import load_runner


class RunnerDeadlineReview(unittest.TestCase):
    def test_release_smoke_cards_are_not_published_as_sweep_seed_cards(self):
        module = load_runner("modal_check")
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            for label in ("h100_a", "tests_cli", "tests_cli_l4", "tests_cli_h100", "tests_cli_b200"):
                payload = json.dumps({"fixture": label}).encode()
                entry = tarfile.TarInfo(f"board_seed/{label}/GPU-12345678/result.json")
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
        with TemporaryDirectory() as td:
            with patch.object(module, "LOCAL", Path(td)), \
                    patch.object(module, "bundle", types.SimpleNamespace(remote=lambda: archive_bytes.getvalue())), \
                    patch.object(module.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                module._download()
            cards = list((Path(td) / "docs" / "board" / "seed").glob("*.json"))
            self.assertEqual([path.name for path in cards], ["modal_h100_a_12345678.json"])


if __name__ == "__main__":
    unittest.main()
