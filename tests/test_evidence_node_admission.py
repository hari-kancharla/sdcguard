"""Small shared node-admission controls. Owner: Hari Krishna Kancharla."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sdcguard import evidence_combined as combined
from sdcguard import evidence_envelope as envelope
from sdcguard import evidence_entry
from sdcguard import evidence_io as io
from sdcguard import evidence_native_mapping as native
from sdcguard import upgrade_combined as upgrade


class SharedNodeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="sdcguard-node-admission-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_whole_snapshot_node_limit_precedes_temporary_reconstruction(self):
        files = {"one.json": b'{"a":1}', "two.json": b'{"b":2}'}
        with patch.object(io, "MAX_JSON_NODES", 5), patch.object(envelope, "TemporaryDirectory") as created:
            with self.assertRaises(io.EvidenceError):
                envelope.evaluate_private(files, [], lambda root: None)
            created.assert_not_called()

    def test_next_leaf_does_not_receive_a_fresh_node_allowance(self):
        first, second = self.root / "first", self.root / "second"
        first.mkdir()
        second.mkdir()
        (first / "data.json").write_bytes(b'{"a":1}')
        (second / "data.json").write_bytes(b'{"b":2}')
        files, directories = {}, []
        with patch.object(io, "MAX_JSON_NODES", 5):
            combined._attach(first, "inputs/native/first", files, directories, 0)
            original_files, original_dirs = dict(files), list(directories)
            with self.assertRaises(io.EvidenceError):
                combined._attach(second, "inputs/native/second", files, directories, 0)
        self.assertEqual(files, original_files)
        self.assertEqual(directories, original_dirs)

    def test_native_normalizer_refuses_shared_node_excess_before_temp_creation(self):
        files = {"manifest.json": io.json_bytes({"schema": native.native.NATIVE_FORMAT}), "extra.json": b'{"a":1}'}
        with patch.object(io, "MAX_JSON_NODES", 4), patch.object(native, "TemporaryDirectory") as created:
            with self.assertRaises(io.EvidenceError):
                native.normalize_snapshot(files, [])
            created.assert_not_called()

    def test_upgrade_capture_keeps_second_node_excess_explicit_and_unretained(self):
        first, second = self.root / "first", self.root / "second"
        first.mkdir()
        second.mkdir()
        (first / "data.json").write_bytes(b'{"a":1}')
        (second / "data.json").write_bytes(b'{"b":2}')
        with patch.object(io, "MAX_JSON_NODES", 5):
            originals, captures, retained, dirs, errors = upgrade._capture_inputs(first, second, {})
        self.assertIn("baseline", originals)
        self.assertNotIn("candidate", originals)
        self.assertEqual(captures["candidate"], ({}, []))
        self.assertIn("node allowance", errors["candidate"])
        self.assertFalse(any(name.startswith("inputs/candidate/") for name in retained))

    def test_public_format_detection_applies_node_admission_before_its_decoder(self):
        (self.root / "manifest.json").write_bytes(b'{"items":[1,2,3]}')
        with patch.object(io, "MAX_JSON_NODES", 4), patch.object(io, "parse_json") as decode:
            with self.assertRaises(io.EvidenceError):
                evidence_entry.packet_format(self.root)
            decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
