"""Small packet IO controls. Owner: Hari Krishna Kancharla."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sdcguard import evidence_io as io
from sdcguard import evidence_native_io as bounded


class NativeEvidenceIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="sdcguard-native-io-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def test_streamed_inventory_retains_empty_directories_without_listdir(self):
        (self.root / "empty").mkdir()
        (self.root / "b").write_bytes(b"second")
        (self.root / "a").write_bytes(b"first")
        with patch.object(bounded.os, "listdir", side_effect=AssertionError("unbounded listing")):
            self.assertEqual(bounded.inventory(self.root), (["a", "b"], ["empty"]))
            snapshot, directories = bounded.capture_snapshot(self.root)
        self.assertEqual(snapshot, {"a": b"first", "b": b"second"})
        self.assertEqual(directories, ["empty"])

    def test_enumeration_stops_at_the_first_excess_member(self):
        for index in range(8):
            (self.root / str(index)).write_bytes(b"")
        scandir = bounded.os.scandir
        consumed = []
        class Counted:
            def __init__(self, directory):
                self.stream = scandir(directory)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.stream.close()
            def __iter__(self):
                for entry in self.stream:
                    consumed.append(entry.name)
                    yield entry
        with patch.object(io, "MAX_FILES", 2), patch.object(bounded.os, "scandir", Counted):
            with self.assertRaises(io.EvidenceError):
                bounded.inventory(self.root)
        self.assertEqual(len(consumed), 3)

    def test_root_inclusive_directory_limit_is_enforced_during_inventory(self):
        (self.root / "a").mkdir()
        with patch.object(io, "MAX_FILES", 2):
            self.assertEqual(bounded.inventory(self.root), ([], ["a"]))
            (self.root / "b").mkdir()
            with self.assertRaises(io.EvidenceError):
                bounded.inventory(self.root)

    def test_changed_same_size_bytes_and_membership_are_rejected(self):
        (self.root / "data").write_bytes(b"abc")
        snapshot, directories = bounded.capture_snapshot(self.root)
        (self.root / "data").write_bytes(b"xyz")
        with self.assertRaises(io.EvidenceError):
            bounded.recheck_snapshot(self.root, snapshot, directories)
        (self.root / "data").write_bytes(b"abc")
        (self.root / "new-empty").mkdir()
        with self.assertRaises(io.EvidenceError):
            bounded.recheck_snapshot(self.root, snapshot, directories)

    def test_recheck_reads_only_the_original_byte_budget(self):
        (self.root / "data").write_bytes(b"abc")
        snapshot, directories = bounded.capture_snapshot(self.root)
        actual_read = bounded.config_inputs._read_regular
        limits = []
        def read(root, parts, limit):
            limits.append(limit)
            return actual_read(root, parts, limit)
        with patch.object(bounded.config_inputs, "_read_regular", read):
            bounded.recheck_snapshot(self.root, snapshot, directories)
        self.assertEqual(limits, [3])
        (self.root / "data").write_bytes(b"abcd")
        with self.assertRaises(io.EvidenceError):
            bounded.recheck_snapshot(self.root, snapshot, directories)

    def test_file_and_cumulative_byte_limits_reject_before_copying_more(self):
        (self.root / "a").write_bytes(b"abc")
        (self.root / "b").write_bytes(b"de")
        with patch.object(io, "MAX_FILE_BYTES", 2):
            with self.assertRaises(io.EvidenceError):
                bounded.capture_snapshot(self.root)
        with patch.object(io, "MAX_TOTAL_BYTES", 4):
            with self.assertRaises(io.EvidenceError):
                bounded.capture_snapshot(self.root)
        with patch.object(io, "MAX_TOTAL_BYTES", 5):
            self.assertEqual(bounded.capture_snapshot(self.root)[0], {"a": b"abc", "b": b"de"})

    def test_links_and_excess_depth_are_refused_without_reading_their_targets(self):
        (self.root / "data").write_bytes(b"abc")
        (self.root / "link").symlink_to(self.root / "data")
        with self.assertRaises(io.EvidenceError):
            bounded.capture_snapshot(self.root)
        (self.root / "link").unlink()
        (self.root / "first").mkdir()
        (self.root / "first/second").mkdir()
        with patch.object(bounded, "MAX_DIRECTORY_DEPTH", 1):
            with self.assertRaises(io.EvidenceError):
                bounded.inventory(self.root)


if __name__ == "__main__":
    unittest.main()
