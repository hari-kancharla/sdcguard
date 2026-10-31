"""Portable native evidence checks. Owner: Hari Krishna Kancharla.

The retained fixture is an ordinary host-recorder CPU control. These tests
start no engine and construct no engine failure or request ordering.
"""
from copy import deepcopy
import builtins
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sdcguard import assignment_report
from sdcguard import evidence_io as io
from sdcguard import evidence_native_mapping as native


FIXTURE = Path(__file__).parent / "data" / "native_mapping_cpu"


class NativeEvidenceMappingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="sdcguard-native-evidence-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.packet = self.root / "native"
        origin = json.loads((FIXTURE / "PROVENANCE.json").read_text())
        self.assertFalse(origin["native_engine_started"])
        for name, expected in origin["files"].items():
            self.assertEqual(hashlib.sha256((FIXTURE / "records" / name).read_bytes()).hexdigest(), expected)
        result = assignment_report.create_packet(FIXTURE / "records", self.packet, native=True)
        self.assertEqual(result["scope_verdict"], "fine")

    def rebind_member(self, name, raw):
        (self.packet / name).write_bytes(raw)
        manifest = io.parse_json((self.packet / "manifest.json").read_bytes())
        manifest["files"][name] = {"bytes": len(raw), "sha256": io.digest(raw)}
        (self.packet / "manifest.json").write_bytes(io.json_bytes(manifest))

    def test_recomputed_mapping_and_permission_scopes_stay_separate(self):
        before = io.snapshot(self.packet)
        directories = io.names(self.packet, directories=True)
        captured = native.capture(self.packet, packet_index=1)
        result = captured["normalized"]
        self.assertEqual(result["scope_verdict"], "fine")
        self.assertEqual(len(result["findings"]), 1)
        finding = result["findings"][0]
        self.assertEqual(finding["rule_id"], assignment_report.contract.PROPERTY)
        self.assertEqual(finding["evidence_kind"], native.EVIDENCE_KIND)
        self.assertEqual(finding["subject"]["packet"], 1)
        self.assertEqual(finding["verdict"], "satisfied")
        self.assertEqual(finding["facts"]["summary"], result["native_report"]["sessions"][0]["assessment"]["summary"])
        for key in ("ownership_verdict", "physical_read_verdict", "content_provenance", "client_attribution"):
            self.assertEqual(result[key], "unknown")
        self.assertEqual(result["runtime_upgrade_witness"], "not established")
        self.assertEqual(captured["files"], before)
        self.assertEqual(captured["directories"], directories)
        self.assertEqual(io.snapshot(self.packet), before)

    def test_copied_packet_has_the_same_identity_and_recomputed_result(self):
        other = self.root / "copied"
        shutil.copytree(self.packet, other)
        self.assertEqual(native.capture(self.packet), native.capture(other))

    def test_raw_inspection_is_not_native_close_evidence(self):
        records = self.root / "raw-records"
        records.mkdir()
        for path in (FIXTURE / "records").glob("*.jsonl"):
            shutil.copyfile(path, records / path.name)
        packet = self.root / "raw-packet"
        assignment_report.create_packet(records, packet)
        with self.assertRaises(native.NativeMappingEvidenceError):
            native.capture(packet)

    def test_changed_and_rehashed_derived_verdict_is_refused(self):
        report = io.parse_json((self.packet / "report.json").read_bytes())
        report.update(verdict="violated", scope_verdict="not fine")
        self.rebind_member("report.json", io.json_bytes(report))
        with self.assertRaises(native.NativeMappingEvidenceError):
            native.capture(self.packet)

    def test_missing_closure_is_not_replaced_by_a_footer(self):
        manifest = io.parse_json((self.packet / "manifest.json").read_bytes())
        name = next(name for name in manifest["files"] if name.startswith("closures/"))
        (self.packet / name).unlink()
        del manifest["files"][name]
        (self.packet / "manifest.json").write_bytes(io.json_bytes(manifest))
        with self.assertRaises(native.NativeMappingEvidenceError):
            native.capture(self.packet)

    def test_changed_and_rehashed_evaluator_identity_is_refused(self):
        values = io.parse_json((self.packet / "metadata/evaluators.json").read_bytes())
        values[next(iter(values))] = "0" * 64
        self.rebind_member("metadata/evaluators.json", io.json_bytes(values))
        with self.assertRaises(native.NativeMappingEvidenceError):
            native.capture(self.packet)

    def test_unknown_projection_preserves_recorded_zero_and_missing_counters(self):
        captured = native.capture(self.packet)
        for counters_present in (True, False):
            with self.subTest(counters_present=counters_present):
                report = deepcopy(captured["normalized"]["native_report"])
                unknown = assignment_report.contract._base()
                unknown["missing"] = ["required identity is absent"]
                if not counters_present:
                    del unknown["witnesses_omitted"]
                    del unknown["witnesses_complete"]
                report["sessions"][0]["assessment"] = unknown
                report.update(verdict="unknown", scope_verdict="inconclusive")
                findings = native._project(report, captured["normalized"]["packet_snapshot_sha256"], 0)
                facts = findings[0]["facts"]
                self.assertEqual(findings[0]["verdict"], "unknown")
                if counters_present:
                    self.assertEqual(facts["witnesses_omitted"], 0)
                    self.assertIs(facts["witnesses_complete"], True)
                else:
                    self.assertIsNone(facts["witnesses_omitted"])
                    self.assertIsNone(facts["witnesses_complete"])
                self.assertEqual(findings[0]["missing"], ["required identity is absent"])

    def test_unsafe_inventory_and_boolean_index_are_refused(self):
        snapshot = io.snapshot(self.packet)
        directories = io.names(self.packet, directories=True)
        with self.assertRaises(native.NativeMappingEvidenceError):
            native.normalize_snapshot(snapshot, directories, packet_index=True)
        with self.assertRaises(native.NativeMappingEvidenceError):
            native.normalize_snapshot(snapshot, sorted(directories + ["manifest.json"]))
        with self.assertRaises(io.EvidenceError):
            native.normalize_snapshot({**snapshot, "../outside": b"data"}, directories)

    def test_links_are_not_read_and_runtime_modules_are_not_imported(self):
        original_import = builtins.__import__
        attempted = []
        forbidden = {"vllm", "torch", "nixl", "modal"}
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in forbidden:
                attempted.append(name)
                raise AssertionError("runtime import in an offline evidence check")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", guarded):
            native.capture(self.packet)
        self.assertEqual(attempted, [])
        (self.packet / "untrusted-link").symlink_to(FIXTURE / "PROVENANCE.json")
        with self.assertRaises(io.EvidenceError):
            native.capture(self.packet)

    def test_directory_shape_is_rejected_before_any_materialization(self):
        snapshot = io.snapshot(self.packet)
        directories = io.names(self.packet, directories=True)
        deep = ["/".join(["nested"] * level) for level in range(1, 34)]
        cases = (
            sorted(directories + ["undeclared_parent/empty"]),
            sorted(directories + deep),
            sorted(directories + ["manifest.json"]),
        )
        for invalid in cases:
            with self.subTest(directories=invalid[-1]):
                with patch.object(native, "TemporaryDirectory") as temporary:
                    with self.assertRaises(io.EvidenceError):
                        native.normalize_snapshot(snapshot, invalid)
                    temporary.assert_not_called()

    def test_root_counts_toward_the_directory_bound(self):
        snapshot = {"manifest.json": io.json_bytes({"schema": assignment_report.NATIVE_FORMAT})}
        with patch.object(io, "MAX_FILES", 4):
            native._validate_snapshot(snapshot, ["a", "b", "c"])
            with patch.object(native, "TemporaryDirectory") as temporary:
                with self.assertRaises(io.EvidenceError):
                    native.normalize_snapshot(snapshot, ["a", "b", "c", "d"])
                temporary.assert_not_called()

    def test_supported_depth_boundary_is_checked_without_creating_paths(self):
        snapshot = {"manifest.json": io.json_bytes({"schema": assignment_report.NATIVE_FORMAT})}
        directories = sorted("/".join(["nested"] * level) for level in range(1, 33))
        native._validate_snapshot(snapshot, directories)
        snapshot["/".join(["nested"] * 33) + "/data"] = b""
        with patch.object(native, "TemporaryDirectory") as temporary:
            with self.assertRaises(io.EvidenceError):
                native.normalize_snapshot(snapshot, directories)
            temporary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
