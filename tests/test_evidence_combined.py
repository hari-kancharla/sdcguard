"""Passive evidence composition controls. Owner: Hari Krishna Kancharla.

The retained native fixture is an ordinary CPU recording. Metadata-only
grouping controls construct no engine execution or native failure trace.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sdcguard import assignment_report
from sdcguard import evidence_bundle as legacy
from sdcguard import evidence_combined as combined
from sdcguard import evidence_entry
from sdcguard import evidence_io as io
from sdcguard import evidence_native_io as bounded

FIXTURE = Path(__file__).parent / "data/native_mapping_cpu/records"
EXAMPLE = Path(__file__).parent.parent / "examples/config-safety/vllm-capacity-shortfall.json"


class CombinedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="sdcguard-combined-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.native = self.root / "native"
        assignment_report.create_packet(FIXTURE, self.native, native=True)

    def create(self, name="combined", **kwargs):
        return combined.create_bundle(self.root / name, **kwargs)

    def rehash(self, directory):
        files, dirs = bounded.capture_snapshot(directory)
        manifest = io.parse_json(files["manifest.json"])
        manifest["files"] = io.file_receipts({name: raw for name, raw in files.items() if name != "manifest.json"})
        manifest["directories"] = dirs
        (directory / "manifest.json").write_bytes(io.json_bytes(manifest))

    def test_native_only_reverifies_and_does_not_select_legacy(self):
        before = bounded.capture_snapshot(self.native)
        result = self.create(native_packets=[("decode", self.native)])
        report = result["report"]
        self.assertEqual(report["status"], "fine")
        self.assertIsNone(report["legacy"])
        self.assertFalse(report["requirements"]["legacy"])
        self.assertEqual(report["requirements"]["native_mapping"], ["decode"])
        checked = combined.verify_bundle(self.root / "combined")
        self.assertEqual(checked["integrity"], "verified", checked)
        self.assertEqual(checked["report"], report)
        self.assertEqual(report["ownership_verdict"], "unknown")
        self.assertEqual(report["physical_read_verdict"], "unknown")
        self.assertEqual(report["runtime_upgrade_witness"], "not established")
        identity = report["native_mapping"][0]["recorded_identities"][0]
        self.assertEqual(len(identity["source_files_sha256"]), 64)
        self.assertEqual(len(identity["configuration_sha256"]), 64)
        self.assertIn("not authenticated", identity["basis"])
        self.assertEqual(bounded.capture_snapshot(self.native), before)

    def test_missing_required_input_is_complete_collection_with_unknown_scope(self):
        result = self.create(require_native=["decode", "decode"])
        self.assertTrue(result["capture_complete"])
        self.assertEqual(result["report"]["status"], "inconclusive")
        row = result["report"]["native_mapping"][0]
        self.assertEqual(row["integrity"], "unavailable")
        self.assertEqual(row["findings"], [])
        checked = combined.verify_bundle(self.root / "combined")
        self.assertEqual(checked["integrity"], "verified")
        self.assertEqual(checked["scope_verdict"], "inconclusive")

    def test_duplicate_assignment_refuses_before_any_input_or_output_work(self):
        with patch.object(combined, "_attach") as attach:
            with self.assertRaises(io.EvidenceError):
                self.create(native_packets=[("decode", self.native), ("decode", self.native)])
            attach.assert_not_called()
        self.assertFalse((self.root / "combined").exists())

    def test_output_inside_any_external_packet_is_refused_without_changing_input(self):
        before = bounded.capture_snapshot(self.native)
        for selection in (
            {"native_packets": [("decode", self.native)]},
            {"legacy_packet": self.native, "require_native": ["decode"]},
            {"recorded_packet": self.native, "require_native": ["decode"]},
        ):
            with self.subTest(selection=list(selection)):
                with patch.object(combined, "_attach") as attach:
                    with self.assertRaises(io.EvidenceError):
                        combined.create_bundle(self.native / "new-child", **selection)
                    attach.assert_not_called()
                self.assertEqual(bounded.capture_snapshot(self.native), before)

    def test_slot_order_is_canonical_and_copies_count_one_raw_stream(self):
        other = self.root / "other"
        shutil.copytree(self.native, other)
        first = self.create("first", native_packets=[("zeta", other), ("alpha", self.native)])["report"]
        second = self.create("second", native_packets=[("alpha", self.native), ("zeta", other)])["report"]
        self.assertEqual(first, second)
        self.assertEqual(first["stream_inventory"]["unique_raw_streams"], 1)
        self.assertEqual(len(first["stream_inventory"]["aliases"]), 1)
        self.assertEqual(first["stream_inventory"]["independent_executions"], "not established")

    def test_known_static_negative_survives_missing_and_invalid_native_leaves(self):
        old = self.root / "legacy"
        self.assertEqual(legacy.create_bundle(old, config=EXAMPLE)["report"]["status"], "not fine")
        bad = self.root / "bad-native"
        shutil.copytree(self.native, bad)
        (bad / "report.json").write_bytes(b"{}")
        result = self.create(native_packets=[("invalid", bad)], require_native=["missing"], legacy_packet=old)
        self.assertEqual(result["report"]["status"], "not fine")
        self.assertEqual(result["report"]["native_mapping"][0]["integrity"], "invalid")
        checked = combined.verify_bundle(self.root / "combined")
        self.assertEqual(checked["integrity"], "verified", checked)
        self.assertEqual(checked["scope_verdict"], "not fine")

    def test_unreadable_input_is_an_unknown_capture_notice(self):
        result = self.create(native_packets=[("missing", self.root / "does-not-exist")])
        self.assertEqual(result["report"]["status"], "inconclusive")
        self.assertEqual(combined.verify_bundle(self.root / "combined")["integrity"], "verified")

    def test_rehashed_report_and_evaluator_edits_do_not_verify(self):
        for label, member in (("report-edit", "report.json"), ("evaluator-edit", "metadata/evaluators.json")):
            self.create(label, native_packets=[("decode", self.native)])
            directory = self.root / label
            value = io.parse_json((directory / member).read_bytes())
            if member == "report.json":
                value["ownership_verdict"] = "satisfied"
            else:
                value[next(iter(value))] = "0" * 64
            (directory / member).write_bytes(io.json_bytes(value))
            self.rehash(directory)
            checked = combined.verify_bundle(directory)
            self.assertEqual(checked["integrity"], "failed", checked)
            self.assertEqual(checked["scope_verdict"], "inconclusive")

    def test_requirement_removal_and_boolean_receipt_are_rejected(self):
        self.create(native_packets=[("decode", self.native)])
        directory = self.root / "combined"
        path = directory / "manifest.json"
        original = io.parse_json(path.read_bytes())
        changed = deepcopy(original)
        changed["profile"]["requirements"]["native_mapping"] = []
        path.write_bytes(io.json_bytes(changed))
        self.assertEqual(combined.verify_bundle(directory)["integrity"], "failed")
        changed = deepcopy(original)
        changed["files"]["report.json"]["bytes"] = True
        path.write_bytes(io.json_bytes(changed))
        self.assertEqual(combined.verify_bundle(directory)["integrity"], "failed")

    def test_each_next_capture_uses_the_remaining_aggregate_allowance(self):
        real = bounded.capture_snapshot
        budgets = []
        def capture(root, **kwargs):
            if kwargs:
                budgets.append(kwargs)
            return real(root, **kwargs)
        with patch.object(bounded, "capture_snapshot", capture):
            self.create(native_packets=[("first", self.native), ("second", self.native)])
        self.assertEqual(len(budgets), 2)
        self.assertLess(budgets[1]["max_bytes"], budgets[0]["max_bytes"])
        self.assertLess(budgets[1]["max_files"], budgets[0]["max_files"])
        self.assertLess(budgets[1]["max_directories"], budgets[0]["max_directories"])
        self.assertEqual(budgets[0]["max_depth"], bounded.MAX_DIRECTORY_DEPTH - 3)

    def test_grouping_ambiguous_recorded_metadata_does_not_merge_streams(self):
        # Parser grouping only. No execution, event trace or isolation result is constructed.
        source = self.create(native_packets=[("first", self.native)])["report"]["native_mapping"][0]
        other = deepcopy(source)
        other["slot"] = "second"
        other["streams"][0]["journal_sha256"] = "0" * 64
        other["findings"][0]["subject"]["slot"] = "second"
        groups = combined._stream_groups([source, other])
        self.assertEqual(len(groups["conflicting_sessions"]), 1)
        self.assertEqual(groups["unique_raw_streams"], 2)
        for section in (source, other):
            self.assertEqual(section["scope_verdict"], "inconclusive")
            self.assertEqual(section["findings"][0]["verdict"], "unknown")

    def test_unavailable_current_component_refresh_cannot_approve_captured_fine(self):
        # Current-check policy only; no component or engine function is executed.
        for captured in ("fine", "not fine"):
            checked = {"scope_verdict": captured, "errors": [], "report": {"legacy": {
                "integrity": "verified", "report": {"profile": {"components": "all"}}}}}
            with patch.object(legacy, "_collect_components", side_effect=ValueError("unavailable")):
                combined.refresh_components(checked)
            self.assertEqual(checked["scope_verdict"], "inconclusive" if captured == "fine" else "not fine")
            self.assertTrue(checked["errors"])
            self.assertEqual(checked["component_execution"]["status"], "unknown")

    def test_semantic_unknown_reasons_survive_the_summary(self):
        sections = [{"slot": "decode", "scope_verdict": "inconclusive", "errors": [], "findings": [
            {"subject": {"slot": "decode"}, "verdict": "unknown", "missing": ["A required observation is absent."]}]}]
        summary = combined._missing(sections, None)
        self.assertEqual(summary[0]["semantic_missing"][0]["missing"], ["A required observation is absent."])

    def test_public_collect_verify_and_missing_exit_codes(self):
        with redirect_stdout(StringIO()) as output:
            code = evidence_entry.main(["collect", "--native-packet", "decode=" + str(self.native),
                                        "--out", str(self.root / "cli")])
        self.assertEqual(code, 0, output.getvalue())
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(evidence_entry.main(["verify", str(self.root / "cli")]), 0)
        self.assertEqual(json.loads(output.getvalue())["integrity"], "verified")
        with redirect_stdout(StringIO()):
            code = evidence_entry.main(["collect", "--require-native", "decode", "--out", str(self.root / "cli-missing")])
        self.assertEqual(code, 2)

    def test_installed_setup_exports_the_native_guide_and_honest_support_scope(self):
        from sdcguard import operator_setup
        destination = self.root / "setup"
        operator_setup.export_setup(destination)
        guide = (destination / "guides/NATIVE_EVIDENCE.md").read_text()
        self.assertIn("--native-packet", guide)
        support = io.parse_json((destination / "SUPPORT.json").read_bytes())
        self.assertEqual(support["native_evidence"]["ownership"], "unknown")
        self.assertIn("inconclusive", support["native_evidence"]["current_runtime_upgrade_gate"])
        self.assertTrue((destination / "guides/ALLOCATION_RECIPIENT.md").is_file())
        self.assertEqual(support["allocation_recipient"]["read_rights"], "unknown")
        self.assertEqual(support["allocation_recipient"]["common_evidence_ingest"], "not implemented")
        self.assertEqual(operator_setup.verify_output(destination)["integrity"], "verified")


if __name__ == "__main__":
    unittest.main()
