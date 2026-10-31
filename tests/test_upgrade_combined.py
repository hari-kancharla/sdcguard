"""Passive upgrade envelope controls. Owner: Hari Krishna Kancharla.

The native input is a retained ordinary host-recorder CPU fixture. Metadata
classification tests do not construct an engine execution or failure trace.
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
from sdcguard import evidence_envelope as envelope
from sdcguard import evidence_io as io
from sdcguard import evidence_native_io as bounded
from sdcguard import upgrade
from sdcguard import upgrade_combined as native_upgrade
from sdcguard import upgrade_entry
from sdcguard import upgrade_policy

FIXTURE = Path(__file__).parent / "data/native_mapping_cpu/records"
NEGATIVE = Path(__file__).parent.parent / "examples/config-safety/vllm-capacity-shortfall.json"


class NativeUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="sdcguard-upgrade-envelope-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.native = self.root / "native"
        assignment_report.create_packet(FIXTURE, self.native, native=True)
        self.base = self.root / "base"
        self.candidate = self.root / "candidate"
        combined.create_bundle(self.base, native_packets=[("decode", self.native)])
        shutil.copytree(self.base, self.candidate)

    def compare(self, label="comparison", baseline=None, candidate=None):
        return native_upgrade.compare_packets(baseline or self.base, candidate or self.candidate, self.root / label)

    def rehash(self, directory):
        files, dirs = bounded.capture_snapshot(directory)
        manifest = io.parse_json(files["manifest.json"])
        manifest["files"] = io.file_receipts({name: raw for name, raw in files.items() if name != "manifest.json"})
        manifest["directories"] = dirs
        (directory / "manifest.json").write_bytes(io.json_bytes(manifest))

    def captures(self):
        return {"baseline": bounded.capture_snapshot(self.base), "candidate": bounded.capture_snapshot(self.candidate)}

    def test_retained_facts_fine_but_current_native_gate_inconclusive(self):
        before = self.captures()
        result = self.compare()
        report = result["report"]
        self.assertEqual(report["captured_scope_verdict"], "fine")
        self.assertEqual(report["comparison_details"]["native_preservation"][0]["status"], "retained evidence")
        self.assertEqual(result["scope_verdict"], "inconclusive")
        self.assertEqual(result["decision"], "blocked")
        self.assertEqual(result["runtime_upgrade_witness"], "not established")
        checked = native_upgrade.verify_comparison(self.root / "comparison")
        self.assertEqual(checked["integrity"], "verified", checked)
        self.assertEqual(checked["report"], report)
        self.assertEqual(checked["scope_verdict"], "inconclusive")
        self.assertEqual(self.captures(), before)
        self.assertEqual(report["ownership_verdict"], "unknown")
        self.assertEqual(report["physical_read_verdict"], "unknown")

    def test_relocated_comparison_reproduces_without_input_paths(self):
        self.compare()
        moved = self.root / "moved"
        shutil.copytree(self.root / "comparison", moved)
        checked = native_upgrade.verify_comparison(moved)
        self.assertEqual(checked["integrity"], "verified", checked)
        self.assertEqual(checked["scope_verdict"], "inconclusive")

    def test_slot_order_does_not_change_semantics(self):
        first, second = self.root / "first", self.root / "second"
        combined.create_bundle(first, native_packets=[("zeta", self.native), ("alpha", self.native)])
        combined.create_bundle(second, native_packets=[("alpha", self.native), ("zeta", self.native)])
        result = self.compare(baseline=first, candidate=second)
        rows = result["report"]["comparison_details"]["native_preservation"]
        self.assertEqual([row["slot"] for row in rows], ["alpha", "zeta"])
        self.assertEqual([row["status"] for row in rows], ["retained evidence"] * 2)
        self.assertEqual(result["scope_verdict"], "inconclusive")

    def test_removed_required_native_slot_is_gap(self):
        first, second = self.root / "first", self.root / "second"
        combined.create_bundle(first, native_packets=[("prefill", self.native), ("decode", self.native)])
        combined.create_bundle(second, native_packets=[("decode", self.native)])
        result = self.compare(baseline=first, candidate=second)
        gaps = result["report"]["comparison_details"]["gaps"]
        self.assertTrue(any(row["code"] == "required_native_slots_changed" and row["removed"] == ["prefill"] for row in gaps))
        self.assertEqual(result["scope_verdict"], "inconclusive")

    def test_missing_required_native_leaf_is_not_removed_or_satisfied(self):
        missing = self.root / "missing"
        combined.create_bundle(missing, require_native=["decode"])
        result = self.compare(candidate=missing)
        self.assertEqual(result["report"]["requirements"]["candidate"]["native_mapping"], ["decode"])
        self.assertEqual(result["report"]["captured_scope_verdict"], "inconclusive")
        self.assertEqual(result["report"]["comparison_details"]["native_preservation"][0]["status"], "inconclusive")

    def test_independent_candidate_negative_survives_invalid_baseline_and_native_gaps(self):
        bad = self.root / "bad"
        bad.mkdir()
        (bad / "manifest.json").write_bytes(b"{broken")
        negative = self.root / "negative"
        combined.create_bundle(negative, config=NEGATIVE, require_native=["missing"])
        with redirect_stdout(StringIO()) as output:
            code = upgrade_entry.main(["compare", str(bad), str(negative), "--out", str(self.root / "negative-comparison")])
        self.assertEqual(code, 1, output.getvalue())
        result = json.loads(output.getvalue())
        self.assertEqual(result["scope_verdict"], "not fine")
        self.assertEqual(result["report"]["inputs"]["baseline"]["verification"]["integrity"], "failed")
        self.assertEqual(result["report"]["inputs"]["candidate"]["verification"]["integrity"], "verified")
        self.assertIsNone(result["report"]["comparison_details"]["legacy_comparison"])
        self.assertTrue(result["report"]["comparison_details"]["gaps"])
        checked = native_upgrade.verify_comparison(self.root / "negative-comparison")
        self.assertEqual(checked["integrity"], "verified", checked)
        self.assertEqual(checked["scope_verdict"], "not fine")

    def test_absent_and_empty_baseline_do_not_skip_a_verified_candidate_negative(self):
        negative = self.root / "negative"
        combined.create_bundle(negative, config=NEGATIVE, require_native=["missing"])
        empty = self.root / "empty"
        empty.mkdir()
        for name, baseline in (("absent", self.root / "absent"), ("empty", empty)):
            with self.subTest(baseline=name):
                result = self.compare(label="comparison-" + name, baseline=baseline, candidate=negative)
                self.assertEqual(result["scope_verdict"], "not fine")
                before = result["report"]["inputs"]["baseline"]
                self.assertEqual(before["verification"]["integrity"], "unavailable" if name == "absent" else "failed")
                if name == "absent":
                    self.assertIsNone(before["packet_sha256"])
                checked = native_upgrade.verify_comparison(self.root / ("comparison-" + name))
                self.assertEqual(checked["integrity"], "verified", checked)
                self.assertEqual(checked["scope_verdict"], "not fine")

    def test_mixed_v1_v2_removal_and_legacy_nested_policy_only(self):
        old = self.root / "old"
        legacy.create_bundle(old, config=NEGATIVE)
        mixed = self.root / "mixed"
        combined.create_bundle(mixed, legacy_packet=old, native_packets=[("decode", self.native)])
        real = upgrade_policy.compare
        seen = []
        def compare(left, right):
            seen.append((left["schema"], right["schema"]))
            return real(left, right)
        with patch.object(upgrade_policy, "compare", compare):
            result = self.compare(baseline=mixed, candidate=old)
        self.assertEqual(seen, [("sdcguard.evidence-report.v1", "sdcguard.evidence-report.v1")])
        self.assertEqual(result["scope_verdict"], "not fine")
        self.assertTrue(any(row["code"] == "required_native_slots_changed" for row in result["report"]["comparison_details"]["gaps"]))

    def test_changed_capture_with_matching_stack_is_incomparable_metadata_only(self):
        # Contract comparison only: alter a verified metadata copy, not raw engine events.
        captures = self.captures()
        checked = combined.verify_snapshot(*captures["baseline"])
        other = deepcopy(checked)
        other["report"]["native_mapping"][0]["streams"][0]["journal_sha256"] = "0" * 64
        with patch.object(native_upgrade, "_verify_input", side_effect=[(combined.FORMAT, checked), (combined.FORMAT, other)]):
            report = native_upgrade._evaluate(captures)
        self.assertEqual(report["captured_scope_verdict"], "inconclusive")
        self.assertTrue(any(row["code"] == "changed_native_capture_incomparable" for row in report["comparison_details"]["gaps"]))
        self.assertEqual(native_upgrade._current_gate(report)["scope_verdict"], "inconclusive")

    def test_native_violation_projection_has_no_regression_attribution(self):
        # Finding projection only. No execution or failure journal is manufactured.
        checked = combined.verify_snapshot(*self.captures()["candidate"])
        section = deepcopy(checked["report"]["native_mapping"][0])
        section["findings"][0]["verdict"] = "violated"
        rows = native_upgrade._native_violations({"decode": section})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["regression_attribution"], "not established")
        section["integrity"] = "invalid"
        self.assertEqual(native_upgrade._native_violations({"decode": section}), [])

    def test_outer_corruption_never_supplies_approved_report(self):
        negative = self.root / "negative"
        combined.create_bundle(negative, config=NEGATIVE, require_native=["missing"])
        result = self.compare(candidate=negative)
        self.assertEqual(result["scope_verdict"], "not fine")
        (self.root / "comparison" / "report.json").write_bytes(b"{}")
        checked = native_upgrade.verify_comparison(self.root / "comparison")
        self.assertEqual(checked["integrity"], "failed")
        self.assertEqual(checked["scope_verdict"], "inconclusive")
        self.assertNotIn("report", checked)

    def test_rehashed_report_html_policy_and_evaluator_edits_refuse(self):
        self.compare()
        for name in ("report.json", "index.html", "metadata/native-policy.json", "metadata/upgrade-evaluators.json"):
            with self.subTest(member=name):
                changed = self.root / ("changed-" + name.replace("/", "-"))
                shutil.copytree(self.root / "comparison", changed)
                (changed / name).write_bytes(b"{}" if name.endswith(".json") else b"changed")
                self.rehash(changed)
                checked = native_upgrade.verify_comparison(changed)
                self.assertEqual(checked["integrity"], "failed", checked)
                self.assertEqual(checked["scope_verdict"], "inconclusive")

    def test_boolean_receipt_and_extra_empty_directory_refuse(self):
        self.compare()
        original = self.root / "comparison"
        changed = self.root / "boolean"
        shutil.copytree(original, changed)
        manifest = io.parse_json((changed / "manifest.json").read_bytes())
        manifest["files"]["report.json"]["bytes"] = True
        (changed / "manifest.json").write_bytes(io.json_bytes(manifest))
        self.assertEqual(native_upgrade.verify_comparison(changed)["integrity"], "failed")
        extra = self.root / "extra"
        shutil.copytree(original, extra)
        (extra / "unselected-empty").mkdir()
        self.rehash(extra)
        self.assertEqual(native_upgrade.verify_comparison(extra)["integrity"], "failed")

    def test_next_capture_receives_remaining_aggregate_budget_and_prefix_depth(self):
        real = bounded.capture_snapshot
        calls = []
        def capture(root, **kwargs):
            if kwargs:
                calls.append(kwargs)
            return real(root, **kwargs)
        with patch.object(bounded, "capture_snapshot", capture):
            self.compare()
        self.assertEqual(len(calls), 2)
        for key in ("max_files", "max_bytes", "max_directories"):
            self.assertLess(calls[1][key], calls[0][key])
        self.assertEqual(calls[0]["max_depth"], bounded.MAX_DIRECTORY_DEPTH - 2)
        self.assertLessEqual(calls[0]["max_bytes"], io.MAX_TOTAL_BYTES - 3 * io.MAX_FILE_BYTES)

    def test_aggregate_file_count_leaves_second_input_unavailable_without_holding_it(self):
        before = self.captures()
        first_count = len(before["baseline"][0])
        with patch.object(io, "MAX_FILES", 2 * first_count + len(native_upgrade.BASE_FILES) - 1):
            result = self.compare()
            candidate = result["report"]["inputs"]["candidate"]
            self.assertEqual(candidate["verification"]["integrity"], "unavailable")
            self.assertEqual(candidate["file_count"], 0)
            self.assertEqual(result["scope_verdict"], "inconclusive")
            checked = native_upgrade.verify_comparison(self.root / "comparison")
            self.assertEqual(checked["integrity"], "verified", checked)
        self.assertEqual(self.captures(), before)

    def test_overlap_and_existing_output_refuse_without_input_mutation(self):
        before = self.captures()
        with patch.object(bounded, "capture_snapshot") as capture:
            for output in (self.base / "inside", self.candidate / "inside", self.base, self.root):
                with self.subTest(output=output), self.assertRaises(io.EvidenceError):
                    native_upgrade.compare_packets(self.base, self.candidate, output)
            capture.assert_not_called()
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "keep").write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            native_upgrade.compare_packets(self.base, self.candidate, existing)
        self.assertEqual((existing / "keep").read_bytes(), b"preserve")
        self.assertEqual(self.captures(), before)

    def test_source_change_during_completion_refuses_and_keeps_completed_snapshot(self):
        real = envelope.complete
        def complete(*args, **kwargs):
            root = real(*args, **kwargs)
            (self.candidate / "report.json").write_bytes(b"{}")
            return root
        with patch.object(envelope, "complete", complete):
            with self.assertRaises(io.EvidenceError):
                self.compare()
        self.assertTrue((self.root / "comparison" / "manifest.json").is_file())
        checked = native_upgrade.verify_comparison(self.root / "comparison")
        self.assertEqual(checked["integrity"], "verified", checked)
        self.assertEqual(checked["scope_verdict"], "inconclusive")

    def test_current_components_are_refreshed_only_outside_saved_report(self):
        captures = self.captures()
        checked = combined.verify_snapshot(*captures["baseline"])
        # Current CPU dispatch only, using a detached ordinary report shape.
        fake_old = {"integrity": "verified", "scope_verdict": "fine", "report": {"profile": {"components": "all"}}}
        checked["report"]["legacy"] = fake_old
        checked["report"]["requirements"]["legacy"] = True
        with patch.object(native_upgrade, "_verify_input", side_effect=[(combined.FORMAT, checked), (combined.FORMAT, deepcopy(checked))]):
            with patch.object(upgrade_policy, "compare", return_value={"gaps": [], "policy_violations": []}):
                with patch.object(combined, "refresh_components") as refresh:
                    report = native_upgrade._evaluate(captures)
                    refresh.assert_not_called()
        original = deepcopy(report)
        def refresh(wrapper):
            wrapper["component_execution"] = {"scope": "source_fragment_cpu", "status": "unknown"}
            wrapper["errors"].append("Current CPU unavailable")
            return wrapper
        with patch.object(combined, "refresh_components", refresh):
            current = native_upgrade._current_gate(report)
        self.assertEqual(report, original)
        self.assertEqual(current["scope_verdict"], "inconclusive")
        self.assertEqual(len(current["current_execution_gaps"]), 3)

    def test_dispatch_keeps_both_v1_commands_unchanged_and_native_current_blocks(self):
        old = self.root / "old"
        legacy.create_bundle(old, config=NEGATIVE)
        argv = ["compare", str(old), str(old), "--out", str(self.root / "legacy-compare")]
        with patch.object(upgrade, "main", return_value=17) as original:
            self.assertEqual(upgrade_entry.main(argv), 17)
            original.assert_called_once_with(argv)
        with patch.object(upgrade_entry.evidence_entry, "packet_format", return_value=upgrade.FORMAT):
            with patch.object(upgrade, "main", return_value=17) as original:
                self.assertEqual(upgrade_entry.main(["verify", "old-comparison"]), 17)
                original.assert_called_once_with(["verify", "old-comparison"])
        with patch.object(upgrade_entry.evidence_entry, "packet_format", return_value="unknown"):
            with patch.object(upgrade, "main") as original, patch.object(native_upgrade, "main", return_value=2) as current:
                self.assertEqual(upgrade_entry.main(["verify", "unknown-comparison"]), 2)
                original.assert_not_called()
                current.assert_called_once_with(["verify", "unknown-comparison"])
        with redirect_stdout(StringIO()) as output:
            code = upgrade_entry.main(["compare", str(self.base), str(self.candidate), "--out", str(self.root / "cli")])
        self.assertEqual(code, 2, output.getvalue())
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(upgrade_entry.main(["verify", str(self.root / "cli")]), 2)
        self.assertEqual(json.loads(output.getvalue())["integrity"], "verified")
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(upgrade_entry.main(["policy", "--native"]), 0)
        self.assertEqual(json.loads(output.getvalue())["policy"]["name"], "preserve-retained-native-facts.v1")


if __name__ == "__main__":
    unittest.main()
