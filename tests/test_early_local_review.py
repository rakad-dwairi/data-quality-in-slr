"""Synthetic early-review checks; no real PDF/model/provider calls."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import early_local_review as early
from slr_pipeline.common import read_csv, read_json, write_json, file_hash


class EarlyReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        blocked = patch("requests.sessions.Session.request", side_effect=AssertionError("No real model/network"))
        blocked.start(); self.addCleanup(blocked.stop)

    @staticmethod
    def decision(value="include", status="assessed", missing=False):
        names = list(early.screening.CORE_CRITERIA) + ["full_text_sufficient"]
        return {"ai_decision": value, "screening_status": status,
                "criteria_json": json.dumps([{"criterion": n, "value": "yes"} for n in names[:-1] if missing] if missing else
                                            [{"criterion": n, "value": "yes"} for n in names]),
                "extraction_json": "[]", "appraisal_json": "[]"}

    def test_primary_checkpoint_read_is_read_only_and_stage_bound(self):
        folder = self.root / "primary/checkpoints"; folder.mkdir(parents=True)
        path = folder / "screening.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE assessments(record_id,cache_key,result_json,created_at,stage,status)")
            for rid, stage, status, date in [("A", "title_abstract", "assessed", "2026-10-24"),
                                              ("OLD", "title_abstract", "assessed", "2026-10-22"),
                                              ("FT", "full_text", "assessed", "2026-10-24"),
                                              ("ERR", "title_abstract", "model_unavailable", "2026-10-24")]:
                db.execute("INSERT INTO assessments VALUES(?,?,?,?,?,?)", (rid, rid, "{}", date, stage, status))
        before = path.read_bytes()
        rows = early.checkpoints(folder.parent, "2026-10-23")
        self.assertEqual([r["record_id"] for r in rows], ["A"])
        self.assertEqual(path.read_bytes(), before)

    def test_priority_retains_promising_uncertainty_without_importing_other_decisions(self):
        records = {rid: {"record_id": rid, "oa_pdf_url": "https://example.org/a.pdf"} for rid in ("A", "B", "C", "D", "E")}
        rows = []
        for rid, decision, yes in [("A", "uncertain", 4), ("B", "include", 5), ("C", "exclude", 3), ("D", "uncertain", 2), ("E", "include", 5)]:
            rows.append({"record_id": rid, "created_at": rid, "cache_key": rid,
                         "result": {"ai_decision": decision, "criteria_json": json.dumps([{"value": "yes"}] * yes)}})
        original = json.dumps(rows, sort_keys=True)
        self.assertEqual([r["record_id"] for r in early.shortlist(rows, records, {"E"})], ["B", "A"])
        self.assertEqual(json.dumps(rows, sort_keys=True), original)

    def test_budget_does_not_increase_without_primary_progress(self):
        self.assertEqual(early.permitted_budget(0, 10, .15, 180), 180)
        self.assertEqual(early.permitted_budget(100, 10, .15, 180), 330)

    def test_protocol_seed_can_be_checked_early_without_inventing_ta_decision(self):
        records = {"R_SEED": {"record_id": "R_SEED", "title": "Synthetic Known Item", "oa_pdf_url": "https://example.org/seed.pdf"},
                   "R_OTHER": {"record_id": "R_OTHER", "title": "Other item", "oa_pdf_url": "https://example.org/other.pdf"}}
        queue = early.shortlist([], records, set(), ["Synthetic Known Item"])
        self.assertEqual(len(queue), 1)
        self.assertTrue(queue[0]["protocol_seed"])
        self.assertEqual(queue[0]["cache_key"], "")
        self.assertNotIn("result", queue[0])
        self.assertEqual(early.shortlist([], records, {"R_SEED"}, ["Synthetic Known Item"]), [])
        self.assertEqual(early.permitted_budget(100, 10, .15, 180), 330)

    def test_blinded_prompt_and_eligibility_only_schema_restore_on_error(self):
        original = early.screening.SYSTEM_PROMPT, early.screening.PROMPT_VERSION, early.screening._schema
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with early.second_pass():
                self.assertIn("blinded eligibility", early.screening.SYSTEM_PROMPT)
                self.assertNotEqual(early.screening.PROMPT_VERSION, original[1])
                schema = early.screening._schema("full_text", ["synthetic"])
                for name in ("extraction", "appraisal", "iso_mapping"):
                    self.assertEqual(schema["properties"][name]["maxItems"], 0)
                raise RuntimeError("synthetic")
        self.assertEqual((early.screening.SYSTEM_PROMPT, early.screening.PROMPT_VERSION, early.screening._schema), original)

    def test_consensus_requires_two_complete_supported_fulltext_inclusions(self):
        yes = self.decision()
        self.assertEqual(early.consensus(yes, yes, True), "double_pass_candidate")
        for second in (self.decision("uncertain"), self.decision("exclude"), self.decision(status="model_unavailable"), self.decision(missing=True)):
            self.assertNotEqual(early.consensus(yes, second, True), "double_pass_candidate")
        self.assertEqual(early.consensus(yes, yes, False), "source_needed")

    def test_missing_source_does_not_call_model(self):
        config = {"search": {"start": "1900-01-01", "end": "2026-10-23"}}
        row = {"record_id": "R_SYNTHETIC", "date": "2020-01-01", "date_status": "within_window"}
        with patch.object(early, "prepare_fulltexts", return_value=[{**row, "text_complete": False, "full_text_error": "synthetic missing PDF"}]), \
                patch.object(early.screening, "screen_records", side_effect=AssertionError("Missing text must not be inferred")):
            result = early.review_one(row, config, self.root / "report", "synthetic_key")
        self.assertEqual(result["summary"]["review_status"], "source_needed")
        self.assertEqual(result["summary"]["human_decision"], "")

    def test_two_passes_receive_source_without_previous_decision(self):
        config = {"model": {"name": "synthetic", "allow_remote": False, "base_url": "http://127.0.0.1:11434"},
                  "protocol": {}, "search": {"start": "1900-01-01", "end": "2026-10-23"}}
        record = {"record_id": "R_SYNTHETIC", "title": "Synthetic report", "date": "2020-01-01", "date_status": "within_window"}
        prepared = {**record, "text_complete": True, "full_text": "Synthetic source body", "full_text_status": "ready"}
        seen = []
        def assess(rows, stage, cfg, folder):
            seen.append((dict(rows[0]), early.screening.SYSTEM_PROMPT))
            return [{**rows[0], **self.decision(), "ai_model_rationale": "Synthetic previous rationale"}]
        with patch.object(early, "prepare_fulltexts", return_value=[prepared]), patch.object(early.screening, "screen_records", side_effect=assess):
            result = early.review_one(record, config, self.root / "report", "synthetic_key")
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0], seen[1][0])
        self.assertNotIn("ai_model_rationale", seen[1][0])
        self.assertNotEqual(seen[0][1], seen[1][1])
        self.assertEqual(result["summary"]["review_status"], "double_pass_candidate")

    def test_export_keeps_uncertainty_and_does_not_create_human_confirmation(self):
        self.root.mkdir(exist_ok=True)
        results = []
        for rid, status in [("R_A", "double_pass_candidate"), ("R_B", "needs_review")]:
            results.append({"summary": {"record_id": rid, "title": "Synthetic title\nTY - EVIL", "review_status": status,
                                        "human_decision": "", "doi": "10.9999/synthetic"},
                            "first": self.decision(), "second": self.decision()})
        status = early.export_pack(self.root, results)
        self.assertEqual(status["double_pass_candidates"], 1)
        self.assertEqual(status["human_confirmed"], 0)
        self.assertFalse(status["publication_ready"])
        self.assertEqual(len(read_csv(self.root / "writing_pack/early_candidates.csv")), 1)
        self.assertEqual(len(read_csv(self.root / "writing_pack/needs_review.csv")), 1)
        self.assertEqual(len(read_csv(self.root / "writing_pack/criterion_evidence.csv")), 24)
        self.assertEqual((self.root / "writing_pack/early_candidates.ris").read_text().count("\nTY  - "), 0)
        self.assertEqual(read_json(self.root / "writing_pack/status.json")["human_confirmed"], 0)

    def fixture_source(self):
        library = self.root / "slr_pipeline"; library.mkdir()
        (library / "synthetic.py").write_text("# synthetic frozen source\n")
        source = self.root / "primary"; (source / "code_snapshot").mkdir(parents=True)
        (source / "code_snapshot/synthetic.py").write_text("# synthetic frozen source\n")
        config = {"model": {"allow_remote": False, "base_url": "http://127.0.0.1:11434"},
                  "screening": {}, "fulltext": {"timeout_seconds": 90}}
        write_json(source / "manifest.json", {"config": config, "fingerprint": "synthetic",
                   "source_hashes": {"slr_pipeline/synthetic.py": file_hash(library / "synthetic.py")}})
        write_json(source / "stage_state.json", {"screen": {"started_utc": "2026-10-24T00:00:00+00:00"}})
        return source

    def test_initialization_freezes_own_config_and_preserves_primary_bytes(self):
        source = self.fixture_source(); output = self.root / "early"
        before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
        config, identity = early.initialize(self.root, source, output, .15, 180)
        self.assertEqual(config["fulltext"]["timeout_seconds"], 30)
        self.assertFalse(config["screening"]["retry_errors"])
        self.assertTrue((output / "code_snapshot/early_local_review.py").exists())
        self.assertEqual(before, {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()})
        self.assertEqual(early.initialize(self.root, source, output, .15, 180)[1], identity)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            early.initialize(self.root, source, output, .1, 180)

    def test_remote_changed_source_and_unsafe_outputs_or_budgets_are_refused(self):
        source = self.fixture_source()
        for share, bootstrap in [(0, 180), (.5, 180), (.15, 1000)]:
            with self.subTest(share=share, bootstrap=bootstrap), self.assertRaises(ValueError):
                early.initialize(self.root, source, self.root / "early", share, bootstrap)
        with self.assertRaisesRegex(ValueError, "separate output"):
            early.initialize(self.root, source, source / "child", .15, 180)
        manifest = read_json(source / "manifest.json"); manifest["config"]["model"]["base_url"] = "https://example.org"
        write_json(source / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "loopback-only"):
            early.initialize(self.root, source, self.root / "early", .15, 180)
        manifest["config"]["model"]["base_url"] = "http://127.0.0.1:11434"
        write_json(source / "manifest.json", manifest)
        (self.root / "slr_pipeline/synthetic.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            early.initialize(self.root, source, self.root / "early", .15, 180)


if __name__ == "__main__":
    unittest.main()
