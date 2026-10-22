"""Synthetic final-export and integrity checks. Supplied unrun for the owner."""
from pathlib import Path
import tempfile
import unittest


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from slr_pipeline import cli, common, reviews
        cls.cli, cls.common, cls.reviews = cli, common, reviews

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {"_project_dir": str(self.root), "reviews_file": "reviews.csv",
                       "search": {"start": "2016-01-01", "end": "2026-10-21"}}
        self.pdf_text = self.root / "synthetic.txt"
        self.pdf_text.write_text("Synthetic test text. " * 100, encoding="utf-8")
        self.ta = {"record_id": "R_fixture", "report_id": "R_fixture", "title": "Synthetic fixture",
                   "abstract": "Synthetic abstract", "date": "2001-01-01", "ai_decision": "include",
                   "screening_status": "assessed", "input_sha256": "ta_fixture_hash"}
        self.ft = {**self.ta, "input_sha256": "ft_fixture_hash", "text_complete": True,
                   "full_text_status": "ready", "full_text_path": str(self.pdf_text),
                   "full_text_file_sha256": self.common.file_hash(self.pdf_text)}

    def save_reviews(self, rows):
        self.common.write_csv(self.root / "reviews.csv", rows, self.reviews.REVIEW_FIELDS)

    def two_inclusions(self):
        return [{"stage": "full_text", "record_id": "R_fixture", "input_sha256": "ft_fixture_hash",
                 "reviewer_id": identity, "role": "reviewer", "decision": "include", "reason_code": "",
                 "notes": "", "reviewed_utc": "2026-10-21T12:00:00Z", "study_id": "S_fixture"}
                for identity in ("person_one", "person_two")]

    def export(self):
        self.cli.export_results(self.config, self.root / "run", ta=[self.ta], ft=[self.ft])
        return self.common.read_csv(self.root / "run/06_final/final_included_papers.csv")

    def test_ai_candidate_is_not_human_confirmed_final(self):
        self.assertEqual(self.export(), [])
        self.assertEqual(len(self.common.read_csv(self.root / "run/06_final/included_candidates.csv")), 1)

    def test_two_valid_human_inclusions_require_intact_fulltext_and_date(self):
        self.save_reviews(self.two_inclusions())
        self.assertEqual(len(self.export()), 1)
        self.pdf_text.write_text("Replaced synthetic evidence", encoding="utf-8")
        self.assertEqual(self.export(), [])
        pending = self.common.read_csv(self.root / "run/06_final/pending_review.csv")
        self.assertEqual(pending[0]["full_text_status"], "changed_or_missing_since_screening")

    def test_outside_window_cannot_enter_final_even_with_review(self):
        self.save_reviews(self.two_inclusions())
        self.ft["date"] = "2030-01-01"
        self.assertEqual(self.export(), [])

    def test_unknown_review_input_is_reported(self):
        self.save_reviews([{**self.two_inclusions()[0], "record_id": "R_typo"}])
        self.export()
        audit = self.common.read_csv(self.root / "run/06_final/review_import_audit.csv")
        self.assertEqual(audit[0]["import_status"], "unknown_or_inactive_stage_record")

    def test_changed_generated_artifact_is_rejected(self):
        out = self.root / "run"
        self.common.write_csv(out / "01_retrieval/records.csv", [self.ta])
        self.cli.end_stage(out, "retrieve", [self.ta])
        self.assertTrue(self.cli.validate_stage(out, "retrieve"))
        (out / "01_retrieval/records.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.cli.validate_stage(out, "retrieve")

    def test_reviewer_identity_whitespace_and_case_do_not_make_two_people(self):
        rows = self.two_inclusions()
        rows[0]["reviewer_id"], rows[1]["reviewer_id"] = "Alice", " ALICE "
        resolved, _ = self.reviews.resolve_reviews([self.ft], "full_text", rows)
        self.assertEqual(resolved[0]["human_decision"], "")


if __name__ == "__main__":
    unittest.main()
