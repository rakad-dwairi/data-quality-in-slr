"""Synthetic checks for the local coordinator; no worker or model is launched."""
from pathlib import Path
import json
import tempfile
import unittest

from complete_local_review import fulltext_preflight, pilot_checks, seed_title_candidates, writing_pack
from slr_pipeline.common import read_csv, write_csv


class LocalCompletionTests(unittest.TestCase):
    def test_fulltext_preflight_skips_unavailable_text_and_checks_one_real_input(self):
        from unittest.mock import Mock
        rows = [{"record_id": "R_A", "effective_decision": "include", "ai_decision": "include", "oa_pdf_url": "https://example.org/a.pdf"},
                {"record_id": "R_B", "effective_decision": "uncertain", "oa_pdf_url": "https://example.org/b.pdf"}]
        prepare = Mock(side_effect=[[{"full_text_status": "unavailable"}],
                                    [{"full_text_status": "ready", "text_complete": True}]])
        screen = Mock(return_value=[{"screening_status": "chunked_complete"}])
        result = fulltext_preflight(rows, {}, Path("synthetic"), prepare, screen)
        self.assertEqual(result["record_id"], "R_B")
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(screen.call_count, 1)
        self.assertFalse(result["scientific_accuracy_validated"])

    def test_fulltext_preflight_failure_does_not_enable_bulk(self):
        from unittest.mock import Mock
        rows = [{"record_id": "R_A", "effective_decision": "include", "oa_pdf_url": "https://example.org/a.pdf"}]
        prepare = Mock(return_value=[{"full_text_status": "ready", "text_complete": True}])
        screen = Mock(return_value=[{"screening_status": "model_or_validation_error"}])
        with self.assertRaisesRegex(RuntimeError, "bulk full-text inference was not enabled"):
            fulltext_preflight(rows, {}, Path("synthetic"), prepare, screen)

    def test_seed_matching_preserves_candidate_identity_without_inventing_validation(self):
        rows = [{"record_id": "R_SYNTHETIC", "title": "Synthetic: Seed-Title", "doi": "10.9999/synthetic"}]
        result = seed_title_candidates(rows, ["Synthetic Seed Title", "Missing seed"])
        self.assertFalse(result["all_seed_titles_found"])
        self.assertEqual(result["matches"]["Synthetic Seed Title"][0]["record_id"], "R_SYNTHETIC")
        self.assertEqual(result["matches"]["Missing seed"], [])
        self.assertFalse(result["author_year_identity_independently_verified"])

    @staticmethod
    def row(title, decision="uncertain", status="assessed", unclear=False):
        return {"title": title, "screening_status": status, "ai_decision": decision,
                "criteria_json": json.dumps([{"criterion": "synthetic", "value": "unclear" if unclear else "yes"}])}

    def test_pilot_pass_is_not_scientific_or_human_validation(self):
        result = pilot_checks([self.row("Synthetic retained"), self.row("Synthetic unrelated", "exclude")],
                              {"Synthetic retained": "retain", "Synthetic unrelated": "out_of_scope"})
        self.assertTrue(result["passed"])
        self.assertFalse(result["scientific_accuracy_validated"])
        self.assertTrue(result["human_review_required"])

    def test_most_criterion_evidence_downgraded_blocks_bulk_even_with_safe_retention(self):
        row = self.row("Synthetic retained")
        row["criteria_json"] = json.dumps([{"criterion": str(i), "value": "yes"} for i in range(5)])
        row["evidence_downgraded_criteria"] = "3"
        self.assertFalse(pilot_checks([row], {"Synthetic retained": "retain"})["passed"])

    def test_potentially_eligible_discard_and_secondary_inclusion_stop_bulk(self):
        for title, category, decision in (("Synthetic candidate", "retain", "exclude"),
                                          ("Synthetic review", "secondary", "include"),
                                          ("Synthetic unrelated", "out_of_scope", "include")):
            with self.subTest(category=category):
                self.assertFalse(pilot_checks([self.row(title, decision)], {title: category})["passed"])

    def test_missing_error_duplicate_and_wholesale_uncertainty_fail(self):
        expected = {"Synthetic A": "retain", "Synthetic B": "retain"}
        for rows in ([], [self.row("Synthetic A", status="model_unavailable"), self.row("Synthetic B")],
                     [self.row("Synthetic A"), self.row("Synthetic A")],
                     [self.row("Synthetic A", unclear=True), self.row("Synthetic B", unclear=True)]):
            self.assertFalse(pilot_checks(rows, expected)["passed"])

    def test_pack_preserves_proposals_and_never_invents_human_inclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            rows = [{"record_id": "R_SYNTHETIC", "title": "Synthetic title\nTY - EVIL",
                     "authors": "Alpha Example;Beta Example", "doi": "10.9999/synthetic", "date": "2020",
                     "ai_decision": "include", "human_decision": ""}]
            write_csv(run / "06_final/included_candidates.csv", rows)
            write_csv(run / "06_final/final_included_papers.csv", [], ["record_id", "human_decision"])
            write_csv(run / "05_extraction/synthetic_evidence.csv", [{"synthetic": "fixture"}])
            original = (run / "06_final/included_candidates.csv").read_bytes()
            result = writing_pack(run)
            self.assertEqual(result["ai_inclusion_candidates"], 1)
            self.assertEqual(result["human_confirmed_reports"], 0)
            self.assertFalse(result["publication_ready"])
            self.assertFalse(result["target_count_used_for_selection"])
            folder = run / "07_writing_pack"
            self.assertEqual(read_csv(folder / "human_confirmed_papers.csv"), [])
            self.assertEqual(read_csv(folder / "candidate_papers.csv")[0]["human_decision"], "")
            self.assertEqual((run / "06_final/included_candidates.csv").read_bytes(), original)
            self.assertEqual((folder / "candidate_references.ris").read_text().count("\nTY  - "), 0)
            self.assertTrue((folder / "evidence_tables/synthetic_evidence.csv").is_file())


if __name__ == "__main__":
    unittest.main()
