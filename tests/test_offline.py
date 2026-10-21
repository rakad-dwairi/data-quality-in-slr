"""Offline regression checks using synthetic records and temporary directories.

These tests do not search databases, download papers, or contact Ollama. They
are supplied for the user to run locally; writing this file does not run them.

Run from the project directory:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock


class OfflineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deferred project imports run only when the user runs the tests.
        from slr_pipeline import common, dedup, retrieval, reviews
        cls.common = common
        cls.dedup = dedup
        cls.retrieval = retrieval
        cls.reviews = reviews

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recsys-slr-offline-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.network = mock.patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("Network calls are forbidden in offline tests"),
        )
        self.network.start()
        self.addCleanup(self.network.stop)

    @staticmethod
    def record(ordinal, **changes):
        # Synthetic metadata for accounting tests; these are not paper claims.
        result = {
            "observation_id": f"O_fixture_{ordinal}",
            "source": "synthetic_export", "source_id": f"fixture_{ordinal}",
            "query_id": "synthetic_query", "raw_ref": f"synthetic.json#record={ordinal}",
            "title": "Synthetic analysis of data conditions in recommendation fixtures",
            "abstract": "Synthetic abstract used only to test file and record accounting.",
            "authors": "Example, Alpha; Example, Beta", "date": "2001-01-01",
            "date_basis": "synthetic_fixture", "doi": "10.9999/fixture-a",
            "related_doi": "", "venue": "Synthetic Test Venue",
            "publication_type": "journal-article", "url": "", "oa_pdf_url": "",
        }
        result.update(changes)
        return result

    def deduplicate(self, rows):
        return self.dedup.deduplicate(rows, self.root, {
            "deduplication": {"similarity": 95, "block_limit": 200, "skip_fuzzy": False}
        })


class DeduplicationTests(OfflineTestCase):
    def test_exact_doi_duplicate_has_reversible_accounting(self):
        rows = [self.record(1), self.record(2, abstract="Longer synthetic abstract for the same fixture report.")]
        kept = self.deduplicate(rows)
        output = self.root / "02_deduplication"
        mapping = self.common.read_csv(output / "record_mapping.csv")
        removed = self.common.read_csv(output / "duplicates_removed.csv")
        summary = self.common.read_json(output / "summary.json")
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["dedup_group_size"], 2)
        self.assertEqual(len(mapping), 2)
        self.assertEqual({row["observation_id"] for row in mapping}, {"O_fixture_1", "O_fixture_2"})
        self.assertEqual({row["record_id"] for row in mapping}, {kept[0]["record_id"]})
        self.assertEqual(Counter(row["role"] for row in mapping),
                         {"representative": 1, "duplicate_merged": 1})
        self.assertEqual(len(removed), 1)
        self.assertEqual(summary["input_observations"], 2)
        self.assertEqual(summary["duplicates_removed"], 1)
        self.assertEqual(summary["reports_remaining"], 1)
        self.assertIsNone(summary["verified_study_count"])
        self.assertEqual(kept[0]["study_id"], "")
        # Input order must not determine a report identity.
        repeated = self.deduplicate(list(reversed(rows)))
        self.assertEqual(repeated[0]["record_id"], kept[0]["record_id"])

    def test_different_known_dois_with_identical_metadata_are_retained(self):
        kept = self.deduplicate([
            self.record(1, doi="10.9999/fixture-a"),
            self.record(2, doi="10.9999/fixture-b"),
        ])
        self.assertEqual(len(kept), 2)
        self.assertEqual(len({row["record_id"] for row in kept}), 2)
        pairs = self.common.read_csv(self.root / "02_deduplication" / "possible_duplicates.csv")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(self.common.read_csv(self.root / "02_deduplication" / "duplicates_removed.csv")), 0)

    def test_missing_metadata_does_not_merge_identical_titles(self):
        changes = {"doi": "", "authors": "", "date": "", "venue": "", "publication_type": ""}
        kept = self.deduplicate([self.record(1, **changes), self.record(2, **changes)])
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(row["dedup_group_size"] == 1 for row in kept))

    def test_metadata_row_cannot_bridge_two_known_dois(self):
        kept = self.deduplicate([
            self.record(1, doi="10.9999/fixture-a"),
            self.record(2, doi=""),
            self.record(3, doi="10.9999/fixture-b"),
        ])
        self.assertEqual(len(kept), 3)
        self.assertEqual(len({row["record_id"] for row in kept}), 3)

    def test_preprint_and_published_report_are_linked_but_not_merged(self):
        kept = self.deduplicate([
            self.record(1, source="arxiv", source_id="0000.00000", publication_type="preprint",
                        date="2000-01-01", doi="10.48550/arxiv.0000.00000", related_doi="10.9999/fixture-a"),
            self.record(2),
        ])
        self.assertEqual(len(kept), 2)
        links = self.common.read_csv(self.root / "02_deduplication" / "version_links.csv")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["link_status"], "possible_related_reports_pending_review")
        self.assertEqual(links[0]["primary_report_id"], "")
        self.assertEqual(links[0]["study_id"], "")

    def test_duplicate_observation_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Repeated observation_id"):
            self.deduplicate([self.record(1), self.record(1, doi="10.9999/fixture-b")])


class ManualImportTests(OfflineTestCase):
    def test_local_csv_import_resumes_with_stable_ids_and_no_provider_calls(self):
        source = self.root / "synthetic_input.csv"
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Title", "Abstract", "Year", "DOI", "EID"])
            writer.writeheader()
            writer.writerow({"Title": "Synthetic local import fixture", "Abstract": "Synthetic evidence text.",
                             "Year": "2001", "DOI": "10.9999/fixture-import", "EID": "fixture-id"})
        config = {
            "_project_dir": str(self.root),
            "search": {"enabled": False, "sources": [], "topics": [],
                       "start": "1900-01-01", "end": "2026-10-21"},
            "imports": [{"source": "synthetic_export", "path": source.name}],
        }
        run = self.root / "run"
        with mock.patch.object(self.retrieval.providers, "Client",
                               side_effect=AssertionError("Manual imports must not create a provider client")):
            first = self.retrieval.collect(config, run)
            second = self.retrieval.collect(config, run)
        self.assertEqual(len(first), 1)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["title"], "Synthetic local import fixture")
        self.assertEqual(first[0]["date"], "2001")
        self.assertEqual(first[0]["import_original_source_id"], "fixture-id")
        self.assertTrue(first[0]["source_id"].startswith("import:"))
        self.assertTrue((run / first[0]["raw_ref"].split("#", 1)[0]).is_file())
        raw_before = self.common.file_hash(source)
        self.assertEqual(raw_before, first[0]["import_file_sha256"])

    def test_identical_import_file_contents_are_counted_once(self):
        source = self.root / "one.csv"
        content = "title,abstract,year,doi\nSynthetic duplicate import fixture,Synthetic text,2001,10.9999/fixture-import\n"
        source.write_text(content, encoding="utf-8")
        duplicate = self.root / "two.csv"
        duplicate.write_text(content, encoding="utf-8")
        config = {
            "_project_dir": str(self.root),
            "search": {"enabled": False, "sources": [], "topics": [],
                       "start": "1900-01-01", "end": "2026-10-21"},
            "imports": [{"source": "fixture", "path": source.name},
                        {"source": "fixture", "path": duplicate.name}],
        }
        records = self.retrieval.collect(config, self.root / "run")
        self.assertEqual(len(records), 1)
        logs = self.common.read_json(self.root / "run" / "01_retrieval" / "search_log.json")
        self.assertEqual([row["status"] for row in logs], ["imported", "duplicate_import_skipped"])


class CommonFileTests(OfflineTestCase):
    def test_csv_roundtrip_protects_formula_cells_and_preserves_unicode_newlines(self):
        rows = [{"record_id": f"R{index}", "value": value} for index, value in enumerate([
            "=SUM(1,2)", "+1", "-2", "@command", "  =1+1", "'Synthetic title",
            "Unicode: é\nSecond line, with comma and \"quote\"", "",
        ])]
        path = self.root / "nested" / "roundtrip.csv"
        self.common.write_csv(path, rows, ["record_id", "value"])
        self.assertEqual(self.common.read_csv(path), rows)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            protected = list(csv.DictReader(handle))
        self.assertEqual(protected[0]["value"], "'=SUM(1,2)")
        self.assertEqual(protected[4]["value"], "'  =1+1")
        self.assertEqual(protected[5]["value"], "'Synthetic title")

    def test_empty_csv_retains_header_and_json_roundtrip_is_exact(self):
        path = self.root / "empty.csv"
        self.common.write_csv(path, [], ["record_id", "decision"])
        self.assertEqual(self.common.read_csv(path), [])
        self.assertEqual(path.read_text(encoding="utf-8-sig").strip(), "record_id,decision")
        data = {"b": [1, None, "é"], "a": {"decision": "uncertain"}}
        json_path = self.root / "nested" / "value.json"
        self.common.write_json(json_path, data)
        self.assertEqual(self.common.read_json(json_path), data)
        self.assertEqual(self.common.digest({"a": 1, "b": 2}), self.common.digest({"b": 2, "a": 1}))


class HumanReviewTests(OfflineTestCase):
    def current_row(self):
        return {"record_id": "R_fixture", "input_sha256": self.common.digest("Synthetic current text"),
                "ai_decision": "include", "title": "Synthetic review fixture"}

    def review(self, reviewer_id, decision="include", role="reviewer", **changes):
        row = {
            "stage": "full_text", "record_id": "R_fixture",
            "input_sha256": self.current_row()["input_sha256"], "reviewer_id": reviewer_id,
            "role": role, "decision": decision,
            "reason_code": "NO_DATA_QUALITY_CONSTRUCT" if decision == "exclude" else "",
            "notes": "Synthetic reason recorded for this test." if decision in {"exclude", "secondary"} else "",
            "reviewed_utc": "2026-10-21T12:00:00+00:00", "study_id": "S_fixture",
        }
        row.update(changes)
        return row

    def resolve(self, reviews):
        rows, audit = self.reviews.resolve_reviews([self.current_row()], "full_text", reviews)
        self.assertEqual(len(rows), 1)
        return rows[0], audit

    def test_one_human_review_never_becomes_a_final_human_decision(self):
        row, audit = self.resolve([self.review("person_a")])
        self.assertEqual(row["human_status"], "pending")
        self.assertEqual(row["human_decision"], "")
        self.assertEqual(row["effective_decision"], "include")  # AI proposal remains separately marked.
        self.assertEqual(audit[0]["import_status"], "valid")

    def test_two_distinct_agreeing_reviewers_are_accepted(self):
        row, audit = self.resolve([self.review("person_a"), self.review("person_b")])
        self.assertEqual(row["human_status"], "agreed")
        self.assertEqual(row["human_decision"], "include")
        self.assertEqual(row["study_id"], "S_fixture")
        self.assertTrue(all(item["import_status"] == "valid" for item in audit))

    def test_stale_text_hash_cannot_count_toward_agreement(self):
        row, audit = self.resolve([
            self.review("person_a"), self.review("person_b", input_sha256="stale-synthetic-hash"),
        ])
        self.assertEqual(row["human_status"], "pending")
        self.assertEqual(row["human_decision"], "")
        self.assertEqual(audit[1]["import_status"], "stale_or_missing_text_hash")

    def test_disagreement_remains_unresolved_without_adjudication(self):
        row, _ = self.resolve([self.review("person_a"), self.review("person_b", "exclude")])
        self.assertEqual(row["human_status"], "disagreement")
        self.assertEqual(row["human_decision"], "")

    def test_a_reviewing_person_cannot_also_adjudicate_the_disagreement(self):
        row, _ = self.resolve([
            self.review("person_a"), self.review("person_b", "exclude"),
            self.review("person_a", role="adjudicator", notes="Synthetic adjudication rationale."),
        ])
        self.assertEqual(row["human_decision"], "")
        self.assertNotEqual(row["human_status"], "adjudicated")

    def test_distinct_third_person_with_rationale_can_adjudicate(self):
        row, _ = self.resolve([
            self.review("person_a"), self.review("person_b", "exclude"),
            self.review("person_c", role="adjudicator", notes="Synthetic adjudication rationale."),
        ])
        self.assertEqual(row["human_status"], "adjudicated")
        self.assertEqual(row["human_decision"], "include")

    def test_exclusion_requires_a_valid_principal_reason(self):
        row, audit = self.resolve([
            self.review("person_a", "exclude", reason_code=""),
            self.review("person_b", "exclude", reason_code=""),
        ])
        self.assertEqual(row["human_decision"], "")
        self.assertTrue(all(item["import_status"] == "principal_exclusion_reason_required" for item in audit))

    def test_same_person_twice_does_not_satisfy_independent_review(self):
        row, _ = self.resolve([self.review("person_a"), self.review("person_a")])
        self.assertEqual(row["human_decision"], "")
        self.assertEqual(row["human_status"], "pending")


if __name__ == "__main__":
    unittest.main()
