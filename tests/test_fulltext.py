"""Synthetic full-text checks supplied for the user to run locally.

Writing this file does not run it. All input files are disposable fixtures; the
tests prohibit network connections and never contact databases or models.
"""
from __future__ import annotations

import csv
import hashlib
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class FullTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deferred project imports occur only if the user runs these tests.
        from slr_pipeline import common, fulltext
        cls.common = common
        cls.fulltext = fulltext

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recsys-fulltext-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "inputs" / "fulltext"
        self.directory.mkdir(parents=True)
        self.manifest = self.root / "inputs" / "fulltext_manifest.csv"
        self.run = self.root / "run"
        self.config = {
            "_project_dir": str(self.root),
            "fulltext": {
                "directory": "inputs/fulltext",
                "manifest": "inputs/fulltext_manifest.csv",
                "download_open_access": False,
                "min_text_chars": 1000,
                "max_download_mb": 1,
                "timeout_seconds": 1,
            },
        }
        self.network = mock.patch(
            "socket.create_connection",
            side_effect=AssertionError("Network connections are forbidden in full-text tests"),
        )
        self.network.start()
        self.addCleanup(self.network.stop)
        self.dns = mock.patch(
            "socket.getaddrinfo",
            side_effect=AssertionError("Live DNS is forbidden in full-text tests"),
        )
        self.dns.start()
        self.addCleanup(self.dns.stop)

    @staticmethod
    def record(record_id="R_fixture", **changes):
        result = {
            "record_id": record_id,
            "title": "Synthetic full-text fixture; not a research paper",
            "doi": "10.9999/fulltext-fixture",
            "oa_pdf_url": "",
            "url": "",
        }
        result.update(changes)
        return result

    def text_file(self, name="R_fixture.txt"):
        path = self.directory / name
        body = (
            "Synthetic fixture: measured data conditions, methods, results and limitations. "
            "This invented text only exercises local extraction and coverage accounting.\n"
        ) * 30
        path.write_text(body, encoding="utf-8")
        return path, body

    def write_manifest(self, rows):
        fields = ["record_id", "doi", "path", "coverage"]
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def prepare(self, records=None):
        return self.fulltext.prepare_fulltexts(records or [self.record()], self.config, self.run)

    def test_complete_local_txt_is_ready_and_source_is_preserved(self):
        path, body = self.text_file()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        records = self.prepare()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["full_text_status"], "ready")
        self.assertIs(record["text_complete"], True)
        self.assertEqual(record["full_text_path"], str(path.resolve()))
        self.assertEqual(record["full_text"], "[TEXT FILE]\n" + body)
        self.assertEqual(record["full_text_sha256"],
                         hashlib.sha256(record["full_text"].encode("utf-8")).hexdigest())
        self.assertEqual(record["full_text_file_sha256"], before)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertTrue(list((self.run / "checkpoints" / "fulltext").glob("*.json")))
        self.assertEqual(self.common.read_csv(self.run / "04_full_text" / "fulltext_needed.csv"), [])
        # The same unchanged text may be reused without altering the source file.
        repeated = self.prepare()[0]
        self.assertEqual(repeated["full_text_sha256"], record["full_text_sha256"])
        self.assertIs(repeated["text_complete"], True)

    def test_manifest_partial_coverage_stays_unresolved_despite_long_text(self):
        path, _ = self.text_file("partial.txt")
        self.write_manifest([{
            "record_id": "R_fixture", "doi": "10.9999/fulltext-fixture",
            "path": str(path.relative_to(self.root)), "coverage": "partial",
        }])
        record = self.prepare()[0]
        self.assertEqual(record["full_text_status"], "partial")
        self.assertIs(record["text_complete"], False)
        self.assertGreater(len(record["full_text"]), 1000)
        self.assertIn("Partial coverage declared", record["full_text_error"])
        needed = self.common.read_csv(self.run / "04_full_text" / "fulltext_needed.csv")
        self.assertEqual([row["record_id"] for row in needed], ["R_fixture"])

    def test_duplicate_manifest_rows_are_ambiguous_and_do_not_select_a_file(self):
        first, _ = self.text_file("first.txt")
        second, _ = self.text_file("second.txt")
        self.write_manifest([
            {"record_id": "R_fixture", "doi": "", "path": str(path), "coverage": "complete"}
            for path in (first, second)
        ])
        record = self.prepare()[0]
        self.assertEqual(record["full_text_status"], "error")
        self.assertIs(record["text_complete"], False)
        self.assertEqual(record["full_text"], "")
        self.assertEqual(record["full_text_path"], "")
        self.assertIn("Multiple manifest rows", record["full_text_error"])

    def test_doi_only_manifest_cannot_choose_between_distinct_reports(self):
        path, _ = self.text_file("shared.txt")
        self.write_manifest([{
            "record_id": "", "doi": "10.9999/fulltext-fixture",
            "path": str(path), "coverage": "complete",
        }])
        records = self.prepare([self.record("R_first"), self.record("R_second")])
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["full_text_status"] == "error" for record in records))
        self.assertTrue(all(record["text_complete"] is False for record in records))
        self.assertTrue(all("multiple reports" in record["full_text_error"] for record in records))

    def test_missing_full_text_is_retained_as_not_retrieved(self):
        original = self.record()
        records = self.prepare([original])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["record_id"], original["record_id"])
        self.assertEqual(records[0]["full_text_status"], "not_retrieved")
        self.assertIs(records[0]["text_complete"], False)
        self.assertNotIn("ai_decision", records[0])
        self.assertNotIn("exclusion_reason", records[0])
        needed = self.common.read_csv(self.run / "04_full_text" / "fulltext_needed.csv")
        self.assertEqual(len(needed), 1)
        self.assertTrue(needed[0]["expected_path"].endswith("R_fixture.pdf"))
        self.assertTrue((self.run / "04_full_text" / "retrieval_attempts.jsonl").is_file())
        self.assertNotIn("full_text_status", original)  # Input record is not mutated.

    def test_malformed_and_credential_urls_are_rejected_before_dns(self):
        urls = [
            "file:///tmp/paper.pdf",
            "https://user:password@example.org/paper.pdf",
            "https://example.org:invalid/paper.pdf",
            "https://example.org/paper.pdf?token=synthetic-secret",
            "https://example.org:444/paper.pdf",
        ]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(self.fulltext.FullTextError):
                self.fulltext._checked_url(url)

    def test_private_and_mixed_dns_results_are_rejected_without_connections(self):
        def addresses(*values):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (value, 443)) for value in values]
        for values in (("127.0.0.1",), ("10.0.0.5",), ("169.254.169.254",),
                       ("93.184.216.34", "192.168.1.2")):
            with self.subTest(addresses=values), mock.patch(
                "socket.getaddrinfo", return_value=addresses(*values)
            ), self.assertRaises(self.fulltext.FullTextError):
                self.fulltext._checked_url("https://synthetic.example/paper.pdf")


if __name__ == "__main__":
    unittest.main()
