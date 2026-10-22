"""Synthetic saved-audit exports: no pipeline, SQLite, credentials, or network."""
from contextlib import ExitStack, redirect_stdout
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import export_screening_diagnostics as diagnostics


class ScreeningDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("socket.create_connection", "socket.socket.connect", "socket.getaddrinfo"):
            self.stack.enter_context(mock.patch(name, side_effect=AssertionError("Network forbidden")))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.run = self.root / "run"
        self.directory = self.run / "checkpoints/model_audit"
        self.directory.mkdir(parents=True)
        (self.run / "03_title_abstract").mkdir()
        (self.run / ".env").write_text("PRIVATE_FIXTURE=must-not-export\n")
        (self.run / "screening.sqlite").write_bytes(b"not-a-database")
        (self.run / "paper.pdf").write_bytes(b"private synthetic document")
        self.output = self.root / "diagnostics.zip"
        self.rows = []

    def add(self, record_id="R_TEST_1", status="assessed", key=None, direct=True, modified=10):
        key = key or hashlib.sha256(record_id.encode()).hexdigest()
        metadata = {"title": "Synthetic study", "authors": "Synthetic Author", "date": "2020",
                    "venue": "Synthetic Venue", "publication_type": "article", "doi": ""}
        evidence = {"record_id": record_id, "stage": "title_abstract", "metadata": metadata,
                    "passages": {"TITLE": metadata["title"], "ABSTRACT": "Synthetic abstract."}}
        values = {**metadata, "abstract": "Synthetic abstract."}
        row = {"record_id": record_id, **values, "screening_status": status,
               "ai_decision": "uncertain", "prompt_sha256": "a" * 64,
               "model": "synthetic:offline", "model_digest": "synthetic-digest",
               "input_sha256": diagnostics._digest({"stage": "title_abstract", "evidence": values})}
        if direct:
            row.update(cache_key=key, assessment_fingerprint=key)
        audit = {"record_id": record_id, "stage": "title_abstract",
                 "assessment_fingerprint": key, "prompt_sha256": row["prompt_sha256"],
                 "model": row["model"], "model_digest": row["model_digest"],
                 "input_evidence": evidence,
                 "request_event_file": "../../.env",  # Deliberately never followed.
                 "result": {"screening_status": status, "ai_decision": "uncertain",
                            "ai_rationale": "Synthetic adjusted rationale.",
                            "evidence_adjustments_json": json.dumps([{"criterion": "test", "action": "unclear",
                                                                     "reason": "synthetic reason"}])},
                 "raw_response": {"message": {"content": json.dumps({"rationale": "Synthetic raw rationale."})}}}
        path = self.directory / (key + ".json")
        path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        os.utime(path, ns=(modified, modified))
        self.rows.append(row)
        self.write_rows()
        return row, audit, path

    def write_rows(self):
        keys = list(dict.fromkeys(key for row in self.rows for key in row))
        with (self.run / "03_title_abstract/screening.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)

    def update(self, audit, path):
        path.write_text(json.dumps(audit) + "\n", encoding="utf-8")

    def export(self, **kwargs):
        return diagnostics.export_diagnostics(self.run, self.output, **kwargs)

    def test_exact_selected_bytes_summary_hash_and_source_unchanged(self):
        row, audit, path = self.add()
        before = {str(path.relative_to(self.run)): path.read_bytes()
                  for path in self.run.rglob("*") if path.is_file()}
        summary = self.export()
        self.assertEqual(summary["selected_count"], 1)
        self.assertEqual(summary["records"][0]["raw_ai_rationale"], "Synthetic raw rationale.")
        self.assertEqual(summary["records"][0]["adjustments"][0]["reason"], "synthetic reason")
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(set(archive.namelist()), {"summary.json", "model_audit/" + path.name})
            self.assertEqual(archive.read("model_audit/" + path.name), path.read_bytes())
            self.assertEqual(json.loads(archive.read("summary.json")), summary)
        self.assertEqual(summary["records"][0]["audit_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        after = {str(path.relative_to(self.run)): path.read_bytes()
                 for path in self.run.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_fallback_uses_newest_matching_evidence_and_stops_early(self):
        row, audit, old_path = self.add(direct=False, modified=10)
        old_path.write_text("invalid JSON, should never be read after newest match")
        key = "b" * 64
        audit["assessment_fingerprint"] = key
        new_path = self.directory / (key + ".json")
        self.update(audit, new_path)
        summary = self.export()
        self.assertTrue(summary["records"][0]["audit_file"].endswith(new_path.name))

    def test_fallback_skips_newer_different_prompt_or_evidence(self):
        row, audit, old_path = self.add(direct=False, modified=10)
        newer = copy.deepcopy(audit)
        newer["assessment_fingerprint"] = "b" * 64
        newer["prompt_sha256"] = "c" * 64
        newer["input_evidence"]["passages"]["ABSTRACT"] = "Old unrelated evidence."
        self.update(newer, self.directory / ("b" * 64 + ".json"))
        self.assertTrue(self.export()["records"][0]["audit_file"].endswith(old_path.name))

    def test_fallback_refuses_equally_new_matching_audits(self):
        row, audit, path = self.add(direct=False, modified=10)
        audit["assessment_fingerprint"] = "b" * 64
        other = self.directory / ("b" * 64 + ".json")
        self.update(audit, other)
        os.utime(other, ns=(10, 10))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_refuses_duplicate_requested_or_csv_record_ids(self):
        row, _, _ = self.add()
        with self.assertRaisesRegex(ValueError, "Repeated"):
            self.export(record_ids=[row["record_id"], row["record_id"]])
        self.rows.append(dict(row))
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.export()

    def test_refuses_unsafe_or_conflicting_direct_keys(self):
        row, _, _ = self.add()
        for cache_key, fingerprint in (("../../.env", "../../.env"), ("a" * 64, "b" * 64)):
            with self.subTest(cache_key=cache_key):
                row.update(cache_key=cache_key, assessment_fingerprint=fingerprint)
                self.write_rows()
                with self.assertRaisesRegex(ValueError, "Invalid or conflicting"):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_refuses_current_identity_hash_or_evidence_mismatches(self):
        row, original, path = self.add()
        for location, key, value in (("audit", "record_id", "R_DIFFERENT"),
                                     ("audit", "stage", "full_text"),
                                     ("audit", "prompt_sha256", "b" * 64),
                                     ("audit", "assessment_fingerprint", "b" * 64),
                                     ("evidence", "record_id", "R_DIFFERENT"),
                                     ("passages", "ABSTRACT", "Changed evidence")):
            with self.subTest(location=location, key=key):
                audit = copy.deepcopy(original)
                target = audit if location == "audit" else audit["input_evidence"]
                if location == "passages":
                    target = target["passages"]
                target[key] = value
                self.update(audit, path)
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.output.exists())
        self.update(original, path)
        row["input_sha256"] = "f" * 64
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "current CSV evidence"):
            self.export()

    def test_explicit_errors_allowed_default_only_assessed_and_limit_applies(self):
        self.add("R_ERROR", status="model_or_validation_error")
        self.add("R_ASSESSED")
        summary = self.export(limit=1)
        self.assertEqual([row["record_id"] for row in summary["records"]], ["R_ASSESSED"])
        self.output.unlink()
        summary = self.export(record_ids=["R_ERROR"], limit=1)
        self.assertEqual(summary["records"][0]["screening_status"], "model_or_validation_error")
        for kwargs in ({"limit": 0}, {"limit": 101}, {"record_ids": ["R_ERROR", "R_ASSESSED"], "limit": 1}):
            with self.assertRaises(ValueError):
                self.export(**kwargs)

    def test_missing_or_unassessed_selection_leaves_no_partial_bundle(self):
        row, _, _ = self.add()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.export(record_ids=[row["record_id"], "R_MISSING"])
        self.assertFalse(self.output.exists())
        row["screening_status"] = "pending_limit"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "No completed"):
            self.export(record_ids=[row["record_id"]])

    def test_refuses_ambiguous_csv_columns(self):
        self.add()
        path = self.run / "03_title_abstract/screening.csv"
        path.write_text("record_id,title,abstract,screening_status,record_id\nR_TEST_1,Title,Abstract,assessed,R_TEST_2\n")
        with self.assertRaisesRegex(ValueError, "duplicate columns"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_existing_output_requires_explicit_overwrite(self):
        self.add()
        self.output.write_bytes(b"keep existing output")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.export()
        self.assertEqual(self.output.read_bytes(), b"keep existing output")
        self.export(overwrite=True)
        self.assertTrue(zipfile.is_zipfile(self.output))

    def test_refuses_symlink_audit_and_output_inside_run(self):
        _, _, path = self.add()
        outside = self.root / "private.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks unavailable on this host")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.export()
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, "outside the source run"):
            diagnostics.export_diagnostics(self.run, self.run / "forbidden.zip")

    def test_cli_has_compact_output_and_no_pipeline_imports(self):
        self.add()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(diagnostics.main(["--run", str(self.run), "--output", str(self.output)]), 0)
        self.assertEqual(len(output.getvalue().splitlines()), 2)
        self.assertIn("database/model requests: 0", output.getvalue())
        import ast
        tree = ast.parse(Path(diagnostics.__file__).read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = [alias.name for node in imports if isinstance(node, ast.Import) for alias in node.names]
        names += [node.module or "" for node in imports if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(name.startswith(("slr_pipeline", "sqlite3", "requests", "socket", "urllib"))
                             for name in names))


if __name__ == "__main__":
    unittest.main()
