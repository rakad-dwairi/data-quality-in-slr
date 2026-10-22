"""Synthetic CLI integration checks; no real review configuration is executed.

All records, full texts, reviewers, model responses and run folders are temporary
fixtures. The complete CLI is exercised with a fake local model and hard network
guards. No databases, actual Ollama server, downloads or user run files are used.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


FIXTURE_QUOTE = (
    "This synthetic target study measures missing mandatory item metadata in a "
    "recommender dataset using missing fields divided by required fields."
)


class SyntheticOllama:
    """Deterministic schema fixtures, not inference or scientific judgements."""

    calls = 0
    calls_by_stage = Counter()
    preflights = 0

    def __init__(self, settings):
        if settings.get("name") != "synthetic:offline-fixture":
            raise AssertionError("Only the synthetic test model is allowed")
        self.settings = settings
        self.model = settings["name"]
        self.context = int(settings["context_tokens"])
        self.output = int(settings["max_output_tokens"])
        self.ready = False
        self.unavailable_error = ""
        self.model_digest = ""

    def preflight(self):
        type(self).preflights += 1
        self.model_digest = "synthetic_offline_model_digest"

    def close(self):
        pass

    def call(self, messages, schema):
        evidence = json.loads(messages[-1]["content"])
        stage = evidence["stage"]
        type(self).calls += 1
        type(self).calls_by_stage[stage] += 1
        passage_id = (
            next(key for key in evidence["passages"] if key.startswith("BODY_"))
            if stage == "full_text" else "ABSTRACT"
        )
        if FIXTURE_QUOTE not in evidence["passages"][passage_id]:
            raise AssertionError("Synthetic response must cite supplied fixture evidence")
        payload = {
            "decision": "uncertain" if evidence.get("source_packet") else "include",
            "rationale": "Synthetic target-study fixture satisfies the test criteria.",
            "study_kind": "primary",
            "criteria": [
                {"criterion": name, "value": "yes", "source_attribution": "own_study",
                 "explanation": "Synthetic explicit target-study evidence.",
                 "passage_id": passage_id, "quote": FIXTURE_QUOTE}
                for name in evidence["required_criteria"]
            ],
        }
        if stage == "full_text":
            payload.update(
                extraction=[{
                    "entity_type": "measure", "entity_label": "synthetic_missingness",
                    "fields": {"source_attribution": "own_study",
                               "author_term": "synthetic missingness",
                               "formula_or_procedure": "missing fields / required fields",
                               "reference_requirement": "synthetic mandatory item fields"},
                    "passage_id": passage_id, "quote": FIXTURE_QUOTE,
                }],
                appraisal=[{
                    "domain": "measurement_transparency", "rating": "yes",
                    "justification": "The fixture explicitly states its synthetic procedure.",
                    "passage_id": passage_id, "quote": FIXTURE_QUOTE,
                }],
                iso_mapping=[],
            )
        return {"done": True, "done_reason": "stop", "prompt_eval_count": 100,
                "message": {"content": json.dumps(payload)}}


class OfflineCLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        deny = AssertionError("Real network/model/provider traffic is forbidden in this test")
        for target in (
            "requests.sessions.Session.request", "socket.create_connection",
            "socket.getaddrinfo", "socket.socket.connect", "socket.socket.connect_ex",
            "http.client.HTTPConnection.request",
        ):
            self.stack.enter_context(mock.patch(target, side_effect=deny))

        from slr_pipeline import checks, cli, common, dedup, fulltext, retrieval, screening
        self.cli, self.common, self.retrieval = cli, common, retrieval
        self.dedup, self.fulltext, self.screening = dedup, fulltext, screening
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="slr-cli-synthetic-"))
        # macOS /var can resolve through /private/var; compare canonical paths.
        self.root = Path(temporary).resolve()
        self.out = self.root / "runs" / "synthetic_review"
        self.stack.enter_context(mock.patch.object(cli, "PROJECT", self.root))
        # Readiness metadata is also synthetic: these fixtures neither discover
        # an installed model nor claim that a provider entitlement is available.
        versions = {"requests": "2.34.2", "rapidfuzz": "3.14.6",
                    "bibtexparser": "1.4.3", "pypdf": "6.21.2"}
        actual_version = checks.importlib.metadata.version
        self.stack.enter_context(mock.patch.object(
            checks.importlib.metadata, "version",
            side_effect=lambda name: versions[name] if name in versions else actual_version(name)))
        self.stack.enter_context(mock.patch.object(screening, "OllamaClient", SyntheticOllama))
        self.stack.enter_context(mock.patch.object(
            retrieval.providers, "Client", side_effect=AssertionError("Imports-only fixtures must not create a provider client")))
        self.stack.enter_context(mock.patch.object(
            fulltext, "_download", side_effect=AssertionError("Offline fixtures must not download full texts")))
        self.stack.enter_context(mock.patch.dict(os.environ, {
            name: "" for name in (
                "OPENALEX_API_KEY", "SCOPUS_API_KEY", "SCOPUS_INST_TOKEN",
                "WOS_API_KEY", "IEEE_API_KEY",
            )
        }))
        SyntheticOllama.calls = 0
        SyntheticOllama.calls_by_stage = Counter()
        SyntheticOllama.preflights = 0
        self._make_fixture()

    def _make_fixture(self):
        source = self.root / "inputs" / "imports" / "synthetic.csv"
        row = {
            "title": "Synthetic recommender metadata quality integration fixture",
            "abstract": FIXTURE_QUOTE, "authors": "Example, Alpha; Example, Beta",
            "date": "2020-01-01", "date_basis": "synthetic_fixture",
            "doi": "10.9999/synthetic-cli-fixture", "venue": "Synthetic Fixture Venue",
            "publication_type": "journal-article", "oa_pdf_url": "",
        }
        self.common.write_csv(source, [{**row, "source_id": "fixture_one"},
                                       {**row, "source_id": "fixture_two"}])
        text = self.root / "inputs" / "fulltext" / "synthetic_report.txt"
        text.parent.mkdir(parents=True, exist_ok=True)
        text.write_text((FIXTURE_QUOTE + "\n") * 12, encoding="utf-8")
        self.common.write_csv(self.root / "inputs" / "fulltext_manifest.csv", [{
            "record_id": "", "doi": row["doi"],
            "path": "inputs/fulltext/synthetic_report.txt", "coverage": "complete",
        }], ["record_id", "doi", "path", "coverage"])
        (self.root / "synthetic_plan.txt").write_text("Synthetic protocol fixture only.", encoding="utf-8")
        self.config = {
            "run_directory": "runs/synthetic_review",
            "search": {"enabled": False, "email": "", "sources": [], "topics": [],
                       "start": "2016-01-01", "end": "2026-10-21",
                       "max_pages": 0, "crossref_rows": 1000},
            "imports": [{"source": "synthetic_manual", "path": "inputs/imports/synthetic.csv"}],
            "deduplication": {"similarity": 95, "block_limit": 200, "skip_fuzzy": False},
            "model": {"name": "synthetic:offline-fixture", "base_url": "http://127.0.0.1:11434",
                      "allow_remote": False, "context_tokens": 32768, "max_output_tokens": 4096,
                      "timeout_seconds": 10, "temperature": 0},
            "screening": {"limit": 0, "retry_errors": True, "require_abstract": True},
            "fulltext": {"directory": "inputs/fulltext", "manifest": "inputs/fulltext_manifest.csv",
                         "download_open_access": False, "min_text_chars": 1000,
                         "max_download_mb": 1, "timeout_seconds": 5},
            "reviews_file": "inputs/reviews.csv",
            "protocol": {"version": "synthetic-offline-only", "frozen": False,
                         "source_plan": "synthetic_plan.txt", "allowed_languages": [],
                         "language_restriction_justification": "", "iso_codebook": {}},
        }
        self.config_path = self.root / "synthetic_config.json"
        self.common.write_json(self.config_path, self.config)

    def command(self, stage, expected=0):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            result = self.cli.main(["--config", str(self.config_path), "--stage", stage])
        self.assertEqual(result, expected, output.getvalue() + error.getvalue())
        self.last_stderr = error.getvalue()
        return output.getvalue()

    def test_import_validation_needs_no_email_credentials_or_run_creation(self):
        self.assertFalse(self.out.exists())
        loaded = self.cli.load_config(self.config_path)
        result = self.retrieval._validate(loaded)
        self.assertFalse(result[0]["enabled"])
        self.assertEqual(result[-1], "")
        self.assertFalse(self.out.exists())
        self.assertEqual(SyntheticOllama.calls, 0)
        self.assertEqual(SyntheticOllama.preflights, 0)

    def test_cli_check_imports_only_needs_no_keys_and_writes_nothing(self):
        before = {str(path.relative_to(self.root)): self.common.file_hash(path)
                  for path in self.root.rglob("*") if path.is_file()}
        report = json.loads(self.command("check"))
        after = {str(path.relative_to(self.root)): self.common.file_hash(path)
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertTrue(report["ready"])
        self.assertEqual(report["check"], "offline_only")
        self.assertEqual(report["database_requests"], 0)
        self.assertEqual(report["model_requests"], 0)
        self.assertEqual(report["writes"], 0)
        self.assertFalse(report["access_verified"])
        self.assertEqual(report["sources"], [])
        self.assertEqual(report["errors"], [])
        self.assertEqual(before, after)
        self.assertFalse(self.out.exists())
        self.assertEqual(SyntheticOllama.calls, 0)
        self.assertEqual(SyntheticOllama.preflights, 0)

    def test_cli_check_missing_email_and_credentials_stops_without_any_stage(self):
        self.config["search"].update(enabled=True, sources=["ieee"], topics=["quality"],
                                     email="", require_all_sources=True)
        self.config["imports"] = []
        self.common.write_json(self.config_path, self.config)
        before = {str(path.relative_to(self.root)): self.common.file_hash(path)
                  for path in self.root.rglob("*") if path.is_file()}
        with ExitStack() as stages:
            mocked = [stages.enter_context(mock.patch.object(owner, name)) for owner, name in (
                (self.cli, "initialize"), (self.cli, "begin_stage"),
                (self.retrieval, "collect"), (self.dedup, "deduplicate"),
                (self.screening, "screen_records"), (self.fulltext, "prepare_fulltexts"),
            )]
            report = json.loads(self.command("check", expected=2))
            for stage in mocked:
                stage.assert_not_called()
        after = {str(path.relative_to(self.root)): self.common.file_hash(path)
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertFalse(report["ready"])
        self.assertTrue(any("contact email" in message for message in report["errors"]))
        self.assertTrue(any("IEEE_API_KEY" in message for message in report["errors"]))
        self.assertFalse(report["access_verified"])
        self.assertEqual(report["database_requests"], 0)
        self.assertEqual(report["model_requests"], 0)
        self.assertEqual(before, after)
        self.assertFalse(self.out.exists())

    def test_incomplete_mocked_retrieval_preserves_status_and_blocks_screening(self):
        self.config["search"].update(enabled=True, email="offline-fixture@example.invalid",
                                     sources=["arxiv", "crossref"], topics=["quality"],
                                     require_all_sources=True)
        self.config["imports"] = []
        self.common.write_json(self.config_path, self.config)
        observation = {"observation_id": "O_SYNTHETIC_PARTIAL", "source": "arxiv",
                       "source_id": "SYNTHETIC_ONLY", "title": "Synthetic partial retrieval fixture",
                       "abstract": FIXTURE_QUOTE, "date": "2001-01-01",
                       "doi": "10.9999/synthetic-cli-fixture"}
        logs = [
            {"source": "arxiv", "query_id": "synthetic_arxiv", "status": "api_query_exhausted", "records": 1},
            {"source": "crossref", "query_id": "synthetic_crossref", "status": "error_unfinished", "records": 0,
             "error": "Synthetic provider unavailable; no request was made"},
        ]

        def fake_collect(config, run_dir):
            self.assertEqual(Path(run_dir), self.out)
            self.assertEqual(Path(config["_project_dir"]), self.root)
            self.common.write_csv(run_dir / "01_retrieval/records.csv", [observation])
            self.common.write_csv(run_dir / "01_retrieval/search_log.csv", logs)
            self.common.write_json(run_dir / "01_retrieval/search_log.json", logs)
            self.common.write_json(run_dir / "01_retrieval/summary.json", {
                "observations": 1, "complete": False, "search_coverage_verified": False,
                "unfinished_sources_or_queries": [{"source": "crossref", "status": "error_unfinished"}],
            })
            return [observation]

        with ExitStack() as stages:
            collect = stages.enter_context(mock.patch.object(self.retrieval, "collect", side_effect=fake_collect))
            dedup = stages.enter_context(mock.patch.object(self.dedup, "deduplicate"))
            screen = stages.enter_context(mock.patch.object(self.screening, "screen_records"))
            fulltext = stages.enter_context(mock.patch.object(self.fulltext, "prepare_fulltexts"))
            self.command("all", expected=1)
            collect.assert_called_once()
            dedup.assert_not_called()
            screen.assert_not_called()
            fulltext.assert_not_called()
        self.assertIn("Retrieval is incomplete", self.last_stderr)
        self.assertEqual(self.common.read_json(self.out / "01_retrieval/search_log.json"), logs)
        self.assertEqual(self.common.read_csv(self.out / "01_retrieval/records.csv")[0]["observation_id"],
                         observation["observation_id"])
        state = self.common.read_json(self.out / "stage_state.json")
        self.assertTrue(state["retrieve"]["valid"])
        for name in ("deduplicate", "screen", "fulltext"):
            self.assertFalse(state[name]["valid"])
        self.assertFalse(self.common.read_json(self.out / "summary.json")["retrieval"]["complete"])
        self.assertEqual(SyntheticOllama.calls, 0)
        self.assertEqual(SyntheticOllama.preflights, 0)

    def test_model_outage_exports_pending_results_and_stops_before_fulltext(self):
        with mock.patch.object(SyntheticOllama, "call", side_effect=self.screening.ModelUnavailable("Synthetic offline server failure")), \
                mock.patch.object(self.fulltext, "prepare_fulltexts") as prepare:
            self.command("all", expected=1)
        prepare.assert_not_called()
        self.assertIn("model unavailable", self.last_stderr)
        rows = self.common.read_csv(self.out / "03_title_abstract/screening.csv")
        self.assertEqual(rows[0]["screening_status"], "model_unavailable")
        self.assertTrue((self.out / "checkpoints/screening.sqlite").exists())

    def test_all_stops_at_title_abstract_limit_but_explicit_screen_can_return(self):
        def pending(records, stage, config, out):
            self.assertEqual(stage, "title_abstract")
            return [{**row, **self.screening._uncertain("Synthetic call limit.", "pending_limit"),
                     "input_sha256": self.screening.screening_input_hash(row, stage)} for row in records]

        with mock.patch.object(self.screening, "screen_records", side_effect=pending), \
                mock.patch.object(self.fulltext, "prepare_fulltexts") as prepare:
            self.command("all", expected=1)
            self.assertIn("reached its configured limit", self.last_stderr)
            prepare.assert_not_called()
            self.command("screen")
            prepare.assert_not_called()

    def test_chunked_model_outage_is_detected_without_changing_uncertain_rows(self):
        row = {"screening_status": "chunked_partial", "ai_decision": "uncertain",
               "chunk_errors_json": json.dumps([{"packet_index": 2, "status": "model_unavailable"}])}
        before = json.dumps(row, sort_keys=True)
        with self.assertRaisesRegex(ValueError, "model unavailable for 1 records"):
            self.cli.require_available_model([row], "Synthetic full-text screening")
        self.assertEqual(json.dumps(row, sort_keys=True), before)

    def test_all_stages_reviews_and_resume_use_only_temporary_synthetic_evidence(self):
        sentinel = self.root / "unrelated_existing_run" / "preserve.txt"
        sentinel.parent.mkdir()
        sentinel.write_text("Synthetic unrelated-run preservation sentinel.", encoding="utf-8")
        original_sentinel = sentinel.read_bytes()
        original_input = (self.root / "inputs/imports/synthetic.csv").read_bytes()

        self.command("all")
        retrieval = self.common.read_json(self.out / "01_retrieval/summary.json")
        dedup = self.common.read_json(self.out / "02_deduplication/summary.json")
        self.assertEqual(retrieval["observations"], 2)
        self.assertEqual(dedup["duplicates_removed"], 1)
        self.assertEqual(dedup["reports_remaining"], 1)
        self.assertEqual(len(self.common.read_csv(self.out / "03_title_abstract/full_text_queue.csv")), 1)
        ft = self.common.read_csv(self.out / "04_full_text/screening.csv")
        self.assertEqual(len(ft), 1)
        self.assertEqual(ft[0]["ai_decision"], "include")
        self.assertEqual(ft[0]["full_text_status"], "ready")
        self.assertEqual(len(self.common.read_csv(self.out / "06_final/included_candidates.csv")), 1)
        self.assertEqual(self.common.read_csv(self.out / "06_final/final_included_papers.csv"), [])
        self.assertEqual(len(self.common.read_csv(self.out / "05_extraction/measures.csv")), 1)
        self.assertEqual(SyntheticOllama.calls_by_stage, {"title_abstract": 1, "full_text": 1})
        for stage in ("retrieve", "deduplicate", "screen", "fulltext"):
            self.assertTrue(self.cli.validate_stage(self.out, stage))

        # These named reviewers are synthetic fixtures, never actual review evidence.
        reviews = [{
            "stage": "full_text", "record_id": ft[0]["record_id"],
            "input_sha256": ft[0]["input_sha256"], "reviewer_id": identity,
            "role": "reviewer", "decision": "include", "reason_code": "",
            "notes": "Synthetic fixture consensus only.", "reviewed_utc": "2026-10-21T12:00:00Z",
            "study_id": "S_SYNTHETIC_CLI_FIXTURE",
        } for identity in ("synthetic_person_one", "synthetic_person_two")]
        self.common.write_csv(self.root / "inputs/reviews.csv", reviews)
        calls_before_export = SyntheticOllama.calls
        self.command("export")
        final = self.common.read_csv(self.out / "06_final/final_included_papers.csv")
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["study_id"], "S_SYNTHETIC_CLI_FIXTURE")
        self.assertEqual(final[0]["human_status"], "agreed")
        self.assertEqual(SyntheticOllama.calls, calls_before_export)

        self.command("all")
        self.assertEqual(SyntheticOllama.calls, calls_before_export)
        self.assertEqual(len(self.common.read_csv(self.out / "06_final/final_included_papers.csv")), 1)
        self.assertEqual(sentinel.read_bytes(), original_sentinel)
        self.assertEqual((self.root / "inputs/imports/synthetic.csv").read_bytes(), original_input)
        self.assertTrue(Path(ft[0]["full_text_path"]).is_relative_to(self.root))
        self.assertTrue((self.out / "checkpoints/screening.sqlite").is_file())


if __name__ == "__main__":
    unittest.main()
