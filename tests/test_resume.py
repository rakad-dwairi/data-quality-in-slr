"""Offline migration checks using temporary synthetic provider-page fixtures.

No actual provider, model, user configuration, or existing review run is used.
Synthetic responses exercise the real cache writer/parser/resume boundary.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


def openalex_page(ordinal, total=2, next_cursor=None):
    return json.dumps({
        "meta": {"count": total, "next_cursor": next_cursor},
        "results": [{
            "id": f"https://openalex.org/W900000000{ordinal}",
            "display_name": f"Synthetic cache migration fixture {ordinal}",
            "doi": f"https://doi.org/10.9999/synthetic-migration-{ordinal}",
            "publication_date": "2001-01-01", "type": "article",
            "abstract_inverted_index": {"Synthetic": [0], "fixture.": [1]},
            "authorships": [], "primary_location": {},
        }],
    })


def crossref_page(ordinal):
    return json.dumps({"message": {"total-results": 1, "items": [{
        "DOI": f"10.9999/synthetic-crossref-{ordinal}",
        "title": [f"Synthetic Crossref migration fixture {ordinal}"],
        "abstract": "Synthetic metadata used only for offline cache tests.",
        "author": [{"given": "Alpha", "family": "Example"}],
        "published": {"date-parts": [[2001, 1, 1]]},
        "container-title": ["Synthetic Fixture Venue"], "type": "journal-article",
        "URL": f"https://doi.org/10.9999/synthetic-crossref-{ordinal}",
    }]}})


def arxiv_page(first=1, count=100, total=101):
    entries = "".join(
        f"""<entry><id>http://arxiv.org/abs/0000.{ordinal:05d}v1</id>
        <title>Synthetic arXiv migration fixture {ordinal}</title>
        <summary>Synthetic metadata used only for offline cache tests.</summary>
        <author><name>Alpha Example</name></author>
        <published>2001-01-01T00:00:00Z</published>
        <link title="pdf" href="http://arxiv.org/pdf/0000.{ordinal:05d}v1"/>
        </entry>""" for ordinal in range(first, first + count)
    )
    return f"""<feed xmlns="http://www.w3.org/2005/Atom"
        xmlns:o="http://a9.com/-/spec/opensearch/1.1/"
        xmlns:x="http://arxiv.org/schemas/atom">
        <o:totalResults>{total}</o:totalResults>{entries}</feed>"""


class FakePageClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.closed = False
        self.session = SimpleNamespace(close=self.close)

    def close(self):
        self.closed = True

    def redact(self, value):
        return value

    def get(self, endpoint, params=None, headers=None):
        self.requests.append({"endpoint": endpoint, "params": dict(params or {})})
        if not self.responses:
            raise AssertionError("Unexpected synthetic provider request; cached pages should be reused")
        return SimpleNamespace(text=self.responses.pop(0), close=lambda: None)


class OfflineResumeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        denied = AssertionError("Real network, provider and model traffic is forbidden in migration tests")
        for target in (
            "requests.sessions.Session.request", "socket.create_connection", "socket.getaddrinfo",
            "socket.socket.connect", "socket.socket.connect_ex", "http.client.HTTPConnection.request",
        ):
            self.stack.enter_context(mock.patch(target, side_effect=denied))
        from slr_pipeline import cli, common, provider_library, resume, retrieval
        self.cli, self.common, self.providers = cli, common, provider_library
        self.resume, self.retrieval = resume, retrieval
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="slr-resume-synthetic-"))
        self.root = Path(temporary).resolve()
        library = self.root / "slr_pipeline"
        library.mkdir()
        # The donor records real adapter source bytes, never fabricated hashes.
        # These are passive files in a temporary project, not imported modules.
        for source in Path(cli.__file__).resolve().parent.glob("*.py"):
            shutil.copyfile(source, library / source.name)
        self.stack.enter_context(mock.patch.object(cli, "PROJECT", self.root))
        self.stack.enter_context(mock.patch.object(
            self.providers, "Client", side_effect=AssertionError("Migration must never construct a provider client")))
        self.stack.enter_context(mock.patch.dict(os.environ, {"OPENALEX_API_KEY": "SYNTHETIC_FIXTURE_ONLY"}))
        self.donor = self.root / "runs" / "donor"
        self.target = self.root / "runs" / "target"
        (self.root / "synthetic_plan.txt").write_text("Synthetic migration protocol only.", encoding="utf-8")
        self.old_config = {
            "_project_dir": str(self.root), "run_directory": "runs/donor",
            "search": {"enabled": True, "email": "fixture@example.invalid",
                       "sources": ["openalex"], "topics": ["quality"],
                       "start": "2016-01-01", "end": "2026-10-21",
                       "max_pages": 1, "crossref_rows": 1000,
                       "scopus_pagination": "cursor", "scopus_view": "STANDARD"},
            "imports": [], "deduplication": {"similarity": 95, "block_limit": 200},
            "model": {"name": "synthetic:never-called", "base_url": "http://127.0.0.1:11434",
                      "context_tokens": 32768, "max_output_tokens": 4096},
            "screening": {"limit": 0, "retry_errors": True},
            "fulltext": {"directory": "inputs/fulltext", "download_open_access": False},
            "protocol": {"version": "synthetic-migration", "source_plan": "synthetic_plan.txt"},
        }
        self.new_config = copy.deepcopy(self.old_config)
        self.new_config["run_directory"] = "runs/target"
        self.new_config["search"].update(max_pages=0, scopus_pagination="offset")

    def snapshot(self, folder):
        return {str(path.relative_to(folder)): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def collect(self, config, directory, responses):
        client = FakePageClient(responses)
        with mock.patch.object(self.providers, "Client", return_value=client), redirect_stdout(io.StringIO()):
            rows = self.retrieval.collect(config, directory)
        self.assertTrue(client.closed)
        self.assertEqual(client.responses, [])
        return rows, client

    def make_donor(self, complete=False):
        config = copy.deepcopy(self.old_config)
        responses = [openalex_page(1, next_cursor="SYNTHETIC_NEXT_CURSOR")]
        if complete:
            config["search"]["max_pages"] = 0
            responses.append(openalex_page(2, next_cursor="SYNTHETIC_END_CURSOR"))
        with self.common.run_lock(self.donor):
            self.cli.initialize(config, self.donor)
            rows, client = self.collect(config, self.donor, responses)
        self.assertEqual(len(rows), 2 if complete else 1)
        self.assertEqual(len(client.requests), 2 if complete else 1)
        # Derived outputs must never cross into a newly prepared run.
        (self.donor / "03_title_abstract").mkdir()
        (self.donor / "03_title_abstract/screening.csv").write_text("Synthetic derived output must not migrate.", encoding="utf-8")
        (self.donor / "stage_state.json").write_text('{"screen":{"valid":true}}', encoding="utf-8")
        (self.donor / "checkpoints").mkdir()
        (self.donor / "checkpoints/screening.sqlite").write_bytes(b"Synthetic non-database sentinel")

    def page_files(self, folder):
        return sorted((folder / "01_retrieval/raw").glob("*/page_*.json"))

    def migrate(self):
        # Transfer preparation validates saved evidence without obtaining or
        # loading any provider credential, even the synthetic collection key.
        with mock.patch("os.getenv", side_effect=AssertionError("Migration must not read environment credentials")), \
                mock.patch.object(self.cli, "load_env", side_effect=AssertionError("Migration must not load .env")), \
                mock.patch("slr_pipeline.checks.env_values", side_effect=AssertionError("Migration must not read .env")):
            return self.resume.prepare_resume(self.new_config, self.donor)

    def assert_refused_without_target(self, message=None):
        before = self.snapshot(self.donor)
        with self.assertRaisesRegex(ValueError, message or "."):
            self.migrate()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.snapshot(self.donor), before)
        self.assertEqual(list((self.root / "runs").glob(".resume-preparing-*")), [])

    def test_capped_openalex_cache_is_copied_and_resumes_only_at_next_cursor(self):
        self.make_donor()
        donor_before = self.snapshot(self.donor)
        pages = self.page_files(self.donor)
        report = self.migrate()
        self.assertEqual(report["copied_pages"], 1)
        self.assertEqual(report["copied_observations"], 1)
        self.assertEqual(report["database_requests"], 0)
        self.assertEqual(report["model_requests"], 0)
        self.assertFalse(report["scopus_cache_reused"])
        self.assertEqual(self.snapshot(self.donor), donor_before)
        self.assertEqual(self.common.read_json(self.target / "cache_migration.json"), report)
        self.assertTrue((self.target / "manifest.json").is_file())
        self.assertTrue((self.target / "code_snapshot/provider_library.py").is_file())
        self.assertFalse((self.target / "stage_state.json").exists())
        self.assertFalse((self.target / "03_title_abstract").exists())
        self.assertFalse((self.target / "checkpoints").exists())
        for page in pages:
            self.assertEqual((self.target / page.relative_to(self.donor)).read_bytes(), page.read_bytes())
        rows, client = self.collect(self.new_config, self.target, [openalex_page(2, next_cursor="SYNTHETIC_END_CURSOR")])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["params"]["cursor"], "SYNTHETIC_NEXT_CURSOR")
        logs = self.common.read_json(self.target / "01_retrieval/search_log.json")
        self.assertEqual(logs[0]["status"], "api_query_exhausted")
        self.assertEqual(logs[0]["cached_pages"], 1)
        self.assertEqual(self.snapshot(self.donor), donor_before)

    def test_completed_donor_cache_needs_no_new_provider_requests(self):
        self.make_donor(complete=True)
        self.migrate()
        rows, client = self.collect(self.new_config, self.target, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(client.requests, [])
        self.assertEqual(self.common.read_json(self.target / "01_retrieval/search_log.json")[0]["cached_pages"], 2)

    def test_completed_crossref_discovery_pages_migrate_byte_identically_without_new_requests(self):
        self.old_config["search"]["sources"] = ["crossref"]
        self.new_config["search"]["sources"] = ["crossref"]
        jobs = self.providers.jobs_for(["crossref"], self.old_config["search"]["topics"])
        with self.common.run_lock(self.donor):
            self.cli.initialize(self.old_config, self.donor)
            rows, client = self.collect(self.old_config, self.donor,
                                        [crossref_page(index) for index in range(1, len(jobs) + 1)])
        self.assertEqual(len(rows), len(jobs))
        self.assertEqual(len(client.requests), len(jobs))
        before = self.snapshot(self.donor)
        pages = self.page_files(self.donor)
        report = self.migrate()
        self.assertEqual(report["copied_pages"], len(jobs))
        self.assertEqual(report["copied_observations"], len(jobs))
        for page in pages:
            self.assertEqual((self.target / page.relative_to(self.donor)).read_bytes(), page.read_bytes())
        resumed, resumed_client = self.collect(self.new_config, self.target, [])
        self.assertEqual(len(resumed), len(jobs))
        self.assertEqual(resumed_client.requests, [])
        logs = self.common.read_json(self.target / "01_retrieval/search_log.json")
        self.assertTrue(all(row["status"] == "ranked_discovery_returned" and row["cached_pages"] == 1 for row in logs))
        # Ranked discovery completion must not turn into verified search closure.
        self.assertFalse(self.common.read_json(self.target / "01_retrieval/summary.json")["search_coverage_verified"])
        self.assertEqual(self.snapshot(self.donor), before)

    def test_partial_arxiv_cache_migrates_and_requests_only_the_next_offset(self):
        self.old_config["search"]["sources"] = ["arxiv"]
        self.new_config["search"]["sources"] = ["arxiv"]
        with self.common.run_lock(self.donor):
            self.cli.initialize(self.old_config, self.donor)
            rows, client = self.collect(self.old_config, self.donor, [arxiv_page()])
        self.assertEqual(len(rows), 100)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["params"]["start"], 0)
        before = self.snapshot(self.donor)
        page = self.page_files(self.donor)[0]
        report = self.migrate()
        self.assertEqual(report["copied_pages"], 1)
        self.assertEqual(report["copied_observations"], 100)
        self.assertEqual((self.target / page.relative_to(self.donor)).read_bytes(), page.read_bytes())
        resumed, resumed_client = self.collect(self.new_config, self.target, [arxiv_page(first=101, count=1)])
        self.assertEqual(len(resumed), 101)
        self.assertEqual(len(resumed_client.requests), 1)
        self.assertEqual(resumed_client.requests[0]["params"]["start"], 100)
        self.assertEqual(resumed_client.requests[0]["params"]["max_results"], 100)
        logs = self.common.read_json(self.target / "01_retrieval/search_log.json")
        self.assertEqual(logs[0]["status"], "api_query_exhausted")
        self.assertEqual(logs[0]["cached_pages"], 1)
        self.assertEqual(self.snapshot(self.donor), before)

    def test_active_source_is_refused_without_destination_or_source_changes(self):
        self.make_donor()
        with self.common.run_lock(self.donor):
            self.assert_refused_without_target("still active")

    def test_missing_source_lock_is_refused_without_recreating_it(self):
        self.make_donor()
        (self.donor / ".lock").unlink()
        self.assert_refused_without_target("original .lock")
        self.assertFalse((self.donor / ".lock").exists())

    def test_existing_destination_is_never_overwritten(self):
        self.make_donor()
        self.target.mkdir()
        sentinel = self.target / "preserve.txt"
        sentinel.write_text("Synthetic destination already exists.", encoding="utf-8")
        target_before, donor_before = self.snapshot(self.target), self.snapshot(self.donor)
        with self.assertRaisesRegex(ValueError, "Destination already exists"):
            self.migrate()
        self.assertEqual(self.snapshot(self.target), target_before)
        self.assertEqual(self.snapshot(self.donor), donor_before)

    def test_normalized_record_tampering_is_rejected_against_raw_response(self):
        self.make_donor()
        page = self.page_files(self.donor)[0]
        saved = json.loads(page.read_text(encoding="utf-8"))
        saved["records"][0]["title"] = "Tampered synthetic normalized title"
        self.common.write_json(page, saved)
        self.assert_refused_without_target("differ from raw response")

    def test_missing_first_committed_page_is_not_a_resumable_cursor_chain(self):
        self.make_donor(complete=True)
        self.page_files(self.donor)[0].unlink()
        self.assert_refused_without_target("Noncontiguous")

    def test_incorrect_request_cursor_is_rejected(self):
        self.make_donor(complete=True)
        page = self.page_files(self.donor)[1]
        saved = json.loads(page.read_text(encoding="utf-8"))
        saved["request_position"] = "SYNTHETIC_WRONG_CURSOR"
        self.common.write_json(page, saved)
        self.assert_refused_without_target("cursor mismatch")

    def test_malformed_raw_response_is_controlled_and_not_copied(self):
        self.make_donor()
        page = self.page_files(self.donor)[0]
        saved = json.loads(page.read_text(encoding="utf-8"))
        saved["raw_response"] = "not-json: SYNTHETIC_UNTRUSTED_RESPONSE_MUST_NOT_APPEAR"
        self.common.write_json(page, saved)
        before = self.snapshot(self.donor)
        with self.assertRaises(ValueError) as failure:
            self.migrate()
        self.assertNotIn("SYNTHETIC_UNTRUSTED_RESPONSE_MUST_NOT_APPEAR", str(failure.exception))
        self.assertFalse(self.target.exists())
        self.assertEqual(self.snapshot(self.donor), before)
        self.assertEqual(list((self.root / "runs").glob(".resume-preparing-*")), [])

    def test_logged_committed_pages_cannot_disappear_with_the_entire_folder(self):
        self.make_donor()
        shutil.rmtree(self.page_files(self.donor)[0].parent)
        self.assert_refused_without_target("Committed pages.*missing")

    def test_uncommitted_temporary_response_is_never_migrated(self):
        self.make_donor()
        page_dir = self.page_files(self.donor)[0].parent
        pending = page_dir / "page_000002.json.tmp"
        pending.write_text("Uncommitted synthetic response, not a complete cache page.", encoding="utf-8")
        before = self.snapshot(self.donor)
        report = self.migrate()
        self.assertEqual(report["copied_pages"], 1)
        self.assertFalse((self.target / pending.relative_to(self.donor)).exists())
        self.assertEqual(self.snapshot(self.donor), before)

    def test_changed_non_scopus_search_date_is_rejected(self):
        self.make_donor()
        self.new_config["search"]["start"] = "1901-01-01"
        self.assert_refused_without_target("source/query/date settings unchanged")

    def test_changed_current_provider_hash_is_rejected(self):
        self.make_donor()
        changed_adapter = self.root / "changed_provider_fixture.py"
        changed_adapter.write_text("# Synthetic altered adapter bytes; never imported.\n", encoding="utf-8")
        with mock.patch.object(self.providers, "__file__", str(changed_adapter)):
            self.assert_refused_without_target("Provider adapter/cache format changed")

    def test_changed_frozen_provider_snapshot_is_rejected(self):
        self.make_donor()
        frozen = self.donor / "code_snapshot/provider_library.py"
        frozen.write_bytes(frozen.read_bytes() + b"\n# Synthetic snapshot tampering.\n")
        self.assert_refused_without_target("snapshot is missing or changed")

    def test_manual_imports_are_not_silently_migrated(self):
        self.make_donor()
        self.new_config["imports"] = [{"source": "synthetic_manual", "path": "unused.csv"}]
        self.assert_refused_without_target("without manual imports")


if __name__ == "__main__":
    unittest.main()
