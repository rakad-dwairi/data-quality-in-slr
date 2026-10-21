"""Provider contracts and retrieval checkpoints, using synthetic responses only.

No database searches or paper/model requests are made. Both Requests and socket
connection entry points are blocked unless a test replaces the provider client
with its own in-memory response fixture.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class ProviderTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from slr_pipeline import provider_library, retrieval
        cls.providers = provider_library
        cls.retrieval = retrieval

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recsys-provider-fixture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for target in ("requests.sessions.Session.request", "socket.socket.connect", "socket.create_connection"):
            patcher = mock.patch(target, side_effect=AssertionError("External network is forbidden in provider tests"))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.environment = mock.patch.dict(os.environ, {
            "OPENALEX_API_KEY": "fixture-openalex", "SCOPUS_API_KEY": "fixture-scopus",
            "WOS_API_KEY": "fixture-wos", "IEEE_API_KEY": "fixture-ieee",
            "SCOPUS_INST_TOKEN": "fixture-institution",
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.args = SimpleNamespace(start=date(2000, 2, 3), end=date(2001, 4, 5),
                                    email="fixture@example.invalid", crossref_rows=123)
        self.query = '("recommender systems" OR "collaborative filtering") AND ("data quality")'

    def config(self, source="scopus", **search_changes):
        search = {"enabled": True, "sources": [source], "topics": ["quality"],
                  "start": "2000-01-01", "end": "2001-12-31", "max_pages": 0,
                  "crossref_rows": 1000, "email": "fixture@example.invalid"}
        search.update(search_changes)
        return {"_project_dir": str(self.root), "search": search, "imports": []}

    def fixture(self, source, number=1, total=1, cursor=None):
        """Small schema-valid fixtures; no real paper metadata or credentials."""
        title = f"Synthetic recommendation data fixture {number}"
        abstract = "Synthetic data-quality evidence used only in offline tests."
        doi = f"10.9999/provider-fixture-{number}"
        if source == "crossref":
            data = {"message": {"total-results": total, "items": [{
                "DOI": doi, "title": [title], "abstract": "<jats:p>" + abstract + "</jats:p>",
                "author": [{"given": "Alpha", "family": "Fixture"}],
                "published": {"date-parts": [[2001, 2, 3]]},
                "container-title": ["Synthetic Journal"], "type": "journal-article",
                "URL": "https://example.invalid/report"}]}}
        elif source == "openalex":
            data = {"meta": {"count": total, "next_cursor": cursor}, "results": [{
                "id": f"https://openalex.org/W{number}", "doi": "https://doi.org/" + doi,
                "display_name": title, "publication_date": "2001-02-03", "type": "article",
                "abstract_inverted_index": {word: [i] for i, word in enumerate(abstract.split())},
                "authorships": [{"author": {"display_name": "Alpha Fixture"}}],
                "primary_location": {"source": {"display_name": "Synthetic Journal"},
                                     "landing_page_url": "https://example.invalid/report"},
                "best_oa_location": {"is_oa": True, "pdf_url": "https://example.invalid/report.pdf"}}]}
        elif source == "scopus":
            data = {"search-results": {"opensearch:totalResults": str(total), "cursor": {"@next": cursor}, "entry": [{
                "eid": f"2-s2.0-{number}", "dc:title": title, "dc:creator": "Alpha Fixture",
                "prism:doi": doi, "prism:coverDate": "2001-02-03", "dc:description": abstract,
                "prism:publicationName": "Synthetic Journal", "subtypeDescription": "Article",
                "link": [{"@ref": "scopus", "@href": "https://example.invalid/report"}]}]}}
        elif source == "wos":
            data = {"metadata": {"total": total}, "hits": [{
                "uid": f"WOS:{number}", "title": title, "identifiers": {"doi": doi},
                "names": {"authors": [{"displayName": "Alpha Fixture"}]},
                "source": {"publishYear": 2001, "sourceTitle": "Synthetic Journal"},
                "types": ["Article"], "links": {"record": "https://example.invalid/report"}}]}
        elif source == "ieee":
            data = {"total_records": total, "articles": [{
                "article_number": number, "title": title, "doi": doi, "abstract": abstract,
                "authors": {"authors": [{"full_name": "Alpha Fixture"}]},
                "publication_year": 2001, "publication_title": "Synthetic Journal",
                "content_type": "Journals", "abstract_url": "https://example.invalid/report"}]}
        elif source == "arxiv":
            return f'''<?xml version="1.0"?>
            <feed xmlns="http://www.w3.org/2005/Atom"
             xmlns:o="http://a9.com/-/spec/opensearch/1.1/" xmlns:x="http://arxiv.org/schemas/atom">
             <o:totalResults>{total}</o:totalResults><entry>
              <id>http://arxiv.org/abs/0000.{number:05d}v2</id><title>{title}</title>
              <summary>{abstract}</summary><author><name>Alpha Fixture</name></author>
              <published>2001-02-03T00:00:00Z</published><x:doi>{doi}</x:doi>
              <link title="pdf" href="http://arxiv.org/pdf/0000.{number:05d}v2"/>
             </entry></feed>'''
        else:
            raise AssertionError("Unknown test source")
        return json.dumps(data)

    def client(self, responses):
        client = mock.Mock()
        client.redact.side_effect = lambda text: text
        client.get.side_effect = [SimpleNamespace(text=text, close=mock.Mock())
                                  if isinstance(text, str) else text for text in responses]
        return client

    def collect(self, config, client, run=None):
        with mock.patch.object(self.providers, "Client", return_value=client), contextlib.redirect_stdout(io.StringIO()):
            return self.retrieval.collect(config, run or self.root / "run")

    def summary(self, run=None):
        return json.loads(((run or self.root / "run") / "01_retrieval/summary.json").read_text())

    def logs(self, run=None):
        return json.loads(((run or self.root / "run") / "01_retrieval/search_log.json").read_text())


class RequestContractTests(ProviderTestCase):
    def test_crossref_is_ranked_date_filtered_discovery(self):
        params, headers = self.providers.request_spec("crossref", self.query, None, self.args)
        self.assertEqual(self.providers.ENDPOINTS["crossref"], "https://api.crossref.org/works")
        self.assertEqual(params["query.bibliographic"], self.query)
        self.assertEqual(params["rows"], 123)
        self.assertEqual(params["filter"], "from-pub-date:2000-02-03,until-pub-date:2001-04-05")
        self.assertEqual(params["mailto"], "fixture@example.invalid")
        self.assertEqual((params["sort"], params["order"]), ("relevance", "desc"))
        self.assertNotIn("apikey", params)
        self.assertNotIn("Authorization", headers)

    def test_arxiv_uses_submission_window_title_abstract_and_zero_offset(self):
        job = self.providers.jobs_for(["arxiv"], ["quality"])[0]
        params, headers = self.providers.request_spec("arxiv", job["query"], None, self.args)
        self.assertEqual(self.providers.ENDPOINTS["arxiv"], "https://export.arxiv.org/api/query")
        self.assertIn('ti:"recommender system"', params["search_query"])
        self.assertIn('abs:"data quality"', params["search_query"])
        self.assertIn("submittedDate:[200002030000 TO 200104052359]", params["search_query"])
        self.assertEqual((params["start"], params["max_results"]), (0, 100))
        self.assertEqual(headers["Accept"], "application/atom+xml")
        self.assertEqual(self.providers.request_spec("arxiv", job["query"], 100, self.args)[0]["start"], 100)

    def test_openalex_preserves_locally_proven_title_abstract_search_contract(self):
        params, headers = self.providers.request_spec("openalex", self.query, None, self.args)
        self.assertEqual(self.providers.ENDPOINTS["openalex"], "https://api.openalex.org/works")
        self.assertEqual(params["search.title_and_abstract"], self.query)
        self.assertEqual(params["filter"], "from_publication_date:2000-02-03,to_publication_date:2001-04-05")
        self.assertEqual(params["cursor"], "*")
        self.assertEqual(headers["Authorization"], "Bearer fixture-openalex")
        self.assertEqual(self.providers.request_spec("openalex", self.query, "fixture-cursor", self.args)[0]["cursor"], "fixture-cursor")

    def test_scopus_is_fielded_with_inclusive_years_and_institution_header(self):
        self.args.scopus_pagination = "offset"
        params, headers = self.providers.request_spec("scopus", self.query, None, self.args)
        self.assertEqual(self.providers.ENDPOINTS["scopus"], "https://api.elsevier.com/content/search/scopus")
        self.assertEqual(params["query"], f"TITLE-ABS-KEY({self.query}) AND PUBYEAR > 1999 AND PUBYEAR < 2002")
        self.assertEqual((params["start"], params["count"], params["view"]), (0, 25, "STANDARD"))
        self.assertEqual(headers["X-ELS-APIKey"], "fixture-scopus")
        self.assertEqual(headers["X-ELS-Insttoken"], "fixture-institution")
        self.assertEqual(self.providers.request_spec("scopus", self.query, 25, self.args)[0]["start"], 25)

    def test_scopus_cursor_default_has_no_offset_and_complete_view_is_configurable(self):
        params, _ = self.providers.request_spec("scopus", self.query, None, self.args)
        self.assertEqual(params["cursor"], "*")
        self.assertNotIn("start", params)
        self.args.scopus_view = "COMPLETE"
        params, _ = self.providers.request_spec("scopus", self.query, "fixture-next", self.args)
        self.assertEqual((params["cursor"], params["view"], params["count"]), ("fixture-next", "COMPLETE", 25))

    def test_scopus_offset_never_requests_past_5000_result_boundary(self):
        self.args.scopus_pagination = "offset"
        params, _ = self.providers.request_spec("scopus", self.query, 4999, self.args)
        self.assertEqual((params["start"], params["count"]), (4999, 1))
        with self.assertRaisesRegex(self.providers.APIError, "offset limit"):
            self.providers.request_spec("scopus", self.query, 5000, self.args)

    def test_wos_starter_uses_topic_query_years_and_one_based_pages(self):
        params, headers = self.providers.request_spec("wos", self.query, None, self.args)
        self.assertEqual(self.providers.ENDPOINTS["wos"], "https://api.clarivate.com/apis/wos-starter/v1/documents")
        self.assertEqual(params["q"], f"TS=({self.query}) AND PY=(2000-2001)")
        self.assertEqual((params["db"], params["limit"], params["page"]), ("WOS", 50, 1))
        self.assertEqual(headers["X-ApiKey"], "fixture-wos")
        self.assertEqual(self.providers.request_spec("wos", self.query, 2, self.args)[0]["page"], 2)

    def test_ieee_has_year_filters_and_one_based_record_offset(self):
        params, headers = self.providers.request_spec("ieee", self.query, None, self.args)
        self.assertEqual(self.providers.ENDPOINTS["ieee"], "https://ieeexploreapi.ieee.org/api/v1/search/articles")
        self.assertEqual(params["querytext"], self.query)
        self.assertEqual((params["start_year"], params["end_year"]), (2000, 2001))
        self.assertEqual((params["start_record"], params["max_records"]), (1, 100))
        self.assertEqual(params["apikey"], "fixture-ieee")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(self.providers.request_spec("ieee", self.query, 101, self.args)[0]["start_record"], 101)

    def test_crossref_anchors_are_distinct_and_include_collaborative_filtering(self):
        jobs = self.providers.jobs_for(["crossref"], ["quality"])
        queries = [job["query"] for job in jobs]
        self.assertEqual(len(queries), len(set(queries)), "Duplicate discovery queries waste retrieval quota")
        for anchor in ("recommender systems", "recommendation systems", "collaborative filtering", "item recommendation"):
            self.assertIn(anchor + " data quality", queries)
        self.assertEqual(len({job["id"] for job in jobs}), len(jobs))

    def test_non_crossref_sources_get_all_configured_topics_without_iso_gate(self):
        sources = [source for source in self.providers.ENDPOINTS if source != "crossref"]
        jobs = self.providers.jobs_for(sources, list(self.providers.TOPICS))
        self.assertEqual(len(jobs), len(sources) * len(self.providers.TOPICS))
        for job in jobs:
            self.assertIn(" AND ", job["query"])
            self.assertIn("recommender system", job["query"])
            if job["topic"] != "iso":
                self.assertNotIn("5259", job["query"])


class ResponseContractTests(ProviderTestCase):
    def test_all_six_response_schemas_normalize_metadata_and_preserve_missing_abstracts(self):
        for source in self.providers.ENDPOINTS:
            with self.subTest(source=source):
                rows, total, cursor = self.providers.unpack(source, self.fixture(source, total=2, cursor="next-fixture"))
                self.assertEqual(len(rows), 1)
                self.assertEqual(total, 2)
                self.assertEqual(rows[0]["title"], "Synthetic recommendation data fixture 1")
                self.assertEqual(rows[0]["authors"], "Alpha Fixture")
                self.assertTrue(rows[0]["source_id"])
                if source == "arxiv":
                    self.assertEqual(rows[0]["doi"], "")
                    self.assertEqual(rows[0]["related_doi"], "10.9999/provider-fixture-1")
                    self.assertEqual(rows[0]["source_id"], "0000.00001")
                    self.assertEqual(rows[0]["publication_type"], "preprint")
                    self.assertTrue(rows[0]["oa_pdf_url"].startswith("https://"))
                else:
                    self.assertEqual(rows[0]["doi"], "10.9999/provider-fixture-1")
                self.assertEqual(rows[0]["date"], "2001" if source in ("wos", "ieee") else "2001-02-03")
                self.assertEqual(bool(rows[0]["abstract"]), source != "wos")
                self.assertEqual(cursor, "next-fixture" if source in ("openalex", "scopus") else None)

    def test_arxiv_error_feed_and_scopus_error_entry_do_not_become_empty_success(self):
        with self.assertRaises(self.providers.APIError):
            self.providers.unpack("arxiv", '<feed xmlns="http://www.w3.org/2005/Atom"/>')
        with self.assertRaises(self.providers.APIError):
            self.providers.unpack("scopus", json.dumps({"search-results": {"opensearch:totalResults": "1", "entry": [{"error": "fixture error"}]}}))

    def test_clean_preserves_comparisons_and_removes_named_xml_tags(self):
        self.assertEqual(self.providers.clean("x < 3 and y > 2"), "x < 3 and y > 2")
        self.assertEqual(self.providers.clean('<jats:p>A &amp; B <i>data</i></jats:p>'), "A & B data")

    def test_client_retries_429_without_real_sleep_or_network(self):
        unavailable = SimpleNamespace(status_code=429, headers={"Retry-After": "1"}, close=mock.Mock())
        success = SimpleNamespace(status_code=200, ok=True)
        client = self.providers.Client("fixture@example.invalid")
        self.addCleanup(client.session.close)
        with mock.patch.object(client.session, "get", side_effect=[unavailable, success]) as get, mock.patch.object(self.providers.time, "sleep"):
            self.assertIs(client.get("https://example.invalid/fixture"), success)
        self.assertEqual(get.call_count, 2)
        unavailable.close.assert_called_once()

    def test_client_redacts_secret_from_403_message(self):
        response = SimpleNamespace(status_code=403, ok=False, text="Denied fixture-ieee", close=mock.Mock())
        client = self.providers.Client("fixture@example.invalid")
        self.addCleanup(client.session.close)
        with mock.patch.object(client.session, "get", return_value=response), mock.patch.object(self.providers.time, "sleep"):
            with self.assertRaises(self.providers.APIError) as caught:
                client.get("https://example.invalid/fixture")
        self.assertNotIn("fixture-ieee", str(caught.exception))
        self.assertIn("[REDACTED]", str(caught.exception))
        self.assertTrue(caught.exception.halt_source)


class RetrievalCheckpointTests(ProviderTestCase):
    def test_five_paginated_sources_use_correct_next_position_and_cached_resume(self):
        for source in ("arxiv", "openalex", "scopus", "wos", "ieee"):
            with self.subTest(source=source):
                run = self.root / source
                first = self.client([self.fixture(source, 1, total=2, cursor="fixture-next"), self.fixture(source, 2, total=2)])
                config = self.config(source)
                rows = self.collect(config, first, run)
                self.assertEqual(len(rows), 2)
                self.assertEqual(first.get.call_count, 2)
                calls = first.get.call_args_list
                self.assertEqual(calls[0].args[0], self.providers.ENDPOINTS[source])
                next_params = calls[1].kwargs["params"]
                if source in ("openalex", "scopus"):
                    self.assertEqual(next_params["cursor"], "fixture-next")
                    self.assertNotIn("start", next_params)
                elif source == "wos":
                    self.assertEqual(next_params["page"], 2)
                else:
                    self.assertEqual(next_params["start_record" if source == "ieee" else "start"], 2 if source == "ieee" else 1)
                self.assertTrue(self.summary(run)["complete"])
                self.assertFalse(self.summary(run)["search_coverage_verified"])
                cached_only = self.client([])
                cached_only.get.side_effect = AssertionError("Completed pages must not be refetched")
                replayed = self.collect(config, cached_only, run)
                self.assertEqual(replayed, rows)
                cached_only.get.assert_not_called()
                self.assertEqual(self.logs(run)[0]["cached_pages"], 2)

    def test_increasing_page_cap_reuses_first_page_and_finishes_remaining_page(self):
        config = self.config("scopus", max_pages=1)
        first = self.client([self.fixture("scopus", 1, total=2, cursor="fixture-next")])
        initial = self.collect(config, first)
        self.assertEqual(len(initial), 1)
        self.assertFalse(self.summary()["complete"])
        self.assertEqual(self.logs()[0]["status"], "capped_unfinished")
        config["search"]["max_pages"] = 0
        remaining = self.client([self.fixture("scopus", 2, total=2)])
        finished = self.collect(config, remaining)
        self.assertEqual(finished[0], initial[0])
        self.assertEqual(len(finished), 2)
        self.assertEqual(remaining.get.call_count, 1)
        self.assertTrue(self.summary()["complete"])

    def test_repeated_page_keeps_prior_evidence_and_marks_incomplete(self):
        body = self.fixture("scopus", 1, total=2, cursor="fixture-next")
        client = self.client([body, body])
        rows = self.collect(self.config("scopus"), client)
        self.assertEqual(len(rows), 1)
        self.assertFalse(self.summary()["complete"])
        self.assertIn("Repeated page", self.logs()[0]["error"])
        self.assertEqual(len(list((self.root / "run/01_retrieval/raw").glob("*/page_*.json"))), 1)

    def test_missing_key_is_retryable_and_does_not_destroy_other_source_records(self):
        config = self.config("scopus", sources=["scopus", "ieee"])
        os.environ.pop("IEEE_API_KEY")
        first = self.client([self.fixture("scopus")])
        rows = self.collect(config, first)
        self.assertEqual(len(rows), 1)
        self.assertFalse(self.summary()["complete"])
        self.assertIn("IEEE_API_KEY", self.logs()[1]["error"])
        os.environ["IEEE_API_KEY"] = "fixture-ieee-restored"
        retry = self.client([self.fixture("ieee", number=2)])
        recovered = self.collect(config, retry)
        self.assertEqual(len(recovered), 2)
        self.assertEqual(retry.get.call_count, 1)
        self.assertEqual(recovered[0], rows[0])
        self.assertTrue(self.summary()["complete"])

    def test_provider_failure_after_first_page_can_resume_without_refetch(self):
        first = self.client([self.fixture("scopus", 1, total=2, cursor="fixture-next"), self.providers.APIError("Fixture provider unavailable", True)])
        rows = self.collect(self.config("scopus"), first)
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.logs()[0]["status"], "error_unfinished")
        retry = self.client([self.fixture("scopus", 2, total=2)])
        recovered = self.collect(self.config("scopus"), retry)
        self.assertEqual(len(recovered), 2)
        self.assertEqual(retry.get.call_count, 1)
        self.assertEqual(recovered[0], rows[0])

    def test_crossref_discovery_never_claims_exhaustive_search(self):
        config = self.config("crossref")
        job = {"id": "crossref_fixture", "source": "crossref", "topic": "quality", "query": "synthetic ranked discovery"}
        client = self.client([self.fixture("crossref", total=50)])
        with mock.patch.object(self.providers, "jobs_for", return_value=[job]):
            rows = self.collect(config, client)
        self.assertEqual(len(rows), 1)
        self.assertEqual(client.get.call_count, 1)
        self.assertEqual(self.logs()[0]["status"], "ranked_discovery_capped")
        self.assertFalse(self.summary()["complete"])
        self.assertFalse(self.summary()["search_coverage_verified"])

    def test_scopus_single_year_overflow_is_incomplete_and_never_requests_a_second_page(self):
        client = self.client(["synthetic offset boundary fixture"])
        rows = [self.providers.record(source_id=str(number), title="Synthetic offset boundary row")
                for number in range(5000)]
        with mock.patch.object(self.providers, "unpack", return_value=(rows, 5001, None)):
            result = self.collect(self.config("scopus", scopus_pagination="offset",
                                              start="2000-01-01", end="2000-12-31"), client)
        self.assertEqual(len(result), 5000)
        self.assertEqual(client.get.call_count, 1)
        self.assertEqual(self.logs()[0]["status"], "single_year_overflow_unfinished")
        self.assertFalse(self.summary()["complete"])

    def test_scopus_missing_or_repeated_cursor_remains_retryable_and_incomplete(self):
        initial = self.client([self.fixture("scopus", 1, total=3, cursor="next-one"),
                               self.fixture("scopus", 2, total=3, cursor="next-one")])
        rows = self.collect(self.config("scopus"), initial)
        self.assertEqual(len(rows), 1)
        self.assertIn("Missing/repeated cursor", self.logs()[0]["error"])
        self.assertFalse(self.summary()["complete"])
        retry = self.client([self.fixture("scopus", 2, total=3, cursor="next-two"),
                             self.fixture("scopus", 3, total=3)])
        recovered = self.collect(self.config("scopus"), retry)
        self.assertEqual(len(recovered), 3)
        self.assertEqual(retry.get.call_count, 2)
        self.assertTrue(self.summary()["complete"])

    def test_scopus_cache_identity_changes_with_view_and_pagination(self):
        first = self.collect(self.config("scopus"), self.client([self.fixture("scopus")]))
        for changes in ({"scopus_view": "COMPLETE"}, {"scopus_pagination": "offset"}):
            client = self.client([self.fixture("scopus")])
            changed = self.collect(self.config("scopus", **changes), client)
            client.get.assert_called_once()
            self.assertNotEqual(changed[0]["raw_ref"], first[0]["raw_ref"])

    def test_ieee_key_is_excluded_from_persisted_request_parameters(self):
        self.collect(self.config("ieee"), self.client([self.fixture("ieee")]))
        files = list((self.root / "run/01_retrieval/raw").glob("*/page_*.json"))
        self.assertEqual(len(files), 1)
        page = json.loads(files[0].read_text())
        self.assertNotIn("apikey", page["request_params"])
        self.assertNotIn("fixture-ieee", files[0].read_text())

    def test_invalid_source_topic_or_date_is_rejected_before_client_creation(self):
        for changes in ({"sources": ["unknown"]}, {"topics": ["unknown"]}, {"start": "2002-01-01"},
                        {"scopus_pagination": "unknown"}, {"scopus_view": "unknown"}):
            with self.subTest(changes=changes), mock.patch.object(self.providers, "Client") as client:
                with self.assertRaises(ValueError):
                    self.retrieval.collect(self.config(**changes), self.root / "invalid")
                client.assert_not_called()


class ScopusYearPartitionTests(ProviderTestCase):
    def partitions(self):
        return json.loads((self.root / "run/01_retrieval/scopus_partitions/scopus_quality.json").read_text())["partitions"]

    def test_offset_splits_disjoint_years_and_excludes_probe_rows_from_observations(self):
        client = self.client([self.fixture("scopus", 99, total=3),
                              self.fixture("scopus", 1, total=1),
                              self.fixture("scopus", 2, total=2),
                              self.fixture("scopus", 3, total=2)])
        with mock.patch.object(self.retrieval, "SCOPUS_OFFSET_LIMIT", 2):
            rows = self.collect(self.config("scopus", scopus_pagination="offset"), client)
        self.assertEqual(len(rows), 3)
        self.assertNotIn("2-s2.0-99", {row["source_id"] for row in rows})
        self.assertEqual({row["source_id"] for row in rows}, {"2-s2.0-1", "2-s2.0-2", "2-s2.0-3"})
        self.assertTrue(self.summary()["complete"])
        self.assertEqual(self.logs()[0]["records"], 3)
        self.assertEqual(self.logs()[0]["pages"], 4)
        parts = self.partitions()
        self.assertEqual([(p["year_start"], p["year_end"]) for p in parts], [(2000, 2001), (2000, 2000), (2001, 2001)])
        self.assertEqual(parts[0]["status"], "split_year_range")
        self.assertEqual(parts[0]["included_observations"], 0)
        self.assertEqual(parts[0]["children_reported_total"], 3)
        requests = [call.kwargs["params"] for call in client.get.call_args_list]
        self.assertEqual([p["start"] for p in requests], [0, 0, 0, 1])
        self.assertTrue(all("cursor" not in params for params in requests))
        self.assertIn("PUBYEAR > 1999 AND PUBYEAR < 2002", requests[0]["query"])
        self.assertIn("PUBYEAR > 1999 AND PUBYEAR < 2001", requests[1]["query"])
        self.assertIn("PUBYEAR > 2000 AND PUBYEAR < 2002", requests[2]["query"])
        self.assertTrue(all(row["parent_query_id"] == "scopus_quality" for row in rows))
        for row in rows:
            cache = self.root / "run" / row["raw_ref"].split("#", 1)[0]
            self.assertTrue(cache.is_file())
            self.assertIn("/years_", row["raw_ref"])

    def test_partition_tree_and_leaf_page_resume_without_repeating_completed_calls(self):
        config = self.config("scopus", scopus_pagination="offset")
        first = self.client([self.fixture("scopus", 99, total=3), self.fixture("scopus", 1),
                             self.fixture("scopus", 2, total=2), self.providers.APIError("Fixture interruption", True)])
        with mock.patch.object(self.retrieval, "SCOPUS_OFFSET_LIMIT", 2):
            before = self.collect(config, first)
            self.assertEqual(len(before), 2)
            self.assertFalse(self.summary()["complete"])
            retry = self.client([self.fixture("scopus", 3, total=2)])
            after = self.collect(config, retry)
            self.assertEqual(len(after), 3)
            self.assertEqual(after[:2], before)
            retry.get.assert_called_once()
            self.assertEqual(retry.get.call_args.kwargs["params"]["start"], 1)
            self.assertEqual(self.logs()[0]["cached_pages"], 3)
            self.assertTrue(self.summary()["complete"])
            replay = self.client([])
            replay.get.side_effect = AssertionError("The complete partition tree must replay from cache")
            self.assertEqual(self.collect(config, replay), after)
            replay.get.assert_not_called()

    def test_nested_splits_cover_every_year_once_at_leaves(self):
        config = self.config("scopus", scopus_pagination="offset", end="2003-12-31")
        client = self.client([self.fixture("scopus", 90, total=4), self.fixture("scopus", 91, total=3),
                              self.fixture("scopus", 1), self.fixture("scopus", 2, total=2),
                              self.fixture("scopus", 3, total=2), self.fixture("scopus", 4)])
        with mock.patch.object(self.retrieval, "SCOPUS_OFFSET_LIMIT", 2):
            rows = self.collect(config, client)
        self.assertEqual(len(rows), 4)
        self.assertTrue(self.summary()["complete"])
        leaves = [part for part in self.partitions() if part["status"] == "api_query_exhausted"]
        years = [year for part in leaves for year in range(part["year_start"], part["year_end"] + 1)]
        self.assertEqual(years, [2000, 2001, 2002, 2003])
        self.assertEqual(len(years), len(set(years)))

    def test_single_year_overflow_is_honestly_incomplete_without_guessing_further_filters(self):
        config = self.config("scopus", scopus_pagination="offset", start="2001-01-01")
        client = self.client([self.fixture("scopus", 1, total=3)])
        with mock.patch.object(self.retrieval, "SCOPUS_OFFSET_LIMIT", 2):
            rows = self.collect(config, client)
        self.assertEqual(len(rows), 1)
        client.get.assert_called_once()
        self.assertEqual(self.logs()[0]["status"], "single_year_overflow_unfinished")
        self.assertFalse(self.summary()["complete"])
        self.assertIn("no verified narrower split", self.partitions()[0]["error"])

    def test_inconsistent_parent_and_child_counts_do_not_claim_complete_retrieval(self):
        client = self.client([self.fixture("scopus", 99, total=4), self.fixture("scopus", 1),
                              self.fixture("scopus", 2, total=2), self.fixture("scopus", 3, total=2)])
        with mock.patch.object(self.retrieval, "SCOPUS_OFFSET_LIMIT", 2):
            rows = self.collect(self.config("scopus", scopus_pagination="offset"), client)
        self.assertEqual(len(rows), 3)
        self.assertFalse(self.summary()["complete"])
        self.assertEqual(self.partitions()[0]["status"], "partition_count_mismatch_unfinished")

    def test_page_cap_remains_explicit_and_can_be_lifted_without_refetching_probes(self):
        config = self.config("scopus", scopus_pagination="offset", max_pages=1)
        first = self.client([self.fixture("scopus", 99, total=3), self.fixture("scopus", 1), self.fixture("scopus", 2, total=2)])
        with mock.patch.object(self.retrieval, "SCOPUS_OFFSET_LIMIT", 2):
            initial = self.collect(config, first)
            self.assertEqual(len(initial), 2)
            self.assertEqual(self.logs()[0]["status"], "capped_unfinished")
            config["search"]["max_pages"] = 0
            retry = self.client([self.fixture("scopus", 3, total=2)])
            final = self.collect(config, retry)
        self.assertEqual(len(final), 3)
        retry.get.assert_called_once()
        self.assertTrue(self.summary()["complete"])

    def test_shared_cache_identity_preserves_the_prior_envelope(self):
        job = self.providers.jobs_for(["openalex"], ["quality"])[0]
        expected = self.retrieval._hash({"version": "1.0.0", "adapter": "fixture-adapter-hash", "job": job,
            "start": "2000-01-01", "end": "2001-12-31", "scopus_pagination": None,
            "scopus_view": None, "crossref_rows": None})
        self.assertEqual(self.retrieval.cache_identity(job, date(2000, 1, 1), date(2001, 12, 31), "fixture-adapter-hash"), expected)


class ImportContractTests(ProviderTestCase):
    def test_all_five_import_formats_preserve_basic_bibliography_without_calls(self):
        fixtures = {
            "csv": "Title,Abstract,Year,DOI\nSynthetic import fixture,Synthetic evidence,2001,10.9999/import\n",
            "json": json.dumps({"records": [{"title": "Synthetic import fixture", "abstract": "Synthetic evidence", "year": "2001", "doi": "10.9999/import"}]}),
            "jsonl": json.dumps({"title": "Synthetic import fixture", "abstract": "Synthetic evidence", "year": "2001", "doi": "10.9999/import"}) + "\n",
            "ris": "TY  - JOUR\nTI  - Synthetic import fixture\nAB  - Synthetic evidence\nPY  - 2001\nDO  - 10.9999/import\nER  - \n",
            "bib": "@article{fixture,title={Synthetic import fixture},abstract={Synthetic evidence},year={2001},doi={10.9999/import}}\n",
        }
        for suffix, text in fixtures.items():
            with self.subTest(suffix=suffix):
                path = self.root / ("input." + suffix)
                path.write_text(text)
                rows = self.retrieval._import_rows(path)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["title"], "Synthetic import fixture")
                self.assertEqual(rows[0]["abstract"], "Synthetic evidence")
                self.assertEqual(rows[0]["date"], "2001")
                self.assertEqual(rows[0]["doi"], "10.9999/import")

    def test_raw_import_provenance_is_accessible_but_old_labels_are_not_decisions(self):
        item = {"title": "Synthetic imported observation", "source": "scopus", "query_id": "source-query",
                "raw_ref": "original/raw.json#record=1", "observation_id": "original-observation",
                "sources": "scopus;openalex", "query_ids": "source-query;other-query",
                "provenance": "earlier-source", "screening_decision": "include"}
        normalized = self.retrieval._canonical(item)
        for field in self.retrieval.IMPORT_PROVENANCE_FIELDS:
            self.assertEqual(normalized[field], item[field.removeprefix("import_original_")])
        self.assertNotIn("screening_decision", normalized)
        row = self.retrieval._observation(normalized, "historical", "new-query", "snapshot.json", 1, date(1900, 1, 1), date(2001, 12, 31))
        self.assertEqual(row["import_original_source"], "scopus")
        self.assertNotEqual(row["observation_id"], item["observation_id"])

    def test_csv_accepts_abstract_larger_than_default_field_limit(self):
        path = self.root / "large.csv"
        abstract = "Synthetic evidence. " * 10000
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["title", "abstract"])
            writer.writeheader()
            writer.writerow({"title": "Synthetic large abstract", "abstract": abstract})
        self.assertEqual(self.retrieval._import_rows(path)[0]["abstract"], abstract.strip())

    def test_changed_import_set_or_bytes_is_rejected_before_provider_client(self):
        path = self.root / "input.json"
        path.write_text(json.dumps([{"title": "Synthetic immutable input"}]))
        config = self.config(enabled=False, sources=[], topics=[])
        config["imports"] = [{"source": "fixture", "path": path.name}]
        self.collect(config, self.client([]))
        old_snapshot = next((self.root / "run/01_retrieval/raw").glob("import_*.json")).read_bytes()
        path.write_text(json.dumps([{"title": "Changed synthetic input"}]))
        with mock.patch.object(self.providers, "Client") as client:
            with self.assertRaisesRegex(ValueError, "content changed"):
                self.retrieval.collect(config, self.root / "run")
            client.assert_not_called()
        self.assertEqual(next((self.root / "run/01_retrieval/raw").glob("import_*.json")).read_bytes(), old_snapshot)
        config["imports"] = []
        with self.assertRaisesRegex(ValueError, "source/path list differs"):
            self.retrieval.collect(config, self.root / "run")

