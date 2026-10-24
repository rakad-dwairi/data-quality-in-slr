"""Offline pending-only batching, provenance preservation, and crash recovery."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pending_title_abstract as pending
from local_source_citations import SourceCitationClient, VERSION as CITATION_VERSION
from slr_pipeline import cli, screening
from slr_pipeline.common import digest, read_csv, read_json
from validate_fast_local_model import BREVITY


class PendingTitleAbstractTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        denied = AssertionError("Real network and model inference are forbidden")
        for name in ("requests.sessions.Session.request", "socket.create_connection", "socket.getaddrinfo",
                     "socket.socket.connect", "socket.socket.connect_ex"):
            self.stack.enter_context(mock.patch(name, side_effect=denied))
        self.stack.enter_context(mock.patch.object(SourceCitationClient, "preflight", side_effect=denied))
        self.stack.enter_context(mock.patch.object(SourceCitationClient, "call", side_effect=denied))
        self.folder = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.run = self.folder / "continuation"
        self.project = Path(screening.__file__).resolve().parents[1]
        self.config = {"run_directory": str(self.run), "_project_dir": str(self.project),
                       "search": {"enabled": False, "start": "1900-01-01", "end": "2026-10-21"},
                       "deduplication": {}, "imports": [], "fulltext": {},
                       "reviews_file": str(self.folder / "reviews.csv"),
                       "date_checks_file": str(self.folder / "dates.csv"),
                       "model": {"name": "synthetic:calibrated-fast", "base_url": "http://127.0.0.1:11435",
                                 "allow_remote": False, "context_tokens": 131072, "max_output_tokens": 4096,
                                 "source_citation_mode": CITATION_VERSION},
                       "protocol": {"version": "synthetic-protocol", "iso_codebook": {}},
                       "screening": {"limit": 0, "retry_errors": False, "require_abstract": True}}
        cli.initialize(self.config, self.run)
        self.records = [{"record_id": "R" + str(n), "title": "Synthetic study " + str(n),
                         "abstract": "Synthetic unresolved data-quality assessment.", "date": "2020",
                         "doi": "", "authors": "Fixture", "source": "fixture", "sources": "fixture"}
                        for n in range(5)]
        self.retained = [{**self.records[0], **screening._uncertain("Original unresolved assessment.", "assessed"),
                          "model": "synthetic:original-35b", "model_digest": "original-installed-digest",
                          "input_sha256": screening.screening_input_hash(self.records[0], "title_abstract"),
                          "cache_key": "a" * 64, "assessment_fingerprint": "a" * 64,
                          "attempt_id": "original-attempt", "assessed_utc": "2020-01-01T00:00:00Z",
                          "retained_from_run": str(self.folder / "donor")}]
        self.ledger = [{"record_id": "R0", "source_cache_key": "a" * 64,
                        "audit_path": str(self.folder / "donor/audit.json"), "result_payload_digest": "b" * 64}]
        self.seen = []
        self.original_globals = screening.OllamaClient, screening.SYSTEM_PROMPT, screening.PROMPT_VERSION

    def synthetic_batch(self, records, stage, config, run):
        self.assertEqual(stage, "title_abstract")
        self.assertIs(screening.OllamaClient, SourceCitationClient)
        self.assertEqual(screening.SYSTEM_PROMPT, pending._BASE_PROMPT + BREVITY)
        self.assertEqual(screening.PROMPT_VERSION, pending._BASE_VERSION + "/concise-source-ids-1")
        self.seen.append([row["record_id"] for row in records])
        names = screening._criterion_names(stage, config["protocol"])
        schema = screening._schema(stage, names)
        system = screening.SYSTEM_PROMPT + "\nTrusted review protocol and required criteria:\n" + screening._json(
            {"protocol": config["protocol"], "required_criteria": names})

        class SavedFixtureReply:
            ready = True
            unavailable_error = ""
            model_digest = "candidate-installed-digest"
            model = config["model"]["name"]
            context = config["model"]["context_tokens"]
            output = config["model"]["max_output_tokens"]

            def call(self, messages, supplied_schema):
                data = {"decision": "uncertain", "study_kind": "primary", "rationale": "Synthetic unresolved evidence.",
                        "criteria": [{"criterion": name, "value": "unclear", "source_attribution": "unclear",
                                      "explanation": "Synthetic absence of definite evidence.", "passage_id": "", "quote": ""}
                                     for name in names]}
                return {"done": True, "done_reason": "stop", "prompt_eval_count": 50,
                        "message": {"role": "assistant", "content": json.dumps(data)}}

        db = screening._open_checkpoint(Path(run) / "checkpoints/screening.sqlite")
        try:
            result = []
            for record in records:
                evidence = {"stage": stage, "required_criteria": names, "record_id": record["record_id"],
                            "metadata": {field: record.get(field, "") for field in
                                         ("title", "authors", "date", "venue", "publication_type", "doi", "source", "sources")},
                            "all_source_text_provided": True, "passages": screening._passages(record, stage)}
                assessed = screening._assess_input(record, stage, config, run, SavedFixtureReply(), db,
                    evidence, system, names, schema, {"limit": 0, "attempted": 0})
                result.append({**record, "model": config["model"]["name"], "prompt_version": screening.PROMPT_VERSION,
                               "input_sha256": screening.screening_input_hash(record, stage), **assessed})
            return result
        finally:
            db.close()

    def execute(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return pending.screen_remaining(self.records, self.retained, self.config, self.run,
                                             retained_ledger=self.ledger, **kwargs)

    def test_disjoint_batches_preserve_retained_rows_and_callback_cannot_mutate_results(self):
        original_retained = copy.deepcopy(self.retained)
        callback_counts = []

        def callback(rows):
            self.assertEqual((screening.OllamaClient, screening.SYSTEM_PROMPT, screening.PROMPT_VERSION), self.original_globals)
            snapshot = read_json(self.run / "03_title_abstract/completed_snapshot.json")
            self.assertEqual(snapshot, rows)
            callback_counts.append(len(rows))
            rows[0]["model"] = "INVALID:callback-mutation"

        with mock.patch.object(screening, "screen_records", side_effect=self.synthetic_batch), \
                mock.patch.object(cli, "export_results") as exports:
            rows = self.execute(batch_size=2, after_batch=callback)
        self.assertEqual(self.seen, [["R1", "R2"], ["R3", "R4"]])
        self.assertEqual(callback_counts, [3, 5])
        self.assertEqual(rows[0], original_retained[0])
        self.assertEqual(self.retained, original_retained)
        self.assertEqual(read_json(self.run / "03_title_abstract/retained_rows.json"), original_retained)
        self.assertEqual(read_json(self.run / "03_title_abstract/retained_ledger.json"), self.ledger)
        self.assertTrue(all(call.kwargs["ft"] == [] for call in exports.call_args_list))
        self.assertEqual(read_json(self.run / "03_title_abstract/progress.json")["phase"], "complete")
        self.assertTrue(cli.validate_stage(self.run, "screen"))

    def test_completed_candidate_checkpoints_resume_without_any_screening_call(self):
        with mock.patch.object(screening, "screen_records", side_effect=self.synthetic_batch), mock.patch.object(cli, "export_results"):
            first = self.execute(batch_size=2)
        with mock.patch.object(screening, "screen_records", side_effect=AssertionError("Completed work must not regenerate")), \
                mock.patch.object(cli, "export_results"):
            resumed = self.execute(batch_size=2)
        self.assertEqual(resumed[0], first[0])
        for old, new in zip(first[1:], resumed[1:]):
            for field in ("model", "model_digest", "cache_key", "attempt_id", "assessed_utc", "criteria_json", "ai_decision"):
                self.assertEqual(old[field], new[field], field)
            self.assertTrue(new["cache_reused"])
            self.assertNotIn("retained_from_run", new)
        self.assertEqual(read_json(self.run / "03_title_abstract/progress.json")["resumed_checkpoint_rows"], 4)

    def test_checkpoint_committed_before_interruption_is_recovered_beyond_old_snapshot(self):
        def interrupt_after_commit(records, stage, config, run):
            self.synthetic_batch(records[:1], stage, config, run)
            raise KeyboardInterrupt

        with mock.patch.object(screening, "screen_records", side_effect=interrupt_after_commit), \
                mock.patch.object(cli, "export_results"), self.assertRaises(KeyboardInterrupt):
            self.execute(batch_size=2)
        self.assertEqual(len(read_json(self.run / "03_title_abstract/completed_snapshot.json")), 1)
        self.assertFalse(read_json(self.run / "stage_state.json")["screen"]["valid"])
        self.seen = []
        with mock.patch.object(screening, "screen_records", side_effect=self.synthetic_batch), mock.patch.object(cli, "export_results"):
            rows = self.execute(batch_size=2)
        self.assertEqual(self.seen, [["R2", "R3"], ["R4"]])
        self.assertEqual(len(rows), 5)
        self.assertEqual(read_json(self.run / "03_title_abstract/progress.json")["resumed_checkpoint_rows"], 1)

    def test_failure_persists_pending_status_and_stops_before_next_batch(self):
        def unavailable(records, stage, config, run):
            return [{**record, **screening._uncertain("Synthetic unavailable model.", "model_unavailable"),
                     "model": config["model"]["name"], "prompt_version": screening.PROMPT_VERSION,
                     "input_sha256": screening.screening_input_hash(record, stage)} for record in records]

        with mock.patch.object(screening, "screen_records", side_effect=unavailable) as screen, \
                mock.patch.object(cli, "export_results"), self.assertRaisesRegex(ValueError, "unresolved assessments"):
            self.execute(batch_size=1)
        self.assertEqual(screen.call_count, 1)
        self.assertEqual(read_json(self.run / "03_title_abstract/progress.json")["phase"], "needs_attention")
        self.assertEqual(len(read_json(self.run / "03_title_abstract/completed_snapshot.json")), 1)
        self.assertEqual(len(read_json(self.run / "03_title_abstract/assessments.json")), 2)

    def test_changed_retained_payload_is_refused_on_resume(self):
        with mock.patch.object(screening, "screen_records", side_effect=self.synthetic_batch), mock.patch.object(cli, "export_results"):
            self.execute()
        self.retained[0]["assessed_utc"] = "2026-10-24T00:00:00Z"
        with mock.patch.object(screening, "screen_records", side_effect=AssertionError("No inference")), \
                self.assertRaisesRegex(ValueError, "input changed"):
            self.execute()

    def test_real_exports_contain_zero_fulltext_assessments_and_no_invented_final_results(self):
        with mock.patch.object(screening, "screen_records", side_effect=self.synthetic_batch):
            rows = self.execute()
        self.assertEqual(len(rows), 5)
        self.assertEqual(read_csv(self.run / "04_full_text/screening.csv"), [])
        self.assertEqual(read_csv(self.run / "06_final/included_candidates.csv"), [])
        flow = read_json(self.run / "06_final/flow_counts.json")
        self.assertEqual(flow["full_text_assessments"], 0)
        self.assertEqual(flow["human_confirmed_included_reports"], 0)

    def test_prompt_context_restores_globals_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with pending.concise_candidate_prompt():
                self.assertIs(screening.OllamaClient, SourceCitationClient)
                raise RuntimeError("synthetic")
        self.assertEqual((screening.OllamaClient, screening.SYSTEM_PROMPT, screening.PROMPT_VERSION), self.original_globals)

    def test_expected_digest_mismatch_stops_before_any_inference(self):
        def mismatched_preflight(client):
            client.model_digest = "different-installed-weights"

        with mock.patch.object(SourceCitationClient, "preflight", mismatched_preflight), \
                mock.patch.object(SourceCitationClient, "call", side_effect=AssertionError("Mismatched weights must not generate")) as generate, \
                mock.patch.object(cli, "export_results"), \
                self.assertRaisesRegex(screening.ModelUnavailable, "approved calibration"):
            self.execute(expected_model_digest="approved-installed-weights")
        generate.assert_not_called()
        identity = read_json(self.run / "03_title_abstract/pending_identity.json")
        self.assertEqual(identity["expected_model_digest"], "approved-installed-weights")
        self.assertEqual((screening.OllamaClient, screening.SYSTEM_PROMPT, screening.PROMPT_VERSION), self.original_globals)
        self.assertEqual(len(read_json(self.run / "03_title_abstract/completed_snapshot.json")), 1)

    def test_changed_retained_input_is_refused_before_screening(self):
        self.records[0]["abstract"] = "Changed donor input."
        with mock.patch.object(screening, "screen_records", side_effect=AssertionError("No inference")), \
                self.assertRaisesRegex(ValueError, "unchanged input"):
            self.execute()


if __name__ == "__main__":
    unittest.main()
