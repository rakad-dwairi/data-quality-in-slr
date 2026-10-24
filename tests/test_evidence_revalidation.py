"""Saved-generation recovery with synthetic evidence and inference forbidden."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


class SavedResponseTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("requests.sessions.Session.request", "socket.create_connection",
                     "socket.socket.connect", "socket.getaddrinfo"):
            self.stack.enter_context(mock.patch(name, side_effect=AssertionError("Network forbidden")))
        from slr_pipeline import screening as s
        self.s = s
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.db = s._open_checkpoint(self.root / "checkpoints/screening.sqlite")
        self.addCleanup(self.db.close)
        self.quote = "Synthetic target study measures a required data property."
        self.record = {"record_id": "R_SYNTHETIC_RECOVERY"}
        self.names = list(s.CORE_CRITERIA)
        self.evidence = {"stage": "title_abstract", "record_id": self.record["record_id"],
                         "passages": {"TITLE": "Synthetic title", "ABSTRACT": self.quote}}
        self.data = {"decision": "include", "study_kind": "primary", "rationale": "Synthetic rationale.",
                     "criteria": [{"criterion": n, "value": "yes", "source_attribution": "own_study",
                                   "explanation": "Synthetic target evidence.", "passage_id": "ABSTRACT",
                                   "quote": self.quote} for n in self.names]}
        self.raw = {"done": True, "done_reason": "stop", "prompt_eval_count": 100,
                    "message": {"content": json.dumps(self.data)}}
        self.config = {"protocol": {}, "model": {"name": "synthetic:offline"},
                       "screening": {"retry_errors": True}}
        self.client = SimpleNamespace(ready=True, unavailable_error="", model="synthetic:offline",
                                      model_digest="synthetic_digest", context=32768, output=8192,
                                      call=mock.Mock(side_effect=AssertionError("Inference forbidden")))
        self.schema = s._schema("title_abstract", self.names)
        self.system = "Synthetic trusted screening instructions."
        self.budget = {"attempted": 0, "limit": 3}

    def arguments(self):
        return (self.record, "title_abstract", self.config, self.root, self.client, self.db,
                self.evidence, self.system, self.names, self.schema, self.budget)

    def seed_old_failure(self, data=None):
        data = copy.deepcopy(self.data if data is None else data)
        if data == self.data:
            data["criteria"][0]["passage_id"] = "TITLE_ABSTRACT"
            data["criteria"][-1]["quote"] = "Unsupported synthetic statement."
        raw = {**self.raw, "message": {"content": json.dumps(data)}}
        prompt_hash = self.s.digest({"system": self.system, "user": self.s._json(self.evidence), "schema": self.schema})
        key = self.s.digest({"prompt_version": self.s.PROMPT_VERSION, "system": self.system,
                             "evidence": self.evidence, "schema": self.schema,
                             "model_settings": self.config["model"], "model_digest": self.client.model_digest})
        result = {**self.s._uncertain("Synthetic original validation failure", "model_or_validation_error"),
                  "attempt_id": "synthetic_original_attempt", "assessed_utc": "2026-10-22T00:00:00+00:00"}
        self.db.execute("INSERT INTO assessments VALUES (?,?,?,?,?,?,?,?)", (key, self.record["record_id"],
                        "title_abstract", result["screening_status"], self.s._json(result), self.s._json(raw),
                        "Synthetic original validation failure", result["assessed_utc"]))
        self.db.commit()
        self.s.write_json(self.root / "checkpoints/model_audit" / f"{key}.json", {
            "record_id": self.record["record_id"], "stage": "title_abstract", "attempt_id": result["attempt_id"],
            "assessment_fingerprint": key, "prompt_sha256": prompt_hash,
            "model": self.client.model, "model_digest": self.client.model_digest,
            "input_evidence": self.evidence, "system_prompt": self.system,
            "raw_response": raw, "result": result,
        })
        immutable = self.root / "checkpoints/model_attempts/synthetic_original_attempt_finished.json"
        self.s.write_json(immutable, {"original": True, "raw_response": raw, "result": result})
        return key, raw, immutable, immutable.read_bytes()

    def call(self):
        with redirect_stdout(io.StringIO()):
            return self.s._assess_input(*self.arguments())

    def test_saved_failed_response_is_revalidated_without_inference_or_budget(self):
        key, raw, immutable, old_bytes = self.seed_old_failure()
        result = self.call()
        self.client.call.assert_not_called()
        self.assertEqual(self.budget["attempted"], 0)
        self.assertEqual(result["assessment_fingerprint"], key)
        self.assertEqual(result["screening_status"], "assessed")
        self.assertEqual(result["ai_decision"], "uncertain")
        self.assertTrue(result["cache_reused"])
        self.assertTrue(result["cached_response_revalidated"])
        self.assertEqual(immutable.read_bytes(), old_bytes)
        stored = self.db.execute("SELECT status,raw_response,error FROM assessments WHERE cache_key=?", (key,)).fetchone()
        self.assertEqual(stored[0], "assessed")
        self.assertEqual(json.loads(stored[1]), raw)
        self.assertEqual(stored[2], "")
        events = list((self.root / "checkpoints/model_attempts").glob("*_revalidated.json"))
        self.assertEqual(len(events), 1)
        event = json.loads(events[0].read_text())
        self.assertEqual(event["model_requests"], 0)
        self.assertEqual(event["generation_attempt_id"], "synthetic_original_attempt")
        self.assertEqual(self.call()["revalidation_id"], result["revalidation_id"])
        self.assertEqual(len(list((self.root / "checkpoints/model_attempts").glob("*_revalidated.json"))), 1)

    def test_failed_response_without_matching_audit_cannot_be_promoted(self):
        key, _, _, _ = self.seed_old_failure()
        path = self.root / "checkpoints/model_audit" / f"{key}.json"
        audit = json.loads(path.read_text())
        audit["input_evidence"]["passages"]["ABSTRACT"] = "Tampered synthetic evidence."
        self.s.write_json(path, audit)
        self.budget.update(attempted=3)
        self.assertEqual(self.call()["screening_status"], "pending_limit")
        self.assertEqual(self.db.execute("SELECT status FROM assessments WHERE cache_key=?", (key,)).fetchone()[0],
                         "model_or_validation_error")
        self.client.call.assert_not_called()

    def test_retry_errors_false_preserves_old_failed_result_without_revalidation(self):
        self.seed_old_failure()
        self.config["screening"]["retry_errors"] = False
        result = self.call()
        self.assertEqual(result["screening_status"], "model_or_validation_error")
        self.assertFalse(list((self.root / "checkpoints/model_attempts").glob("*_revalidated.json")))
        self.client.call.assert_not_called()

    def test_stale_assessed_policy_revalidates_without_inference_and_preserves_generation_time(self):
        self.seed_old_failure()
        first = self.call()
        with mock.patch.object(self.s, "EVIDENCE_VALIDATION_VERSION", "SYNTHETIC_UPDATED_POLICY"):
            second = self.call()
            self.assertNotEqual(first["revalidation_id"], second["revalidation_id"])
            self.assertEqual(second["generation_created_utc"], first["generation_created_utc"])
            self.assertEqual(second["evidence_validation_version"], "SYNTHETIC_UPDATED_POLICY")
            self.assertEqual(self.call()["revalidation_id"], second["revalidation_id"])
        self.assertEqual(self.budget["attempted"], 0)
        self.client.call.assert_not_called()

    def test_structural_error_remains_failed_and_does_not_bypass_limit(self):
        data = copy.deepcopy(self.data)
        data["criteria"].pop()
        self.seed_old_failure(data)
        self.budget.update(attempted=3)
        self.assertEqual(self.call()["screening_status"], "pending_limit")
        self.client.call.assert_not_called()

    def test_fresh_response_is_normalized_and_raw_reply_is_preserved(self):
        data = copy.deepcopy(self.data)
        data["criteria"][0]["passage_id"] = "TITLE_ABSTRACT"
        raw = {**self.raw, "message": {"content": json.dumps(data)}}
        self.client.call.side_effect = None
        self.client.call.return_value = raw
        result = self.call()
        self.assertEqual(result["screening_status"], "assessed")
        self.assertTrue(json.loads(result["evidence_adjustments_json"]))
        self.assertEqual(json.loads(result["criteria_json"])[0]["passage_id"], "ABSTRACT")
        self.assertEqual(self.budget["attempted"], 1)
        audit = self.s.read_json(self.root / "checkpoints/model_audit" / f"{result['cache_key']}.json")
        self.assertEqual(audit["raw_response"], raw)
        self.assertEqual(json.loads(raw["message"]["content"])["criteria"][0]["passage_id"], "TITLE_ABSTRACT")

    def test_incomplete_generation_is_never_recovered_as_an_assessment(self):
        for change in ({"done": False}, {"done_reason": "length"}, {"prompt_eval_count": 32768}):
            with self.subTest(change=change), self.assertRaises(self.s.AssessmentError):
                self.s._validated_response({**self.raw, **change}, self.evidence, "title_abstract", self.names, {}, 32768, 8192)

    def test_downgraded_judgements_expose_original_rationale_and_distinct_issue_counts(self):
        data = copy.deepcopy(self.data)
        data["criteria"][0].update(source_attribution="prior_work", quote="Invented synthetic quote.")
        raw = {**self.raw, "message": {"content": json.dumps(data)}}
        result = self.s._validated_response(raw, self.evidence, "title_abstract", self.names, {}, 32768, 8192)
        self.assertEqual(result["ai_decision"], "uncertain")
        self.assertEqual(result["ai_model_decision"], "include")
        self.assertEqual(result["ai_model_rationale"], data["rationale"])
        self.assertIn("Original model rationale (unverified): " + data["rationale"], result["ai_rationale"])
        self.assertTrue(result["evidence_review_required"])
        self.assertEqual(result["evidence_downgraded_criteria"], 1)
        self.assertIn("target_report_attribution_not_established: 1", result["evidence_issue_summary"])
        self.assertIn("unsupported_or_ambiguous_source_quote: 1", result["evidence_issue_summary"])
        self.assertEqual(json.loads(result["criteria_json"])[0]["value"], "unclear")

    def test_source_anchor_repair_alone_does_not_claim_unsupported_scientific_judgement(self):
        data = copy.deepcopy(self.data)
        data["criteria"][0]["passage_id"] = "INVENTED_PASSAGE_ID"
        raw = {**self.raw, "message": {"content": json.dumps(data)}}
        result = self.s._validated_response(raw, self.evidence, "title_abstract", self.names, {}, 32768, 8192)
        self.assertEqual(result["ai_decision"], "include")
        self.assertFalse(result["evidence_review_required"])
        self.assertEqual(result["evidence_downgraded_criteria"], 0)
        self.assertEqual(result["evidence_issue_summary"], "unique_exact_quote_in_actual_passage: 1")

    def test_csv_exports_current_generation_identity_and_review_diagnostics(self):
        from slr_pipeline.cli import SCREEN, FULL
        fields = {"ai_study_kind", "ai_model_decision", "ai_model_rationale", "cache_key",
                  "assessment_fingerprint", "attempt_id", "assessed_utc", "prompt_version",
                  "evidence_downgraded_criteria", "evidence_review_required", "evidence_issue_summary"}
        self.assertTrue(fields.issubset(SCREEN))
        self.assertTrue(fields.issubset(FULL))
        self.assertEqual(len(SCREEN), len(set(SCREEN)))

    def test_transport_failure_names_timeout_without_exposing_exception_details(self):
        client = self.s.OllamaClient({"name": "synthetic:offline"})
        self.addCleanup(client.close)
        failure = self.s.requests.ReadTimeout("SYNTHETIC_PRIVATE_EXCEPTION_DETAIL")
        with mock.patch.object(client.session, "post", side_effect=failure) as post, \
                mock.patch.object(self.s.time, "sleep"):
            with self.assertRaises(self.s.ModelUnavailable) as error:
                client.call([], {})
        self.assertEqual(post.call_count, 2)
        self.assertIn("ReadTimeout", str(error.exception))
        self.assertNotIn("SYNTHETIC_PRIVATE_EXCEPTION_DETAIL", str(error.exception))


if __name__ == "__main__":
    unittest.main()
