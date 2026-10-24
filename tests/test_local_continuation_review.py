"""Synthetic continuation proof: no databases, remote requests or real models."""
from contextlib import ExitStack, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import early_local_review as early
import local_continuation_review as continuation
from local_source_citations import SourceCitationClient
from slr_pipeline import screening
from slr_pipeline.common import digest, file_hash, read_csv, read_json, write_csv, write_json


QUOTE = "The target study empirically measures item metadata completeness in a recommender dataset against a defined reference requirement."
DIGEST = "synthetic-original-strong-model-digest"


def answer(evidence):
    passage_id, text = next((key, value) for key, value in evidence["passages"].items() if key.startswith("BODY_"))
    quote = QUOTE if QUOTE in text else text[:400]
    return {"decision": "uncertain" if evidence.get("source_packet") else "include",
            "study_kind": "primary", "rationale": "Synthetic fixture proposal, never a paper claim.",
            "criteria": [{"criterion": name, "value": "yes", "source_attribution": "own_study",
                          "explanation": "Synthetic fixture evidence.", "passage_id": passage_id, "quote": quote}
                         for name in evidence["required_criteria"]],
            "extraction": [], "appraisal": [], "iso_mapping": []}


class FakeStrongClient(SourceCitationClient):
    calls = []
    preflights = 0

    def preflight(self):
        type(self).preflights += 1
        self.model_digest = DIGEST

    def call(self, messages, schema):
        evidence = json.loads(messages[1]["content"])
        FakeStrongClient.calls.append({"model": self.model, "evidence": copy.deepcopy(evidence),
                                       "system": messages[0]["content"]})
        return {"done": True, "done_reason": "stop", "prompt_eval_count": 100,
                "message": {"role": "assistant", "content": json.dumps(answer(evidence))}}


class ReviewContinuationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in ("requests.sessions.Session.request", "socket.create_connection", "socket.getaddrinfo",
                       "socket.socket.connect", "socket.socket.connect_ex"):
            self.stack.enter_context(mock.patch(target, side_effect=AssertionError("No real network or inference")))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.source = self.root / "early"
        self.output = self.root / "continuation"
        (self.source / "code_snapshot").mkdir(parents=True)
        self.project = Path(continuation.__file__).resolve().parent
        for name in (*continuation._CORE_FILES, "early_local_review.py", "local_source_citations.py"):
            source = self.project / name if name in {"early_local_review.py", "local_source_citations.py"} else self.project / "slr_pipeline" / name
            shutil.copy2(source, self.source / "code_snapshot" / name)
        self.config = {
            "search": {"start": "1900-01-01", "end": "2026-10-21"},
            "model": {"name": "qwen3.5:35b", "base_url": "http://127.0.0.1:11434", "allow_remote": False,
                      "source_citation_mode": "source-ids-1", "context_tokens": 32768,
                      "max_output_tokens": 4096, "timeout_seconds": 30, "temperature": 0},
            "screening": {"limit": 0, "retry_errors": False, "require_abstract": True},
            "fulltext": {"min_text_chars": 1000, "download_open_access": False,
                         "directory": str(self.root / "inputs"), "manifest": str(self.root / "missing-manifest.csv")},
            "protocol": {"version": "synthetic-frozen-original", "iso_codebook": {}},
        }
        write_json(self.source / "manifest.json", {"identity": {"version": "synthetic-early"}, "config": self.config})
        self.record = {"record_id": "R_FIXTURE", "title": "Synthetic recommendation study", "abstract": QUOTE,
                       "authors": "Synthetic Author", "date": "2020-01-01", "doi": "10.9999/continuation-fixture",
                       "venue": "Synthetic venue", "publication_type": "article", "source": "synthetic", "sources": "synthetic"}
        FakeStrongClient.calls = []
        FakeStrongClient.preflights = 0

    def source_snapshot(self):
        return {str(path.relative_to(self.source)): path.read_bytes() for path in self.source.rglob("*") if path.is_file()}

    def prepare_saved(self, record=None, length=10):
        record = record or self.record
        folder = self.source / "reports" / record["record_id"]
        pdf = folder / "04_full_text/pdfs" / (record["record_id"] + ".txt")
        pdf.parent.mkdir(parents=True)
        body = (QUOTE + "\n") * length
        pdf.write_text(body, encoding="utf-8")
        file_sha = file_hash(pdf)
        text = "[TEXT FILE]\n" + body
        text_sha = hashlib.sha256(text.encode()).hexdigest()
        cache = {"extractor_version": "fulltext-1", "file_sha256": file_sha, "text_sha256": text_sha,
                 "text": text, "page_count": 1, "sparse_pages": [], "failed_pages": [], "characters": len(body)}
        write_json(folder / "checkpoints/fulltext" / ("fulltext-1_" + file_sha + "_txt.json"), cache)
        write_csv(folder / "04_full_text/fulltext_manifest.csv", [{"record_id": record["record_id"], "path": str(pdf),
                   "file_sha256": file_sha, "text_sha256": text_sha, "text_complete": True}])
        return folder, {**record, "full_text": text, "full_text_path": str(pdf), "full_text_sha256": text_sha,
                        "full_text_file_sha256": file_sha, "full_text_status": "ready", "text_complete": True}

    def add_pass(self, folder, prepared, pass_name):
        context = early.second_pass() if pass_name == "second" else nullcontext()
        with context:
            names = screening._criterion_names("full_text", self.config["protocol"])
            evidence = {"stage": "full_text", "required_criteria": names, "record_id": prepared["record_id"],
                        "metadata": {key: prepared.get(key, "") for key in ("title", "authors", "date", "venue", "publication_type", "doi", "source", "sources")},
                        "all_source_text_provided": True, "passages": screening._passages(prepared, "full_text")}
            schema = screening._schema("full_text", names)
            system = screening.SYSTEM_PROMPT + "\nTrusted review protocol and required criteria:\n" + screening._json({"protocol": self.config["protocol"], "required_criteria": names})
            raw = {"done": True, "done_reason": "stop", "prompt_eval_count": 100,
                   "message": {"role": "assistant", "content": json.dumps(answer(evidence))}}
            result = screening._validated_response(raw, evidence, "full_text", names, self.config["protocol"], 32768, 4096)
            result.update(attempt_id="synthetic-original-" + pass_name, assessed_utc="2026-10-23T00:00:00Z")
            key = digest({"prompt_version": screening.PROMPT_VERSION, "system": system, "evidence": evidence,
                          "schema": schema, "model_settings": self.config["model"], "model_digest": DIGEST})
            prompt_hash = digest({"system": system, "user": screening._json(evidence), "schema": schema})
            saved = {**early.stripped(prepared), **result, "model": "qwen3.5:35b", "model_digest": DIGEST,
                     "prompt_version": screening.PROMPT_VERSION, "cache_key": key, "assessment_fingerprint": key,
                     "prompt_sha256": prompt_hash, "input_sha256": screening.screening_input_hash(prepared, "full_text")}
            write_json(folder / (pass_name + "_result.json"), saved)
            write_json(folder / pass_name / "checkpoints/model_audit" / (key + ".json"), {
                "record_id": prepared["record_id"], "stage": "full_text", "assessment_fingerprint": key,
                "prompt_sha256": prompt_hash, "model": "qwen3.5:35b", "model_digest": DIGEST,
                "prompt_version": screening.PROMPT_VERSION, "input_evidence": evidence, "system_prompt": system,
                "raw_response": raw, "result": result, "attempt_id": result["attempt_id"],
                "created_at": result["assessed_utc"], "error": ""})
            return saved

    def add_completed(self, status="double_pass_candidate"):
        folder, prepared = self.prepare_saved()
        first, second = self.add_pass(folder, prepared, "first"), self.add_pass(folder, prepared, "second")
        summary = {key: prepared.get(key, "") for key in early.FIELDS}
        summary.update(review_status=status, first_decision="include", second_decision="include",
                       date_window_status="within_window", source_ta_cache_key="original-ta-key", human_decision="")
        result = {"summary": summary, "first": first, "second": second, "elapsed_seconds": 123.5}
        write_json(folder / "result.json", result)
        return folder, result

    def test_completed_donor_is_referenced_unchanged_without_preflight_or_generation(self):
        _, original = self.add_completed()
        before = self.source_snapshot()
        with mock.patch.object(continuation, "SourceCitationClient", side_effect=AssertionError("No client for completed work")), \
                mock.patch.object(continuation, "prepare_fulltexts", side_effect=AssertionError("No retrieval for completed work")):
            worker = continuation.ReviewContinuation(self.source, self.output)
            self.assertEqual(worker.completed, {"R_FIXTURE"})
            self.assertEqual(worker.review(self.record, "different-ta-key"), original)
            status = worker.export()
        self.assertEqual(before, self.source_snapshot())
        self.assertFalse((self.output / "reports/R_FIXTURE").exists())
        self.assertEqual(status["double_pass_candidates"], 1)
        ledger = read_csv(self.output / "continuation_ledger.csv")
        self.assertEqual(ledger[0]["origin"], "retained_reference")
        self.assertEqual(ledger[0]["first_cache_key"], original["first"]["cache_key"])
        self.assertEqual(read_json(self.output / "accounting.json")["retained_completed_reports"], 1)

    def test_partial_first_pass_is_reused_and_only_missing_blinded_pass_is_generated(self):
        folder, prepared = self.prepare_saved()
        first = self.add_pass(folder, prepared, "first")
        before = self.source_snapshot()
        with mock.patch.object(continuation, "SourceCitationClient", FakeStrongClient), \
                mock.patch.object(continuation, "prepare_fulltexts", side_effect=AssertionError("Do not redownload a prepared source")):
            worker = continuation.ReviewContinuation(self.source, self.output)
            result = worker.review(self.record, "new-ta-key")
        self.assertEqual(len(FakeStrongClient.calls), 1)
        self.assertIn("blinded eligibility verification", FakeStrongClient.calls[0]["system"])
        self.assertEqual(result["first"], first)
        self.assertEqual(result["first"]["cache_key"], first["cache_key"])
        self.assertEqual(result["summary"]["review_status"], "double_pass_candidate")
        self.assertEqual(before, self.source_snapshot())
        self.assertEqual(read_json(self.output / "reports/R_FIXTURE/first_result.json"), first)
        self.assertTrue((self.output / "reports/R_FIXTURE/checkpoints/fulltext").is_dir())
        # A fresh coordinator instance reuses its completed result with no client.
        with mock.patch.object(continuation, "SourceCitationClient", side_effect=AssertionError("No repeated generation")):
            resumed = continuation.ReviewContinuation(self.source, self.output)
            self.assertEqual(resumed.review(self.record), result)

    def test_previously_failed_source_is_not_retried_or_sent_to_a_model(self):
        folder = self.source / "reports" / self.record["record_id"]
        write_csv(folder / "04_full_text/retrieval_log.csv", [{"record_id": self.record["record_id"], "status": "error"}])
        before = self.source_snapshot()
        with mock.patch.object(continuation, "SourceCitationClient", side_effect=AssertionError("Missing source must not be inferred")), \
                mock.patch.object(continuation, "prepare_fulltexts", side_effect=AssertionError("Do not retry failed source retrieval")):
            worker = continuation.ReviewContinuation(self.source, self.output)
            result = worker.review(self.record)
            self.assertEqual(worker.review(self.record), result)
        self.assertEqual(result["summary"]["review_status"], "source_needed")
        self.assertEqual(before, self.source_snapshot())
        self.assertEqual(result["first"], {})

    def test_changed_pdf_or_changed_input_cannot_become_a_retained_candidate(self):
        folder, result = self.add_completed()
        worker = continuation.ReviewContinuation(self.source, self.output)
        with self.assertRaisesRegex(ValueError, "differs from its completed frozen input"):
            worker.review({**self.record, "title": "Changed title"})
        Path(result["summary"]["full_text_path"]).write_text("Changed source", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            worker.export()

    def test_saved_pass_prompt_config_or_result_mismatch_fails_without_regeneration(self):
        folder, prepared = self.prepare_saved()
        saved = self.add_pass(folder, prepared, "first")
        saved["ai_decision"] = "exclude"
        write_json(folder / "first_result.json", saved)
        with mock.patch.object(continuation, "SourceCitationClient", side_effect=AssertionError("Never regenerate a mismatched completed pass")), \
                self.assertRaisesRegex(ValueError, "differs from its original audited result"):
            worker = continuation.ReviewContinuation(self.source, self.output)
            worker.review(self.record)

    def test_installed_model_digest_mismatch_stops_before_missing_pass_generation(self):
        folder, prepared = self.prepare_saved()
        first = self.add_pass(folder, prepared, "first")
        class DifferentInstalledDigest(FakeStrongClient):
            def preflight(self):
                self.model_digest = "different-synthetic-installed-model"
        before = self.source_snapshot()
        with mock.patch.object(continuation, "SourceCitationClient", DifferentInstalledDigest):
            worker = continuation.ReviewContinuation(self.source, self.output)
            with self.assertRaisesRegex(screening.ModelUnavailable, "differs from the frozen completed reviews"):
                worker.review(self.record)
        self.assertEqual(FakeStrongClient.calls, [])
        self.assertEqual(before, self.source_snapshot())
        self.assertEqual(read_json(self.output / "reports/R_FIXTURE/first_result.json"), first)

    def test_fast_config_remote_or_changed_frozen_prompt_is_rejected(self):
        faster = copy.deepcopy(self.config)
        faster["model"]["name"] = "qwen3.5:4b"
        with self.assertRaisesRegex(ValueError, "exact early frozen strong configuration"):
            continuation.ReviewContinuation(self.source, self.output, faster)
        manifest = read_json(self.source / "manifest.json")
        manifest["config"]["model"]["allow_remote"] = True
        write_json(self.source / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "loopback"):
            continuation.ReviewContinuation(self.source, self.output)

    def test_long_complete_source_uses_all_core_packets_and_strong_blinded_passes(self):
        self.prepare_saved(length=700)
        before = self.source_snapshot()
        original_client, original_fits = screening.OllamaClient, screening._fits
        with mock.patch.object(continuation, "SourceCitationClient", FakeStrongClient), \
                mock.patch.object(continuation, "prepare_fulltexts", side_effect=AssertionError("Never redownload existing text")):
            worker = continuation.ReviewContinuation(self.source, self.output)
            result = worker.review(self.record, "synthetic-ta-key")
        self.assertEqual(result["first"]["screening_status"], "chunked_complete")
        self.assertEqual(result["second"]["screening_status"], "chunked_complete")
        self.assertGreater(result["first"]["chunk_count"], 1)
        self.assertEqual(result["first"]["chunk_count"], result["first"]["chunks_completed"])
        self.assertEqual(result["second"]["chunk_count"], result["second"]["chunks_completed"])
        self.assertTrue(all(call["model"] == "qwen3.5:35b" for call in FakeStrongClient.calls))
        self.assertEqual(result["summary"]["review_status"], "double_pass_candidate")
        for blinded in (False, True):
            calls = [call for call in FakeStrongClient.calls if ("blinded eligibility verification" in call["system"]) == blinded]
            covered = "".join(text for call in calls for key, text in call["evidence"]["passages"].items() if key.startswith("BODY_"))
            self.assertEqual(hashlib.sha256(covered.encode()).hexdigest(), result["first"]["full_text_sha256"])
        self.assertEqual(before, self.source_snapshot())
        self.assertIs(screening.OllamaClient, original_client)
        self.assertIs(screening._fits, original_fits)


if __name__ == "__main__":
    unittest.main()
