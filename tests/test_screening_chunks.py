"""Offline screening regression-test SOURCE, supplied for the user to run.

These fixtures are synthetic and do not describe actual publications. The tests
partition text, combine proposals, and use a fake model with temporary local
SQLite checkpoints. They never call an actual model, search API, or download.
Run locally with: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class ScreeningChunkTests(unittest.TestCase):
    def setUp(self):
        self.network = mock.patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("Real network/model calls are forbidden in these tests"),
        )
        self.network.start()
        self.addCleanup(self.network.stop)
        # Deferred project import: writing or inspecting this file runs no code.
        from slr_pipeline import screening
        self.screening = screening
        self.names = list(screening.CORE_CRITERIA) + ["full_text_sufficient"]

    @staticmethod
    def record(body):
        return {
            "record_id": "R_SYNTHETIC_PACKET_FIXTURE", "title": "Synthetic fixture title",
            "abstract": "Synthetic fixture abstract.", "full_text": body,
            "text_complete": True,
        }

    def evidence(self, record):
        return {
            "stage": "full_text", "record_id": record["record_id"],
            "required_criteria": self.names, "metadata": {"title": record["title"]},
            "all_source_text_provided": True,
            "passages": self.screening._passages(record, "full_text"),
        }

    def packet_result(self, *, values=None, status="assessed", kind="primary",
                      extraction=None, appraisal=None, fingerprint="synthetic_packet_1"):
        values = values or {}
        criteria = [{
            "criterion": name, "value": values.get(name, "yes"),
            "source_attribution": "own_study",
            "explanation": "Synthetic criterion fixture.",
            "passage_id": "BODY_00000000_00000100",
            "quote": "Synthetic target-study method statement.",
        } for name in self.names]
        return {
            "ai_decision": "uncertain", "ai_study_kind": kind,
            "ai_rationale": "Synthetic bounded source packet.",
            "screening_status": status, "criteria_json": json.dumps(criteria),
            "evidence_json": json.dumps(criteria),
            "extraction_json": json.dumps(extraction or []),
            "appraisal_json": json.dumps(appraisal or []),
            "iso_mapping_json": "[]", "assessment_fingerprint": fingerprint,
            "cache_reused": False,
        }

    def test_multilingual_packet_partition_preserves_every_character_and_bound(self):
        body = ("Synthetic αβ. 段落 بيانات 🙂\n" * 1600) + "\n  trailing whitespace  \n"
        record = self.record(body)
        client = SimpleNamespace(context=12000, output=1024)
        packets = self.screening._packets(record, self.evidence(record), "Trusted synthetic system.", client)
        self.assertGreater(len(packets), 1)
        covered, expected_start = [], 0
        for index, packet in enumerate(packets, 1):
            self.assertEqual(packet["source_packet"]["index"], index)
            self.assertEqual(packet["source_packet"]["total"], len(packets))
            self.assertFalse(packet["all_source_text_provided"])
            self.assertTrue(self.screening._fits(client, "Trusted synthetic system.", packet))
            self.assertEqual(packet["passages"]["TITLE"], record["title"])
            self.assertEqual(packet["passages"]["ABSTRACT"], record["abstract"])
            for key, passage in packet["passages"].items():
                if not key.startswith("BODY_"):
                    continue
                match = re.fullmatch(r"BODY_(\d+)_(\d+)", key)
                self.assertIsNotNone(match)
                start, end = map(int, match.groups())
                self.assertEqual(start, expected_start)
                self.assertEqual(passage, body[start:end])
                self.assertEqual(end - start, len(passage))
                covered.append(passage)
                expected_start = end
        self.assertEqual("".join(covered), body)
        self.assertEqual(expected_start, len(body))

    def test_whitespace_only_body_segments_are_not_dropped(self):
        body = "prefix" + (" \n\t" * 5000) + "suffix"
        record = self.record(body)
        client = SimpleNamespace(context=9500, output=1024)
        packets = self.screening._packets(record, self.evidence(record), "system", client)
        covered = "".join(value for packet in packets for key, value in packet["passages"].items()
                          if key.startswith("BODY_"))
        self.assertEqual(covered, body)

    def test_protocol_and_metadata_that_already_exceed_context_raise(self):
        record = self.record("Synthetic body.")
        client = SimpleNamespace(context=2400, output=1024)
        with self.assertRaises(self.screening.AssessmentError):
            self.screening._packets(record, self.evidence(record), "system", client)

    def test_positive_whole_report_proposal_requires_all_packets(self):
        first = self.packet_result()
        second = self.packet_result(fingerprint="synthetic_packet_2")
        complete = self.screening._combine_packets([first, second], self.names, 2)
        self.assertEqual(complete["screening_status"], "chunked_complete")
        self.assertEqual(complete["chunks_completed"], 2)
        self.assertEqual(complete["ai_decision"], "include")
        partial = self.screening._combine_packets([first], self.names, 2)
        self.assertEqual(partial["screening_status"], "chunked_partial")
        self.assertEqual(partial["ai_decision"], "uncertain")

    def test_a_pending_packet_blocks_even_an_explicit_exclusion(self):
        first = self.packet_result(values={"recommender_context": "no"})
        second = self.packet_result(status="pending_limit", fingerprint="synthetic_pending")
        combined = self.screening._combine_packets([first, second], self.names, 2)
        self.assertEqual(combined["ai_decision"], "uncertain")
        self.assertEqual(combined["screening_status"], "chunked_partial")
        self.assertEqual(combined["chunks_completed"], 1)
        self.assertEqual(json.loads(combined["chunk_errors_json"])[0]["status"], "pending_limit")

    def test_packet_count_mismatch_cannot_be_complete(self):
        results = [self.packet_result(), self.packet_result(status="pending_limit")]
        combined = self.screening._combine_packets(results, self.names, 1)
        self.assertEqual(combined["ai_decision"], "uncertain")
        self.assertEqual(combined["screening_status"], "chunked_partial")

    def test_zero_packet_total_cannot_be_complete(self):
        combined = self.screening._combine_packets([], self.names, 0)
        self.assertEqual(combined["ai_decision"], "uncertain")
        self.assertEqual(combined["screening_status"], "chunked_partial")

    def test_contradictory_criterion_quotes_remain_unclear_with_both_citations(self):
        first = self.packet_result(values={"investigated_data_property": "yes"})
        second = self.packet_result(values={"investigated_data_property": "no"}, fingerprint="synthetic_packet_2")
        combined = self.screening._combine_packets([first, second], self.names, 2)
        criterion = next(item for item in json.loads(combined["criteria_json"])
                         if item["criterion"] == "investigated_data_property")
        self.assertEqual(criterion["value"], "unclear")
        self.assertEqual({item["value"] for item in criterion["packet_evidence"]}, {"yes", "no"})
        self.assertEqual(combined["ai_decision"], "uncertain")

    def test_conflicting_study_kinds_block_inclusion(self):
        combined = self.screening._combine_packets([
            self.packet_result(kind="primary"),
            self.packet_result(kind="secondary", fingerprint="synthetic_packet_2"),
        ], self.names, 2)
        self.assertEqual(combined["ai_study_kind"], "unclear")
        self.assertEqual(combined["ai_decision"], "uncertain")

    def test_extraction_merge_preserves_distinct_facts_without_double_counting_identical_payload(self):
        first_fact = {
            "entity_type": "measure", "entity_label": "Synthetic measure A",
            "fields": {"formula_or_procedure": "Synthetic numerator / synthetic denominator", "source_attribution": "own_study"},
            "passage_id": "BODY_00000000_00000100", "quote": "Synthetic method A.",
        }
        duplicate_fact = {**first_fact, "passage_id": "BODY_00000100_00000200"}
        distinct_fact = {**first_fact, "entity_label": "Synthetic measure B",
                         "fields": {"formula_or_procedure": "A different synthetic procedure", "source_attribution": "own_study"}}
        combined = self.screening._combine_packets([
            self.packet_result(extraction=[first_fact]),
            self.packet_result(extraction=[duplicate_fact, distinct_fact], fingerprint="synthetic_packet_2"),
        ], self.names, 2)
        observations = json.loads(combined["extraction_json"])
        self.assertEqual(len(observations), 2)
        self.assertEqual({item["entity_label"] for item in observations}, {"Synthetic measure A", "Synthetic measure B"})
        self.assertTrue(all(item["quote"] and item["passage_id"] and item["packet_index"] for item in observations))

    def test_conflicting_appraisal_ratings_are_not_averaged(self):
        judgement = {"domain": "construct_clarity", "rating": "yes",
                     "justification": "Synthetic supported judgement.",
                     "passage_id": "BODY_00000000_00000100", "quote": "Synthetic definition."}
        combined = self.screening._combine_packets([
            self.packet_result(appraisal=[judgement]),
            self.packet_result(appraisal=[{**judgement, "rating": "no"}], fingerprint="synthetic_packet_2"),
        ], self.names, 2)
        appraisal = json.loads(combined["appraisal_json"])
        self.assertEqual({item["domain"] for item in appraisal}, set(self.screening.APPRAISAL_DOMAINS))
        clarity = next(item for item in appraisal if item["domain"] == "construct_clarity")
        self.assertEqual(clarity["rating"], "unclear")
        self.assertEqual({item["rating"] for item in clarity["packet_evidence"]}, {"yes", "no"})
        self.assertNotIn("quality_score", combined)

    def test_partial_report_does_not_turn_appraisal_into_a_completed_judgement(self):
        judgement = {"domain": "construct_clarity", "rating": "yes",
                     "justification": "Synthetic supported judgement.",
                     "passage_id": "BODY_00000000_00000100", "quote": "Synthetic definition."}
        combined = self.screening._combine_packets([self.packet_result(appraisal=[judgement])], self.names, 2)
        self.assertTrue(all(item["rating"] == "unclear" for item in json.loads(combined["appraisal_json"])))


    def test_identical_packet_is_reused_and_model_identity_change_invalidates_cache(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-screening-cache-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        db = self.screening._open_checkpoint(root / "screening.sqlite")
        self.addCleanup(db.close)
        quote = "Synthetic target-study methodological fixture."
        record = self.record(quote)
        evidence = self.evidence(record)
        body_id = next(key for key in evidence["passages"] if key.startswith("BODY_"))
        response_data = {
            "decision": "uncertain", "study_kind": "primary", "rationale": "Synthetic source packet.",
            "criteria": [{"criterion": name, "value": "yes", "source_attribution": "own_study", "explanation": "Synthetic grounded fixture.",
                          "passage_id": body_id, "quote": quote} for name in self.names],
            "extraction": [], "appraisal": [], "iso_mapping": [],
        }

        class FakeModel:
            ready = True
            unavailable_error = ""
            model_digest = "synthetic_model_digest_A"
            model = "synthetic:fixture"
            output = 1024
            context = 32000
            calls = 0

            def call(self, messages, schema):
                self.calls += 1
                return {"done": True, "done_reason": "stop", "prompt_eval_count": 40,
                        "message": {"content": json.dumps(response_data)}}

        client = FakeModel()
        config = {"model": {"name": client.model}, "protocol": {}, "screening": {"retry_errors": True}}
        schema = self.screening._schema("full_text", self.names)
        budget = {"attempted": 0, "limit": 0}
        arguments = (record, "full_text", config, root, client, db, evidence, "synthetic trusted system", self.names, schema, budget)
        first = self.screening._assess_input(*arguments)
        second = self.screening._assess_input(*arguments)
        self.assertEqual(first["screening_status"], "assessed")
        self.assertFalse(first["cache_reused"])
        self.assertTrue(second["cache_reused"])
        self.assertEqual(client.calls, 1)
        self.assertEqual(budget["attempted"], 1)
        client.model_digest = "synthetic_model_digest_B"
        third = self.screening._assess_input(*arguments)
        self.assertFalse(third["cache_reused"])
        self.assertEqual(client.calls, 2)
        self.assertNotEqual(first["assessment_fingerprint"], third["assessment_fingerprint"])

    def test_prior_work_evidence_cannot_establish_the_target_study_eligibility(self):
        quote = "Synthetic cited study, rather than the target study, assessed a condition."
        body_id = "BODY_00000000_00000100"
        data = {
            "decision": "include", "study_kind": "primary", "rationale": "Synthetic fixture.",
            "criteria": [{"criterion": name, "value": "yes", "source_attribution": "own_study",
                          "explanation": "Synthetic fixture.", "passage_id": body_id, "quote": quote}
                         for name in self.names],
            "extraction": [], "appraisal": [], "iso_mapping": [],
        }
        data["criteria"][0]["source_attribution"] = "prior_work"
        with self.assertRaises(self.screening.AssessmentError):
            self.screening._validate(data, {body_id: quote}, "full_text", self.names, {})

    def test_a_quote_outside_the_cited_body_is_rejected(self):
        quote = "Synthetic valid source statement."
        body_id = "BODY_00000000_00000100"
        data = {
            "decision": "include", "study_kind": "primary", "rationale": "Synthetic fixture.",
            "criteria": [{"criterion": name, "value": "yes", "source_attribution": "own_study",
                          "explanation": "Synthetic fixture.", "passage_id": body_id, "quote": quote}
                         for name in self.names],
            "extraction": [], "appraisal": [], "iso_mapping": [],
        }
        data["criteria"][0]["quote"] = "A fabricated quotation absent from the supplied source."
        with self.assertRaises(self.screening.AssessmentError):
            self.screening._validate(data, {body_id: quote}, "full_text", self.names, {})

    def test_full_text_operational_criterion_cannot_rely_on_title_only(self):
        quote = "Synthetic target-study operational statement."
        body_id = "BODY_00000000_00000100"
        data = {
            "decision": "include", "study_kind": "primary", "rationale": "Synthetic fixture.",
            "criteria": [{"criterion": name, "value": "yes", "source_attribution": "own_study",
                          "explanation": "Synthetic fixture.", "passage_id": body_id, "quote": quote}
                         for name in self.names],
            "extraction": [], "appraisal": [], "iso_mapping": [],
        }
        next(item for item in data["criteria"] if item["criterion"] == "operational_assessment")["passage_id"] = "TITLE"
        with self.assertRaises(self.screening.AssessmentError):
            self.screening._validate(data, {body_id: quote, "TITLE": quote}, "full_text", self.names, {})

    def test_extraction_requires_explicit_target_study_attribution(self):
        quote = "Synthetic target-study methodological statement."
        body_id = "BODY_00000000_00000100"
        data = {
            "decision": "include", "study_kind": "primary", "rationale": "Synthetic fixture.",
            "criteria": [{"criterion": name, "value": "yes", "source_attribution": "own_study",
                          "explanation": "Synthetic fixture.", "passage_id": body_id, "quote": quote}
                         for name in self.names],
            "extraction": [{"entity_type": "measure", "entity_label": "Synthetic measure",
                            "fields": {"formula_or_procedure": "Synthetic procedure."},
                            "passage_id": body_id, "quote": quote}],
            "appraisal": [], "iso_mapping": [],
        }
        with self.assertRaises(self.screening.AssessmentError):
            self.screening._validate(data, {body_id: quote}, "full_text", self.names, {})


if __name__ == "__main__":
    unittest.main()
