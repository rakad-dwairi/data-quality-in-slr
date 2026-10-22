"""Offline contract checks for revised scope/attribution instructions.

Synthetic replies test the accepted evidence semantics and generation identity.
They do not claim that a live model will follow the revised prompt correctly.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from slr_pipeline import screening
from slr_pipeline.evidence import normalize_criteria


class ScreeningPromptGuidanceTests(unittest.TestCase):
    def setUp(self):
        network = mock.patch("requests.sessions.Session.request",
                             side_effect=AssertionError("Real network/model calls are forbidden"))
        network.start()
        self.addCleanup(network.stop)
        self.names = list(screening.CORE_CRITERIA)

    def reply(self, *, decision="uncertain", kind="primary", observations=None):
        observations = observations or {}
        criteria = []
        for name in self.names:
            observed = observations.get(name, {})
            criteria.append({"criterion": name, "value": observed.get("value", "unclear"),
                             "source_attribution": observed.get("source_attribution", "unclear"),
                             "explanation": observed.get("explanation", "Synthetic missing detail remains unclear."),
                             "passage_id": observed.get("passage_id", "ABSTRACT" if observed.get("quote") else ""),
                             "quote": observed.get("quote", "")})
        return {"decision": decision, "study_kind": kind,
                "rationale": "Synthetic evidence proposal, pending independent review.", "criteria": criteria}

    @staticmethod
    def observed(value, quote, attribution="own_study"):
        return {"value": value, "quote": quote, "source_attribution": attribution,
                "explanation": "Synthetic target-report evidence for this criterion."}

    def validate(self, data, source):
        passages = {"ABSTRACT": source}
        normalized, adjustments = normalize_criteria(data, passages, "title_abstract")
        result = screening._validate(normalized, passages, "title_abstract", self.names, {})
        return result, normalized, adjustments

    def test_schema_guidance_preserves_field_shape_and_attribution_enums(self):
        schema = screening._schema("title_abstract", self.names)
        self.assertEqual(set(schema["required"]), {"decision", "rationale", "study_kind", "criteria"})
        item = schema["properties"]["criteria"]["items"]
        self.assertEqual(set(item["required"]),
                         {"criterion", "value", "source_attribution", "explanation", "passage_id", "quote"})
        self.assertEqual(item["properties"]["source_attribution"]["enum"],
                         ["own_study", "prior_work", "unclear"])
        self.assertIn("independent of topical eligibility", item["properties"]["source_attribution"]["description"])
        self.assertIn("exact key", item["properties"]["passage_id"]["description"])
        self.assertIn("contiguous", item["properties"]["quote"]["description"])
        self.assertIn("independent of review scope", schema["properties"]["study_kind"]["description"])

    def test_out_of_scope_original_analysis_can_exclude_without_five_negatives(self):
        aim = "We estimate river-water contamination from sensor samples."
        original = "We analysed 240 original samples in a controlled field experiment."
        data = self.reply(decision="exclude", observations={
            "recommender_context": self.observed("no", aim),
            "primary_research": self.observed("yes", original),
        })
        result, normalized, adjustments = self.validate(data, aim + " " + original)
        self.assertEqual(result["ai_decision"], "exclude")
        self.assertEqual(result["ai_reason_code"], "NOT_RECSYS")
        self.assertEqual(result["ai_study_kind"], "primary")
        self.assertEqual([item["value"] for item in normalized["criteria"]],
                         ["no", "unclear", "unclear", "unclear", "yes"])
        self.assertEqual(adjustments, [])

    def test_course_recommender_is_retained_when_property_method_is_unreported(self):
        aim = "We develop a course recommendation system."
        original = "We evaluate the system in a new student experiment."
        result, normalized, adjustments = self.validate(self.reply(observations={
            "recommender_context": self.observed("yes", aim),
            "primary_research": self.observed("yes", original),
        }), aim + " " + original)
        self.assertEqual(result["ai_decision"], "uncertain")
        self.assertEqual(result["ai_study_kind"], "primary")
        self.assertEqual(normalized["criteria"][0]["value"], "yes")
        self.assertEqual(adjustments, [])

    def test_in_scope_original_data_condition_experiment_can_be_inclusion_candidate(self):
        statements = {
            "recommender_context": "We study a course recommendation system.",
            "investigated_data_property": "We investigate missing student interaction data.",
            "operational_assessment": "We randomly remove 10 percent of recorded interactions and compare conditions.",
            "eligible_data_object": "The measured object is the system's student-item interaction data.",
            "primary_research": "We conduct a new controlled experiment on our observed interaction dataset.",
        }
        result, _, adjustments = self.validate(self.reply(decision="include", observations={
            name: self.observed("yes", quote) for name, quote in statements.items()
        }), " ".join(statements.values()))
        self.assertEqual(result["ai_decision"], "include")
        self.assertEqual(result["ai_reason_code"], "INCLUDED_CANDIDATE")
        self.assertEqual(adjustments, [])

    def test_cited_only_property_does_not_become_target_report_evidence(self):
        aim = "We develop a course recommendation system."
        cited = "Smith et al. measured recommendation-data sparsity."
        original = "We evaluate a new architecture in a student experiment."
        result, normalized, adjustments = self.validate(self.reply(decision="include", observations={
            "recommender_context": self.observed("yes", aim),
            "investigated_data_property": self.observed("yes", cited, "prior_work"),
            "primary_research": self.observed("yes", original),
        }), aim + " " + cited + " " + original)
        property_item = normalized["criteria"][1]
        self.assertEqual(result["ai_decision"], "uncertain")
        self.assertEqual(property_item["value"], "unclear")
        self.assertEqual(property_item["source_attribution"], "prior_work")
        self.assertTrue(adjustments)

    def test_literature_synthesis_requires_explicit_secondary_report_evidence(self):
        source = "This report systematically reviews prior studies of recommendation-data quality."
        result, _, adjustments = self.validate(self.reply(decision="secondary", kind="secondary", observations={
            "primary_research": self.observed("no", source),
        }), source)
        self.assertEqual(result["ai_decision"], "secondary")
        self.assertEqual(result["ai_reason_code"], "SECONDARY_RESEARCH")
        self.assertEqual(adjustments, [])

    def test_prompt_version_changes_generation_identity_without_overwriting_old_events(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-prompt-guidance-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        db = screening._open_checkpoint(root / "checkpoints/screening.sqlite")
        self.addCleanup(db.close)
        source = "We develop a course recommendation system."
        reply = self.reply(observations={"recommender_context": self.observed("yes", source)})
        record = {"record_id": "R_SYNTHETIC_PROMPT_REVISION", "title": "Synthetic course recommendation fixture",
                  "abstract": source}
        evidence = {"stage": "title_abstract", "record_id": record["record_id"],
                    "required_criteria": self.names, "passages": screening._passages(record, "title_abstract")}
        schema = screening._schema("title_abstract", self.names)

        class SyntheticModel:
            ready = True
            unavailable_error = ""
            model_digest = "synthetic_model_digest"
            model = "synthetic:fixture"
            output = 1024
            context = 32768
            calls = 0

            def call(self, messages, supplied_schema):
                self.calls += 1
                self.last_messages, self.last_schema = messages, supplied_schema
                return {"done": True, "done_reason": "stop", "prompt_eval_count": 80,
                        "message": {"content": json.dumps(reply)}}

        client = SyntheticModel()
        config = {"model": {"name": client.model}, "protocol": {}, "screening": {"retry_errors": True}}
        budget = {"attempted": 0, "limit": 0}
        args = (record, "title_abstract", config, root, client, db, evidence,
                screening.SYSTEM_PROMPT, self.names, schema, budget)
        # Hold every input equal: the explicit prompt version alone must separate
        # old generations from this revision, even before prompt/schema hashes.
        with mock.patch.object(screening, "PROMPT_VERSION", "2026-10-21.3"):
            old = screening._assess_input(*args)
        old_events = {path: path.read_bytes() for path in (root / "checkpoints/model_attempts").glob("*.json")}
        current = screening._assess_input(*args)
        repeated = screening._assess_input(*args)
        self.assertEqual(client.calls, 2)
        self.assertEqual(budget["attempted"], 2)
        self.assertNotEqual(old["assessment_fingerprint"], current["assessment_fingerprint"])
        self.assertFalse(current["cache_reused"])
        self.assertTrue(repeated["cache_reused"])
        self.assertEqual(client.last_messages[0], {"role": "system", "content": screening.SYSTEM_PROMPT})
        self.assertEqual(client.last_schema, schema)
        self.assertEqual({path: path.read_bytes() for path in old_events}, old_events)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM assessments").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
