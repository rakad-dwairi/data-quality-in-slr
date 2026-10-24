"""Unsupported full-text facts are omitted, never promoted to verified evidence."""
import copy
import json
import unittest

from slr_pipeline import screening
from slr_pipeline.evidence import _APPRAISAL_DOMAINS, normalize_criteria


class FullTextEvidenceRecoveryTests(unittest.TestCase):
    source = "Synthetic own experiment measures recommendation interaction data."
    body = "BODY_00000000_00000100"

    def data(self):
        names = list(screening.CORE_CRITERIA) + ["full_text_sufficient"]
        return {"decision": "include", "study_kind": "primary", "rationale": "Synthetic implementation fixture only.",
                "criteria": [{"criterion": name, "value": "yes", "source_attribution": "own_study",
                              "explanation": "Synthetic grounded fixture.", "passage_id": self.body, "quote": self.source}
                             for name in names], "extraction": [], "appraisal": [], "iso_mapping": []}

    def result(self, data):
        raw = {"done": True, "done_reason": "stop", "message": {"content": json.dumps(data)}}
        return screening._validated_response(raw, {"passages": {self.body: self.source}}, "full_text",
                                            list(screening.CORE_CRITERIA) + ["full_text_sufficient"], {}, 32768, 8192)

    def observation(self, quote=None):
        return {"entity_type": "measure", "entity_label": "Synthetic condition",
                "fields": {"source_attribution": "own_study", "author_term": "synthetic fixture"},
                "passage_id": self.body, "quote": self.source if quote is None else quote}

    def test_unsupported_fact_is_omitted_and_audited_without_weakening_eligibility(self):
        data = self.data(); observation = self.observation("Invented unsupported scientific result.")
        data["extraction"] = [observation]; original = copy.deepcopy(data)
        result = self.result(data)
        self.assertEqual(result["ai_decision"], "include")
        self.assertEqual(json.loads(result["extraction_json"]), [])
        self.assertEqual(result["evidence_omitted_observations"], 1)
        self.assertTrue(result["evidence_review_required"])
        self.assertEqual(json.loads(result["evidence_adjustments_json"])[0]["original"], observation)
        self.assertEqual(data, original)
        self.assertNotIn("human_decision", result)

    def test_unsupported_appraisal_becomes_unclear_and_original_rating_remains_auditable(self):
        data = self.data(); data["appraisal"] = [{"domain": "design_strength", "rating": "no",
            "justification": "Synthetic unsupported judgement.", "passage_id": self.body, "quote": "Invented quote."}]
        result = self.result(data)
        appraisal = next(x for x in json.loads(result["appraisal_json"]) if x["domain"] == "design_strength")
        self.assertEqual(appraisal["rating"], "unclear")
        self.assertEqual(appraisal["quote"], "")
        self.assertEqual(result["evidence_appraisals_unclear"], 1)
        self.assertEqual(json.loads(result["evidence_adjustments_json"])[0]["original"]["rating"], "no")

    def test_prior_work_is_never_relabelled_as_own_work(self):
        data = self.data(); observation = self.observation(); observation["fields"]["source_attribution"] = "prior_work"
        data["extraction"] = [observation]
        result = self.result(data)
        self.assertEqual(json.loads(result["extraction_json"]), [])
        audit = json.loads(result["evidence_adjustments_json"])[0]
        self.assertEqual(audit["original"]["fields"]["source_attribution"], "prior_work")

    def test_valid_fact_is_preserved_exactly(self):
        data = self.data(); data["extraction"] = [self.observation()]
        result = self.result(data)
        self.assertEqual(json.loads(result["extraction_json"]), data["extraction"])
        self.assertFalse(result["evidence_review_required"])
        self.assertEqual(result["evidence_omitted_observations"], 0)

    def test_structural_errors_are_not_hidden_by_source_omission(self):
        for change in ({"entity_type": "UNKNOWN"}, {"entity_label": []}, {"fields": {"numeric": 123}}):
            data = self.data(); observation = self.observation("Unsupported quote."); observation.update(change)
            data["extraction"] = [observation]
            with self.subTest(change=change), self.assertRaises(screening.AssessmentError):
                self.result(data)

    def test_unsupported_mapping_is_omitted_but_grounded_mapping_still_requires_codebook(self):
        data = self.data(); mapping = {"construct": "Synthetic construct", "characteristic": "Synthetic mapping",
            "relationship": "supported", "justification": "Synthetic only.", "passage_id": self.body, "quote": "Invented quote."}
        data["iso_mapping"] = [mapping]
        result = self.result(data)
        self.assertEqual(json.loads(result["iso_mapping_json"]), [])
        self.assertEqual(result["evidence_omitted_iso_proposals"], 1)
        mapping["quote"] = self.source
        with self.assertRaisesRegex(screening.AssessmentError, "authoritative codebook"):
            self.result(data)

    def test_valid_unclear_appraisal_needs_no_fabricated_citation(self):
        data = self.data(); data["appraisal"] = [{"domain": "design_strength", "rating": "unclear",
            "justification": "Unreported in this packet.", "passage_id": "", "quote": ""}]
        result = self.result(data)
        self.assertEqual(result["evidence_appraisals_unclear"], 0)
        self.assertEqual(json.loads(result["evidence_adjustments_json"]), [])
        self.assertEqual(_APPRAISAL_DOMAINS, set(screening.APPRAISAL_DOMAINS))


if __name__ == "__main__":
    unittest.main()
