"""Exact ordered source-fragment restoration; no inference or fuzzy evidence."""
import copy
import json
import unittest

from slr_pipeline import screening
from slr_pipeline.evidence import _unique_ellipsis_span, normalize_criteria


class EllipsisEvidenceTests(unittest.TestCase):
    first = "We investigate the synthetic data condition."
    last = "We measure the recorded experimental outcomes."

    def data(self, quote):
        return {"criteria": [{"criterion": "recommender_context", "value": "yes", "source_attribution": "own_study",
                              "explanation": "Synthetic evidence only.", "passage_id": "ABSTRACT", "quote": quote}]}

    def test_omitted_context_is_restored_as_one_complete_original_span(self):
        source = self.first + " The intervening qualifications remain visible. " + self.last
        original = self.data(self.first + " ... " + self.last)
        before = copy.deepcopy(original)
        result, audit = normalize_criteria(original, {"ABSTRACT": source}, "title_abstract")
        self.assertEqual(original, before)
        self.assertEqual(result["criteria"][0]["value"], "yes")
        self.assertEqual(result["criteria"][0]["quote"], source)
        self.assertEqual(audit[0]["reason"], "unique_ordered_ellipsis_original_span")
        self.assertEqual(audit[0]["source_span"]["text"], source)
        self.assertEqual(audit[0]["original"]["quote"], before["criteria"][0]["quote"])

    def test_unicode_ellipsis_and_whitespace_preserve_every_nonwhitespace_character(self):
        source = self.first.replace(" ", "\n") + " Original omitted source context. " + self.last
        span = _unique_ellipsis_span(source, self.first + " \u2026 " + self.last)
        self.assertEqual(source[slice(*span)], source)

    def test_ambiguous_reversed_short_leading_and_trailing_fragments_are_refused(self):
        source = self.first + " Context. " + self.last
        for candidate, text in ((self.last + " ... " + self.first, source),
                                 (self.first + " ... " + self.last, source + self.first),
                                 ("We ... " + self.last, source),
                                 ("... " + self.last, source), (self.first + " ...", source)):
            with self.subTest(candidate=candidate):
                self.assertIsNone(_unique_ellipsis_span(text, candidate))

    def test_numeric_case_punctuation_and_oversized_omissions_are_not_repaired(self):
        source = self.first + " Context. " + self.last
        for quote, text in ((self.first.lower() + " ... " + self.last, source),
                            (self.first.replace(".", "!") + " ... " + self.last, source),
                            ("We observe 35 synthetic measurements. ... " + self.last,
                             "We observe 36 synthetic measurements. Context. " + self.last),
                            (self.first + " ... " + self.last, self.first + "x" * 1001 + self.last)):
            self.assertIsNone(_unique_ellipsis_span(text, quote))

    def test_wrong_or_unknown_passage_is_not_combined_with_fragment_guessing(self):
        source = self.first + " Context. " + self.last
        for passage in ("TITLE", "INVENTED"):
            data = self.data(self.first + " ... " + self.last)
            data["criteria"][0]["passage_id"] = passage
            result, _ = normalize_criteria(data, {"TITLE": "Different passage.", "ABSTRACT": source}, "title_abstract")
            self.assertEqual(result["criteria"][0]["value"], "unclear")

    def test_prior_work_remains_unclear_even_when_source_anchor_is_restored(self):
        source = self.first + " Context. " + self.last
        data = self.data(self.first + " ... " + self.last)
        data["criteria"][0]["source_attribution"] = "prior_work"
        result, _ = normalize_criteria(data, {"ABSTRACT": source}, "title_abstract")
        self.assertEqual(result["criteria"][0]["value"], "unclear")
        self.assertEqual(result["criteria"][0]["source_attribution"], "prior_work")

    def test_appraisal_anchor_restored_without_altering_rating_or_scientific_fields(self):
        source = self.first + " Full original qualification. " + self.last
        names = list(screening.CORE_CRITERIA) + ["full_text_sufficient"]
        body = "BODY_00000000_00000500"
        data = {"decision": "uncertain", "study_kind": "primary", "rationale": "Synthetic fixture.",
                "criteria": [{"criterion": name, "value": "unclear", "source_attribution": "unclear",
                              "explanation": "Unreported.", "quote": "", "passage_id": ""} for name in names],
                "extraction": [], "iso_mapping": [],
                "appraisal": [{"domain": "construct_clarity", "rating": "yes", "justification": "Synthetic fixture.",
                               "passage_id": body, "quote": self.first + " ... " + self.last}]}
        normalized, audit = normalize_criteria(data, {body: source}, "full_text")
        result = screening._validate(normalized, {body: source}, "full_text", names, {})
        self.assertEqual(result["ai_decision"], "uncertain")
        self.assertEqual(normalized["appraisal"][0]["rating"], "yes")
        self.assertEqual(normalized["appraisal"][0]["quote"], source)
        self.assertEqual(audit[0]["citation_path"], ["appraisal", 0])

    def test_malformed_noncriterion_citation_is_left_for_structural_validation(self):
        data = {"criteria": [], "appraisal": [{"passage_id": [], "quote": "Untrusted malformed citation"}]}
        normalized, audit = normalize_criteria(data, {"ABSTRACT": "Synthetic source"}, "full_text")
        self.assertEqual(normalized, data)
        self.assertEqual(audit, [])


if __name__ == "__main__":
    unittest.main()
