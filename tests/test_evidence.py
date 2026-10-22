"""Synthetic, offline regression checks for conservative citation recovery.

No real paper metadata, external services, model inference, or database access
is needed.  These tests exercise only source strings and validation functions.
"""
from __future__ import annotations

import copy
import unittest
from unittest import mock

from slr_pipeline.evidence import normalize_criteria


def criterion(name="recommender_context", value="yes", attribution="own_study",
              passage_id="ABSTRACT", quote="Synthetic target-report statement."):
    return {"criterion": name, "value": value, "source_attribution": attribution,
            "explanation": "Synthetic explanation.", "passage_id": passage_id, "quote": quote}


def response(items, decision="include", kind="primary"):
    return {"decision": decision, "study_kind": kind,
            "rationale": "Synthetic response, not a scientific conclusion.", "criteria": items}


class EvidenceNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.network = mock.patch("requests.sessions.Session.request",
                                  side_effect=AssertionError("Network/model calls forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def normalize(self, item, passages=None, stage="title_abstract"):
        return normalize_criteria(response([item]), passages or {"ABSTRACT": "Synthetic target-report statement."}, stage)

    def test_valid_response_is_deep_copied_and_unchanged(self):
        original = response([criterion()])
        expected = copy.deepcopy(original)
        result, adjustments = normalize_criteria(original, {"ABSTRACT": "Synthetic target-report statement."}, "title_abstract")
        self.assertEqual(result, expected)
        self.assertEqual(adjustments, [])
        self.assertIsNot(result, original)
        self.assertIsNot(result["criteria"], original["criteria"])
        result["criteria"][0]["value"] = "unclear"
        self.assertEqual(original, expected)

    def test_unknown_id_with_unique_literal_quote_is_repaired(self):
        item = criterion(passage_id="TITLE_ABSTRACT", quote="Synthetic report.")
        passages = {"TITLE": "Synthetic title", "ABSTRACT": "Prefix. Synthetic report. Tail."}
        result, adjustments = self.normalize(item, passages)
        self.assertEqual(result["criteria"][0]["passage_id"], "ABSTRACT")
        self.assertEqual(result["criteria"][0]["value"], "yes")
        span = adjustments[0]["source_span"]
        self.assertEqual(passages["ABSTRACT"][span["start"]:span["end"]], "Synthetic report.")
        self.assertEqual(adjustments[0]["original"]["passage_id"], "TITLE_ABSTRACT")

    def test_known_but_mismatched_id_is_not_redirected(self):
        result, _ = self.normalize(criterion(passage_id="TITLE", quote="Synthetic report."),
                                   {"TITLE": "Other title", "ABSTRACT": "Synthetic report."})
        self.assertEqual(result["criteria"][0]["value"], "unclear")
        self.assertEqual(result["criteria"][0]["quote"], "")

    def test_ambiguous_unknown_id_across_passages_is_not_repaired(self):
        result, _ = self.normalize(criterion(passage_id="UNKNOWN", quote="Same phrase"),
                                   {"TITLE": "Same phrase", "ABSTRACT": "Same phrase"})
        self.assertEqual(result["criteria"][0]["value"], "unclear")

    def test_repeated_literal_occurrences_do_not_supply_unique_repair_span(self):
        result, _ = self.normalize(criterion(passage_id="UNKNOWN", quote="Repeat"),
                                   {"ABSTRACT": "Repeat and Repeat"})
        self.assertEqual(result["criteria"][0]["value"], "unclear")

    def test_already_valid_repeated_quote_needs_no_repair(self):
        result, adjustments = self.normalize(criterion(quote="Repeat"), {"ABSTRACT": "Repeat and Repeat"})
        self.assertEqual(result["criteria"][0]["value"], "yes")
        self.assertEqual(adjustments, [])

    def test_whitespace_only_repairs_return_original_substring_and_offsets(self):
        source = "Prefix: as$x$-augmentation improves rank-Nscore on  synthetic\nrecords. Tail."
        quote = "as $x$-augmentation improves rank-N score on synthetic records."
        original = criterion(quote=quote)
        result, adjustments = self.normalize(original, {"ABSTRACT": source})
        fixed = result["criteria"][0]
        self.assertEqual(fixed["value"], "yes")
        self.assertEqual(fixed["quote"], "as$x$-augmentation improves rank-Nscore on  synthetic\nrecords.")
        span = adjustments[0]["source_span"]
        self.assertEqual(source[span["start"]:span["end"]], fixed["quote"])
        self.assertEqual(adjustments[0]["original"]["quote"], quote)
        self.assertEqual(original["quote"], quote)

    def test_unicode_whitespace_is_mapped_without_other_unicode_normalization(self):
        source = "Prefix. α\u00a0β\tγ\nδ. Suffix."
        result, _ = self.normalize(criterion(quote="α β γ δ."), {"ABSTRACT": source})
        self.assertEqual(result["criteria"][0]["quote"], "α\u00a0β\tγ\nδ.")

    def test_whitespace_repair_needs_unique_span_within_passage(self):
        result, _ = self.normalize(criterion(quote="a b."), {"ABSTRACT": "ab. and a\tb."})
        self.assertEqual(result["criteria"][0]["value"], "unclear")
        self.assertEqual(result["criteria"][0]["passage_id"], "")

    def test_unknown_id_and_whitespace_guess_are_not_combined(self):
        result, _ = self.normalize(criterion(passage_id="UNKNOWN", quote="rank-N score."),
                                   {"ABSTRACT": "rank-Nscore."})
        self.assertEqual(result["criteria"][0]["value"], "unclear")

    def test_non_whitespace_changes_are_never_repaired(self):
        cases = [
            ("A synthetic result is 5.", "A synthetic result is 6."),
            ("A synthetic result is 5.", "a synthetic result is 5."),
            ("There is no effect.", "There is effect."),
            ("A result (with detail) improves.", "A result improves."),
            ("A result, improves.", "A result improves."),
            ("A result—improves.", "A result-improves."),
            ("A synthetic ﬁeld.", "A synthetic field."),
            ("A synthetic é.", "A synthetic e\u0301."),
            ("A synthetic zero\u200bwidth.", "A synthetic zerowidth."),
        ]
        for source, quote in cases:
            with self.subTest(source=source, quote=quote):
                result, _ = self.normalize(criterion(quote=quote), {"ABSTRACT": source})
                self.assertEqual(result["criteria"][0]["value"], "unclear")
                self.assertEqual(result["criteria"][0]["quote"], "")

    def test_prior_work_or_unclear_attribution_is_never_relabelled(self):
        for attribution in ("prior_work", "unclear"):
            for value in ("yes", "no"):
                with self.subTest(attribution=attribution, value=value):
                    result, adjustments = self.normalize(criterion(value=value, attribution=attribution))
                    item = result["criteria"][0]
                    self.assertEqual(item["value"], "unclear")
                    self.assertEqual(item["source_attribution"], attribution)
                    self.assertEqual(item["quote"], "Synthetic target-report statement.")
                    self.assertEqual(adjustments[0]["original"]["value"], value)

    def test_already_unclear_prior_work_citation_is_kept_if_exact(self):
        result, adjustments = self.normalize(criterion(value="unclear", attribution="prior_work"))
        self.assertEqual(result["criteria"][0]["source_attribution"], "prior_work")
        self.assertEqual(adjustments, [])

    def test_unsupported_unclear_quote_is_cleared(self):
        result, adjustments = self.normalize(criterion(value="unclear", quote="Absent source quote."))
        self.assertEqual(result["criteria"][0]["quote"], "")
        self.assertEqual(result["criteria"][0]["passage_id"], "")
        self.assertEqual(adjustments[0]["original"]["quote"], "Absent source quote.")

    def test_blank_evidence_cannot_support_a_definite_judgement(self):
        result, _ = self.normalize(criterion(quote=" \n\t"))
        self.assertEqual(result["criteria"][0]["value"], "unclear")
        self.assertEqual(result["criteria"][0]["quote"], "")

    def test_fulltext_methods_require_body_after_anchor_recovery(self):
        for name in ("investigated_data_property", "operational_assessment", "full_text_sufficient"):
            with self.subTest(name=name):
                result, adjustments = self.normalize(criterion(name=name), stage="full_text")
                self.assertEqual(result["criteria"][0]["value"], "unclear")
                self.assertEqual(result["criteria"][0]["quote"], "")
                self.assertEqual(adjustments[-1]["reason"], "full_text_body_evidence_required")

    def test_fulltext_body_whitespace_anchor_is_allowed(self):
        source = "Synthetic target-report\nstatement."
        result, _ = self.normalize(criterion(name="operational_assessment", passage_id="BODY_0_34"),
                                   {"BODY_0_34": source}, stage="full_text")
        self.assertEqual(result["criteria"][0]["value"], "yes")
        self.assertEqual(result["criteria"][0]["quote"], source)

    def test_fulltext_context_can_still_cite_title(self):
        result, _ = self.normalize(criterion(passage_id="TITLE"),
                                   {"TITLE": "Synthetic target-report statement."}, stage="full_text")
        self.assertEqual(result["criteria"][0]["value"], "yes")

    def test_structural_errors_are_left_for_strict_validation(self):
        cases = [None, [], {}, {"criteria": "bad"}, {"criteria": [None]},
                 response([{"criterion": "recommender_context"}]),
                 response([{**criterion(), "value": "maybe"}]),
                 response([{**criterion(), "source_attribution": "invented"}]),
                 response([{**criterion(), "quote": ["bad"]}])]
        for original in cases:
            with self.subTest(original=original):
                result, adjustments = normalize_criteria(original, {"ABSTRACT": "Synthetic target-report statement."}, "title_abstract")
                self.assertEqual(result, original)
                self.assertEqual(adjustments, [])

    def test_invalid_passage_container_is_not_silently_fixed(self):
        original = response([criterion()])
        for passages in (None, [], {"ABSTRACT": 123}):
            result, adjustments = normalize_criteria(original, passages, "title_abstract")
            self.assertEqual(result, original)
            self.assertEqual(adjustments, [])

    def test_normalization_is_idempotent(self):
        original = response([criterion(quote="Synthetic target-report statement.")])
        passages = {"ABSTRACT": "Synthetic target-report\nstatement."}
        first, adjustments = normalize_criteria(original, passages, "title_abstract")
        second, repeated = normalize_criteria(first, passages, "title_abstract")
        self.assertTrue(adjustments)
        self.assertEqual(second, first)
        self.assertEqual(repeated, [])


class EvidenceStrictValidationTests(unittest.TestCase):
    def setUp(self):
        self.network = mock.patch("requests.sessions.Session.request",
                                  side_effect=AssertionError("Network/model calls forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)
        from slr_pipeline import screening
        self.screening = screening
        self.names = list(screening.CORE_CRITERIA)

    def validate(self, data, passages):
        normalized, adjustments = normalize_criteria(data, passages, "title_abstract")
        return self.screening._validate(normalized, passages, "title_abstract", self.names, {}), normalized, adjustments

    def test_synthetic_primary_response_preserves_four_yes_and_invalid_fifth_unclear(self):
        source = "We use as$x$-augmentation on rank-Nscore. Our experiment compares methods (including controls) across three datasets."
        items = [criterion(name=name, quote="We use as $x$-augmentation on rank-N score.") for name in self.names[:-1]]
        items.append(criterion(name="primary_research", quote="Our experiment compares methods across three datasets."))
        assessed, normalized, adjustments = self.validate(response(items), {"ABSTRACT": source})
        self.assertEqual(assessed["screening_status"], "assessed")
        self.assertEqual(assessed["ai_decision"], "uncertain")
        self.assertEqual([item["value"] for item in normalized["criteria"]], ["yes"] * 4 + ["unclear"])
        self.assertEqual(len([item for item in adjustments if item["action"] == "quote_anchor_repaired"]), 4)

    def test_synthetic_secondary_include_contradiction_cannot_include_or_exclude(self):
        source = "We review synthetic recommendation research. This report is a synthetic review."
        items = [criterion(passage_id="TITLE_ABSTRACT", quote="We review synthetic recommendation research.")]
        items += [criterion(name=name, value="no", attribution="prior_work", quote="This report is a synthetic review.") for name in self.names[1:]]
        data = response(items, decision="include", kind="secondary")
        assessed, normalized, _ = self.validate(data, {"ABSTRACT": source})
        self.assertEqual(assessed["ai_decision"], "uncertain")
        self.assertEqual(assessed["ai_study_kind"], "secondary")
        self.assertEqual(normalized["decision"], "include")
        self.assertEqual(normalized["criteria"][0]["passage_id"], "ABSTRACT")
        self.assertEqual([item["value"] for item in normalized["criteria"]], ["yes"] + ["unclear"] * 4)

    def test_synthetic_editorial_prior_work_negatives_cannot_automatically_exclude(self):
        source = "Introduction to a synthetic topic collection."
        data = response([criterion(name=name, value="no", attribution="prior_work", quote=source)
                         for name in self.names], decision="exclude", kind="secondary")
        assessed, normalized, _ = self.validate(data, {"ABSTRACT": source})
        self.assertEqual(assessed["ai_decision"], "uncertain")
        self.assertTrue(all(item["value"] == "unclear" for item in normalized["criteria"]))

    def test_malformed_values_still_fail_strict_validation(self):
        data = response([criterion(name=name) for name in self.names])
        data["criteria"][0]["value"] = "maybe"
        normalized, _ = normalize_criteria(data, {"ABSTRACT": "Synthetic target-report statement."}, "title_abstract")
        with self.assertRaises(self.screening.AssessmentError):
            self.screening._validate(normalized, {"ABSTRACT": "Synthetic target-report statement."}, "title_abstract", self.names, {})


if __name__ == "__main__":
    unittest.main()
