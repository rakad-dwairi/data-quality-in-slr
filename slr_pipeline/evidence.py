"""Conservative source-anchor recovery for individual screening criteria.

This module does not assess scientific eligibility, invent source text, or call
a model.  It preserves the raw response by working on a deep copy.  Structural
validation remains the caller's responsibility; malformed fields are left for
the strict validator rather than being converted into valid-looking replies.
"""
from __future__ import annotations

from copy import deepcopy
import re


_VALUES = {"yes", "no", "unclear"}
_ATTRIBUTIONS = {"own_study", "prior_work", "unclear"}
_BODY_REQUIRED = {
    "investigated_data_property", "operational_assessment", "full_text_sufficient",
}
_FIELDS = ("criterion", "value", "source_attribution", "explanation", "passage_id", "quote")
_ENTITIES = {"study", "dataset", "data_object", "measure", "validation", "outcome"}
_APPRAISAL_DOMAINS = {"construct_clarity", "measurement_transparency", "reference_appropriateness",
                      "design_strength", "confounding_control", "evaluation_validity", "statistical_evidence",
                      "robustness", "generalisability", "reproducibility"}
_RELATIONSHIPS = {"explicit", "supported", "tentative", "proxy_non_equivalent", "unsupported"}


def _snapshot(criterion: dict) -> dict:
    return {key: criterion[key] for key in ("value", "source_attribution", "passage_id", "quote")}


def _occurrences(text: str, quote: str):
    """Yield all literal occurrences, including overlapping matches."""
    start = 0
    while True:
        index = text.find(quote, start)
        if index < 0:
            return
        yield index, index + len(quote)
        start = index + 1


def _unique_exact(quote: str, passages: dict) -> tuple[str, int, int] | None:
    match = None
    for passage_id, text in passages.items():
        for start, end in _occurrences(text, quote):
            if match is not None:
                return None
            match = (passage_id, start, end)
    return match


def _compact(text: str) -> tuple[str, list[int]]:
    characters, offsets = [], []
    for offset, character in enumerate(text):
        if not character.isspace():
            characters.append(character)
            offsets.append(offset)
    return "".join(characters), offsets


def _unique_whitespace_span(text: str, quote: str) -> tuple[int, int] | None:
    """Locate one original contiguous span with identical non-whitespace text.

    No case folding, Unicode normalization, punctuation repair, word deletion,
    or fuzzy matching is performed.  This repairs a source anchor only: spacing
    can affect meaning, so it is not a verification of the model's judgement.
    """
    compact_quote, _ = _compact(quote)
    if not compact_quote:
        return None
    compact_text, offsets = _compact(text)
    locations = _occurrences(compact_text, compact_quote)
    first = next(locations, None)
    if first is None or next(locations, None) is not None:
        return None
    start, end = first
    return offsets[start], offsets[end - 1] + 1


def _unique_ellipsis_span(text: str, quote: str) -> tuple[int, int] | None:
    """Restore omitted source context between unique ordered prose fragments.

    Every non-ellipsis character must match (allowing whitespace only). Short,
    ambiguous, reordered, overlapping or widely separated fragments are refused.
    The result is the entire original span, never a stitched quotation.
    """
    fragments = re.split(r"\.{3}|\u2026", quote)
    if not 2 <= len(fragments) <= 4:
        return None
    fragments = [part.strip() for part in fragments]
    if any(len(re.sub(r"\s", "", part)) < 12 or len(part.split()) < 3 for part in fragments):
        return None
    spans = [_unique_whitespace_span(text, part) for part in fragments]
    if any(span is None for span in spans):
        return None
    if any(right[0] < left[1] or right[0] - left[1] > 1000 for left, right in zip(spans, spans[1:])):
        return None
    start, end = spans[0][0], spans[-1][1]
    return (start, end) if end - start <= 3000 else None


def _known_passage_span(text, quote):
    span = _unique_whitespace_span(text, quote)
    if span is not None:
        return span, "unique_whitespace_only_original_span"
    span = _unique_ellipsis_span(text, quote)
    return span, "unique_ordered_ellipsis_original_span"


def normalize_criteria(data, passages, stage):
    """Return ``(deepcopy_of_data, adjustments)`` without deciding eligibility.

    A mismatched quote can be anchored only in either of these bounded ways:
    an unknown passage ID with a unique literal quote occurrence, or a known
    passage with one unique whitespace-only source span. Explicit ellipses may
    also identify unique ordered prose fragments in that same passage; omitted
    context is restored as the full contiguous original span. Unknown-ID and
    approximate-anchor guesses are not combined. Valid quotes are unchanged.

    Unsupported evidence and definite judgements attributed to prior/unclear
    work become ``unclear``.  Attribution is never relabelled.  Invalid citations
    are cleared in the normalized copy, while the audit retains their original
    values and exact original-source offsets for successful anchor repairs.
    Offsets are zero-based Unicode character positions within the passage,
    with an exclusive end, matching Python string slicing.
    """
    normalized = deepcopy(data)
    adjustments = []
    if not isinstance(normalized, dict) or not isinstance(normalized.get("criteria"), list):
        return normalized, adjustments
    if (not isinstance(passages, dict)
            or any(not isinstance(key, str) or not isinstance(value, str)
                   for key, value in passages.items())):
        return normalized, adjustments

    for index, criterion in enumerate(normalized["criteria"]):
        # Do not hide missing keys, unknown enums, or incorrect field types.
        if (not isinstance(criterion, dict)
                or any(not isinstance(criterion.get(key), str) for key in _FIELDS)
                or criterion["value"] not in _VALUES
                or criterion["source_attribution"] not in _ATTRIBUTIONS):
            continue

        def audit(action, reason, before, source_span=None):
            entry = {
                "criterion_index": index, "criterion": criterion["criterion"],
                "action": action, "reason": reason,
                "original": before, "normalized": _snapshot(criterion),
            }
            if source_span is not None:
                entry["source_span"] = source_span
            adjustments.append(entry)

        if criterion["value"] != "unclear" and criterion["source_attribution"] != "own_study":
            before = _snapshot(criterion)
            criterion["value"] = "unclear"
            audit("downgraded_to_unclear", "target_report_attribution_not_established", before)

        passage_id, quote = criterion["passage_id"], criterion["quote"]
        supported = bool(quote.strip()) and passage_id in passages and quote in passages[passage_id]
        if not supported and quote.strip():
            if passage_id not in passages:
                match = _unique_exact(quote, passages)
                if match is not None:
                    corrected_id, start, end = match
                    before = _snapshot(criterion)
                    criterion["passage_id"] = corrected_id
                    supported = True
                    audit("passage_id_repaired", "unique_exact_quote_in_actual_passage", before,
                          {"passage_id": corrected_id, "start": start, "end": end,
                           "text": passages[corrected_id][start:end]})
            else:
                span, repair_reason = _known_passage_span(passages[passage_id], quote)
                if span is not None:
                    start, end = span
                    original_span = passages[passage_id][start:end]
                    before = _snapshot(criterion)
                    criterion["quote"] = original_span
                    supported = True
                    audit("quote_anchor_repaired", repair_reason, before,
                          {"passage_id": passage_id, "start": start, "end": end,
                           "text": original_span})

        if not supported:
            before = _snapshot(criterion)
            criterion.update(value="unclear", passage_id="", quote="")
            if before != _snapshot(criterion):
                audit("downgraded_to_unclear", "unsupported_or_ambiguous_source_quote", before)
            continue

        if (stage == "full_text" and criterion["criterion"] in _BODY_REQUIRED
                and not criterion["passage_id"].startswith("BODY_")):
            before = _snapshot(criterion)
            criterion.update(value="unclear", passage_id="", quote="")
            audit("downgraded_to_unclear", "full_text_body_evidence_required", before)

    # Do not discard an otherwise usable packet because one observation lacks
    # source support. Omit unsupported facts/mappings, and leave unsupported
    # appraisal judgements unclear. Structural errors still reach the validator.
    for group in ("extraction", "appraisal", "iso_mapping"):
        items = normalized.get(group)
        if not isinstance(items, list):
            continue
        retained = []
        for index, item in enumerate(items):
            valid = isinstance(item, dict) and all(isinstance(item.get(key), str) for key in ("passage_id", "quote"))
            if valid and group == "extraction":
                valid = item.get("entity_type") in _ENTITIES and isinstance(item.get("entity_label"), str) and isinstance(item.get("fields"), dict) and all(isinstance(v, str) for v in item["fields"].values())
            elif valid and group == "appraisal":
                valid = item.get("domain") in _APPRAISAL_DOMAINS and item.get("rating") in {"yes", "partly", "no", "not_applicable", "unclear"} and isinstance(item.get("justification"), str)
            elif valid:
                valid = item.get("relationship") in _RELATIONSHIPS and all(isinstance(item.get(key), str) for key in ("construct", "characteristic", "justification"))
            if not valid:
                retained.append(item)
                continue
            passage_id, quote = item.get("passage_id"), item.get("quote")
            supported = bool(quote.strip()) and quote in passages.get(passage_id, "")
            if not supported and quote.strip() and passage_id in passages:
                span, reason = _known_passage_span(passages[passage_id], quote)
                if span is not None:
                    start, end = span
                    before = {"passage_id": passage_id, "quote": quote}
                    item["quote"] = passages[passage_id][start:end]
                    supported = True
                    adjustments.append({"criterion": f"{group}[{index}]", "citation_path": [group, index],
                                        "action": "quote_anchor_repaired", "reason": reason,
                                        "original": before, "normalized": {"passage_id": passage_id, "quote": item["quote"]},
                                        "source_span": {"passage_id": passage_id, "start": start, "end": end, "text": item["quote"]}})
            needs_citation = group != "appraisal" or item["rating"] != "unclear" or bool(quote.strip())
            non_target = group == "extraction" and item["fields"].get("source_attribution") != "own_study"
            if (needs_citation and not supported) or non_target:
                before = deepcopy(item)
                reason = "target_report_attribution_not_established" if non_target else "unsupported_or_ambiguous_source_quote"
                action = {"extraction": "extraction_observation_omitted", "iso_mapping": "iso_proposal_omitted",
                          "appraisal": "appraisal_downgraded_to_unclear"}[group]
                if group == "appraisal":
                    item.update(rating="unclear", passage_id="", quote="",
                                justification="Source support was not verified; the original model judgement remains in the audit.")
                    retained.append(item)
                adjustments.append({"criterion": f"{group}[{index}]", "citation_path": [group, index],
                                    "action": action, "reason": reason, "original": before,
                                    "normalized": deepcopy(item) if group == "appraisal" else None})
            else:
                retained.append(item)
        normalized[group] = retained
    return normalized, adjustments
