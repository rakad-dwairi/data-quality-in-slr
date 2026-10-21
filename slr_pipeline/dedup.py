"""Bibliographic deduplication with a reversible observation-to-report map.

Only identifier or corroborated exact-metadata matches can remove an
observation from the screening queue. Similar titles, conflicting identifiers,
and related publication versions remain separate reports for review.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, utcnow, write_csv, write_json
from .dedup_library import Deduplicator, fuzz, normalize_doi, normalized, pieces


BIBLIOGRAPHIC_FIELDS = [
    "record_id", "report_id", "study_id", "study_link_status", "title", "abstract",
    "authors", "date", "date_basis", "doi", "related_doi", "venue",
    "publication_type", "url", "oa_pdf_url", "source", "source_id", "sources",
    "query_id", "query_ids", "raw_ref", "provenance", "observation_id",
    "observation_ids_json", "source_identifiers_json", "dedup_group_size",
    "dedup_possible_match_count", "field_variants_json",
]


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _prepare(rows: list[dict]) -> list[dict]:
    prepared = []
    occurrences = Counter()
    observation_ids = set()
    for original in rows:
        row = {str(key): _text(value) for key, value in original.items()}
        # Spreadsheet-protection quotes are not stripped here: a genuine title
        # can begin with an apostrophe. Original metadata always survives.
        for field in ("title", "abstract", "authors", "date", "doi", "related_doi",
                      "venue", "publication_type", "url", "oa_pdf_url", "source",
                      "source_id", "query_id", "raw_ref"):
            row.setdefault(field, "")
        fingerprint = digest(row)
        occurrences[fingerprint] += 1
        row.setdefault("observation_id", "O" + digest([
            fingerprint, occurrences[fingerprint]
        ])[:24])
        if not row["observation_id"]:
            row["observation_id"] = "O" + digest([
                fingerprint, occurrences[fingerprint]
            ])[:24]
        if row["observation_id"] in observation_ids:
            raise ValueError("Repeated observation_id in retrieval input: " + row["observation_id"])
        observation_ids.add(row["observation_id"])
        prepared.append(row)
    return prepared


def _anchor(engine: Deduplicator, members: set[int]):
    dois = sorted({engine.meta[i]["doi"] for i in members if engine.meta[i]["doi"]})
    if len(dois) == 1:
        return ["doi", dois[0]]
    provider_keys = sorted(set().union(*(engine.meta[i]["provider_keys"] for i in members)))
    if provider_keys:
        return ["provider", provider_keys[0]]
    metadata = sorted({tuple(engine.meta[i][field] for field in
                             ("title", "year", "author", "venue")) for i in members})
    if len(metadata) == 1 and all(metadata[0]):
        return ["metadata", metadata[0]]
    return ["observations", sorted(engine.rows[i]["observation_id"] for i in members)]


def deduplicate(rows: list[dict], run_dir: Path, config: dict) -> list[dict]:
    """Write stage-2 audit files and return independent report representatives.

    IDs are deterministic for an unchanged cluster. A unique DOI provides the
    preferred stable anchor. Ambiguous clusters sharing an anchor receive a
    content-derived suffix instead of accidentally sharing a screening ID.
    """
    settings = config.get("deduplication", {})
    similarity = float(settings.get("similarity", 95.0))
    block_limit = int(settings.get("block_limit", 200))
    if not 85 <= similarity <= 100 or block_limit < 2:
        raise ValueError("Deduplication similarity must be 85–100 and block_limit >= 2")
    output = Path(run_dir) / "02_deduplication"
    output.mkdir(parents=True, exist_ok=True)
    prepared = _prepare(rows)
    engine = Deduplicator(prepared, block_limit=block_limit)
    engine.exact_passes()
    engine.related_reports()
    fuzzy_summary = {"fuzzy_pair_comparisons": 0, "rows_without_eligible_fuzzy_block": 0}
    if not settings.get("skip_fuzzy", False):
        fuzzy_summary = engine.fuzzy_pass(similarity)

    anchors = {root: _anchor(engine, members) for root, members in engine.members.items()}
    anchor_counts = Counter(digest(anchor) for anchor in anchors.values())
    ids = {}
    for root, anchor in anchors.items():
        if anchor_counts[digest(anchor)] > 1:
            anchor = [anchor, sorted(prepared[i]["observation_id"]
                                     for i in engine.members[root])]
        ids[root] = "R" + digest(anchor)[:24]
    if len(ids.values()) != len(set(ids.values())):
        raise ValueError("Observation identifiers collide; cannot assign distinct report IDs")

    representatives, mapping, removed, groups = [], [], [], []
    by_root = {}
    union_fields = {"sources", "query_ids", "date_values", "date_statuses", "title_variants"}
    for root, members in sorted(engine.members.items(), key=lambda item: ids[item[0]]):
        ordered = sorted(members, key=lambda i: (
            -bool(engine.meta[i]["doi"]), -len(prepared[i]["abstract"]),
            -sum(bool(value.strip()) for value in prepared[i].values()),
            prepared[i]["observation_id"], digest(prepared[i]),
        ))
        chosen = ordered[0]
        combined = dict(prepared[chosen])
        fields = sorted(set().union(*(set(prepared[i]) for i in members)))
        variants = {}
        for field in fields:
            values = list(dict.fromkeys(prepared[i].get(field, "") for i in ordered
                                        if prepared[i].get(field, "")))
            if len(values) > 1:
                variants[field] = values
            if field in union_fields:
                combined[field] = "; ".join(sorted(set().union(*(pieces(v) for v in values))))
            elif values:
                combined[field] = values[0]
        combined["doi"] = engine.meta[chosen]["doi"] or combined.get("doi", "")
        combined["sources"] = "; ".join(sorted(set().union(*(
            pieces(prepared[i].get("sources", "")) | pieces(prepared[i]["source"])
            for i in members
        ))))
        combined["query_ids"] = "; ".join(sorted(set().union(*(
            pieces(prepared[i].get("query_ids", "")) | pieces(prepared[i]["query_id"])
            for i in members
        ))))
        provenance = sorted({prepared[i]["raw_ref"] for i in members if prepared[i]["raw_ref"]})
        combined.update(
            record_id=ids[root], report_id=ids[root], study_id="",
            study_link_status="not_reviewed",
            observation_ids_json=json.dumps(sorted(prepared[i]["observation_id"] for i in members)),
            source_identifiers_json=json.dumps(sorted({
                (prepared[i]["source"], prepared[i]["source_id"]) for i in members
            }), ensure_ascii=False),
            provenance=json.dumps(provenance, ensure_ascii=False),
            dedup_group_size=len(members), dedup_possible_match_count=0,
            field_variants_json=json.dumps(variants, ensure_ascii=False, sort_keys=True),
        )
        representatives.append(combined)
        by_root[root] = combined
        for i in ordered:
            mapping.append({
                "observation_id": prepared[i]["observation_id"], "record_id": ids[root],
                "report_id": ids[root], "representative_observation_id": prepared[chosen]["observation_id"],
                "source": prepared[i]["source"], "source_id": prepared[i]["source_id"],
                "query_id": prepared[i]["query_id"], "raw_ref": prepared[i]["raw_ref"],
                "role": "representative" if i == chosen else "duplicate_merged",
            })
            if i != chosen:
                removed.append({**prepared[i], "record_id": ids[root],
                                "representative_observation_id": prepared[chosen]["observation_id"]})
        groups.append({"record_id": ids[root], "representative_observation_id": prepared[chosen]["observation_id"],
                       "field_variants": variants,
                       "observations": [prepared[i] for i in ordered]})

    review = defaultdict(set)
    for (left, right), reasons in engine.review.items():
        a, b = engine.root(left), engine.root(right)
        if a != b:
            review[tuple(sorted((a, b), key=lambda i: ids[i]))].update(reasons)
    possible, version_links = [], []
    degree = Counter()
    for (left, right), reasons in sorted(review.items(), key=lambda item: tuple(ids[i] for i in item[0])):
        a, b = by_root[left], by_root[right]
        row = {
            "left_record_id": ids[left], "right_record_id": ids[right],
            "left_title": a["title"], "right_title": b["title"],
            "left_doi": a["doi"], "right_doi": b["doi"],
            "left_date": a["date"], "right_date": b["date"],
            "left_publication_type": a["publication_type"],
            "right_publication_type": b["publication_type"],
            "title_similarity": round(fuzz.ratio(normalized(a["title"]), normalized(b["title"])), 2),
            "reasons": "; ".join(sorted(reasons)),
            "review_decision": "", "reviewer": "", "reviewer_notes": "",
        }
        possible.append(row)
        degree[ids[left]] += 1
        degree[ids[right]] += 1
        explicit_relation = "related_doi_link_not_an_automatic_duplicate" in reasons
        left_kinds = {engine.meta[i]["kind"] for i in engine.members[left]} - {""}
        right_kinds = {engine.meta[i]["kind"] for i in engine.members[right]} - {""}
        different_kinds = bool(left_kinds and right_kinds and left_kinds != right_kinds)
        exact_title = normalized(a["title"]) == normalized(b["title"]) and bool(a["title"])
        if explicit_relation or (different_kinds and exact_title):
            version_links.append({
                **row, "link_basis": "provider_related_doi" if explicit_relation else "same_title_different_report_kind",
                "link_status": "possible_related_reports_pending_review", "study_id": "",
                "primary_report_id": "", "relationship": "",
            })
    for row in representatives:
        row["dedup_possible_match_count"] = degree[row["record_id"]]

    all_fields = list(dict.fromkeys(BIBLIOGRAPHIC_FIELDS + sorted(set().union(*(set(row) for row in representatives))
                                                              if representatives else set())))
    write_csv(output / "records.csv", representatives, all_fields)
    removed_fields = list(dict.fromkeys(["record_id", "representative_observation_id"] + sorted(
        set().union(*(set(row) for row in prepared)) if prepared else set())))
    write_csv(output / "duplicates_removed.csv", removed, removed_fields)
    write_csv(output / "record_mapping.csv", mapping, [
        "observation_id", "record_id", "report_id", "representative_observation_id", "source", "source_id",
        "query_id", "raw_ref", "role",
    ])
    pair_fields = ["left_record_id", "right_record_id", "left_title", "right_title", "left_doi", "right_doi",
                   "left_date", "right_date", "left_publication_type", "right_publication_type", "title_similarity",
                   "reasons", "review_decision", "reviewer", "reviewer_notes"]
    write_csv(output / "possible_duplicates.csv", possible, pair_fields)
    write_csv(output / "version_links.csv", version_links, pair_fields + [
        "link_basis", "link_status", "study_id", "primary_report_id", "relationship",
    ])
    merge_log = [{
        "left_observation_id": prepared[event["left_index"]]["observation_id"],
        "right_observation_id": prepared[event["right_index"]]["observation_id"],
        "rule": event["rule"], "record_id": ids[engine.root(event["left_index"])],
    } for event in engine.events]
    write_csv(output / "merge_log.csv", merge_log, [
        "left_observation_id", "right_observation_id", "rule", "record_id",
    ])
    write_json(output / "group_details.json", groups)
    write_json(output / "large_blocks.json", [{
        "rule": block["rule"], "key": block["key"],
        "observation_ids": [prepared[i]["observation_id"] for i in block["record_indices"]],
        "status": "not_auto_merged_manual_review_required",
    } for block in engine.large_blocks])
    if len(prepared) != len(representatives) + len(removed) or len(mapping) != len(prepared):
        raise RuntimeError("Deduplication accounting failed; outputs cannot be used")
    write_json(output / "summary.json", {
        "completed_utc": utcnow(), "input_observations": len(prepared),
        "duplicates_removed": len(removed), "reports_remaining": len(representatives),
        "possible_duplicate_pairs_retained": len(possible), "reports_in_possible_pairs": len(degree),
        "possible_version_links": len(version_links), "large_match_blocks_not_auto_merged": len(engine.large_blocks),
        "merge_rules": dict(Counter(event["rule"] for event in engine.events)),
        "fuzzy_matches_auto_merged": 0, "verified_study_count": None,
        "similarity": similarity, "block_limit": block_limit,
        "fuzzy_check_skipped": bool(settings.get("skip_fuzzy", False)), **fuzzy_summary,
        "notes": [
            "Counts describe bibliographic reports, not verified unique studies.",
            "Conflicting DOI/type/version/notice clusters remain in the screening queue.",
            "Related reports and fuzzy matches require review; neither is silently removed.",
        ],
    })
    return representatives
