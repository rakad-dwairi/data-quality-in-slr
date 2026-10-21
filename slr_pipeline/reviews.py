"""Import independent human decisions without confusing AI with human review."""
from collections import Counter, defaultdict
from datetime import datetime

from .common import read_csv, write_csv, write_json

REVIEW_FIELDS = ["stage", "record_id", "input_sha256", "reviewer_id", "role",
                 "decision", "reason_code", "notes", "reviewed_utc", "study_id"]
DECISIONS = {"include", "exclude", "uncertain", "secondary"}
REASONS = {"NOT_RECSYS", "NO_DATA_QUALITY_CONSTRUCT", "NO_OPERATIONAL_ASSESSMENT",
           "WRONG_STUDY_TYPE", "DUPLICATE_COMPANION_REPORT", "INSUFFICIENT_FULL_TEXT",
           "LANGUAGE_RESTRICTION"}


def load_reviews(path):
    return read_csv(path) if path.exists() else []


def resolve_reviews(rows, stage, reviews):
    """Two named independent reviewers agree, or a third resolves a disagreement."""
    index = defaultdict(list)
    for original_review in reviews:
        review = dict(original_review)
        review["reviewer_id"] = review.get("reviewer_id", "").strip().casefold()
        if review.get("stage") == stage and review.get("decision", "").strip():
            index[review.get("record_id", "")].append(review)
    resolved, audit = [], []
    for original in rows:
        row = dict(original)
        valid = []
        for review in index.get(row["record_id"], []):
            problem = ""
            if not review.get("input_sha256") or review["input_sha256"] != row.get("input_sha256"):
                problem = "stale_or_missing_text_hash"
            elif not review.get("reviewer_id", "").strip() or not review.get("reviewed_utc", "").strip():
                problem = "missing_reviewer_or_date"
            elif not valid_timestamp(review["reviewed_utc"]):
                problem = "reviewed_utc_requires_iso_datetime_with_timezone"
            elif review.get("role") not in {"reviewer", "adjudicator"}:
                problem = "role_must_be_reviewer_or_adjudicator"
            elif review.get("decision") not in DECISIONS:
                problem = "invalid_decision"
            elif review["decision"] == "exclude" and review.get("reason_code") not in REASONS:
                problem = "principal_exclusion_reason_required"
            elif review["decision"] in {"exclude", "secondary"} and not review.get("notes", "").strip():
                problem = "decision_justification_required"
            audit.append({**review, "import_status": problem or "valid"})
            if not problem:
                valid.append(review)
        identities = Counter(r["reviewer_id"] for r in valid)
        # Duplicate decisions from the same person are ambiguous, never silently last-write-wins.
        regular = [r for r in valid if r["role"] == "reviewer" and identities[r["reviewer_id"]] == 1]
        row.update(human_decision="", human_reason_code="", human_status="pending",
                   human_reviewers="", study_id="")
        agreement = len(regular) >= 2 and len({r["decision"] for r in regular}) == 1
        if agreement and regular[0]["decision"] == "exclude":
            agreement = len({r.get("reason_code") for r in regular}) == 1
        accepted = regular[0] if agreement else None
        if len(regular) >= 2 and not agreement:
            row["human_status"] = "disagreement"
            adjudicators = [r for r in valid if r["role"] == "adjudicator"
                            and identities[r["reviewer_id"]] == 1 and r.get("notes", "").strip()
                            and r["reviewer_id"] not in {x["reviewer_id"] for x in regular}]
            if len(adjudicators) == 1:
                accepted = adjudicators[0]
        if accepted:
            row.update(human_decision=accepted["decision"], human_reason_code=accepted.get("reason_code", ""),
                       human_status="agreed" if agreement else "adjudicated",
                       human_reviewers="; ".join(r["reviewer_id"] for r in valid))
            studies = {r.get("study_id", "").strip() for r in regular if r.get("study_id", "").strip()}
            if accepted["role"] == "adjudicator" and accepted.get("study_id", "").strip():
                row["study_id"] = accepted["study_id"].strip()
            elif len(studies) == 1:
                row["study_id"] = next(iter(studies))
        row["effective_decision"] = row["human_decision"] or row.get("ai_decision", "uncertain")
        resolved.append(row)
    return resolved, audit


def write_worksheets(directory, rows, stage):
    """These templates are regenerated; completed reviews belong in inputs/reviews.csv."""
    fields = REVIEW_FIELDS + ["title", "abstract", "doi", "full_text_path"]
    for reviewer in ("reviewer_1", "reviewer_2"):
        sheet = [{**{k: "" for k in REVIEW_FIELDS}, "stage": stage,
                  "record_id": row["record_id"], "input_sha256": row.get("input_sha256", ""),
                  "role": "reviewer", "title": row.get("title", ""), "abstract": row.get("abstract", ""), "doi": row.get("doi", ""),
                  "full_text_path": row.get("full_text_path", "")} for row in rows]
        # Blind templates: no AI label and no other reviewer's response.
        write_csv(directory / f"{reviewer}_template.csv", sheet, fields)


def agreement_summary(rows, reviews, stage):
    """Agreement among exactly two named reviewers with current input hashes only."""
    current = {r["record_id"]: r.get("input_sha256") for r in rows}
    _, validated = resolve_reviews(rows, stage, reviews)
    grouped = defaultdict(list)
    for r in validated:
        if (r.get("stage") == stage and r.get("role") == "reviewer"
                and r.get("import_status") == "valid"
                and r.get("decision") in DECISIONS and r.get("reviewer_id")
                and r.get("reviewed_utc") and r.get("input_sha256")
                and r.get("input_sha256") == current.get(r.get("record_id"))):
            grouped[r["record_id"]].append(r)
    pairs = []
    reviewer_pair = None
    for record_id in sorted(grouped):
        group = grouped[record_id]
        if len(group) != 2 or len({r["reviewer_id"] for r in group}) != 2:
            continue
        group.sort(key=lambda r: r["reviewer_id"])
        identities = tuple(r["reviewer_id"] for r in group)
        if reviewer_pair is None:
            reviewer_pair = identities
        if identities == reviewer_pair:
            pairs.append((group[0]["decision"], group[1]["decision"]))
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n if n else None
    left, right = Counter(a for a, b in pairs), Counter(b for a, b in pairs)
    expected = sum(left[k] * right[k] for k in DECISIONS) / (n * n) if n else None
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else None
    return {"stage": stage, "reviewers": reviewer_pair, "paired_records": n,
            "raw_agreement": observed, "cohens_kappa": kappa,
            "note": "Only independent human labels count; undefined kappa is null. No universal acceptance threshold."}


def valid_timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.tzinfo is not None
    except (TypeError, ValueError):
        return False
