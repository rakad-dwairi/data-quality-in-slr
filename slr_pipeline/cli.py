"""Orchestration only; stages are called explicitly through main()."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from . import __version__
from .common import (digest, file_hash, project_path, read_csv, read_json, run_lock,
                     utcnow, write_csv, write_json)
from .reviews import (REVIEW_FIELDS, agreement_summary, load_reviews,
                      resolve_reviews, valid_timestamp, write_worksheets)

PROJECT = Path(__file__).resolve().parents[1]
BIB = ["record_id", "report_id", "study_id", "title", "abstract", "authors", "date", "date_basis", "date_status", "date_window_status", "doi", "related_doi",
       "date_verification_evidence", "date_verified_by", "date_verified_utc",
       "venue", "publication_type", "source", "sources", "url", "oa_pdf_url", "provenance"]
SCREEN = BIB + ["ai_decision", "ai_reason_code", "ai_rationale", "ai_study_kind",
                "ai_model_decision", "ai_model_rationale", "screening_status", "input_sha256",
                "criteria_json", "evidence_json", "model", "model_digest", "prompt_version", "prompt_sha256",
                "cache_key", "assessment_fingerprint", "attempt_id", "assessed_utc",
                "evidence_validation_version", "evidence_adjustments_json", "evidence_downgraded_criteria",
                "evidence_review_required", "evidence_issue_summary",
                "human_decision", "human_reason_code", "human_status", "human_reviewers", "effective_decision"]
FULL = SCREEN + ["full_text_path", "full_text_sha256", "full_text_file_sha256", "full_text_status", "full_text_error", "text_complete",
                 "chunk_count", "chunks_completed", "chunk_errors_json", "chunk_fingerprints_json",
                 "extraction_json", "appraisal_json", "iso_mapping_json"]


def load_env(path):
    """Literal KEY=value parser; never executes shell syntax or prints values."""
    from .checks import env_values
    for key, value in env_values(path).items():
        os.environ.setdefault(key, value)


def load_config(path):
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    config["_project_dir"] = str(PROJECT)
    config["_config_path"] = str(path.resolve())
    for name in ("search", "deduplication", "model", "screening", "fulltext", "protocol"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing configuration object: {name}")
    if not isinstance(config.get("imports", []), list):
        raise ValueError("imports must be a list of source/path objects.")
    if config["protocol"].get("allowed_languages") and not config["protocol"].get("language_restriction_justification"):
        raise ValueError("A language restriction needs a protocol justification.")
    if int(config["screening"].get("limit", 0)) < 0:
        raise ValueError("screening.limit must be >= 0.")
    if not config.get("run_directory"):
        raise ValueError("Set run_directory to a dedicated output directory.")
    return config


def sanitized_config(config):
    result = {k: v for k, v in config.items() if not k.startswith("_")}
    # Credentials have no supported config slots. Reject accidental additions instead of recording them.
    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if re.search(r"(?:api[_-]?key|password|secret|access[_-]?token|authorization)", key, re.I):
                    raise ValueError("Do not put credentials in JSON. Use environment variables or .env.")
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
    inspect(result)
    return result


def initialize(config, out):
    from .checks import configuration_fingerprint
    sanitized_config(config)
    snapshot, source_hashes, fingerprint = configuration_fingerprint(config)
    manifest_path = out / "manifest.json"
    old = read_json(manifest_path, {})
    if old and old.get("fingerprint") != fingerprint:
        raise ValueError("This run was created with different code/configuration. Use a new run_directory to retain the old audit trail.")
    if not old:
        write_json(manifest_path, {"version": __version__, "created_utc": utcnow(),
                                  "fingerprint": fingerprint, "source_hashes": source_hashes,
                                  "config": snapshot, "execution_status": "prepared"})
        # Freeze actual implementation text, without credentials or user data.
        from .common import atomic_text
        for source in sorted((PROJECT / "slr_pipeline").glob("*.py")):
            # Preserve Windows CRLF bytes: read_text's universal-newline mode
            # previously made snapshots disagree with their raw source hashes.
            with source.open(encoding="utf-8", newline="") as handle:
                atomic_text(out / "code_snapshot" / source.name, handle.read())
        protocol_path = project_path(config, config["protocol"].get("source_plan", ""))
        write_json(out / "protocol_snapshot.json", {"protocol": config["protocol"],
                   "source_plan_sha256": file_hash(protocol_path) if protocol_path.is_file() else None,
                   "search_start_note": "1900 is a declared broad lower bound, not verified database-inception coverage.",
                   "scientific_status": "Draft until team calibration, search validation, extraction review and ISO coding are completed."})


def require_rows(out, relative):
    path = out / relative
    if not path.exists():
        raise ValueError(f"Missing {relative}. Run its preceding stage first, or use --stage all.")
    stage = "retrieve" if relative.startswith("01_") else "deduplicate"
    if not validate_stage(out, stage):
        raise ValueError(f"{relative} is stale or incomplete. Rerun its stage or --stage all.")
    return read_csv(path)


def begin_stage(out, stage):
    state = read_json(out / "stage_state.json", {})
    order = ["retrieve", "deduplicate", "screen", "fulltext"]
    for name in order[order.index(stage):]:
        state.setdefault(name, {})["valid"] = False
    state[stage]["started_utc"] = utcnow()
    write_json(out / "stage_state.json", state)


def end_stage(out, stage, rows):
    state = read_json(out / "stage_state.json", {})
    artifact = {"retrieve": "01_retrieval/records.csv", "deduplicate": "02_deduplication/records.csv",
                "screen": "03_title_abstract/assessments.json", "fulltext": "04_full_text/assessments.json"}[stage]
    state[stage] = {"valid": True, "rows": len(rows), "digest": digest(without_body(rows)),
                    "artifact": artifact, "artifact_sha256": file_hash(out / artifact), "ended_utc": utcnow()}
    write_json(out / "stage_state.json", state)


def validate_stage(out, stage):
    entry = read_json(out / "stage_state.json", {}).get(stage, {})
    if not entry.get("valid"):
        return False
    artifact = out / entry.get("artifact", "missing")
    if not artifact.is_file() or file_hash(artifact) != entry.get("artifact_sha256"):
        raise ValueError(f"Generated {stage} artifact changed or is missing. Restore it or rerun that stage; put review decisions in inputs/ instead.")
    return True


def without_body(rows):
    return [{k: v for k, v in row.items() if k != "full_text"} for row in rows]


def snapshot_stage(out, rows, stage):
    directory = out / ("03_title_abstract" if stage == "title_abstract" else "04_full_text")
    write_json(directory / "assessments.json", without_body(rows))


def reviewed_stages(config, out, ta=None, ft=None):
    reviews = load_reviews(project_path(config, config.get("reviews_file", "inputs/reviews.csv")))
    if ta is None:
        ta = read_json(out / "03_title_abstract/assessments.json", []) if validate_stage(out, "screen") else []
    if ft is None:
        ft = read_json(out / "04_full_text/assessments.json", []) if validate_stage(out, "fulltext") else []
    ta, audit_ta = resolve_reviews(ta, "title_abstract", reviews)
    active = {r["record_id"] for r in ta if advances_to_fulltext(r)}
    # Prior FT results for records subsequently excluded in TA are retained in snapshots,
    # but never enter the current final export.
    ft, audit_ft = resolve_reviews([r for r in ft if r["record_id"] in active], "full_text", reviews)
    for row in ft:
        if row.get("full_text_status") == "ready":
            source = Path(row.get("full_text_path", ""))
            expected = row.get("full_text_file_sha256")
            try:
                intact = bool(expected) and source.is_file() and file_hash(source) == expected
            except OSError:
                intact = False
            if not intact:
                row.update(full_text_status="changed_or_missing_since_screening", text_complete=False,
                           full_text_error="Original full-text file changed or is missing; rerun --stage fulltext before inclusion.")
    known = {"title_abstract": {r["record_id"] for r in ta}, "full_text": {r["record_id"] for r in ft}}
    unmatched = []
    for review in reviews:
        stage = review.get("stage")
        if stage not in known or review.get("record_id") not in known.get(stage, set()):
            unmatched.append({**review, "import_status": "unknown_or_inactive_stage_record"})
        elif not review.get("decision", "").strip():
            unmatched.append({**review, "import_status": "blank_decision_pending"})
    return ta, ft, reviews, audit_ta + audit_ft + unmatched


def advances_to_fulltext(row):
    return row.get("effective_decision") in {"include", "uncertain"} and (
        bool(row.get("human_decision")) or row.get("screening_status") not in {"pending_limit", "model_unavailable"})


def require_available_model(rows, stage):
    """An exported pending queue is not a completed model screening run."""
    failed = 0
    for row in rows:
        unavailable = row.get("screening_status") == "model_unavailable"
        if not unavailable and row.get("chunk_errors_json"):
            try:
                unavailable = any(error.get("status") == "model_unavailable"
                                  for error in json.loads(row["chunk_errors_json"]))
            except (ValueError, TypeError, AttributeError):
                pass
        failed += int(unavailable)
    if failed:
        raise ValueError(f"{stage}: model unavailable for {failed} records. Pending CSVs and completed checkpoints were saved. "
                         "Fix Ollama, then resume this screening stage; downstream stages were not started.")


def date_window_status(row, config):
    """Keep uncertain publication dates visible; do not fabricate a month/day."""
    import calendar
    values = {row.get("date", "")}
    try:
        values.update(json.loads(row.get("field_variants_json") or "{}").get("date", []))
    except (ValueError, TypeError):
        return "check_conflicting_or_invalid_dates"
    start, end = date.fromisoformat(config["search"]["start"]), date.fromisoformat(config["search"]["end"])
    statuses = set()
    for value in values:
        try:
            if not re.fullmatch(r"\d{4}(?:-\d{2}){0,2}", value):
                raise ValueError
            parts = [int(p) for p in value.split("-")]
            if len(parts) == 1:
                low, high = date(parts[0], 1, 1), date(parts[0], 12, 31)
            elif len(parts) == 2:
                low, high = date(*parts, 1), date(*parts, calendar.monthrange(*parts)[1])
            else:
                low = high = date(*parts)
            statuses.add("outside_window" if high < start or low > end else
                         "within_window" if start <= low and high <= end else "check_boundary_date")
        except (ValueError, TypeError):
            statuses.add("check_missing_or_invalid_date")
    return next(iter(statuses)) if len(statuses) == 1 else "check_conflicting_dates"


def truth(value):
    return value is True or str(value).lower() == "true"


def extraction_tables(out, candidates):
    folder = out / "05_extraction"
    fields_by_kind = {
        "study": ["objective", "research_questions", "paradigm", "task", "domain", "evaluation_setting", "authors_limitations", "reviewer_concerns"],
        "dataset": ["name", "version", "source", "collection_period", "target_population", "sampling", "exposure_mechanism", "counts", "missingness", "preprocessing"],
        "data_object": ["type", "feedback_type", "interaction_type", "unit", "lifecycle_stage", "iso_target_data_role", "context_of_use"],
        "measure": ["author_term", "operational_definition", "formula_or_procedure", "numerator", "denominator", "reference_requirement", "measurement_level", "unit", "aggregation", "parameters", "threshold", "threshold_justification", "direction", "uncertainty", "implementation"],
        "validation": ["validation_type", "dataset", "intervention", "dose", "comparator", "statistical_method", "estimate", "uncertainty", "replication"],
        "outcome": ["metric", "estimate", "uncertainty", "algorithm", "baselines", "split", "candidates", "negative_sampling", "leakage_controls", "tuning", "repeated_runs", "effect_direction", "heterogeneity"]}
    tables = {kind: [] for kind in fields_by_kind}
    appraisal, mappings = [], []
    base_fields = ["entity_id", "record_id", "report_id", "study_id", "entity_label", "passage_id", "quote", "fields_json", "provenance", "human_status"]
    for paper in candidates:
        base = {"record_id": paper["record_id"], "report_id": paper.get("report_id", paper["record_id"]),
                "study_id": paper.get("study_id", ""), "provenance": "ai_proposed", "human_status": "pending"}
        for index, entity in enumerate(json.loads(paper.get("extraction_json") or "[]")):
            if not isinstance(entity, dict) or entity.get("entity_type") not in tables:
                continue
            kind = entity["entity_type"]
            properties = entity.get("fields", {})
            if not isinstance(properties, dict):
                continue
            tables[kind].append({**base, **{k: properties.get(k, "not_reported") for k in fields_by_kind[kind]},
                "entity_id": f"{paper['record_id']}-{kind}-{index+1:03d}", "entity_label": entity.get("entity_label", ""),
                "passage_id": entity.get("passage_id", ""), "quote": entity.get("quote", ""), "fields_json": properties})
        for assessment in json.loads(paper.get("appraisal_json") or "[]"):
            if isinstance(assessment, dict):
                appraisal.append({**assessment, **base})
        for mapping in json.loads(paper.get("iso_mapping_json") or "[]"):
            if isinstance(mapping, dict):
                mappings.append({**mapping, **base, "mapping_status": "provisional_requires_independent_coding",
                                 "authorised_iso_clause": "", "measure_id": ""})
    filenames = {"study": "studies", "dataset": "datasets", "data_object": "data_objects",
                 "measure": "measures", "validation": "validations", "outcome": "outcomes"}
    for kind, entries in tables.items():
        write_csv(folder / (filenames[kind] + ".csv"), entries, base_fields + fields_by_kind[kind])
    write_csv(folder / "quality_appraisal.csv", appraisal,
              ["record_id", "report_id", "study_id", "domain", "rating", "justification", "passage_id", "quote", "provenance", "human_status"])
    write_csv(folder / "iso_mapping_candidates.csv", mappings,
              ["record_id", "report_id", "study_id", "measure_id", "construct", "characteristic", "relationship", "justification", "passage_id", "quote", "mapping_status", "authorised_iso_clause", "provenance", "human_status"])


def export_results(config, out, ta=None, ft=None):
    ta, ft, reviews, audit = reviewed_stages(config, out, ta, ft)
    for row in ta + ft:
        row["date_window_status"] = date_window_status(row, config)
    checks_path = project_path(config, config.get("date_checks_file", "inputs/date_checks.csv"))
    checks = load_reviews(checks_path)
    check_counts = Counter(r.get("record_id") for r in checks)
    for row in ft:
        matches = [r for r in checks if r.get("record_id") == row["record_id"]]
        if check_counts[row["record_id"]] == 1:
            check = matches[0]
            if (check.get("input_sha256") == row.get("input_sha256") and check.get("reviewer_id", "").strip()
                    and valid_timestamp(check.get("reviewed_utc")) and check.get("evidence", "").strip()
                    and check.get("date_window_status") in {"within_window", "outside_window"}):
                row["date_window_status"] = check["date_window_status"]
                row["date_verification_evidence"] = check["evidence"]
                row["date_verified_by"] = check["reviewer_id"].strip()
                row["date_verified_utc"] = check["reviewed_utc"]
    verified_dates = {r["record_id"]: r for r in ft if r.get("date_verification_evidence")}
    for row in ta:
        if row["record_id"] in verified_dates:
            for key in ("date_window_status", "date_verification_evidence", "date_verified_by", "date_verified_utc"):
                row[key] = verified_dates[row["record_id"]][key]
    ta_dir, ft_dir, final_dir = out / "03_title_abstract", out / "04_full_text", out / "06_final"
    write_csv(ta_dir / "screening.csv", ta, SCREEN)
    write_csv(ta_dir / "excluded_provisional.csv", [r for r in ta if r["effective_decision"] == "exclude"], SCREEN)
    write_csv(ta_dir / "secondary_literature.csv", [r for r in ta if r["effective_decision"] == "secondary"], SCREEN)
    write_csv(ta_dir / "pending_human_review.csv", [r for r in ta if not r.get("human_decision") or r["human_decision"] == "uncertain"], SCREEN)
    advancing = [r for r in ta if advances_to_fulltext(r)]
    write_csv(ta_dir / "full_text_queue.csv", advancing, SCREEN)
    write_csv(ft_dir / "screening.csv", ft, FULL)
    write_csv(ft_dir / "excluded_provisional.csv", [r for r in ft if r["effective_decision"] == "exclude"], FULL)
    write_csv(ft_dir / "secondary_literature.csv", [r for r in ft if r["effective_decision"] == "secondary"], FULL)
    candidates = [r for r in ft if r["effective_decision"] == "include"
                  and r.get("full_text_status") == "ready" and truth(r.get("text_complete"))]
    final = [r for r in candidates if r.get("human_decision") == "include" and r["date_window_status"] == "within_window"]
    final_ids = {r["record_id"] for r in final}
    pending = [r for r in ft if r["record_id"] not in final_ids and
               r.get("human_decision") not in {"exclude", "secondary"}]
    # Records without FT assessments also remain explicitly unresolved.
    assessed_ids = {r["record_id"] for r in ft}
    pending += [{**r, "full_text_status": "not_yet_assessed"} for r in advancing if r["record_id"] not in assessed_ids]
    write_csv(final_dir / "included_candidates.csv", candidates, FULL)
    write_csv(final_dir / "final_included_papers.csv", final, FULL)
    write_csv(final_dir / "pending_review.csv", pending, FULL)
    write_csv(final_dir / "date_checks_needed.csv", [r for r in ta if r["date_window_status"] != "within_window"], SCREEN)
    write_csv(final_dir / "date_checks_template.csv",
              [{"record_id": r["record_id"], "input_sha256": r.get("input_sha256", ""), "title": r.get("title", ""),
                "date": r.get("date", ""), "date_window_status": "", "reviewer_id": "", "reviewed_utc": "", "evidence": ""}
               for r in ft if r["date_window_status"] != "within_window"],
              ["record_id", "input_sha256", "title", "date", "date_window_status", "reviewer_id", "reviewed_utc", "evidence"])
    write_csv(final_dir / "date_verification_inputs.csv", checks)
    write_csv(final_dir / "human_confirmed_exclusions.csv",
              [{**r, "stage": "title_abstract"} for r in ta if r["human_decision"] == "exclude"] +
              [{**r, "stage": "full_text"} for r in ft if r["human_decision"] == "exclude"], ["stage"] + FULL)
    write_csv(final_dir / "review_import_audit.csv", audit, REVIEW_FIELDS + ["import_status"])
    for stage, rows, directory in (("title_abstract", ta, ta_dir), ("full_text", ft, ft_dir)):
        write_worksheets(directory, rows, stage)
        write_json(directory / "human_agreement.json", agreement_summary(rows, reviews, stage))
    extraction_tables(out, candidates)
    state = read_json(out / "stage_state.json", {})
    retrieval = read_json(out / "01_retrieval/summary.json", {}) if state.get("retrieve", {}).get("valid") else {"status": "stale_or_not_run"}
    dedup = read_json(out / "02_deduplication/summary.json", {}) if state.get("deduplicate", {}).get("valid") else {"status": "stale_or_not_run"}
    ta_unresolved = [r for r in ta if not r.get("human_decision") or r["human_decision"] == "uncertain"]
    ft_unresolved = [r for r in ft if not r.get("human_decision") or r["human_decision"] == "uncertain"]
    flow = {
        "generated_utc": utcnow(), "unit": "bibliographic reports, not verified unique studies",
        "stage_state": state,
        "retrieval": retrieval, "deduplication": dedup,
        "title_abstract_records": len(ta), "title_abstract_ai_decisions": dict(Counter(r.get("ai_decision", "pending") for r in ta)),
        "title_abstract_human_unresolved": len(ta_unresolved), "full_text_queue": len(advancing),
        "full_text_assessments": len(ft), "full_text_ready": sum(r.get("full_text_status") == "ready" for r in ft),
        "full_text_not_retrieved_or_incomplete": sum(r.get("full_text_status") != "ready" for r in ft),
        "full_text_human_unresolved": len(ft_unresolved), "ai_included_candidates": len(candidates),
        "human_confirmed_included_reports": len(final),
        "human_confirmed_study_ids": len({r["study_id"] for r in final if r.get("study_id")}),
        "reports_missing_verified_study_id": sum(not r.get("study_id") for r in final),
        "verified_final_study_count": None,
        "human_full_text_exclusion_reasons": dict(Counter(r["human_reason_code"] for r in ft if r["human_decision"] == "exclude")),
        "prisma_closed": False, "publication_ready": False,
        "remaining_gates": ["Team protocol freeze and calibration", "Seed recall, database coverage and backward/forward snowballing verification",
                            "Independent human screening and disagreement resolution", "Report-to-study/version linkage",
                            "Review extracted evidence, appraisal and authorised ISO clause mapping", "Synthesis and reporting review"]}
    write_json(final_dir / "flow_counts.json", flow)
    write_json(out / "summary.json", flow)
    print(f"AI inclusion candidates: {len(candidates)}; human-confirmed reports: {len(final)}; pending full-text/review: {len(pending)}", flush=True)
    print(f"CSV outputs: {out}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RecSys SLR: local retrieval, deduplication, screening and auditable CSV exports.")
    parser.add_argument("--config", type=Path, default=PROJECT / "config/review.json")
    parser.add_argument("--stage", choices=["check", "plan", "retrieve", "deduplicate", "screen", "fulltext", "export", "status", "all"], default="all")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        out = project_path(config, config["run_directory"])
        if args.stage == "status":
            print(json.dumps(read_json(out / "summary.json", {"status": "No completed exports yet."}), indent=2))
            return 0
        sanitized_config(config)
        if args.stage in {"check", "plan", "all", "retrieve"}:
            from .checks import check_configuration
            report = check_configuration(config, stage=args.stage)
            if args.stage == "check":
                print(json.dumps(report, indent=2))
                return 0 if report["ready"] else 2
            if not report["ready"]:
                raise ValueError("Offline readiness check failed before starting any stages:\n- " + "\n- ".join(report["errors"]) + "\nUse --stage check for the full report.")
        load_env(PROJECT / ".env")
        with run_lock(out):
            initialize(config, out)
            if args.stage == "plan":
                print(f"Prepared {out}. No database, full-text download or model requests.")
                return 0
            if args.stage in {"all", "retrieve"}:
                from .retrieval import collect
                begin_stage(out, "retrieve")
                export_results(config, out)
                print("1/5 Retrieving/importing bibliographic observations", flush=True)
                observations = collect(config, out)
                if not observations:
                    raise ValueError("No records retrieved/imported. Check search_log.csv and configure providers/imports.")
                end_stage(out, "retrieve", observations)
                if config["search"].get("require_all_sources", True):
                    logs = read_json(out / "01_retrieval/search_log.json", [])
                    allowed = {"api_query_exhausted", "imported", "duplicate_import_skipped", "ranked_discovery_capped", "ranked_discovery_returned"}
                    failed = [entry for entry in logs if entry.get("status") not in allowed]
                    if failed:
                        export_results(config, out)
                        raise ValueError("Retrieval is incomplete for selected sources/queries. Successful records were saved; inspect 01_retrieval/search_log.csv and resume before screening.")
            if args.stage in {"all", "deduplicate"}:
                from .dedup import deduplicate
                if args.stage != "all":
                    observations = require_rows(out, "01_retrieval/records.csv")
                begin_stage(out, "deduplicate")
                export_results(config, out)
                print("2/5 Deduplicating bibliographic reports", flush=True)
                records = deduplicate(observations, out, config)
                end_stage(out, "deduplicate", records)
            if args.stage in {"all", "screen"}:
                from .screening import screen_records
                if args.stage != "all":
                    records = require_rows(out, "02_deduplication/records.csv")
                begin_stage(out, "screen")
                export_results(config, out)
                print("3/5 Screening titles and abstracts with the local model", flush=True)
                ta = screen_records(records, "title_abstract", config, out)
                snapshot_stage(out, ta, "title_abstract")
                end_stage(out, "screen", ta)
                export_results(config, out, ta=ta)
                require_available_model(ta, "Title/abstract screening")
                if args.stage == "all" and any(row.get("screening_status") == "pending_limit" for row in ta):
                    raise ValueError("Title/abstract screening reached its configured limit. Saved results can be resumed; "
                                     "full-text preparation was not started while title/abstract work remains pending.")
            if args.stage in {"all", "fulltext"}:
                from .fulltext import prepare_fulltexts
                from .screening import screen_records
                ta, _, _, _ = reviewed_stages(config, out)
                if not read_json(out / "stage_state.json", {}).get("screen", {}).get("valid"):
                    raise ValueError("Run --stage screen first.")
                begin_stage(out, "fulltext")
                export_results(config, out)
                queue = [r for r in ta if advances_to_fulltext(r)]
                print(f"4/5 Preparing and screening full text for {len(queue)} reports", flush=True)
                prepared = prepare_fulltexts(queue, config, out)
                ft = screen_records(prepared, "full_text", config, out)
                snapshot_stage(out, ft, "full_text")
                end_stage(out, "fulltext", ft)
                export_results(config, out, ft=ft)
                require_available_model(ft, "Full-text screening")
            if args.stage in {"export", "deduplicate", "retrieve"}:
                export_results(config, out)
            manifest = read_json(out / "manifest.json")
            manifest.update(last_stage=args.stage, last_updated_utc=utcnow(), execution_status="stage_returned")
            write_json(out / "manifest.json", manifest)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Completed retrieval pages and model calls are checkpointed; repeat the same command to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Do not expose HTTP credential URLs or raw provider error bodies.
        if isinstance(exc, (ValueError, FileNotFoundError, ImportError)) or type(exc).__name__ == "ModelUnavailable":
            message = str(exc)
            for name in ("OPENALEX_API_KEY", "SCOPUS_API_KEY", "SCOPUS_INST_TOKEN", "WOS_API_KEY", "IEEE_API_KEY"):
                value = os.getenv(name)
                if value:
                    message = message.replace(value, "[REDACTED]")
        else:
            message = "Unexpected failure (" + type(exc).__name__ + "). Check private stage logs and resume after correction."
        print("Stopped: " + message, file=sys.stderr)
        return 1
