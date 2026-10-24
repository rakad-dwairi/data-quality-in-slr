#!/usr/bin/env python3
"""Bounded local candidate-model checks, using saved source material only."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time

from complete_local_review import SEED_TITLES, pilot_checks, seed_title_candidates
from local_source_citations import SourceCitationClient, VERSION as CITATION_VERSION
from slr_pipeline import screening
from slr_pipeline.common import digest, file_hash, read_csv, read_json, utcnow, write_csv, write_json

BREVITY = """\nPresentation only: give each criterion explanation in one short sentence
of at most 20 words, and the rationale in at most 45 words. Apply every original
criterion without changing its meaning. Preserve uncertainty when evidence is
missing. Source citation IDs supply exact quotations; do not repeat quotations
in the explanation. No paper-count target determines a decision.\n"""


def configured(config, model, endpoint, context=32768):
    value = copy.deepcopy(config)
    value["model"] = {**value["model"], "name": model, "base_url": endpoint,
                      "allow_remote": False, "context_tokens": context, "max_output_tokens": 4096,
                      "source_citation_mode": CITATION_VERSION}
    value["screening"] = {**value["screening"], "limit": 0, "retry_errors": False}
    value["search"]["enabled"] = False
    return value


def quality_checks(rows, labels):
    basic = pilot_checks(rows, labels)
    errors = list(basic["errors"])
    by_title = {r["title"]: r for r in rows}
    for title, label in labels.items():
        row = by_title.get(title, {})
        if label == "out_of_scope" and row.get("ai_decision") != "exclude":
            errors.append("Explicit unrelated control was not excluded: " + title)
        criteria = {c["criterion"]: c["value"] for c in json.loads(row.get("criteria_json") or "[]")}
        # The core prioritizes other explicit failed criteria before
        # primary_research. A correctly recognized review can therefore have
        # an operational decision of exclude. Test the research classification
        # itself, rather than imposing a different decision-routing rule.
        if label == "secondary" and (row.get("ai_study_kind") != "secondary"
                or criteria.get("primary_research") != "no" or row.get("ai_decision") == "include"):
            errors.append("Explicit secondary-literature control was not identified: " + title)
        if label == "retain":
            if criteria.get("recommender_context") != "yes":
                errors.append("Recommender context not recognised in positive control: " + title)
    return {**basic, "passed": not errors, "errors": errors,
            "interpretation": "A bounded failure check, not a population accuracy or recall estimate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435")
    parser.add_argument("--context", type=int, default=32768)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent; source = args.source_run.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(source / "manifest.json")
    config = configured(manifest["config"], args.model, args.endpoint, args.context)
    config["_project_dir"] = str(root)
    if args.endpoint not in {"http://127.0.0.1:11435", "http://127.0.0.1:11434"}:
        raise ValueError("Only existing local loopback endpoints are allowed")
    service = root / "runs/local_completion_service"
    # Use the normalized records from the previous pilot, including stable IDs.
    # The hand-curated import CSV intentionally predates identifier assignment.
    inputs = read_csv(root / "runs/mac_model_pilot_final_20261023/02_deduplication/records.csv")
    labels = read_json(service / "pilot_expectations.json")
    corpus = read_csv(source / "02_deduplication/records.csv")
    leads = seed_title_candidates(corpus)["matches"]
    by_id = {r["record_id"]: r for r in corpus}
    for title in SEED_TITLES:
        candidates = [by_id[r["record_id"]] for r in leads[title] if by_id[r["record_id"]].get("abstract", "").strip()]
        if candidates:
            row = candidates[0]
            if row["title"] not in labels:
                inputs.append(row); labels[row["title"]] = "retain"
    write_csv(out / "source_inputs.csv", inputs)
    write_json(out / "expected_categories.json", labels)
    config["run_directory"] = str(out)
    write_json(out / "configuration.json", {k: v for k, v in config.items() if not k.startswith("_")})
    write_json(out / "provenance.json", {"created_utc": utcnow(), "source_run": str(source),
               "protocol_sha256": digest(config["protocol"]), "source_manifest_sha256": file_hash(source / "manifest.json"),
               "source_inputs_sha256": file_hash(out / "source_inputs.csv"),
               "same_eligibility_protocol": config["protocol"] == manifest["config"]["protocol"],
               "candidate_config_sha256": file_hash(out / "configuration.json"),
               "runner_sha256": file_hash(Path(__file__)),
               "source_citation_adapter_sha256": file_hash(root / "local_source_citations.py"),
               "eligibility_prompt": screening.SYSTEM_PROMPT, "presentation_appendix": BREVITY,
               "human_accuracy_validated": False, "database_requests": 0, "paid_ai_calls": 0})
    screening.OllamaClient = SourceCitationClient
    screening.SYSTEM_PROMPT += BREVITY
    screening.PROMPT_VERSION += "/concise-source-ids-1"
    started = time.monotonic()
    rows = screening.screen_records(inputs, "title_abstract", config, out)
    write_csv(out / "screening.csv", rows)
    write_json(out / "assessments.json", rows)
    checks = quality_checks(rows, labels)
    durations, token_rates = [], []
    audits = sorted((read_json(p) for p in (out / "checkpoints/model_audit").glob("*.json")),
                    key=lambda a: a.get("created_at", ""))
    for audit in audits:
        raw = audit.get("raw_response", {})
        if raw.get("total_duration"):
            durations.append(raw["total_duration"] / 1e9)
        if raw.get("eval_duration"):
            token_rates.append(raw.get("eval_count", 0) / (raw["eval_duration"] / 1e9))
    median = statistics.median(durations[1:] or durations) if durations else None
    checks.update(completed_utc=utcnow(), wall_seconds=round(time.monotonic() - started, 2),
                  candidate_median_seconds=round(median, 3) if median else None,
                  baseline_seconds=11.5, speed_gate_passed=median is not None and median < 9.2,
                  generation_tokens_per_second=round(statistics.median(token_rates), 2) if token_rates else None)
    checks["adoptable"] = checks["passed"] and checks["speed_gate_passed"]
    write_json(out / "quality_report.json", checks)
    print(json.dumps(checks, indent=2))
    return 0 if checks["adoptable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
