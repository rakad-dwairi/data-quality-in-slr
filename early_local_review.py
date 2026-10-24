#!/usr/bin/env python3
"""Paced, local-only full-text reviews of a running SLR's saved shortlist.

The primary process/configuration/code are not changed. This worker owns its
outputs, reads primary checkpoints in read-only mode, and uses the same resident
Ollama model for two blinded passes. No search adapter or paid AI is invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import time

from slr_pipeline import screening
from slr_pipeline.cli import date_window_status
from slr_pipeline.common import digest, file_hash, read_csv, read_json, run_lock, utcnow, write_csv, write_json, atomic_text
from slr_pipeline.fulltext import prepare_fulltexts
from local_source_citations import SourceCitationClient, wire_request, VERSION as CITATION_VERSION

VERSION = "early-local-2"
KNOWN_ITEM_TITLES = (
    "Data Quality Matters in Recommender Systems",
    "Collaborative Prediction and Ranking with Non-Random Missing Data",
    "Recommendations as Treatments: Debiasing Learning and Evaluation",
    "Data quality in recommender systems: the impact of completeness of item content data on prediction accuracy of recommender systems",
    "Denoising Implicit Feedback for Recommendation",
)
SECOND_INSTRUCTIONS = """
This is a separate, blinded eligibility verification pass. You have not been
given any earlier decision. Evaluate only this target report's source text
against the trusted protocol. Distinguish background problems from the target
study's actual investigation. Check operational assessment particularly closely:
require a source description of measurement, manipulation, diagnosis or empirical
assessment of the DATA property. Copy one short exact phrase from its actual
BODY passage. Do not paraphrase a method as a quotation. An absence of reporting
is unclear, never no. Apply all the same eligibility rules; do not raise the bar
to require downstream improvement or explicit ISO terminology. Do not infer a
publication decision from the title alone. Return empty extraction, appraisal
and iso_mapping arrays: this pass verifies eligibility, without duplicating the
first pass's extraction. You are an AI proposal, never a human reviewer.
"""
FIELDS = ["record_id", "title", "authors", "date", "doi", "url", "sources",
          "review_status", "first_decision", "second_decision", "date_window_status",
          "full_text_path", "full_text_file_sha256", "full_text_sha256",
          "first_audit_directory", "second_audit_directory", "source_ta_cache_key", "shortlist_basis",
          "reason", "human_decision"]


def checkpoints(source, started):
    """A short read transaction; never open the primary database for writing."""
    path = Path(source) / "checkpoints/screening.sqlite"
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT record_id,cache_key,result_json,created_at FROM assessments "
                          "WHERE stage='title_abstract' AND status='assessed' AND created_at>=? "
                          "ORDER BY created_at", (started,)).fetchall()
    return [{"record_id": rid, "cache_key": key, "result": json.loads(payload), "created_at": created}
            for rid, key, payload, created in rows]


def shortlist(rows, records, done, seed_titles=()):
    candidates = []
    normalize = lambda value: re.sub(r"\W", "", str(value).casefold())
    known = {normalize(t) for t in seed_titles}
    seen = set()
    for item in rows:
        rid, result = item["record_id"], item["result"]
        if rid in done or rid not in records:
            continue
        record = records[rid]
        if not record.get("oa_pdf_url"):
            continue
        criteria = json.loads(result.get("criteria_json") or "[]")
        yes = sum(c.get("value") == "yes" for c in criteria)
        decision = result.get("ai_decision")
        if decision == "include" or (decision == "uncertain" and yes >= 4):
            candidates.append((normalize(record.get("title")) not in known, decision != "include", -yes, item["created_at"], item))
            seen.add(rid)
    # Protocol known items are existing bibliographic leads, not manufactured
    # TA decisions. They allow early source checks before random-ID screening
    # reaches them. All other records remain in the original screening queue.
    for rid, record in records.items():
        if rid not in done and rid not in seen and record.get("oa_pdf_url") and normalize(record.get("title")) in known:
            candidates.append((False, True, 0, rid, {"record_id": rid, "cache_key": "", "created_at": "", "protocol_seed": True}))
    # This changes processing order only. No other primary records are excluded.
    return [x[-1] for x in sorted(candidates, key=lambda x: x[:-1])]


def permitted_budget(completed, baseline_seconds, share, bootstrap):
    # Credit comes from completed primary records, not wall time. A stalled
    # primary process cannot accumulate an unlimited background workload.
    return bootstrap + completed * baseline_seconds * share


@contextmanager
def second_pass():
    """Process-local prompt/schema changes; no source file is edited."""
    original = screening.SYSTEM_PROMPT, screening.PROMPT_VERSION, screening._schema
    def schema(stage, names):
        value = original[2](stage, names)
        if stage == "full_text":
            for name in ("extraction", "appraisal", "iso_mapping"):
                value["properties"][name] = {"type": "array", "items": {}, "maxItems": 0}
        return value
    screening.SYSTEM_PROMPT = original[0] + "\n" + SECOND_INSTRUCTIONS
    screening.PROMPT_VERSION = original[1] + "/blinded-eligibility-1"
    screening._schema = schema
    try:
        yield
    finally:
        screening.SYSTEM_PROMPT, screening.PROMPT_VERSION, screening._schema = original


def consensus(first, second, text_complete):
    valid = {"assessed", "chunked_complete"}
    if not text_complete:
        return "source_needed"
    if first.get("screening_status") not in valid or second.get("screening_status") not in valid:
        return "assessment_error"
    if first.get("ai_decision") == second.get("ai_decision") == "include":
        for result in (first, second):
            criteria = json.loads(result.get("criteria_json") or "[]")
            names = [c.get("criterion") for c in criteria]
            if (not set(screening.CORE_CRITERIA + ("full_text_sufficient",)).issubset(names)
                    or len(names) != len(set(names)) or any(c.get("value") != "yes" for c in criteria)):
                return "needs_review"
        return "double_pass_candidate"
    if first.get("ai_decision") == second.get("ai_decision") and first.get("ai_decision") in {"exclude", "secondary"}:
        return "agreement_to_not_include_provisional"
    return "needs_review"


def stripped(record):
    return {key: value for key, value in record.items() if key != "full_text"}


def review_one(record, config, folder, ta_key):
    rid = record["record_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", rid):
        raise ValueError("Unsafe report identifier")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    prepared = prepare_fulltexts([record], config, folder)[0]
    summary = {**{key: prepared.get(key, "") for key in FIELDS}, "human_decision": "",
               "source_ta_cache_key": ta_key, "date_window_status": date_window_status(record, config),
               "shortlist_basis": "source_title_abstract" if ta_key else "protocol_known_item_title_lead",
               "first_audit_directory": str(folder / "first/checkpoints/model_audit"),
               "second_audit_directory": str(folder / "second/checkpoints/model_audit")}
    if prepared.get("text_complete") is not True:
        summary.update(review_status="source_needed", reason=prepared.get("full_text_error", ""))
        return {"summary": summary, "first": {}, "second": {}}
    # Bound the extra GPU work per report. Long texts remain in the main queue,
    # which supports complete packet coverage; this early worker never truncates.
    names = screening._criterion_names("full_text", config.get("protocol", {}))
    system = screening.SYSTEM_PROMPT + "\nTrusted review protocol and required criteria:\n" + screening._json({"protocol": config["protocol"], "required_criteria": names})
    client = screening.OllamaClient(config["model"])
    evidence = {"stage": "full_text", "required_criteria": names, "record_id": rid,
                "metadata": {field: record.get(field, "") for field in ("title", "authors", "date", "venue", "publication_type", "doi", "source", "sources")},
                "all_source_text_provided": True, "passages": screening._passages(prepared, "full_text")}
    sent, _, _ = wire_request([{"role": "system", "content": system + SECOND_INSTRUCTIONS},
                               {"role": "user", "content": screening._json(evidence)}], screening._schema("full_text", names))
    fits = sum(len(m["content"].encode("utf-8")) for m in sent) + 2048 + client.output <= client.context
    client.close()
    if not fits:
        summary.update(review_status="deferred_long_document", reason="Complete report exceeds the early worker's one-request bound; main full-text screening retains it.")
        return {"summary": summary, "first": {}, "second": {}}
    first = screening.screen_records([prepared], "full_text", config, folder / "first")[0]
    write_json(folder / "first_result.json", stripped(first))
    if first.get("screening_status") != "assessed":
        summary.update(review_status="assessment_error", first_decision=first.get("ai_decision"), reason=first.get("ai_rationale", ""))
        return {"summary": summary, "first": stripped(first), "second": {}}
    # No previous decision/rationale/extraction is passed to this model request.
    with second_pass():
        second = screening.screen_records([prepared], "full_text", config, folder / "second")[0]
    write_json(folder / "second_result.json", stripped(second))
    status = consensus(first, second, True)
    summary.update(review_status=status, first_decision=first.get("ai_decision"), second_decision=second.get("ai_decision"),
                   reason="Two blinded prompts on the same local model; AI proposals, not human confirmation.")
    if status == "double_pass_candidate" and summary["date_window_status"] != "within_window":
        summary.update(review_status="needs_date_verification", reason="Both AI passes include; date eligibility still requires verification.")
    return {"summary": summary, "first": stripped(first), "second": stripped(second)}


def export_pack(output, results):
    output = Path(output); pack = output / "writing_pack"; pack.mkdir(exist_ok=True)
    summaries = [r["summary"] for r in results]
    candidates = [s for s in summaries if s.get("review_status") == "double_pass_candidate"]
    write_csv(pack / "early_candidates.csv", candidates, FIELDS)
    write_csv(pack / "reviewed_papers.csv", summaries, FIELDS)
    write_csv(pack / "needs_review.csv", [s for s in summaries if s not in candidates], FIELDS)
    evidence, observations, appraisal = [], [], []
    for result in results:
        rid = result["summary"]["record_id"]
        for pass_name in ("first", "second"):
            data = result.get(pass_name, {})
            evidence.extend({"record_id": rid, "review_pass": pass_name, **item} for item in json.loads(data.get("criteria_json") or "[]"))
        first = result.get("first", {})
        observations.extend({"record_id": rid, "provenance": "ai_proposed_unverified", **item} for item in json.loads(first.get("extraction_json") or "[]"))
        appraisal.extend({"record_id": rid, "provenance": "ai_proposed_unverified", **item} for item in json.loads(first.get("appraisal_json") or "[]"))
    write_csv(pack / "criterion_evidence.csv", evidence)
    write_csv(pack / "extracted_observations.csv", observations)
    write_csv(pack / "appraisal.csv", appraisal)
    ris = []
    for row in candidates:
        clean = lambda value: " ".join(str(value or "").split())
        ris += ["TY  - GEN", "ID  - " + clean(row["record_id"]), "TI  - " + clean(row.get("title"))]
        ris += ["AU  - " + clean(author) for author in str(row.get("authors", "")).split(";") if author.strip()]
        if row.get("doi"): ris.append("DO  - " + clean(row["doi"]))
        ris += ["N1  - Two AI passes using the same model; not human-confirmed.", "ER  - ", ""]
    atomic_text(pack / "early_candidates.ris", "\n".join(ris))
    status = {"updated_utc": utcnow(), "reports_processed": len(summaries), "double_pass_candidates": len(candidates),
              "statuses": dict(Counter(s.get("review_status") for s in summaries)), "human_confirmed": 0,
              "same_model_two_blinded_prompts": True, "publication_ready": False}
    write_json(pack / "status.json", status)
    atomic_text(pack / "README.md", "# Early local AI evidence pack\n\n"
                "Updated after each report, while the original collection continues screening.\n\n"
                "early_candidates.csv contains only complete-text, in-window reports supported by two blinded AI passes. "
                "Both use the same local model and can share errors; this is not independent human review. "
                "reviewed_papers.csv and needs_review.csv preserve disagreements, missing texts and other unresolved cases. "
                "criterion_evidence.csv links every model criterion to its source quotation. "
                "Extracted observations/appraisal remain unverified AI proposals. Authorised ISO mapping remains pending. "
                "Processing priority is not an exclusion rule or a target count. "
                "Report identity, related versions, the final study set and extraction still need verification before publication.\n")
    return status


def initialize(root, source, output, share, bootstrap):
    root, source, output = Path(root).resolve(), Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("The early worker must have a separate output directory")
    if not 0 < share <= 0.2 or not 0 <= bootstrap <= 600:
        raise ValueError("Extra review share must be at most 20%, with at most 600 bootstrap seconds")
    manifest = read_json(source / "manifest.json")
    for relative, expected in manifest["source_hashes"].items():
        if file_hash(root / relative) != expected:
            raise ValueError("Current library differs from the running source's frozen implementation")
    config = copy.deepcopy(manifest["config"])
    if config["model"].get("allow_remote") is not False or config["model"]["base_url"] != "http://127.0.0.1:11434":
        raise ValueError("The early worker requires the existing loopback-only model")
    config["model"]["source_citation_mode"] = CITATION_VERSION
    config["screening"] = {**config["screening"], "limit": 0, "retry_errors": False}
    config["fulltext"]["timeout_seconds"] = min(config["fulltext"]["timeout_seconds"], 30)
    state = read_json(source / "stage_state.json")
    output.mkdir(parents=True, exist_ok=True)
    carryover = read_json(output / "budget_carryover.json", {"spent_seconds": 0})
    previous_spend = carryover.get("spent_seconds")
    if type(previous_spend) not in {int, float} or not math.isfinite(previous_spend) or previous_spend < 0:
        raise ValueError("Prior local-review budget must be finite and nonnegative")
    identity = {"version": VERSION, "source": str(source), "source_fingerprint": manifest["fingerprint"],
                "script_sha256": file_hash(Path(__file__)), "share": share, "bootstrap_seconds": bootstrap,
                "citation_adapter_sha256": file_hash(Path(__file__).with_name("local_source_citations.py")),
                "budget_carryover": carryover,
                "baseline_seconds_per_primary_record": 11.5, "primary_started": state["screen"]["started_utc"],
                "second_instructions_sha256": digest(SECOND_INSTRUCTIONS)}
    path = output / "manifest.json"
    if path.exists() and read_json(path)["identity"] != identity:
        raise ValueError("Early-run identity changed; preserve this run and choose a new output directory")
    if not path.exists():
        write_json(path, {"created_utc": utcnow(), "identity": identity, "config": config,
                          "note": "Reads primary SQLite only; never invokes search adapters or fabricates human review."})
        shutil.copytree(source / "code_snapshot", output / "code_snapshot")
        shutil.copy2(Path(__file__), output / "code_snapshot/early_local_review.py")
        shutil.copy2(Path(__file__).with_name("local_source_citations.py"), output / "code_snapshot/local_source_citations.py")
    config["_project_dir"] = str(root)
    return config, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--share", type=float, default=.15)
    parser.add_argument("--bootstrap-seconds", type=float, default=180)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent; source = args.from_run.resolve(); output = args.output.resolve()
    config, identity = initialize(root, source, output, args.share, args.bootstrap_seconds)
    # Process-local client replacement; the original running process and source
    # files retain their own unchanged client implementation/configuration.
    screening.OllamaClient = SourceCitationClient
    records = {r["record_id"]: r for r in read_csv(source / "02_deduplication/records.csv")}
    with run_lock(output):
        while True:
            results = [read_json(p) for p in sorted((output / "reports").glob("*/result.json"))]
            done = {r["summary"]["record_id"] for r in results}
            primary = checkpoints(source, identity["primary_started"])
            spent = identity["budget_carryover"]["spent_seconds"] + sum(r.get("elapsed_seconds", 0) for r in results)
            allowance = permitted_budget(len(primary), identity["baseline_seconds_per_primary_record"], args.share, args.bootstrap_seconds)
            queue = shortlist(primary, records, done, KNOWN_ITEM_TITLES)
            phase = "waiting_for_new_shortlist" if not queue else "paced_wait"
            primary_state = read_json(source / "stage_state.json")
            if primary_state.get("screen", {}).get("valid"):
                # Let the established coordinator have the model for bulk FT.
                phase = "primary_title_abstract_finished_handoff_pending"
            progress = {"updated_utc": utcnow(), "phase": phase, "primary_checkpoints": len(primary),
                        "reports_processed": len(results), "eligible_waiting": len(queue), "spent_seconds": round(spent, 1),
                        "budget_seconds": round(allowance, 1), "next_eligible_after_primary_records": max(0, int((spent - args.bootstrap_seconds) / (11.5 * args.share)) + 1)}
            write_json(output / "progress.json", progress)
            export_pack(output, results)
            if phase == "primary_title_abstract_finished_handoff_pending":
                return 0
            if queue and spent < allowance:
                item = queue[0]; rid = item["record_id"]; folder = output / "reports" / rid
                folder.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                write_json(output / "progress.json", {**progress, "phase": "reviewing", "record_id": rid})
                result = review_one(records[rid], config, folder, item["cache_key"])
                result["elapsed_seconds"] = round(time.monotonic() - started, 2)
                write_json(folder / "result.json", result)
                print(f"[early_review] {rid}: {result['summary']['review_status']}", flush=True)
                if result["summary"]["review_status"] == "assessment_error":
                    export_pack(output, results + [result])
                    write_json(output / "progress.json", {**progress, "phase": "needs_attention", "record_id": rid,
                                                          "reason": "A local review failed; automatic extra inference stopped."})
                    return 2
                continue
            if not args.watch:
                return 0
            time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
