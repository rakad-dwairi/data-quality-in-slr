#!/usr/bin/env python3
"""Continue the explicitly authorised Mac run without per-paper Codex requests.

This coordinator uses the existing local-only pipeline. A small automatic pilot
is a failure check, not scientific validation. Human inclusion is never invented.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata

import requests

from slr_pipeline.common import read_csv, read_json, utcnow, write_csv, write_json


SEED_TITLES = (
    "Data Quality Matters in Recommender Systems",
    "Collaborative Prediction and Ranking with Non-Random Missing Data",
    "Recommendations as Treatments: Debiasing Learning and Evaluation",
    "Data quality in recommender systems: the impact of completeness of item content data on prediction accuracy of recommender systems",
    "Denoising Implicit Feedback for Recommendation",
)


def seed_title_candidates(rows, titles=SEED_TITLES):
    def normalized(value):
        return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower())
    matches = {}
    for title in titles:
        matches[title] = [{key: row.get(key, "") for key in ("record_id", "title", "authors", "date", "doi")}
                          for row in rows if normalized(row.get("title", "")) == normalized(title)]
    return {"all_seed_titles_found": all(matches.values()), "matches": matches,
            "author_year_identity_independently_verified": False,
            "note": "Known-item title leads from the supplied plan; not final inclusion or calibrated search recall."}


def pilot_checks(rows, expected):
    errors = []
    by_title = {row.get("title"): row for row in rows}
    if len(rows) != len(expected) or len(by_title) != len(rows):
        errors.append("Pilot row count or title identities differ from the supplied inputs")
    all_unclear = 0
    for title, category in expected.items():
        row = by_title.get(title, {})
        if row.get("screening_status") != "assessed":
            errors.append(f"Pilot response not assessed: {title}")
        criteria = json.loads(row.get("criteria_json") or "[]")
        if category == "retain" and int(row.get("evidence_downgraded_criteria") or 0) > len(criteria) // 2:
            errors.append(f"Most criterion judgements lost their source support: {title}")
        if not criteria or all(item.get("value") == "unclear" for item in criteria):
            all_unclear += 1
        if category == "retain" and row.get("ai_decision") not in {"include", "uncertain"}:
            errors.append(f"Potentially eligible calibration lead was discarded: {title}")
        if category == "out_of_scope" and row.get("ai_decision") == "include":
            errors.append(f"Unrelated calibration lead was included: {title}")
        if category == "secondary" and row.get("ai_decision") == "include":
            errors.append(f"Secondary calibration lead was included as primary evidence: {title}")
    if all_unclear > max(1, len(expected) // 4):
        errors.append("Too many pilot replies have no definite supported criterion")
    return {"passed": not errors, "errors": errors, "records": len(rows),
            "all_criteria_unclear": all_unclear,
            "scientific_accuracy_validated": False, "human_review_required": True}


def writing_pack(run):
    candidates = read_csv(run / "06_final/included_candidates.csv")
    final = read_csv(run / "06_final/final_included_papers.csv")
    folder = run / "07_writing_pack"
    folder.mkdir(exist_ok=True)
    write_csv(folder / "candidate_papers.csv", candidates)
    write_csv(folder / "human_confirmed_papers.csv", final)
    lines = []
    for row in candidates:
        clean = lambda value: " ".join(str(value or "").split())
        lines += ["TY  - GEN", "ID  - " + clean(row.get("record_id")),
                  "TI  - " + clean(row.get("title"))]
        lines += ["AU  - " + clean(author) for author in str(row.get("authors", "")).split(";") if author.strip()]
        for field, tag in (("date", "PY"), ("venue", "T2"), ("doi", "DO"), ("url", "UR")):
            if row.get(field):
                lines.append(tag + "  - " + clean(row[field]))
        lines += ["N1  - AI inclusion candidate; independent evidence review pending.", "ER  - ", ""]
    (folder / "candidate_references.ris").write_text("\n".join(lines), encoding="utf-8")
    extraction = run / "05_extraction"
    if extraction.is_dir():
        target = folder / "evidence_tables"
        target.mkdir(exist_ok=True)
        for path in extraction.glob("*.csv"):
            shutil.copy2(path, target / path.name)
    counts = {"created_utc": utcnow(), "ai_inclusion_candidates": len(candidates),
              "human_confirmed_reports": len(final), "publication_ready": False,
              "source_run": str(run), "target_count_used_for_selection": False}
    write_json(folder / "status.json", counts)
    (folder / "README.md").write_text(
        "# Evidence pack for writing\n\n"
        f"AI full-text inclusion candidates: {len(candidates)}. Human-confirmed reports: {len(final)}.\n\n"
        "candidate_papers.csv contains provisional, evidence-linked model proposals. "
        "candidate_references.ris can be imported into a reference manager. "
        "Evidence tables remain AI extractions requiring source checks. "
        "PDF locations are in the CSV and the original run is retained.\n\n"
        "Use the research plan for scope, methods and synthesis. Check each candidate against its "
        "full text, resolve proposed exclusions and unavailable texts, link report versions, "
        "verify extraction/appraisal and authorised ISO mappings before claiming a final SLR. "
        "Missing or uncertain evidence is never silently counted as included. "
        "No paper-count target determines selection and no human approvals were invented.\n",
        encoding="utf-8")
    return counts


def fulltext_preflight(rows, config, run, prepare, screen):
    """Check one available actual full text before enabling the bulk FT stage."""
    queue = [row for row in rows if row.get("effective_decision") in {"include", "uncertain"}
             and row.get("screening_status") != "pending_limit" and row.get("oa_pdf_url")]
    queue.sort(key=lambda row: row.get("ai_decision") != "include")
    for row in queue[:10]:
        prepared = prepare([row], config, run)
        if not prepared or prepared[0].get("full_text_status") != "ready" or prepared[0].get("text_complete") is not True:
            continue
        assessed = screen(prepared, "full_text", config, run)
        result = assessed[0] if assessed else {}
        if result.get("screening_status") not in {"assessed", "chunked_complete"}:
            raise RuntimeError("Full-text pilot did not validate; bulk full-text inference was not enabled")
        return {"record_id": row["record_id"], "status": result["screening_status"],
                "scientific_accuracy_validated": False, "cache_reusable": True}
    raise RuntimeError("No usable full text found in the bounded download pilot; inspect source access before bulk work")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--pilot-config", type=Path, default=Path("config/mac_completion_pilot.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    service = args.service.resolve()
    config_path = root / "config/mac_completion.json"
    config = read_json(config_path)
    model = config["model"]["name"]
    if config["model"].get("allow_remote") or config["model"]["base_url"] != "http://127.0.0.1:11434":
        raise ValueError("The completion coordinator requires the loopback-only local model")

    def status(phase, **extra):
        write_json(service / "completion_progress.json", {"updated_utc": utcnow(),
                   "pid": os.getpid(), "phase": phase, **extra})

    def stage(config_file, name, log_name):
        status(name, config=str(config_file))
        with (service / log_name).open("ab") as log:
            result = subprocess.run([str(root / ".venv/bin/python"), "-u", str(root / "run_slr.py"),
                                     "--config", str(config_file), "--stage", name],
                                    cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Stage {name} stopped with exit {result.returncode}; inspect {log_name}")

    session = requests.Session()
    session.trust_env = False
    try:
        status("waiting_for_model_download", model=model)
        download_deadline = time.monotonic() + 86400
        download_attempts = 0
        while True:
            response = session.get("http://127.0.0.1:11434/api/tags", timeout=15)
            response.raise_for_status()
            if any(item.get("name", item.get("model")) == model for item in response.json().get("models", [])):
                break
            if time.monotonic() > download_deadline:
                raise RuntimeError("Model download did not finish within 24 hours")
            running = False
            try:
                os.kill(int((service / "model_download.pid").read_text().strip()), 0)
                running = True
            except (OSError, ValueError):
                pass
            if not running:
                if download_attempts >= 3:
                    raise RuntimeError("Model download stopped after three resumable retries")
                environment = os.environ.copy()
                environment["OLLAMA_HOST"] = "127.0.0.1:11434"
                with (service / "model_download.log").open("ab") as log:
                    download = subprocess.Popen(["/opt/homebrew/opt/ollama/bin/ollama", "pull", model],
                                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                env=environment, start_new_session=True)
                (service / "model_download.pid").write_text(str(download.pid) + "\n")
                download_attempts += 1
            time.sleep(15)
        pilot = root / args.pilot_config
        for name in ("retrieve", "deduplicate", "screen"):
            stage(pilot, name, "pilot.log")
        rows = read_csv(root / read_json(pilot)["run_directory"] / "03_title_abstract/screening.csv")
        result = pilot_checks(rows, read_json(service / "pilot_expectations.json"))
        write_json(service / "pilot_checks.json", result)
        if not result["passed"]:
            raise RuntimeError("Local-model pilot failed a basic failure check; bulk inference was not enabled")
        status("waiting_for_retrieval_and_deduplication")
        while True:
            progress = read_json(service / "progress.json", {})
            if progress.get("phase") == "needs_attention":
                raise RuntimeError(f"Primary worker stopped during {progress.get('failed_stage')}")
            if progress.get("phase") in {"waiting_for_local_model_calibration", "screen", "title_abstract_complete_waiting_for_review"}:
                break
            time.sleep(15)
        seeds = seed_title_candidates(read_csv(root / config["run_directory"] / "02_deduplication/records.csv"))
        write_json(service / "seed_title_recall.json", seeds)
        if not seeds["all_seed_titles_found"]:
            raise RuntimeError("One or more known-item seed titles were not retrieved; inspect seed_title_recall.json before bulk screening")
        write_json(service / "pilot_passed.json", result)
        status("waiting_for_retrieval_and_title_abstract_completion", model=model)
        while True:
            progress = read_json(service / "progress.json", {})
            if progress.get("phase") == "needs_attention":
                raise RuntimeError(f"Primary worker stopped during {progress.get('failed_stage')}")
            if progress.get("phase") == "title_abstract_complete_waiting_for_review":
                break
            time.sleep(15)
        from slr_pipeline.cli import load_config
        from slr_pipeline.fulltext import prepare_fulltexts
        from slr_pipeline.screening import screen_records
        config = load_config(config_path)
        run = root / config["run_directory"]
        status("fulltext_preflight")
        report = fulltext_preflight(read_csv(run / "03_title_abstract/screening.csv"),
                                    config, run, prepare_fulltexts, screen_records)
        write_json(service / "fulltext_preflight.json", report)
        stage(config_path, "fulltext", "fulltext.log")
        counts = writing_pack(root / config["run_directory"])
        if counts["ai_inclusion_candidates"] or counts["human_confirmed_reports"]:
            status("ai_processing_finished_human_review_required", **counts)
        else:
            status("needs_attention", error="No inclusion candidates; inspect unresolved full texts and assessments", **counts)
    except Exception as exc:
        status("needs_attention", error=str(exc))
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
