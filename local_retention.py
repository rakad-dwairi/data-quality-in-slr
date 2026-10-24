"""Read-only retention of exact, completed assessments from a frozen local run.

The caller coordinates the stopped checkpoint handoff. This module never
constructs a provider/model client, changes source artifacts, or rekeys replies.
"""
from __future__ import annotations

from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from slr_pipeline import screening
from slr_pipeline.checks import configuration_fingerprint
from slr_pipeline.common import digest, file_hash

STAGE = "title_abstract"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_OPERATIONAL = {"attempt_id", "assessed_utc", "revalidation_id",
                "generation_created_utc", "cached_response_revalidated", "cache_reused"}


def _read_json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("A retention input is missing or is a symlink: " + str(path))
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError("A retention JSON input is unreadable: " + str(path)) from exc
    if not isinstance(value, dict):
        raise ValueError("A retention JSON input must be an object: " + str(path))
    return value


def _frozen_config(source):
    manifest = _read_json(source / "manifest.json")
    config = manifest.get("config")
    if not isinstance(config, dict) or any(not isinstance(config.get(k), dict)
                                          for k in ("model", "protocol", "screening")):
        raise ValueError("The donor frozen model/protocol configuration is incomplete")
    project = Path(screening.__file__).resolve().parents[1]
    _, hashes, fingerprint = configuration_fingerprint({**config, "_project_dir": str(project)})
    if hashes != manifest.get("source_hashes"):
        raise ValueError("Current core hashes differ from the donor frozen implementation")
    if fingerprint != manifest.get("fingerprint"):
        raise ValueError("The donor frozen manifest fingerprint is inconsistent")
    return config


def retain_title_abstract(source: Path, records: list[dict]) -> dict:
    """Return operative rows/ledger plus explicitly counted non-operative history.

    ``historical_rows`` counts every non-operative TA checkpoint, including
    failures and incompatible generations. ``history`` gives each one's reason.
    An internally inconsistent audit or multiple compatible replies is an error,
    rather than permission to regenerate completed work under a new identity.
    """
    source = Path(source).resolve()
    config = _frozen_config(source)
    model, protocol = config["model"], config["protocol"]
    names = screening._criterion_names(STAGE, protocol)
    schema = screening._schema(STAGE, names)
    system = screening.SYSTEM_PROMPT + "\nTrusted review protocol and required criteria:\n" + screening._json(
        {"protocol": protocol, "required_criteria": names})
    indexed = {}
    for record in records:
        rid = str(record.get("record_id", ""))
        if not rid or rid in indexed:
            raise ValueError("Every retention record must have a unique nonempty record_id")
        indexed[rid] = record
    database = source / "checkpoints/screening.sqlite"
    if database.is_symlink() or not database.is_file():
        raise ValueError("The donor checkpoint database is missing or is a symlink")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=5)) as db:
        db.execute("PRAGMA query_only=ON")
        checkpoints = db.execute(
            "SELECT cache_key,record_id,status,result_json,raw_response,error,created_at "
            "FROM assessments WHERE stage=? ORDER BY record_id,cache_key", (STAGE,)).fetchall()
    retained, ledger, history = {}, {}, []
    for key, rid, status, payload, raw_payload, error, created in checkpoints:
        item = {"record_id": rid, "cache_key": key, "status": status}
        if status != "assessed" or rid not in indexed:
            history.append({**item, "reason": "not_assessed" if status != "assessed" else "outside_cohort"})
            continue
        if not isinstance(key, str) or not _SHA.fullmatch(key):
            raise ValueError("An assessed checkpoint has an invalid generation key")
        audit_path = source / "checkpoints/model_audit" / (key + ".json")
        audit = _read_json(audit_path)
        try:
            result, raw = json.loads(payload), json.loads(raw_payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("An assessed checkpoint payload is malformed: " + key) from exc
        expected_audit = {"record_id": rid, "stage": STAGE, "assessment_fingerprint": key,
                          "result": result, "raw_response": raw, "error": error, "created_at": created}
        if (not isinstance(result, dict) or not isinstance(raw, dict)
                or any(audit.get(k) != value for k, value in expected_audit.items())
                or not result.get("attempt_id") or audit.get("attempt_id") != result["attempt_id"]):
            raise ValueError("Assessment audit differs from its checkpoint: " + key)
        record = indexed[rid]
        evidence = {"stage": STAGE, "required_criteria": names, "record_id": rid,
                    "metadata": {field: record.get(field, "") for field in
                                 ("title", "authors", "date", "venue", "publication_type", "doi", "source", "sources")},
                    "all_source_text_provided": True, "passages": screening._passages(record, STAGE)}
        incompatibility = next((reason for matches, reason in (
            (audit.get("model") == str(model.get("name", "qwen3:8b")), "different_model"),
            (audit.get("prompt_version") == screening.PROMPT_VERSION, "different_prompt_version"),
            (audit.get("input_evidence") == evidence, "different_input"),
            (audit.get("system_prompt") == system, "different_system_prompt")) if not matches), None)
        if incompatibility:
            history.append({**item, "reason": incompatibility, "audit_path": str(audit_path)})
            continue
        model_digest = audit.get("model_digest")
        if not isinstance(model_digest, str) or not model_digest:
            raise ValueError("A compatible assessment lacks its original model digest: " + key)
        fingerprint = digest({"prompt_version": screening.PROMPT_VERSION, "system": system,
                              "evidence": evidence, "schema": schema, "model_settings": model,
                              "model_digest": model_digest})
        if fingerprint != key:
            history.append({**item, "reason": "different_generation_identity", "audit_path": str(audit_path)})
            continue
        prompt_hash = digest({"system": system, "user": screening._json(evidence), "schema": schema})
        if audit.get("prompt_sha256") != prompt_hash or ("schema" in audit and audit["schema"] != schema):
            raise ValueError("A compatible assessment has an inconsistent prompt/schema audit: " + key)
        try:
            validated = screening._validated_response(raw, evidence, STAGE, names, protocol,
                int(model.get("context_tokens", 32768)), int(model.get("max_output_tokens", 4096)))
        except (screening.AssessmentError, ValueError, TypeError, KeyError, OverflowError) as exc:
            raise ValueError("A compatible saved response fails original strict validation: " + key) from exc
        scientific = {k: value for k, value in result.items() if k not in _OPERATIONAL}
        if scientific != validated:
            raise ValueError("A compatible saved response no longer reproduces its scientific result: " + key)
        if rid in retained:
            raise ValueError("Ambiguous compatible completed assessments for record: " + rid)
        provenance = {"model": audit["model"], "model_digest": model_digest,
                      "cache_key": key, "assessment_fingerprint": key, "prompt_sha256": prompt_hash,
                      "prompt_version": audit["prompt_version"],
                      "input_sha256": screening.screening_input_hash(record, STAGE),
                      "retained_from_run": str(source)}
        retained[rid] = {**copy.deepcopy(record), **copy.deepcopy(result), **provenance}
        ledger[rid] = {"record_id": rid, "stage": STAGE, **provenance, "source_cache_key": key,
                       "result_payload_digest": digest(result),
                       "result_json_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                       "audit_path": str(audit_path), "audit_sha256": file_hash(audit_path),
                       "source_created_at": created, "attempt_id": result["attempt_id"],
                       "assessed_utc": result.get("assessed_utc", "")}
    return {"rows": [retained[rid] for rid in indexed if rid in retained],
            "ledger": [ledger[rid] for rid in indexed if rid in ledger],
            "historical_rows": len(history), "history": history, "checkpoint_rows": len(checkpoints)}
