#!/usr/bin/env python3
"""Export a few saved title/abstract audits using only the Python standard library.

This script never imports the pipeline, opens SQLite, loads .env, follows paths
from audit contents, or makes network/model requests. The source run is read only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_RECORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_STAGE = "title_abstract"
_STATUSES = {"assessed", "model_or_validation_error", "model_unavailable"}
_FIELDS = ("title", "abstract", "authors", "date", "venue", "publication_type", "doi")


def _no_symlinks(path: Path) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f"Symlink paths are not allowed: {item}")
    return path


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode("utf-8")).hexdigest()


def _cell(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value if value is not None else "")


def _read_rows(path: Path) -> list[dict]:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"record_id", "title", "abstract", "screening_status"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Screening CSV is missing required columns")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Screening CSV has ambiguous duplicate columns")
        # Reverse only the spreadsheet escaping used by the CSV writer.
        return [{key: value[1:] if value and value.startswith("'") and
                 value[1:].lstrip().startswith(("=", "+", "-", "@")) else (value or "")
                 for key, value in row.items() if key is not None}
                for row in reader]


def _select_rows(rows: list[dict], record_ids: list[str], limit: int) -> list[dict]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Repeated record IDs are not allowed")
    if any(not _RECORD.fullmatch(record_id) for record_id in record_ids):
        raise ValueError("Record IDs must be plain identifiers")
    if len(record_ids) > limit:
        raise ValueError("Requested record IDs exceed limit")
    if record_ids:
        selected = []
        for record_id in record_ids:
            found = [row for row in rows if row["record_id"] == record_id]
            if len(found) != 1:
                raise ValueError(f"Expected exactly one CSV row for {record_id}")
            if found[0]["screening_status"] not in _STATUSES:
                raise ValueError(f"No completed assessment or model error for {record_id}")
            selected.append(found[0])
    else:
        selected = [row for row in rows if row["screening_status"] == "assessed"][:limit]
    if not selected:
        raise ValueError("No assessed records selected; use --record-id for saved model errors")
    ids = [row["record_id"] for row in selected]
    if len(ids) != len(set(ids)) or any(not _RECORD.fullmatch(item) for item in ids):
        raise ValueError("Selected CSV record IDs are duplicate or invalid")
    if any(sum(row["record_id"] == item for row in rows) != 1 for item in ids):
        raise ValueError("Selected record IDs have duplicate CSV rows")
    return selected


def _load_audit(path: Path) -> tuple[dict, bytes]:
    _no_symlinks(path)
    if not _HASH.fullmatch(path.stem) or path.suffix != ".json" or not path.is_file():
        raise ValueError("Audit filename must be a plain SHA-256 hash plus .json")
    raw = path.read_bytes()
    try:
        audit = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid audit JSON: {path.name}") from exc
    if not isinstance(audit, dict) or audit.get("assessment_fingerprint") != path.stem:
        raise ValueError(f"Audit fingerprint does not match filename: {path.name}")
    return audit, raw


def _matches(audit: dict, row: dict) -> bool:
    if audit.get("record_id") != row["record_id"] or audit.get("stage") != _STAGE:
        return False
    evidence = audit.get("input_evidence")
    result = audit.get("result")
    if not isinstance(evidence, dict) or not isinstance(result, dict):
        return False
    if evidence.get("record_id") != row["record_id"] or evidence.get("stage") != _STAGE:
        return False
    passages, metadata = evidence.get("passages"), evidence.get("metadata", {})
    if not isinstance(passages, dict) or not isinstance(metadata, dict):
        return False
    values = {field: (passages.get(field.upper(), "") if field in {"title", "abstract"}
                      else metadata.get(field, "")) for field in _FIELDS}
    if any(not isinstance(passages.get(key, ""), str) for key in ("TITLE", "ABSTRACT")):
        return False
    if any(_cell(values[field]) != row.get(field, "") for field in _FIELDS if field in row):
        return False
    if metadata.get("title", values["title"]) != values["title"]:
        return False
    if result.get("screening_status") != row["screening_status"]:
        return False
    for key in ("prompt_sha256", "model_digest", "model"):
        if row.get(key) and audit.get(key) != row[key]:
            return False
    if row.get("ai_decision") and result.get("ai_decision") != row["ai_decision"]:
        return False
    if row.get("input_sha256") and _digest({"stage": _STAGE, "evidence": values}) != row["input_sha256"]:
        return False
    return True


def _find_audits(directory: Path, rows: list[dict]) -> list[tuple[dict, bytes, str]]:
    found, pending = {}, []
    for row in rows:
        keys = [row[key] for key in ("cache_key", "assessment_fingerprint") if row.get(key)]
        if keys:
            if len(set(keys)) != 1 or not _HASH.fullmatch(keys[0]):
                raise ValueError(f"Invalid or conflicting audit key for {row['record_id']}")
            path = directory / f"{keys[0]}.json"
            audit, raw = _load_audit(path)
            if not _matches(audit, row):
                raise ValueError(f"Audit does not match current CSV evidence for {row['record_id']}")
            found[row["record_id"]] = (audit, raw, path.name)
        else:
            pending.append(row)
    if pending:
        candidates = []
        for path in directory.iterdir():
            if _HASH.fullmatch(path.stem) and path.suffix == ".json":
                _no_symlinks(path)
                if path.is_file():
                    candidates.append((path.stat().st_mtime_ns, path))
        candidates.sort(key=lambda item: (-item[0], item[1].name))
        selected_times = {}
        for modified, path in candidates:
            # Inspect a complete timestamp group so equally new matches cannot
            # silently choose one assessment over another.
            if all(row["record_id"] in found for row in pending) and modified < min(selected_times.values()):
                break
            audit, raw = _load_audit(path)
            for row in pending:
                record_id = row["record_id"]
                if not _matches(audit, row):
                    continue
                if record_id in found:
                    if modified == selected_times[record_id]:
                        raise ValueError(f"Ambiguous equally new audits for {record_id}")
                    continue
                found[record_id] = (audit, raw, path.name)
                selected_times[record_id] = modified
        missing = [row["record_id"] for row in pending if row["record_id"] not in found]
        if missing:
            raise ValueError("No saved audit matches current CSV evidence for: " + ", ".join(missing))
    selected = [found[row["record_id"]] for row in rows]
    if len({name for _, _, name in selected}) != len(selected):
        raise ValueError("Selected audits resolve to duplicate files")
    return selected


def _summary(row: dict, audit: dict, raw: bytes, name: str) -> dict:
    response = audit.get("raw_response", {})
    message = response.get("message", {}) if isinstance(response, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    try:
        reply = json.loads(content)
    except (ValueError, TypeError):
        reply = {}
    result = audit["result"]
    adjustments = result.get("evidence_adjustments_json", "[]")
    try:
        adjustments = json.loads(adjustments) if isinstance(adjustments, str) else adjustments
    except ValueError:
        adjustments = []
    return {"record_id": row["record_id"], "title": row["title"],
            "screening_status": row["screening_status"], "ai_decision": result.get("ai_decision", ""),
            "ai_rationale": result.get("ai_rationale", ""),
            "raw_ai_rationale": reply.get("rationale", "") if isinstance(reply, dict) else "",
            "adjustments": [{key: item.get(key, "") for key in ("criterion", "action", "reason")}
                            for item in adjustments if isinstance(item, dict)] if isinstance(adjustments, list) else [],
            "audit_file": "model_audit/" + name,
            "audit_sha256": hashlib.sha256(raw).hexdigest()}


def export_diagnostics(run: Path, output: Path, record_ids=None, limit=8, overwrite=False) -> dict:
    run, output = _no_symlinks(Path(run)), _no_symlinks(Path(output))
    if not run.is_dir():
        raise ValueError("Run directory does not exist")
    if output == run or run in output.parents:
        raise ValueError("Diagnostic ZIP must be outside the source run")
    if output.suffix.lower() != ".zip" or not output.parent.is_dir():
        raise ValueError("Output must be a .zip file in an existing directory")
    if output.exists() and (not overwrite or not output.is_file()):
        raise ValueError("Output already exists; choose another name or use --overwrite")
    csv_path = _no_symlinks(run / "03_title_abstract/screening.csv")
    directory = _no_symlinks(run / "checkpoints/model_audit")
    rows = _select_rows(_read_rows(csv_path), list(record_ids or []), limit)
    audits = _find_audits(directory, rows)
    summary = {"stage": _STAGE, "selected_count": len(rows),
               "records": [_summary(row, *audit) for row, audit in zip(rows, audits)],
               "database_requests": 0, "model_requests": 0}
    descriptor, temporary = tempfile.mkstemp(prefix=".screening-diagnostics-", suffix=".zip", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for _, raw, name in audits:
                    archive.writestr("model_audit/" + name, raw)
                archive.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        if overwrite:
            os.replace(temporary, output)
        else:
            # Publish a complete file exclusively; never leave a partial ZIP.
            os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/database_review_screening_calibration"))
    parser.add_argument("--output", type=Path, default=Path("screening_failure_samples_calibration.zip"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = export_diagnostics(args.run, args.output, args.record_id, args.limit, args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Diagnostic export stopped: {exc}\n")
    print(f"Exported {summary['selected_count']} saved title/abstract audits; database/model requests: 0.")
    print(f"ZIP: {args.output.absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
