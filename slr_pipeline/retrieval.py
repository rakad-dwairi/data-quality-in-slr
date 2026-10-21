"""Resumable bibliographic retrieval and immutable manual-export ingestion.

Importing this module does not contact providers. Only ``collect`` performs the
searches explicitly enabled in the supplied configuration.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from . import provider_library as providers


VERSION = "1.0.0"
SCOPUS_OFFSET_LIMIT = 5000
SCOPUS_PARTITION_SCHEMA = 1
IMPORT_PROVENANCE_FIELDS = [
    "import_original_source", "import_original_query_id", "import_original_raw_ref",
    "import_original_observation_id", "import_original_sources",
    "import_original_query_ids", "import_original_provenance",
]
FIELDS = [
    "observation_id", "source", *providers.FIELDS, "query_id", "raw_ref",
    "date_status", "import_file_sha256", "import_original_source_id",
    "parent_query_id", "partition_year_start", "partition_year_end",
    *IMPORT_PROVENANCE_FIELDS,
]
LOG_FIELDS = [
    "source", "query_id", "query", "status", "pages", "records",
    "provider_total_last", "cached_pages", "error", "started_utc", "ended_utc",
    "year_partitions", "partition_log",
]


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def cache_identity(job: dict, start, end, adapter_hash: str,
                   scopus_pagination: str = "cursor", scopus_view: str = "STANDARD",
                   crossref_rows: int = 1000) -> str:
    """Existing job cache envelope, shared with offline cache migration tools."""
    source = job["source"]
    return _hash({"version": VERSION, "adapter": adapter_hash, "job": job,
                  "start": str(start), "end": str(end),
                  "scopus_pagination": scopus_pagination if source == "scopus" else None,
                  "scopus_view": scopus_view if source == "scopus" else None,
                  "crossref_rows": crossref_rows if source == "crossref" else None})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values = {}
            for key in fields:
                value = str(row.get(key, "") if row.get(key) is not None else "")
                values[key] = "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value
            writer.writerow(values)
    tmp.replace(path)


def _safe_error(exc: Exception, client=None) -> str:
    # Provider exceptions can contain URLs including credential parameters.
    message = str(exc) if isinstance(exc, (ValueError, providers.APIError)) else type(exc).__name__
    if client:
        message = client.redact(message)
    for name in [*providers.KEYS.values(), "SCOPUS_INST_TOKEN"]:
        secret = os.getenv(name, "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return re.sub(r"(?i)(apikey|api_key|token|authorization)=([^&\s]+)",
                  r"\1=[REDACTED]", message)[:1000]


def _unescape_export(value):
    """Undo the formula guard used by our own CSV exports only."""
    if isinstance(value, str) and len(value) > 1 and value[0] == "'" and value[1] in "=+-@":
        return value[1:]
    return value


def _canonical(item: dict) -> dict:
    # Familiar provider-export headings are accepted alongside our own schema.
    lowered = {str(k).strip().lower().replace(" ", "_"): v for k, v in item.items() if k is not None}
    aliases = {
        "title": ("title", "document_title", "article_title", "ti"),
        "authors": ("authors", "author_full_names", "author", "au"),
        "date": ("date", "publication_date", "year", "publication_year", "py"),
        "doi": ("doi", "do"),
        "abstract": ("abstract", "description", "ab"),
        "venue": ("venue", "source_title", "journal", "publication_title", "so"),
        "source_id": ("source_id", "eid", "accession_number", "ut", "id"),
        "publication_type": ("publication_type", "document_type", "type", "dt"),
        "url": ("url", "link"),
    }
    values = {}
    for field in providers.FIELDS:
        values[field] = next((lowered.get(k) for k in aliases.get(field, (field,))
                              if lowered.get(k) is not None and lowered.get(k) != ""), "")
        if isinstance(values[field], (list, dict)):
            values[field] = json.dumps(values[field], ensure_ascii=False)
        values[field] = _unescape_export(values[field])
    values["date_basis"] = values["date_basis"] or "imported_metadata"
    result = providers.record(**values)
    # Keep earlier retrieval origins accessible without promoting any historical
    # screening or eligibility labels into this run's decisions.
    for field in IMPORT_PROVENANCE_FIELDS:
        original_key = field.removeprefix("import_original_")
        value = lowered.get(field) or lowered.get(original_key, "")
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        result[field] = str(_unescape_export(value) if value is not None else "")
    return result


def _import_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in (".ris", ".bib"):
        return providers.imported(path)
    if suffix == ".csv":
        limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(limit)
                break
            except OverflowError:
                limit //= 10
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [_canonical(row) for row in csv.DictReader(handle)]
    if suffix in (".json", ".jsonl"):
        with path.open(encoding="utf-8-sig") as handle:
            data = [json.loads(line) for line in handle if line.strip()] if suffix == ".jsonl" else json.load(handle)
        if isinstance(data, dict):
            data = data.get("records", data.get("observations", data.get("items")))
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError("JSON imports must contain a record list, records, observations, or items")
        return [_canonical(row) for row in data]
    raise ValueError("Supported imports: .csv, .json, .jsonl, .ris, .bib")


def _freeze_imports(imports: list[dict], project: Path, out: Path) -> list[dict]:
    """Freeze the ordered source/path/content set before any provider calls.

    Each invocation verifies both live inputs and preserved snapshots. Adding,
    replacing, moving, or deleting an input requires a distinct run directory;
    old snapshots and manifests are never rewritten to accept a changed corpus.
    """
    manifest_path = out / "import_manifest.json"
    configured = []
    for spec in imports:
        path = Path(str(spec["path"])).expanduser()
        if not path.is_absolute():
            path = project / path
        configured.append({"source": str(spec["source"]),
                           "path": str(spec["path"]),
                           "resolved_path": str(path.resolve())})
    previous = None
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))["imports"]
            prior_configured = [{k: row[k] for k in ("source", "path", "resolved_path")} for row in previous]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Import manifest is unreadable. Preserve this run and use a new run directory.") from exc
        if prior_configured != configured:
            raise ValueError("The configured import source/path list differs from this run's frozen manifest. "
                             "Use a new run directory for changed inputs; existing snapshots are preserved.")
    frozen = []
    for index, spec in enumerate(configured):
        try:
            content = Path(spec["resolved_path"]).read_bytes()
        except OSError as exc:
            raise ValueError("A configured import file is missing or unreadable: " + spec["resolved_path"] +
                             ". Restore the original input or use a new run directory; existing snapshots are preserved.") from exc
        checksum = hashlib.sha256(content).hexdigest()
        if previous is not None and previous[index].get("sha256") != checksum:
            raise ValueError("Import content changed for " + spec["resolved_path"] +
                             ". Use a new run directory for changed inputs; existing snapshots are preserved.")
        suffix = Path(spec["resolved_path"]).suffix.lower()
        snapshot = out / "raw" / ("import_" + checksum + suffix)
        if snapshot.exists():
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != checksum:
                raise ValueError("An immutable import snapshot changed. Preserve this run and use a new run directory.")
        elif previous is not None:
            raise ValueError("An immutable import snapshot is missing. Restore it or use a new run directory.")
        else:
            tmp = snapshot.with_name(snapshot.name + ".tmp")
            tmp.write_bytes(content)
            tmp.replace(snapshot)
        frozen.append({**spec, "sha256": checksum,
                       "snapshot": str(snapshot.relative_to(out))})
    if previous is None:
        _json(manifest_path, {"version": 1, "frozen_utc": _now(), "imports": frozen})
    return frozen


def _observation(row: dict, source: str, query_id: str, raw_ref: str,
                 ordinal: int, start: date, end: date) -> dict:
    item = providers.record(**row)
    item.update({field: row.get(field, "") for field in IMPORT_PROVENANCE_FIELDS})
    # The observation identity includes the evidence location, so different
    # query hits remain separately countable before deduplication.
    item.update(source=source, query_id=query_id, raw_ref=f"{raw_ref}#record={ordinal}",
                date_status=providers.date_status(item["date"], start, end))
    item["observation_id"] = "O" + _hash([source, query_id, raw_ref, ordinal, item])[:24]
    return item


def _validate(config: dict):
    search = config.get("search", {})
    if not isinstance(search, dict):
        raise ValueError("search must be an object")
    enabled = search.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("search.enabled must be true or false")
    if not isinstance(search.get("require_all_sources", True), bool):
        raise ValueError("search.require_all_sources must be true or false")
    sources = search.get("sources", [])
    topics = search.get("topics", list(providers.TOPICS))
    if not isinstance(sources, list) or any(source not in providers.ENDPOINTS for source in sources):
        raise ValueError("search.sources must be a list of: " + ", ".join(providers.ENDPOINTS))
    if not isinstance(topics, list) or any(topic not in providers.TOPICS for topic in topics):
        raise ValueError("search.topics must be a list of: " + ", ".join(providers.TOPICS))
    start = date.fromisoformat(search.get("start", "1900-01-01"))
    if not search.get("end"):
        raise ValueError("Set an explicit search.end date for reproducibility")
    end = date.fromisoformat(search["end"])
    if start > end:
        raise ValueError("search.start must not be after search.end")
    max_pages = search.get("max_pages", 0)
    crossref_rows = search.get("crossref_rows", 1000)
    if type(max_pages) is not int or max_pages < 0 or type(crossref_rows) is not int or not 1 <= crossref_rows <= 1000:
        raise ValueError("max_pages must be >=0 and crossref_rows must be 1..1000")
    if search.get("scopus_pagination", "cursor") not in ("cursor", "offset"):
        raise ValueError("search.scopus_pagination must be cursor or offset")
    if search.get("scopus_view", "STANDARD") not in ("STANDARD", "COMPLETE"):
        raise ValueError("search.scopus_view must be STANDARD or COMPLETE")
    email = search.get("email", "").strip()
    if enabled and (not sources or not topics or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email)):
        raise ValueError("Enabled search requires sources, topics and a valid contact email")
    imports = config.get("imports", [])
    if not isinstance(imports, list) or any(not isinstance(x, dict) or not x.get("source") or not x.get("path") for x in imports):
        raise ValueError("imports must be a list of {source, path} objects")
    return search, list(dict.fromkeys(sources)), list(dict.fromkeys(topics)), start, end, max_pages, crossref_rows, email


def _collect_scopus_offset(job: dict, args, identity: str, page_dir: Path,
                           run_dir: Path, client, disabled: dict, log: dict,
                           observations: list[dict], max_pages: int, export) -> None:
    """Exhaust disjoint year partitions without requesting an entitled cursor.

    An over-limit range is probed once and split into [lo,mid], [mid+1,hi].
    Probe responses remain evidence; only leaf responses become observations.
    A single-year overflow is retained as explicitly incomplete evidence.
    Existing non-Scopus cache envelopes and adapter hashes are unaffected.
    """
    states: list[dict] = []
    partition_log = Path(run_dir) / "01_retrieval/scopus_partitions" / (job["id"] + ".json")
    log["partition_log"] = str(partition_log.relative_to(run_dir))

    def save():
        log["year_partitions"] = len(states)
        _json(partition_log, {"schema": SCOPUS_PARTITION_SCHEMA, "parent_identity": identity,
                             "parent_query_id": job["id"], "partitions": states})
        _csv(partition_log.with_suffix(".csv"), states, [
            "query_id", "parent_query_id", "year_start", "year_end", "query",
            "status", "provider_total", "children_reported_total", "pages",
            "retrieved_records", "included_observations", "cached_pages", "error"])
        export()

    def visit(low: int, high: int):
        partition_id = f"{job['id']}__years_{low}_{high}"
        part_identity = _hash({"parent_identity": identity, "schema": SCOPUS_PARTITION_SCHEMA,
                               "year_start": low, "year_end": high})
        folder = page_dir / f"years_{low}_{high}"
        folder.mkdir(parents=True, exist_ok=True)
        part = {"query_id": partition_id, "parent_query_id": job["id"],
                "year_start": low, "year_end": high,
                "query": f"TITLE-ABS-KEY({job['query']}) AND PUBYEAR > {low - 1} AND PUBYEAR < {high + 1}",
                "status": "running", "pages": 0, "retrieved_records": 0,
                "included_observations": 0, "cached_pages": 0}
        states.append(part)
        partition_args = SimpleNamespace(**{**vars(args),
            "start": max(args.start, date(low, 1, 1)),
            "end": min(args.end, date(high, 12, 31))})
        position = None
        signatures = set()
        first_total = None
        try:
            while True:
                number = part["pages"] + 1
                cache = folder / f"page_{number:06d}.json"
                from_cache = cache.exists()
                if from_cache:
                    saved = json.loads(cache.read_text(encoding="utf-8"))
                    if saved.get("identity") != part_identity or saved.get("request_position") != position:
                        raise ValueError("Scopus partition cache identity/offset mismatch; use a new run")
                    rows, total = saved["records"], saved["provider_total"]
                    if saved.get("request_params", {}).get("query") != part["query"]:
                        raise ValueError("Scopus partition cached query mismatch; use a new run")
                    part["cached_pages"] += 1
                    log["cached_pages"] += 1
                else:
                    if "scopus" in disabled:
                        raise providers.APIError(disabled["scopus"], True)
                    if not os.getenv(providers.KEYS["scopus"]):
                        raise providers.APIError("Missing environment variable: SCOPUS_API_KEY", True)
                    params, headers = providers.request_spec("scopus", job["query"], position, partition_args)
                    response = client.get(providers.ENDPOINTS["scopus"], params=params, headers=headers)
                    try:
                        body = client.redact(response.text)
                    finally:
                        response.close()
                    rows, total, cursor = providers.unpack("scopus", body)
                    saved = {"identity": part_identity, "parent_identity": identity,
                             "partition_schema": SCOPUS_PARTITION_SCHEMA,
                             "retrieved_utc": _now(), "source": "scopus",
                             "query_id": partition_id, "parent_query_id": job["id"],
                             "year_start": low, "year_end": high, "query": job["query"],
                             "endpoint": providers.ENDPOINTS["scopus"],
                             "request_params": params, "request_position": position,
                             "next_cursor": cursor, "provider_total": total,
                             "records": rows, "raw_response": body}
                if type(total) is not int or total < 0:
                    raise providers.APIError("Unexpected Scopus partition total; retrieval unfinished")
                if first_total is None:
                    first_total = total
                    part["provider_total"] = total
                    if low == args.start.year and high == args.end.year:
                        log["provider_total_last"] = total
                elif total != first_total:
                    raise providers.APIError("Scopus partition total changed between pages; retrieval unfinished")
                signature = _hash([row.get("source_id") or row.get("doi") or row.get("title") for row in rows])
                if rows and signature in signatures:
                    raise providers.APIError("Repeated Scopus partition page; retrieval unfinished")
                if not rows and part["retrieved_records"] < total:
                    raise providers.APIError("Empty Scopus partition page before total; retrieval unfinished")
                if not from_cache:
                    _json(cache, saved)
                signatures.add(signature)
                part["pages"] = number
                part["retrieved_records"] += len(rows)
                log["pages"] += 1
                if total > SCOPUS_OFFSET_LIMIT and low < high:
                    midpoint = (low + high) // 2
                    part["status"] = "split_year_range"
                    save()
                    left = visit(low, midpoint)
                    right = visit(midpoint + 1, high)
                    if left is not None and right is not None:
                        part["children_reported_total"] = left + right
                        if left + right != total:
                            part.update(status="partition_count_mismatch_unfinished",
                                        error="Child year-range totals do not sum to parent reported total; retrieval unfinished")
                    save()
                    return total
                for ordinal, row in enumerate(rows, 1):
                    item = _observation(row, "scopus", partition_id, str(cache.relative_to(run_dir)),
                                        ordinal, args.start, args.end)
                    item.update(parent_query_id=job["id"], partition_year_start=low, partition_year_end=high)
                    observations.append(item)
                part["included_observations"] += len(rows)
                log["records"] += len(rows)
                print(f"  years {low}-{high} page {number}: {len(rows)} observations ({'cached' if from_cache else 'new'}); total {total}", flush=True)
                if total > SCOPUS_OFFSET_LIMIT:
                    part.update(status="single_year_overflow_unfinished",
                                error=f"Scopus PUBYEAR={low} has {total} results, above offset limit {SCOPUS_OFFSET_LIMIT}; no verified narrower split")
                    save()
                    return total
                if not rows or part["retrieved_records"] >= total:
                    part["status"] = "api_query_exhausted"
                    save()
                    return total
                if max_pages and number >= max_pages:
                    part["status"] = "capped_unfinished"
                    save()
                    return total
                position = (position or 0) + len(rows)
                save()
        except Exception as exc:
            error = _safe_error(exc, client)
            part.update(status="error_unfinished", error=error)
            if isinstance(exc, providers.APIError) and exc.halt_source:
                disabled["scopus"] = error
            save()
            return first_total

    try:
        visit(args.start.year, args.end.year)
    finally:
        incomplete = [part for part in states if part["status"] not in ("api_query_exhausted", "split_year_range")]
        if incomplete:
            log["status"] = ("single_year_overflow_unfinished"
                             if any(part["status"] == "single_year_overflow_unfinished" for part in incomplete)
                             else "capped_unfinished" if all(part["status"] == "capped_unfinished" for part in incomplete)
                             else "error_unfinished")
            log["error"] = "; ".join(part.get("error", part["status"]) for part in incomplete)[:1000]
        else:
            log["status"] = "api_query_exhausted"
        save()


def collect(config: dict, run_dir: Path) -> list[dict]:
    """Collect preserved observations; resuming reuses completed pages and files.

    Page cache keys include the exact query, date range and adapter source hash.
    Credentials and the local page cap are excluded from cache identity, so a
    failed key can be corrected and a capped search extended without refetching.
    Provider retrieval is discovery coverage, never proof of search closure.
    """
    search, sources, topics, start, end, max_pages, crossref_rows, email = _validate(config)
    project = Path(config.get("_project_dir", Path.cwd()))
    out = Path(run_dir) / "01_retrieval"
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    observations: list[dict] = []
    logs: list[dict] = []
    import_checksums: set[str] = set()
    imports = config.get("imports", [])
    frozen_imports = _freeze_imports(imports, project, out)
    enabled = search.get("enabled", False)
    jobs = providers.jobs_for(sources, topics) if enabled else []
    adapter_hash = hashlib.sha256(Path(providers.__file__).read_bytes()).hexdigest()
    _json(out / "search_plan.json", {
        "pipeline_retrieval_version": VERSION, "adapter_sha256": adapter_hash,
        "start": str(start), "end": str(end), "search_enabled": enabled,
        "sources": sources, "topics": {k: providers.TOPICS[k] for k in topics},
        "recommender_terms": providers.RS_TERMS, "jobs": jobs,
        "max_pages": max_pages, "crossref_rows": crossref_rows,
        "scopus_pagination": search.get("scopus_pagination", "cursor"),
        "scopus_view": search.get("scopus_view", "STANDARD"),
        "imports": imports,
        "notes": ["API query exhaustion does not establish review search closure.",
                  "Crossref is ranked supplementary discovery, not exhaustive Boolean retrieval.",
                  "ACM exports and citation chasing are separate documented sources.",
                  "No inferred publication status, screening decision or ISO mapping is imported."]})

    def export():
        _csv(out / "records.csv", observations, FIELDS)
        _csv(out / "search_log.csv", logs, LOG_FIELDS)
        _json(out / "search_log.json", logs)
        unfinished = [row for row in logs if row["status"] not in (
            "api_query_exhausted", "imported", "duplicate_import_skipped")]
        _json(out / "summary.json", {
            "updated_utc": _now(), "observations": len(observations),
            "search_jobs": len(jobs), "manual_import_files": len(imports),
            "configured_items": len(jobs) + len(imports), "items_logged": len(logs),
            "complete": bool(logs) and len(logs) == len(jobs) + len(imports) and not unfinished,
            "search_coverage_verified": False,
            "unfinished_sources_or_queries": [{"source": row["source"],
                "query_id": row.get("query_id", ""), "status": row["status"]} for row in unfinished],
            "note": "Complete only describes configured retrieval jobs, not database coverage or protocol closure."})

    # Preserve imports before starting potentially slow API queries.
    for spec in frozen_imports:
        path = Path(spec["resolved_path"])
        log = {"source": str(spec["source"]), "query_id": "", "query": "manual import: " + str(path),
               "status": "import_started", "pages": 0, "records": 0, "started_utc": _now()}
        logs.append(log)
        try:
            checksum = spec["sha256"]
            log["query_id"] = "import_" + checksum[:24]
            if checksum in import_checksums:
                log["status"] = "duplicate_import_skipped"
                continue
            raw_file = out / spec["snapshot"]
            rows = _import_rows(raw_file)
            if not rows:
                raise ValueError("No records were parsed from import")
            for ordinal, row in enumerate(rows, 1):
                original_id = row.get("source_id", "")
                # RIS/BibTeX entry keys and generic CSV IDs are not guaranteed
                # provider identifiers; namespace to the immutable export.
                row["source_id"] = f"import:{checksum[:24]}:{ordinal}:{original_id}"
                item = _observation(row, str(spec["source"]), log["query_id"],
                                    str(raw_file.relative_to(run_dir)), ordinal, start, end)
                item.update(import_file_sha256=checksum, import_original_source_id=original_id)
                observations.append(item)
            import_checksums.add(checksum)
            log.update(status="imported", pages=1, records=len(rows))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            log.update(status="import_error_unfinished", error=_safe_error(exc))
            print(f"Import incomplete ({path.name}): {log['error']}", flush=True)
        finally:
            log["ended_utc"] = _now()
            export()

    args = SimpleNamespace(start=start, end=end, email=email, crossref_rows=crossref_rows,
                           scopus_pagination=search.get("scopus_pagination", "cursor"),
                           scopus_view=search.get("scopus_view", "STANDARD"))
    client = providers.Client(email) if jobs else None
    disabled: dict[str, str] = {}
    try:
        for job_index, job in enumerate(jobs, 1):
            source = job["source"]
            log = {"source": source, "query_id": job["id"], "query": job["query"],
                   "status": "running", "pages": 0, "records": 0, "cached_pages": 0, "started_utc": _now()}
            logs.append(log)
            identity = cache_identity(job, start, end, adapter_hash,
                                      args.scopus_pagination, args.scopus_view, crossref_rows)
            page_dir = raw / (job["id"] + "_" + identity[:24])
            page_dir.mkdir(parents=True, exist_ok=True)
            position = None
            signatures = set()
            cursor_positions = {"*"}
            print(f"Retrieval [{job_index}/{len(jobs)}] {job['id']}", flush=True)
            try:
                if source == "scopus" and args.scopus_pagination == "offset":
                    _collect_scopus_offset(job, args, identity, page_dir, Path(run_dir), client,
                                           disabled, log, observations, max_pages, export)
                    continue
                while True:
                    page_number = log["pages"] + 1
                    cache = page_dir / f"page_{page_number:06d}.json"
                    from_cache = cache.exists()
                    if from_cache:
                        saved = json.loads(cache.read_text(encoding="utf-8"))
                        if saved.get("identity") != identity or saved.get("request_position") != position:
                            raise ValueError("Page cache identity/cursor mismatch; preserve folder and use a new run")
                        rows, total, cursor = saved["records"], saved["provider_total"], saved["next_cursor"]
                        log["cached_pages"] += 1
                    else:
                        if source in disabled:
                            raise providers.APIError(disabled[source], True)
                        key_name = providers.KEYS.get(source)
                        if key_name and not os.getenv(key_name):
                            raise providers.APIError("Missing environment variable: " + key_name, True)
                        params, headers = providers.request_spec(source, job["query"], position, args)
                        response = client.get(providers.ENDPOINTS[source], params=params, headers=headers)
                        try:
                            body = client.redact(response.text)
                        finally:
                            response.close()
                        rows, total, cursor = providers.unpack(source, body)
                        saved = {"identity": identity, "retrieved_utc": _now(),
                                 "source": source, "query_id": job["id"], "query": job["query"],
                                 "endpoint": providers.ENDPOINTS[source],
                                 "request_params": {k: v for k, v in params.items() if k.lower() not in ("apikey", "api_key", "token")},
                                 "request_position": position, "next_cursor": cursor,
                                 "provider_total": total, "records": rows, "raw_response": body}
                    if not isinstance(total, int) or total < 0:
                        raise providers.APIError("Unexpected provider total; retrieval unfinished")
                    signature = _hash([row.get("source_id") or row.get("doi") or row.get("title") for row in rows])
                    if rows and signature in signatures:
                        raise providers.APIError("Repeated page; retrieval unfinished")
                    if not rows and log["records"] < total:
                        raise providers.APIError("Empty page before reported total; retrieval unfinished")
                    new_count = log["records"] + len(rows)
                    uses_cursor = source == "openalex" or (source == "scopus" and args.scopus_pagination == "cursor")
                    if uses_cursor and rows and new_count < total and (not cursor or cursor in cursor_positions):
                        raise providers.APIError("Missing/repeated cursor before reported total; retrieval unfinished")
                    if not from_cache:
                        # Raw response, normalized rows and cursor commit together.
                        _json(cache, saved)
                    signatures.add(signature)
                    for ordinal, row in enumerate(rows, 1):
                        observations.append(_observation(row, source, job["id"],
                                            str(cache.relative_to(run_dir)), ordinal, start, end))
                    log.update(pages=page_number, records=new_count, provider_total_last=total)
                    print(f"  page {page_number}: {len(rows)} observations ({'cached' if from_cache else 'new'})", flush=True)
                    export()
                    if source == "crossref":
                        log["status"] = "ranked_discovery_capped" if total > len(rows) else "ranked_discovery_returned"
                        break
                    if not rows or new_count >= total:
                        log["status"] = "api_query_exhausted"
                        break
                    if source == "scopus" and args.scopus_pagination == "offset" and (position or 0) + len(rows) >= 5000:
                        log["status"] = "provider_offset_cap_unfinished"
                        log["error"] = "Scopus offset limit (5000); use cursor pagination with an eligible API entitlement."
                        break
                    if max_pages and page_number >= max_pages:
                        log["status"] = "capped_unfinished"
                        break
                    if uses_cursor:
                        position = cursor
                        cursor_positions.add(cursor)
                    elif source == "wos":
                        position = (position or 1) + 1
                    else:
                        initial = 1 if source == "ieee" else 0
                        position = (position or initial) + len(rows)
            except Exception as exc:
                error = _safe_error(exc, client)
                log.update(status="error_unfinished", error=error)
                if isinstance(exc, providers.APIError) and exc.halt_source:
                    disabled[source] = error
                print(f"  Retrieval incomplete: {error}", flush=True)
            finally:
                log["ended_utc"] = _now()
                export()
    except KeyboardInterrupt:
        export()
        raise
    finally:
        if client:
            client.session.close()
    export()
    if not observations:
        raise ValueError("No observations available. Add imports or enable a successful search; inspect 01_retrieval/search_log.csv")
    return observations
