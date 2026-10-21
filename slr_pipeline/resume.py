"""Offline, byte-preserving transfer of retrieval pages into a fresh run.

This module never constructs a provider/model client or loads credentials. It
does not resume an active process: the owner must stop it before preparation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from xml.etree.ElementTree import ParseError

from . import provider_library as providers
from .common import digest, file_hash, project_path, read_json, utcnow, write_json
from .retrieval import VERSION, _hash, _validate, cache_identity


@contextmanager
def _stopped_source(source):
    """Hold the existing lock without creating or rewriting donor files."""
    lock = source / ".lock"
    if not source.is_dir() or not lock.is_file() or lock.is_symlink():
        raise ValueError("Source must be an existing pipeline run with its original .lock file.")
    with lock.open("r+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Source run is still active. Press Ctrl+C in its terminal, wait for it to exit, then retry preparation.") from None
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _public_search(search):
    return {k: v for k, v in search.items()
            if k not in {"scopus_pagination", "scopus_view", "max_pages"}}


def _expected_params(source, query, position, args):
    # The three unchanged adapters are deliberately supported explicitly. No
    # environment variables or authentication headers are needed to check them.
    if source == "openalex":
        return {"search.title_and_abstract": query, "per_page": 100,
                "cursor": position or "*",
                "filter": f"from_publication_date:{args.start},to_publication_date:{args.end}"}
    if source == "arxiv":
        return {"search_query": f"({query}) AND submittedDate:[{args.start:%Y%m%d}0000 TO {args.end:%Y%m%d}2359]",
                "start": position or 0, "max_results": 100,
                "sortBy": "submittedDate", "sortOrder": "ascending"}
    if source == "crossref":
        return {"query.bibliographic": query, "rows": args.crossref_rows,
                "filter": f"from-pub-date:{args.start},until-pub-date:{args.end}",
                "sort": "relevance", "order": "desc", "mailto": args.email}
    raise ValueError("Cache migration supports only unchanged OpenAlex, arXiv and Crossref adapters.")


def _copy_chain(source_run, destination, job, identity, args):
    relative = Path("01_retrieval/raw") / (job["id"] + "_" + identity[:24])
    directory = source_run / relative
    if not directory.exists():
        return []
    if (directory.is_symlink() or not directory.is_dir()
            or not directory.resolve().is_relative_to(source_run)
            or (source_run / "01_retrieval").is_symlink()
            or (source_run / "01_retrieval/raw").is_symlink()):
        raise ValueError("Cache directory must be a real directory: " + str(relative))
    pages = sorted(directory.glob("page_*.json"))
    position, count, complete = None, 0, False
    signatures, cursors, copied = set(), {"*"}, []
    for index, page in enumerate(pages, 1):
        if page.name != f"page_{index:06d}.json" or page.is_symlink() or complete:
            raise ValueError("Noncontiguous or unexpected committed cache pages: " + str(relative))
        content = page.read_bytes()
        saved = json.loads(content)
        expected = {"identity": identity, "source": job["source"], "query_id": job["id"],
                    "query": job["query"], "endpoint": providers.ENDPOINTS[job["source"]],
                    "request_position": position,
                    "request_params": _expected_params(job["source"], job["query"], position, args)}
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Cache identity, query or cursor mismatch: " + str(relative / page.name))
        try:
            rows, total, cursor = providers.unpack(job["source"], saved["raw_response"])
        except (ValueError, KeyError, TypeError, AttributeError, ParseError, providers.APIError):
            raise ValueError("Cached raw response cannot be validated: " + str(relative / page.name)) from None
        if rows != saved["records"] or total != saved["provider_total"] or cursor != saved["next_cursor"]:
            raise ValueError("Cached normalized records differ from raw response: " + str(relative / page.name))
        if type(total) is not int or total < 0 or not isinstance(rows, list):
            raise ValueError("Invalid provider total/records in " + str(relative / page.name))
        signature = _hash([r.get("source_id") or r.get("doi") or r.get("title") for r in rows])
        if (rows and signature in signatures) or (not rows and count < total):
            raise ValueError("Repeated or prematurely empty page: " + str(relative / page.name))
        count += len(rows)
        complete = job["source"] == "crossref" or not rows or count >= total
        if job["source"] == "openalex":
            if not complete:
                if not cursor or cursor in cursors:
                    raise ValueError("Missing/repeated cached cursor: " + str(relative / page.name))
                cursors.add(cursor)
                position = cursor
        else:
            position = (position or 0) + len(rows)
        signatures.add(signature)
        target = destination / relative / page.name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Independent copies survive deletion of the donor after preparation.
        target.write_bytes(content)
        checksum = hashlib.sha256(content).hexdigest()
        if file_hash(target) != checksum:
            raise ValueError("Copied cache failed its checksum: " + str(relative / page.name))
        copied.append({"path": str(relative / page.name), "sha256": checksum,
                       "bytes": len(content), "source": job["source"],
                       "query_id": job["id"], "records": len(rows)})
    return copied


def prepare_resume(config: dict, source_run: Path) -> dict:
    """Prepare a new frozen run using validated cached pages; no network calls."""
    from .cli import initialize, sanitized_config

    sanitized_config(config)
    search, sources, topics, start, end, _, rows, email = _validate(config)
    source = Path(source_run).expanduser().resolve()
    target = project_path(config, config["run_directory"])
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and destination must be separate run directories.")
    if target.exists():
        raise ValueError("Destination already exists. To resume it, repeat run_slr.py with the new config; preparation never overwrites runs.")
    if config.get("imports"):
        raise ValueError("This cache transfer supports API-only runs, without manual imports.")
    with _stopped_source(source):
        manifest = read_json(source / "manifest.json", {})
        plan = read_json(source / "01_retrieval/search_plan.json", {})
        old_config = manifest.get("config", {})
        old_search = old_config.get("search", {})
        old_hashes = manifest.get("source_hashes", {})
        if not old_config or not old_hashes or old_config.get("imports"):
            raise ValueError("Source must contain a frozen API-only manifest and search plan.")
        frozen_identity = json.loads(json.dumps(old_config))
        frozen_identity.get("screening", {}).pop("limit", None)
        frozen_identity.get("screening", {}).pop("retry_errors", None)
        frozen_identity.get("search", {}).pop("max_pages", None)
        if manifest.get("fingerprint") != digest({"config": frozen_identity, "source_hashes": old_hashes}):
            raise ValueError("Source manifest fingerprint is inconsistent.")
        for name, checksum in old_hashes.items():
            if Path(name).parent != Path("slr_pipeline") or Path(name).suffix != ".py":
                raise ValueError("Unexpected code snapshot path in source manifest.")
            snapshot = source / "code_snapshot" / Path(name).name
            if not snapshot.is_file() or snapshot.is_symlink() or file_hash(snapshot) != checksum:
                raise ValueError("Source code snapshot is missing or changed: " + name)
        adapter_hash = file_hash(Path(providers.__file__))
        if (old_hashes.get("slr_pipeline/provider_library.py") != adapter_hash or
                plan.get("adapter_sha256") != adapter_hash or
                plan.get("pipeline_retrieval_version") != VERSION):
            raise ValueError("Provider adapter/cache format changed. Refusing unverified cache reuse.")
        if _public_search(old_search) != _public_search(search):
            raise ValueError("Keep source/query/date settings unchanged to reuse downloads; only Scopus pagination/view may change.")
        jobs = providers.jobs_for(sources, topics) if search.get("enabled") else []
        if (plan.get("jobs") != jobs or plan.get("start") != str(start) or plan.get("end") != str(end)
                or plan.get("imports") or plan.get("sources") != sources
                or plan.get("topics") != {topic: providers.TOPICS[topic] for topic in topics}):
            raise ValueError("Source search plan differs from its frozen configuration.")
        if set(sources) - {"scopus", "openalex", "arxiv", "crossref"}:
            raise ValueError("This transfer supports Scopus, OpenAlex, arXiv and Crossref only.")
        source_logs = read_json(source / "01_retrieval/search_log.json", [])
        if not isinstance(source_logs, list) or any(not isinstance(row, dict) for row in source_logs):
            raise ValueError("Source search log is malformed.")
        recorded_pages = {row.get("query_id"): row.get("pages", 0) for row in source_logs}
        args = SimpleNamespace(start=start, end=end, email=email, crossref_rows=rows)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".resume-preparing-", dir=target.parent))
        try:
            copied = []
            for job in jobs:
                if job["source"] == "scopus":
                    continue  # New pagination is a different retrieval contract.
                identity = cache_identity(job, start, end, adapter_hash,
                                          search.get("scopus_pagination", "cursor"),
                                          search.get("scopus_view", "STANDARD"), rows)
                chain = _copy_chain(source, staging, job, identity, args)
                expected_pages = recorded_pages.get(job["id"], 0)
                if type(expected_pages) is not int or len(chain) < expected_pages:
                    raise ValueError("Committed pages recorded in the source log are missing for " + job["id"])
                copied.extend(chain)
            initialize(config, staging)
            report = {"prepared_utc": utcnow(), "source_run": str(source), "destination_run": str(target),
                      "source_manifest_sha256": file_hash(source / "manifest.json"),
                      "source_fingerprint": manifest["fingerprint"], "adapter_sha256": adapter_hash,
                      "copied_pages": len(copied), "copied_observations": sum(p["records"] for p in copied),
                      "files": copied, "scopus_cache_reused": False,
                      "database_requests": 0, "model_requests": 0,
                      "note": "Byte-identical independent copies of committed page caches only. Derived outputs and assessments will be rebuilt. Source run was not modified."}
            write_json(staging / "cache_migration.json", report)
            # Publish only a fully validated/frozen destination. The donor stays
            # locked until publication, so it cannot advance during the copy.
            if target.exists():
                raise ValueError("Destination appeared during preparation; it was not overwritten.")
            staging.rename(target)
            return report
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def main(argv=None):
    from .cli import load_config

    parser = argparse.ArgumentParser(description="Prepare a clean run from stopped-run downloads, entirely offline.")
    parser.add_argument("--from-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        report = prepare_resume(config, args.from_run)
        print(f"Prepared {report['destination_run']}: reused {report['copied_pages']} pages / {report['copied_observations']} retrieval observations.")
        print("No database or model requests. Original run preserved. Start run_slr.py with the same config; repeat that command to resume.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Raw cache parser errors can contain response snippets: show only
        # controlled validation messages, never JSON/body contents.
        message = str(exc) if type(exc) is ValueError else type(exc).__name__
        print("Preparation stopped: " + message)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
