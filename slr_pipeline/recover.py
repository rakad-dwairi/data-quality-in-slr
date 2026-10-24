"""Prepare a separate screening recovery run using verified local artifacts only.

No searches, downloads, model calls, credentials, or screening stages are run.
The original run stays locked and unchanged while its reusable evidence is copied.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import sys
import tempfile

from . import __version__
from .checks import configuration_fingerprint
from .common import digest, file_hash, utcnow, write_json
from .resume import _stopped_source


ALLOWED_CODE_CHANGES = {"screening.py", "cli.py", "evidence.py", "recover.py"}
STAGE_ARTIFACTS = {"retrieve": "01_retrieval/records.csv",
                   "deduplicate": "02_deduplication/records.csv"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _path(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _exists(path):
    return path.exists() or path.is_symlink()


def _json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("A required frozen recovery input is missing or is a symlink.")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError):
        raise ValueError("A frozen recovery JSON input is unreadable or malformed.") from None


def _tree_files(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Recovery evidence folders must be real local directories.")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Symlinks and special files cannot be copied into a recovery run.")
        if path.is_file():
            files.append(path)
    return files


def _source_names(hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("The donor manifest has no frozen implementation hashes.")
    names = {}
    for original, checksum in hashes.items():
        if not isinstance(original, str) or not isinstance(checksum, str) or not _SHA.fullmatch(checksum):
            raise ValueError("The donor implementation hash registry is malformed.")
        path = PurePosixPath(original.replace("\\", "/"))
        if len(path.parts) != 2 or path.parts[0] != "slr_pipeline" or path.suffix != ".py":
            raise ValueError("The donor manifest contains an unexpected implementation path.")
        if path.name in names:
            raise ValueError("The donor manifest repeats an implementation path.")
        names[path.name] = checksum
    return names


def _verify_manifest(source, root):
    from .cli import sanitized_config

    manifest = _json(source / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        raise ValueError("The donor must contain its original frozen configuration.")
    config = manifest["config"]
    sanitized_config(config)
    if any(not isinstance(config.get(key), dict) for key in
           ("search", "deduplication", "model", "screening", "fulltext", "protocol")):
        raise ValueError("The donor frozen configuration is incomplete.")
    original_hashes = manifest.get("source_hashes")
    old_hashes = _source_names(original_hashes)
    identity = copy.deepcopy(config)
    identity["screening"].pop("limit", None)
    identity["screening"].pop("retry_errors", None)
    identity["search"].pop("max_pages", None)
    if manifest.get("fingerprint") != digest({"config": identity, "source_hashes": original_hashes}):
        raise ValueError("The donor frozen manifest fingerprint is inconsistent.")
    snapshots = source / "code_snapshot"
    _tree_files(snapshots)
    newline_verification = []
    for name, checksum in old_hashes.items():
        path = snapshots / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("A donor frozen implementation snapshot is missing or changed.")
        content = path.read_bytes()
        observed = hashlib.sha256(content).hexdigest()
        if observed != checksum:
            # Legacy cli.initialize hashed original Windows CRLF source, then
            # read_text() converted its frozen snapshot to LF. Verify exactly
            # that one reversible preimage; never ignore arbitrary byte edits.
            if (b"\r" in content or b"\n" not in content or
                    hashlib.sha256(content.replace(b"\n", b"\r\n")).hexdigest() != checksum):
                raise ValueError("A donor frozen implementation snapshot is missing or changed.")
            newline_verification.append({
                "path": "code_snapshot/" + name, "snapshot_sha256": observed,
                "manifest_source_sha256": checksum,
                "method": "legacy_lf_snapshot_verified_by_exact_uniform_crlf_reconstruction",
            })
    current = {path.name: file_hash(path) for path in (root / "slr_pipeline").glob("*.py")}
    if not current or not {"cli.py", "screening.py", "provider_library.py", "retrieval.py"}.issubset(current):
        raise ValueError("The selected project is missing the current pipeline implementation.")
    changes = sorted(name for name in set(old_hashes) | set(current)
                     if old_hashes.get(name) != current.get(name))
    if set(changes) - ALLOWED_CODE_CHANGES:
        raise ValueError("Recovery allows implementation changes only in screening.py, cli.py, evidence.py and recover.py.")
    return manifest, changes, newline_verification


def _row_count(path, required_field):
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if required_field not in (reader.fieldnames or []):
            raise ValueError("A completed donor stage has an unexpected CSV schema.")
        return sum(1 for _ in reader)


def _verify_stages(source):
    state = _json(source / "stage_state.json")
    if not isinstance(state, dict):
        raise ValueError("The donor stage state is malformed.")
    verified = {}
    for stage, relative in STAGE_ARTIFACTS.items():
        entry = state.get(stage, {})
        if (not isinstance(entry, dict) or entry.get("valid") is not True
                or type(entry.get("rows")) is not int or entry["rows"] < 0
                or not isinstance(entry.get("digest"), str) or not _SHA.fullmatch(entry["digest"])
                or str(entry.get("artifact", "")).replace("\\", "/") != relative):
            raise ValueError("Recovery requires completed, valid retrieval and deduplication stages.")
        artifact = source / relative
        if (artifact.is_symlink() or not artifact.is_file()
                or file_hash(artifact) != entry.get("artifact_sha256")):
            raise ValueError("A completed donor stage artifact changed or is missing.")
        count = _row_count(artifact, "observation_id" if stage == "retrieve" else "record_id")
        if count != entry["rows"]:
            raise ValueError("A completed donor stage row count does not match its frozen state.")
        verified[stage] = {"artifact": relative, "rows": count,
                           "artifact_sha256": entry["artifact_sha256"], "source_digest": entry["digest"],
                           "digest_verification": "Source-attested typed-row digest preserved; CSV hash and row count independently verified."}
    return state, verified


def _copy_file(source, destination, staging, records):
    checksum = file_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_hash(destination) != checksum or file_hash(source) != checksum:
        raise ValueError("A recovery copy changed during verification.")
    records.append({"path": destination.relative_to(staging).as_posix(),
                    "source_sha256": checksum, "sha256": checksum,
                    "bytes": destination.stat().st_size, "method": "byte_copy"})


def _copy_tree(source, destination, staging, records):
    files = _tree_files(source)
    destination.mkdir(parents=True, exist_ok=True)
    for path in files:
        _copy_file(path, destination / path.relative_to(source), staging, records)


def _database_stats(connection):
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise ValueError("A checkpoint database failed its integrity check.")
    tables = [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    counts = {name: connection.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
              for name in tables}
    statuses = []
    if "assessments" in tables:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(assessments)")}
        if {"stage", "status"}.issubset(columns):
            statuses = [{"stage": str(stage), "status": str(status), "rows": count}
                        for stage, status, count in connection.execute(
                            "SELECT stage,status,COUNT(*) FROM assessments GROUP BY stage,status ORDER BY stage,status")]
    return {"tables": counts, "assessment_counts": statuses}


def _copy_database(source, destination, staging, records):
    """Read a temporary DB/WAL mirror so SQLite cannot modify donor sidecars."""
    checksum = file_hash(source)
    sidecars = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sqlite-readonly-", dir=staging) as folder:
        mirror = Path(folder) / source.name
        shutil.copyfile(source, mirror)
        for suffix in ("-wal", "-shm"):
            original = source.with_name(source.name + suffix)
            if original.exists():
                if original.is_symlink() or not original.is_file():
                    raise ValueError("A checkpoint sidecar is not a real local file.")
                sidecars.append({"name": original.name, "sha256": file_hash(original), "bytes": original.stat().st_size})
                shutil.copyfile(original, mirror.with_name(mirror.name + suffix))
        try:
            reader = sqlite3.connect(mirror.as_uri() + "?mode=ro", uri=True)
            writer = None
            try:
                reader.execute("PRAGMA query_only=ON")
                before = _database_stats(reader)
                writer = sqlite3.connect(destination)
                reader.backup(writer)
                writer.execute("PRAGMA journal_mode=DELETE")
                after = _database_stats(writer)
                if before != after:
                    raise ValueError("Checkpoint backup row counts do not match the source.")
            finally:
                if writer is not None:
                    writer.close()
                reader.close()
        except sqlite3.Error:
            raise ValueError("A checkpoint SQLite database could not be backed up safely.") from None
    if file_hash(source) != checksum or any(
            file_hash(source.with_name(item["name"])) != item["sha256"] for item in sidecars):
        raise ValueError("A donor checkpoint changed during its read-only backup.")
    records.append({"path": destination.relative_to(staging).as_posix(),
                    "source_sha256": checksum, "sha256": file_hash(destination),
                    "bytes": destination.stat().st_size, "method": "sqlite_readonly_backup",
                    "source_sidecars": sidecars, "counts": before})
    return {"path": destination.relative_to(staging).as_posix(), **before}


def _copy_checkpoints(source, destination, staging, records):
    if not _exists(source):
        return []
    files = _tree_files(source)
    databases = []
    for path in files:
        if path.name.endswith(("-wal", "-shm")):
            if not path.with_name(path.name[:-4]).is_file():
                raise ValueError("An orphaned checkpoint WAL/SHM file cannot be recovered safely.")
            continue
        target = destination / path.relative_to(source)
        with path.open("rb") as handle:
            sqlite_header = handle.read(16) == b"SQLite format 3\x00"
        if sqlite_header or path.suffix.casefold() in {".sqlite", ".sqlite3", ".db"}:
            databases.append(_copy_database(path, target, staging, records))
        else:
            _copy_file(path, target, staging, records)
    return databases


def _initialize_current(config, staging, root):
    # Match cli.initialize's manifest contract, while honoring the explicit
    # project argument and preserving exact source bytes on Windows as well.
    snapshot, hashes, fingerprint = configuration_fingerprint(config)
    write_json(staging / "manifest.json", {"version": __version__, "created_utc": utcnow(),
               "fingerprint": fingerprint, "source_hashes": hashes,
               "config": snapshot, "execution_status": "prepared"})
    for source in sorted((root / "slr_pipeline").glob("*.py")):
        target = staging / "code_snapshot" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        expected = hashes.get(str(source.relative_to(root)))
        if file_hash(target) != expected or file_hash(source) != expected:
            raise ValueError("The current implementation snapshot could not be verified.")
    protocol_path = _path(root, config["protocol"].get("source_plan", "."))
    write_json(staging / "protocol_snapshot.json", {"protocol": config["protocol"],
               "source_plan_sha256": file_hash(protocol_path) if protocol_path.is_file() else None,
               "scientific_status": "Screening recovery; original protocol is unchanged and scientific review remains pending."})


def prepare_recovery(source_run, run_directory, config_out, limit=3, project=None, context_tokens=None):
    """Freeze a new screening run from stopped local evidence; never run stages."""
    from .cli import PROJECT

    if type(limit) is not int or limit < 0:
        raise ValueError("The screening limit must be a nonnegative integer; zero means all pending work.")
    if context_tokens is not None and (type(context_tokens) is not int or context_tokens <= 0):
        raise ValueError("The context override must be a positive integer.")
    root = Path(project if project is not None else PROJECT).expanduser().resolve()
    source, target, config_path = (_path(root, value) for value in (source_run, run_directory, config_out))
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("The donor and recovery destination must be separate run directories.")
    if source == config_path or source in config_path.parents or target == config_path or target in config_path.parents:
        raise ValueError("The recovery config must be outside both donor and destination run directories.")
    if _exists(target) or _exists(config_path):
        raise ValueError("The recovery destination and output config must both be new; existing paths are never overwritten.")
    with _stopped_source(source):
        manifest, code_changes, newline_verification = _verify_manifest(source, root)
        source_state, stages = _verify_stages(source)
        # Validate whole reusable trees before any destination directory is made.
        for name in ("01_retrieval", "02_deduplication"):
            _tree_files(source / name)
        if _exists(source / "checkpoints"):
            _tree_files(source / "checkpoints")
        pdfs = source / "04_full_text/pdfs"
        if _exists(pdfs):
            _tree_files(pdfs)
        config = copy.deepcopy(manifest["config"])
        original_context = config["model"]["context_tokens"]
        if context_tokens is not None:
            if context_tokens <= config["model"]["max_output_tokens"] + 2048:
                raise ValueError("The context override must leave more than 2048 tokens beyond the output reserve.")
            config["model"]["context_tokens"] = context_tokens
        config["run_directory"] = str(run_directory)
        config["screening"].update(limit=limit, retry_errors=True)
        config["_project_dir"] = str(root)
        config["_config_path"] = str(config_path)
        public_config = {key: value for key, value in config.items() if not key.startswith("_")}
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".screening-recovery-", dir=target.parent))
        config_created = False
        published = False
        records = []
        try:
            for name in ("01_retrieval", "02_deduplication"):
                _copy_tree(source / name, staging / name, staging, records)
            databases = _copy_checkpoints(source / "checkpoints", staging / "checkpoints", staging, records)
            if _exists(pdfs):
                _copy_tree(pdfs, staging / "04_full_text/pdfs", staging, records)
            origin = staging / "recovery_origin"
            for name in ("manifest.json", "stage_state.json", "protocol_snapshot.json"):
                original = source / name
                if original.exists():
                    _copy_file(original, origin / name, staging, records)
            _copy_tree(source / "code_snapshot", origin / "code_snapshot", staging, records)
            _initialize_current(config, staging, root)
            state = {name: copy.deepcopy(source_state[name]) for name in STAGE_ARTIFACTS}
            state.update(screen={"valid": False}, fulltext={"valid": False})
            write_json(staging / "stage_state.json", state)
            # Verify the actual copied artifacts against the preserved stage
            # attestations as well as checking the donor before copying.
            _verify_stages(staging)
            inventory = [{"path": path.relative_to(staging).as_posix(), "sha256": file_hash(path),
                          "bytes": path.stat().st_size} for path in _tree_files(staging)]
            report = {"prepared_utc": utcnow(), "source_run": str(source),
                      "run_directory": str(target), "config_out": str(config_path),
                      "source_manifest_sha256": file_hash(source / "manifest.json"),
                      "source_fingerprint": manifest["fingerprint"], "allowed_code_changes": code_changes,
                      "legacy_newline_verification": newline_verification,
                      "verified_stages": stages, "checkpoint_databases": databases,
                      "copied_files": records, "files": inventory,
                      "database_requests": 0, "model_requests": 0, "pipeline_stages_executed": 0,
                      "screening_limit": limit,
                      "context_override": None if context_tokens is None else {
                          "previous": original_context, "current": context_tokens,
                          "note": "Changed context creates a different model-generation identity; saved replies are not silently reused across it."},
                      "note": "Only completed retrieval/deduplication state and local caches were reused. Screening labels were not imported; final inclusion and scientific review remain pending."}
            write_json(staging / "screening_recovery.json", report)
            if _exists(target) or _exists(config_path):
                raise ValueError("A recovery output appeared during preparation; it was not overwritten.")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            config_created = True
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                json.dump(public_config, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if _exists(target):
                raise ValueError("The recovery destination appeared during preparation; it was not overwritten.")
            staging.rename(target)
            published = True
            return report
        finally:
            if not published and config_created:
                config_path.unlink()
            if staging.exists():
                shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a separate screening recovery from verified local caches, without running the pipeline.")
    parser.add_argument("--from-run", default="runs/database_review_verified")
    parser.add_argument("--run-directory", default="runs/database_review_screening_recovery")
    parser.add_argument("--config-out", default="config/screening_recovery.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--context-tokens", type=int, default=None,
                        help="Optional new local model context; retrieval and PDF caches remain reusable.")
    args = parser.parse_args(argv)
    try:
        report = prepare_recovery(args.from_run, args.run_directory, args.config_out, args.limit,
                                  context_tokens=args.context_tokens)
        print("Prepared recovery config: " + report["config_out"])
        print("Recovery run: " + report["run_directory"])
        print("Original run preserved. No database/model requests or pipeline stages executed.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        # Controlled validation messages only; do not echo cached response text.
        message = str(exc) if type(exc) is ValueError else type(exc).__name__
        print("Recovery preparation stopped: " + message)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
