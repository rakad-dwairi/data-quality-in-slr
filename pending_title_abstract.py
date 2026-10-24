"""Pending-only local TA screening, with immutable donor retention and batch snapshots.

The coordinator owns initialization, its run lock and any process handoff. This
module never invokes retrieval, deduplication, downloads or full-text screening.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from pathlib import Path

from local_retention import _frozen_config, retain_title_abstract
from local_source_citations import SourceCitationClient, VERSION as CITATION_VERSION
from validate_fast_local_model import BREVITY
from slr_pipeline import cli, screening
from slr_pipeline.checks import configuration_fingerprint
from slr_pipeline.common import digest, file_hash, read_json, utcnow, write_json

_SUFFIX = "/concise-source-ids-1"
_BASE_VERSION = screening.PROMPT_VERSION.removesuffix(_SUFFIX)
_BASE_PROMPT = screening.SYSTEM_PROMPT.removesuffix(BREVITY)
_UNRESOLVED = {"pending_limit", "model_unavailable", "model_or_validation_error", "context_configuration_error"}


@contextmanager
def concise_candidate_prompt(expected_model_digest=None):
    """Use exactly the candidate calibration interface, restoring globals on exit."""
    if expected_model_digest is not None and (not isinstance(expected_model_digest, str) or not expected_model_digest):
        raise ValueError("expected_model_digest must be a nonempty string or None")
    client_class = SourceCitationClient
    if expected_model_digest is not None:
        class DigestBoundClient(SourceCitationClient):
            def preflight(self):
                super().preflight()
                if self.model_digest != expected_model_digest:
                    raise screening.ModelUnavailable("Local model digest differs from the approved calibration; no generation was started")
        client_class = DigestBoundClient
    original = screening.OllamaClient, screening.SYSTEM_PROMPT, screening.PROMPT_VERSION
    screening.OllamaClient = client_class
    screening.SYSTEM_PROMPT = _BASE_PROMPT + BREVITY
    screening.PROMPT_VERSION = _BASE_VERSION + _SUFFIX
    try:
        yield
    finally:
        screening.OllamaClient, screening.SYSTEM_PROMPT, screening.PROMPT_VERSION = original


def _persist_exact(path, value):
    if path.exists():
        if path.is_symlink() or read_json(path) != value:
            raise ValueError("Saved pending-screening input changed: " + str(path))
    else:
        write_json(path, value)


def _index(rows, label):
    indexed = {}
    for row in rows:
        rid = str(row.get("record_id", ""))
        if not rid or rid in indexed:
            raise ValueError(label + " must have unique nonempty record IDs")
        indexed[rid] = row
    return indexed


def screen_remaining(records, retained_rows, config, run, *, retained_ledger=None,
                     batch_size=32, after_batch=None, expected_model_digest=None):
    """Return combined rows, preserving supplied donor rows without modification.

    Persist donor rows/optional ledger exactly. New committed checkpoints are
    verified and recovered before choosing pending IDs. Atomic completed and
    assessment snapshots update per batch; core CSV exports occur initially,
    at completion, and before a controlled error. ``after_batch`` receives a
    deep copy of completed rows after persistence and outside the prompt context.
    No caller-owned input is mutated. Failed assessments stop for inspection.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    run = Path(run).resolve()
    records, retained_rows, config = copy.deepcopy(records), copy.deepcopy(retained_rows), copy.deepcopy(config)
    indexed, retained = _index(records, "Cohort records"), _index(retained_rows, "Retained rows")
    if not set(retained).issubset(indexed):
        raise ValueError("Retained record IDs must belong to the original cohort")
    for rid, row in retained.items():
        if (row.get("screening_status") != "assessed"
                or row.get("input_sha256") != screening.screening_input_hash(indexed[rid], "title_abstract")):
            raise ValueError("Retained rows must be completed and bound to the unchanged input: " + rid)
    frozen = _frozen_config(run)
    project = Path(screening.__file__).resolve().parents[1]
    _, _, fingerprint = configuration_fingerprint({**config, "_project_dir": str(project)})
    manifest = read_json(run / "manifest.json")
    if fingerprint != manifest["fingerprint"] or config["model"] != frozen["model"]:
        raise ValueError("The pending runner must use the continuation's exact frozen configuration")
    model = config["model"]
    if (model.get("allow_remote") is not False
            or model.get("base_url") not in {"http://127.0.0.1:11434", "http://127.0.0.1:11435"}
            or model.get("source_citation_mode") != CITATION_VERSION
            or int(config["screening"].get("limit", 0)) != 0
            or config["screening"].get("retry_errors") is not False):
        raise ValueError("Pending screening requires the bounded local candidate interface and no automatic error retries")
    directory = run / "03_title_abstract"
    _persist_exact(directory / "retained_rows.json", retained_rows)
    ledger_path = directory / "retained_ledger.json"
    if retained_ledger is None:
        retained_ledger = read_json(ledger_path, [])
    retained_ledger = copy.deepcopy(retained_ledger)
    if retained_ledger and set(_index(retained_ledger, "Retention ledger")) != set(retained):
        raise ValueError("The supplied retention ledger must cover exactly the retained rows")
    _persist_exact(ledger_path, retained_ledger)
    _persist_exact(directory / "pending_identity.json", {
        "cohort_digest": digest(records), "retained_rows_digest": digest(retained_rows),
        "retained_ledger_digest": digest(retained_ledger), "run_fingerprint": fingerprint,
        "expected_model_digest": expected_model_digest,
        "prompt_version": _BASE_VERSION + _SUFFIX, "presentation_appendix_sha256": digest(BREVITY),
        "helper_sha256": file_hash(Path(__file__)),
        "citation_adapter_sha256": file_hash(Path(__file__).with_name("local_source_citations.py")),
        "calibration_runner_sha256": file_hash(Path(__file__).with_name("validate_fast_local_model.py"))})
    eligible = [row for rid, row in indexed.items() if rid not in retained]
    recovered = {"rows": [], "historical_rows": 0}
    if (run / "checkpoints/screening.sqlite").exists():
        with concise_candidate_prompt(expected_model_digest):
            recovered = retain_title_abstract(run, eligible)
    if expected_model_digest is not None and any(row.get("model_digest") != expected_model_digest for row in recovered["rows"]):
        raise ValueError("A saved candidate checkpoint has a different digest from the approved calibration")
    operative = copy.deepcopy(retained)
    for row in recovered["rows"]:
        restored = copy.deepcopy(row)
        restored.pop("retained_from_run", None)  # This is a current-run cache, not a donor assessment.
        restored["cache_reused"] = True
        operative[restored["record_id"]] = restored
    pending = [row for rid, row in indexed.items() if rid not in operative]
    generation_count = 0
    cli.begin_stage(run, "screen")

    def combined():
        return [copy.deepcopy(operative[rid]) for rid in indexed if rid in operative]

    def persist(phase, export=False):
        rows = combined()
        completed = [row for row in rows if row.get("screening_status") not in _UNRESOLVED]
        cli.snapshot_stage(run, rows, "title_abstract")
        write_json(directory / "completed_snapshot.json", completed)
        write_json(directory / "progress.json", {
            "updated_utc": utcnow(), "phase": phase, "cohort_records": len(indexed),
            "retained_rows": len(retained), "resumed_checkpoint_rows": len(recovered["rows"]),
            "newly_completed_rows": generation_count, "completed_rows": len(completed),
            "remaining_records": len(indexed) - len(completed),
            "historical_checkpoint_rows": recovered["historical_rows"],
            "completed_snapshot_sha256": file_hash(directory / "completed_snapshot.json"),
            "assessments_sha256": file_hash(directory / "assessments.json"),
            "database_requests": 0, "paid_ai_calls": 0})
        if export:
            # No full-text assessment is inferred from a title/abstract result.
            cli.export_results(config, run, ta=copy.deepcopy(rows), ft=[])
        return completed

    persist("screening", export=True)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        try:
            with concise_candidate_prompt(expected_model_digest):
                results = screening.screen_records(copy.deepcopy(batch), "title_abstract", config, run)
        except BaseException:
            # Committed assessments remain recoverable even if this batch never returned.
            persist("interrupted")
            raise
        outputs = _index(results, "Batch results")
        if set(outputs) != {row["record_id"] for row in batch}:
            raise ValueError("Candidate batch returned a different record cohort")
        for rid, row in outputs.items():
            if (row.get("model") != str(model.get("name", "qwen3:8b"))
                    or row.get("prompt_version") != _BASE_VERSION + _SUFFIX
                    or row.get("input_sha256") != screening.screening_input_hash(indexed[rid], "title_abstract")):
                raise ValueError("Candidate batch returned inconsistent original generation provenance: " + rid)
            operative[rid] = copy.deepcopy(row)
            if row.get("screening_status") == "assessed" and not row.get("cache_reused"):
                generation_count += 1
        failed = [row for row in results if row.get("screening_status") in _UNRESOLVED]
        completed = persist("needs_attention" if failed else "screening", export=bool(failed))
        if after_batch is not None:
            after_batch(copy.deepcopy(completed))
        if failed:
            raise ValueError("Pending title/abstract screening stopped with unresolved assessments; saved work is preserved")
    rows = combined()
    cli.end_stage(run, "screen", rows)
    persist("complete", export=True)
    return rows
