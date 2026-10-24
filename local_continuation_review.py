"""Continue the frozen strong-model full-text reviews without rewriting donors.

This is an optional local coordinator, not a core pipeline modification. Import
and completed-result reuse perform no inference or retrieval. Only ``review``
may prepare a previously unattempted source and generate a missing strong-model
pass. Callers must serialize reviews; prompt/client substitutions are scoped.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
from threading import RLock
from types import SimpleNamespace
from uuid import uuid4

import early_local_review as early
from local_source_citations import SourceCitationClient, VERSION as CITATION_VERSION, wire_request
from slr_pipeline import screening
from slr_pipeline.cli import date_window_status
from slr_pipeline.common import digest, file_hash, read_csv, read_json, utcnow, write_json, write_csv
from slr_pipeline.fulltext import prepare_fulltexts

VERSION = "strong-review-continuation-1"
STRONG_MODEL = "qwen3.5:35b"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
_LOCK = RLock()
_OPERATIONAL = {"attempt_id", "assessed_utc", "revalidation_id", "generation_created_utc",
                "cached_response_revalidated", "cache_reused"}
_CORE_FILES = ("screening.py", "fulltext.py", "common.py", "evidence.py")


def _object(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked continuation evidence: " + str(path))
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Continuation evidence must be a JSON object: " + str(path))
    return value


def _public_config(config):
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _adapter_fits(client, system, evidence):
    """Use core packet coverage, including the real citation-wire overhead."""
    names = evidence["required_criteria"]
    sent, _, _ = wire_request([
        {"role": "system", "content": system},
        {"role": "user", "content": screening._json(evidence)},
    ], screening._schema("full_text", names))
    return sum(len(message["content"].encode("utf-8")) for message in sent) + 2048 + client.output <= client.context


@contextmanager
def _strong_context(expected_digest=""):
    """Use the original source-citation client, restoring all global bindings."""
    with _LOCK:
        original_client, original_fits = screening.OllamaClient, screening._fits
        class PinnedStrongClient(SourceCitationClient):
            def preflight(self):
                super().preflight()
                expected = expected_digest() if callable(expected_digest) else expected_digest
                if expected and self.model_digest != expected:
                    raise screening.ModelUnavailable("The installed strong model differs from the frozen completed reviews; no new pass was generated.")
        screening.OllamaClient, screening._fits = PinnedStrongClient, _adapter_fits
        try:
            yield
        finally:
            screening.OllamaClient, screening._fits = original_client, original_fits


class ReviewContinuation:
    """Retain old results by reference and finish missing passes in a new run.

    ``completed`` is a set of report IDs; ``results`` is a list in stable ID
    order. Both validate existing evidence without constructing a model client.
    The optional config must equal the donor's frozen configuration. A faster
    title/abstract configuration cannot be passed through to full-text review.
    """
    def __init__(self, source_early: Path, output: Path, config=None):
        self.source = Path(source_early).resolve()
        self.output = Path(output).resolve()
        self.project = Path(__file__).resolve().parent
        if self.source == self.output or self.source in self.output.parents or self.output in self.source.parents:
            raise ValueError("Continuation source and output must be separate, non-nested directories")
        self.source_manifest_path = self.source / "manifest.json"
        manifest = _object(self.source_manifest_path)
        frozen = manifest.get("config")
        if not isinstance(frozen, dict) or any(not isinstance(frozen.get(name), dict)
                                              for name in ("model", "screening", "fulltext", "protocol", "search")):
            raise ValueError("The early run has no complete frozen review configuration")
        if config is not None and _public_config(config) != _public_config(frozen):
            raise ValueError("Continuation requires the exact early frozen strong configuration")
        model = frozen["model"]
        if (model.get("name") != STRONG_MODEL or model.get("allow_remote") is not False
                or model.get("base_url") != "http://127.0.0.1:11434"
                or model.get("source_citation_mode") != CITATION_VERSION):
            raise ValueError("Detailed review requires the frozen loopback qwen3.5:35b source-citation model")
        self.config = copy.deepcopy(frozen)
        self.config["_project_dir"] = str(self.project)
        self._code = {}
        for name in (*_CORE_FILES, "early_local_review.py", "local_source_citations.py"):
            frozen_path = self.source / "code_snapshot" / name
            actual = self.project / name if name.endswith("local_review.py") or name == "local_source_citations.py" else self.project / "slr_pipeline" / name
            if not frozen_path.is_file() or file_hash(frozen_path) != file_hash(actual):
                raise ValueError("Current code differs from the early frozen review implementation: " + name)
            self._code[str(actual)] = file_hash(actual)
        self.identity = {"version": VERSION, "source_early": str(self.source),
                         "source_manifest_sha256": file_hash(self.source_manifest_path),
                         "frozen_config_sha256": digest(_public_config(frozen)),
                         "code_hashes": self._code, "continuation_script_sha256": file_hash(Path(__file__)),
                         "retention_policy": "Old results referenced unchanged; exact completed passes never regenerated"}
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "manifest.json"
        if path.exists():
            if _object(path).get("identity") != self.identity:
                raise ValueError("Continuation identity changed; preserve this output and choose a new directory")
        else:
            write_json(path, {"created_utc": utcnow(), "identity": self.identity,
                              "config": _public_config(frozen), "publication_ready": False})
        self._references = {}
        self._pass_events = read_json(self.output / "accounting.json", {}).get("pass_events", [])
        self._expected_digest = ""
        # Validate retained candidates before any export can call them eligible.
        self.results

    def _check_frozen(self):
        if file_hash(self.source_manifest_path) != self.identity["source_manifest_sha256"]:
            raise ValueError("The frozen early manifest changed during continuation")
        if any(file_hash(Path(path)) != expected for path, expected in self._code.items()):
            raise ValueError("The frozen strong review implementation changed during continuation")
        for path, reference in self._references.items():
            if not Path(path).is_file() or file_hash(Path(path)) != reference["result_sha256"]:
                raise ValueError("An immutable completed review is missing or changed: " + path)

    def _text_from_cache(self, folder, record, hint=None):
        """Restore exact extracted text without retrieval or parsing a PDF again."""
        hint = hint or {}
        manifest = Path(folder) / "04_full_text/fulltext_manifest.csv"
        rows = read_csv(manifest) if manifest.is_file() else []
        found = [row for row in rows if row.get("record_id") == record["record_id"]]
        if len(found) > 1:
            raise ValueError("Ambiguous saved full-text manifest for " + record["record_id"])
        row = found[0] if found else {}
        expected_file = hint.get("full_text_file_sha256") or row.get("file_sha256", "")
        expected_text = hint.get("full_text_sha256") or row.get("text_sha256", "")
        old_path = hint.get("full_text_path") or row.get("path", "")
        if not expected_file:
            return None
        if not old_path:
            raise ValueError("A completed full-text extraction has no preserved source path")
        path = Path(old_path)
        copied = Path(folder) / "04_full_text/pdfs" / (record["record_id"] + path.suffix.lower())
        if copied.is_file():
            path = copied
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected_file:
            raise ValueError("Saved full-text file is missing or changed for " + record["record_id"])
        caches = []
        for candidate in sorted((Path(folder) / "checkpoints/fulltext").glob("*.json")):
            data = _object(candidate)
            if data.get("file_sha256") == expected_file:
                caches.append(data)
        if len(caches) != 1:
            raise ValueError("Missing or ambiguous exact extraction cache for " + record["record_id"])
        cache = caches[0]
        text = cache.get("text")
        if (not isinstance(text, str) or not expected_text or cache.get("text_sha256") != expected_text
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_text):
            raise ValueError("Saved extracted source text changed for " + record["record_id"])
        complete = hint.get("text_complete") is True if "text_complete" in hint else str(row.get("text_complete", "")).lower() == "true"
        complete = bool(complete and cache.get("page_count") and not cache.get("sparse_pages")
                        and not cache.get("failed_pages") and cache.get("characters", 0) >= self.config["fulltext"].get("min_text_chars", 1000))
        return {**copy.deepcopy(record), "full_text": text, "full_text_sha256": expected_text,
                "full_text_file_sha256": expected_file, "full_text_path": str(path.resolve()),
                "full_text_status": "ready" if complete else "partial", "text_complete": complete,
                "full_text_pages": cache.get("page_count", 0), "full_text_characters": cache.get("characters", 0)}

    def _evidence(self, prepared):
        names = screening._criterion_names("full_text", self.config["protocol"])
        return {"stage": "full_text", "required_criteria": names, "record_id": prepared["record_id"],
                "metadata": {key: prepared.get(key, "") for key in ("title", "authors", "date", "venue", "publication_type", "doi", "source", "sources")},
                "all_source_text_provided": True, "passages": screening._passages(prepared, "full_text")}

    def _check_audit(self, saved, folder, evidence, system, schema):
        key = saved.get("assessment_fingerprint") or saved.get("cache_key")
        if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("Saved pass lacks its original assessment key")
        audit = _object(Path(folder) / "checkpoints/model_audit" / (key + ".json"))
        model_digest = saved.get("model_digest") or audit.get("model_digest")
        expected = digest({"prompt_version": screening.PROMPT_VERSION, "system": system, "evidence": evidence,
                           "schema": schema, "model_settings": self.config["model"], "model_digest": model_digest})
        prompt_hash = digest({"system": system, "user": screening._json(evidence), "schema": schema})
        if (not model_digest or key != expected or audit.get("assessment_fingerprint") != key
                or audit.get("model") != STRONG_MODEL or audit.get("model_digest") != model_digest
                or audit.get("input_evidence") != evidence or audit.get("system_prompt") != system
                or audit.get("prompt_sha256") != prompt_hash or audit.get("prompt_version") != screening.PROMPT_VERSION
                or audit.get("record_id") != evidence["record_id"] or audit.get("stage") != "full_text"):
            raise ValueError("Saved full-text pass input/prompt/config identity does not match its frozen audit")
        if self._expected_digest and self._expected_digest != model_digest:
            raise ValueError("Saved full-text passes have different strong model identities")
        self._expected_digest = model_digest
        audited = audit.get("result")
        if not isinstance(audited, dict) or any(saved.get(key) != value for key, value in audited.items()):
            raise ValueError("Saved full-text pass differs from its original audited result")
        validated = screening._validated_response(audit.get("raw_response", {}), evidence, "full_text",
            evidence["required_criteria"], self.config["protocol"],
            self.config["model"]["context_tokens"], self.config["model"]["max_output_tokens"])
        if {key: value for key, value in audited.items() if key not in _OPERATIONAL} != validated:
            raise ValueError("Saved full-text response does not reproduce its original result; refusing to relabel it")
        return {**audited, "assessment_fingerprint": expected, "cache_key": expected,
                "model_digest": model_digest, "prompt_sha256": prompt_hash}

    def _check_pass(self, saved, folder, prepared, pass_name):
        if not saved:
            return
        if (saved.get("model") != STRONG_MODEL or saved.get("input_sha256") != screening.screening_input_hash(prepared, "full_text")
                or saved.get("full_text_file_sha256") != prepared["full_text_file_sha256"]
                or saved.get("full_text_sha256") != prepared["full_text_sha256"]):
            raise ValueError("Saved full-text pass differs from the exact source/model input")
        with _LOCK:
            context = early.second_pass() if pass_name == "second" else _nullcontext()
            with context:
                if saved.get("prompt_version") != screening.PROMPT_VERSION:
                    raise ValueError("Saved full-text pass has a different frozen prompt version")
                evidence = self._evidence(prepared)
                names = evidence["required_criteria"]
                system = screening.SYSTEM_PROMPT + "\nTrusted review protocol and required criteria:\n" + screening._json({"protocol": self.config["protocol"], "required_criteria": names})
                schema = screening._schema("full_text", names)
                if saved.get("screening_status") == "chunked_complete":
                    client = SimpleNamespace(context=self.config["model"]["context_tokens"], output=self.config["model"]["max_output_tokens"])
                    old_fits = screening._fits
                    screening._fits = _adapter_fits
                    try:
                        packets = screening._packets(prepared, evidence, system + screening.PACKET_RULES, client)
                    finally:
                        screening._fits = old_fits
                    keys = json.loads(saved.get("chunk_fingerprints_json") or "[]")
                    if len(keys) != len(packets):
                        raise ValueError("Saved full-text packet coverage is incomplete")
                    packets_results = []
                    for key, packet in zip(keys, packets):
                        audit = _object(Path(folder) / "checkpoints/model_audit" / (key + ".json"))
                        packets_results.append(self._check_audit({**audit.get("result", {}), "assessment_fingerprint": key,
                            "model_digest": audit.get("model_digest")}, folder, packet, system + screening.PACKET_RULES, schema))
                    combined = screening._combine_packets(packets_results, names, len(packets))
                    if any(saved.get(key) != value for key, value in combined.items() if key != "cache_reused"):
                        raise ValueError("Saved full-text aggregate differs from its preserved packets")
                elif saved.get("screening_status") == "assessed":
                    self._check_audit(saved, folder, evidence, system, schema)
                else:
                    # Technical failures remain as originally saved, without
                    # being recycled as a successful pass or regenerated.
                    if saved.get("ai_decision") != "uncertain":
                        raise ValueError("A failed full-text pass has a non-uncertain decision")

    def _validate_result(self, result, folder):
        summary = result.get("summary", {})
        rid = summary.get("record_id", "")
        if not _SAFE_ID.fullmatch(rid) or rid != Path(folder).name:
            raise ValueError("Saved review has an inconsistent report identity")
        first, second = result.get("first", {}), result.get("second", {})
        hint = first or second or summary
        prepared = self._text_from_cache(folder, first or second or summary, hint)
        if prepared:
            summary_path = Path(str(summary.get("full_text_path", "")))
            if (summary.get("full_text_file_sha256") != prepared["full_text_file_sha256"]
                    or summary.get("full_text_sha256") != prepared["full_text_sha256"]
                    or not summary_path.is_file() or summary_path.is_symlink()
                    or file_hash(summary_path) != prepared["full_text_file_sha256"]):
                raise ValueError("Review summary does not point to its exact preserved source")
            for name, data in (("first", first), ("second", second)):
                if data:
                    path = Path(folder) / (name + "_result.json")
                    if not path.is_file() or _object(path) != data:
                        raise ValueError("A completed review differs from its saved original pass")
                    self._check_pass(data, Path(folder) / name, prepared, name)
            status = early.consensus(first, second, prepared["text_complete"])
            if summary.get("review_status") == "double_pass_candidate" and (
                    status != "double_pass_candidate" or summary.get("date_window_status") != "within_window"):
                raise ValueError("A retained candidate lacks strict complete-text/date/two-pass support")
        elif summary.get("review_status") == "double_pass_candidate" or first or second:
            raise ValueError("A retained full-text decision has no verifiable preserved source")
        return result

    @property
    def results(self):
        self._check_frozen()
        results, identities = {}, {}
        for origin, parent in (("retained_reference", self.source), ("continuation", self.output)):
            for path in sorted((parent / "reports").glob("*/result.json")):
                key = str(path)
                checksum = file_hash(path)
                if key in self._references and self._references[key]["result_sha256"] != checksum:
                    raise ValueError("An immutable completed review changed: " + key)
                result = self._validate_result(_object(path), path.parent)
                rid = result["summary"]["record_id"]
                if rid in results:
                    raise ValueError("A completed report appears in both donor and continuation outputs")
                results[rid] = result
                identities[key] = {"record_id": rid, "origin": origin, "result_path": key,
                    "result_sha256": checksum, "first_cache_key": result.get("first", {}).get("cache_key", ""),
                    "second_cache_key": result.get("second", {}).get("cache_key", ""),
                    "review_status": result["summary"].get("review_status", ""),
                    "model": STRONG_MODEL, "source_modified": False}
        self._references.update(identities)
        return [copy.deepcopy(results[rid]) for rid in sorted(results)]

    @property
    def completed(self):
        return {result["summary"]["record_id"] for result in self.results}

    def _copy_partial(self, source, target):
        if target.exists():
            return
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("A partial donor report contains symlinks")
        before = {str(path.relative_to(source)): file_hash(path) for path in source.rglob("*") if path.is_file()}
        temporary = target.with_name(target.name + ".copying-" + uuid4().hex)
        try:
            shutil.copytree(source, temporary)
            after = {str(path.relative_to(source)): file_hash(path) for path in source.rglob("*") if path.is_file()}
            copied = {str(path.relative_to(temporary)): file_hash(path) for path in temporary.rglob("*") if path.is_file()}
            if before != after or before != copied:
                raise ValueError("The partial donor changed while being copied; preserve it and retry after its writer is stopped")
            temporary.replace(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        # A partial report has no result.json; it may contain a downloaded PDF,
        # exact extraction cache, immutable model events and first/second DBs.

    def _summary(self, prepared, folder, ta_key):
        return {**{key: prepared.get(key, "") for key in early.FIELDS}, "human_decision": "",
                "source_ta_cache_key": ta_key, "date_window_status": date_window_status(prepared, self.config),
                "shortlist_basis": "source_title_abstract" if ta_key else "protocol_known_item_title_lead",
                "first_audit_directory": str(folder / "first/checkpoints/model_audit"),
                "second_audit_directory": str(folder / "second/checkpoints/model_audit")}

    def review(self, record, ta_cache_key=""):
        self._check_frozen()
        rid = record.get("record_id", "")
        if not isinstance(rid, str) or not _SAFE_ID.fullmatch(rid):
            raise ValueError("Unsafe report identifier")
        for result in self.results:
            if result["summary"]["record_id"] == rid:
                saved = result.get("first") or result.get("second")
                if saved:
                    reference = next(item for item in self._references.values() if item["record_id"] == rid)
                    old_folder = Path(reference["result_path"]).parent
                    prepared = self._text_from_cache(old_folder, record, saved)
                    if saved.get("input_sha256") != screening.screening_input_hash(prepared, "full_text"):
                        raise ValueError("The requested report differs from its completed frozen input")
                # No preflight, generation, source retry, relabelling or copy.
                return result
        folder = self.output / "reports" / rid
        donor = self.source / "reports" / rid
        if donor.is_dir():
            self._copy_partial(donor, folder)
        folder.mkdir(parents=True, exist_ok=True)
        first = _object(folder / "first_result.json") if (folder / "first_result.json").exists() else {}
        second = _object(folder / "second_result.json") if (folder / "second_result.json").exists() else {}
        prepared = self._text_from_cache(folder, record, first or second)
        attempted = (folder / "04_full_text/retrieval_log.csv").exists() or (folder / "04_full_text/retrieval_attempts.jsonl").exists()
        if prepared is None and attempted:
            prepared = {**copy.deepcopy(record), "text_complete": False, "full_text_status": "not_retrieved",
                        "full_text_error": "Source retrieval was already attempted; no automatic redownload. Supply a verified local source separately."}
        elif prepared is None:
            prepared = prepare_fulltexts([record], self.config, folder)[0]
        summary = self._summary(prepared, folder, ta_cache_key)
        if prepared.get("text_complete") is not True:
            summary.update(review_status="source_needed", reason=prepared.get("full_text_error", "Incomplete preserved source text"))
            result = {"summary": summary, "first": first, "second": second}
        else:
            for name, saved in (("first", first), ("second", second)):
                if saved:
                    self._check_pass(saved, folder / name, prepared, name)
                    self._pass_events.append({"record_id": rid, "pass": name, "action": "reused_exact_saved_pass",
                                              "cache_key": saved.get("cache_key", ""), "utc": utcnow()})
            with _strong_context(lambda: self._expected_digest):
                if not first:
                    first = screening.screen_records([prepared], "full_text", self.config, folder / "first")[0]
                    first = early.stripped(first)
                    write_json(folder / "first_result.json", first)
                    self._pass_events.append({"record_id": rid, "pass": "first", "action": "continued_core_screen_records", "utc": utcnow()})
                    self._check_pass(first, folder / "first", prepared, "first")
                if first.get("screening_status") in {"assessed", "chunked_complete"} and not second:
                    with early.second_pass():
                        second = screening.screen_records([prepared], "full_text", self.config, folder / "second")[0]
                    second = early.stripped(second)
                    write_json(folder / "second_result.json", second)
                    self._pass_events.append({"record_id": rid, "pass": "second", "action": "continued_blinded_core_screen_records", "utc": utcnow()})
            status = early.consensus(first, second, True)
            summary.update(review_status=status, first_decision=first.get("ai_decision", ""), second_decision=second.get("ai_decision", ""),
                           reason="Frozen strong model, source-citation interface and two blinded prompts; AI proposals, never human confirmation.")
            if status == "double_pass_candidate" and summary["date_window_status"] != "within_window":
                summary.update(review_status="needs_date_verification", reason="Both strong AI passes include; date eligibility remains unresolved.")
            result = {"summary": summary, "first": first, "second": second}
        self._validate_result(result, folder)
        write_json(folder / "result.json", result)
        self.export()
        return copy.deepcopy(result)

    def export(self):
        results = self.results
        status = early.export_pack(self.output, results)
        references = sorted(self._references.values(), key=lambda row: row["record_id"])
        write_csv(self.output / "continuation_ledger.csv", references)
        write_json(self.output / "accounting.json", {"updated_utc": utcnow(), "version": VERSION,
                   "retained_completed_reports": sum(row["origin"] == "retained_reference" for row in references),
                   "continuation_completed_reports": sum(row["origin"] == "continuation" for row in references),
                   "pass_events": self._pass_events, "source_modified": False,
                   "model": STRONG_MODEL, "same_model_blinded_passes": True, "human_confirmed": 0})
        return status


@contextmanager
def _nullcontext():
    yield
