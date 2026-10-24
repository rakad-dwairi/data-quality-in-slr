"""Synthetic offline proof that completed work keeps its original identity."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import local_retention
from slr_pipeline import screening
from slr_pipeline.checks import configuration_fingerprint
from slr_pipeline.common import digest, write_json


QUOTE = "The target study measures item metadata in a recommender dataset."


class LocalRetentionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        denied = AssertionError("Inference and network are forbidden in retention tests")
        for name in ("requests.sessions.Session.request", "socket.create_connection", "socket.getaddrinfo",
                     "socket.socket.connect", "socket.socket.connect_ex"):
            self.stack.enter_context(mock.patch(name, side_effect=denied))
        self.stack.enter_context(mock.patch.object(screening, "OllamaClient", side_effect=denied))
        self.source = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve() / "donor"
        (self.source / "checkpoints/model_audit").mkdir(parents=True)
        self.config = {"run_directory": "donor", "search": {"enabled": True, "max_pages": 0},
                       "model": {"name": "synthetic:original-35b", "base_url": "http://127.0.0.1:11434",
                                 "allow_remote": False, "context_tokens": 32768, "max_output_tokens": 4096},
                       "protocol": {"version": "synthetic-protocol", "iso_codebook": {}},
                       "screening": {"limit": 0, "retry_errors": True, "require_abstract": True}}
        project = Path(screening.__file__).resolve().parents[1]
        _, hashes, fingerprint = configuration_fingerprint({**self.config, "_project_dir": str(project)})
        self.manifest = {"config": copy.deepcopy(self.config), "source_hashes": hashes, "fingerprint": fingerprint}
        write_json(self.source / "manifest.json", self.manifest)
        self.record = {"record_id": "R_SYNTHETIC", "title": "Synthetic study", "abstract": QUOTE,
                       "authors": "Synthetic Author", "date": "2020", "venue": "Synthetic Venue",
                       "publication_type": "article", "doi": "", "source": "fixture", "sources": "fixture"}
        self.dbpath = self.source / "checkpoints/screening.sqlite"
        with sqlite3.connect(self.dbpath) as db:
            db.execute("CREATE TABLE assessments (cache_key TEXT PRIMARY KEY, record_id TEXT, stage TEXT, "
                       "status TEXT, result_json TEXT, raw_response TEXT, error TEXT, created_at TEXT)")

    def add_assessment(self, record=None, model=None, model_digest="original-installed-digest", status="assessed"):
        record = record or self.record
        model = model or self.config["model"]
        names = screening._criterion_names("title_abstract", self.config["protocol"])
        evidence = {"stage": "title_abstract", "required_criteria": names, "record_id": record["record_id"],
                    "metadata": {field: record.get(field, "") for field in
                                 ("title", "authors", "date", "venue", "publication_type", "doi", "source", "sources")},
                    "all_source_text_provided": True, "passages": screening._passages(record, "title_abstract")}
        schema = screening._schema("title_abstract", names)
        system = screening.SYSTEM_PROMPT + "\nTrusted review protocol and required criteria:\n" + screening._json(
            {"protocol": self.config["protocol"], "required_criteria": names})
        data = {"decision": "uncertain", "study_kind": "primary", "rationale": "Synthetic unresolved method.",
                "criteria": [{"criterion": name, "value": "unclear" if name == "operational_assessment" else "yes",
                              "source_attribution": "own_study", "explanation": "Synthetic evidence.",
                              "passage_id": "ABSTRACT", "quote": QUOTE} for name in names]}
        raw = {"done": True, "done_reason": "stop", "prompt_eval_count": 100,
               "message": {"role": "assistant", "content": json.dumps(data)}}
        result = screening._validated_response(raw, evidence, "title_abstract", names, self.config["protocol"],
                                               model["context_tokens"], model["max_output_tokens"])
        result.update(attempt_id="original-attempt", assessed_utc="2020-01-01T00:00:00Z")
        if status != "assessed":
            result["screening_status"] = status
        key = digest({"prompt_version": screening.PROMPT_VERSION, "system": system, "evidence": evidence,
                      "schema": schema, "model_settings": model, "model_digest": model_digest})
        audit = {"record_id": record["record_id"], "stage": "title_abstract", "assessment_fingerprint": key,
                 "prompt_sha256": digest({"system": system, "user": screening._json(evidence), "schema": schema}),
                 "model": model["name"], "model_digest": model_digest, "prompt_version": screening.PROMPT_VERSION,
                 "input_evidence": evidence, "system_prompt": system, "raw_response": raw, "result": result,
                 "attempt_id": result["attempt_id"], "created_at": result["assessed_utc"], "error": ""}
        # Deliberate whitespace proves the ledger also hashes the exact stored JSON.
        payload = json.dumps(result, indent=3)
        with sqlite3.connect(self.dbpath) as db:
            db.execute("INSERT INTO assessments VALUES (?,?,?,?,?,?,?,?)",
                       (key, record["record_id"], "title_abstract", status, payload, json.dumps(raw), "", audit["created_at"]))
        path = self.source / "checkpoints/model_audit" / (key + ".json")
        write_json(path, audit)
        return key, result, audit, path, payload

    def retain(self, records=None):
        return local_retention.retain_title_abstract(self.source, [self.record] if records is None else records)

    def snapshot(self):
        return {str(path.relative_to(self.source)): path.read_bytes()
                for path in self.source.rglob("*") if path.is_file()}

    def test_uncertain_preserves_every_result_field_original_identity_and_timestamp(self):
        key, result, audit, _, payload = self.add_assessment()
        supplied = {**self.record, "model": "WRONG:new-fast-model", "model_digest": "wrong-new-digest"}
        before = self.snapshot()
        retained = self.retain([supplied])
        self.assertEqual(retained["checkpoint_rows"], 1)
        self.assertEqual(retained["historical_rows"], 0)
        row = retained["rows"][0]
        for field, value in result.items():
            self.assertEqual(row[field], value, field)
        self.assertEqual(row["ai_decision"], "uncertain")
        self.assertEqual(row["model"], self.config["model"]["name"])
        self.assertEqual(row["model_digest"], audit["model_digest"])
        self.assertEqual(row["assessed_utc"], "2020-01-01T00:00:00Z")
        self.assertEqual(row["retained_from_run"], str(self.source))
        self.assertEqual(row["cache_key"], key)
        self.assertEqual(row["input_sha256"], screening.screening_input_hash(self.record, "title_abstract"))
        ledger = retained["ledger"][0]
        self.assertEqual(ledger["source_cache_key"], key)
        self.assertEqual(ledger["result_payload_digest"], digest(result))
        self.assertEqual(ledger["result_json_sha256"], local_retention.hashlib.sha256(payload.encode()).hexdigest())
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(supplied["model"], "WRONG:new-fast-model")

    def test_failed_old_model_old_settings_and_outside_cohort_have_explicit_accounting(self):
        self.add_assessment()
        self.add_assessment(model={**self.config["model"], "name": "synthetic:historical-8b"})
        self.add_assessment(model={**self.config["model"], "context_tokens": 65536})
        self.add_assessment(model_digest="failed-digest", status="model_or_validation_error")
        self.add_assessment(record={**self.record, "record_id": "R_OUTSIDE"})
        retained = self.retain()
        self.assertEqual(len(retained["rows"]), 1)
        self.assertEqual(retained["checkpoint_rows"], 5)
        self.assertEqual(retained["historical_rows"], 4)
        self.assertEqual({row["reason"] for row in retained["history"]},
                         {"different_model", "different_generation_identity", "not_assessed", "outside_cohort"})

    def test_changed_logical_input_is_explicitly_incompatible(self):
        self.add_assessment()
        retained = self.retain([{**self.record, "abstract": "Changed supplied abstract."}])
        self.assertEqual(retained["rows"], [])
        self.assertEqual(retained["historical_rows"], 1)
        self.assertEqual(retained["history"][0]["reason"], "different_input")

    def test_tampered_audit_cannot_silently_regenerate_a_completed_reply(self):
        _, _, audit, path, _ = self.add_assessment()
        audit["result"]["ai_decision"] = "include"
        write_json(path, audit)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "differs from its checkpoint"):
            self.retain()
        self.assertEqual(self.snapshot(), before)

    def test_tampered_prompt_hash_is_refused(self):
        _, _, audit, path, _ = self.add_assessment()
        audit["prompt_sha256"] = "f" * 64
        write_json(path, audit)
        with self.assertRaisesRegex(ValueError, "prompt/schema"):
            self.retain()

    def test_mutated_scientific_result_and_matching_audit_still_fail_revalidation(self):
        key, result, audit, path, _ = self.add_assessment()
        result["ai_rationale"] = "Unsupported edited conclusion."
        audit["result"] = result
        write_json(path, audit)
        with sqlite3.connect(self.dbpath) as db:
            db.execute("UPDATE assessments SET result_json=? WHERE cache_key=?", (json.dumps(result), key))
        with self.assertRaisesRegex(ValueError, "scientific result"):
            self.retain()

    def test_two_original_installed_digests_are_ambiguous_rather_than_latest_wins(self):
        self.add_assessment(model_digest="first-original-installed-digest")
        self.add_assessment(model_digest="second-original-installed-digest")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.retain()

    def test_current_core_hash_or_frozen_configuration_mismatch_is_refused(self):
        self.manifest["source_hashes"]["slr_pipeline/screening.py"] = "f" * 64
        write_json(self.source / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "core hashes"):
            self.retain()

    def test_duplicate_input_ids_are_refused(self):
        with self.assertRaisesRegex(ValueError, "unique nonempty"):
            self.retain([self.record, copy.deepcopy(self.record)])


if __name__ == "__main__":
    unittest.main()
