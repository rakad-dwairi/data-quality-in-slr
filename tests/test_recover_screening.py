"""Screening recovery checks using temporary synthetic evidence and checkpoints.

No real review run/configuration, provider, download or model endpoint is used.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock


QUOTE = "The synthetic target study measures required item metadata in a recommender dataset."


class SyntheticModel:
    ready = True
    unavailable_error = ""
    model_digest = "synthetic_screening_recovery_digest"
    model = "synthetic:donor-only"
    context = 32768
    output = 4096

    def __init__(self, forbid=False):
        self.forbid = forbid
        self.calls = 0

    def call(self, messages, schema):
        if self.forbid:
            raise AssertionError("A successful copied checkpoint must avoid inference")
        self.calls += 1
        evidence = json.loads(messages[-1]["content"])
        data = {"decision": "include", "study_kind": "primary", "rationale": "Synthetic fixture only.",
                "criteria": [{"criterion": name, "value": "yes", "source_attribution": "own_study",
                              "explanation": "Synthetic explicit target-study evidence.",
                              "passage_id": "ABSTRACT", "quote": QUOTE}
                             for name in evidence["required_criteria"]]}
        return {"done": True, "done_reason": "stop", "prompt_eval_count": 100,
                "message": {"content": json.dumps(data)}}


class ScreeningRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        denied = AssertionError("Network/provider/model endpoints are forbidden in recovery tests")
        for name in ("requests.sessions.Session.request", "socket.create_connection", "socket.getaddrinfo",
                     "socket.socket.connect", "socket.socket.connect_ex", "http.client.HTTPConnection.request"):
            self.stack.enter_context(mock.patch(name, side_effect=denied))
        from slr_pipeline import checks, cli, common, fulltext, recover, retrieval, screening
        self.cli, self.common, self.recover, self.screening = cli, common, recover, screening
        self.checks, self.retrieval = checks, retrieval
        folder = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="slr-screening-recovery-fixture-"))
        self.root = Path(folder).resolve()
        self.source = self.root / "runs/windows_donor"
        self.target = self.root / "runs/recovered"
        self.config_out = self.root / "config/recovery.json"
        self.stack.enter_context(mock.patch.object(cli, "PROJECT", self.root))
        self.stack.enter_context(mock.patch.object(retrieval.providers, "Client", side_effect=denied))
        self.stack.enter_context(mock.patch.object(fulltext, "_download", side_effect=denied))
        self.stack.enter_context(mock.patch.object(screening.OllamaClient, "preflight", side_effect=denied))
        self.stack.enter_context(mock.patch.object(screening.OllamaClient, "call", side_effect=denied))
        versions = {"requests": "2.34.2", "rapidfuzz": "3.14.6", "bibtexparser": "1.4.3", "pypdf": "6.21.2"}
        actual_version = checks.importlib.metadata.version
        self.stack.enter_context(mock.patch.object(checks.importlib.metadata, "version",
            side_effect=lambda name: versions[name] if name in versions else actual_version(name)))
        self._make_project_and_donor(Path(recover.__file__).resolve().parent)

    def _make_project_and_donor(self, actual_library):
        library = self.root / "slr_pipeline"
        library.mkdir()
        for path in actual_library.glob("*.py"):
            shutil.copyfile(path, library / path.name)
        self.source.mkdir(parents=True)
        with self.common.run_lock(self.source):
            pass
        self.config = {
            "run_directory": "runs/windows_donor",
            "search": {"enabled": True, "email": "fixture@example.invalid", "sources": ["crossref"],
                       "topics": ["quality"], "start": "1900-01-01", "end": "2026-10-21",
                       "max_pages": 0, "crossref_rows": 1000, "require_all_sources": True},
            "imports": [], "deduplication": {"similarity": 95, "block_limit": 200},
            "model": {"name": "synthetic:donor-only", "base_url": "http://127.0.0.1:11434",
                      "allow_remote": False, "context_tokens": 32768, "max_output_tokens": 4096,
                      "timeout_seconds": 10},
            "screening": {"limit": 0, "retry_errors": False, "require_abstract": True},
            "fulltext": {"directory": "inputs/fulltext", "download_open_access": False},
            "reviews_file": "inputs/reviews.csv",
            "protocol": {"version": "SYNTHETIC_DONOR_PROTOCOL", "scope": "Synthetic fixture only.",
                         "source_plan": "synthetic_plan.txt", "iso_codebook": {}},
        }
        (self.root / "synthetic_plan.txt").write_text("Synthetic plan, never a manuscript.", encoding="utf-8")
        # A different local config must never replace the donor's frozen settings.
        self.common.write_json(self.root / "config/review.json", {"model": {"name": "WRONG_LOCAL_CONFIG"}})
        hashes = {}
        for path in library.glob("*.py"):
            if path.name in {"recover.py", "evidence.py"}:
                continue  # Simulate the old run before these new modules existed.
            content = path.read_bytes()
            if path.name in {"screening.py", "cli.py"}:
                content += b"\n# Synthetic original implementation snapshot.\n"
            snapshot = self.source / "code_snapshot" / path.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(content)
            hashes["slr_pipeline/" + path.name] = self.common.file_hash(snapshot)
        identity = copy.deepcopy(self.config)
        identity["screening"].pop("limit", None)
        identity["screening"].pop("retry_errors", None)
        identity["search"].pop("max_pages", None)
        self.common.write_json(self.source / "manifest.json", {
            "version": "1.0.0", "config": self.config, "source_hashes": hashes,
            "fingerprint": self.common.digest({"config": identity, "source_hashes": hashes}),
        })
        self.common.write_json(self.source / "protocol_snapshot.json", {"protocol": self.config["protocol"]})
        self.observation = {"observation_id": "O_SYNTHETIC_RECOVERY", "source": "crossref",
            "source_id": "10.9999/synthetic-screening-recovery", "title": "Synthetic recovery fixture",
            "abstract": QUOTE, "authors": "Alpha Example", "date": "2001-01-01",
            "doi": "10.9999/synthetic-screening-recovery", "publication_type": "journal-article",
            "venue": "Synthetic Fixture Venue", "raw_ref": "01_retrieval/raw/synthetic.json#record=1"}
        self.report = {**self.observation, "record_id": "R_SYNTHETIC_RECOVERY",
                       "report_id": "R_SYNTHETIC_RECOVERY", "study_id": "", "dedup_group_size": 1}
        self.common.write_json(self.source / "01_retrieval/raw/synthetic.json", {"synthetic": True})
        self.common.write_csv(self.source / "01_retrieval/records.csv", [self.observation])
        self.common.write_csv(self.source / "02_deduplication/records.csv", [self.report])
        self.common.write_csv(self.source / "02_deduplication/record_mapping.csv", [{"observation_id": self.observation["observation_id"],
                                                                                   "record_id": self.report["record_id"]}])
        state = {}
        for stage, artifact, rows in (
            ("retrieve", "01_retrieval/records.csv", [self.observation]),
            ("deduplicate", "02_deduplication/records.csv", [self.report]),
        ):
            state[stage] = {"valid": True, "rows": len(rows), "digest": self.common.digest(rows),
                            "artifact": artifact, "artifact_sha256": self.common.file_hash(self.source / artifact)}
        state.update(screen={"valid": True}, fulltext={"valid": True})
        self.common.write_json(self.source / "stage_state.json", state)
        self.common.write_json(self.source / "03_title_abstract/assessments.json", [{"ai_decision": "exclude", "synthetic": True}])
        self.common.write_csv(self.source / "04_full_text/screening.csv", [{"ai_decision": "exclude", "synthetic": True}])
        pdf = self.source / "04_full_text/pdfs/synthetic.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF-1.4\nSynthetic cached PDF bytes: never parsed as a real report.\n")
        self.common.write_json(self.source / "checkpoints/fulltext/synthetic.json", {"text": "Synthetic extracted cache only."})
        self.db = self.screening._open_checkpoint(self.source / "checkpoints/screening.sqlite")
        self.addCleanup(self.db.close)
        self.names = self.screening._criterion_names("title_abstract", self.config["protocol"])
        self.evidence = {"stage": "title_abstract", "record_id": self.report["record_id"],
                         "required_criteria": self.names, "passages": {"ABSTRACT": QUOTE}}
        self.system = "Synthetic trusted screening fixture system."
        result = self.screening._assess_input(self.report, "title_abstract", self.config, self.source,
            SyntheticModel(), self.db, self.evidence, self.system, self.names,
            self.screening._schema("title_abstract", self.names), {"limit": 0, "attempted": 0})
        self.assertEqual(result["screening_status"], "assessed")
        self.cached_key = result["cache_key"]
        self.db.execute("INSERT INTO assessments VALUES (?,?,?,?,?,?,?,?)", (
            "synthetic_failed_checkpoint", self.report["record_id"], "title_abstract", "model_or_validation_error",
            "{}", "{}", "Synthetic previous error", "2026-10-21T12:00:00Z"))
        self.db.commit()

    def snapshot(self, folder):
        return {str(path.relative_to(folder)): path.read_bytes()
                for path in folder.rglob("*") if path.is_file()}

    def prepare(self, limit=3):
        with mock.patch.object(self.cli, "load_env", side_effect=AssertionError("Recovery must not load credentials")), \
                mock.patch("slr_pipeline.checks.env_values", side_effect=AssertionError("Recovery must not read .env")), \
                mock.patch("os.getenv", side_effect=AssertionError("Recovery must not read API keys")):
            return self.recover.prepare_recovery("runs/windows_donor", "runs/recovered", "config/recovery.json", limit, self.root)

    def assert_refused(self, message):
        before = self.snapshot(self.source)
        with self.assertRaisesRegex(ValueError, message):
            self.prepare()
        self.assertFalse(self.target.exists())
        self.assertFalse(self.config_out.exists())
        self.assertEqual(self.snapshot(self.source), before)
        self.assertEqual(list((self.root / "runs").glob(".screening-recovery-*")), [])

    def test_recovery_derives_donor_config_and_copies_evidence_without_labels_or_redownload(self):
        before = self.snapshot(self.source)
        result = self.prepare()
        self.assertEqual(self.snapshot(self.source), before)
        recovered = self.common.read_json(self.config_out)
        for key, value in self.config.items():
            if key not in {"run_directory", "screening"}:
                self.assertEqual(recovered[key], value)
        self.assertEqual(recovered["run_directory"], "runs/recovered")
        self.assertEqual(recovered["screening"], {**self.config["screening"], "limit": 3, "retry_errors": True})
        for relative, data in before.items():
            if relative.startswith(("01_retrieval/", "02_deduplication/", "04_full_text/pdfs/")):
                self.assertEqual((self.target / relative).read_bytes(), data)
        self.assertFalse((self.target / "03_title_abstract").exists())
        self.assertFalse((self.target / "04_full_text/screening.csv").exists())
        self.assertEqual((self.target / "recovery_origin/manifest.json").read_bytes(), before["manifest.json"])
        self.assertEqual((self.target / "recovery_origin/stage_state.json").read_bytes(), before["stage_state.json"])
        state = self.common.read_json(self.target / "stage_state.json")
        for name in ("retrieve", "deduplicate"):
            self.assertTrue(self.cli.validate_stage(self.target, name))
            self.assertIn("Source-attested", result["verified_stages"][name]["digest_verification"])
        self.assertFalse(state["screen"]["valid"])
        self.assertFalse(state["fulltext"]["valid"])
        self.assertEqual(result["database_requests"], 0)
        self.assertEqual(result["model_requests"], 0)
        self.assertEqual(result["pipeline_stages_executed"], 0)
        self.assertFalse(any(path.name.endswith(("-wal", "-shm")) for path in self.target.rglob("*")))
        self.assertEqual(result["checkpoint_databases"][0]["tables"]["assessments"], 2)
        for item in result["files"]:
            self.assertEqual(self.common.file_hash(self.target / item["path"]), item["sha256"])

    def test_current_fingerprint_readiness_and_cli_export_accept_the_prepared_run(self):
        self.prepare()
        loaded = self.cli.load_config(self.config_out)
        check = self.checks.check_configuration(loaded, environ={})
        self.assertTrue(check["ready"], check["errors"])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.cli.main(["--config", str(self.config_out), "--stage", "export"]), 0)
        self.assertEqual(self.common.read_csv(self.target / "06_final/final_included_papers.csv"), [])

    def test_legacy_windows_lf_snapshots_verify_against_crlf_source_hashes(self):
        # Simulate Git's Windows checkout bytes and the old universal-newline
        # snapshot writer without executing or altering any real Windows run.
        for path in (self.root / "slr_pipeline").glob("*.py"):
            content = path.read_bytes().replace(b"\r\n", b"\n")
            path.write_bytes(content.replace(b"\n", b"\r\n"))
        manifest = self.common.read_json(self.source / "manifest.json")
        windows_hashes = {}
        for original in manifest["source_hashes"]:
            name = original.replace("\\", "/").split("/")[-1]
            frozen = (self.source / "code_snapshot" / name).read_bytes()
            self.assertNotIn(b"\r", frozen)
            windows_hashes[original.replace("/", "\\")] = hashlib.sha256(
                frozen.replace(b"\n", b"\r\n")).hexdigest()
        manifest["source_hashes"] = windows_hashes
        identity = copy.deepcopy(manifest["config"])
        identity["screening"].pop("limit", None)
        identity["screening"].pop("retry_errors", None)
        identity["search"].pop("max_pages", None)
        manifest["fingerprint"] = self.common.digest({"config": identity, "source_hashes": windows_hashes})
        self.common.write_json(self.source / "manifest.json", manifest)
        before = self.snapshot(self.source)
        report = self.prepare()
        self.assertEqual(self.snapshot(self.source), before)
        self.assertEqual(len(report["legacy_newline_verification"]), len(windows_hashes))
        for item in report["legacy_newline_verification"]:
            self.assertEqual(item["method"], "legacy_lf_snapshot_verified_by_exact_uniform_crlf_reconstruction")
            frozen = (self.target / "recovery_origin" / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(frozen.replace(b"\n", b"\r\n")).hexdigest(),
                             item["manifest_source_sha256"])
        current = self.common.read_json(self.target / "manifest.json")
        for name, checksum in current["source_hashes"].items():
            snapshot = self.target / "code_snapshot" / name.replace("\\", "/").split("/")[-1]
            self.assertIn(b"\r\n", snapshot.read_bytes())
            self.assertEqual(self.common.file_hash(snapshot), checksum)

    def test_copied_successful_sqlite_checkpoint_is_reused_without_inference(self):
        self.prepare()
        config = self.cli.load_config(self.config_out)
        copied = self.screening._open_checkpoint(self.target / "checkpoints/screening.sqlite")
        try:
            self.assertEqual(copied.execute("SELECT COUNT(*) FROM assessments").fetchone()[0], 2)
            self.assertEqual(copied.execute("SELECT status FROM assessments WHERE cache_key=?", (self.cached_key,)).fetchone()[0], "assessed")
            model, budget = SyntheticModel(forbid=True), {"limit": 3, "attempted": 0}
            result = self.screening._assess_input(self.report, "title_abstract", config, self.target,
                model, copied, self.evidence, self.system, self.names,
                self.screening._schema("title_abstract", self.names), budget)
            self.assertTrue(result["cache_reused"])
            self.assertEqual(model.calls, 0)
            self.assertEqual(budget["attempted"], 0)
        finally:
            copied.close()

    def test_chained_recovery_retains_evidence_but_changed_prompt_never_reuses_old_labels(self):
        self.prepare()
        # A calibration starts from the stopped recovery, with a newly frozen
        # implementation. Only this temporary synthetic project is changed.
        code = self.root / "slr_pipeline/screening.py"
        code.write_bytes(code.read_bytes() + b"\n# Synthetic changed screening prompt implementation.\n")
        with self.common.run_lock(self.target):
            pass  # Simulate the completed owner's screening trial.
        before = self.snapshot(self.target)
        calibration = self.root / "runs/calibration"
        config_path = self.root / "config/calibration.json"
        report = self.recover.prepare_recovery("runs/recovered", "runs/calibration",
                                              "config/calibration.json", 8, self.root)
        self.assertEqual(self.snapshot(self.target), before)
        config = self.cli.load_config(config_path)
        self.assertEqual(config["model"], self.config["model"])
        self.assertEqual(config["protocol"], self.config["protocol"])
        self.assertEqual(config["screening"]["limit"], 8)
        self.assertTrue(config["screening"]["retry_errors"])
        self.assertEqual(report["allowed_code_changes"], ["screening.py"])
        self.assertEqual(report["database_requests"], 0)
        self.assertEqual(report["model_requests"], 0)
        for relative, data in before.items():
            if relative.startswith(("01_retrieval/", "02_deduplication/", "04_full_text/pdfs/",
                                    "checkpoints/model_audit/", "checkpoints/model_attempts/",
                                    "checkpoints/fulltext/")):
                self.assertEqual((calibration / relative).read_bytes(), data)
        self.assertFalse((calibration / "03_title_abstract").exists())
        self.assertFalse(self.common.read_json(calibration / "stage_state.json")["screen"]["valid"])
        self.assertNotEqual(self.common.read_json(calibration / "manifest.json")["fingerprint"],
                            self.common.read_json(self.target / "manifest.json")["fingerprint"])

        copied = self.screening._open_checkpoint(calibration / "checkpoints/screening.sqlite")
        try:
            old = copied.execute("SELECT * FROM assessments WHERE cache_key=?", (self.cached_key,)).fetchone()
            self.assertIsNotNone(old)
            old_version = self.screening.PROMPT_VERSION
            # Either the complete system prompt or its explicit version must
            # independently invalidate the old generation identity.
            for system, version in ((self.system + "\nSynthetic calibrated guidance.", old_version),
                                    (self.system, old_version + ".synthetic-calibration")):
                with self.subTest(prompt_changed=system != self.system, version_changed=version != old_version), \
                        mock.patch.object(self.screening, "PROMPT_VERSION", version):
                    model, budget = SyntheticModel(), {"limit": 8, "attempted": 0}
                    result = self.screening._assess_input(self.report, "title_abstract", config, calibration,
                        model, copied, self.evidence, system, self.names,
                        self.screening._schema("title_abstract", self.names), budget)
                    self.assertFalse(result["cache_reused"])
                    self.assertNotEqual(result["cache_key"], self.cached_key)
                    self.assertEqual(model.calls, 1)
                    self.assertEqual(budget["attempted"], 1)
                    cached = self.screening._assess_input(self.report, "title_abstract", config, calibration,
                        SyntheticModel(forbid=True), copied, self.evidence, system, self.names,
                        self.screening._schema("title_abstract", self.names), budget)
                    self.assertTrue(cached["cache_reused"])
                    self.assertEqual(cached["cache_key"], result["cache_key"])
                    self.assertEqual(budget["attempted"], 1)
            self.assertEqual(copied.execute("SELECT * FROM assessments WHERE cache_key=?",
                                           (self.cached_key,)).fetchone(), old)
            self.assertEqual(copied.execute("SELECT COUNT(*) FROM assessments").fetchone()[0], 4)
        finally:
            copied.close()

    def test_cli_screen_uses_copied_deduplication_without_collecting_or_fulltext(self):
        self.prepare()

        def fake_screen(records, stage, config, run_dir):
            self.assertEqual(stage, "title_abstract")
            self.assertEqual(Path(run_dir), self.target)
            self.assertEqual([row["record_id"] for row in records], [self.report["record_id"]])
            return [{**records[0], "ai_decision": "uncertain", "screening_status": "assessed",
                     "input_sha256": "synthetic_recovery_input_hash"}]

        with mock.patch.object(self.retrieval, "collect", side_effect=AssertionError("No retrieval stage is allowed")), \
                mock.patch.object(self.screening, "screen_records", side_effect=fake_screen) as screen, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.cli.main(["--config", str(self.config_out), "--stage", "screen"]), 0)
            screen.assert_called_once()

    def test_active_donor_refuses_before_creating_outputs(self):
        with self.common.run_lock(self.source):
            self.assert_refused("still active")

    def test_changed_unapproved_implementation_is_refused(self):
        path = self.root / "slr_pipeline/provider_library.py"
        path.write_bytes(path.read_bytes() + b"\n# Synthetic unexpected adapter change.\n")
        self.assert_refused("allows implementation changes only")

    def test_tampered_frozen_snapshot_is_refused(self):
        path = self.source / "code_snapshot/retrieval.py"
        path.write_bytes(path.read_bytes() + b"\n# Synthetic donor tampering.\n")
        self.assert_refused("snapshot is missing or changed")

    def test_tampered_deduplicated_artifact_is_refused(self):
        (self.source / "02_deduplication/records.csv").write_text("Tampered synthetic artifact", encoding="utf-8")
        self.assert_refused("stage artifact changed")

    def test_mismatched_row_count_is_refused(self):
        state = self.common.read_json(self.source / "stage_state.json")
        state["deduplicate"]["rows"] = 2
        self.common.write_json(self.source / "stage_state.json", state)
        self.assert_refused("row count")

    def test_existing_destination_or_output_config_is_never_replaced(self):
        self.target.mkdir()
        sentinel = self.target / "keep.txt"
        sentinel.write_text("Synthetic existing destination", encoding="utf-8")
        before = self.snapshot(self.target)
        with self.assertRaisesRegex(ValueError, "must both be new"):
            self.prepare()
        self.assertEqual(self.snapshot(self.target), before)
        shutil.rmtree(self.target)
        self.config_out.write_text("Synthetic existing config", encoding="utf-8")
        before = self.config_out.read_bytes()
        with self.assertRaisesRegex(ValueError, "must both be new"):
            self.prepare()
        self.assertEqual(self.config_out.read_bytes(), before)
        self.assertFalse(self.target.exists())

    def test_broken_or_missing_completed_stage_is_refused(self):
        state = self.common.read_json(self.source / "stage_state.json")
        state["retrieve"]["valid"] = False
        self.common.write_json(self.source / "stage_state.json", state)
        self.assert_refused("requires completed, valid")


if __name__ == "__main__":
    unittest.main()
