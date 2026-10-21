"""Deterministic readiness checks using synthetic configuration and no network.

These tests never inspect the user's .env or keys, start a pipeline, or query a
provider/model. Dependency versions are mocked to isolate readiness behaviour.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class ReadinessCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Imports are inert. No provider client or model client is constructed.
        import requests
        from slr_pipeline import checks, provider_library, retrieval
        cls.checks = checks
        cls.providers = provider_library
        cls.retrieval = retrieval

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recsys-readiness-check-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        package = self.root / "slr_pipeline"
        package.mkdir()
        self.source = package / "fixture.py"
        self.source.write_text("# Synthetic implementation identity, never executed.\n", encoding="utf-8")
        self.config = {
            "_project_dir": str(self.root),
            "_config_path": str(self.root / "config.json"),
            "run_directory": "runs/synthetic",
            "search": {
                "enabled": True, "require_all_sources": True,
                "email": "fixture@example.org",
                "sources": ["scopus", "wos", "ieee", "openalex", "arxiv", "crossref"],
                "topics": ["quality", "missingness", "noise", "distribution", "selection", "preprocessing", "time", "iso"],
                "start": "1900-01-01", "end": "2026-10-21",
                "max_pages": 0, "crossref_rows": 1000,
                "scopus_pagination": "cursor", "scopus_view": "STANDARD",
            },
            "imports": [],
            "deduplication": {"similarity": 95, "block_limit": 200, "skip_fuzzy": False},
            "model": {
                "name": "synthetic-readiness-fixture", "base_url": "http://127.0.0.1:11434",
                "allow_remote": False, "context_tokens": 32768,
                "max_output_tokens": 8192, "timeout_seconds": 300,
            },
            "screening": {"limit": 0, "retry_errors": True, "require_abstract": True},
            "fulltext": {"download_open_access": False},
            "protocol": {"version": "synthetic-v1", "allowed_languages": []},
        }
        self.credentials = {
            "SCOPUS_API_KEY": "synthetic-scopus-value",
            "WOS_API_KEY": "synthetic-wos-value",
            "IEEE_API_KEY": "synthetic-ieee-value",
            "OPENALEX_API_KEY": "synthetic-openalex-value",
        }
        versions = {"requests": "2.32.5", "rapidfuzz": "3.10.1",
                    "bibtexparser": "1.4.3", "pypdf": "6.0.0"}
        self.patches = [
            mock.patch.object(self.checks.importlib.metadata, "version", side_effect=versions.__getitem__),
            mock.patch.object(self.checks.importlib.util, "find_spec", return_value=object()),
            mock.patch.object(self.checks.shutil, "which", return_value="/synthetic/bin/ollama"),
            mock.patch("socket.create_connection", side_effect=AssertionError("No network in readiness checks")),
            mock.patch("socket.getaddrinfo", side_effect=AssertionError("No DNS in readiness checks")),
            mock.patch("socket.socket.connect", side_effect=AssertionError("No sockets in readiness checks")),
            mock.patch("socket.socket.connect_ex", side_effect=AssertionError("No sockets in readiness checks")),
            mock.patch("requests.sessions.Session.request", side_effect=AssertionError("No HTTP in readiness checks")),
            mock.patch.object(self.providers, "Client", side_effect=AssertionError("No provider client in readiness checks")),
            mock.patch.object(self.retrieval, "collect", side_effect=AssertionError("Do not run the retrieval pipeline")),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_remote_opt_in_must_be_a_real_boolean(self):
        self.config["model"].update(base_url="https://example.org", allow_remote="false")
        report = self.check()
        self.assertFalse(report["ready"])
        self.assertIn("model.allow_remote must be a JSON boolean", "\n".join(report["errors"]))
        from slr_pipeline.screening import OllamaClient
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            OllamaClient(self.config["model"])

    def test_malformed_manifest_is_an_error_report_without_writes(self):
        run = self.root / self.config["run_directory"]
        run.mkdir(parents=True)
        path = run / "manifest.json"
        path.write_text("not valid JSON", encoding="utf-8")
        report = self.check()
        self.assertFalse(report["ready"])
        self.assertIn("malformed manifest", "\n".join(report["errors"]))
        self.assertEqual(path.read_text(), "not valid JSON")

    def test_source_completeness_option_must_be_boolean(self):
        self.config["search"]["require_all_sources"] = "false"
        report = self.check()
        self.assertFalse(report["ready"])
        self.assertIn("require_all_sources must be true or false", "\n".join(report["errors"]))

    def check(self, config=None, environment=None, stage="all"):
        return self.checks.check_configuration(
            self.config if config is None else config,
            environ=self.credentials if environment is None else environment,
            stage=stage,
        )

    def assert_no_values_leaked(self, report, values=None):
        serialized = json.dumps(report)
        for value in values or self.credentials.values():
            self.assertNotIn(value, serialized)

    def test_all_six_sources_report_query_counts_and_key_presence_without_access_claim(self):
        report = self.check()
        self.assertTrue(report["ready"], report["errors"])
        self.assertFalse(report["access_verified"])
        self.assertEqual((report["database_requests"], report["model_requests"], report["writes"]), (0, 0, 0))
        sources = {item["source"]: item for item in report["sources"]}
        self.assertEqual(set(sources), {"scopus", "wos", "ieee", "openalex", "arxiv", "crossref"})
        # Eight Boolean topic jobs per index; 48 discovery terms x four anchors.
        self.assertEqual({source: item["query_count"] for source, item in sources.items()},
                         {"scopus": 8, "wos": 8, "ieee": 8, "openalex": 8, "arxiv": 8, "crossref": 192})
        for source, variable in (("scopus", "SCOPUS_API_KEY"), ("wos", "WOS_API_KEY"),
                                 ("ieee", "IEEE_API_KEY"), ("openalex", "OPENALEX_API_KEY")):
            self.assertEqual(sources[source]["credential_variable"], variable)
            self.assertIs(sources[source]["credential_present"], True)
        self.assertIsNone(sources["crossref"]["credential_variable"])
        self.assertIsNone(sources["arxiv"]["credential_present"])
        self.assertTrue(all(item["live_access"] == "not_checked_offline" for item in sources.values()))
        self.assert_no_values_leaked(report)

    def test_missing_email_and_every_selected_database_key_are_listed_together(self):
        self.config["search"]["email"] = ""
        report = self.check(environment={})
        self.assertFalse(report["ready"])
        errors = "\n".join(report["errors"])
        self.assertIn("valid contact email", errors)
        for variable in self.credentials:
            self.assertIn(variable, errors)
        self.assertEqual(len(report["sources"]), 6)
        self.assertTrue(all(item["credential_present"] is False
                            for item in report["sources"] if item["credential_variable"]))

    def test_imports_only_needs_neither_email_nor_provider_credentials(self):
        self.config["search"].update(enabled=False, email="", sources=[], topics=[])
        source = self.root / "synthetic.csv"
        source.write_text("title,abstract\nSynthetic fixture,Synthetic evidence\n", encoding="utf-8")
        self.config["imports"] = [{"source": "manual-fixture", "path": source.name}]
        report = self.check(environment={})
        self.assertTrue(report["ready"], report["errors"])
        self.assertEqual(report["sources"], [])
        self.assertEqual(report["database_requests"], 0)
        self.assertFalse((self.root / "runs").exists())

    def test_non_strict_missing_keys_warn_without_failing_when_email_is_valid(self):
        self.config["search"]["require_all_sources"] = False
        report = self.check(environment={})
        self.assertTrue(report["ready"], report["errors"])
        self.assertEqual(report["errors"], [])
        warnings = "\n".join(report["warnings"])
        for variable in self.credentials:
            self.assertIn(variable, warnings)

    def test_environment_precedes_dotenv_and_neither_value_appears_in_report(self):
        file_values = {key: "synthetic-file-" + str(index) for index, key in enumerate(self.credentials)}
        (self.root / ".env").write_text(
            "\n".join(f'{key}="{value}"' for key, value in file_values.items()) + "\n",
            encoding="utf-8",
        )
        report = self.check(environment={"SCOPUS_API_KEY": ""})
        sources = {item["source"]: item for item in report["sources"]}
        self.assertFalse(report["ready"])
        # An explicitly empty process variable wins over the nonempty file value,
        # matching the runtime's setdefault behaviour without exposing either.
        self.assertIs(sources["scopus"]["credential_present"], False)
        self.assertIs(sources["ieee"]["credential_present"], True)
        self.assert_no_values_leaked(report, file_values.values())
        ready = self.check(environment={"SCOPUS_API_KEY": "synthetic-env-preferred"})
        self.assertTrue(ready["ready"], ready["errors"])
        self.assert_no_values_leaked(ready, [*file_values.values(), "synthetic-env-preferred"])

    def test_duplicate_and_invalid_dotenv_errors_do_not_leak_names_or_values(self):
        cases = [
            ("IEEE_API_KEY=synthetic-first\nIEEE_API_KEY=synthetic-second\n", "Duplicate .env variable"),
            ("UNSUPPORTED_SYNTHETIC_NAME=synthetic-secret-value\n", "Unsupported .env variable"),
            ("synthetic-invalid-secret-no-equals\n", "Invalid .env line"),
        ]
        for content, error in cases:
            with self.subTest(error=error):
                (self.root / ".env").write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error) as caught:
                    self.checks.env_values(self.root / ".env")
                self.assertNotIn("synthetic-", str(caught.exception))
                self.assertNotIn("UNSUPPORTED_SYNTHETIC_NAME", str(caught.exception))
                report = self.check()
                self.assertFalse(report["ready"])
                self.assertIn(error, "\n".join(report["errors"]))
                self.assertNotIn(content.strip(), json.dumps(report))

    def test_dotenv_parser_treats_shell_syntax_as_literal_data(self):
        literal = "$(synthetic-command-not-to-execute)`synthetic-literal`"
        path = self.root / ".env"
        path.write_text(f"# Fixture comment\nSCOPUS_API_KEY='{literal}'\n", encoding="utf-8")
        self.assertEqual(self.checks.env_values(path), {"SCOPUS_API_KEY": literal})
        self.assert_no_values_leaked(self.check(), [literal])

    def test_check_does_not_mutate_environment_configuration_or_create_a_run(self):
        (self.root / ".env").write_text("IEEE_API_KEY=synthetic-file-key\n", encoding="utf-8")
        before_config = copy.deepcopy(self.config)
        before_files = sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*"))
        with mock.patch.dict(os.environ, {"SYNTHETIC_UNRELATED": "preserve-this"}, clear=True):
            before_environment = dict(os.environ)
            report = self.checks.check_configuration(self.config, stage="check")
            self.assertEqual(dict(os.environ), before_environment)
        self.assertEqual(self.config, before_config)
        self.assertEqual(sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*")), before_files)
        self.assertFalse((self.root / "runs").exists())
        self.assertEqual(report["writes"], 0)
        self.assertEqual(report["checked_for_stage"], "check")

    def test_remote_model_is_refused_without_explicit_remote_setting(self):
        self.config["model"]["base_url"] = "https://synthetic.example.org/model"
        report = self.check()
        self.assertFalse(report["ready"])
        self.assertIn("Remote model endpoint is disabled.", report["errors"])
        self.assertEqual(report["model_requests"], 0)

    def test_incompatible_manifest_is_detected_without_rewriting_it(self):
        run = self.root / "runs" / "synthetic"
        run.mkdir(parents=True)
        manifest = run / "manifest.json"
        content = '{"fingerprint": "different-synthetic-frozen-identity"}\n'
        manifest.write_text(content, encoding="utf-8")
        report = self.check()
        self.assertFalse(report["ready"])
        self.assertTrue(any("different frozen code/configuration" in error for error in report["errors"]))
        self.assertEqual(manifest.read_text(encoding="utf-8"), content)
        self.assertEqual(list(run.iterdir()), [manifest])

    def test_compatible_manifest_is_accepted_without_live_verification(self):
        _, _, fingerprint = self.checks.configuration_fingerprint(self.config)
        run = self.root / "runs" / "synthetic"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
        report = self.check()
        self.assertTrue(report["ready"], report["errors"])
        self.assertFalse(report["access_verified"])

    def test_fingerprint_ignores_only_operational_limits_and_retries(self):
        before = copy.deepcopy(self.config)
        snapshot, hashes, fingerprint = self.checks.configuration_fingerprint(self.config)
        self.assertEqual(self.config, before)
        self.assertNotIn("_project_dir", snapshot)
        self.assertEqual(set(hashes), {"slr_pipeline/fixture.py"})
        changed = copy.deepcopy(self.config)
        changed["screening"].update(limit=17, retry_errors=False)
        changed["search"]["max_pages"] = 3
        self.assertEqual(self.checks.configuration_fingerprint(changed)[2], fingerprint)
        for section, key, value in (("screening", "require_abstract", False),
                                    ("model", "context_tokens", 65536),
                                    ("protocol", "version", "synthetic-v2"),
                                    ("search", "end", "2026-10-20"),
                                    ("search", "require_all_sources", False)):
            with self.subTest(section=section, key=key):
                modified = copy.deepcopy(self.config)
                modified[section][key] = value
                self.assertNotEqual(self.checks.configuration_fingerprint(modified)[2], fingerprint)
        self.source.write_text("# Changed synthetic implementation identity.\n", encoding="utf-8")
        self.assertNotEqual(self.checks.configuration_fingerprint(self.config)[2], fingerprint)


if __name__ == "__main__":
    unittest.main()
