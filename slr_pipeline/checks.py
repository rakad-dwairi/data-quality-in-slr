"""Read-only readiness checks. Never query a provider, model, DNS or socket."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import ipaddress
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from .common import digest, file_hash, project_path, read_json

ENV_NAMES = {"OPENALEX_API_KEY", "SCOPUS_API_KEY", "SCOPUS_INST_TOKEN", "WOS_API_KEY", "IEEE_API_KEY"}


def env_values(path):
    """Parse literal values without mutating the process or evaluating shell code."""
    values = {}
    path = Path(path)
    if not path.exists():
        return values
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line {number}; use literal KEY=value entries.")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in ENV_NAMES:
            raise ValueError(f"Unsupported .env variable on line {number}; use the names in .env.example.")
        if key in values:
            raise ValueError(f"Duplicate .env variable on line {number}; keep one value per name.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def configuration_fingerprint(config):
    snapshot = {k: v for k, v in config.items() if not k.startswith("_")}
    identity = json.loads(json.dumps(snapshot))
    identity.get("screening", {}).pop("limit", None)
    identity.get("screening", {}).pop("retry_errors", None)
    identity.get("search", {}).pop("max_pages", None)
    root = Path(config["_project_dir"])
    hashes = {str(path.relative_to(root)): file_hash(path)
              for path in sorted((root / "slr_pipeline").glob("*.py"))}
    return snapshot, hashes, digest({"config": identity, "source_hashes": hashes})


def check_configuration(config, environ=None, stage="all"):
    """Return public metadata/booleans only. ready never means live access verified."""
    root = Path(config["_project_dir"])
    errors, warnings = [], []
    environment = dict(os.environ if environ is None else environ)
    try:
        for key, value in env_values(root / ".env").items():
            environment.setdefault(key, value)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    dependencies = {}
    for package, module, low, high in (("requests", "requests", (2, 32, 3), (3,)),
                                        ("rapidfuzz", "rapidfuzz", (3, 9), (4,)),
                                        ("bibtexparser", "bibtexparser", (1, 4, 3), (1, 4, 4)),
                                        ("pypdf", "pypdf", (5,), (7,))):
        try:
            version = importlib.metadata.version(package)
            numbers = tuple(int(x) for x in re.match(r"\d+(?:\.\d+)*", version).group().split("."))
            installed = importlib.util.find_spec(module) is not None
            valid = installed and low <= numbers < high
        except (importlib.metadata.PackageNotFoundError, AttributeError, ValueError, ImportError):
            version, valid = None, False
        dependencies[package] = {"version": version, "compatible": valid}
        if not valid:
            errors.append(f"Install compatible dependencies with this interpreter: python -m pip install -r requirements.txt ({package}).")

    search = config.get("search", {})
    enabled = search.get("enabled", False)
    selected = search.get("sources", [])
    sources, queries = [], []
    try:
        from . import provider_library as providers
        from .retrieval import _validate
        try:
            _validate(config)
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
        valid_sources = isinstance(selected, list) and all(source in providers.ENDPOINTS for source in selected)
        topics = search.get("topics", [])
        valid_topics = isinstance(topics, list) and all(topic in providers.TOPICS for topic in topics)
        if enabled and valid_sources and valid_topics:
            queries = providers.jobs_for(list(dict.fromkeys(selected)), list(dict.fromkeys(search.get("topics", []))))
        counts = Counter(job["source"] for job in queries)
        for source in selected if valid_sources else []:
            key = providers.KEYS.get(source)
            present = bool(environment.get(key, "").strip()) if key else None
            if enabled and key and not present:
                message = f"{source}: missing {key} in this environment and project .env."
                if search.get("require_all_sources", True):
                    errors.append(message)
                else:
                    warnings.append(message + " It will be logged as unavailable.")
            sources.append({"source": source, "selected": True, "enabled": bool(enabled),
                            "endpoint": providers.ENDPOINTS[source], "query_count": counts[source],
                            "credential_variable": key, "credential_present": present,
                            "live_access": "not_checked_offline"})
    except (ImportError, ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
        # Report the selected sources/credential presence even when email validation fails.
        keys = {"openalex": "OPENALEX_API_KEY", "scopus": "SCOPUS_API_KEY", "wos": "WOS_API_KEY", "ieee": "IEEE_API_KEY"}
        if not sources and isinstance(selected, list):
            for source in selected:
                key = keys.get(source)
                present = bool(environment.get(key, "").strip()) if key else None
                sources.append({"source": source, "selected": True, "enabled": bool(enabled),
                                "credential_variable": key, "credential_present": present,
                                "live_access": "not_checked_offline"})
                if enabled and key and not present:
                    errors.append(f"{source}: missing {key} in this environment and project .env.")
    imports = config.get("imports", [])
    for spec in imports:
        if not isinstance(spec, dict) or not spec.get("path"):
            errors.append("Every import needs a source and path.")
            continue
        if not project_path(config, spec["path"]).is_file():
            errors.append(f"Import does not exist: {spec['path']}")
    if not enabled and not imports:
        errors.append("Enable database searching or supply at least one import.")
    model = config.get("model", {})
    try:
        if not isinstance(model.get("allow_remote", False), bool):
            raise ValueError("model.allow_remote must be a JSON boolean (true or false).")
        parsed = urlsplit(str(model.get("base_url", "http://127.0.0.1:11434")))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("model.base_url must be HTTP(S) without credentials/query/fragment.")
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            local = parsed.hostname.lower() == "localhost"
        if not local and not model.get("allow_remote", False):
            raise ValueError("Remote model endpoint is disabled.")
        context, output = int(model.get("context_tokens", 32768)), int(model.get("max_output_tokens", 4096))
        if context <= 2048 or output <= 0 or output >= context or float(model.get("timeout_seconds", 300)) <= 0:
            raise ValueError("Invalid model context/output/timeout settings.")
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
    out = project_path(config, config["run_directory"])
    try:
        manifest = read_json(out / "manifest.json", {})
        if not isinstance(manifest, dict):
            raise ValueError("Expected a manifest object")
    except (OSError, ValueError):
        manifest = {}
        errors.append("The chosen run_directory has an unreadable or malformed manifest. Preserve it and choose a new run_directory.")
    if manifest:
        _, _, fingerprint = configuration_fingerprint(config)
        if manifest.get("fingerprint") != fingerprint:
            errors.append("The chosen run_directory contains a different frozen code/configuration snapshot. Use a new run_directory; the existing run is preserved.")
    if enabled and "wos" in selected:
        warnings.append("WoS uses the Starter API; Core Collection/API entitlements cannot be verified offline.")
    if enabled and "crossref" in selected:
        warnings.append("Crossref is supplementary ranked discovery with a per-query result cap, not exhaustive Boolean retrieval.")
    if enabled and "scopus" in selected:
        warnings.append("Scopus cursor/COMPLETE-view access depends on entitlements; missing abstracts remain unresolved.")
    warnings.append("ACM Digital Library and citation-chasing records require manual exports/imports; they are not API adapters.")
    return {"check": "offline_only", "ready": not errors, "checked_for_stage": stage,
            "database_requests": 0, "model_requests": 0, "writes": 0,
            "sources": sources, "dependencies": dependencies,
            "model": {"name": model.get("name", "qwen3:8b"), "ollama_executable_found": bool(shutil.which("ollama")),
                      "server_and_model_availability": "not_checked_offline"},
            "errors": list(dict.fromkeys(errors)), "warnings": warnings,
            "access_verified": False}
