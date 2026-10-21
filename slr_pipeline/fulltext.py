"""Private local full-text collection and extraction, with optional OA downloads.

No work occurs at import time. Missing/unreadable text is a retrieval problem,
never a scientific exclusion. This module does not contact a model or upload text.
"""
from __future__ import annotations

import csv
import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit, urlunsplit

from .common import utcnow, write_csv, write_json


EXTRACTOR_VERSION = "fulltext-1"
_SENSITIVE_QUERY = re.compile(
    r"(?:api.?key|token|secret|password|credential|authorization|signature|^key$|^auth$|^sig$)", re.I
)
_REDIRECTS = {301, 302, 303, 307, 308}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


class FullTextError(ValueError):
    """A message safe to include in the local audit log."""


def _doi(value: str) -> str:
    value = unquote(str(value or "").strip()).casefold()
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", value)
    return value if re.fullmatch(r"10\.\d{4,9}/[^\s<>]+", value) else ""


def _local_path(value: str, base: Path) -> Path:
    if not value or "://" in value or value.startswith("//"):
        raise FullTextError("Expected a local filesystem path, not a URL")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _file_sha(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _label(record_id: str) -> str:
    # IDs in an imported CSV are untrusted; never make them filesystem paths.
    return record_id if _SAFE_ID.fullmatch(record_id) else "R" + hashlib.sha256(record_id.encode()).hexdigest()[:24]


def _checked_url(raw: str) -> tuple[str, str, int, str]:
    try:
        parsed = urlsplit(raw)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise FullTextError("Invalid OA PDF URL") from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise FullTextError("OA PDF URL must use public HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise FullTextError("Credential-bearing OA PDF URL blocked")
    if any(_SENSITIVE_QUERY.search(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
        raise FullTextError("Credential-bearing OA PDF query blocked")
    if port not in {80, 443}:
        raise FullTextError("Nonstandard OA download port blocked")
    if any(ord(char) < 32 for char in raw) or "\\" in raw:
        raise FullTextError("Unsafe characters in OA PDF URL")
    try:
        host = parsed.hostname.encode("idna").decode("ascii")
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        raise FullTextError("OA host could not be resolved") from None
    candidates = sorted({entry[4][0] for entry in addresses})
    if not candidates or any(not ipaddress.ip_address(address).is_global for address in candidates):
        raise FullTextError("Private, local, reserved or mixed-address OA host blocked")
    request_path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    return host, candidates[0], port, request_path


def _display_url(raw: str) -> str:
    """Do not leak credentials from imported URLs into retrieval worksheets."""
    try:
        parsed = urlsplit(raw)
        if parsed.username is not None or parsed.password is not None:
            return "[credential-bearing URL omitted]"
        if any(_SENSITIVE_QUERY.search(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
            return "[credential-bearing URL omitted]"
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
    except ValueError:
        return "[invalid URL omitted]"


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, address: str, port: int, timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._approved_address = address

    def connect(self):
        self.sock = socket.create_connection((self._approved_address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str, port: int, timeout: float):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._approved_address = address

    def connect(self):
        raw = socket.create_connection((self._approved_address, self.port), self.timeout)
        try:
            # Keep certificate/SNI validation against the original hostname while
            # connecting only to the public IP we just validated (no DNS rebind).
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


def _download(raw_url: str, target: Path, max_bytes: int, timeout: float) -> None:
    """Download one public PDF, revalidating/pinning each redirect destination."""
    current = raw_url
    for _ in range(6):
        host, address, port, request_path = _checked_url(current)
        cls = _PinnedHTTPSConnection if urlsplit(current).scheme == "https" else _PinnedHTTPConnection
        connection = cls(host, address, port, timeout)
        temporary = None
        try:
            connection.request("GET", request_path, headers={
                "User-Agent": "RecSysSLR/1.0 (private open-access full-text collection)",
                "Accept": "application/pdf, application/octet-stream;q=0.8",
                "Accept-Encoding": "identity",
            })
            response = connection.getresponse()
            if response.status in _REDIRECTS:
                location = response.getheader("Location")
                if not location:
                    raise FullTextError("OA redirect had no destination")
                destination = urljoin(current, location)
                if urlsplit(current).scheme == "https" and urlsplit(destination).scheme != "https":
                    raise FullTextError("HTTPS-to-HTTP OA redirect blocked")
                current = destination
                continue
            if response.status != 200:
                raise FullTextError(f"OA download returned HTTP {response.status}")
            content_type = (response.getheader("Content-Type") or "").split(";", 1)[0].strip().lower()
            if content_type not in {"application/pdf", "application/x-pdf", "application/octet-stream"}:
                raise FullTextError("OA response did not have a PDF content type")
            if (response.getheader("Content-Encoding") or "identity").lower() != "identity":
                raise FullTextError("Compressed OA HTTP response not supported")
            length = response.getheader("Content-Length")
            if length:
                try:
                    declared_size = int(length)
                except ValueError:
                    raise FullTextError("Invalid OA response size") from None
                if declared_size < 0:
                    raise FullTextError("Invalid OA response size")
                if declared_size > max_bytes:
                    raise FullTextError("OA PDF exceeds configured size limit")
            target.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                while True:
                    block = response.read(min(65536, max_bytes - total + 1))
                    if not block:
                        break
                    if total == 0 and not block.lstrip().startswith(b"%PDF-"):
                        raise FullTextError("OA response body is not a PDF")
                    total += len(block)
                    if total > max_bytes:
                        raise FullTextError("OA PDF exceeds configured size limit")
                    output.write(block)
                if total == 0:
                    raise FullTextError("OA response body was empty")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            temporary = None
            return
        except FullTextError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            # Exception strings and HTTP response bodies can contain URLs or keys.
            raise FullTextError(f"OA network/HTTP failure ({type(exc).__name__})") from None
        finally:
            connection.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise FullTextError("OA redirect limit exceeded")


def _extract(path: Path, cache_dir: Path) -> dict:
    file_hash = _file_sha(path)
    cache_path = cache_dir / f"{EXTRACTOR_VERSION}_{file_hash}_{path.suffix.lower().lstrip('.')}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("file_sha256") == file_hash and cached.get("extractor_version") == EXTRACTOR_VERSION:
                return cached
        except (OSError, ValueError):
            pass
    result = {
        "file_sha256": file_hash, "extractor_version": EXTRACTOR_VERSION,
        "text": "", "page_count": 0, "sparse_pages": [], "failed_pages": [], "characters": 0,
    }
    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8-sig")
        result.update(text="[TEXT FILE]\n" + text, page_count=1, characters=len(text.strip()))
        if len(re.sub(r"\s+", "", text)) < 50:
            result["sparse_pages"] = [1]
    elif path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise FullTextError("PDF extraction requires pypdf; install requirements.txt") from None
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise FullTextError("Encrypted PDF requires a readable local copy")
        chunks = []
        characters = 0
        result["page_count"] = len(reader.pages)
        for index, page in enumerate(reader.pages, 1):
            try:
                text = page.extract_text() or ""
            except Exception:
                result["failed_pages"].append(index)
                text = ""
            characters += len(text.strip())
            if len(re.sub(r"\s+", "", text)) < 50:
                result["sparse_pages"].append(index)
            chunks.append(f"[PAGE {index}]\n{text}")
        result.update(text="\n\n".join(chunks), characters=characters)
    else:
        raise FullTextError("Only local .pdf and UTF-8 .txt full texts are supported")
    result["text_sha256"] = hashlib.sha256(result["text"].encode("utf-8")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, result)
    return result


def _manifest(path: Path, records: list[dict]) -> dict[str, dict]:
    if not path.exists():
        return {}
    by_doi = defaultdict(list)
    record_ids = {str(record.get("record_id", "")) for record in records}
    record_dois = {str(record.get("record_id", "")): _doi(record.get("doi", "")) for record in records}
    for record in records:
        identifier = _doi(record.get("doi", ""))
        if identifier:
            by_doi[identifier].append(str(record.get("record_id", "")))
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "path" not in (reader.fieldnames or []):
            raise FullTextError("Full-text manifest needs a path column")
        if not {"record_id", "doi"}.intersection(reader.fieldnames or []):
            raise FullTextError("Full-text manifest needs record_id or doi")
        for row in reader:
            record_id = str(row.get("record_id", "")).strip()
            identifier = _doi(row.get("doi", ""))
            if not record_id:
                matches = by_doi.get(identifier, [])
                if len(matches) != 1:
                    # A DOI ambiguity is never resolved by selecting the first row.
                    if len(matches) > 1:
                        for match in matches:
                            result[match] = {"error": "Manifest DOI matches multiple reports; use record_id"}
                    continue
                record_id = matches[0]
            if record_id not in record_ids:
                continue
            if identifier and record_dois[record_id] and identifier != record_dois[record_id]:
                row = {"error": "Manifest record_id and DOI disagree"}
            if record_id in result:
                result[record_id] = {"error": "Multiple manifest rows for this report; provide one explicit path"}
            else:
                result[record_id] = row
    return result


def prepare_fulltexts(records: list[dict], config: dict, run_dir: Path) -> list[dict]:
    """Return records with complete local text where possible and an audited queue.

    A successful PDF text extraction is not an identity/eligibility judgement. A
    supplied file must be the correct report, with all substantive pages. Scanned
    or partial files stay unresolved until the user supplies a better/OCR copy.
    """
    settings = config.get("fulltext", {})
    base = Path(config.get("_project_dir", ".")).resolve()
    directory = _local_path(str(settings.get("directory", "inputs/fulltext")), base)
    manifest_path = _local_path(str(settings.get("manifest", "inputs/fulltext_manifest.csv")), base)
    output = Path(run_dir) / "04_full_text"
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(run_dir) / "checkpoints" / "fulltext"
    downloads = output / "pdfs"
    minimum = int(settings.get("min_text_chars", 1000))
    max_bytes = int(float(settings.get("max_download_mb", 50)) * 1024 * 1024)
    timeout = float(settings.get("timeout_seconds", 90))
    if minimum < 1 or max_bytes < 1 or timeout <= 0:
        raise FullTextError("Full-text size, timeout and minimum text settings must be positive")
    mappings = _manifest(manifest_path, records)
    results, log, manifest_rows, needed = [], [], [], []
    for original in records:
        record = dict(original)
        record_id = str(record.get("record_id", ""))
        if not record_id:
            raise FullTextError("Full-text input contains a report without record_id")
        label = _label(record_id)
        record.update(full_text="", full_text_sha256="", full_text_file_sha256="",
                      full_text_path="", text_complete=False, full_text_status="not_retrieved",
                      full_text_error="", full_text_pages=0, full_text_characters=0)
        attempt = {"record_id": record_id, "title": record.get("title", ""),
                   "doi": record.get("doi", ""), "attempted_utc": utcnow(),
                   "method": "local", "oa_pdf_url": _display_url(str(record.get("oa_pdf_url", "")))}
        path = None
        coverage = "complete"
        try:
            mapping = mappings.get(record_id)
            if mapping:
                if mapping.get("error"):
                    raise FullTextError(mapping["error"])
                coverage = str(mapping.get("coverage", "complete") or "complete").strip().lower()
                if coverage not in {"complete", "partial"}:
                    raise FullTextError("Manifest coverage must be complete or partial")
                path = _local_path(str(mapping.get("path", "")).strip(), base)
                if not path.is_file():
                    raise FullTextError("Manifest file is missing or is not a regular file")
                attempt["method"] = "manifest"
            else:
                candidates = [directory / f"{label}.pdf", directory / f"{label}.txt"]
                existing = [candidate for candidate in candidates if candidate.is_file()]
                if len(existing) > 1:
                    raise FullTextError("Both PDF and TXT match this report; choose one in the manifest")
                if existing:
                    path = existing[0].resolve()
                elif (downloads / f"{label}.pdf").is_file():
                    path = (downloads / f"{label}.pdf").resolve()
                    attempt["method"] = "cached_oa_pdf"
                elif settings.get("download_open_access", False) and record.get("oa_pdf_url"):
                    path = downloads / f"{label}.pdf"
                    attempt["method"] = "oa_download"
                    _download(str(record["oa_pdf_url"]), path, max_bytes, timeout)
                    path = path.resolve()
            if path is not None:
                record["full_text_path"] = str(path)
                data = _extract(path, cache_dir)
                record.update(full_text=data["text"], full_text_sha256=data["text_sha256"],
                              full_text_file_sha256=data["file_sha256"], full_text_pages=data["page_count"],
                              full_text_characters=data["characters"])
                record["text_complete"] = bool(data["page_count"] and data["characters"] >= minimum
                                               and not data["sparse_pages"] and not data["failed_pages"]
                                               and coverage == "complete")
                if record["text_complete"]:
                    record["full_text_status"] = "ready"
                elif data["characters"] < minimum:
                    record["full_text_status"] = "needs_ocr" if data["characters"] < 100 else "partial"
                    record["full_text_error"] = "Insufficient extracted text; supply full text/OCR and verify coverage"
                else:
                    record["full_text_status"] = "partial"
                    record["full_text_error"] = (
                        "Partial coverage declared" if coverage == "partial" else
                        "Sparse/unreadable pages require inspection or OCR; text coverage is unresolved"
                    )
                attempt.update(sparse_pages=json.dumps(data["sparse_pages"]),
                               failed_pages=json.dumps(data["failed_pages"]))
            else:
                record["full_text_error"] = "No matching local full text; obtain an authorised PDF or UTF-8 TXT copy"
        except FullTextError as exc:
            record.update(full_text_status="error", full_text_error=str(exc), text_complete=False)
        except Exception as exc:
            record.update(full_text_status="error", text_complete=False,
                          full_text_error=f"Full-text read/extraction failure ({type(exc).__name__})")
        attempt.update(status=record["full_text_status"], path=record["full_text_path"],
                       text_complete=record["text_complete"], error=record["full_text_error"],
                       file_sha256=record["full_text_file_sha256"], text_sha256=record["full_text_sha256"])
        log.append(attempt)
        with (output / "retrieval_attempts.jsonl").open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(attempt, ensure_ascii=False) + "\n")
            audit.flush()
            os.fsync(audit.fileno())
        manifest_rows.append({"record_id": record_id, "doi": record.get("doi", ""),
                              "path": record["full_text_path"], "coverage": coverage,
                              "status": record["full_text_status"], "text_complete": record["text_complete"],
                              "file_sha256": record["full_text_file_sha256"],
                              "text_sha256": record["full_text_sha256"],
                              "pages": record["full_text_pages"], "characters": record["full_text_characters"]})
        if not record["text_complete"]:
            needed.append({"record_id": record_id, "doi": record.get("doi", ""),
                           "title": record.get("title", ""), "status": record["full_text_status"],
                           "expected_path": str(directory / f"{label}.pdf"),
                           "existing_path": record["full_text_path"], "reason": record["full_text_error"],
                           "oa_pdf_url": _display_url(str(record.get("oa_pdf_url", ""))),
                           "doi_url": f"https://doi.org/{_doi(record.get('doi', ''))}" if _doi(record.get("doi", "")) else ""})
        results.append(record)
    write_csv(output / "retrieval_log.csv", log, ["record_id", "title", "doi", "attempted_utc", "method", "oa_pdf_url", "status", "path", "text_complete", "error", "file_sha256", "text_sha256", "sparse_pages", "failed_pages"])
    write_csv(output / "fulltext_manifest.csv", manifest_rows, ["record_id", "doi", "path", "coverage", "status", "text_complete", "file_sha256", "text_sha256", "pages", "characters"])
    write_csv(output / "fulltext_needed.csv", needed, ["record_id", "doi", "title", "status", "expected_path", "existing_path", "reason", "oa_pdf_url", "doi_url"])
    return results
