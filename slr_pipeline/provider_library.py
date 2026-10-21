#!/usr/bin/env python3
"""Preserved provider adapters and bibliography parsers; no CLI or import-time requests.
Python 3.10+. Required: requests. For .bib imports: bibtexparser==1.4.3.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

VERSION = "pipeline-1.0"

RS_TERMS = [
    "recommender system",
    "recommender systems",
    "recommendation system",
    "recommendation systems",
    "collaborative filtering",
    "item recommendation",
]

# These are starting search blocks, not a validated final SLR strategy.
# ISO terminology is an additional route, NOT a requirement for every paper.
TOPICS = {
    "quality": [
        "data quality",
        "dataset quality",
        "data characteristics",
        "data-centric",
        "item content completeness",
    ],
    "missingness": [
        "sparsity",
        "sparse data",
        "missing feedback",
        "missing data",
        "completeness",
        "cold start",
        "non-random missing",
    ],
    "noise": [
        "rating noise",
        "noisy feedback",
        "data noise",
        "data accuracy",
        "shilling",
        "data poisoning",
        "data corruption",
        "corrupted data",
    ],
    "distribution": [
        "item popularity",
        "popularity bias",
        "long tail",
        "user activity",
        "rating distribution",
        "imbalance",
        "representativeness",
        "data diversity",
    ],
    "selection": [
        "selection bias",
        "exposure bias",
        "sampling bias",
        "missing not at random",
        "negative sampling",
    ],
    "preprocessing": [
        "data cleaning",
        "data preprocessing",
        "dataset filtering",
        "data pruning",
        "dataset sampling",
    ],
    "time": [
        "temporal drift",
        "concept drift",
        "temporal split",
        "data leakage",
        "data freshness",
        "currentness",
        "timeliness",
        "distribution shift",
    ],
    "iso": [
        "5259",
        "ISO/IEC 5259",
    ],
}

KEYS = {
    "openalex": "OPENALEX_API_KEY",
    "scopus": "SCOPUS_API_KEY",
    "wos": "WOS_API_KEY",
    "ieee": "IEEE_API_KEY",
}

ENDPOINTS = {
    "crossref": "https://api.crossref.org/works",
    "arxiv": "https://export.arxiv.org/api/query",
    "openalex": "https://api.openalex.org/works",
    "scopus": "https://api.elsevier.com/content/search/scopus",
    "wos": "https://api.clarivate.com/apis/wos-starter/v1/documents",
    "ieee": "https://ieeexploreapi.ieee.org/api/v1/search/articles",
}

FIELDS = [
    "source_id",
    "title",
    "authors",
    "date",
    "date_basis",
    "doi",
    "related_doi",
    "abstract",
    "venue",
    "publication_type",
    "url",
    "oa_pdf_url",
]


def clean(value) -> str:
    # Recognize tag names rather than arbitrary angle-bracketed prose, so
    # mathematical comparisons such as "x < 3 and y > 2" remain intact.
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9:_.-]*(?:\s+[^<>]*?)?\s*/?>", " ", str(value or ""))
    return " ".join(html.unescape(text).split())


def norm(value) -> str:
    text = unicodedata.normalize("NFKC", clean(value)).casefold()
    return " ".join(re.findall(r"\w+", text))


def doi(value) -> str:
    return re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
        "",
        unquote(clean(value)),
        flags=re.I,
    ).strip().lower()


def digest(value) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def record(**values) -> dict:
    result = {k: clean(values.get(k, "")) for k in FIELDS}
    result["doi"] = doi(result["doi"])
    result["related_doi"] = doi(result["related_doi"])
    return result


def date_status(value: str, start: date, end: date) -> str:
    """Do not invent a day for year-only or month-only metadata."""
    try:
        if not re.fullmatch(r"\d{4}(?:-\d{2}){0,2}", value):
            return "check_missing_or_unparsed_date"

        parts = list(map(int, value.split("-")))
        year = parts[0]

        if len(parts) == 1:
            low = date(year, 1, 1)
            high = date(year, 12, 31)
        elif len(parts) == 2:
            month = parts[1]
            low = date(year, month, 1)
            high = date(year, month, calendar.monthrange(year, month)[1])
        else:
            low = high = date(*parts)

        if high < start or low > end:
            return "outside_window"

        if start <= low and high <= end:
            return "within_window"

        return "check_boundary_date"

    except ValueError:
        return "check_invalid_date"


def boolean(terms: list[str], arxiv: bool = False) -> str:
    if arxiv:
        pieces = [f'(ti:"{term}" OR abs:"{term}")' for term in terms]
    else:
        pieces = [f'"{term}"' for term in terms]

    return "(" + " OR ".join(pieces) + ")"


def jobs_for(sources: list[str], topics: list[str]) -> list[dict]:
    jobs = []

    for source in sources:
        for topic in topics:
            if source == "crossref":
                # Crossref is ranked discovery, not Boolean database searching.
                anchors = ["recommender systems", "recommendation systems",
                           "collaborative filtering", "item recommendation"]
                pairs = [
                    (anchor, term)
                    for anchor in anchors
                    for term in TOPICS[topic]
                ]

                for i, (anchor, term) in enumerate(pairs, 1):
                    jobs.append({
                        "id": f"{source}_{topic}_{i}",
                        "source": source,
                        "topic": topic,
                        "query": f"{anchor} {term}",
                    })
            else:
                query = boolean(RS_TERMS, source == "arxiv")
                query += " AND "
                query += boolean(TOPICS[topic], source == "arxiv")

                jobs.append({
                    "id": f"{source}_{topic}",
                    "source": source,
                    "topic": topic,
                    "query": query,
                })

    return jobs


class APIError(RuntimeError):
    def __init__(self, message: str, halt_source: bool = False):
        super().__init__(message)
        self.halt_source = halt_source


class Client:
    def __init__(self, email: str):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            f"RecSysSLR/{VERSION} (mailto:{email})"
        )
        self.last = 0.0
        self.secrets = [
            os.getenv(key, "")
            for key in [*KEYS.values(), "SCOPUS_INST_TOKEN"]
        ]

    def redact(self, text: str) -> str:
        for secret in self.secrets:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text

    def get(self, url: str, params=None, headers=None):
        hostname = urlparse(url).hostname or ""
        delay = 3.2 if "arxiv.org" in hostname else 1.1

        for attempt in range(5):
            time.sleep(max(0, self.last + delay - time.monotonic()))
            self.last = time.monotonic()

            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=(15, 90),
                )
            except requests.RequestException as exc:
                if attempt == 4:
                    # Request exception text can expose credential-bearing URLs.
                    raise APIError(
                        f"Network failure: {type(exc).__name__}",
                        True,
                    ) from None

                time.sleep(2 ** (attempt + 1))
                continue

            if response.status_code == 429 or 500 <= response.status_code < 600:
                wait = 2 ** (attempt + 2)
                retry = response.headers.get("Retry-After", "")

                try:
                    wait = max(wait, float(retry))
                except ValueError:
                    try:
                        retry_time = parsedate_to_datetime(retry).timestamp()
                        wait = max(wait, retry_time - time.time())
                    except (TypeError, ValueError, OverflowError):
                        pass

                code = response.status_code
                response.close()

                # Do not bypass a long provider-requested cooldown.
                if attempt == 4 or wait > 120:
                    raise APIError(
                        f"HTTP {code}: rate limit/server unavailable; "
                        "retrieval unfinished",
                        True,
                    )

                time.sleep(wait)
                continue

            if not response.ok:
                code = response.status_code
                detail = self.redact(
                    clean(unquote(self.redact(response.text)))
                )[:600]
                response.close()
                raise APIError(
                    f"HTTP {code}: {detail or 'Empty error response'}",
                    code in (401, 402, 403),
                )

            return response

        raise APIError("Retries exhausted", True)


def request_spec(source: str, query: str, position, args):
    headers = {"Accept": "application/json"}

    year_filter = (
        f"PUBYEAR > {args.start.year - 1} "
        f"AND PUBYEAR < {args.end.year + 1}"
    )

    if source == "crossref":
        params = {
            "query.bibliographic": query,
            "rows": args.crossref_rows,
            "filter": (
                f"from-pub-date:{args.start},"
                f"until-pub-date:{args.end}"
            ),
            "mailto": args.email,
            "sort": "relevance",
            "order": "desc",
        }

    elif source == "arxiv":
        dated_query = (
            f"({query}) AND submittedDate:"
            f"[{args.start:%Y%m%d}0000 TO {args.end:%Y%m%d}2359]"
        )
        params = {
            "search_query": dated_query,
            "start": position or 0,
            "max_results": 100,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        headers["Accept"] = "application/atom+xml"

    elif source == "openalex":
        params = {
            "search.title_and_abstract": query,
            "per_page": 100,
            "cursor": position or "*",
            "filter": (
                f"from_publication_date:{args.start},"
                f"to_publication_date:{args.end}"
            ),
        }
        headers["Authorization"] = "Bearer " + os.environ[KEYS[source]]

    elif source == "scopus":
        params = {
            "query": f"TITLE-ABS-KEY({query}) AND {year_filter}",
            "count": 25,
            "view": getattr(args, "scopus_view", "STANDARD"),
        }
        pagination = getattr(args, "scopus_pagination", "cursor")
        if pagination == "cursor":
            params["cursor"] = position or "*"
        elif pagination == "offset":
            offset = position or 0
            if offset >= 5000:
                raise APIError("Scopus offset limit reached (5000); retrieval unfinished. "
                               "Use cursor pagination with an eligible API entitlement.")
            params["start"] = offset
            params["count"] = min(params["count"], 5000 - offset)
        else:
            raise ValueError("scopus_pagination must be cursor or offset")
        headers["X-ELS-APIKey"] = os.environ[KEYS[source]]

        if os.getenv("SCOPUS_INST_TOKEN"):
            headers["X-ELS-Insttoken"] = os.environ["SCOPUS_INST_TOKEN"]

    elif source == "wos":
        params = {
            "q": (
                f"TS=({query}) "
                f"AND PY=({args.start.year}-{args.end.year})"
            ),
            "db": "WOS",
            "limit": 50,
            "page": position or 1,
        }
        headers["X-ApiKey"] = os.environ[KEYS[source]]

    else:
        params = {
            "querytext": query,
            "apikey": os.environ[KEYS[source]],
            "start_year": args.start.year,
            "end_year": args.end.year,
            "start_record": position or 1,
            "max_records": 100,
            "format": "json",
            "sort_field": "article_number",
            "sort_order": "asc",
        }

    return params, headers


def unpack(source: str, text: str):
    """Return records, provider-reported total, and OpenAlex cursor."""
    records = []
    next_cursor = None

    if source == "arxiv":
        ns = {
            "a": "http://www.w3.org/2005/Atom",
            "x": "http://arxiv.org/schemas/atom",
            "o": "http://a9.com/-/spec/opensearch/1.1/",
        }

        root = ET.fromstring(text)
        total_text = root.findtext("o:totalResults", namespaces=ns)

        if total_text is None:
            raise APIError("arXiv returned an error feed or unexpected XML")

        for entry in root.findall("a:entry", ns):
            def get(tag):
                return entry.findtext(tag, default="", namespaces=ns)

            url = get("a:id").replace("http://", "https://")

            if "/api/errors" in url:
                raise APIError("arXiv query error")

            pdf = next(
                (
                    link.get("href", "")
                    for link in entry.findall("a:link", ns)
                    if link.get("title") == "pdf"
                ),
                "",
            )

            authors = "; ".join(
                author.findtext("a:name", default="", namespaces=ns)
                for author in entry.findall("a:author", ns)
            )

            records.append(record(
                source_id=re.sub(
                    r"v\d+$",
                    "",
                    url.split("/abs/")[-1],
                ),
                title=get("a:title"),
                abstract=get("a:summary"),
                url=url,
                authors=authors,
                date=get("a:published")[:10],
                date_basis="arxiv_first_submission",
                # Do not merge the preprint into its published article.
                related_doi=get("x:doi"),
                publication_type="preprint",
                venue=get("x:journal_ref"),
                oa_pdf_url=pdf.replace("http://", "https://"),
            ))

        return records, int(total_text), None

    data = json.loads(text)

    if not isinstance(data, dict):
        raise APIError("Unexpected JSON response")

    if source == "crossref":
        message = data["message"]
        items = message["items"]
        total = int(message["total-results"])

        for item in items:
            dates = []
            basis = ""

            for name in [
                "published",
                "published-online",
                "published-print",
                "issued",
            ]:
                dates = (
                    (item.get(name) or {}).get("date-parts") or [[]]
                )[0]

                if dates:
                    basis = "crossref_" + name
                    break

            published = "-".join(
                f"{int(number):04d}" if i == 0 else f"{int(number):02d}"
                for i, number in enumerate(dates)
            )

            authors = "; ".join(
                clean(
                    author.get("given", "")
                    + " "
                    + author.get("family", "")
                ) or clean(author.get("name"))
                for author in (item.get("author") or [])
            )

            records.append(record(
                source_id=item.get("DOI"),
                doi=item.get("DOI"),
                title=" ".join(item.get("title") or []),
                abstract=item.get("abstract"),
                authors=authors,
                date=published,
                date_basis=basis,
                venue="; ".join(item.get("container-title") or []),
                publication_type=item.get("type"),
                url=item.get("URL"),
            ))

    elif source == "openalex":
        items = data["results"]
        total = int(data["meta"]["count"])
        next_cursor = data["meta"].get("next_cursor")

        for item in items:
            pairs = [
                (position, word)
                for word, positions in (
                    item.get("abstract_inverted_index") or {}
                ).items()
                for position in positions
            ]

            location = item.get("primary_location") or {}
            oa = item.get("best_oa_location") or {}

            authors = "; ".join(
                (author.get("author") or {}).get("display_name", "")
                for author in (item.get("authorships") or [])
            )

            records.append(record(
                source_id=item.get("id"),
                doi=item.get("doi"),
                title=item.get("display_name"),
                abstract=" ".join(word for _, word in sorted(pairs)),
                authors=authors,
                date=item.get("publication_date"),
                date_basis="openalex_publication_date",
                venue=(location.get("source") or {}).get("display_name"),
                publication_type=item.get("type"),
                url=location.get("landing_page_url") or item.get("id"),
                oa_pdf_url=oa.get("pdf_url") if oa.get("is_oa") else "",
            ))

    elif source == "scopus":
        message = data["search-results"]
        total = int(message["opensearch:totalResults"])
        next_cursor = (message.get("cursor") or {}).get("@next")
        items = message.get("entry", []) if total else []

        for item in items:
            if "error" in item:
                raise APIError("Scopus returned an error entry")

            url = next(
                (
                    link.get("@href", "")
                    for link in (item.get("link") or [])
                    if link.get("@ref") == "scopus"
                ),
                "",
            )

            records.append(record(
                source_id=item.get("eid") or item.get("dc:identifier"),
                doi=item.get("prism:doi"),
                title=item.get("dc:title"),
                authors=item.get("dc:creator"),
                date=item.get("prism:coverDate"),
                date_basis="scopus_cover_date",
                abstract=item.get("dc:description"),
                venue=item.get("prism:publicationName"),
                publication_type=item.get("subtypeDescription"),
                url=url,
            ))

    elif source == "wos":
        items = data["hits"]
        total = int(data["metadata"]["total"])

        for item in items:
            source_data = item.get("source") or {}

            authors = "; ".join(
                author.get("displayName", "")
                for author in (
                    (item.get("names") or {}).get("authors") or []
                )
            )

            records.append(record(
                source_id=item.get("uid"),
                title=item.get("title"),
                doi=(item.get("identifiers") or {}).get("doi"),
                authors=authors,
                date=source_data.get("publishYear"),
                date_basis="wos_publication_year",
                venue=source_data.get("sourceTitle"),
                publication_type="; ".join(item.get("types") or []),
                url=(item.get("links") or {}).get("record"),
            ))

    else:
        total = int(
            data["total_records"]
            if "total_records" in data
            else data["totalfound"]
        )
        items = data.get("articles", [])

        for item in items:
            authors = "; ".join(
                author.get("full_name", "")
                for author in (
                    (item.get("authors") or {}).get("authors") or []
                )
            )

            records.append(record(
                source_id=item.get("article_number"),
                title=item.get("title"),
                doi=item.get("doi"),
                abstract=item.get("abstract"),
                authors=authors,
                date=item.get("publication_year"),
                date_basis="ieee_publication_year",
                venue=item.get("publication_title"),
                publication_type=item.get("content_type"),
                url=item.get("abstract_url"),
            ))

    return records, total, next_cursor


def imported(path: Path) -> list[dict]:
    if path.suffix.lower() == ".bib":
        try:
            import bibtexparser
        except ImportError:
            raise ValueError(
                "For BibTeX imports install: "
                "pip install bibtexparser==1.4.3"
            ) from None

        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False

        with path.open(encoding="utf-8-sig") as file:
            entries = bibtexparser.load(file, parser=parser).entries

        return [
            record(
                source_id=item.get("ID"),
                title=item.get("title"),
                authors=item.get("author"),
                date=item.get("year"),
                doi=item.get("doi"),
                abstract=item.get("abstract"),
                venue=item.get("journal") or item.get("booktitle"),
                publication_type=item.get("ENTRYTYPE"),
                url=item.get("url"),
                date_basis="imported_bibtex_year",
            )
            for item in entries
        ]

    if path.suffix.lower() == ".ris":
        entries = []
        current = defaultdict(list)
        tag = ""

        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"^([A-Z0-9]{2})  -\s?(.*)$", line)

            if match:
                tag, value = match.groups()

                if tag == "ER":
                    entries.append(current)
                    current = defaultdict(list)
                    tag = ""
                else:
                    current[tag].append(value)

            elif tag and line.strip():
                current[tag][-1] += " " + line.strip()

        if current:
            raise ValueError("RIS record is missing its ER terminator")

        rows = []

        for item in entries:
            def get(*tags):
                return next(
                    (item[tag][0] for tag in tags if item.get(tag)),
                    "",
                )

            rows.append(record(
                source_id=get("ID", "AN"),
                title=get("TI", "T1"),
                authors="; ".join(
                    item.get("AU") or item.get("A1") or []
                ),
                date=get("PY", "Y1", "DA")[:4],
                date_basis="imported_ris_year",
                doi=get("DO"),
                abstract=get("AB", "N2"),
                venue=get("JO", "JF", "T2"),
                publication_type=get("TY"),
                url=get("UR"),
            ))

        return rows

    raise ValueError("Supported imports: .ris and .bib")

