#!/usr/bin/env python3
"""Conservative matcher adapted from the preserved step2_deduplicate.py.

No command-line entry point and no work on import. The pipeline wrapper writes
the outputs. Fuzzy matches are always candidates for manual review.
"""
from __future__ import annotations

import html
import itertools
import re
import unicodedata
from collections import defaultdict
from urllib.parse import unquote, urlsplit

from rapidfuzz import fuzz

PROTECTED = ("screening_decision", "exclusion_reason", "iso_mapping")
MATCH_RULES = (
    "same_normalized_doi",
    "same_provider_identifier",
    "same_title_year_first_author_and_venue",
)


def normalized(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"</?(?:b|i|em|strong|sup|sub|span)(?:\s[^>]*)?>", " ",
                   value, flags=re.I)
    value = unicodedata.normalize("NFKC", value).casefold()
    # Keep digits and important operators: 'part 1' != 'part 2', C++ != C#.
    return " ".join(re.findall(r"\w+|[+<>=#]", value))


def normalize_doi(value: str) -> str:
    value = html.unescape(value or "").strip().strip("<>")
    if re.match(r"^https?://(?:dx\.)?doi\.org/", value, re.I):
        value = unquote(urlsplit(value).path.lstrip("/"))
    else:
        value = unquote(re.sub(r"^doi\s*:\s*", "", value, flags=re.I))
    value = value.strip().lower()
    # Do not guess where a malformed identifier ends or silently trim its suffix.
    return value if re.fullmatch(r'10\.\d{4,9}/[^\s<>"\x00-\x1f]+', value) else ""


def pieces(value: str) -> set[str]:
    return {x.strip() for x in (value or "").split(";") if x.strip()}


def first_author(value: str) -> str:
    first = re.split(r";|\s+and\s+", value or "", maxsplit=1, flags=re.I)[0]
    words = normalized(first).split()
    # Accept reversed name order, but do not equate initials with guessed names.
    return " ".join(sorted(words)) if len(words) >= 2 else ""


def publication_kind(row: dict) -> str:
    kind = normalized(row.get("publication_type", ""))
    venue = normalized(row.get("venue", ""))
    is_preprint = (
        "preprint" in kind or "posted content" in kind or "arxiv" in venue
        or pieces(row.get("sources", "")) == {"arxiv"}
        or row.get("date_basis", "").startswith("arxiv_")
        or normalize_doi(row.get("doi", "")).startswith("10.48550/arxiv.")
    )
    if is_preprint:
        return "preprint"
    if any(x in kind for x in ("conference", "proceeding", "inproceedings")):
        return "conference"
    if any(x in kind for x in ("journal", "article")):
        return "journal"
    return ""


def provider_keys(row: dict) -> set[str]:
    keys = set()
    for raw in (row.get("source_id", ""), row.get("url", "")):
        value = raw.strip()
        match = re.fullmatch(r"2-s2\.0-(\d+)", value)
        if match:
            keys.add("scopus:" + match[1])
        if re.fullmatch(r"WOS:\d+", value, re.I):
            keys.add(value.upper())
        if value.startswith(("http://", "https://")):
            parsed = urlsplit(value)
            host = (parsed.hostname or "").lower().removeprefix("www.")
            path = parsed.path.rstrip("/")
            if host == "openalex.org" and re.fullmatch(r"/W\d+", path, re.I):
                keys.add("openalex:" + path[1:].upper())
            if host == "ieeexplore.ieee.org":
                match = re.fullmatch(r"/document/(\d+)", path)
                if match:
                    keys.add("ieee:" + match[1])
            if host in ("arxiv.org", "export.arxiv.org"):
                match = re.fullmatch(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?", path)
                if match:
                    keys.add("arxiv:" + re.sub(r"v\d+$", "", match[1]))
    sources = pieces(row.get("sources", ""))
    if row.get("source"):
        sources.add(row["source"])
    source_id = row.get("source_id", "").strip()
    # API-native identifiers are scoped by provider. A bare integer from one
    # provider must never collide with an integer from another provider.
    source = row.get("source", "").strip().casefold()
    if source in {"scopus", "wos", "ieee", "openalex", "arxiv", "crossref"} and source_id:
        key = source_id
        if source == "arxiv":
            key = re.sub(r"v\d+$", "", key)
        elif source == "openalex":
            key = key.rstrip("/").rsplit("/", 1)[-1].upper()
        elif source == "crossref":
            key = normalize_doi(key)
        if key:
            keys.add(source + ":" + key)
    if sources == {"ieee"} and source_id.isdigit():
        keys.add("ieee:" + source_id)
    if sources == {"arxiv"} and re.fullmatch(
        r"(?:\d{4}\.\d{4,5}|[a-z.-]+/\d{7})(?:v\d+)?", source_id, re.I
    ):
        keys.add("arxiv:" + re.sub(r"v\d+$", "", source_id))
    return keys


def metadata(row: dict) -> dict:
    title = normalized(row.get("title", ""))
    year = re.match(r"^(\d{4})(?:\D|$)", row.get("date", ""))
    return {
        "title": title,
        "doi": normalize_doi(row.get("doi", "")),
        "year": year[1] if year else "",
        "author": first_author(row.get("authors", "")),
        "venue": normalized(row.get("venue", "")),
        "kind": publication_kind(row),
        "provider_keys": provider_keys(row),
        "notice": bool(re.match(
            r"^(?:correction|erratum|corrigendum|retraction|withdrawal|"
            r"expression of concern)\b", title
        )),
    }


class Deduplicator:
    def __init__(self, rows: list[dict], block_limit: int = 200):
        self.rows = rows
        self.meta = [metadata(row) for row in rows]
        self.parent = list(range(len(rows)))
        self.members = {i: {i} for i in range(len(rows))}
        self.events = []
        self.review = defaultdict(set)
        self.large_blocks = []
        self.block_limit = block_limit

    def root(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def conflicts(self, indices: set[int]) -> list[str]:
        """Cluster-wide checks prevent a no-DOI row bridging different DOIs."""
        reasons = []
        values = lambda name: {self.meta[i][name] for i in indices if self.meta[i][name]}
        if len(values("doi")) > 1:
            reasons.append("different_dois")
        if len(values("kind")) > 1:
            reasons.append("different_publication_types_or_versions")
        if len({self.meta[i]["notice"] for i in indices}) > 1:
            reasons.append("notice_vs_original_paper")
        years = [int(x) for x in values("year")]
        if years and max(years) - min(years) > 2:
            reasons.append("publication_year_gap_over_two")
        titles = sorted(values("title"))
        number_sets = {tuple(re.findall(r"\d+(?:\.\d+)?", t)) for t in titles}
        if len(number_sets) > 1:
            reasons.append("different_title_numbers")
        if any(fuzz.ratio(a, b) < 85 for a, b in itertools.combinations(titles, 2)):
            reasons.append("inconsistent_titles")
        for field in PROTECTED:
            if len({self.rows[i].get(field, "").strip().casefold()
                    for i in indices if self.rows[i].get(field, "").strip()}) > 1:
                reasons.append("conflicting_" + field)
        return reasons

    def flag(self, a: int, b: int, reason: str) -> None:
        if a != b:
            self.review[tuple(sorted((a, b)))].add(reason)

    def apply_bucket(self, indices: list[int], rule: str, key: str) -> None:
        roots = sorted({self.root(i) for i in indices})
        if len(roots) < 2:
            return
        if len(roots) > self.block_limit:
            self.large_blocks.append({"rule": rule, "key": key, "record_indices": indices})
            return
        all_members = set().union(*(self.members[root] for root in roots))
        conflicts = self.conflicts(all_members)
        if conflicts:
            # Leave the ENTIRE ambiguous bucket unmerged, not an arbitrary pair.
            for a, b in itertools.combinations(roots, 2):
                self.flag(a, b, rule + ":" + ";".join(conflicts))
            return
        destination = roots[0]
        for other in roots[1:]:
            self.events.append({"left_index": destination, "right_index": other, "rule": rule})
            self.parent[other] = destination
            self.members[destination].update(self.members.pop(other))

    def exact_passes(self) -> None:
        for field, rule in (("doi", MATCH_RULES[0]), ("provider_keys", MATCH_RULES[1])):
            buckets = defaultdict(list)
            for i, meta in enumerate(self.meta):
                keys = meta[field] if field == "provider_keys" else [meta[field]]
                for key in keys:
                    if key:
                        buckets[key].append(i)
            for key, indices in sorted(buckets.items()):
                self.apply_bucket(indices, rule, key)

        titles = defaultdict(list)
        for i, meta in enumerate(self.meta):
            if len(meta["title"]) >= 30 and len(meta["title"].split()) >= 5:
                titles[meta["title"]].append(i)
        for title, indices in sorted(titles.items()):
            roots = sorted({self.root(i) for i in indices})
            if len(roots) < 2:
                continue
            if len(roots) > self.block_limit:
                self.large_blocks.append({"rule": "exact_title", "key": title, "record_indices": indices})
                continue
            # A missing DOI must not attach arbitrarily to one of two DOI records.
            title_dois = {self.meta[j]["doi"] for r in roots for j in self.members[r]
                          if self.meta[j]["doi"]}
            buckets = defaultdict(list)
            for i in indices:
                meta = self.meta[i]
                if all(meta[k] for k in ("year", "author", "venue")) and not meta["notice"]:
                    buckets[(meta["year"], meta["author"], meta["venue"])].append(i)
            if len(title_dois) <= 1:
                for key, members in sorted(buckets.items()):
                    self.apply_bucket(members, MATCH_RULES[2], repr(key))
            remaining = sorted({self.root(i) for i in indices})
            for a, b in itertools.combinations(remaining, 2):
                reason = "same_title_but_insufficient_or_conflicting_metadata"
                self.flag(a, b, reason)

    def related_reports(self) -> None:
        doi_index = defaultdict(list)
        for i, meta in enumerate(self.meta):
            if meta["doi"]:
                doi_index[meta["doi"]].append(i)
        for i, row in enumerate(self.rows):
            related = normalize_doi(row.get("related_doi", ""))
            for other in doi_index.get(related, []):
                self.flag(i, other, "related_doi_link_not_an_automatic_duplicate")

    def fuzzy_pass(self, threshold: float) -> dict:
        # Compare all title variants in input, but never remove a fuzzy-only match.
        index = defaultdict(set)
        for i, meta in enumerate(self.meta):
            if len(meta["title"]) >= 30:
                for word in set(meta["title"].split()):
                    if len(word) >= 4:
                        index[word].add(i)
        comparisons = 0
        without_block = 0
        for i, meta in enumerate(self.meta):
            words = sorted(
                (w for w in set(meta["title"].split()) if 1 < len(index.get(w, ())) <= self.block_limit),
                key=lambda w: (len(index[w]), w),
            )[:4]
            if not words:
                without_block += 1
                continue
            candidates = set().union(*(index[w] for w in words))
            for j in sorted(candidates):
                if j == i or self.root(i) == self.root(j):
                    continue
                # Checking both directions avoids losing pairs due to asymmetric blocks.
                comparisons += 1
                score = fuzz.ratio(meta["title"], self.meta[j]["title"], score_cutoff=threshold)
                if score >= threshold:
                    self.flag(i, j, "similar_title_manual_check")
            if (i + 1) % 5000 == 0:
                print(f"  Similar-title check: {i + 1:,}/{len(self.rows):,}", flush=True)
        return {"fuzzy_pair_comparisons": comparisons, "rows_without_eligible_fuzzy_block": without_block}

