# Offline verification — 23 October 2026

Latest recovery result: **170 tests passed, 0 failures, 0 errors, 0 blocked network attempts**. The initial verification passed 98 tests; the Scopus recovery suite passed 122. The latest suite also covers conservative evidence normalization, saved-response revalidation, screening recovery, model-outage stops and Windows source snapshots. Verification used Python 3.12, with real socket connections and DNS disabled before discovery.

```bash
source .venv/bin/activate
python tests/run_offline.py
```

Verification did not launch the real review pipeline. All stage execution inside tests used temporary synthetic records/text, mocked provider responses and fake model responses. No real database requests, PDF downloads or model inference took place. During the initial verification, existing review-run files were hash-compared around read-only readiness checks and remained unchanged; the subsequent `database_review_verified` run was launched by the owner.

## What was checked

- All six adapters: endpoint, authentication placement, query fields, date filters, response normalization and missing-abstract handling.
- Pagination and cached resume for OpenAlex, Scopus, WoS Starter, IEEE and arXiv; ranked/capped Crossref discovery remains explicitly non-exhaustive.
- Missing credentials, provider failures, rate limits, repeated pages/cursors, local caps, retry/resume, credential redaction and frozen input preservation.
- Five manual import formats, large CSV fields, source provenance, conservative deduplication and report-version conflicts.
- Synthetic CLI flow through all stages, separate AI/human decisions, human-confirmed export and resume without repeated fake model calls.
- Complete long-document packet coverage, exact quotation and own-study attribution checks, contradictory evidence, incomplete packets and review gates.
- Local TXT handling, missing/partial full texts, guarded OA URL validation, source-file integrity and date-window checks.
- Read-only `--stage check`, dependency compatibility, credential presence without secret disclosure, no environment mutation, malformed manifests, boolean options and frozen-run fingerprints.

## Fixes made during this verification

1. Crossref now uses four explicit distinct recommendation anchors. The old positional slicing omitted collaborative filtering and repeated item-recommendation searches after the vocabulary expansion.
2. Scopus uses cursor pagination by default and supports `STANDARD`/`COMPLETE` selection. Offset compatibility mode records incomplete retrieval at its 5,000-result limit. These contracts were checked against [Elsevier's published API limits](https://dev.elsevier.com/api_key_settings.html) and [official Scopus API guide](https://supportcontent.elsevier.com/Support%20Hub/DaaS/37062-Scopus_API_Guide_V1_20230907.pdf); no Scopus query was submitted.
3. `--stage check` validates setup without starting stages or creating/fixing a run. The same checks now run before real retrieval initialization, so a blank contact email cannot create another failed frozen run.
4. Selected missing credentials stop startup by default. Incomplete selected-source retrieval preserves successful records and stops before screening; partial coverage requires an explicit `require_all_sources: false` choice.
5. Remote-model opt-in and source-completeness flags require JSON booleans; a string such as `"false"` cannot enable a remote model. Malformed frozen manifests appear as readiness errors.
6. The new-search configuration uses a fresh output path, preserving the earlier empty/failed run and its frozen snapshot.

## What this does not establish

An offline test cannot prove that an API key is active, that a subscription includes cursor/response-view access, that a network/VPN is correctly configured, or that a live provider still returns the mocked contract. Ollama server/model availability was not queried. Key presence is not entitlement verification.

The six tested API adapters are Scopus, WoS Starter, IEEE Xplore, OpenAlex, arXiv and Crossref. The current review configuration enables only **Scopus, OpenAlex, arXiv and Crossref**, as requested; WoS and IEEE remain disabled. ACM Digital Library and backward/forward citation chasing remain manual export/import sources. WoS Starter is not independently verified Core Collection coverage.

Run this safe check after editing your email, keys and selected sources:

```bash
python run_slr.py --config config/review.json --stage check
```

Exit 0 means offline readiness passes; exit 2 means setup still needs attention. Live account access and scientific screening accuracy remain unverified until the owner elects to run the real workflow and calibrate it.

## Recovery from the owner-started Scopus cursor failure

The owner's saved log showed `403 ENTITLEMENTS_ERROR: Use of the cursor parameter is restricted`. The recovery code uses offset pagination and splits searches above 5,000 into disjoint publication-year ranges. It preserves parent probes without adding their duplicate rows to the screening corpus. A single-year overflow or inconsistent partition totals remains unfinished, keeping the strict screening gate intact.

The new `prepare_resume.py` is an entirely offline transfer for a stopped API-only run. It requires the donor's exclusive run lock, verifies frozen code and query identity, checks saved normalized records against raw responses, validates cursor/offset chains, and copies committed pages independently into a new frozen run. It refuses an active donor, an existing destination, changed adapters/searches, malformed or missing pages, and manual imports. Source run data and configuration remain unchanged; only the new run is written. Derived outputs are rebuilt from the copied retrieval evidence.

The recovery configuration passes guarded read-only readiness with all four requested sources selected, zero errors and zero database/model requests. The current provider adapter hash was checked against the original frozen snapshot and is unchanged, so unaffected download identities remain compatible. The active owner process was not stopped or restarted, and migration was tested only with temporary synthetic runs. No live offset entitlement or screening-accuracy claim follows from these tests.

Use [the recovery instructions](restart_after_scopus.md) after stopping the original process yourself. Do not delete the donor before the one-time preparation succeeds.

## Recovery from the supplied screening failures

All three owner-supplied failed title/abstract responses were replayed offline through the integrated cache-revalidation path with model construction and inference forbidden. Matching generation keys, prompts, inputs, model digests and raw replies were preserved. Each became a valid `uncertain` assessment: unsupported claims were downgraded to `unclear`, and no paper was included, excluded or human-confirmed by this repair. The private sample archive remained unchanged and is excluded from the update package.

The 170-test suite covers unique exact passage-ID recovery, unique whitespace-only source spans, rejection of punctuation/paraphrase/ambiguous matches, preserved own-study attribution rules, unchanged strict validation, zero-inference cache recovery, immutable original attempt events, original-run preservation, committed SQLite WAL records, source/artifact tampering, model-outage stops, bounded screening and legacy/new Windows newline handling. Two existing synthetic export fixtures were updated to use a year inside the configured review window; production date criteria were unchanged.

Use [the screening recovery instructions](screening_recovery.md) to prepare a separate run from the Windows donor's frozen configuration and cached evidence, then run a three-assessment trial yourself. No real review pipeline, database search, PDF download or model request was executed during this verification. Live Windows execution and scientific screening accuracy still require the owner's bounded trial and review.
