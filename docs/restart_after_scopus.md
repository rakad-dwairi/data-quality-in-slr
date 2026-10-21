# Restart after the Scopus cursor restriction

The original run `runs/database_review_verified` used Scopus cursor pagination and received `403 ENTITLEMENTS_ERROR: Use of the cursor parameter is restricted`. The fix does not require changing credentials. It cannot grant additional API entitlements; the next owner-started run will establish whether the account permits ordinary offset searches.

`config/review_resume.json` enables the same Scopus, OpenAlex, arXiv and Crossref searches, dates, topics, contact email and local model. WoS and IEEE remain disabled. It changes Scopus pagination to `offset` and selects the fresh output directory `runs/database_review_resumed`. The original configuration and run directory remain preserved.

## Commands

1. In the terminal running the original search, press **Ctrl+C once**. Wait for the interruption message and the shell prompt. Preparation refuses to proceed if that process still holds its run lock.
2. Keep the original run directory in place. In your project terminal, run:

```bash
cd "/Users/macbookpro/Workspace/Research/SLR/Data Quality in RecSys/recsys-slr-search"
.venv/bin/python prepare_resume.py \
  --from-run runs/database_review_verified \
  --config config/review_resume.json && \
.venv/bin/python run_slr.py --config config/review_resume.json --stage all
```

The `&&` starts the workflow only if offline preparation succeeds. Preparation prints the number of copied pages and retrieval observations. It never contacts databases, downloads papers or calls a model. The following `run_slr.py --stage all` command is the real workflow and makes the remaining selected database requests, then uses the configured local model.

3. To resume this new run after any later interruption, repeat **only**:

```bash
.venv/bin/python run_slr.py --config config/review_resume.json --stage all
```

Preparation is a one-time transfer. It refuses to overwrite an existing destination. If preparation already succeeded, run the command in step 3; do not delete the destination to repeat preparation.

Optional read-only setup validation after preparation:

```bash
.venv/bin/python run_slr.py --config config/review_resume.json --stage check
```

## What is reused

Completed OpenAlex, arXiv and Crossref response pages are copied independently, preserving their exact bytes and relative evidence paths. A partial query continues at its next saved cursor/offset; a complete query uses its cache. An in-flight response that was never committed may need fetching once. Searches that had not started still need their first requests.

Scopus had no successful pages in the observed run. Its changed pagination uses a separate cache and starts afresh. Failed requests are not treated as completed evidence. `cache_migration.json` in the new run records the donor fingerprint, each copied page's hash and the total reused observations. Derived CSVs, deduplication decisions, screening assessments and final outputs are rebuilt using the fixed code.

Keep the original directory as the previous audit trail. The new cached pages are independent copies, but deleting the source removes the original logs and frozen implementation. Never delete it before preparation succeeds. No automatic cleanup or deletion is performed.

## Scopus behavior

Each topic starts with an ordinary offset query. If its reported total exceeds 5,000, the workflow preserves that probe and recursively splits the publication-year interval into two disjoint halves. Only leaf-query records enter the retrieval CSV, avoiding duplicate inclusion of probe results. Every split, exact fielded query and partial result is logged under `01_retrieval/scopus_partitions/`.

Offset requests stay inside Scopus's [5,000-result limit](https://dev.elsevier.com/api_key_settings.html). The year filters use the official numeric [PUBYEAR operators](https://dev.elsevier.com/sc_search_tips.html). A single year above the limit, inconsistent totals, a failed page or a missing/repeated page remains unfinished. With `require_all_sources: true`, these conditions stop the workflow before screening and preserve successful pages for resumption.

## Offline verification

```bash
.venv/bin/python tests/run_offline.py
```

Tests use temporary synthetic records and mocked responses, with real socket and DNS access blocked before discovery. They cover Scopus range splitting, pagination, interrupted/cached resume, a clean cache transfer, active-run refusal and tamper detection. Offline success verifies code behavior; it does not prove live subscription entitlement or scientific screening accuracy.
