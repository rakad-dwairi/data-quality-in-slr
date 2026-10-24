# RecSys data-quality SLR

The owner-authorised Mac completion is described in [local completion](docs/local_completion.md). It uses the saved search caches and a loopback-only model; the long processing runs in Python without per-paper Codex requests.

The active Mac run now uses a [faster pending-only continuation](docs/faster_local_continuation.md), retaining original completed assessments and full-text reviews. Its resume command and model comparison are documented there.

For the latest Windows screening samples, use [the eight-assessment calibration](docs/screening_calibration.md). The saved replies confuse a paper's own work, research type and topical eligibility; the revised prompt needs a new bounded trial. Completed searches, PDFs and audit history are retained. The [earlier screening recovery](docs/screening_recovery.md) documents the first repair. Do not rerun database retrieval for either repair.

Run **`run_slr.py`** to collect/import records, deduplicate, screen titles/abstracts, obtain available full texts, screen complete documents, and export evidence tables. The workflow follows the scope of the PDF in the parent folder. It never aims for a predetermined number of included papers.

**Verification status (23 October 2026):** the offline suite has been run with real network access blocked. It covers synthetic CLI stages, all six provider request/response contracts, pagination, resume, extraction/review gates and configuration checks. The real review, database queries, paper downloads and model inference have **not** been run during verification. Live account/API entitlements and scientific screening accuracy remain unverified. See `docs/offline_verification.md`.

## Folder map

```text
run_slr.py                  The one command-line entry point
config/review.json          New multi-database search and local screening
config/review_resume.json   Scopus offset recovery, reusing current downloads
prepare_resume.py           Offline copy from a stopped run into a fresh run
config/existing_records.json Import old raw records, with no database searches
slr_pipeline/               Implementation, split into readable modules
inputs/imports/             Your Scopus/WoS/ACM/etc. exports
inputs/fulltext/            Your private PDFs or complete UTF-8 TXT copies
inputs/*.example.csv        Blank input formats
tests/                     Offline tests supplied for you to run
runs/                      Created by you when you run the workflow
archive/legacy_scripts/    Original competing scripts, preserved
archive/previous_runs/     Original retrieval/screening results, preserved
archive/preservation_manifest.json  Original file sizes and SHA-256 hashes
```

The PDF, existing virtual environment and IDE settings remain in place. The archive is historical evidence, not the active pipeline; old internal relative paths may require adjustment if you choose to reuse an archived script. New imports deliberately ignore old screening labels and apply the new protocol from the raw metadata.

## Set up once

Python 3.10+ is required. In Terminal:

```bash
cd "/Users/macbookpro/Workspace/Research/SLR/Data Quality in RecSys/recsys-slr-search"
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If that virtual environment is unavailable, create it first with `python3 -m venv .venv`.

Install/start [Ollama](https://ollama.com/) locally, then download the configured model:

```bash
ollama pull qwen3:8b
```

Keep the Ollama desktop app running, or run `ollama serve` in another terminal. The initial model is the same model family used in the old scripts; it is **not a scientifically validated choice for this review**. You can choose another installed model in the JSON before starting a new run. Hardware speed and memory determine runtime; 25,000 records can take substantial time.

Inference uses the local Ollama `/api/chat` endpoint with a JSON schema, as described in the [official structured-output documentation](https://docs.ollama.com/capabilities/structured-outputs). No paid model API is configured. Database providers can still apply their own quotas/charges. Remote model URLs require an explicit `model.allow_remote` change and would send paper text to that endpoint.

## Choose your input and run

**For a new database search**, edit `config/review.json` first:

- Put your contact address in `search.email`.
- Set the final search date in `search.end`; the supplied date is 2026-10-21.
- Select your licensed/available `search.sources`. The supplied configuration currently enables **Scopus, OpenAlex, arXiv and Crossref**; WoS and IEEE adapters remain available but disabled. `require_all_sources: true` stops before starting if a selected credential is missing, and stops before screening if retrieval fails or remains unfinished. Successful retrieval records are preserved for resume. Set this flag to `false` only if you intentionally want to proceed with documented partial coverage.
- Add your provider credentials privately as environment variables or in a local `.env` file. The supported names are in `.env.example`; do not put secrets in the JSON. Existing environment variables take precedence over `.env`.
- Select `run_directory` for this review. Keep that same directory to resume it.

To create `.env` without replacing an existing one:

```bash
test -f .env || cp .env.example .env
```

Then edit `.env` privately. Do not paste your keys into this chat. Check readiness first; this command makes **no database/model/DNS requests** and creates no run artifacts:

```bash
python run_slr.py --config config/review.json --stage check
```

It reports selected sources, query counts, dependency versions, credential presence (never values), and configuration/frozen-run conflicts. Exit code 0 means the offline setup passes; exit code 2 means settings need correction. It cannot establish whether a key is active or an account has the required API entitlement. When ready, run:

```bash
python run_slr.py --config config/review.json --stage all
```

`max_pages: 0` has no local page cap; provider limits still apply. Crossref is capped ranked discovery, not an exhaustive Boolean search. The broad `1900-01-01` lower bound replaces the old arbitrary trailing-ten-year default; it is a declared search bound, not independently verified inception coverage. Exact queries and source-specific limitations are recorded.

Scopus supports `scopus_pagination: "cursor"` when the account permits it. For the current account's restricted-cursor error, use the separate `config/review_resume.json` recovery configuration and [restart instructions](docs/restart_after_scopus.md). Its `"offset"` mode splits searches over 5,000 results into disjoint publication-year ranges, preserving probe evidence and exhausting the smaller ranges. A single year exceeding the limit stays explicitly unfinished. `scopus_view: "STANDARD"` preserves the conservative entitlement choice; use `"COMPLETE"` only if access permits it. See the [Elsevier API limits](https://dev.elsevier.com/api_key_settings.html) and [PUBYEAR syntax](https://dev.elsevier.com/sc_search_tips.html).

**To start with the records already in this folder**, no database credentials are needed:

```bash
python run_slr.py --config config/existing_records.json --stage all
```

That configuration imports the preserved multi-source and IEEE raw observations. It does not import prior AI labels. It still uses local Ollama and attempts available OA PDF downloads; set `fulltext.download_open_access` to `false` before starting if you want only local full texts.

**To add database exports**, copy CSV, RIS, BibTeX, JSON or JSONL files into `inputs/imports/`. Add entries such as the following to `imports`, with a source label identifying the actual export:

```json
{"source": "acm_dl_manual", "path": "inputs/imports/acm_export.ris"}
```

Use `search.enabled: false` for imports only. Scopus/WoS exports containing abstracts are especially useful: some API responses lack abstracts, and the WoS Starter adapter does not populate them. ACM and citation-chasing are manual imports. WoS Starter access is not evidence of a complete Web of Science Core Collection search.

## Outputs, in order

Each run creates numbered folders:

| Stage | Main CSVs | Meaning |
|---|---|---|
| `01_retrieval` | `records.csv`, `search_log.csv` | Original observations; exact searches, caps and failures |
| `02_deduplication` | `records.csv`, `duplicates_removed.csv`, `record_mapping.csv`, `possible_duplicates.csv`, `version_links.csv` | Unique report queue, reversible merges and unresolved related reports |
| `03_title_abstract` | `screening.csv`, `excluded_provisional.csv`, `secondary_literature.csv`, `full_text_queue.csv` | Title/abstract proposals, criterion evidence and routing |
| `04_full_text` | `retrieval_log.csv`, `fulltext_needed.csv`, `fulltext_manifest.csv`, `screening.csv`, `excluded_provisional.csv` | Full-text availability, detailed evidence and proposals |
| `05_extraction` | `studies.csv`, `datasets.csv`, `data_objects.csv`, `measures.csv`, `validations.csv`, `outcomes.csv`, `quality_appraisal.csv`, `iso_mapping_candidates.csv` | Grounded AI extraction proposals for eligible candidates |
| `06_final` | `included_candidates.csv`, `final_included_papers.csv`, `pending_review.csv`, `human_confirmed_exclusions.csv`, `date_checks_needed.csv` | Automated shortlist, human-confirmed reports, and outstanding work |

`06_final/flow_counts.json` and `summary.json` distinguish provisional automation counts from human decisions. They do not pretend the PRISMA flow or study count is scientifically closed. A report and a study are different: preprints, conference versions and extensions may be related without being identical reports.

The AI shortlist is `included_candidates.csv`. `final_included_papers.csv` requires independently confirmed full-text inclusion and a date inside the search window. It can correctly be empty while human review is pending. No model-generated response counts as a human review.

## Missing PDFs and long papers

Where `oa_pdf_url` exists, the workflow tries that public PDF URL, subject to size/type/network checks. It does not bypass paywalls. For unavailable papers, obtain an authorised full text and place it at:

```text
inputs/fulltext/<record_id>.pdf
```

Alternatively supply a complete UTF-8 `.txt` file with that name, or copy `inputs/fulltext_manifest.example.csv` to `inputs/fulltext_manifest.csv` and fill `record_id,path,doi,coverage`. Paths are project-relative or absolute. Use `coverage=partial` for fragments. Exact IDs or a unique DOI match are required; titles are never guessed.

Scanned/unreadable/partial PDFs remain pending. Supply OCR or a better copy and verify that it contains the full report. Long documents are processed in bounded packets covering the entire extracted text; packet-level decisions do not become whole-report decisions. A report remains uncertain if any required packet fails or is still pending, or if evidence contradicts itself. Extraction and quotation checks cannot guarantee that a PDF is the correct report or that interpretation is scientifically sound.

After adding full texts:

```bash
python run_slr.py --config config/review.json --stage fulltext
```

Use `config/existing_records.json` instead if that is your active run.

## Resume and individual stages

Repeat the same command to resume. Completed retrieval pages, full-text extraction and successful model calls are cached. A changed installed model digest causes new model assessments. Interrupting with Ctrl-C retains committed work; an in-flight call may need repeating.

Available stages: `check`, `plan`, `retrieve`, `deduplicate`, `screen`, `fulltext`, `export`, `status`, `all`. `check` is read-only setup validation. `plan` saves configuration/code provenance without database/model/download calls after readiness succeeds. `status` reads existing output. `export` incorporates human decisions without inference or downloads.

```bash
python run_slr.py --config config/review.json --stage plan
python run_slr.py --config config/review.json --stage status
```

Upstream reruns mark downstream outputs stale until regenerated. Import paths/content, protocol, model configuration and code are frozen per run; choose a **new `run_directory`** if these change. Restore a changed input file to resume its original run. `screening.limit`, `screening.retry_errors` and `search.max_pages` can change within a run. A nonzero screening limit bounds model attempts for each invoked stage; long-document packets also count. Zero means process all available work. Identical successful cached calls do not consume this attempt limit.

For the Scopus fix prepared while the original search was active, stop the original process yourself, then use `prepare_resume.py` to copy compatible downloaded pages into the new frozen run. Do not delete the original before preparation. The helper validates locks, adapter/query identity, raw response normalization and cursor chains, then makes independent byte-identical copies; it does not make database or model requests. See [the exact commands and recovery behavior](docs/restart_after_scopus.md).

Generated stage inputs have verified file hashes. Do not edit their CSVs or `assessments.json` in place; use the files under `inputs/` for reviews, date checks and full-text mappings. A changed generated input stops downstream processing until restored or regenerated.

Final exports also recheck the original full-text file hash. Replaced or missing PDFs/TXT copies return to the pending queue until you rerun full-text screening; old human labels do not silently certify changed evidence.

## Human review and scientific coding

Each screening folder has two blind `reviewer_*_template.csv` files containing title, abstract and full-text path where available. Copy these elsewhere before completing them: generated templates are overwritten. Fill them independently with real reviewer IDs, ISO timestamps with timezone, and `include`, `exclude`, `secondary` or `uncertain`. Keep `stage,record_id,input_sha256` unchanged. Combine completed rows into `inputs/reviews.csv`, using the header in `inputs/reviews.example.csv`.

Two distinct reviewers must agree. Exclusions require the same valid principal reason and a justification. For disagreement, a distinct third reviewer may add `role=adjudicator` with a decision and rationale. Reviewer IDs are normalized for independence checks. The file records decisions you actually made; never fill it with simulated human labels. Re-export:

```bash
python run_slr.py --config config/review.json --stage export
```

If human TA review reinstates an AI-excluded record, run `--stage fulltext` to process it. Human agreement statistics are exported separately; AI agreement is not reviewer reliability. Unknown/boundary/conflicting dates remain in `date_checks_needed.csv`; for assessed full texts, copy `date_checks_template.csv` to `inputs/date_checks.csv` and document the verified window status, source evidence, named reviewer and timestamp. Re-export to apply it.

The ten appraisal domains stay separate, with evidence and no arbitrary aggregate score. Extracted entities require human checking and relational coding: dataset→object→measure→validation/outcome links and companion-report study IDs are not inferred as verified facts. Use `fields_json` to retain additional properties beyond the convenience columns. ISO mapping is disabled by default because no authorised ISO codebook is supplied; any enabled mapping remains provisional until independently coded against authorised clauses. See `docs/protocol.md` for the plan-to-code decisions and remaining scientific work.

Private PDFs, full-text caches, credentials and working review data must stay private; there is no publication/upload command in this tool.

## Offline checks you can run

```bash
python tests/run_offline.py
```

This runner blocks real socket connections and DNS before test discovery. Tests use synthetic records, temporary files, mocked database responses and a fake local model; they are not evidence of live access or screening accuracy. Run a real calibration sample, compare false exclusions with independent human labels, and document any protocol amendments before the definitive review. No top-tier publication outcome can be guaranteed by this pipeline.
 this
