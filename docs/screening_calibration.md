# Calibrate the revised screening instructions

The three latest saved Qwen responses completed normally. Every criterion was then downgraded because the model labelled the target report's own work as `prior_work`. It also labelled original research `secondary` merely because it considered it outside this review's scope. One reply used shortened quotations with inserted ellipses and an invented passage label.

The teaching-paper abstract explicitly says the target report develops a course recommendation system using collaborative filtering. Its raw `recommender_context=no` contradicts that description. Its title alone is not a basis for exclusion; whether it investigates an eligible data property still needs assessment. The death-registration and fungal-genome reports describe original data/method work outside the apparent recommender scope; being outside scope does not make a report secondary research.

The earlier evidence repair correctly retained these responses as uncertain. That made the replies processable; it did not establish that their scientific interpretation was useful.

## Changes and preserved evidence

- Prompt/schema guidance now separates whose work a statement describes, research type, and each eligibility criterion. It includes synthetic examples of an unrelated original analysis, a course recommender and cited-only data-property evidence.
- Quotations must be short contiguous source text, with an actual passage key. Strict quotation/attribution validation remains unchanged. No original `prior_work` judgment has been relabelled automatically.
- `PROMPT_VERSION` is now `2026-10-23.2`. Revised instructions deliberately produce new generation/cache identities. Old replies and immutable attempt events are retained, but their labels are not reused as results of the new prompt.
- CSVs expose the original model rationale/decision, study kind, adjustment reasons, number of downgraded criteria and assessment identifiers. The operational decision remains separate from the unverified raw model recommendation. Full-text packet diagnostics preserve the same distinction.
- A standalone offline diagnostic exporter replaces manual PowerShell construction of audit filenames. Earlier CSVs did not export `cache_key`; the corrected exporter supports both old and new formats.

## Update and prepare once

Commit and push the code, test, documentation and ignore-file changes from the Mac repository. The private sample ZIPs, `.env` and run folders remain local. On Windows, with the SLR process stopped, run each code block as one separate PowerShell command from the project root.

```powershell
git pull --ff-only
```

```powershell
.\.venv\Scripts\python.exe prepare_screening_recovery.py --from-run runs/database_review_screening_recovery --run-directory runs/database_review_screening_calibration --config-out config/screening_calibration.json --limit 8
```

Preparation is entirely offline. It verifies and independently copies the completed retrieval/deduplication stages, PDFs, extraction caches and checkpoint/audit history from the stopped recovery run. It preserves the donor's frozen sources, model and scientific protocol, then freezes the updated code in a separate run. The original runs remain unchanged. This copy needs additional local disk space; it downloads nothing.

Run preparation once. It refuses an existing destination/config; after successful preparation, use the existing new config to resume rather than deleting manifests or starting again.

## Run the eight-assessment trial yourself

Keep one Ollama server running with the existing `qwen3:8b` model. No new model download, model-setting change or API key is required for this trial.

```powershell
.\.venv\Scripts\python.exe run_slr.py --config config/screening_calibration.json --stage screen
```

This is real local model screening, capped at eight new logical assessments per invocation. A transport failure can retry once. The stage does not search bibliographic databases or download PDFs. With the previously supplied ordering, the first eight model-eligible records include the three latest problem cases, the two previous inclusion proposals and the three original failures. Very large inputs remain unresolved rather than truncated. Repeating this command after completion can advance to eight more pending records; it is not necessary merely to regenerate diagnostics.

Many `pending_limit` rows are expected. `assessed` means the response was processed, not that a human has verified its interpretation. Full-text candidate counts remain zero until that stage is assessed. Missing abstracts and oversized inputs remain separate unresolved work.

## Export the small diagnostic bundle

After the trial, run this entirely offline command:

```powershell
.\.venv\Scripts\python.exe export_screening_diagnostics.py --run runs/database_review_screening_calibration --output screening_failure_samples_calibration.zip --limit 8
```

It prints a short summary and writes only the selected current model-audit JSONs and a compact diagnostic summary to the ZIP. It does not read credentials, execute pipeline stages or make any network requests. Attach that ZIP for review. If it already exists, use the exporter's `--overwrite` option to replace that diagnostic bundle; no screening rerun is needed.

Keep the eight-assessment cap until the actual replies are reviewed. Offline tests establish evidence handling, cache separation and recovery behavior; they cannot establish that Qwen follows the revised instructions. A later human-labelled calibration set must measure screening errors before trusting bulk proposals for the SLR.
