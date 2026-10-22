# Recover screening without repeating searches

**Later update:** the first bounded retry completed, but three new replies exposed confusion between source attribution, research type and review scope. Use [the new calibration instructions](screening_calibration.md) for the next trial. This page records the earlier recovery; its unchanged-prompt cache reuse claim does not apply to the revised generation prompt.

The supplied Windows samples show valid JSON responses rejected because of invented passage IDs, whitespace differences in quotations, unsupported punctuation/text edits, and definite judgements attributed to prior work. These are evidence-validation failures, not an account quota. The repair separates recoverable source anchors from unsupported scientific judgements.

## What changed

- An unknown passage ID is corrected only when the complete literal quote occurs exactly once across the supplied passages.
- Within a valid passage, a whitespace-only match is accepted only for one unique contiguous source span. The stored quotation becomes the exact original substring, with zero-based character offsets and an end-exclusive endpoint in the adjustment audit. Case, punctuation, numbers and all other non-whitespace characters must remain identical. No fuzzy matching or paraphrase repair is used.
- Unsupported or ambiguous citations and definite judgements attributed to prior/unclear work become `unclear`. Attribution is never changed to `own_study`. Valid criteria are retained instead of discarding the entire response.
- The original strict validator still checks the resulting response. Full-text methods still require body-text evidence. Structural errors, incomplete generations, extraction/appraisal errors and unsupported ISO mappings are not silently accepted.
- Saved failed responses can be revalidated from their exact input-bound audit without inference. Original request/finished events and raw responses remain intact; a new `revalidated` event records the operation. Successful generation caches remain reusable because generation prompts, schemas and model settings have not changed.
- CSVs now include `evidence_validation_version` and `evidence_adjustments_json`. A model outage is reported after saving pending results and stops the workflow before downstream work. A bounded `all` run also stops before full-text preparation while title/abstract records remain `pending_limit`.

All three supplied samples were replayed offline through the integrated cache-recovery path. They became valid **uncertain** assessments; none became an included study, exclusion or verified human decision. One retained four supported criteria, another retained one, and the remaining sample retained no definite criteria. This is recovery of evidence handling, not proof of scientific accuracy.

## Apply the update on Windows

Close the SLR process if one is running. Keep the original `runs/database_review_verified` folder, its manifest, PDFs and checkpoints. Keep one Ollama server instance and close other applications using large models. The earlier diagnostics showed DeepSeek 70B resident while this pipeline requests Qwen 8B; do not launch a second `ollama serve` on the occupied port.

Copy/extract the supplied patch ZIP into the existing project root. Its paths are relative to that root; it contains only code, tests and documentation, with no credentials or review data. In PowerShell:

```powershell
Set-Location "C:\Users\Rakad\Documents\GitHub\data-quality-in-slr"
```

If the ZIP is in Downloads, extraction is one command (adjust its location if saved elsewhere):

```powershell
Expand-Archive -LiteralPath "$HOME\Downloads\screening_recovery_patch_20261023.zip" -DestinationPath "." -Force
```

Do not pull a new model or change context/timeout/protocol settings as part of this update. The next stage uses the existing configured model and can reuse matching generation caches.

## Prepare the recovery once, entirely offline

```powershell
.\.venv\Scripts\python.exe prepare_screening_recovery.py
```

This command makes no provider requests, model requests or downloads and executes no pipeline stages. It:

1. Requires the donor's existing exclusive run lock and refuses active/missing donors or existing destinations.
2. Reads the Windows donor's frozen configuration, preserving its date range, sources, protocol and model. It does not copy the Mac's working review configuration.
3. Verifies completed retrieval/deduplication artifact hashes and row counts. Their typed-row digests are preserved as source-attested metadata rather than incorrectly recomputed from string-valued CSVs.
4. Copies those completed stages, model audits/attempts, extraction caches and downloaded PDFs independently. SQLite checkpoints are backed up from a temporary read-only mirror, including committed WAL entries without modifying donor sidecars.
5. Freezes the updated code in `runs/database_review_screening_recovery`, preserves original provenance under `recovery_origin`, and writes `config/screening_recovery.json` with a three-assessment limit. Derived screening/final labels are rebuilt.

The original run is not modified. Original model-attempt files can contain paths back to that run; retain it as the generation audit trail. A preparation report with file hashes is written to `screening_recovery.json` inside the new run.

The helper accepts only this screening/CLI recovery code change set. For legacy Windows snapshots, it can verify the old writer's LF-only snapshot against the exact hash of its uniformly reconstructed CRLF source, recording that verification explicitly. It does not accept arbitrary newline/content edits. New snapshots preserve original bytes.

Preparation is one-time. If the destination/config already exists after successful preparation, skip preparation and use the stage command below. Never delete a frozen manifest to bypass a refusal.

## Run a bounded title/abstract retry yourself

```powershell
.\.venv\Scripts\python.exe run_slr.py --config config/screening_recovery.json --stage screen
```

This is the real screening command: it lists the configured Ollama model and may generate at most three **new logical assessments** (each transport attempt can retry once). Revalidation of saved replies and reuse of successful replies do not use that budget. Generation reuse requires the same installed model digest. There are no bibliographic searches or PDF downloads in this stage.

You may see `revalidated saved response (no inference)` for the supplied failed cases. The stage still traverses and exports the full metadata queue, so many `pending_limit` rows are expected. Recovered `assessed` rows can correctly remain `uncertain`.

Show the short result summary:

```powershell
Import-Csv ".\runs\database_review_screening_recovery\03_title_abstract\screening.csv" | Group-Object screening_status | Select-Object Name, Count
```

If there are new structural/model failures, inspect their `ai_rationale` and adjustment audit before increasing the limit. Full-text work comes after the title/abstract retry is healthy. Missing abstracts, unavailable PDFs, OCR problems and oversized metadata remain separate unresolved work.

To resume the bounded retry, repeat the same `--stage screen` command. Once it is healthy, change only `screening.limit` in the recovery config to zero to process all pending title/abstract work. Changing that operational limit preserves its run identity. After title/abstract processing, run `--stage fulltext` using the same recovery config; cached PDFs/text are reused, but unavailable OA PDFs may be retried. Do not use `--stage all` to recover screening, since that invokes retrieval again.

## Verification and its limits

```powershell
.\.venv\Scripts\python.exe tests/run_offline.py
```

Real socket connections and DNS are blocked before discovery. Tests cover synthetic quote repairs/downgrades, unchanged strict validation, saved-response revalidation with inference forbidden, immutable attempt history, original-run preservation, SQLite WAL backup, source/artifact tamper detection, bounded stages, model-outage stops and Windows newline compatibility. They do not establish live Windows performance or the scientific accuracy of model screening.

The supplied audit timings reported approximately 60–247 seconds loading Qwen and 6–7 seconds generating each of the three responses. These are per-call observations; they do not establish why loading was slow. Ollama documents these fields as nanosecond durations in its [chat API reference](https://docs.ollama.com/api/chat).
