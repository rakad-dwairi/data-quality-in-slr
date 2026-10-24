# Early local AI review

The owner authorised `early_local_review.py` on 24 October 2026. It runs beside
the original Mac pipeline and reads its SQLite checkpoints in read-only mode.
It does not edit the primary worker, coordinator, configuration, library or
checkpoint database. No search adapters or paid/cloud model are invoked.

It prioritises title/abstract inclusion proposals, uncertain reports with at
least four supported criteria, and the plan's five known-item title leads from
the existing collection. Known items do not receive invented TA decisions.
Title matches do not establish report identity. Only saved OA PDF links enter
this early queue; every other record stays in the original pipeline. Processing
order does not determine the final study set.

Every usable complete PDF receives full-text assessment/extraction and a second
eligibility-only pass. The second pass receives no first decision, rationale,
extraction or TA recommendation. Both use the resident local `qwen3.5:35b`, with
distinct audited prompts/cache identities. The source-ID adapter gives each exact source span a label; models select labels and the code inserts the original text. The actual wire prompts, schema and raw ID response remain in the audit alongside the resolved validation representation. It does not rewrite scientific judgements or weaken the unchanged validator. These are two blinded prompts on the
same model, not independent models or human reviewers. The second pass omits
duplicate extraction; the pack uses first-pass extraction/appraisal.

The GPU has one request slot. The extra work can slow overall TA throughput;
it brings the first papers forward rather than promising a shorter full run.
Budget accrues at 15% of the primary baseline (11.5 seconds per completed
record), plus 180 startup seconds. One report can overshoot this soft allowance
because active calls are not cancelled. A stalled primary run earns no more
budget. Long texts are deferred to main-pipeline packet processing, without
truncation. OA downloads time out after 30 seconds per request. Missing sources
remain unresolved; a model error stops this worker for inspection. It stops
starting reports when the main TA stage finishes, leaving bulk FT to the
existing coordinator. Completed results and PDFs are cached separately and
are not automatically merged into the active primary database.

## Output

`runs/mac_early_review_evidence_20261024/writing_pack/` updates after each paper:

- `early_candidates.csv` / `.ris`: both passes propose inclusion, every required
  criterion is supported, text is complete and the date is within the window.
- `reviewed_papers.csv`: all processed reports, PDF and audit paths.
- `needs_review.csv`: disagreements, uncertainty, unavailable/long texts and
  date checks. Agreed non-inclusions remain provisional.
- `criterion_evidence.csv`: both passes' criteria and source quotations.
- `extracted_observations.csv` / `appraisal.csv`: unverified first-pass proposals.
- `status.json` / `README.md`: counts and methodological limitations.

No human approvals, final study count or authorised ISO mappings are invented.
Identity/version linkage, exclusion errors, extraction and final inclusion need
verification before publication. Early batches are a prioritised incomplete
subset and cannot be presented as the final synthesis.

Progress: `runs/mac_early_review_evidence_20261024/progress.json`.
Log: `runs/local_completion_service/early_review_evidence.log`.
The worker has a separate OS lock, frozen identity and per-report audit. Keep
the Mac powered, online and with its lid open. `caffeinate` does not provide
automatic restart after reboot. There is no per-paper Codex request or scheduled
Codex polling; this chat still consumes Codex usage.

To resume only after this worker has stopped, from the repository root:

```bash
.venv/bin/python -u early_local_review.py \
  --from-run runs/mac_completion_final_20261023 \
  --output runs/mac_early_review_evidence_20261024 --watch
```

Completed results/checkpoints are reused. Inspect `needs_attention` before
retrying failed model assessments; existing failures are not auto-regenerated.

Validation: **237 offline tests passed, zero failures/errors and zero blocked
network attempts**. Sixteen new synthetic checks cover read-only access,
priorities, known-item provenance, budget, prompt restoration, blinded inputs,
evidence requirements, missing texts, exports and frozen local-only setup.
These tests do not measure model screening accuracy.

The initial diagnostic helper is preserved in `runs/mac_early_review_20261024`. It was stopped after its third report checkpoint, while the primary process kept running. Its PDFs/text caches were copied into the revised helper; its decisions were not imported. `budget_carryover.json` charges those diagnostic reviews against the revised worker's paced budget.
