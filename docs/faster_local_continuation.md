# Faster local continuation

Two compact alternatives were tested on saved material: Qwen3.5 4B and Qwen3
4B Instruct. They were rejected for quality/validation failures. No compact
model was adopted just because it was smaller.

The existing GGUF Qwen3.5 35B MoE, with concise source-ID screening responses,
passed a bounded twelve-case failure check. It retained the potential primary
studies, recognized the secondary controls, and excluded the unrelated controls.
Its median model request took 7.041 seconds, against the earlier 11.5-second
baseline. This suggests about 39% less model time for title/abstract screening;
whole-run speed depends on documents, extraction and full-text review.

This is a small operational calibration, not a population accuracy estimate.
The original plan, eligibility criteria, source corpus and original completed
assessments stay fixed. The concise interface changes answer presentation and
citation selection. Full text still uses the original strong model and original
blinded review prompts. Two prompts on one model can share errors.

The initial quality gate incorrectly required every secondary control to have
an operational decision of `secondary`. The unchanged core prioritizes other
explicit failed criteria, and can assign `exclude` to a correctly classified
secondary report. The corrected gate checks `ai_study_kind=secondary`, supported
`primary_research=no`, and refusal of primary inclusion. The original reports,
responses and a separate offline correction record are retained. A synthetic
regression test reproduces this exact core priority. The compact-model rejection
still holds after this correction.

`continue_local_review.py` prepares a separate checkpoint continuation only
from stopped source workers and a passed, provenance-verified quality check.
It copies the saved collection and records an exact retention ledger. The
pending-only helper never passes retained IDs back through the model. It also
recovers committed candidate checkpoints when a previous batch was interrupted
before the combined snapshot was written. The tested model digest is pinned.
The original seven early full-text reports and completed partial passes are
retained, including unresolved reports; missing sources stay unresolved.

The active continuation folder contains `continuation_origin.json`, its frozen
configuration and coordinator sources, original model audit references, per-stage
CSV/JSON outputs and `evidence_review/writing_pack`. The writing pack lists strict
two-pass AI candidates separately from unresolved cases; it does not invent human
confirmations or force a total of 50–200 papers.

Resume only with the continuation coordinator, using the project virtualenv:

```sh
.venv/bin/python -u continue_local_review.py --run runs/mac_fast_remaining_20261024
```

Only one coordinator may own that run. The standard `run_slr.py --stage screen`
command does not implement the mixed-model retention ledger. Do not use it to
resume this specific continuation. No retrieval/search stage is called by the
continuation. New authorized OA full-text downloads and local inference are
performed only for still-pending reports, and completed model results are cached.

Run synthetic checks without any database or model connections:

```sh
.venv/bin/python tests/run_offline.py
```
