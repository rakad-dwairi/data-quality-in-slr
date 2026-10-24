# Owner-authorised local completion

An additional [early local review worker](early_local_review.md) produces
two-pass evidence batches before the whole title/abstract stage finishes. It
has separate outputs and does not edit the running primary pipeline.

On 23 October 2026 the owner authorised finishing the four-source review on the Mac, including remaining database searches and local inference. Paid AI API calls remain disabled. This supersedes the earlier request to perform offline checks only.

The working configuration is the private `config/mac_completion.json`. It derives from the original Mac run's frozen configuration, including its inception-to-21-October-2026 search range. Scopus uses offset pagination with year partitioning. Scopus, OpenAlex, arXiv and Crossref are selected; the source plan's other databases and citation chasing remain coverage limitations until independently addressed. Crossref is ranked supplementary discovery, not exhaustive Boolean coverage.

The collection in `runs/mac_completion_20261023` returned 248,927 observations across 216 jobs: Scopus 16,294; OpenAlex 36,076; arXiv 4,557; Crossref 192,000. The OpenAlex pages were already cached. Deduplication removed 173,802 duplicate observations, retaining 75,125 bibliographic reports, including 10,780 possible duplicate pairs that still need review. These are not verified unique studies. All selected jobs returned; the retrieval summary correctly remains non-exhaustive because all 192 Crossref ranked discovery searches reached their 1,000-result cap.

The active screening configuration points to `runs/mac_completion_final_20261023`. Verified offline recovery preserves the completed retrieval/deduplication artifacts, raw responses, model audits and downloaded PDFs from earlier runs. It freezes the current evidence policy and changes local context to 131,072 tokens; different context settings create different generation identities. None of these recoveries repeat database searches. Of the retained reports, 34,567 have abstracts and 40,558 do not. Missing abstracts remain unresolved and cannot establish exclusion.

The task created its own Python 3.12 virtual environment and installed a local Ollama service bound to `127.0.0.1:11434`. The initial automatic MLX model build returned HTTP 501 for structured-output requests, so its pilot stopped without eligibility exclusions. Backend preparation downloaded the explicit `qwen3.5:35b-a3b-q4_K_M` GGUF build. The local `qwen3.5:35b` alias uses those weights with repetition/presence/frequency penalties disabled for repeated literal quotations; the creation request is saved in `local_model_definition.json`. The original tags remain separately available, and `model_backend.json` records the source tag, sampling override and actual custom manifest digest. Digest changes prevent reuse of previous attempts. No cloud model or paid inference provider is configured. The model is kept resident and only one model/request is active at a time. The larger context fits entirely in GPU memory on this Mac and avoids unnecessarily splitting typical complete papers under the conservative byte-based input bound.

Evidence validation preserves original generations. An explicit ellipsis can be resolved only to a unique, ordered span in its known source passage, retaining all intervening original text and recording the change. Unsupported extraction observations and ISO proposals are omitted with their originals audited; unsupported appraisal ratings become unclear. Structural errors still fail validation, and definite eligibility criteria still require their own supported source evidence. Case, punctuation and numbers are not fuzzily repaired. An empty authoritative ISO codebook still prevents model-generated ISO mappings.

The final eight-case pilot returned eight assessed responses, passed the basic retention/secondary/unrelated failure checks and had no report with every criterion unclear. All five plan seed titles were present among deduplicated records. A real complete-PDF test (`Denoising Implicit Feedback for Recommendation`) was processed in one request at the larger context. Five of its six eligibility criteria retained source support; an unsupported operational-assessment quotation kept the report uncertain. This is explicitly not a validated inclusion or a measured screening-accuracy estimate. Prompt revision `2026-10-23.3` asks for short literal phrases and actual methods/results evidence for full-text sufficiency; old prompt generations are not silently reused.

## Coordinator and progress

`complete_local_review.py` coordinates the explicitly authorised local work. Its runtime files are private in `runs/local_completion_service/`:

- `progress.json` and `retrieve.log`: retrieval/deduplication/title-abstract worker.
- `completion_progress.json` and `coordinator.log`: model download, pilot, full-text and writing-pack coordinator.
- `model_download.log`, `ollama.log`: local model service.
- `gguf_download.log`, `backend_preparation.log`, `model_backend.json`: explicit backend preparation and model identity.
- `pilot.log`, `pilot_checks.json`: eight saved-source examples assessed by the new local model. Older model decisions are not imported.
- `seed_title_recall.json`: title-match leads for the five known items in the supplied plan. Missing seed titles stop bulk screening; title matches alone do not independently verify author/year identity or establish calibrated recall.
- `screen.log`, `fulltext.log`: the long model stages.

The coordinator refuses a remote inference endpoint. It stops on pipeline failure, missing known-item titles, or basic pilot failures, including loss of source support for most criteria in any pilot report. Before bulk full-text inference, a bounded full-text preflight must successfully process one available actual PDF; matching checkpoints are reused in the later full-text stage. These pilots are failure checks, not human-labelled accuracy evaluations. Matching checkpoints and downloaded full texts are reusable after a stopped process. The local workers have no scheduled Codex polling or per-paper Codex requests. The chat and later reviews still consume Codex usage.

The worker uses `caffeinate` while it is alive. Keep the Mac powered, online and its lid open. Closing the lid, rebooting or shutting down the Mac can interrupt processing; checkpoints preserve completed work but processes do not automatically restart after reboot.

## Output for writing

After full-text processing returns, the coordinator creates `runs/mac_completion_final_20261023/07_writing_pack/` with:

- `candidate_papers.csv`: source-linked AI full-text inclusion proposals.
- `candidate_references.ris`: reference-manager import file.
- `evidence_tables/`: copied extraction/appraisal/mapping tables, where present.
- `human_confirmed_papers.csv`: only actual recorded human confirmations.
- `status.json` and `README.md`: counts and remaining verification work.

No target of 50–200 is used to select papers. No automated stage fills human-review fields or guarantees publication readiness. Missing abstracts, unavailable/incomplete PDFs, uncertain criteria, report/version linkage, extraction correctness and authorised ISO correspondence remain visible. AI candidate completion is not scientific closure of the SLR.

Verification after adding the coordinator and evidence recovery: **221 offline tests passed, zero failures/errors, zero blocked network attempts**. Synthetic tests cover pilot failure gates, known-item title leads, full-text preflight stops, bounded evidence restoration, honest omission/downgrades, frozen context overrides and writing-pack preservation without invented human inclusion. Live retrieval and local model execution are separate owner-authorised operations, not part of those offline tests. This corpus requires a long local run; no finish time or inclusion count is guaranteed.
