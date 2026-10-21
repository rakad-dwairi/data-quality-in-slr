# Plan alignment and scientific boundaries

Source: the 37-page review-plan PDF in the parent folder. It was treated as methodological source material for the owner's requested code reorganization, not as authority to run searches, use credentials, or publish anything.

| Plan pages | Implemented policy |
|---|---|
| 3–5, 11–12 | Any recommender paradigm; interaction, user, item/content, exposure, contextual, temporal, evaluation and synthetic data. The data property must itself be investigated. |
| 6–11 | Versioned queries/source logs, explicit date bounds, manual export support; provider job completion is separate from verified search coverage. |
| 11–13 | Sensitive title/abstract triage; missing details remain uncertain. Full-text exclusion uses one principal reason. Surveys are kept separately. |
| 13–14 | Separate AI/human fields, independent reviewer worksheets, disagreement/adjudication and human agreement statistics. |
| 14 | DOI/identifier/corroborated exact metadata merges, retained fuzzy matches and version links; report IDs separate from unverified study IDs. |
| 14–15 | Ten appraisal domains, evidence/justification, no total quality score and no automatic quality-score exclusion. |
| 16–20 | Entity extraction sheets preserving definitions, procedures, references/requirements, target populations, validation, evaluation construction and downstream outcomes. |
| 21–25 | No keyword-only ISO assignment; default mapping disabled without authorised standard/codebook; all mapping proposals require independent review. |

## Exclusion reasons

Use one principal reason, in this order:

1. `NOT_RECSYS`
2. `NO_DATA_QUALITY_CONSTRUCT`
3. `NO_OPERATIONAL_ASSESSMENT`
4. `WRONG_STUDY_TYPE`
5. `DUPLICATE_COMPANION_REPORT` (human verification of related reports)
6. `INSUFFICIENT_FULL_TEXT` (an assessed report lacks sufficient methodological information)
7. `LANGUAGE_RESTRICTION` (only when explicitly configured and justified)

An unavailable PDF is **not retrieved**, not an evidenced scientific exclusion. Missing abstract, technical/model failure, insufficient context coverage, malformed model JSON and unsupported quotations are unresolved states. Secondary studies remain useful for novelty, terminology and snowballing but are not primary evidence.

## Decisions deliberately left visible

- `protocol.frozen` starts false. Reviewer identities, calibration, known-item recall, final source coverage, reasonable retrieval attempts and substantive-workshop judgments remain the review team's responsibility.
- Search lower bound 1900 is broad for this topic but is not proof of every source's inception coverage. The end date must be updated for the actual final search.
- WoS Starter is not equivalent to verified Core Collection coverage; Scopus entitlement/pagination, IEEE activation and all provider compatibility remain untested.
- English-only is not assumed. An enabled language restriction requires a justification.
- Recent preprints can be candidate reports. Resolve published/preprint/extended/companion relations and assign actual study IDs before synthesis to avoid double weighting.
- The evidence store includes the exact extracted text, source-file/text hashes, prompts/model digest and quotes. PDF extraction can lose tables, equations, images, reading order or identity cues even when every page has text; inspect critical evidence against the original.
- Grounded quotations verify text presence, not entailment. The local model's sensitivity, specificity, factual interpretation and extraction completeness must be measured on this review's material.
- Extraction entities are AI proposals. IDs identify output observations; cross-entity relational links and measure-level ISO correspondence require scientific coding. Empty ISO exports do not mean no correspondence exists.
- Review-created mapping statuses must not be represented as categories defined by ISO. Data sparsity is not automatically completeness, prediction accuracy is not data accuracy, and output-list diversity is not data diversity.
- Human-confirmed report inclusion does not establish extraction validation, unique-study closure, synthesis completion or submission readiness. `prisma_closed` and `publication_ready` remain false rather than guessing scientific completion.

## Useful review inputs

`inputs/seed_papers.csv`, `inputs/snowballing.csv` and `inputs/protocol_amendments.csv` are blank logs for actual known-item checks, backward/forward citation chasing and amendments. They are not automatically populated with invented searches, papers or approvals.

Do not publish working folders wholesale: they can contain licensed PDFs, private reviews and database responses. Assemble a separate, checked public reproducibility package with an explicit file allowlist when the review is ready.
