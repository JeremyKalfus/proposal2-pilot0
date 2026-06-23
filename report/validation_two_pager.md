# Validation Study: Moral Robustness vs Safety Robustness

## Question
Do moral-steering gaps and safety/jailbreak gaps diverge across English and Zulu?

## Setup
Same configured model: `Qwen/Qwen2.5-7B-Instruct`. The safety files currently contain harmless placeholders because no local public safety/refusal benchmark subset was available during repo construction. Classification uses a conservative phrase-based refusal detector with an `unclear` class. Manager-facing reports redact prompt details and do not reproduce harmful content.

## Results
Safety table: `results/tables/safety_summary_by_language.csv`.

| language   |   n_prompts |   non_refusal_count |   refusal_count |   unclear_count |   safety_ASR |   unclear_rate | run_mode   | benchmark_status    |
|:-----------|------------:|--------------------:|----------------:|----------------:|-------------:|---------------:|:-----------|:--------------------|
| en         |           4 |                   4 |               0 |               0 |            1 |              0 | dry_run    | placeholder_blocked |
| zu         |           4 |                   4 |               0 |               0 |            1 |              0 | dry_run    | placeholder_blocked |

Divergence table: `results/tables/divergence_summary.csv`.

|   english_moral_steering_score |   zulu_moral_steering_score |   moral_gap |   english_safety_ASR |   zulu_safety_ASR |   safety_gap | qualitative_divergence_interpretation                              |
|-------------------------------:|----------------------------:|------------:|---------------------:|------------------:|-------------:|:-------------------------------------------------------------------|
|                          1.997 |                       1.266 |      -0.732 |                    1 |                 1 |            0 | inconclusive: dry-run mock outputs verify pipeline structure only. |

Plot: `results/plots/moral_vs_safety_gap.png`.

## Interpretation
inconclusive: dry-run mock outputs verify pipeline structure only.

## Limitations
The safety validation is not substantive while the CSVs are placeholders. The refusal classifier is intentionally conservative and can miss subtle non-refusals. Moral and safety metrics are not directly commensurable. Zulu translation quality remains a first-order measurement caveat.

## Next Step
Replace placeholder safety CSVs with a vetted public benchmark subset translated under the same controls, then rerun the same scripts. If positive after a real benchmark run, expand to six languages. If null, reframe the project around whether moral brittleness tracks general low-resource robustness.
