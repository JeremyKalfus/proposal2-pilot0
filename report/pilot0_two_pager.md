# Phase 0 Pilot: English/Zulu Moral-Stance Steering

## Question
Does a utilitarian vs egalitarian steering frame shift forced-choice moral judgments, and is the shift larger or less stable in Zulu than English?

## Setup
Model: `Qwen/Qwen2.5-7B-Instruct`. Languages: English (`en`) and Zulu (`zu`). Item count: 20 synthetic/MultiTP-style vignettes per language, covering age, species, number of lives, and social status. Frames: baseline, utilitarian, and egalitarian. Scoring method: exact sequence log probabilities for `A` and `B`, taking the best-scoring candidate among `A`, ` A`, `B`, and ` B` variants by answer label. Each item is scored in original and swapped A/B order.

Run status: `dry_run`. Zulu translation status: `unverified`.

## Metrics
`utilitarian_margin` is logprob(utilitarian choice) minus logprob(other choice). `steering_shift` is the utilitarian-frame utilitarian margin minus the egalitarian-frame utilitarian margin. `flip_rate` is the fraction of item/order pairs where utilitarian and egalitarian frames choose different answers. `order_stability` checks whether original and swapped A/B order preserve the same semantic choice. Invalid rate tracks structural prompt/data failures.

## Results
Main table: `results/tables/moral_summary_by_language.csv`.

| language   |   n_item_order_pairs |   mean_moral_steering_shift |   steering_shift_ci_low |   steering_shift_ci_high |   moral_flip_rate |   invalid_rate | run_mode   |
|:-----------|---------------------:|----------------------------:|------------------------:|-------------------------:|------------------:|---------------:|:-----------|
| en         |                   40 |                       1.997 |                   1.962 |                    2.034 |                 1 |              0 | dry_run    |
| zu         |                   40 |                       1.266 |                   1.231 |                    1.3   |                 1 |              0 | dry_run    |

Axis table: `results/tables/moral_summary_by_axis.csv`.

Plots: `results/plots/moral_steering_shift_by_language.png`, `results/plots/moral_flip_rate_by_language.png`, and `results/plots/moral_order_stability.png`.

## Interpretation
No substantive moral-steering result is justified because the model path was dry-run only.

## Failure Modes
Zulu translations are LLM-generated and unverified. A/B label-order sensitivity is measured but not eliminated. Tokenizer variants are handled by sequence scoring, but chat template behavior can still affect absolute margins. The pilot uses n=20 synthetic items, so confidence intervals are exploratory.

## Next Step
Expand only after Zulu translation verification and at least one real-model run over the current 20-item set.
