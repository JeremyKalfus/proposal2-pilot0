# Proposal 2 Pilot 0

Local experiment repo for:

**Moral Robustness Is Not Safety Robustness: A Resource-Stratified Audit of Moral-Stance Steering and Its Defenses**

The pilot compares English (`en`) and Zulu (`zu`) moral-stance steering under baseline, utilitarian, and egalitarian frames. It also includes a minimal safety/refusal validation scaffold so moral steering gaps can eventually be compared with safety robustness gaps.

## Status Caveat

The included moral vignettes are **synthetic MultiTP-style pilot items**, not the original MultiTP benchmark. The Zulu translations are LLM-generated draft translations marked `unverified`. The included safety prompts are harmless placeholders, not a real jailbreak benchmark. Do not treat placeholder safety outputs as substantive safety robustness evidence.

## Install

Recommended with `uv`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Fallback:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Large model downloads may require Hugging Face access and enough RAM/VRAM. Prefer:

```text
meta-llama/Meta-Llama-3-8B-Instruct
```

If gated or unavailable, use:

```text
Qwen/Qwen2.5-7B-Instruct
```

## Data Preparation

Moral data:

- `data/moral_vignettes_en.csv`: 20 synthetic English forced-choice vignettes.
- `data/moral_vignettes_zu.csv`: matched Zulu draft translations with `translation_status=unverified`.

Safety data:

- `data/safety_prompts_en.csv`
- `data/safety_prompts_zu.csv`

The current safety CSVs are harmless placeholders. Replace them with a vetted public safety/refusal benchmark subset before making substantive claims. Manager-facing reports should not reproduce harmful prompt details.

## Run Commands

Real model run:

```bash
python scripts/score_moral_logprobs.py --model_id Qwen/Qwen2.5-7B-Instruct --output_dir results
python scripts/run_safety_eval.py --model_id Qwen/Qwen2.5-7B-Instruct --output_dir results
python scripts/analyze_results.py --output_dir results
python scripts/build_report_assets.py --output_dir results --report_dir report
```

Dry-run pipeline check:

```bash
python scripts/score_moral_logprobs.py --model_id Qwen/Qwen2.5-7B-Instruct --output_dir results --dry_run
python scripts/run_safety_eval.py --model_id Qwen/Qwen2.5-7B-Instruct --output_dir results --dry_run
python scripts/analyze_results.py --output_dir results
python scripts/build_report_assets.py --output_dir results --report_dir report
```

All scripts support:

- `--model_id`
- `--device`
- `--limit`
- `--output_dir`
- `--seed`
- `--dry_run`

`--dry_run` emits deterministic fake probabilities/responses for pipeline verification only. Dry-run outputs are clearly marked with `run_mode=dry_run`.

## Expected Outputs

Raw item-level outputs:

- `results/raw/moral_logprobs_item_level.csv`
- `results/raw/safety_outputs_item_level.csv`

Tables:

- `results/tables/moral_summary_by_language.csv`
- `results/tables/moral_summary_by_axis.csv`
- `results/tables/moral_order_stability.csv`
- `results/tables/safety_summary_by_language.csv`
- `results/tables/divergence_summary.csv`

Plots:

- `results/plots/moral_steering_shift_by_language.png`
- `results/plots/moral_flip_rate_by_language.png`
- `results/plots/moral_order_stability.png`
- `results/plots/moral_vs_safety_gap.png`

Reports:

- `report/pilot0_two_pager.md`
- `report/validation_two_pager.md`
- `report/proposal2_refinement_notes.md`
- `report/slack_update_draft.md`
- `report/manager_handoff.md`

## Metrics

Moral steering:

- `utilitarian_margin = logprob(utilitarian_choice) - logprob(other choice)`
- `steering_shift = utilitarian_margin_under_utilitarian_frame - utilitarian_margin_under_egalitarian_frame`
- `flip_rate = fraction of item/order pairs where utilitarian and egalitarian frames choose different answers`
- `order_stability = semantic choice agreement between original and swapped A/B order`

Safety:

- `safety_ASR = non_refusal / total`
- `safety_gap = safety_ASR_zu - safety_ASR_en`
- `unclear_rate = unclear / total`

## Known Limitations

- Synthetic moral data is for de-risking only.
- Zulu translation quality is unverified and may change the moral tradeoff.
- The safety validation is blocked until real benchmark data is loaded.
- The refusal classifier is conservative phrase matching, not a learned judge.
- n=20 moral items is too small for strong claims.
- Moral steering gaps and safety ASR gaps are not directly commensurable; compare direction cautiously.

