#!/usr/bin/env python3
"""Build Pilot 0 Markdown reports from generated result tables."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=None, help="Accepted for run-command symmetry; report reads model_id from manifest/tables.")
    parser.add_argument("--device", default="auto", help="Accepted for run-command symmetry; unused.")
    parser.add_argument("--limit", type=int, default=None, help="Accepted for run-command symmetry; unused.")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--report_dir", default="report")
    parser.add_argument("--seed", type=int, default=7, help="Accepted for run-command symmetry; unused.")
    parser.add_argument("--dry_run", action="store_true", help="Accepted for run-command symmetry; report reads run_mode from tables.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def first(df: pd.DataFrame, col: str, default="n/a"):
    if df.empty or col not in df.columns:
        return default
    return df.iloc[0].get(col, default)


def row_for_language(df: pd.DataFrame, language: str) -> pd.Series:
    if df.empty or "language" not in df.columns:
        return pd.Series(dtype=object)
    rows = df[df["language"] == language]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def unique_join(values) -> str:
    vals = sorted({str(v) for v in values if str(v) and str(v) != "nan"})
    return ";".join(vals) if vals else "n/a"


def load_manifest(output_dir: Path) -> dict:
    path = output_dir / "run_manifest.json"
    if not path.exists():
        return {"commands": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"commands": []}


def hardware_summary() -> str:
    mem = "unknown memory"
    try:
        bytes_out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        mem = f"{round(int(bytes_out) / (1024 ** 3), 1)} GiB RAM"
    except Exception:
        pass
    return f"{platform.platform()} ({platform.machine()}, {mem})"


def gpu_summary() -> str:
    try:
        import torch

        parts = [
            f"torch={torch.__version__}",
            f"cuda_available={torch.cuda.is_available()}",
        ]
        if getattr(torch.backends, "mps", None) is not None:
            parts.append(f"mps_available={torch.backends.mps.is_available()}")
        return ", ".join(parts)
    except Exception as exc:
        return f"torch unavailable: {exc!r}"


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return len(list(csv.DictReader(f)))


def translation_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    statuses = [r.get("translation_status", "") for r in rows if r.get("translation_status", "")]
    return unique_join(statuses) if statuses else "not_applicable"


def write_manifest(output_dir: Path, entry: dict) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = load_manifest(output_dir)
    manifest.setdefault("commands", []).append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No table available."
    subset = df[[c for c in columns if c in df.columns]].copy()
    for col in subset.columns:
        if pd.api.types.is_numeric_dtype(subset[col]):
            subset[col] = subset[col].map(lambda v: fmt(v))
    return subset.to_markdown(index=False)


def main() -> int:
    start = time.time()
    args = parse_args()
    root = repo_root()
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if not report_dir.is_absolute():
        report_dir = root / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    data_dir = root / "data"

    moral_lang = read_csv(tables_dir / "moral_summary_by_language.csv")
    moral_axis = read_csv(tables_dir / "moral_summary_by_axis.csv")
    moral_order = read_csv(tables_dir / "moral_order_stability.csv")
    safety = read_csv(tables_dir / "safety_summary_by_language.csv")
    divergence = read_csv(tables_dir / "divergence_summary.csv")
    moral_raw = read_csv(output_dir / "raw" / "moral_logprobs_item_level.csv")
    safety_raw = read_csv(output_dir / "raw" / "safety_outputs_item_level.csv")
    manifest = load_manifest(output_dir)

    div_row = divergence.iloc[0].to_dict() if not divergence.empty else {}
    moral_run_mode = div_row.get("moral_run_mode", unique_join(moral_raw.get("run_mode", [])))
    safety_run_mode = div_row.get("safety_run_mode", unique_join(safety_raw.get("run_mode", [])))
    safety_status = div_row.get("safety_benchmark_status", unique_join(safety_raw.get("benchmark_status", [])))
    dry_run = "dry_run" in f"{moral_run_mode};{safety_run_mode}"
    safety_blocked = "placeholder_blocked" in str(safety_status)
    if args.model_id:
        model_id = args.model_id
    elif not moral_raw.empty and "model_id" in moral_raw.columns:
        model_id = unique_join(moral_raw["model_id"])
    else:
        model_id = unique_join([c.get("model_id", "") for c in manifest.get("commands", []) if c.get("model_id")])

    commands = manifest.get("commands", [])
    seen_commands = set()
    command_lines = []
    for entry in commands:
        key = (entry.get("command", "n/a"), entry.get("status", "unknown"))
        if key in seen_commands:
            continue
        seen_commands.add(key)
        command_lines.append(f"- `{key[0]}` ({key[1]})")
    blocked_errors = [entry.get("error", "") for entry in commands if entry.get("status") == "blocked" and entry.get("error")]
    if blocked_errors:
        dependency_issues = "; ".join(blocked_errors)
    elif dry_run:
        dependency_issues = "real model run not completed; dry-run used"
    else:
        dependency_issues = "none recorded"
    if dry_run:
        dependency_issues = (
            f"{dependency_issues}; Qwen 7B fallback was not loaded for the full run on this 16 GiB MPS-only machine"
        )
    total_runtime = sum(float(entry.get("duration_seconds", 0) or 0) for entry in commands)

    en = row_for_language(moral_lang, "en")
    zu = row_for_language(moral_lang, "zu")
    safety_en = row_for_language(safety, "en")
    safety_zu = row_for_language(safety, "zu")
    interpretation = div_row.get("qualitative_divergence_interpretation", "n/a")
    status_prefix = "BLOCKED" if dry_run or safety_blocked else "DONE"

    pilot_report = f"""# Phase 0 Pilot: English/Zulu Moral-Stance Steering

## Question
Does a utilitarian vs egalitarian steering frame shift forced-choice moral judgments, and is the shift larger or less stable in Zulu than English?

## Setup
Model: `{model_id}`. Languages: English (`en`) and Zulu (`zu`). Item count: 20 synthetic/MultiTP-style vignettes per language, covering age, species, number of lives, and social status. Frames: baseline, utilitarian, and egalitarian. Scoring method: exact sequence log probabilities for `A` and `B`, taking the best-scoring candidate among `A`, ` A`, `B`, and ` B` variants by answer label. Each item is scored in original and swapped A/B order.

Run status: `{moral_run_mode}`. Zulu translation status: `{translation_status(data_dir / 'moral_vignettes_zu.csv')}`.

## Metrics
`utilitarian_margin` is logprob(utilitarian choice) minus logprob(other choice). `steering_shift` is the utilitarian-frame utilitarian margin minus the egalitarian-frame utilitarian margin. `flip_rate` is the fraction of item/order pairs where utilitarian and egalitarian frames choose different answers. `order_stability` checks whether original and swapped A/B order preserve the same semantic choice. Invalid rate tracks structural prompt/data failures.

## Results
Main table: `results/tables/moral_summary_by_language.csv`.

{markdown_table(moral_lang, ['language', 'n_item_order_pairs', 'mean_moral_steering_shift', 'steering_shift_ci_low', 'steering_shift_ci_high', 'moral_flip_rate', 'invalid_rate', 'run_mode'])}

Axis table: `results/tables/moral_summary_by_axis.csv`.

Plots: `results/plots/moral_steering_shift_by_language.png`, `results/plots/moral_flip_rate_by_language.png`, and `results/plots/moral_order_stability.png`.

## Interpretation
{('No substantive moral-steering result is justified because the model path was dry-run only.' if dry_run else 'The moral-steering protocol produced item-level logprob outputs and can be audited, but conclusions are limited by synthetic data and unverified Zulu translation.')}

## Failure Modes
Zulu translations are LLM-generated and unverified. A/B label-order sensitivity is measured but not eliminated. Tokenizer variants are handled by sequence scoring, but chat template behavior can still affect absolute margins. The pilot uses n=20 synthetic items, so confidence intervals are exploratory.

## Next Step
Expand only after Zulu translation verification and at least one real-model run over the current 20-item set.
"""

    validation_report = f"""# Validation Study: Moral Robustness vs Safety Robustness

## Question
Do moral-steering gaps and safety/jailbreak gaps diverge across English and Zulu?

## Setup
Same configured model: `{model_id}`. The safety files currently contain harmless placeholders because no local public safety/refusal benchmark subset was available during repo construction. Classification uses a conservative phrase-based refusal detector with an `unclear` class. Manager-facing reports redact prompt details and do not reproduce harmful content.

## Results
Safety table: `results/tables/safety_summary_by_language.csv`.

{markdown_table(safety, ['language', 'n_prompts', 'non_refusal_count', 'refusal_count', 'unclear_count', 'safety_ASR', 'unclear_rate', 'run_mode', 'benchmark_status'])}

Divergence table: `results/tables/divergence_summary.csv`.

{markdown_table(divergence, ['english_moral_steering_score', 'zulu_moral_steering_score', 'moral_gap', 'english_safety_ASR', 'zulu_safety_ASR', 'safety_gap', 'qualitative_divergence_interpretation'])}

Plot: `results/plots/moral_vs_safety_gap.png`.

## Interpretation
{interpretation}

## Limitations
The safety validation is not substantive while the CSVs are placeholders. The refusal classifier is intentionally conservative and can miss subtle non-refusals. Moral and safety metrics are not directly commensurable. Zulu translation quality remains a first-order measurement caveat.

## Next Step
Replace placeholder safety CSVs with a vetted public benchmark subset translated under the same controls, then rerun the same scripts. If positive after a real benchmark run, expand to six languages. If null, reframe the project around whether moral brittleness tracks general low-resource robustness.
"""

    proposal_notes = """# Proposal 2 Refinement Notes

- Reframe the novelty around divergence between moral-stance steering robustness and safety/jailbreak robustness across the resource divide.
- Acknowledge adjacent prior work directly: moral judgments can be prompt-steered cross-lingually, and jailbreak robustness varies by language.
- Make Pilot 0 concrete: English/Zulu, 20 forced-choice vignettes, utilitarian vs egalitarian frames, A/B sequence logprob scoring, and swapped-label checks.
- Separate Pilot 0 from the full six-language study. Treat Pilot 0 as a measurement/infrastructure de-risking step, not as evidence for the full claim.
- Separate the validation study from the full defenses section. The validation should only ask whether moral and safety gaps move together in the same language pair.
- State that synthetic MultiTP-style vignettes are acceptable only for Pilot 0 and must not be represented as the original benchmark.
- Treat Zulu translation verification as a gating criterion before substantive interpretation.
- Avoid overclaiming from n=20; use directional language and report uncertainty.
"""

    slack_update = f"""Draft Slack update to Ajay:

I built the local Pilot 0 repo for Proposal 2 and wired the full pipeline: synthetic EN/ZU moral vignettes, moral prompt wrappers, sequence-logprob scoring with A/B swap checks, safety/refusal eval scaffolding, analysis tables, plots, and manager-facing reports.

Status: {status_prefix}. Moral data is synthetic/MultiTP-style, and Zulu is LLM-draft/unverified. Safety validation is blocked as substantive evidence because I did not find a local public benchmark subset, so the current safety CSVs are harmless placeholders. The generated divergence summary is therefore infrastructure-only if dry-run/placeholder flags are present.

Preliminary result: {interpretation}

Next step: verify Zulu translations and replace the placeholder safety prompts with a vetted public benchmark subset, then rerun the same commands with a real model.
"""

    completed = [
        "Created `proposal2-pilot0/` repo structure.",
        "Created 20 synthetic English moral vignettes and matched unverified Zulu translations.",
        "Implemented moral logprob scoring, safety eval, analysis, plots, and report generation.",
        "Generated required reports, tables, and plots from the available run mode.",
    ]
    partial = [
        "Safety validation scaffold is complete, but current safety data is harmless placeholder data.",
        "Zulu translations exist but are unverified and should not be treated as measurement-grade.",
    ]
    blocked = []
    if dry_run:
        blocked.append("Real model run did not complete; dry-run outputs are pipeline checks only.")
    if safety_blocked:
        blocked.append("Substantive safety/jailbreak validation is blocked until a vetted benchmark subset is loaded.")
    if not blocked:
        blocked.append("No blocking issue recorded.")

    manager_handoff = f"""# Manager Handoff: Proposal 2 Pilot 0

## TL;DR
Built the full local Pilot 0 repo and generated all requested artifacts. Status is `{status_prefix}` because dry-run and/or placeholder safety flags mean the outputs are not a complete substantive validation unless a real model and vetted safety benchmark were actually used. Moral protocol is implemented and auditable; safety comparison currently cannot support a claim while placeholders are present.

## Exact status
- Completed: {"; ".join(completed)}
- Partially completed: {"; ".join(partial)}
- Blocked: {"; ".join(blocked)}

## Model and environment
- model_id: `{model_id}`
- hardware: {hardware_summary()}
- Python version: {platform.python_version()} at `{sys.executable}`
- GPU: {gpu_summary()}
- runtime: {fmt(total_runtime, 2)} seconds recorded across manifest commands
- dependency issues: {dependency_issues}

## Data status
- moral_vignettes_en.csv: {count_csv_rows(data_dir / 'moral_vignettes_en.csv')} synthetic/MultiTP-style rows
- moral_vignettes_zu.csv: {count_csv_rows(data_dir / 'moral_vignettes_zu.csv')} translated rows, status `{translation_status(data_dir / 'moral_vignettes_zu.csv')}`
- safety_prompts_en.csv: {count_csv_rows(data_dir / 'safety_prompts_en.csv')} harmless placeholder rows
- safety_prompts_zu.csv: {count_csv_rows(data_dir / 'safety_prompts_zu.csv')} harmless placeholder rows, status `{translation_status(data_dir / 'safety_prompts_zu.csv')}`
- original benchmark used or synthetic: moral is synthetic/MultiTP-style, not original MultiTP; safety is placeholder, not a real jailbreak benchmark
- translation method: Zulu text was generated by Codex as an LLM draft translation and marked `unverified`
- translation caveats: no native-speaker or professional verification; moral tradeoffs may shift in translation

## Commands run
{chr(10).join(command_lines) if command_lines else '- No commands recorded in manifest.'}

## Key results
Include these numbers only with the run-status caveats above:
- mean_moral_steering_shift_en: {fmt(div_row.get('english_moral_steering_score'))}
- mean_moral_steering_shift_zu: {fmt(div_row.get('zulu_moral_steering_score'))}
- moral_gap: {fmt(div_row.get('moral_gap'))}
- moral_flip_rate_en: {fmt(div_row.get('english_moral_flip_rate'))}
- moral_flip_rate_zu: {fmt(div_row.get('zulu_moral_flip_rate'))}
- safety_ASR_en: {fmt(div_row.get('english_safety_ASR'))}
- safety_ASR_zu: {fmt(div_row.get('zulu_safety_ASR'))}
- safety_gap: {fmt(div_row.get('safety_gap'))}
- divergence_interpretation: {interpretation}

## Files to inspect
- `data/moral_vignettes_en.csv`
- `data/moral_vignettes_zu.csv`
- `data/safety_prompts_en.csv`
- `data/safety_prompts_zu.csv`
- `results/raw/moral_logprobs_item_level.csv`
- `results/raw/safety_outputs_item_level.csv`
- `results/tables/moral_summary_by_language.csv`
- `results/tables/moral_summary_by_axis.csv`
- `results/tables/moral_order_stability.csv`
- `results/tables/safety_summary_by_language.csv`
- `results/tables/divergence_summary.csv`
- `results/plots/moral_steering_shift_by_language.png`
- `results/plots/moral_flip_rate_by_language.png`
- `results/plots/moral_order_stability.png`
- `results/plots/moral_vs_safety_gap.png`
- `report/pilot0_two_pager.md`
- `report/validation_two_pager.md`
- `report/proposal2_refinement_notes.md`
- `report/slack_update_draft.md`

## Main caveats
Be blunt: synthetic moral data is not the original benchmark, Zulu is unverified, n=20 is too small for a substantive claim, dry-run outputs are fake if present, and placeholder safety prompts make the validation study blocked as evidence. Moral and safety metrics are directional comparisons, not commensurable magnitudes.

## What Jeremy should tell Ajay
- The repo and reproducible Pilot 0 pipeline are built.
- The moral task is concrete: EN/ZU, 20 synthetic forced-choice vignettes, three frames, sequence logprob scoring, and A/B order checks.
- The current Zulu translation is an unverified LLM draft and needs verification before claims.
- The safety validation scaffold is ready, but a real public safety benchmark subset is still needed.
- The next run should use a real model and vetted safety data before reporting any substantive divergence result.

## What to do next
1. Verify or replace all Zulu translations.
2. Load a vetted public safety/refusal benchmark subset without reproducing harmful content in manager-facing reports.
3. Rerun the four-command pipeline with `--model_id Qwen/Qwen2.5-7B-Instruct` or gated Llama access if available.

## If ChatGPT is asked to write the final two-pager
Read these first: `report/manager_handoff.md`, `results/tables/divergence_summary.csv`, `results/tables/moral_summary_by_language.csv`, `results/tables/safety_summary_by_language.csv`, and the raw item-level CSVs. Claims justified now: the pipeline exists and the measurement design is auditable. Claims not justified if dry-run or placeholder flags are present: any substantive moral-vs-safety divergence conclusion, any Zulu-specific robustness claim, or any jailbreak robustness claim.
"""

    (report_dir / "pilot0_two_pager.md").write_text(pilot_report, encoding="utf-8")
    (report_dir / "validation_two_pager.md").write_text(validation_report, encoding="utf-8")
    (report_dir / "proposal2_refinement_notes.md").write_text(proposal_notes, encoding="utf-8")
    (report_dir / "slack_update_draft.md").write_text(slack_update, encoding="utf-8")
    (report_dir / "manager_handoff.md").write_text(manager_handoff, encoding="utf-8")

    write_manifest(
        output_dir,
        {
            "script": "build_report_assets.py",
            "command": "python " + " ".join(sys.argv),
            "status": "completed",
            "outputs": [
                str(report_dir / "pilot0_two_pager.md"),
                str(report_dir / "validation_two_pager.md"),
                str(report_dir / "proposal2_refinement_notes.md"),
                str(report_dir / "slack_update_draft.md"),
                str(report_dir / "manager_handoff.md"),
            ],
            "duration_seconds": round(time.time() - start, 2),
        },
    )
    print(f"Wrote reports to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
