#!/usr/bin/env python3
"""Score forced-choice moral vignettes with answer-sequence log probabilities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path


FRAMES = {
    "baseline": "moral_baseline.txt",
    "utilitarian": "moral_utilitarian.txt",
    "egalitarian": "moral_egalitarian.txt",
}

ANSWER_VARIANTS = {
    "A": ["A", " A"],
    "B": ["B", " B"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--limit", type=int, default=None, help="Optional per-language item limit.")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry_run", action="store_true", help="Use deterministic fake logprobs; no model download.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--prompt_dir", default="prompts")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def strip_option_label(text: str) -> str:
    stripped = (text or "").strip()
    if len(stripped) >= 2 and stripped[0] in {"A", "B"} and stripped[1] in {":", "."}:
        return stripped[2:].strip()
    return stripped


def relabel_option(label: str, option_text: str) -> str:
    return f"{label}: {strip_option_label(option_text)}"


def swapped_choice(choice: str) -> str:
    return {"A": "B", "B": "A"}.get(choice, "")


def format_user_prompt(row: dict[str, str]) -> str:
    return (
        "Scenario:\n"
        f"{row['scenario'].strip()}\n\n"
        "Options:\n"
        f"A: {strip_option_label(row['option_a'])}\n"
        f"B: {strip_option_label(row['option_b'])}\n\n"
        "Answer with A or B only."
    )


def make_rows_for_order(row: dict[str, str], order_condition: str) -> dict[str, str]:
    out = dict(row)
    out["order_condition"] = order_condition
    if order_condition == "swapped":
        original_a = row.get("option_a", "")
        original_b = row.get("option_b", "")
        out["option_a"] = relabel_option("A", original_b)
        out["option_b"] = relabel_option("B", original_a)
        out["utilitarian_choice"] = swapped_choice(row.get("utilitarian_choice", ""))
        out["egalitarian_choice"] = swapped_choice(row.get("egalitarian_choice", ""))
    return out


def format_valid(row: dict[str, str]) -> bool:
    required = ["id", "axis", "scenario", "option_a", "option_b", "utilitarian_choice", "egalitarian_choice"]
    if any(not str(row.get(k, "")).strip() for k in required):
        return False
    return row.get("utilitarian_choice") in {"A", "B"} and row.get("egalitarian_choice") in {"A", "B"}


def softmax_pair(logprob_a: float, logprob_b: float) -> tuple[float, float]:
    m = max(logprob_a, logprob_b)
    ea = math.exp(logprob_a - m)
    eb = math.exp(logprob_b - m)
    denom = ea + eb
    return ea / denom, eb / denom


def margin_for_choice(choice: str, logprob_a: float, logprob_b: float) -> float:
    if choice == "A":
        return logprob_a - logprob_b
    if choice == "B":
        return logprob_b - logprob_a
    return float("nan")


def original_label_for_choice(order_condition: str, chosen: str) -> str:
    if chosen not in {"A", "B"}:
        return ""
    if order_condition == "original":
        return chosen
    return swapped_choice(chosen)


def fake_logprobs(row: dict[str, str], language: str, frame: str, seed: int) -> tuple[float, float, str]:
    digest = hashlib.sha256(f"{seed}|{row['id']}|{language}|{frame}|{row['order_condition']}".encode()).hexdigest()
    jitter = (int(digest[:8], 16) / 0xFFFFFFFF) - 0.5
    strength = 1.10 if language == "en" else 0.72
    if row["order_condition"] == "swapped":
        strength -= 0.10
    logprob_a = -0.72 + jitter * 0.08
    logprob_b = -0.72 - jitter * 0.08
    if frame == "utilitarian":
        target = row["utilitarian_choice"]
        boost = strength
    elif frame == "egalitarian":
        target = row["egalitarian_choice"]
        boost = strength * 0.9
    else:
        target = row["utilitarian_choice"] if int(digest[8:10], 16) % 2 == 0 else row["egalitarian_choice"]
        boost = 0.18
    if target == "A":
        logprob_a += boost / 2
        logprob_b -= boost / 2
    else:
        logprob_a -= boost / 2
        logprob_b += boost / 2
    return logprob_a, logprob_b, "DRY_RUN_FAKE_LOGPROBS"


def select_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is false.")
    if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("Requested MPS but torch.backends.mps.is_available() is false.")
    return requested


def load_model(model_id: str, device_request: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = select_device(device_request)
    dtype = torch.float32
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device in {"cuda", "mps"}:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def build_model_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"


def sequence_logprob(model, tokenizer, device: str, prompt_text: str, candidate_text: str) -> float:
    import torch

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    candidate_ids = tokenizer(candidate_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    total = 0.0
    for i in range(candidate_ids.shape[1]):
        token_id = candidate_ids[0, i]
        pos = prompt_len + i - 1
        total += float(log_probs[0, pos, token_id].detach().cpu())
    return total


def best_answer_logprob(model, tokenizer, device: str, prompt_text: str, answer: str) -> tuple[float, str, dict[str, list[int]]]:
    scores = []
    tokenizations: dict[str, list[int]] = {}
    for variant in ANSWER_VARIANTS[answer]:
        token_ids = tokenizer(variant, add_special_tokens=False).input_ids
        tokenizations[variant] = list(token_ids)
        scores.append((sequence_logprob(model, tokenizer, device, prompt_text, variant), variant))
    return max(scores, key=lambda item: item[0])[0], max(scores, key=lambda item: item[0])[1], tokenizations


def write_manifest(output_dir: Path, entry: dict) -> None:
    manifest_path = output_dir / "run_manifest.json"
    manifest = {"commands": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"commands": []}
    manifest.setdefault("commands", []).append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    start = time.time()
    args = parse_args()
    random.seed(args.seed)
    root = repo_root()
    data_dir = Path(args.data_dir)
    prompt_dir = Path(args.prompt_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    if not prompt_dir.is_absolute():
        prompt_dir = root / prompt_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)

    prompts = {frame: read_text(prompt_dir / filename) for frame, filename in FRAMES.items()}
    language_files = {
        "en": data_dir / "moral_vignettes_en.csv",
        "zu": data_dir / "moral_vignettes_zu.csv",
    }
    data_by_language = {}
    for language, path in language_files.items():
        rows = load_csv(path)
        if args.limit is not None:
            rows = rows[: args.limit]
        data_by_language[language] = rows

    model = tokenizer = device = None
    load_error = None
    if not args.dry_run:
        try:
            model, tokenizer, device = load_model(args.model_id, args.device)
        except Exception as exc:
            load_error = repr(exc)
            print(f"ERROR: could not load model {args.model_id}: {exc}", file=sys.stderr)
            print("Use --dry_run to verify the pipeline without model access.", file=sys.stderr)
            write_manifest(
                output_dir,
                {
                    "script": "score_moral_logprobs.py",
                    "command": "python " + " ".join(sys.argv),
                    "model_id": args.model_id,
                    "dry_run": args.dry_run,
                    "status": "blocked",
                    "error": load_error,
                    "duration_seconds": round(time.time() - start, 2),
                },
            )
            return 2

    output_rows: list[dict[str, object]] = []
    for language, rows in data_by_language.items():
        for raw_row in rows:
            for order_condition in ["original", "swapped"]:
                ordered_row = make_rows_for_order(raw_row, order_condition)
                valid = format_valid(ordered_row)
                for frame, system_prompt in prompts.items():
                    notes = ordered_row.get("notes", "")
                    if ordered_row.get("translation_status"):
                        notes = f"{notes} translation_status={ordered_row['translation_status']}"
                    if not valid:
                        logprob_a = logprob_b = float("nan")
                        prob_a = prob_b = float("nan")
                        chosen = ""
                        variant_a = variant_b = ""
                        tokenization_note = "FORMAT_INVALID"
                    elif args.dry_run:
                        logprob_a, logprob_b, tokenization_note = fake_logprobs(ordered_row, language, frame, args.seed)
                        prob_a, prob_b = softmax_pair(logprob_a, logprob_b)
                        chosen = "A" if prob_a >= prob_b else "B"
                        variant_a = "DRY_RUN"
                        variant_b = "DRY_RUN"
                    else:
                        user_prompt = format_user_prompt(ordered_row)
                        prompt_text = build_model_prompt(tokenizer, system_prompt, user_prompt)
                        logprob_a, variant_a, toks_a = best_answer_logprob(model, tokenizer, device, prompt_text, "A")
                        logprob_b, variant_b, toks_b = best_answer_logprob(model, tokenizer, device, prompt_text, "B")
                        prob_a, prob_b = softmax_pair(logprob_a, logprob_b)
                        chosen = "A" if prob_a >= prob_b else "B"
                        tokenization_note = f"A_tokens={toks_a}; B_tokens={toks_b}"

                    output_rows.append(
                        {
                            "id": ordered_row.get("id", ""),
                            "language": language,
                            "axis": ordered_row.get("axis", ""),
                            "frame": frame,
                            "order_condition": order_condition,
                            "option_a": ordered_row.get("option_a", ""),
                            "option_b": ordered_row.get("option_b", ""),
                            "utilitarian_choice": ordered_row.get("utilitarian_choice", ""),
                            "egalitarian_choice": ordered_row.get("egalitarian_choice", ""),
                            "logprob_A": logprob_a,
                            "logprob_B": logprob_b,
                            "prob_A": prob_a,
                            "prob_B": prob_b,
                            "chosen": chosen,
                            "chosen_original_label": original_label_for_choice(order_condition, chosen),
                            "utilitarian_margin": margin_for_choice(ordered_row.get("utilitarian_choice", ""), logprob_a, logprob_b),
                            "egalitarian_margin": margin_for_choice(ordered_row.get("egalitarian_choice", ""), logprob_a, logprob_b),
                            "format_valid": valid,
                            "model_id": args.model_id,
                            "run_mode": "dry_run" if args.dry_run else "real",
                            "candidate_variant_A": variant_a,
                            "candidate_variant_B": variant_b,
                            "notes": f"{notes} {tokenization_note}".strip(),
                        }
                    )

    output_path = output_dir / "raw" / "moral_logprobs_item_level.csv"
    fieldnames = [
        "id",
        "language",
        "axis",
        "frame",
        "order_condition",
        "option_a",
        "option_b",
        "utilitarian_choice",
        "egalitarian_choice",
        "logprob_A",
        "logprob_B",
        "prob_A",
        "prob_B",
        "chosen",
        "chosen_original_label",
        "utilitarian_margin",
        "egalitarian_margin",
        "format_valid",
        "model_id",
        "run_mode",
        "candidate_variant_A",
        "candidate_variant_B",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    write_manifest(
        output_dir,
        {
            "script": "score_moral_logprobs.py",
            "command": "python " + " ".join(sys.argv),
            "model_id": args.model_id,
            "dry_run": args.dry_run,
            "device": device or args.device,
            "status": "completed",
            "rows": len(output_rows),
            "output": str(output_path),
            "duration_seconds": round(time.time() - start, 2),
        },
    )
    print(f"Wrote {output_path} ({len(output_rows)} rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

