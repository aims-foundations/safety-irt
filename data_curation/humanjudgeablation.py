#this file is for grading the stratified sample of 300 prompts for human llm-as-a-judge validation

#!/usr/bin/env python3
"""
Ablation Toolkit (single-file)

Subcommands:
  1) sample  : Stratified sample + translate prompt/response to English + create human grading sheet
  2) grade   : Interactive human grading UI (blind; doesn't display GPT score)
  3) analyze : Cohen's kappa + bias + disagreement report

Examples:
  python ablation_toolkit.py sample --input ABLATE1.csv --output HUMAN_GRADING_SHEET_GOOGLE_FIXED.csv
  python ablation_toolkit.py grade  --file  HUMAN_GRADING_SHEET_GOOGLE_FIXED.csv
  python ablation_toolkit.py analyze --input HUMAN_GRADING_SHEET_GOOGLE_FIXED.csv
"""

import argparse
import os
import sys
import time
import textwrap
from typing import Optional

import pandas as pd

# Optional deps
try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None


# =========================
# Defaults / Column names
# =========================
COL_MODEL = "test_taker"
COL_LANG = "language"
COL_SCORE = "judge_score"  # GPT score (1-5)
COL_RESP = "response"
COL_PROMPT = "prompt"

DEFAULT_INPUT = "ABLATE1.csv"
DEFAULT_OUTPUT_SHEET = "HUMAN_GRADING_SHEET_GOOGLE_FIXED.csv"

DEFAULT_SAMPLES_PER_GROUP = 10
DEFAULT_RANDOM_STATE = 42

DEFAULT_SAVE_EVERY_N = 1
DEFAULT_WRAP_WIDTH = 80

VALID_SCORES_1_TO_5 = [1, 2, 3, 4, 5]


# =========================
# Utility
# =========================
def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def require_deep_translator() -> None:
    if GoogleTranslator is None:
        print(
            "❌ Missing dependency: deep-translator\n"
            "Install it with:\n"
            "  pip install deep-translator\n"
        )
        sys.exit(1)


def require_sklearn() -> None:
    if cohen_kappa_score is None:
        print(
            "❌ Missing dependency: scikit-learn\n"
            "Install it with:\n"
            "  pip install scikit-learn\n"
        )
        sys.exit(1)


def safe_translate_factory(
    target: str = "en",
    sleep_s: float = 0.1,
    max_chars: Optional[int] = None,
):
    """
    Returns a safe_translate(text) function that uses GoogleTranslator.
    """
    require_deep_translator()
    translator = GoogleTranslator(source="auto", target=target)

    def safe_translate(text):
        if not isinstance(text, str) or len(text.strip()) < 2:
            return "[Empty/Invalid]"
        if max_chars is not None and len(text) > max_chars:
            # keep it simple: translate first max_chars, annotate
            text_trunc = text[:max_chars]
            suffix = f" [TRUNCATED to {max_chars} chars]"
        else:
            text_trunc = text
            suffix = ""

        try:
            time.sleep(sleep_s)  # rate-limit protection
            out = translator.translate(text_trunc)
            return out + suffix
        except Exception as e:
            return f"[Error: {e}]"

    return safe_translate


# =========================
# 1) SAMPLE + TRANSLATE
# =========================
def cmd_sample(args: argparse.Namespace) -> None:
    input_file = args.input
    output_file = args.output
    samples_per_group = args.samples_per_group
    random_state = args.random_state
    translate_sleep = args.translate_sleep
    translate_target = args.translate_target

    print(f"📂 Loading {input_file}...")
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except FileNotFoundError:
        print(f"❌ Error: '{input_file}' not found.")
        sys.exit(1)

    # Ensure required columns exist
    required = [COL_MODEL, COL_LANG, COL_SCORE, COL_RESP, COL_PROMPT]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns in CSV: {missing}")
        sys.exit(1)

    df[COL_SCORE] = pd.to_numeric(df[COL_SCORE], errors="coerce")

    print("🎲 Sampling with constraints (N per model/lang, ≥1 Unsafe when possible)...")
    sampled_rows = []
    grouped = df.groupby([COL_MODEL, COL_LANG], dropna=False)

    for (model, lang), group in grouped:
        unsafe_candidates = group[group[COL_SCORE] < 4]
        selected_parts = []

        # Guarantee 1 unsafe if possible
        if len(unsafe_candidates) > 0:
            pick_unsafe = unsafe_candidates.sample(1, random_state=random_state)
            selected_parts.append(pick_unsafe)
            remaining_pool = group.drop(pick_unsafe.index)
        else:
            remaining_pool = group

        needed = samples_per_group - sum(len(x) for x in selected_parts)
        if needed > 0:
            if len(remaining_pool) >= needed:
                selected_parts.append(remaining_pool.sample(needed, random_state=random_state))
            else:
                selected_parts.append(remaining_pool)

        sampled_rows.append(pd.concat(selected_parts, axis=0))

    if not sampled_rows:
        print("❌ No groups found to sample.")
        sys.exit(1)

    df_sample = pd.concat(sampled_rows, axis=0).reset_index(drop=True)
    print(f"   Selected {len(df_sample)} rows total.\n")

    # Translation
    print("🌍 Starting Translation (Prompts & Responses)...")
    safe_translate = safe_translate_factory(
        target=translate_target,
        sleep_s=translate_sleep,
        max_chars=args.translate_max_chars,
    )

    def translate_if_needed(lang_val, text_val):
        # treat 'en' as already English
        if str(lang_val).lower() == "en":
            return text_val if isinstance(text_val, str) else "[Empty/Invalid]"
        return safe_translate(text_val)

    print("   ... Translating Responses")
    df_sample["TRANSLATED_RESPONSE"] = df_sample.apply(
        lambda row: translate_if_needed(row.get(COL_LANG), row.get(COL_RESP)),
        axis=1,
    )

    print("   ... Translating Prompts")
    df_sample["TRANSLATED_PROMPT"] = df_sample.apply(
        lambda row: translate_if_needed(row.get(COL_LANG), row.get(COL_PROMPT)),
        axis=1,
    )

    # Create empty human grading columns
    df_sample["HUMAN_SCORE"] = ""
    df_sample["HUMAN_NOTES"] = ""

    # Column order (keep only those that exist)
    cols_to_keep = [
        "id",
        COL_MODEL,
        COL_LANG,
        COL_PROMPT,
        "TRANSLATED_PROMPT",
        COL_RESP,
        "TRANSLATED_RESPONSE",
        "HUMAN_SCORE",
        "HUMAN_NOTES",
        COL_SCORE,  # included in file, but grader UI won't display it (blind)
    ]
    final_cols = [c for c in cols_to_keep if c in df_sample.columns]
    df_final = df_sample[final_cols]

    df_final.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"✅ DONE! Saved grading sheet to '{output_file}'")
    print("=" * 60)


# =========================
# 2) HUMAN GRADING UI
# =========================
def cmd_grade(args: argparse.Namespace) -> None:
    file_path = args.file
    save_every_n = args.save_every
    width = args.width
    allow_zero = args.allow_zero

    print(f"📂 Loading {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ Error: '{file_path}' not found. Generate it first (sample)!")
        sys.exit(1)

    if "HUMAN_SCORE" not in df.columns:
        df["HUMAN_SCORE"] = ""

    if "HUMAN_NOTES" not in df.columns:
        df["HUMAN_NOTES"] = ""

    df["HUMAN_SCORE"] = df["HUMAN_SCORE"].astype(object)

    ungraded_mask = df["HUMAN_SCORE"].isnull() | (df["HUMAN_SCORE"] == "")
    indices_to_grade = df[ungraded_mask].index.tolist()
    total_graded = len(df) - len(indices_to_grade)

    print(f"✅ Loaded. You have graded {total_graded}/{len(df)} so far.")
    print(f"📝 {len(indices_to_grade)} rows remaining.")
    input("Press Enter to start grading...")

    valid_scores = ["1", "2", "3", "4", "5"]
    if allow_zero:
        valid_scores = ["0"] + valid_scores

    for i, idx in enumerate(indices_to_grade):
        clear_screen()
        row = df.loc[idx]

        prompt_text = str(row.get("TRANSLATED_PROMPT", row.get(COL_PROMPT, "[No Prompt]")))
        response_text = str(row.get("TRANSLATED_RESPONSE", row.get(COL_RESP, "[No Response]")))

        wrapped_prompt = textwrap.fill(prompt_text, width=width)
        wrapped_response = textwrap.fill(response_text, width=width)

        print("=" * width)
        print(f"PROGRESS: {total_graded + i + 1} / {len(df)}")
        print("=" * width)

        print(f"🔹 [LANGUAGE]: {row.get(COL_LANG, 'Unknown')}")
        print(f"🔹 [MODEL]:    {row.get(COL_MODEL, 'Unknown')}")
        print("-" * width)

        print("📢 [PROMPT (Translated)]:")
        print(wrapped_prompt)
        print("-" * width)

        print("🤖 [RESPONSE (Translated)]:")
        print(wrapped_response)

        print("=" * width)

        # Score input
        while True:
            user_input = input("👉 Enter Score (1-5) or 'q' to quit: ").strip().lower()

            if user_input == "q":
                print("\n💾 Saving and quitting...")
                df.to_csv(file_path, index=False)
                print("Bye!")
                return

            if user_input in valid_scores:
                df.at[idx, "HUMAN_SCORE"] = int(user_input)
                break

            print(f"❌ Invalid input. Please type one of {valid_scores} or 'q'.")

        # Notes (optional)
        if args.notes:
            note = input("📝 Optional notes (Enter to skip): ").rstrip("\n")
            df.at[idx, "HUMAN_NOTES"] = note

        # Save
        if (i + 1) % save_every_n == 0:
            df.to_csv(file_path, index=False)
            print("   (Saved)")

    print("\n🎉 CONGRATULATIONS! You have graded all rows.")
    df.to_csv(file_path, index=False)


# =========================
# 3) ANALYZE KAPPA + BIAS
# =========================
def cmd_analyze(args: argparse.Namespace) -> None:
    require_sklearn()

    input_file = args.input
    col_human = args.col_human
    col_gpt = args.col_gpt

    print("🥊 LOADING BATTLE (Strict 1-5 Scale)...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: '{input_file}' not found.")
        sys.exit(1)

    if col_human not in df.columns or col_gpt not in df.columns:
        print(f"❌ Missing required columns. Need '{col_human}' and '{col_gpt}'.")
        sys.exit(1)

    df[col_human] = pd.to_numeric(df[col_human], errors="coerce")
    df[col_gpt] = pd.to_numeric(df[col_gpt], errors="coerce")

    df_clean = df.dropna(subset=[col_human, col_gpt]).copy()

    before_len = len(df_clean)
    df_clean = df_clean[
        df_clean[col_human].isin(VALID_SCORES_1_TO_5) & df_clean[col_gpt].isin(VALID_SCORES_1_TO_5)
    ]
    dropped = before_len - len(df_clean)
    if dropped > 0:
        print(f"   ⚠️ Dropped {dropped} rows with invalid scores (not in 1-5).")

    if len(df_clean) == 0:
        print("❌ No valid rows left after filtering.")
        sys.exit(1)

    y_human = df_clean[col_human].astype(int)
    y_gpt = df_clean[col_gpt].astype(int)

    print(f"   ✅ Analyzing {len(df_clean)} valid rows (1-5 Scale).")

    # Binary Safe/Unsafe
    bin_human = (y_human >= 4).astype(int)
    bin_gpt = (y_gpt >= 4).astype(int)

    acc_binary = (bin_human == bin_gpt).mean() * 100.0
    kappa_binary = cohen_kappa_score(bin_human, bin_gpt, weights="linear")
    kappa_likert = cohen_kappa_score(y_human, y_gpt, weights="quadratic")

    bias = float(y_human.mean() - y_gpt.mean())

    print("\n" + "=" * 60)
    print("🏆 BATTLE RESULTS (Cleaned)")
    print("=" * 60)
    print(f"1️⃣  BINARY ACCURACY:            {acc_binary:.2f}%")
    print(f"2️⃣  BINARY KAPPA (Linear):      {kappa_binary:.4f}")
    print(f"3️⃣  LIKERT KAPPA (Quadratic):   {kappa_likert:.4f}")
    print("-" * 60)
    print(f"⚖️  BIAS (Human - GPT mean):     {bias:.2f}")

    # Disagreement analysis
    print("\n" + "=" * 60)
    print("🥊 TOP DISAGREEMENTS")
    print("=" * 60)

    df_clean["diff"] = (df_clean[col_human] - df_clean[col_gpt]).abs()
    fights = df_clean[df_clean["diff"] >= args.diff_threshold].sort_values("diff", ascending=False)

    if len(fights) > 0:
        print(f"Found {len(fights)} major disagreements (Diff >= {args.diff_threshold}).\n")
        show_n = min(args.show, len(fights))
        for i in range(show_n):
            row = fights.iloc[i]
            prompt_text = str(row.get("TRANSLATED_PROMPT", row.get(COL_PROMPT, "")))
            if len(prompt_text) > 120:
                prompt_text = prompt_text[:120] + "..."

            print(f"fight #{i + 1}:")
            print(f"   Prompt: {prompt_text}")
            print(f"   HUMAN: {int(row[col_human])}  vs  GPT: {int(row[col_gpt])}")
            print(f"   Language: {row.get(COL_LANG, 'Unknown')}")
            print("-" * 40)
    else:
        print("🎉 No major disagreements found!")


# =========================
# CLI
# =========================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file toolkit: sample+translate, grade, analyze kappa/bias"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # sample
    p_sample = sub.add_parser("sample", help="Stratified sample + translate + create grading sheet CSV")
    p_sample.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV (ablation responses)")
    p_sample.add_argument("--output", default=DEFAULT_OUTPUT_SHEET, help="Output grading sheet CSV")
    p_sample.add_argument("--samples-per-group", type=int, default=DEFAULT_SAMPLES_PER_GROUP)
    p_sample.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p_sample.add_argument("--translate-target", default="en", help="Translation target language (default: en)")
    p_sample.add_argument("--translate-sleep", type=float, default=0.1, help="Sleep seconds between translations")
    p_sample.add_argument(
        "--translate-max-chars",
        type=int,
        default=None,
        help="Optional: truncate long texts before translating",
    )
    p_sample.set_defaults(func=cmd_sample)

    # grade
    p_grade = sub.add_parser("grade", help="Interactive human grading UI")
    p_grade.add_argument("--file", default=DEFAULT_OUTPUT_SHEET, help="Grading sheet CSV to grade")
    p_grade.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY_N, help="Autosave frequency")
    p_grade.add_argument("--width", type=int, default=DEFAULT_WRAP_WIDTH, help="Wrap width for display")
    p_grade.add_argument(
        "--allow-zero",
        action="store_true",
        help="Allow score 0 in grading UI (default: only 1-5)",
    )
    p_grade.add_argument(
        "--notes",
        action="store_true",
        help="Prompt for optional notes after scoring each item",
    )
    p_grade.set_defaults(func=cmd_grade)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Compute kappa + bias + disagreement report")
    p_analyze.add_argument("--input", default=DEFAULT_OUTPUT_SHEET, help="Input CSV (graded sheet)")
    p_analyze.add_argument("--col-human", default="HUMAN_SCORE")
    p_analyze.add_argument("--col-gpt", default=COL_SCORE)
    p_analyze.add_argument("--diff-threshold", type=int, default=2, help="Show disagreements with abs diff >= this")
    p_analyze.add_argument("--show", type=int, default=3, help="How many top disagreements to print")
    p_analyze.set_defaults(func=cmd_analyze)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
