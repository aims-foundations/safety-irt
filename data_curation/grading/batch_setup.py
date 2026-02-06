"""Batch grading setup: prepare data, merge CSVs, estimate cost, create JSONL.

Usage:
    python -m data_curation.grading.batch_setup merge --files a.csv b.csv --output merged.csv
    python -m data_curation.grading.batch_setup add-prompts --prompts multijail.csv --results results.csv --output out.csv
    python -m data_curation.grading.batch_setup estimate-cost --input graded.csv
    python -m data_curation.grading.batch_setup create-jsonl --input graded.csv --output batch.jsonl --model gpt-5.2
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_curation.shared.grading_prompt import (
    POLICY_DESCRIPTION,
    SCORE_BREAKDOWN,
    format_grading_prompt,
)


# ─── merge ────────────────────────────────────────────────────────────────────

def cmd_merge(args):
    """Merge multiple test-taker CSVs into one file."""
    dataframes = []
    total_expected = 0

    print("Starting Merge Process...")
    for filename in args.files:
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            continue
        df = pd.read_csv(filename)
        print(f"Loaded '{filename}': {len(df)} rows")
        if 'test_taker' not in df.columns:
            print(f"Warning: 'test_taker' column missing in {filename}.")
        dataframes.append(df)
        total_expected += len(df)

    if not dataframes:
        print("No data found. Exiting.")
        return

    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    combined_df.to_csv(args.output, index=False)

    print("\n" + "=" * 40)
    print("MERGE COMPLETE")
    print("=" * 40)
    print(f"Files Merged:      {len(dataframes)}")
    print(f"Total Rows Saved:  {len(combined_df)}")
    print(f"Output File:       {args.output}")

    if len(combined_df) == total_expected:
        print("Row count matches perfectly.")
    else:
        print(f"Row count mismatch! Expected {total_expected}, got {len(combined_df)}")

    if 'test_taker' in combined_df.columns:
        print("\nModels included:")
        print(combined_df['test_taker'].unique())


# ─── add-prompts ──────────────────────────────────────────────────────────────

def cmd_add_prompts(args):
    """Merge prompt text into a results CSV that only has id + language."""
    if not os.path.exists(args.prompts):
        print(f"Error: Source file '{args.prompts}' not found.")
        return
    if not os.path.exists(args.results):
        print(f"Error: Results file '{args.results}' not found.")
        return

    print("Loading files...")
    df_prompts = pd.read_csv(args.prompts)[['id', 'language', 'prompt']]
    print(f"Loaded {len(df_prompts)} prompts.")

    df_results = pd.read_csv(args.results)
    print(f"Loaded {len(df_results)} model responses.")

    print("\nMerging data...")
    df_merged = pd.merge(df_results, df_prompts, on=['id', 'language'], how='left')

    missing_prompts = df_merged['prompt'].isna().sum()
    if missing_prompts > 0:
        print(f"Warning: {missing_prompts} rows could not be matched to a prompt.")
    else:
        print("All rows successfully matched to their prompt!")

    desired_order = ['id', 'language', 'prompt', 'test_taker', 'response', 'finish_reason']
    final_order = [col for col in desired_order if col in df_merged.columns]
    df_merged = df_merged[final_order]

    df_merged.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")
    print(df_merged.head())


# ─── estimate-cost ────────────────────────────────────────────────────────────

def cmd_estimate_cost(args):
    """Estimate OpenAI Batch API input token cost."""
    try:
        import tiktoken
    except ImportError:
        print("tiktoken not installed. Run: pip install tiktoken")
        return

    df = pd.read_csv(args.input)
    df["response"] = df["response"].fillna("[NO RESPONSE]").astype(str)
    df["prompt"] = df["prompt"].fillna("[NO PROMPT]").astype(str)
    df["response"] = df["response"].str.replace("\n", " ", regex=False)
    df["prompt"] = df["prompt"].str.replace("\n", " ", regex=False)

    enc = tiktoken.get_encoding("cl100k_base")

    total_input_tokens = 0
    per_row_tokens = []

    for _, row in df.iterrows():
        user_content = format_grading_prompt(row["prompt"], row["response"])
        tokens = len(enc.encode(user_content))
        per_row_tokens.append(tokens)
        total_input_tokens += tokens

    avg_tokens = total_input_tokens / len(per_row_tokens)
    p50 = sorted(per_row_tokens)[len(per_row_tokens) // 2]
    p90 = sorted(per_row_tokens)[int(len(per_row_tokens) * 0.9)]

    print("==== Batch Token Estimate (INPUT) ====")
    print(f"Rows: {len(per_row_tokens)}")
    print(f"Total input tokens (approx): {total_input_tokens:,}")
    print(f"Avg tokens per row: {avg_tokens:.1f}")
    print(f"Median (p50): {p50}")
    print(f"90th percentile (p90): {p90}")

    input_cost_per_m = args.cost_per_million
    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_m

    print(f"\n==== Cost Estimate (INPUT ONLY) ====")
    print(f"Price per 1M tokens: ${input_cost_per_m}")
    print(f"Estimated input cost: ${input_cost:.2f}")
    print("\nNOTE: Output tokens not included. Add ~50-80 output tokens per row for full estimate.")


# ─── create-jsonl ─────────────────────────────────────────────────────────────

def cmd_create_jsonl(args):
    """Create a Batch API JSONL file from a CSV of prompts + responses."""
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    df = pd.read_csv(args.input)
    df["response"] = df["response"].fillna("[NO RESPONSE]").astype(str)
    df["prompt"] = df["prompt"].fillna("[NO PROMPT]").astype(str)
    df["response"] = df["response"].str.replace("\n", " ", regex=False)
    df["prompt"] = df["prompt"].str.replace("\n", " ", regex=False)

    print(f"Processing {len(df)} rows for Batch API (Model: {args.model})...")

    with open(args.output, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            user_content = format_grading_prompt(row["prompt"], row["response"])

            request_body = {
                "custom_id": f"req_{row['id']}_{idx}" if "id" in df.columns else f"req_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "user", "content": user_content}
                    ],
                    "max_completion_tokens": 256,
                    "reasoning_effort": "low",
                },
            }

            f.write(json.dumps(request_body, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} requests to {args.output}")
    print("Ready to upload.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch grading setup utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # merge
    p_merge = sub.add_parser("merge", help="Merge multiple test-taker CSVs")
    p_merge.add_argument("--files", nargs="+", required=True, help="CSV files to merge")
    p_merge.add_argument("--output", required=True, help="Output CSV path")
    p_merge.set_defaults(func=cmd_merge)

    # add-prompts
    p_add = sub.add_parser("add-prompts", help="Add prompt text to a results CSV")
    p_add.add_argument("--prompts", default="multijail.csv", help="Source prompts CSV")
    p_add.add_argument("--results", required=True, help="Results CSV (missing prompts)")
    p_add.add_argument("--output", required=True, help="Output CSV path")
    p_add.set_defaults(func=cmd_add_prompts)

    # estimate-cost
    p_cost = sub.add_parser("estimate-cost", help="Estimate Batch API input token cost")
    p_cost.add_argument("--input", required=True, help="Input CSV with prompt + response")
    p_cost.add_argument("--cost-per-million", type=float, default=1.75,
                        help="USD per 1M input tokens (default: 1.75)")
    p_cost.set_defaults(func=cmd_estimate_cost)

    # create-jsonl
    p_jsonl = sub.add_parser("create-jsonl", help="Create Batch API JSONL file")
    p_jsonl.add_argument("--input", required=True, help="Input CSV with prompt + response")
    p_jsonl.add_argument("--output", default="batch_grading_requests.jsonl", help="Output JSONL path")
    p_jsonl.add_argument("--model", default="gpt-5.2", help="Judge model (default: gpt-5.2)")
    p_jsonl.set_defaults(func=cmd_create_jsonl)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
