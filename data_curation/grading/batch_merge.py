"""Parse batch grading JSONL results and merge scores into the original CSV.

Usage:
    python -m data_curation.grading.batch_merge --original data.csv --results batch_results.jsonl --output graded.csv
"""

import argparse
import json
import re

import pandas as pd


def merge_batch_results(original_csv, batch_jsonl, output_file):
    """Parse batch JSONL and merge judge_score + judge_reason into the original CSV."""
    print("Loading data...")
    try:
        df = pd.read_csv(original_csv)
        print(f"Loaded {len(df)} rows from {original_csv}.")
    except FileNotFoundError:
        print(f"Error: Could not find {original_csv}")
        return

    df['judge_score'] = None
    df['judge_reason'] = None

    success_count = 0
    parse_error_count = 0

    print("Parsing Batch JSONL...")

    with open(batch_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)

            custom_id = data['custom_id']
            try:
                row_idx = int(custom_id.split('_')[-1])
            except ValueError:
                print(f"Warning: Could not parse index from {custom_id}")
                continue

            if data['response']['status_code'] != 200:
                print(f"API Error on row {row_idx}: {data['response']}")
                continue

            content = data['response']['body']['choices'][0]['message']['content']

            score_match = re.search(r'#score:\s*(\d)', content)
            reason_match = re.search(r'#reason:\s*(.*)', content)

            if score_match:
                df.at[row_idx, 'judge_score'] = int(score_match.group(1))
            else:
                parse_error_count += 1

            if reason_match:
                df.at[row_idx, 'judge_reason'] = reason_match.group(1).strip()

            success_count += 1

    df.to_csv(output_file, index=False)

    print("\n" + "=" * 40)
    print("       GRADING MERGE COMPLETE")
    print("=" * 40)
    print(f"Total Rows Processed: {success_count}")
    print(f"Parse Errors (Missing Score): {parse_error_count}")
    print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Merge batch grading results into CSV")
    parser.add_argument("--original", required=True, help="Original CSV with prompts + responses")
    parser.add_argument("--results", required=True, help="Batch API output JSONL")
    parser.add_argument("--output", required=True, help="Output CSV with judge scores")
    args = parser.parse_args()

    merge_batch_results(args.original, args.results, args.output)


if __name__ == "__main__":
    main()
