"""Post-processing utilities for test-taker response CSVs."""

import argparse
import os
import pandas as pd


def clean_responses(input_file, output_file):
    """Fill NaN responses and save cleaned CSV."""
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}")
        return

    df['response'] = df['response'].fillna("")
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 40)
    print("        CLEANING REPORT")
    print("=" * 40)
    print(f"Total Rows: {len(df)}")

    empty_count = len(df[df['response'].str.strip() == ""])
    if empty_count > 0:
        print(f"Warning: {empty_count} rows have empty responses.")
    else:
        print("No empty responses detected.")

    print(f"Cleaned data saved to: {output_file}")


def analyze_length_quality(input_file):
    """Categorize responses by length and print a quality report."""
    df = pd.read_csv(input_file)
    df['response'] = df['response'].fillna("")

    empty_mask = df['response'].str.strip() == ""
    short_mask = (df['response'].str.len() < 50) & (~empty_mask)
    valid_mask = df['response'].str.len() >= 50

    print("=" * 40)
    print("      RESPONSE LENGTH ANALYSIS")
    print("=" * 40)
    print(f"Total Rows:       {len(df)}")
    print(f"Empty:            {df[empty_mask].shape[0]}")
    print(f"Short (<50c):     {df[short_mask].shape[0]}")
    print(f"Valid (>=50c):    {df[valid_mask].shape[0]}")

    if df[short_mask].shape[0] > 0:
        print("\nExamples of Short Responses:")
        print(df[short_mask][['test_taker', 'response']].head(5))


def merge_csvs(file_list, output_file):
    """Concatenate multiple CSVs and sort by id + test_taker."""
    dataframes = []
    print(f"Starting merge of {len(file_list)} files...")

    for filename in file_list:
        if not os.path.exists(filename):
            print(f"Warning: File not found: {filename} (Skipping)")
            continue
        try:
            df = pd.read_csv(filename)
            print(f"Loaded {filename}: {len(df)} rows")
            dataframes.append(df)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    if not dataframes:
        print("No data found to merge.")
        return

    master_df = pd.concat(dataframes, ignore_index=True)
    if 'id' in master_df.columns:
        master_df = master_df.sort_values(by=['id', 'test_taker'])

    master_df.to_csv(output_file, index=False)

    print("\n" + "=" * 40)
    print("MERGE COMPLETE")
    print("=" * 40)
    print(f"Total Rows:     {len(master_df)}")
    if 'test_taker' in master_df.columns:
        print(f"Unique Models:  {master_df['test_taker'].nunique()}")
    print(f"Saved to:       {output_file}")


def count_tokens(input_file, encoding_name="cl100k_base"):
    """Count tokens per response using tiktoken and print summary stats."""
    try:
        import tiktoken
    except ImportError:
        print("tiktoken not installed. Run: pip install tiktoken")
        return

    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        return

    def _count(text, enc):
        if pd.isna(text):
            return 0
        return len(enc.encode(str(text)))

    try:
        encoding = tiktoken.get_encoding(encoding_name)
        print(f"Using tokenizer: {encoding_name}")
        df['token_count'] = df['response'].apply(lambda x: _count(x, encoding))
    except Exception:
        print("tiktoken error. Using word count approximation.")
        df['token_count'] = df['response'].apply(
            lambda x: len(str(x).split()) * 1.3 if pd.notna(x) else 0
        )

    print(f"\nGLOBAL AVERAGE: {df['token_count'].mean():.2f} tokens/response")

    if 'test_taker' in df.columns:
        print("\nAverage Tokens per Test Taker:")
        breakdown = df.groupby('test_taker')['token_count'].mean().sort_values(ascending=False)
        print(breakdown.to_string(float_format="%.1f"))


def main():
    parser = argparse.ArgumentParser(description="Post-processing utilities for test-taker CSVs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_clean = sub.add_parser("clean", help="Fill NaN responses and save cleaned CSV")
    p_clean.add_argument("--input", required=True, help="Input CSV")
    p_clean.add_argument("--output", required=True, help="Output CSV")

    p_length = sub.add_parser("analyze-length", help="Categorize responses by length")
    p_length.add_argument("--input", required=True, help="Input CSV")

    p_merge = sub.add_parser("merge", help="Concatenate multiple CSVs")
    p_merge.add_argument("--files", nargs="+", required=True, help="CSV files to merge")
    p_merge.add_argument("--output", required=True, help="Output CSV")

    p_tokens = sub.add_parser("count-tokens", help="Count tokens per response")
    p_tokens.add_argument("--input", required=True, help="Input CSV")
    p_tokens.add_argument("--encoding", default="cl100k_base", help="Tokenizer encoding name")

    args = parser.parse_args()

    if args.cmd == "clean":
        clean_responses(args.input, args.output)
    elif args.cmd == "analyze-length":
        analyze_length_quality(args.input)
    elif args.cmd == "merge":
        merge_csvs(args.files, args.output)
    elif args.cmd == "count-tokens":
        count_tokens(args.input, args.encoding)


if __name__ == "__main__":
    main()
