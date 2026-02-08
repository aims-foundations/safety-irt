"""LLM-as-Judge ablation with Gemini 2.5 Pro.

Grades a subset of responses using Gemini as an alternative judge,
then compares with the primary GPT judge scores.

Usage:
    python -m data_curation.ablation.gemini_judge extract --input FULLDATA.csv --output ABLATE1.csv
    python -m data_curation.ablation.gemini_judge grade --input ABLATE1.csv --output ABLATE1_GEMINI_GRADED.csv
"""

import argparse
import asyncio
import os
import re
import sys

import pandas as pd
from tqdm.asyncio import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_curation.shared.grading_prompt import format_grading_prompt

# Target models for ablation (subset of full dataset)
TARGET_MODELS = [
    "deepseek-chat_Standard",
    "grok-4-1-fast-non-reasoning_Chaos",
    "gpt-4o-mini_Standard",
]


def cmd_extract(args):
    """Extract the three ablation models from the full dataset."""
    print(f"Loading {args.input}...")

    if not os.path.exists(args.input):
        print(f"Error: Could not find {args.input}")
        return

    df = pd.read_csv(args.input)
    filtered_df = df[df['test_taker'].isin(TARGET_MODELS)].copy()

    print("\nCounts found per model (Should be ~3150 each):")
    print(filtered_df['test_taker'].value_counts())

    filtered_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(filtered_df)} rows to: {args.output}")


async def grade_with_gemini(args):
    """Grade responses using Gemini 2.5 Pro as judge."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai not installed. Run: pip install google-genai")
        return

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY environment variable.")
        return

    judge_model = args.model
    new_column = "Judge_score_gemini"

    client = genai.Client(api_key=api_key)
    semaphore = asyncio.Semaphore(4)

    async def grade_row(index, row):
        row_id = row.get('id')
        lang = row.get('language', '??')
        if pd.isna(row_id):
            row_id = f"Row_{index}"

        prompt_text = format_grading_prompt(
            str(row.get("prompt", "")),
            str(row.get("response", "")),
        )

        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        ]

        gen_config = types.GenerateContentConfig(
            max_output_tokens=4096,
            safety_settings=safety_settings,
        )

        async with semaphore:
            retries = 5
            for attempt in range(retries):
                try:
                    resp = await client.aio.models.generate_content(
                        model=judge_model,
                        contents=prompt_text,
                        config=gen_config,
                    )

                    content = resp.text if resp.text else ""
                    score_match = re.search(r'(?:#|(?<=\W))score[:\s]*(\d)', content, re.IGNORECASE)

                    if score_match:
                        score = int(score_match.group(1))
                        tqdm.write(f"[ID:{row_id}|{lang}] Score: {score}")
                        return score
                    else:
                        tqdm.write(f"[ID:{row_id}|{lang}] Parsing Failed: '{content[:40]}...'")
                        return None

                except Exception as e:
                    wait_time = (2 ** attempt) + 1
                    if "429" in str(e):
                        if attempt < retries - 1:
                            await asyncio.sleep(wait_time)
                            continue

                    if attempt == retries - 1:
                        tqdm.write(f"[ID:{row_id}|{lang}] API Error: {e}")
                        return None

                    await asyncio.sleep(1)

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)

    if "prompt" not in df.columns or "response" not in df.columns:
        print("Error: Missing 'prompt' or 'response' columns.")
        return

    print(f"Rows to grade: {len(df)}")
    print(f"Judge Model:   {judge_model}")
    print("Sending requests...")

    coroutines = [grade_row(i, row) for i, row in df.iterrows()]
    results = []
    chunk_size = 50

    for i in tqdm(range(0, len(coroutines), chunk_size), desc="Grading Batches"):
        batch = coroutines[i:i + chunk_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        await asyncio.sleep(1)

    df[new_column] = results
    df.to_csv(args.output, index=False)

    print("\n" + "=" * 40)
    print(f"Grading Complete! Saved to: {args.output}")
    print("Score Distribution:")
    print(df[new_column].value_counts().sort_index())


def cmd_grade(args):
    """Entry point for the grade subcommand."""
    asyncio.run(grade_with_gemini(args))


def main():
    parser = argparse.ArgumentParser(description="Gemini judge ablation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="Extract ablation models from full dataset")
    p_extract.add_argument("--input", default="FULLDATA.csv", help="Full dataset CSV")
    p_extract.add_argument("--output", default="ABLATE1.csv", help="Output CSV")
    p_extract.set_defaults(func=cmd_extract)

    p_grade = sub.add_parser("grade", help="Grade with Gemini judge")
    p_grade.add_argument("--input", default="ABLATE1.csv", help="Input CSV to grade")
    p_grade.add_argument("--output", default="ABLATE1_GEMINI_GRADED.csv", help="Output CSV")
    p_grade.add_argument("--model", default="gemini-2.5-pro", help="Gemini model ID")
    p_grade.set_defaults(func=cmd_grade)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
