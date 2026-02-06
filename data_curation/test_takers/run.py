"""Unified test-taker response collection runner.

Usage:
    python -m data_curation.test_takers.run --config gpt
    python -m data_curation.test_takers.run --config gemini --input multijail.csv
    python -m data_curation.test_takers.run --config claude_3 --dry-run
"""

import argparse
import asyncio
import importlib
import os
import random
import sys

import pandas as pd
from tqdm.asyncio import tqdm

# Add parent to path for imports when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_curation.shared.async_helpers import (
    create_csv_saver,
    build_resume_keys,
    run_generation_loop,
)


def load_config(config_name):
    """Dynamically import a config module from test_takers/configs/."""
    module = importlib.import_module(f"data_curation.test_takers.configs.{config_name}")
    return module


async def process_single_request(
    config_module, client, model, prompt, prompt_id, language,
    variant_name, settings, pass_index, semaphore,
    result_buffer, save_buffer_fn, retries
):
    """Send one API request with retries and buffer the result."""
    test_taker_name = f"{model}_{variant_name}"

    async with semaphore:
        for attempt in range(retries):
            try:
                response_text, finish_reason = await config_module.call_api(
                    client, model, prompt, settings
                )

                result_data = {
                    "id": prompt_id,
                    "language": language,
                    "model": model,
                    "variant": variant_name,
                    "test_taker": test_taker_name,
                    "pass": pass_index,
                    "response": response_text,
                    "finish_reason": finish_reason,
                    "temperature": settings.get("temp", "default"),
                    "top_p": settings.get("top_p", "default"),
                }

                result_buffer.append(result_data)
                buffer_size = config_module.CONFIG.get("buffer_size", 50)
                if len(result_buffer) >= buffer_size:
                    await save_buffer_fn()
                return result_data

            except Exception as e:
                error_str = str(e)
                status_code = getattr(e, 'status_code', None)

                # Rate limit / overloaded: backoff and retry
                if status_code in [429, 529] or "overloaded" in error_str.lower():
                    wait_time = (5 * (attempt + 1)) + random.uniform(0, 3)
                    if attempt > 1:
                        tqdm.write(f"Rate limited ({status_code}) - {test_taker_name} waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue

                # Content filter: record and move on
                if status_code in [400, 403] and ("content_filter" in error_str or status_code == 403):
                    filter_data = {
                        "id": prompt_id, "language": language, "model": model,
                        "variant": variant_name, "test_taker": test_taker_name,
                        "pass": pass_index,
                        "response": f"[BLOCKED_BY_API] {error_str}",
                        "finish_reason": "content_filter",
                        "temperature": settings.get("temp", "default"),
                        "top_p": settings.get("top_p", "default"),
                    }
                    result_buffer.append(filter_data)
                    return filter_data

                # Model not found: record and move on
                if status_code == 404:
                    fail_data = {
                        "id": prompt_id, "language": language, "model": model,
                        "variant": variant_name, "test_taker": test_taker_name,
                        "pass": pass_index,
                        "response": "ERROR_MODEL_NOT_FOUND",
                        "finish_reason": "error",
                        "temperature": settings.get("temp", "default"),
                        "top_p": settings.get("top_p", "default"),
                    }
                    result_buffer.append(fail_data)
                    return fail_data

                tqdm.write(f"Error {test_taker_name}: {e}")

                # Last attempt: record failure
                if attempt == retries - 1:
                    fail_data = {
                        "id": prompt_id, "language": language, "model": model,
                        "variant": variant_name, "test_taker": test_taker_name,
                        "pass": pass_index,
                        "response": "ERROR_FAILED_RETRIES",
                        "finish_reason": "error",
                        "temperature": settings.get("temp", "default"),
                        "top_p": settings.get("top_p", "default"),
                    }
                    result_buffer.append(fail_data)
                    return fail_data

                await asyncio.sleep(2 ** attempt + 1)


async def main(config_name, input_file, dry_run=False):
    config_module = load_config(config_name)
    cfg = config_module.CONFIG

    output_file = cfg["output_file"]
    models = cfg["models"]
    variants = cfg["variants"]
    num_passes = cfg.get("num_passes", 2)
    retries = cfg.get("retries", 3)
    use_pass = num_passes > 1

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    df = pd.read_csv(input_file)
    print(f"Config:     {config_name}")
    print(f"Models:     {models}")
    print(f"Variants:   {list(variants.keys())}")
    print(f"Passes:     {num_passes}")
    print(f"Input rows: {len(df)}")

    # Build resume keys
    completed_keys = build_resume_keys(output_file, use_pass=use_pass)

    # Count tasks
    tasks_info = []
    skipped_count = 0

    for _, row in df.iterrows():
        for model in models:
            for variant_name, settings in variants.items():
                test_taker_name = f"{model}_{variant_name}"
                for pass_index in range(num_passes):
                    if use_pass:
                        unique_key = (row['id'], row['language'], test_taker_name, pass_index)
                    else:
                        unique_key = (row['id'], row['language'], test_taker_name)

                    if unique_key in completed_keys:
                        skipped_count += 1
                        continue

                    tasks_info.append((model, row['prompt'], row['id'], row['language'],
                                       variant_name, settings, pass_index))

    total_expected = len(df) * len(models) * len(variants) * num_passes
    print(f"Target total:     {total_expected}")
    print(f"Already complete: {skipped_count}")
    print(f"Remaining:        {len(tasks_info)}")

    if dry_run:
        print("\n[DRY RUN] Would process the above tasks. Exiting.")
        return

    if not tasks_info:
        print("All tasks are already complete!")
        return

    # Create client and async infrastructure
    client = config_module.create_client()
    semaphore = asyncio.Semaphore(cfg.get("semaphore_limit", 10))
    result_buffer, save_buffer_fn, _ = create_csv_saver(output_file, cfg.get("buffer_size", 50))

    # Create async tasks
    tasks = [
        asyncio.create_task(
            process_single_request(
                config_module, client, model, prompt, prompt_id, language,
                variant_name, settings, pass_index, semaphore,
                result_buffer, save_buffer_fn, retries
            )
        )
        for model, prompt, prompt_id, language, variant_name, settings, pass_index in tasks_info
    ]

    # Run
    await run_generation_loop(tasks, desc=f"{cfg['name']} Generation")
    await save_buffer_fn()
    print(f"\nDone! Results saved to {output_file}")


def cli():
    parser = argparse.ArgumentParser(
        description="Unified test-taker response collection"
    )
    parser.add_argument(
        "--config", required=True,
        help="Config module name (e.g., gpt, gemini, claude_3, grok, deepseek, "
             "claude_4_5_low_creativity, claude_4_5_high_risk)"
    )
    parser.add_argument(
        "--input", default="multijail.csv",
        help="Input CSV file (default: multijail.csv)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load config and count tasks without making API calls"
    )
    args = parser.parse_args()

    asyncio.run(main(args.config, args.input, args.dry_run))


if __name__ == "__main__":
    cli()
