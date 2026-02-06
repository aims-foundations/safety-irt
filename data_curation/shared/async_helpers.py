"""Shared async infrastructure for test-taker response collection."""

import asyncio
import os
import pandas as pd
from tqdm.asyncio import tqdm


def create_csv_saver(output_file, buffer_size=50):
    """Create a buffer-based CSV saver.

    Returns (result_buffer, save_buffer_fn, save_lock).
    Append results to result_buffer; call save_buffer_fn() when
    len(result_buffer) >= buffer_size or at the end of a run.
    """
    result_buffer = []
    save_lock = asyncio.Lock()

    async def save_buffer():
        async with save_lock:
            if result_buffer:
                df_save = pd.DataFrame(result_buffer)
                file_exists = os.path.exists(output_file)
                df_save.to_csv(
                    output_file, mode='a', header=not file_exists,
                    index=False, encoding="utf-8"
                )
                result_buffer.clear()

    return result_buffer, save_buffer, save_lock


def build_resume_keys(output_file, use_pass=True):
    """Read an existing output CSV and return a set of completed keys.

    Keys are (id, language, test_taker) or (id, language, test_taker, pass)
    depending on use_pass.
    """
    completed_keys = set()

    if not os.path.exists(output_file):
        print("Starting fresh.")
        return completed_keys

    print(f"Checking existing progress in '{output_file}'...")
    try:
        existing_df = pd.read_csv(
            output_file, encoding="utf-8",
            on_bad_lines='skip', engine='python'
        )

        if use_pass and 'pass' in existing_df.columns:
            for _, row in existing_df.iterrows():
                key = (row['id'], row['language'], row['test_taker'], row['pass'])
                completed_keys.add(key)
        elif not use_pass:
            for _, row in existing_df.iterrows():
                key = (row['id'], row['language'], row['test_taker'])
                completed_keys.add(key)
        else:
            print("Existing file missing 'pass' column. Starting fresh advised.")

        print(f"Resuming... {len(completed_keys)} responses already found.")
    except Exception as e:
        print(f"Could not read existing file: {e}. Starting fresh.")

    return completed_keys


async def run_generation_loop(tasks, desc="Generation", heartbeat_interval=1000):
    """Run a list of asyncio tasks with a tqdm progress bar and heartbeat logging."""
    if not tasks:
        print("All tasks are already complete!")
        return

    print(f"Queueing {len(tasks)} tasks...")

    for i, f in enumerate(tqdm(
        asyncio.as_completed(tasks), total=len(tasks),
        desc=desc, miniters=100
    )):
        await f
        if i % heartbeat_interval == 0 and i > 0:
            tqdm.write(f"Alive: Completed {i} requests...")
