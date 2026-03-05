"""Download xsafety.csv from the MaxZ119/safetyirt HuggingFace dataset repo."""

import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "MaxZ119/safetyirt"
FILENAME = "xsafety.csv"
OUTPUT_FILE = "xsafety.csv"


def prepare_xsafety_data():
    print(f"Downloading '{FILENAME}' from HuggingFace repo '{REPO_ID}'...")
    try:
        cached_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
        )
    except Exception as e:
        print(f"Error downloading file: {e}")
        return

    shutil.copy(cached_path, OUTPUT_FILE)
    print(f"Saved to '{OUTPUT_FILE}'.")

    import pandas as pd
    df = pd.read_csv(OUTPUT_FILE)
    print(f"Rows: {len(df)},  Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    prepare_xsafety_data()
