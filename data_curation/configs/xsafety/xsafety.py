"""Download XSafety benchmark directly from GitHub and produce a parallel CSV.

No git clone needed. Fetches from raw.githubusercontent.com.

Usage:
    python xsafety_to_csv.py --output xsafety.csv

Requires: pip install requests openpyxl
"""

import argparse, io, os, sys
import requests
import pandas as pd

RAW = "https://raw.githubusercontent.com/Jarviswang94/Multilingual_safety_benchmark/main"

CATEGORIES = [
    "Crimes_And_Illegal_Activities",
    "Ethics_And_Morality",
    "Goal_Hijacking",
    "Inquiry_With_Unsafe_Opinion",
    "Insult",
    "Mental_Health",
    "Physical_Harm",
    "Privacy_And_Property",
    "Prompt_Leaking",
    "Reverse_Exposure",
    "Role_Play_Instruction",
    "Unfairness_And_Discrimination",
    "Unsafe_Instruction_Topic",
    "commonsense",
]

LANGUAGES = ["en", "zh", "ar", "bn", "de", "fr", "hi", "ja", "ru", "sp"]

# Exact filenames per language (from repo scan)
FILES = {}

# English: _en or _n suffix
FILES["en"] = {c: f"{c}_en.csv" if c == "Crimes_And_Illegal_Activities" else
                  (f"{c}.csv" if c == "commonsense" else f"{c}_n.csv")
               for c in CATEGORIES}

# Bengali: xlsx with _n suffix (except commonsense)
FILES["bn"] = {c: f"{c}.csv" if c == "commonsense" else f"{c}_n.xlsx"
               for c in CATEGORIES}

# Russian: typo in Crimes filename
FILES["ru"] = {c: f"{c}.csv" for c in CATEGORIES}
FILES["ru"]["Crimes_And_Illegal_Activities"] = "Crimes_And_Illegal_Activitie.csv"

# Standard langs: {category}.csv
for lang in ["zh", "ar", "de", "fr", "hi", "ja", "sp"]:
    FILES[lang] = {c: f"{c}.csv" for c in CATEGORIES}


def fetch(lang, filename):
    """Download file, return list of prompt strings."""
    url = f"{RAW}/{lang}/{filename}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"    FAIL {r.status_code}: {lang}/{filename}")
        return None
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(io.StringIO(r.text), header=None)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(r.content), header=None)
    else:
        return None
    return df.iloc[:, 0].dropna().astype(str).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="xsafety.csv")
    parser.add_argument("--per-category", type=int, default=None,
                        help="Random sample N prompts per category (e.g. 22 for ~308 total)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    gid = 0

    for cat in CATEGORIES:
        label = cat.replace("_", " ")

        en = fetch("en", FILES["en"][cat])
        if en is None:
            print(f"  SKIP {label}: English failed")
            continue

        n = len(en)
        print(f"  {label}: {n} prompts", end="")

        data = {"en": en}
        for lang in LANGUAGES:
            if lang == "en":
                continue
            prompts = fetch(lang, FILES[lang][cat])
            if prompts is not None:
                data[lang] = prompts
                if len(prompts) != n:
                    print(f"  [{lang}:{len(prompts)}]", end="")

        print(f"  ({len(data)} langs)")

        for i in range(n):
            gid += 1
            for lang in LANGUAGES:
                p = data.get(lang)
                if p is None or i >= len(p) or not p[i].strip():
                    continue
                rows.append({
                    "id": gid,
                    "category": label,
                    "language": lang,
                    "prompt": p[i],
                    "prompt_en": en[i] if lang != "en" else "",
                })

    if not rows:
        print("ERROR: No rows.")
        return

    df = pd.DataFrame(rows)

    # Random subset: pick N prompt IDs per category, keep all languages for those IDs
    if args.per_category:
        import numpy as np
        rng = np.random.RandomState(args.seed)
        keep_ids = []
        for cat in df["category"].unique():
            cat_ids = df[df["category"] == cat]["id"].unique()
            n = min(args.per_category, len(cat_ids))
            sampled = rng.choice(cat_ids, size=n, replace=False)
            keep_ids.extend(sampled)
        df = df[df["id"].isin(set(keep_ids))].copy()
        print(f"\n  Subsampled: {args.per_category}/category, {len(set(keep_ids))} prompts, {len(df)} rows")

    df.to_csv(args.output, index=False)

    print(f"\nTotal rows:      {len(df)}")
    print(f"Unique prompts:  {df['id'].nunique()}")
    print(f"Languages:       {sorted(df['language'].unique())}")
    print(f"Categories:      {df['category'].nunique()}")
    print(f"Saved:           {args.output}")

    print(f"\nPer-language:")
    for lang, cnt in df.groupby("language").size().sort_index().items():
        print(f"  {lang}: {cnt}")

    en_ids = set(df[df["language"] == "en"]["id"])
    print(f"\nAlignment:")
    for lang in LANGUAGES:
        if lang == "en":
            continue
        lid = set(df[df["language"] == lang]["id"])
        miss = en_ids - lid
        if miss:
            print(f"  {lang}: missing {len(miss)}")
            md = df[(df["language"] == "en") & (df["id"].isin(miss))]
            for c, g in md.groupby("category"):
                print(f"       {c}: {len(g)}")
        else:
            print(f"  {lang}: OK ({len(lid)})")


if __name__ == "__main__":
    main()
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
