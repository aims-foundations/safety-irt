#!/usr/bin/env python3
"""
Extract prompt pairs (English + target language) for human translation validation.
Outputs one CSV per language with columns: id, tags, prompt_en, prompt_target
Plus a combined sheet with all languages.
"""
import pandas as pd
import os

MULTIJAIL = "multijail.csv"  # change path as needed
OUT_DIR = "human_validation_pairs"
os.makedirs(OUT_DIR, exist_ok=True)

# How many to sample per language (None = all 315)
SAMPLE_N = None  # set e.g. 50 for a smaller batch
SEED = 42

def main():
    mj = pd.read_csv(MULTIJAIL)
    mj["id"] = mj["id"].astype(str).str.strip()
    print(f"Loaded {len(mj)} rows, {mj['language'].nunique()} languages")

    # Get English prompts
    en = mj[mj["language"] == "en"][["id", "tags", "prompt"]].copy()
    en = en.rename(columns={"prompt": "prompt_en"})
    en = en.drop_duplicates("id")
    print(f"  {len(en)} English prompts")

    languages = sorted(mj["language"].unique())
    languages = [l for l in languages if l != "en"]

    all_pairs = []

    for lang in languages:
        target = mj[mj["language"] == lang][["id", "prompt"]].copy()
        target = target.rename(columns={"prompt": "prompt_target"})
        target = target.drop_duplicates("id")

        pairs = en.merge(target, on="id", how="inner")
        pairs["language"] = lang
        pairs["translation_quality"] = ""  # blank column for human annotator

        if SAMPLE_N and len(pairs) > SAMPLE_N:
            pairs = pairs.sample(SAMPLE_N, random_state=SEED)

        pairs = pairs.sort_values("id", key=lambda x: x.astype(int))

        # Save per-language file
        out_cols = ["id", "tags", "language", "prompt_en", "prompt_target", "translation_quality"]
        path = os.path.join(OUT_DIR, f"pairs_{lang}.csv")
        pairs[out_cols].to_csv(path, index=False)
        print(f"  {lang}: {len(pairs)} pairs → {path}")

        all_pairs.append(pairs[out_cols])

    # Combined file
    combined = pd.concat(all_pairs, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "all_pairs_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined: {len(combined)} pairs → {combined_path}")


if __name__ == "__main__":
    main()