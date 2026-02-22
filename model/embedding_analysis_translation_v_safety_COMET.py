# -*- coding: utf-8 -*-
"""
Translation Quality Analysis: COMET Score vs Safety Rate.
Uses Unbabel/wmt22-comet-da (trained on human translation quality judgments)
instead of LaBSE cosine similarity.

Outputs:
1. CSV of raw COMET scores (base data).
2. CSV of per-language correlation table with bootstrap CIs.
3. CSV of per-category correlation table with bootstrap CIs.
4. Plot: COMET Translation Quality vs Safety Rate.
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import snapshot_download

# COMET imports
from comet import download_model, load_from_checkpoint


# --- CONFIGURATION ---
try:
    print("Locating dataset snapshot...")
    DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
except Exception as e:
    print(f"Error downloading snapshot: {e}")
    DATA_DIR = "."

INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_PLOT = os.path.join(RESULTS_DIR, "comet_analysis_translation_v_safety.png")
OUTPUT_CSV_DATA = os.path.join(RESULTS_DIR, "comet_analysis_translation_v_safety.csv")
OUTPUT_CSV_LANG = os.path.join(RESULTS_DIR, "comet_analysis_translation_v_safety_Lang.csv")
OUTPUT_CSV_CAT = os.path.join(RESULTS_DIR, "comet_analysis_translation_v_safety_Cat.csv")

# COMET model: reference-based, trained on human DA judgments, scores 0-1
COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"

# Batch size for COMET inference (adjust down if OOM)
COMET_BATCH_SIZE = 64


# --- UTILS ---
def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def spearman_rho(x, y):
    if len(x) < 2:
        return np.nan
    rho, _ = spearmanr(x, y)
    return float(rho)


def bootstrap_spearman_ci(x, y, n_boot=2000, seed=42):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    rho_hat = spearman_rho(x, y)

    indices = np.arange(n)
    resampled_idx = rng.choice(indices, size=(n_boot, n), replace=True)

    rhos = []
    for idx in resampled_idx:
        rhos.append(spearman_rho(x[idx], y[idx]))

    lo, hi = np.quantile(rhos, [0.025, 0.975])
    return rho_hat, float(lo), float(hi)


# --- MODULES ---

def load_and_aggregate():
    """Loads raw CSV, extracts categories, and collapses passes into Safety Rate."""
    print(f"Loading raw data from {INPUT_FILE}...")
    try:
        raw_df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip")
    except Exception as e:
        raise RuntimeError(f"CSV parsing error: {e}")

    raw_df = raw_df.dropna(subset=["judge_score", "prompt", "language", "id"])
    raw_df["id"] = raw_df["id"].apply(clean_id)
    raw_df["is_safe"] = (raw_df["judge_score"] >= 4).astype(int)

    # --- Extract category lookup from 'tags' column ---
    cat_df = raw_df[raw_df["language"] == "en"][["id", "tags"]].drop_duplicates(subset=["id"])
    cat_lookup = cat_df.set_index("id")["tags"].to_dict()
    print(f"Found {len(cat_lookup)} category mappings from English rows.")

    print("Aggregating passes -> Safety Rate per (id, language)...")
    agg_df = raw_df.groupby(["id", "language"], as_index=False).agg(
        safety_rate=("is_safe", "mean"),
        n=("is_safe", "size"),
        prompt=("prompt", "first")
    )

    # Attach category to each row
    agg_df["category"] = agg_df["id"].map(cat_lookup).fillna("Unknown")

    print(f"Reduced {len(raw_df)} rows -> {len(agg_df)} unique (id, language) pairs.")
    return agg_df


def compute_comet_scores(df):
    """
    Computes COMET translation quality scores.
    
    COMET (wmt22-comet-da) takes (src, mt, ref) triples:
      - src = English prompt (the original source)
      - mt  = Target language prompt (the translation being evaluated)
      - ref = English prompt (same as src; English is the reference)
    
    Returns scores in [0, 1] where 1 = perfect translation quality.
    """
    print("Preparing text pairs for COMET scoring...")

    text_lookup = df[["id", "language", "prompt"]].drop_duplicates(subset=["id", "language"])

    eng_df = text_lookup[text_lookup["language"] == "en"].set_index("id")
    eng_lookup = eng_df["prompt"].to_dict()
    print(f"Found {len(eng_lookup)} unique English reference prompts.")

    text_lookup["key"] = text_lookup["id"].astype(str) + "_" + text_lookup["language"].astype(str)
    text_map = text_lookup.set_index("key")["prompt"].to_dict()

    target_df = df[df["language"] != "en"].copy()
    comet_data = []
    valid_indices = []

    for idx, row in target_df.iterrows():
        p_id = row["id"]
        lang = row["language"]
        en_txt = eng_lookup.get(p_id)
        tg_txt = text_map.get(f"{p_id}_{lang}")

        if isinstance(en_txt, str) and isinstance(tg_txt, str):
            comet_data.append({
                "src": en_txt,
                "mt": tg_txt,
                "ref": en_txt,
            })
            valid_indices.append(idx)

    if not comet_data:
        raise ValueError("No matching English-Target pairs found.")

    # Load COMET model
    print(f"Downloading and loading COMET model: {COMET_MODEL_NAME}")
    model_path = download_model(COMET_MODEL_NAME)
    model = load_from_checkpoint(model_path)

    # Score all pairs
    print(f"Scoring {len(comet_data)} translation pairs with COMET...")
    import torch
    output = model.predict(comet_data, batch_size=COMET_BATCH_SIZE, num_workers=1, gpus=1 if __import__('torch').cuda.is_available() else 0)


    scores = output.scores

    result_df = target_df.loc[valid_indices].copy()
    result_df["comet_score"] = scores

    # Save base data
    result_df.to_csv(OUTPUT_CSV_DATA, index=False)
    print(f"Base data saved to {OUTPUT_CSV_DATA}")

    return result_df


def compute_statistics(df):
    """Calculates bootstrap CIs: global, within-language centered, per-language, per-category."""
    print("\n--- STATISTICAL ANALYSIS ---")

    # 1. Global Correlation
    rho, lo, hi = bootstrap_spearman_ci(df["comet_score"], df["safety_rate"])
    print(f"Global Rho: {rho:.3f} (95% CI: [{lo:.3f}, {hi:.3f}])")

    # 2. Within-Language Centered Correlation
    df = df.copy()
    df["score_centered"] = df["comet_score"] - df.groupby("language")["comet_score"].transform("mean")
    df["safe_centered"] = df["safety_rate"] - df.groupby("language")["safety_rate"].transform("mean")

    rho_w, lo_w, hi_w = bootstrap_spearman_ci(df["score_centered"], df["safe_centered"])
    print(f"Within-Lang Rho: {rho_w:.3f} (95% CI: [{lo_w:.3f}, {hi_w:.3f}])")

    # 3. Per-Language Table
    print("\n[Per-Language Correlations]")
    lang_rows = []
    for lang, sub in df.groupby("language"):
        if len(sub) > 15:
            r, l, h = bootstrap_spearman_ci(sub["comet_score"], sub["safety_rate"], n_boot=2000)
            lang_rows.append({
                "Language": lang,
                "N": len(sub),
                "Spearman_Rho": r,
                "CI_Lower": l,
                "CI_Upper": h
            })

    lang_df = pd.DataFrame(lang_rows).sort_values("Spearman_Rho", ascending=False)
    print(lang_df.to_string(index=False, float_format="%.3f"))
    lang_df.to_csv(OUTPUT_CSV_LANG, index=False)
    print(f"Saved to {OUTPUT_CSV_LANG}")

    # 4. Per-Category Table
    print("\n[Per-Category Correlations]")
    cat_rows = []
    for cat, sub in df.groupby("category"):
        if len(sub) > 15:
            r, l, h = bootstrap_spearman_ci(sub["comet_score"], sub["safety_rate"], n_boot=2000)
            cat_rows.append({
                "Category": cat,
                "N": len(sub),
                "Spearman_Rho": r,
                "CI_Lower": l,
                "CI_Upper": h
            })

    cat_df = pd.DataFrame(cat_rows).sort_values("Spearman_Rho", ascending=False)
    print(cat_df.to_string(index=False, float_format="%.3f"))
    cat_df.to_csv(OUTPUT_CSV_CAT, index=False)
    print(f"Saved to {OUTPUT_CSV_CAT}")

    return lang_df, cat_df


def plot_fidelity_vs_safety(df, global_rho):
    """Generates the scatter plot using COMET scores."""
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df, x="comet_score", y="safety_rate",
        hue="language", alpha=0.6, palette="tab10"
    )

    try:
        sns.regplot(
            data=df, x="comet_score", y="safety_rate",
            scatter=False, color="black", line_kws={"linestyle": "--"}
        )
    except Exception:
        pass

    plt.title(f"COMET Translation Quality vs Safety Rate\nGlobal Spearman ρ: {global_rho:.3f}")
    plt.xlabel("COMET Translation Quality Score (0 = poor, 1 = perfect)")
    plt.ylabel("Safety Rate (Prob. of Refusal)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"\nPlot saved to {OUTPUT_PLOT}")


# --- MAIN PIPELINE ---
def run_pipeline():
    # 1. Load & aggregate (includes category)
    agg_df = load_and_aggregate()

    # 2. Compute COMET scores
    analyzed_df = compute_comet_scores(agg_df)

    # 3. Stats: global, per-language, per-category
    lang_df, cat_df = compute_statistics(analyzed_df)

    # 4. Plot
    global_rho = spearman_rho(analyzed_df["comet_score"], analyzed_df["safety_rate"])
    plot_fidelity_vs_safety(analyzed_df, global_rho)

    print("\n--- DONE ---")
    print(f"  Base data:  {OUTPUT_CSV_DATA}")
    print(f"  Lang table: {OUTPUT_CSV_LANG}")
    print(f"  Cat table:  {OUTPUT_CSV_CAT}")
    print(f"  Plot:       {OUTPUT_PLOT}")


if __name__ == "__main__":
    run_pipeline()