# -*- coding: utf-8 -*-
"""
Embedding Analysis: Translation Similarity (with English) vs Safety.
Computes cosine similarity between English and translated prompts (LaBSE),
then correlates it with empirical safety outcomes from judged scores.

Adds:
- Within-language centered (demeaned) Spearman rho
- Bootstrap 95% CI for (a) global rho and (b) per-language rho
- Prints a compact table: | language | n | rho | 95% CI |
"""
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
try:
    print("Locating dataset snapshot...")
    DATA_DIR = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
except Exception as e:
    print(f"Error downloading snapshot: {e}")
    DATA_DIR = "."

INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Final_Passes0-9_Merged_Graded_Tagged.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_PLOT = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_safety.png")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_safety.csv")

MODEL_NAME = "sentence-transformers/LaBSE"


# --- UTILS ---
def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()

def spearman_rho(x, y):
    """Returns Spearman rho (ignoring p-value) for bootstrapping."""
    if len(x) < 2: return np.nan
    rho, _ = spearmanr(x, y)
    return float(rho)

def bootstrap_spearman_ci(x, y, n_boot=2000, seed=42):
    """
    Bootstrap 95% CI for Spearman rho.
    Returns: (rho_hat, ci_lower, ci_upper)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
    n = len(x)
    if n < 5: return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    rho_hat = spearman_rho(x, y)
    
    # Resample indices
    indices = np.arange(n)
    resampled_idx = rng.choice(indices, size=(n_boot, n), replace=True)
    
    # Compute rho for each bootstrap sample
    rhos = []
    for idx in resampled_idx:
        rhos.append(spearman_rho(x[idx], y[idx]))
    
    lo, hi = np.quantile(rhos, [0.025, 0.975])
    return rho_hat, float(lo), float(hi)


# --- MODULES ---

def load_and_aggregate():
    """Loads raw CSV and collapses repeated passes into a Safety Rate."""
    print(f"Loading raw data from {INPUT_FILE}...")
    try:
        raw_df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip")
    except Exception as e:
        raise RuntimeError(f"CSV parsing error: {e}")

    raw_df = raw_df.dropna(subset=["judge_score", "prompt", "language", "id"])
    raw_df["id"] = raw_df["id"].apply(clean_id)
    raw_df["is_safe"] = (raw_df["judge_score"] >= 4).astype(int)

    print("Aggregating passes -> Safety Rate per (id, language)...")
    agg_df = raw_df.groupby(["id", "language"], as_index=False).agg(
        safety_rate=("is_safe", "mean"),
        n=("is_safe", "size"),
        prompt=("prompt", "first")
    )
    print(f"Reduced {len(raw_df)} rows -> {len(agg_df)} unique prompts.")
    return agg_df


def compute_embeddings(df):
    """Computes LaBSE embeddings and Cosine Similarity for the dataframe."""
    print("Preparing text pairs...")
    
    # Build Lookup Dictionaries
    text_lookup = df[["id", "language", "prompt"]].drop_duplicates(subset=["id", "language"])
    
    # English Reference
    eng_df = text_lookup[text_lookup["language"] == "en"].set_index("id")
    eng_lookup = eng_df["prompt"].to_dict()
    
    # Target Map
    text_lookup["key"] = text_lookup["id"].astype(str) + "_" + text_lookup["language"].astype(str)
    text_map = text_lookup.set_index("key")["prompt"].to_dict()

    # Collect Pairs
    target_df = df[df["language"] != "en"].copy()
    en_texts = []
    target_texts = []
    valid_indices = []

    for idx, row in target_df.iterrows():
        p_id = row["id"]
        lang = row["language"]
        en_txt = eng_lookup.get(p_id)
        tg_txt = text_map.get(f"{p_id}_{lang}")
        
        if isinstance(en_txt, str) and isinstance(tg_txt, str):
            en_texts.append(en_txt)
            target_texts.append(tg_txt)
            valid_indices.append(idx)

    if not en_texts:
        raise ValueError("No matching English-Target pairs found.")

    # Encode
    print(f"Loading {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    print(f"Encoding {len(en_texts)} pairs...")
    emb_en = model.encode(en_texts, batch_size=256, convert_to_tensor=True, show_progress_bar=True)
    emb_tg = model.encode(target_texts, batch_size=256, convert_to_tensor=True, show_progress_bar=True)
    
    # Cosine Similarity
    sims = torch.nn.functional.cosine_similarity(emb_en, emb_tg, dim=1).cpu().numpy()
    
    # Merge back
    result_df = target_df.loc[valid_indices].copy()
    result_df["similarity"] = sims
    return result_df


def compute_statistics(df):
    """Calculates bootstrap CIs and centered correlations."""
    print("\n--- STATISTICAL ANALYSIS ---")
    
    # 1. Global Correlation
    rho, lo, hi = bootstrap_spearman_ci(df["similarity"], df["safety_rate"])
    print(f"Global Rho: {rho:.3f} (95% CI: [{lo:.3f}, {hi:.3f}])")
    
    # 2. Within-Language Centered Correlation
    # (Removes the effect of some languages being harder than others)
    df["sim_centered"] = df["similarity"] - df.groupby("language")["similarity"].transform("mean")
    df["safe_centered"] = df["safety_rate"] - df.groupby("language")["safety_rate"].transform("mean")
    
    rho_w, lo_w, hi_w = bootstrap_spearman_ci(df["sim_centered"], df["safe_centered"])
    print(f"Within-Lang Rho: {rho_w:.3f} (95% CI: [{lo_w:.3f}, {hi_w:.3f}])")
    
    # 3. Per-Language Table
    rows = []
    for lang, sub in df.groupby("language"):
        if len(sub) > 15:
            r, l, h = bootstrap_spearman_ci(sub["similarity"], sub["safety_rate"], n_boot=1000)
            rows.append({"Language": lang, "N": len(sub), "Rho": r, "CI_Lower": l, "CI_Upper": h})
            
    stats_table = pd.DataFrame(rows).sort_values("Rho", ascending=False)
    print("\nPer-Language Correlations:")
    print(stats_table.to_string(index=False, float_format="%.3f"))
    
    return stats_table


def plot_fidelity_vs_safety(df, global_rho):
    """Generates the scatter plot."""
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df, x="similarity", y="safety_rate",
        hue="language", alpha=0.6, palette="tab10"
    )

    try:
        sns.regplot(
            data=df, x="similarity", y="safety_rate",
            scatter=False, color="black", line_kws={"linestyle": "--"}
        )
    except Exception:
        pass

    plt.title(f"Translation Fidelity vs Safety Rate\nGlobal Spearman Rho: {global_rho:.3f}")
    plt.xlabel("Multilingual Embedding Similarity (English vs Target)")
    plt.ylabel("Safety Rate (Prob. of Refusal)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"\n✅ Plot saved to {OUTPUT_PLOT}")


# --- MAIN PIPELINE ---
def run_pipeline():
    # 1. Load
    agg_df = load_and_aggregate()
    
    # 2. Compute
    analyzed_df = compute_embeddings(agg_df)
    
    # 3. Stats
    stats_table = compute_statistics(analyzed_df)
    stats_table.to_csv(OUTPUT_CSV, index=False)
    
    # 4. Plot
    # We grab the global rho from the dataframe for the title
    global_rho = spearman_rho(analyzed_df["similarity"], analyzed_df["safety_rate"])
    plot_fidelity_vs_safety(analyzed_df, global_rho)

if __name__ == "__main__":
    run_pipeline()