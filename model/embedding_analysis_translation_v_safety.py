# -*- coding: utf-8 -*-
"""
Multi-Metric Translation Quality Analysis vs Safety Rate.

Runs 4 translation quality metrics and correlates each with safety rate:
  1. LaBSE        — cross-lingual embedding cosine similarity
  2. COMET        — Unbabel/wmt22-comet-da (reference-based, trained on human DA)
  3. CometKiwi    — Unbabel/wmt22-cometkiwi-da (reference-free quality estimation)
  4. XCOMET-XL    — Unbabel/XCOMET-XL (3.5B, state-of-the-art, reference-based)

Output:
  - Per-metric CSV with scores + safety rates
  - Per-language and per-category Spearman tables with bootstrap CIs
  - Combined comparison plot
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

SEED = 42

import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import snapshot_download

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- CONFIGURATION ---
try:
    print("Locating dataset snapshot...")
    DATA_DIR = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
except Exception as e:
    print(f"Error downloading snapshot: {e}")
    DATA_DIR = "."

INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_CSV_DATA = os.path.join(RESULTS_DIR, "multimetric_translation_v_safety.csv")
OUTPUT_CSV_LANG = os.path.join(RESULTS_DIR, "multimetric_translation_v_safety_Lang.csv")
OUTPUT_CSV_CAT = os.path.join(RESULTS_DIR, "multimetric_translation_v_safety_Cat.csv")
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "Analysis_MultiMetric_vs_SafetyRate.png")

BATCH_SIZE_LABSE = 256
BATCH_SIZE_COMET = 64
BATCH_SIZE_COMETKIWI = 64
BATCH_SIZE_XCOMET = 8


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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    rho_hat = spearman_rho(x, y)
    indices = np.arange(n)
    resampled_idx = rng.choice(indices, size=(n_boot, n), replace=True)

    rhos = [spearman_rho(x[idx], y[idx]) for idx in resampled_idx]
    lo, hi = np.quantile(rhos, [0.025, 0.975])
    return rho_hat, float(lo), float(hi)


# =========================================================================
#  STEP 1: Load & aggregate
# =========================================================================
def load_and_aggregate():
    print(f"\nLoading raw data from {INPUT_FILE}...")
    raw_df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip")
    raw_df = raw_df.dropna(subset=["judge_score", "prompt", "language", "id"])
    raw_df["id"] = raw_df["id"].apply(clean_id)
    raw_df["is_safe"] = (raw_df["judge_score"] >= 4).astype(int)

    cat_df = raw_df[raw_df["language"] == "en"][["id", "tags"]].drop_duplicates(subset=["id"])
    cat_lookup = cat_df.set_index("id")["tags"].to_dict()
    print(f"  {len(cat_lookup)} category mappings from English rows.")

    agg_df = raw_df.groupby(["id", "language"], as_index=False).agg(
        safety_rate=("is_safe", "mean"),
        n=("is_safe", "size"),
        prompt=("prompt", "first"),
    )
    agg_df["category"] = agg_df["id"].map(cat_lookup).fillna("Unknown")
    print(f"  {len(raw_df)} rows -> {len(agg_df)} unique (id, language) pairs.")
    return agg_df


def build_pairs(df):
    print("\nBuilding text pairs...")
    text_lookup = df[["id", "language", "prompt"]].drop_duplicates(subset=["id", "language"])
    eng_lookup = text_lookup[text_lookup["language"] == "en"].set_index("id")["prompt"].to_dict()
    print(f"  {len(eng_lookup)} English reference prompts.")

    text_lookup = text_lookup.copy()
    text_lookup["key"] = text_lookup["id"].astype(str) + "_" + text_lookup["language"].astype(str)
    text_map = text_lookup.set_index("key")["prompt"].to_dict()

    target_df = df[df["language"] != "en"].copy()
    en_texts, target_texts, valid_indices = [], [], []

    for idx, row in target_df.iterrows():
        p_id = row["id"]
        lang = row["language"]
        en_txt = eng_lookup.get(p_id)
        tg_txt = text_map.get(f"{p_id}_{lang}")

        if isinstance(en_txt, str) and isinstance(tg_txt, str):
            en_texts.append(en_txt)
            target_texts.append(tg_txt)
            valid_indices.append(idx)

    print(f"  {len(en_texts)} valid text pairs.")
    return en_texts, target_texts, valid_indices, target_df


# =========================================================================
#  STEP 2: Metric scoring functions
# =========================================================================

def score_labse(en_texts, target_texts):
    print("\n--- [1/4] LaBSE ---")
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    en_emb = model.encode(en_texts, batch_size=BATCH_SIZE_LABSE, convert_to_tensor=True, device=device)
    tgt_emb = model.encode(target_texts, batch_size=BATCH_SIZE_LABSE, convert_to_tensor=True, device=device)
    sims = torch.nn.functional.cosine_similarity(en_emb, tgt_emb, dim=1).cpu().numpy()
    print(f"  Done. Mean={sims.mean():.4f}, Std={sims.std():.4f}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sims


def score_comet(en_texts, target_texts):
    print("\n--- [2/4] COMET (wmt22-comet-da) ---")
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    data = [{"src": s, "mt": t, "ref": s} for s, t in zip(en_texts, target_texts)]
    gpus = 1 if torch.cuda.is_available() else 0
    output = model.predict(data, batch_size=BATCH_SIZE_COMET, num_workers=1, gpus=gpus)
    scores = np.array(output.scores)
    print(f"  Done. Mean={scores.mean():.4f}, Std={scores.std():.4f}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def score_cometkiwi(en_texts, target_texts):
    print("\n--- [3/4] CometKiwi (reference-free) ---")
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)
    # Reference-free: only src + mt
    data = [{"src": s, "mt": t} for s, t in zip(en_texts, target_texts)]
    gpus = 1 if torch.cuda.is_available() else 0
    output = model.predict(data, batch_size=BATCH_SIZE_COMETKIWI, num_workers=1, gpus=gpus)
    scores = np.array(output.scores)
    print(f"  Done. Mean={scores.mean():.4f}, Std={scores.std():.4f}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def score_xcomet(en_texts, target_texts):
    print("\n--- [4/4] XCOMET-XL (3.5B params — this will be slow) ---")
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    data = [{"src": s, "mt": t, "ref": s} for s, t in zip(en_texts, target_texts)]
    gpus = 1 if torch.cuda.is_available() else 0
    output = model.predict(data, batch_size=BATCH_SIZE_XCOMET, num_workers=1, gpus=gpus)
    scores = np.array(output.scores)
    print(f"  Done. Mean={scores.mean():.4f}, Std={scores.std():.4f}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


# =========================================================================
#  STEP 3: Run all metrics
# =========================================================================
def compute_all_scores():
    agg_df = load_and_aggregate()
    en_texts, target_texts, valid_indices, target_df = build_pairs(agg_df)

    if not en_texts:
        print("No text pairs found.")
        return None

    result_df = target_df.loc[valid_indices].copy()

    result_df["labse"] = score_labse(en_texts, target_texts)
    result_df["comet"] = score_comet(en_texts, target_texts)
    result_df["cometkiwi"] = score_cometkiwi(en_texts, target_texts)

    try:
        result_df["xcomet_xl"] = score_xcomet(en_texts, target_texts)
    except Exception as e:
        print(f"  ⚠️  XCOMET-XL failed (likely OOM): {e}")
        print(f"  Skipping XCOMET-XL.")
        result_df["xcomet_xl"] = np.nan

    result_df.to_csv(OUTPUT_CSV_DATA, index=False)
    print(f"\n✅ All scores saved to {OUTPUT_CSV_DATA}")
    return result_df


# =========================================================================
#  STEP 4: Correlations + plots
# =========================================================================
METRIC_COLS = ["labse", "comet", "cometkiwi", "xcomet_xl"]
METRIC_LABELS = {
    "labse": "LaBSE",
    "comet": "COMET",
    "cometkiwi": "CometKiwi",
    "xcomet_xl": "XCOMET-XL",
}


def compute_correlations_and_plot():
    if not os.path.exists(OUTPUT_CSV_DATA):
        print(f"❌ {OUTPUT_CSV_DATA} not found.")
        return

    df = pd.read_csv(OUTPUT_CSV_DATA)
    available_metrics = [m for m in METRIC_COLS if m in df.columns and df[m].notna().any()]
    print(f"\nAvailable metrics: {available_metrics}")

    # --- Per-Language with bootstrap CIs ---
    lang_rows = []
    for metric in available_metrics:
        for lang, sub in df.groupby("language"):
            sub_valid = sub[sub[metric].notna()]
            if len(sub_valid) > 15:
                r, lo, hi = bootstrap_spearman_ci(sub_valid[metric].values, sub_valid["safety_rate"].values)
                lang_rows.append({
                    "Metric": METRIC_LABELS.get(metric, metric),
                    "Language": lang, "N": len(sub_valid),
                    "Spearman_Rho": r, "CI_Lower": lo, "CI_Upper": hi,
                })

    lang_df = pd.DataFrame(lang_rows)
    lang_df.to_csv(OUTPUT_CSV_LANG, index=False)
    print(f"Language correlations saved to {OUTPUT_CSV_LANG}")

    # --- Per-Category with bootstrap CIs ---
    cat_rows = []
    for metric in available_metrics:
        for cat, sub in df.groupby("category"):
            sub_valid = sub[sub[metric].notna()]
            if len(sub_valid) > 15:
                r, lo, hi = bootstrap_spearman_ci(sub_valid[metric].values, sub_valid["safety_rate"].values)
                cat_rows.append({
                    "Metric": METRIC_LABELS.get(metric, metric),
                    "Category": cat, "N": len(sub_valid),
                    "Spearman_Rho": r, "CI_Lower": lo, "CI_Upper": hi,
                })

    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(OUTPUT_CSV_CAT, index=False)
    print(f"Category correlations saved to {OUTPUT_CSV_CAT}")

    # --- Global stats ---
    print("\n--- Global Correlations (metric vs safety_rate) ---")
    global_stats = []
    for metric in available_metrics:
        sub = df[df[metric].notna()]
        r, lo, hi = bootstrap_spearman_ci(sub[metric].values, sub["safety_rate"].values)
        label = METRIC_LABELS.get(metric, metric)
        global_stats.append({"Metric": label, "Spearman_Rho": r, "CI_Lower": lo, "CI_Upper": hi})
        print(f"  {label:12s}: ρ={r:.3f} [{lo:.3f}, {hi:.3f}]")

    # --- Within-language centered ---
    print("\n--- Within-Language Centered Correlations ---")
    for metric in available_metrics:
        sub = df[df[metric].notna()].copy()
        sub["m_centered"] = sub[metric] - sub.groupby("language")[metric].transform("mean")
        sub["s_centered"] = sub["safety_rate"] - sub.groupby("language")["safety_rate"].transform("mean")
        r, lo, hi = bootstrap_spearman_ci(sub["m_centered"].values, sub["s_centered"].values)
        label = METRIC_LABELS.get(metric, metric)
        print(f"  {label:12s}: ρ={r:.3f} [{lo:.3f}, {hi:.3f}]")

    global_df = pd.DataFrame(global_stats).sort_values("Spearman_Rho")

    # --- PLOT ---
    print(f"\nGenerating plot to {OUTPUT_PLOT}...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 2]})

    # Panel 1: Global Spearman per metric with CI whiskers
    colors = ["#d62728" if x > 0 else "#1f77b4" for x in global_df["Spearman_Rho"]]
    bars = axes[0].barh(global_df["Metric"], global_df["Spearman_Rho"], color=colors)

    for i, (_, row) in enumerate(global_df.iterrows()):
        axes[0].plot([row["CI_Lower"], row["CI_Upper"]], [i, i], color="black", linewidth=1.5)

    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Global: Metric vs Safety Rate", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Spearman's ρ (with 95% CI)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # Panel 2: Heatmap of per-language rho by metric
    pivot = lang_df.pivot_table(index="Language", columns="Metric", values="Spearman_Rho")
    col_order = [METRIC_LABELS.get(m, m) for m in available_metrics if METRIC_LABELS.get(m, m) in pivot.columns]
    pivot = pivot[col_order]

    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                ax=axes[1], linewidths=0.5, cbar_kws={"label": "Spearman ρ"})
    axes[1].set_title("Per-Language: Metric vs Safety Rate", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Plot saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_CSV_DATA):
        compute_all_scores()

    compute_correlations_and_plot()