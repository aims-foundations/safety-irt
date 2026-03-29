# -*- coding: utf-8 -*-
"""
Multi-Metric Translation Quality Analysis vs Safety Cost (Tau).

Runs 4 translation quality metrics and correlates each with τ:
  1. LaBSE        — cross-lingual embedding cosine similarity
  2. COMET        — Unbabel/wmt22-comet-da (reference-based, trained on human DA)
  3. CometKiwi    — Unbabel/wmt22-cometkiwi-da (reference-free quality estimation)
  4. XCOMET-XL    — Unbabel/XCOMET-XL (3.5B, state-of-the-art, reference-based)

Output:
  - Per-metric CSV with scores
  - Per-language and per-category Spearman tables
  - Combined bar chart + heatmap comparing all metrics
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fig_style import *

import ast
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
import seaborn as sns
from huggingface_hub import snapshot_download

apply_style()

SEED = 42
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

IRT_RESULTS_FILE = os.path.join(RESULTS_DIR, "bayesian_irt_results_binary.csv")
MULTIJAIL_FILE = os.path.join(DATA_DIR, "multijail.csv")

OUTPUT_CSV_DATA = os.path.join(RESULTS_DIR, "multimetric_translation_v_DIF.csv")
OUTPUT_CSV_LANG = os.path.join(RESULTS_DIR, "multimetric_translation_v_DIF_Lang.csv")
OUTPUT_CSV_CAT = os.path.join(RESULTS_DIR, "multimetric_translation_v_DIF_Cat.csv")
OUTPUT_CSV_SUMMARY = os.path.join(RESULTS_DIR, "multimetric_translation_v_DIF_Summary.csv")
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "Analysis_MultiMetric_vs_SafetyTax.png")


BATCH_SIZE_LABSE = 256
BATCH_SIZE_COMET = 64
BATCH_SIZE_COMETKIWI = 64
BATCH_SIZE_XCOMET = 8  # 3.5B model — keep small


# --- UTILS ---
def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def parse_tags_cell(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            out = ast.literal_eval(s)
            if isinstance(out, list):
                return [str(t).strip() for t in out if str(t).strip()]
        except Exception:
            pass
    return [s]


def detect_category_column(df):
    candidates = ["tags", "tag", "category", "categories", "label", "labels"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower()
        if "tag" in lc or "cat" in lc:
            return c
    raise KeyError(f"Could not find tags/category column. Columns: {list(df.columns)}")


def build_multijail_category_lookup():
    if not os.path.exists(MULTIJAIL_FILE):
        raise FileNotFoundError(f"multijail.csv not found at {MULTIJAIL_FILE}")

    mj = pd.read_csv(MULTIJAIL_FILE)
    cat_col = detect_category_column(mj)
    mj["id"] = mj["id"].apply(clean_id)
    mj["language"] = mj["language"].astype(str).str.strip()
    mj["category_list"] = mj[cat_col].apply(parse_tags_cell)

    lookup_by_id_lang = {}
    for _, r in mj.iterrows():
        key = (r["id"], r["language"])
        if key not in lookup_by_id_lang:
            lookup_by_id_lang[key] = list(r["category_list"])
        else:
            lookup_by_id_lang[key] = sorted(set(lookup_by_id_lang[key]).union(r["category_list"]))

    lookup_by_id = {}
    for _, r in mj.iterrows():
        i = r["id"]
        if i not in lookup_by_id:
            lookup_by_id[i] = list(r["category_list"])
        else:
            lookup_by_id[i] = sorted(set(lookup_by_id[i]).union(r["category_list"]))

    print(f"  Loaded multijail tags from column '{cat_col}'.")
    return lookup_by_id_lang, lookup_by_id


# =========================================================================
#  STEP 1: Build text pairs + metadata
# =========================================================================
def build_pairs():
    print("\n=== Building text pairs ===")

    irt_df = pd.read_csv(IRT_RESULTS_FILE)
    for old in ["prompt", "prompt_id", "item"]:
        if old in irt_df.columns:
            irt_df.rename(columns={old: "id"}, inplace=True)
            break
    if "id" not in irt_df.columns:
        irt_df.rename(columns={irt_df.columns[0]: "id"}, inplace=True)

    irt_df["id"] = irt_df["id"].apply(clean_id)
    irt_df["language"] = irt_df["language"].astype(str).str.strip()
    print(f"  {len(irt_df)} IRT rows loaded.")

    raw_df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip")
    raw_df["id"] = raw_df["id"].apply(clean_id)
    raw_df["language"] = raw_df["language"].astype(str).str.strip()

    text_lookup = raw_df[["id", "language", "prompt"]].drop_duplicates(subset=["id", "language"])
    eng_lookup = text_lookup[text_lookup["language"] == "en"].set_index("id")["prompt"].to_dict()
    print(f"  {len(eng_lookup)} English reference prompts.")

    lookup_by_id_lang, lookup_by_id = build_multijail_category_lookup()

    text_lookup = text_lookup.copy()
    text_lookup["key"] = text_lookup["id"] + "_" + text_lookup["language"]
    text_map = text_lookup.set_index("key")["prompt"].to_dict()

    target_rows = irt_df[irt_df["language"] != "en"].copy()
    en_texts, target_texts, meta_rows = [], [], []

    for _, row in target_rows.iterrows():
        p_id = row["id"]
        lang = row["language"]
        tau = row.get("tau", row.get("Safety_Tax", row.get("diff", 0)))
        en_text = eng_lookup.get(p_id)
        target_text = text_map.get(f"{p_id}_{lang}")

        if not (en_text and target_text):
            continue

        cat_list = lookup_by_id_lang.get((p_id, lang), lookup_by_id.get(p_id, []))
        en_texts.append(str(en_text))
        target_texts.append(str(target_text))
        meta_rows.append({
            "language": lang, "id": p_id, "category": cat_list,
            "tau": tau, "en_text": en_text, "target_text": target_text,
        })

    print(f"  {len(en_texts)} valid text pairs built.")
    return en_texts, target_texts, meta_rows


# =========================================================================
#  STEP 2: Metric scoring functions
# =========================================================================

def score_labse(en_texts, target_texts):
    """LaBSE: cross-lingual sentence embedding cosine similarity."""
    print("\n--- [1/4] LaBSE ---")
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/LaBSE", device=dev)

    en_emb = model.encode(en_texts, batch_size=BATCH_SIZE_LABSE, convert_to_tensor=True, device=dev)
    tgt_emb = model.encode(target_texts, batch_size=BATCH_SIZE_LABSE, convert_to_tensor=True, device=dev)
    sims = torch.nn.functional.cosine_similarity(en_emb, tgt_emb, dim=1).cpu().numpy()

    print(f"  Done. Mean={sims.mean():.4f}, Std={sims.std():.4f}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sims


def score_comet(en_texts, target_texts):
    """COMET (wmt22-comet-da): reference-based, trained on human DA judgments."""
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
    """CometKiwi (wmt22-cometkiwi-da): reference-FREE. Only needs src + mt."""
    print("\n--- [3/4] CometKiwi (reference-free) ---")
    from comet import download_model, load_from_checkpoint

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

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
    """XCOMET-XL (3.5B): state-of-the-art, reference-based, error span detection."""
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
#  STEP 3: Run all metrics and save
# =========================================================================
def compute_all_scores():
    en_texts, target_texts, meta_rows = build_pairs()
    if not en_texts:
        print("No text pairs found.")
        return None

    out = pd.DataFrame(meta_rows)

    out["labse"] = score_labse(en_texts, target_texts)
    out["comet"] = score_comet(en_texts, target_texts)
    out["cometkiwi"] = score_cometkiwi(en_texts, target_texts)

    try:
        out["xcomet_xl"] = score_xcomet(en_texts, target_texts)
    except Exception as e:
        print(f"  XCOMET-XL failed (likely OOM): {e}")
        print(f"  Skipping XCOMET-XL. You may need a GPU or more RAM.")
        out["xcomet_xl"] = np.nan

    out.to_csv(OUTPUT_CSV_DATA, index=False)
    print(f"\nAll scores saved to {OUTPUT_CSV_DATA}")
    return out


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

def summarize_metric_across_languages(df, metric, min_n=10):
    """
    Summarize within-language associations for one metric.

    Returns:
      - mean_rho: unweighted mean of per-language Spearman rho
      - median_rho: median of per-language Spearman rho
      - weighted_mean_rho: mean weighted by language sample size
      - n_languages: number of languages included
      - pos_langs / neg_langs / null_langs: sign breakdown
      - sig_pos / sig_neg: significant sign breakdown
      - pooled_rho / pooled_p / pooled_n: pooled correlation across all rows
    """
    per_lang = []
    for lang in sorted(df["language"].dropna().unique()):
        sub = df[(df["language"] == lang) & df[metric].notna() & df["tau"].notna()].copy()
        if len(sub) <= min_n:
            continue
        rho, p = spearmanr(sub[metric], sub["tau"])
        per_lang.append({
            "language": lang,
            "rho": float(rho),
            "p": float(p),
            "n": int(len(sub)),
        })

    pooled = df[df[metric].notna() & df["tau"].notna()].copy()
    pooled_rho, pooled_p = spearmanr(pooled[metric], pooled["tau"]) if len(pooled) > min_n else (np.nan, np.nan)

    if not per_lang:
        return {
            "mean_rho": np.nan,
            "median_rho": np.nan,
            "weighted_mean_rho": np.nan,
            "n_languages": 0,
            "pos_langs": 0,
            "neg_langs": 0,
            "null_langs": 0,
            "sig_pos": 0,
            "sig_neg": 0,
            "pooled_rho": float(pooled_rho) if pd.notna(pooled_rho) else np.nan,
            "pooled_p": float(pooled_p) if pd.notna(pooled_p) else np.nan,
            "pooled_n": int(len(pooled)),
        }

    per_lang_df = pd.DataFrame(per_lang)
    weights = per_lang_df["n"].to_numpy(dtype=float)
    rhos = per_lang_df["rho"].to_numpy(dtype=float)

    return {
        "mean_rho": float(np.mean(rhos)),
        "median_rho": float(np.median(rhos)),
        "weighted_mean_rho": float(np.average(rhos, weights=weights)),
        "n_languages": int(len(per_lang_df)),
        "pos_langs": int((per_lang_df["rho"] > 0).sum()),
        "neg_langs": int((per_lang_df["rho"] < 0).sum()),
        "null_langs": int((per_lang_df["rho"] == 0).sum()),
        "sig_pos": int(((per_lang_df["rho"] > 0) & (per_lang_df["p"] < 0.05)).sum()),
        "sig_neg": int(((per_lang_df["rho"] < 0) & (per_lang_df["p"] < 0.05)).sum()),
        "pooled_rho": float(pooled_rho),
        "pooled_p": float(pooled_p),
        "pooled_n": int(len(pooled)),
    }
def compute_correlations_and_plot():
    if not os.path.exists(OUTPUT_CSV_DATA):
        print(f"{OUTPUT_CSV_DATA} not found. Run compute_all_scores() first.")
        return

    df = pd.read_csv(OUTPUT_CSV_DATA)
    available_metrics = [m for m in METRIC_COLS if m in df.columns and df[m].notna().any()]
    print(f"\nAvailable metrics: {available_metrics}")

    if not available_metrics:
        print("No available metric columns found in the CSV.")
        return

    print("\nRows per language:")
    print(df["language"].value_counts().sort_index())

    # --- Per-Language correlations ---
    lang_rows = []
    for metric in available_metrics:
        print(f"\n=== {metric} ===")

        pooled_sub = df[df[metric].notna() & df["tau"].notna()].copy()
        pooled_rho, pooled_p = spearmanr(pooled_sub[metric], pooled_sub["tau"])
        print("Pooled across all languages:")
        print(f"rho={pooled_rho:.4f}, p={pooled_p:.4g}, n={len(pooled_sub)}")

        print("Per language:")
        for lang in sorted(df["language"].dropna().unique()):
            sub = df[(df["language"] == lang) & df[metric].notna() & df["tau"].notna()].copy()
            if len(sub) > 10:
                r, p = spearmanr(sub[metric], sub["tau"])
                print(f"  {lang}: rho={r:.4f}, p={p:.4g}, n={len(sub)}")
                lang_rows.append({
                    "Metric": METRIC_LABELS.get(metric, metric),
                    "Metric_Key": metric,
                    "Language": lang,
                    "Spearman_Rho": float(r),
                    "P_Value": float(p),
                    "Count": int(len(sub)),
                })

        print("Language means:")
        print(df.groupby("language")[[metric, "tau"]].mean().sort_index())

    lang_df = pd.DataFrame(lang_rows)
    lang_df.to_csv(OUTPUT_CSV_LANG, index=False)
    print(f"\nLanguage correlations saved to {OUTPUT_CSV_LANG}")

    # --- Per-Category correlations (explode multi-label) ---
    df_copy = df.copy()
    df_copy["category_list"] = df_copy["category"].apply(parse_tags_cell)
    df_ex = df_copy.explode("category_list").rename(columns={"category_list": "Category"})
    df_ex = df_ex[df_ex["Category"].notna() & (df_ex["Category"] != "")]

    cat_rows = []
    for metric in available_metrics:
        for cat in sorted(df_ex["Category"].unique()):
            sub = df_ex[(df_ex["Category"] == cat) & df_ex[metric].notna() & df_ex["tau"].notna()].copy()
            if len(sub) > 10:
                r, p = spearmanr(sub[metric], sub["tau"])
                cat_rows.append({
                    "Metric": METRIC_LABELS.get(metric, metric),
                    "Metric_Key": metric,
                    "Category": cat,
                    "Spearman_Rho": float(r),
                    "P_Value": float(p),
                    "Count": int(len(sub)),
                })

    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(OUTPUT_CSV_CAT, index=False)
    print(f"Category correlations saved to {OUTPUT_CSV_CAT}")

    # --- Representative summary across languages ---
    summary_rows = []
    for metric in available_metrics:
        s = summarize_metric_across_languages(df, metric, min_n=10)
        summary_rows.append({
            "Metric": METRIC_LABELS.get(metric, metric),
            "Metric_Key": metric,
            "Mean_PerLanguage_Rho": s["mean_rho"],
            "Median_PerLanguage_Rho": s["median_rho"],
            "WeightedMean_PerLanguage_Rho": s["weighted_mean_rho"],
            "Num_Languages": s["n_languages"],
            "Positive_Languages": s["pos_langs"],
            "Negative_Languages": s["neg_langs"],
            "Zero_Languages": s["null_langs"],
            "Sig_Positive_Languages": s["sig_pos"],
            "Sig_Negative_Languages": s["sig_neg"],
            "Pooled_Rho_AllRows": s["pooled_rho"],
            "Pooled_P_AllRows": s["pooled_p"],
            "Pooled_N_AllRows": s["pooled_n"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Mean_PerLanguage_Rho")
    summary_df.to_csv(OUTPUT_CSV_SUMMARY, index=False)
    print(f"Representative summary saved to {OUTPUT_CSV_SUMMARY}")

    print("\nRepresentative summary (mean within-language rho):")
    print(summary_df[[
        "Metric",
        "Mean_PerLanguage_Rho",
        "Median_PerLanguage_Rho",
        "WeightedMean_PerLanguage_Rho",
        "Positive_Languages",
        "Negative_Languages",
        "Sig_Positive_Languages",
        "Sig_Negative_Languages",
        "Pooled_Rho_AllRows",
    ]].to_string(index=False))

    # --- PLOT: 2-panel (representative summary + per-language heatmap) ---
    print(f"\nGenerating plot to {OUTPUT_PLOT}...")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(FULL_WIDTH * 1.08, FULL_WIDTH * 0.45),
        gridspec_kw={"width_ratios": [1.35, 2]}
    )

    # Panel 1: mean per-language rho (more representative than pooled rho)
    ax = axes[0]
    plot_df = summary_df.copy().reset_index(drop=True)

    # Build cleaner y-axis labels
    plot_df["Metric_Label"] = [
        f'{row["Metric"]} ({int(row["Negative_Languages"])}/{int(row["Num_Languages"])} neg)'
        for _, row in plot_df.iterrows()
    ]

    colors = [C_RED if x > 0 else C_BLUE for x in plot_df["Mean_PerLanguage_Rho"]]

    ax.barh(
        plot_df["Metric_Label"],
        plot_df["Mean_PerLanguage_Rho"],
        color=colors,
        edgecolor="black",
        linewidth=0.3
    )

    # Overlay pooled rho as black dots
    ax.scatter(
        plot_df["Pooled_Rho_AllRows"],
        plot_df["Metric_Label"],
        color="black",
        s=32,
        zorder=3
    )

    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("Mean within-language ρ")
    ax.set_xlabel("Spearman ρ")

    # Give extra room on the right for pooled-rho dots
    vals = np.r_[plot_df["Mean_PerLanguage_Rho"].values,
                 plot_df["Pooled_Rho_AllRows"].values]
    xmin = np.nanmin(vals)
    xmax = np.nanmax(vals)
    xr = xmax - xmin if xmax > xmin else 0.1
    ax.set_xlim(xmin - 0.15 * xr, xmax + 0.25 * xr)

    # Panel 2: heatmap of per-language rho by metric
    ax = axes[1]
    pivot = lang_df.pivot_table(index="Language", columns="Metric", values="Spearman_Rho")

    col_order = [METRIC_LABELS.get(m, m) for m in available_metrics if METRIC_LABELS.get(m, m) in pivot.columns]
    pivot = pivot[col_order]

    present = [l for l in NON_EN_LANGS if l in pivot.index]
    remainder = [l for l in pivot.index if l not in present]
    pivot = pivot.reindex(present + remainder)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=CMAP_DIV,
        center=0,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": LABELS["rho"], "shrink": 0.8}
    )
    ax.set_title("Per-language ρ")
    ax.set_ylabel("")

    savefig(fig, OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_CSV_DATA):
        compute_all_scores()

    compute_correlations_and_plot()
