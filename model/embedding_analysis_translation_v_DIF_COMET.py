# -*- coding: utf-8 -*-
"""
Translation Quality Analysis: COMET Score vs Safety Cost (Tau).
Uses Unbabel/wmt22-comet-da (trained on human translation quality judgments)
instead of LaBSE cosine similarity.

1. Computes COMET translation quality scores.
2. Calculates Spearman's Rho for Categories and Languages.
3. Generates the 'Semantic Hazard vs Filter Benefit' Bar Charts.

Output schema:
language,id,category,tau,en_text,target_text,comet_score

Where `category` is a multi-label list (stringified in CSV), sourced from multijail.csv
and later exploded for per-tag analysis.
"""
import os
import sys
import ast

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

IRT_RESULTS_FILE = os.path.join(RESULTS_DIR, "bayesian_irt_results_binary.csv")
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "Analysis_COMET_Fidelity_vs_SafetyTax.png")
OUTPUT_CSV_DATA = os.path.join(RESULTS_DIR, "comet_analysis_translation_v_DIF.csv")
OUTPUT_CSV_LANG = os.path.join(RESULTS_DIR, "comet_analysis_translation_v_DIF_Lang.csv")
OUTPUT_CSV_CAT = os.path.join(RESULTS_DIR, "comet_analysis_translation_v_DIF_Cat.csv")

# COMET model identifier (reference-based, trained on human DA judgments, scores 0-1)
COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"

# Multijail tags source
MULTIJAIL_FILE = os.path.join(DATA_DIR, "multijail.csv")

# Batch size for COMET inference (adjust down if OOM)
COMET_BATCH_SIZE = 64


# --- UTILS ---
def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def parse_tags_cell(x):
    """
    Convert a tag cell into a list[str].
    Handles:
      - "['Fraud & deception', 'Theft']"
      - already-a-list
      - single tag strings
      - NaN
    """
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


def detect_category_column(df: pd.DataFrame) -> str:
    """
    Find the column in multijail.csv that contains tags/categories.
    """
    candidates = ["tags", "tag", "category", "categories", "label", "labels"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower()
        if "tag" in lc or "cat" in lc:
            return c
    raise KeyError(f"Could not find a tags/category column in multijail.csv. Columns: {list(df.columns)}")


def build_multijail_category_lookup():
    """
    Returns two dicts:
      - lookup_by_id_lang: (id, language) -> list[str]
      - lookup_by_id: id -> list[str]  (fallback)
    """
    if not os.path.exists(MULTIJAIL_FILE):
        raise FileNotFoundError(
            f"multijail.csv not found at {MULTIJAIL_FILE}.\n"
            f"Put multijail.csv next to this script or update MULTIJAIL_FILE."
        )

    mj = pd.read_csv(MULTIJAIL_FILE)
    if "id" not in mj.columns or "language" not in mj.columns:
        raise KeyError(f"multijail.csv must contain 'id' and 'language' columns. Found: {list(mj.columns)}")

    cat_col = detect_category_column(mj)

    mj = mj.copy()
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

    print(f"Loaded multijail.csv tags from column '{cat_col}'.")
    return lookup_by_id_lang, lookup_by_id


def compute_comet_scores():
    """
    Step 1: Compute COMET translation quality scores.
    
    COMET (wmt22-comet-da) is a reference-based metric trained on human
    Direct Assessment judgments from WMT17-20. It takes (src, mt, ref) triples
    and outputs a quality score in [0, 1].
    
    For our use case:
      - src = English prompt (the original source text)
      - mt  = Target language prompt (the "translation")
      - ref = English prompt (same as src; English is the reference)
    
    This is the standard MT evaluation approach when evaluating translations
    back against their source.
    
    Saves OUTPUT_CSV_DATA with schema:
      language,id,category,tau,en_text,target_text,comet_score
    """
    print("--- Starting COMET Translation Quality Analysis ---")

    if not os.path.exists(IRT_RESULTS_FILE):
        raise FileNotFoundError(f"Results not found at {IRT_RESULTS_FILE}.")

    irt_df = pd.read_csv(IRT_RESULTS_FILE)

    # Robust ID Column Detection
    if "prompt" in irt_df.columns:
        irt_df.rename(columns={"prompt": "id"}, inplace=True)
    elif "prompt_id" in irt_df.columns:
        irt_df.rename(columns={"prompt_id": "id"}, inplace=True)
    elif "item" in irt_df.columns:
        irt_df.rename(columns={"item": "id"}, inplace=True)

    if "id" not in irt_df.columns:
        irt_df.rename(columns={irt_df.columns[0]: "id"}, inplace=True)

    if "language" not in irt_df.columns:
        raise KeyError(f"IRT results file must include 'language'. Found: {list(irt_df.columns)}")

    irt_df["id"] = irt_df["id"].apply(clean_id)
    irt_df["language"] = irt_df["language"].astype(str).str.strip()
    print(f"Loaded {len(irt_df)} IRT parameter rows.")

    # Load Raw Text
    print(f"Loading raw text from {INPUT_FILE}...")
    raw_df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip")
    raw_df["id"] = raw_df["id"].apply(clean_id)
    raw_df["language"] = raw_df["language"].astype(str).str.strip()

    # Create Lookups
    text_lookup = raw_df[["id", "language", "prompt"]].drop_duplicates(subset=["id", "language"])

    eng_df = text_lookup[text_lookup["language"] == "en"].set_index("id")
    eng_lookup = eng_df["prompt"].to_dict()
    print(f"Found {len(eng_lookup)} unique English reference prompts.")

    # Load multijail categories
    lookup_by_id_lang, lookup_by_id = build_multijail_category_lookup()

    # Build text map for target texts
    text_lookup = text_lookup.copy()
    text_lookup["key"] = text_lookup["id"] + "_" + text_lookup["language"]
    text_map = text_lookup.set_index("key")["prompt"].to_dict()

    # Build pairs (non-English rows in IRT)
    target_rows = irt_df[irt_df["language"] != "en"].copy()

    comet_data = []  # List of dicts for COMET: {"src": ..., "mt": ..., "ref": ...}
    meta_rows = []

    print("Building text pairs...")
    for _, row in target_rows.iterrows():
        p_id = row["id"]
        lang = row["language"]

        tau = row.get("tau", row.get("Safety_Tax", row.get("diff", 0)))

        en_text = eng_lookup.get(p_id)
        target_text = text_map.get(f"{p_id}_{lang}")

        if not (en_text and target_text):
            continue

        cat_list = lookup_by_id_lang.get((p_id, lang), lookup_by_id.get(p_id, []))

        # COMET expects: src (source), mt (machine/human translation), ref (reference)
        # English is both the source and reference; target language is the translation
        comet_data.append({
            "src": str(en_text),
            "mt": str(target_text),
            "ref": str(en_text),
        })

        meta_rows.append({
            "language": lang,
            "id": p_id,
            "category": cat_list,
            "tau": tau,
            "en_text": en_text,
            "target_text": target_text,
        })

    if not comet_data:
        print("No matching text pairs found.")
        return None

    # Load COMET model
    print(f"Downloading and loading COMET model: {COMET_MODEL_NAME}")
    model_path = download_model(COMET_MODEL_NAME)
    model = load_from_checkpoint(model_path)

    # Score all pairs
    print(f"Scoring {len(comet_data)} translation pairs with COMET...")
    output = model.predict(comet_data, batch_size=COMET_BATCH_SIZE, num_workers=1, gpus=1 if __import__('torch').cuda.is_available() else 0)

    # output.scores is a list of per-segment scores
    scores = output.scores

    out = pd.DataFrame(meta_rows)
    out["comet_score"] = scores

    # Enforce column order
    out = out[["language", "id", "category", "tau", "en_text", "target_text", "comet_score"]]
    out.to_csv(OUTPUT_CSV_DATA, index=False)
    print(f"✅ Data processed and saved to {OUTPUT_CSV_DATA}")

    return out


def plot_and_report():
    """
    Step 2: Spearman stats + bar charts.
    Uses comet_score instead of LaBSE similarity.
    """
    if not os.path.exists(OUTPUT_CSV_DATA):
        print(f"❌ Output file {OUTPUT_CSV_DATA} not found.")
        return

    print(f"Loading analysis data from {OUTPUT_CSV_DATA}...")
    df = pd.read_csv(OUTPUT_CSV_DATA)

    required = {"language", "id", "category", "tau", "comet_score"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Analysis CSV missing required columns: {missing}. Found: {list(df.columns)}")

    print("\nCalculating correlations...")

    # --- Language stats ---
    lang_stats = []
    for lang in df["language"].unique():
        sub = df[df["language"] == lang]
        if len(sub) > 10:
            r, p = spearmanr(sub["comet_score"], sub["tau"])
            lang_stats.append({"Language": lang, "Spearman_Rho": r, "P_Value": p, "Count": len(sub)})
    lang_df = pd.DataFrame(lang_stats).sort_values("Spearman_Rho")
    lang_df.to_csv(OUTPUT_CSV_LANG, index=False)

    # --- Category stats (explode multi-label) ---
    df = df.copy()
    df["category_list"] = df["category"].apply(parse_tags_cell)
    df_ex = df.explode("category_list").rename(columns={"category_list": "Category"})
    df_ex = df_ex[df_ex["Category"].notna() & (df_ex["Category"] != "")]

    cat_stats = []
    for cat in df_ex["Category"].unique():
        sub = df_ex[df_ex["Category"] == cat]
        if len(sub) > 10:
            r, p = spearmanr(sub["comet_score"], sub["tau"])
            cat_stats.append({"Category": cat, "Spearman_Rho": r, "P_Value": p, "Count": len(sub)})
    cat_df = pd.DataFrame(cat_stats).sort_values("Spearman_Rho")
    cat_df.to_csv(OUTPUT_CSV_CAT, index=False)

    # --- PLOT ---
    print(f"Generating Bar Charts to {OUTPUT_PLOT}...")

    df_cat = cat_df.copy()
    df_lang = lang_df.copy()

    df_cat["Significant"] = df_cat["P_Value"] < 0.05
    df_lang["Significant"] = df_lang["P_Value"] < 0.05

    fig, axes = plt.subplots(1, 2, figsize=(20, 12), gridspec_kw={"width_ratios": [2, 1]})

    # Categories
    colors_cat = ["#d62728" if x > 0 else "#1f77b4" for x in df_cat["Spearman_Rho"]]
    alphas_cat = [1.0 if sig else 0.3 for sig in df_cat["Significant"]]
    bars = axes[0].barh(df_cat["Category"], df_cat["Spearman_Rho"], color=colors_cat)
    for bar, a in zip(bars, alphas_cat):
        bar.set_alpha(a)

    axes[0].set_title("Correlation: COMET Translation Quality vs. Safety Tax (By Category)",
                       fontsize=16, fontweight="bold")
    axes[0].set_xlabel(
        "Spearman's Rho\n(Left/Blue: Better Translation → Safer | Right/Red: Better Translation → More Dangerous)",
        fontsize=12,
    )
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    for i, (rho, p) in enumerate(zip(df_cat["Spearman_Rho"], df_cat["P_Value"])):
        if p < 0.05:
            offset = 0.01 if rho > 0 else -0.05
            axes[0].text(rho + offset, i, "★", va="center", fontsize=14, fontweight="bold", color="black")

    # Languages
    colors_lang = ["#d62728" if x > 0 else "#1f77b4" for x in df_lang["Spearman_Rho"]]
    alphas_lang = [1.0 if sig else 0.3 for sig in df_lang["Significant"]]
    bars2 = axes[1].barh(df_lang["Language"], df_lang["Spearman_Rho"], color=colors_lang)
    for bar, a in zip(bars2, alphas_lang):
        bar.set_alpha(a)

    axes[1].set_title("Correlation: COMET Quality vs. Safety Tax (By Language)",
                       fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Spearman's Rho", fontsize=12)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    for i, (rho, p) in enumerate(zip(df_lang["Spearman_Rho"], df_lang["P_Value"])):
        if p < 0.05:
            offset = 0.01 if rho > 0 else -0.05
            axes[1].text(rho + offset, i, "★", va="center", fontsize=14, fontweight="bold", color="black")

    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=4, label="Filter Benefit (High Quality → Safer)"),
        Line2D([0], [0], color="#d62728", lw=4, label="Semantic Hazard (High Quality → More Dangerous)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=15,
               label="Statistically Significant (p < 0.05)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.08), ncol=3, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Graph Saved: {OUTPUT_PLOT}")


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_CSV_DATA):
        compute_comet_scores()

    plot_and_report()