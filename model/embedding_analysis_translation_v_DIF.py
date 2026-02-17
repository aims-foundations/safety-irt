# -*- coding: utf-8 -*-
"""
Embedding Analysis: Translation Similarity vs Safety Cost (Tau).
1. Computes LaBSE embeddings and cosine similarity.
2. Calculates Spearman's Rho for Categories and Languages.
3. Generates the 'Semantic Hazard vs Filter Benefit' Bar Charts.

Output schema:
language,id,category,tau,en_text,target_text,similarity

Where `category` is a multi-label list (stringified in CSV), sourced from multijail.csv
and later exploded for per-tag analysis.
"""
import os
import sys
import ast

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from huggingface_hub import snapshot_download


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
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "Analysis_Fidelity_vs_SafetyTax.png")
OUTPUT_CSV_DATA = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_DIF.csv")
OUTPUT_CSV_LANG = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_DIF_Lang.csv")
OUTPUT_CSV_CAT = os.path.join(RESULTS_DIR, "embedding_analysis_translation_v_DIF_Cat.csv")

# NEW: your multijail tags source
# Put multijail.csv next to this script OR change this path
MULTIJAIL_FILE = os.path.join(DATA_DIR, "multijail.csv")


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
    Tries common names.
    """
    candidates = ["tags", "tag", "category", "categories", "label", "labels"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try any column containing 'tag' or 'cat'
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

    # (id, language) -> tags
    lookup_by_id_lang = {}
    for _, r in mj.iterrows():
        key = (r["id"], r["language"])
        # if duplicates exist, union them
        if key not in lookup_by_id_lang:
            lookup_by_id_lang[key] = list(r["category_list"])
        else:
            lookup_by_id_lang[key] = sorted(set(lookup_by_id_lang[key]).union(r["category_list"]))

    # id -> tags fallback (union across langs)
    lookup_by_id = {}
    for _, r in mj.iterrows():
        i = r["id"]
        if i not in lookup_by_id:
            lookup_by_id[i] = list(r["category_list"])
        else:
            lookup_by_id[i] = sorted(set(lookup_by_id[i]).union(r["category_list"]))

    print(f"Loaded multijail.csv tags from column '{cat_col}'.")
    return lookup_by_id_lang, lookup_by_id


def compute_embeddings():
    """
    Step 1: Compute Embeddings & Similarity.
    Saves OUTPUT_CSV_DATA with schema:
      language,id,category,tau,en_text,target_text,similarity
    """
    print("--- Starting Embedding Analysis ---")

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

    # Ensure language exists in IRT results
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

    # English reference
    eng_df = text_lookup[text_lookup["language"] == "en"].set_index("id")
    eng_lookup = eng_df["prompt"].to_dict()
    print(f"Found {len(eng_lookup)} unique English reference prompts.")

    # Load multijail categories
    lookup_by_id_lang, lookup_by_id = build_multijail_category_lookup()

    # Load Model
    model_name = "sentence-transformers/LaBSE"
    print(f"Loading Model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # Build text map for target texts
    text_lookup = text_lookup.copy()
    text_lookup["key"] = text_lookup["id"] + "_" + text_lookup["language"]
    text_map = text_lookup.set_index("key")["prompt"].to_dict()

    # Build pairs (non-English rows in IRT)
    target_rows = irt_df[irt_df["language"] != "en"].copy()

    en_texts, target_texts, meta_rows = [], [], []

    print("Building text pairs...")
    for _, row in target_rows.iterrows():
        p_id = row["id"]
        lang = row["language"]

        # Prioritize 'tau', then 'Safety_Tax', then 'diff'
        tau = row.get("tau", row.get("Safety_Tax", row.get("diff", 0)))

        en_text = eng_lookup.get(p_id)
        target_text = text_map.get(f"{p_id}_{lang}")

        if not (en_text and target_text):
            continue

        # category from multijail.csv (try id+lang; fallback to id; else empty)
        cat_list = lookup_by_id_lang.get((p_id, lang), lookup_by_id.get(p_id, []))

        en_texts.append(en_text)
        target_texts.append(target_text)
        meta_rows.append(
            {
                "language": lang,
                "id": p_id,
                "category": cat_list,  # list -> will be stringified in CSV
                "tau": tau,
                "en_text": en_text,
                "target_text": target_text,
            }
        )

    if not en_texts:
        print("No matching text pairs found.")
        return None

    # Encode & cosine similarity
    BATCH_SIZE = 256
    print(f"Encoding {len(en_texts)} pairs...")
    en_emb = model.encode(en_texts, batch_size=BATCH_SIZE, convert_to_tensor=True, device=device)
    tgt_emb = model.encode(target_texts, batch_size=BATCH_SIZE, convert_to_tensor=True, device=device)
    sims = torch.nn.functional.cosine_similarity(en_emb, tgt_emb, dim=1).cpu().numpy()

    out = pd.DataFrame(meta_rows)
    out["similarity"] = sims

    # enforce EXACT column order
    out = out[["language", "id", "category", "tau", "en_text", "target_text", "similarity"]]
    out.to_csv(OUTPUT_CSV_DATA, index=False)
    print(f"✅ Data processed and saved to {OUTPUT_CSV_DATA}")

    return out


def plot_and_report():
    """
    Step 2: Spearman stats + bar charts.
    Category handling:
      - `category` is a list (stored as string in CSV)
      - we parse it back -> explode -> per-tag Spearman
    """
    if not os.path.exists(OUTPUT_CSV_DATA):
        print(f"❌ Output file {OUTPUT_CSV_DATA} not found.")
        return

    print(f"Loading analysis data from {OUTPUT_CSV_DATA}...")
    df = pd.read_csv(OUTPUT_CSV_DATA)

    required = {"language", "id", "category", "tau", "similarity"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Analysis CSV missing required columns: {missing}. Found: {list(df.columns)}")

    print("\nCalculating correlations...")

    # --- Language stats (no explode) ---
    lang_stats = []
    for lang in df["language"].unique():
        sub = df[df["language"] == lang]
        if len(sub) > 10:
            r, p = spearmanr(sub["similarity"], sub["tau"])
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
            r, p = spearmanr(sub["similarity"], sub["tau"])
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

    axes[0].set_title("Correlation: Translation Fidelity vs. Safety Tax (By Category)", fontsize=16, fontweight="bold")
    axes[0].set_xlabel(
        "Spearman's Rho\n(Left/Blue: Better Translation = Safer | Right/Red: Better Translation = More Dangerous)",
        fontsize=12,
    )
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    for i, (rho, p) in enumerate(zip(df_cat["Spearman_Rho"], df_cat["P_Value"])):
        if p < 0.05:
            offset = 0.01 if rho > 0 else -0.05
            axes[0].text(rho + offset, i, "*", va="center", fontsize=16, fontweight="bold", color="black")

    # Languages
    colors_lang = ["#d62728" if x > 0 else "#1f77b4" for x in df_lang["Spearman_Rho"]]
    alphas_lang = [1.0 if sig else 0.3 for sig in df_lang["Significant"]]
    bars2 = axes[1].barh(df_lang["Language"], df_lang["Spearman_Rho"], color=colors_lang)
    for bar, a in zip(bars2, alphas_lang):
        bar.set_alpha(a)

    axes[1].set_title("Correlation: Fidelity vs. Safety Tax (By Language)", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Spearman's Rho", fontsize=12)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    for i, (rho, p) in enumerate(zip(df_lang["Spearman_Rho"], df_lang["P_Value"])):
        if p < 0.05:
            offset = 0.01 if rho > 0 else -0.05
            axes[1].text(rho + offset, i, "*", va="center", fontsize=16, fontweight="bold", color="black")

    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=4, label="Filter Benefit (High Fidelity -> Safer)"),
        Line2D([0], [0], color="#d62728", lw=4, label="Semantic Hazard (High Fidelity -> More Dangerous)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=15,
               label="Statistically Significant (p < 0.05)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.08), ncol=3, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"✅ Graph Saved: {OUTPUT_PLOT}")


if __name__ == "__main__":
    # If you changed schema/inputs, delete the old analysis CSV so it recomputes.
    # rm -f results/embedding_analysis_translation_v_DIF.csv

    if not os.path.exists(OUTPUT_CSV_DATA):
        compute_embeddings()

    plot_and_report()
