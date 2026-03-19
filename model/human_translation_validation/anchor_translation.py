# -*- coding: utf-8 -*-
"""
Anchor Translation Quality Check
=================================
Validates that semantic-equivalence anchors actually have high human-rated
translation quality. If anchors have poor translations, the identification
strategy is compromised.

Also compares anchor vs non-anchor translation quality distributions
to verify that anchors are indeed better-translated.

Inputs:
  - human_translation_quality.csv  (human TQ ratings)
  - anchors.csv                    (current anchor set)
  - bayesian_irt_results_binary.csv (IRT results for τ)

Outputs:
  - anchor_tq_report.csv     — per-anchor TQ scores
  - anchor_tq_summary.csv    — anchor vs non-anchor comparison
  - anchor_tq_plot.png       — visualization
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from fig_style import (apply_style, savefig, C_RED, C_BLUE, C_GREY,
                           FULL_WIDTH, DPI)
    _HAS_FS = True
except ImportError:
    _HAS_FS = False

from huggingface_hub import snapshot_download

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = snapshot_download(repo_id="MaxZ119/safetyirt",
                                repo_type="dataset", token=False)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_human_TQ")
os.makedirs(RESULTS_DIR, exist_ok=True)

HUMAN_TQ_FILE   = os.path.join(DATA_DIR, "human_translation_validation", "human_translation_quality.csv")
ANCHOR_FILE   = os.path.join(DATA_DIR, "anchors", "anchors.csv")
IRT_RESULTS   = os.path.join(RESULTS_DIR, "bayesian_irt_results_binary.csv")

OUT_REPORT  = os.path.join(RESULTS_DIR, "anchor_tq_report.csv")
OUT_SUMMARY = os.path.join(RESULTS_DIR, "anchor_tq_summary.csv")
OUT_PLOT    = os.path.join(RESULTS_DIR, "anchor_tq_plot")


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def main():
    if _HAS_FS:
        apply_style()

    print("=" * 60)
    print("ANCHOR TRANSLATION QUALITY CHECK")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    if not os.path.exists(HUMAN_TQ_FILE):
        raise FileNotFoundError(f"Human TQ not found: {HUMAN_TQ_FILE}")

    htq = pd.read_csv(HUMAN_TQ_FILE)
    htq["id"] = htq["id"].apply(clean_id)
    htq["language"] = htq["language"].astype(str).str.strip()
    htq["translation_quality"] = pd.to_numeric(
        htq["translation_quality"], errors="coerce")
    print(f"  Human TQ: {len(htq)} ratings")

    anchors_df = pd.read_csv(ANCHOR_FILE)
    anchors_df["id"] = anchors_df["id"].apply(clean_id)
    anchor_ids = set(anchors_df["id"].unique())
    print(f"  Anchors: {len(anchor_ids)} items")

    # ── Flag anchors in TQ data ───────────────────────────────────
    htq["is_anchor"] = htq["id"].isin(anchor_ids)

    # How many anchors have human TQ ratings?
    rated_anchors = htq[htq["is_anchor"]]["id"].unique()
    unrated = anchor_ids - set(rated_anchors)
    print(f"  Anchors with human TQ: {len(rated_anchors)}/{len(anchor_ids)}")
    if unrated:
        print(f"  Unrated anchors (not in human TQ languages): "
              f"{len(unrated)} items")

    # ── Anchor vs Non-anchor TQ comparison ────────────────────────
    print(f"\n{'─' * 50}")
    print("ANCHOR vs NON-ANCHOR TRANSLATION QUALITY")
    print(f"{'─' * 50}")

    summary_rows = []

    # Global
    a = htq[htq["is_anchor"]]["translation_quality"].dropna()
    na = htq[~htq["is_anchor"]]["translation_quality"].dropna()

    if len(a) > 0 and len(na) > 0:
        u_stat, u_p = mannwhitneyu(a, na, alternative="two-sided")
        print(f"\n  Global:")
        print(f"    Anchors:     mean={a.mean():.2f}, "
              f"median={a.median():.1f}, n={len(a)}")
        print(f"    Non-anchors: mean={na.mean():.2f}, "
              f"median={na.median():.1f}, n={len(na)}")
        print(f"    Mann-Whitney U: U={u_stat:.0f}, p={u_p:.4f}")

        summary_rows.append({
            "scope": "global",
            "anchor_mean": a.mean(), "anchor_median": a.median(),
            "anchor_n": len(a),
            "nonanchor_mean": na.mean(), "nonanchor_median": na.median(),
            "nonanchor_n": len(na),
            "U_stat": u_stat, "p_value": u_p,
        })

    # Per language
    for lang in sorted(htq["language"].unique()):
        lsub = htq[htq["language"] == lang]
        la = lsub[lsub["is_anchor"]]["translation_quality"].dropna()
        lna = lsub[~lsub["is_anchor"]]["translation_quality"].dropna()

        if len(la) >= 5 and len(lna) >= 5:
            u, p = mannwhitneyu(la, lna, alternative="two-sided")
            sig = "*" if p < 0.05 else ""
            print(f"\n  {lang}: anchors={la.mean():.2f} (n={len(la)}) vs "
                  f"non-anchors={lna.mean():.2f} (n={len(lna)}) "
                  f"p={p:.4f} {sig}")

            summary_rows.append({
                "scope": lang,
                "anchor_mean": la.mean(), "anchor_median": la.median(),
                "anchor_n": len(la),
                "nonanchor_mean": lna.mean(), "nonanchor_median": lna.median(),
                "nonanchor_n": len(lna),
                "U_stat": u, "p_value": p,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"\n  Saved: {OUT_SUMMARY}")

    # ── Per-anchor detail report ──────────────────────────────────
    anchor_detail = htq[htq["is_anchor"]].copy()
    if "tags" in anchor_detail.columns:
        cols = ["id", "language", "translation_quality", "tags",
                "prompt_en", "prompt_target"]
    else:
        cols = ["id", "language", "translation_quality",
                "prompt_en", "prompt_target"]
    available = [c for c in cols if c in anchor_detail.columns]
    anchor_detail = anchor_detail[available].sort_values(
        ["id", "language"])
    anchor_detail.to_csv(OUT_REPORT, index=False)
    print(f"  Saved: {OUT_REPORT}")

    # Flag problematic anchors (low TQ)
    low_tq_anchors = anchor_detail[
        anchor_detail["translation_quality"] <= 2
    ]
    if len(low_tq_anchors) > 0:
        print(f"\n  ⚠ WARNING: {len(low_tq_anchors)} anchor×language pairs "
              f"with TQ ≤ 2:")
        print(low_tq_anchors[["id", "language",
                               "translation_quality"]].to_string(index=False))
    else:
        print(f"\n  All rated anchors have TQ ≥ 3")

    # ── TQ distribution of anchors ────────────────────────────────
    if len(a) > 0:
        print(f"\n  Anchor TQ distribution:")
        print(a.value_counts().sort_index().to_string())

    # ── Plot ──────────────────────────────────────────────────────
    _cb = C_BLUE if _HAS_FS else "#0072B2"
    _cr = C_RED if _HAS_FS else "#D55E00"

    fig, axes = plt.subplots(1, 2,
                              figsize=(FULL_WIDTH if _HAS_FS else 5.5, 2.5))

    # Panel 1: Side-by-side histograms
    ax = axes[0]
    bins = np.arange(0.5, 6.5, 1)
    ax.hist(a, bins=bins, alpha=0.7, color=_cb, label="Anchor",
            density=True, edgecolor="black", linewidth=0.3)
    ax.hist(na, bins=bins, alpha=0.5, color=_cr, label="Non-anchor",
            density=True, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Human Translation Quality")
    ax.set_ylabel("Density")
    ax.set_title("TQ Distribution: Anchor vs Non-anchor")
    ax.legend()
    ax.set_xticks([1, 2, 3, 4, 5])

    # Panel 2: Per-language comparison
    ax = axes[1]
    langs = sorted(htq["language"].unique())
    x = np.arange(len(langs))
    w = 0.35

    a_means = [htq[(htq["language"] == l) & htq["is_anchor"]]
               ["translation_quality"].mean() for l in langs]
    na_means = [htq[(htq["language"] == l) & ~htq["is_anchor"]]
                ["translation_quality"].mean() for l in langs]

    ax.bar(x - w/2, a_means, w, label="Anchor", color=_cb,
           edgecolor="black", linewidth=0.3)
    ax.bar(x + w/2, na_means, w, label="Non-anchor", color=_cr,
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Mean TQ")
    ax.set_title("Mean TQ by Language")
    ax.legend()

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        fig.savefig(OUT_PLOT + ext, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_PLOT}.png/.pdf")

    print("\nDone.")


if __name__ == "__main__":
    main()