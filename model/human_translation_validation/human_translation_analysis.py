# -*- coding: utf-8 -*-
"""
Human Translation Quality vs Safety Outcomes
=============================================
Uses independent human grader ratings (3 graders × 315 prompts × 3 languages)
to validate H2 vs H3 with ground-truth translation quality.

Analyses:
  1. Human TQ vs τ (DIF/CSG) — does translation quality explain safety gaps?
  2. Human TQ vs safety_rate — does translation quality predict raw safety?
  3. H3 evidence: high-τ prompts with verified-good translations
     (translation is faithful BUT prompt still fails → conceptual mismatch)
  4. Comparison with automated metrics (LaBSE, COMET, etc.)
  5. Per-category and per-language breakdowns

Inputs:
  - human_translation_quality.csv  (id, tags, language, prompt_en,
                                     prompt_target, translation_quality)
  - bayesian_irt_results_binary.csv (IRT τ estimates)
  - Master_Passes0-9_Dataset.csv    (raw responses for safety_rate)

Outputs:
  - human_tq_vs_tau.csv             — merged data with correlations
  - human_tq_analysis_summary.csv   — per-language and global correlations
  - h3_evidence_prompts.csv         — high-τ + high-TQ prompts (H3 candidates)
  - human_tq_vs_tau_plot.png        — visualization
  - human_vs_automated_comparison.csv — human TQ vs LaBSE/COMET agreement
"""

import os
import sys
import ast
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from fig_style import (apply_style, savefig, make_fig, make_fig_grid,
                           C_RED, C_BLUE, C_PURPLE, C_GREY, CMAP_DIV,
                           LABELS, NON_EN_LANGS, FULL_WIDTH, DPI)
    _HAS_FS = True
except ImportError:
    _HAS_FS = False

from huggingface_hub import snapshot_download

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = snapshot_download(repo_id="MaxZ119/safetyirt",
                                repo_type="dataset", token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "processed_data",
                            "Master_Passes0-9_Dataset.csv")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_DIR_input = os.path.join("model/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Input files
HUMAN_TQ_FILE   = os.path.join(DATA_DIR, "human_translation_validation", "human_translation_quality.csv")
IRT_RESULTS     = os.path.join(RESULTS_DIR_input, "bayesian_irt_results_binary.csv")
AUTOMATED_FILE  = os.path.join(RESULTS_DIR, "multimetric_translation_v_DIF.csv")

# Output files
OUT_MERGED      = os.path.join(RESULTS_DIR, "human_tq_vs_tau.csv")
OUT_SUMMARY     = os.path.join(RESULTS_DIR, "human_tq_analysis_summary.csv")
OUT_H3          = os.path.join(RESULTS_DIR, "h3_evidence_prompts.csv")
OUT_PLOT        = os.path.join(RESULTS_DIR, "human_tq_vs_tau_plot")
OUT_COMPARISON  = os.path.join(RESULTS_DIR, "human_vs_automated_comparison.csv")

# Thresholds for H3 identification
H3_TOP_N = 100           # pull from top N highest |τ| prompts
H3_TQ_THRESHOLD = 5     # translation_quality ≥ this = "good translation"


SEED = 42
np.random.seed(SEED)


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def parse_tags(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s.startswith("["):
        try:
            out = ast.literal_eval(s)
            return [str(t).strip() for t in out if str(t).strip()]
        except Exception:
            pass
    return [s] if s else []


def bootstrap_spearman(x, y, n_boot=2000, seed=42):
    """Spearman ρ with bootstrap 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, np.nan, np.nan, np.nan
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        r, _ = spearmanr(x[idx], y[idx])
        rhos.append(r)
    lo, hi = np.quantile(rhos, [0.025, 0.975])
    return float(rho), float(p), float(lo), float(hi)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load and merge data
# ══════════════════════════════════════════════════════════════════════════════

def load_and_merge():
    """Merge human TQ ratings with IRT τ estimates and raw safety rates."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # ── Human translation quality ─────────────────────────────────
    if not os.path.exists(HUMAN_TQ_FILE):
        raise FileNotFoundError(
            f"Human TQ file not found: {HUMAN_TQ_FILE}\n"
            f"Expected columns: id, tags, language, prompt_en, "
            f"prompt_target, translation_quality")

    htq = pd.read_csv(HUMAN_TQ_FILE)
    htq["id"] = htq["id"].apply(clean_id)
    htq["language"] = htq["language"].astype(str).str.strip()
    htq["translation_quality"] = pd.to_numeric(
        htq["translation_quality"], errors="coerce")
    print(f"  Human TQ: {len(htq)} ratings, "
          f"languages={sorted(htq['language'].unique())}")
    print(f"  TQ distribution:\n{htq['translation_quality'].value_counts().sort_index().to_string()}")

    # ── IRT results (τ) ───────────────────────────────────────────
    if not os.path.exists(IRT_RESULTS):
        raise FileNotFoundError(f"IRT results not found: {IRT_RESULTS}")

    irt = pd.read_csv(IRT_RESULTS)
    # Normalise column names
    for old in ["prompt", "prompt_id", "item"]:
        if old in irt.columns:
            irt.rename(columns={old: "id"}, inplace=True)
    for old in ["Safety_Tax", "diff", "tau"]:
        if old in irt.columns:
            irt.rename(columns={old: "tau"}, inplace=True)

    irt["id"] = irt["id"].apply(clean_id)
    irt["language"] = irt["language"].astype(str).str.strip()
    print(f"  IRT results: {len(irt)} rows")

    # ── Raw safety rates ──────────────────────────────────────────
    raw = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip")
    raw["judge_score"] = pd.to_numeric(raw["judge_score"], errors="coerce")
    raw = raw[raw["judge_score"] > 0].dropna(subset=["judge_score"]).copy()
    raw["is_safe"] = (raw["judge_score"] >= 4).astype(int)
    raw["id"] = raw["id"].apply(clean_id)

    safety = raw.groupby(["id", "language"], as_index=False).agg(
        safety_rate=("is_safe", "mean"),
        n_responses=("is_safe", "size"),
    )
    print(f"  Safety rates: {len(safety)} (id, language) pairs")

    # ── Merge ─────────────────────────────────────────────────────
    merged = htq.merge(
        irt[["id", "language", "tau"]],
        on=["id", "language"], how="left"
    )
    merged = merged.merge(
        safety[["id", "language", "safety_rate", "n_responses"]],
        on=["id", "language"], how="left"
    )

    n_with_tau = merged["tau"].notna().sum()
    n_with_sr  = merged["safety_rate"].notna().sum()
    print(f"\n  Merged: {len(merged)} rows")
    print(f"  With τ:          {n_with_tau}")
    print(f"  With safety_rate: {n_with_sr}")

    # ── Optionally merge automated metrics ────────────────────────
    if os.path.exists(AUTOMATED_FILE):
        auto = pd.read_csv(AUTOMATED_FILE)
        auto["id"] = auto["id"].apply(clean_id)
        auto["language"] = auto["language"].astype(str).str.strip()
        auto_cols = ["id", "language"]
        for c in ["labse", "comet", "cometkiwi", "xcomet_xl"]:
            if c in auto.columns:
                auto_cols.append(c)
        if len(auto_cols) > 2:
            merged = merged.merge(
                auto[auto_cols], on=["id", "language"], how="left")
            print(f"  Automated metrics joined: "
                  f"{[c for c in auto_cols if c not in ['id','language']]}")

    merged.to_csv(OUT_MERGED, index=False)
    print(f"\n  Saved: {OUT_MERGED}")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Correlation analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_correlations(df):
    """Compute correlations: human TQ vs τ, human TQ vs safety_rate."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    rows = []

    # ── Global correlations ───────────────────────────────────────
    for target, target_label in [("tau", "τ (CSG)"),
                                  ("safety_rate", "Safety Rate")]:
        sub = df[df[target].notna() & df["translation_quality"].notna()]
        if len(sub) < 10:
            continue
        rho, p, lo, hi = bootstrap_spearman(
            sub["translation_quality"], sub[target])
        rows.append({
            "scope": "global", "language": "all", "target": target_label,
            "n": len(sub), "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
        })
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  Human TQ vs {target_label}: "
              f"ρ={rho:.3f} [{lo:.3f}, {hi:.3f}] p={p:.4f} {sig}")

    # ── Per-language correlations ─────────────────────────────────
    print()
    for lang in sorted(df["language"].unique()):
        for target, target_label in [("tau", "τ"),
                                      ("safety_rate", "safety_rate")]:
            sub = df[(df["language"] == lang)
                     & df[target].notna()
                     & df["translation_quality"].notna()]
            if len(sub) < 10:
                continue
            rho, p, lo, hi = bootstrap_spearman(
                sub["translation_quality"], sub[target])
            rows.append({
                "scope": "per_language", "language": lang,
                "target": target_label, "n": len(sub),
                "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
            })
            sig = "*" if p < 0.05 else ""
            print(f"  {lang:>3} | TQ vs {target_label:>12}: "
                  f"ρ={rho:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"n={len(sub):>3} {sig}")

    # ── Per-category correlations (explode tags) ──────────────────
    if "tags" in df.columns:
        df_c = df.copy()
        df_c["tag_list"] = df_c["tags"].apply(parse_tags)
        df_ex = df_c.explode("tag_list").rename(columns={"tag_list": "category"})
        df_ex = df_ex[df_ex["category"].notna() & (df_ex["category"] != "")]

        print(f"\n  Per-category (exploded to {len(df_ex)} rows):")
        for target, target_label in [("tau", "τ")]:
            for cat in sorted(df_ex["category"].unique()):
                sub = df_ex[(df_ex["category"] == cat)
                            & df_ex[target].notna()
                            & df_ex["translation_quality"].notna()]
                if len(sub) < 10:
                    continue
                rho, p, lo, hi = bootstrap_spearman(
                    sub["translation_quality"], sub[target])
                rows.append({
                    "scope": "per_category", "language": cat,
                    "target": target_label, "n": len(sub),
                    "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
                })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"\n  Saved: {OUT_SUMMARY}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: H3 evidence — high τ + good translation
# ══════════════════════════════════════════════════════════════════════════════
def find_h3_evidence(df):
    """
    Among the top H3_TOP_N highest-|τ| prompts, identify those with
    verified-good translations (TQ ≥ H3_TQ_THRESHOLD).

    These are H3 candidates: translation is faithful, but the safety
    concept doesn't transfer cross-lingually → conceptual mismatch.
    """
    print("\n" + "=" * 60)
    print(f"H3 EVIDENCE: Top {H3_TOP_N} |τ| prompts with TQ ≥ {H3_TQ_THRESHOLD}")
    print("=" * 60)

    sub = df[df["tau"].notna() & df["translation_quality"].notna()].copy()
    sub["abs_tau"] = sub["tau"].abs()
    sub = sub.sort_values("abs_tau", ascending=False)

    top_tau = sub.head(H3_TOP_N)
    h3 = top_tau[top_tau["translation_quality"] >= H3_TQ_THRESHOLD]

    print(f"  Total rated prompts with τ: {len(sub)}")
    print(f"  Top {H3_TOP_N} by |τ|:          τ range [{top_tau['abs_tau'].min():.2f}, {top_tau['abs_tau'].max():.2f}]")
    print(f"  Of those, TQ ≥ {H3_TQ_THRESHOLD}:       {len(h3)} → H3 candidates")

    if len(h3) > 0:
        print(f"\n  Top H3 candidates:")
        display_cols = ["id", "language", "translation_quality", "tau",
                        "tags", "prompt_en"]
        available = [c for c in display_cols if c in h3.columns]
        print(h3[available].head(15).to_string(index=False))

        if "tags" in h3.columns:
            all_tags = []
            for t in h3["tags"]:
                all_tags.extend(parse_tags(t))
            if all_tags:
                tag_counts = pd.Series(all_tags).value_counts()
                print(f"\n  H3 category distribution:")
                print(tag_counts.head(10).to_string())

    # Reverse: bad translation among top-τ (H2 evidence)
    h2 = top_tau[top_tau["translation_quality"] <= 2]
    print(f"\n  H2 candidates (top |τ| + TQ ≤ 2): {len(h2)}")

    # Neutral: good translation + low τ
    neutral = sub[(sub["translation_quality"] >= H3_TQ_THRESHOLD)
                  & (sub["abs_tau"] < 0.5)]
    print(f"  Neutral (TQ ≥ {H3_TQ_THRESHOLD} + |τ| < 0.5):    {len(neutral)}")

    h3.to_csv(OUT_H3, index=False)
    print(f"\n  Saved: {OUT_H3}")
    return h3
# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Compare human vs automated metrics
# ══════════════════════════════════════════════════════════════════════════════

def compare_human_vs_automated(df):
    """Correlate human TQ with automated metrics (LaBSE, COMET, etc.)."""
    print("\n" + "=" * 60)
    print("HUMAN vs AUTOMATED METRIC AGREEMENT")
    print("=" * 60)

    auto_cols = [c for c in ["labse", "comet", "cometkiwi", "xcomet_xl"]
                 if c in df.columns and df[c].notna().any()]

    if not auto_cols:
        print("  No automated metrics found in merged data — skipping")
        return None

    rows = []
    for metric in auto_cols:
        sub = df[df[metric].notna() & df["translation_quality"].notna()]
        if len(sub) < 10:
            continue
        rho, p, lo, hi = bootstrap_spearman(
            sub["translation_quality"], sub[metric])
        rows.append({
            "metric": metric, "scope": "global",
            "n": len(sub), "rho": rho, "p": p,
            "ci_lo": lo, "ci_hi": hi,
        })
        print(f"  Human TQ vs {metric:>12}: "
              f"ρ={rho:.3f} [{lo:.3f}, {hi:.3f}] n={len(sub)}")

        # Per language
        for lang in sorted(sub["language"].unique()):
            lsub = sub[sub["language"] == lang]
            if len(lsub) < 10:
                continue
            rho_l, p_l, lo_l, hi_l = bootstrap_spearman(
                lsub["translation_quality"], lsub[metric])
            rows.append({
                "metric": metric, "scope": lang,
                "n": len(lsub), "rho": rho_l, "p": p_l,
                "ci_lo": lo_l, "ci_hi": hi_l,
            })

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_COMPARISON, index=False)
    print(f"\n  Saved: {OUT_COMPARISON}")
    return comp


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_analysis(df, summary):
    """Generate figure: human TQ vs τ scatter + summary bars."""
    if _HAS_FS:
        apply_style()

    _cb = C_BLUE if _HAS_FS else "#0072B2"
    _cr = C_RED if _HAS_FS else "#D55E00"
    _cg = C_GREY if _HAS_FS else "#999999"

    langs = sorted(df["language"].unique())
    n_langs = len(langs)

    fig, axes = plt.subplots(1, n_langs + 1,
                              figsize=(FULL_WIDTH if _HAS_FS else 7,
                                       2.2),
                              gridspec_kw={"width_ratios":
                                           [1] * n_langs + [1.2]})

    # In plot_analysis, replace the H3 zone highlighting block with:

    # ── Scatter panels: TQ vs τ per language ──────────────────────
    sub = df[df["tau"].notna() & df["translation_quality"].notna()]

    # Get the |τ| cutoff from top-N for shading
    abs_taus = sub["tau"].abs().sort_values(ascending=False)
    tau_cutoff = abs_taus.iloc[min(H3_TOP_N - 1, len(abs_taus) - 1)] if len(abs_taus) > 0 else 1.0

    for i, lang in enumerate(langs):
        ax = axes[i]
        lsub = sub[sub["language"] == lang]

        tq_jitter = lsub["translation_quality"] + np.random.uniform(
            -0.15, 0.15, len(lsub))

        ax.scatter(tq_jitter, lsub["tau"], s=12, alpha=0.4,
                   color=_cb, edgecolors="none")
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

        # Shade H3 zone: top-N τ region + high TQ
        ax.axhspan(tau_cutoff, ax.get_ylim()[1] if len(lsub) else 5,
                    xmin=0.6, alpha=0.08, color=_cr)
        ax.axhspan(-5, -tau_cutoff,
                    xmin=0.6, alpha=0.08, color=_cr)

        rho, p = spearmanr(lsub["translation_quality"], lsub["tau"])
        sig = "*" if p < 0.05 else ""
        ax.set_title(f"{lang}\nρ={rho:.2f}{sig}")
        ax.set_xlabel("Human TQ")
        if i == 0:
            ax.set_ylabel("τ (CSG)")
        ax.set_xticks([1, 2, 3, 4, 5])

    # ── Summary bar: global correlations ──────────────────────────
    ax = axes[-1]
    glob = summary[(summary["scope"] == "global")].copy()
    if len(glob) > 0:
        colors = [_cb if "τ" in t else _cr for t in glob["target"]]
        bars = ax.barh(glob["target"], glob["rho"], color=colors,
                       edgecolor="black", linewidth=0.3)
        for j, row in glob.iterrows():
            ax.plot([row["ci_lo"], row["ci_hi"]],
                    [glob["target"].tolist().index(row["target"])] * 2,
                    color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title("Global ρ")
        ax.set_xlabel("Spearman ρ")

    plt.tight_layout()

    for ext in [".png", ".pdf"]:
        fig.savefig(OUT_PLOT + ext, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUT_PLOT}.png/.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("HUMAN TRANSLATION QUALITY ANALYSIS")
    print("  H2: Bad translation → high τ?")
    print("  H3: Good translation + high τ → conceptual mismatch?")
    print("=" * 60)

    df = load_and_merge()
    summary = run_correlations(df)
    h3 = find_h3_evidence(df)
    compare_human_vs_automated(df)
    plot_analysis(df, summary)

    # ── Key findings ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    sub = df[df["tau"].notna() & df["translation_quality"].notna()]
    if len(sub) > 0:
        rho, p = spearmanr(sub["translation_quality"], sub["tau"])
        print(f"  Human TQ vs τ:        ρ = {rho:+.3f} (p = {p:.4f})")
        print(f"  H3 candidates:        {len(h3)} prompts")
        print(f"  (good translation + high safety gap)")

        # Mean τ by TQ bin
        sub_c = sub.copy()
        sub_c["tq_bin"] = pd.cut(sub_c["translation_quality"],
                                  bins=[0, 2, 3, 5],
                                  labels=["Low(1-2)", "Mid(3)", "High(4-5)"])
        print(f"\n  Mean |τ| by human TQ bin:")
        for bin_label, grp in sub_c.groupby("tq_bin", observed=True):
            print(f"    {bin_label}: mean|τ|={grp['tau'].abs().mean():.3f} "
                  f"(n={len(grp)})")

    print("\nDone.")


if __name__ == "__main__":
    main()