# -*- coding: utf-8 -*-
"""
Logistic-Regression DIF Validation of Lord's-χ² Selected Anchors
=================================================================
Companion to mh_anchor_validation.py. Where MH primarily catches UNIFORM DIF,
logistic-regression (LR) DIF (Swaminathan & Rogers 1990; Zumbo 1999) catches
both uniform and NON-UNIFORM DIF. Non-uniform DIF parallels the 2PL
discrimination term (α_i) in our IRT model — so this is the closer
non-IRT analogue to what our model is actually estimating.

Like MH, LR-DIF conditions on the observed total score and therefore does
NOT require an IRT-calibrated scale.

Method (Zumbo 1999):
  For each (item × non-English language) pair, fit nested logistic regressions
  on the combined English + target-language responses for that item:
    Model 0: safe ~ M                   (no DIF)
    Model 1: safe ~ M + G               (uniform DIF — main effect of group)
    Model 2: safe ~ M + G + M:G         (uniform + non-uniform DIF)
  Where M = item-purified total score, G = group indicator (0=en, 1=target).

  Likelihood ratio tests:
    Total DIF      : χ²(2) = −2 (logL_0 − logL_2)
    Uniform DIF    : χ²(1) = −2 (logL_0 − logL_1)
    Non-uniform DIF: χ²(1) = −2 (logL_1 − logL_2)

  Effect size: Nagelkerke ΔR² (Zumbo classification rule):
    R² < 0.130           → negligible (analogue of ETS A)
    0.130 ≤ R² < 0.260   → moderate   (analogue of ETS B)
    R² ≥ 0.260           → large      (analogue of ETS C)

Outputs (irt_validations/results_logreg_dif_validation/):
  logreg_per_item_lang.csv       — per (item × lang) LRT statistics
  logreg_per_item_summary.csv    — aggregated per-item DIF summary
  logreg_anchor_vs_nonanchor.csv — distribution comparison
  logreg_validation_summary.txt  — plain-text summary

Usage:
  python logreg_dif_anchor_validation.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist, spearmanr
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors", "anchors.csv")
LORDS_CHI2_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model", "results_dif_stratified",
                                "dif_agreement_scores.csv")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_logreg_dif_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

REFERENCE_LANG = "en"
NON_EN_LANGS   = ["ar", "bn", "it", "jv", "ko", "sw", "th", "vi", "zh"]

# Zumbo (1999) ΔR² thresholds
R2_B_THRESHOLD = 0.130
R2_C_THRESHOLD = 0.260
SIG_ALPHA      = 0.05


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def load_responses():
    """
    Person = (test_taker × pass). Each person has up to 315 responses per
    language with total scores in the 0–315 range — matches the granularity
    needed for LR-DIF to have enough variance per item.
    """
    df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip")
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    df = df[df["judge_score"] > 0].dropna(subset=["judge_score"]).copy()
    df["safe"] = (df["judge_score"] >= 4).astype(int)
    df["id"]   = df["id"].apply(clean_id)

    sc = "test_taker" if "test_taker" in df.columns else "model"
    if "pass" in df.columns:
        df["person"] = df[sc].astype(str) + "_p" + df["pass"].astype(str)
    else:
        df["person"] = df[sc].astype(str)

    totals = (df.groupby(["person", "language"])["safe"].sum()
                .reset_index().rename(columns={"safe": "total_score"}))
    df = df.merge(totals, on=["person", "language"])
    return df


def load_anchor_ids():
    adf = pd.read_csv(ANCHOR_FILE)
    adf["id"] = adf["id"].apply(clean_id)
    return set(adf["id"].unique())


def load_lords_chi2():
    if not os.path.exists(LORDS_CHI2_FILE):
        return None
    ldf = pd.read_csv(LORDS_CHI2_FILE)
    ldf["prompt_id"] = ldf["prompt_id"].apply(clean_id)
    return ldf[["prompt_id", "mean_chi2"]].rename(columns={"mean_chi2": "lords_chi2"})


def load_candidate_ids():
    """
    Set of items that passed Lord's variance filter. See MH script for the
    same logic — non-anchor items outside this pool are near-saturated and
    contaminate the headline comparison.
    """
    if not os.path.exists(LORDS_CHI2_FILE):
        return None
    ldf = pd.read_csv(LORDS_CHI2_FILE)
    return {clean_id(x) for x in ldf["prompt_id"].tolist()}


# ── LR-DIF core ──────────────────────────────────────────────────────────────

def nagelkerke_r2(model_result, n):
    """Nagelkerke pseudo-R² from a statsmodels Logit result."""
    ll_full = model_result.llf
    ll_null = model_result.llnull
    cs   = 1 - np.exp((2.0 / n) * (ll_null - ll_full))
    norm = 1 - np.exp((2.0 / n) * ll_null)
    return cs / norm if norm > 0 else np.nan


def fit_logit(y, X):
    """Fit logistic regression; return result or None on failure."""
    try:
        # Use a small regularization-equivalent fallback if needed
        return sm.Logit(y, X).fit(disp=False, maxiter=100)
    except Exception:
        return None


def lr_dif_for_item_lang(item_df, ref_lang, foc_lang):
    """
    Fit nested logits and return uniform / non-uniform / total LRTs plus ΔR².
    Returns dict of statistics (NaN-filled on failure).
    """
    sub = item_df[item_df["language"].isin([ref_lang, foc_lang])].copy()
    sub["match"] = sub["total_score"] - sub["safe"]
    sub["G"]     = (sub["language"] == foc_lang).astype(int)

    y = sub["safe"].values.astype(float)
    n = len(y)
    if n < 20 or y.sum() in (0, n):
        return {k: np.nan for k in ["chi2_total", "p_total", "chi2_uniform",
                                     "p_uniform", "chi2_nonuniform",
                                     "p_nonuniform", "delta_r2"]}

    M  = sub["match"].values.astype(float)
    G  = sub["G"].values.astype(float)
    MG = M * G

    X0 = sm.add_constant(M, has_constant="add")               # M
    X1 = sm.add_constant(np.column_stack([M, G]),
                         has_constant="add")                  # M + G
    X2 = sm.add_constant(np.column_stack([M, G, MG]),
                         has_constant="add")                  # M + G + M*G

    r0 = fit_logit(y, X0)
    r1 = fit_logit(y, X1)
    r2 = fit_logit(y, X2)
    if r0 is None or r1 is None or r2 is None:
        return {k: np.nan for k in ["chi2_total", "p_total", "chi2_uniform",
                                     "p_uniform", "chi2_nonuniform",
                                     "p_nonuniform", "delta_r2"]}

    chi2_total = -2 * (r0.llf - r2.llf)
    chi2_unif  = -2 * (r0.llf - r1.llf)
    chi2_nonu  = -2 * (r1.llf - r2.llf)
    p_total = 1 - chi2_dist.cdf(chi2_total, df=2)
    p_unif  = 1 - chi2_dist.cdf(chi2_unif,  df=1)
    p_nonu  = 1 - chi2_dist.cdf(chi2_nonu,  df=1)

    r2_0 = nagelkerke_r2(r0, n)
    r2_2 = nagelkerke_r2(r2, n)
    dr2  = (r2_2 - r2_0) if (not np.isnan(r2_0) and not np.isnan(r2_2)) else np.nan

    return {
        "chi2_total":      chi2_total,
        "p_total":         p_total,
        "chi2_uniform":    chi2_unif,
        "p_uniform":       p_unif,
        "chi2_nonuniform": chi2_nonu,
        "p_nonuniform":    p_nonu,
        "delta_r2":        dr2,
    }


def classify_zumbo(delta_r2, p_total):
    if np.isnan(delta_r2) or np.isnan(p_total):
        return "NA"
    if p_total >= SIG_ALPHA or delta_r2 < R2_B_THRESHOLD:
        return "A"
    if delta_r2 < R2_C_THRESHOLD:
        return "B"
    return "C"


# ── Main computation ─────────────────────────────────────────────────────────

def compute_lr_dif_all_items(df):
    prompts = sorted(df["id"].unique())
    rows = []
    print(f"\nComputing LR-DIF for {len(prompts)} items × {len(NON_EN_LANGS)} languages "
          f"({len(prompts) * len(NON_EN_LANGS):,} cells)...")
    for n_i, pid in enumerate(prompts, 1):
        item_df = df[df["id"] == pid]
        for lang in NON_EN_LANGS:
            stats = lr_dif_for_item_lang(item_df, REFERENCE_LANG, lang)
            row = {"id": pid, "language": lang, **stats}
            row["zumbo_class"] = classify_zumbo(stats["delta_r2"], stats["p_total"])
            rows.append(row)
        if n_i % 50 == 0:
            print(f"  {n_i}/{len(prompts)} items done...")
    return pd.DataFrame(rows)


def summarize_per_item(lr_df):
    """
    Parallel to the MH summary: lead with continuous DIF magnitudes and the
    per-item proportion of languages classified A (pct_lang_A). The worst-case
    class is kept but not the headline.
    """
    rows = []
    for pid, sub in lr_df.groupby("id"):
        valid = sub.dropna(subset=["chi2_total"])
        n_valid = len(valid)
        if n_valid == 0:
            continue
        n_unif_sig  = int((valid["p_uniform"]    < SIG_ALPHA).sum())
        n_nonu_sig  = int((valid["p_nonuniform"] < SIG_ALPHA).sum())
        n_total_sig = int((valid["p_total"]      < SIG_ALPHA).sum())
        n_A = int((valid["zumbo_class"] == "A").sum())
        n_B = int((valid["zumbo_class"] == "B").sum())
        n_C = int((valid["zumbo_class"] == "C").sum())

        if n_C > 0:
            overall = "C"
        elif n_B > 0:
            overall = "B"
        elif n_A > 0:
            overall = "A"
        else:
            overall = "NA"

        rows.append({
            "id":                  pid,
            "n_valid_langs":       n_valid,
            "mean_chi2_total":     valid["chi2_total"].mean(),
            "median_chi2_total":   valid["chi2_total"].median(),
            "mean_delta_r2":       valid["delta_r2"].mean(),
            "median_delta_r2":     valid["delta_r2"].median(),
            "max_delta_r2":        valid["delta_r2"].max(),
            "n_lang_uniform_sig":  n_unif_sig,
            "n_lang_nonunif_sig":  n_nonu_sig,
            "n_lang_total_sig":    n_total_sig,
            "n_lang_A":            n_A,
            "n_lang_B":            n_B,
            "n_lang_C":            n_C,
            "pct_lang_A":          round(100 * n_A / n_valid, 2),
            "overall_class":       overall,
        })
    return pd.DataFrame(rows)


def compare_anchor_vs_nonanchor(summary_df, anchor_ids, candidate_ids=None):
    """
    If `candidate_ids` is provided, restrict to anchors + candidate non-anchors
    (apples-to-apples). Otherwise compare against all 275 non-anchors.
    """
    summary_df = summary_df.copy()
    summary_df["is_anchor"]    = summary_df["id"].isin(anchor_ids)
    summary_df["is_candidate"] = (
        summary_df["id"].isin(candidate_ids) if candidate_ids is not None
        else True
    )
    if candidate_ids is not None:
        cmp_df = summary_df[summary_df["is_anchor"] | summary_df["is_candidate"]].copy()
        non_anchor_label = "candidate non-anchor (variance-filter passed)"
    else:
        cmp_df = summary_df
        non_anchor_label = "non-anchor (all, includes saturated items)"

    rows = []
    for is_anc, label in [(True, "anchor (Lord's selected)"),
                          (False, non_anchor_label)]:
        sub = cmp_df[cmp_df["is_anchor"] == is_anc]
        n = len(sub)
        if n == 0:
            continue
        rows.append({
            "group":                       label,
            "n_items":                     n,
            # ── headline ──
            "mean_chi2_acrossitems":       round(sub["mean_chi2_total"].mean(),   3),
            "median_chi2_acrossitems":     round(sub["mean_chi2_total"].median(), 3),
            "median_delta_r2":             round(sub["mean_delta_r2"].median(),   4),
            "mean_pct_lang_A":             round(sub["pct_lang_A"].mean(),        2),
            "median_pct_lang_A":           round(sub["pct_lang_A"].median(),      2),
            "mean_unif_sig":               round(sub["n_lang_uniform_sig"].mean(), 2),
            "mean_nonunif_sig":            round(sub["n_lang_nonunif_sig"].mean(), 2),
            # ── worst-case, completeness only ──
            "pct_class_A_worstcase":       round(100 * (sub["overall_class"] == "A").mean(), 1),
            "pct_class_B_worstcase":       round(100 * (sub["overall_class"] == "B").mean(), 1),
            "pct_class_C_worstcase":       round(100 * (sub["overall_class"] == "C").mean(), 1),
        })
    return pd.DataFrame(rows), summary_df


def compute_lords_lr_rank_correlation(summary_df, lords_df):
    if lords_df is None:
        return None
    merged = summary_df.merge(lords_df, left_on="id", right_on="prompt_id",
                              how="inner").dropna(subset=["mean_chi2_total", "lords_chi2"])
    if len(merged) < 5:
        return None
    rho, p = spearmanr(merged["mean_chi2_total"], merged["lords_chi2"])
    return {
        "n":            len(merged),
        "spearman_rho": round(float(rho), 4),
        "p_value":      float(p),
    }


def write_summary(lr_df, summary_df, compare_df, rank_corr, n_anchors,
                  n_nonanchors, lords_loaded):
    path = os.path.join(RESULTS_DIR, "logreg_validation_summary.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("LOGISTIC-REGRESSION DIF VALIDATION OF LORD'S-χ² SELECTED ANCHORS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Purpose: Complements the MH check by also catching NON-UNIFORM DIF,")
    lines.append("which parallels the 2PL discrimination term α_i in our IRT model.")
    lines.append("LR-DIF conditions on observed total score, so no IRT calibration.")
    lines.append("")
    lines.append(f"Anchors evaluated     : {n_anchors}")
    lines.append(f"Non-anchors evaluated : {n_nonanchors}")
    lines.append(f"Languages (vs English): {len(NON_EN_LANGS)}  ({', '.join(NON_EN_LANGS)})")
    lines.append("")
    lines.append("-" * 70)
    lines.append("HEADLINE: rank correlation with Lord's χ²")
    lines.append("-" * 70)
    if rank_corr is None:
        if not lords_loaded:
            lines.append("  (Skipped — Lord's-χ² file not found at")
            lines.append("   model/results_dif_stratified/dif_agreement_scores.csv.")
            lines.append("   Run `python model/anchors.py` to enable.)")
        else:
            lines.append("  (Skipped — insufficient overlap.)")
    else:
        lines.append(f"  Spearman ρ = {rank_corr['spearman_rho']:+.3f}  "
                     f"(n = {rank_corr['n']}, p = {rank_corr['p_value']:.2e})")
        lines.append("  Positive ρ means Lord's and LR-DIF rank items consistently.")
    lines.append("")
    lines.append("-" * 70)
    lines.append("ITEM-LEVEL DIF MAGNITUDE  (anchors vs non-anchors)")
    lines.append("-" * 70)
    for _, r in compare_df.iterrows():
        lines.append(f"  {r['group']:<33s} n={r['n_items']:<4}  "
                     f"mean χ²={r['mean_chi2_acrossitems']:>7.2f}  "
                     f"median χ²={r['median_chi2_acrossitems']:>7.2f}  "
                     f"median ΔR²={r['median_delta_r2']:>6.4f}")
    lines.append("")
    lines.append("  pct_lang_A : within-item proportion of languages classified Zumbo A")
    lines.append("")
    for _, r in compare_df.iterrows():
        lines.append(f"  {r['group']:<33s} n={r['n_items']:<4}  "
                     f"mean pct_A={r['mean_pct_lang_A']:>5.1f}%  "
                     f"median pct_A={r['median_pct_lang_A']:>5.1f}%")
    lines.append("")
    lines.append("-" * 70)
    lines.append("UNIFORM vs NON-UNIFORM DIF (mean # of 9 languages flagged per item)")
    lines.append("-" * 70)
    for _, r in compare_df.iterrows():
        lines.append(f"  {r['group']:<33s}  uniform sig: {r['mean_unif_sig']:.2f}   "
                     f"non-uniform sig: {r['mean_nonunif_sig']:.2f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("WORST-CASE ZUMBO ΔR² CLASSIFICATION  (completeness only)")
    lines.append("-" * 70)
    lines.append(f"  A: ΔR² < {R2_B_THRESHOLD} OR LRT not significant")
    lines.append(f"  B: {R2_B_THRESHOLD} ≤ ΔR² < {R2_C_THRESHOLD} AND LRT significant")
    lines.append(f"  C: ΔR² ≥ {R2_C_THRESHOLD} AND LRT significant")
    lines.append("")
    for _, r in compare_df.iterrows():
        lines.append(f"  {r['group']:<33s} n={r['n_items']:<4}  "
                     f"A: {r['pct_class_A_worstcase']:>5.1f}%  "
                     f"B: {r['pct_class_B_worstcase']:>5.1f}%  "
                     f"C: {r['pct_class_C_worstcase']:>5.1f}%")
    lines.append("")
    lines.append("-" * 70)
    lines.append("RANK CORRELATION DETAIL (already shown above as headline)")
    lines.append("-" * 70)
    if rank_corr is None:
        if not lords_loaded:
            lines.append("  (Skipped — Lord's-χ² file not found at")
            lines.append("   model/results_dif_stratified/dif_agreement_scores.csv.")
            lines.append("   Run `python model/anchors.py` to enable.)")
        else:
            lines.append("  (Skipped — insufficient overlap.)")
    else:
        lines.append(f"  Spearman ρ = {rank_corr['spearman_rho']:+.3f}  "
                     f"(n = {rank_corr['n']}, p = {rank_corr['p_value']:.2e})")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSummary written → {path}")


def main():
    print("Loading response data and anchor set...")
    df            = load_responses()
    anchor_ids    = load_anchor_ids()
    lords_df      = load_lords_chi2()
    candidate_ids = load_candidate_ids()

    print(f"  {len(df):,} rows | {df['id'].nunique()} prompts | "
          f"{df['language'].nunique()} languages | {len(anchor_ids)} anchors")
    if candidate_ids is not None:
        print(f"  Lord's candidate pool (variance-filter passed): {len(candidate_ids)}")
    else:
        print("  [WARN] Lord's candidate pool not available — non-anchor group will")
        print("         include all 275 non-anchor items (mix of candidates and saturated).")

    lr_df = compute_lr_dif_all_items(df)
    lr_df.to_csv(os.path.join(RESULTS_DIR, "logreg_per_item_lang.csv"), index=False)

    summary_df = summarize_per_item(lr_df)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "logreg_per_item_summary.csv"), index=False)

    compare_df, summary_df = compare_anchor_vs_nonanchor(
        summary_df, anchor_ids, candidate_ids=candidate_ids)
    compare_df.to_csv(os.path.join(RESULTS_DIR, "logreg_anchor_vs_nonanchor.csv"), index=False)

    rc_input = (summary_df[summary_df["is_candidate"] | summary_df["is_anchor"]]
                if candidate_ids is not None else summary_df)
    rank_corr = compute_lords_lr_rank_correlation(rc_input, lords_df)

    n_anchors    = int(summary_df["is_anchor"].sum())
    n_nonanchors = len(summary_df) - n_anchors
    write_summary(lr_df, summary_df, compare_df, rank_corr,
                  n_anchors, n_nonanchors, lords_df is not None)

    print(f"\nAll outputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
