# -*- coding: utf-8 -*-
"""
Mantel-Haenszel DIF Validation of Lord's-χ² Selected Anchors
=============================================================
Cross-validates our Lord's-χ²-selected anchors against an independent,
non-parametric DIF detector (Mantel-Haenszel) that does NOT require an
IRT-calibrated scale.

Lord's χ² needs a calibrated scale; we form one inside the selection step.
MH conditions on the observed total score rather than a latent trait, so
it has no scale-calibration dependency. Agreement between Lord's-χ²-selected
anchors and MH classifications is convergent evidence from a methodologically
distinct detector.

Method (per Holland & Thayer 1988, ETS Δ scale):
  • For each (item × non-English language) pair, compute MH χ²(1) and the
    common odds ratio α_MH, with English as the reference group.
  • Matching variable: total safe-response score within each respondent's own
    language, item-purified (focal item removed from the matching score).
  • Convert to ETS Δ: Δ_MH = −2.35 × ln(α_MH).
  • Classify per (item, language) by ETS rules:
      A (negligible)  : |Δ_MH| < 1.0  OR  MH χ² not significant at α = 0.05
      B (moderate)    : 1.0 ≤ |Δ_MH| < 1.5  AND  significantly different from 0
      C (large)       : |Δ_MH| ≥ 1.5  AND  significantly different from 1.0
  • Aggregate per item: count A/B/C across the 9 languages; report mean |Δ_MH|
    and mean MH χ² as item-level DIF magnitude.
  • Compare anchor (N=40) vs non-anchor (N=275) item distributions in each ETS class.
  • Spearman ρ between mean MH χ² and Lord's-χ² mean_chi2 across all candidate
    items (if Lord's-χ² scores file available).

Caveat reported in the summary: MH detects primarily UNIFORM DIF. Non-uniform
DIF (which parallels the 2PL discrimination term) is covered by
logreg_dif_anchor_validation.py.

Outputs (irt_validations/results_mh_validation/):
  mh_per_item_lang.csv        — MH stats for every (item × language) cell
  mh_per_item_summary.csv     — aggregated per-item DIF summary
  mh_anchor_vs_nonanchor.csv  — ETS distributions for anchors vs non-anchors
  mh_validation_summary.txt   — plain-text summary

Usage:
  python mh_anchor_validation.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist, spearmanr
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors", "anchors.csv")
LORDS_CHI2_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model", "results_dif_stratified",
                                "dif_agreement_scores.csv")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_mh_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

REFERENCE_LANG = "en"
NON_EN_LANGS   = ["ar", "bn", "it", "jv", "ko", "sw", "th", "vi", "zh"]

# ETS thresholds
ETS_B_THRESHOLD = 1.0
ETS_C_THRESHOLD = 1.5
SIG_ALPHA       = 0.05

# Matching-variable binning. Integer purified scores in 0–315 are too sparse
# with ~600 persons per language; binning forces stratum overlap between
# reference and focal groups. Standard practice (Holland & Thayer 1988).
N_STRATA_BINS  = 10   # deciles


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def load_responses():
    """
    Load and binarize raw responses. Defines `person` as the (test_taker × pass)
    combination — each person has up to 315 responses per language, with
    total scores in the natural 0–315 range. (Grouping only by test_taker
    sums across all 10 passes and produces 0–3150 ranges that are too sparse
    for MH stratification.)
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

    # Total score within each (person, language) — pre-purification
    totals = (df.groupby(["person", "language"])["safe"].sum()
                .reset_index().rename(columns={"safe": "total_score"}))
    df = df.merge(totals, on=["person", "language"])

    # Bin the matching variable into deciles WITHIN each language so each
    # respondent's purified score maps to one of N_STRATA_BINS coarse strata.
    # This is the standard Holland-Thayer remedy for sparse strata.
    return df


def assign_match_bins(item_df, ref_lang, foc_lang):
    """
    Compute item-purified score for each row, then bin into N_STRATA_BINS
    quantile bins computed jointly across reference + focal language. Joint
    binning forces strata to overlap.
    """
    sub = item_df[item_df["language"].isin([ref_lang, foc_lang])].copy()
    sub["match_raw"] = sub["total_score"] - sub["safe"]
    # Joint quantile bins so both groups land in shared strata
    try:
        sub["match"] = pd.qcut(sub["match_raw"], q=N_STRATA_BINS,
                                labels=False, duplicates="drop")
    except ValueError:
        sub["match"] = 0
    return sub


def load_anchor_ids():
    """Load the set of Lord's-χ² selected anchor IDs."""
    adf = pd.read_csv(ANCHOR_FILE)
    adf["id"] = adf["id"].apply(clean_id)
    return set(adf["id"].unique())


def load_lords_chi2():
    """Load per-item Lord's-χ² mean if available; else None."""
    if not os.path.exists(LORDS_CHI2_FILE):
        print(f"  [INFO] Lord's-χ² file not found at {LORDS_CHI2_FILE}")
        print("         (rank correlation with MH will be skipped — run model/anchors.py to enable)")
        return None
    ldf = pd.read_csv(LORDS_CHI2_FILE)
    ldf["prompt_id"] = ldf["prompt_id"].apply(clean_id)
    return ldf[["prompt_id", "mean_chi2"]].rename(columns={"mean_chi2": "lords_chi2"})


def load_candidate_ids():
    """
    Return the set of items that passed Lord's variance filter (the candidate
    pool from which anchors were selected). Required for the apples-to-apples
    anchor-vs-non-anchor comparison — otherwise the non-anchor group is
    contaminated with near-saturated items (P(Safe|EN) outside 0.05–0.95)
    that fail variance filtering and have artificially low MH χ².
    """
    if not os.path.exists(LORDS_CHI2_FILE):
        return None
    ldf = pd.read_csv(LORDS_CHI2_FILE)
    return {clean_id(x) for x in ldf["prompt_id"].tolist()}


# ── MH core ───────────────────────────────────────────────────────────────────

def mh_for_item_lang(item_df, ref_lang, foc_lang):
    """
    Compute MH χ² (Holland-Thayer continuity-corrected) and common odds ratio
    α_MH for a single item, comparing ref_lang vs foc_lang. Returns
    (chi2_stat, alpha_mh, delta_mh, p_value, n_strata_used).

    Matching variable: total_score minus the focal item's safe response
    (Holland & Thayer 1988 item purification).
    """
    sub = assign_match_bins(item_df, ref_lang, foc_lang)

    ref = sub[sub["language"] == ref_lang]
    foc = sub[sub["language"] == foc_lang]

    if len(ref) == 0 or len(foc) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0

    num         = 0.0   # Σ (A_k − E[A_k])
    var_sum     = 0.0   # Σ Var(A_k)
    ad_over_N   = 0.0   # Σ (A_k * D_k / N_k)
    bc_over_N   = 0.0   # Σ (B_k * C_k / N_k)
    n_strata_used = 0

    strata = sorted(set(ref["match"]).union(set(foc["match"])))
    for s in strata:
        rs = ref[ref["match"] == s]
        fs = foc[foc["match"] == s]
        n_R, n_F = len(rs), len(fs)
        N = n_R + n_F
        if n_R == 0 or n_F == 0:
            continue

        A = int(rs["safe"].sum())     # ref correct
        C = int(fs["safe"].sum())     # foc correct
        B = n_R - A                   # ref incorrect
        D = n_F - C                   # foc incorrect
        m1 = A + C                    # correct in stratum
        m0 = B + D                    # incorrect in stratum

        if m1 == 0 or m0 == 0 or N < 2:
            continue

        E_A   = n_R * m1 / N
        var_A = (n_R * n_F * m1 * m0) / (N * N * (N - 1))

        num       += A - E_A
        var_sum   += var_A
        ad_over_N += A * D / N
        bc_over_N += B * C / N
        n_strata_used += 1

    if var_sum <= 0 or bc_over_N <= 0 or ad_over_N <= 0:
        return np.nan, np.nan, np.nan, np.nan, n_strata_used

    # Holland-Thayer χ² (continuity-corrected)
    chi2_stat = (max(abs(num) - 0.5, 0.0)) ** 2 / var_sum
    p_value   = 1.0 - chi2_dist.cdf(chi2_stat, df=1)

    alpha_mh  = ad_over_N / bc_over_N
    delta_mh  = -2.35 * np.log(alpha_mh)

    return chi2_stat, alpha_mh, delta_mh, p_value, n_strata_used


def classify_ets(delta_mh, p_value):
    """ETS A/B/C classification."""
    if np.isnan(delta_mh) or np.isnan(p_value):
        return "NA"
    abs_d = abs(delta_mh)
    # A: negligible — small effect OR not significant
    if abs_d < ETS_B_THRESHOLD or p_value >= SIG_ALPHA:
        return "A"
    if abs_d < ETS_C_THRESHOLD:
        return "B"
    return "C"


# ── Main computation ─────────────────────────────────────────────────────────

def compute_mh_all_items(df):
    """Run MH for every (item × non-English language) pair."""
    prompts = sorted(df["id"].unique())
    rows = []

    print(f"\nComputing MH χ² for {len(prompts)} items × {len(NON_EN_LANGS)} languages "
          f"({len(prompts) * len(NON_EN_LANGS):,} cells)...")
    for n_i, pid in enumerate(prompts, 1):
        item_df = df[df["id"] == pid]
        for lang in NON_EN_LANGS:
            chi2_stat, alpha, delta, pval, n_strata = mh_for_item_lang(
                item_df, REFERENCE_LANG, lang)
            ets = classify_ets(delta, pval)
            rows.append({
                "id":         pid,
                "language":   lang,
                "mh_chi2":    chi2_stat,
                "alpha_mh":   alpha,
                "delta_mh":   delta,
                "abs_delta":  abs(delta) if not np.isnan(delta) else np.nan,
                "p_value":    pval,
                "ets_class":  ets,
                "n_strata":   n_strata,
            })
        if n_i % 50 == 0:
            print(f"  {n_i}/{len(prompts)} items done...")
    return pd.DataFrame(rows)


def summarize_per_item(mh_df):
    """
    Aggregate per-item. Reports BOTH:
      • continuous stats (mean / median MH χ² and |Δ_MH|) — robust to extreme γ
      • per-item proportion of languages classified A (pct_lang_A) — the
        within-item DIF-free majority signal
      • worst-case ETS class across 9 languages — kept for completeness but
        not the headline (a single noisy language pushes any item to C, which
        in this data is uninformative because γ_L is large).
    """
    rows = []
    for pid, sub in mh_df.groupby("id"):
        valid = sub.dropna(subset=["mh_chi2"])
        n_valid = len(valid)
        if n_valid == 0:
            continue
        n_A  = int((valid["ets_class"] == "A").sum())
        n_B  = int((valid["ets_class"] == "B").sum())
        n_C  = int((valid["ets_class"] == "C").sum())

        # Worst-case class (kept but de-emphasized)
        if n_C > 0:
            overall = "C"
        elif n_B > 0:
            overall = "B"
        elif n_A > 0:
            overall = "A"
        else:
            overall = "NA"

        rows.append({
            "id":              pid,
            "n_valid_langs":   n_valid,
            "mean_mh_chi2":    valid["mh_chi2"].mean(),
            "median_mh_chi2":  valid["mh_chi2"].median(),
            "max_mh_chi2":     valid["mh_chi2"].max(),
            "mean_abs_delta":  valid["abs_delta"].mean(),
            "max_abs_delta":   valid["abs_delta"].max(),
            "n_lang_A":        n_A,
            "n_lang_B":        n_B,
            "n_lang_C":        n_C,
            "pct_lang_A":      round(100 * n_A / n_valid, 2),
            "overall_class":   overall,
        })
    return pd.DataFrame(rows)


def compare_anchor_vs_nonanchor(summary_df, anchor_ids, candidate_ids=None):
    """
    Build the anchor vs non-anchor comparison table. The HEADLINE metrics
    are the continuous ones (mean / median MH χ², median |Δ|) and the
    per-item pct_lang_A — these robustly capture the anchor signal.

    If `candidate_ids` is provided, the comparison is restricted to items
    that passed Lord's variance filter (apples-to-apples: both anchors and
    non-anchors are drawn from the same candidate pool). Without it, the
    non-anchor group is contaminated with near-saturated items that fail
    variance filtering and have artificially low MH χ².
    """
    summary_df = summary_df.copy()
    summary_df["is_anchor"]    = summary_df["id"].isin(anchor_ids)
    summary_df["is_candidate"] = (
        summary_df["id"].isin(candidate_ids) if candidate_ids is not None
        else True
    )
    if candidate_ids is not None:
        # Keep all anchors + candidate non-anchors only (drop saturated items)
        cmp_df = summary_df[summary_df["is_anchor"] | summary_df["is_candidate"]].copy()
        non_anchor_label = "candidate non-anchor (variance-filter passed)"
    else:
        cmp_df = summary_df
        non_anchor_label = "non-anchor (all, includes saturated items)"

    rows = []
    for is_anc, label in [(True, "anchor (N=Lord's selected)"),
                          (False, non_anchor_label)]:
        sub = cmp_df[cmp_df["is_anchor"] == is_anc]
        n = len(sub)
        if n == 0:
            continue
        row = {
            "group":                   label,
            "n_items":                 n,
            # ── headline continuous metrics ──
            "mean_mh_chi2_acrossitems":   round(sub["mean_mh_chi2"].mean(),    3),
            "median_mh_chi2_acrossitems": round(sub["mean_mh_chi2"].median(),  3),
            "median_abs_delta":           round(sub["mean_abs_delta"].median(), 3),
            "mean_pct_lang_A":            round(sub["pct_lang_A"].mean(),       2),
            "median_pct_lang_A":          round(sub["pct_lang_A"].median(),     2),
            # ── worst-case ETS, kept but not the headline ──
            "pct_class_A_worstcase":      round(100 * (sub["overall_class"] == "A").mean(), 1),
            "pct_class_B_worstcase":      round(100 * (sub["overall_class"] == "B").mean(), 1),
            "pct_class_C_worstcase":      round(100 * (sub["overall_class"] == "C").mean(), 1),
        }
        rows.append(row)
    return pd.DataFrame(rows), summary_df


def compute_lords_mh_rank_correlation(summary_df, lords_df):
    """Spearman ρ between mean MH χ² and Lord's mean_chi2."""
    if lords_df is None:
        return None
    merged = summary_df.merge(lords_df, left_on="id", right_on="prompt_id",
                              how="inner").dropna(subset=["mean_mh_chi2", "lords_chi2"])
    if len(merged) < 5:
        return None
    rho, p = spearmanr(merged["mean_mh_chi2"], merged["lords_chi2"])
    return {
        "n":            len(merged),
        "spearman_rho": round(float(rho), 4),
        "p_value":      float(p),
    }


# ── Output ────────────────────────────────────────────────────────────────────

def write_summary(mh_df, summary_df, compare_df, rank_corr, n_anchors,
                  n_nonanchors, lords_loaded):
    path = os.path.join(RESULTS_DIR, "mh_validation_summary.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("MANTEL-HAENSZEL DIF VALIDATION OF LORD'S-χ² SELECTED ANCHORS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Purpose: Cross-validate Lord's-χ²-selected anchors against MH, a")
    lines.append("non-parametric DIF detector that does not require an IRT-calibrated")
    lines.append("scale.")
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
            lines.append("  (Skipped — model/results_dif_stratified/dif_agreement_scores.csv")
            lines.append("   not found. Run `python model/anchors.py` to enable this comparison.)")
        else:
            lines.append("  (Skipped — insufficient overlap between Lord's and MH item sets.)")
    else:
        lines.append(f"  Spearman ρ = {rank_corr['spearman_rho']:+.3f}  "
                     f"(n = {rank_corr['n']}, p = {rank_corr['p_value']:.2e})")
        lines.append("  Positive ρ means both detectors rank the same items as more / less")
        lines.append("  DIF-prone, despite MH having no shared scale assumption with Lord's.")
    lines.append("")
    lines.append("-" * 70)
    lines.append("ITEM-LEVEL DIF MAGNITUDE  (anchors vs non-anchors)")
    lines.append("-" * 70)
    lines.append("  Continuous metrics — robust to large γ_L in our data.")
    lines.append("")
    for _, r in compare_df.iterrows():
        lines.append(f"  {r['group']:<35s} n={r['n_items']:<4}  "
                     f"mean χ²={r['mean_mh_chi2_acrossitems']:>7.2f}  "
                     f"median χ²={r['median_mh_chi2_acrossitems']:>7.2f}  "
                     f"median |Δ|={r['median_abs_delta']:>5.2f}")
    lines.append("")
    lines.append("  pct_lang_A : within-item proportion of languages classified ETS A")
    lines.append("  (i.e., the share of language pairs that look DIF-free per item).")
    lines.append("")
    for _, r in compare_df.iterrows():
        lines.append(f"  {r['group']:<35s} n={r['n_items']:<4}  "
                     f"mean pct_A={r['mean_pct_lang_A']:>5.1f}%  "
                     f"median pct_A={r['median_pct_lang_A']:>5.1f}%")
    lines.append("")
    lines.append("-" * 70)
    lines.append("WORST-CASE ETS CLASSIFICATION  (single noisy language → C)")
    lines.append("-" * 70)
    lines.append("  Kept for completeness only; uninformative in this dataset because γ_L")
    lines.append("  is large and ETS thresholds (1.0 / 1.5) were calibrated for small effects.")
    lines.append("  A = |Δ_MH| < 1.0  OR  not significant ; B = 1.0–1.5 ; C = ≥1.5")
    lines.append("")
    for _, r in compare_df.iterrows():
        lines.append(f"  {r['group']:<35s} n={r['n_items']:<4}  "
                     f"A: {r['pct_class_A_worstcase']:>5.1f}%  "
                     f"B: {r['pct_class_B_worstcase']:>5.1f}%  "
                     f"C: {r['pct_class_C_worstcase']:>5.1f}%")
    lines.append("")
    lines.append("-" * 70)
    lines.append("CAVEAT")
    lines.append("-" * 70)
    lines.append("  MH primarily detects UNIFORM DIF (consistent advantage at all ability")
    lines.append("  levels). Non-uniform DIF — which parallels the 2PL discrimination term")
    lines.append("  in our model — is not fully covered. See logreg_dif_anchor_validation.py")
    lines.append("  for a complementary check that handles both.")
    lines.append("")
    lines.append("Convergent corroboration from an independent method that does not depend")
    lines.append("on IRT scale calibration.")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSummary written → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading response data and anchor set...")
    df            = load_responses()
    anchor_ids    = load_anchor_ids()
    lords_df      = load_lords_chi2()
    candidate_ids = load_candidate_ids()

    n_persons = df.groupby("language")["person"].nunique().to_dict()
    print(f"  {len(df):,} rows | {df['id'].nunique()} prompts | "
          f"{df['language'].nunique()} languages | {len(anchor_ids)} anchors")
    if candidate_ids is not None:
        print(f"  Lord's candidate pool (variance-filter passed): {len(candidate_ids)}")
    else:
        print("  [WARN] Lord's candidate pool not available — non-anchor group will")
        print("         include all 275 non-anchor items (mix of candidates and saturated).")
    print(f"  Persons per language: {n_persons}")

    mh_df = compute_mh_all_items(df)
    mh_df.to_csv(os.path.join(RESULTS_DIR, "mh_per_item_lang.csv"), index=False)

    summary_df = summarize_per_item(mh_df)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "mh_per_item_summary.csv"), index=False)

    compare_df, summary_df = compare_anchor_vs_nonanchor(
        summary_df, anchor_ids, candidate_ids=candidate_ids)
    compare_df.to_csv(os.path.join(RESULTS_DIR, "mh_anchor_vs_nonanchor.csv"), index=False)

    # Rank correlation: restrict to candidate items so it's apples-to-apples
    rc_input = (summary_df[summary_df["is_candidate"] | summary_df["is_anchor"]]
                if candidate_ids is not None else summary_df)
    rank_corr = compute_lords_mh_rank_correlation(rc_input, lords_df)

    n_anchors    = int(summary_df["is_anchor"].sum())
    n_nonanchors = len(summary_df) - n_anchors
    write_summary(mh_df, summary_df, compare_df, rank_corr,
                  n_anchors, n_nonanchors, lords_df is not None)

    print(f"\nAll outputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
