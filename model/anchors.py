# -*- coding: utf-8 -*-
"""
Stratified-Variance + Agreement-Rank Anchor Selection
======================================================
Combines two ideas into a single, non-iterative anchor selection pipeline:

  1. Variance stratification (Goldilocks filter)
     Keep only prompts where P(Safe|EN) ∈ (VARIANCE_LO, VARIANCE_HI).
     This excludes saturated items.  A softer ceiling (0.95 instead of 0.80)
     is offered as an alternative to widen the candidate pool.

  2. Agreement-rank selection
     For each candidate item, compute Lord's χ²(2) against every focal
     language using all candidates as a provisional equating set.
     Average the χ² statistic across all 9 languages → "DIF agreement score".
     Items with the lowest score are maximally invariant across languages.
     Select the top N_ANCHORS items.

Outputs (in results_dif_stratified/):
  variance_filter_stats.csv   — P(Safe|EN) and filter flag per prompt
  dif_agreement_scores.csv    — per-item mean χ², rank, selected flag
  soft_anchor_priors.csv      — prompt_id + prior_sigma for irt.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from huggingface_hub import snapshot_download

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_dif_stratified")
os.makedirs(RESULTS_DIR, exist_ok=True)

REFERENCE_LANG = "en"
ALL_LANGS      = ["zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]

# Variance filter bounds — two options:
#   Strict  (0.20, 0.80): pure Goldilocks
#   Softer  (0.05, 0.95): avoid extreme saturation, wider candidate pool
VARIANCE_LO = 0.05   # lower bound on P(Safe|EN)
VARIANCE_HI = 0.95   # upper bound on P(Safe|EN)

# Number of anchors to select from the ranked candidates
N_ANCHORS = 40

# Soft-anchor prior σ for confirmed anchors in irt.py
ANCHOR_PRIOR_SIGMA = 0.01

MIN_PERSONS = 10
EM_OUTER    = 10
EM_THETA    = 10
MIN_DELTA_B = 0.5    # ETS B-level effect size gate


# ── 2PL estimation ────────────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _update_theta(X, a, b):
    N = X.shape[0]
    theta = np.zeros(N)
    mask = ~np.isnan(X)
    for _ in range(EM_THETA):
        logits = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
        P = _sigmoid(logits)
        resid = np.where(mask, X - P, 0.0)
        grad  = (a[np.newaxis, :] * resid).sum(axis=1)
        info  = (a[np.newaxis, :] ** 2 * P * (1 - P) * mask).sum(axis=1)
        theta += grad / np.maximum(info, 1e-6)
        theta  = np.clip(theta, -4.0, 4.0)
    return theta


def _item_nll_and_grad(params, X, theta):
    I = X.shape[1]
    a = np.exp(params[:I])
    b = params[I:]
    diff  = theta[:, np.newaxis] - b[np.newaxis, :]
    P     = _sigmoid(a[np.newaxis, :] * diff)
    mask  = ~np.isnan(X)
    ll    = np.where(mask, X * np.log(P + 1e-8) + (1 - X) * np.log(1 - P + 1e-8), 0.0)
    resid = np.where(mask, X - P, 0.0)
    g_la  = -(resid * a[np.newaxis, :] * diff).sum(axis=0)
    g_b   =  (resid * a[np.newaxis, :]).sum(axis=0)
    return -ll.sum(), np.concatenate([g_la, g_b])


def _item_fisher_info(a, b, theta):
    P  = _sigmoid(a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :]))
    PQ = P * (1 - P)
    diff = theta[:, np.newaxis] - b[np.newaxis, :]
    J_aa = ((a[np.newaxis, :] * diff) ** 2 * PQ).sum(axis=0)
    J_bb = (a[np.newaxis, :] ** 2 * PQ).sum(axis=0)
    J_ab = -(a[np.newaxis, :] ** 2 * diff * PQ).sum(axis=0)
    det  = np.where(np.abs(J_aa * J_bb - J_ab ** 2) < 1e-10,
                    1e-10, J_aa * J_bb - J_ab ** 2)
    var_a  = a ** 2 * (J_bb / det)
    cov_ab = a      * (-J_ab / det)
    se_a   = np.sqrt(np.maximum(var_a, 1e-8))
    se_b   = np.sqrt(np.maximum(J_aa / det, 1e-8))
    return se_a, se_b, cov_ab


def fit_2pl(X):
    N, I = X.shape
    prop = np.clip(np.nanmean(X, axis=0), 0.05, 0.95)
    b = -np.log(prop / (1 - prop))
    a = np.ones(I)
    theta = np.zeros(N)
    for _ in range(EM_OUTER):
        theta = _update_theta(X, a, b)
        mu, sd = theta.mean(), theta.std()
        theta = (theta - mu) / max(sd, 1e-6)
        res = minimize(_item_nll_and_grad,
                       np.concatenate([np.log(np.maximum(a, 0.01)), b]),
                       args=(X, theta), method="L-BFGS-B", jac=True,
                       options={"maxiter": 300, "ftol": 1e-7, "gtol": 1e-5})
        a = np.exp(np.clip(res.x[:I], -3, 3))
        b = np.clip(res.x[I:], -5, 5)
    se_a, se_b, cov_ab = _item_fisher_info(a, b, theta)
    return a, b, se_a, se_b, cov_ab


def mean_equating(b_ref, b_foc, candidate_mask):
    """Mean-only equating (A=1). Appropriate when anchor b-spread is low."""
    br = b_ref[candidate_mask]
    bf = b_foc[candidate_mask]
    if len(br) < 1:
        return 1.0, 0.0
    return 1.0, float(br.mean() - bf.mean())


def lords_chi_square(a_r, b_r, se_a_r, se_b_r, cov_ab_r,
                     a_f, b_f, se_a_f, se_b_f, cov_ab_f):
    S_aa = se_a_r ** 2 + se_a_f ** 2
    S_bb = se_b_r ** 2 + se_b_f ** 2
    S_ab = cov_ab_r + cov_ab_f
    da, db = a_r - a_f, b_r - b_f
    det  = np.where(np.abs(S_aa * S_bb - S_ab ** 2) < 1e-10,
                    1e-10, S_aa * S_bb - S_ab ** 2)
    stat = (da ** 2 * S_bb - 2.0 * da * db * S_ab + db ** 2 * S_aa) / det
    return stat, 1.0 - chi2.cdf(stat, df=2)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_response_matrices():
    df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip",
                     usecols=["id", "language", "test_taker", "judge_score", "pass"])
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    df = df[df["judge_score"] > 0].dropna(subset=["judge_score"]).copy()
    df["binary"]     = (df["judge_score"] >= 4).astype(float)
    df["person_key"] = df["test_taker"].astype(str) + "_p" + df["pass"].astype(str)
    df["id"]         = df["id"].astype(str)
    matrices = {}
    for lang in [REFERENCE_LANG] + ALL_LANGS:
        sub = df[df["language"] == lang]
        if sub.empty:
            continue
        matrices[lang] = sub.pivot_table(
            index="person_key", columns="id", values="binary", aggfunc="first")
        print(f"  {lang}: {matrices[lang].shape[0]} persons × {matrices[lang].shape[1]} prompts")
    return matrices


# ── Step 1: Variance filter ───────────────────────────────────────────────────

def variance_filter(mat_ref):
    """
    Filter prompts by P(Safe|θ=0) using 2PL-corrected probability.
    Returns candidate_ids (set) and stats DataFrame.
    """
    print(f"\nStep 1 — Variance filter (fitting English 2PL for ability correction):")
    X = mat_ref.values.astype(float)
    prompt_ids = mat_ref.columns.tolist()
    a, b, _, _, _ = fit_2pl(X)

    p_safe_2pl = _sigmoid(-b)          # P(Safe | θ=0) = σ(−b)
    p_safe_raw = np.nanmean(X, axis=0)

    in_window = (p_safe_2pl > VARIANCE_LO) & (p_safe_2pl < VARIANCE_HI)
    candidate_ids = set(np.array(prompt_ids)[in_window])

    n_total = len(prompt_ids)
    n_cand  = in_window.sum()
    print(f"  P(Safe|θ=0) ∈ ({VARIANCE_LO}, {VARIANCE_HI}): "
          f"{n_cand}/{n_total} candidates ({100*n_cand/n_total:.1f}%)")

    stats_df = pd.DataFrame({
        "prompt_id":    prompt_ids,
        "p_safe_raw":   p_safe_raw,
        "p_safe_2pl":   p_safe_2pl,
        "in_window":    in_window,
    })
    return candidate_ids, stats_df, dict(zip(prompt_ids, b)), dict(zip(prompt_ids, a))


# ── Step 2: Agreement-rank selection ─────────────────────────────────────────

def compute_agreement_scores(mat_ref, matrices, candidate_ids,
                              ref_b_dict, ref_a_dict):
    """
    For each candidate item, compute Lord's χ²(2) against every focal language
    using all candidates as the provisional equating set.  Average the χ²
    statistic across languages to get a single "DIF agreement score" per item.

    Items with the lowest score are the most cross-linguistically invariant.

    Parameters
    ----------
    ref_b_dict, ref_a_dict : pre-computed English 2PL parameters (avoids refitting)
    """
    print(f"\nStep 2 — Agreement-rank scoring across {len(ALL_LANGS)} languages:")

    # Collect per-language χ² for each candidate prompt
    # chi2_matrix[prompt_id][lang] = Lord's χ² statistic
    chi2_records = {pid: {} for pid in candidate_ids}

    for lang in ALL_LANGS:
        if lang not in matrices:
            continue

        mat_foc = matrices[lang]
        common  = sorted(set(mat_ref.columns) & set(mat_foc.columns))

        if mat_foc.shape[0] < MIN_PERSONS:
            print(f"  {lang}: insufficient persons — skipping")
            continue

        # Candidate mask over common prompts
        cand_mask = np.array([pid in candidate_ids for pid in common])
        n_cand    = cand_mask.sum()
        print(f"  {lang}: {n_cand} candidates", end="", flush=True)

        # Align pre-computed English parameters to common prompts
        b_r   = np.array([ref_b_dict[p] for p in common])
        a_r   = np.array([ref_a_dict[p] for p in common])

        # Recompute English SEs on common prompt subset for Lord's statistic
        X_ref_common = mat_ref[common].values.astype(float)
        _, _, se_a_r, se_b_r, cov_ab_r = fit_2pl(X_ref_common)
        # Overwrite a/b with pre-computed values for consistency
        b_r = np.array([ref_b_dict[p] for p in common])
        a_r = np.array([ref_a_dict[p] for p in common])

        # Fit focal language
        X_foc = mat_foc[common].values.astype(float)
        a_f, b_f, se_a_f, se_b_f, cov_ab_f = fit_2pl(X_foc)

        # Mean-only equating using all candidates as provisional anchor set
        A, B = mean_equating(b_r, b_f, cand_mask)
        b_f_lnk  = A * b_f + B
        se_b_f_l = abs(A) * se_b_f
        se_a_f_l = se_a_f / abs(A)
        cov_ab_fl = cov_ab_f

        # Lord's χ² for all common prompts
        stat, _ = lords_chi_square(
            a_r, b_r, se_a_r, se_b_r, cov_ab_r,
            a_f, b_f_lnk, se_a_f_l, se_b_f_l, cov_ab_fl,
        )

        # Store χ² for candidate items only
        for idx, pid in enumerate(common):
            if pid in chi2_records:
                chi2_records[pid][lang] = float(stat[idx])

        n_flagged = int(((stat > chi2.ppf(0.95, df=2)) &
                         (np.abs(b_r - b_f_lnk) > MIN_DELTA_B) &
                         cand_mask).sum())
        print(f"  →  {n_flagged} items above χ²(0.95) threshold")

    # ── Build agreement score table ───────────────────────────────────────────
    rows = []
    for pid in sorted(candidate_ids):
        lang_stats = chi2_records[pid]
        if not lang_stats:
            continue
        values = list(lang_stats.values())
        rows.append({
            "prompt_id":        pid,
            "n_languages":      len(values),
            "mean_chi2":        np.mean(values),
            "median_chi2":      np.median(values),
            "max_chi2":         np.max(values),
            **{f"chi2_{lang}": lang_stats.get(lang, np.nan) for lang in ALL_LANGS},
        })

    scores_df = pd.DataFrame(rows).sort_values("mean_chi2").reset_index(drop=True)
    scores_df["rank"]     = scores_df.index + 1
    scores_df["selected"] = scores_df["rank"] <= N_ANCHORS

    return scores_df


# ── Step 3: Soft-anchor prior output ─────────────────────────────────────────

def build_soft_anchor_priors(anchor_ids, all_prompt_ids):
    rows = []
    for pid in all_prompt_ids:
        if pid in anchor_ids:
            rows.append({"prompt_id": pid, "prior_sigma": ANCHOR_PRIOR_SIGMA,
                         "is_anchor": True})
        else:
            rows.append({"prompt_id": pid, "prior_sigma": float("nan"),
                         "is_anchor": False})
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STRATIFIED-VARIANCE + AGREEMENT-RANK ANCHOR SELECTION")
    print(f"  Variance window: P(Safe|θ=0) ∈ ({VARIANCE_LO}, {VARIANCE_HI})")
    print(f"  Languages: {', '.join(ALL_LANGS)}")
    print(f"  Target anchors: top {N_ANCHORS} by mean Lord's χ² across languages")
    print(f"  Equating: mean-only (A=1, location shift only)")
    print(f"  Anchor prior σ: {ANCHOR_PRIOR_SIGMA}")
    print("=" * 70)

    print("\nLoading data...")
    matrices = load_response_matrices()

    if REFERENCE_LANG not in matrices:
        raise ValueError(f"Reference language '{REFERENCE_LANG}' not found")

    mat_ref = matrices[REFERENCE_LANG]
    print(f"\nReference: {mat_ref.shape[0]} persons × {mat_ref.shape[1]} prompts")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    candidate_ids, filter_stats, ref_b_dict, ref_a_dict = variance_filter(mat_ref)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    scores_df = compute_agreement_scores(
        mat_ref, matrices, candidate_ids, ref_b_dict, ref_a_dict)

    selected_ids = set(scores_df.loc[scores_df["selected"], "prompt_id"].astype(str))
    print(f"\n  Selected {len(selected_ids)} anchors (top {N_ANCHORS} by mean χ²)")

    top = scores_df[scores_df["selected"]][["prompt_id", "mean_chi2", "median_chi2", "max_chi2"]]
    print(f"\n  Top 10 anchors (lowest mean χ²):")
    print(f"  {'Rank':>5} {'prompt_id':>12} {'mean χ²':>10} {'max χ²':>10}")
    print("  " + "─" * 42)
    for _, row in top.head(10).iterrows():
        print(f"  {scores_df.index[scores_df['prompt_id']==row['prompt_id']][0]+1:>5} "
              f"{row['prompt_id']:>12} {row['mean_chi2']:>10.3f} {row['max_chi2']:>10.3f}")

    # ── Per-language summary over selected anchors ────────────────────────────
    selected_df = scores_df[scores_df["selected"]]
    print(f"\n  Per-language χ² across the {N_ANCHORS} selected anchors:")
    print(f"  {'Language':<10} {'mean χ²':>10} {'max χ²':>10}")
    print("  " + "─" * 32)
    for lang in ALL_LANGS:
        col = f"chi2_{lang}"
        if col in selected_df.columns:
            print(f"  {lang:<10} {selected_df[col].mean():>10.3f} {selected_df[col].max():>10.3f}")

    # ── Step 3 ────────────────────────────────────────────────────────────────
    all_prompt_ids = sorted(mat_ref.columns.astype(str).tolist())
    priors_df = build_soft_anchor_priors(selected_ids, all_prompt_ids)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving outputs to {RESULTS_DIR}/")

    filter_stats.to_csv(
        os.path.join(RESULTS_DIR, "variance_filter_stats.csv"), index=False)
    print("  variance_filter_stats.csv")

    scores_df.to_csv(
        os.path.join(RESULTS_DIR, "dif_agreement_scores.csv"), index=False)
    print(f"  dif_agreement_scores.csv  ({len(scores_df)} candidates ranked)")

    priors_df.to_csv(
        os.path.join(RESULTS_DIR, "soft_anchor_priors.csv"), index=False)
    print(f"  soft_anchor_priors.csv    ({len(selected_ids)} anchors, σ={ANCHOR_PRIOR_SIGMA})")

    print("\nDone.")


if __name__ == "__main__":
    main()