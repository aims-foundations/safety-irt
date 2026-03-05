# -*- coding: utf-8 -*-
"""
Iterative Forward Anchor Selection with MTT Ranking
====================================================
Kopf, Zeileis & Strobl (2015) forward anchor construction using
Mean Test-statistic Threshold (MTT) ranking, with Lord's χ²(2) as
the DIF test statistic.

Contrast with dif_iterative_purification.py (backward elimination):
  - Backward: start with ALL items as anchors, iteratively REMOVE DIF items
  - Forward:  start with ONE item, iteratively ADD DIF-free items

The forward approach is theoretically superior when DIF prevalence is
high (as in this dataset), because backward purification starts from a
contaminated anchor set, biasing all subsequent tests.

Algorithm
---------
Phase 1 — MTT ranking (single-anchor screening):
  For each candidate item j:
    1. Use item j alone as anchor
    2. Equate reference↔focal scales on that single item
    3. Compute Lord's χ²(2) for all other items
  For each item i, collect its test statistics across all J-1 single-
  anchor runs. The MTT criterion counts how often item i's statistic
  exceeds a threshold derived from the ordered mean statistics across
  all items. Items with fewer exceedances are ranked as better (more
  likely DIF-free) anchor candidates.

Phase 2 — Iterative forward construction:
  1. Start anchor set A = {top-ranked item by MTT}
  2. Equate scales using A, test all non-anchor items for DIF
  3. Among items NOT currently flagged as DIF, add the next-best-ranked
     item (per MTT ordering) to A
  4. Repeat until |A| ≥ number of items currently deemed DIF-free,
     or no more items can be added

Phase 3 — Cross-language consensus (same as backward script):
  Items that survive as anchors in ALL language pairs → final anchors

Outputs (in results_dif_forward_mtt/):
  mtt_rankings_per_language.csv   — MTT exceedance counts per item × language
  forward_anchors_per_language.csv — items surviving forward selection per pair
  forward_consensus_anchors.csv   — items invariant across ALL language pairs
  forward_convergence.csv         — iteration trace per language pair
  anchor_method_comparison.csv    — side-by-side with backward purification

Usage:
  python irt_validations/dif_forward_anchor_mtt.py

References:
  Kopf, J., Zeileis, A., & Strobl, C. (2015). Anchor selection
    strategies for DIF analysis: Review, assessment, and new approaches.
    Educational and Psychological Measurement, 75(1), 22–56.
"""

import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from huggingface_hub import snapshot_download

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset",
                                token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_dif_forward_mtt")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to backward purification results (for comparison table)
BACKWARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results_dif_purification")

REFERENCE_LANG = "en"
ALL_LANGS      = ["zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]

ALPHA_DIF   = 0.05   # BH-corrected significance level
MAX_ITER    = 300      # max forward iterations (we add 1 item/iter, so need headroom)
MIN_ANCHORS = 1       # forward starts from 1, so floor is 1
MIN_PERSONS = 10
MIN_DELTA_B = 0.5     # effect size gate (same as backward script)

EM_OUTER = 10
EM_THETA = 10


# ══════════════════════════════════════════════════════════════════════════════
# 2PL ENGINE — identical to dif_iterative_purification.py
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _update_theta(X, a, b, n_steps=EM_THETA):
    N = X.shape[0]
    theta = np.zeros(N)
    mask = ~np.isnan(X)
    for _ in range(n_steps):
        logits = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
        P = _sigmoid(logits)
        resid = np.where(mask, X - P, 0.0)
        grad  = (a[np.newaxis, :] * resid).sum(axis=1)
        info  = (a[np.newaxis, :] ** 2 * P * (1 - P) * mask).sum(axis=1)
        info  = np.maximum(info, 1e-6)
        theta += grad / info
        theta  = np.clip(theta, -4.0, 4.0)
    return theta


def _item_nll_and_grad(params, X, theta):
    I = X.shape[1]
    log_a, b = params[:I], params[I:]
    a = np.exp(log_a)
    diff   = theta[:, np.newaxis] - b[np.newaxis, :]
    logits = a[np.newaxis, :] * diff
    P      = _sigmoid(logits)
    mask   = ~np.isnan(X)
    eps    = 1e-8
    ll  = np.where(mask, X * np.log(P + eps) + (1 - X) * np.log(1 - P + eps), 0.0)
    nll = -ll.sum()
    resid = np.where(mask, X - P, 0.0)
    grad_log_a = -(resid * a[np.newaxis, :] * diff).sum(axis=0)
    grad_b     =  (resid * a[np.newaxis, :]).sum(axis=0)
    return nll, np.concatenate([grad_log_a, grad_b])


def _item_fisher_info(a, b, theta):
    logits = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
    P  = _sigmoid(logits)
    PQ = P * (1 - P)
    diff = theta[:, np.newaxis] - b[np.newaxis, :]
    J_aa = ((a[np.newaxis, :] * diff) ** 2 * PQ).sum(axis=0)
    J_bb = (a[np.newaxis, :] ** 2 * PQ).sum(axis=0)
    J_ab = -(a[np.newaxis, :] ** 2 * diff * PQ).sum(axis=0)
    det = J_aa * J_bb - J_ab ** 2
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)
    var_log_a   =  J_bb / det
    var_b       =  J_aa / det
    cov_log_a_b = -J_ab / det
    var_a  = a ** 2 * var_log_a
    cov_ab = a * cov_log_a_b
    se_a   = np.sqrt(np.maximum(var_a, 1e-8))
    se_b   = np.sqrt(np.maximum(var_b, 1e-8))
    return se_a, se_b, cov_ab


def fit_2pl(X, n_em=EM_OUTER):
    N, I = X.shape
    prop_correct = np.clip(np.nanmean(X, axis=0), 0.05, 0.95)
    b = -np.log(prop_correct / (1 - prop_correct))
    a = np.ones(I)
    theta = np.zeros(N)
    for _ in range(n_em):
        theta = _update_theta(X, a, b)
        mu, sd = theta.mean(), max(theta.std(), 1e-6)
        theta = (theta - mu) / sd
        params0 = np.concatenate([np.log(np.maximum(a, 0.01)), b])
        result = minimize(_item_nll_and_grad, params0, args=(X, theta),
                          method="L-BFGS-B", jac=True,
                          options={"maxiter": 300, "ftol": 1e-7, "gtol": 1e-5})
        params = result.x
        a = np.exp(np.clip(params[:I], -3, 3))
        b = np.clip(params[I:], -5, 5)
    se_a, se_b, cov_ab = _item_fisher_info(a, b, theta)
    return a, b, se_a, se_b, cov_ab, theta


def mean_sigma_equating(b_ref, b_foc, anchor_mask):
    b_r, b_f = b_ref[anchor_mask], b_foc[anchor_mask]
    if len(b_r) < 2 or b_f.std() < 1e-8:
        return 1.0, 0.0
    A = b_r.std() / b_f.std()
    B = b_r.mean() - A * b_f.mean()
    return float(A), float(B)


def mean_sigma_equating_single(b_ref_j, b_foc_j):
    """Single-item equating: only shift, no rescaling (A=1, B=b_ref - b_foc)."""
    return 1.0, float(b_ref_j - b_foc_j)


def apply_equating(a_f, b_f, se_a_f, se_b_f, cov_ab_f, A, B):
    a_linked      = a_f / A
    b_linked      = A * b_f + B
    se_a_linked   = se_a_f / abs(A)
    se_b_linked   = abs(A) * se_b_f
    cov_ab_linked = cov_ab_f
    return a_linked, b_linked, se_a_linked, se_b_linked, cov_ab_linked


def lords_chi_square(a_r, b_r, se_a_r, se_b_r, cov_ab_r,
                     a_f, b_f, se_a_f, se_b_f, cov_ab_f):
    S_aa = se_a_r ** 2 + se_a_f ** 2
    S_bb = se_b_r ** 2 + se_b_f ** 2
    S_ab = cov_ab_r + cov_ab_f
    da, db = a_r - a_f, b_r - b_f
    det = S_aa * S_bb - S_ab ** 2
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)
    stat = (da ** 2 * S_bb - 2.0 * da * db * S_ab + db ** 2 * S_aa) / det
    pvals = 1.0 - chi2.cdf(stat, df=2)
    return stat, pvals


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_response_matrices() -> dict:
    print(f"Loading {INPUT_FILE} ...")
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
        mat = sub.pivot_table(index="person_key", columns="id",
                               values="binary", aggfunc="first")
        matrices[lang] = mat
        print(f"  {lang}: {mat.shape[0]} persons × {mat.shape[1]} prompts")
    return matrices


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: MTT RANKING
# ══════════════════════════════════════════════════════════════════════════════

def compute_mtt_ranking(prompt_ids, lang_name, params):
    a_r, b_r, se_a_r, se_b_r, cov_ab_r, a_f, b_f, se_a_f, se_b_f, cov_ab_f = params
    """
    Mean Test-statistic Threshold (MTT) ranking.

    For each item j used as single-item anchor:
      - Equate scales using only item j (shift-only: A=1, B=b_ref_j - b_foc_j)
      - Compute Lord's χ² for all other items

    For each item i, collect its I-1 test statistics (one per single-anchor run,
    excluding runs where i is the anchor). Then:
      1. Compute the ordered mean test statistics across all items
      2. Set threshold = value at the ⌈I/2⌉-th position of the ordered means
      3. Count how often item i's statistics exceed this threshold

    Items with FEWER exceedances → ranked as better anchor candidates.

    Returns
    -------
    ranking : list of (prompt_idx, exceedance_count) sorted ascending by count
    stat_matrix : (I, I) array — stat_matrix[j, i] = Lord's χ² for item i
                  when item j is the single anchor (diagonal = NaN)
    """
    I = len(prompt_ids)
    print(f"\n  MTT ranking for {lang_name}: {I} items, {I} single-anchor runs")

    # ── Calibrate both groups once (parameters are fixed; only equating changes) ──
    # a_r, b_r, se_a_r, se_b_r, cov_ab_r, _ = fit_2pl(X_ref)
    # a_f, b_f, se_a_f, se_b_f, cov_ab_f, _ = fit_2pl(X_foc)

    # ── Single-anchor runs ────────────────────────────────────────────────────
    stat_matrix = np.full((I, I), np.nan)

    for j in range(I):
        # Equate using only item j
        A, B = mean_sigma_equating_single(b_r[j], b_f[j])
        a_f_lnk, b_f_lnk, se_a_f_lnk, se_b_f_lnk, cov_ab_f_lnk = apply_equating(
            a_f, b_f, se_a_f, se_b_f, cov_ab_f, A, B
        )

        # Lord's χ² for all items (anchor item's own test is not meaningful)
        stat_j, _ = lords_chi_square(
            a_r, b_r, se_a_r, se_b_r, cov_ab_r,
            a_f_lnk, b_f_lnk, se_a_f_lnk, se_b_f_lnk, cov_ab_f_lnk
        )
        stat_j[j] = np.nan  # exclude self
        stat_matrix[j, :] = stat_j

    # ── MTT criterion ─────────────────────────────────────────────────────────
    # For each item i: mean of its test statistics across all single-anchor runs
    item_means = np.nanmean(stat_matrix, axis=0)  # (I,) — average across anchors

    # Threshold: median of the ordered mean statistics
    # (Kopf et al. use the ⌈I/2⌉-th ordered value, which is the median)
    sorted_means = np.sort(item_means)
    threshold = sorted_means[len(sorted_means) // 2]

    # For each item i: count exceedances across single-anchor runs
    exceedance_counts = np.zeros(I, dtype=int)
    for i in range(I):
        stats_for_i = stat_matrix[:, i]  # (I,) — one per anchor item j
        valid = ~np.isnan(stats_for_i)
        exceedance_counts[i] = int((stats_for_i[valid] > threshold).sum())

    # Rank: fewer exceedances = better anchor candidate
    # Break ties using mean statistic (lower = better)
    ranking = sorted(range(I), key=lambda i: (exceedance_counts[i], item_means[i]))

    print(f"    MTT threshold: {threshold:.2f}")
    print(f"    Exceedance range: [{exceedance_counts.min()}, {exceedance_counts.max()}]")
    print(f"    Top-5 anchor candidates: "
          f"{[prompt_ids[r] for r in ranking[:5]]}")

    return ranking, exceedance_counts, item_means, stat_matrix


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: ITERATIVE FORWARD CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def forward_anchor_selection(prompt_ids, mtt_ranking, lang_name, params):
    a_r, b_r, se_a_r, se_b_r, cov_ab_r, a_f, b_f, se_a_f, se_b_f, cov_ab_f = params
    
    """
    Iterative forward anchor construction (Kopf et al., 2015, §3.3).

    1. Start A = {best-ranked item by MTT}
    2. Equate on A, test all items for DIF (BH + effect size gate)
    3. Count n_free = items NOT flagged as DIF
    4. If |A| < n_free: add the next-best MTT-ranked item that is
       currently DIF-free to A, go to 2
    5. Stop when |A| ≥ n_free (anchor set is at least as large as
       the DIF-free set) or no more items can be added

    Returns
    -------
    anchor_mask  : (I,) boolean — True = item in final anchor set
    dif_mask     : (I,) boolean — True = item flagged as DIF at final iteration
    conv_df      : convergence trace
    final_stats  : dict with Lord's χ², p-values, etc. at final iteration
    """
    I = len(prompt_ids)
    print(f"\n  Forward construction for {lang_name}:")

    # ── Calibrate both groups (fixed throughout — only equating anchor changes) ──
    # a_r, b_r, se_a_r, se_b_r, cov_ab_r, _ = fit_2pl(X_ref)
    # a_f, b_f, se_a_f, se_b_f, cov_ab_f, _ = fit_2pl(X_foc)

    # ── Initialise: anchor = single best-ranked item ──────────────────────────
    anchor_indices = {mtt_ranking[0]}
    mtt_queue = list(mtt_ranking[1:])  # remaining items in MTT order

    conv_rows = []
    stat = pvals = pvals_bh = delta_b = dif_mask = None

    for iteration in range(1, MAX_ITER + 1):
        anchor_mask = np.zeros(I, dtype=bool)
        anchor_mask[list(anchor_indices)] = True
        n_anchors = len(anchor_indices)

        # ── Equate on current anchor set ──────────────────────────────────────
        if n_anchors == 1:
            j = list(anchor_indices)[0]
            A, B = mean_sigma_equating_single(b_r[j], b_f[j])
        else:
            A, B = mean_sigma_equating(b_r, b_f, anchor_mask)

        a_f_lnk, b_f_lnk, se_a_f_lnk, se_b_f_lnk, cov_ab_f_lnk = apply_equating(
            a_f, b_f, se_a_f, se_b_f, cov_ab_f, A, B
        )

        # ── Lord's χ² + BH + effect size gate ────────────────────────────────
        stat, pvals = lords_chi_square(
            a_r, b_r, se_a_r, se_b_r, cov_ab_r,
            a_f_lnk, b_f_lnk, se_a_f_lnk, se_b_f_lnk, cov_ab_f_lnk
        )
        _, pvals_bh, _, _ = multipletests(pvals, alpha=ALPHA_DIF, method="fdr_bh")
        delta_b = np.abs(b_r - b_f_lnk)
        dif_mask = (pvals_bh < ALPHA_DIF) & (delta_b > MIN_DELTA_B)

        n_dif  = int(dif_mask.sum())
        n_free = I - n_dif  # items currently deemed DIF-free

        conv_rows.append({
            "iteration":  iteration,
            "n_anchors":  n_anchors,
            "n_dif":      n_dif,
            "n_free":     n_free,
        })

        print(f"    iter {iteration:2d}: {n_anchors:3d} anchors, "
              f"{n_dif:3d} DIF, {n_free:3d} free", end="")

        # ── Stopping rule: |A| ≥ n_free ──────────────────────────────────────
        if n_anchors >= n_free:
            print(f"  → converged (|A| ≥ n_free)")
            break

        # ── Add next-best MTT-ranked item that is currently DIF-free ─────────
        added = False
        while mtt_queue:
            candidate = mtt_queue.pop(0)
            if not dif_mask[candidate]:
                anchor_indices.add(candidate)
                print(f"  + item {prompt_ids[candidate]}")
                added = True
                break

        if not added:
            # All remaining MTT candidates are currently DIF — stop
            print(f"  → no more DIF-free candidates in MTT queue")
            break

    # ── Final anchor mask ─────────────────────────────────────────────────────
    # The anchor set is the items we've been growing.
    # But we should also verify: any item currently DIF-free could be an anchor.
    # The Kopf et al. criterion says stop when |A| ≥ n_free, meaning the
    # anchor set has grown to include all reliably DIF-free items.
    final_anchor = np.zeros(I, dtype=bool)
    final_anchor[list(anchor_indices)] = True

    n_final = final_anchor.sum()
    n_dif_final = dif_mask.sum() if dif_mask is not None else 0
    print(f"  Final: {n_final} anchors, {n_dif_final} DIF items")

    final_stats = {
        "a_ref": a_r, "b_ref": b_r, "se_a_ref": se_a_r, "se_b_ref": se_b_r,
        "a_foc_linked": a_f_lnk, "b_foc_linked": b_f_lnk,
        "delta_b": delta_b, "lords_chi2": stat,
        "p_raw": pvals, "p_bh": pvals_bh, "dif_mask": dif_mask,
    }

    return final_anchor, dif_mask, pd.DataFrame(conv_rows), final_stats


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE FOR ONE LANGUAGE PAIR
# ══════════════════════════════════════════════════════════════════════════════

def process_one_language(mat_ref, mat_foc, lang_name):
    """Run MTT ranking + forward construction for one focal language vs English."""

    common_prompts = sorted(mat_ref.columns.intersection(mat_foc.columns))
    if len(common_prompts) == 0:
        print(f"  {lang_name}: no common prompts — skipping")
        return None

    X_ref = mat_ref[common_prompts].values.astype(float)
    X_foc = mat_foc[common_prompts].values.astype(float)
    I = len(common_prompts)

    print(f"  {lang_name}: N_ref={X_ref.shape[0]}, N_foc={X_foc.shape[0]}, I={I}")

    if X_ref.shape[0] < MIN_PERSONS or X_foc.shape[0] < MIN_PERSONS:
        print(f"  WARNING: insufficient persons — skipping")
        return None

    # Fit ONCE for both phases
    a_r, b_r, se_a_r, se_b_r, cov_ab_r, _ = fit_2pl(X_ref)
    a_f, b_f, se_a_f, se_b_f, cov_ab_f, _ = fit_2pl(X_foc)
    params = (a_r, b_r, se_a_r, se_b_r, cov_ab_r,
              a_f, b_f, se_a_f, se_b_f, cov_ab_f)

    # Phase 1: MTT ranking
    mtt_ranking, exceedance_counts, item_means, _ = compute_mtt_ranking(
        common_prompts, lang_name, params
    )

    # Phase 2: Forward construction
    anchor_mask, dif_mask, conv_df, final_stats = forward_anchor_selection(
        common_prompts, mtt_ranking, lang_name, params
    )

    # Build result table
    result_df = pd.DataFrame({
        "prompt_id":          common_prompts,
        "language":           lang_name,
        "mtt_exceedances":    exceedance_counts,
        "mtt_mean_stat":      item_means,
        "mtt_rank":           [mtt_ranking.index(i) + 1 for i in range(I)],
        "a_ref":              final_stats["a_ref"],
        "b_ref":              final_stats["b_ref"],
        "a_foc_linked":       final_stats["a_foc_linked"],
        "b_foc_linked":       final_stats["b_foc_linked"],
        "delta_b":            final_stats["delta_b"],
        "lords_chi2":         final_stats["lords_chi2"],
        "p_raw":              final_stats["p_raw"],
        "p_bh":               final_stats["p_bh"],
        "is_dif":             final_stats["dif_mask"],
        "is_anchor":          anchor_mask,
    })

    conv_df["language"] = lang_name

    return {
        "result_df":  result_df,
        "anchor_set": set(result_df.loc[result_df["is_anchor"], "prompt_id"]),
        "conv_df":    conv_df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON WITH BACKWARD PURIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def compare_with_backward(forward_anchor_sets, backward_dir):
    """Compare forward MTT anchors with backward Lord's χ² anchors."""

    backward_file = os.path.join(backward_dir, "dif_anchors_per_language.csv")
    if not os.path.exists(backward_file):
        print(f"\n  Backward results not found at {backward_file} — skipping comparison")
        return None

    backward_df = pd.read_csv(backward_file)
    backward_sets = {}
    for lang, grp in backward_df.groupby("language"):
        backward_sets[lang] = set(grp["prompt_id"].astype(str))

    rows = []
    for lang in sorted(forward_anchor_sets.keys()):
        fwd = forward_anchor_sets[lang]
        bwd = backward_sets.get(lang, set())

        overlap    = fwd & bwd
        fwd_only   = fwd - bwd
        bwd_only   = bwd - fwd
        jaccard    = len(overlap) / max(len(fwd | bwd), 1)

        rows.append({
            "language":         lang,
            "n_forward":        len(fwd),
            "n_backward":       len(bwd),
            "n_overlap":        len(overlap),
            "n_forward_only":   len(fwd_only),
            "n_backward_only":  len(bwd_only),
            "jaccard_index":    round(jaccard, 3),
        })

    comp_df = pd.DataFrame(rows)
    return comp_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ITERATIVE FORWARD ANCHOR SELECTION — MTT Ranking")
    print(f"  Kopf, Zeileis & Strobl (2015)")
    print(f"  Prompts: 315  |  Languages: {len(ALL_LANGS)}  |  "
          f"α_DIF={ALPHA_DIF} (BH)  |  |Δb|>{MIN_DELTA_B}")
    print("=" * 70)

    matrices = load_response_matrices()

    if REFERENCE_LANG not in matrices:
        raise ValueError(f"Reference language '{REFERENCE_LANG}' not in dataset")

    mat_ref = matrices[REFERENCE_LANG]
    print(f"\nReference group ({REFERENCE_LANG}): "
          f"{mat_ref.shape[0]} persons × {mat_ref.shape[1]} prompts\n")

    all_results    = []
    all_conv       = []
    anchor_sets    = {}

    for lang in ALL_LANGS:
        if lang not in matrices:
            print(f"\nSkipping {lang} (no data)")
            continue

        print(f"\n{'─' * 60}")
        print(f"English  vs  {lang.upper()}")

        out = process_one_language(mat_ref, matrices[lang], lang)
        if out is None:
            continue

        all_results.append(out["result_df"])
        all_conv.append(out["conv_df"])
        anchor_sets[lang] = out["anchor_set"]

        n_anchors = len(out["anchor_set"])
        n_dif = out["result_df"]["is_dif"].sum()
        print(f"  Summary: {n_anchors} anchors, {n_dif} DIF items "
              f"({n_dif / len(out['result_df']) * 100:.1f}% DIF rate)")

    # ── Cross-language consensus ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CROSS-LANGUAGE CONSENSUS ANCHORS (Forward MTT)")
    print(f"{'=' * 70}")

    if anchor_sets:
        consensus = set.intersection(*anchor_sets.values())
        print(f"Items invariant across ALL {len(anchor_sets)} language pairs: "
              f"{len(consensus)} / {mat_ref.shape[1]}")

        pid_counts = Counter(pid for anch in anchor_sets.values() for pid in anch)
        majority_threshold = len(anchor_sets) / 2
        majority_anchors = {pid for pid, cnt in pid_counts.items()
                            if cnt >= majority_threshold}
        print(f"Items anchors in ≥ 50% of pairs:                       "
              f"{len(majority_anchors)} / {mat_ref.shape[1]}")

        print(f"\n{'Language':<8} {'Anchors':>8} {'DIF items':>10} {'DIF rate':>10}")
        print("─" * 42)
        for lang in ALL_LANGS:
            if lang not in anchor_sets:
                continue
            n_anch = len(anchor_sets[lang])
            total  = mat_ref.shape[1]
            n_dif_ = total - n_anch
            print(f"{lang:<8} {n_anch:>8} {n_dif_:>10} {n_dif_ / total * 100:>9.1f}%")

    # ── Comparison with backward purification ─────────────────────────────────
    comp_df = compare_with_backward(anchor_sets, BACKWARD_DIR)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\nSaving outputs to {RESULTS_DIR}/")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(
            os.path.join(RESULTS_DIR, "mtt_rankings_per_language.csv"), index=False)
        print("  mtt_rankings_per_language.csv")

    if all_conv:
        pd.concat(all_conv, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "forward_convergence.csv"), index=False)
        print("  forward_convergence.csv")

    if anchor_sets:
        rows = [{"prompt_id": pid, "language": lang}
                for lang, anch in anchor_sets.items() for pid in sorted(anch)]
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, "forward_anchors_per_language.csv"), index=False)
        print("  forward_anchors_per_language.csv")

        pd.DataFrame({"prompt_id": sorted(consensus)}).to_csv(
            os.path.join(RESULTS_DIR, "forward_consensus_anchors.csv"), index=False)
        print(f"  forward_consensus_anchors.csv  ({len(consensus)} items)")

        pd.DataFrame({"prompt_id": sorted(majority_anchors)}).to_csv(
            os.path.join(RESULTS_DIR, "forward_majority_anchors.csv"), index=False)
        print(f"  forward_majority_anchors.csv   ({len(majority_anchors)} items)")

    if comp_df is not None:
        comp_df.to_csv(
            os.path.join(RESULTS_DIR, "anchor_method_comparison.csv"), index=False)
        print("  anchor_method_comparison.csv")
        print(f"\n{'=' * 70}")
        print("BACKWARD vs FORWARD COMPARISON")
        print(f"{'=' * 70}")
        print(comp_df.to_string(index=False))

        # Overall Jaccard
        if anchor_sets:
            fwd_all = set.union(*anchor_sets.values()) if anchor_sets else set()
            bwd_file = os.path.join(BACKWARD_DIR, "dif_anchors_per_language.csv")
            if os.path.exists(bwd_file):
                bdf = pd.read_csv(bwd_file)
                bwd_all = set(bdf["prompt_id"].astype(str))
                overlap = fwd_all & bwd_all
                j = len(overlap) / max(len(fwd_all | bwd_all), 1)
                print(f"\n  Overall (union across langs): "
                      f"Jaccard = {j:.3f}, "
                      f"overlap = {len(overlap)}, "
                      f"forward-only = {len(fwd_all - bwd_all)}, "
                      f"backward-only = {len(bwd_all - fwd_all)}")

    print("\nDone.")


if __name__ == "__main__":
    main()