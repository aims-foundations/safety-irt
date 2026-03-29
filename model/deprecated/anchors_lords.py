# -*- coding: utf-8 -*-
"""
Iterative Purification with Lord's Chi-Square DIF Detection
============================================================
Identifies anchor prompts (CSG ≈ 0) across all 315 prompts × 9 non-English
languages using classical 2PL IRT + iterative purification.

This is run as a preprocessing step to determine which prompts should have
tau_mask = 0 (anchors) in the main Bayesian IRT model (model/irt.py).

Algorithm (for each focal language L vs English reference):
  1. Start: all 315 prompts in anchor set
  2. Fit 2PL IRT separately for English and L (joint MLE over items, EM over persons)
  3. Link scales via mean-sigma equating on anchor items
  4. Compute Lord's χ²(2) per item — compares (a_R, b_R) vs (a_F, b_F)
  5. Flag DIF items (BH-corrected p < ALPHA_DIF)
  6. Remove flagged items from anchor set → go to 2
  7. Stop when no new items flagged OR anchor set < MIN_ANCHORS
  8. Cross-language consensus: items that survive ALL language pairs → final anchors

Mean-sigma equating on the anchor subset removes the language main effect
(analogous to γ_L in the Bayesian model), so Lord's χ² tests item-specific
invariance (analogous to τ_{iL} = 0).

Response matrix:
  Each row = (test_taker × pass) combination  →  max sample size per language
  Columns = prompt IDs (315)
  Values  = binary: 1=safe (judge_score ≥ 4), 0=unsafe

Outputs (in results_dif_purification/):
  dif_per_language.csv        — Lord's χ², p-values, DIF flag per item × language
  dif_anchors_per_language.csv — items surviving purification per language pair
  dif_consensus_anchors.csv   — items invariant across ALL language pairs
  dif_convergence.csv         — iteration trace per language pair

Usage:
  python irt_validations/dif_iterative_purification.py
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
DATA_DIR   = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_dif_purification")
os.makedirs(RESULTS_DIR, exist_ok=True)

REFERENCE_LANG = "en"
ALL_LANGS      = ["zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]

ALPHA_DIF   = 0.05   # BH-corrected significance level for flagging DIF
MAX_ITER    = 20     # maximum purification iterations per language pair
MIN_ANCHORS = 20     # stop if anchor set drops below this (identification floor)
MIN_PERSONS = 10     # minimum persons per group required to attempt calibration

EM_OUTER    = 10     # EM cycles (theta ↔ items)
EM_THETA    = 10     # Newton-Raphson steps for theta update

# Effect size gate (ETS "B-level" threshold; Zumbo, 1999)
# Items are flagged as DIF only if BOTH statistically significant (BH p < α)
# AND practically meaningful (|Δb| > MIN_DELTA_B on the linked scale).
# Without this, Lord's χ² with N=610 persons is overpowered and flags
# trivially small parameter differences as "significant."
MIN_DELTA_B = 0.5

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_response_matrices() -> dict[str, pd.DataFrame]:
    """
    Load binary response matrices from the HF dataset.

    Persons = (test_taker × pass) combinations — maximises sample size.
    This is intentional: more persons → better 2PL item parameter estimation.

    Returns
    -------
    dict : lang_code → DataFrame(index=person_key, columns=prompt_id, values ∈ {0,1,NaN})
    """
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip",
                     usecols=["id", "language", "test_taker", "judge_score", "pass"])

    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    df = df[df["judge_score"] > 0].dropna(subset=["judge_score"]).copy()
    df["binary"] = (df["judge_score"] >= 4).astype(float)

    # person key = test_taker + "_pass" + pass number
    df["person_key"] = df["test_taker"].astype(str) + "_p" + df["pass"].astype(str)
    df["id"] = df["id"].astype(str)

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


# ── 2PL EM Estimation ─────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _update_theta(X: np.ndarray, a: np.ndarray, b: np.ndarray,
                  n_steps: int = EM_THETA) -> np.ndarray:
    """
    Newton-Raphson update of person parameters θ given fixed item parameters.

    X : (N, I)  binary response matrix (NaN = missing)
    a : (I,)    discrimination (positive)
    b : (I,)    difficulty
    Returns θ : (N,)
    """
    N = X.shape[0]
    theta = np.zeros(N)
    mask = ~np.isnan(X)

    for _ in range(n_steps):
        logits = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
        P = _sigmoid(logits)                                  # (N, I)
        resid = np.where(mask, X - P, 0.0)                   # (N, I)
        grad  = (a[np.newaxis, :] * resid).sum(axis=1)       # (N,)
        info  = (a[np.newaxis, :] ** 2 * P * (1 - P) * mask).sum(axis=1)
        info  = np.maximum(info, 1e-6)
        theta += grad / info
        theta  = np.clip(theta, -4.0, 4.0)

    return theta


def _item_nll_and_grad(params: np.ndarray, X: np.ndarray,
                       theta: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Negative log-likelihood AND analytical gradient for all items given fixed θ.

    Parameterisation: params = [log_a_1, ..., log_a_I, b_1, ..., b_I]
    so a_i = exp(log_a_i) ensures positivity.

    Gradient derivation (2PL, logit = a·(θ-b), P = σ(logit), resid = x - P):

        ∂NLL/∂(log_a_i) = -Σ_n resid_{ni} · a_i · (θ_n - b_i)
        ∂NLL/∂b_i        = -Σ_n resid_{ni} · (-a_i) = +a_i · Σ_n resid_{ni}

    Providing the analytical gradient eliminates ~630 finite-difference
    evaluations that L-BFGS-B would otherwise need (2 × I parameters),
    giving roughly 100× speedup on the M-step.

    Returns
    -------
    nll  : float          — negative log-likelihood
    grad : (2I,) ndarray  — [∂NLL/∂log_a, ∂NLL/∂b]
    """
    I = X.shape[1]
    log_a = params[:I]
    b     = params[I:]
    a     = np.exp(log_a)

    diff   = theta[:, np.newaxis] - b[np.newaxis, :]       # (N, I)
    logits = a[np.newaxis, :] * diff                        # (N, I)
    P      = _sigmoid(logits)                               # (N, I)
    mask   = ~np.isnan(X)
    eps    = 1e-8

    # ── NLL ───────────────────────────────────────────────────────────────────
    ll  = np.where(mask, X * np.log(P + eps) + (1 - X) * np.log(1 - P + eps), 0.0)
    nll = -ll.sum()

    # ── Analytical gradient ───────────────────────────────────────────────────
    resid = np.where(mask, X - P, 0.0)                     # (N, I)

    # ∂NLL/∂(log_a_i) = -(∂LL/∂logit)·(∂logit/∂log_a) = -(x-P)·a·(θ-b)
    grad_log_a = -(resid * a[np.newaxis, :] * diff).sum(axis=0)     # (I,)

    # ∂NLL/∂b_i = -(∂LL/∂logit)·(∂logit/∂b) = -(x-P)·(-a) = +a·(x-P)
    grad_b = (resid * a[np.newaxis, :]).sum(axis=0)                  # (I,)

    return nll, np.concatenate([grad_log_a, grad_b])


def _item_fisher_info(a: np.ndarray, b: np.ndarray,
                      theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytical per-item Fisher information matrix for (log_a, b), fully inverted.

    The 2×2 information matrix for item i (log_a parameterisation):
        J_aa = Σ_j (a·(θ_j-b))² · P_j·Q_j
        J_bb = Σ_j  a²           · P_j·Q_j
        J_ab = -Σ_j a²·(θ_j-b)  · P_j·Q_j

    Inverse gives the full covariance matrix Cov(log_a, b).
    Delta method transforms to Cov(a, b):
        Var(a)     = a² · Var(log_a)
        Cov(a, b)  = a  · Cov(log_a, b)

    Returns
    -------
    se_a   : (I,)  SE of discrimination a
    se_b   : (I,)  SE of difficulty b
    cov_ab : (I,)  Cov(a, b) — needed for full Lord's statistic
    """
    logits = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])  # (N, I)
    P  = _sigmoid(logits)
    PQ = P * (1 - P)

    diff = theta[:, np.newaxis] - b[np.newaxis, :]   # (N, I)

    J_aa = ((a[np.newaxis, :] * diff) ** 2 * PQ).sum(axis=0)    # (I,)
    J_bb = (a[np.newaxis, :] ** 2 * PQ).sum(axis=0)             # (I,)
    J_ab = -(a[np.newaxis, :] ** 2 * diff * PQ).sum(axis=0)     # (I,)

    det = J_aa * J_bb - J_ab ** 2
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)

    # Full 2×2 inverse: Cov(log_a, b)
    var_log_a    =  J_bb / det           # (I,)
    var_b        =  J_aa / det           # (I,)
    cov_log_a_b  = -J_ab / det           # (I,)

    # Delta method: transform from (log_a, b) to (a, b)
    var_a   = a ** 2 * var_log_a         # Var(a)     = a² · Var(log_a)
    cov_ab  = a      * cov_log_a_b       # Cov(a, b)  = a  · Cov(log_a, b)

    se_a   = np.sqrt(np.maximum(var_a, 1e-8))
    se_b   = np.sqrt(np.maximum(var_b, 1e-8))

    return se_a, se_b, cov_ab


def fit_2pl(X: np.ndarray, n_em: int = EM_OUTER) -> tuple:
    """
    Fit 2PL IRT model via EM-style alternation.

    X : (N, I) binary response matrix (NaN = missing)
    Returns (a, b, se_a, se_b, cov_ab, theta)
        a, b       : (I,) item parameters
        se_a, se_b : (I,) standard errors
        cov_ab     : (I,) Cov(a, b) per item — for full Lord's statistic
        theta      : (N,) person parameters
    """
    N, I = X.shape

    # Initialise
    prop_correct = np.nanmean(X, axis=0)
    prop_correct = np.clip(prop_correct, 0.05, 0.95)
    b = -np.log(prop_correct / (1 - prop_correct))  # logit difficulty
    a = np.ones(I)
    theta = np.zeros(N)

    for em_iter in range(n_em):
        # E-step: update theta given items
        theta = _update_theta(X, a, b)
        # Normalise to (0, 1) identification constraint
        mu, sd = theta.mean(), theta.std()
        if sd < 1e-6:
            sd = 1.0
        theta = (theta - mu) / sd

        # M-step: update item params given theta via L-BFGS-B
        #         jac=True → function returns (nll, gradient) tuple
        #         Analytical gradient eliminates ~630 finite-difference evals
        params0 = np.concatenate([np.log(np.maximum(a, 0.01)), b])
        result = minimize(
            _item_nll_and_grad, params0, args=(X, theta),
            method="L-BFGS-B", jac=True,
            options={"maxiter": 300, "ftol": 1e-7, "gtol": 1e-5},
        )
        params = result.x
        a = np.exp(np.clip(params[:I], -3, 3))  # clip to [0.05, 20]
        b = np.clip(params[I:], -5, 5)

    # Final SE + covariance from analytical Fisher information
    se_a, se_b, cov_ab = _item_fisher_info(a, b, theta)

    return a, b, se_a, se_b, cov_ab, theta


# ── Scale Linking ─────────────────────────────────────────────────────────────

def mean_sigma_equating(b_ref: np.ndarray, b_foc: np.ndarray,
                         anchor_mask: np.ndarray) -> tuple[float, float]:
    """
    Mean-sigma equating: find A, B such that b_foc* = A·b_foc + B ≈ b_ref on anchors.
    (Kolen & Brennan, 2004, §4.2)

    A = SD(b_ref[anchors]) / SD(b_foc[anchors])
    B = mean(b_ref[anchors]) - A · mean(b_foc[anchors])
    """
    b_r = b_ref[anchor_mask]
    b_f = b_foc[anchor_mask]
    if len(b_r) < 2 or b_f.std() < 1e-8:
        return 1.0, 0.0
    A = b_r.std() / b_f.std()
    B = b_r.mean() - A * b_f.mean()
    return float(A), float(B)


def apply_equating(a_f: np.ndarray, b_f: np.ndarray,
                   se_a_f: np.ndarray, se_b_f: np.ndarray,
                   cov_ab_f: np.ndarray,
                   A: float, B: float) -> tuple:
    """
    Transform focal-group item parameters to the reference scale.

    Transformation (Kolen & Brennan, 2004):
        b_f* = A · b_f + B
        a_f* = a_f / A          (discrimination in inverse-theta units)

    Standard errors via first-order delta method:
        SE(b_f*) = |A| · SE(b_f)
        SE(a_f*) = SE(a_f) / |A|

    Covariance is scale-invariant:
        Cov(a/A, A·b+B) = (1/A)·A · Cov(a,b) = Cov(a,b)
    """
    a_linked      = a_f / A
    b_linked      = A * b_f + B
    se_a_linked   = se_a_f / abs(A)
    se_b_linked   = abs(A) * se_b_f
    cov_ab_linked = cov_ab_f          # invariant under linear scale transformation
    return a_linked, b_linked, se_a_linked, se_b_linked, cov_ab_linked


# ── Lord's Chi-Square ─────────────────────────────────────────────────────────

def lords_chi_square(a_r, b_r, se_a_r, se_b_r, cov_ab_r,
                     a_f, b_f, se_a_f, se_b_f, cov_ab_f) -> tuple[np.ndarray, np.ndarray]:
    """
    Lord's (1980) chi-square using the full 2×2 covariance matrix.

    For each item i, with d = (Δa, Δb) and Σ = Σ_ref + Σ_foc:

        χ²_i = d' Σ⁻¹ d  ~  χ²(2)  under H₀

    where
        Σ = [[ Var(a_R) + Var(a_F),     Cov(a_R,b_R) + Cov(a_F,b_F) ],
             [ Cov(a_R,b_R) + Cov(a_F,b_F),  Var(b_R) + Var(b_F)    ]]

    Including the off-diagonal Cov(a,b) gives the correct quadratic form
    and avoids inflating/deflating the statistic when a and b are correlated
    (which they are in 2PL — harder items tend to have higher discrimination).

    Parameters
    ----------
    All inputs : (I,) arrays for I items.

    Returns
    -------
    stat  : (I,) chi-square statistics
    pvals : (I,) p-values (upper tail of χ²(2))
    """
    # Build combined covariance matrix elements (vectorised over I items)
    S_aa = se_a_r ** 2 + se_a_f ** 2          # Var(a_R) + Var(a_F)
    S_bb = se_b_r ** 2 + se_b_f ** 2          # Var(b_R) + Var(b_F)
    S_ab = cov_ab_r + cov_ab_f                # Cov(a_R,b_R) + Cov(a_F,b_F)

    da = a_r - a_f
    db = b_r - b_f

    # det(Σ) = S_aa·S_bb - S_ab²
    det = S_aa * S_bb - S_ab ** 2
    det = np.where(np.abs(det) < 1e-10, 1e-10, det)

    # Quadratic form d' Σ⁻¹ d using analytical 2×2 inverse:
    #   Σ⁻¹ = (1/det) · [[ S_bb, -S_ab], [-S_ab,  S_aa]]
    stat = (da ** 2 * S_bb - 2.0 * da * db * S_ab + db ** 2 * S_aa) / det

    pvals = 1.0 - chi2.cdf(stat, df=2)
    return stat, pvals


# ── Iterative Purification ────────────────────────────────────────────────────

def purify_one_language(mat_ref: pd.DataFrame, mat_foc: pd.DataFrame,
                         lang_name: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Run iterative purification for one focal language vs English.

    Parameters
    ----------
    mat_ref  : reference (English) response matrix
    mat_foc  : focal language response matrix
    lang_name: language code for reporting

    Returns
    -------
    result_df    : per-item DIF statistics at final iteration
    anchor_mask  : boolean (I,) — True = item survived as anchor
    conv_df      : convergence trace (iteration, n_anchors, n_dif)
    """
    # Align to prompts present in both languages
    common_prompts = sorted(mat_ref.columns.intersection(mat_foc.columns))
    if len(common_prompts) == 0:
        print(f"  {lang_name}: no common prompts — skipping")
        return pd.DataFrame(), np.array([], dtype=bool), pd.DataFrame()

    X_ref = mat_ref[common_prompts].values.astype(float)   # (N_ref, I)
    X_foc = mat_foc[common_prompts].values.astype(float)   # (N_foc, I)
    I = len(common_prompts)

    print(f"  {lang_name}: N_ref={X_ref.shape[0]}, N_foc={X_foc.shape[0]}, I={I}")

    if X_ref.shape[0] < MIN_PERSONS or X_foc.shape[0] < MIN_PERSONS:
        print(f"  WARNING: insufficient persons for {lang_name} — skipping")
        return pd.DataFrame(), np.ones(I, dtype=bool), pd.DataFrame()

    # Initialise: ALL items as anchors
    anchor_mask = np.ones(I, dtype=bool)

    conv_rows      = []
    a_r = b_r = se_a_r = se_b_r = cov_ab_r = None
    a_f = b_f = se_a_f = se_b_f = cov_ab_f = None
    a_f_lnk = b_f_lnk = None
    stat = pvals = pvals_bh = dif_mask = None

    for iteration in range(1, MAX_ITER + 1):
        n_anchors = anchor_mask.sum()
        print(f"    iter {iteration:2d}: {n_anchors:3d} anchors", end="", flush=True)

        if n_anchors < MIN_ANCHORS:
            print(f" — anchor set < {MIN_ANCHORS}, stopping")
            break

        # ── Calibration ──────────────────────────────────────────────────────
        a_r, b_r, se_a_r, se_b_r, cov_ab_r, _ = fit_2pl(X_ref)
        a_f, b_f, se_a_f, se_b_f, cov_ab_f, _ = fit_2pl(X_foc)

        # ── Scale linking ─────────────────────────────────────────────────────
        A, B = mean_sigma_equating(b_r, b_f, anchor_mask)
        a_f_lnk, b_f_lnk, se_a_f_lnk, se_b_f_lnk, cov_ab_f_lnk = apply_equating(
            a_f, b_f, se_a_f, se_b_f, cov_ab_f, A, B
        )

        # ── Lord's chi-square (full 2×2 covariance) ───────────────────────────
        stat, pvals = lords_chi_square(a_r, b_r, se_a_r, se_b_r, cov_ab_r,
                                       a_f_lnk, b_f_lnk,
                                       se_a_f_lnk, se_b_f_lnk, cov_ab_f_lnk)

        # ── BH correction + effect size gate ─────────────────────────────────
        _, pvals_bh, _, _ = multipletests(pvals, alpha=ALPHA_DIF, method="fdr_bh")
        delta_b  = np.abs(b_r - b_f_lnk)
        dif_mask = (pvals_bh < ALPHA_DIF) & (delta_b > MIN_DELTA_B)

        n_dif = int(dif_mask.sum())
        print(f"  →  {n_dif:3d} DIF items")

        conv_rows.append({
            "iteration":        iteration,
            "n_anchors_before": int(n_anchors),
            "n_dif_flagged":    n_dif,
        })

        new_anchor_mask = anchor_mask & ~dif_mask

        # ── Convergence check ─────────────────────────────────────────────────
        if np.array_equal(new_anchor_mask, anchor_mask):
            print(f"    Converged at iteration {iteration}")
            break

        anchor_mask = new_anchor_mask

    # ── Build result table ────────────────────────────────────────────────────
    result_df = pd.DataFrame({
        "prompt_id":     common_prompts,
        "language":      lang_name,
        "a_ref":         a_r,
        "b_ref":         b_r,
        "se_a_ref":      se_a_r,
        "se_b_ref":      se_b_r,
        "a_foc_linked":  a_f_lnk  if a_f_lnk  is not None else np.nan,
        "b_foc_linked":  b_f_lnk  if b_f_lnk  is not None else np.nan,
        "delta_b":       np.abs(b_r - b_f_lnk) if b_f_lnk is not None else np.nan,
        "lords_chi2":    stat     if stat     is not None else np.nan,
        "p_raw":         pvals    if pvals    is not None else np.nan,
        "p_bh":          pvals_bh if pvals_bh is not None else np.nan,
        "is_dif":        dif_mask if dif_mask is not None else False,
        "is_anchor":     anchor_mask,
    })

    return result_df, anchor_mask, pd.DataFrame(conv_rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ITERATIVE PURIFICATION — Lord's Chi-Square DIF Detection")
    print(f"  Prompts: 315  |  Languages: {len(ALL_LANGS)}  |  "
          f"α_DIF={ALPHA_DIF} (BH)  |  |Δb|>{MIN_DELTA_B}")
    print("=" * 70)

    matrices = load_response_matrices()

    if REFERENCE_LANG not in matrices:
        raise ValueError(f"Reference language '{REFERENCE_LANG}' not in dataset")

    mat_ref = matrices[REFERENCE_LANG]
    print(f"\nReference group ({REFERENCE_LANG}): "
          f"{mat_ref.shape[0]} persons × {mat_ref.shape[1]} prompts\n")

    all_results   = []
    all_conv      = []
    anchor_sets   = {}   # lang → set of anchor prompt IDs

    for lang in ALL_LANGS:
        if lang not in matrices:
            print(f"\nSkipping {lang} (no data in dataset)")
            continue

        print(f"\n{'─' * 60}")
        print(f"English  vs  {lang.upper()}")

        result_df, anchor_mask, conv_df = purify_one_language(
            mat_ref, matrices[lang], lang_name=lang
        )

        if result_df.empty:
            continue

        all_results.append(result_df)
        conv_df["language"] = lang
        all_conv.append(conv_df)

        final_anchors = result_df.loc[result_df["is_anchor"], "prompt_id"].tolist()
        anchor_sets[lang] = set(final_anchors)
        n_dif = result_df["is_dif"].sum()
        print(f"  Final: {len(final_anchors)} anchors, {n_dif} DIF items "
              f"({n_dif / len(result_df) * 100:.1f}% DIF rate)")

    # ── Cross-language consensus ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CROSS-LANGUAGE CONSENSUS ANCHORS")
    print(f"{'=' * 70}")

    if anchor_sets:
        consensus = set.intersection(*anchor_sets.values())
        print(f"Items invariant across ALL {len(anchor_sets)} language pairs: "
              f"{len(consensus)} / {mat_ref.shape[1]}")

        # Majority-vote anchors (anchor in ≥ 50% of language pairs)
        pid_counts = Counter(pid for anchors in anchor_sets.values() for pid in anchors)
        majority_threshold = len(anchor_sets) / 2
        majority_anchors = {pid for pid, cnt in pid_counts.items()
                            if cnt >= majority_threshold}
        print(f"Items that are anchors in ≥ 50% of pairs:              "
              f"{len(majority_anchors)} / {mat_ref.shape[1]}")

        # Per-language summary
        print(f"\n{'Language':<8} {'Anchors':>8} {'DIF items':>10} {'DIF rate':>10}")
        print("─" * 42)
        for lang in ALL_LANGS:
            if lang not in anchor_sets:
                continue
            n_anch = len(anchor_sets[lang])
            total  = mat_ref.shape[1]
            n_dif_ = total - n_anch
            print(f"{lang:<8} {n_anch:>8} {n_dif_:>10} {n_dif_ / total * 100:>9.1f}%")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print(f"\nSaving outputs to {RESULTS_DIR}/")

    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "dif_per_language.csv"), index=False)
        print("  dif_per_language.csv")

    if all_conv:
        pd.concat(all_conv, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "dif_convergence.csv"), index=False)
        print("  dif_convergence.csv")

    if anchor_sets:
        # Per-language anchor table
        rows = [{"prompt_id": pid, "language": lang}
                for lang, anchors in anchor_sets.items() for pid in sorted(anchors)]
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, "dif_anchors_per_language.csv"), index=False)
        print("  dif_anchors_per_language.csv")

        # Consensus
        pd.DataFrame({"prompt_id": sorted(consensus)}).to_csv(
            os.path.join(RESULTS_DIR, "dif_consensus_anchors.csv"), index=False)
        print(f"  dif_consensus_anchors.csv  ({len(consensus)} items)")

        # Majority vote
        pd.DataFrame({"prompt_id": sorted(majority_anchors)}).to_csv(
            os.path.join(RESULTS_DIR, "dif_majority_anchors.csv"), index=False)
        print(f"  dif_majority_anchors.csv   ({len(majority_anchors)} items)")

    print("\nDone.")


if __name__ == "__main__":
    main()