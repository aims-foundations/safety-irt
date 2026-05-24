# -*- coding: utf-8 -*-
"""
IRT vs Non-IRT Baseline Comparison
====================================
Addresses reviewer concern that framework justification should include
comparison against non-IRT latent-variable decompositions.

Three baselines compared against the paper's 2PL IRT with τ (DIF):

  1. Bradley-Terry (BT)
     Fit logistic regression with model + item×language fixed effects.
     BT ability estimates ≡ model intercepts; compare to IRT θ via Spearman ρ.
     Shows: BT can rank models but cannot decompose γ, τ, β, α.

  2. No-DIF mixed logistic regression (GLMM-style)
     Fit logistic regression: safe ~ model + prompt + language
     (no prompt×language interaction, i.e. τ=0 for all items).
     Compare log-loss to IRT predictions.
     Shows: ignoring DIF (τ) inflates prediction error.

  3. IRT ↔ GLMM equivalence note
     Mathematical note (no code needed): 1PL IRT = Rasch = GLMM with
     random item and person intercepts (De Boeck et al., 2011).
     2PL extends this with random slopes. Our model IS a GLMM.

Outputs (irt_validations/results_irt_vs_baselines/):
  bt_vs_irt_theta.csv           — BT ability vs IRT θ per test-taker
  nodif_vs_irt_logloss.csv      — log-loss comparison table
  bt_scatter.pdf/png            — BT ability vs IRT θ scatter
  baselines_summary.txt         — plain-text summary for rebuttal

Usage:
  python irt_vs_baselines.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

try:
    from fig_style import apply_style, savefig, FULL_WIDTH, C_RED, C_BLUE, C_PURPLE
    apply_style()
except ImportError:
    C_RED, C_BLUE, C_PURPLE = "#c0392b", "#2471a3", "#7d3c98"
    FULL_WIDTH = 5.5
    def savefig(fig, path, **kw):
        fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)

try:
    from huggingface_hub import snapshot_download
    DATA_DIR = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
except Exception:
    DATA_DIR = "."

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.join(SCRIPT_DIR, "..")
IRT_CSV    = os.path.join(REPO_ROOT, "model", "results", "bayesian_irt_results_binary.csv")
THETA_CSV  = os.path.join(DATA_DIR, "results", "results_jsr_theta_posthoc", "1_jsr_vs_theta_all_models.csv")
RAW_CSV    = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "results_irt_vs_baselines")
os.makedirs(OUT_DIR, exist_ok=True)

SAFE_THRESHOLD = 4.0   # judge_score >= 4 → safe


# ── Data loading ──────────────────────────────────────────────────────────────
def load_raw(max_pass=None):
    """Load raw response data, aggregate 10 passes to binary majority."""
    df = pd.read_csv(RAW_CSV)
    df = df.dropna(subset=["judge_score"])
    df["safe"] = (df["judge_score"] >= SAFE_THRESHOLD).astype(int)
    df["item_lang"] = df["id"].astype(str) + "_" + df["language"]
    # Aggregate per model × item × language (mean over passes, then threshold)
    agg = df.groupby(["test_taker", "model", "id", "language", "item_lang"])["safe"].mean().reset_index()
    agg["safe_bin"] = (agg["safe"] >= 0.5).astype(int)
    agg["id"] = agg["id"].astype(str)
    return agg


def load_irt_params():
    irt = pd.read_csv(IRT_CSV)
    irt = irt.rename(columns={"prompt": "id", "Safety_Tax": "tau",
                               "Base_Difficulty": "beta", "gamma_L": "gamma"})
    irt["id"] = irt["id"].astype(str)
    return irt


def load_theta():
    theta = pd.read_csv(THETA_CSV)
    # Use 2PL entries if available, else 1PL
    t2 = theta[theta["irt_model"] == "2PL"]
    if len(t2) == 0:
        t2 = theta
    return t2[["test_taker", "theta", "JSR"]].copy()


# ── 1. Bradley-Terry baseline ─────────────────────────────────────────────────
def fit_bradley_terry(agg):
    """
    Logistic regression: safe_bin ~ model_dummies + item_lang_dummies
    Model intercepts relative to first model = BT abilities.
    """
    print("Fitting Bradley-Terry (logistic regression with model + item-lang FE)...")
    X_model = pd.get_dummies(agg["test_taker"], drop_first=True, prefix="m")
    X_item  = pd.get_dummies(agg["item_lang"],  drop_first=True, prefix="il")
    X = pd.concat([X_model, X_item], axis=1).astype(np.float32)
    y = agg["safe_bin"].values

    lr = LogisticRegression(C=1e4, max_iter=300, solver="saga", n_jobs=-1, verbose=0)
    lr.fit(X, y)

    # Extract model coefficients
    model_cols = [c for c in X.columns if c.startswith("m_")]
    coef_map   = dict(zip(X.columns, lr.coef_[0]))
    ref_taker  = agg["test_taker"].unique()[0]

    rows = []
    for col in model_cols:
        taker = col[len("m_"):]
        rows.append({"test_taker": taker, "bt_ability": coef_map.get(col, np.nan)})
    # Reference model has BT ability = 0
    rows.append({"test_taker": ref_taker, "bt_ability": 0.0})

    bt_df = pd.DataFrame(rows)
    print(f"  BT fit complete. {len(bt_df)} model entries.")
    return bt_df, lr, X.columns.tolist()


# ── 2. No-DIF logistic regression ────────────────────────────────────────────
def fit_nodif(agg):
    """
    safe ~ model + prompt_id + language  (no prompt×language interaction → τ=0)
    """
    print("Fitting no-DIF logistic regression (model + prompt + language FE)...")
    X_model  = pd.get_dummies(agg["test_taker"], drop_first=True, prefix="m")
    X_prompt = pd.get_dummies(agg["id"].astype(str), drop_first=True, prefix="p")
    X_lang   = pd.get_dummies(agg["language"],   drop_first=True, prefix="l")
    X = pd.concat([X_model, X_prompt, X_lang], axis=1).astype(np.float32)
    y = agg["safe_bin"].values

    lr = LogisticRegression(C=1e4, max_iter=300, solver="saga", n_jobs=-1, verbose=0)
    lr.fit(X, y)
    proba_nodif = lr.predict_proba(X)[:, 1]
    ll_nodif = log_loss(y, proba_nodif)
    print(f"  No-DIF log-loss = {ll_nodif:.4f}")
    return lr, proba_nodif, ll_nodif


# ── 3. IRT predicted probabilities ───────────────────────────────────────────
def irt_predictions(agg, irt, theta_df):
    """
    Compute IRT P(safe) = σ(α[(θ_j + δ) − (β_i + γ_L + τ_iL)])
    We ignore δ_jL (model-language aptitude) since it's not in the output CSV.
    Using θ from the 2PL results file, β, γ, τ, α from IRT CSV.
    """
    irt_sub = irt[["id", "language", "tau", "beta", "gamma", "alpha"]].copy()
    irt_sub["id"] = irt_sub["id"].astype(str)

    # Merge IRT params into agg
    merged = agg.merge(irt_sub, on=["id", "language"], how="inner")
    # Map theta
    merged = merged.merge(theta_df[["test_taker", "theta"]], on="test_taker", how="inner")

    # IRT logit: α*(θ - β - γ - τ)
    logit = merged["alpha"] * (merged["theta"] - merged["beta"] - merged["gamma"] - merged["tau"])
    proba_irt = 1 / (1 + np.exp(-logit.clip(-20, 20)))

    y = merged["safe_bin"].values
    ll_irt = log_loss(y, proba_irt.clip(1e-7, 1 - 1e-7))
    print(f"  IRT log-loss     = {ll_irt:.4f}  (n={len(y)})")
    return merged, proba_irt, ll_irt, y


# ── Plot: BT ability vs IRT θ ─────────────────────────────────────────────────
def plot_bt_vs_theta(merged_bt, rho, p):
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.55, FULL_WIDTH * 0.55))
    ax.scatter(merged_bt["bt_ability"], merged_bt["theta"],
               s=20, color=C_BLUE, alpha=0.8, linewidths=0)
    ax.set_xlabel("Bradley-Terry ability (log-odds)", fontsize=8)
    ax.set_ylabel(r"IRT $\theta_j$", fontsize=8)
    ax.set_title(f"BT ability vs IRT θ\nSpearman ρ = {rho:.3f} (p = {p:.3e})", fontsize=8)
    fig.tight_layout()
    savefig(fig, os.path.join(OUT_DIR, "bt_scatter"))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    agg      = load_raw()
    irt      = load_irt_params()
    theta_df = load_theta()
    print(f"  {len(agg)} model×item×language rows; {agg['test_taker'].nunique()} test-takers.")

    # 1. Bradley-Terry
    bt_df, bt_lr, bt_cols = fit_bradley_terry(agg)
    # Merge with IRT theta
    merged_bt = bt_df.merge(theta_df, on="test_taker", how="inner")
    rho, p = spearmanr(merged_bt["bt_ability"], merged_bt["theta"])
    print(f"\nBT ability vs IRT θ: Spearman ρ = {rho:.4f}, p = {p:.4e}, n = {len(merged_bt)}")
    merged_bt.to_csv(os.path.join(OUT_DIR, "bt_vs_irt_theta.csv"), index=False)
    plot_bt_vs_theta(merged_bt, rho, p)

    # 2. No-DIF logistic regression
    nodif_lr, proba_nodif, ll_nodif = fit_nodif(agg)

    # 3. IRT predictions (subset matched to theta file)
    merged_irt, proba_irt, ll_irt, y_irt = irt_predictions(agg, irt, theta_df)

    # Compare on same subset
    agg_sub = agg[agg["test_taker"].isin(theta_df["test_taker"])].copy()
    agg_sub["id"] = agg_sub["id"].astype(str)
    agg_sub = agg_sub.merge(irt[["id", "language", "tau", "beta", "gamma", "alpha"]],
                             on=["id", "language"], how="inner")

    X_nd_model  = pd.get_dummies(agg_sub["test_taker"], drop_first=True, prefix="m")
    X_nd_prompt = pd.get_dummies(agg_sub["id"], drop_first=True, prefix="p")
    X_nd_lang   = pd.get_dummies(agg_sub["language"], drop_first=True, prefix="l")
    X_nd = pd.concat([X_nd_model, X_nd_prompt, X_nd_lang], axis=1).astype(np.float32)
    # Re-fit no-DIF on same subset
    from sklearn.linear_model import LogisticRegression as LR
    nd2 = LR(C=1e4, max_iter=300, solver="saga", n_jobs=-1)
    nd2.fit(X_nd, agg_sub["safe_bin"].values)
    proba_nd2 = nd2.predict_proba(X_nd)[:, 1]
    ll_nodif_sub = log_loss(agg_sub["safe_bin"].values, proba_nd2)

    # Baseline: always-safe
    base_rate = agg_sub["safe_bin"].mean()
    ll_baseline = log_loss(agg_sub["safe_bin"].values,
                           np.full(len(agg_sub), base_rate))

    logloss_df = pd.DataFrame([
        {"model": "Intercept-only (base rate)", "log_loss": round(ll_baseline, 4)},
        {"model": "No-DIF logistic (model+prompt+language)", "log_loss": round(ll_nodif_sub, 4)},
        {"model": "2PL IRT with τ (DIF)", "log_loss": round(ll_irt, 4)},
    ])
    logloss_df.to_csv(os.path.join(OUT_DIR, "nodif_vs_irt_logloss.csv"), index=False)

    print("\n=== Log-loss comparison ===")
    print(logloss_df.to_string(index=False))

    # Summarise
    lines = [
        "=== IRT vs Baselines Summary ===\n",
        f"Test-takers: {agg['test_taker'].nunique()} | Items: {agg['id'].nunique()} | Languages: {agg['language'].nunique()}",
        f"Obs (aggregated): {len(agg)}\n",
        "1. Bradley-Terry vs IRT θ",
        f"   Spearman ρ = {rho:.4f}  (p = {p:.2e},  n = {len(merged_bt)})",
        "   Interpretation: BT ability and IRT θ agree on model ordering but BT",
        "   collapses all language variation into a single ability — cannot recover γ or τ.\n",
        "2. Log-loss comparison (same held-in subset)",
        f"   Intercept-only:                     {ll_baseline:.4f}",
        f"   No-DIF logistic (model+prompt+lang): {ll_nodif_sub:.4f}",
        f"   2PL IRT with τ (DIF):               {ll_irt:.4f}",
        f"   ΔLL (no-DIF → IRT): {ll_nodif_sub - ll_irt:.4f}  ({100*(ll_nodif_sub-ll_irt)/ll_nodif_sub:.1f}% reduction)\n",
        "3. IRT ↔ GLMM equivalence",
        "   1PL IRT (Rasch) is a GLMM with random item and person intercepts.",
        "   2PL adds random slopes (item discrimination). Our Bayesian SVI model",
        "   IS a structured GLMM; the comparison is within, not against, that class.",
        "   Ref: De Boeck et al. (2011) Psychometrika; Wilson et al. (2008).",
    ]
    txt_path = os.path.join(OUT_DIR, "baselines_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary → {txt_path}")


if __name__ == "__main__":
    main()
