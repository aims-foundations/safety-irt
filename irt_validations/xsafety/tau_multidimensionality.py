# -*- coding: utf-8 -*-
"""
τ Multi-Dimensionality Experiment — XSafety
============================================
Tests whether τ (Cross-Lingual Safety Gap) absorbs residual multi-dimensionality
beyond the dominant unidimensionality captured by θ.

Three analyses:
  1. PCA / EFA on the τ_{i,L} matrix (prompts × languages)
     — If τ has >1 factor with eigenvalue > 1, it encodes structured variance
       beyond a single safety-tax dimension.

  2. Regression of τ on category (tags) × language fixed effects
     — R² from OLS quantifies how much τ variance is explained by systematic
       category-language interactions vs. idiosyncratic prompt-level noise.

  3. EFA on IRT residuals: no-τ model vs. full model
     — Compute binary residuals r_{j,i,L} = y − P̂ for each observation,
       then aggregate to model × category averages and run EFA on the
       resulting correlation matrix.  Reduced off-diagonal structure in the
       full-model residuals implies τ was absorbing factorially structured
       variance.

Outputs (saved to results/):
  tau_pca_scree.png
  tau_pca_loadings.csv       (prompts × PC scores)
  tau_pca_lang_heatmap.png   (language loadings on PC1/PC2)
  tau_regression_summary.txt
  tau_efa_residuals_nontau.png   (EFA factor structure without τ)
  tau_efa_residuals_full.png     (EFA factor structure with τ)
  tau_efa_residuals_comparison.csv
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
try:
    from fig_style import apply_style, savefig as fs_savefig
    apply_style()
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import pearsonr
from huggingface_hub import snapshot_download

# ── paths ────────────────────────────────────────────────────────────────────

DATA_DIR    = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "xsafety", "xsafety_pass_graded.csv")
XSAFETY_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model", "xsafety", "xsafety_results")
IRT_CSV     = os.path.join(XSAFETY_MODEL_DIR, "bayesian_irt_results_binary.csv")
IRT_PT      = os.path.join(XSAFETY_MODEL_DIR, "irt_params_binary_2pl.pt")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_save = fs_savefig if _HAS_FIG_STYLE else \
    lambda f, p: (f.savefig(p, dpi=300, bbox_inches="tight"), plt.close(f))


# ── helpers ───────────────────────────────────────────────────────────────────

def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def load_tau_matrix(irt_csv: str) -> pd.DataFrame:
    """Return wide τ matrix: rows=prompt_id, cols=languages (non-English, non-anchor)."""
    df = pd.read_csv(irt_csv)

    for col in ["prompt", "prompt_id", "item"]:
        if col in df.columns:
            df.rename(columns={col: "prompt_id"}, inplace=True)
            break
    df["prompt_id"] = df["prompt_id"].apply(clean_id)

    tau_col    = "Safety_Tax" if "Safety_Tax" in df.columns else "tau"
    anchor_col = "Is_Anchor"  if "Is_Anchor"  in df.columns else "is_anchor"

    mask = df["language"] != "en"
    if anchor_col in df.columns:
        mask = mask & (~df[anchor_col].astype(bool))
    tau_long = df[mask][["prompt_id", "language", tau_col]].copy()
    tau_long  = tau_long.rename(columns={tau_col: "tau"})

    tau_wide = tau_long.pivot(index="prompt_id", columns="language", values="tau")
    tau_wide = tau_wide.dropna(how="all")
    print(f"τ matrix: {tau_wide.shape[0]} prompts × {tau_wide.shape[1]} languages")
    print(f"  Mean missingness per language: "
          f"{tau_wide.isna().mean().mean():.1%}")
    return tau_wide


def load_raw_data(input_file: str):
    """Load XSafety graded CSV, return df with binary score column."""
    df = pd.read_csv(input_file, engine="python", on_bad_lines="skip")
    df["id"] = df["id"].apply(clean_id)
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    df = df[df["judge_score"] > 0].dropna(subset=["judge_score"]).copy()
    df["safe"] = (df["judge_score"] >= 4).astype(float)
    student_col = "test_taker" if "test_taker" in df.columns else "model"
    df["student"] = df[student_col]
    cat_col = "tags" if "tags" in df.columns else "category"
    df["category"] = df[cat_col].astype(str)
    print(f"Raw data: {len(df):,} observations, "
          f"{df['student'].nunique()} models, "
          f"{df['id'].nunique()} prompts, "
          f"{df['language'].nunique()} languages")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — PCA on τ matrix
# ══════════════════════════════════════════════════════════════════════════════

def analysis_pca_tau(tau_wide: pd.DataFrame):
    """
    PCA on the (prompts × languages) τ matrix.

    Interpretation:
      PC1 eigenvalue >> others  → τ is essentially one-dimensional (uniform safety tax)
      Multiple large PCs         → τ captures structured multi-dimensionality
      Language loadings on PC1  → which languages drive the dominant gap direction
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1 — PCA on τ_{i,L} matrix")
    print("=" * 60)

    # Fill missing values with column mean (languages with sparse τ)
    tau_filled = tau_wide.fillna(tau_wide.mean())
    scaler = StandardScaler()
    X = scaler.fit_transform(tau_filled.values)   # shape: prompts × langs

    pca = PCA()
    pca.fit(X)

    ev  = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    n_langs = tau_wide.shape[1]

    print(f"\nEigenvalues (Kaiser criterion: >1 → retain):")
    for k in range(min(n_langs, 10)):
        bar = "█" * int(evr[k] * 40)
        print(f"  PC{k+1:2d}  λ={ev[k]:6.2f}  ({evr[k]:.1%})  {bar}")

    n_factors_kaiser = int((ev > 1).sum())
    print(f"\n  → {n_factors_kaiser} component(s) with λ > 1  (Kaiser criterion)")
    print(f"  → PC1 explains {evr[0]:.1%} of τ variance")
    print(f"  → PC1+PC2 explain {evr[:2].sum():.1%} of τ variance")

    # Scree plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, n_langs + 1), ev, "o-", color="#2166AC", lw=2)
    ax.axhline(1, ls="--", color="#D6604D", lw=1.2, label="Kaiser criterion (λ=1)")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("PCA Scree Plot — τ Matrix (XSafety)")
    ax.legend()
    _save(fig, os.path.join(RESULTS_DIR, "tau_pca_scree.png"))
    print("  Saved: tau_pca_scree.png")

    # Language loadings (components × languages)
    loadings = pd.DataFrame(
        pca.components_[:4].T,
        index=tau_wide.columns,
        columns=[f"PC{k+1}" for k in range(4)],
    )
    print(f"\nLanguage loadings on PC1/PC2:")
    print(loadings[["PC1", "PC2"]].to_string())

    # Heatmap of language loadings
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        loadings.T,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0,
        linewidths=0.4, ax=ax,
    )
    ax.set_title("Language Loadings on First 4 PCs of τ")
    ax.set_ylabel("Principal Component")
    ax.set_xlabel("Language")
    fig.tight_layout()
    _save(fig, os.path.join(RESULTS_DIR, "tau_pca_lang_heatmap.png"))
    print("  Saved: tau_pca_lang_heatmap.png")

    # Prompt-level PC scores
    scores_df = pd.DataFrame(
        pca.transform(X)[:, :4],
        index=tau_wide.index,
        columns=[f"PC{k+1}" for k in range(4)],
    )
    scores_df.to_csv(os.path.join(RESULTS_DIR, "tau_pca_loadings.csv"))
    print("  Saved: tau_pca_loadings.csv")

    return pca, loadings, scores_df


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Regression of τ on category × language
# ══════════════════════════════════════════════════════════════════════════════

def analysis_regression_tau(tau_wide: pd.DataFrame, raw_df: pd.DataFrame):
    """
    OLS regression of τ_iL on (category FE) + (language FE) + (category×language interaction).

    R² from each nested model shows how much systematic structure (vs. idiosyncratic
    prompt-level noise) exists in τ.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2 — Regression of τ on category + language")
    print("=" * 60)

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  statsmodels not available — skipping regression analysis")
        return

    # Get prompt-level category from raw data
    cat_map = (
        raw_df[["id", "category"]]
        .drop_duplicates(subset="id")
        .set_index("id")["category"]
        .to_dict()
    )

    tau_long = tau_wide.reset_index().melt(
        id_vars="prompt_id", var_name="language", value_name="tau"
    ).dropna(subset=["tau"])
    tau_long["category"] = tau_long["prompt_id"].map(cat_map).fillna("Unknown")

    n_total = len(tau_long)
    tau_var  = tau_long["tau"].var()
    print(f"\n  τ observations: {n_total:,}  |  Var(τ) = {tau_var:.4f}")

    results = {}

    # Model A: language only
    ma = smf.ols("tau ~ C(language)", data=tau_long).fit()
    results["language_only"] = ma.rsquared

    # Model B: category only
    mb = smf.ols("tau ~ C(category)", data=tau_long).fit()
    results["category_only"] = mb.rsquared

    # Model C: language + category (additive)
    mc = smf.ols("tau ~ C(language) + C(category)", data=tau_long).fit()
    results["language_plus_category"] = mc.rsquared

    # Model D: language × category interaction
    md = smf.ols("tau ~ C(language) * C(category)", data=tau_long).fit()
    results["language_x_category"] = md.rsquared

    print("\n  OLS R² for nested models:")
    print(f"    Language FE only             : R²={results['language_only']:.4f}")
    print(f"    Category FE only             : R²={results['category_only']:.4f}")
    print(f"    Language + Category (additive): R²={results['language_plus_category']:.4f}")
    print(f"    Language × Category (interact): R²={results['language_x_category']:.4f}")
    unexplained = 1 - results["language_x_category"]
    print(f"\n    Unexplained prompt-idiosyncratic τ variance: {unexplained:.1%}")
    print(f"    → If high (>60%), τ is mostly idiosyncratic noise around systematic trends.")
    print(f"    → If low (<40%), τ encodes structured multi-dimensionality.")

    # Mean |τ| by category
    cat_stats = (
        tau_long.groupby("category")["tau"]
        .agg(mean_tau="mean", mean_abs_tau=lambda x: x.abs().mean(), n="count")
        .sort_values("mean_abs_tau", ascending=False)
    )
    print(f"\n  Mean |τ| by category (top 10):")
    print(cat_stats.head(10).to_string())

    # Mean τ heatmap (category × language)
    pivot = tau_long.pivot_table(values="tau", index="category", columns="language", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(8, len(tau_wide.columns) + 2), max(6, len(pivot) * 0.5 + 2)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.3, ax=ax)
    ax.set_title("Mean τ (Cross-Lingual Safety Gap) by Category × Language")
    ax.set_xlabel("Language")
    ax.set_ylabel("Category")
    fig.tight_layout()
    _save(fig, os.path.join(RESULTS_DIR, "tau_category_language_heatmap.png"))
    print("  Saved: tau_category_language_heatmap.png")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "tau_regression_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("τ Regression Summary — XSafety\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"N observations: {n_total:,}\n")
        f.write(f"Var(τ): {tau_var:.4f}\n\n")
        f.write("OLS R² nested models:\n")
        for name, r2 in results.items():
            f.write(f"  {name:<35}: {r2:.4f}\n")
        f.write(f"\nFull interaction model F-stat: {md.fvalue:.2f}  p={md.f_pvalue:.2e}\n")
        f.write(f"\nMean |τ| by category:\n{cat_stats.to_string()}\n")
    print(f"  Saved: tau_regression_summary.txt")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — EFA on IRT residuals (no-τ vs full model)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_residuals_from_params(raw_df: pd.DataFrame, params: dict,
                                   include_tau: bool) -> pd.DataFrame:
    """
    Compute binary residuals r = y − σ(logit) for each observation
    using extracted IRT parameters.  Returns df with columns:
      student, category, language, residual
    """
    import torch
    torch.set_grad_enabled(False)

    students  = raw_df["student"].unique()
    prompts   = raw_df["id"].unique()
    languages = raw_df["language"].unique()

    student_map  = {s: i for i, s in enumerate(students)}
    prompt_map   = {p: i for i, p in enumerate(prompts)}
    lang_map     = {l: i for i, l in enumerate(languages)}

    def _get(key, default_shape):
        v = params.get(key)
        if v is None:
            return torch.zeros(default_shape)
        if isinstance(v, torch.Tensor):
            return v.float()
        return torch.tensor(v, dtype=torch.float32)

    n_s = len(students)
    n_p = len(prompts)
    n_l = len(languages)

    theta = _get("theta", (n_s,))
    beta  = _get("beta",  (n_p,))
    alpha = _get("alpha", (n_p,)).clamp(min=0.1)
    gamma = _get("gamma", (n_l,))
    delta = _get("delta", (n_s, n_l))
    tau   = _get("tau",   (n_p, n_l)) if include_tau else torch.zeros(n_p, n_l)

    s_idx = torch.tensor([student_map.get(s, 0) for s in raw_df["student"]], dtype=torch.long)
    p_idx = torch.tensor([prompt_map.get(p, 0)  for p in raw_df["id"]],      dtype=torch.long)
    l_idx = torch.tensor([lang_map.get(l, 0)    for l in raw_df["language"]], dtype=torch.long)

    ability    = theta[s_idx] + delta[s_idx, l_idx]
    difficulty = beta[p_idx] + gamma[l_idx] + tau[p_idx, l_idx]
    logits     = alpha[p_idx] * (ability - difficulty)
    p_hat      = torch.sigmoid(logits).numpy()

    residuals = raw_df["safe"].values - p_hat
    return residuals


def analysis_efa_residuals(raw_df: pd.DataFrame):
    """
    EFA on IRT residuals aggregated to model × category averages.

    Loads saved .pt parameters, computes residuals with and without τ,
    then compares the factor structure of the two residual correlation matrices.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 3 — EFA on IRT residuals (no-τ vs full model)")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("  torch not available — skipping residual EFA")
        return

    if not os.path.exists(IRT_PT):
        print(f"  IRT params not found at: {IRT_PT}")
        print("  Run model/xsafety/irt.py and upload results first.")
        return

    print(f"  Loading IRT params from: {os.path.basename(IRT_PT)}")
    params = torch.load(IRT_PT, map_location="cpu", weights_only=False)
    if isinstance(params, dict) and "params" in params:
        params = params["params"]

    # Flatten any extra batch/sample dimensions (take posterior mean if shape > expected)
    def _flatten(key, ndim):
        v = params.get(key)
        if v is None:
            return None
        t = v.float()
        while t.dim() > ndim:
            t = t.mean(0)
        params[key] = t

    _flatten("theta", 1)
    _flatten("beta",  1)
    _flatten("alpha", 1)
    _flatten("gamma", 1)
    _flatten("delta", 2)
    _flatten("tau",   2)

    results_rows = []
    for include_tau, label in [(False, "no_tau"), (True, "full")]:
        print(f"\n  Computing residuals ({label})...")
        resid = _compute_residuals_from_params(raw_df, params, include_tau=include_tau)
        raw_df_copy = raw_df.copy()
        raw_df_copy["residual"] = resid

        # Aggregate: mean residual per (student × category)
        agg = (
            raw_df_copy.groupby(["student", "category"])["residual"]
            .mean()
            .reset_index()
        )
        mat = agg.pivot(index="student", columns="category", values="residual")
        mat = mat.dropna(axis=1, thresh=int(0.8 * len(mat)))
        mat = mat.fillna(mat.mean())

        if mat.shape[1] < 3:
            print(f"  Too few categories ({mat.shape[1]}) for EFA — skipping {label}")
            continue

        # KMO
        try:
            kmo_all, kmo_model = calculate_kmo(mat.values)
            print(f"  KMO ({label}): {kmo_model:.3f}")
        except Exception:
            kmo_model = np.nan

        # Parallel analysis: eigenvalues of actual vs random correlation matrices
        corr = np.corrcoef(mat.values.T)
        eigs = np.sort(np.linalg.eigvalsh(corr))[::-1]

        # Fit EFA with 1 and 2 factors for comparison
        for n_f in [1, 2]:
            if mat.shape[1] <= n_f:
                continue
            try:
                fa = FactorAnalyzer(n_factors=n_f, rotation="varimax", method="minres")
                fa.fit(mat.values)
                loadings = pd.DataFrame(
                    fa.loadings_,
                    index=mat.columns,
                    columns=[f"F{k+1}" for k in range(n_f)],
                )
                communalities = fa.get_communalities()
                mean_h2 = communalities.mean()
                print(f"  EFA {n_f}F ({label}): mean communality h²={mean_h2:.3f}")
                results_rows.append({
                    "model": label,
                    "n_factors": n_f,
                    "kmo": kmo_model,
                    "mean_h2": mean_h2,
                    "eigenvalue_1": eigs[0],
                    "eigenvalue_2": eigs[1] if len(eigs) > 1 else np.nan,
                    "dominance_ratio": eigs[0] / eigs[1] if len(eigs) > 1 and eigs[1] > 0 else np.nan,
                })

                if n_f == 2:
                    fig, ax = plt.subplots(figsize=(max(7, mat.shape[1] * 0.5 + 2), 5))
                    sns.heatmap(
                        loadings, annot=True, fmt=".2f",
                        cmap="RdBu_r", center=0, linewidths=0.3, ax=ax,
                        vmin=-1, vmax=1,
                    )
                    ax.set_title(f"EFA 2-Factor Loadings on Residuals ({label})")
                    ax.set_xlabel("Factor")
                    ax.set_ylabel("Category")
                    fig.tight_layout()
                    _save(fig, os.path.join(RESULTS_DIR, f"tau_efa_residuals_{label}.png"))
                    print(f"  Saved: tau_efa_residuals_{label}.png")

            except Exception as e:
                print(f"  EFA {n_f}F ({label}) failed: {e}")

    if results_rows:
        comp = pd.DataFrame(results_rows)
        comp.to_csv(os.path.join(RESULTS_DIR, "tau_efa_residuals_comparison.csv"), index=False)
        print("\n  Residual EFA comparison:")
        print(comp.to_string(index=False))

        # Interpretation
        no_tau = comp[comp["model"] == "no_tau"]
        full   = comp[comp["model"] == "full"]
        if len(no_tau) and len(full):
            dr_no  = no_tau[no_tau["n_factors"] == 2]["dominance_ratio"].values
            dr_ful = full[full["n_factors"] == 2]["dominance_ratio"].values
            if len(dr_no) and len(dr_ful):
                print(f"\n  Dominance ratio (λ1/λ2):")
                print(f"    no-τ model : {dr_no[0]:.2f}")
                print(f"    full model : {dr_ful[0]:.2f}")
                if dr_ful[0] > dr_no[0]:
                    print("  → Full model residuals are MORE unidimensional: "
                          "τ absorbed structured multi-dimensional variance.")
                else:
                    print("  → No clear improvement: τ did not absorb additional "
                          "factorial structure.")

        print("  Saved: tau_efa_residuals_comparison.csv")
    return results_rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("τ MULTI-DIMENSIONALITY EXPERIMENT — XSafety")
    print("=" * 60)

    if not os.path.exists(IRT_CSV):
        raise FileNotFoundError(
            f"IRT results not found: {IRT_CSV}\n"
            "Run model/xsafety/irt.py and upload results to HuggingFace first."
        )
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"XSafety data not found: {INPUT_FILE}"
        )

    tau_wide = load_tau_matrix(IRT_CSV)
    raw_df   = load_raw_data(INPUT_FILE)

    # 1. PCA on τ matrix
    pca, loadings, scores_df = analysis_pca_tau(tau_wide)

    # 2. Regression of τ on category × language
    reg_results = analysis_regression_tau(tau_wide, raw_df)

    # 3. EFA on IRT residuals
    efa_results = analysis_efa_residuals(raw_df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ev  = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    n_kaiser = int((ev > 1).sum())
    print(f"  PCA on τ: {n_kaiser} component(s) with λ > 1")
    print(f"  PC1 explains {evr[0]:.1%} of τ variance")
    print(f"  See results/ directory for all outputs.")


if __name__ == "__main__":
    main()
