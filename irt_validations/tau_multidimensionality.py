# -*- coding: utf-8 -*-
"""
τ Multi-Dimensionality Experiment — MultiJail (Master Passes 0-9)
==================================================================
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
     — Compute binary residuals r = y − P̂ for each observation,
       aggregate to model × category averages, and run EFA on the
       resulting correlation matrix.  If full-model residuals are more
       unidimensional than no-τ residuals, τ was absorbing factorial structure.

Outputs (saved to results/):
  tau_pca_scree.png
  tau_pca_loadings.csv
  tau_pca_lang_heatmap.png
  tau_category_language_heatmap.png
  tau_regression_summary.txt
  tau_efa_residuals_no_tau.png
  tau_efa_residuals_full.png
  tau_efa_residuals_comparison.csv
"""

import os
import sys
import ast
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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
from huggingface_hub import snapshot_download

# ── paths ─────────────────────────────────────────────────────────────────────

DATA_DIR   = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
IRT_CSV    = os.path.join(DATA_DIR, "resultsmar19", "results",
                           "bayesian_irt_results_binary.csv")
IRT_PT     = os.path.join(DATA_DIR, "resultsmar19", "results",
                           "irt_params_binary_2pl.pt")

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


def load_raw_data(input_file: str) -> pd.DataFrame:
    """
    Load Master_Passes0-9_Dataset.csv.
    - Tags are stored as list-strings → ast.literal_eval + explode
    - Binary score: judge_score >= 4 = safe
    - Aggregate across passes (mean per student×prompt×language×tag)
    """
    df = pd.read_csv(input_file, low_memory=False)
    df["id"] = df["id"].apply(clean_id)
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    df = df[df["judge_score"] > 0].dropna(subset=["judge_score"]).copy()
    df["safe"] = (df["judge_score"] >= 4).astype(float)

    student_col = "test_taker" if "test_taker" in df.columns else "model"
    df["student"] = df[student_col]

    # Tags: stored as list-strings, e.g. "['crime', 'violence']"
    df["tags"] = df["tags"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    df = df.explode("tags")
    df = df.rename(columns={"tags": "category"})
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["category"] != ""]

    print(f"Raw data: {len(df):,} obs (post-explode), "
          f"{df['student'].nunique()} models, "
          f"{df['id'].nunique()} prompts, "
          f"{df['language'].nunique()} languages, "
          f"{df['category'].nunique()} categories")
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

    tau_filled = tau_wide.fillna(tau_wide.mean())
    scaler = StandardScaler()
    X = scaler.fit_transform(tau_filled.values)   # prompts × languages

    pca = PCA()
    pca.fit(X)

    ev  = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    n_langs = tau_wide.shape[1]

    print(f"\nEigenvalues (Kaiser criterion: λ > 1 → retain):")
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
    ax.set_title("PCA Scree Plot — τ Matrix (MultiJail)")
    ax.legend()
    fig.tight_layout()
    _save(fig, os.path.join(RESULTS_DIR, "tau_pca_scree.png"))
    print("  Saved: tau_pca_scree.png")

    # Language loadings on first 4 PCs
    n_show = min(4, n_langs)
    loadings = pd.DataFrame(
        pca.components_[:n_show].T,
        index=tau_wide.columns,
        columns=[f"PC{k+1}" for k in range(n_show)],
    )
    print(f"\nLanguage loadings on PC1/PC2:")
    print(loadings[["PC1", "PC2"]].to_string())

    fig, ax = plt.subplots(figsize=(max(8, n_langs + 2), 4), layout="constrained")
    sns.heatmap(
        loadings.T,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0,
        linewidths=0.4, ax=ax,
    )
    ax.set_title("Language Loadings on First PCs of τ  (MultiJail)")
    ax.set_ylabel("Principal Component")
    ax.set_xlabel("Language")
    _save(fig, os.path.join(RESULTS_DIR, "tau_pca_lang_heatmap.png"))
    print("  Saved: tau_pca_lang_heatmap.png")

    scores_df = pd.DataFrame(
        pca.transform(X)[:, :n_show],
        index=tau_wide.index,
        columns=[f"PC{k+1}" for k in range(n_show)],
    )
    scores_df.to_csv(os.path.join(RESULTS_DIR, "tau_pca_loadings.csv"))
    print("  Saved: tau_pca_loadings.csv")

    return pca, loadings, scores_df


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Regression of τ on category × language
# ══════════════════════════════════════════════════════════════════════════════

def analysis_regression_tau(tau_wide: pd.DataFrame, raw_df: pd.DataFrame):
    """
    OLS regression of τ_iL on (category FE) + (language FE) + interaction.

    R² from nested models shows how much τ is systematic (category/language
    structure) vs. idiosyncratic prompt-level noise.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2 — Regression of τ on category + language")
    print("=" * 60)

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  statsmodels not available — skipping regression analysis")
        return None

    # Prompt → dominant category (most frequent tag across all rows)
    cat_map = (
        raw_df.groupby("id")["category"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )

    tau_long = (
        tau_wide.reset_index()
        .melt(id_vars="prompt_id", var_name="language", value_name="tau")
        .dropna(subset=["tau"])
    )
    tau_long["category"] = tau_long["prompt_id"].map(cat_map).fillna("Unknown")

    n_total = len(tau_long)
    tau_var  = tau_long["tau"].var()
    print(f"\n  τ observations: {n_total:,}  |  Var(τ) = {tau_var:.4f}")

    results = {}

    ma = smf.ols("tau ~ C(language)", data=tau_long).fit()
    results["language_only"] = ma.rsquared

    mb = smf.ols("tau ~ C(category)", data=tau_long).fit()
    results["category_only"] = mb.rsquared

    mc = smf.ols("tau ~ C(language) + C(category)", data=tau_long).fit()
    results["language_plus_category"] = mc.rsquared

    md = smf.ols("tau ~ C(language) * C(category)", data=tau_long).fit()
    results["language_x_category"] = md.rsquared

    print("\n  OLS R² for nested models:")
    print(f"    Language FE only              : R²={results['language_only']:.4f}")
    print(f"    Category FE only              : R²={results['category_only']:.4f}")
    print(f"    Language + Category (additive): R²={results['language_plus_category']:.4f}")
    print(f"    Language × Category (interact): R²={results['language_x_category']:.4f}")
    unexplained = 1 - results["language_x_category"]
    print(f"\n    Unexplained prompt-idiosyncratic τ variance: {unexplained:.1%}")
    print(f"    → High (>60%): τ is mostly idiosyncratic noise around systematic trends.")
    print(f"    → Low  (<40%): τ encodes structured multi-dimensionality.")

    # Mean |τ| by category
    cat_stats = (
        tau_long.groupby("category")["tau"]
        .agg(mean_tau="mean", mean_abs_tau=lambda x: x.abs().mean(), n="count")
        .sort_values("mean_abs_tau", ascending=False)
    )
    print(f"\n  Mean |τ| by category:")
    print(cat_stats.to_string())

    # Category × language heatmap of mean τ
    pivot = tau_long.pivot_table(
        values="tau", index="category", columns="language", aggfunc="mean"
    )
    fig, ax = plt.subplots(
        figsize=(max(8, len(tau_wide.columns) + 2), max(5, len(pivot) * 0.5 + 2)),
        layout="constrained"
    )
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.3, ax=ax)
    ax.set_title("Mean τ (Cross-Lingual Safety Gap) by Category × Language  (MultiJail)")
    ax.set_xlabel("Language")
    ax.set_ylabel("Category")
    _save(fig, os.path.join(RESULTS_DIR, "tau_category_language_heatmap.png"))
    print("  Saved: tau_category_language_heatmap.png")

    summary_path = os.path.join(RESULTS_DIR, "tau_regression_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("τ Regression Summary — MultiJail\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"N observations: {n_total:,}\n")
        f.write(f"Var(τ): {tau_var:.4f}\n\n")
        f.write("OLS R² nested models:\n")
        for name, r2 in results.items():
            f.write(f"  {name:<35}: {r2:.4f}\n")
        f.write(f"\nFull interaction model F-stat: {md.fvalue:.2f}  p={md.f_pvalue:.2e}\n")
        f.write(f"\nMean |τ| by category:\n{cat_stats.to_string()}\n")
    print("  Saved: tau_regression_summary.txt")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — EFA on IRT residuals (no-τ vs full model)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_residuals_from_csv(raw_df: pd.DataFrame,
                                irt_df: pd.DataFrame,
                                theta_df: pd.DataFrame,
                                delta_df: pd.DataFrame,
                                include_tau: bool) -> np.ndarray:
    """
    Binary residuals r = y − σ(logit) using posterior means from local CSVs.

    Parameters loaded from:
      irt_df   — bayesian_irt_results_binary.csv
                 columns: prompt, language, Base_Difficulty (β), gamma_L (γ),
                          Safety_Tax (τ), alpha (α)
      theta_df — theta_person_params.csv  columns: test_taker, theta (θ)
      delta_df — delta_person_params.csv  columns: test_taker, language, delta (δ)
    """
    # Build lookup dicts
    theta_map = theta_df.set_index("test_taker")["theta"].to_dict()

    delta_map = delta_df.set_index(["test_taker", "language"])["delta"].to_dict()

    # irt_df is indexed by (prompt, language)
    irt_idx = irt_df.set_index(["prompt", "language"])

    y      = raw_df["safe"].values
    p_hats = np.zeros(len(raw_df), dtype=np.float64)

    students  = raw_df["student"].values
    prompts   = raw_df["id"].values
    languages = raw_df["language"].values

    for k in range(len(raw_df)):
        s, p, l = students[k], prompts[k], languages[k]

        theta = theta_map.get(s, 0.0)
        delta = delta_map.get((s, l), 0.0)

        try:
            row   = irt_idx.loc[(p, l)]
            beta  = row["Base_Difficulty"]
            gamma = row["gamma_L"]
            alpha = max(row["alpha"], 0.1)
            tau   = row["Safety_Tax"] if include_tau else 0.0
        except KeyError:
            # Prompt-language pair absent from IRT results (e.g. English)
            p_hats[k] = 0.5
            continue

        logit    = alpha * ((theta + delta) - (beta + gamma + tau))
        p_hats[k] = 1.0 / (1.0 + np.exp(-logit))

    return y - p_hats


def analysis_efa_residuals(raw_df: pd.DataFrame):
    """
    EFA on IRT residuals aggregated to (model × category) means.

    Loads posterior means from local CSVs (bayesian_irt_results_binary.csv,
    theta_person_params.csv, delta_person_params.csv) so that the no-τ
    condition correctly zeros out Safety_Tax while keeping all other params
    identical.

    A higher dominance ratio (λ₁/λ₂) in the full model means τ absorbed
    multi-dimensional structure, leaving more unidimensional residuals.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 3 — EFA on IRT residuals (no-τ vs full model)")
    print("=" * 60)

    local_irt_csv   = os.path.join(RESULTS_DIR, "bayesian_irt_results_binary.csv")
    local_theta_csv = os.path.join(RESULTS_DIR, "theta_person_params.csv")
    local_delta_csv = os.path.join(RESULTS_DIR, "delta_person_params.csv")

    for path in [local_irt_csv, local_theta_csv, local_delta_csv]:
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            print("  Run model/irt.py locally first.")
            return None

    irt_df   = pd.read_csv(local_irt_csv)
    theta_df = pd.read_csv(local_theta_csv)
    delta_df = pd.read_csv(local_delta_csv)

    # Normalise prompt id to string
    irt_df["prompt"] = irt_df["prompt"].apply(clean_id)

    print(f"  Loaded IRT params: {len(irt_df)} prompt-language rows, "
          f"{len(theta_df)} models, {len(delta_df)} model-language rows")

    results_rows = []

    for include_tau, label in [(False, "no_tau"), (True, "full")]:
        print(f"\n  Computing residuals ({label})...")
        resid = _compute_residuals_from_csv(
            raw_df, irt_df, theta_df, delta_df, include_tau=include_tau
        )

        tmp = raw_df.copy()
        tmp["residual"] = resid

        # Aggregate to student × category mean residual
        agg = (
            tmp.groupby(["student", "category"])["residual"]
            .mean()
            .reset_index()
        )
        mat = agg.pivot(index="student", columns="category", values="residual")
        mat = mat.dropna(axis=1, thresh=int(0.8 * len(mat)))
        mat = mat.fillna(mat.mean())

        if mat.shape[1] < 3:
            print(f"  Too few categories ({mat.shape[1]}) — skipping {label}")
            continue

        try:
            kmo_all, kmo_model = calculate_kmo(mat.values)
            print(f"  KMO ({label}): {kmo_model:.3f}")
        except Exception:
            kmo_model = np.nan

        corr = np.corrcoef(mat.values.T)
        eigs = np.sort(np.linalg.eigvalsh(corr))[::-1]
        dom  = eigs[0] / eigs[1] if len(eigs) > 1 and eigs[1] > 0 else np.nan
        print(f"  Eigenvalues ({label}): λ1={eigs[0]:.2f}  λ2={eigs[1]:.2f}  "
              f"dominance ratio={dom:.2f}")

        for n_f in [1, 2]:
            if mat.shape[1] <= n_f:
                continue
            try:
                fa = FactorAnalyzer(n_factors=n_f, rotation="varimax", method="minres")
                fa.fit(mat.values)
                mean_h2 = fa.get_communalities().mean()
                print(f"  EFA {n_f}F ({label}): mean h²={mean_h2:.3f}")
                results_rows.append({
                    "model":          label,
                    "n_factors":      n_f,
                    "kmo":            kmo_model,
                    "mean_h2":        mean_h2,
                    "eigenvalue_1":   eigs[0],
                    "eigenvalue_2":   eigs[1] if len(eigs) > 1 else np.nan,
                    "dominance_ratio": dom,
                })

                if n_f == 2:
                    loadings = pd.DataFrame(
                        fa.loadings_,
                        index=mat.columns,
                        columns=["F1", "F2"],
                    )
                    fig, ax = plt.subplots(
                        figsize=(5, max(5, mat.shape[1] * 0.45 + 2)),
                        layout="constrained"
                    )
                    sns.heatmap(
                        loadings, annot=True, fmt=".2f",
                        cmap="RdBu_r", center=0,
                        linewidths=0.3, vmin=-1, vmax=1, ax=ax,
                    )
                    ax.set_title(f"EFA 2F Loadings on Residuals ({label})  MultiJail")
                    ax.set_xlabel("Factor")
                    ax.set_ylabel("Category")
                    _save(fig, os.path.join(RESULTS_DIR,
                                            f"tau_efa_residuals_{label}.png"))
                    print(f"  Saved: tau_efa_residuals_{label}.png")

            except Exception as e:
                print(f"  EFA {n_f}F ({label}) failed: {e}")

    if results_rows:
        comp = pd.DataFrame(results_rows)
        comp.to_csv(os.path.join(RESULTS_DIR, "tau_efa_residuals_comparison.csv"),
                    index=False)
        print("\n  Residual EFA comparison:")
        print(comp.to_string(index=False))

        no_tau_row = comp[(comp["model"] == "no_tau") & (comp["n_factors"] == 2)]
        full_row   = comp[(comp["model"] == "full")   & (comp["n_factors"] == 2)]
        if len(no_tau_row) and len(full_row):
            dr_no  = no_tau_row["dominance_ratio"].values[0]
            dr_ful = full_row["dominance_ratio"].values[0]
            print(f"\n  Dominance ratio λ1/λ2:")
            print(f"    no-τ model : {dr_no:.2f}")
            print(f"    full model : {dr_ful:.2f}")
            if dr_ful > dr_no:
                print("  → Full model residuals MORE unidimensional: "
                      "τ absorbed structured multi-dimensional variance.")
            else:
                print("  → No improvement: τ did not absorb additional factorial structure.")

        print("  Saved: tau_efa_residuals_comparison.csv")

    return results_rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("τ MULTI-DIMENSIONALITY EXPERIMENT — MultiJail")
    print("=" * 60)

    if not os.path.exists(IRT_CSV):
        raise FileNotFoundError(
            f"IRT results not found: {IRT_CSV}\n"
            "Run model/irt.py and upload results to HuggingFace first."
        )
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Data not found: {INPUT_FILE}")

    tau_wide = load_tau_matrix(IRT_CSV)
    raw_df   = load_raw_data(INPUT_FILE)

    pca, loadings, scores_df = analysis_pca_tau(tau_wide)
    reg_results              = analysis_regression_tau(tau_wide, raw_df)
    efa_results              = analysis_efa_residuals(raw_df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ev  = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    n_kaiser = int((ev > 1).sum())
    print(f"  PCA on τ: {n_kaiser} component(s) with λ > 1")
    print(f"  PC1 explains {evr[0]:.1%} of τ variance")
    print(f"  All outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
