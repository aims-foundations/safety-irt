# -*- coding: utf-8 -*-
"""
GLMM Baseline vs. 2PL IRT Decomposition.
==========================================
Fits a mixed-effects logistic regression:

  logit P(safe_ijL) = μ + u_j + v_i + w_L + x_iL

  u_j  ~ N(0, σ²_model)    — model random effect        (≈ IRT θ_j)
  v_i  ~ N(0, σ²_prompt)   — prompt random effect        (≈ IRT β_i)
  w_L  ~ N(0, σ²_lang)     — language random effect      (≈ IRT γ_L)
  x_iL ~ N(0, σ²_inter)    — prompt×language interaction (≈ IRT τ_iL)

IRT results are loaded directly from HuggingFace (results_figures/results/).
GLMM is fitted fresh using statsmodels BinomialBayesMixedGLM (variational Bayes).

Outputs (results_glmm_baseline/):
  param_correspondence.csv      — ρ/r/RMSE for θ, γ, τ
  theta_comparison.csv          — per-model u_j vs θ_j
  gamma_comparison.csv          — per-language w_L vs γ_L
  tau_comparison.csv            — per-(prompt,lang) x_iL vs τ_iL
  predictive_metrics.csv        — AUC / accuracy / Brier
  param_comparison.png          — scatter grid: θ / γ / τ
  predictive_comparison.png     — bar chart: AUC / accuracy / Brier
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import csc_matrix, hstack as sp_hstack
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from huggingface_hub import snapshot_download

# ── fig_style ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from fig_style import apply_style, C_RED, C_BLUE, C_PURPLE, LANG_ORDER
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False
    C_RED, C_BLUE, C_PURPLE = '#c0392b', '#2471a3', '#7d3c98'
    LANG_ORDER = None

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR     = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset",
                                  token=False)
INPUT_FILE   = os.path.join(DATA_DIR, "processed_data",
                             "Master_Passes0-9_Dataset.csv")
IRT_CSV      = os.path.join(DATA_DIR, "results_figures", "results",
                             "bayesian_irt_results_binary.csv")
IRT_PT       = os.path.join(DATA_DIR, "results_figures", "results",
                             "irt_params_binary_2pl.pt")
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "results_glmm_baseline")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── helpers ───────────────────────────────────────────────────────────────────

def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def one_hot_sparse(values, categories):
    """Sparse one-hot matrix aligned to given category order."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    codes   = np.array([cat_idx[v] for v in values])
    n, q    = len(values), len(categories)
    mat     = csc_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, q))
    return mat


# ── load data ─────────────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['score'] = (df['judge_score'] >= 4).astype(np.float32)
    df['id']    = df['id'].apply(clean_id)
    sc = 'test_taker' if 'test_taker' in df.columns else 'model'

    # Preserve training-time ordering (df[col].unique() not sorted)
    students  = list(df[sc].unique())
    prompts   = list(df['id'].unique())
    languages = list(df['language'].unique())

    print(f"  {len(df):,} rows | {len(students)} models | "
          f"{len(prompts)} prompts | {len(languages)} languages")
    return df, sc, students, prompts, languages


# ── load IRT results from HuggingFace ────────────────────────────────────────

def load_irt(students, prompts, languages):
    """
    Loads pre-computed IRT parameters from HF results_figures/results/:
      θ  — from AutoNormal.locs.theta in .pt   (shape: n_students,)
      γ  — from AutoNormal.locs.gamma_raw       (shape: n_langs,)
      τ  — from bayesian_irt_results_binary.csv (Safety_Tax column)
      β  — from Base_Difficulty column
      α  — from alpha column
    """
    print("Loading IRT results from HuggingFace...")

    # ── θ and γ from .pt ──────────────────────────────────────────
    state      = torch.load(IRT_PT, weights_only=False)
    params     = state['params']
    theta_locs = params['AutoNormal.locs.theta'].detach().cpu().numpy()   # (n_students,)
    gamma_raw  = params['AutoNormal.locs.gamma_raw'].detach().cpu().numpy() # (n_langs,)

    # Apply gamma_mask: English = 0
    l_map = {l: i for i, l in enumerate(languages)}
    gamma_mask = np.ones(len(languages))
    if 'en' in l_map:
        gamma_mask[l_map['en']] = 0.0
    gamma = gamma_raw * gamma_mask

    theta_dict = {s: theta_locs[i] for i, s in enumerate(students)}
    gamma_dict = {l: gamma[l_map[l]] for l in languages}

    print(f"  θ: {len(students)} models  |  range [{theta_locs.min():.2f}, {theta_locs.max():.2f}]")
    print(f"  γ: {len(languages)} languages")

    # ── τ, β, α from CSV ──────────────────────────────────────────
    irt_df = pd.read_csv(IRT_CSV)
    irt_df['prompt'] = irt_df['prompt'].apply(clean_id)

    # Build lookup: (prompt, language) → Safety_Tax / Base_Difficulty / alpha
    tau_lookup  = {}
    beta_lookup = {}
    alph_lookup = {}
    for _, row in irt_df.iterrows():
        key = (str(row['prompt']), row['language'])
        tau_lookup[key]  = row['Safety_Tax']
        beta_lookup[key] = row['Base_Difficulty']
        alph_lookup[key] = row['alpha']

    non_en = [l for l in languages if l != 'en']
    print(f"  τ: {len(tau_lookup)} (prompt, language) pairs from CSV")

    return {
        'theta':      theta_locs,
        'theta_dict': theta_dict,
        'gamma':      gamma,
        'gamma_dict': gamma_dict,
        'tau_lookup': tau_lookup,
        'beta_lookup':beta_lookup,
        'alph_lookup':alph_lookup,
        'l_map':      l_map,
        'non_en':     non_en,
    }


# ── fit GLMM ─────────────────────────────────────────────────────────────────

def fit_glmm(df, sc, students, prompts, languages):
    """
    BinomialBayesMixedGLM with four variance components:
      0: model (student)  →  u_j  ≈  θ_j
      1: prompt           →  v_i  ≈  β_i
      2: language         →  w_L  ≈  γ_L  (English column dropped)
      3: prompt×language  →  x_iL ≈  τ_iL (English cells dropped)
    """
    print("\nBuilding GLMM design matrices (sparse)...")
    non_en = [l for l in languages if l != 'en']

    Z_model  = one_hot_sparse(df[sc].values,          students)
    Z_prompt = one_hot_sparse(df['id'].values,         prompts)
    Z_lang   = one_hot_sparse(df['language'].values,   non_en +['en'])  # need all for indexing
    # Drop English column
    en_col   = len(non_en)          # English is appended last
    Z_lang   = one_hot_sparse(df['language'].values, non_en + ['en'])[:, :len(non_en)]

    inter_cats   = [f"{p}:{l}" for p in prompts for l in non_en]
    inter_labels = (df['id'].astype(str) + ':' + df['language'].astype(str)).values
    # Keep only non-English rows in interaction (English obs get a zero row implicitly)
    valid_inter  = set(inter_cats)
    inter_labels_safe = np.where(np.isin(inter_labels, list(valid_inter)),
                                  inter_labels, inter_cats[0])   # dummy for en rows
    Z_inter = one_hot_sparse(inter_labels_safe, inter_cats)
    # Zero out rows that correspond to English (no interaction for reference lang)
    en_mask = (df['language'] == 'en').values
    Z_inter = Z_inter.copy()
    Z_inter[en_mask, :] = 0

    Z     = sp_hstack([Z_model, Z_prompt, Z_lang, Z_inter], format='csc')
    ident = np.array([0]*len(students) + [1]*len(prompts) +
                     [2]*len(non_en)   + [3]*len(inter_cats))

    print(f"  Z: {Z.shape[0]:,} rows × {Z.shape[1]:,} cols | "
          f"{Z.nnz:,} non-zeros")

    endog  = df['score'].values.astype(float)
    exog   = np.ones((len(df), 1))

    print("  Fitting (variational Bayes)...")
    glmm   = BinomialBayesMixedGLM(endog, exog, Z, ident, vcp_p=1, fe_p=2)
    result = glmm.fit_vb()

    intercept = result.params[0]
    re        = result.random_effects          # (Z.shape[1],)

    ns  = len(students)
    np_ = len(prompts)
    nl  = len(non_en)

    model_re  = re[:ns]
    prompt_re = re[ns:ns+np_]
    lang_re   = re[ns+np_:ns+np_+nl]
    inter_re  = re[ns+np_+nl:]

    print(f"  Intercept: {intercept:.3f}")

    # In-sample predicted probabilities
    probs_glmm = result.predict()

    return {
        'model_re':   model_re,    # (ns,) aligned to students
        'prompt_re':  prompt_re,   # (np_,) aligned to prompts
        'lang_re':    lang_re,     # (nl,) aligned to non_en
        'inter_re':   inter_re,    # aligned to inter_cats
        'inter_cats': inter_cats,
        'non_en':     non_en,
        'probs':      probs_glmm,
        'intercept':  intercept,
        'result':     result,
    }


# ── align and compare ─────────────────────────────────────────────────────────

def build_comparison_tables(irt, glmm, students, prompts, languages):
    non_en = glmm['non_en']
    l_map  = irt['l_map']

    # ── θ ──────────────────────────────────────────────────────────
    theta_rows = []
    for i, s in enumerate(students):
        theta_rows.append({
            'model':    s,
            'theta_IRT': irt['theta'][i],
            'u_GLMM':   glmm['model_re'][i],
        })
    theta_df = pd.DataFrame(theta_rows)

    # ── γ ──────────────────────────────────────────────────────────
    gamma_rows = []
    for k, lang in enumerate(non_en):
        gamma_rows.append({
            'language':  lang,
            'gamma_IRT': irt['gamma'][l_map[lang]],
            'w_GLMM':    glmm['lang_re'][k],
        })
    gamma_df = pd.DataFrame(gamma_rows)

    # ── τ ──────────────────────────────────────────────────────────
    inter_idx = {s: i for i, s in enumerate(glmm['inter_cats'])}
    tau_rows  = []
    for prompt in prompts:
        for lang in non_en:
            key     = (str(prompt), lang)
            ikey    = f"{prompt}:{lang}"
            if key not in irt['tau_lookup']:
                continue
            if ikey not in inter_idx:
                continue
            tau_rows.append({
                'prompt':    prompt,
                'language':  lang,
                'tau_IRT':   irt['tau_lookup'][key],
                'x_GLMM':    glmm['inter_re'][inter_idx[ikey]],
                'beta_IRT':  irt['beta_lookup'].get(key, np.nan),
                'alpha_IRT': irt['alph_lookup'].get(key, np.nan),
            })
    tau_df = pd.DataFrame(tau_rows)

    return theta_df, gamma_df, tau_df


def compute_metrics(theta_df, gamma_df, tau_df):
    rows = []

    # θ
    rho, _ = spearmanr(theta_df['theta_IRT'], theta_df['u_GLMM'])
    r,   _ = pearsonr(theta_df['theta_IRT'],  theta_df['u_GLMM'])
    rmse   = np.sqrt(np.mean((theta_df['theta_IRT'] - theta_df['u_GLMM'])**2))
    rows.append({'param': 'θ (model ability)',     'n': len(theta_df),
                 'spearman_rho': rho, 'pearson_r': r, 'rmse': rmse})

    # γ
    if len(gamma_df) >= 3:
        rho, _ = spearmanr(gamma_df['gamma_IRT'], gamma_df['w_GLMM'])
        r,   _ = pearsonr(gamma_df['gamma_IRT'],  gamma_df['w_GLMM'])
    else:
        rho = r = np.nan
    rmse = np.sqrt(np.mean((gamma_df['gamma_IRT'] - gamma_df['w_GLMM'])**2))
    rows.append({'param': 'γ (language shift)',    'n': len(gamma_df),
                 'spearman_rho': rho, 'pearson_r': r, 'rmse': rmse})

    # τ
    rho, _ = spearmanr(tau_df['tau_IRT'], tau_df['x_GLMM'])
    r,   _ = pearsonr(tau_df['tau_IRT'],  tau_df['x_GLMM'])
    rmse   = np.sqrt(np.mean((tau_df['tau_IRT'] - tau_df['x_GLMM'])**2))
    rows.append({'param': 'τ (cross-lingual gap)', 'n': len(tau_df),
                 'spearman_rho': rho, 'pearson_r': r, 'rmse': rmse})

    return pd.DataFrame(rows)


def compute_predictive_metrics(df, irt_probs, glmm_probs):
    y = df['score'].values
    rows = []
    for name, probs in [('IRT 2PL', irt_probs), ('GLMM', glmm_probs)]:
        p = np.clip(probs, 1e-7, 1-1e-7)
        rows.append({
            'model':    name,
            'AUC':      roc_auc_score(y, p),
            'Accuracy': accuracy_score(y, (p >= 0.5).astype(int)),
            'Brier':    brier_score_loss(y, p),
            'LogLoss':  -np.mean(y*np.log(p) + (1-y)*np.log(1-p)),
        })
    return pd.DataFrame(rows).set_index('model')


# ── IRT in-sample predictions ─────────────────────────────────────────────────

def irt_predictions(df, sc, students, prompts, languages, irt):
    """Compute IRT predicted probabilities from saved parameters."""
    state     = torch.load(IRT_PT, weights_only=False)
    params    = state['params']

    s_map = {s: i for i, s in enumerate(students)}
    p_map = {p: i for i, p in enumerate(prompts)}
    l_map = {l: i for i, l in enumerate(languages)}

    # Anchor mask for tau
    from huggingface_hub import snapshot_download as _sd
    anchor_file = os.path.join(DATA_DIR, "anchors", "anchors.csv")
    tau_mask = np.ones((len(prompts), len(languages)))
    if 'en' in l_map:
        tau_mask[:, l_map['en']] = 0.0
    if os.path.exists(anchor_file):
        adf = pd.read_csv(anchor_file)
        adf['id'] = adf['id'].apply(clean_id)
        for pid in adf['id'].unique():
            if pid in p_map:
                tau_mask[p_map[pid], :] = 0.0

    theta_locs = params['AutoNormal.locs.theta'].detach().cpu().numpy()
    beta_locs  = params['AutoNormal.locs.beta'].detach().cpu().numpy()
    alpha_locs = params['AutoNormal.locs.alpha'].detach().cpu().numpy()  # log-space
    gamma_raw  = params['AutoNormal.locs.gamma_raw'].detach().cpu().numpy()
    tau_raw    = params['AutoNormal.locs.tau_raw'].detach().cpu().numpy()

    gamma_mask = np.ones(len(languages))
    if 'en' in l_map:
        gamma_mask[l_map['en']] = 0.0
    gamma = gamma_raw * gamma_mask
    tau   = tau_raw   * tau_mask
    alpha = np.exp(alpha_locs)      # stored as log in AutoNormal

    si = df[sc].map(s_map).values
    pi = df['id'].map(p_map).values
    li = df['language'].map(l_map).values

    logits = alpha[pi] * (theta_locs[si] - (beta_locs[pi] + gamma[li] + tau[pi, li]))
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_param_comparison(theta_df, gamma_df, tau_df, metrics_df):
    fig = plt.figure(figsize=(14, 4.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── θ ──────────────────────────────────────────────────────────
    ax  = fig.add_subplot(gs[0])
    rho = metrics_df.loc[metrics_df['param'] == 'θ (model ability)', 'spearman_rho'].iloc[0]
    ax.scatter(theta_df['theta_IRT'], theta_df['u_GLMM'],
               s=40, color=C_BLUE, alpha=0.75, edgecolors='none')
    for _, row in theta_df.iterrows():
        ax.annotate(row['model'], (row['theta_IRT'], row['u_GLMM']),
                    fontsize=4.5, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')
    lo = min(theta_df['theta_IRT'].min(), theta_df['u_GLMM'].min()) - 0.1
    hi = max(theta_df['theta_IRT'].max(), theta_df['u_GLMM'].max()) + 0.1
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('IRT  θ_j', fontsize=9)
    ax.set_ylabel('GLMM  u_j', fontsize=9)
    ax.set_title(f'Model Ability  (ρ = {rho:.3f})', fontsize=9, fontweight='bold')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')

    # ── γ ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    r  = metrics_df.loc[metrics_df['param'] == 'γ (language shift)', 'pearson_r'].iloc[0]
    ax.scatter(gamma_df['gamma_IRT'], gamma_df['w_GLMM'],
               s=60, color=C_RED, alpha=0.85, edgecolors='black', linewidths=0.5)
    for _, row in gamma_df.iterrows():
        ax.annotate(row['language'], (row['gamma_IRT'], row['w_GLMM']),
                    fontsize=7.5, xytext=(4, 4), textcoords='offset points')
    lo = min(gamma_df['gamma_IRT'].min(), gamma_df['w_GLMM'].min()) - 0.05
    hi = max(gamma_df['gamma_IRT'].max(), gamma_df['w_GLMM'].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('IRT  γ_L', fontsize=9)
    ax.set_ylabel('GLMM  w_L', fontsize=9)
    ax.set_title(f'Language Shift  (r = {r:.3f})', fontsize=9, fontweight='bold')

    # ── τ ── colored by language ────────────────────────────────────
    ax    = fig.add_subplot(gs[2])
    r     = metrics_df.loc[metrics_df['param'] == 'τ (cross-lingual gap)', 'pearson_r'].iloc[0]
    langs = tau_df['language'].unique()
    cmap  = plt.cm.get_cmap('tab10', len(langs))
    lang_color = {l: cmap(i) for i, l in enumerate(langs)}
    for lang, grp in tau_df.groupby('language'):
        ax.scatter(grp['tau_IRT'], grp['x_GLMM'],
                   s=6, alpha=0.35, color=lang_color[lang],
                   edgecolors='none', label=lang)
    lo = min(tau_df['tau_IRT'].min(), tau_df['x_GLMM'].min()) - 0.05
    hi = max(tau_df['tau_IRT'].max(), tau_df['x_GLMM'].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('IRT  τ_iL', fontsize=9)
    ax.set_ylabel('GLMM  x_iL', fontsize=9)
    ax.set_title(f'Cross-lingual Gap  (r = {r:.3f})', fontsize=9, fontweight='bold')
    ax.legend(fontsize=5, ncol=2, loc='upper left')

    fig.suptitle('GLMM Random Effects vs. IRT Parameters', fontsize=11, fontweight='bold')
    path = os.path.join(RESULTS_DIR, "param_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: param_comparison.png")


def plot_predictive(pred_df):
    metrics = ['AUC', 'Accuracy', 'Brier', 'LogLoss']
    models  = pred_df.index.tolist()
    colors  = [C_BLUE, C_RED]
    lower_better = {'Brier', 'LogLoss'}

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, metric in zip(axes, metrics):
        vals = [pred_df.loc[m, metric] for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric, fontweight='bold', fontsize=10)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        note = '↓ better' if metric in lower_better else '↑ better'
        ax.set_ylabel(note, fontsize=8)
        ax.grid(axis='y', alpha=0.25)
        ax.set_ylim(0, max(vals) * 1.15)

    fig.suptitle('Predictive Performance: GLMM vs. IRT 2PL (in-sample)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "predictive_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: predictive_comparison.png")


def plot_model_ranking(theta_df):
    """Rank scatter: both models rank the same 61 models — do they agree?"""
    theta_df = theta_df.copy()
    theta_df['rank_IRT']  = theta_df['theta_IRT'].rank(ascending=False)
    theta_df['rank_GLMM'] = theta_df['u_GLMM'].rank(ascending=False)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(theta_df['rank_IRT'], theta_df['rank_GLMM'],
               s=35, color=C_BLUE, alpha=0.75, edgecolors='black', linewidths=0.4)
    for _, row in theta_df.iterrows():
        if abs(row['rank_IRT'] - row['rank_GLMM']) > 5:   # annotate large disagreements
            ax.annotate(row['model'], (row['rank_IRT'], row['rank_GLMM']),
                        fontsize=5, xytext=(4, 4), textcoords='offset points')
    n = len(theta_df)
    ax.plot([1, n], [1, n], 'k--', lw=0.8, alpha=0.4)
    rho, _ = spearmanr(theta_df['rank_IRT'], theta_df['rank_GLMM'])
    ax.set_xlabel('Safety rank — IRT 2PL  (1 = safest)', fontsize=9)
    ax.set_ylabel('Safety rank — GLMM  (1 = safest)', fontsize=9)
    ax.set_title(f'Model Safety Ranking Agreement\nSpearman ρ = {rho:.3f}', fontweight='bold')
    ax.set_aspect('equal')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "model_ranking_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: model_ranking_comparison.png")


def plot_tau_by_language(tau_df):
    """Per-language τ scatter: IRT vs GLMM, one panel per language."""
    non_en = sorted(tau_df['language'].unique())
    ncols  = min(3, len(non_en))
    nrows  = (len(non_en) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 3.8*nrows),
                             squeeze=False)
    for idx, lang in enumerate(non_en):
        ax  = axes[idx // ncols][idx % ncols]
        sub = tau_df[tau_df['language'] == lang]
        r, _ = pearsonr(sub['tau_IRT'], sub['x_GLMM']) if len(sub) >= 3 else (np.nan, None)
        ax.scatter(sub['tau_IRT'], sub['x_GLMM'],
                   s=10, alpha=0.5, color=C_PURPLE, edgecolors='none')
        lo = min(sub['tau_IRT'].min(), sub['x_GLMM'].min()) - 0.05
        hi = max(sub['tau_IRT'].max(), sub['x_GLMM'].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7, alpha=0.4)
        ax.set_xlabel('IRT  τ_iL', fontsize=8)
        ax.set_ylabel('GLMM  x_iL', fontsize=8)
        ax.set_title(f'{lang}  (r = {r:.3f})', fontsize=8, fontweight='bold')

    for j in range(idx+1, nrows*ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle('τ vs x_iL  by Language', fontsize=11, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "tau_by_language.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: tau_by_language.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if _HAS_FIG_STYLE:
        apply_style()

    print("=" * 65)
    print("GLMM BASELINE vs. 2PL IRT")
    print("=" * 65)

    df, sc, students, prompts, languages = load_data()
    irt  = load_irt(students, prompts, languages)
    glmm = fit_glmm(df, sc, students, prompts, languages)

    # Align parameters
    print("\nBuilding comparison tables...")
    theta_df, gamma_df, tau_df = build_comparison_tables(
        irt, glmm, students, prompts, languages)

    # Metrics
    metrics_df  = compute_metrics(theta_df, gamma_df, tau_df)
    irt_probs   = irt_predictions(df, sc, students, prompts, languages, irt)
    pred_df     = compute_predictive_metrics(df, irt_probs, glmm['probs'])

    # Save CSVs
    theta_df.to_csv(os.path.join(RESULTS_DIR, "theta_comparison.csv"),   index=False)
    gamma_df.to_csv(os.path.join(RESULTS_DIR, "gamma_comparison.csv"),   index=False)
    tau_df.to_csv(  os.path.join(RESULTS_DIR, "tau_comparison.csv"),     index=False)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "param_correspondence.csv"), index=False)
    pred_df.to_csv(   os.path.join(RESULTS_DIR, "predictive_metrics.csv"))

    # Print summaries
    print(f"\n{'=' * 65}")
    print("PARAMETER CORRESPONDENCE  (GLMM vs IRT)")
    print(f"{'=' * 65}")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n{'=' * 65}")
    print("PREDICTIVE PERFORMANCE  (in-sample)")
    print(f"{'=' * 65}")
    print(pred_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Plots
    print("\nGenerating figures...")
    plot_param_comparison(theta_df, gamma_df, tau_df, metrics_df)
    plot_predictive(pred_df)
    plot_model_ranking(theta_df)
    plot_tau_by_language(tau_df)

    print(f"\n{'=' * 65}")
    print("WHAT IRT ADDS OVER GLMM")
    print(f"{'=' * 65}")
    print("  1. Discrimination α_i  — which prompts are most diagnostic")
    print("     (no GLMM equivalent; GLMM treats all prompts equally)")
    print("  2. Horseshoe prior on τ — most CSGs shrunk to ≈ 0 (sparse)")
    print("     GLMM uses isotropic Normal → inflated interaction estimates")
    print("  3. Anchor identification — γ and τ jointly identified via")
    print("     DIF-validated constraints; GLMM can conflate the two")

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
