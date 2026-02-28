# -*- coding: utf-8 -*-
"""
Experiment B: Test-Retest Reliability and Robustness Analysis
=============================================================
Validates that IRT parameter estimates (θ, β, τ) are stable across
independent samples of the same data-generating process. Exploits the
10 generation passes in the dataset.

Sub-experiments:
  B1 — Response consistency across passes
       Computes empirical P(safe) per (model, prompt, language) triple.
       Shows bimodal distribution and per-language entropy.
       Produces: B1_response_consistency.png, B1_psafe_by_language_violin.png

  B2 — Split-half reliability
       Randomly partitions observations into two halves, fits IRT on each,
       correlates matched θ, β, τ. Reports Spearman-Brown correction.
       Note: Currently splits rows, not passes — see code comment.
       Produces: B2_split_half_reliability.png, B2_split_half_summary.csv

  B3 — Intraclass correlation coefficients (ICC)
       Computes ICC(1,1), ICC(2,1), ICC(2,k) per language from pass-level
       agreement on safety scores.
       Produces: B3_icc_by_language.png, B3_icc_results.csv

  B4 — Stochastic safety profiles
       Classifies triples into deterministic/boundary categories. Shows that
       low-resource languages cluster at high entropy / high boundary %.
       Produces: B4_stochastic_profiles.png, B4_prompt_entropy_ranking.csv

  B5 — Calibration against empirical response rates
       Compares IRT-predicted P(safe) to observed rates across 10 passes.
       Overall r, per-language r, residual distribution, example ICCs.
       Produces: B5_calibration.png, B5_example_ICCs.png

  B6 — Temperature variance decomposition
       Decomposes total response variance into between-temperature (2.6%)
       and within-temperature components. Validates variant design.
       Produces: B6_temperature_decomposition.png

  B7 — Pass-to-pass τ stability
       Fits IRT on 3 independent data partitions, correlates τ estimates.
       Per-language stability and scatter plots.
       Produces: B7_tau_stability.png, B7_tau_diff_distribution.png
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
# ── fig_style integration ──
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
try:
    from fig_style import (apply_style, savefig as fs_savefig, make_fig, make_fig_grid,
                           C_RED, C_BLUE, C_PURPLE, COLORS_3, CMAP_DIV, CMAP_SEQ,
                           FAM_COLORS as FS_FAM_COLORS, FAM_ORDER as FS_FAM_ORDER,
                           LABELS, LANG_ORDER, FULL_WIDTH, DPI, ASPECT,
                           get_family, get_family_color, add_identity_line)
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False
    print("[WARN] fig_style.py not found - using defaults")
from tqdm import tqdm
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import os
import warnings
import re
import pickle

warnings.filterwarnings('ignore')

from huggingface_hub import snapshot_download

DATA_DIR = snapshot_download(
    repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False
)
INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors", "anchors.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_experiment_B")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_POSTERIOR_SAMPLES = 500
SEED = 42
MAX_TRAINING_STEPS = 6000          # match A and D
CONVERGENCE_WINDOW = 200
CONVERGENCE_THRESHOLD = 1e-4
MIN_TRAINING_STEPS = 1500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def check_convergence(losses, window, threshold, min_steps):
    """
    Check if training has converged.
    Returns True if relative improvement over last `window` steps < threshold.
    """
    if len(losses) < min_steps:
        return False
    if len(losses) < 2 * window:
        return False

    # Compare mean loss of last window vs previous window
    recent = np.mean(losses[-window:])
    previous = np.mean(losses[-2 * window:-window])

    if previous == 0:
        return True

    relative_improvement = (previous - recent) / abs(previous)
    return relative_improvement < threshold


def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def extract_pass_number(test_taker_str):
    match = re.search(r'pass[_-]?(\d+)', str(test_taker_str), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_base_model(test_taker_str):
    return re.sub(r'[_-]?pass[_-]?\d+', '', str(test_taker_str), flags=re.IGNORECASE).strip()


def extract_temperature_setting(test_taker_str):
    s = str(test_taker_str)
    if 'Low_Creativity' in s or 'Low-Creativity' in s:
        return 'Low_Creativity'
    elif 'High_Risk' in s or 'High-Risk' in s:
        return 'High_Risk'
    elif 'Chaos' in s:
        return 'Chaos'
    elif 'Standard' in s:
        return 'Standard'
    elif 'reasoning' in s.lower():
        return 'Reasoning'
    return 'Unknown'


def _to_scalar(val):
    if hasattr(val, 'item'):
        return float(val.item())
    if hasattr(val, '__len__'):
        return float(val.flat[0])
    return float(val)


def compute_icc(df_wide):
    data = df_wide.values
    n, k = data.shape
    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]
    n = data.shape[0]
    if n < 2 or k < 2:
        return np.nan, np.nan, np.nan
    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1) if k > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1e-10
    icc_21 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    icc_2k = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
    icc_11 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
    return icc_11, icc_21, icc_2k


# ══════════════════════════════════════════════════════════════════════════
# IRT MODEL + UNIFIED FITTER (shape-safe)
# ══════════════════════════════════════════════════════════════════════════

def model_2pl(student_idx, prompt_idx, lang_idx, obs=None,
              num_students=None, num_prompts=None, num_langs=None,
              tau_mask=None, gamma_mask=None):
    """2PL: P(safe) = σ(α_i · (θ_j − (β_i + γ_L + τ_iL)))"""
    theta = pyro.sample("theta",
        dist.Normal(torch.zeros(num_students, device=device), 1.0).to_event(1))
    beta = pyro.sample("beta",
        dist.Normal(torch.zeros(num_prompts, device=device), 1.0).to_event(1))
    log_alpha = pyro.sample("log_alpha",
        dist.Normal(torch.zeros(num_prompts, device=device), 0.5).to_event(1))
    alpha = pyro.deterministic("alpha", torch.exp(log_alpha))
    gamma_raw = pyro.sample("gamma_raw",
        dist.Normal(torch.zeros(num_langs, device=device), 1.0).to_event(1))
    gamma = pyro.deterministic("gamma", gamma_raw * gamma_mask)
    tau_scale = pyro.sample("tau_scale",
        dist.HalfCauchy(torch.ones(1, device=device)).to_event(1))
    tau_raw = pyro.sample("tau_raw",
        dist.StudentT(1.0, torch.zeros(num_prompts, num_langs, device=device),
                      tau_scale).to_event(2))
    tau = pyro.deterministic("tau", tau_raw * tau_mask)
    delta_raw = pyro.sample("delta_raw",
        dist.Normal(torch.zeros(num_students, num_langs, device=device), 0.5).to_event(2))
    delta_mask = gamma_mask.unsqueeze(0).expand(num_students, -1)
    delta = pyro.deterministic("delta", delta_raw * delta_mask)

    with pyro.plate("data", len(student_idx)):
        ability = theta[student_idx] + delta[student_idx, lang_idx]
        difficulty = beta[prompt_idx] + gamma[lang_idx] + tau[prompt_idx, lang_idx]
        logits = alpha[prompt_idx] * (ability - difficulty)
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)

def fit_irt(df_subset, anchor_ids, label="full", cache_key=None):
    if cache_key:
        cache_path = os.path.join(RESULTS_DIR, f"_cache_{cache_key}.pkl")
        if os.path.exists(cache_path):
            print(f"    ★ Loading cached IRT [{label}] from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    pyro.clear_param_store()

    student_col = 'test_taker' if 'test_taker' in df_subset.columns else 'model'
    students = sorted(df_subset[student_col].unique())
    prompts = sorted(df_subset['id'].unique())
    languages = sorted(df_subset['language'].unique())

    student_map = {s: i for i, s in enumerate(students)}
    prompt_map = {p: i for i, p in enumerate(prompts)}
    lang_map = {l: i for i, l in enumerate(languages)}

    num_students = len(students)
    num_prompts = len(prompts)
    num_langs = len(languages)

    student_idx = torch.tensor(df_subset[student_col].map(student_map).values, dtype=torch.long).to(device)
    prompt_idx = torch.tensor(df_subset['id'].map(prompt_map).values, dtype=torch.long).to(device)
    lang_idx = torch.tensor(df_subset['language'].map(lang_map).values, dtype=torch.long).to(device)
    score_obs = torch.tensor(df_subset['score'].values, dtype=torch.float32).to(device)

    tau_mask = torch.ones((num_prompts, num_langs), device=device)
    gamma_mask = torch.ones(num_langs, device=device)
    if 'en' in lang_map:
        en_i = lang_map['en']
        tau_mask[:, en_i] = 0.0
        gamma_mask[en_i] = 0.0
    for pid in prompts:
        if pid in anchor_ids and pid in prompt_map:
            tau_mask[prompt_map[pid], :] = 0.0

    # CHANGED 1: use model_2pl, hide alpha from guide
    guide = pyro.infer.autoguide.AutoNormal(
        pyro.poutine.block(model_2pl, hide=["obs", "tau", "gamma", "delta", "alpha"]))
    optimizer = ClippedAdam({"lr": 0.005, "clip_norm": 10.0})  # CHANGED 2: lr 0.01 → 0.005 (2PL needs lower)
    svi = SVI(model_2pl, guide, optimizer, loss=Trace_ELBO())

    losses = []
    converged_at = None
    pbar = tqdm(range(MAX_TRAINING_STEPS), desc=f"IRT [{label}]", leave=False)

    for step in pbar:
        loss = svi.step(student_idx, prompt_idx, lang_idx, score_obs,
                        num_students, num_prompts, num_langs, tau_mask, gamma_mask)
        losses.append(loss)
        if step % 200 == 0:
            pbar.set_description(f"[{label}] Loss: {loss:.1f}")
        if check_convergence(losses, CONVERGENCE_WINDOW, CONVERGENCE_THRESHOLD, MIN_TRAINING_STEPS):
            converged_at = step + 1
            pbar.close()
            break

    if converged_at is None:
        converged_at = MAX_TRAINING_STEPS

    # CHANGED 3: add alpha to return_sites
    predictive = Predictive(model_2pl, guide=guide, num_samples=N_POSTERIOR_SAMPLES,
                            return_sites=["theta", "beta", "gamma", "tau", "delta", "alpha"])
    samples = predictive(student_idx, prompt_idx, lang_idx, None,
                         num_students, num_prompts, num_langs, tau_mask, gamma_mask)

    theta_mean = samples['theta'].detach().cpu().numpy().mean(axis=0).reshape(num_students).astype(np.float64)
    theta_std = samples['theta'].detach().cpu().numpy().std(axis=0).reshape(num_students).astype(np.float64)
    beta_mean = samples['beta'].detach().cpu().numpy().mean(axis=0).reshape(num_prompts).astype(np.float64)
    beta_std = samples['beta'].detach().cpu().numpy().std(axis=0).reshape(num_prompts).astype(np.float64)
    gamma_mean = samples['gamma'].detach().cpu().numpy().mean(axis=0).reshape(num_langs).astype(np.float64)
    tau_mean = samples['tau'].detach().cpu().numpy().mean(axis=0).reshape(num_prompts, num_langs).astype(np.float64)
    tau_std = samples['tau'].detach().cpu().numpy().std(axis=0).reshape(num_prompts, num_langs).astype(np.float64)
    delta_mean = samples['delta'].detach().cpu().numpy().mean(axis=0).reshape(num_students, num_langs).astype(np.float64)
    # CHANGED 3b: extract alpha
    alpha_mean = samples['alpha'].detach().cpu().numpy().mean(axis=0).reshape(num_prompts).astype(np.float64)

    result = {
        'theta_mean': theta_mean, 'theta_std': theta_std,
        'beta_mean': beta_mean, 'beta_std': beta_std,
        'gamma_mean': gamma_mean,
        'tau_mean': tau_mean, 'tau_std': tau_std,
        'delta_mean': delta_mean,
        'alpha_mean': alpha_mean,  # CHANGED 3c: include in results
        'student_map': student_map, 'prompt_map': prompt_map, 'lang_map': lang_map,
        'num_students': num_students, 'num_prompts': num_prompts, 'num_langs': num_langs,
        'losses': losses, 'converged_at': converged_at,
    }
    if cache_key:
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"    ★ Cached IRT [{label}] to {cache_path}")
    return result

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_and_prepare_data():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    print(f"  Raw rows: {len(df):,}")
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['score'] = (df['judge_score'] >= 4).astype(np.float32)
    df['id'] = df['id'].apply(clean_id)
    print(f"  Valid rows (score > 0): {len(df):,}")

    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    df['pass_num'] = df[student_col].apply(extract_pass_number)
    df['base_model'] = df[student_col].apply(extract_base_model)
    df['temp_setting'] = df[student_col].apply(extract_temperature_setting)

    print(f"  Pass distribution:\n{df['pass_num'].value_counts().sort_index()}")
    print(f"  Unique base models: {df['base_model'].nunique()}")
    print(f"  Unique test_takers: {df[student_col].nunique()}")
    print(f"  Languages: {sorted(df['language'].unique())}")

    anchor_ids = set()
    if os.path.exists(ANCHOR_FILE):
        anchors_df = pd.read_csv(ANCHOR_FILE)
        anchors_df['id'] = anchors_df['id'].apply(clean_id)
        anchor_ids = set(anchors_df['id'].unique())
        print(f"  Loaded {len(anchor_ids)} anchor prompts")

    return df, anchor_ids



# ══════════════════════════════════════════════════════════════════════════
# B1: Response Consistency
# ══════════════════════════════════════════════════════════════════════════


def b1_response_consistency(df):
    print("\n" + "=" * 70)
    print("B1: RESPONSE CONSISTENCY ANALYSIS")
    print("=" * 70)

    consistency = df.groupby(['base_model', 'id', 'language']).agg(
        n_passes=('score', 'count'), n_safe=('score', 'sum'),
        mean_safe=('score', 'mean'), std_safe=('score', 'std')
    ).reset_index()
    consistency['std_safe'] = consistency['std_safe'].fillna(0)
    p = consistency['mean_safe'].clip(1e-10, 1 - 1e-10)
    consistency['entropy'] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    print(f"\nTotal triples: {len(consistency):,}")
    print(f"Always safe (P=1.0):  {(consistency['mean_safe'] == 1.0).mean():.1%}")
    print(f"Always unsafe (P=0.0): {(consistency['mean_safe'] == 0.0).mean():.1%}")
    print(f"Mixed (0<P<1):        {((consistency['mean_safe'] > 0) & (consistency['mean_safe'] < 1)).mean():.1%}")
    print(f"Mean entropy:          {consistency['entropy'].mean():.4f}")

    _c1, _c2, _c3 = (C_BLUE, C_RED, C_PURPLE) if _HAS_FIG_STYLE else ('#2471a3', '#c0392b', '#7d3c98')
    _save = fs_savefig if _HAS_FIG_STYLE else lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))

    fig, axes = make_fig(n_panels=3, height_override=2.2) if _HAS_FIG_STYLE \
        else plt.subplots(1, 3, figsize=(5.5, 2.2))[::1]  # ensure tuple
    if not isinstance(axes, np.ndarray): axes = np.array([axes])
    axes[0].hist(consistency['mean_safe'], bins=50, edgecolor='black',
                 linewidth=0.3, alpha=0.7, color=_c1)
    axes[0].set_xlabel('Empirical $P$(safe)'); axes[0].set_ylabel('Count')
    axes[0].set_title('Response Consistency')
    axes[0].axvline(0.5, color=_c2, ls='--', alpha=0.7, lw=0.6)
    axes[1].hist(consistency['entropy'], bins=50, edgecolor='black',
                 linewidth=0.3, alpha=0.7, color=_c2)
    axes[1].set_xlabel('Entropy (bits)'); axes[1].set_ylabel('Count')
    axes[1].set_title('Response Uncertainty')
    lang_ent = consistency.groupby('language')['entropy'].mean().sort_values(ascending=False)
    axes[2].barh(lang_ent.index, lang_ent.values,
                 color=sns.color_palette("viridis", len(lang_ent)),
                 edgecolor='black', linewidth=0.3)
    axes[2].set_xlabel('Mean Entropy'); axes[2].set_title('Uncertainty by Language')
    axes[2].invert_yaxis()
    _save(fig, os.path.join(RESULTS_DIR, "B1_response_consistency.png"))
    print(f"  Saved: B1_response_consistency")

    fig, ax = make_fig(n_panels=1, height_override=2.8) if _HAS_FIG_STYLE \
        else plt.subplots(figsize=(5.5, 2.8))
    if isinstance(ax, np.ndarray): ax = ax[0]
    mixed = consistency[consistency['n_passes'] >= 2]
    lang_order = mixed.groupby('language')['mean_safe'].mean().sort_values().index.tolist()
    sns.violinplot(data=mixed, x='language', y='mean_safe', order=lang_order,
                   palette='Set2', inner='quartile', ax=ax, cut=0, linewidth=0.5)
    ax.axhline(0.5, color=_c2, ls='--', alpha=0.5, lw=0.6)
    ax.set_title('$P$(safe) by Language')
    _save(fig, os.path.join(RESULTS_DIR, "B1_psafe_by_language_violin.png"))
    print(f"  Saved: B1_psafe_by_language_violin")

    consistency.to_csv(os.path.join(RESULTS_DIR, "B1_response_consistency.csv"), index=False)
    return consistency


# ══════════════════════════════════════════════════════════════════════════
# B2: Split-Half Reliability (FIXED)
# ══════════════════════════════════════════════════════════════════════════

def b2_split_half_reliability(df, anchor_ids):
    print("\n" + "=" * 70)
    print("B2: SPLIT-HALF RELIABILITY OF θ")
    print("=" * 70)

    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'

    np.random.seed(SEED)
    indices = np.random.permutation(len(df))
    mid = len(indices) // 2
    df_h1 = df.iloc[indices[:mid]].copy()
    df_h2 = df.iloc[indices[mid:]].copy()

    common_students = set(df_h1[student_col].unique()) & set(df_h2[student_col].unique())
    common_prompts_set = set(df_h1['id'].unique()) & set(df_h2['id'].unique())
    common_langs_set = set(df_h1['language'].unique()) & set(df_h2['language'].unique())

    df_h1 = df_h1[df_h1[student_col].isin(common_students) & df_h1['id'].isin(common_prompts_set) & df_h1['language'].isin(common_langs_set)]
    df_h2 = df_h2[df_h2[student_col].isin(common_students) & df_h2['id'].isin(common_prompts_set) & df_h2['language'].isin(common_langs_set)]

    print(f"  Half 1: {len(df_h1):,}, Half 2: {len(df_h2):,}, Common students: {len(common_students)}")

    r1 = fit_irt(df_h1, anchor_ids, label="half-1", cache_key="b2_half1")
    r2 = fit_irt(df_h2, anchor_ids, label="half-2", cache_key="b2_half2")

    # Match θ
    theta_pairs = []
    for s in common_students:
        if s in r1['student_map'] and s in r2['student_map']:
            theta_pairs.append({
                'student': s,
                'theta_h1': _to_scalar(r1['theta_mean'][r1['student_map'][s]]),
                'theta_h2': _to_scalar(r2['theta_mean'][r2['student_map'][s]]),
            })
    theta_df = pd.DataFrame(theta_pairs)
    if len(theta_df) < 3:
        print("  ERROR: Too few matched students."); return None

    r_theta, p_theta = pearsonr(theta_df['theta_h1'], theta_df['theta_h2'])
    sb_theta = (2 * r_theta) / (1 + r_theta)
    print(f"  θ: r={r_theta:.4f}, SB={sb_theta:.4f}")

    # Match β
    beta_pairs = []
    for p in common_prompts_set:
        if p in r1['prompt_map'] and p in r2['prompt_map']:
            beta_pairs.append({
                'prompt': p,
                'beta_h1': _to_scalar(r1['beta_mean'][r1['prompt_map'][p]]),
                'beta_h2': _to_scalar(r2['beta_mean'][r2['prompt_map'][p]]),
            })
    beta_df = pd.DataFrame(beta_pairs)
    r_beta = sb_beta = np.nan
    if len(beta_df) >= 3:
        r_beta, _ = pearsonr(beta_df['beta_h1'], beta_df['beta_h2'])
        sb_beta = (2 * r_beta) / (1 + r_beta)
        print(f"  β: r={r_beta:.4f}, SB={sb_beta:.4f}")

    # Match τ
    tau_pairs = []
    for p in common_prompts_set:
        for lang in common_langs_set:
            if lang == 'en': continue
            if (p in r1['prompt_map'] and p in r2['prompt_map'] and
                    lang in r1['lang_map'] and lang in r2['lang_map']):
                pi1, pi2 = r1['prompt_map'][p], r2['prompt_map'][p]
                li1, li2 = r1['lang_map'][lang], r2['lang_map'][lang]
                tau_pairs.append({
                    'prompt': p, 'language': lang,
                    'tau_h1': _to_scalar(r1['tau_mean'][pi1, li1]),
                    'tau_h2': _to_scalar(r2['tau_mean'][pi2, li2]),
                })
    tau_df = pd.DataFrame(tau_pairs)
    r_tau = sb_tau = np.nan
    if len(tau_df) >= 3:
        r_tau, _ = pearsonr(tau_df['tau_h1'], tau_df['tau_h2'])
        sb_tau = (2 * r_tau) / (1 + r_tau)
        print(f"  τ: r={r_tau:.4f}, SB={sb_tau:.4f}")

    # Plot
    _c1, _c2, _c3 = (C_BLUE, C_RED, C_PURPLE) if _HAS_FIG_STYLE else ('#2471a3', '#c0392b', '#7d3c98')
    _save = fs_savefig if _HAS_FIG_STYLE else lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))
    _L = LABELS if _HAS_FIG_STYLE else {'theta_short': 'θ', 'beta_short': 'β', 'tau_short': 'τ'}

    fig, axes = make_fig(n_panels=3, height_override=2.2) if _HAS_FIG_STYLE \
        else plt.subplots(1, 3, figsize=(5.5, 2.2))
    if not isinstance(axes, np.ndarray): axes = np.array([axes])
    ax = axes[0]
    ax.scatter(theta_df['theta_h1'], theta_df['theta_h2'], alpha=0.6, s=10,
               edgecolors='black', linewidth=0.2, c=_c1)
    lims = [min(theta_df['theta_h1'].min(), theta_df['theta_h2'].min()) - 0.3,
            max(theta_df['theta_h1'].max(), theta_df['theta_h2'].max()) + 0.3]
    ax.plot(lims, lims, color=_c2, ls='--', alpha=0.7, lw=0.6)
    ax.set_xlabel(f'{_L["theta_short"]} (Half 1)')
    ax.set_ylabel(f'{_L["theta_short"]} (Half 2)')
    ax.set_title(f'Split-Half: {_L["theta_short"]}\n$r$ = {r_theta:.3f}, SB = {sb_theta:.3f}')

    ax = axes[1]
    if len(beta_df) >= 3:
        ax.scatter(beta_df['beta_h1'], beta_df['beta_h2'], alpha=0.4, s=4,
                   color=_c3)
        lims_b = [min(beta_df['beta_h1'].min(), beta_df['beta_h2'].min()) - 0.3,
                  max(beta_df['beta_h1'].max(), beta_df['beta_h2'].max()) + 0.3]
        ax.plot(lims_b, lims_b, color=_c2, ls='--', alpha=0.7, lw=0.6)
        ax.set_title(f'Split-Half: {_L["beta_short"]}\n$r$ = {r_beta:.3f}, SB = {sb_beta:.3f}')
    ax.set_xlabel(f'{_L["beta_short"]} (Half 1)')
    ax.set_ylabel(f'{_L["beta_short"]} (Half 2)')

    ax = axes[2]
    if len(tau_df) >= 3:
        lang_colors = {l: plt.cm.Set2(i / max(1, len(common_langs_set)))
                       for i, l in enumerate(sorted(common_langs_set))}
        for lang in sorted(common_langs_set):
            if lang == 'en': continue
            sub = tau_df[tau_df['language'] == lang]
            ax.scatter(sub['tau_h1'], sub['tau_h2'], alpha=0.3, s=3,
                       label=lang, color=lang_colors.get(lang))
        lims_t = [min(tau_df['tau_h1'].min(), tau_df['tau_h2'].min()) - 0.3,
                  max(tau_df['tau_h1'].max(), tau_df['tau_h2'].max()) + 0.3]
        ax.plot(lims_t, lims_t, color=_c2, ls='--', alpha=0.7, lw=0.6)
        ax.set_title(f'Split-Half: {_L["tau_short"]}\n$r$ = {r_tau:.3f}, SB = {sb_tau:.3f}')
        ax.legend(fontsize=4, ncol=3, loc='upper left')
    ax.set_xlabel(f'{_L["tau_short"]} (Half 1)')
    ax.set_ylabel(f'{_L["tau_short"]} (Half 2)')

    _save(fig, os.path.join(RESULTS_DIR, "B2_split_half_reliability.png"))
    print(f"  Saved: B2_split_half_reliability")

    pd.DataFrame([
        {'parameter': 'θ', 'pearson_r': r_theta, 'spearman_brown': sb_theta, 'n': len(theta_df)},
        {'parameter': 'β', 'pearson_r': r_beta, 'spearman_brown': sb_beta, 'n': len(beta_df)},
        {'parameter': 'τ', 'pearson_r': r_tau, 'spearman_brown': sb_tau, 'n': len(tau_df)},
    ]).to_csv(os.path.join(RESULTS_DIR, "B2_split_half_summary.csv"), index=False)

    return {'theta_df': theta_df, 'beta_df': beta_df, 'tau_df': tau_df}


# ══════════════════════════════════════════════════════════════════════════
# B3: ICC Analysis
# ══════════════════════════════════════════════════════════════════════════

def b3_icc_analysis(df):
    print("\n" + "=" * 70)
    print("B3: INTRACLASS CORRELATION COEFFICIENTS")
    print("=" * 70)
    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    pass_col = student_col if df['pass_num'].isna().all() else 'pass_num'
    if pass_col == student_col:
        print("  Pass numbers unavailable. Using test_taker as pass proxy.")
    grouped = df.groupby(['base_model', 'id', 'language', pass_col])['score'].mean().reset_index()
    icc_results = []
    for lang in sorted(df['language'].unique()):
        lang_data = grouped[grouped['language'] == lang]
        try:
            wide = lang_data.pivot_table(index=['base_model', 'id'], columns=pass_col, values='score')
        except: continue
        if wide.shape[1] < 2 or wide.shape[0] < 10: continue
        wide_clean = wide.dropna(thresh=max(2, wide.shape[1] // 2))
        if len(wide_clean) < 10: continue
        icc_11, icc_21, icc_2k = compute_icc(wide_clean)
        icc_results.append({'language': lang, 'ICC(1,1)': icc_11, 'ICC(2,1)': icc_21, 'ICC(2,k)': icc_2k, 'n_triples': len(wide_clean), 'n_passes': wide_clean.shape[1]})
        print(f"  {lang}: ICC(2,1) = {icc_21:.4f}, ICC(2,k) = {icc_2k:.4f}")
    icc_df = pd.DataFrame(icc_results)
    if len(icc_df) == 0:
        print("  WARNING: Could not compute ICC."); return None
    _c1, _c2 = (C_BLUE, C_RED) if _HAS_FIG_STYLE else ('#2471a3', '#c0392b')
    _save = fs_savefig if _HAS_FIG_STYLE else lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))

    fig, ax = make_fig(n_panels=1, height_override=3.0) if _HAS_FIG_STYLE \
        else plt.subplots(figsize=(5.5, 3.0))
    if isinstance(ax, np.ndarray): ax = ax[0]
    icc_sorted = icc_df.sort_values('ICC(2,1)', ascending=True)
    colors = [_c1 if v > 0.75 else '#e67e22' if v > 0.5 else _c2
              for v in icc_sorted['ICC(2,1)']]
    bars = ax.barh(icc_sorted['language'], icc_sorted['ICC(2,1)'],
                   color=colors, edgecolor='black', linewidth=0.3)
    ax.axvline(0.75, color=_c1, ls='--', alpha=0.5, lw=0.6, label='Excellent')
    ax.axvline(0.5, color='#e67e22', ls='--', alpha=0.5, lw=0.6, label='Moderate')
    for bar, val in zip(bars, icc_sorted['ICC(2,1)']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=6)
    ax.set_xlabel('ICC(2,1)'); ax.set_title('Test-Retest Reliability by Language')
    ax.legend(fontsize=5); ax.set_xlim(0, 1.05)
    _save(fig, os.path.join(RESULTS_DIR, "B3_icc_by_language.png"))
    icc_df.to_csv(os.path.join(RESULTS_DIR, "B3_icc_results.csv"), index=False)
    return icc_df


# ══════════════════════════════════════════════════════════════════════════
# B4: Stochastic Safety Profiles
# ══════════════════════════════════════════════════════════════════════════

def b4_stochastic_profiles(df, consistency_df):
    print("\n" + "=" * 70)
    print("B4: STOCHASTIC SAFETY PROFILES")
    print("=" * 70)
    multi_pass = consistency_df[consistency_df['n_passes'] >= 3].copy()
    multi_pass['is_boundary'] = (multi_pass['mean_safe'] > 0.2) & (multi_pass['mean_safe'] < 0.8)
    n_total = len(multi_pass)
    print(f"  Det. Safe: {(multi_pass['mean_safe'] == 1.0).sum():,} ({(multi_pass['mean_safe'] == 1.0).mean():.1%})")
    print(f"  Det. Unsafe: {(multi_pass['mean_safe'] == 0.0).sum():,}")
    print(f"  Boundary: {multi_pass['is_boundary'].sum():,} ({multi_pass['is_boundary'].mean():.1%})")

    boundary_by_lang = multi_pass.groupby('language').agg(total=('is_boundary', 'count'), n_boundary=('is_boundary', 'sum'), mean_entropy=('entropy', 'mean')).reset_index()
    boundary_by_lang['pct_boundary'] = boundary_by_lang['n_boundary'] / boundary_by_lang['total']
    for _, row in boundary_by_lang.sort_values('pct_boundary', ascending=False).iterrows():
        print(f"    {row['language']}: {row['pct_boundary']:.1%} boundary")

    boundary_by_model = multi_pass.groupby('base_model').agg(total=('is_boundary', 'count'), n_boundary=('is_boundary', 'sum'), mean_entropy=('entropy', 'mean')).reset_index()
    boundary_by_model['pct_boundary'] = boundary_by_model['n_boundary'] / boundary_by_model['total']
    print(f"\n  Top 10 boundary models:")
    for _, row in boundary_by_model.sort_values('pct_boundary', ascending=False).head(10).iterrows():
        print(f"    {row['base_model']:<50}: {row['pct_boundary']:.1%}")

    _c1, _c2, _c3 = (C_BLUE, C_RED, C_PURPLE) if _HAS_FIG_STYLE else ('#2471a3', '#c0392b', '#7d3c98')
    _save = fs_savefig if _HAS_FIG_STYLE else lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))

    fig, axes = make_fig(n_panels=2, height_override=3.0) if _HAS_FIG_STYLE \
        else plt.subplots(1, 2, figsize=(5.5, 3.0))
    if not isinstance(axes, np.ndarray): axes = np.array([axes])
    lang_comp = []
    for lang in sorted(multi_pass['language'].unique()):
        ld = multi_pass[multi_pass['language'] == lang]; total = len(ld)
        lang_comp.append({'language': lang,
            'Det. Safe': (ld['mean_safe'] == 1.0).sum()/total,
            'Mostly Safe': ((ld['mean_safe'] >= 0.8) & (ld['mean_safe'] < 1.0)).sum()/total,
            'Boundary': ld['is_boundary'].sum()/total,
            'Mostly Unsafe': ((ld['mean_safe'] > 0) & (ld['mean_safe'] <= 0.2)).sum()/total,
            'Det. Unsafe': (ld['mean_safe'] == 0.0).sum()/total})
    pd.DataFrame(lang_comp).set_index('language').plot(
        kind='barh', stacked=True, ax=axes[0],
        color=[_c1, '#82c6b5', '#f0d070', '#e67e22', _c2])
    axes[0].set_title('Composition by Language')
    
    # CHANGED: Moved legend to upper center beneath the plot to avoid overlapping bars
    axes[0].legend(fontsize=4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    for _, row in boundary_by_lang.iterrows():
        axes[1].scatter(row['pct_boundary'], row['mean_entropy'],
                        s=25, color=_c3, edgecolors='black', linewidth=0.3)
        axes[1].annotate(row['language'],
                         (row['pct_boundary'], row['mean_entropy']),
                         fontsize=5, xytext=(2, 2), textcoords='offset points')
    axes[1].set_xlabel('\\% Boundary'); axes[1].set_ylabel('Mean Entropy')
    axes[1].set_title('Uncertainty Profile')
    
    fig.tight_layout()
    _save(fig, os.path.join(RESULTS_DIR, "B4_stochastic_profiles.png"))

    prompt_entropy = multi_pass.groupby('id').agg(mean_entropy=('entropy', 'mean'), mean_psafe=('mean_safe', 'mean'), n_models=('base_model', 'nunique'), n_langs=('language', 'nunique')).reset_index().sort_values('mean_entropy', ascending=False)
    print(f"\n  Top 10 uncertain prompts:")
    for _, row in prompt_entropy.head(10).iterrows():
        print(f"    Prompt {row['id']}: entropy={row['mean_entropy']:.3f}, P(safe)={row['mean_psafe']:.2f}")
    multi_pass.to_csv(os.path.join(RESULTS_DIR, "B4_stochastic_profiles.csv"), index=False)
    prompt_entropy.to_csv(os.path.join(RESULTS_DIR, "B4_prompt_entropy_ranking.csv"), index=False)
    return multi_pass, prompt_entropy


# ══════════════════════════════════════════════════════════════════════════
# B5: Empirical ICC Validation (FIXED)
# ══════════════════════════════════════════════════════════════════════════

def b5_empirical_icc_validation(df, consistency_df, anchor_ids):
    print("\n" + "=" * 70)
    print("B5: EMPIRICAL ICC VALIDATION")
    print("=" * 70)

    results = fit_irt(df, anchor_ids, label="full-B5", cache_key="b5_full")

    validation_rows = []
    skipped = {'no_student': 0, 'no_prompt': 0, 'no_lang': 0}

    for _, row in tqdm(consistency_df.iterrows(), total=len(consistency_df), desc="  Validating", leave=False):
        base_model, prompt_id, lang = row['base_model'], str(row['id']), row['language']
        empirical_p, n_passes = float(row['mean_safe']), int(row['n_passes'])
        if n_passes < 2: continue

        matching_students = [s for s in results['student_map'].keys() if extract_base_model(s) == base_model]
        if not matching_students: skipped['no_student'] += 1; continue
        if prompt_id not in results['prompt_map']: skipped['no_prompt'] += 1; continue
        if lang not in results['lang_map']: skipped['no_lang'] += 1; continue

        p_idx = results['prompt_map'][prompt_id]
        l_idx = results['lang_map'][lang]
        beta_i = _to_scalar(results['beta_mean'][p_idx])
        gamma_l = _to_scalar(results['gamma_mean'][l_idx])
        tau_il = _to_scalar(results['tau_mean'][p_idx, l_idx])

        thetas = []
        for s in matching_students:
            s_idx = results['student_map'][s]
            th = _to_scalar(results['theta_mean'][s_idx])
            dl = _to_scalar(results['delta_mean'][s_idx, l_idx])
            thetas.append(th + dl)
        if not thetas: continue

        theta_avg = np.mean(thetas)
        alpha_i = _to_scalar(results['alpha_mean'][p_idx])
        logit = alpha_i * (theta_avg - (beta_i + gamma_l + tau_il))
        irt_p = 1.0 / (1.0 + np.exp(-logit))

        validation_rows.append({'base_model': base_model, 'prompt_id': prompt_id, 'language': lang, 'empirical_p': empirical_p, 'irt_p': irt_p, 'n_passes': n_passes, 'theta': theta_avg, 'difficulty': beta_i + gamma_l + tau_il})

    val_df = pd.DataFrame(validation_rows)
    print(f"  Validation pairs: {len(val_df):,}, Skipped: {skipped}")
    if len(val_df) == 0: print("  ERROR: No pairs."); return None, None

    r_pearson, _ = pearsonr(val_df['empirical_p'], val_df['irt_p'])
    r_spearman, _ = spearmanr(val_df['empirical_p'], val_df['irt_p'])
    rmse = np.sqrt(np.mean((val_df['empirical_p'] - val_df['irt_p']) ** 2))
    mae = np.mean(np.abs(val_df['empirical_p'] - val_df['irt_p']))
    print(f"  r={r_pearson:.4f}, ρ={r_spearman:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    for lang in sorted(val_df['language'].unique()):
        sub = val_df[val_df['language'] == lang]
        if len(sub) >= 10:
            r_l, _ = pearsonr(sub['empirical_p'], sub['irt_p'])
            print(f"    {lang}: r={r_l:.4f}, n={len(sub):,}")

    # Plot calibration
    _c1, _c2, _c3 = (C_BLUE, C_RED, C_PURPLE) if _HAS_FIG_STYLE else ('#2471a3', '#c0392b', '#7d3c98')
    _save = fs_savefig if _HAS_FIG_STYLE else lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))
    _L = LABELS if _HAS_FIG_STYLE else {'theta_short': 'θ'}

    fig, axes = make_fig(n_panels=3, height_override=2.2) if _HAS_FIG_STYLE \
        else plt.subplots(1, 3, figsize=(5.5, 2.2))
    if not isinstance(axes, np.ndarray): axes = np.array([axes])
    ax = axes[0]
    ax.scatter(val_df['irt_p'], val_df['empirical_p'], alpha=0.02, s=1, color=_c1)
    ax.plot([0, 1], [0, 1], color=_c2, ls='--', lw=0.6, label='Perfect')
    val_df['irt_bin'] = pd.cut(val_df['irt_p'], bins=np.linspace(0, 1, 21), labels=False)
    cal = val_df.groupby('irt_bin').agg(mean_irt=('irt_p', 'mean'),
                                         mean_emp=('empirical_p', 'mean')).dropna()
    ax.plot(cal['mean_irt'], cal['mean_emp'], 'ko-', markersize=3, linewidth=1, label='Binned')
    ax.set_xlabel('IRT $P$(safe)'); ax.set_ylabel('Empirical $P$(safe)')
    ax.set_title(f'Calibration\n$r$={r_pearson:.3f}, RMSE={rmse:.3f}')
    ax.legend(fontsize=5)

    ax = axes[1]
    palette = sns.color_palette("Set2", val_df['language'].nunique())
    for i, lang in enumerate(sorted(val_df['language'].unique())):
        sub = val_df[val_df['language'] == lang]
        if len(sub) >= 10:
            r_l, _ = pearsonr(sub['empirical_p'], sub['irt_p'])
            ax.scatter(sub['irt_p'], sub['empirical_p'], alpha=0.05, s=1,
                       color=palette[i], label=f'{lang} ({r_l:.2f})')
    ax.plot([0, 1], [0, 1], color=_c2, ls='--', lw=0.6)
    ax.set_xlabel('IRT $P$(safe)'); ax.set_ylabel('Empirical $P$(safe)')
    ax.set_title('By Language')
    
    ax.legend(fontsize=3.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.35), markerscale=5)

    ax = axes[2]; val_df['residual'] = val_df['empirical_p'] - val_df['irt_p']
    ax.hist(val_df['residual'], bins=100, edgecolor='black', linewidth=0.2,
            alpha=0.7, color=_c1)
    ax.axvline(0, color=_c2, ls='--', lw=0.6)
    ax.set_xlabel('Residual'); ax.set_ylabel('Count')
    ax.set_title(f'Residuals\nMean={val_df["residual"].mean():.4f}, '
                 f'Std={val_df["residual"].std():.4f}')
                 
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35, wspace=0.35)
    _save(fig, os.path.join(RESULTS_DIR, "B5_calibration.png"))

    # Example ICCs
    prompt_var = val_df.groupby('prompt_id')['empirical_p'].std().sort_values(ascending=False)
    example_prompts = prompt_var.head(6).index.tolist()
    fig, axes_icc = make_fig_grid(2, 3, height_override=1.8) if _HAS_FIG_STYLE \
        else plt.subplots(2, 3, figsize=(5.5, 3.6))
    axes_flat = axes_icc.flatten()
    for idx, pid in enumerate(example_prompts):
        ax = axes_flat[idx]
        p_data = val_df[val_df['prompt_id'] == pid].sort_values('theta')
        if len(p_data) < 5: continue
        for lang in sorted(p_data['language'].unique()):
            sub = p_data[p_data['language'] == lang]
            ax.scatter(sub['theta'], sub['empirical_p'], alpha=0.6, s=6, label=lang)
        if pid in results['prompt_map']:
            pi = results['prompt_map'][pid]
            theta_range = np.linspace(p_data['theta'].min() - 0.5,
                                      p_data['theta'].max() + 0.5, 200)
            diff_en = _to_scalar(results['beta_mean'][pi])
            alpha_i = _to_scalar(results['alpha_mean'][pi])
            ax.plot(theta_range,
                    1/(1+np.exp(-alpha_i*(theta_range - diff_en))),
                    'k-', linewidth=1, label='IRT (en)')
            for show_lang in ['bn', 'jv', 'sw']:
                if show_lang in results['lang_map']:
                    li = results['lang_map'][show_lang]
                    diff_l = diff_en + _to_scalar(results['gamma_mean'][li]) + \
                             _to_scalar(results['tau_mean'][pi, li])
                    ax.plot(theta_range,
                            1/(1+np.exp(-alpha_i*(theta_range - diff_l))),
                            '--', linewidth=0.8, label=f'IRT ({show_lang})')
                    break
        ax.set_xlabel(_L.get('theta_short', 'θ'))
        ax.set_ylabel('$P$(safe)')
        ax.set_title(f'Prompt {pid}')
        
        # CHANGED: Legend shifted to upper left, and limited to 1 column so it misses the curve
        ax.legend(fontsize=3.5, ncol=1, loc='upper left') 
        ax.set_ylim(-0.05, 1.05)
        
    fig.tight_layout()
    _save(fig, os.path.join(RESULTS_DIR, "B5_example_ICCs.png"))

    val_df.to_csv(os.path.join(RESULTS_DIR, "B5_validation_data.csv"), index=False)
    pd.DataFrame([{'pearson_r': r_pearson, 'spearman_rho': r_spearman, 'rmse': rmse, 'mae': mae, 'n': len(val_df)}]).to_csv(os.path.join(RESULTS_DIR, "B5_calibration_summary.csv"), index=False)
    print(f"  Saved: B5_calibration.png, B5_example_ICCs.png")
    return val_df, results


# ══════════════════════════════════════════════════════════════════════════
# B6: Temperature Decomposition (unchanged logic, uses df directly)
# ══════════════════════════════════════════════════════════════════════════

def b6_temperature_decomposition(df):
    print("\n" + "=" * 70)
    print("B6: TEMPERATURE VS. STOCHASTIC VARIANCE DECOMPOSITION")
    print("=" * 70)
    def get_model_family(name):
        name = str(name).lower()
        if 'gpt' in name or 'o3' in name or 'o4' in name: return 'GPT'
        elif 'claude' in name: return 'Claude'
        elif 'gemini' in name: return 'Gemini'
        elif 'grok' in name: return 'Grok'
        elif 'deepseek' in name: return 'DeepSeek'
        return 'Other'
    df = df.copy(); df['model_family'] = df['base_model'].apply(get_model_family)
    agg = df.groupby(['base_model', 'temp_setting', 'id', 'language'])['score'].agg(['mean', 'std', 'count']).reset_index()
    agg.columns = ['base_model', 'temp_setting', 'id', 'language', 'mean_score', 'std_score', 'n_obs']
    def strip_temp(m):
        for t in ['_Low_Creativity', '_Standard', '_High_Risk', '_Chaos', '_Low-Creativity', '_High-Risk']: m = m.replace(t, '')
        return m
    agg['core_model'] = agg['base_model'].apply(strip_temp)
    decomp = []
    for (core, pid, lang), group in agg.groupby(['core_model', 'id', 'language']):
        if len(group) < 2: continue
        bv = group['mean_score'].var(); wv = (group['std_score'].fillna(0)**2).mean()
        tv = bv + wv if (bv + wv) > 0 else 1e-10
        decomp.append({'core_model': core, 'prompt_id': pid, 'language': lang, 'between_temp_var': bv, 'within_temp_var': wv, 'total_var': tv, 'between_frac': bv/tv, 'model_family': get_model_family(core)})
    decomp_df = pd.DataFrame(decomp)
    if len(decomp_df) == 0: print("  WARNING: No decomposition."); return None
    print(f"  Mean between-temp fraction: {decomp_df['between_frac'].mean():.3f}")
    for fam, row in decomp_df.groupby('model_family')['between_frac'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False).iterrows():
        print(f"    {fam:<12}: mean={row['mean']:.3f}")

    _c1, _c2 = (C_BLUE, C_RED) if _HAS_FIG_STYLE else ('#2471a3', '#c0392b')
    _fc = FS_FAM_COLORS if _HAS_FIG_STYLE else {'Claude': '#7d3c98', 'GPT': '#2471a3',
        'Gemini': '#c0392b', 'Grok': '#e67e22', 'DeepSeek': '#27ae60', 'Other': '#7f8c8d'}
    _save = fs_savefig if _HAS_FIG_STYLE else lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))

    fig, axes = make_fig(n_panels=3, height_override=2.2) if _HAS_FIG_STYLE \
        else plt.subplots(1, 3, figsize=(5.5, 2.2))
    if not isinstance(axes, np.ndarray): axes = np.array([axes])
    axes[0].hist(decomp_df['between_frac'], bins=50, edgecolor='black',
                 linewidth=0.2, alpha=0.7, color=_c1)
    axes[0].axvline(decomp_df['between_frac'].mean(), color=_c2, ls='--',
                    lw=0.6, label=f'Mean={decomp_df["between_frac"].mean():.3f}')
    axes[0].set_xlabel('Between-Temp Fraction')
    axes[0].set_title('Variance Decomposition'); axes[0].legend(fontsize=5)
    fam_decomp = decomp_df.groupby('model_family')['between_frac'].agg(['mean']).sort_values('mean', ascending=False)
    fam_order = fam_decomp.index.tolist()
    bp = axes[1].boxplot(
        [decomp_df[decomp_df['model_family'] == f]['between_frac'].values for f in fam_order],
        labels=fam_order, patch_artist=True, widths=0.5,
        medianprops=dict(color='black', lw=0.8),
        flierprops=dict(markersize=2))
    for patch, fam in zip(bp['boxes'], fam_order):
        patch.set_facecolor(_fc.get(fam, '#7f8c8d'))
        patch.set_linewidth(0.4)
    axes[1].set_title('By Model Family')
    axes[1].tick_params(axis='x', labelsize=5, rotation=30)
    temp_jsr = df.groupby(['temp_setting', 'language']).agg(jsr=('score', lambda x: 1 - x.mean())).reset_index()
    temp_order = [t for t in ['Low_Creativity', 'Standard', 'High_Risk', 'Chaos']
                  if t in temp_jsr['temp_setting'].values]
    for lang in sorted(temp_jsr['language'].unique()):
        ld = temp_jsr[temp_jsr['language'] == lang].set_index('temp_setting').reindex(temp_order)
        if not ld['jsr'].isna().all():
            axes[2].plot(range(len(temp_order)), ld['jsr'].values,
                         'o-', label=lang, alpha=0.7, markersize=3, lw=0.8)
    axes[2].set_xticks(range(len(temp_order)))
    axes[2].set_xticklabels(temp_order, rotation=30, fontsize=5)
    axes[2].set_ylabel('JSR'); axes[2].set_title('JSR by Temperature')
    axes[2].legend(fontsize=3.5, ncol=3)
    _save(fig, os.path.join(RESULTS_DIR, "B6_temperature_decomposition.png"))
    decomp_df.to_csv(os.path.join(RESULTS_DIR, "B6_variance_decomposition.csv"), index=False)
    return decomp_df


# ══════════════════════════════════════════════════════════════════════════
# B7: Pass-to-Pass τ Stability (uses fit_irt)
# ══════════════════════════════════════════════════════════════════════════

def b7_tau_stability(df, anchor_ids):
    print("\n" + "=" * 70)
    print("B7: PASS-TO-PASS STABILITY OF τ")
    print("=" * 70)
    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    available_passes = sorted(df['pass_num'].dropna().unique())
    use_random = len(available_passes) < 3
    if use_random:
        print("  Falling back to random 3-way split.")
        np.random.seed(SEED); df = df.copy(); df['pass_group'] = np.random.choice(['A', 'B', 'C'], size=len(df))
        groups = ['A', 'B', 'C']
    else:
        groups = available_passes[:min(5, len(available_passes))]

    tau_by_pass, theta_by_pass, beta_by_pass = {}, {}, {}
    for group in groups:
        subset = df[df['pass_group'] == group] if use_random else df[df['pass_num'] == group]
        subset = subset.copy()
        if len(subset) < 100: continue
        print(f"\n  Group {group}: {len(subset):,} rows")
        r = fit_irt(subset, anchor_ids, label=f"pass-{group}", cache_key=f"b7_pass_{group}")
        tau_by_pass[group] = {'tau_mean': r['tau_mean'], 'prompt_map': r['prompt_map'], 'lang_map': r['lang_map']}
        theta_by_pass[group] = {'theta_mean': r['theta_mean'], 'student_map': r['student_map']}
        beta_by_pass[group] = {'beta_mean': r['beta_mean'], 'prompt_map': r['prompt_map']}

    if len(tau_by_pass) < 2: print("  ERROR: Need ≥2 passes."); return None
    pass_keys = sorted(tau_by_pass.keys(), key=str)

    n_pk = len(pass_keys); corr_mat = np.full((n_pk, n_pk), np.nan)
    pairwise = []
    for i, p1 in enumerate(pass_keys):
        corr_mat[i, i] = 1.0
        for j, p2 in enumerate(pass_keys):
            if i >= j: continue
            cp = set(tau_by_pass[p1]['prompt_map']) & set(tau_by_pass[p2]['prompt_map'])
            cl = (set(tau_by_pass[p1]['lang_map']) & set(tau_by_pass[p2]['lang_map'])) - {'en'}
            v1, v2 = [], []
            for pr in cp:
                for la in cl:
                    pi1, pi2 = tau_by_pass[p1]['prompt_map'][pr], tau_by_pass[p2]['prompt_map'][pr]
                    li1, li2 = tau_by_pass[p1]['lang_map'][la], tau_by_pass[p2]['lang_map'][la]
                    t1, t2 = tau_by_pass[p1]['tau_mean'], tau_by_pass[p2]['tau_mean']
                    if pi1 < t1.shape[0] and li1 < t1.shape[1] and pi2 < t2.shape[0] and li2 < t2.shape[1]:
                        v1.append(_to_scalar(t1[pi1, li1])); v2.append(_to_scalar(t2[pi2, li2]))
            if len(v1) >= 10:
                a1, a2 = np.array(v1), np.array(v2)
                rp, _ = pearsonr(a1, a2); rs, _ = spearmanr(a1, a2); rm = np.sqrt(np.mean((a1 - a2)**2))
                corr_mat[i, j] = corr_mat[j, i] = rp
                pairwise.append({'pass_1': p1, 'pass_2': p2, 'pearson_r': rp, 'spearman_r': rs, 'rmse': rm, 'n': len(v1)})
                print(f"  τ ({p1} vs {p2}): r={rp:.4f}, ρ={rs:.4f}, n={len(v1)}")
    pairwise_df = pd.DataFrame(pairwise)
    if len(pairwise_df) > 0: print(f"  Mean τ r: {pairwise_df['pearson_r'].mean():.4f}")

    # θ, β stability
    theta_corrs, beta_corrs = [], []
    for i, p1 in enumerate(pass_keys):
        for j, p2 in enumerate(pass_keys):
            if i >= j: continue
            cs = set(theta_by_pass[p1]['student_map']) & set(theta_by_pass[p2]['student_map'])
            if len(cs) >= 3:
                t1 = [float(theta_by_pass[p1]['theta_mean'][theta_by_pass[p1]['student_map'][s]]) for s in cs]
                t2 = [float(theta_by_pass[p2]['theta_mean'][theta_by_pass[p2]['student_map'][s]]) for s in cs]
                theta_corrs.append({'pass_1': p1, 'pass_2': p2, 'pearson_r': pearsonr(t1, t2)[0]})
            cpids = set(beta_by_pass[p1]['prompt_map']) & set(beta_by_pass[p2]['prompt_map'])
            if len(cpids) >= 10:
                b1 = [float(beta_by_pass[p1]['beta_mean'][beta_by_pass[p1]['prompt_map'][p]]) for p in cpids]
                b2 = [float(beta_by_pass[p2]['beta_mean'][beta_by_pass[p2]['prompt_map'][p]]) for p in cpids]
                beta_corrs.append({'pass_1': p1, 'pass_2': p2, 'pearson_r': pearsonr(b1, b2)[0]})
    theta_corr_df, beta_corr_df = pd.DataFrame(theta_corrs), pd.DataFrame(beta_corrs)
    if len(theta_corr_df) > 0: print(f"  θ stability: {theta_corr_df['pearson_r'].mean():.4f}")
    if len(beta_corr_df) > 0: print(f"  β stability: {beta_corr_df['pearson_r'].mean():.4f}")

    # Per-language scatter data
    scatter_all_df = pd.DataFrame()
    lang_stability_df = pd.DataFrame()
    if len(pass_keys) >= 2:
        p1k, p2k = pass_keys[0], pass_keys[1]
        cp12 = set(tau_by_pass[p1k]['prompt_map']) & set(tau_by_pass[p2k]['prompt_map'])
        cl12 = (set(tau_by_pass[p1k]['lang_map']) & set(tau_by_pass[p2k]['lang_map'])) - {'en'}
        scat, lstab = [], []
        for la in sorted(cl12):
            v1l, v2l = [], []
            for pr in cp12:
                pi1, pi2 = tau_by_pass[p1k]['prompt_map'][pr], tau_by_pass[p2k]['prompt_map'][pr]
                li1, li2 = tau_by_pass[p1k]['lang_map'][la], tau_by_pass[p2k]['lang_map'][la]
                tm1, tm2 = tau_by_pass[p1k]['tau_mean'], tau_by_pass[p2k]['tau_mean']
                if pi1 < tm1.shape[0] and li1 < tm1.shape[1] and pi2 < tm2.shape[0] and li2 < tm2.shape[1]:
                    a, b = _to_scalar(tm1[pi1, li1]), _to_scalar(tm2[pi2, li2])
                    v1l.append(a); v2l.append(b); scat.append({'language': la, 'prompt': pr, 'tau_pass1': a, 'tau_pass2': b})
            if len(v1l) >= 5:
                rl, _ = pearsonr(v1l, v2l)
                lstab.append({'language': la, 'pearson_r': rl, 'n_prompts': len(v1l), 'mean_abs_diff': float(np.mean(np.abs(np.array(v1l) - np.array(v2l))))})
                print(f"    {la}: r={rl:.4f}")
        scatter_all_df = pd.DataFrame(scat)
        if len(scatter_all_df) > 0:
            scatter_all_df['tau_pass1'] = scatter_all_df['tau_pass1'].astype(np.float64)
            scatter_all_df['tau_pass2'] = scatter_all_df['tau_pass2'].astype(np.float64)
        lang_stability_df = pd.DataFrame(lstab)

    # Plotting
    _c1, _c2, _c3 = (C_BLUE, C_RED, C_PURPLE) if _HAS_FIG_STYLE else ('#2471a3', '#c0392b', '#7d3c98')
    _save = fs_savefig if _HAS_FIG_STYLE else lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))
    _L = LABELS if _HAS_FIG_STYLE else {'theta_short': 'θ', 'beta_short': 'β', 'tau_short': 'τ'}

    fig, all_axes = make_fig_grid(4, 3, height_override=1.8) if _HAS_FIG_STYLE \
        else plt.subplots(4, 3, figsize=(5.5, 7.2))

    # Row 0, col 0: correlation heatmap
    ax = all_axes[0, 0]; dm = corr_mat.copy(); dm[np.isnan(dm)] = 0
    sns.heatmap(dm, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=[str(p) for p in pass_keys],
                yticklabels=[str(p) for p in pass_keys],
                vmin=0, vmax=1, ax=ax, annot_kws={'fontsize': 5},
                cbar_kws={'shrink': 0.8})
    ax.set_title(f'{_L["tau_short"]} Correlation Across Passes')
    ax.tick_params(labelsize=5)

    # Row 0, col 1: per-language stability bars
    ax = all_axes[0, 1]
    if len(lang_stability_df) > 0:
        ls = lang_stability_df.sort_values('pearson_r', ascending=True)
        cols = [_c1 if v > 0.8 else _c3 if v > 0.6 else _c2
                for v in ls['pearson_r']]
        bars = ax.barh(ls['language'], ls['pearson_r'], color=cols,
                       edgecolor='black', linewidth=0.3)
        for bar, val in zip(bars, ls['pearson_r']):
            ax.text(max(0, bar.get_width() + 0.01),
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=4)
        ax.axvline(0.8, color=_c1, ls='--', alpha=0.5, lw=0.5)
        ax.axvline(0.6, color=_c3, ls='--', alpha=0.5, lw=0.5)
        ax.set_title(f'{_L["tau_short"]} Stability by Language')
        ax.set_xlim(0, 1.15)

    # Row 0, col 2: parameter stability summary
    ax = all_axes[0, 2]
    pn, pm, pc = [], [], []
    if len(theta_corr_df) > 0:
        pn.append(_L['theta_short']); pm.append(theta_corr_df['pearson_r'].mean()); pc.append(_c1)
    if len(beta_corr_df) > 0:
        pn.append(_L['beta_short']); pm.append(beta_corr_df['pearson_r'].mean()); pc.append(_c3)
    if len(pairwise_df) > 0:
        pn.append(_L['tau_short']); pm.append(pairwise_df['pearson_r'].mean()); pc.append(_c2)
    if pn:
        bars = ax.bar(pn, pm, color=pc, edgecolor='black', width=0.5, linewidth=0.3)
        for bar, val in zip(bars, pm):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=5)
        ax.set_ylim(0, 1.1); ax.axhline(0.8, color='gray', ls='--', alpha=0.5, lw=0.5)
        ax.set_title('Parameter Stability')

    # Rows 1-2: per-language scatter (up to 6)
    if len(scatter_all_df) > 0:
        langs_sorted = sorted(scatter_all_df['language'].unique())
        pal = sns.color_palette("Set2", min(9, len(langs_sorted)))
        for idx in range(min(9, len(langs_sorted))):
            ax = all_axes[1 + idx//3, idx%3]; la = langs_sorted[idx]
            ld = scatter_all_df[scatter_all_df['language'] == la]
            xv = ld['tau_pass1'].values.astype(np.float64)
            yv = ld['tau_pass2'].values.astype(np.float64)
            ax.scatter(xv, yv, alpha=0.4, s=4, color=pal[idx], edgecolors='none')
            av = np.concatenate([xv, yv])
            lo, hi = np.percentile(av, 1) - 0.2, np.percentile(av, 99) + 0.2
            ax.plot([lo, hi], [lo, hi], color=_c2, ls='--', alpha=0.7, lw=0.5)
            if len(xv) >= 10:
                sl, ic, rv, _, _ = stats.linregress(xv, yv)
                ax.plot(np.linspace(lo, hi, 100),
                        sl * np.linspace(lo, hi, 100) + ic,
                        'k-', alpha=0.6, lw=0.6, label=f'$r$={rv:.3f}')
            ax.set_title(f'{la.upper()} (n={len(ld)})')
            ax.legend(fontsize=4)
        for idx in range(min(9, len(langs_sorted)), 9):
            all_axes[1 + idx//3, idx%3].set_visible(False)
    else:
        for r in range(1, 3):
            for c in range(3):
                all_axes[r, c].set_visible(False)

    _save(fig, os.path.join(RESULTS_DIR, "B7_tau_stability.png"))

    # Tau diff distribution (separate figure)
    if len(scatter_all_df) > 0:
        scatter_all_df['abs_diff'] = np.abs(
            scatter_all_df['tau_pass1'].values - scatter_all_df['tau_pass2'].values)
        fig2, ax2 = make_fig(n_panels=2, height_override=2.2) if _HAS_FIG_STYLE \
            else plt.subplots(1, 2, figsize=(5.5, 2.2))
        if not isinstance(ax2, np.ndarray): ax2 = np.array([ax2])
        ax2[0].hist(scatter_all_df['abs_diff'], bins=60, edgecolor='black',
                    linewidth=0.2, alpha=0.7, color=_c1)
        ax2[0].axvline(scatter_all_df['abs_diff'].mean(), color=_c2, ls='--',
                       lw=0.6, label=f'Mean={scatter_all_df["abs_diff"].mean():.3f}')
        ax2[0].set_title(r'$|\Delta\tau|$ Distribution')
        ax2[0].set_xlabel(r'$|\Delta\tau|$'); ax2[0].legend(fontsize=5)
        ld2 = scatter_all_df.groupby('language')['abs_diff'].agg(
            ['mean', 'std']).sort_values('mean', ascending=True)
        ax2[1].barh(ld2.index, ld2['mean'], xerr=ld2['std'],
                    color=sns.color_palette("viridis", len(ld2)),
                    edgecolor='black', linewidth=0.3, capsize=2)
        ax2[1].set_title(r'$|\Delta\tau|$ by Language')
        ax2[1].set_xlabel(r'$|\Delta\tau|$')
        _save(fig2, os.path.join(RESULTS_DIR, "B7_tau_diff_distribution.png"))

    pairwise_df.to_csv(os.path.join(RESULTS_DIR, "B7_tau_pairwise_correlations.csv"), index=False)
    if len(lang_stability_df) > 0: lang_stability_df.to_csv(os.path.join(RESULTS_DIR, "B7_tau_stability_by_language.csv"), index=False)
    if len(scatter_all_df) > 0: scatter_all_df.to_csv(os.path.join(RESULTS_DIR, "B7_tau_scatter_data.csv"), index=False)

    summary_rows = []
    if len(theta_corr_df) > 0: summary_rows.append({'Parameter': 'θ', 'Mean r': theta_corr_df['pearson_r'].mean(), 'Min r': theta_corr_df['pearson_r'].min(), 'Max r': theta_corr_df['pearson_r'].max()})
    if len(beta_corr_df) > 0: summary_rows.append({'Parameter': 'β', 'Mean r': beta_corr_df['pearson_r'].mean(), 'Min r': beta_corr_df['pearson_r'].min(), 'Max r': beta_corr_df['pearson_r'].max()})
    if len(pairwise_df) > 0: summary_rows.append({'Parameter': 'τ', 'Mean r': pairwise_df['pearson_r'].mean(), 'Min r': pairwise_df['pearson_r'].min(), 'Max r': pairwise_df['pearson_r'].max()})
    summary_df = pd.DataFrame(summary_rows); summary_df.to_csv(os.path.join(RESULTS_DIR, "B7_stability_summary_table.csv"), index=False)
    print(f"\n  Summary:\n{summary_df.to_string(index=False)}")
    return {'pairwise_df': pairwise_df, 'lang_stability_df': lang_stability_df, 'scatter_all_df': scatter_all_df, 'theta_corr_df': theta_corr_df, 'beta_corr_df': beta_corr_df, 'summary_df': summary_df}


# ══════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_all_experiments():
    print("=" * 70)
    print("EXPERIMENT B: TEST-RETEST RELIABILITY (UNIFIED)")
    print("=" * 70)

    df, anchor_ids = load_and_prepare_data()

    consistency_df = b1_response_consistency(df)
    icc_df = b3_icc_analysis(df)
    stochastic_df, prompt_entropy_df = b4_stochastic_profiles(df, consistency_df)
    decomp_df = b6_temperature_decomposition(df)
    b7_results = b7_tau_stability(df, anchor_ids)
    b2_results = b2_split_half_reliability(df, anchor_ids)
    val_df, irt_results = b5_empirical_icc_validation(df, consistency_df, anchor_ids)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENT B COMPLETE")
    print("=" * 70)

    print("\n── KEY RESULTS ──")
    if consistency_df is not None:
        print(f"  Det. safe: {(consistency_df['mean_safe'] == 1.0).mean():.1%}")
        print(f"  Boundary:  {((consistency_df['mean_safe'] > 0.2) & (consistency_df['mean_safe'] < 0.8)).mean():.1%}")
    if b2_results:
        r = pearsonr(b2_results['theta_df']['theta_h1'], b2_results['theta_df']['theta_h2'])[0]
        print(f"  Split-half θ SB: {(2*r)/(1+r):.3f}")
    if b7_results and len(b7_results['pairwise_df']) > 0:
        print(f"  Mean τ pass r: {b7_results['pairwise_df']['pearson_r'].mean():.3f}")
    if val_df is not None:
        print(f"  IRT calibration r: {pearsonr(val_df['empirical_p'], val_df['irt_p'])[0]:.3f}")


if __name__ == "__main__":
    if _HAS_FIG_STYLE: apply_style()
    run_all_experiments()
