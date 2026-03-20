# -*- coding: utf-8 -*-
"""
Experiment D: Predictive Validation via Cross-Validation
========================================================
Tests whether IRT parameter estimates generalize to held-out data by
comparing the full IRT model against an ablated variant (τ=0) and five
non-parametric baselines under three cross-validation regimes.

Cross-validation strategies:
  D1 — Leave-One-Family-Out (LOFO)
       Holds out all test-takers from one model family (GPT). Tests model generalization: can IRT
       predict safety for an entirely unseen model family?
       Produces: D1_LOFO_results.csv, D_LOFO_heatmap.png

  D2 — Leave-One-Language-Out (LOLO)
       Holds out all data for one language. Tests language generalization.
       Key result: τ contributes Δ=0.000 (expected, language-specific).
       Lookup baselines collapse to AUC=0.500; IRT maintains 0.75–0.91.
       Produces: D2_LOLO_results.csv, D_LOLO_heatmap.png

  D3 — Random 80/20
       Standard interpolation benchmark. IRT full achieves AUC=0.931.
       Produces: D3_LOPO_results.csv, D_Random_heatmap.png

  D4 — Plots and summary
       Generates all cross-experiment visualizations:
       - D_tau_ablation.png: ΔAUC across all three CV regimes
       - D_calibration_roc.png: calibration + ROC on held-out 85/15 split
       - D_convergence_by_fold.png: convergence step per fold
       - D_{type}_barplot.png: multi-metric bar charts per CV type
       - D_summary_table.csv: master results table

Baselines compared:
  1. Global Rate         — marginal P(safe) for all observations
  2. Language Rate       — conditions on language only
  3. Model Rate          — conditions on test-taker only
  4. Model×Lang Rate     — conditions on (test-taker, language) pair
  5. Prompt×Lang Rate    — conditions on (prompt, language) pair (lookup table)
  6. IRT (no τ)          — full model with τ forced to zero
  7. IRT (full)          — complete model with all parameters
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ── fig_style integration ──────────────────────────────────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
try:
    from fig_style import (apply_style, savefig as fs_savefig, make_fig, make_fig_grid,
                           C_RED, C_BLUE, C_PURPLE, COLORS_3, CMAP_DIV,
                           FAM_COLORS as FS_FAM_COLORS, LABELS, FULL_WIDTH,
                           add_identity_line)
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False
    print("[WARN] fig_style.py not found — using fallback styling")
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, accuracy_score,
    roc_curve, average_precision_score, log_loss
)
from sklearn.calibration import calibration_curve
import os
import warnings
import re
import pickle
from collections import defaultdict

warnings.filterwarnings('ignore')

from huggingface_hub import snapshot_download

DATA_DIR = snapshot_download(
    repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False
)
INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors", "anchors.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_experiment_D")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Convergence config (SAME as Experiment A) ──
MAX_TRAINING_STEPS = 6000
CONVERGENCE_WINDOW = 200
CONVERGENCE_THRESHOLD = 1e-4
MIN_TRAINING_STEPS = 1500
N_POSTERIOR_SAMPLES = 300
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def extract_pass_number(test_taker_str):
    match = re.search(r'pass[_-]?(\d+)', str(test_taker_str), re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_base_model(test_taker_str):
    return re.sub(r'[_-]?pass[_-]?\d+', '', str(test_taker_str), flags=re.IGNORECASE).strip()


def get_model_family(name):
    name = str(name).lower()
    if any(x in name for x in ['gpt', 'o3-mini', 'o4-mini', 'gpt-5']):
        return 'GPT'
    elif 'claude' in name:
        return 'Claude'
    elif 'gemini' in name:
        return 'Gemini'
    elif 'grok' in name:
        return 'Grok'
    elif 'deepseek' in name:
        return 'DeepSeek'
    return 'Other'


def _to_scalar(val):
    if hasattr(val, 'item'):
        return float(val.item())
    if hasattr(val, '__len__'):
        return float(val.flat[0])
    return float(val)


def check_convergence(losses, window, threshold, min_steps):
    """Check if training has converged by comparing rolling means."""
    if len(losses) < min_steps or len(losses) < 2 * window:
        return False
    recent = np.mean(losses[-window:])
    previous = np.mean(losses[-2 * window:-window])
    if previous == 0:
        return True
    return (previous - recent) / abs(previous) < threshold


# ══════════════════════════════════════════════════════════════════════════
# IRT MODEL
# ══════════════════════════════════════════════════════════════════════════

def model_2pl(student_idx, prompt_idx, lang_idx, obs=None,
              num_students=None, num_prompts=None, num_langs=None,
              tau_mask=None, gamma_mask=None, anchor_mask_tensor=None):
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
    tau_scale_per = torch.where(
        anchor_mask_tensor > 0.5,
        torch.full((num_prompts, num_langs), 0.01, device=device),
        tau_scale.expand(num_prompts, num_langs))
    tau_raw = pyro.sample("tau_raw",
        dist.StudentT(1.0, torch.zeros(num_prompts, num_langs, device=device),
                    tau_scale_per).to_event(2))
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

# ══════════════════════════════════════════════════════════════════════════
# UNIFIED FITTER WITH CONVERGENCE-BASED STOPPING
# ══════════════════════════════════════════════════════════════════════════

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
    anchor_mask_tensor = torch.zeros((num_prompts, num_langs), device=device)
    for pid in prompts:
        if pid in anchor_ids and pid in prompt_map:
            anchor_mask_tensor[prompt_map[pid], :] = 1.0

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
                        num_students, num_prompts, num_langs, tau_mask, gamma_mask, anchor_mask_tensor)
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
                         num_students, num_prompts, num_langs, tau_mask, gamma_mask, anchor_mask_tensor)

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
        print(f"    ★ Cached IRT [{label}] → {cache_path}")
    return result

# ══════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def irt_predict_proba(irt_results, test_df):
    """Vectorized 2PL prediction for test set."""
    student_col = 'test_taker' if 'test_taker' in test_df.columns else 'model'
    n = len(test_df)
    theta_arr = np.zeros(n, dtype=np.float64)
    beta_arr = np.zeros(n, dtype=np.float64)
    gamma_arr = np.zeros(n, dtype=np.float64)
    tau_arr = np.zeros(n, dtype=np.float64)
    delta_arr = np.zeros(n, dtype=np.float64)
    alpha_arr = np.ones(n, dtype=np.float64)  # default α=1 for unseen items

    students = test_df[student_col].values
    prompts = test_df['id'].values
    langs = test_df['language'].values
    s_map = irt_results['student_map']
    p_map = irt_results['prompt_map']
    l_map = irt_results['lang_map']

    for i in range(n):
        s, p, l = students[i], prompts[i], langs[i]
        if s in s_map and s_map[s] < len(irt_results['theta_mean']):
            theta_arr[i] = _to_scalar(irt_results['theta_mean'][s_map[s]])
        if p in p_map and p_map[p] < len(irt_results['beta_mean']):
            pi = p_map[p]
            beta_arr[i] = _to_scalar(irt_results['beta_mean'][pi])
            alpha_arr[i] = _to_scalar(irt_results['alpha_mean'][pi])  # CHANGED
        if l in l_map and l_map[l] < len(irt_results['gamma_mean']):
            gamma_arr[i] = _to_scalar(irt_results['gamma_mean'][l_map[l]])
        if p in p_map and l in l_map:
            pi, li = p_map[p], l_map[l]
            if pi < irt_results['tau_mean'].shape[0] and li < irt_results['tau_mean'].shape[1]:
                tau_arr[i] = _to_scalar(irt_results['tau_mean'][pi, li])
        if s in s_map and l in l_map:
            si, li = s_map[s], l_map[l]
            if si < irt_results['delta_mean'].shape[0] and li < irt_results['delta_mean'].shape[1]:
                delta_arr[i] = _to_scalar(irt_results['delta_mean'][si, li])

    logits = alpha_arr * ((theta_arr + delta_arr) - (beta_arr + gamma_arr + tau_arr))
    return 1.0 / (1.0 + np.exp(-logits))


def irt_predict_no_tau(irt_results, test_df):
    """2PL prediction with τ forced to zero (ablation)."""
    student_col = 'test_taker' if 'test_taker' in test_df.columns else 'model'
    n = len(test_df)
    theta_arr = np.zeros(n, dtype=np.float64)
    beta_arr = np.zeros(n, dtype=np.float64)
    gamma_arr = np.zeros(n, dtype=np.float64)
    delta_arr = np.zeros(n, dtype=np.float64)
    alpha_arr = np.ones(n, dtype=np.float64)


    students = test_df[student_col].values
    prompts = test_df['id'].values
    langs = test_df['language'].values
    s_map = irt_results['student_map']
    p_map = irt_results['prompt_map']
    l_map = irt_results['lang_map']

    for i in range(n):
        s, p, l = students[i], prompts[i], langs[i]
        if s in s_map and s_map[s] < len(irt_results['theta_mean']):
            theta_arr[i] = _to_scalar(irt_results['theta_mean'][s_map[s]])
        if p in p_map and p_map[p] < len(irt_results['beta_mean']):
            pi = p_map[p]
            beta_arr[i] = _to_scalar(irt_results['beta_mean'][pi])
            alpha_arr[i] = _to_scalar(irt_results['alpha_mean'][pi])  # CHANGED
        if l in l_map and l_map[l] < len(irt_results['gamma_mean']):
            gamma_arr[i] = _to_scalar(irt_results['gamma_mean'][l_map[l]])
        if s in s_map and l in l_map:
            si, li = s_map[s], l_map[l]
            if si < irt_results['delta_mean'].shape[0] and li < irt_results['delta_mean'].shape[1]:
                delta_arr[i] = _to_scalar(irt_results['delta_mean'][si, li])
    # irt_predict_proba (full):
    logits = alpha_arr * ((theta_arr + delta_arr) - (beta_arr + gamma_arr))
    return 1.0 / (1.0 + np.exp(-logits))

# ══════════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════════

def baseline_global(train_df, test_df):
    return np.full(len(test_df), train_df['score'].mean())

def baseline_language(train_df, test_df):
    rates = train_df.groupby('language')['score'].mean().to_dict()
    g = train_df['score'].mean()
    return test_df['language'].map(lambda l: rates.get(l, g)).values.astype(np.float64)

def baseline_model(train_df, test_df):
    sc = 'test_taker' if 'test_taker' in train_df.columns else 'model'
    rates = train_df.groupby(sc)['score'].mean().to_dict()
    g = train_df['score'].mean()
    return test_df[sc].map(lambda m: rates.get(m, g)).values.astype(np.float64)

def baseline_model_lang(train_df, test_df):
    sc = 'test_taker' if 'test_taker' in train_df.columns else 'model'
    ml = train_df.groupby([sc, 'language'])['score'].mean().to_dict()
    lr = train_df.groupby('language')['score'].mean().to_dict()
    g = train_df['score'].mean()
    preds = []
    for _, row in test_df.iterrows():
        key = (row[sc], row['language'])
        preds.append(ml.get(key, lr.get(row['language'], g)))
    return np.array(preds, dtype=np.float64)

def baseline_prompt_lang(train_df, test_df):
    pl = train_df.groupby(['id', 'language'])['score'].mean().to_dict()
    lr = train_df.groupby('language')['score'].mean().to_dict()
    g = train_df['score'].mean()
    preds = []
    for _, row in test_df.iterrows():
        key = (row['id'], row['language'])
        preds.append(pl.get(key, lr.get(row['language'], g)))
    return np.array(preds, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, label=""):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.clip(np.array(y_pred, dtype=np.float64), 1e-7, 1 - 1e-7)
    y_bin = (y_pred >= 0.5).astype(int)
    m = {'label': label, 'n': len(y_true), 'accuracy': accuracy_score(y_true, y_bin)}
    if len(np.unique(y_true)) > 1:
        m['auc_roc'] = roc_auc_score(y_true, y_pred)
        m['avg_precision'] = average_precision_score(y_true, y_pred)
    else:
        m['auc_roc'] = m['avg_precision'] = np.nan
    m['brier'] = brier_score_loss(y_true, y_pred)
    m['log_loss'] = log_loss(y_true, y_pred)
    try:
        pt, pp = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')
        bc, _ = np.histogram(y_pred, bins=10, range=(0, 1))
        tot = bc.sum()
        m['ece'] = np.sum(np.abs(pt - pp) * bc[:len(pt)] / tot) if tot > 0 else np.nan
    except:
        m['ece'] = np.nan
    return m


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    print(f"  Raw: {len(df):,}")
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['score'] = (df['judge_score'] >= 4).astype(np.float32)
    df['id'] = df['id'].apply(clean_id)
    sc = 'test_taker' if 'test_taker' in df.columns else 'model'
    df['pass_num'] = df[sc].apply(extract_pass_number)
    df['base_model'] = df[sc].apply(extract_base_model)
    df['model_family'] = df['base_model'].apply(get_model_family)
    print(f"  Valid: {len(df):,}, Families: {sorted(df['model_family'].unique())}, "
          f"Languages: {sorted(df['language'].unique())}")

    anchor_ids = set()
    if os.path.exists(ANCHOR_FILE):
        adf = pd.read_csv(ANCHOR_FILE)
        adf['id'] = adf['id'].apply(clean_id)
        anchor_ids = set(adf['id'].unique())
        print(f"  Anchors: {len(anchor_ids)}")
    return df, anchor_ids


# ══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def run_fold(train_df, test_df, anchor_ids, fold_name, fold_type):
    """Run one CV fold: fit IRT + all baselines, return metrics."""
    sc = 'test_taker' if 'test_taker' in train_df.columns else 'model'

    if len(test_df) < 50 or len(np.unique(test_df['score'])) < 2:
        print(f"    SKIP: insufficient test data ({len(test_df)} rows)")
        return []

    # Fit IRT with convergence-based stopping
    irt = fit_irt(train_df, anchor_ids, label=f"{fold_type}-{fold_name}")
    print(f"    Converged at step {irt['converged_at']}")

    y_true = test_df['score'].values.astype(np.float64)

    predictions = {
        'IRT (full)': irt_predict_proba(irt, test_df),
        'IRT (no τ)': irt_predict_no_tau(irt, test_df),
        'Global Rate': baseline_global(train_df, test_df),
        'Language Rate': baseline_language(train_df, test_df),
        'Model Rate': baseline_model(train_df, test_df),
        'Model×Lang Rate': baseline_model_lang(train_df, test_df),
        'Prompt×Lang Rate': baseline_prompt_lang(train_df, test_df),
    }

    results = []
    for method, y_pred in predictions.items():
        m = compute_metrics(y_true, y_pred, label=method)
        m['held_out'] = fold_name
        m['fold_type'] = fold_type
        m['converged_at'] = irt['converged_at']
        results.append(m)
        print(f"      {method:<22}: AUC={m['auc_roc']:.4f}, "
              f"Brier={m['brier']:.4f}, Acc={m['accuracy']:.4f}")

    return results


def d1_leave_one_family_out(df, anchor_ids):
    print("\n" + "=" * 70)
    print("D1: LEAVE-ONE-FAMILY-OUT")
    print("=" * 70)
    cache_csv = os.path.join(RESULTS_DIR, "D1_LOFO_results.csv")
    if os.path.exists(cache_csv):
        print(f"  ★ Loading cached D1 from {cache_csv}")
        return pd.read_csv(cache_csv)
    sc = 'test_taker' if 'test_taker' in df.columns else 'model'
    families = sorted(df['model_family'].unique())
    all_results = []
    for fam in families:
        print(f"\n  Held-out: {fam}")
        train = df[df['model_family'] != fam]
        test = df[df['model_family'] == fam]
        print(f"    Train: {len(train):,} ({train[sc].nunique()} TT), Test: {len(test):,}")
        all_results.extend(run_fold(train, test, anchor_ids, fam, 'LOFO'))
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "D1_LOFO_results.csv"), index=False)
    return results_df


def d2_leave_one_language_out(df, anchor_ids):
    print("\n" + "=" * 70)
    print("D2: LEAVE-ONE-LANGUAGE-OUT")
    print("=" * 70)
    cache_csv = os.path.join(RESULTS_DIR, "D2_LOLO_results.csv")
    if os.path.exists(cache_csv):
        print(f"  ★ Loading cached D2 from {cache_csv}")
        return pd.read_csv(cache_csv)
    languages = sorted(df['language'].unique())
    all_results = []
    for lang in languages:
        print(f"\n  Held-out: {lang}")
        train = df[df['language'] != lang]
        test = df[df['language'] == lang]
        print(f"    Train: {len(train):,} ({train['language'].nunique()} langs), Test: {len(test):,}")
        all_results.extend(run_fold(train, test, anchor_ids, lang, 'LOLO'))
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "D2_LOLO_results.csv"), index=False)
    return results_df


def d3_leave_one_pass_out(df, anchor_ids):
    print("\n" + "=" * 70)
    print("D3: LEAVE-ONE-PASS-OUT")
    print("=" * 70)
    cache_csv = os.path.join(RESULTS_DIR, "D3_LOPO_results.csv")
    if os.path.exists(cache_csv):
        print(f"  ★ Loading cached D3 from {cache_csv}")
        return pd.read_csv(cache_csv)
    available = sorted(df['pass_num'].dropna().unique())
    all_results = []

    if len(available) < 2:
        print("  No pass numbers detected. Using random 80/20 × 5 folds.")
        for fold in range(5):
            print(f"\n  Fold {fold}")
            np.random.seed(SEED + fold)
            mask = np.random.rand(len(df)) < 0.8
            train, test = df[mask], df[~mask]
            print(f"    Train: {len(train):,}, Test: {len(test):,}")
            all_results.extend(run_fold(train, test, anchor_ids, f'fold_{fold}', 'Random'))
    else:
        test_passes = available[:min(5, len(available))]
        for p in test_passes:
            print(f"\n  Held-out: pass {p}")
            train = df[df['pass_num'] != p]
            test = df[df['pass_num'] == p]
            print(f"    Train: {len(train):,}, Test: {len(test):,}")
            all_results.extend(run_fold(train, test, anchor_ids, f'pass_{p}', 'LOPO'))

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "D3_LOPO_results.csv"), index=False)
    return results_df


# ══════════════════════════════════════════════════════════════════════════
# D4: PLOTS AND SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def d4_plots(d1_df, d2_df, d3_df, df, anchor_ids):
    print("\n" + "=" * 70)
    print("D4: PLOTS AND SUMMARY")
    print("=" * 70)

    all_results = pd.concat([d1_df, d2_df, d3_df], ignore_index=True)
    all_results.to_csv(os.path.join(RESULTS_DIR, "D_all_results.csv"), index=False)

    method_order = ['Global Rate', 'Language Rate', 'Model Rate',
                    'Model×Lang Rate', 'Prompt×Lang Rate', 'IRT (no τ)', 'IRT (full)']

    # ── Style constants (resolved once) ────────────────────────────
    if _HAS_FIG_STYLE:
        _c1, _c2, _c3 = C_BLUE, C_RED, C_PURPLE
        _sv = fs_savefig
        _L  = LABELS
    else:
        _c1, _c2, _c3 = '#2471a3', '#c0392b', '#7d3c98'
        _sv = lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))
        _L  = {'tau_short': r'$\tau$'}

    method_colors = {
        'Global Rate': '#bdc3c7', 'Language Rate': '#95a5a6',
        'Model Rate': '#7f8c8d', 'Model×Lang Rate': '#5d6d7e',
        'Prompt×Lang Rate': '#aab7b8',
        'IRT (no τ)': _c3,      # purple = ablated reference
        'IRT (full)': _c1,       # blue = main model ("good")
    }

    # ── Figure D.1: Bar plots per fold type ───────────────────────
    for fold_type, title in [('LOFO', 'Leave-One-Family-Out'),
                              ('LOLO', 'Leave-One-Language-Out'),
                              ('Random', 'Random 80/20')]:
        sub = all_results[all_results['fold_type'] == fold_type]
        if len(sub) == 0:
            continue

        if _HAS_FIG_STYLE:
            fig, axes = make_fig(n_panels=4, height_override=2.5)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.5))

        for ax_idx, (metric, mlabel) in enumerate([
            ('auc_roc', 'AUC-ROC'), ('brier', 'Brier Score'),
            ('log_loss', 'Log Loss'), ('accuracy', 'Accuracy')
        ]):
            ax = axes[ax_idx]
            means = sub.groupby('label')[metric].mean()
            stds = sub.groupby('label')[metric].std().fillna(0)
            present = [m for m in method_order if m in means.index]
            vals = [means[m] for m in present]
            errs = [stds[m] for m in present]
            cols = [method_colors.get(m, '#666') for m in present]
            bars = ax.bar(range(len(present)), vals, yerr=errs, color=cols,
                          edgecolor='black', linewidth=0.3, capsize=1.5,
                          error_kw={'linewidth': 0.5})
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels(present, rotation=55, ha='right')
            ax.set_ylabel(mlabel)
            ax.set_title(mlabel)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=4)

        _sv(fig, os.path.join(RESULTS_DIR, f"D_{fold_type}_barplot.png"))
        print(f"  Saved: D_{fold_type}_barplot")

    # ── Figure D.2: Heatmaps ─────────────────────────────────────
    for fold_type, fold_name in [('LOFO', 'Family'), ('LOLO', 'Language'),
                                  ('Random', 'Fold')]:
        sub = all_results[all_results['fold_type'] == fold_type]
        if len(sub) == 0:
            continue
        pivot = sub.pivot_table(index='label', columns='held_out', values='auc_roc')
        present = [m for m in method_order if m in pivot.index]
        pivot = pivot.reindex(present)

        if _HAS_FIG_STYLE:
            fig, ax = make_fig(n_panels=1, height_override=2.8)
        else:
            fig, ax = plt.subplots(figsize=(5.5, 2.8))

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0.5, vmax=1.0, ax=ax, linewidths=0.3,
                    annot_kws={'fontsize': 4, 'color': 'black'},
                    cbar_kws={'label': 'AUC-ROC', 'shrink': 0.8})
        ax.set_title(f'AUC-ROC × Held-Out {fold_name}')
        _sv(fig, os.path.join(RESULTS_DIR, f"D_{fold_type}_heatmap.png"))
        print(f"  Saved: D_{fold_type}_heatmap")

    # ── Figure D.3: τ ablation ───────────────────────────────────
    _tau = _L.get('tau_short', r'$\tau$')
    if _HAS_FIG_STYLE:
        fig, axes = make_fig(n_panels=3, height_override=2.5)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.5))

    for ax_idx, ft in enumerate(['LOFO', 'LOLO', 'Random']):
        ax = axes[ax_idx]
        sub = all_results[all_results['fold_type'] == ft]
        if len(sub) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        full = sub[sub['label'] == 'IRT (full)'].set_index('held_out')['auc_roc']
        notau = sub[sub['label'] == 'IRT (no τ)'].set_index('held_out')['auc_roc']
        common = full.index.intersection(notau.index)
        if len(common) == 0: continue

        delta = full[common] - notau[common]
        cols = [_c1 if d > 0 else _c2 for d in delta.values]
        ax.bar(range(len(delta)), delta.values, color=cols,
               edgecolor='black', linewidth=0.3)
        ax.set_xticks(range(len(delta)))
        ax.set_xticklabels(common, rotation=45, ha='right')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(delta.mean(), color=_c3, ls='--', alpha=0.7, lw=0.6,
                    label=f'Mean $\\Delta$={delta.mean():.4f}')
        ax.set_ylabel(r'$\Delta$AUC (full $-$ no ' + _tau + ')')
        ax.set_title(f'{ft}: {_tau} Contribution')
        ax.margins(y=0.25)  # Adds 25% padding to the top/bottom of the y-axis
        ax.legend(fontsize=5, loc='upper left')  # Anchors the legend so it won't float over data

    _sv(fig, os.path.join(RESULTS_DIR, "D_tau_ablation.png"))
    print(f"  Saved: D_tau_ablation")

    # ── Figure D.4: Calibration + ROC ────────────────────────────
    print("  Generating calibration curves...")
    np.random.seed(SEED)
    mask = np.random.rand(len(df)) < 0.85
    train_cal, test_cal = df[mask], df[~mask]
    if len(test_cal) > 100 and len(np.unique(test_cal['score'])) > 1:
        irt_cal = fit_irt(train_cal, anchor_ids, label="calibration")
        print(f"    Calibration model converged at step {irt_cal['converged_at']}")
        y_true_cal = test_cal['score'].values.astype(np.float64)
        methods_cal = {
            'IRT (full)': irt_predict_proba(irt_cal, test_cal),
            'IRT (no τ)': irt_predict_no_tau(irt_cal, test_cal),
            'Language Rate': baseline_language(train_cal, test_cal),
            'Prompt×Lang Rate': baseline_prompt_lang(train_cal, test_cal),
        }
        cal_colors = {'IRT (full)': _c1, 'IRT (no τ)': _c3,
                      'Language Rate': '#95a5a6', 'Prompt×Lang Rate': '#7f8c8d'}

        if _HAS_FIG_STYLE:
            fig, axes = make_fig(n_panels=2, height_override=2.5)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

        ax = axes[0]
        ax.plot([0, 1], [0, 1], 'k--', lw=0.5, label='Perfect')
        for name, yp in methods_cal.items():
            yp_c = np.clip(yp, 1e-7, 1 - 1e-7)
            try:
                pt, pp = calibration_curve(y_true_cal, yp_c, n_bins=10,
                                           strategy='uniform')
                ax.plot(pp, pt, 'o-', label=name, markersize=2,
                        color=cal_colors.get(name))
            except: pass
        ax.set_xlabel('Predicted $P$(safe)')
        ax.set_ylabel('Observed $P$(safe)')
        ax.set_title('Calibration')
        ax.legend(fontsize=5)

        ax = axes[1]
        for name, yp in methods_cal.items():
            yp_c = np.clip(yp, 1e-7, 1 - 1e-7)
            try:
                fpr, tpr, _ = roc_curve(y_true_cal, yp_c)
                auc_val = roc_auc_score(y_true_cal, yp_c)
                ax.plot(fpr, tpr, label=f'{name} ({auc_val:.3f})',
                        color=cal_colors.get(name))
            except: pass
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=0.5)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title('ROC Curves')
        ax.legend(fontsize=5, loc='lower right')

        _sv(fig, os.path.join(RESULTS_DIR, "D_calibration_roc.png"))
        print(f"  Saved: D_calibration_roc")

    # ── Figure D.5: Convergence across folds ─────────────────────
    if 'converged_at' in all_results.columns:
        if _HAS_FIG_STYLE:
            fig, ax = make_fig(n_panels=1, height_override=2.2)
        else:
            fig, ax = plt.subplots(figsize=(5.5, 2.2))

        conv_data = all_results[all_results['label'] == 'IRT (full)'][
            ['fold_type', 'held_out', 'converged_at']].drop_duplicates()
        ft_colors = {'LOFO': _c1, 'LOLO': _c2, 'LOPO': _c3, 'Random': '#7f8c8d'}
        for ft in conv_data['fold_type'].unique():
            sub = conv_data[conv_data['fold_type'] == ft]
            ax.scatter(sub['held_out'], sub['converged_at'], s=15, label=ft,
                       edgecolors='black', linewidth=0.3,
                       color=ft_colors.get(ft, '#7f8c8d'))
        ax.axhline(MAX_TRAINING_STEPS, color=_c2, ls='--', alpha=0.5,
                    lw=0.6, label=f'Max ({MAX_TRAINING_STEPS})')
        ax.set_ylabel('Converged at Step')
        ax.set_title('IRT Convergence Across Folds')
        ax.legend(fontsize=5)
        ax.tick_params(axis='x', rotation=45)

        _sv(fig, os.path.join(RESULTS_DIR, "D_convergence_by_fold.png"))
        print(f"  Saved: D_convergence_by_fold")

    # ── Summary Table ──
    summary_lines = []
    for ft in ['LOFO', 'LOLO', 'Random']:
        sub = all_results[all_results['fold_type'] == ft]
        if len(sub) == 0: continue
        for method in method_order:
            ms = sub[sub['label'] == method]
            if len(ms) == 0: continue
            summary_lines.append({
                'Validation': ft, 'Method': method,
                'AUC-ROC': f"{ms['auc_roc'].mean():.3f}±{ms['auc_roc'].std():.3f}",
                'Brier': f"{ms['brier'].mean():.4f}±{ms['brier'].std():.4f}",
                'LogLoss': f"{ms['log_loss'].mean():.3f}±{ms['log_loss'].std():.3f}",
                'Accuracy': f"{ms['accuracy'].mean():.3f}",
                'AUC_mean': ms['auc_roc'].mean(),
            })
    summary_df = pd.DataFrame(summary_lines)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "D_summary_table.csv"), index=False)

    print(f"\n  ── Summary ──")
    print(summary_df[['Validation', 'Method', 'AUC-ROC', 'Brier', 'Accuracy']].to_string(index=False))

    # ── Convergence summary ──
    if 'converged_at' in all_results.columns:
        conv_summary = all_results[all_results['label'] == 'IRT (full)'].groupby('fold_type')['converged_at'].agg(['mean', 'min', 'max', 'count'])
        print(f"\n  ── Convergence Summary ──")
        print(conv_summary.to_string())
        conv_summary.to_csv(os.path.join(RESULTS_DIR, "D_convergence_summary.csv"))

    return summary_df


# ══════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_experiment_d():
    print("=" * 70)
    print("EXPERIMENT D: PREDICTIVE VALIDATION")
    print(f"  Config: max_steps={MAX_TRAINING_STEPS}, convergence_window={CONVERGENCE_WINDOW}, "
          f"threshold={CONVERGENCE_THRESHOLD}, min_steps={MIN_TRAINING_STEPS}")
    print("=" * 70)

    df, anchor_ids = load_data()

    d1 = d1_leave_one_family_out(df, anchor_ids)
    d2 = d2_leave_one_language_out(df, anchor_ids)
    d3 = d3_leave_one_pass_out(df, anchor_ids)
    summary = d4_plots(d1, d2, d3, df, anchor_ids)

    print("\n" + "=" * 70)
    print("EXPERIMENT D COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)

    all_r = pd.concat([d1, d2, d3], ignore_index=True)
    print("\n── KEY NUMBERS ──")
    for m in ['IRT (full)', 'IRT (no τ)', 'Prompt×Lang Rate', 'Language Rate', 'Global Rate']:
        s = all_r[all_r['label'] == m]
        if len(s) > 0:
            print(f"  {m:<22}: AUC={s['auc_roc'].mean():.3f}±{s['auc_roc'].std():.3f}, "
                  f"Brier={s['brier'].mean():.4f}")

    irt_f = all_r[all_r['label'] == 'IRT (full)']['auc_roc'].mean()
    irt_nt = all_r[all_r['label'] == 'IRT (no τ)']['auc_roc'].mean()
    print(f"\n  τ contribution: {irt_f - irt_nt:+.4f} AUC")

    if 'converged_at' in all_r.columns:
        irt_rows = all_r[all_r['label'] == 'IRT (full)']
        print(f"  Avg convergence: {irt_rows['converged_at'].mean():.0f} steps "
              f"(range: {irt_rows['converged_at'].min():.0f}–{irt_rows['converged_at'].max():.0f})")


if __name__ == "__main__":
    if _HAS_FIG_STYLE:
        apply_style()
    run_experiment_d()
