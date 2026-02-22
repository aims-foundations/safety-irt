# -*- coding: utf-8 -*-
"""
Experiment A: IRT Model Comparison and Fit Diagnostics
======================================================
Uses convergence-based training for fair model comparison.
All models train up to the same maximum, stopping when converged.
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
from tqdm import tqdm
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
import os
import warnings
import re

warnings.filterwarnings('ignore')

from huggingface_hub import snapshot_download

DATA_DIR = snapshot_download(
    repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False
)
INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors", "anchors.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_experiment_A")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Convergence-based training config (SHARED across all models) ──
MAX_TRAINING_STEPS = 6000       # Hard ceiling for all models
CONVERGENCE_WINDOW = 200        # Rolling window to check improvement
CONVERGENCE_THRESHOLD = 1e-4    # Relative improvement threshold (0.01%)
MIN_TRAINING_STEPS = 1500       # Minimum steps before early stopping kicks in
N_POSTERIOR_SAMPLES = 400
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


def _to_scalar(val):
    if hasattr(val, 'item'):
        return float(val.item())
    if hasattr(val, '__len__'):
        return float(val.flat[0])
    return float(val)


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


# ══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

def model_1pl(student_idx, prompt_idx, lang_idx, obs=None,
              num_students=None, num_prompts=None, num_langs=None,
              tau_mask=None, gamma_mask=None):
    """1PL: P(safe) = σ(θ_j − (β_i + γ_L + τ_iL))"""
    theta = pyro.sample("theta",
        dist.Normal(torch.zeros(num_students, device=device), 1.0).to_event(1))
    beta = pyro.sample("beta",
        dist.Normal(torch.zeros(num_prompts, device=device), 1.0).to_event(1))
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
        logits = ability - difficulty
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)


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


def model_grm(student_idx, prompt_idx, lang_idx, obs=None,
              num_students=None, num_prompts=None, num_langs=None,
              tau_mask=None, gamma_mask=None, num_categories=5):
    """Graded Response Model (Samejima 1969) on Likert 1–5."""
    K = num_categories
    n_thresh = K - 1

    theta = pyro.sample("theta",
        dist.Normal(torch.zeros(num_students, device=device), 1.0).to_event(1))
    beta_base = pyro.sample("beta_base",
        dist.Normal(torch.zeros(num_prompts, device=device), 1.5).to_event(1))
    beta_increments = pyro.sample("beta_increments",
        dist.HalfNormal(torch.ones(num_prompts, n_thresh - 1, device=device) * 0.8).to_event(2))

    cumulative = torch.cumsum(beta_increments, dim=-1)
    first_thresh = beta_base.unsqueeze(-1)
    remaining = first_thresh + cumulative
    thresholds = torch.cat([first_thresh, remaining], dim=-1)
    thresholds = pyro.deterministic("thresholds", thresholds)

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
        lang_shift = gamma[lang_idx] + tau[prompt_idx, lang_idx]

        item_thresh = thresholds[prompt_idx]
        a_i = alpha[prompt_idx].unsqueeze(-1)
        ab = ability.unsqueeze(-1)
        ls = lang_shift.unsqueeze(-1)

        cum_probs = torch.sigmoid(a_i * (ab - (item_thresh + ls)))
        ones = torch.ones(cum_probs.shape[0], 1, device=device)
        zeros = torch.zeros(cum_probs.shape[0], 1, device=device)
        cum_extended = torch.cat([ones, cum_probs, zeros], dim=-1)
        cat_probs = cum_extended[:, :-1] - cum_extended[:, 1:]
        cat_probs = cat_probs.clamp(min=1e-8)
        cat_probs = cat_probs / cat_probs.sum(dim=-1, keepdim=True)

        pyro.sample("obs", dist.Categorical(probs=cat_probs), obs=obs)


# ══════════════════════════════════════════════════════════════════════════
# UNIFIED FITTER WITH CONVERGENCE-BASED STOPPING
# ══════════════════════════════════════════════════════════════════════════

def fit_model(model_fn, df_subset, anchor_ids, label="model",
              learning_rate=None, return_sites=None, extra_kwargs=None):
    """
    Fit any IRT model with convergence-based early stopping.
    Handles different parameter names across 1PL, 2PL, and GRM.
    """
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

    # GRM uses grm_score (0-indexed categorical), others use binary score
    if model_fn == model_grm:
        score_col = 'grm_score'
        score_dtype = torch.long
    else:
        score_col = 'score'
        score_dtype = torch.float32

    student_idx = torch.tensor(df_subset[student_col].map(student_map).values, dtype=torch.long).to(device)
    prompt_idx = torch.tensor(df_subset['id'].map(prompt_map).values, dtype=torch.long).to(device)
    lang_idx = torch.tensor(df_subset['language'].map(lang_map).values, dtype=torch.long).to(device)
    score_obs = torch.tensor(df_subset[score_col].values, dtype=score_dtype).to(device)

    # Masks
    tau_mask = torch.ones((num_prompts, num_langs), device=device)
    gamma_mask = torch.ones(num_langs, device=device)
    if 'en' in lang_map:
        en_i = lang_map['en']
        tau_mask[:, en_i] = 0.0
        gamma_mask[en_i] = 0.0
    for pid in prompts:
        if pid in anchor_ids and pid in prompt_map:
            tau_mask[prompt_map[pid], :] = 0.0

    model_kwargs = dict(
        num_students=num_students, num_prompts=num_prompts, num_langs=num_langs,
        tau_mask=tau_mask, gamma_mask=gamma_mask
    )
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)

    # Auto-select learning rate
    if learning_rate is None:
        learning_rate = 0.01 if model_fn == model_1pl else 0.005

    # Determine which sites to hide from the guide (deterministic + obs)
    hide_sites = ["obs", "tau", "gamma", "delta"]
    if model_fn in (model_2pl, model_grm):
        hide_sites.append("alpha")
    if model_fn == model_grm:
        hide_sites.append("thresholds")

    guide = pyro.infer.autoguide.AutoNormal(
        pyro.poutine.block(model_fn, hide=hide_sites)
    )
    optimizer = ClippedAdam({"lr": learning_rate, "clip_norm": 10.0})
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    # Training with convergence check
    losses = []
    converged_at = None
    pbar = tqdm(range(MAX_TRAINING_STEPS), desc=f"Fit [{label}]")

    for step in pbar:
        loss = svi.step(student_idx, prompt_idx, lang_idx, score_obs, **model_kwargs)
        losses.append(loss)
        if step % 100 == 0:
            pbar.set_description(f"[{label}] Loss: {loss:.1f}")
        if check_convergence(losses, CONVERGENCE_WINDOW, CONVERGENCE_THRESHOLD, MIN_TRAINING_STEPS):
            converged_at = step + 1
            pbar.close()
            print(f"    [{label}] Converged at step {converged_at} "
                  f"(window={CONVERGENCE_WINDOW}, threshold={CONVERGENCE_THRESHOLD})")
            break

    if converged_at is None:
        converged_at = MAX_TRAINING_STEPS
        print(f"    [{label}] Reached max steps ({MAX_TRAINING_STEPS})")

    # ── Build return_sites based on model type ──
    if return_sites is None:
        if model_fn == model_grm:
            return_sites = ["theta", "beta_base", "beta_increments",
                            "gamma", "tau", "delta", "alpha", "thresholds"]
        elif model_fn == model_2pl:
            return_sites = ["theta", "beta", "gamma", "tau", "delta", "alpha"]
        else:
            return_sites = ["theta", "beta", "gamma", "tau", "delta"]

    predictive = Predictive(model_fn, guide=guide, num_samples=N_POSTERIOR_SAMPLES,
                            return_sites=return_sites)
    samples = predictive(student_idx, prompt_idx, lang_idx, None, **model_kwargs)

    # ── Shape-safe extraction ──
    results = {
        'student_map': student_map, 'prompt_map': prompt_map, 'lang_map': lang_map,
        'num_students': num_students, 'num_prompts': num_prompts, 'num_langs': num_langs,
        'losses': losses, 'final_loss': losses[-1],
        'converged_at': converged_at,
        'student_idx': student_idx, 'prompt_idx': prompt_idx,
        'lang_idx': lang_idx, 'score_obs': score_obs,
        'tau_mask': tau_mask, 'gamma_mask': gamma_mask,
        'learning_rate': learning_rate,
    }

    # theta
    results['theta_mean'] = samples['theta'].detach().cpu().numpy().mean(axis=0).reshape(num_students).astype(np.float64)
    results['theta_std'] = samples['theta'].detach().cpu().numpy().std(axis=0).reshape(num_students).astype(np.float64)

    # beta — only exists in 1PL and 2PL, NOT in GRM
    if 'beta' in samples:
        results['beta_mean'] = samples['beta'].detach().cpu().numpy().mean(axis=0).reshape(num_prompts).astype(np.float64)
        results['beta_std'] = samples['beta'].detach().cpu().numpy().std(axis=0).reshape(num_prompts).astype(np.float64)
    elif 'beta_base' in samples:
        # GRM: use beta_base as the "effective difficulty" (first threshold)
        results['beta_mean'] = samples['beta_base'].detach().cpu().numpy().mean(axis=0).reshape(num_prompts).astype(np.float64)
        results['beta_std'] = samples['beta_base'].detach().cpu().numpy().std(axis=0).reshape(num_prompts).astype(np.float64)
    else:
        results['beta_mean'] = np.zeros(num_prompts, dtype=np.float64)
        results['beta_std'] = np.zeros(num_prompts, dtype=np.float64)

    # gamma
    results['gamma_mean'] = samples['gamma'].detach().cpu().numpy().mean(axis=0).reshape(num_langs).astype(np.float64)

    # tau
    results['tau_mean'] = samples['tau'].detach().cpu().numpy().mean(axis=0).reshape(num_prompts, num_langs).astype(np.float64)
    results['tau_std'] = samples['tau'].detach().cpu().numpy().std(axis=0).reshape(num_prompts, num_langs).astype(np.float64)

    # delta
    results['delta_mean'] = samples['delta'].detach().cpu().numpy().mean(axis=0).reshape(num_students, num_langs).astype(np.float64)

    # alpha — exists in 2PL and GRM
    if 'alpha' in samples:
        results['alpha_mean'] = samples['alpha'].detach().cpu().numpy().mean(axis=0).reshape(num_prompts).astype(np.float64)
        results['alpha_std'] = samples['alpha'].detach().cpu().numpy().std(axis=0).reshape(num_prompts).astype(np.float64)
    else:
        results['alpha_mean'] = np.ones(num_prompts, dtype=np.float64)
        results['alpha_std'] = np.zeros(num_prompts, dtype=np.float64)

    # thresholds — only GRM
    if 'thresholds' in samples:
        thresh_raw = samples['thresholds'].detach().cpu().numpy()
        n_thresh = thresh_raw.shape[-1]
        results['thresholds_mean'] = thresh_raw.mean(axis=0).reshape(num_prompts, n_thresh).astype(np.float64)
        results['thresholds_std'] = thresh_raw.std(axis=0).reshape(num_prompts, n_thresh).astype(np.float64)

    # Parameter counts
    n_params = num_students + num_prompts + num_langs + num_prompts * num_langs + num_students * num_langs
    if model_fn == model_2pl:
        n_params += num_prompts  # alpha
    if model_fn == model_grm:
        n_params += num_prompts * 3 + num_prompts  # 3 extra thresholds per item + alpha
    results['n_params'] = n_params
    results['n_data'] = len(student_idx)

    # Assertions
    assert results['theta_mean'].shape == (num_students,), f"theta: {results['theta_mean'].shape}"
    assert results['beta_mean'].shape == (num_prompts,), f"beta: {results['beta_mean'].shape}"
    assert results['gamma_mean'].shape == (num_langs,), f"gamma: {results['gamma_mean'].shape}"
    assert results['tau_mean'].shape == (num_prompts, num_langs), f"tau: {results['tau_mean'].shape}"
    assert results['delta_mean'].shape == (num_students, num_langs), f"delta: {results['delta_mean'].shape}"

    print(f"    [{label}] Extracted: θ={results['theta_mean'].shape}, β={results['beta_mean'].shape}, "
          f"γ={results['gamma_mean'].shape}, τ={results['tau_mean'].shape}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════

def compute_predicted_probs(results, model_type='1pl'):
    s_idx = results['student_idx'].cpu().numpy()
    p_idx = results['prompt_idx'].cpu().numpy()
    l_idx = results['lang_idx'].cpu().numpy()
    n = len(s_idx)
    probs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        si, pi, li = s_idx[i], p_idx[i], l_idx[i]
        ability = _to_scalar(results['theta_mean'][si]) + _to_scalar(results['delta_mean'][si, li])
        difficulty = _to_scalar(results['beta_mean'][pi]) + _to_scalar(results['gamma_mean'][li]) + _to_scalar(results['tau_mean'][pi, li])
        a = _to_scalar(results['alpha_mean'][pi]) if model_type == '2pl' else 1.0
        probs[i] = expit(a * (ability - difficulty))
    return probs


def compute_log_likelihood(results, model_type='1pl'):
    probs = compute_predicted_probs(results, model_type)
    obs = results['score_obs'].cpu().numpy().astype(np.float64)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    return float(np.sum(obs * np.log(probs) + (1 - obs) * np.log(1 - probs)))


def compute_bic(ll, n_params, n_data):
    return -2 * ll + n_params * np.log(n_data)


def compute_aic(ll, n_params):
    return -2 * ll + 2 * n_params


def compute_item_fit(results, model_type='1pl'):
    probs = compute_predicted_probs(results, model_type)
    obs = results['score_obs'].cpu().numpy().astype(np.float64)
    p_idx = results['prompt_idx'].cpu().numpy()
    prompts = sorted(results['prompt_map'].keys(), key=lambda x: results['prompt_map'][x])

    items = []
    for pi in range(results['num_prompts']):
        mask = (p_idx == pi)
        if mask.sum() < 10:
            continue
        p_i, o_i = probs[mask], obs[mask]
        var_i = np.clip(p_i * (1 - p_i), 1e-10, None)
        residuals = o_i - p_i
        z_sq = (residuals ** 2) / var_i
        outfit = float(np.mean(z_sq))
        infit = float(np.sum(residuals ** 2) / np.sum(var_i))
        pbis = float(pearsonr(o_i, p_i)[0]) if np.std(o_i) > 0 and np.std(p_i) > 0 else np.nan
        items.append({
            'prompt': prompts[pi] if pi < len(prompts) else str(pi),
            'prompt_idx': pi, 'n_obs': int(mask.sum()),
            'p_value': float(o_i.mean()), 'infit': infit, 'outfit': outfit,
            'point_biserial': pbis,
            'beta': _to_scalar(results['beta_mean'][pi]),
            'alpha': _to_scalar(results['alpha_mean'][pi]) if model_type == '2pl' else 1.0,
        })
    return pd.DataFrame(items)


def compute_person_fit(results, model_type='1pl'):
    probs = compute_predicted_probs(results, model_type)
    obs = results['score_obs'].cpu().numpy().astype(np.float64)
    s_idx = results['student_idx'].cpu().numpy()
    students = sorted(results['student_map'].keys(), key=lambda x: results['student_map'][x])

    persons = []
    for si in range(results['num_students']):
        mask = (s_idx == si)
        if mask.sum() < 10:
            continue
        p_i, o_i = probs[mask], obs[mask]
        var_i = np.clip(p_i * (1 - p_i), 1e-10, None)
        residuals = o_i - p_i
        z_sq = (residuals ** 2) / var_i
        persons.append({
            'student': students[si] if si < len(students) else str(si),
            'student_idx': si, 'n_obs': int(mask.sum()),
            'theta': _to_scalar(results['theta_mean'][si]),
            'infit': float(np.sum(residuals ** 2) / np.sum(var_i)),
            'outfit': float(np.mean(z_sq)),
        })
    return pd.DataFrame(persons)


def compute_item_information(theta_range, beta, alpha=1.0):
    p = expit(alpha * (theta_range - beta))
    return alpha ** 2 * p * (1 - p)


def compute_test_information(theta_range, results, model_type='1pl'):
    info = np.zeros_like(theta_range)
    for pi in range(results['num_prompts']):
        b = _to_scalar(results['beta_mean'][pi])
        a = _to_scalar(results['alpha_mean'][pi]) if model_type == '2pl' else 1.0
        info += compute_item_information(theta_range, b, a)
    return info


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    print(f"  Raw rows: {len(df):,}")
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['id'] = df['id'].apply(clean_id)
    df['score'] = (df['judge_score'] >= 4).astype(np.float32)
    df['grm_score'] = (df['judge_score'].clip(1, 5) - 1).astype(np.int64)

    student_col = 'test_taker' if 'test_taker' in df.columns else 'model'
    df['model_family'] = df[student_col].apply(get_model_family)
    print(f"  Valid rows: {len(df):,}")
    print(f"  Likert distribution:\n{df['judge_score'].value_counts().sort_index()}")

    anchor_ids = set()
    if os.path.exists(ANCHOR_FILE):
        adf = pd.read_csv(ANCHOR_FILE)
        adf['id'] = adf['id'].apply(clean_id)
        anchor_ids = set(adf['id'].unique())
        print(f"  Anchors: {len(anchor_ids)}")
    return df, anchor_ids


# ══════════════════════════════════════════════════════════════════════════
# A1: 1PL vs 2PL
# ══════════════════════════════════════════════════════════════════════════

def a1_compare_1pl_2pl(df, anchor_ids):
    print("\n" + "=" * 70)
    print("A1: 1PL vs 2PL MODEL COMPARISON")
    print(f"    Max steps: {MAX_TRAINING_STEPS}, Convergence: "
          f"window={CONVERGENCE_WINDOW}, threshold={CONVERGENCE_THRESHOLD}")
    print("=" * 70)

    print("\n  Fitting 1PL...")
    r1 = fit_model(model_1pl, df, anchor_ids, label="1PL")

    print("\n  Fitting 2PL...")
    r2 = fit_model(model_2pl, df, anchor_ids, label="2PL")

    ll_1pl = compute_log_likelihood(r1, '1pl')
    ll_2pl = compute_log_likelihood(r2, '2pl')
    bic_1pl = compute_bic(ll_1pl, r1['n_params'], r1['n_data'])
    bic_2pl = compute_bic(ll_2pl, r2['n_params'], r2['n_data'])
    aic_1pl = compute_aic(ll_1pl, r1['n_params'])
    aic_2pl = compute_aic(ll_2pl, r2['n_params'])

    print(f"\n  {'Metric':<25} {'1PL':>15} {'2PL':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Log-Likelihood':<25} {ll_1pl:>15.1f} {ll_2pl:>15.1f}")
    print(f"  {'AIC':<25} {aic_1pl:>15.1f} {aic_2pl:>15.1f}")
    print(f"  {'BIC':<25} {bic_1pl:>15.1f} {bic_2pl:>15.1f}")
    print(f"  {'N params':<25} {r1['n_params']:>15} {r2['n_params']:>15}")
    print(f"  {'Converged at step':<25} {r1['converged_at']:>15} {r2['converged_at']:>15}")
    print(f"  {'Learning rate':<25} {r1['learning_rate']:>15} {r2['learning_rate']:>15}")
    print(f"  {'Final ELBO':<25} {r1['final_loss']:>15.1f} {r2['final_loss']:>15.1f}")

    alpha = r2['alpha_mean']
    print(f"\n  2PL α: mean={np.mean(alpha):.3f}, std={np.std(alpha):.3f}, "
          f"range=[{np.min(alpha):.3f}, {np.max(alpha):.3f}]")
    print(f"  α < 0.5: {(alpha < 0.5).sum()}, α > 2.0: {(alpha > 2.0).sum()}")

    common = set(r1['student_map']) & set(r2['student_map'])
    th1 = [_to_scalar(r1['theta_mean'][r1['student_map'][s]]) for s in common]
    th2 = [_to_scalar(r2['theta_mean'][r2['student_map'][s]]) for s in common]
    r_theta, _ = pearsonr(th1, th2)
    common_p = set(r1['prompt_map']) & set(r2['prompt_map'])
    b1 = [_to_scalar(r1['beta_mean'][r1['prompt_map'][p]]) for p in common_p]
    b2 = [_to_scalar(r2['beta_mean'][r2['prompt_map'][p]]) for p in common_p]
    r_beta, _ = pearsonr(b1, b2)
    print(f"  θ correlation (1PL vs 2PL): {r_theta:.4f}")
    print(f"  β correlation (1PL vs 2PL): {r_beta:.4f}")

    comparison = {
        'll_1pl': ll_1pl, 'll_2pl': ll_2pl,
        'bic_1pl': bic_1pl, 'bic_2pl': bic_2pl,
        'aic_1pl': aic_1pl, 'aic_2pl': aic_2pl,
        'r_theta': r_theta, 'r_beta': r_beta,
        'converged_1pl': r1['converged_at'], 'converged_2pl': r2['converged_at'],
    }
    return r1, r2, comparison


# ══════════════════════════════════════════════════════════════════════════
# A2: GRM
# ══════════════════════════════════════════════════════════════════════════

def a2_graded_response(df, anchor_ids):
    print("\n" + "=" * 70)
    print("A2: GRADED RESPONSE MODEL (LIKERT 1-5)")
    print("=" * 70)

    print("\n  Fitting GRM...")
    r_grm = fit_model(model_grm, df, anchor_ids, label="GRM",
                      return_sites=["theta", "beta_base", "beta_increments",
                                    "gamma", "tau", "delta", "alpha", "thresholds"],
                      extra_kwargs={'num_categories': 5})

    print(f"  Converged at step: {r_grm['converged_at']}")

    if 'thresholds_mean' in r_grm:
        thresh = r_grm['thresholds_mean']
        print(f"\n  GRM Thresholds (shape {thresh.shape}):")
        for k in range(thresh.shape[1]):
            print(f"    Threshold {k+1}: mean={np.mean(thresh[:, k]):.3f}, "
                  f"range=[{np.min(thresh[:, k]):.3f}, {np.max(thresh[:, k]):.3f}]")
    return r_grm


# ══════════════════════════════════════════════════════════════════════════
# A3–A7: FIT STATISTICS AND PLOTS
# ══════════════════════════════════════════════════════════════════════════

def a3_a7_plots(r1, r2, r_grm, comparison, df):
    print("\n" + "=" * 70)
    print("A3-A7: FIT DIAGNOSTICS AND PLOTS")
    print("=" * 70)

    # Item and person fit
    item_fit_1pl = compute_item_fit(r1, '1pl')
    item_fit_2pl = compute_item_fit(r2, '2pl')
    person_fit_1pl = compute_person_fit(r1, '1pl')
    person_fit_2pl = compute_person_fit(r2, '2pl')

    item_fit_1pl.to_csv(os.path.join(RESULTS_DIR, "A3_item_fit_1pl.csv"), index=False)
    item_fit_2pl.to_csv(os.path.join(RESULTS_DIR, "A3_item_fit_2pl.csv"), index=False)
    person_fit_1pl.to_csv(os.path.join(RESULTS_DIR, "A4_person_fit_1pl.csv"), index=False)
    person_fit_2pl.to_csv(os.path.join(RESULTS_DIR, "A4_person_fit_2pl.csv"), index=False)

    print(f"  1PL: infit mean={item_fit_1pl['infit'].mean():.3f}, misfit={((item_fit_1pl['infit'] > 1.3) | (item_fit_1pl['outfit'] > 1.3)).sum()}")
    print(f"  2PL: infit mean={item_fit_2pl['infit'].mean():.3f}, misfit={((item_fit_2pl['infit'] > 1.3) | (item_fit_2pl['outfit'] > 1.3)).sum()}")

    # ── Figure A.1: Model Comparison (4 panels) ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Convergence (with convergence markers)
    ax = axes[0, 0]
    w = 50
    for res, lbl, col in [(r1, '1PL', '#3498db'), (r2, '2PL', '#e74c3c')]:
        losses = res['losses']
        if len(losses) > w:
            sm = np.convolve(losses, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(losses)), sm, color=col, label=lbl, linewidth=2)
        ax.axvline(res['converged_at'], color=col, linestyle=':', alpha=0.7,
                   label=f'{lbl} converged ({res["converged_at"]})')
    if r_grm is not None:
        losses_g = r_grm['losses']
        if len(losses_g) > w:
            sm_g = np.convolve(losses_g, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(losses_g)), sm_g, color='#2ecc71', label='GRM', linewidth=2)
        ax.axvline(r_grm['converged_at'], color='#2ecc71', linestyle=':', alpha=0.7,
                   label=f'GRM converged ({r_grm["converged_at"]})')
    ax.set_xlabel('Step'); ax.set_ylabel('ELBO Loss')
    ax.set_title('Training Convergence\n(dotted = convergence point)', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # AIC/BIC
    ax = axes[0, 1]
    x = np.arange(2); w_bar = 0.3
    ax.bar(x - w_bar/2, [comparison['aic_1pl'], comparison['bic_1pl']], w_bar, label='1PL', color='#3498db', edgecolor='black')
    ax.bar(x + w_bar/2, [comparison['aic_2pl'], comparison['bic_2pl']], w_bar, label='2PL', color='#e74c3c', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(['AIC', 'BIC'])
    ax.set_title('Model Selection (lower = better)', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # α distribution
    ax = axes[1, 0]
    ax.hist(r2['alpha_mean'], bins=40, edgecolor='black', alpha=0.7, color='#e74c3c')
    ax.axvline(1.0, color='blue', linestyle='--', linewidth=2, label='α=1 (1PL)')
    ax.axvline(np.mean(r2['alpha_mean']), color='black', linestyle='-', linewidth=2, label=f'Mean={np.mean(r2["alpha_mean"]):.2f}')
    ax.set_xlabel('Discrimination (α)'); ax.set_ylabel('Count')
    ax.set_title('2PL Discrimination', fontweight='bold'); ax.legend()

    # θ agreement
    ax = axes[1, 1]
    common = set(r1['student_map']) & set(r2['student_map'])
    th1 = [_to_scalar(r1['theta_mean'][r1['student_map'][s]]) for s in common]
    th2 = [_to_scalar(r2['theta_mean'][r2['student_map'][s]]) for s in common]
    ax.scatter(th1, th2, s=50, alpha=0.7, edgecolors='black', linewidth=0.5, c='steelblue')
    lims = [min(min(th1), min(th2)) - 0.3, max(max(th1), max(th2)) + 0.3]
    ax.plot(lims, lims, 'r--', label=f'r={comparison["r_theta"]:.3f}')
    ax.set_xlabel('θ (1PL)'); ax.set_ylabel('θ (2PL)')
    ax.set_title('Ability Agreement', fontweight='bold')
    ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    plt.suptitle('A1: Model Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(RESULTS_DIR, "A1_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(); print(f"  Saved: A1_model_comparison.png")

    # ── Figure A.2: Item Fit ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, fdf, lbl in [(axes[0], item_fit_1pl, '1PL'), (axes[1], item_fit_2pl, '2PL')]:
        ax.scatter(fdf['infit'], fdf['outfit'], alpha=0.5, s=30, c=fdf['p_value'], cmap='RdYlGn', edgecolors='black', linewidth=0.3)
        ax.axvline(1.3, color='red', linestyle='--', alpha=0.5); ax.axhline(1.3, color='red', linestyle='--', alpha=0.5)
        ax.axvline(0.7, color='orange', linestyle='--', alpha=0.5); ax.axhline(0.7, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='gray', linestyle='-', alpha=0.3); ax.axhline(1.0, color='gray', linestyle='-', alpha=0.3)
        n_mis = ((fdf['infit'] > 1.3) | (fdf['outfit'] > 1.3)).sum()
        ax.set_xlabel('Infit MNSQ'); ax.set_ylabel('Outfit MNSQ')
        ax.set_title(f'{lbl}: Item Fit ({n_mis} misfitting)', fontweight='bold'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "A2_item_fit.png"), dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Saved: A2_item_fit.png")

    # ── Figure A.3: Person Fit ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fam_colors = {'GPT': '#3498db', 'Claude': '#9b59b6', 'Gemini': '#2ecc71', 'Grok': '#e74c3c', 'DeepSeek': '#f39c12', 'Other': '#95a5a6'}
    for ax, pf, lbl in [(axes[0], person_fit_1pl, '1PL'), (axes[1], person_fit_2pl, '2PL')]:
        c = [fam_colors.get(get_model_family(s), '#666') for s in pf['student']]
        ax.scatter(pf['theta'], pf['infit'], alpha=0.7, s=60, c=c, edgecolors='black', linewidth=0.5)
        ax.axhline(1.3, color='red', linestyle='--', alpha=0.5); ax.axhline(0.7, color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('θ'); ax.set_ylabel('Infit MNSQ'); ax.set_title(f'{lbl}: Person Fit', fontweight='bold')
        for fam, col in fam_colors.items():
            if any(get_model_family(s) == fam for s in pf['student']): ax.scatter([], [], c=col, s=60, label=fam)
        ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "A3_person_fit.png"), dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Saved: A3_person_fit.png")

    # ── Figure A.4: Information Functions ──
    theta_range = np.linspace(-4, 4, 200)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    ti_1pl = compute_test_information(theta_range, r1, '1pl')
    ti_2pl = compute_test_information(theta_range, r2, '2pl')
    ax = axes[0]; ax.plot(theta_range, ti_1pl, 'b-', linewidth=2, label='1PL'); ax.plot(theta_range, ti_2pl, 'r-', linewidth=2, label='2PL')
    ax.set_xlabel('θ'); ax.set_ylabel('I(θ)'); ax.set_title('Test Information', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]; prompts_list = sorted(r2['prompt_map'].keys(), key=lambda x: r2['prompt_map'][x])
    alpha_sorted = np.argsort(r2['alpha_mean'])
    for pi in alpha_sorted[:3]:
        b, a = _to_scalar(r2['beta_mean'][pi]), _to_scalar(r2['alpha_mean'][pi])
        ax.plot(theta_range, compute_item_information(theta_range, b, a), '--', alpha=0.7, label=f'α={a:.2f}')
    for pi in alpha_sorted[-3:]:
        b, a = _to_scalar(r2['beta_mean'][pi]), _to_scalar(r2['alpha_mean'][pi])
        ax.plot(theta_range, compute_item_information(theta_range, b, a), '-', linewidth=2, label=f'α={a:.2f}')
    ax.set_xlabel('θ'); ax.set_ylabel('Item Info'); ax.set_title('Item Information (high vs low α)', fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = axes[2]; ax.scatter(r2['beta_mean'], r2['alpha_mean'], alpha=0.5, s=30, edgecolors='black', linewidth=0.3, c='steelblue')
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='α=1'); ax.set_xlabel('β'); ax.set_ylabel('α')
    ax.set_title('Difficulty vs Discrimination', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "A4_information_functions.png"), dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Saved: A4_information_functions.png")

    # ── Figure A.5: ICC Comparison ──
    fig, axes_icc = plt.subplots(2, 4, figsize=(20, 9)); axes_flat = axes_icc.flatten()
    alpha_order = np.argsort(r2['alpha_mean']); selected = list(alpha_order[:4]) + list(alpha_order[-4:])
    for idx, pi in enumerate(selected):
        ax = axes_flat[idx]; pname = prompts_list[pi] if pi < len(prompts_list) else str(pi)
        b1_val = _to_scalar(r1['beta_mean'][pi]); b2_val = _to_scalar(r2['beta_mean'][pi]); a2 = _to_scalar(r2['alpha_mean'][pi])
        ax.plot(theta_range, expit(theta_range - b1_val), 'b-', linewidth=2, label='1PL')
        ax.plot(theta_range, expit(a2 * (theta_range - b2_val)), 'r-', linewidth=2, label='2PL')
        ax.set_title(f'P{pname} (α={a2:.2f})', fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.05); ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle('ICCs: 1PL vs 2PL (left=low α, right=high α)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(os.path.join(RESULTS_DIR, "A5_ICC_comparison.png"), dpi=300, bbox_inches='tight'); plt.close()
    print(f"  Saved: A5_ICC_comparison.png")

    # ── Figure A.6: GRM Category Curves ──
    if r_grm is not None and 'thresholds_mean' in r_grm:
        fig, axes_g = plt.subplots(2, 4, figsize=(20, 9)); af = axes_g.flatten()
        grm_prompts = sorted(r_grm['prompt_map'].keys(), key=lambda x: r_grm['prompt_map'][x])
        step_g = max(1, r_grm['num_prompts'] // 8)
        selected_g = list(range(0, r_grm['num_prompts'], step_g))[:8]
        cat_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
        for idx, pi in enumerate(selected_g):
            if idx >= 8: break
            ax = af[idx]; thresh = r_grm['thresholds_mean'][pi]; a = _to_scalar(r_grm['alpha_mean'][pi])
            n_t = len(thresh); cum = np.zeros((len(theta_range), n_t + 2)); cum[:, 0] = 1.0
            for k in range(n_t): cum[:, k+1] = expit(a * (theta_range - thresh[k]))
            for c in range(n_t + 1):
                cp = cum[:, c] - cum[:, c+1]; ax.fill_between(theta_range, 0, cp, alpha=0.3, color=cat_colors[c])
                ax.plot(theta_range, cp, color=cat_colors[c], linewidth=1.5, label=f'Score {c+1}')
            pn = grm_prompts[pi] if pi < len(grm_prompts) else str(pi)
            ax.set_title(f'P{pn} (α={a:.2f})', fontsize=10, fontweight='bold')
            if idx == 0: ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.2)
        for idx in range(len(selected_g), 8): af[idx].set_visible(False)
        plt.suptitle('GRM Category Response Functions', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(os.path.join(RESULTS_DIR, "A6_GRM_category_curves.png"), dpi=300, bbox_inches='tight'); plt.close()
        print(f"  Saved: A6_GRM_category_curves.png")

    # ── Summary Table ──
    summary = pd.DataFrame([
        {'Model': '1PL', 'LL': comparison['ll_1pl'], 'AIC': comparison['aic_1pl'], 'BIC': comparison['bic_1pl'],
         'N_params': r1['n_params'], 'Converged_step': r1['converged_at'], 'LR': r1['learning_rate'],
         'Mean_Infit': item_fit_1pl['infit'].mean(), 'Misfit_Items': ((item_fit_1pl['infit'] > 1.3) | (item_fit_1pl['outfit'] > 1.3)).sum(),
         'Person_Misfit': (person_fit_1pl['infit'] > 1.3).sum()},
        {'Model': '2PL', 'LL': comparison['ll_2pl'], 'AIC': comparison['aic_2pl'], 'BIC': comparison['bic_2pl'],
         'N_params': r2['n_params'], 'Converged_step': r2['converged_at'], 'LR': r2['learning_rate'],
         'Mean_Infit': item_fit_2pl['infit'].mean(), 'Misfit_Items': ((item_fit_2pl['infit'] > 1.3) | (item_fit_2pl['outfit'] > 1.3)).sum(),
         'Person_Misfit': (person_fit_2pl['infit'] > 1.3).sum(),
         'Mean_Alpha': np.mean(r2['alpha_mean']), 'Std_Alpha': np.std(r2['alpha_mean'])},
        {'Model': 'GRM', 'N_params': r_grm['n_params'] if r_grm else np.nan,
         'Converged_step': r_grm['converged_at'] if r_grm else np.nan,
         'LR': r_grm['learning_rate'] if r_grm else np.nan},
    ])
    summary.to_csv(os.path.join(RESULTS_DIR, "A_summary_table.csv"), index=False)

    # θ agreement across all models
    agreement = [{'Comparison': 'θ (1PL vs 2PL)', 'Pearson r': comparison['r_theta']},
                 {'Comparison': 'β (1PL vs 2PL)', 'Pearson r': comparison['r_beta']}]
    if r_grm is not None:
        cs = set(r1['student_map']) & set(r_grm['student_map'])
        if len(cs) >= 3:
            t1g = [_to_scalar(r1['theta_mean'][r1['student_map'][s]]) for s in cs]
            tg = [_to_scalar(r_grm['theta_mean'][r_grm['student_map'][s]]) for s in cs]
            r_1g, _ = pearsonr(t1g, tg)
            agreement.append({'Comparison': 'θ (1PL vs GRM)', 'Pearson r': r_1g})
            print(f"  θ (1PL vs GRM): r={r_1g:.4f}")
    pd.DataFrame(agreement).to_csv(os.path.join(RESULTS_DIR, "A_parameter_agreement.csv"), index=False)

    print(f"\n  Summary:\n{summary[['Model', 'LL', 'AIC', 'BIC', 'N_params', 'Converged_step']].to_string(index=False)}")
    return summary


# ══════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════

def run_experiment_a():
    print("=" * 70)
    print("EXPERIMENT A: IRT MODEL COMPARISON")
    print(f"  Config: max_steps={MAX_TRAINING_STEPS}, convergence_window={CONVERGENCE_WINDOW}, "
          f"threshold={CONVERGENCE_THRESHOLD}, min_steps={MIN_TRAINING_STEPS}")
    print("=" * 70)

    df, anchor_ids = load_data()
    r1, r2, comparison = a1_compare_1pl_2pl(df, anchor_ids)
    r_grm = a2_graded_response(df, anchor_ids)
    summary = a3_a7_plots(r1, r2, r_grm, comparison, df)

    print("\n" + "=" * 70)
    print("EXPERIMENT A COMPLETE")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)

    print("\n── KEY TAKEAWAYS ──")
    print(f"  1PL converged at step {r1['converged_at']}, 2PL at {r2['converged_at']}")
    if r_grm: print(f"  GRM converged at step {r_grm['converged_at']}")
    if comparison['bic_2pl'] < comparison['bic_1pl']:
        print(f"  2PL preferred by BIC (ΔBIC = {comparison['bic_1pl'] - comparison['bic_2pl']:.0f})")
    else:
        print(f"  1PL preferred by BIC (ΔBIC = {comparison['bic_2pl'] - comparison['bic_1pl']:.0f})")
        print(f"  → Justifies the simpler 1PL model")
    print(f"  θ highly correlated across models (r={comparison['r_theta']:.3f}) → rankings robust")


if __name__ == "__main__":
    run_experiment_a()