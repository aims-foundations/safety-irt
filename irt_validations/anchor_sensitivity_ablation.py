# -*- coding: utf-8 -*-
"""
Anchor Sensitivity Ablation (2PL Bayesian IRT).
================================================
Runs the same model under six anchor conditions and compares θ/γ/τ stability.

Anchors are selected via iterative purification with Lord's χ²(2) test
(96.5% agreement with MTT forward selection, Kopf et al. 2015).
anchors.csv = items DIF-free across all languages after BH-corrected purification.

Conditions (τ prior held fixed = StudentT df=1 Horseshoe throughout):
  lords_dif        -- Lord's iterative purification anchors (reference)
  lords_small      -- random 50% subsample of Lord's anchors
  lords_large      -- random 150% supersample of Lord's anchors
  random_small     -- random 50% of |lords_dif|, drawn from all prompts
  random_matched   -- random |lords_dif| prompts, drawn from all prompts
  category_balanced-- stratified by prompt category (if available)

Stability metrics (all vs. strict reference):
  θ: Spearman ρ + RMSE
  γ: Pearson r + max |Δγ|
  τ: RMSE (non-anchor, non-English cells)

Outputs (results_anchor_sensitivity/):
  anchor_conditions.csv          -- anchor IDs per condition
  params_all_conditions.csv      -- θ/γ/τ for every condition
  stability_summary.csv          -- correlation / RMSE table
  theta_stability.png            -- scatter grid θ_cond vs θ_strict
  gamma_stability.png            -- grouped bar γ per language
  tau_stability.png              -- RMSE of τ per language × condition
  convergence.png                -- ELBO loss curves
"""

import os
import sys
import warnings
import itertools
warnings.filterwarnings('ignore')

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from huggingface_hub import snapshot_download

# ── fig_style integration ────────────────────────────────────────────────────
_sys = sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from fig_style import (apply_style, savefig as fs_savefig, make_fig_grid,
                           C_RED, C_BLUE, C_PURPLE, COLORS_3, LANG_ORDER)
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False
    C_RED, C_BLUE, C_PURPLE = '#c0392b', '#2471a3', '#7d3c98'
    COLORS_3 = [C_BLUE, C_RED, C_PURPLE]
    LANG_ORDER = None

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = snapshot_download(repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False)
INPUT_FILE  = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "anchors", "anchors.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_anchor_sensitivity")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
MAX_STEPS    = 4000
CONV_WINDOW  = 200
CONV_THRESH  = 1e-4
MIN_STEPS    = 1000
N_SAMPLES    = 500
SEED         = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── condition palette ─────────────────────────────────────────────────────────
COND_COLORS = {
    'lords_dif':         '#2c3e50',
    'lords_small':       C_BLUE,
    'lords_large':       C_PURPLE,
    'random_small':      '#e67e22',
    'random_matched':    '#27ae60',
    'category_balanced': C_RED,
}
REFERENCE_COND = 'lords_dif'


# ── helpers ──────────────────────────────────────────────────────────────────

def clean_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def check_convergence(losses, window, threshold, min_steps):
    if len(losses) < min_steps or len(losses) < 2 * window:
        return False
    recent   = np.mean(losses[-window:])
    previous = np.mean(losses[-2*window:-window])
    if previous == 0:
        return True
    return (previous - recent) / abs(previous) < threshold


# ── 2PL model (Horseshoe τ prior, identical to main model/irt.py) ────────────

def model_2pl(student_idx, prompt_idx, lang_idx, obs=None,
              num_students=None, num_prompts=None, num_langs=None,
              tau_mask=None, gamma_mask=None):
    theta = pyro.sample("theta",
        dist.Normal(torch.zeros(num_students, device=device), 1.0).to_event(1))
    beta = pyro.sample("beta",
        dist.Normal(torch.zeros(num_prompts, device=device), 1.0).to_event(1))
    alpha = pyro.sample("alpha",
        dist.LogNormal(0.5 * torch.ones(num_prompts, device=device),
                       0.5 * torch.ones(num_prompts, device=device)).to_event(1))
    gamma_raw = pyro.sample("gamma_raw",
        dist.Normal(torch.zeros(num_langs, device=device), 1.0).to_event(1))
    gamma = pyro.deterministic("gamma", gamma_raw * gamma_mask)

    tau_scale = pyro.sample("tau_scale",
        dist.HalfCauchy(torch.ones(1, device=device)).to_event(1))
    tau_raw = pyro.sample("tau_raw",
        dist.StudentT(1.0,
                      torch.zeros(num_prompts, num_langs, device=device),
                      tau_scale).to_event(2))
    tau = pyro.deterministic("tau", tau_raw * tau_mask)

    delta_raw = pyro.sample("delta_raw",
        dist.Normal(torch.zeros(num_students, num_langs, device=device), 0.5).to_event(2))
    delta_mask = gamma_mask.unsqueeze(0).expand(num_students, -1)
    delta = pyro.deterministic("delta", delta_raw * delta_mask)

    with pyro.plate("data", len(student_idx)):
        ability     = theta[student_idx] + delta[student_idx, lang_idx]
        difficulty  = beta[prompt_idx] + gamma[lang_idx] + tau[prompt_idx, lang_idx]
        logits      = alpha[prompt_idx] * (ability - difficulty)
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=obs)


# ── anchor set builder ────────────────────────────────────────────────────────

def build_anchor_sets(df, rng=None):
    """
    Returns dict: condition_name -> set of prompt IDs (strings).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    all_prompts = sorted(df['id'].unique())

    # Load anchors.csv
    if not os.path.exists(ANCHOR_FILE):
        raise FileNotFoundError(f"Anchor file not found: {ANCHOR_FILE}")

    adf = pd.read_csv(ANCHOR_FILE)
    adf['id'] = adf['id'].apply(clean_id)

    # Lord's DIF anchors: all items in anchors.csv are DIF-free by construction
    lords_ids = set(adf['id'].unique())
    n_lords   = len(lords_ids)
    n_small   = max(10, int(n_lords * 0.5))
    n_large   = int(n_lords * 1.5)

    lords_arr = np.array(sorted(lords_ids))
    lords_perm = rng.permutation(len(lords_arr))

    # Subsample / supersample of the Lord's anchor set itself
    lords_small_ids = set(lords_arr[lords_perm[:n_small]])
    lords_large_ids = lords_ids | set(
        np.array(all_prompts)[rng.permutation(len(all_prompts))[:(n_large - n_lords)]])

    # Random conditions: sample from ALL prompts (no DIF screening)
    all_arr = np.array(all_prompts)
    perm    = rng.permutation(len(all_arr))

    random_small_ids   = set(all_arr[perm[:n_small]])
    random_matched_ids = set(all_arr[perm[:n_lords]])

    # Category-balanced: stratified sample by prompt tag/category
    cat_col = 'tags' if 'tags' in df.columns else \
              next((c for c in df.columns if 'category' in c.lower()), None)
    if cat_col:
        # tags may be stored as string repr of a list — extract first tag
        tag_series = df.drop_duplicates('id').set_index('id')[cat_col].astype(str)
        tag_map = tag_series.str.strip("[]'\"").str.split("'").str[0].str.strip("[], ")
        categories = tag_map.unique()
        per_cat = max(1, n_lords // len(categories))
        cat_ids = set()
        for cat in categories:
            cat_prompts = [p for p in all_prompts if tag_map.get(p) == cat]
            if not cat_prompts:
                continue
            sampled = rng.choice(cat_prompts,
                                 size=min(per_cat, len(cat_prompts)),
                                 replace=False)
            cat_ids.update(sampled)
        category_balanced_ids = cat_ids
        print(f"  Category-balanced: {len(cat_ids)} anchors across {len(categories)} tags")
    else:
        # Fallback: evenly-spaced sample
        step = max(1, len(all_prompts) // n_lords)
        category_balanced_ids = set(all_arr[::step][:n_lords])
        print(f"  [WARN] No tags/category column found — using evenly-spaced sample")

    conditions = {
        'lords_dif':         lords_ids,
        'lords_small':       lords_small_ids,
        'lords_large':       lords_large_ids,
        'random_small':      random_small_ids,
        'random_matched':    random_matched_ids,
        'category_balanced': category_balanced_ids,
    }

    print("\n  Anchor set sizes:")
    for name, ids in conditions.items():
        ref = " ← REFERENCE" if name == REFERENCE_COND else ""
        print(f"    {name:20s}: {len(ids):4d} prompts{ref}")

    return conditions


# ── fitting ───────────────────────────────────────────────────────────────────

def fit_condition(df, anchor_ids, cond_name):
    """
    Fit 2PL model for a given anchor set. Returns dict of extracted parameters.
    """
    pyro.clear_param_store()

    sc        = 'test_taker' if 'test_taker' in df.columns else 'model'
    students  = sorted(df[sc].unique())
    prompts   = sorted(df['id'].unique())
    languages = sorted(df['language'].unique())

    s_map = {s: i for i, s in enumerate(students)}
    p_map = {p: i for i, p in enumerate(prompts)}
    l_map = {l: i for i, l in enumerate(languages)}
    ns, np_, nl = len(students), len(prompts), len(languages)

    s_idx = torch.tensor(df[sc].map(s_map).values, dtype=torch.long).to(device)
    p_idx = torch.tensor(df['id'].map(p_map).values, dtype=torch.long).to(device)
    l_idx = torch.tensor(df['language'].map(l_map).values, dtype=torch.long).to(device)
    obs   = torch.tensor(df['score'].values, dtype=torch.float32).to(device)

    tau_mask   = torch.ones((np_, nl), device=device)
    gamma_mask = torch.ones(nl, device=device)
    if 'en' in l_map:
        ei = l_map['en']
        tau_mask[:, ei]  = 0.0
        gamma_mask[ei]   = 0.0

    # Apply anchor constraints (τ_iL = 0 for anchor items)
    n_applied = 0
    for pid in prompts:
        if pid in anchor_ids:
            tau_mask[p_map[pid], :] = 0.0
            n_applied += 1
    print(f"  [{cond_name}] {n_applied}/{len(anchor_ids)} anchors matched in data")

    guide = pyro.infer.autoguide.AutoNormal(
        pyro.poutine.block(model_2pl, hide=["obs", "tau", "gamma", "delta"]))
    optimizer = ClippedAdam({"lr": 0.01, "clip_norm": 10.0})
    svi = SVI(model_2pl, guide, optimizer, loss=Trace_ELBO())

    losses = []
    pbar   = tqdm(range(MAX_STEPS), desc=f"[{cond_name}]", leave=False)
    converged_at = MAX_STEPS

    for step in pbar:
        loss = svi.step(s_idx, p_idx, l_idx, obs,
                        ns, np_, nl, tau_mask, gamma_mask)
        losses.append(loss)
        if step % 200 == 0:
            pbar.set_description(f"[{cond_name}] Loss: {loss:.1f}")
        if check_convergence(losses, CONV_WINDOW, CONV_THRESH, MIN_STEPS):
            converged_at = step + 1
            pbar.close()
            break

    print(f"  [{cond_name}] Converged at step {converged_at}")

    # Posterior samples — include theta, gamma, tau
    pred  = Predictive(model_2pl, guide=guide, num_samples=N_SAMPLES,
                       return_sites=["theta", "gamma", "tau", "beta", "alpha"])
    samps = pred(s_idx, p_idx, l_idx, None, ns, np_, nl, tau_mask, gamma_mask)

    theta_mean = samps['theta'].mean(0).detach().cpu().numpy().reshape(ns)
    gamma_mean = samps['gamma'].mean(0).detach().cpu().numpy().reshape(nl)
    tau_mean   = samps['tau'].mean(0).detach().cpu().numpy().reshape(np_, nl)
    tau_std    = samps['tau'].std(0).detach().cpu().numpy().reshape(np_, nl)
    beta_mean  = samps['beta'].mean(0).detach().cpu().numpy().reshape(np_)
    alpha_mean = samps['alpha'].mean(0).detach().cpu().numpy().reshape(np_)

    return {
        'condition':     cond_name,
        'theta':         theta_mean,      # (ns,)
        'gamma':         gamma_mean,      # (nl,)
        'tau_mean':      tau_mean,        # (np_, nl)
        'tau_std':       tau_std,         # (np_, nl)
        'beta':          beta_mean,       # (np_,)
        'alpha':         alpha_mean,      # (np_,)
        'tau_mask':      tau_mask.cpu().numpy(),
        'students':      students,
        'prompts':       prompts,
        'languages':     languages,
        's_map':         s_map,
        'p_map':         p_map,
        'l_map':         l_map,
        'losses':        losses,
        'converged_at':  converged_at,
        'n_anchors':     n_applied,
    }


# ── stability metrics ─────────────────────────────────────────────────────────

def compute_stability(results, ref_cond=REFERENCE_COND):
    """
    Compare each condition vs. reference on θ, γ, τ.
    Returns a DataFrame of stability metrics.
    """
    ref = results[ref_cond]
    rows = []

    for cond_name, res in results.items():
        # ── θ stability ──────────────────────────────────────────
        # Both runs use the same student ordering since data is identical
        theta_r = ref['theta']
        theta_c = res['theta']
        rho_theta, _ = spearmanr(theta_r, theta_c)
        rmse_theta   = np.sqrt(np.mean((theta_r - theta_c) ** 2))

        # ── γ stability ──────────────────────────────────────────
        # Only non-English languages
        non_en_idx = [i for l, i in ref['l_map'].items() if l != 'en']
        gamma_r = ref['gamma'][non_en_idx]
        gamma_c = res['gamma'][non_en_idx]
        r_gamma, _ = pearsonr(gamma_r, gamma_c) if len(gamma_r) >= 3 else (np.nan, np.nan)
        maxd_gamma  = np.max(np.abs(gamma_r - gamma_c))

        # ── τ stability ──────────────────────────────────────────
        # Compare only free (non-anchor, non-English) cells
        # Use reference anchor mask to define "free" consistently
        tau_r = ref['tau_mean']
        tau_c = res['tau_mean']
        free_mask = (ref['tau_mask'] > 0)   # True = not constrained to 0
        if free_mask.any():
            rmse_tau = np.sqrt(np.mean((tau_r[free_mask] - tau_c[free_mask]) ** 2))
            mae_tau  = np.mean(np.abs(tau_r[free_mask] - tau_c[free_mask]))
        else:
            rmse_tau = mae_tau = np.nan

        rows.append({
            'condition':      cond_name,
            'n_anchors':      res['n_anchors'],
            'converged_at':   res['converged_at'],
            'spearman_theta': rho_theta,
            'rmse_theta':     rmse_theta,
            'pearson_gamma':  r_gamma,
            'max_delta_gamma': maxd_gamma,
            'rmse_tau':       rmse_tau,
            'mae_tau':        mae_tau,
        })

    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_theta_stability(results, ref_cond=REFERENCE_COND):
    """
    Grid of scatter plots: θ_cond vs θ_strict for each non-reference condition.
    """
    ref     = results[ref_cond]
    theta_r = ref['theta']
    others  = [c for c in results if c != ref_cond]
    n       = len(others)

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, cond in enumerate(others):
        ax      = axes[idx // ncols][idx % ncols]
        theta_c = results[cond]['theta']
        rho, _  = spearmanr(theta_r, theta_c)
        rmse    = np.sqrt(np.mean((theta_r - theta_c) ** 2))

        ax.scatter(theta_r, theta_c, s=18, alpha=0.55,
                   color=COND_COLORS.get(cond, '#555555'),
                   edgecolors='none')
        lims = [min(theta_r.min(), theta_c.min()) - 0.1,
                max(theta_r.max(), theta_c.max()) + 0.1]
        ax.plot(lims, lims, color='black', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel(f'θ  ({ref_cond})', fontsize=9)
        ax.set_ylabel(f'θ  ({cond})', fontsize=9)
        ax.set_title(f'{cond}\nρ = {rho:.3f},  RMSE = {rmse:.3f}', fontsize=9)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect('equal')

    # Hide unused axes
    for j in range(idx + 1, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f'θ Stability vs. {ref_cond} anchors', fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "theta_stability.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: theta_stability.png")


def plot_gamma_stability(results, ref_cond=REFERENCE_COND):
    """
    Grouped bar chart of γ_L per language per condition.
    Reference condition drawn with hatching for easy comparison.
    """
    ref    = results[ref_cond]
    l_map  = ref['l_map']
    langs  = [l for l in (LANG_ORDER or sorted(l_map.keys())) if l in l_map and l != 'en']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: grouped bars
    ax    = axes[0]
    x     = np.arange(len(langs))
    width = 0.8 / len(results)
    for i, (cond, res) in enumerate(results.items()):
        vals = [res['gamma'][l_map[l]] for l in langs]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=cond, color=COND_COLORS.get(cond, '#888'),
                      edgecolor='black', linewidth=0.4,
                      hatch='//' if cond == ref_cond else '')
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=9)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('γ_L', fontsize=10)
    ax.set_title('Language Shift γ by Anchor Condition', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.2)

    # Right: deviation from reference
    ax = axes[1]
    for cond, res in results.items():
        if cond == ref_cond:
            continue
        deviations = [res['gamma'][l_map[l]] - ref['gamma'][l_map[l]] for l in langs]
        ax.plot(langs, deviations, 'o-', label=cond,
                color=COND_COLORS.get(cond, '#888'), markersize=5, linewidth=1.5)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.set_ylabel(f'Δγ  (cond − {ref_cond})', fontsize=10)
    ax.set_title('γ Deviation from Reference', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle('γ Stability Across Anchor Conditions', fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "gamma_stability.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: gamma_stability.png")


def plot_tau_stability(results, ref_cond=REFERENCE_COND):
    """
    RMSE of τ vs. reference, broken down by language.
    Also shows distribution of τ values per condition.
    """
    ref    = results[ref_cond]
    l_map  = ref['l_map']
    langs  = [l for l in (LANG_ORDER or sorted(l_map.keys())) if l in l_map and l != 'en']
    others = [c for c in results if c != ref_cond]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: per-language RMSE bar chart
    ax    = axes[0]
    x     = np.arange(len(langs))
    width = 0.8 / len(others)
    for i, cond in enumerate(others):
        rmse_per_lang = []
        for lang in langs:
            li       = l_map[lang]
            free_row = ref['tau_mask'][:, li] > 0
            t_ref    = ref['tau_mean'][free_row, li]
            t_cond   = results[cond]['tau_mean'][free_row, li]
            rmse     = np.sqrt(np.mean((t_ref - t_cond) ** 2)) if free_row.any() else np.nan
            rmse_per_lang.append(rmse)
        ax.bar(x + i * width - 0.4 + width / 2, rmse_per_lang, width,
               label=cond, color=COND_COLORS.get(cond, '#888'),
               edgecolor='black', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=9)
    ax.set_ylabel(f'RMSE(τ vs {ref_cond})', fontsize=10)
    ax.set_title('τ RMSE by Language vs. Reference', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.2)

    # Right: τ distribution per condition (violin/box)
    ax = axes[1]
    tau_data   = []
    cond_labels = []
    for cond, res in results.items():
        free = res['tau_mask'] > 0
        vals = res['tau_mean'][free]
        tau_data.append(vals)
        cond_labels.append(cond)

    parts = ax.violinplot(tau_data, positions=range(len(cond_labels)),
                          showmedians=True, widths=0.6)
    for i, (body, cond) in enumerate(zip(parts['bodies'], cond_labels)):
        body.set_facecolor(COND_COLORS.get(cond, '#888'))
        body.set_alpha(0.6)
    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, rotation=20, ha='right', fontsize=8)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_ylabel('τ_iL (free cells)', fontsize=10)
    ax.set_title('τ Distribution per Condition', fontweight='bold')
    ax.grid(axis='y', alpha=0.2)

    fig.suptitle('τ Stability Across Anchor Conditions', fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "tau_stability.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: tau_stability.png")


def plot_convergence(results):
    """ELBO loss curves for all conditions."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for cond, res in results.items():
        losses = np.array(res['losses'])
        ax.plot(losses, alpha=0.8, label=cond,
                color=COND_COLORS.get(cond, '#888'), linewidth=1.2)
    ax.set_xlabel('SVI Step', fontsize=10)
    ax.set_ylabel('ELBO Loss', fontsize=10)
    ax.set_title('Training Convergence per Anchor Condition', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "convergence.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: convergence.png")


# ── save parameter CSVs ───────────────────────────────────────────────────────

def save_params_csv(results):
    """Save θ, γ, τ for every condition to a long-format CSV."""
    rows = []
    ref  = results[REFERENCE_COND]

    for cond, res in results.items():
        l_map = res['l_map']
        p_map = res['p_map']
        s_map = res['s_map']

        # θ rows
        for student, si in s_map.items():
            rows.append({'condition': cond, 'param': 'theta',
                         'student': student, 'language': None, 'prompt': None,
                         'value': res['theta'][si]})

        # γ rows
        for lang, li in l_map.items():
            rows.append({'condition': cond, 'param': 'gamma',
                         'student': None, 'language': lang, 'prompt': None,
                         'value': res['gamma'][li]})

        # τ rows (non-anchor, non-English)
        for prompt, pi in p_map.items():
            for lang, li in l_map.items():
                if lang == 'en':
                    continue
                if res['tau_mask'][pi, li] == 0:
                    continue
                rows.append({'condition': cond, 'param': 'tau',
                             'student': None, 'language': lang, 'prompt': prompt,
                             'value': res['tau_mean'][pi, li]})

    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "params_all_conditions.csv"), index=False)
    print("  Saved: params_all_conditions.csv")


def save_anchor_conditions(anchor_sets):
    """Save anchor IDs per condition."""
    rows = []
    for cond, ids in anchor_sets.items():
        for pid in sorted(ids):
            rows.append({'condition': cond, 'prompt_id': pid})
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "anchor_conditions.csv"), index=False)
    print("  Saved: anchor_conditions.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if _HAS_FIG_STYLE:
        apply_style()

    print("=" * 65)
    print("ANCHOR SENSITIVITY ABLATION")
    print("=" * 65)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['score'] = (df['judge_score'] >= 4).astype(np.float32)
    df['id']    = df['id'].apply(clean_id)
    sc = 'test_taker' if 'test_taker' in df.columns else 'model'
    print(f"  {len(df):,} rows | {df[sc].nunique()} models | "
          f"{df['id'].nunique()} prompts | {df['language'].nunique()} languages")

    # Build anchor sets
    print("\nBuilding anchor conditions...")
    rng         = np.random.default_rng(SEED)
    anchor_sets = build_anchor_sets(df, rng)
    save_anchor_conditions(anchor_sets)

    # Fit each condition
    results = {}
    for cond_name, anchor_ids in anchor_sets.items():
        ref_tag = " ← REFERENCE" if cond_name == REFERENCE_COND else ""
        print(f"\n{'─' * 55}")
        print(f"Fitting: {cond_name}{ref_tag}  (n_anchors = {len(anchor_ids)})")
        print(f"{'─' * 55}")
        results[cond_name] = fit_condition(df, anchor_ids, cond_name)

    # Stability metrics
    print("\nComputing stability metrics...")
    stability = compute_stability(results)
    stability.to_csv(os.path.join(RESULTS_DIR, "stability_summary.csv"), index=False)

    print(f"\n{'=' * 65}")
    print("STABILITY SUMMARY  (vs. reference: strict)")
    print(f"{'=' * 65}")
    print(stability.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save params
    print("\nSaving parameter CSVs...")
    save_params_csv(results)

    # Plots
    print("\nGenerating figures...")
    plot_theta_stability(results)
    plot_gamma_stability(results)
    plot_tau_stability(results)
    plot_convergence(results)

    # Summary for paper
    print(f"\n{'=' * 65}")
    print("KEY FINDINGS")
    print(f"{'=' * 65}")
    ref_row = stability[stability['condition'] == REFERENCE_COND].iloc[0]
    for _, row in stability.iterrows():
        if row['condition'] == REFERENCE_COND:
            continue
        print(f"\n  vs. {row['condition']:20s}:")
        print(f"    θ  Spearman ρ = {row['spearman_theta']:.3f}  "
              f"RMSE = {row['rmse_theta']:.3f}")
        print(f"    γ  Pearson  r = {row['pearson_gamma']:.3f}  "
              f"max|Δγ| = {row['max_delta_gamma']:.3f}")
        print(f"    τ  RMSE = {row['rmse_tau']:.3f}")

    print(f"\nAll outputs in: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
