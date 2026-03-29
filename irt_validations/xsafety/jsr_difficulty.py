# -*- coding: utf-8 -*-
"""
Post-hoc Analysis: JSR vs Theta — XSafety.
Adapted from irt_validations/jsr_difficulty.py:
  - No B experiment / B5 validation data (XSafety single pass only)
  - Uses XSafety_Dataset.csv
  - ANCHOR_FILE from local xsafety results_dif_stratified/

Expected input files:
  Experiment A: results_experiment_A/
    - A4_person_fit_1pl.csv   (student, theta columns)
    - A4_person_fit_2pl.csv

  Raw data:
    - XSafety_Dataset.csv (from HuggingFace)

NOTE ON GRM:
  Set FIT_GRM = True at the top to re-fit GRM, or leave False to
  use only 1PL + 2PL from saved files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, linregress
import os
import re
import warnings
import torch

warnings.filterwarnings('ignore')

from huggingface_hub import snapshot_download

# ── fig_style integration ──
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
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
# ──────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURE THESE PATHS TO MATCH YOUR SETUP
# ══════════════════════════════════════════════════════════════════════════

DATA_DIR    = snapshot_download(
    repo_id="safety-irt/safety-data", repo_type="dataset", token=False
)
INPUT_FILE  = os.path.join(DATA_DIR, "xsafety", "xsafety_pass_graded.csv")
ANCHOR_FILE = os.path.join(DATA_DIR, "xsafety", "xsafety_anchors.csv")

EXP_A_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "results_experiment_A")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_jsr_theta_posthoc")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set True to re-fit GRM (~10 min on GPU). False = only 1PL + 2PL.
FIT_GRM = False

SEED = 42
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


def get_model_family(name):
    name = str(name).lower()
    if any(x in name for x in ['gpt', 'o3-mini', 'o4-mini', 'gpt-5']):
        return 'GPT'
    elif 'claude'   in name: return 'Claude'
    elif 'gemini'   in name: return 'Gemini'
    elif 'grok'     in name: return 'Grok'
    elif 'deepseek' in name: return 'DeepSeek'
    return 'Other'


FAM_COLORS = FS_FAM_COLORS if _HAS_FIG_STYLE else {
    'Claude':   '#7d3c98', 'GPT':      '#2471a3',
    'Gemini':   '#c0392b', 'Grok':     '#e67e22',
    'DeepSeek': '#27ae60', 'Other':    '#7f8c8d',
}

_m1 = C_BLUE   if _HAS_FIG_STYLE else '#2471a3'
_m2 = C_PURPLE if _HAS_FIG_STYLE else '#7d3c98'
_m3 = C_RED    if _HAS_FIG_STYLE else '#c0392b'
MODEL_STYLES = {
    '1PL': dict(linestyle='-',  color=_m1),
    '2PL': dict(linestyle='--', color=_m2),
    'GRM': dict(linestyle=':',  color=_m3),
}

_save = fs_savefig if _HAS_FIG_STYLE else \
    lambda f, p: (f.savefig(p, dpi=300, bbox_inches='tight'), plt.close(f))


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load raw data (for JSR computation)
# ══════════════════════════════════════════════════════════════════════════

def load_raw_data():
    print("Loading raw response data ...")
    df = pd.read_csv(INPUT_FILE, engine='python', on_bad_lines='skip')
    df['judge_score'] = pd.to_numeric(df['judge_score'], errors='coerce')
    df = df[df['judge_score'] > 0].dropna(subset=['judge_score']).copy()
    df['score']     = (df['judge_score'] >= 4).astype(float)
    df['grm_score'] = (df['judge_score'].clip(1, 5) - 1).astype(int)
    df['id']        = df['id'].apply(clean_id)
    sc = 'test_taker' if 'test_taker' in df.columns else 'model'
    df['model_family'] = df[sc].apply(get_model_family)
    df['student_col']  = sc
    print(f"  Rows: {len(df):,}  |  Languages: {sorted(df['language'].unique())}")
    return df


def build_jsr_tables(df):
    """
    Empirical JSR computed once, used by all IRT models.
    Returns
      jsr_overall — one row per test-taker (JSR pooled across all languages)
      jsr_lang    — one row per (test-taker x language)
    """
    sc = df['student_col'].iloc[0]

    jsr_overall = (
        df.groupby(sc)['score']
          .agg(n_obs='count', mean_safe='mean')
          .reset_index()
          .rename(columns={sc: 'test_taker'})
    )
    jsr_overall['JSR']          = 1.0 - jsr_overall['mean_safe']
    jsr_overall['model_family'] = jsr_overall['test_taker'].apply(get_model_family)

    jsr_lang = (
        df.groupby([sc, 'language'])['score']
          .agg(n_obs='count', mean_safe='mean')
          .reset_index()
          .rename(columns={sc: 'test_taker'})
    )
    jsr_lang['JSR_lang']     = 1.0 - jsr_lang['mean_safe']
    jsr_lang['model_family'] = jsr_lang['test_taker'].apply(get_model_family)

    print(f"  JSR table: {len(jsr_overall)} models  |  "
          f"lang table: {len(jsr_lang)} (model x language) pairs")
    return jsr_overall, jsr_lang


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Load IRT parameters from saved CSVs
# ══════════════════════════════════════════════════════════════════════════

def load_person_fit_csv(path, model_name):
    """
    Experiment A saves person_fit CSVs with columns:
      student, student_idx, n_obs, theta, infit, outfit
    We only need: student (renamed to test_taker), theta
    """
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found -- skipping {model_name}")
        return None
    df = pd.read_csv(path)

    if 'student' not in df.columns or 'theta' not in df.columns:
        print(f"  WARNING: {path} missing 'student' or 'theta' column")
        return None

    out = df[['student', 'theta']].copy()
    out = out.rename(columns={'student': 'test_taker'})
    print(f"  [{model_name}] Loaded {len(out)} rows from {os.path.basename(path)}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# OPTIONAL — Re-fit GRM if FIT_GRM = True
# ══════════════════════════════════════════════════════════════════════════

def fit_grm_if_needed(df, anchor_ids):
    """Re-fit GRM and return theta DataFrame."""
    import pyro
    import pyro.distributions as dist_pyro
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.optim import ClippedAdam
    from tqdm import tqdm
    from pyro.infer.autoguide import AutoNormal, init_to_feasible

    print("\nFitting GRM (this may take several minutes) ...")
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    pyro.clear_param_store()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc        = 'test_taker' if 'test_taker' in df.columns else 'model'
    students  = sorted(df[sc].unique())
    prompts   = sorted(df['id'].unique())
    languages = sorted(df['language'].unique())

    student_map = {s: i for i, s in enumerate(students)}
    prompt_map  = {p: i for i, p in enumerate(prompts)}
    lang_map    = {l: i for i, l in enumerate(languages)}
    ns, np_, nl = len(students), len(prompts), len(languages)

    student_idx = torch.tensor(df[sc].map(student_map).values,
                               dtype=torch.long).to(device)
    prompt_idx  = torch.tensor(df['id'].map(prompt_map).values,
                               dtype=torch.long).to(device)
    lang_idx    = torch.tensor(df['language'].map(lang_map).values,
                               dtype=torch.long).to(device)
    score_obs   = torch.tensor(df['grm_score'].values,
                               dtype=torch.long).to(device)

    tau_mask   = torch.ones((np_, nl), device=device)
    gamma_mask = torch.ones(nl, device=device)
    if 'en' in lang_map:
        ei = lang_map['en']
        tau_mask[:, ei] = 0.0
        gamma_mask[ei]  = 0.0

    for pid in prompts:
        if pid in anchor_ids and pid in prompt_map:
            tau_mask[prompt_map[pid], :] = 0.0

    n_thresh = 4  # K=5 categories -> 4 thresholds

    def grm(s_idx, p_idx, l_idx, obs=None):
        theta     = pyro.sample("theta",
            dist_pyro.Normal(
                torch.zeros(ns, device=device), 1.).to_event(1))
        beta_base = pyro.sample("beta_base",
            dist_pyro.Normal(
                torch.zeros(np_, device=device), 1.5).to_event(1))
        beta_inc  = pyro.sample("beta_increments",
            dist_pyro.HalfNormal(
                torch.ones(np_, n_thresh - 1, device=device) * 0.8
            ).to_event(2))
        thresh = pyro.deterministic("thresholds",
            torch.cat([
                beta_base.unsqueeze(-1),
                beta_base.unsqueeze(-1) + torch.cumsum(beta_inc, -1)
            ], -1))
        log_alpha = pyro.sample("log_alpha",
            dist_pyro.Normal(
                torch.zeros(np_, device=device), 0.5).to_event(1))
        alpha     = pyro.deterministic("alpha", torch.exp(log_alpha))
        gamma_raw = pyro.sample("gamma_raw",
            dist_pyro.Normal(
                torch.zeros(nl, device=device), 1.).to_event(1))
        gamma     = pyro.deterministic("gamma", gamma_raw * gamma_mask)
        tau_scale = pyro.sample("tau_scale",
            dist_pyro.HalfCauchy(
                torch.ones(1, device=device)).to_event(1))
        tau_raw   = pyro.sample("tau_raw",
            dist_pyro.StudentT(
                1., torch.zeros(np_, nl, device=device),
                tau_scale).to_event(2))
        tau       = pyro.deterministic("tau", tau_raw * tau_mask)
        delta_raw = pyro.sample("delta_raw",
            dist_pyro.Normal(
                torch.zeros(ns, nl, device=device), 0.5).to_event(2))
        dmask     = gamma_mask.unsqueeze(0).expand(ns, -1)
        delta     = pyro.deterministic("delta", delta_raw * dmask)

        with pyro.plate("data", len(s_idx)):
            ab    = theta[s_idx] + delta[s_idx, l_idx]
            ls    = gamma[l_idx] + tau[p_idx, l_idx]
            it    = thresh[p_idx]
            ai    = alpha[p_idx].unsqueeze(-1)
            cp    = torch.sigmoid(
                ai * (ab.unsqueeze(-1) - (it + ls.unsqueeze(-1))))
            ones  = torch.ones(cp.shape[0],  1, device=device)
            zeros = torch.zeros(cp.shape[0], 1, device=device)
            ce    = torch.cat([ones, cp, zeros], -1)
            pp    = (ce[:, :-1] - ce[:, 1:]).clamp(min=1e-8)
            pp    = pp / pp.sum(-1, keepdim=True)
            pyro.sample("obs", dist_pyro.Categorical(probs=pp), obs=obs)

    hide  = ["obs", "tau", "gamma", "delta", "alpha", "thresholds"]
    guide = AutoNormal(pyro.poutine.block(grm, hide=hide),
                       init_loc_fn=init_to_feasible())
    opt   = ClippedAdam({"lr": 0.005, "clip_norm": 10.0})
    svi   = SVI(grm, guide, opt, loss=Trace_ELBO())

    MAX_STEPS = 6000
    WIN, THR, MIN_S = 200, 1e-4, 1500
    losses = []
    for step in tqdm(range(MAX_STEPS), desc="GRM"):
        loss = svi.step(student_idx, prompt_idx, lang_idx, score_obs)
        losses.append(loss)
        if len(losses) >= 2 * WIN and len(losses) >= MIN_S:
            prev = np.mean(losses[-2 * WIN:-WIN])
            rec  = np.mean(losses[-WIN:])
            if prev != 0 and (prev - rec) / abs(prev) < THR:
                print(f"  GRM converged at step {step + 1}")
                break

    pred = Predictive(grm, guide=guide, num_samples=300,
                      return_sites=["theta", "delta"])
    samp = pred(student_idx, prompt_idx, lang_idx, None)

    theta_mean = samp['theta'].detach().cpu().numpy().mean(0).reshape(ns)
    theta_std  = samp['theta'].detach().cpu().numpy().std(0).reshape(ns)

    theta_df = pd.DataFrame({
        'test_taker': students,
        'theta':      theta_mean,
        'theta_std':  theta_std,
    })

    print(f"  GRM: {len(theta_df)} students fitted")
    return theta_df


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Assemble theta tables
# ══════════════════════════════════════════════════════════════════════════

def assemble_theta_tables(jsr_overall, jsr_lang, df_raw, anchor_ids):
    """
    Load 1PL and 2PL theta from Experiment A person-fit CSVs.
    No B5 delta data (XSafety has no B experiment).
    delta_lookup is empty; language analysis will be skipped.
    """
    theta_dfs = []

    # ── 1PL ──────────────────────────────────────────────────────────────────
    pf1 = load_person_fit_csv(
        os.path.join(EXP_A_DIR, "A4_person_fit_1pl.csv"), "1PL")
    if pf1 is not None:
        merged = jsr_overall.merge(
            pf1[['test_taker', 'theta']],
            on='test_taker', how='inner'
        )
        merged['irt_model'] = '1PL'
        theta_dfs.append(
            merged[['test_taker', 'JSR', 'theta',
                     'model_family', 'irt_model', 'n_obs']].copy()
        )
        print(f"  [1PL] Merged {len(merged)} rows into theta table")

    # ── 2PL ──────────────────────────────────────────────────────────────────
    pf2 = load_person_fit_csv(
        os.path.join(EXP_A_DIR, "A4_person_fit_2pl.csv"), "2PL")
    if pf2 is not None:
        merged2 = jsr_overall.merge(
            pf2[['test_taker', 'theta']],
            on='test_taker', how='inner'
        )
        merged2['irt_model'] = '2PL'
        theta_dfs.append(
            merged2[['test_taker', 'JSR', 'theta',
                      'model_family', 'irt_model', 'n_obs']].copy()
        )
        print(f"  [2PL] Merged {len(merged2)} rows into theta table")

    # ── GRM ──────────────────────────────────────────────────────────────────
    if FIT_GRM:
        grm_theta = fit_grm_if_needed(df_raw, anchor_ids)
        merged_grm = jsr_overall.merge(
            grm_theta[['test_taker', 'theta']],
            on='test_taker', how='inner'
        )
        merged_grm['irt_model'] = 'GRM'
        theta_dfs.append(
            merged_grm[['test_taker', 'JSR', 'theta',
                         'model_family', 'irt_model', 'n_obs']].copy()
        )
        print(f"  [GRM] Merged {len(merged_grm)} rows into theta table")
    else:
        print("  GRM skipped (FIT_GRM=False). "
              "Set FIT_GRM=True at the top to include it.")

    if not theta_dfs:
        raise RuntimeError(
            "No theta data loaded. Check that EXP_A_DIR points to your "
            "results_experiment_A folder and that A4_person_fit_*.csv exist."
        )

    combined_theta = pd.concat(theta_dfs, ignore_index=True)
    print(f"\n  Combined theta table: {len(combined_theta)} rows "
          f"across {combined_theta['irt_model'].nunique()} model(s): "
          f"{combined_theta['irt_model'].unique().tolist()}")

    # No B experiment for XSafety → delta_lookup is empty
    delta_lookup = {}
    print("  NOTE: No B experiment for XSafety. "
          "Per-language (θ−δ) analysis will be skipped.")

    return combined_theta, delta_lookup


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Build language analysis DataFrame (empty for XSafety)
# ══════════════════════════════════════════════════════════════════════════

def build_lang_analysis(jsr_lang, delta_lookup):
    """
    delta_lookup is empty for XSafety (no B experiment).
    Returns empty DataFrame — downstream plots gracefully skip.
    """
    if not delta_lookup:
        print("  No delta data available — language analysis skipped.")
        return pd.DataFrame()

    lang_dfs = []
    for mname, d_df in delta_lookup.items():
        col = 'theta_eff' if 'theta_eff' in d_df.columns else 'theta_minus_delta'
        merged = jsr_lang.merge(
            d_df[['test_taker', 'language', col]],
            on=['test_taker', 'language'], how='inner'
        )
        merged = merged.rename(columns={col: 'theta_minus_delta'})
        merged['irt_model'] = mname
        lang_dfs.append(
            merged[['test_taker', 'language', 'JSR_lang',
                     'theta_minus_delta', 'n_obs',
                     'model_family', 'irt_model']].copy()
        )

    if not lang_dfs:
        return pd.DataFrame()

    combined = pd.concat(lang_dfs, ignore_index=True)
    print(f"  Language analysis table: {len(combined)} rows "
          f"across {combined['irt_model'].unique().tolist()}")
    return combined


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════

def scatter_with_ols(ax, x, y, families, style, label,
                     alpha_pts=0.75, show_families=True):
    """Scatter coloured by model family + OLS regression line."""
    for fam, col in FAM_COLORS.items():
        mask = np.array(families) == fam
        if mask.sum():
            ax.scatter(x[mask], y[mask],
                       color=col, s=18, alpha=alpha_pts,
                       edgecolors='black', linewidths=0.25,
                       label=fam if show_families else None)

    r_p = r_s = sl = ic = np.nan
    if len(x) >= 3:
        sl, ic, _, _, _ = linregress(x, y)
        xr = np.linspace(x.min() - 0.15, x.max() + 0.15, 300)
        ax.plot(xr, sl * xr + ic, linewidth=1.0,
                label=f'OLS [{label}]', **style)
        r_p, _ = pearsonr(x, y)
        r_s, _ = spearmanr(x, y)
    return r_p, r_s, sl, ic


# ══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Overall JSR vs θ
# ══════════════════════════════════════════════════════════════════════════

def plot_jsr_vs_theta(combined_theta):
    model_names = [m for m in ['1PL', '2PL', 'GRM']
                   if m in combined_theta['irt_model'].values]
    n = len(model_names)
    if n == 0:
        print("  No theta data to plot.")
        return pd.DataFrame()

    n_total = n + 1
    if _HAS_FIG_STYLE:
        fig, axes = make_fig(n_panels=n_total, height_override=3.5)
    else:
        fig, axes = plt.subplots(1, n_total, figsize=(5.5, 3.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    corr_rows = []
    for idx, mname in enumerate(model_names):
        sub = combined_theta[
            combined_theta['irt_model'] == mname
        ].dropna(subset=['theta', 'JSR'])

        ax    = axes[idx]
        style = MODEL_STYLES.get(mname, {})

        r_p, r_s, sl, ic = scatter_with_ols(
            ax,
            sub['theta'].values,
            sub['JSR'].values,
            sub['model_family'].values,
            style=style,
            label=mname,
            show_families=(idx == 0),
        )

        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_xlabel(r'$\theta$ (ability)')
        ax.set_ylabel('JSR')
        ax.set_title(f'{mname}: $r$={r_p:.3f}, n={len(sub)}')

        if idx == 0:
            handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=c, markersize=3.5, label=f)
                for f, c in FAM_COLORS.items()
            ]
            ax.legend(handles=handles, fontsize=5, ncol=2,
                      loc='upper left')

        corr_rows.append(dict(
            irt_model=mname, pearson_r=r_p,
            spearman_r=r_s, n=len(sub)
        ))

        if not np.isnan(sl):
            all_x = combined_theta['theta'].dropna().values
            xr = np.linspace(
                np.nanpercentile(all_x, 1) - 0.1,
                np.nanpercentile(all_x, 99) + 0.1, 200
            )
            axes[n].plot(xr, sl * xr + ic, linewidth=1.0,
                         label=f'{mname} $r$={r_p:.3f}', **style)

    ax_right = axes[n]
    ax_right.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax_right.set_xlabel(r'$\theta$')
    ax_right.set_ylabel('JSR')
    ax_right.set_title('OLS Comparison')
    ax_right.legend(fontsize=5)

    out_path = os.path.join(RESULTS_DIR, "1_jsr_vs_theta.png")
    _save(fig, out_path)
    print("  Saved: 1_jsr_vs_theta")

    return pd.DataFrame(corr_rows)


# ══════════════════════════════════════════════════════════════════════════
# PLOT 2 — theta rank agreement across IRT models
# ══════════════════════════════════════════════════════════════════════════

def plot_rank_agreement(combined_theta):
    model_names = [m for m in ['1PL', '2PL', 'GRM']
                   if m in combined_theta['irt_model'].values]
    pairs = [
        (model_names[i], model_names[j])
        for i in range(len(model_names))
        for j in range(i + 1, len(model_names))
    ]
    if not pairs:
        print("  Only one IRT model present -- rank agreement plot skipped.")
        return

    n_pairs = len(pairs)
    if _HAS_FIG_STYLE:
        fig, axes_raw = make_fig(n_panels=n_pairs, height_override=3.5)
    else:
        fig, axes_raw = plt.subplots(1, n_pairs, figsize=(5.5, 3.5))
    if n_pairs == 1 and not isinstance(axes_raw, np.ndarray):
        axes_all = [axes_raw]
    else:
        axes_all = list(axes_raw) if isinstance(axes_raw, np.ndarray) else [axes_raw]

    for idx, (m1, m2) in enumerate(pairs):
        ax = axes_all[idx]
        d1 = (combined_theta[combined_theta['irt_model'] == m1]
              .set_index('test_taker')['theta'])
        d2 = (combined_theta[combined_theta['irt_model'] == m2]
              .set_index('test_taker')['theta'])
        common = d1.index.intersection(d2.index)
        if len(common) < 3:
            ax.set_title(f'{m1} vs {m2}: too few shared models')
            continue

        x    = d1[common].values
        y    = d2[common].values
        fams = (
            combined_theta[
                (combined_theta['irt_model'] == m1) &
                (combined_theta['test_taker'].isin(common))
            ]
            .set_index('test_taker')['model_family'][common]
            .values
        )

        for fam, col in FAM_COLORS.items():
            mask = np.array(fams) == fam
            if mask.sum():
                ax.scatter(x[mask], y[mask],
                           color=col, s=18, alpha=0.8,
                           edgecolors='black', linewidths=0.25,
                           label=fam)

        lims = [min(x.min(), y.min()) - 0.2,
                max(x.max(), y.max()) + 0.2]
        ax.plot(lims, lims, 'k--', alpha=0.5, lw=0.5, label='Identity')

        r_p, _ = pearsonr(x, y)
        r_s, _ = spearmanr(x, y)
        ax.set_xlabel(rf'$\theta$ [{m1}]')
        ax.set_ylabel(rf'$\theta$ [{m2}]')
        ax.set_title(f'{m1} vs {m2}: $r$={r_p:.3f}')
        ax.legend(fontsize=4, ncol=2)

    _save(fig, os.path.join(RESULTS_DIR, "3_theta_rank_agreement.png"))
    print("  Saved: 3_theta_rank_agreement")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    if _HAS_FIG_STYLE:
        apply_style()
    print("=" * 68)
    print("POST-HOC: JSR vs theta — XSafety")
    print("Loading from saved Experiment A results -- no re-fitting")
    print("=" * 68)

    # ── 1. Raw data for JSR ───────────────────────────────────────────────────
    df_raw = load_raw_data()
    jsr_overall, jsr_lang = build_jsr_tables(df_raw)

    anchor_ids = set()
    if os.path.exists(ANCHOR_FILE):
        adf = pd.read_csv(ANCHOR_FILE)
        adf['id'] = adf['id'].apply(clean_id)
        anchor_ids = set(adf['id'].unique())

    # ── 2. Load IRT parameters ────────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("Loading IRT parameters from saved CSVs ...")
    combined_theta, delta_lookup = assemble_theta_tables(
        jsr_overall, jsr_lang, df_raw, anchor_ids
    )

    # ── 3. Analysis 1: Overall JSR vs theta ───────────────────────────────────
    print("\n" + "-" * 50)
    print("Analysis 1: Overall JSR vs theta")

    combined_theta.to_csv(
        os.path.join(RESULTS_DIR, "1_jsr_vs_theta_all_models.csv"),
        index=False
    )
    print("  Saved: 1_jsr_vs_theta_all_models.csv")

    for mname, sub in combined_theta.groupby('irt_model'):
        sub = sub.dropna(subset=['theta', 'JSR'])
        if len(sub) >= 3:
            r_p, _ = pearsonr(sub['theta'], sub['JSR'])
            r_s, _ = spearmanr(sub['theta'], sub['JSR'])
            print(f"  [{mname}]  r={r_p:.4f}  rho={r_s:.4f}  n={len(sub)}")

    corr1 = plot_jsr_vs_theta(combined_theta)
    corr1.to_csv(
        os.path.join(RESULTS_DIR, "1_jsr_theta_correlations.csv"),
        index=False
    )
    print("  Saved: 1_jsr_theta_correlations.csv")

    # ── 4. Analysis 2: Language JSR vs (theta - delta) ───────────────────────
    # NOTE: Skipped for XSafety — no B experiment (no delta by language).
    print("\n" + "-" * 50)
    print("Analysis 2: Per-language JSR vs (theta - delta)")
    lang_df = build_lang_analysis(jsr_lang, delta_lookup)

    if len(lang_df) == 0:
        lang_df.to_csv(
            os.path.join(RESULTS_DIR,
                         "2_jsr_vs_theta_minus_delta_all_models.csv"),
            index=False
        )

    # ── 5. Rank agreement ─────────────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("Analysis 3: theta rank agreement across IRT models")
    plot_rank_agreement(combined_theta)

    print(f"\nAll outputs in: {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
