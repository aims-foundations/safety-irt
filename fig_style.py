# -*- coding: utf-8 -*-
"""
fig_style.py — Shared figure style for all plots in the paper.
================================================================
Usage (at top of any plotting script):

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from fig_style import *

Then:
    apply_style()                          # call once at script start
    fig, ax = make_fig(n_panels=3)         # 3 panels on 1 row, correct width
    ax[0].set_xlabel(LABELS['tau'])        # Greek-letter label

"""

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ══════════════════════════════════════════════════════════════════════
# 1. VENUE SETTINGS
# ══════════════════════════════════════════════════════════════════════

# COLM 2026: single-column, \textwidth ≈ 5.5in
FULL_WIDTH  = 5.5     # inches  (single-column text width)
SINGLE_COL  = 5.5     # same — single-column venue
DPI         = 300
ASPECT      = 0.5     # default height/width per panel (flatter for 1-row layouts)


def apply_style():
    """Call once at script start. Sets rcParams for publication figures."""

    # ── Try tueplots first (preferred) ───────────────────────────
    try:
        from tueplots import bundles, figsizes
        # tueplots may not have colm2026; icml2024 is closest base
        plt.rcParams.update(bundles.icml2024())
        plt.rcParams.update({
            "figure.figsize": (FULL_WIDTH, FULL_WIDTH * ASPECT),
        })
    except ImportError:
        print("[fig_style] tueplots not installed — using manual rcParams. "
              "Install with: pip install tueplots")
        plt.rcParams.update({
            "font.family":     "serif",
            "font.serif":      ["Computer Modern Roman", "Times New Roman",
                                "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size":        8,
            "axes.labelsize":   8,
            "axes.titlesize":   9,
            "xtick.labelsize":  7,
            "ytick.labelsize":  7,
            "legend.fontsize":  7,
            "figure.figsize":   (FULL_WIDTH, FULL_WIDTH * ASPECT),
        })

    # ── Overrides that apply regardless of tueplots ──────────────
    plt.rcParams.update({
        "figure.dpi":            DPI,
        "savefig.dpi":           DPI,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.02,
        "text.usetex":           False,    # mathtext only (no LaTeX install)
        "mathtext.fontset":      "cm",     # Computer Modern — matches LaTeX
        "figure.constrained_layout.use": True,   # prevent label overlap
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.grid":             False,
        "axes.linewidth":        0.5,
        "xtick.major.width":     0.5,
        "ytick.major.width":     0.5,
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        "legend.frameon":        False,
        "legend.handlelength":   1.2,
        "lines.linewidth":      1.0,
        "lines.markersize":     4,
        "patch.linewidth":      0.5,
    })


# ══════════════════════════════════════════════════════════════════════
# 2. FIGURE FACTORY
# ══════════════════════════════════════════════════════════════════════

def make_fig(n_panels=1, aspect=None, width=FULL_WIDTH,
             height_override=None, sharex=False, sharey=False):
    """
    Create a (1, n_panels) figure at the correct venue width.
    All panels on ONE ROW — use make_fig_grid() only if you must break rows.

    Returns (fig, axes):
        n_panels=1  → axes is a single Axes object
        n_panels>1  → axes is a list of Axes

    Parameters
    ----------
    n_panels : int
        Number of side-by-side panels.
    aspect : float or None
        Height / width ratio *per panel*.
        Default: auto-scaled (0.5 for 1-3 panels, taller for 4+).
    width : float
        Total figure width in inches. Default FULL_WIDTH.
    height_override : float or None
        If set, use this exact height (inches).
    """
    if aspect is None:
        # Panels get narrower → need proportionally taller to stay readable
        aspect = ASPECT if n_panels <= 3 else min(0.7, ASPECT + 0.05 * (n_panels - 3))

    panel_w = width / n_panels
    h = height_override if height_override else panel_w * aspect

    fig, axes = plt.subplots(1, n_panels, figsize=(width, h),
                             squeeze=False, sharex=sharex, sharey=sharey)
    axes = axes[0]  # flatten from (1, N) to (N,)

    if n_panels == 1:
        return fig, axes[0]
    return fig, axes


def make_fig_grid(nrows, ncols, aspect=ASPECT, width=FULL_WIDTH,
                  height_override=None, **kwargs):
    """
    For the rare case you NEED a grid (e.g., 10-language facet).
    Returns (fig, axes_2d).
    """
    panel_w = width / ncols
    panel_h = height_override if height_override else panel_w * aspect
    h = panel_h * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, h),
                             squeeze=False, **kwargs)
    return fig, axes


# ══════════════════════════════════════════════════════════════════════
# 3. COLOR PALETTES
# ══════════════════════════════════════════════════════════════════════

# ── Core semantic colors (3-color, colorblind-safe) ──────────────
#    Main paper: use only these three for most figures.
C_RED    = '#c0392b'   # "bad"  — harder, unsafe, negative δ, positive τ
C_BLUE   = '#2471a3'   # "good" — easier, safe, positive δ, negative τ
C_PURPLE = '#7d3c98'   # neutral / reference / third condition

COLORS_3 = [C_RED, C_BLUE, C_PURPLE]

# Semantic aliases — red = bad, blue = good throughout.
C_POS = C_RED          # positive τ  → harder in target lang → bad
C_NEG = C_BLUE         # negative τ  → easier in target lang → good
C_REF = C_PURPLE       # reference / baseline

# ── Horseshoe sensitivity priors ─────────────────────────────────
PRIOR_COLORS = {
    'horseshoe': C_RED,
    'moderate':  C_PURPLE,
    'normal':    C_BLUE,
}

# ── Model family colors (5-color, colorblind-safe) ───────────────
#    Used for: appendix delta heatmaps, bump charts, family-level figures.
#    Main paper figures should prefer COLORS_3; use FAM_COLORS only when
#    distinguishing families is the point of the figure.
#    Okabe-Ito inspired: distinct under all CVD types.
FAM_COLORS = {
    'Claude':   '#7d3c98',   # purple
    'GPT':      '#2471a3',   # blue
    'Gemini':   '#c0392b',   # red
    'Grok':     '#e67e22',   # orange
    'DeepSeek': '#27ae60',   # green
    'Other':    '#7f8c8d',   # gray
}

FAM_ORDER = ['Claude', 'GPT', 'Gemini', 'Grok', 'DeepSeek']

# ── Diverging colormap for heatmaps (red ↔ blue) ────────────────
CMAP_DIV = 'RdBu_r'     # red = negative/harder, blue = positive/easier
CMAP_SEQ = 'Blues'       # sequential for counts / magnitudes


# ══════════════════════════════════════════════════════════════════════
# 4. GREEK-LETTER LABELS  &  LANGUAGE / FAMILY ORDERINGS
# ══════════════════════════════════════════════════════════════════════

# Greek labels — use: ax.set_xlabel(LABELS['tau'])
LABELS = {
    'theta':       r'$\theta_j$',
    'theta_short': r'$\theta$',
    'delta':       r'$\delta_{jL}$',
    'delta_short': r'$\delta$',
    'beta':        r'$\beta_i$',
    'beta_short':  r'$\beta$',
    'gamma':       r'$\gamma_L$',
    'gamma_short': r'$\gamma$',
    'tau':         r'$\tau_{iL}$',
    'tau_short':   r'$\tau$',
    'alpha':       r'$\alpha_i$',
    'alpha_short': r'$\alpha$',
    'abs_tau':     r'$|\tau_{iL}|$',
    'mean_tau':    r'$\overline{\tau}_{\cdot L}$',
    'sigma':       r'$\sigma$',
    'rho':         r'$\rho$',
}

# Axis titles — ready to paste
TITLES = {
    'theta_vs_jsr':      r'$\theta$ Rank vs JSR Rank',
    'gamma_by_lang':     r'$\gamma_L$ by Language',
    'tau_dist':          r'$\tau_{iL}$ Distribution',
    'delta_heatmap':     r'$\delta_{jL}$ by Family $\times$ Language',
    'gamma_tau_scatter': r'$\gamma_L$ vs $\overline{\tau}_{\cdot L}$',
}

# ── Consistent axis orderings ────────────────────────────────────
LANG_ORDER = ['en', 'ar', 'bn', 'it', 'jv', 'ko', 'sw', 'th', 'vi', 'zh']

LANG_LABELS = {
    'en': 'English', 'ar': 'Arabic',    'bn': 'Bengali',
    'it': 'Italian', 'jv': 'Javanese',  'ko': 'Korean',
    'sw': 'Swahili', 'th': 'Thai',      'vi': 'Vietnamese',
    'zh': 'Chinese',
}

NON_EN_LANGS = [l for l in LANG_ORDER if l != 'en']


# ══════════════════════════════════════════════════════════════════════
# 5. HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════

def savefig(fig, path, formats=None, **kwargs):
    """
    Save figure. Defaults to both PDF (vector, for paper) and PNG (preview).

    Parameters
    ----------
    path : str
        Output path. Extension is replaced per format.
    formats : list of str or None
        E.g. ['pdf', 'png']. Default: both.
    """
    if formats is None:
        formats = ['pdf', 'png']
    base, _ = os.path.splitext(path)
    for fmt in formats:
        out = f"{base}.{fmt}"
        fig.savefig(out, dpi=DPI, bbox_inches='tight',
                    pad_inches=0.02, format=fmt, **kwargs)
        print(f"  Saved: {out}")
    plt.close(fig)


def add_identity_line(ax, **kwargs):
    """Add y=x reference line to a scatter plot."""
    lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]),
            min(ax.get_xlim()[1], ax.get_ylim()[1])]
    kw = dict(color='gray', ls='--', lw=0.7, alpha=0.6, zorder=0)
    kw.update(kwargs)
    ax.plot(lims, lims, **kw)


def add_zero_line(ax, axis='x', **kwargs):
    """Add a zero reference line."""
    kw = dict(color='gray', ls='-', lw=0.5, alpha=0.5, zorder=0)
    kw.update(kwargs)
    if axis in ('x', 'both'):
        ax.axhline(0, **kw)
    if axis in ('y', 'both'):
        ax.axvline(0, **kw)


def annotate_r(ax, x, y, prefix='', loc='top-right'):
    """Add Pearson r annotation to a scatter panel."""
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    txt = f'{prefix}$r = {r:.2f}$'
    if p < 0.001:
        txt += r', $p < .001$'
    elif p < 0.05:
        txt += f', $p = {p:.3f}$'
    else:
        txt += f', $p = {p:.2f}$'

    positions = {
        'top-right':    (0.97, 0.95, 'right', 'top'),
        'top-left':     (0.03, 0.95, 'left',  'top'),
        'bottom-right': (0.97, 0.05, 'right', 'bottom'),
        'bottom-left':  (0.03, 0.05, 'left',  'bottom'),
    }
    x_, y_, ha, va = positions.get(loc, positions['top-right'])
    ax.text(x_, y_, txt, transform=ax.transAxes, ha=ha, va=va,
            fontsize=plt.rcParams['legend.fontsize'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='none', alpha=0.8))


def get_family(name):
    """Map a model/test-taker name to its family string."""
    name_lower = str(name).lower()
    if any(x in name_lower for x in ['gpt', 'o3-mini', 'o4-mini', 'gpt-5']):
        return 'GPT'
    elif 'claude'   in name_lower: return 'Claude'
    elif 'gemini'   in name_lower: return 'Gemini'
    elif 'grok'     in name_lower: return 'Grok'
    elif 'deepseek' in name_lower: return 'DeepSeek'
    return 'Other'


def get_family_color(name):
    """Map a model name to its family color."""
    return FAM_COLORS.get(get_family(name), FAM_COLORS['Other'])


def family_legend(ax=None, **kwargs):
    """Add a compact family-color legend."""
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=FAM_COLORS[f], label=f) for f in FAM_ORDER]
    target = ax if ax else plt.gca()
    kw = dict(fontsize=plt.rcParams['legend.fontsize'],
              handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    kw.update(kwargs)
    return target.legend(handles=handles, **kw)


# ══════════════════════════════════════════════════════════════════════
# 6. QUICK TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    apply_style()

    # Demo: 3-panel figure
    fig, axes = make_fig(n_panels=3)
    for i, ax in enumerate(axes):
        x = np.linspace(-3, 3, 100)
        ax.plot(x, np.sin(x + i), color=COLORS_3[i], label=f'Panel {i+1}')
        ax.set_xlabel(LABELS['theta'])
        ax.set_ylabel(r'$P(\mathrm{safe})$')
        ax.legend()
    fig.suptitle('Demo: 3-panel, full-width, serif, Greek labels')

    savefig(fig, 'fig_style_demo.png')
    print(f"\nStyle config OK.")
    print(f"  Width: {FULL_WIDTH}in  |  DPI: {DPI}  |  Aspect: {ASPECT}")
    print(f"  Core colors: red={C_RED}, blue={C_BLUE}, purple={C_PURPLE}")
    print(f"  Languages: {LANG_ORDER}")