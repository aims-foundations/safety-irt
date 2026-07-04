"""
Phase 5 — plot the headline figures from the Phase 4 grid.

Reads phase_4_results.csv (written by phase_4.run_grid / its __main__) or takes a
results DataFrame directly, and produces:

  fig 1 (headline)  contamination vs DIF proportion, one panel per direction,
                    three lines: Floor / Method / Oracle (+/- SD over reps).
                    The rebuttal reads off panel 2 (unbalanced): does Method
                    track Oracle rather than Floor at the realistic 40-50% end?

  fig 2 (diagnostics)  rank AUC and, if present, DIF detection (hit / false-alarm)
                       vs proportion, per direction.

Colab: run after phase_4. Uses the repo's fig_style if importable, else falls
back to plain matplotlib (same pattern as irt.py).
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── fig_style integration (../../fig_style.py), with graceful fallback ────────
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))
try:
    from fig_style import apply_style, savefig as _fs_savefig, make_fig, C_RED, C_BLUE, C_PURPLE
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False
    C_RED, C_BLUE, C_PURPLE = "#c0392b", "#2471a3", "#7d3c98"

    def make_fig(n_panels=1, **kw):
        fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.0), squeeze=False)
        axes = axes[0]
        return (fig, axes[0]) if n_panels == 1 else (fig, axes)

def _save_multi(fig, out_base, extra=None):
    """Save png+pdf with a tight bbox that includes out-of-axes artists (e.g. a
    figure-level legend), so nothing is clipped. Keeps fig_style's rcParams."""
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight",
                    bbox_extra_artists=tuple(extra or ()))
    plt.close(fig)

_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(_RESULTS_DIR, "phase_4_results.csv")
try:
    from phase_2 import DIRECTIONS            # keep panels in sync with the grid
except ModuleNotFoundError:
    DIRECTIONS = ["balanced", "unbalanced", "realistic"]


def _load(results):
    if results is None:
        results = RESULTS_CSV
    return pd.read_csv(results) if isinstance(results, str) else results.copy()


def _line(ax, sub, mean_col, sd_col, color, label, ls="-", marker="o"):
    x = sub["proportion"].to_numpy() * 100
    y = sub[mean_col].to_numpy()
    ax.plot(x, y, color=color, ls=ls, marker=marker, ms=3.5, lw=1.2, label=label)
    if sd_col and sd_col in sub.columns:
        sd = sub[sd_col].to_numpy()
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.15, lw=0)


def _panels(df):
    """Directions actually present in the results, in canonical order."""
    return [d for d in DIRECTIONS if (df["direction"] == d).any()]


def plot_contamination(results=None, save=True):
    df = _load(results)
    dirs = _panels(df)
    if _HAS_FIG_STYLE:
        apply_style()
    fig, axes = make_fig(len(dirs), sharey=True, width=7.5, height_override=2.4)
    if len(dirs) == 1:
        axes = [axes]

    for ax, direction in zip(axes, dirs):
        sub = df[df["direction"] == direction].sort_values("proportion")
        _line(ax, sub, "contam_floor_mean",  "contam_floor_sd",  C_RED,    "Floor (random)")
        _line(ax, sub, "contam_method_mean", "contam_method_sd", C_BLUE,   "Method (χ²-first)")
        _line(ax, sub, "contam_oracle_mean", "contam_oracle_sd", "black",  "Oracle (true DIF-free)",
              ls="--", marker="s")
        # Floor's expectation = the DIF proportion itself (a random anchor is DIF w.p. proportion)
        xs = sub["proportion"].to_numpy() * 100
        ax.plot(xs, xs / 100, color=C_RED, ls=":", lw=0.7, alpha=0.6)
        ax.set_title(direction)
        ax.set_xlabel("DIF proportion (%)")
        ax.set_ylim(-0.03, 1.0)
    axes[0].set_ylabel("Anchor contamination")
    # single legend ABOVE the panels so it never overlaps the rising lines
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                     bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)

    if save:
        out = os.path.join(_RESULTS_DIR, "phase_5_contamination")
        _save_multi(fig, out, extra=[leg])
        print(f"saved: {out}")
    return fig


def plot_diagnostics(results=None, save=True):
    df = _load(results)
    dirs = _panels(df)
    has_det = "hit_rate_mean" in df.columns
    if _HAS_FIG_STYLE:
        apply_style()
    fig, axes = make_fig(len(dirs), sharey=True, width=7.5, height_override=2.4)
    if len(dirs) == 1:
        axes = [axes]

    for ax, direction in zip(axes, dirs):
        sub = df[df["direction"] == direction].sort_values("proportion")
        _line(ax, sub, "rank_auc_mean", "rank_auc_sd", C_PURPLE, "Rank AUC", marker="o")
        if has_det:
            _line(ax, sub, "hit_rate_mean",    "hit_rate_sd",    C_BLUE, "Hit rate (power)",  marker="^")
            _line(ax, sub, "false_alarm_mean", "false_alarm_sd", C_RED,  "False alarm",        marker="v")
        ax.axhline(0.5, color="gray", ls=":", lw=0.7)   # chance AUC
        ax.set_title(direction)
        ax.set_xlabel("DIF proportion (%)")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Rate")
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                     bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)

    if save:
        out = os.path.join(_RESULTS_DIR, "phase_5_diagnostics")
        _save_multi(fig, out, extra=[leg])
        print(f"saved: {out}")
    return fig


if __name__ == "__main__":
    plot_contamination()
    plot_diagnostics()
