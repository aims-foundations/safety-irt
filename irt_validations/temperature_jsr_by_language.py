# -*- coding: utf-8 -*-
"""
JSR vs Temperature/Top-p Variant by Language
=============================================
Plots mean JSR for each temperature variant (Low_Creativity, Standard,
High_Risk, Chaos) with one line per language, using the full
Master_Passes0-9_Dataset.csv (all 1.9M responses).

Variants and their settings:
  Low_Creativity : temp=0.4, top_p=1.00
  Standard       : temp=0.7, top_p=0.90
  High_Risk      : temp=1.0, top_p=0.95
  Chaos          : temp=1.3, top_p=1.00

Output:
  results_temperature/JSR_vs_temperature_by_language.png
  results_temperature/JSR_vs_temperature_by_language.csv
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
warnings.filterwarnings('ignore')

# ── fig_style integration ──────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from fig_style import apply_style, savefig as fs_savefig, make_fig, FULL_WIDTH, DPI
    _HAS_FIG_STYLE = True
except ImportError:
    _HAS_FIG_STYLE = False
    print("[WARN] fig_style.py not found — using fallback styling")

from huggingface_hub import snapshot_download

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_temperature")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ordered by increasing temperature
VARIANT_ORDER = ["Low_Creativity", "Standard", "High_Risk", "Chaos"]
VARIANT_LABELS = {
    "Low_Creativity": "Low Creativity\n(T=0.4, p=1.0)",
    "Standard":       "Standard\n(T=0.7, p=0.9)",
    "High_Risk":      "High Risk\n(T=1.0, p=0.95)",
    "Chaos":          "Chaos\n(T=1.3, p=1.0)",
}

# 10 distinct colors for languages
LANG_COLORS = {
    "en": "#e41a1c",
    "zh": "#377eb8",
    "ar": "#4daf4a",
    "it": "#984ea3",
    "ko": "#ff7f00",
    "vi": "#a65628",
    "th": "#f781bf",
    "bn": "#999999",
    "sw": "#17becf",
    "jv": "#bcbd22",
}

LANG_LABELS = {
    "en": "English",
    "zh": "Chinese",
    "ar": "Arabic",
    "it": "Italian",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "bn": "Bengali",
    "sw": "Swahili",
    "jv": "Javanese",
}


def load_and_aggregate():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, engine="python", on_bad_lines="skip",
                     usecols=["language", "test_taker", "judge_score"])
    print(f"  Loaded {len(df):,} rows.")

    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    df = df[df["judge_score"] > 0].dropna(subset=["judge_score"])
    df["is_safe"] = (df["judge_score"] < 4).astype(float)

    # Extract variant from test_taker (e.g. "gpt-4o-mini_High_Risk" → "High_Risk")
    df["variant"] = df["test_taker"].str.extract(
        r"(Low_Creativity|Standard|High_Risk|Chaos)"
    )
    df = df.dropna(subset=["variant"])

    print(f"  Valid rows after filtering: {len(df):,}")
    print(f"  Variants found: {sorted(df['variant'].unique())}")
    print(f"  Languages found: {sorted(df['language'].unique())}")

    agg = (
        df.groupby(["language", "variant"])["is_safe"]
        .agg(jsr="mean", n="count")
        .reset_index()
    )
    return agg


def plot(agg):
    languages = sorted(agg["language"].unique())
    x_positions = list(range(len(VARIANT_ORDER)))

    if _HAS_FIG_STYLE:
        apply_style()
        fig, ax = make_fig(n_panels=1, height_override=3.0)
        if isinstance(ax, np.ndarray):
            ax = ax[0]
        _save = lambda f, p: fs_savefig(f, p)
        _dpi = DPI
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        _save = lambda f, p: (f.savefig(p, dpi=300, bbox_inches="tight"), plt.close(f))
        _dpi = 300

    for lang in languages:
        sub = agg[agg["language"] == lang].set_index("variant")
        y_vals = [sub.loc[v, "jsr"] if v in sub.index else np.nan for v in VARIANT_ORDER]
        color = LANG_COLORS.get(lang, "#333333")
        label = LANG_LABELS.get(lang, lang.upper())

        ax.plot(x_positions, y_vals,
                marker="o", markersize=5, linewidth=1.5,
                color=color, label=label, zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], fontsize=7)
    ax.set_ylabel("JSR (P(judge_score < 4), i.e. unsafe)")
    ax.set_title("JSR vs Temperature/Top-p Variant by Language")
    ax.set_ylim(
        max(0, agg["jsr"].min() - 0.02),
        min(1.01, agg["jsr"].max() + 0.02)
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.5)
    ax.legend(fontsize=6, loc="best", framealpha=0.8,
              ncol=2, title="Language", title_fontsize=6)

    out_png = os.path.join(RESULTS_DIR, "JSR_vs_temperature_by_language.png")
    out_pdf = os.path.join(RESULTS_DIR, "JSR_vs_temperature_by_language.pdf")
    _save(fig, out_png)
    # also save pdf
    fig2, ax2 = (plt.subplots(figsize=(6, 4)) if not _HAS_FIG_STYLE
                 else (None, None))
    print(f"  Saved: {out_png}")

    # Save a clean PDF copy
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    for lang in languages:
        sub = agg[agg["language"] == lang].set_index("variant")
        y_vals = [sub.loc[v, "jsr"] if v in sub.index else np.nan for v in VARIANT_ORDER]
        ax3.plot(x_positions, y_vals,
                 marker="o", markersize=5, linewidth=1.5,
                 color=LANG_COLORS.get(lang, "#333333"),
                 label=LANG_LABELS.get(lang, lang.upper()), zorder=3)
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], fontsize=7)
    ax3.set_ylabel("JSR (P(judge_score < 4), i.e. unsafe)")
    ax3.set_title("JSR vs Temperature/Top-p Variant by Language")
    ax3.set_ylim(max(0, agg["jsr"].min() - 0.02), min(1.01, agg["jsr"].max() + 0.02))
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax3.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.5)
    ax3.legend(fontsize=6, loc="best", framealpha=0.8,
               ncol=2, title="Language", title_fontsize=6)
    fig3.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {out_pdf}")


def plot_faceted(agg):
    """1×10 row — one panel per language, sorted lowest to highest JSR."""
    lang_mean_jsr = agg.groupby("language")["jsr"].mean()
    languages = lang_mean_jsr.sort_values().index.tolist()
    x_positions = list(range(len(VARIANT_ORDER)))
    y_min = max(0,    agg["jsr"].min() - 0.02)
    y_max = min(1.01, agg["jsr"].max() + 0.02)

    ncols = len(languages)
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(2.2 * ncols, 3.2),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    x_labels_short = ["Low\nCreat.", "Std.", "High\nRisk", "Chaos"]

    for idx, lang in enumerate(languages):
        ax = axes[idx]
        sub = agg[agg["language"] == lang].set_index("variant")
        y_vals = [sub.loc[v, "jsr"] if v in sub.index else np.nan
                  for v in VARIANT_ORDER]
        color = LANG_COLORS.get(lang, "#333333")

        ax.plot(x_positions, y_vals,
                marker="o", markersize=4, linewidth=1.4,
                color=color, zorder=3)
        ax.set_title(LANG_LABELS.get(lang, lang.upper()), fontsize=8, pad=3)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels_short, fontsize=6)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.5)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

        # y tick labels and label only on leftmost panel
        if idx != 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("JSR", fontsize=8, labelpad=6)

    # Layout panels, then add white space at bottom for x-axis title
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.subplots_adjust(bottom=0.28)

    fig.text(0.5, 0.10, "Variant", ha="center", fontsize=8)
    fig.suptitle("JSR vs Variant by Language", fontsize=10, y=0.98)

    out_png = os.path.join(RESULTS_DIR, "JSR_vs_temperature_faceted.png")
    out_pdf = os.path.join(RESULTS_DIR, "JSR_vs_temperature_faceted.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")


def main():
    agg = load_and_aggregate()

    # Print summary table
    pivot = agg.pivot_table(index="language", columns="variant", values="jsr")
    pivot = pivot.reindex(columns=VARIANT_ORDER)
    pivot.index = [LANG_LABELS.get(l, l) for l in pivot.index]
    print("\nMean JSR by Language × Variant:")
    print(pivot.round(4).to_string())

    # Save CSV
    out_csv = os.path.join(RESULTS_DIR, "JSR_vs_temperature_by_language.csv")
    agg.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    plot(agg)
    plot_faceted(agg)
    print("\nDone.")


if __name__ == "__main__":
    main()
