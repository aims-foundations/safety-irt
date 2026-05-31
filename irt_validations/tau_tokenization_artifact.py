# -*- coding: utf-8 -*-
"""
Tokenization Artifact Analysis: Does token-length ratio drive τ?
=================================================================
Tests whether BPE tokenization differences between English and target-language
prompts explain cross-lingual safety gaps (τ_iL).

If tokenization were a confound, prompts that expand more in the target
language (higher BPE token ratio) should systematically show higher τ.

Token counts are computed with tiktoken cl100k_base — the same encoding used
by GPT-4.1-mini (the judge model), making this the most mechanistically
relevant measure: it captures what the judge actually processes.

Outputs (irt_validations/results_tokenization/):
  tokenization_tau_correlation.csv   — per-language ρ table
  tokenization_tau_summary.csv       — pooled + mean-within-language summary
  tokenization_tau_scatter.pdf/.png  — scatter grid per language

Usage:
  python tau_tokenization_artifact.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
from scipy.stats import spearmanr

try:
    from fig_style import apply_style, savefig, FULL_WIDTH, C_RED, C_BLUE
    apply_style()
except ImportError:
    C_RED, C_BLUE = "#c0392b", "#5dade2"
    FULL_WIDTH = 5.5
    def savefig(fig, path):
        fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.join(SCRIPT_DIR, "..")
TQ_CSV      = os.path.join(REPO_ROOT, "model", "results", "multimetric_translation_v_DIF.csv")
OUT_DIR     = os.path.join(SCRIPT_DIR, "results_tokenization")
os.makedirs(OUT_DIR, exist_ok=True)

NON_EN_LANGS = ["ar", "bn", "it", "jv", "ko", "sw", "th", "vi", "zh"]
ENCODING     = "o200k_base"   # GPT-4o / GPT-5.2 tokenizer


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    if not os.path.exists(TQ_CSV):
        raise FileNotFoundError(
            f"Expected pre-computed translation+tau CSV at:\n  {TQ_CSV}\n"
            "Run model/embedding_analysis_translation_v_CSG.py first."
        )
    df = pd.read_csv(TQ_CSV)
    required = {"en_text", "target_text", "tau", "language", "id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {TQ_CSV}: {missing}")
    df = df[df["language"].isin(NON_EN_LANGS)].copy()
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    enc = tiktoken.get_encoding(ENCODING)

    def count_tokens(text):
        try:
            return len(enc.encode(str(text)))
        except Exception:
            return np.nan

    print(f"  Counting BPE tokens ({ENCODING}) for en_text and target_text...")
    df = df.copy()
    df["en_tokens"]   = df["en_text"].map(count_tokens).astype(float)
    df["tgt_tokens"]  = df["target_text"].map(count_tokens).astype(float)
    df["token_ratio"] = df["tgt_tokens"] / df["en_tokens"].replace(0, np.nan)
    return df


# ── Per-language correlation ───────────────────────────────────────────────────
def per_language_corr(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang in NON_EN_LANGS:
        sub = df[df["language"] == lang].dropna(subset=["token_ratio", "tau"])
        if len(sub) < 10:
            continue
        rho, p = spearmanr(sub["token_ratio"], sub["tau"])
        rows.append({
            "language": lang,
            "rho":      round(float(rho), 4),
            "p":        round(float(p),   4),
            "n":        len(sub),
            "sig":      p < 0.05,
        })
    return pd.DataFrame(rows)


def pooled_corr(df: pd.DataFrame):
    sub = df.dropna(subset=["token_ratio", "tau"])
    rho, p = spearmanr(sub["token_ratio"], sub["tau"])
    return float(rho), float(p), len(sub)


def mean_within_lang_rho(lang_df: pd.DataFrame) -> float:
    return float(lang_df["rho"].mean())


# ── Plot: scatter grid ────────────────────────────────────────────────────────
def plot_scatter_grid(df: pd.DataFrame, lang_corr: pd.DataFrame,
                      pooled_rho: float, pooled_p: float):
    label = f"BPE token-count ratio (tgt / en, {ENCODING})"
    langs = [l for l in NON_EN_LANGS if l in df["language"].unique()]
    ncols = 3
    nrows = int(np.ceil(len(langs) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FULL_WIDTH * 1.1, nrows * FULL_WIDTH * 0.35),
                             sharey=False)
    axes = np.array(axes).flatten()

    corr_lookup = lang_corr.set_index("language")[["rho", "p", "sig"]].to_dict("index")

    for i, lang in enumerate(langs):
        ax  = axes[i]
        sub = df[df["language"] == lang].dropna(subset=["token_ratio", "tau"])
        info = corr_lookup.get(lang, {"rho": np.nan, "p": np.nan, "sig": False})
        color = C_RED if info["sig"] else C_BLUE
        ax.scatter(sub["token_ratio"], sub["tau"], s=6, alpha=0.5, color=color, linewidths=0)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        sig_str = "*" if info["sig"] else ""
        ax.set_title(f"{lang}  ρ={info['rho']:+.3f}{sig_str}", fontsize=7)
        ax.set_xlabel("token ratio", fontsize=6)
        ax.set_ylabel("τ", fontsize=6)
        ax.tick_params(labelsize=5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    pooled_str = f"Pooled ρ = {pooled_rho:+.3f}  (p = {pooled_p:.3f})"
    fig.suptitle(f"BPE token ratio vs τ — {ENCODING}\n{pooled_str}", fontsize=8, y=1.01)
    fig.tight_layout()
    savefig(fig, os.path.join(OUT_DIR, "tokenization_tau_scatter_token_ratio"))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} prompt×language pairs across {df['language'].nunique()} languages.")

    df = add_length_features(df)

    lang_corr              = per_language_corr(df)
    pool_rho, pool_p, pool_n = pooled_corr(df)
    mean_rho               = mean_within_lang_rho(lang_corr)
    n_sig                  = lang_corr["sig"].sum()

    print(f"\n{'='*60}")
    print(f"BPE token-count ratio ({ENCODING}) vs τ")
    print(f"{'='*60}")
    print(f"  Pooled ρ = {pool_rho:+.4f}  (p = {pool_p:.4f},  n = {pool_n})")
    print(f"  Mean within-language ρ = {mean_rho:+.4f}")
    print(f"  Languages with p < 0.05: {n_sig} / {len(lang_corr)}")
    print(lang_corr[["language", "rho", "p", "sig", "n"]].to_string(index=False))

    # Save tables
    lang_corr.to_csv(os.path.join(OUT_DIR, "tokenization_tau_correlation.csv"), index=False)

    summary = pd.DataFrame([{
        "encoding":             ENCODING,
        "pooled_rho":           round(pool_rho,  4),
        "pooled_p":             round(pool_p,    4),
        "pooled_n":             pool_n,
        "mean_within_lang_rho": round(mean_rho,  4),
        "n_sig_languages":      int(n_sig),
        "n_languages":          len(lang_corr),
        "R2_pct":               round(pool_rho**2 * 100, 2),
    }])
    summary.to_csv(os.path.join(OUT_DIR, "tokenization_tau_summary.csv"), index=False)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    print(f"\nInterpretation: pooled |ρ| < 0.05 and R² < 0.1% indicates BPE")
    print("tokenization length differences are not a meaningful confounder of τ.")

    plot_scatter_grid(df, lang_corr, pool_rho, pool_p)
    print(f"\nPlot saved → results_tokenization/tokenization_tau_scatter_token_ratio.png")


if __name__ == "__main__":
    main()
