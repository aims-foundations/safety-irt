# -*- coding: utf-8 -*-
"""
Translation Quality Validation for Low-Resource Languages
==========================================================
Addresses the reviewer concern that native-speaker TQ validation covers only
zh/th/bn and is absent for Javanese (jv) and Swahili (sw).

This script provides three proxy validations:

  1. Human-TQ robustness check (bn, th, zh)
     Shows that filtering to high-TQ prompts (TQ >= 4) barely changes mean |τ|,
     providing indirect evidence that the same would hold for sw/jv.

  2. Automated inter-metric agreement by language
     If LaBSE, COMET, CometKiwi, XCOMET-XL agree with each other similarly in
     sw/jv as in bn/th, the metrics are not systematically less reliable there.

  3. Automated-TQ-filter robustness for sw/jv
     Removes bottom-quartile COMET items from sw/jv and checks whether
     mean |τ| changes substantially (proxy for high-TQ filtering).

Outputs (irt_validations/results_tq_lowresource/):
  tq_robustness_human.csv        — human-TQ high-TQ filter effect on |τ|
  tq_intermetric_agreement.csv   — mean inter-metric ρ by language
  tq_robustness_automated.csv    — automated-TQ filter effect on |τ| (all langs)
  tq_lowresource_summary.txt     — plain-text summary for rebuttal

Usage:
  python tq_lowresource_validation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from huggingface_hub import snapshot_download
    DATA_DIR = snapshot_download(repo_id="safety-irt/safety-data", repo_type="dataset", token=False)
except Exception:
    DATA_DIR = "."

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.join(SCRIPT_DIR, "..", "..")
TQ_CSV      = os.path.join(REPO_ROOT, "model", "results", "multimetric_translation_v_DIF.csv")
HUMAN_TQ    = os.path.join(DATA_DIR, "human_translation_validation", "human_translation_quality.csv")
OUT_DIR     = os.path.join(SCRIPT_DIR, "results_tq_lowresource")
os.makedirs(OUT_DIR, exist_ok=True)

METRICS     = ["labse", "comet", "cometkiwi", "xcomet_xl"]
METRIC_PAIRS = [("labse","comet"), ("labse","cometkiwi"), ("labse","xcomet_xl"),
                ("comet","cometkiwi"), ("comet","xcomet_xl"), ("cometkiwi","xcomet_xl")]
LANG_ORDER  = ["zh", "it", "ko", "vi", "ar", "th", "bn", "sw", "jv"]
VALIDATED   = ["zh", "th", "bn"]
LOW_RES     = ["sw", "jv"]


def load_data():
    df = pd.read_csv(TQ_CSV)
    df["id"] = df["id"].astype(str)
    df = df[df["language"].isin(LANG_ORDER)].copy()

    htq = None
    if os.path.exists(HUMAN_TQ):
        htq = pd.read_csv(HUMAN_TQ)
        htq["id"] = htq["id"].astype(str)
        htq = htq.merge(
            df[["id", "language"] + METRICS + ["tau"]],
            on=["id", "language"], how="inner"
        )
    return df, htq


# ── Analysis 1: Human-TQ robustness ──────────────────────────────────────────
def human_tq_robustness(htq: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang in VALIDATED:
        sub = htq[htq["language"] == lang].dropna(subset=["translation_quality", "tau"])
        high = sub[sub["translation_quality"] >= 4]
        rho, p = spearmanr(sub["translation_quality"], sub["tau"])
        rows.append({
            "language":        lang,
            "n_all":           len(sub),
            "mean_abs_tau_all": round(sub["tau"].abs().mean(), 3),
            "n_high_tq":       len(high),
            "mean_abs_tau_high_tq": round(high["tau"].abs().mean(), 3),
            "delta_abs_tau":   round(high["tau"].abs().mean() - sub["tau"].abs().mean(), 3),
            "human_tq_vs_tau_rho": round(rho, 3),
            "human_tq_vs_tau_p":   round(p, 4),
        })
    return pd.DataFrame(rows)


# ── Analysis 1b: Human TQ vs automated metrics (Bengali bridge) ───────────────
def human_vs_automated_metrics(htq: pd.DataFrame) -> pd.DataFrame:
    """
    For validated languages (zh/th/bn), compute Spearman ρ between human TQ
    and each automated metric. Bengali is lower-resource than zh/th; high
    agreement there supports metric reliability in sw/jv.
    """
    rows = []
    for lang in VALIDATED:
        sub = htq[htq["language"] == lang].dropna(subset=["translation_quality"] + METRICS)
        metric_rhos = {}
        rho_vals = []
        for m in METRICS:
            r, p = spearmanr(sub["translation_quality"], sub[m])
            metric_rhos[f"rho_human_{m}"] = round(float(r), 3)
            metric_rhos[f"p_human_{m}"]   = round(float(p), 4)
            rho_vals.append(r)
        rows.append({
            "language": lang,
            "n": len(sub),
            "mean_rho_human_vs_automated": round(float(np.mean(rho_vals)), 3),
            **metric_rhos,
        })
    return pd.DataFrame(rows)


# ── Analysis 2: Inter-metric agreement ───────────────────────────────────────
def intermetric_agreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang in LANG_ORDER:
        sub = df[df["language"] == lang].dropna(subset=METRICS)
        rhos = []
        pair_rhos = {}
        for m1, m2 in METRIC_PAIRS:
            r, _ = spearmanr(sub[m1], sub[m2])
            rhos.append(r)
            pair_rhos[f"rho_{m1}_{m2}"] = round(r, 3)
        rows.append({
            "language": lang,
            "n": len(sub),
            "mean_inter_metric_rho": round(np.mean(rhos), 3),
            "validated": lang in VALIDATED,
            **pair_rhos,
        })
    return pd.DataFrame(rows)


# ── Analysis 3: Automated-TQ filter robustness ───────────────────────────────
def automated_tq_robustness(df: pd.DataFrame, metric: str = "comet",
                             quantile: float = 0.25) -> pd.DataFrame:
    rows = []
    for lang in LANG_ORDER:
        sub = df[df["language"] == lang].dropna(subset=[metric, "tau"])
        cutoff = sub[metric].quantile(quantile)
        high = sub[sub[metric] >= cutoff]
        rows.append({
            "language":            lang,
            "filter_metric":       metric,
            "cutoff_quantile":     quantile,
            "cutoff_value":        round(cutoff, 4),
            "n_all":               len(sub),
            "mean_abs_tau_all":    round(sub["tau"].abs().mean(), 3),
            "n_filtered":          len(high),
            "mean_abs_tau_filtered": round(high["tau"].abs().mean(), 3),
            "delta_abs_tau":       round(high["tau"].abs().mean() - sub["tau"].abs().mean(), 3),
            "validated":           lang in VALIDATED,
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df, htq = load_data()
    print(f"  {len(df)} prompt×language pairs; human TQ: {len(htq) if htq is not None else 0} rows")

    # 1. Human TQ robustness
    if htq is not None:
        rob_human = human_tq_robustness(htq)
        rob_human.to_csv(os.path.join(OUT_DIR, "tq_robustness_human.csv"), index=False)
        print("\n=== 1. Human-TQ robustness (validated languages: zh, th, bn) ===")
        print(rob_human[["language","n_all","mean_abs_tau_all","n_high_tq",
                          "mean_abs_tau_high_tq","delta_abs_tau",
                          "human_tq_vs_tau_rho","human_tq_vs_tau_p"]].to_string(index=False))

        # 1b. Human TQ vs automated metrics
        hvm = human_vs_automated_metrics(htq)
        hvm.to_csv(os.path.join(OUT_DIR, "tq_human_vs_automated.csv"), index=False)
        print("\n=== 1b. Human TQ vs automated metrics (Bengali bridge) ===")
        print(hvm[["language","n","mean_rho_human_vs_automated"] +
                   [f"rho_human_{m}" for m in METRICS]].to_string(index=False))
    else:
        print("  Human TQ file not found — skipping analysis 1.")
        rob_human = None
        hvm = None

    # 2. Inter-metric agreement
    agree = intermetric_agreement(df)
    agree.to_csv(os.path.join(OUT_DIR, "tq_intermetric_agreement.csv"), index=False)
    print("\n=== 2. Inter-metric agreement by language ===")
    print(agree[["language","n","mean_inter_metric_rho","validated"]].to_string(index=False))

    # 3. Automated TQ filter robustness
    rob_auto = automated_tq_robustness(df, metric="comet", quantile=0.25)
    rob_auto.to_csv(os.path.join(OUT_DIR, "tq_robustness_automated.csv"), index=False)
    print("\n=== 3. Automated-TQ filter robustness (remove bottom-quartile COMET) ===")
    print(rob_auto[["language","mean_abs_tau_all","mean_abs_tau_filtered",
                    "delta_abs_tau","validated"]].to_string(index=False))

    # Summary text
    lines = ["=== TQ Low-Resource Validation Summary ===\n"]

    if rob_human is not None:
        lines.append("1. Human-TQ robustness (zh/th/bn):")
        for _, r in rob_human.iterrows():
            lines.append(
                f"   {r.language}: mean|τ| all={r.mean_abs_tau_all} → "
                f"high-TQ (≥4)={r.mean_abs_tau_high_tq} "
                f"(Δ={r.delta_abs_tau:+.3f}); "
                f"human TQ vs τ: ρ={r.human_tq_vs_tau_rho}, p={r.human_tq_vs_tau_p}"
            )
        lines.append("")

    if hvm is not None:
        lines.append("1b. Human TQ vs automated metrics (Bengali bridge):")
        for _, r in hvm.iterrows():
            rho_parts = ", ".join(
                f"{m}={r[f'rho_human_{m}']}" for m in METRICS
            )
            lines.append(
                f"   {r.language} (n={r.n}): mean ρ={r.mean_rho_human_vs_automated}  [{rho_parts}]"
            )
        lines.append("")

    lines.append("2. Inter-metric agreement (mean Spearman ρ across 6 metric pairs):")
    for _, r in agree.iterrows():
        tag = "(validated)" if r.validated else "(automated only)"
        lines.append(f"   {r.language}: {r.mean_inter_metric_rho:.3f} {tag}")
    lines.append("")

    lines.append("3. Automated-TQ filter robustness (remove bottom-25% COMET):")
    for _, r in rob_auto.iterrows():
        tag = "(validated)" if r.validated else "(automated only)"
        lines.append(
            f"   {r.language}: {r.mean_abs_tau_all} → {r.mean_abs_tau_filtered} "
            f"(Δ={r.delta_abs_tau:+.3f}) {tag}"
        )

    summary_path = os.path.join(OUT_DIR, "tq_lowresource_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
