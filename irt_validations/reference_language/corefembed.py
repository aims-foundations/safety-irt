# -*- coding: utf-8 -*-
"""
Aggregate refembed.py outputs across reference languages.

For each reference language run of refembed.py we have a per-row CSV of translation-quality
metrics (LaBSE / COMET / CometKiwi / XCOMET-XL) paired with τ, plus per-language and per-
category Spearman summaries. This script loads those and reports how stable each metric's
relationship to τ is under reference-language perturbation.

Usage:
    python corefembed.py
    python corefembed.py --langs en zh ar
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RESULTS_ROOT = os.path.dirname(os.path.abspath(__file__))

METRIC_KEYS = ["labse", "comet", "cometkiwi", "xcomet_xl"]
METRIC_LABELS = {
    "labse": "LaBSE",
    "comet": "COMET",
    "cometkiwi": "CometKiwi",
    "xcomet_xl": "XCOMET-XL",
}


def load_refembed_results(ref_lang):
    """Load refembed.py outputs for one reference language."""
    d = os.path.join(RESULTS_ROOT, f"results_ref_{ref_lang}")
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"No results directory for ref={ref_lang}. Run refembed.py --ref-lang {ref_lang} first."
        )

    paths = {
        "data": os.path.join(d, f"multimetric_translation_v_DIF_ref_{ref_lang}.csv"),
        "lang": os.path.join(d, f"multimetric_translation_v_DIF_Lang_ref_{ref_lang}.csv"),
        "cat": os.path.join(d, f"multimetric_translation_v_DIF_Cat_ref_{ref_lang}.csv"),
        "summary": os.path.join(d, f"multimetric_translation_v_DIF_Summary_ref_{ref_lang}.csv"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing refembed outputs for ref={ref_lang}: {missing}. "
            f"Run refembed.py --ref-lang {ref_lang} first."
        )

    return {
        "ref_lang": ref_lang,
        "data": pd.read_csv(paths["data"]),
        "lang": pd.read_csv(paths["lang"]),
        "cat": pd.read_csv(paths["cat"]),
        "summary": pd.read_csv(paths["summary"]),
    }


def pooled_metric_vs_tau(df, metric, min_n=10):
    """Pooled Spearman ρ between a metric column and τ across all rows."""
    sub = df[df[metric].notna() & df["tau"].notna()].copy()
    if len(sub) <= min_n:
        return np.nan, np.nan, len(sub)
    rho, p = spearmanr(sub[metric], sub["tau"])
    return float(rho), float(p), int(len(sub))


def _banner(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+",
                        default=["ar", "bn", "en", "it", "ko", "sw", "th", "vi", "zh", "jv"],
                        help="Reference languages to aggregate")
    args = parser.parse_args()

    # ── Load whatever's available ──
    all_data = {}
    for lang in args.langs:
        try:
            all_data[lang] = load_refembed_results(lang)
            n_rows = len(all_data[lang]["data"])
            print(f"  Loaded ref={lang}: {n_rows} (prompt, target_lang) rows")
        except FileNotFoundError as e:
            print(f"  SKIP {lang}: {e}")

    if len(all_data) < 2:
        print("Need refembed outputs for at least 2 reference languages. Exiting.")
        return

    refs = sorted(all_data.keys())
    avail_metrics = []
    for m in METRIC_KEYS:
        for d in all_data.values():
            if m in d["data"].columns and d["data"][m].notna().any():
                avail_metrics.append(m)
                break

    # ──────────────────────────────────────────────────────────────────────
    # 1. POOLED Spearman ρ(metric, τ) per (metric, ref_lang)
    # ──────────────────────────────────────────────────────────────────────
    _banner("1. POOLED Spearman ρ(metric, τ) — one value per (metric × ref_lang)")
    pooled_rows = []
    pooled_table = {m: {} for m in avail_metrics}
    for ref in refs:
        df = all_data[ref]["data"]
        for metric in avail_metrics:
            if metric not in df.columns:
                continue
            rho, p, n = pooled_metric_vs_tau(df, metric)
            pooled_table[metric][ref] = rho
            pooled_rows.append({
                "metric": METRIC_LABELS.get(metric, metric),
                "metric_key": metric,
                "ref_lang": ref,
                "pooled_rho": rho,
                "pooled_p": p,
                "n": n,
            })

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_path = os.path.join(RESULTS_ROOT, "corefembed_pooled_rho_by_ref.csv")
    pooled_df.to_csv(pooled_path, index=False)

    # Print as wide pivot: metric rows × ref_lang columns
    print(f"  {'Metric':<12} | " + " | ".join(f"{r:>7}" for r in refs))
    print(f"  {'-'*12}-+-" + "-+-".join(["-"*7] * len(refs)))
    for metric in avail_metrics:
        vals = [pooled_table[metric].get(r, np.nan) for r in refs]
        cells = " | ".join(f"{v:>+7.3f}" if not np.isnan(v) else f"{'—':>7}" for v in vals)
        print(f"  {METRIC_LABELS.get(metric, metric):<12} | {cells}")
    print(f"  → saved: {pooled_path}")

    # ──────────────────────────────────────────────────────────────────────
    # 2. Per-metric stability summary (across refs)
    # ──────────────────────────────────────────────────────────────────────
    _banner("2. PER-METRIC STABILITY: distribution of pooled ρ across reference languages")
    print(f"  {'Metric':<12} | {'mean ρ':>8} | {'std':>6} | {'min ρ':>7} | {'max ρ':>7} | {'range':>6}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")
    stability_rows = []
    for metric in avail_metrics:
        vals = np.array([v for v in pooled_table[metric].values() if not np.isnan(v)])
        if len(vals) == 0:
            continue
        row = {
            "metric": METRIC_LABELS.get(metric, metric),
            "metric_key": metric,
            "n_refs": len(vals),
            "mean_rho": vals.mean(),
            "std_rho": vals.std(),
            "min_rho": vals.min(),
            "max_rho": vals.max(),
            "range": vals.max() - vals.min(),
        }
        stability_rows.append(row)
        print(f"  {row['metric']:<12} | {row['mean_rho']:>+8.3f} | {row['std_rho']:>6.3f} | "
              f"{row['min_rho']:>+7.3f} | {row['max_rho']:>+7.3f} | {row['range']:>6.3f}")
    stab_path = os.path.join(RESULTS_ROOT, "corefembed_metric_stability.csv")
    pd.DataFrame(stability_rows).to_csv(stab_path, index=False)
    print(f"  → saved: {stab_path}")

    # ──────────────────────────────────────────────────────────────────────
    # 3. Per-language ρ stability vs English (Spearman of per-lang ρ vectors)
    # ──────────────────────────────────────────────────────────────────────
    if "en" in all_data:
        _banner("3. PER-LANGUAGE ρ VECTOR vs ENGLISH (does the per-language pattern survive?)")
        en_lang = all_data["en"]["lang"]
        rows = []
        for metric_label in en_lang["Metric"].unique():
            en_sub = en_lang[en_lang["Metric"] == metric_label].set_index("Language")["Spearman_Rho"]
            for ref in refs:
                if ref == "en":
                    continue
                other_lang = all_data[ref]["lang"]
                ot_sub = other_lang[other_lang["Metric"] == metric_label].set_index("Language")["Spearman_Rho"]
                common = en_sub.index.intersection(ot_sub.index)
                if len(common) < 3:
                    continue
                rho, p = spearmanr(en_sub.loc[common], ot_sub.loc[common])
                rows.append({
                    "metric": metric_label,
                    "ref_other": ref,
                    "n_langs_compared": len(common),
                    "spearman_rho_of_per_lang_rhos": float(rho),
                    "p": float(p),
                })
        if rows:
            stab2 = pd.DataFrame(rows)
            stab2_path = os.path.join(RESULTS_ROOT, "corefembed_per_lang_stability_vs_en.csv")
            stab2.to_csv(stab2_path, index=False)
            # Print: rows = metric, columns = ref_other, values = Spearman ρ
            print(f"  Spearman ρ of (per-language ρ vector) between English-ref and other-ref.")
            print(f"  {'Metric':<12} | " + " | ".join(
                f"{r:>7}" for r in refs if r != "en"))
            print(f"  {'-'*12}-+-" + "-+-".join(["-"*7] * (len(refs) - 1)))
            for metric_label in stab2["metric"].unique():
                sub = stab2[stab2["metric"] == metric_label].set_index("ref_other")["spearman_rho_of_per_lang_rhos"]
                cells = " | ".join(
                    f"{sub.get(r, np.nan):>+7.3f}" if r in sub.index else f"{'—':>7}"
                    for r in refs if r != "en"
                )
                print(f"  {metric_label:<12} | {cells}")
            print(f"  → saved: {stab2_path}")
        else:
            print("  No per-language comparisons available.")
    else:
        print("\n  English ref not loaded — skipping per-language stability table.")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Per-category ρ stability vs English
    # ──────────────────────────────────────────────────────────────────────
    if "en" in all_data:
        _banner("4. PER-CATEGORY ρ VECTOR vs ENGLISH")
        en_cat = all_data["en"]["cat"]
        rows = []
        for metric_label in en_cat["Metric"].unique():
            en_sub = en_cat[en_cat["Metric"] == metric_label].set_index("Category")["Spearman_Rho"]
            for ref in refs:
                if ref == "en":
                    continue
                other_cat = all_data[ref]["cat"]
                ot_sub = other_cat[other_cat["Metric"] == metric_label].set_index("Category")["Spearman_Rho"]
                common = en_sub.index.intersection(ot_sub.index)
                if len(common) < 3:
                    continue
                rho, p = spearmanr(en_sub.loc[common], ot_sub.loc[common])
                rows.append({
                    "metric": metric_label,
                    "ref_other": ref,
                    "n_cats_compared": len(common),
                    "spearman_rho_of_per_cat_rhos": float(rho),
                    "p": float(p),
                })
        if rows:
            cat_stab = pd.DataFrame(rows)
            cat_path = os.path.join(RESULTS_ROOT, "corefembed_per_cat_stability_vs_en.csv")
            cat_stab.to_csv(cat_path, index=False)
            print(f"  Spearman ρ of (per-category ρ vector) between English-ref and other-ref.")
            print(f"  {'Metric':<12} | " + " | ".join(
                f"{r:>7}" for r in refs if r != "en"))
            print(f"  {'-'*12}-+-" + "-+-".join(["-"*7] * (len(refs) - 1)))
            for metric_label in cat_stab["metric"].unique():
                sub = cat_stab[cat_stab["metric"] == metric_label].set_index("ref_other")["spearman_rho_of_per_cat_rhos"]
                cells = " | ".join(
                    f"{sub.get(r, np.nan):>+7.3f}" if r in sub.index else f"{'—':>7}"
                    for r in refs if r != "en"
                )
                print(f"  {metric_label:<12} | {cells}")
            print(f"  → saved: {cat_path}")
        else:
            print("  No per-category comparisons available.")
    else:
        print("\n  English ref not loaded — skipping per-category stability table.")

    # ──────────────────────────────────────────────────────────────────────
    # 5. Bottom line
    # ──────────────────────────────────────────────────────────────────────
    _banner("BOTTOM LINE")
    print(f"  Per-metric pooled ρ range across {len(refs)} reference languages:")
    for row in stability_rows:
        verdict = ("stable" if row["range"] < 0.05 else
                   "moderate" if row["range"] < 0.1 else
                   "sensitive")
        print(f"    {row['metric']:<12}: {row['mean_rho']:+.3f} ± {row['std_rho']:.3f} "
              f"(range {row['range']:.3f} — {verdict})")
    print()


if __name__ == "__main__":
    main()
