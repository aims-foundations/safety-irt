"""Variant similarity ablation.

Subcommands:
  cohen         : Pairwise Cohen's kappa between variants within each model family
  fleiss        : Fleiss' kappa across variants within each family
  doppelgangers : Cross-family model pairs with similar JSR
  theta         : θ stability across temperature variants
  theta-doppel  : Cross-family θ vs JSR comparison

Usage:
    python variant_ablation.py theta --theta-csv theta_person_params.csv
    python variant_ablation.py theta-doppel --theta-csv theta_person_params.csv
"""

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

# ─── Data ────────────────────────────────────────────────────────────────────

DATA_DIR = snapshot_download(
    repo_id="MaxZ119/safetyirt", repo_type="dataset", token=False
)
INPUT_FILE = os.path.join(DATA_DIR, "processed_data", "Master_Passes0-9_Dataset.csv")

VARIANT_SUFFIXES = [
    "_Low_Creativity", "_Standard_Real", "_Standard",
    "_High_Risk", "_Chaos", "_Reasoning_Default", "_Default",
]

VARIANT_ORDER = {
    "Low_Creativity": 0, "Standard": 1, "Standard_Real": 1,
    "High_Risk": 2, "Chaos": 3, "Reasoning_Default": 1, "Default": 1,
}

FAMILY_COLORS = {
    "claude": "#E07B39", "gpt": "#74AA9C", "gemini": "#4285F4",
    "grok": "#1DA1F2", "deepseek": "#A855F7",
}


def _load_graded():
    """Load graded CSV, remove invalids, binarize."""
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df = df.dropna(subset=["test_taker", "judge_score", "prompt"])
    df["judge_score"] = pd.to_numeric(df["judge_score"], errors="coerce")
    df = df[df["judge_score"] != 0].copy()
    df["is_jailbreak"] = (df["judge_score"] <= 3).astype(int)
    return df


def _get_model_family(name):
    for s in VARIANT_SUFFIXES:
        if name.endswith(s):
            return name.replace(s, "")
    return name


def _get_variant_label(name):
    for s in VARIANT_SUFFIXES:
        if name.endswith(s):
            return s.lstrip("_")
    return "Unknown"


def _get_provider(family_name):
    fn = family_name.lower()
    for key in ["claude", "haiku", "sonnet", "opus"]:
        if key in fn:
            return "claude"
    for key in ["gpt", "4o", "4.1"]:
        if key in fn:
            return "gpt"
    for key in ["gemini", "flash"]:
        if key in fn:
            return "gemini"
    if "grok" in fn:
        return "grok"
    if "deepseek" in fn:
        return "deepseek"
    return "other"


# ─── Cohen's Kappa ───────────────────────────────────────────────────────────

def cmd_cohen(args):
    from sklearn.metrics import cohen_kappa_score
    df = _load_graded()
    df["model_family"] = df["test_taker"].apply(_get_model_family)

    pair_results = []
    for family, group in df.groupby("model_family"):
        pivot = group.pivot_table(index="prompt", columns="test_taker",
                                  values="is_jailbreak", aggfunc="first")
        variants = pivot.columns.tolist()
        if len(variants) < 2:
            continue
        for v1, v2 in combinations(variants, 2):
            pair_data = pivot[[v1, v2]].dropna()
            if len(pair_data) > 0:
                kappa = cohen_kappa_score(pair_data[v1], pair_data[v2])
                pair_results.append({
                    "Model Family": family,
                    "Variant A": v1.replace(family, "").lstrip("_"),
                    "Variant B": v2.replace(family, "").lstrip("_"),
                    "Cohen Kappa": kappa, "Sample Size": len(pair_data),
                })

    results_df = pd.DataFrame(pair_results).sort_values(
        by=["Model Family", "Cohen Kappa"], ascending=[True, False])

    for _, row in results_df.iterrows():
        print(f"{row['Model Family']:<30} | {row['Variant A']:<20} | {row['Variant B']:<20} | {row['Cohen Kappa']:.4f}")

    high = results_df[results_df["Cohen Kappa"] > 0.90]
    print(f"\nHighly redundant pairs (κ > 0.90): {len(high)}")
    if not high.empty:
        print(high[["Model Family", "Variant A", "Variant B", "Cohen Kappa"]].to_string(index=False))


# ─── Fleiss' Kappa ───────────────────────────────────────────────────────────

def _calculate_fleiss_kappa(pivot_df):
    n_total = pivot_df.shape[1]
    n_items = pivot_df.shape[0]
    if n_total < 2 or n_items == 0:
        return np.nan
    count_1 = pivot_df.sum(axis=1)
    count_0 = n_total - count_1
    P_i = ((count_0**2 + count_1**2) - n_total) / (n_total * (n_total - 1))
    P_bar = P_i.mean()
    p_0 = count_0.sum() / (n_items * n_total)
    p_1 = count_1.sum() / (n_items * n_total)
    P_e = p_0**2 + p_1**2
    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def cmd_fleiss(args):
    df = _load_graded()
    df["model_family"] = df["test_taker"].apply(_get_model_family)

    results = []
    for family, group in df.groupby("model_family"):
        pivot = group.pivot_table(index="prompt", columns="test_taker",
                                  values="is_jailbreak", aggfunc="first").dropna()
        if pivot.shape[1] > 1 and pivot.shape[0] > 0:
            kappa = _calculate_fleiss_kappa(pivot)
            jsrs = pivot.mean() * 100
            results.append({
                "Model Family": family, "Variants": pivot.shape[1],
                "Fleiss Kappa": kappa, "JSR Spread (%)": jsrs.max() - jsrs.min(),
            })

    results_df = pd.DataFrame(results).sort_values("Fleiss Kappa", ascending=False)
    for _, row in results_df.iterrows():
        k = row["Fleiss Kappa"]
        interp = "Identical" if k > 0.90 else "Very Similar" if k > 0.75 else "Distinct"
        print(f"{row['Model Family']:<35} | {k:.4f} | {row['Variants']} variants | {interp}")


# ─── Doppelgangers ───────────────────────────────────────────────────────────

def cmd_doppelgangers(args):
    from sklearn.metrics import cohen_kappa_score
    df = _load_graded()
    pivot = df.pivot_table(index="prompt", columns="test_taker",
                           values="is_jailbreak", aggfunc="first")
    jsr_series = pivot.mean() * 100
    models = pivot.columns.tolist()

    results = []
    for model_a, model_b in combinations(models, 2):
        base_a, base_b = _get_model_family(model_a), _get_model_family(model_b)
        if base_a == base_b:
            continue
        diff = abs(jsr_series[model_a] - jsr_series[model_b])
        if diff <= args.jsr_threshold:
            pair_data = pivot[[model_a, model_b]].dropna()
            if len(pair_data) > 50:
                kappa = cohen_kappa_score(pair_data[model_a], pair_data[model_b])
                results.append({
                    "Model A": model_a, "Model B": model_b,
                    "Diff (%)": round(diff, 2), "Kappa": round(kappa, 4),
                })

    results_df = pd.DataFrame(results).sort_values("Kappa", ascending=False)
    for _, row in results_df.head(20).iterrows():
        print(f"{row['Model A']:<30} | {row['Model B']:<30} | Δ{row['Diff (%)']:<5} | κ={row['Kappa']}")


# ─── Theta Stability ─────────────────────────────────────────────────────────

def cmd_theta(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    theta_df = pd.read_csv(args.theta_csv)
    print(f"Loaded {len(theta_df)} test-takers")

    theta_df["base_model"] = theta_df["test_taker"].apply(_get_model_family)
    theta_df["variant"] = theta_df["test_taker"].apply(_get_variant_label)
    theta_df["provider"] = theta_df["base_model"].apply(_get_provider)
    theta_df["variant_order"] = theta_df["variant"].map(VARIANT_ORDER).fillna(1).astype(int)
    theta_df = theta_df.sort_values(["base_model", "variant_order"])

    family_stats = []
    for family, grp in theta_df.groupby("base_model"):
        t = grp["theta"].values
        n = len(t)
        family_stats.append({
            "Base Model": family, "Provider": grp["provider"].iloc[0], "N": n,
            "θ Mean": t.mean(), "θ SD": t.std(ddof=1) if n >= 2 else 0.0,
            "θ Range": t.max() - t.min() if n >= 2 else 0.0,
            "Variants": ", ".join(grp["variant"].tolist()),
        })
    stats_df = pd.DataFrame(family_stats).sort_values("θ Mean", ascending=False)

    print(f"\n{'Base Model':<40} | {'N':>2} | {'θ Mean':>7} | {'θ SD':>6} | {'θ Range':>7}")
    print("-" * 75)
    for _, r in stats_df.iterrows():
        print(f"{r['Base Model']:<40} | {r['N']:>2} | {r['θ Mean']:>7.3f} | {r['θ SD']:>6.4f} | {r['θ Range']:>7.4f}")

    multi = stats_df[stats_df["N"] >= 2]
    between_var = theta_df.groupby("base_model")["theta"].mean().var()
    within_var = multi["θ SD"].apply(lambda x: x**2).mean()
    icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 1.0

    print(f"\n  Models ≥2 variants: {len(multi)}")
    print(f"  Mean within-family θ SD:    {multi['θ SD'].mean():.4f}")
    print(f"  Max within-family θ range:  {multi['θ Range'].max():.4f}")
    print(f"  ICC (between/total):        {icc:.4f}")
    print(f"  Variance from temperature:  {(1-icc)*100:.1f}%")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [3, 1.2, 1.5]})

    ax = axes[0]
    vlabels = ["Low_Creativity", "Standard", "High_Risk", "Chaos"]
    vx = {v: i for i, v in enumerate(vlabels)}
    for family, grp in theta_df.groupby("base_model"):
        color = FAMILY_COLORS.get(grp["provider"].iloc[0], "#999999")
        pg = grp[grp["variant"].isin(vlabels)].sort_values("variant_order")
        if len(pg) < 2:
            continue
        ax.plot([vx[v] for v in pg["variant"]], pg["theta"].values,
                "-o", color=color, alpha=0.7, linewidth=1.5, markersize=5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Low\nCreativity", "Standard", "High\nRisk", "Chaos"])
    ax.set_ylabel("θ")
    ax.set_title("(A) θ Across Temperature Variants", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[Line2D([0], [0], color=c, lw=2, label=p.title())
                       for p, c in FAMILY_COLORS.items()], loc="lower left", fontsize=9)

    ax2 = axes[1]
    for _, r in multi.iterrows():
        ax2.scatter(r["θ Mean"], r["θ SD"], s=r["N"]*40,
                    color=FAMILY_COLORS.get(r["Provider"], "#999"), edgecolors="black", lw=0.5, alpha=0.8)
    ax2.axhline(multi["θ SD"].mean(), color="red", ls="--", alpha=0.5)
    ax2.set_xlabel("Mean θ"); ax2.set_ylabel("Within-Family θ SD")
    ax2.set_title("(B) θ Dispersion", fontweight="bold"); ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    bars = ax3.barh(["Between-Model", "Within-Model\n(Temperature)"],
                    [icc*100, (1-icc)*100], color=["#2196F3", "#FF9800"], edgecolor="black", lw=0.5)
    ax3.set_xlabel("% of θ Variance"); ax3.set_xlim(0, 105)
    ax3.set_title("(C) Variance Decomposition", fontweight="bold")
    for bar, val in zip(bars, [icc*100, (1-icc)*100]):
        ax3.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", fontweight="bold")
    ax3.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(args.theta_csv) or ".", "theta_variant_stability.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Figure saved: {out}")


# ─── Delta Stability ─────────────────────────────────────────────────────────

def cmd_delta(args):
    """δ stability across temperature variants, per language."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    delta_df = pd.read_csv(args.delta_csv)
    print(f"Loaded {len(delta_df)} rows ({delta_df['test_taker'].nunique()} test-takers, {delta_df['language'].nunique()} languages)")

    delta_df["base_model"] = delta_df["test_taker"].apply(_get_model_family)
    delta_df["variant"] = delta_df["test_taker"].apply(_get_variant_label)
    delta_df["provider"] = delta_df["base_model"].apply(_get_provider)

    languages = sorted(delta_df["language"].unique())

    # Per-family, per-language δ stats
    rows = []
    for lang in languages:
        lang_df = delta_df[delta_df["language"] == lang]
        for family, grp in lang_df.groupby("base_model"):
            vals = grp["delta"].values
            n = len(vals)
            if n < 2:
                continue
            rows.append({
                "language": lang, "base_model": family,
                "provider": grp["provider"].iloc[0], "n": n,
                "δ_mean": vals.mean(), "δ_sd": vals.std(ddof=1),
                "δ_range": vals.max() - vals.min(),
            })
    stats_df = pd.DataFrame(rows)

    # Per-language ICC
    lang_icc = []
    for lang in languages:
        ls = stats_df[stats_df["language"] == lang]
        if len(ls) < 2:
            continue
        lang_delta = delta_df[delta_df["language"] == lang]
        between = lang_delta.groupby("base_model")["delta"].mean().var()
        within = ls["δ_sd"].apply(lambda x: x**2).mean()
        icc = between / (between + within) if (between + within) > 0 else 1.0
        lang_icc.append({
            "language": lang, "families": len(ls),
            "mean_within_sd": ls["δ_sd"].mean(),
            "max_range": ls["δ_range"].max(),
            "between_var": between, "within_var": within, "ICC": icc,
        })
    icc_df = pd.DataFrame(lang_icc).sort_values("ICC", ascending=False)

    print(f"\n{'Language':<6} | {'Families':>8} | {'Mean δ SD':>9} | {'Max Range':>9} | {'ICC':>6} | {'Temp %':>6}")
    print("-" * 60)
    for _, r in icc_df.iterrows():
        print(f"{r['language']:<6} | {r['families']:>8} | {r['mean_within_sd']:>9.4f} | {r['max_range']:>9.4f} | {r['ICC']:>6.4f} | {(1-r['ICC'])*100:>5.1f}%")

    # Overall summary
    overall_between = stats_df.groupby(["language", "base_model"])["δ_mean"].first().groupby("base_model").var().mean()
    overall_within = stats_df["δ_sd"].apply(lambda x: x**2).mean()
    overall_icc = overall_between / (overall_between + overall_within) if (overall_between + overall_within) > 0 else 1.0

    print(f"\n  Overall mean within-family δ SD:  {stats_df['δ_sd'].mean():.4f}")
    print(f"  Overall max within-family range:  {stats_df['δ_range'].max():.4f}")
    print(f"  Mean ICC across languages:        {icc_df['ICC'].mean():.4f}")
    print(f"  Min ICC:                          {icc_df['ICC'].min():.4f} ({icc_df.loc[icc_df['ICC'].idxmin(), 'language']})")

    # Within-family vs cross-family Δδ (same as theta analysis)
    within_deltas = []
    cross_deltas = []
    for lang in languages:
        lang_delta = delta_df[delta_df["language"] == lang]
        takers = lang_delta.set_index("test_taker")["delta"].to_dict()
        for (a, da), (b, db) in combinations(takers.items(), 2):
            diff = abs(da - db)
            if _get_model_family(a) == _get_model_family(b):
                within_deltas.append(diff)
            else:
                cross_deltas.append(diff)
    within_deltas = np.array(within_deltas)
    cross_deltas = np.array(cross_deltas)

    print(f"\n  Within-family mean Δδ:  {within_deltas.mean():.4f}")
    print(f"  Cross-family mean Δδ:   {cross_deltas.mean():.4f}")
    print(f"  Ratio:                  {cross_deltas.mean()/within_deltas.mean():.1f}x")

    # Figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={"width_ratios": [2, 1.5, 1.5]})

    # Panel A: δ across variants per language (one subplot showing all langs for a few families)
    ax = axes[0]
    vlabels = ["Low_Creativity", "Standard", "High_Risk", "Chaos"]
    vx = {v: i for i, v in enumerate(vlabels)}

    # Pick top 4 languages by variance to show
    lang_var = delta_df.groupby("language")["delta"].var().sort_values(ascending=False)
    show_langs = lang_var.head(4).index.tolist()
    markers = ["o", "s", "^", "D"]

    for family, fam_grp in delta_df.groupby("base_model"):
        color = FAMILY_COLORS.get(fam_grp["provider"].iloc[0], "#999999")
        for li, lang in enumerate(show_langs):
            lg = fam_grp[(fam_grp["language"] == lang) & (fam_grp["variant"].isin(vlabels))]
            lg = lg.sort_values("variant", key=lambda x: x.map(VARIANT_ORDER))
            if len(lg) < 2:
                continue
            xs = [vx[v] + li * 0.05 for v in lg["variant"]]  # slight offset per lang
            ax.plot(xs, lg["delta"].values, "-", color=color, alpha=0.3, linewidth=0.8,
                    marker=markers[li], markersize=4)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["Low\nCreativity", "Standard", "High\nRisk", "Chaos"])
    ax.set_ylabel("δ (person-language shift)")
    ax.set_title(f"(A) δ Across Variants ({', '.join(show_langs)})", fontweight="bold")
    ax.grid(True, alpha=0.3)
    # Lang markers legend
    from matplotlib.lines import Line2D as L2D
    lang_handles = [L2D([0], [0], color="gray", marker=markers[i], ls="", label=l)
                    for i, l in enumerate(show_langs)]
    prov_handles = [L2D([0], [0], color=c, lw=2, label=p.title()) for p, c in FAMILY_COLORS.items()]
    ax.legend(handles=prov_handles + lang_handles, fontsize=7, ncol=2, loc="best")

    # Panel B: ICC per language
    ax2 = axes[1]
    colors_bar = [FAMILY_COLORS.get("gemini", "#4285F4")] * len(icc_df)
    bars = ax2.barh(icc_df["language"], icc_df["ICC"], color="#4285F4",
                    edgecolor="black", lw=0.5, alpha=0.8)
    ax2.set_xlabel("ICC")
    ax2.set_title("(B) ICC per Language", fontweight="bold")
    ax2.set_xlim(0.9, 1.005)
    ax2.grid(True, axis="x", alpha=0.3)

    # Panel C: Δδ distributions
    ax3 = axes[2]
    bins = np.linspace(0, max(cross_deltas.max(), within_deltas.max()) * 0.5, 30)
    ax3.hist(within_deltas, bins=bins, alpha=0.7, color="#2196F3", edgecolor="black", lw=0.5,
             label=f"Within-family (n={len(within_deltas)})", density=True)
    ax3.hist(cross_deltas, bins=bins, alpha=0.7, color="#FF9800", edgecolor="black", lw=0.5,
             label=f"Cross-family (n={len(cross_deltas)})", density=True)
    ax3.axvline(within_deltas.mean(), color="#2196F3", ls="--", lw=1.5)
    ax3.axvline(cross_deltas.mean(), color="#FF9800", ls="--", lw=1.5)
    ax3.set_xlabel("Δδ"); ax3.set_ylabel("Density")
    ax3.set_title("(C) Within vs Cross-Family Δδ", fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(args.delta_csv) or ".", "delta_variant_stability.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Figure saved: {out}")


# ─── Theta Doppelgangers ─────────────────────────────────────────────────────

def cmd_theta_doppel(args):
    """Cross-family: do models with similar JSR have similar θ?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.metrics import cohen_kappa_score

    theta_df = pd.read_csv(args.theta_csv)
    theta_df["base_model"] = theta_df["test_taker"].apply(_get_model_family)
    theta_df["provider"] = theta_df["base_model"].apply(_get_provider)
    theta_map = dict(zip(theta_df["test_taker"], theta_df["theta"]))

    df = _load_graded()
    jsr_map = df.groupby("test_taker")["is_jailbreak"].mean().to_dict()

    merged = pd.DataFrame([
        {"test_taker": tt, "theta": theta_map[tt], "jsr": jsr_map[tt]*100,
         "base_model": _get_model_family(tt), "provider": _get_provider(_get_model_family(tt))}
        for tt in theta_map if tt in jsr_map
    ])
    print(f"Merged {len(merged)} test-takers with both θ and JSR")

    pivot = df.pivot_table(index="prompt", columns="test_taker",
                           values="is_jailbreak", aggfunc="first")

    # Cross-family pairs with similar JSR
    results = []
    for ma, mb in combinations(merged["test_taker"].tolist(), 2):
        ra = merged[merged["test_taker"] == ma].iloc[0]
        rb = merged[merged["test_taker"] == mb].iloc[0]
        if ra["base_model"] == rb["base_model"]:
            continue
        jsr_diff = abs(ra["jsr"] - rb["jsr"])
        if jsr_diff > args.jsr_threshold:
            continue
        theta_diff = abs(ra["theta"] - rb["theta"])
        kappa = np.nan
        if ma in pivot.columns and mb in pivot.columns:
            pair = pivot[[ma, mb]].dropna()
            if len(pair) > 50:
                try: kappa = cohen_kappa_score(pair[ma], pair[mb])
                except: pass
        results.append({
            "Model A": ma, "Model B": mb,
            "ΔJSR": round(jsr_diff, 2), "θ A": round(ra["theta"], 3),
            "θ B": round(rb["theta"], 3), "Δθ": round(theta_diff, 3),
            "κ": round(kappa, 4) if not np.isnan(kappa) else "—",
        })

    results_df = pd.DataFrame(results).sort_values("Δθ", ascending=False)

    if not results_df.empty:
        print(f"\nCROSS-FAMILY PAIRS WITH SIMILAR JSR (Δ ≤ {args.jsr_threshold}%)")
        print(f"{'Model A':<38} | {'Model B':<38} | {'ΔJSR':>5} | {'θ A':>6} | {'θ B':>6} | {'Δθ':>5} | {'κ':>6}")
        print("-" * 115)
        for _, r in results_df.head(25).iterrows():
            print(f"{r['Model A']:<38} | {r['Model B']:<38} | {r['ΔJSR']:>5} | {r['θ A']:>6} | {r['θ B']:>6} | {r['Δθ']:>5} | {r['κ']:>6}")

        dt = results_df["Δθ"].astype(float)
        print(f"\n  Pairs: {len(results_df)}  |  Mean Δθ: {dt.mean():.3f}  |  Max Δθ: {dt.max():.3f}  |  Δθ>0.3: {(dt>0.3).sum()}  |  Δθ>0.5: {(dt>0.5).sum()}")

    # Within-family Δθ for comparison
    within = []
    for _, grp in theta_df.groupby("base_model"):
        t = grp["theta"].values
        if len(t) >= 2:
            within.extend(abs(a-b) for a, b in combinations(t, 2))
    within = np.array(within)
    cross = results_df["Δθ"].astype(float).values if not results_df.empty else np.array([0])

    print(f"\n  Within-family mean Δθ: {within.mean():.4f}")
    print(f"  Cross-family mean Δθ:  {cross.mean():.4f}")
    print(f"  Ratio:                 {cross.mean()/within.mean():.1f}x")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for prov, color in FAMILY_COLORS.items():
        sub = merged[merged["provider"] == prov]
        ax.scatter(sub["theta"], sub["jsr"], c=color, s=50, alpha=0.8,
                   edgecolors="black", lw=0.3, zorder=3, label=prov.title())
    for _, grp in merged.groupby("base_model"):
        if len(grp) >= 2:
            g = grp.sort_values("theta")
            ax.plot(g["theta"], g["jsr"], "-", color="gray", alpha=0.3, lw=1, zorder=1)
    ax.set_xlabel("θ (IRT Safety Ability)"); ax.set_ylabel("JSR (%)")
    ax.set_title("(A) JSR vs θ", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.invert_yaxis()

    ax2 = axes[1]
    bins = np.linspace(0, max(cross.max(), within.max())*1.1, 30)
    ax2.hist(within, bins=bins, alpha=0.7, color="#2196F3", edgecolor="black", lw=0.5,
             label=f"Within-family (n={len(within)})", density=True)
    if len(cross) > 1:
        ax2.hist(cross, bins=bins, alpha=0.7, color="#FF9800", edgecolor="black", lw=0.5,
                 label=f"Cross-family (n={len(cross)})", density=True)
    ax2.axvline(within.mean(), color="#2196F3", ls="--", lw=1.5)
    ax2.axvline(cross.mean(), color="#FF9800", ls="--", lw=1.5)
    ax2.set_xlabel("Δθ"); ax2.set_ylabel("Density")
    ax2.set_title("(B) Within vs Cross-Family Δθ", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(args.theta_csv) or ".", "theta_doppelganger_analysis.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    print(f"Figure saved: {out}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Variant similarity ablation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cohen").set_defaults(func=cmd_cohen)
    sub.add_parser("fleiss").set_defaults(func=cmd_fleiss)

    p_d = sub.add_parser("doppelgangers")
    p_d.add_argument("--jsr-threshold", type=float, default=0.5)
    p_d.set_defaults(func=cmd_doppelgangers)

    p_t = sub.add_parser("theta")
    p_t.add_argument("--theta-csv", required=True)
    p_t.set_defaults(func=cmd_theta)

    p_dl = sub.add_parser("delta")
    p_dl.add_argument("--delta-csv", required=True)
    p_dl.set_defaults(func=cmd_delta)

    p_td = sub.add_parser("theta-doppel")
    p_td.add_argument("--theta-csv", required=True)
    p_td.add_argument("--jsr-threshold", type=float, default=1.0)
    p_td.set_defaults(func=cmd_theta_doppel)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
