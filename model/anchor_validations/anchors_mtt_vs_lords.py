# -*- coding: utf-8 -*-
"""
Compare Anchor Sets: Backward (2PL/Lord's χ²) vs Forward (Rasch/MTT)
=====================================================================
Reads outputs from both DIF anchor selection methods and computes
overlap at three levels:
  1. Consensus anchors (invariant across ALL language pairs)
  2. Majority anchors (anchor in ≥ 50% of pairs)
  3. Per-language anchor sets

Usage:
  python irt_validations/compare_anchor_methods.py
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKWARD_DIR = os.path.join(BASE_DIR, "model/results_dif_purification")
FORWARD_DIR  = os.path.join(BASE_DIR, "model/results_dif_forward_mtt")
OUTPUT_DIR   = os.path.join(BASE_DIR, "model/results_anchor_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_set(path, col="prompt_id"):
    """Load a CSV and return set of prompt IDs."""
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path}")
        return None
    df = pd.read_csv(path)
    return set(df[col].astype(str))


def load_per_language(path, col="prompt_id"):
    """Load per-language anchor CSV → dict of {lang: set(prompt_ids)}."""
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path}")
        return None
    df = pd.read_csv(path)
    return {lang: set(grp[col].astype(str))
            for lang, grp in df.groupby("language")}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def overlap_report(name, set_bwd, set_fwd):
    """Print and return overlap statistics for two sets."""
    overlap   = set_bwd & set_fwd
    bwd_only  = set_bwd - set_fwd
    fwd_only  = set_fwd - set_bwd
    j         = jaccard(set_bwd, set_fwd)

    print(f"\n{'─' * 50}")
    print(f"{name}")
    print(f"{'─' * 50}")
    print(f"  Backward (2PL/Lord's χ²):     {len(set_bwd):>4} items")
    print(f"  Forward  (Rasch/MTT):         {len(set_fwd):>4} items")
    print(f"  Overlap (both methods):       {len(overlap):>4} items")
    print(f"  Backward-only:                {len(bwd_only):>4} items")
    print(f"  Forward-only:                 {len(fwd_only):>4} items")
    print(f"  Jaccard index:                {j:>7.3f}")

    if set_bwd:
        print(f"  % of backward in overlap:     "
              f"{len(overlap)/len(set_bwd)*100:>6.1f}%")
    if set_fwd:
        print(f"  % of forward in overlap:      "
              f"{len(overlap)/len(set_fwd)*100:>6.1f}%")

    return {
        "n_backward": len(set_bwd),
        "n_forward":  len(set_fwd),
        "n_overlap":  len(overlap),
        "n_bwd_only": len(bwd_only),
        "n_fwd_only": len(fwd_only),
        "jaccard":    round(j, 4),
        "overlap_ids":  sorted(overlap),
        "bwd_only_ids": sorted(bwd_only),
        "fwd_only_ids": sorted(fwd_only),
    }


def main():
    print("=" * 60)
    print("ANCHOR METHOD COMPARISON")
    print("  Backward: 2PL + Lord's χ² + iterative purification")
    print("  Forward:  Rasch + Wald + MTT + forward construction")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. CONSENSUS ANCHORS (invariant across ALL language pairs)
    # ══════════════════════════════════════════════════════════════════════════
    bwd_consensus = load_set(
        os.path.join(BACKWARD_DIR, "dif_consensus_anchors.csv"))
    fwd_consensus = load_set(
        os.path.join(FORWARD_DIR, "forward_consensus_anchors.csv"))

    consensus_stats = None
    if bwd_consensus is not None and fwd_consensus is not None:
        consensus_stats = overlap_report(
            "CONSENSUS ANCHORS (all language pairs)", 
            bwd_consensus, fwd_consensus)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. MAJORITY ANCHORS (≥ 50% of language pairs)
    # ══════════════════════════════════════════════════════════════════════════
    bwd_majority = load_set(
        os.path.join(BACKWARD_DIR, "dif_majority_anchors.csv"))
    fwd_majority = load_set(
        os.path.join(FORWARD_DIR, "forward_majority_anchors.csv"))

    majority_stats = None
    if bwd_majority is not None and fwd_majority is not None:
        majority_stats = overlap_report(
            "MAJORITY ANCHORS (≥ 50% of pairs)",
            bwd_majority, fwd_majority)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. PER-LANGUAGE COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    bwd_per_lang = load_per_language(
        os.path.join(BACKWARD_DIR, "dif_anchors_per_language.csv"))
    fwd_per_lang = load_per_language(
        os.path.join(FORWARD_DIR, "forward_anchors_per_language.csv"))

    lang_rows = []
    if bwd_per_lang is not None and fwd_per_lang is not None:
        all_langs = sorted(set(bwd_per_lang) | set(fwd_per_lang))

        print(f"\n{'=' * 60}")
        print("PER-LANGUAGE ANCHOR OVERLAP")
        print(f"{'=' * 60}")
        print(f"\n{'Lang':<6} {'Bwd':>5} {'Fwd':>5} {'Both':>5} "
              f"{'Bwd-only':>8} {'Fwd-only':>8} {'Jaccard':>8}")
        print("─" * 52)

        for lang in all_langs:
            bwd = bwd_per_lang.get(lang, set())
            fwd = fwd_per_lang.get(lang, set())
            ov  = bwd & fwd
            j   = jaccard(bwd, fwd)

            print(f"{lang:<6} {len(bwd):>5} {len(fwd):>5} {len(ov):>5} "
                  f"{len(bwd-fwd):>8} {len(fwd-bwd):>8} {j:>8.3f}")

            lang_rows.append({
                "language":       lang,
                "n_backward":     len(bwd),
                "n_forward":      len(fwd),
                "n_overlap":      len(ov),
                "n_backward_only": len(bwd - fwd),
                "n_forward_only":  len(fwd - bwd),
                "jaccard":        round(j, 4),
            })

        # Mean Jaccard
        jaccards = [r["jaccard"] for r in lang_rows]
        print(f"\n  Mean Jaccard across languages: {np.mean(jaccards):.3f} "
              f"± {np.std(jaccards):.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. SAVE OUTPUTS
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nSaving to {OUTPUT_DIR}/")

    if consensus_stats is not None:
        # Overlap item list
        pd.DataFrame({"prompt_id": consensus_stats["overlap_ids"]}
                      ).to_csv(os.path.join(OUTPUT_DIR,
                               "consensus_overlap.csv"), index=False)
        pd.DataFrame({"prompt_id": consensus_stats["bwd_only_ids"]}
                      ).to_csv(os.path.join(OUTPUT_DIR,
                               "consensus_backward_only.csv"), index=False)
        pd.DataFrame({"prompt_id": consensus_stats["fwd_only_ids"]}
                      ).to_csv(os.path.join(OUTPUT_DIR,
                               "consensus_forward_only.csv"), index=False)
        print(f"  consensus_overlap.csv          "
              f"({len(consensus_stats['overlap_ids'])} items)")
        print(f"  consensus_backward_only.csv    "
              f"({len(consensus_stats['bwd_only_ids'])} items)")
        print(f"  consensus_forward_only.csv     "
              f"({len(consensus_stats['fwd_only_ids'])} items)")

    # Summary table
    summary_rows = []
    if consensus_stats:
        summary_rows.append({"level": "consensus", **{k: v for k, v in
                             consensus_stats.items() if k.endswith("_ids") is False
                             and not isinstance(v, list)}})
    if majority_stats:
        summary_rows.append({"level": "majority", **{k: v for k, v in
                             majority_stats.items() if not isinstance(v, list)}})
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(OUTPUT_DIR, "method_comparison_summary.csv"),
            index=False)
        print("  method_comparison_summary.csv")

    if lang_rows:
        pd.DataFrame(lang_rows).to_csv(
            os.path.join(OUTPUT_DIR, "per_language_comparison.csv"),
            index=False)
        print("  per_language_comparison.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()