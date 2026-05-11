"""
Analyze head_ablation_multiblimp results across languages.

Reads outputs/head_ablation_multiblimp/<lang_key>/summary.json (and
deltas.json) for each language and produces:

  - Per-language console table: baseline Δ, top-vs-ctrl collective effect,
    sign-split, zero-vs-mean comparison, Spearman ρ of rank vs effect.
  - analysis_summary.csv in --output_dir with one row per language.

Usage
-----
    python analyze.py \\
        --output_dir /projectnb/mcnet/jbrin/lang-probing/outputs/head_ablation_multiblimp \\
        [--langs fra,eng,...]

All 8 language keys (ara,eng,deu,fra,heb,hin,spa,tur) are used by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional: scipy for t-test and spearmanr
# ---------------------------------------------------------------------------
try:
    from scipy.stats import spearmanr, ttest_ind
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn(
        "scipy not found; Spearman correlation and t-test will not be computed. "
        "Install scipy for full statistics.",
        stacklevel=1,
    )

ALL_LANGS = ["ara", "eng", "deu", "fra", "heb", "hin", "spa", "tur"]

CSV_COLUMNS = [
    "lang_key",
    "n_pairs",
    "baseline_delta",
    "top_collective_k5_delta_change",
    "ctrl_collective_k5_delta_change",
    "top_minus_ctrl_collective",
    "signedPOS_delta_change",
    "signedNEG_delta_change",
    "zero_minus_mean_topk5",
    "spearman_rank_vs_effect",
    "spearman_p_value",
    "top_head_mean_dc",
    "top_head_std_dc",
    "ctrl_head_mean_dc",
    "ctrl_head_std_dc",
    "top_vs_ctrl_t_stat",
    "top_vs_ctrl_p_value",
    "mode",
]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _find_condition(conditions: list, name: str) -> dict | None:
    """Return the first condition dict whose 'name' matches exactly."""
    for c in conditions:
        if c.get("name") == name:
            return c
    return None


def _find_condition_prefix(conditions: list, prefix: str) -> list[dict]:
    """Return all conditions whose name starts with prefix."""
    return [c for c in conditions if c.get("name", "").startswith(prefix)]


# ---------------------------------------------------------------------------
# Per-language analysis
# ---------------------------------------------------------------------------

def analyze_language(lang_key: str, output_dir: Path) -> dict | None:
    lang_dir = output_dir / lang_key
    summary_path = lang_dir / "summary.json"

    if not summary_path.exists():
        print(f"  [SKIP] {lang_key}: {summary_path} not found", file=sys.stderr)
        return None

    data = _load_json(summary_path)
    meta = data.get("meta", {})
    conditions = data.get("conditions", [])

    top_k_collective = meta.get("top_k_collective", 5)
    n_pairs = meta.get("n_pairs_used", meta.get("max_pairs", None))
    mode = meta.get("mode", "unknown")

    # ------------------------------------------------------------------ #
    # Baseline
    # ------------------------------------------------------------------ #
    baseline_cond = _find_condition(conditions, "baseline")
    if baseline_cond is None:
        print(f"  [SKIP] {lang_key}: no 'baseline' condition found", file=sys.stderr)
        return None
    baseline_delta = baseline_cond.get("mean_delta", float("nan"))

    # ------------------------------------------------------------------ #
    # Per-head delta_change (top and ctrl), matched by rank
    # ------------------------------------------------------------------ #
    top_rank_conds = sorted(
        _find_condition_prefix(conditions, "top_rank"),
        key=lambda c: c["name"],
    )
    ctrl_rank_conds = sorted(
        _find_condition_prefix(conditions, "ctrl_rank"),
        key=lambda c: c["name"],
    )

    # Pair up by index (they are built 1:1 by rank)
    paired = list(zip(top_rank_conds, ctrl_rank_conds))
    n_pairs_heads = len(paired)

    top_dc_list: list[float] = []
    ctrl_dc_list: list[float] = []
    for tc, cc in paired:
        top_dc_list.append(tc.get("mean_delta", float("nan")) - baseline_delta)
        ctrl_dc_list.append(cc.get("mean_delta", float("nan")) - baseline_delta)

    # GCM rank = 1..N (ascending position in list)
    ranks = list(range(1, n_pairs_heads + 1))

    # Spearman: rank vs top-head delta_change
    spearman_r = float("nan")
    spearman_p = float("nan")
    if _SCIPY and len(top_dc_list) >= 3:
        finite_ranks = [r for r, v in zip(ranks, top_dc_list) if v == v]  # NaN filter
        finite_vals = [v for v in top_dc_list if v == v]
        if len(finite_vals) >= 3:
            result = spearmanr(finite_ranks, finite_vals)
            spearman_r = float(result.statistic)
            spearman_p = float(result.pvalue)

    # Aggregate top vs ctrl
    def _mean_finite(lst):
        vals = [v for v in lst if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    def _std_finite(lst):
        vals = [v for v in lst if v == v]
        if len(vals) < 2:
            return float("nan")
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        return var ** 0.5

    top_head_mean_dc = _mean_finite(top_dc_list)
    top_head_std_dc = _std_finite(top_dc_list)
    ctrl_head_mean_dc = _mean_finite(ctrl_dc_list)
    ctrl_head_std_dc = _std_finite(ctrl_dc_list)

    t_stat = float("nan")
    t_pval = float("nan")
    if _SCIPY:
        finite_top = [v for v in top_dc_list if v == v]
        finite_ctrl = [v for v in ctrl_dc_list if v == v]
        if len(finite_top) >= 2 and len(finite_ctrl) >= 2:
            t_result = ttest_ind(finite_top, finite_ctrl, equal_var=False)
            t_stat = float(t_result.statistic)
            t_pval = float(t_result.pvalue)

    # ------------------------------------------------------------------ #
    # Collective conditions
    # ------------------------------------------------------------------ #
    def _dc(name):
        cond = _find_condition(conditions, name)
        if cond is None:
            return float("nan")
        return cond.get("mean_delta", float("nan")) - baseline_delta

    top_coll_k5_dc = _dc(f"top_collective_k{top_k_collective}_mean")
    ctrl_coll_k5_dc = _dc(f"ctrl_collective_k{top_k_collective}_mean")
    top_minus_ctrl = (
        top_coll_k5_dc - ctrl_coll_k5_dc
        if (top_coll_k5_dc == top_coll_k5_dc and ctrl_coll_k5_dc == ctrl_coll_k5_dc)
        else float("nan")
    )

    # Sign-split: find by prefix (n varies with data)
    def _dc_prefix(prefix):
        matches = _find_condition_prefix(conditions, prefix)
        if not matches:
            return float("nan")
        # take the first match (there should only be one)
        return matches[0].get("mean_delta", float("nan")) - baseline_delta

    signed_pos_dc = _dc_prefix("top_collective_signedPOS_")
    signed_neg_dc = _dc_prefix("top_collective_signedNEG_")

    # Zero vs mean spot check
    zero_cond_name = f"top_collective_k{top_k_collective}_zero"
    zero_delta = _find_condition(conditions, zero_cond_name)
    mean_delta_topk5_val = _find_condition(conditions, f"top_collective_k{top_k_collective}_mean")

    zero_minus_mean = float("nan")
    if zero_delta is not None and mean_delta_topk5_val is not None:
        z = zero_delta.get("mean_delta", float("nan"))
        m = mean_delta_topk5_val.get("mean_delta", float("nan"))
        if z == z and m == m:
            zero_minus_mean = z - m  # positive = zero ablation hurts more

    # ------------------------------------------------------------------ #
    # Return row
    # ------------------------------------------------------------------ #
    return {
        "lang_key": lang_key,
        "n_pairs": n_pairs,
        "baseline_delta": baseline_delta,
        "top_collective_k5_delta_change": top_coll_k5_dc,
        "ctrl_collective_k5_delta_change": ctrl_coll_k5_dc,
        "top_minus_ctrl_collective": top_minus_ctrl,
        "signedPOS_delta_change": signed_pos_dc,
        "signedNEG_delta_change": signed_neg_dc,
        "zero_minus_mean_topk5": zero_minus_mean,
        "spearman_rank_vs_effect": spearman_r,
        "spearman_p_value": spearman_p,
        "top_head_mean_dc": top_head_mean_dc,
        "top_head_std_dc": top_head_std_dc,
        "ctrl_head_mean_dc": ctrl_head_mean_dc,
        "ctrl_head_std_dc": ctrl_head_std_dc,
        "top_vs_ctrl_t_stat": t_stat,
        "top_vs_ctrl_p_value": t_pval,
        "mode": mode,
        # Extra for printing — not written to CSV as separate fields:
        "_top_dc_list": top_dc_list,
        "_ctrl_dc_list": ctrl_dc_list,
        "_paired_count": n_pairs_heads,
    }


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]):
    header = (
        f"{'lang':>4}  {'n_pairs':>7}  {'baseline':>8}  "
        f"{'top_coll5_dc':>12}  {'ctrl_coll5_dc':>13}  "
        f"{'top-ctrl':>8}  {'sPOS_dc':>7}  {'sNEG_dc':>7}  "
        f"{'zero-mean':>9}  {'spearman_r':>10}  {'sp_p':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        def _fmt(v, width=8, decimals=4):
            if v != v:  # NaN
                return " " * (width - 2) + "--"
            return f"{v:{width}.{decimals}f}"

        print(
            f"{r['lang_key']:>4}  "
            f"{str(r['n_pairs'] or '?'):>7}  "
            f"{_fmt(r['baseline_delta'], 8, 4)}  "
            f"{_fmt(r['top_collective_k5_delta_change'], 12, 4)}  "
            f"{_fmt(r['ctrl_collective_k5_delta_change'], 13, 4)}  "
            f"{_fmt(r['top_minus_ctrl_collective'], 8, 4)}  "
            f"{_fmt(r['signedPOS_delta_change'], 7, 4)}  "
            f"{_fmt(r['signedNEG_delta_change'], 7, 4)}  "
            f"{_fmt(r['zero_minus_mean_topk5'], 9, 4)}  "
            f"{_fmt(r['spearman_rank_vs_effect'], 10, 4)}  "
            f"{_fmt(r['spearman_p_value'], 6, 4)}"
        )
    print(sep)
    print(
        "Columns: top_coll5_dc / ctrl_coll5_dc = mean_delta(condition) - mean_delta(baseline).\n"
        "         More negative = ablation hurts model more.\n"
        "         spearman_r = Spearman ρ of GCM rank (1..K) vs per-head delta_change."
    )


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], out_path: Path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Analyze head_ablation_multiblimp results across languages."
    )
    ap.add_argument(
        "--output_dir",
        default="/projectnb/mcnet/jbrin/lang-probing/outputs/head_ablation_multiblimp",
        help="Root directory containing per-language subdirs.",
    )
    ap.add_argument(
        "--langs",
        default=",".join(ALL_LANGS),
        help="Comma-separated list of language keys to process (default: all 8).",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    lang_keys = [k.strip() for k in args.langs.split(",") if k.strip()]

    rows = []
    for lang_key in lang_keys:
        print(f"Analyzing {lang_key}...")
        row = analyze_language(lang_key, output_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No data found for any language. Exiting.")
        sys.exit(1)

    print(f"\n=== Cross-language summary ({len(rows)} languages) ===")
    print_table(rows)

    # Per-language detail: top vs ctrl per-head summary
    print("\n=== Per-language per-head statistics ===")
    for r in rows:
        n_heads = r["_paired_count"]
        if _SCIPY and r["top_vs_ctrl_p_value"] == r["top_vs_ctrl_p_value"]:
            stat_str = (
                f"  t({n_heads}) = {r['top_vs_ctrl_t_stat']:+.3f}, "
                f"p = {r['top_vs_ctrl_p_value']:.4f}"
            )
        else:
            stat_str = "  (scipy not available or insufficient data)"
        print(
            f"  {r['lang_key']:>4}: top heads Δdc = {r['top_head_mean_dc']:+.4f} "
            f"(±{r['top_head_std_dc']:.4f}), "
            f"ctrl Δdc = {r['ctrl_head_mean_dc']:+.4f} (±{r['ctrl_head_std_dc']:.4f})"
            f"{stat_str}"
        )

    csv_path = output_dir / "analysis_summary.csv"
    write_csv(rows, csv_path)


if __name__ == "__main__":
    main()
