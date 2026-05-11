"""
Visualize head_ablation_multiblimp results.

Per-language plots (saved to --out_img_dir/<lang>_*.png):
  <lang>_scatter_rank_vs_effect.png  — GCM rank vs delta_change, top (blue) + ctrl (gray)
  <lang>_bars_per_head.png           — paired top/ctrl bars ordered by GCM rank
  <lang>_sign_split.png              — signedPOS vs signedNEG collective bars
  <lang>_mean_vs_zero.png            — mean-ablation vs zero-ablation of top-5 collective

Cross-language plots:
  cross_lang_correlations.png        — Spearman r per language
  cross_lang_summary.png             — top_collective_k5_delta_change per language + ctrl overlay

Usage
-----
    python visualize.py \\
        --output_dir /projectnb/mcnet/jbrin/lang-probing/outputs/head_ablation_multiblimp \\
        --out_img_dir /projectnb/mcnet/jbrin/lang-probing/img/head_ablation_multiblimp \\
        [--langs fra,eng,...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch use
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import spearmanr
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn(
        "scipy not found; Spearman r annotations will be omitted.",
        stacklevel=1,
    )

ALL_LANGS = ["ara", "eng", "deu", "fra", "heb", "hin", "spa", "tur"]

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
TOP_COLOR = "#2166ac"       # blue
CTRL_COLOR = "#bdbdbd"      # gray
POS_COLOR = "#d73027"       # red
NEG_COLOR = "#4575b4"       # deep blue
ZERO_COLOR = "#fc8d59"      # orange
MEAN_COLOR = "#2166ac"      # blue (same as TOP_COLOR for mean ablation)
BASELINE_COLOR = "#969696"  # mid gray
FIGSIZE_SCATTER = (7, 5)
FIGSIZE_BARS = (10, 5)
FIGSIZE_SMALL = (5, 4)
DPI = 130


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _find_condition(conditions: list, name: str) -> dict | None:
    for c in conditions:
        if c.get("name") == name:
            return c
    return None


def _find_condition_prefix(conditions: list, prefix: str) -> list[dict]:
    return [c for c in conditions if c.get("name", "").startswith(prefix)]


def _dc(cond: dict | None, baseline_delta: float) -> float:
    if cond is None:
        return float("nan")
    md = cond.get("mean_delta", float("nan"))
    if md != md:
        return float("nan")
    return md - baseline_delta


def _is_finite(v: float) -> bool:
    return v == v and not math.isinf(v)


# ---------------------------------------------------------------------------
# Plot 1: scatter rank vs effect
# ---------------------------------------------------------------------------

def plot_scatter_rank_vs_effect(
    lang_key: str,
    top_dc_list: list[float],
    ctrl_dc_list: list[float],
    top_head_labels: list[str],
    spearman_r: float,
    spearman_p: float,
    out_path: Path,
):
    ranks = list(range(1, len(top_dc_list) + 1))

    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)

    # Top heads
    ax.scatter(ranks, top_dc_list, color=TOP_COLOR, s=55, zorder=3, label="Top heads (GCM)")
    # Ctrl heads
    ax.scatter(ranks, ctrl_dc_list, color=CTRL_COLOR, s=35, zorder=2, marker="^",
               label="Stratified controls", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("GCM rank (1 = highest importance)", fontsize=11)
    ax.set_ylabel("Δ = mean_delta(ablation) − mean_delta(baseline)", fontsize=10)

    title = f"{lang_key.upper()} — rank vs ablation effect (per head)"
    if _SCIPY and _is_finite(spearman_r):
        p_str = f"{spearman_p:.3f}" if _is_finite(spearman_p) else "n/a"
        title += f"\nSpearman r = {spearman_r:.3f}  (p = {p_str})"
    ax.set_title(title, fontsize=11)

    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: paired bars per head ordered by GCM rank
# ---------------------------------------------------------------------------

def plot_bars_per_head(
    lang_key: str,
    top_dc_list: list[float],
    ctrl_dc_list: list[float],
    top_head_labels: list[str],
    ctrl_head_labels: list[str],
    out_path: Path,
):
    n = len(top_dc_list)
    if n == 0:
        return

    x = np.arange(n)
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=FIGSIZE_BARS)
    bars_top = ax.bar(x - bar_w / 2, top_dc_list, bar_w, color=TOP_COLOR,
                      label="Top head", alpha=0.85)
    bars_ctrl = ax.bar(x + bar_w / 2, ctrl_dc_list, bar_w, color=CTRL_COLOR,
                       label="Control head", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"r{i+1}\n{lbl}" for i, lbl in enumerate(top_head_labels)],
        fontsize=7, rotation=0
    )
    ax.set_xlabel("GCM rank / head label", fontsize=10)
    ax.set_ylabel("Δ = mean_delta(ablation) − mean_delta(baseline)", fontsize=10)
    ax.set_title(f"{lang_key.upper()} — per-head ablation effect vs. stratified controls", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: sign split
# ---------------------------------------------------------------------------

def plot_sign_split(
    lang_key: str,
    conditions: list[dict],
    baseline_delta: float,
    top_k_collective: int,
    out_path: Path,
):
    def _get_dc_prefix(prefix):
        matches = _find_condition_prefix(conditions, prefix)
        if not matches:
            return float("nan"), "?"
        c = matches[0]
        n_str = c["name"].split("_n")[-1].split("_")[0] if "_n" in c["name"] else "?"
        return _dc(c, baseline_delta), n_str

    pos_dc, pos_n = _get_dc_prefix("top_collective_signedPOS_")
    neg_dc, neg_n = _get_dc_prefix("top_collective_signedNEG_")
    top_k5_dc = _dc(_find_condition(conditions, f"top_collective_k{top_k_collective}_mean"), baseline_delta)

    labels = []
    values = []
    colors = []
    if _is_finite(pos_dc):
        labels.append(f"POS heads\n(n={pos_n})")
        values.append(pos_dc)
        colors.append(POS_COLOR)
    if _is_finite(neg_dc):
        labels.append(f"NEG heads\n(n={neg_n})")
        values.append(neg_dc)
        colors.append(NEG_COLOR)

    if not values:
        return  # nothing to plot

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, alpha=0.85, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Reference: top-k5 collective
    if _is_finite(top_k5_dc):
        ax.axhline(top_k5_dc, color=TOP_COLOR, linewidth=1.2, linestyle=":",
                   alpha=0.8, label=f"top-{top_k_collective} collective ({top_k5_dc:+.3f})")
        ax.legend(fontsize=8, framealpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Δ = mean_delta(ablation) − baseline", fontsize=10)
    ax.set_title(f"{lang_key.upper()} — sign-split collective ablation", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: mean-ablation vs zero-ablation of top-5 collective
# ---------------------------------------------------------------------------

def plot_mean_vs_zero(
    lang_key: str,
    conditions: list[dict],
    baseline_delta: float,
    top_k_collective: int,
    out_path: Path,
):
    mean_cond = _find_condition(conditions, f"top_collective_k{top_k_collective}_mean")
    zero_cond = _find_condition(conditions, f"top_collective_k{top_k_collective}_zero")

    if mean_cond is None or zero_cond is None:
        return  # condition missing (smoke run)

    mean_dc = _dc(mean_cond, baseline_delta)
    zero_dc = _dc(zero_cond, baseline_delta)

    if not (_is_finite(mean_dc) and _is_finite(zero_dc)):
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    labels = [f"Mean ablation\n(top-{top_k_collective})", f"Zero ablation\n(top-{top_k_collective})"]
    values = [mean_dc, zero_dc]
    colors = [MEAN_COLOR, ZERO_COLOR]
    ax.bar([0, 1], values, color=colors, alpha=0.85, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Δ = mean_delta(ablation) − baseline", fontsize=10)
    ax.set_title(f"{lang_key.upper()} — mean-ablation vs zero-ablation (top-{top_k_collective})", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-language: Spearman correlations bar chart
# ---------------------------------------------------------------------------

def plot_cross_lang_correlations(
    lang_rows: list[dict],
    out_path: Path,
):
    valid = [(r["lang_key"], r["spearman_r"]) for r in lang_rows
             if _is_finite(r["spearman_r"])]
    if not valid:
        print("  [SKIP] cross_lang_correlations: no valid Spearman r values", file=sys.stderr)
        return

    langs, rs = zip(*valid)
    x = np.arange(len(langs))

    fig, ax = plt.subplots(figsize=(max(6, len(langs) * 0.9), 4))
    colors = [TOP_COLOR if r < 0 else CTRL_COLOR for r in rs]
    ax.bar(x, rs, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel("Spearman r  (GCM rank vs per-head delta_change)", fontsize=10)
    ax.set_title("Cross-language: Spearman correlation of GCM rank with ablation effect", fontsize=11)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Cross-language: top_collective_k5 vs ctrl bar chart
# ---------------------------------------------------------------------------

def plot_cross_lang_summary(
    lang_rows: list[dict],
    out_path: Path,
):
    valid = [
        r for r in lang_rows
        if _is_finite(r.get("top_coll_k5_dc", float("nan")))
    ]
    if not valid:
        print("  [SKIP] cross_lang_summary: no collective data", file=sys.stderr)
        return

    langs = [r["lang_key"] for r in valid]
    top_vals = [r["top_coll_k5_dc"] for r in valid]
    ctrl_vals = [r.get("ctrl_coll_k5_dc", float("nan")) for r in valid]
    x = np.arange(len(langs))
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(max(7, len(langs) * 1.0), 5))
    ax.bar(x - bar_w / 2, top_vals, bar_w, color=TOP_COLOR, alpha=0.85, label="Top-5 collective (mean abl.)")
    # ctrl overlay — only finite values
    ctrl_x = [xi + bar_w / 2 for xi, v in zip(x, ctrl_vals) if _is_finite(v)]
    ctrl_v = [v for v in ctrl_vals if _is_finite(v)]
    if ctrl_x:
        ax.bar(ctrl_x, ctrl_v, bar_w, color=CTRL_COLOR, alpha=0.85, label="Ctrl-5 collective (mean abl.)")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel("Δ = mean_delta(condition) − mean_delta(baseline)", fontsize=10)
    ax.set_title("Cross-language: top-5 collective ablation effect vs. control", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Per-language driver
# ---------------------------------------------------------------------------

def process_language(lang_key: str, output_dir: Path, img_dir: Path) -> dict | None:
    summary_path = output_dir / lang_key / "summary.json"
    if not summary_path.exists():
        print(f"  [SKIP] {lang_key}: {summary_path} not found", file=sys.stderr)
        return None

    data = _load_json(summary_path)
    meta = data.get("meta", {})
    conditions = data.get("conditions", [])

    top_k_collective = meta.get("top_k_collective", 5)
    lang_name = meta.get("target_lang_name", lang_key.upper())

    # Baseline
    baseline_cond = _find_condition(conditions, "baseline")
    if baseline_cond is None:
        print(f"  [SKIP] {lang_key}: no baseline condition", file=sys.stderr)
        return None
    baseline_delta = baseline_cond.get("mean_delta", float("nan"))

    # Per-head pairs (sorted by rank)
    top_rank_conds = sorted(
        _find_condition_prefix(conditions, "top_rank"),
        key=lambda c: c["name"],
    )
    ctrl_rank_conds = sorted(
        _find_condition_prefix(conditions, "ctrl_rank"),
        key=lambda c: c["name"],
    )
    paired = list(zip(top_rank_conds, ctrl_rank_conds))

    top_dc_list = [tc.get("mean_delta", float("nan")) - baseline_delta for tc, _ in paired]
    ctrl_dc_list = [cc.get("mean_delta", float("nan")) - baseline_delta for _, cc in paired]

    # Build head labels: "L30.H25" from condition name "top_rank01_L30H25_mean"
    def _extract_label(name: str) -> str:
        # name like "top_rank01_L30H25_mean"
        try:
            part = name.split("_")[2]  # "L30H25"
            layer = part.split("H")[0][1:]  # "30"
            head = part.split("H")[1]       # "25"
            return f"L{layer}.H{head}"
        except (IndexError, ValueError):
            return name

    top_head_labels = [_extract_label(tc["name"]) for tc, _ in paired]
    ctrl_head_labels = [_extract_label(cc["name"]) for _, cc in paired]

    # Spearman
    spearman_r = float("nan")
    spearman_p = float("nan")
    if _SCIPY:
        finite_ranks = [i + 1 for i, v in enumerate(top_dc_list) if _is_finite(v)]
        finite_vals = [v for v in top_dc_list if _is_finite(v)]
        if len(finite_vals) >= 3:
            result = spearmanr(finite_ranks, finite_vals)
            spearman_r = float(result.statistic)
            spearman_p = float(result.pvalue)

    # ---- Plots ----
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scatter
    if paired:
        scatter_path = img_dir / f"{lang_key}_scatter_rank_vs_effect.png"
        plot_scatter_rank_vs_effect(
            lang_key, top_dc_list, ctrl_dc_list,
            top_head_labels, spearman_r, spearman_p, scatter_path
        )
        print(f"  Saved {scatter_path}")

    # 2. Bars per head
    if paired:
        bars_path = img_dir / f"{lang_key}_bars_per_head.png"
        plot_bars_per_head(
            lang_key, top_dc_list, ctrl_dc_list,
            top_head_labels, ctrl_head_labels, bars_path
        )
        print(f"  Saved {bars_path}")

    # 3. Sign split
    sign_path = img_dir / f"{lang_key}_sign_split.png"
    plot_sign_split(lang_key, conditions, baseline_delta, top_k_collective, sign_path)
    if sign_path.exists():
        print(f"  Saved {sign_path}")

    # 4. Mean vs zero
    mvz_path = img_dir / f"{lang_key}_mean_vs_zero.png"
    plot_mean_vs_zero(lang_key, conditions, baseline_delta, top_k_collective, mvz_path)
    if mvz_path.exists():
        print(f"  Saved {mvz_path}")

    # Return summary row for cross-language plots
    top_coll_k5_dc = _dc(
        _find_condition(conditions, f"top_collective_k{top_k_collective}_mean"),
        baseline_delta,
    )
    ctrl_coll_k5_dc = _dc(
        _find_condition(conditions, f"ctrl_collective_k{top_k_collective}_mean"),
        baseline_delta,
    )
    return {
        "lang_key": lang_key,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "top_coll_k5_dc": top_coll_k5_dc,
        "ctrl_coll_k5_dc": ctrl_coll_k5_dc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Visualize head_ablation_multiblimp results."
    )
    ap.add_argument(
        "--output_dir",
        default="/projectnb/mcnet/jbrin/lang-probing/outputs/head_ablation_multiblimp",
        help="Root directory containing per-language subdirs.",
    )
    ap.add_argument(
        "--out_img_dir",
        default="/projectnb/mcnet/jbrin/lang-probing/img/head_ablation_multiblimp",
        help="Directory where all plots are saved.",
    )
    ap.add_argument(
        "--langs",
        default=",".join(ALL_LANGS),
        help="Comma-separated list of language keys (default: all 8).",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    img_dir = Path(args.out_img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    lang_keys = [k.strip() for k in args.langs.split(",") if k.strip()]

    lang_rows = []
    for lang_key in lang_keys:
        print(f"Processing {lang_key}...")
        row = process_language(lang_key, output_dir, img_dir)
        if row is not None:
            lang_rows.append(row)

    if not lang_rows:
        print("No data found for any language. Exiting.")
        sys.exit(1)

    print(f"\nGenerating cross-language plots ({len(lang_rows)} languages)...")
    plot_cross_lang_correlations(
        lang_rows, img_dir / "cross_lang_correlations.png"
    )
    plot_cross_lang_summary(
        lang_rows, img_dir / "cross_lang_summary.png"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
