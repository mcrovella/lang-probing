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
  cross_lang_sign_split.png          — POS/NEG signed-IE collectives across languages
  cross_lang_mean_vs_zero.png        — top-5 mean-vs-zero ablation across languages
  cross_lang_per_head_summary.png    — mean single-head top vs ctrl effects across languages

Usage
-----
    python visualize.py \\
        --output_dir /projectnb/mcnet/jbrin/lang-probing/outputs/head_ablation_multiblimp \\
        --out_img_dir /projectnb/mcnet/jbrin/lang-probing/experiments/head_ablation_multiblimp/img \\
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


def _mean_std_finite(vals: list[float]) -> tuple[float, float]:
    finite = [v for v in vals if _is_finite(v)]
    if not finite:
        return float("nan"), float("nan")
    mean = float(sum(finite) / len(finite))
    if len(finite) < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
    return mean, float(var ** 0.5)


def _dc_prefix_with_n(
    conditions: list[dict],
    prefix: str,
    baseline_delta: float,
) -> tuple[float, int | None]:
    matches = _find_condition_prefix(conditions, prefix)
    if not matches:
        return float("nan"), None
    cond = matches[0]
    n = None
    if "_n" in cond.get("name", ""):
        try:
            n = int(cond["name"].split("_n")[-1].split("_")[0])
        except ValueError:
            n = None
    return _dc(cond, baseline_delta), n


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
# Plot 5: sign-split with matched-size controls
# ---------------------------------------------------------------------------

def plot_sign_split_matched(
    lang_key: str,
    conditions: list[dict],
    baseline_delta: float,
    top_k_collective: int,
    out_path: Path,
):
    """4-bar plot: POS-IE collective, matched-POS ctrl, NEG-IE collective, matched-NEG ctrl."""

    def _get_dc_prefix(prefix):
        matches = _find_condition_prefix(conditions, prefix)
        if not matches:
            return float("nan"), "?"
        c = matches[0]
        n_str = c["name"].split("_n")[-1].split("_")[0] if "_n" in c["name"] else "?"
        return _dc(c, baseline_delta), n_str

    pos_dc, pos_n = _get_dc_prefix("top_collective_signedPOS_")
    neg_dc, neg_n = _get_dc_prefix("top_collective_signedNEG_")
    mpos_dc, mpos_n = _get_dc_prefix("ctrl_collective_matched_POS_")
    mneg_dc, mneg_n = _get_dc_prefix("ctrl_collective_matched_NEG_")

    # Only save a "matched" plot when matched controls are actually present.
    # Otherwise the filename overstates the evidence in older 46-condition runs.
    has_matched = _is_finite(mpos_dc) or _is_finite(mneg_dc)
    if not has_matched:
        return

    labels = []
    values = []
    colors = []
    hatches = []

    if _is_finite(pos_dc):
        labels.append(f"POS-IE\n(n={pos_n})")
        values.append(pos_dc)
        colors.append(POS_COLOR)
        hatches.append("")
    if _is_finite(mpos_dc):
        labels.append(f"Matched-POS ctrl\n(n={mpos_n})")
        values.append(mpos_dc)
        colors.append(POS_COLOR)
        hatches.append("///")
    if _is_finite(neg_dc):
        labels.append(f"NEG-IE\n(n={neg_n})")
        values.append(neg_dc)
        colors.append(NEG_COLOR)
        hatches.append("")
    if _is_finite(mneg_dc):
        labels.append(f"Matched-NEG ctrl\n(n={mneg_n})")
        values.append(mneg_dc)
        colors.append(NEG_COLOR)
        hatches.append("///")

    if not values:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(values) * 1.4), 4))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, hatch=hatches, alpha=0.8, width=0.55,
                  edgecolor="white")
    # Re-draw edges for hatched bars
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_edgecolor("black")
            bar.set_linewidth(0.7)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="baseline (0)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Δ = mean_delta(ablation) − baseline", fontsize=10)
    ax.set_title(f"{lang_key.upper()} — sign-split collective vs matched-size controls", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: ctrl k=5 seed distribution
# ---------------------------------------------------------------------------

def plot_ctrl_seed_distribution(
    lang_key: str,
    conditions: list[dict],
    baseline_delta: float,
    top_k_collective: int,
    out_path: Path,
):
    """Strip/scatter showing ctrl_collective_k5_mean + all seed variants,
    with top_collective_k5_mean marked separately."""

    # Original ctrl k5 (the fixed one from the main run)
    orig_ctrl_cond = _find_condition(conditions, f"ctrl_collective_k{top_k_collective}_mean")
    orig_ctrl_dc = _dc(orig_ctrl_cond, baseline_delta)

    # Extra seed variants
    seed_conds = _find_condition_prefix(conditions, f"ctrl_collective_k{top_k_collective}_seed")
    seed_dcs = [_dc(c, baseline_delta) for c in seed_conds]
    seed_labels = [c["name"].replace(f"ctrl_collective_k{top_k_collective}_seed", "s").replace("_mean", "")
                   for c in seed_conds]

    # Need at least one seed to make the plot meaningful
    if not seed_dcs:
        return

    # Top collective
    top_coll_cond = _find_condition(conditions, f"top_collective_k{top_k_collective}_mean")
    top_coll_dc = _dc(top_coll_cond, baseline_delta)

    # Build combined ctrl distribution (orig + seeds)
    all_ctrl_dcs = []
    if _is_finite(orig_ctrl_dc):
        all_ctrl_dcs.append(orig_ctrl_dc)
    all_ctrl_dcs.extend([v for v in seed_dcs if _is_finite(v)])

    ctrl_mean = float("nan")
    ctrl_std = float("nan")
    if len(all_ctrl_dcs) >= 1:
        ctrl_mean = sum(all_ctrl_dcs) / len(all_ctrl_dcs)
    if len(all_ctrl_dcs) >= 2:
        var = sum((v - ctrl_mean) ** 2 for v in all_ctrl_dcs) / (len(all_ctrl_dcs) - 1)
        ctrl_std = var ** 0.5

    fig, ax = plt.subplots(figsize=(max(5, len(all_ctrl_dcs) * 0.7 + 2), 4))

    # Plot each ctrl point
    x_ctrl = list(range(len(all_ctrl_dcs)))
    labels_ctrl = (
        (["orig"] if _is_finite(orig_ctrl_dc) else []) +
        [lbl for lbl, v in zip(seed_labels, seed_dcs) if _is_finite(v)]
    )
    ax.scatter(x_ctrl, all_ctrl_dcs, color=CTRL_COLOR, s=70, zorder=3,
               label=f"ctrl-{top_k_collective} seeds (n={len(all_ctrl_dcs)})")

    # Annotate ctrl mean and std band
    if _is_finite(ctrl_mean):
        ax.axhline(ctrl_mean, color=CTRL_COLOR, linewidth=1.5, linestyle="-",
                   alpha=0.9, label=f"ctrl mean = {ctrl_mean:+.4f}")
        if _is_finite(ctrl_std):
            ax.axhspan(ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                       color=CTRL_COLOR, alpha=0.15,
                       label=f"±1 SD ({ctrl_std:.4f})")

    # Top collective reference
    if _is_finite(top_coll_dc):
        ax.axhline(top_coll_dc, color=TOP_COLOR, linewidth=2.0, linestyle="--",
                   alpha=0.9, label=f"top-{top_k_collective} collective = {top_coll_dc:+.4f}")

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.4)

    ax.set_xticks(x_ctrl)
    ax.set_xticklabels(labels_ctrl, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Δ = mean_delta(ablation) − baseline", fontsize=10)
    ax.set_title(
        f"{lang_key.upper()} — ctrl k={top_k_collective} seed distribution"
        + (f"\nmean={ctrl_mean:+.4f}  std={ctrl_std:.4f}" if _is_finite(ctrl_std) else ""),
        fontsize=11,
    )
    ax.legend(fontsize=8, framealpha=0.8)
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
# Cross-language: sign-split POS/NEG collectives
# ---------------------------------------------------------------------------

def plot_cross_lang_sign_split(
    lang_rows: list[dict],
    out_path: Path,
):
    valid = [
        r for r in lang_rows
        if _is_finite(r.get("signed_pos_dc", float("nan")))
        or _is_finite(r.get("signed_neg_dc", float("nan")))
    ]
    if not valid:
        print("  [SKIP] cross_lang_sign_split: no sign-split data", file=sys.stderr)
        return

    langs = [r["lang_key"] for r in valid]
    x = np.arange(len(langs))
    width = 0.18
    series = [
        ("POS-IE", "signed_pos_dc", POS_COLOR, ""),
        ("POS ctrl", "matched_pos_dc", POS_COLOR, "///"),
        ("NEG-IE", "signed_neg_dc", NEG_COLOR, ""),
        ("NEG ctrl", "matched_neg_dc", NEG_COLOR, "///"),
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(langs) * 1.15), 5))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for offset, (label, key, color, hatch) in zip(offsets, series):
        values = [r.get(key, float("nan")) for r in valid]
        finite_x = [xi + offset for xi, v in zip(x, values) if _is_finite(v)]
        finite_v = [v for v in values if _is_finite(v)]
        if not finite_v:
            continue
        bars = ax.bar(
            finite_x,
            finite_v,
            width,
            color=color,
            hatch=hatch,
            alpha=0.82,
            label=label,
            edgecolor="black" if hatch else "white",
            linewidth=0.6 if hatch else 0.0,
        )
        for bar in bars:
            if hatch:
                bar.set_linewidth(0.7)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel("Δ = mean_delta(condition) − mean_delta(baseline)", fontsize=10)
    ax.set_title("Cross-language: signed-IE collective ablations", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.85, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Cross-language: top-k mean ablation vs zero ablation
# ---------------------------------------------------------------------------

def plot_cross_lang_mean_vs_zero(
    lang_rows: list[dict],
    out_path: Path,
):
    valid = [
        r for r in lang_rows
        if _is_finite(r.get("top_coll_k5_dc", float("nan")))
        or _is_finite(r.get("top_coll_k5_zero_dc", float("nan")))
    ]
    if not valid:
        print("  [SKIP] cross_lang_mean_vs_zero: no mean/zero data", file=sys.stderr)
        return

    langs = [r["lang_key"] for r in valid]
    mean_vals = [r.get("top_coll_k5_dc", float("nan")) for r in valid]
    zero_vals = [r.get("top_coll_k5_zero_dc", float("nan")) for r in valid]
    x = np.arange(len(langs))
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(max(7, len(langs) * 1.0), 5))
    ax.bar(x - bar_w / 2, mean_vals, bar_w, color=MEAN_COLOR, alpha=0.85, label="Mean ablation")
    finite_zero_x = [xi + bar_w / 2 for xi, v in zip(x, zero_vals) if _is_finite(v)]
    finite_zero_v = [v for v in zero_vals if _is_finite(v)]
    if finite_zero_v:
        ax.bar(finite_zero_x, finite_zero_v, bar_w, color=ZERO_COLOR, alpha=0.85, label="Zero ablation")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel("Δ = mean_delta(condition) − mean_delta(baseline)", fontsize=10)
    ax.set_title("Cross-language: top-5 mean ablation vs zero ablation", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Cross-language: average per-head top vs ctrl effects
# ---------------------------------------------------------------------------

def plot_cross_lang_per_head_summary(
    lang_rows: list[dict],
    out_path: Path,
):
    valid = [
        r for r in lang_rows
        if _is_finite(r.get("top_head_mean_dc", float("nan")))
        or _is_finite(r.get("ctrl_head_mean_dc", float("nan")))
    ]
    if not valid:
        print("  [SKIP] cross_lang_per_head_summary: no per-head data", file=sys.stderr)
        return

    langs = [r["lang_key"] for r in valid]
    x = np.arange(len(langs))
    bar_w = 0.38
    top_means = [r.get("top_head_mean_dc", float("nan")) for r in valid]
    ctrl_means = [r.get("ctrl_head_mean_dc", float("nan")) for r in valid]
    top_err = [0.0 if not _is_finite(r.get("top_head_std_dc", float("nan"))) else r["top_head_std_dc"] for r in valid]
    ctrl_err = [0.0 if not _is_finite(r.get("ctrl_head_std_dc", float("nan"))) else r["ctrl_head_std_dc"] for r in valid]

    fig, ax = plt.subplots(figsize=(max(7, len(langs) * 1.0), 5))
    ax.bar(
        x - bar_w / 2,
        top_means,
        bar_w,
        yerr=top_err,
        color=TOP_COLOR,
        alpha=0.85,
        capsize=3,
        label="Top heads",
    )
    ax.bar(
        x + bar_w / 2,
        ctrl_means,
        bar_w,
        yerr=ctrl_err,
        color=CTRL_COLOR,
        alpha=0.85,
        capsize=3,
        label="Control heads",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel("Mean per-head Δ change (error bars = SD over heads)", fontsize=10)
    ax.set_title("Cross-language: average single-head ablation effects", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.85)
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

    # 5. Sign-split with matched-size controls (new)
    ssm_path = img_dir / f"{lang_key}_sign_split_matched.png"
    plot_sign_split_matched(lang_key, conditions, baseline_delta, top_k_collective, ssm_path)
    if ssm_path.exists():
        print(f"  Saved {ssm_path}")

    # 6. Ctrl k=5 seed distribution (new; only created when seeds are present)
    csd_path = img_dir / f"{lang_key}_ctrl_seed_distribution.png"
    plot_ctrl_seed_distribution(lang_key, conditions, baseline_delta, top_k_collective, csd_path)
    if csd_path.exists():
        print(f"  Saved {csd_path}")

    # Return summary row for cross-language plots
    top_coll_k5_dc = _dc(
        _find_condition(conditions, f"top_collective_k{top_k_collective}_mean"),
        baseline_delta,
    )
    ctrl_coll_k5_dc = _dc(
        _find_condition(conditions, f"ctrl_collective_k{top_k_collective}_mean"),
        baseline_delta,
    )
    top_coll_k5_zero_dc = _dc(
        _find_condition(conditions, f"top_collective_k{top_k_collective}_zero"),
        baseline_delta,
    )
    signed_pos_dc, signed_pos_n = _dc_prefix_with_n(
        conditions, "top_collective_signedPOS_", baseline_delta
    )
    signed_neg_dc, signed_neg_n = _dc_prefix_with_n(
        conditions, "top_collective_signedNEG_", baseline_delta
    )
    matched_pos_dc, matched_pos_n = _dc_prefix_with_n(
        conditions, "ctrl_collective_matched_POS_", baseline_delta
    )
    matched_neg_dc, matched_neg_n = _dc_prefix_with_n(
        conditions, "ctrl_collective_matched_NEG_", baseline_delta
    )
    top_head_mean_dc, top_head_std_dc = _mean_std_finite(top_dc_list)
    ctrl_head_mean_dc, ctrl_head_std_dc = _mean_std_finite(ctrl_dc_list)

    return {
        "lang_key": lang_key,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "top_coll_k5_dc": top_coll_k5_dc,
        "ctrl_coll_k5_dc": ctrl_coll_k5_dc,
        "top_coll_k5_zero_dc": top_coll_k5_zero_dc,
        "signed_pos_dc": signed_pos_dc,
        "signed_neg_dc": signed_neg_dc,
        "matched_pos_dc": matched_pos_dc,
        "matched_neg_dc": matched_neg_dc,
        "signed_pos_n": signed_pos_n,
        "signed_neg_n": signed_neg_n,
        "matched_pos_n": matched_pos_n,
        "matched_neg_n": matched_neg_n,
        "top_head_mean_dc": top_head_mean_dc,
        "top_head_std_dc": top_head_std_dc,
        "ctrl_head_mean_dc": ctrl_head_mean_dc,
        "ctrl_head_std_dc": ctrl_head_std_dc,
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
        default="/projectnb/mcnet/jbrin/lang-probing/experiments/head_ablation_multiblimp/img",
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
    plot_cross_lang_sign_split(
        lang_rows, img_dir / "cross_lang_sign_split.png"
    )
    plot_cross_lang_mean_vs_zero(
        lang_rows, img_dir / "cross_lang_mean_vs_zero.png"
    )
    plot_cross_lang_per_head_summary(
        lang_rows, img_dir / "cross_lang_per_head_summary.png"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
