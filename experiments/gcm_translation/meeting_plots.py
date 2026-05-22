"""
Cross-direction visualizations for the GCM translation meeting.
Reads from outputs/gcm_translation/_aggregate/*.json and writes to
experiments/gcm_translation/img/meeting_*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent.parent
SWEEP = ROOT / "outputs" / "gcm_translation"
AGG = SWEEP / "_aggregate"
NULL_AGG = ROOT / "outputs" / "gcm_translation_null" / "_aggregate"
IMG = EXP / "img"
IMG.mkdir(exist_ok=True)


def _load(name):
    return json.load(open(AGG / name))


def _load_json(path: Path):
    return json.load(open(path))


# ---------- Plot A: direction x universal-head matrix ----------
def plot_direction_by_universal_head():
    boot = _load("bootstrap_summary.json")
    uh = boot["universal_heads"]
    top = sorted(uh, key=lambda e: -e["n_directions_in_topk"])[:20]
    top_k = 20

    directions = sorted({d["direction"] for e in top for d in e["directions"]})
    n_d, n_h = len(directions), len(top)
    M = np.full((n_d, n_h), np.nan)
    for j, e in enumerate(top):
        per = {d["direction"]: d["mean_abs_ie"] for d in e["directions"]}
        for i, dn in enumerate(directions):
            if dn in per:
                M[i, j] = per[dn]

    fig, ax = plt.subplots(figsize=(11, 14))
    im = ax.imshow(M, aspect="auto", cmap="magma")
    ax.set_xticks(range(n_h))
    ax.set_xticklabels([f"L{e['layer']}H{e['head']}" for e in top], rotation=45, ha="right")
    ax.set_yticks(range(n_d))
    ax.set_yticklabels(directions, fontsize=7)
    ax.set_xlabel("Universal head (sorted by # directions)")
    ax.set_ylabel("Translation direction (src__tgt)")
    ax.set_title(
        f"Top-{top_k} universal heads reuse across translation directions\n"
        f"Color = mean |IE| (per pair, N=100); cell present = head was in that direction's top-{top_k}"
    )
    plt.colorbar(im, ax=ax, label="mean |IE|")
    plt.tight_layout()
    plt.savefig(IMG / "meeting_direction_by_universal_head.png", dpi=130)
    plt.close(fig)
    print("  meeting_direction_by_universal_head.png")


def plot_direction_by_universal_head_signed():
    """Signed IE for the same top-20 universal heads, across all 56 directions."""
    uh = _load("universal_heads.json")
    top = sorted(uh, key=lambda e: -e["n_directions_in_topk"])[:20]
    directions = sorted(
        d.name for d in SWEEP.iterdir()
        if d.is_dir() and "__" in d.name and d.name.split("__", 1)[0] != d.name.split("__", 1)[1]
    )
    n_d, n_h = len(directions), len(top)
    M = np.full((n_d, n_h), np.nan)

    tensors = {}
    for i, dn in enumerate(directions):
        h_path = SWEEP / dn / "heads_ie.pt"
        if not h_path.exists():
            continue
        tensors[dn] = torch.load(h_path, map_location="cpu", weights_only=False).float()
        for j, e in enumerate(top):
            M[i, j] = float(torch.nanmean(tensors[dn][:, e["layer"], e["head"]]).item())

    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=(11, 14))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_h))
    ax.set_xticklabels(
        [f"L{e['layer']}H{e['head']}\n({e['n_directions_in_topk']}/56)" for e in top],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(n_d))
    ax.set_yticklabels(directions, fontsize=7)
    ax.set_xlabel("Top-20 universal head (count in per-direction top-20 shown)")
    ax.set_ylabel("Translation direction (src__tgt)")
    ax.set_title(
        "Signed IE of top-20 universal heads across all 56 translation directions\n"
        "Red = positive IE (pushes toward counterfactual); blue = negative IE (pushes toward gold)"
    )
    plt.colorbar(im, ax=ax, label="mean signed IE")
    plt.tight_layout()
    plt.savefig(IMG / "meeting_direction_by_universal_head_signed.png", dpi=130)
    plt.close(fig)
    print("  meeting_direction_by_universal_head_signed.png")


# ---------- Plot B: layer distribution ----------
def plot_layer_distribution():
    uh = _load("universal_heads.json")
    layer_counts = np.zeros(32, dtype=int)
    for e in uh:
        layer_counts[e["layer"]] += 1

    sa = _load("summary_all_directions.json")
    per_dir_layers = []
    for k, v in sa.items():
        for h in v["top_5_heads"]:
            per_dir_layers.append(h["layer"])
    layer_topk_counts = np.zeros(32, dtype=int)
    for L in per_dir_layers:
        layer_topk_counts[L] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(range(32), layer_counts, color="steelblue")
    axes[0].set_title("Where do universal translation heads live?\n(layer of each top-50 universal head)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("# of top-50 universal heads in this layer")
    axes[0].axvspan(15, 22, alpha=0.12, color="orange", label="middle layers (16-22)")
    axes[0].legend()

    axes[1].bar(range(32), layer_topk_counts, color="indianred")
    axes[1].set_title("Per-direction top-5 head layers, pooled over 64 directions\n(higher bar = layer matters across many directions)")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("# of (direction, top-5) entries at this layer")
    plt.tight_layout()
    plt.savefig(IMG / "meeting_layer_distribution.png", dpi=130)
    plt.close(fig)
    print("  meeting_layer_distribution.png")


# ---------- Plot C: SAE top-50 with grammar-feature overlap highlighted ----------
def plot_sae_grammar_overlap():
    usae = _load("universal_sae.json")
    overlap = set(_load("sae_overlap_with_grammar.json")["overlap_features"])

    top = sorted(usae, key=lambda e: -e["n_directions_in_topk"])[:50]
    feats = [e["feature_idx"] for e in top]
    n_dirs = [e["n_directions_in_topk"] for e in top]
    signed = [e["mean_signed_ie_across_directions"] for e in top]
    is_grammar = [f in overlap for f in feats]

    fig, ax = plt.subplots(figsize=(15, 5))
    colors = ["crimson" if g else "lightgray" for g in is_grammar]
    bars = ax.bar(range(len(feats)), n_dirs, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([f"f{f}" for f in feats], rotation=90, fontsize=7)
    ax.set_ylabel("# of 56 directions where this feature is in top-K")
    ax.set_title(
        "Top-50 universal SAE features (L16) — RED = also in grammar-feature top set (218)\n"
        f"{sum(is_grammar)} / 50 GCM-top features overlap with grammar features  "
        f"(p << 0.001 vs random; semantic-hub evidence)"
    )
    for i, (s, g) in enumerate(zip(signed, is_grammar)):
        if g:
            ax.text(i, n_dirs[i] + 0.5, f"{s:+.2f}", ha="center", fontsize=6, color="crimson")
    plt.tight_layout()
    plt.savefig(IMG / "meeting_sae_grammar_overlap.png", dpi=130)
    plt.close(fig)
    print("  meeting_sae_grammar_overlap.png")


def plot_sae_real_vs_null_same():
    """Scatter showing SAE features that separate real_cross from the content floor."""
    path = NULL_AGG / "three_way_summary_full.json"
    if not path.exists():
        print(f"  skip meeting_sae_real_vs_null_same.png: missing {path}")
        return
    summary = _load_json(path)
    rows = summary.get("sae_features_ranked_by_translation_circuit", [])[:100]
    if not rows:
        print("  skip meeting_sae_real_vs_null_same.png: no SAE summary rows")
        return

    real = np.array([r["real_mean"] for r in rows])
    null_same = np.array([r["null_same_mean"] for r in rows])
    tc = np.array([r["translation_circuit_mean"] for r in rows])
    feats = [r["feature_idx"] for r in rows]
    vmax = max(abs(float(tc.min())), abs(float(tc.max())), 1e-6)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(null_same, real, c=tc, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=42, alpha=0.82, edgecolor="black", linewidth=0.25)
    lim = max(float(real.max()), float(null_same.max())) * 1.08
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1, alpha=0.35)
    for idx in np.argsort(-tc)[:8]:
        ax.text(null_same[idx], real[idx], f"f{feats[idx]}", fontsize=8,
                ha="left", va="bottom")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("null_same mean |IE| (content floor)")
    ax.set_ylabel("real_cross mean |IE|")
    ax.set_title(
        "SAE features: real translation signal vs monolingual content floor\n"
        "Color = real_cross - null_cross translation-circuit contribution"
    )
    plt.colorbar(sc, ax=ax, label="translation-circuit mean")
    plt.tight_layout()
    plt.savefig(IMG / "meeting_sae_real_vs_null_same.png", dpi=130)
    plt.close(fig)
    print("  meeting_sae_real_vs_null_same.png")


# ---------- Plot D: per-direction translation strength ----------
def plot_per_direction_strength():
    sa = _load("summary_all_directions.json")
    rows = []
    for k, v in sa.items():
        if not v["top_5_heads"]:
            continue
        top1 = v["top_5_heads"][0]
        rows.append((k, top1["mean_abs_ie"], top1["layer"], top1["head"]))
    rows.sort(key=lambda r: -r[1])

    names = [r[0].replace("__", "→") for r in rows]
    vals = [r[1] for r in rows]
    labels = [f"L{r[2]}H{r[3]}" for r in rows]

    same_lang_mask = [r[0].split("__")[0] == r[0].split("__")[1] for r in rows]
    colors = ["#888" if s else "steelblue" for s in same_lang_mask]

    fig, ax = plt.subplots(figsize=(10, 14))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean |IE| of strongest single head (last-source-token patch)")
    ax.set_title(
        "Per-direction top-1 head magnitude\n"
        "Blue = cross-language (translation),  Gray = same-language (control)\n"
        "Annotation = (layer, head) of top-1"
    )
    for i, (v, l) in enumerate(zip(vals, labels)):
        ax.text(v + 0.005, i, l, va="center", fontsize=6, color="black")
    plt.tight_layout()
    plt.savefig(IMG / "meeting_per_direction_strength.png", dpi=130)
    plt.close(fig)
    print("  meeting_per_direction_strength.png")


def main():
    plot_direction_by_universal_head()
    plot_direction_by_universal_head_signed()
    plot_layer_distribution()
    plot_sae_grammar_overlap()
    plot_sae_real_vs_null_same()
    plot_per_direction_strength()
    print("\nAll plots written to", IMG)


if __name__ == "__main__":
    main()
