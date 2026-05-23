"""
Plotting for GCM translation attribution.

Per-direction (one file per (src, tgt), saved into a subdir per plot type):
  - heads_heatmap/<src>__<tgt>_heads_heatmap.png               — [L, H] mean-abs IE
  - heads_signed_heatmap/<src>__<tgt>_heads_signed_heatmap.png — [L, H] signed mean IE
  - top_heads_bar/<src>__<tgt>_top_heads_bar.png               — top-20 heads, error bars
  - top_sae_bar/<src>__<tgt>_top_sae_bar.png                   — top-50 SAE features

Cross-direction / global (run after analyze.py — saved at the root of img/):
  - universal_heads_heatmap.png  — n_directions_in_topk per (layer, head)

Output convention: figures are saved under the experiment-local
`experiments/gcm_translation/img/` (not the repo-level `img/`) so the
experiment folder is self-contained and portable. Per-direction plots
are bucketed into per-plot-type subdirs to keep the directory navigable
when sweeping many language pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_IMG_DIR = EXP_DIR / "img"
import torch


PER_DIRECTION_SUBDIRS = (
    "heads_heatmap",
    "heads_signed_heatmap",
    "top_heads_bar",
    "top_sae_bar",
)


def plot_direction(direction_dir: Path, img_dir: Path):
    name = direction_dir.name
    for sub in PER_DIRECTION_SUBDIRS:
        (img_dir / sub).mkdir(parents=True, exist_ok=True)

    heads_path = direction_dir / "heads_ie.pt"
    sae_path = direction_dir / "sae_ie.pt"
    top_path = direction_dir / "top_rankings.json"

    if heads_path.exists():
        heads = torch.load(heads_path, map_location="cpu").float()  # [N, L, H]
        mean_abs = heads.abs().mean(dim=0).numpy()                   # [L, H]

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(mean_abs, aspect="auto", cmap="magma")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{name}  mean |IE|  (N={heads.shape[0]} pairs)")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(img_dir / "heads_heatmap" / f"{name}_heads_heatmap.png", dpi=120)
        plt.close(fig)

        # signed mean for direction info
        mean_signed = heads.mean(dim=0).numpy()
        fig, ax = plt.subplots(figsize=(12, 8))
        v = max(abs(mean_signed.min()), abs(mean_signed.max()))
        im = ax.imshow(mean_signed, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=+v)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{name}  signed mean IE  (red=favors original/correct)")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(img_dir / "heads_signed_heatmap" / f"{name}_heads_signed_heatmap.png", dpi=120)
        plt.close(fig)

    if top_path.exists():
        with open(top_path) as f:
            top = json.load(f)

        # Top-20 heads bar with error bars
        if "heads_top_k_by_mean_abs_ie" in top and heads_path.exists():
            tops = top["heads_top_k_by_mean_abs_ie"][:20]
            labels = [f"L{e['layer']}H{e['head']}" for e in tops]
            means = np.array([e["mean_abs_ie"] for e in tops])
            # Std across pairs
            stds = []
            for e in tops:
                hh = heads[:, e["layer"], e["head"]].abs().numpy()
                stds.append(hh.std())
            stds = np.array(stds)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(range(len(labels)), means, yerr=stds, color="steelblue", alpha=0.85)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("mean |IE| (+- std across pairs)")
            ax.set_title(f"{name}  top-20 heads")
            plt.tight_layout()
            plt.savefig(img_dir / "top_heads_bar" / f"{name}_top_heads_bar.png", dpi=120)
            plt.close(fig)

        # Top-50 SAE bar
        if "sae_top_k_by_mean_abs_ie" in top:
            tops = top["sae_top_k_by_mean_abs_ie"][:50]
            labels = [f"f{e['feature_idx']}" for e in tops]
            means = np.array([e["mean_abs_ie"] for e in tops])
            colors = ["crimson" if e["mean_signed_ie"] > 0 else "navy" for e in tops]
            fig, ax = plt.subplots(figsize=(16, 4))
            ax.bar(range(len(labels)), means, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_ylabel("mean |IE|")
            ax.set_title(f"{name}  top-50 SAE features  (red=favors original, blue=favors counterfactual)")
            plt.tight_layout()
            plt.savefig(img_dir / "top_sae_bar" / f"{name}_top_sae_bar.png", dpi=120)
            plt.close(fig)

    print(f"  plotted {name}")


def plot_universality(aggregate_dir: Path, img_dir: Path, n_layers: int = 32, n_heads: int = 32):
    img_dir.mkdir(parents=True, exist_ok=True)

    uh_path = aggregate_dir / "universal_heads.json"
    if uh_path.exists():
        with open(uh_path) as f:
            uh = json.load(f)
        grid = np.zeros((n_layers, n_heads), dtype=np.float32)
        for entry in uh:
            grid[entry["layer"], entry["head"]] = entry["n_directions_in_topk"]
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title("Universal translation heads: # directions where this head is in top-K")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(img_dir / "universal_heads_heatmap.png", dpi=120)
        plt.close(fig)
        print("  plotted universal_heads_heatmap.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default="outputs/gcm_translation")
    p.add_argument("--img_dir", default=str(DEFAULT_IMG_DIR))
    p.add_argument("--per_direction", action="store_true", default=True)
    p.add_argument("--universality", action="store_true", default=True)
    args = p.parse_args()

    sweep = Path(args.sweep_dir)
    img = Path(args.img_dir)

    if args.per_direction:
        for d in sorted(sweep.iterdir()):
            if d.is_dir() and "__" in d.name:
                plot_direction(d, img)

    if args.universality:
        agg = sweep / "_aggregate"
        if agg.exists():
            plot_universality(agg, img)
        else:
            print(f"  no aggregate dir at {agg}; run analyze.py first")


if __name__ == "__main__":
    main()
