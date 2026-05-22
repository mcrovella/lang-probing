"""Summarize per-direction head-IE sparsity.

The paper draft mentions that top translation heads are far above the median
head. This script makes that claim reproducible from the raw ``heads_ie.pt``
tensors.

Outputs:
  outputs/gcm_translation/_aggregate/head_ie_sparsity_summary.csv
  experiments/gcm_translation/img/head_ie_top_median_ratio.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def summarize_direction(direction_dir: Path) -> dict | None:
    path = direction_dir / "heads_ie.pt"
    if not path.exists():
        return None
    heads = torch.load(path, map_location="cpu", weights_only=True).float()
    keep = ~torch.isnan(heads).any(dim=(1, 2))
    heads = heads[keep]
    if heads.numel() == 0:
        return None

    mean_abs = heads.abs().mean(dim=0)
    flat = mean_abs.flatten()
    top_val, top_idx = flat.max(dim=0)
    median_val = flat.median()
    top20_mean = flat.topk(min(20, flat.numel())).values.mean()
    n_heads = mean_abs.shape[1]

    return {
        "direction": direction_dir.name,
        "n_pairs": int(heads.shape[0]),
        "top_layer": int(top_idx.item() // n_heads),
        "top_head": int(top_idx.item() % n_heads),
        "top_mean_abs_ie": float(top_val.item()),
        "median_head_mean_abs_ie": float(median_val.item()),
        "top_over_median": float((top_val / median_val).item()) if median_val.item() else float("nan"),
        "top20_mean_abs_ie": float(top20_mean.item()),
        "top20_mean_over_median": float((top20_mean / median_val).item()) if median_val.item() else float("nan"),
    }


def plot_summary(df: pd.DataFrame, out_path: Path) -> None:
    df = df.sort_values("top_over_median", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(8, 0.15 * len(df))))
    ax.barh(df["direction"].str.replace("__", "->"), df["top_over_median"], color="steelblue")
    ax.set_xlabel("Top head mean |IE| / median head mean |IE|")
    ax.set_ylabel("Translation direction")
    ax.set_title("Per-direction head-IE sparsity (top head vs median head)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", default="outputs/gcm_translation")
    parser.add_argument("--out_dir", default="outputs/gcm_translation/_aggregate")
    parser.add_argument("--img_dir", default="experiments/gcm_translation/img")
    parser.add_argument("--include_same_lang", action="store_true")
    args = parser.parse_args()

    rows = []
    for d in sorted(Path(args.sweep_dir).iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        src, tgt = d.name.split("__", 1)
        if src == tgt and not args.include_same_lang:
            continue
        row = summarize_direction(d)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("No directions with heads_ie.pt found.")

    df = pd.DataFrame(rows).sort_values("direction")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "head_ie_sparsity_summary.csv"
    df.to_csv(csv_path, index=False)
    plot_summary(df, Path(args.img_dir) / "head_ie_top_median_ratio.png")

    print(f"Wrote {csv_path}")
    print(
        "top_over_median: "
        f"min={df['top_over_median'].min():.1f} "
        f"median={df['top_over_median'].median():.1f} "
        f"max={df['top_over_median'].max():.1f}"
    )


if __name__ == "__main__":
    main()
