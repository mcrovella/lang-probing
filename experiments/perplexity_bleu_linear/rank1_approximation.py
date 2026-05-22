"""Rank-1 SVD approximation of the BLEU matrix.

The paper claims "rank-1 approximation is 88% faithful" for Llama. This
script reconstructs that analysis:

1. Load the joined BLEU/PPL CSV.
2. Pivot BLEU into a (src, tgt) matrix.
3. SVD; compute the rank-k approximation's faithfulness = 1 - ||M - M_k||_F / ||M||_F.
4. Plot faithfulness vs rank (fig `linear_effects_ranks.png`).
5. Scatter rank-1 predicted vs actual BLEU (fig `linear_effects_{model}.png`).

Faithfulness here is the Frobenius-norm-ratio metric, equivalently the
fraction of matrix energy captured by the rank-k components.

Run:
    python experiments/perplexity_bleu_linear/rank1_approximation.py \\
        --model llama \\
        --output_dir outputs/perplexity_bleu_linear/bleu_and_ppl/rank1 \\
        --img_dir img/perplexity_bleu_linear

Data source: outputs/perplexity_bleu_linear/bleu_and_ppl/combined_results_{model}.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lang_probing_src.config import OUTPUTS_DIR, IMG_DIR


logger = logging.getLogger(__name__)


def load_bleu_matrix(model: str, data_dir: Path):
    """Load combined_results CSV and pivot to (src, tgt) BLEU matrix.

    Returns (matrix, src_labels, tgt_labels).
    """
    path = data_dir / f"combined_results_{model}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    df = pd.read_csv(path)
    pivot = df.pivot(index="src", columns="tgt", values="bleu")
    return pivot.values, list(pivot.index), list(pivot.columns)


def fill_matrix_for_svd(M: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return an SVD-ready matrix and a short record of any imputation."""
    info = {"n_nan": int(np.isnan(M).sum()), "imputation": "none"}
    if not np.isnan(M).any():
        return M, info

    col_means = np.nanmean(M, axis=0)
    global_mean = float(np.nanmean(M))
    col_means = np.where(np.isnan(col_means), global_mean, col_means)
    inds = np.where(np.isnan(M))
    M_filled = M.copy()
    M_filled[inds] = np.take(col_means, inds[1])
    info["imputation"] = "column_mean"
    info["n_imputed"] = int(len(inds[0]))
    return M_filled, info


def rank_k_approximation(M: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation via truncated SVD."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def faithfulness_at_rank(M: np.ndarray, k: int) -> float:
    """1 - ||M - M_k||_F / ||M||_F, in [0, 1]."""
    M_k = rank_k_approximation(M, k)
    num = np.linalg.norm(M - M_k, ord="fro")
    den = np.linalg.norm(M, ord="fro")
    return float(1.0 - num / den) if den > 0 else float("nan")


def plot_faithfulness_vs_rank(M: np.ndarray, model: str, save_path: Path) -> np.ndarray:
    """Plot the error-vs-rank curve and return the faithfulness vector."""
    ranks = np.arange(1, min(M.shape) + 1)
    faith = np.array([faithfulness_at_rank(M, int(k)) for k in ranks])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, 1.0 - faith, "o-", label="relative error $1 - f_k$")
    ax.axhline(1.0 - faith[0], color="tab:red", linestyle="--", alpha=0.5,
               label=f"rank-1 residual = {1.0 - faith[0]:.3f}")
    ax.set_xlabel("Rank $k$ of SVD approximation")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title(f"BLEU matrix: relative error vs rank ({model})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", save_path)
    return faith


def plot_rank1_vs_actual(M: np.ndarray, model: str, faith1: float,
                         save_path: Path) -> None:
    """Scatter: rank-1 predicted vs actual BLEU."""
    M1 = rank_k_approximation(M, 1)
    mask = ~np.isnan(M) & ~np.isnan(M1)
    actual = M[mask]
    predicted = M1[mask]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, predicted, alpha=0.5, s=20)
    lim = (float(np.nanmin([actual.min(), predicted.min()])) - 1,
           float(np.nanmax([actual.max(), predicted.max()])) + 1)
    ax.plot(lim, lim, color="tab:red", linestyle="--", alpha=0.7, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual BLEU")
    ax.set_ylabel("Rank-1 predicted BLEU")
    ax.set_title(f"Rank-1 SVD: faithfulness = {faith1 * 100:.1f}% ({model})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", save_path)


def _matrix_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("#f2f2f2")
    return cmap


def plot_bleu_heatmap(
    M: np.ndarray,
    src_labels: list[str],
    tgt_labels: list[str],
    model: str,
    save_path: Path,
) -> None:
    """Raw BLEU src x tgt grid from the same pivot used by SVD."""
    masked = np.ma.masked_invalid(M)
    fig_w = max(7.0, 0.45 * len(tgt_labels))
    fig_h = max(6.0, 0.38 * len(src_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(masked, aspect="auto", cmap=_matrix_cmap("viridis"))
    ax.set_xticks(range(len(tgt_labels)))
    ax.set_xticklabels(tgt_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(src_labels)))
    ax.set_yticklabels(src_labels, fontsize=8)
    ax.set_xlabel("Target language")
    ax.set_ylabel("Source language")
    ax.set_title(f"Raw BLEU matrix ({model}): {len(src_labels)} x {len(tgt_labels)}")
    plt.colorbar(im, ax=ax, label="BLEU")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", save_path)


def plot_actual_vs_rank1_heatmaps(
    M_actual: np.ndarray,
    M_filled: np.ndarray,
    src_labels: list[str],
    tgt_labels: list[str],
    model: str,
    faith1: float,
    save_path: Path,
) -> None:
    """Side-by-side actual BLEU and rank-1 reconstruction heatmaps."""
    M1 = rank_k_approximation(M_filled, 1)
    vmin = float(np.nanmin([np.nanmin(M_actual), np.nanmin(M1)]))
    vmax = float(np.nanmax([np.nanmax(M_actual), np.nanmax(M1)]))

    fig_w = max(12.0, 0.8 * len(tgt_labels))
    fig_h = max(6.0, 0.38 * len(src_labels))
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)
    mats = [
        (np.ma.masked_invalid(M_actual), "Actual BLEU"),
        (M1, f"Rank-1 reconstruction ({faith1 * 100:.1f}% faithful)"),
    ]
    for ax, (mat, title) in zip(axes, mats):
        im = ax.imshow(mat, aspect="auto", cmap=_matrix_cmap("viridis"), vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(tgt_labels)))
        ax.set_xticklabels(tgt_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Target")
        ax.set_title(title)
    axes[0].set_yticks(range(len(src_labels)))
    axes[0].set_yticklabels(src_labels, fontsize=7)
    axes[0].set_ylabel("Source")
    fig.suptitle(
        f"BLEU matrix and rank-1 SVD ({model}; {len(src_labels)} x {len(tgt_labels)})",
        y=1.02,
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), label="BLEU", shrink=0.84)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", save_path)


def plot_residual_heatmap(
    M_actual: np.ndarray,
    M_filled: np.ndarray,
    src_labels: list[str],
    tgt_labels: list[str],
    model: str,
    save_path: Path,
) -> None:
    """Actual minus rank-1 predicted BLEU heatmap."""
    M1 = rank_k_approximation(M_filled, 1)
    residual = M_filled - M1
    vmax = float(np.nanmax(np.abs(residual)))

    fig_w = max(7.0, 0.45 * len(tgt_labels))
    fig_h = max(6.0, 0.38 * len(src_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(residual, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(tgt_labels)))
    ax.set_xticklabels(tgt_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(src_labels)))
    ax.set_yticklabels(src_labels, fontsize=8)
    ax.set_xlabel("Target language")
    ax.set_ylabel("Source language")
    ax.set_title(f"Rank-1 residuals: actual - predicted BLEU ({model})")
    plt.colorbar(im, ax=ax, label="BLEU residual")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", save_path)


def leave_one_language_out_summary(
    M_filled: np.ndarray,
    src_labels: list[str],
    tgt_labels: list[str],
) -> pd.DataFrame:
    """Drop each language from both source and target axes, then recompute rank-1 faithfulness."""
    rows = []
    shared = sorted(set(src_labels) & set(tgt_labels))
    for lang in shared:
        src_keep = [i for i, label in enumerate(src_labels) if label != lang]
        tgt_keep = [j for j, label in enumerate(tgt_labels) if label != lang]
        sub = M_filled[np.ix_(src_keep, tgt_keep)]
        rows.append({
            "dropped_language": lang,
            "n_src": len(src_keep),
            "n_tgt": len(tgt_keep),
            "rank1_faithfulness": faithfulness_at_rank(sub, 1),
        })
    return pd.DataFrame(rows).sort_values("rank1_faithfulness", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["llama", "aya"], default="llama")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(OUTPUTS_DIR) / "perplexity_bleu_linear" / "bleu_and_ppl",
        help="Dir containing combined_results_{model}.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(OUTPUTS_DIR) / "perplexity_bleu_linear" / "bleu_and_ppl" / "rank1",
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        default=Path(IMG_DIR) / "perplexity_bleu_linear",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.img_dir.mkdir(parents=True, exist_ok=True)

    M, src_labels, tgt_labels = load_bleu_matrix(args.model, args.data_dir)
    logger.info("BLEU matrix: %s (nan count=%d)",
                M.shape, int(np.isnan(M).sum()))

    # SVD requires a complete matrix.
    M_filled, imputation = fill_matrix_for_svd(M)
    if imputation["n_nan"]:
        logger.info("Imputed %d NaN cells with %s for SVD",
                    imputation.get("n_imputed", 0), imputation["imputation"])

    # Save singular values
    U, S, Vt = np.linalg.svd(M_filled, full_matrices=False)
    pd.DataFrame({"rank": np.arange(1, len(S) + 1), "singular_value": S}).to_csv(
        args.output_dir / f"singular_values_{args.model}.csv", index=False
    )

    # Faithfulness curve + plot
    faith = plot_faithfulness_vs_rank(
        M_filled, args.model, args.img_dir / f"linear_effects_ranks_{args.model}.png"
    )

    # Rank-1 predicted vs actual scatter
    plot_rank1_vs_actual(
        M_filled, args.model, float(faith[0]),
        args.img_dir / f"linear_effects_{args.model}.png",
    )

    # Raw matrix + rank-1 matrix figures from the exact same pivot.
    plot_bleu_heatmap(
        M, src_labels, tgt_labels, args.model,
        args.img_dir / f"bleu_matrix_{args.model}.png",
    )
    plot_actual_vs_rank1_heatmaps(
        M, M_filled, src_labels, tgt_labels, args.model, float(faith[0]),
        args.img_dir / f"bleu_matrix_rank1_{args.model}.png",
    )
    plot_residual_heatmap(
        M, M_filled, src_labels, tgt_labels, args.model,
        args.img_dir / f"rank1_residual_heatmap_{args.model}.png",
    )

    # CSV summary
    summary = pd.DataFrame({
        "rank": np.arange(1, len(faith) + 1),
        "faithfulness": faith,
        "relative_error": 1.0 - faith,
    })
    summary.to_csv(args.output_dir / f"faithfulness_{args.model}.csv", index=False)

    lolo = leave_one_language_out_summary(M_filled, src_labels, tgt_labels)
    lolo.to_csv(args.output_dir / f"leave_one_language_out_{args.model}.csv", index=False)

    matrix_summary = {
        "model": args.model,
        "data_csv": str(args.data_dir / f"combined_results_{args.model}.csv"),
        "n_src": len(src_labels),
        "n_tgt": len(tgt_labels),
        "n_cells": int(np.prod(M.shape)),
        "n_nan": imputation["n_nan"],
        "imputation": imputation,
        "rank1_faithfulness": float(faith[0]),
        "src_labels": src_labels,
        "tgt_labels": tgt_labels,
    }
    with open(args.output_dir / f"matrix_summary_{args.model}.json", "w") as f:
        json.dump(matrix_summary, f, indent=2)

    print(f"\n=== Rank-1 faithfulness for {args.model}: {faith[0] * 100:.2f}% ===")
    print(f"    matrix shape: {len(src_labels)} x {len(tgt_labels)}")
    print(f"    (1 - ||M - M_1||_F / ||M||_F)")
    print(f"Top-5 singular values: {S[:5].round(3)}")
    print(f"\nSaved:")
    print(f"  img:  {args.img_dir / f'linear_effects_ranks_{args.model}.png'}")
    print(f"  img:  {args.img_dir / f'linear_effects_{args.model}.png'}")
    print(f"  img:  {args.img_dir / f'bleu_matrix_{args.model}.png'}")
    print(f"  img:  {args.img_dir / f'bleu_matrix_rank1_{args.model}.png'}")
    print(f"  img:  {args.img_dir / f'rank1_residual_heatmap_{args.model}.png'}")
    print(f"  csv:  {args.output_dir / f'faithfulness_{args.model}.csv'}")
    print(f"  csv:  {args.output_dir / f'singular_values_{args.model}.csv'}")
    print(f"  csv:  {args.output_dir / f'leave_one_language_out_{args.model}.csv'}")
    print(f"  json: {args.output_dir / f'matrix_summary_{args.model}.json'}")


if __name__ == "__main__":
    main()
