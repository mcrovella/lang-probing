"""Recover rank-1 BLEU latents and regress them against FLORES perplexity.

Motivation: the rank-1 SVD of the BLEU matrix is 88% faithful for Llama and
82% for Aya, while a linear-in-PPL model achieves R^2 of only ~0.02 / ~0.10.
This contrast suggests that BLEU is well-approximated by a per-language
src-competence * tgt-competence product, but that FLORES perplexity is a
poor proxy for those competence latents. This script tests that directly.

Pipeline:

1. Build the (src, tgt) BLEU matrix from `data/pair_metrics.csv`.
2. Impute the NaN diagonal with column means (same convention as
   `rank1_approximation.py`) so SVD is well-defined.
3. Run SVD; extract rank-1 src and target latents
       u_s = U[:,0] * sqrt(sigma_1),    v_t = V[:,0] * sqrt(sigma_1).
   Apply a sign convention so that both vectors correlate positively with
   row/column mean BLEU (SVD signs are otherwise arbitrary).
4. For each language, pull FLORES perplexity from `pair_metrics.csv`.
5. Compute Pearson and Spearman correlation between
       u_s   vs   { P_s, log P_s, 1/P_s, 1/sqrt(P_s) }
       v_t   vs   { P_t, log P_t, 1/P_t, 1/sqrt(P_t) }
6. Save the correlation table and per-transform scatter plots.

Outputs:
    outputs/perplexity_bleu_linear/rank1_latent_regression/
        latent_vs_ppl_correlations.csv
        latents_{model}.csv
    img/perplexity_bleu_linear/
        rank1_latent_vs_ppl_{model}.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_CSV = REPO_ROOT / "data" / "pair_metrics.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "rank1_latent_regression"
DEFAULT_IMG_DIR = REPO_ROOT / "img" / "perplexity_bleu_linear"


TRANSFORMS = {
    "raw": (lambda p: p, "P"),
    "log": (lambda p: np.log(p), "log P"),
    "inv": (lambda p: 1.0 / p, "1/P"),
    "inv_sqrt": (lambda p: 1.0 / np.sqrt(p), "1/sqrt(P)"),
}


def build_bleu_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Pivot to a square (src, tgt) matrix with a shared, sorted language axis."""
    langs = sorted(set(df["src"]) | set(df["tgt"]))
    pivot = (
        df.pivot_table(index="src", columns="tgt", values="bleu", aggfunc="mean")
        .reindex(index=langs, columns=langs)
    )
    return pivot.values, langs


def impute_column_means(M: np.ndarray) -> Tuple[np.ndarray, int]:
    """Fill NaN cells with column means (global mean as a backstop)."""
    col_means = np.nanmean(M, axis=0)
    global_mean = float(np.nanmean(M))
    col_means = np.where(np.isnan(col_means), global_mean, col_means)
    nan_idx = np.where(np.isnan(M))
    M_filled = M.copy()
    M_filled[nan_idx] = np.take(col_means, nan_idx[1])
    return M_filled, len(nan_idx[0])


def rank1_latents(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (u, v, sigma_1) with u, v absorbing sqrt(sigma_1) for symmetry."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    sigma1 = float(S[0])
    sqrt_s1 = np.sqrt(sigma1)
    u = U[:, 0] * sqrt_s1
    v = Vt[0, :] * sqrt_s1
    return u, v, sigma1


def sign_correct(u: np.ndarray, v: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flip signs so u, v are positively correlated with their row/col mean BLEU."""
    row_mean = np.nanmean(M, axis=1)
    col_mean = np.nanmean(M, axis=0)
    if np.corrcoef(u, row_mean)[0, 1] < 0:
        u = -u
        v = -v
    if np.corrcoef(v, col_mean)[0, 1] < 0:
        # Flipping both keeps the outer product u v^T invariant, so we can only flip
        # once. If after the row-mean flip v still disagrees with col_mean, that
        # means the rank-1 component genuinely has opposite-sign tgt latents.
        pass
    return u, v


def per_language_perplexity(df: pd.DataFrame, role: str) -> Dict[str, float]:
    """Collapse pair-level PPL to one value per language (src or tgt)."""
    col = "src_flores_perplexity" if role == "src" else "tgt_flores_perplexity"
    key = "src" if role == "src" else "tgt"
    sub = df[[key, col]].dropna()
    return sub.groupby(key)[col].mean().to_dict()


def correlate(latent: np.ndarray, ppl: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Pearson and Spearman correlations for each transform of ppl."""
    out: Dict[str, Dict[str, float]] = {}
    for name, (fn, _) in TRANSFORMS.items():
        t = fn(ppl)
        mask = np.isfinite(latent) & np.isfinite(t)
        if mask.sum() < 3:
            out[name] = {"pearson_r": float("nan"), "pearson_p": float("nan"),
                         "spearman_r": float("nan"), "spearman_p": float("nan"),
                         "n": int(mask.sum())}
            continue
        pr, pp = stats.pearsonr(latent[mask], t[mask])
        sr, sp = stats.spearmanr(latent[mask], t[mask])
        out[name] = {"pearson_r": float(pr), "pearson_p": float(pp),
                     "spearman_r": float(sr), "spearman_p": float(sp),
                     "n": int(mask.sum())}
    return out


def plot_grid(model: str,
              u: np.ndarray, v: np.ndarray,
              p_src: np.ndarray, p_tgt: np.ndarray,
              langs: List[str], save_path: Path) -> None:
    fig, axes = plt.subplots(2, len(TRANSFORMS), figsize=(4 * len(TRANSFORMS), 8))
    for col, (name, (fn, label)) in enumerate(TRANSFORMS.items()):
        # source row
        ax = axes[0, col]
        t = fn(p_src)
        mask = np.isfinite(u) & np.isfinite(t)
        if mask.sum() >= 3:
            r, _ = stats.pearsonr(u[mask], t[mask])
            ax.scatter(t[mask], u[mask], alpha=0.7)
            ax.set_title(f"u_s vs {label}\nPearson r={r:.2f}", fontsize=10)
        else:
            ax.set_title(f"u_s vs {label}\n(n<3)", fontsize=10)
        ax.set_xlabel(label)
        ax.set_ylabel("u_s (src latent)")

        # target row
        ax = axes[1, col]
        t = fn(p_tgt)
        mask = np.isfinite(v) & np.isfinite(t)
        if mask.sum() >= 3:
            r, _ = stats.pearsonr(v[mask], t[mask])
            ax.scatter(t[mask], v[mask], alpha=0.7, color="tab:orange")
            ax.set_title(f"v_t vs {label}\nPearson r={r:.2f}", fontsize=10)
        else:
            ax.set_title(f"v_t vs {label}\n(n<3)", fontsize=10)
        ax.set_xlabel(label)
        ax.set_ylabel("v_t (tgt latent)")

    fig.suptitle(f"Rank-1 BLEU latents vs FLORES perplexity transforms ({model})", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_model(df_model: pd.DataFrame, model: str,
                  out_dir: Path, img_dir: Path) -> List[Dict]:
    M, langs = build_bleu_matrix(df_model)
    M_filled, n_imputed = impute_column_means(M)
    u, v, sigma1 = rank1_latents(M_filled)
    u, v = sign_correct(u, v, M)

    p_src_map = per_language_perplexity(df_model, "src")
    p_tgt_map = per_language_perplexity(df_model, "tgt")
    p_src = np.array([p_src_map.get(l, np.nan) for l in langs], dtype=float)
    p_tgt = np.array([p_tgt_map.get(l, np.nan) for l in langs], dtype=float)

    # Save the recovered latents alongside the per-lang PPL.
    latents_df = pd.DataFrame({
        "model": model,
        "lang": langs,
        "u_src_latent": u,
        "v_tgt_latent": v,
        "src_flores_perplexity": p_src,
        "tgt_flores_perplexity": p_tgt,
    })
    latents_df.to_csv(out_dir / f"latents_{model}.csv", index=False)

    src_corrs = correlate(u, p_src)
    tgt_corrs = correlate(v, p_tgt)

    rows: List[Dict] = []
    for role, corrs in (("src", src_corrs), ("tgt", tgt_corrs)):
        for transform, stat in corrs.items():
            rows.append({
                "model": model,
                "role": role,
                "transform": transform,
                **stat,
            })

    plot_grid(model, u, v, p_src, p_tgt, langs,
              img_dir / f"rank1_latent_vs_ppl_{model}.png")

    print(f"\n[{model}] sigma_1 = {sigma1:.3f}, n_imputed_cells = {n_imputed}")
    print(f"[{model}] src-latent vs PPL correlations:")
    for transform, stat in src_corrs.items():
        print(f"  {transform:9s}  Pearson r = {stat['pearson_r']:+.3f} "
              f"(p={stat['pearson_p']:.3g})   Spearman = {stat['spearman_r']:+.3f}")
    print(f"[{model}] tgt-latent vs PPL correlations:")
    for transform, stat in tgt_corrs.items():
        print(f"  {transform:9s}  Pearson r = {stat['pearson_r']:+.3f} "
              f"(p={stat['pearson_p']:.3g})   Spearman = {stat['spearman_r']:+.3f}")

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.img_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pair_csv)
    all_rows: List[Dict] = []
    for model in sorted(pairs["model"].unique()):
        df_model = pairs[pairs["model"] == model]
        all_rows.extend(analyze_model(df_model, model, args.out_dir, args.img_dir))

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.out_dir / "latent_vs_ppl_correlations.csv", index=False)
    print(f"\nWrote: {args.out_dir/'latent_vs_ppl_correlations.csv'}")
    print(f"Wrote: {args.img_dir}/rank1_latent_vs_ppl_*.png")


if __name__ == "__main__":
    main()
