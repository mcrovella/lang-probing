"""Correlate additive-model per-language effects (alpha, beta) with FLORES PPL.

This is Experiment 2 redone with the additive-RE coefficients in place of
the rank-1 SVD latents. We expect the alphas and betas to be more
interpretable competence factors than the SVD vectors because they are
explicitly the per-language deviations above/below the grand-mean BLEU.

Scatter plots are annotated with language codes so outliers can be
identified visually.

Outputs:
    outputs/perplexity_bleu_linear/alpha_beta_vs_ppl/
        correlations.csv
    img/perplexity_bleu_linear/
        alpha_beta_vs_ppl_{model}.png   # 2x4 grid (src/tgt x 4 transforms),
                                         #   each point labeled by lang code
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_CSV = REPO_ROOT / "data" / "pair_metrics.csv"
DEFAULT_COEF_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "additive_random_effects"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "alpha_beta_vs_ppl"
DEFAULT_IMG_DIR = REPO_ROOT / "img" / "perplexity_bleu_linear"


TRANSFORMS = {
    "raw": (lambda p: p, "$P$"),
    "log": (lambda p: np.log(p), "$\\log P$"),
    "inv": (lambda p: 1.0 / p, "$1/P$"),
    "inv_sqrt": (lambda p: 1.0 / np.sqrt(p), "$1/\\sqrt{P}$"),
}


def per_language_perplexity(df: pd.DataFrame, role: str) -> Dict[str, float]:
    col = "src_flores_perplexity" if role == "src" else "tgt_flores_perplexity"
    key = "src" if role == "src" else "tgt"
    sub = df[[key, col]].dropna()
    return sub.groupby(key)[col].mean().to_dict()


def correlate(latent: np.ndarray, ppl: np.ndarray, transform_fn) -> Dict:
    t = transform_fn(ppl)
    mask = np.isfinite(latent) & np.isfinite(t)
    n = int(mask.sum())
    if n < 3:
        return {"n": n, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(latent[mask], t[mask])
    sr, sp = stats.spearmanr(latent[mask], t[mask])
    return {"n": n, "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


def plot_annotated(model: str, langs: List[str],
                   alpha: np.ndarray, beta: np.ndarray,
                   p_src: np.ndarray, p_tgt: np.ndarray,
                   save_path: Path) -> None:
    fig, axes = plt.subplots(2, len(TRANSFORMS), figsize=(4.5 * len(TRANSFORMS), 9))
    for col_i, (name, (fn, label)) in enumerate(TRANSFORMS.items()):
        # source row
        ax = axes[0, col_i]
        t = fn(p_src)
        mask = np.isfinite(alpha) & np.isfinite(t)
        ax.scatter(t[mask], alpha[mask], alpha=0.7, s=20)
        for k in np.where(mask)[0]:
            ax.annotate(langs[k], (t[k], alpha[k]), fontsize=7, alpha=0.75,
                        xytext=(3, 2), textcoords="offset points")
        if mask.sum() >= 3:
            r, p = stats.pearsonr(alpha[mask], t[mask])
            ax.set_title(f"$\\alpha_s$ vs {label}\nPearson r={r:+.2f} (p={p:.3f})",
                         fontsize=10)
        ax.axhline(0, color="grey", lw=0.5, alpha=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\alpha_s$ (src effect)")
        ax.grid(alpha=0.25)

        # target row
        ax = axes[1, col_i]
        t = fn(p_tgt)
        mask = np.isfinite(beta) & np.isfinite(t)
        ax.scatter(t[mask], beta[mask], alpha=0.7, s=20, color="tab:orange")
        for k in np.where(mask)[0]:
            ax.annotate(langs[k], (t[k], beta[k]), fontsize=7, alpha=0.75,
                        xytext=(3, 2), textcoords="offset points")
        if mask.sum() >= 3:
            r, p = stats.pearsonr(beta[mask], t[mask])
            ax.set_title(f"$\\beta_t$ vs {label}\nPearson r={r:+.2f} (p={p:.3f})",
                         fontsize=10)
        ax.axhline(0, color="grey", lw=0.5, alpha=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\beta_t$ (tgt effect)")
        ax.grid(alpha=0.25)

    fig.suptitle(f"Additive-model language effects vs FLORES PPL ({model})", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyze_model(df_model: pd.DataFrame, model: str,
                  coef_dir: Path, out_dir: Path, img_dir: Path) -> List[Dict]:
    coefs = pd.read_csv(coef_dir / f"coefficients_{model}.csv")
    langs = coefs["lang"].tolist()
    alpha = coefs["alpha_src"].to_numpy(dtype=float)
    beta = coefs["beta_tgt"].to_numpy(dtype=float)

    p_src_map = per_language_perplexity(df_model, "src")
    p_tgt_map = per_language_perplexity(df_model, "tgt")
    p_src = np.array([p_src_map.get(l, np.nan) for l in langs], dtype=float)
    p_tgt = np.array([p_tgt_map.get(l, np.nan) for l in langs], dtype=float)

    rows: List[Dict] = []
    for role, vec, ppl in (("src", alpha, p_src), ("tgt", beta, p_tgt)):
        for name, (fn, _) in TRANSFORMS.items():
            stat = correlate(vec, ppl, fn)
            rows.append({"model": model, "role": role, "transform": name, **stat})

    plot_annotated(model, langs, alpha, beta, p_src, p_tgt,
                   img_dir / f"alpha_beta_vs_ppl_{model}.png")

    print(f"\n[{model}] alpha (src effect) vs PPL transforms:")
    for r in [r for r in rows if r["role"] == "src"]:
        print(f"  {r['transform']:9s}  Pearson r={r['pearson_r']:+.3f} (p={r['pearson_p']:.3g})  "
              f"Spearman={r['spearman_r']:+.3f}")
    print(f"[{model}] beta (tgt effect) vs PPL transforms:")
    for r in [r for r in rows if r["role"] == "tgt"]:
        print(f"  {r['transform']:9s}  Pearson r={r['pearson_r']:+.3f} (p={r['pearson_p']:.3g})  "
              f"Spearman={r['spearman_r']:+.3f}")

    # Identify and report the largest absolute outliers wrt log P (the most
    # informative monotone proxy).
    log_p_src = np.log(p_src)
    log_p_tgt = np.log(p_tgt)
    src_mask = np.isfinite(alpha) & np.isfinite(log_p_src)
    tgt_mask = np.isfinite(beta) & np.isfinite(log_p_tgt)
    if src_mask.sum() >= 3:
        sl = np.polyfit(log_p_src[src_mask], alpha[src_mask], 1)
        src_resid = alpha[src_mask] - np.polyval(sl, log_p_src[src_mask])
        idx_src = np.where(src_mask)[0]
        order_src = idx_src[np.argsort(-np.abs(src_resid))]
        worst_src = [(langs[i], float(alpha[i] - np.polyval(sl, log_p_src[i])))
                     for i in order_src[:3]]
        print(f"[{model}] top-3 alpha outliers vs log P_s (signed residual):", worst_src)
    if tgt_mask.sum() >= 3:
        tl = np.polyfit(log_p_tgt[tgt_mask], beta[tgt_mask], 1)
        tgt_resid = beta[tgt_mask] - np.polyval(tl, log_p_tgt[tgt_mask])
        idx_tgt = np.where(tgt_mask)[0]
        order_tgt = idx_tgt[np.argsort(-np.abs(tgt_resid))]
        worst_tgt = [(langs[i], float(beta[i] - np.polyval(tl, log_p_tgt[i])))
                     for i in order_tgt[:3]]
        print(f"[{model}] top-3 beta outliers vs log P_t  (signed residual):", worst_tgt)

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    p.add_argument("--coef-dir", type=Path, default=DEFAULT_COEF_DIR)
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
        all_rows.extend(analyze_model(df_model, model, args.coef_dir,
                                      args.out_dir, args.img_dir))

    pd.DataFrame(all_rows).to_csv(args.out_dir / "correlations.csv", index=False)
    print(f"\nWrote: {args.out_dir/'correlations.csv'}")
    print(f"Wrote: {args.img_dir}/alpha_beta_vs_ppl_*.png")


if __name__ == "__main__":
    main()
