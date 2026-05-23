"""Drop English and refit the additive RE model + PPL correlations.

Experiment 5 identified English as a consistent large positive outlier
in both alpha (src effect) and beta (tgt effect). It contributes
disproportionately to BLEU-vs-PPL residuals and may be dominating the
PPL correlation magnitudes. This script asks the obvious question: how
much of the alpha/beta vs FLORES PPL correlation is outlier-driven by
English, and how much is systematic across the other 23 languages?

Procedure:
  1. Drop all pairs where src == "eng" or tgt == "eng" from
     `data/pair_metrics.csv`.
  2. Refit the additive RE model on the remaining 23x22 = 506 pairs;
     compute new alpha (23-lang src effects) and beta (23-lang tgt
     effects). Mean / SD / R^2 are recomputed.
  3. Collapse FLORES PPL per language for the 23 surviving langs.
  4. Compute Pearson and Spearman correlations between alpha vs PPL
     transforms and beta vs PPL transforms.
  5. Load the original 24-lang correlations (from `alpha_beta_vs_ppl/`)
     for a side-by-side comparison table.
  6. Produce annotated scatter plots over the 23-lang set.

Outputs:
    outputs/perplexity_bleu_linear/drop_english_ablation/
        coefficients_no_eng_{model}.csv
        fit_summary_no_eng.csv
        correlations_no_eng.csv
        comparison_with_full.csv     # 24-lang vs 23-lang side by side
    img/perplexity_bleu_linear/
        alpha_beta_vs_ppl_no_eng_{model}.png
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
DEFAULT_FULL_CORR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "alpha_beta_vs_ppl" / "correlations.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "drop_english_ablation"
DEFAULT_IMG_DIR = REPO_ROOT / "img" / "perplexity_bleu_linear"


TRANSFORMS = {
    "raw": (lambda p: p, "$P$"),
    "log": (lambda p: np.log(p), "$\\log P$"),
    "inv": (lambda p: 1.0 / p, "$1/P$"),
    "inv_sqrt": (lambda p: 1.0 / np.sqrt(p), "$1/\\sqrt{P}$"),
}


def fit_additive_ols(df_long: pd.DataFrame, langs: List[str]
                     ) -> Tuple[float, Dict[str, float], Dict[str, float], Dict]:
    """Same OLS additive fit as in additive_random_effects.py."""
    sub = df_long.dropna(subset=["bleu", "src", "tgt"]).copy()
    n = len(sub)
    n_src = n_tgt = len(langs)
    src_idx = {s: i for i, s in enumerate(langs)}
    tgt_idx = {t: j for j, t in enumerate(langs)}

    n_cols = 1 + (n_src - 1) + (n_tgt - 1)
    X = np.zeros((n, n_cols))
    X[:, 0] = 1.0
    src_codes = sub["src"].map(src_idx).to_numpy()
    tgt_codes = sub["tgt"].map(tgt_idx).to_numpy()
    for k in range(n):
        i = src_codes[k]; j = tgt_codes[k]
        if i < n_src - 1:
            X[k, 1 + i] = 1.0
        if j < n_tgt - 1:
            X[k, 1 + (n_src - 1) + j] = 1.0

    y = sub["bleu"].to_numpy(dtype=float)
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta_hat
    intercept_raw = float(beta_hat[0])
    src_effects_raw = np.concatenate([beta_hat[1:n_src], [0.0]])
    tgt_effects_raw = np.concatenate([beta_hat[n_src:], [0.0]])
    src_mean = float(np.mean(src_effects_raw))
    tgt_mean = float(np.mean(tgt_effects_raw))
    mu = intercept_raw + src_mean + tgt_mean
    alphas = {langs[i]: float(src_effects_raw[i] - src_mean) for i in range(n_src)}
    betas = {langs[j]: float(tgt_effects_raw[j] - tgt_mean) for j in range(n_tgt)}

    rss = float(((y - y_hat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    info = {"n_obs": n, "mu": mu, "y_mean": float(y.mean()),
            "y_std": float(y.std()), "r2_additive_centered": r2}
    return mu, alphas, betas, info


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
        ax.set_xlabel(label); ax.set_ylabel(r"$\alpha_s$ (src effect)")
        ax.grid(alpha=0.25)

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
        ax.set_xlabel(label); ax.set_ylabel(r"$\beta_t$ (tgt effect)")
        ax.grid(alpha=0.25)

    fig.suptitle(f"Additive language effects vs FLORES PPL ({model}, English dropped)", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def analyze_model(df_full: pd.DataFrame, model: str,
                  out_dir: Path, img_dir: Path) -> Tuple[List[Dict], Dict]:
    df_model = df_full[df_full["model"] == model]
    df_no_eng = df_model[(df_model["src"] != "eng") & (df_model["tgt"] != "eng")].copy()
    langs = sorted(set(df_no_eng["src"]) | set(df_no_eng["tgt"]))
    assert "eng" not in langs, "English should be removed"

    mu, alphas, betas, info = fit_additive_ols(df_no_eng, langs)
    info["model"] = model
    info["n_langs_after_drop"] = len(langs)

    coef_df = pd.DataFrame({
        "model": model, "lang": langs,
        "alpha_src": [alphas[l] for l in langs],
        "beta_tgt": [betas[l] for l in langs],
    })
    coef_df["mu"] = mu
    coef_df.to_csv(out_dir / f"coefficients_no_eng_{model}.csv", index=False)

    p_src_map = per_language_perplexity(df_no_eng, "src")
    p_tgt_map = per_language_perplexity(df_no_eng, "tgt")
    p_src = np.array([p_src_map.get(l, np.nan) for l in langs], dtype=float)
    p_tgt = np.array([p_tgt_map.get(l, np.nan) for l in langs], dtype=float)
    alpha = np.array([alphas[l] for l in langs], dtype=float)
    beta = np.array([betas[l] for l in langs], dtype=float)

    rows: List[Dict] = []
    for role, vec, ppl in (("src", alpha, p_src), ("tgt", beta, p_tgt)):
        for name, (fn, _) in TRANSFORMS.items():
            stat = correlate(vec, ppl, fn)
            rows.append({"model": model, "role": role, "transform": name, **stat})

    plot_annotated(model, langs, alpha, beta, p_src, p_tgt,
                   img_dir / f"alpha_beta_vs_ppl_no_eng_{model}.png")

    print(f"\n[{model}] after dropping English: n_obs={info['n_obs']}, "
          f"n_langs={info['n_langs_after_drop']}, mu={mu:.3f}, "
          f"R^2_additive={info['r2_additive_centered']:.4f}")
    print(f"[{model}] alpha (src effect) vs PPL transforms (no English):")
    for r in [r for r in rows if r["role"] == "src"]:
        print(f"  {r['transform']:9s}  Pearson r={r['pearson_r']:+.3f} "
              f"(p={r['pearson_p']:.3g})  Spearman={r['spearman_r']:+.3f}")
    print(f"[{model}] beta (tgt effect) vs PPL transforms (no English):")
    for r in [r for r in rows if r["role"] == "tgt"]:
        print(f"  {r['transform']:9s}  Pearson r={r['pearson_r']:+.3f} "
              f"(p={r['pearson_p']:.3g})  Spearman={r['spearman_r']:+.3f}")

    return rows, info


def build_comparison(full_corr_path: Path, new_rows: List[Dict]) -> pd.DataFrame:
    """Side-by-side: 24-lang Pearson r vs 23-lang Pearson r."""
    full = pd.read_csv(full_corr_path)
    new = pd.DataFrame(new_rows)

    full_slim = full[["model", "role", "transform", "n", "pearson_r", "pearson_p",
                      "spearman_r"]].rename(columns={
        "n": "n_full", "pearson_r": "pearson_r_full",
        "pearson_p": "pearson_p_full", "spearman_r": "spearman_r_full"})
    new_slim = new[["model", "role", "transform", "n", "pearson_r", "pearson_p",
                    "spearman_r"]].rename(columns={
        "n": "n_no_eng", "pearson_r": "pearson_r_no_eng",
        "pearson_p": "pearson_p_no_eng", "spearman_r": "spearman_r_no_eng"})
    merged = full_slim.merge(new_slim, on=["model", "role", "transform"])
    merged["delta_pearson_r"] = merged["pearson_r_no_eng"] - merged["pearson_r_full"]
    merged["delta_abs_pearson_r"] = merged["pearson_r_no_eng"].abs() - merged["pearson_r_full"].abs()
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    p.add_argument("--full-corr-csv", type=Path, default=DEFAULT_FULL_CORR,
                   help="Path to existing 24-lang correlations CSV from alpha_beta_vs_ppl.py")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.img_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pair_csv)
    all_rows: List[Dict] = []
    summaries: List[Dict] = []
    for model in sorted(pairs["model"].unique()):
        rows, info = analyze_model(pairs, model, args.out_dir, args.img_dir)
        all_rows.extend(rows)
        summaries.append(info)

    pd.DataFrame(all_rows).to_csv(args.out_dir / "correlations_no_eng.csv", index=False)
    pd.DataFrame(summaries).to_csv(args.out_dir / "fit_summary_no_eng.csv", index=False)

    if args.full_corr_csv.exists():
        comparison = build_comparison(args.full_corr_csv, all_rows)
        comparison.to_csv(args.out_dir / "comparison_with_full.csv", index=False)

        # Pretty-print the comparison: most useful single view.
        print("\n=== Pearson r side-by-side: 24-lang (full) vs 23-lang (no eng) ===")
        pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
        cols = ["model", "role", "transform", "pearson_r_full", "pearson_r_no_eng",
                "delta_pearson_r", "delta_abs_pearson_r"]
        print(comparison[cols].to_string(index=False))
        print(f"\nWrote: {args.out_dir/'comparison_with_full.csv'}")

    print(f"\nWrote: {args.out_dir/'correlations_no_eng.csv'}")
    print(f"Wrote: {args.out_dir/'fit_summary_no_eng.csv'}")
    print(f"Wrote: {args.img_dir}/alpha_beta_vs_ppl_no_eng_*.png")


if __name__ == "__main__":
    main()
