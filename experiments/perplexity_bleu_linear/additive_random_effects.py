"""Fit additive two-way model: BLEU_ij = mu + alpha_i + beta_j + eps.

This is the random-effects-style decomposition the user asked for, fit by
OLS with sum-to-zero constraints. Compared like-for-like against the
rank-1 SVD on the SAME R^2 scale (centered total sum of squares) and
also reports the original uncentered "faithfulness" metric for context.

Key claim being tested: how much BLEU variance is captured by per-language
src and tgt effects with no interaction term, versus rank-1 (which is a
multiplicative interaction with no main effects).

Outputs:
    outputs/perplexity_bleu_linear/additive_random_effects/
        coefficients_{model}.csv     # mu and per-lang alpha, beta
        fit_summary.csv              # R^2 comparison across decompositions
        bleu_matrix_{model}.csv      # the source pivot (for downstream use)

The closed-form additive fit assumes nearly-balanced data; we use OLS on
the long-format data instead so the 24 missing (diagonal) cells are
handled exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_CSV = REPO_ROOT / "data" / "pair_metrics.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "additive_random_effects"


def build_bleu_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    langs = sorted(set(df["src"]) | set(df["tgt"]))
    pivot = (
        df.pivot_table(index="src", columns="tgt", values="bleu", aggfunc="mean")
        .reindex(index=langs, columns=langs)
    )
    return pivot.values, langs


def fit_additive_ols(df_long: pd.DataFrame, langs: List[str]
                     ) -> Tuple[float, Dict[str, float], Dict[str, float], pd.DataFrame]:
    """Fit BLEU = mu + alpha_i + beta_j by OLS on long-format data.

    Uses one-hot dummies with the last src and tgt dropped for identifiability,
    then re-centers the recovered effects to sum to zero.
    """
    sub = df_long.dropna(subset=["bleu", "src", "tgt"]).copy()
    n = len(sub)

    src_idx = {s: i for i, s in enumerate(langs)}
    tgt_idx = {t: j for j, t in enumerate(langs)}
    n_src = n_tgt = len(langs)

    # Dummies: drop the last src and last tgt as references.
    n_cols = 1 + (n_src - 1) + (n_tgt - 1)
    X = np.zeros((n, n_cols))
    X[:, 0] = 1.0
    src_codes = sub["src"].map(src_idx).to_numpy()
    tgt_codes = sub["tgt"].map(tgt_idx).to_numpy()
    for k in range(n):
        i = src_codes[k]
        j = tgt_codes[k]
        if i < n_src - 1:
            X[k, 1 + i] = 1.0
        if j < n_tgt - 1:
            X[k, 1 + (n_src - 1) + j] = 1.0

    y = sub["bleu"].to_numpy(dtype=float)
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta_hat

    intercept_raw = float(beta_hat[0])
    src_effects_raw = np.concatenate([beta_hat[1:n_src], [0.0]])  # reference = 0
    tgt_effects_raw = np.concatenate([beta_hat[n_src:], [0.0]])

    # Re-center to sum-to-zero: move the mean of src and tgt effects into mu.
    src_mean = float(np.mean(src_effects_raw))
    tgt_mean = float(np.mean(tgt_effects_raw))
    mu = intercept_raw + src_mean + tgt_mean
    alphas = {langs[i]: float(src_effects_raw[i] - src_mean) for i in range(n_src)}
    betas = {langs[j]: float(tgt_effects_raw[j] - tgt_mean) for j in range(n_tgt)}

    fit_df = pd.DataFrame({
        "src": sub["src"].to_numpy(),
        "tgt": sub["tgt"].to_numpy(),
        "bleu": y,
        "bleu_pred_additive": y_hat,
    })
    return mu, alphas, betas, fit_df


def impute_column_means(M: np.ndarray) -> np.ndarray:
    col_means = np.nanmean(M, axis=0)
    global_mean = float(np.nanmean(M))
    col_means = np.where(np.isnan(col_means), global_mean, col_means)
    nan_idx = np.where(np.isnan(M))
    M_filled = M.copy()
    M_filled[nan_idx] = np.take(col_means, nan_idx[1])
    return M_filled


def rank1_matrix(M: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return float(S[0]) * np.outer(U[:, 0], Vt[0])


def evaluate_decompositions(M_obs: np.ndarray,
                            M_additive: np.ndarray,
                            mu: float, alphas: Dict[str, float], betas: Dict[str, float],
                            langs: List[str]) -> Dict:
    """Compute all R^2 / faithfulness metrics on the observed (non-NaN) cells."""
    mask = ~np.isnan(M_obs)
    y = M_obs[mask]
    y_mean = y.mean()
    tss_centered = float(((y - y_mean) ** 2).sum())
    norm_uncentered = float(np.linalg.norm(y))  # ||M||_F restricted to obs cells

    # Additive prediction matrix.
    alpha_vec = np.array([alphas[l] for l in langs])
    beta_vec = np.array([betas[l] for l in langs])
    M_add = mu + alpha_vec[:, None] + beta_vec[None, :]
    add_pred = M_add[mask]
    rss_add = float(((y - add_pred) ** 2).sum())
    r2_add_centered = 1.0 - rss_add / tss_centered
    faith_add = 1.0 - np.linalg.norm(y - add_pred) / norm_uncentered

    # Rank-1 SVD on uncentered, imputed matrix.
    M_filled = impute_column_means(M_obs)
    M1 = rank1_matrix(M_filled)
    svd1_pred = M1[mask]
    rss_svd1 = float(((y - svd1_pred) ** 2).sum())
    r2_svd1_centered = 1.0 - rss_svd1 / tss_centered
    faith_svd1 = 1.0 - np.linalg.norm(y - svd1_pred) / norm_uncentered

    # Rank-1 of centered matrix (apples-to-apples for the rank-1 part above the mean).
    M_centered = M_filled - y_mean
    M1_centered = rank1_matrix(M_centered)
    centered_pred = (M1_centered + y_mean)[mask]
    rss_centered_svd = float(((y - centered_pred) ** 2).sum())
    r2_svd1_on_centered = 1.0 - rss_centered_svd / tss_centered

    # AMMI-style: additive + rank-1 of residuals.
    M_resid_for_svd = M_filled - M_add
    M1_resid = rank1_matrix(M_resid_for_svd)
    ammi_pred = (M_add + M1_resid)[mask]
    rss_ammi = float(((y - ammi_pred) ** 2).sum())
    r2_ammi_centered = 1.0 - rss_ammi / tss_centered

    return {
        "n_obs_cells": int(mask.sum()),
        "y_mean": float(y_mean),
        "y_std": float(y.std()),
        "tss_centered": tss_centered,
        "norm_uncentered": norm_uncentered,
        "r2_additive_centered": r2_add_centered,
        "r2_rank1_centered": r2_svd1_centered,
        "r2_rank1_on_centered_matrix": r2_svd1_on_centered,
        "r2_ammi_centered": r2_ammi_centered,
        "faithfulness_additive_uncentered": faith_add,
        "faithfulness_rank1_uncentered": faith_svd1,
    }


def analyze_model(df_model: pd.DataFrame, model: str, out_dir: Path) -> Dict:
    M, langs = build_bleu_matrix(df_model)
    mu, alphas, betas, _fit_df = fit_additive_ols(df_model, langs)

    # Save per-language coefficients.
    coef_df = pd.DataFrame({
        "model": model,
        "lang": langs,
        "alpha_src": [alphas[l] for l in langs],
        "beta_tgt": [betas[l] for l in langs],
    })
    coef_df["mu"] = mu
    coef_df.to_csv(out_dir / f"coefficients_{model}.csv", index=False)
    pd.DataFrame(M, index=langs, columns=langs).to_csv(
        out_dir / f"bleu_matrix_{model}.csv"
    )

    metrics = evaluate_decompositions(M, None, mu, alphas, betas, langs)
    metrics["model"] = model
    metrics["mu"] = mu
    metrics["n_langs"] = len(langs)
    metrics["n_params_additive"] = 1 + 2 * (len(langs) - 1)  # df after sum-to-zero
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pair_csv)
    summaries: List[Dict] = []
    for model in sorted(pairs["model"].unique()):
        df_model = pairs[pairs["model"] == model]
        metrics = analyze_model(df_model, model, args.out_dir)
        summaries.append(metrics)

    summary = pd.DataFrame(summaries)
    cols_order = [
        "model", "n_langs", "n_obs_cells", "n_params_additive",
        "mu", "y_mean", "y_std",
        "r2_additive_centered", "r2_rank1_centered", "r2_rank1_on_centered_matrix",
        "r2_ammi_centered", "faithfulness_additive_uncentered", "faithfulness_rank1_uncentered",
    ]
    summary = summary[cols_order]
    summary.to_csv(args.out_dir / "fit_summary.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x: .4f}")
    print("\n=== Additive RE vs Rank-1 SVD comparison (R^2 on centered total SS) ===")
    print(summary.to_string(index=False))
    print(f"\nWrote: {args.out_dir/'fit_summary.csv'}")
    print(f"Wrote: {args.out_dir/'coefficients_{{llama,aya}}.csv'}")


if __name__ == "__main__":
    main()
