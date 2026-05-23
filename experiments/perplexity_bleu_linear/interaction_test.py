"""Test whether a src*tgt FLORES-perplexity interaction term improves BLEU prediction.

For each model (llama, aya), fits two OLS models over all directed language pairs:

    (a) bleu ~ 1 + src_ppl + tgt_ppl
    (b) bleu ~ 1 + src_ppl + tgt_ppl + src_ppl * tgt_ppl

Reports R^2, adjusted R^2, the nested-model F-test for the interaction
term, and the t-test on the interaction coefficient (equivalent when one
parameter is added). Optionally repeats on log-transformed perplexities.

Reads `data/pair_metrics.csv`. Writes a tidy results table to
`outputs/perplexity_bleu_linear/interaction_test/interaction_test.csv`
and per-model coefficient tables alongside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_CSV = REPO_ROOT / "data" / "pair_metrics.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "interaction_test"


def fit_ols(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Plain OLS with classical inference. Returns coeffs, SEs, t, p, RSS, dof_resid, R^2."""
    n, p = X.shape
    # Coefficients via lstsq (numerically stable).
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    dof_resid = n - p
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof_resid if dof_resid > 0 else float("nan")

    # SE(beta) = sqrt(sigma2 * diag((X'X)^-1))
    sigma2 = rss / dof_resid if dof_resid > 0 else float("nan")
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    t = beta / se
    pval = 2.0 * stats.t.sf(np.abs(t), df=dof_resid)

    return {
        "beta": beta,
        "se": se,
        "t": t,
        "p": pval,
        "rss": rss,
        "tss": tss,
        "r2": r2,
        "adj_r2": adj_r2,
        "dof_resid": dof_resid,
        "n": n,
        "p_features": p,
    }


def nested_f_test(small: Dict, big: Dict) -> Tuple[float, float, int, int]:
    """F-test comparing nested OLS models. `big` adds parameter(s) to `small`."""
    df1 = small["dof_resid"] - big["dof_resid"]
    df2 = big["dof_resid"]
    if df1 <= 0:
        raise ValueError("`big` must add parameters relative to `small`.")
    num = (small["rss"] - big["rss"]) / df1
    den = big["rss"] / df2
    f_stat = num / den
    p_val = float(stats.f.sf(f_stat, df1, df2))
    return float(f_stat), p_val, int(df1), int(df2)


def prepare(df: pd.DataFrame, transform: str) -> Tuple[np.ndarray, np.ndarray]:
    sub = df.dropna(subset=["bleu", "src_flores_perplexity", "tgt_flores_perplexity"]).copy()
    s = sub["src_flores_perplexity"].to_numpy(dtype=float)
    t = sub["tgt_flores_perplexity"].to_numpy(dtype=float)
    if transform == "log":
        if (s <= 0).any() or (t <= 0).any():
            raise ValueError("non-positive perplexity encountered for log transform")
        s = np.log(s)
        t = np.log(t)
    elif transform != "raw":
        raise ValueError(f"unknown transform: {transform}")
    y = sub["bleu"].to_numpy(dtype=float)
    return s, t, y


def design(s: np.ndarray, t: np.ndarray, with_interaction: bool) -> Tuple[np.ndarray, List[str]]:
    cols = [np.ones_like(s), s, t]
    names = ["intercept", "src_ppl", "tgt_ppl"]
    if with_interaction:
        cols.append(s * t)
        names.append("src_ppl:tgt_ppl")
    return np.column_stack(cols), names


def analyze_model(df_model: pd.DataFrame, model: str, transform: str) -> Tuple[Dict, pd.DataFrame]:
    s, t, y = prepare(df_model, transform)
    X_a, names_a = design(s, t, with_interaction=False)
    X_b, names_b = design(s, t, with_interaction=True)

    fit_a = fit_ols(X_a, y)
    fit_b = fit_ols(X_b, y)
    f_stat, f_p, df1, df2 = nested_f_test(fit_a, fit_b)

    # t-test on the interaction coef (last column) is equivalent to the F-test here.
    inter_idx = names_b.index("src_ppl:tgt_ppl")
    summary = {
        "model": model,
        "transform": transform,
        "n": fit_a["n"],
        "r2_no_interaction": fit_a["r2"],
        "adj_r2_no_interaction": fit_a["adj_r2"],
        "r2_with_interaction": fit_b["r2"],
        "adj_r2_with_interaction": fit_b["adj_r2"],
        "delta_r2": fit_b["r2"] - fit_a["r2"],
        "F_stat": f_stat,
        "F_df1": df1,
        "F_df2": df2,
        "F_p_value": f_p,
        "interaction_coef": float(fit_b["beta"][inter_idx]),
        "interaction_se": float(fit_b["se"][inter_idx]),
        "interaction_t": float(fit_b["t"][inter_idx]),
        "interaction_p": float(fit_b["p"][inter_idx]),
    }

    # Per-coefficient tables for both fits, stacked.
    coef_rows = []
    for spec, fit, names in [("no_interaction", fit_a, names_a), ("with_interaction", fit_b, names_b)]:
        for i, name in enumerate(names):
            coef_rows.append({
                "model": model,
                "transform": transform,
                "spec": spec,
                "term": name,
                "coef": float(fit["beta"][i]),
                "se": float(fit["se"][i]),
                "t": float(fit["t"][i]),
                "p": float(fit["p"][i]),
            })
    coef_df = pd.DataFrame(coef_rows)
    return summary, coef_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--transforms",
        nargs="+",
        choices=("raw", "log"),
        default=["raw", "log"],
        help="Which perplexity transforms to fit (default: both).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(args.pair_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []
    coef_frames: List[pd.DataFrame] = []
    for model in sorted(pairs["model"].unique()):
        df_model = pairs[pairs["model"] == model]
        for transform in args.transforms:
            summary, coef_df = analyze_model(df_model, model, transform)
            summaries.append(summary)
            coef_frames.append(coef_df)

    summary_df = pd.DataFrame(summaries)
    coef_df = pd.concat(coef_frames, ignore_index=True)
    summary_df.to_csv(args.out_dir / "interaction_test.csv", index=False)
    coef_df.to_csv(args.out_dir / "coefficients.csv", index=False)

    # Pretty print.
    pd.set_option("display.float_format", lambda x: f"{x: .6g}")
    print("\n=== Nested-model F-test: does src_ppl * tgt_ppl improve fit? ===")
    print(summary_df.to_string(index=False))
    print(f"\nWrote: {args.out_dir/'interaction_test.csv'}")
    print(f"Wrote: {args.out_dir/'coefficients.csv'}")


if __name__ == "__main__":
    main()
