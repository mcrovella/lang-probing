"""Investigate the src/tgt asymmetry across Experiments 2, 3, and 5.

Observed puzzle:
  - Latent-vs-PPL (Exp 2 / Exp 5): src signal > tgt signal in Spearman.
  - Log-multiplicative fit (Exp 3): tgt PPL coef larger and more significant than src.

These can both be true. The two analyses measure different things:
  - The per-LANGUAGE study (Exp 2/5) collapses each row/col into one
    "competence" scalar and asks how monotone-related it is to PPL.
    Sample size n = 24.
  - The per-PAIR regression (Exp 3) asks how much, on average, swapping
    in a higher-PPL src or tgt changes BLEU on a given pair. Sample
    size n ~ 550 and the question is conditional on the other axis.

This script reports six diagnostics that together explain why the two
analyses can disagree:

  D1  Variance decomposition of BLEU:
      between-src var, between-tgt var, residual.
  D2  Per-language correlations:
      mean BLEU as src vs P_s   /   mean BLEU as tgt vs P_t.
  D3  Per-pair univariate correlations:
      BLEU vs log P_s   /   BLEU vs log P_t.
  D4  Per-pair univariate R^2 with each predictor alone (log scale).
  D5  Per-pair partial correlations (controlling for the other axis).
  D6  Sequential ANOVA: marginal SS gained by src dummies vs tgt dummies
      after the other side and the intercept.

Outputs:
    outputs/perplexity_bleu_linear/src_tgt_asymmetry/
        diagnostics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_CSV = REPO_ROOT / "data" / "pair_metrics.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "src_tgt_asymmetry"


def ols_r2(X: np.ndarray, y: np.ndarray) -> Dict:
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    rss = float(((y - yhat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    return {"beta": beta, "rss": rss, "tss": tss, "r2": r2, "n": n, "p": p}


def lang_dummies(series: pd.Series, langs: List[str]) -> np.ndarray:
    """Sum-to-zero coded dummies for a categorical: drop the last category."""
    idx = {l: i for i, l in enumerate(langs)}
    codes = series.map(idx).to_numpy()
    n = len(codes)
    k = len(langs) - 1
    D = np.zeros((n, k))
    for r, c in enumerate(codes):
        if c < k:
            D[r, c] = 1.0
    return D


def analyze_model(df_model: pd.DataFrame, model: str) -> List[Dict]:
    sub = df_model.dropna(subset=["bleu", "src_flores_perplexity", "tgt_flores_perplexity"]).copy()
    sub = sub[(sub["bleu"] > 0)].copy()
    sub["log_bleu"] = np.log(sub["bleu"].to_numpy(dtype=float))
    sub["log_src"] = np.log(sub["src_flores_perplexity"].to_numpy(dtype=float))
    sub["log_tgt"] = np.log(sub["tgt_flores_perplexity"].to_numpy(dtype=float))

    langs = sorted(set(sub["src"]) | set(sub["tgt"]))
    rows: List[Dict] = []

    # D1: variance decomposition of BLEU
    src_means = sub.groupby("src")["bleu"].mean()
    tgt_means = sub.groupby("tgt")["bleu"].mean()
    var_total = float(sub["bleu"].var(ddof=0))
    var_src = float(src_means.var(ddof=0))   # variance of per-src means
    var_tgt = float(tgt_means.var(ddof=0))
    rows.append({"diagnostic": "D1_variance_decomposition", "model": model,
                 "metric": "var_total_bleu", "value": var_total})
    rows.append({"diagnostic": "D1_variance_decomposition", "model": model,
                 "metric": "var_between_src_lang_means", "value": var_src})
    rows.append({"diagnostic": "D1_variance_decomposition", "model": model,
                 "metric": "var_between_tgt_lang_means", "value": var_tgt})
    rows.append({"diagnostic": "D1_variance_decomposition", "model": model,
                 "metric": "ratio_tgt_to_src_var", "value": var_tgt / var_src if var_src > 0 else float("nan")})

    # D2: per-language correlations
    src_ppl = sub.groupby("src")["src_flores_perplexity"].mean()
    tgt_ppl = sub.groupby("tgt")["tgt_flores_perplexity"].mean()
    common_src = sorted(set(src_means.index) & set(src_ppl.index))
    common_tgt = sorted(set(tgt_means.index) & set(tgt_ppl.index))
    sm = src_means.loc[common_src].to_numpy()
    sp = src_ppl.loc[common_src].to_numpy()
    tm = tgt_means.loc[common_tgt].to_numpy()
    tp = tgt_ppl.loc[common_tgt].to_numpy()
    for vec, ppl, label in [(sm, sp, "src_meanBLEU_vs_logP_src"),
                            (tm, tp, "tgt_meanBLEU_vs_logP_tgt")]:
        log_ppl = np.log(ppl)
        r_pear, p_pear = stats.pearsonr(vec, log_ppl)
        r_spe, p_spe = stats.spearmanr(vec, log_ppl)
        rows.append({"diagnostic": "D2_per_language_corr", "model": model,
                     "metric": f"{label}_pearson_r", "value": float(r_pear)})
        rows.append({"diagnostic": "D2_per_language_corr", "model": model,
                     "metric": f"{label}_pearson_p", "value": float(p_pear)})
        rows.append({"diagnostic": "D2_per_language_corr", "model": model,
                     "metric": f"{label}_spearman_r", "value": float(r_spe)})

    # D3: per-pair univariate correlations
    for x_col, label in [("log_src", "log_P_src"), ("log_tgt", "log_P_tgt")]:
        x = sub[x_col].to_numpy()
        y = sub["log_bleu"].to_numpy()
        r_pear, p_pear = stats.pearsonr(x, y)
        rows.append({"diagnostic": "D3_per_pair_univariate_corr", "model": model,
                     "metric": f"log_BLEU_vs_{label}_pearson_r", "value": float(r_pear)})
        rows.append({"diagnostic": "D3_per_pair_univariate_corr", "model": model,
                     "metric": f"log_BLEU_vs_{label}_pearson_p", "value": float(p_pear)})

    # D4: per-pair univariate R^2 (single predictor + intercept)
    y = sub["log_bleu"].to_numpy()
    n = len(y)
    for col_name, label in [("log_src", "log_P_src_only"), ("log_tgt", "log_P_tgt_only")]:
        X = np.column_stack([np.ones(n), sub[col_name].to_numpy()])
        fit = ols_r2(X, y)
        rows.append({"diagnostic": "D4_univariate_R2", "model": model,
                     "metric": f"r2_logBLEU_{label}", "value": float(fit["r2"])})

    # Also report the bivariate R^2 for reference (same as Exp 3).
    X_both = np.column_stack([np.ones(n), sub["log_src"].to_numpy(),
                              sub["log_tgt"].to_numpy()])
    fit_both = ols_r2(X_both, y)
    rows.append({"diagnostic": "D4_univariate_R2", "model": model,
                 "metric": "r2_logBLEU_both", "value": float(fit_both["r2"])})

    # D5: per-pair partial correlations
    # partial corr(BLEU, log_P_src | log_P_tgt) = corr of residuals from
    # regressing each on log_P_tgt.
    def partial(corr_var: str, control: str) -> float:
        Xc = np.column_stack([np.ones(n), sub[control].to_numpy()])
        b_y, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        r_y = y - Xc @ b_y
        x = sub[corr_var].to_numpy()
        b_x, *_ = np.linalg.lstsq(Xc, x, rcond=None)
        r_x = x - Xc @ b_x
        return float(stats.pearsonr(r_x, r_y)[0])

    rows.append({"diagnostic": "D5_per_pair_partial_corr", "model": model,
                 "metric": "partial_log_BLEU_vs_log_P_src_given_tgt", "value": partial("log_src", "log_tgt")})
    rows.append({"diagnostic": "D5_per_pair_partial_corr", "model": model,
                 "metric": "partial_log_BLEU_vs_log_P_tgt_given_src", "value": partial("log_tgt", "log_src")})

    # D6: sequential SS gained by src dummies vs tgt dummies, fitting on raw
    # BLEU (Exp 4 setting). Compare:
    #   intercept only -> +src dummies -> +tgt dummies   vs
    #   intercept only -> +tgt dummies -> +src dummies
    y_raw = sub["bleu"].to_numpy()
    int_only = np.ones((n, 1))
    D_src = lang_dummies(sub["src"], langs)
    D_tgt = lang_dummies(sub["tgt"], langs)
    rss0 = float(((y_raw - y_raw.mean()) ** 2).sum())

    fit_src = ols_r2(np.hstack([int_only, D_src]), y_raw)
    fit_tgt = ols_r2(np.hstack([int_only, D_tgt]), y_raw)
    fit_full = ols_r2(np.hstack([int_only, D_src, D_tgt]), y_raw)

    rows.append({"diagnostic": "D6_sequential_SS", "model": model,
                 "metric": "r2_intercept_only", "value": 0.0})
    rows.append({"diagnostic": "D6_sequential_SS", "model": model,
                 "metric": "r2_src_dummies_alone", "value": float(fit_src["r2"])})
    rows.append({"diagnostic": "D6_sequential_SS", "model": model,
                 "metric": "r2_tgt_dummies_alone", "value": float(fit_tgt["r2"])})
    rows.append({"diagnostic": "D6_sequential_SS", "model": model,
                 "metric": "r2_both_additive", "value": float(fit_full["r2"])})
    rows.append({"diagnostic": "D6_sequential_SS", "model": model,
                 "metric": "ss_share_src_alone", "value": float(fit_src["r2"])})
    rows.append({"diagnostic": "D6_sequential_SS", "model": model,
                 "metric": "ss_share_tgt_alone", "value": float(fit_tgt["r2"])})
    rows.append({"diagnostic": "D6_sequential_SS", "model": model,
                 "metric": "marginal_gain_src_after_tgt", "value": float(fit_full["r2"] - fit_tgt["r2"])})
    rows.append({"diagnostic": "D6_sequential_SS", "model": model,
                 "metric": "marginal_gain_tgt_after_src", "value": float(fit_full["r2"] - fit_src["r2"])})

    # Print a focused summary
    print(f"\n=== [{model}] src/tgt asymmetry diagnostics ===")
    def get(metric):
        for r in rows:
            if r["model"] == model and r["metric"] == metric:
                return r["value"]
        return float("nan")

    print(f"  D1 between-src lang variance:        {get('var_between_src_lang_means'):.3f}")
    print(f"  D1 between-tgt lang variance:        {get('var_between_tgt_lang_means'):.3f}")
    print(f"  D1 var_tgt / var_src ratio:          {get('ratio_tgt_to_src_var'):.3f}")
    print(f"  D2 corr(mean_BLEU_as_src, log P_s):  Pearson r={get('src_meanBLEU_vs_logP_src_pearson_r'):+.3f} "
          f"(p={get('src_meanBLEU_vs_logP_src_pearson_p'):.3g})  Spearman={get('src_meanBLEU_vs_logP_src_spearman_r'):+.3f}")
    print(f"  D2 corr(mean_BLEU_as_tgt, log P_t):  Pearson r={get('tgt_meanBLEU_vs_logP_tgt_pearson_r'):+.3f} "
          f"(p={get('tgt_meanBLEU_vs_logP_tgt_pearson_p'):.3g})  Spearman={get('tgt_meanBLEU_vs_logP_tgt_spearman_r'):+.3f}")
    print(f"  D3 per-pair univariate r(logBLEU, log P_s) = {get('log_BLEU_vs_log_P_src_pearson_r'):+.3f}")
    print(f"  D3 per-pair univariate r(logBLEU, log P_t) = {get('log_BLEU_vs_log_P_tgt_pearson_r'):+.3f}")
    print(f"  D4 per-pair R^2(logBLEU ~ log P_s):  {get('r2_logBLEU_log_P_src_only'):.4f}")
    print(f"  D4 per-pair R^2(logBLEU ~ log P_t):  {get('r2_logBLEU_log_P_tgt_only'):.4f}")
    print(f"  D4 per-pair R^2(logBLEU ~ both):     {get('r2_logBLEU_both'):.4f}")
    print(f"  D5 partial r(logBLEU, log P_s | log P_t) = {get('partial_log_BLEU_vs_log_P_src_given_tgt'):+.3f}")
    print(f"  D5 partial r(logBLEU, log P_t | log P_s) = {get('partial_log_BLEU_vs_log_P_tgt_given_src'):+.3f}")
    print(f"  D6 R^2 src dummies alone:            {get('r2_src_dummies_alone'):.4f}")
    print(f"  D6 R^2 tgt dummies alone:            {get('r2_tgt_dummies_alone'):.4f}")
    print(f"  D6 R^2 both additive:                {get('r2_both_additive'):.4f}")
    print(f"  D6 marginal gain src after tgt:      {get('marginal_gain_src_after_tgt'):.4f}")
    print(f"  D6 marginal gain tgt after src:      {get('marginal_gain_tgt_after_src'):.4f}")

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pair_csv)
    all_rows: List[Dict] = []
    for model in sorted(pairs["model"].unique()):
        df_model = pairs[pairs["model"] == model]
        all_rows.extend(analyze_model(df_model, model))

    pd.DataFrame(all_rows).to_csv(args.out_dir / "diagnostics.csv", index=False)
    print(f"\nWrote: {args.out_dir/'diagnostics.csv'}")


if __name__ == "__main__":
    main()
