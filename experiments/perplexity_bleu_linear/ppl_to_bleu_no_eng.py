"""Rerun the BLEU ~ PPL fits with English dropped, plus chain-implied R^2.

Premise: In Experiment 8, dropping English roughly doubled the
Pearson correlation between additive language effects (alpha, beta) and
FLORES PPL transforms for Llama, and made several correlations crossed
p < 0.05 for both models. Additive effects themselves explain ~90% of
centered BLEU variance (Experiment 4 / 7). So PPL should, via the chain
PPL -> alpha,beta -> BLEU, predict BLEU somewhat better as well.

This script:

  (A) Re-runs the regressions of Experiment 1 (BLEU ~ P_s + P_t,
      with/without interaction) and Experiment 3 (log BLEU ~ log P_s +
      log P_t) on the no-English subset (n = 506 pairs).
  (B) Computes the chain-implied R^2 from PPL-vs-alpha/beta
      correlations and per-language variances. Compares to observed.

The chain math: for the additive model BLEU_ij ~= mu + alpha_i + beta_j,
ignoring the multiplicative residual, the per-pair variance of BLEU is
approximately var(alpha across pairs) + var(beta across pairs). A linear
fit BLEU ~ f(P_s) + g(P_t) can explain at most:

    R^2_chain ~= r^2(alpha, f(P)) * var(alpha)/var(BLEU)
                + r^2(beta,  g(P)) * var(beta) /var(BLEU)

assuming independence of the two PPL-axis contributions (which holds
in this design: P_s and P_t are nearly uncorrelated across pairs).

Outputs:
    outputs/perplexity_bleu_linear/ppl_to_bleu_no_eng/
        observed_fits.csv         # raw / log / +interaction R^2 + F-test
        chain_implied_r2.csv      # predicted R^2 from PPL <-> alpha/beta
        comparison_with_full.csv  # no-eng vs full-set R^2 side by side
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
DEFAULT_COEF_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "drop_english_ablation"
DEFAULT_FULL_INTERACTION = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "interaction_test" / "interaction_test.csv"
DEFAULT_FULL_LOGMULT = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "log_multiplicative" / "fit_summary.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "ppl_to_bleu_no_eng"


# ---------------------------------------------------------------- OLS helpers --

def fit_ols(X: np.ndarray, y: np.ndarray) -> Dict:
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    rss = float(((y - yhat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    dof = n - p
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    sigma2 = rss / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    t = beta / se
    pval = 2.0 * stats.t.sf(np.abs(t), df=dof)
    return {"beta": beta, "se": se, "t": t, "p": pval,
            "rss": rss, "tss": tss, "r2": r2, "adj_r2": adj_r2,
            "n": n, "p_features": p, "dof_resid": dof}


def f_test_nested(small: Dict, big: Dict) -> Tuple[float, float]:
    df1 = small["dof_resid"] - big["dof_resid"]
    df2 = big["dof_resid"]
    num = (small["rss"] - big["rss"]) / df1
    den = big["rss"] / df2
    F = num / den
    p = float(stats.f.sf(F, df1, df2))
    return float(F), p


# ----------------------------------------------------- observed fits (no eng) --

def fit_observed(df_no_eng: pd.DataFrame, model: str) -> Dict:
    """Run all four fits on the no-eng subset for a given model."""
    sub_raw = df_no_eng[df_no_eng["model"] == model].dropna(
        subset=["bleu", "src_flores_perplexity", "tgt_flores_perplexity"]).copy()
    sub_log = sub_raw[(sub_raw["bleu"] > 0) & (sub_raw["src_flores_perplexity"] > 0)
                       & (sub_raw["tgt_flores_perplexity"] > 0)].copy()
    sub_log["log_bleu"] = np.log(sub_log["bleu"].to_numpy(dtype=float))
    sub_log["log_src"] = np.log(sub_log["src_flores_perplexity"].to_numpy(dtype=float))
    sub_log["log_tgt"] = np.log(sub_log["tgt_flores_perplexity"].to_numpy(dtype=float))

    # ---- raw: BLEU ~ P_s + P_t [+ P_s * P_t]
    s = sub_raw["src_flores_perplexity"].to_numpy(dtype=float)
    t = sub_raw["tgt_flores_perplexity"].to_numpy(dtype=float)
    y = sub_raw["bleu"].to_numpy(dtype=float)
    Xa = np.column_stack([np.ones_like(s), s, t])
    Xb = np.column_stack([np.ones_like(s), s, t, s * t])
    fit_raw_add = fit_ols(Xa, y)
    fit_raw_int = fit_ols(Xb, y)
    F_raw, p_raw = f_test_nested(fit_raw_add, fit_raw_int)

    # ---- log: log BLEU ~ log P_s + log P_t [+ interaction]
    ls = sub_log["log_src"].to_numpy()
    lt = sub_log["log_tgt"].to_numpy()
    ly = sub_log["log_bleu"].to_numpy()
    Xal = np.column_stack([np.ones_like(ls), ls, lt])
    Xbl = np.column_stack([np.ones_like(ls), ls, lt, ls * lt])
    fit_log_add = fit_ols(Xal, ly)
    fit_log_int = fit_ols(Xbl, ly)
    F_log, p_log = f_test_nested(fit_log_add, fit_log_int)

    return {
        "model": model,
        "n_raw": fit_raw_add["n"], "n_log": fit_log_add["n"],
        # raw additive
        "r2_raw_additive": fit_raw_add["r2"],
        "coef_raw_intercept": float(fit_raw_add["beta"][0]),
        "coef_raw_P_s": float(fit_raw_add["beta"][1]),
        "coef_raw_P_t": float(fit_raw_add["beta"][2]),
        "p_raw_P_s": float(fit_raw_add["p"][1]),
        "p_raw_P_t": float(fit_raw_add["p"][2]),
        # raw + interaction
        "r2_raw_with_interaction": fit_raw_int["r2"],
        "F_raw_interaction": F_raw, "p_F_raw_interaction": p_raw,
        # log additive (Experiment 3 form)
        "r2_log_additive": fit_log_add["r2"],
        "coef_log_intercept": float(fit_log_add["beta"][0]),
        "coef_log_P_s": float(fit_log_add["beta"][1]),
        "coef_log_P_t": float(fit_log_add["beta"][2]),
        "p_log_P_s": float(fit_log_add["p"][1]),
        "p_log_P_t": float(fit_log_add["p"][2]),
        # log + interaction
        "r2_log_with_interaction": fit_log_int["r2"],
        "F_log_interaction": F_log, "p_F_log_interaction": p_log,
    }


# ----------------------------------------------------- chain-implied R^2 --

def per_language_ppl(df: pd.DataFrame, role: str) -> Dict[str, float]:
    col = "src_flores_perplexity" if role == "src" else "tgt_flores_perplexity"
    key = "src" if role == "src" else "tgt"
    sub = df[[key, col]].dropna()
    return sub.groupby(key)[col].mean().to_dict()


def chain_implied_r2(df_no_eng: pd.DataFrame, coef_dir: Path, model: str,
                     transform: str = "log") -> Dict:
    """Predict the R^2 of the direct PPL->BLEU regression from chain components.

    For the additive matrix BLEU_ij ~ mu + alpha_i + beta_j, the per-pair
    variance of BLEU breaks into per-pair variance of alpha + per-pair
    variance of beta (+ residual). Plugging in r^2(alpha, f(P)) and
    r^2(beta, g(P)):

        R^2_chain = r^2_alpha * var(alpha_pair)/var(BLEU)
                   + r^2_beta  * var(beta_pair) /var(BLEU)
    """
    sub = df_no_eng[df_no_eng["model"] == model].copy()

    coefs = pd.read_csv(coef_dir / f"coefficients_no_eng_{model}.csv")
    langs = coefs["lang"].tolist()
    alpha = dict(zip(langs, coefs["alpha_src"].to_numpy(dtype=float)))
    beta = dict(zip(langs, coefs["beta_tgt"].to_numpy(dtype=float)))

    ppl_src = per_language_ppl(sub, "src")
    ppl_tgt = per_language_ppl(sub, "tgt")

    common = sorted(set(langs) & set(ppl_src) & set(ppl_tgt))
    a_vec = np.array([alpha[l] for l in common])
    b_vec = np.array([beta[l] for l in common])
    p_src = np.array([ppl_src[l] for l in common])
    p_tgt = np.array([ppl_tgt[l] for l in common])

    if transform == "log":
        ts, tt = np.log(p_src), np.log(p_tgt)
    elif transform == "raw":
        ts, tt = p_src, p_tgt
    elif transform == "inv":
        ts, tt = 1.0 / p_src, 1.0 / p_tgt
    elif transform == "inv_sqrt":
        ts, tt = 1.0 / np.sqrt(p_src), 1.0 / np.sqrt(p_tgt)
    else:
        raise ValueError(transform)

    r_alpha, _ = stats.pearsonr(a_vec, ts)
    r_beta, _ = stats.pearsonr(b_vec, tt)
    r2_alpha = r_alpha ** 2
    r2_beta = r_beta ** 2

    # Per-pair variances: each language appears equally often as src and tgt,
    # so var(alpha across pairs) ~ var(alpha across langs).
    sub_pair = sub.dropna(subset=["bleu", "src_flores_perplexity",
                                  "tgt_flores_perplexity"]).copy()
    alpha_pairs = sub_pair["src"].map(alpha).to_numpy(dtype=float)
    beta_pairs = sub_pair["tgt"].map(beta).to_numpy(dtype=float)
    bleu = sub_pair["bleu"].to_numpy(dtype=float)
    var_alpha = float(alpha_pairs.var(ddof=0))
    var_beta = float(beta_pairs.var(ddof=0))
    var_bleu = float(bleu.var(ddof=0))

    r2_chain = (r2_alpha * var_alpha + r2_beta * var_beta) / var_bleu

    return {
        "model": model, "transform": transform,
        "n_langs": len(common),
        "r_alpha_vs_f(P_s)": float(r_alpha), "r2_alpha": r2_alpha,
        "r_beta_vs_g(P_t)": float(r_beta), "r2_beta": r2_beta,
        "var_alpha_per_pair": var_alpha,
        "var_beta_per_pair": var_beta,
        "var_bleu_per_pair": var_bleu,
        "predicted_r2_PPL_to_BLEU": float(r2_chain),
    }


# -------------------------------------------------- comparison vs full set --

def comparison_table(observed_rows: List[Dict],
                     full_interaction_csv: Path,
                     full_logmult_csv: Path) -> pd.DataFrame:
    full_int = pd.read_csv(full_interaction_csv)
    full_log = pd.read_csv(full_logmult_csv)

    rows: List[Dict] = []
    for obs in observed_rows:
        model = obs["model"]
        # raw additive (Exp 1 main-effects form)
        full_row = full_int[(full_int["model"] == model)
                            & (full_int["transform"] == "raw")].iloc[0]
        rows.append({"model": model, "form": "BLEU ~ P_s + P_t",
                     "r2_full": float(full_row["r2_no_interaction"]),
                     "r2_no_eng": obs["r2_raw_additive"],
                     "delta_r2": obs["r2_raw_additive"] - float(full_row["r2_no_interaction"])})
        rows.append({"model": model, "form": "BLEU ~ P_s + P_t + P_s*P_t",
                     "r2_full": float(full_row["r2_with_interaction"]),
                     "r2_no_eng": obs["r2_raw_with_interaction"],
                     "delta_r2": obs["r2_raw_with_interaction"] - float(full_row["r2_with_interaction"])})

        full_log_row = full_log[full_log["model"] == model].iloc[0]
        rows.append({"model": model, "form": "log BLEU ~ log P_s + log P_t",
                     "r2_full": float(full_log_row["r2_log_scale"]),
                     "r2_no_eng": obs["r2_log_additive"],
                     "delta_r2": obs["r2_log_additive"] - float(full_log_row["r2_log_scale"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main --

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    p.add_argument("--coef-dir", type=Path, default=DEFAULT_COEF_DIR)
    p.add_argument("--full-interaction", type=Path, default=DEFAULT_FULL_INTERACTION)
    p.add_argument("--full-logmult", type=Path, default=DEFAULT_FULL_LOGMULT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pair_csv)
    df_no_eng = pairs[(pairs["src"] != "eng") & (pairs["tgt"] != "eng")].copy()

    obs_rows = []
    chain_rows = []
    for model in sorted(df_no_eng["model"].unique()):
        obs = fit_observed(df_no_eng, model)
        obs_rows.append(obs)

        for transform in ("raw", "log", "inv", "inv_sqrt"):
            chain_rows.append(chain_implied_r2(df_no_eng, args.coef_dir,
                                               model, transform))

    pd.DataFrame(obs_rows).to_csv(args.out_dir / "observed_fits.csv", index=False)
    pd.DataFrame(chain_rows).to_csv(args.out_dir / "chain_implied_r2.csv", index=False)
    cmp_df = comparison_table(obs_rows, args.full_interaction, args.full_logmult)
    cmp_df.to_csv(args.out_dir / "comparison_with_full.csv", index=False)

    # Pretty print
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")

    print("\n=== Observed R^2: BLEU <- PPL fits, no English (Llama / Aya) ===")
    for obs in obs_rows:
        print(f"\n[{obs['model']}] n_raw={obs['n_raw']}, n_log={obs['n_log']}")
        print(f"  raw additive    R^2 = {obs['r2_raw_additive']:.4f}  "
              f"(coef P_s p={obs['p_raw_P_s']:.3g}; P_t p={obs['p_raw_P_t']:.3g})")
        print(f"  raw + interact  R^2 = {obs['r2_raw_with_interaction']:.4f}  "
              f"(F-test interaction p={obs['p_F_raw_interaction']:.3g})")
        print(f"  log additive    R^2 = {obs['r2_log_additive']:.4f}  "
              f"(coef logP_s p={obs['p_log_P_s']:.3g}; logP_t p={obs['p_log_P_t']:.3g})")
        print(f"  log + interact  R^2 = {obs['r2_log_with_interaction']:.4f}  "
              f"(F-test interaction p={obs['p_F_log_interaction']:.3g})")

    print("\n=== Side-by-side: full (24-lang) vs no-eng (23-lang) ===")
    print(cmp_df.to_string(index=False))

    print("\n=== Chain-implied R^2 from PPL <-> alpha,beta correlations ===")
    chain_df = pd.DataFrame(chain_rows)
    cols = ["model", "transform", "r2_alpha", "r2_beta",
            "var_alpha_per_pair", "var_beta_per_pair", "var_bleu_per_pair",
            "predicted_r2_PPL_to_BLEU"]
    print(chain_df[cols].to_string(index=False))

    print(f"\nWrote: {args.out_dir/'observed_fits.csv'}")
    print(f"Wrote: {args.out_dir/'chain_implied_r2.csv'}")
    print(f"Wrote: {args.out_dir/'comparison_with_full.csv'}")


if __name__ == "__main__":
    main()
