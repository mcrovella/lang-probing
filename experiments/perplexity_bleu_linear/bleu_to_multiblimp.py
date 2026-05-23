"""Repeat Experiment 9 with Multi-BLiMP accuracy as the per-language predictor.

Background. Experiment 9 ran the chain
  PPL  ->  alpha, beta  ->  BLEU
and showed (a) dropping English lifts Llama R^2 from 0.02 to 0.11,
(b) the chain math predicts observed R^2 to within ~0.5 pp, (c) the
ceiling is bounded by how well PPL predicts alpha, beta.

Multi-BLiMP accuracy is, in principle, a more direct measure of
per-language grammatical competence. This script substitutes Multi-BLiMP
for PPL and asks whether it is a stronger proxy via the same chain.

Important caveat. Multi-BLiMP covers only 17 of 24 languages
(the missing ones are ara, ind, jpn, kor, vie, zho_simpl, zho_trad).
Several of those were the largest Llama outliers in Experiments 5/8.
So the Multi-BLiMP subset already filters out most outliers; to make
a fair PPL-vs-MBL comparison we fit BOTH predictors on the same
17-language (272-pair) subset, and separately on the no-English
sub-subset (16-language, 240 pairs).

Procedure for each (subset, model):
  1. Refit the additive RE model on this subset -> new alpha, beta
     (the relevant decomposition for the chain math on THIS subset).
  2. Compute alpha/beta vs Multi-BLiMP accuracy correlations under four
     transforms: raw, 1 - a, log(1 - a), logit(a).
  3. Fit BLEU ~ acc_s + acc_t and BLEU ~ acc_s + acc_t + acc_s * acc_t.
     Fit log BLEU ~ log(1 - acc_s) + log(1 - acc_t) (log analog of the
     log-multiplicative form from Experiment 3).
  4. For the same subset, fit BLEU ~ P_s + P_t and log BLEU ~ log P_s +
     log P_t (apples-to-apples PPL baseline).
  5. Compute chain-implied R^2 for both predictors and compare to
     observed.

Outputs:
    outputs/perplexity_bleu_linear/bleu_to_multiblimp/
        observed_fits.csv          # MBL and PPL fits on each subset
        alpha_beta_vs_mbl.csv      # correlations of alpha, beta with MBL transforms
        chain_implied_r2.csv       # chain predictions for both predictors
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
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "bleu_to_multiblimp"


MBL_TRANSFORMS = {
    "raw":      (lambda a: a,                "$a$"),
    "err":      (lambda a: 1.0 - a,          "$1 - a$"),
    "log_err":  (lambda a: np.log(1.0 - a),  "$\\log(1 - a)$"),
    "logit":    (lambda a: np.log(a / (1.0 - a)), "logit$(a)$"),
}

PPL_TRANSFORMS = {
    "raw":      (lambda p: p,                  "$P$"),
    "log":      (lambda p: np.log(p),          "$\\log P$"),
    "inv":      (lambda p: 1.0 / p,            "$1/P$"),
    "inv_sqrt": (lambda p: 1.0 / np.sqrt(p),   "$1/\\sqrt{P}$"),
}


# ----------------------------------------------------- OLS / F-test helpers --

def fit_ols(X: np.ndarray, y: np.ndarray) -> Dict:
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    rss = float(((y - yhat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    dof = n - p
    sigma2 = rss / dof if dof > 0 else float("nan")
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    t = beta / se
    pval = 2.0 * stats.t.sf(np.abs(t), df=dof)
    return {"beta": beta, "se": se, "t": t, "p": pval,
            "rss": rss, "tss": tss, "r2": r2,
            "n": n, "p_features": p, "dof_resid": dof}


def f_test_nested(small: Dict, big: Dict) -> Tuple[float, float]:
    df1 = small["dof_resid"] - big["dof_resid"]
    df2 = big["dof_resid"]
    num = (small["rss"] - big["rss"]) / df1
    den = big["rss"] / df2
    F = num / den
    return float(F), float(stats.f.sf(F, df1, df2))


# ----------------------------------------------- additive RE refit on subset --

def fit_additive_re(sub: pd.DataFrame, langs: List[str]) -> Dict[str, Dict[str, float]]:
    """OLS additive fit on a long-format dataframe restricted to `langs`."""
    n = len(sub); n_lang = len(langs)
    src_idx = {l: i for i, l in enumerate(langs)}
    tgt_idx = {l: i for i, l in enumerate(langs)}
    n_cols = 1 + (n_lang - 1) + (n_lang - 1)
    X = np.zeros((n, n_cols))
    X[:, 0] = 1.0
    src_codes = sub["src"].map(src_idx).to_numpy()
    tgt_codes = sub["tgt"].map(tgt_idx).to_numpy()
    for k in range(n):
        i = src_codes[k]; j = tgt_codes[k]
        if i < n_lang - 1: X[k, 1 + i] = 1.0
        if j < n_lang - 1: X[k, 1 + (n_lang - 1) + j] = 1.0
    y = sub["bleu"].to_numpy(dtype=float)
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept_raw = float(beta_hat[0])
    src_eff_raw = np.concatenate([beta_hat[1:n_lang], [0.0]])
    tgt_eff_raw = np.concatenate([beta_hat[n_lang:], [0.0]])
    src_mean = float(np.mean(src_eff_raw))
    tgt_mean = float(np.mean(tgt_eff_raw))
    mu = intercept_raw + src_mean + tgt_mean
    alphas = {langs[i]: float(src_eff_raw[i] - src_mean) for i in range(n_lang)}
    betas = {langs[i]: float(tgt_eff_raw[i] - tgt_mean) for i in range(n_lang)}
    y_hat = X @ beta_hat
    rss = float(((y - y_hat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    return {"mu": mu, "alpha": alphas, "beta": betas,
            "r2_additive_centered": r2, "n_obs": n, "n_lang": n_lang}


# --------------------------------------------- per-language scalar helpers --

def per_lang_scalar(df: pd.DataFrame, role: str, col: str) -> Dict[str, float]:
    key = "src" if role == "src" else "tgt"
    sub = df[[key, col]].dropna()
    return sub.groupby(key)[col].mean().to_dict()


def correlate(latent: np.ndarray, x: np.ndarray) -> Dict:
    mask = np.isfinite(latent) & np.isfinite(x)
    n = int(mask.sum())
    if n < 3:
        return {"n": n, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan}
    pr, pp = stats.pearsonr(latent[mask], x[mask])
    sr, _ = stats.spearmanr(latent[mask], x[mask])
    return {"n": n, "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr)}


# ------------------------------------------------------- core per-subset --

def analyze_subset(df_full: pd.DataFrame, subset_name: str, model: str,
                   drop_eng: bool) -> Dict:
    """Run all analyses on (Multi-BLiMP-available, optionally no-eng) subset."""
    sub_all = df_full[df_full["model"] == model].copy()
    sub = sub_all.dropna(subset=[
        "bleu", "src_multiblimp_accuracy", "tgt_multiblimp_accuracy",
        "src_flores_perplexity", "tgt_flores_perplexity",
    ]).copy()
    if drop_eng:
        sub = sub[(sub["src"] != "eng") & (sub["tgt"] != "eng")].copy()
    langs = sorted(set(sub["src"]) | set(sub["tgt"]))

    # (1) Refit additive RE on this subset
    fit_re = fit_additive_re(sub, langs)
    alpha = fit_re["alpha"]; beta = fit_re["beta"]; mu = fit_re["mu"]

    # (2) Per-language Multi-BLiMP accuracy and PPL
    acc_src = per_lang_scalar(sub, "src", "src_multiblimp_accuracy")
    acc_tgt = per_lang_scalar(sub, "tgt", "tgt_multiblimp_accuracy")
    ppl_src = per_lang_scalar(sub, "src", "src_flores_perplexity")
    ppl_tgt = per_lang_scalar(sub, "tgt", "tgt_flores_perplexity")
    a_src = np.array([acc_src[l] for l in langs])
    a_tgt = np.array([acc_tgt[l] for l in langs])
    p_src = np.array([ppl_src[l] for l in langs])
    p_tgt = np.array([ppl_tgt[l] for l in langs])
    alpha_vec = np.array([alpha[l] for l in langs])
    beta_vec = np.array([beta[l] for l in langs])

    # alpha/beta vs Multi-BLiMP transforms
    corr_rows = []
    for transform, (fn, _) in MBL_TRANSFORMS.items():
        for role, vec, sx in (("src", alpha_vec, a_src), ("tgt", beta_vec, a_tgt)):
            x = fn(sx)
            stat = correlate(vec, x)
            corr_rows.append({"subset": subset_name, "model": model, "role": role,
                              "transform": transform, **stat})

    # (3) Observed PPL and MBL fits on this subset
    n = len(sub)
    bleu = sub["bleu"].to_numpy(dtype=float)
    a_s_pairs = sub["src_multiblimp_accuracy"].to_numpy()
    a_t_pairs = sub["tgt_multiblimp_accuracy"].to_numpy()
    p_s_pairs = sub["src_flores_perplexity"].to_numpy()
    p_t_pairs = sub["tgt_flores_perplexity"].to_numpy()

    # raw Multi-BLiMP
    X_mbl_add = np.column_stack([np.ones(n), a_s_pairs, a_t_pairs])
    X_mbl_int = np.column_stack([np.ones(n), a_s_pairs, a_t_pairs, a_s_pairs * a_t_pairs])
    fit_mbl_add = fit_ols(X_mbl_add, bleu)
    fit_mbl_int = fit_ols(X_mbl_int, bleu)
    F_mbl, p_F_mbl = f_test_nested(fit_mbl_add, fit_mbl_int)

    # log analog: log BLEU ~ log(1-a_s) + log(1-a_t)
    log_bleu = np.log(bleu)
    log_err_s = np.log(1.0 - a_s_pairs)
    log_err_t = np.log(1.0 - a_t_pairs)
    X_mbl_logadd = np.column_stack([np.ones(n), log_err_s, log_err_t])
    X_mbl_logint = np.column_stack([np.ones(n), log_err_s, log_err_t, log_err_s * log_err_t])
    fit_mbl_logadd = fit_ols(X_mbl_logadd, log_bleu)
    fit_mbl_logint = fit_ols(X_mbl_logint, log_bleu)
    F_mbl_log, p_F_mbl_log = f_test_nested(fit_mbl_logadd, fit_mbl_logint)

    # PPL baseline on the SAME subset
    X_ppl_add = np.column_stack([np.ones(n), p_s_pairs, p_t_pairs])
    X_ppl_int = np.column_stack([np.ones(n), p_s_pairs, p_t_pairs, p_s_pairs * p_t_pairs])
    fit_ppl_add = fit_ols(X_ppl_add, bleu)
    fit_ppl_int = fit_ols(X_ppl_int, bleu)
    F_ppl, p_F_ppl = f_test_nested(fit_ppl_add, fit_ppl_int)

    log_p_s = np.log(p_s_pairs); log_p_t = np.log(p_t_pairs)
    X_ppl_logadd = np.column_stack([np.ones(n), log_p_s, log_p_t])
    fit_ppl_logadd = fit_ols(X_ppl_logadd, log_bleu)

    # (4) Chain-implied R^2
    alpha_pair = sub["src"].map(alpha).to_numpy()
    beta_pair = sub["tgt"].map(beta).to_numpy()
    var_alpha = float(alpha_pair.var(ddof=0))
    var_beta = float(beta_pair.var(ddof=0))
    var_bleu = float(bleu.var(ddof=0))

    chain_rows = []
    # Multi-BLiMP transforms
    for transform, (fn, _) in MBL_TRANSFORMS.items():
        r_a, _ = stats.pearsonr(alpha_vec, fn(a_src))
        r_b, _ = stats.pearsonr(beta_vec, fn(a_tgt))
        r2_chain = (r_a**2 * var_alpha + r_b**2 * var_beta) / var_bleu
        chain_rows.append({
            "subset": subset_name, "model": model, "predictor": "multiblimp",
            "transform": transform,
            "r_alpha": float(r_a), "r2_alpha": float(r_a**2),
            "r_beta": float(r_b), "r2_beta": float(r_b**2),
            "var_alpha_per_pair": var_alpha,
            "var_beta_per_pair": var_beta,
            "var_bleu_per_pair": var_bleu,
            "predicted_r2": float(r2_chain),
        })
    # PPL transforms
    for transform, (fn, _) in PPL_TRANSFORMS.items():
        r_a, _ = stats.pearsonr(alpha_vec, fn(p_src))
        r_b, _ = stats.pearsonr(beta_vec, fn(p_tgt))
        r2_chain = (r_a**2 * var_alpha + r_b**2 * var_beta) / var_bleu
        chain_rows.append({
            "subset": subset_name, "model": model, "predictor": "ppl",
            "transform": transform,
            "r_alpha": float(r_a), "r2_alpha": float(r_a**2),
            "r_beta": float(r_b), "r2_beta": float(r_b**2),
            "var_alpha_per_pair": var_alpha,
            "var_beta_per_pair": var_beta,
            "var_bleu_per_pair": var_bleu,
            "predicted_r2": float(r2_chain),
        })

    observed_rows = [
        {"subset": subset_name, "model": model, "predictor": "multiblimp",
         "form": "BLEU ~ acc_s + acc_t", "r2": fit_mbl_add["r2"],
         "p_coef_src": float(fit_mbl_add["p"][1]),
         "p_coef_tgt": float(fit_mbl_add["p"][2]),
         "coef_src": float(fit_mbl_add["beta"][1]),
         "coef_tgt": float(fit_mbl_add["beta"][2]),
         "n": n, "additive_re_r2": fit_re["r2_additive_centered"]},
        {"subset": subset_name, "model": model, "predictor": "multiblimp",
         "form": "BLEU ~ acc_s + acc_t + acc_s*acc_t", "r2": fit_mbl_int["r2"],
         "F_interaction": float(F_mbl), "p_F_interaction": float(p_F_mbl),
         "n": n, "additive_re_r2": fit_re["r2_additive_centered"]},
        {"subset": subset_name, "model": model, "predictor": "multiblimp",
         "form": "log BLEU ~ log(1-acc_s) + log(1-acc_t)", "r2": fit_mbl_logadd["r2"],
         "p_coef_src": float(fit_mbl_logadd["p"][1]),
         "p_coef_tgt": float(fit_mbl_logadd["p"][2]),
         "coef_src": float(fit_mbl_logadd["beta"][1]),
         "coef_tgt": float(fit_mbl_logadd["beta"][2]),
         "n": n, "additive_re_r2": fit_re["r2_additive_centered"]},
        {"subset": subset_name, "model": model, "predictor": "ppl",
         "form": "BLEU ~ P_s + P_t", "r2": fit_ppl_add["r2"],
         "p_coef_src": float(fit_ppl_add["p"][1]),
         "p_coef_tgt": float(fit_ppl_add["p"][2]),
         "coef_src": float(fit_ppl_add["beta"][1]),
         "coef_tgt": float(fit_ppl_add["beta"][2]),
         "n": n, "additive_re_r2": fit_re["r2_additive_centered"]},
        {"subset": subset_name, "model": model, "predictor": "ppl",
         "form": "BLEU ~ P_s + P_t + P_s*P_t", "r2": fit_ppl_int["r2"],
         "F_interaction": float(F_ppl), "p_F_interaction": float(p_F_ppl),
         "n": n, "additive_re_r2": fit_re["r2_additive_centered"]},
        {"subset": subset_name, "model": model, "predictor": "ppl",
         "form": "log BLEU ~ log P_s + log P_t", "r2": fit_ppl_logadd["r2"],
         "p_coef_src": float(fit_ppl_logadd["p"][1]),
         "p_coef_tgt": float(fit_ppl_logadd["p"][2]),
         "coef_src": float(fit_ppl_logadd["beta"][1]),
         "coef_tgt": float(fit_ppl_logadd["beta"][2]),
         "n": n, "additive_re_r2": fit_re["r2_additive_centered"]},
    ]

    return {"observed_rows": observed_rows, "corr_rows": corr_rows,
            "chain_rows": chain_rows, "langs": langs,
            "alphas": alpha, "betas": beta, "mu": mu,
            "r2_additive": fit_re["r2_additive_centered"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pair_csv)

    all_obs: List[Dict] = []
    all_corr: List[Dict] = []
    all_chain: List[Dict] = []
    re_summary: List[Dict] = []

    for subset_name, drop_eng in [("full_mbl", False), ("no_eng_mbl", True)]:
        for model in sorted(pairs["model"].unique()):
            out = analyze_subset(pairs, subset_name, model, drop_eng)
            all_obs.extend(out["observed_rows"])
            all_corr.extend(out["corr_rows"])
            all_chain.extend(out["chain_rows"])
            re_summary.append({"subset": subset_name, "model": model,
                               "n_langs": len(out["langs"]),
                               "r2_additive_centered": out["r2_additive"]})

    obs_df = pd.DataFrame(all_obs)
    corr_df = pd.DataFrame(all_corr)
    chain_df = pd.DataFrame(all_chain)
    obs_df.to_csv(args.out_dir / "observed_fits.csv", index=False)
    corr_df.to_csv(args.out_dir / "alpha_beta_vs_mbl.csv", index=False)
    chain_df.to_csv(args.out_dir / "chain_implied_r2.csv", index=False)
    pd.DataFrame(re_summary).to_csv(args.out_dir / "additive_re_on_subset.csv", index=False)

    # ---- Pretty-print key tables ----
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")

    print("\n=== Additive RE refit on each subset ===")
    print(pd.DataFrame(re_summary).to_string(index=False))

    print("\n=== Observed R^2: BLEU <- {Multi-BLiMP, PPL} ===")
    pivoted = obs_df.pivot_table(index=["subset", "model", "form"], values="r2").reset_index()
    print(pivoted.to_string(index=False))

    print("\n=== alpha/beta vs Multi-BLiMP correlations ===")
    print(corr_df[corr_df["predictor"] if "predictor" in corr_df.columns else slice(None)]
          .to_string(index=False) if False else corr_df.to_string(index=False))

    print("\n=== Chain-implied R^2 (Multi-BLiMP vs PPL on same subset) ===")
    cols = ["subset", "model", "predictor", "transform", "r2_alpha", "r2_beta",
            "predicted_r2"]
    print(chain_df[cols].to_string(index=False))

    print(f"\nWrote: {args.out_dir/'observed_fits.csv'}")
    print(f"Wrote: {args.out_dir/'alpha_beta_vs_mbl.csv'}")
    print(f"Wrote: {args.out_dir/'chain_implied_r2.csv'}")
    print(f"Wrote: {args.out_dir/'additive_re_on_subset.csv'}")


if __name__ == "__main__":
    main()
