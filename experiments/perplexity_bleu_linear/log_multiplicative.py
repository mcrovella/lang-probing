"""Log-multiplicative fit: log(BLEU) ~ a + b*log(P_s) + c*log(P_t).

The rank-1 SVD story is multiplicative: BLEU(s, t) ~ u_s * v_t. The natural
linear-model translation is therefore additive in log space:

    log(BLEU) = a + b * log(P_s) + c * log(P_t) + epsilon
    equivalently BLEU = exp(a) * P_s^b * P_t^c

This script fits exactly that form per model. The previous interaction test
fit BLEU itself (additive + multiplicative interaction), which is the wrong
functional form for a multiplicative latent story. This is the right one.

Outputs:
    outputs/perplexity_bleu_linear/log_multiplicative/
        fit_summary.csv         # per-model R^2, F-test, coefs
        coefficients.csv        # term-level coef/SE/t/p
        predictions_{model}.csv # per-pair actual + predicted on both scales
    img/perplexity_bleu_linear/
        log_multiplicative_{model}.png    # 2-panel: log-scale + back-transformed
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
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "perplexity_bleu_linear" / "log_multiplicative"
DEFAULT_IMG_DIR = REPO_ROOT / "img" / "perplexity_bleu_linear"


def fit_ols(X: np.ndarray, y: np.ndarray) -> Dict:
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    dof_resid = n - p
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof_resid if dof_resid > 0 else float("nan")
    sigma2 = rss / dof_resid
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    t = beta / se
    pval = 2.0 * stats.t.sf(np.abs(t), df=dof_resid)
    # F-test vs intercept-only
    f_stat = ((tss - rss) / (p - 1)) / (rss / dof_resid)
    f_p = float(stats.f.sf(f_stat, p - 1, dof_resid))
    return {
        "beta": beta, "se": se, "t": t, "p": pval, "rss": rss, "tss": tss,
        "r2": r2, "adj_r2": adj_r2, "dof_resid": dof_resid, "n": n,
        "F_stat": float(f_stat), "F_p": f_p, "F_df1": p - 1, "F_df2": dof_resid,
    }


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["bleu", "src_flores_perplexity", "tgt_flores_perplexity"]).copy()
    sub = sub[(sub["bleu"] > 0) & (sub["src_flores_perplexity"] > 0)
              & (sub["tgt_flores_perplexity"] > 0)].copy()
    sub["log_bleu"] = np.log(sub["bleu"].to_numpy(dtype=float))
    sub["log_src"] = np.log(sub["src_flores_perplexity"].to_numpy(dtype=float))
    sub["log_tgt"] = np.log(sub["tgt_flores_perplexity"].to_numpy(dtype=float))
    return sub


def plot_predictions(model: str, sub: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    lo = min(sub["log_bleu"].min(), sub["log_bleu_pred"].min()) - 0.1
    hi = max(sub["log_bleu"].max(), sub["log_bleu_pred"].max()) + 0.1
    ax.scatter(sub["log_bleu"], sub["log_bleu_pred"], alpha=0.4, s=14)
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.7, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("log(BLEU) actual")
    ax.set_ylabel("log(BLEU) predicted")
    ax.set_title(f"Log-scale fit ({model})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    lo = min(sub["bleu"].min(), sub["bleu_pred"].min()) - 0.5
    hi = max(sub["bleu"].max(), sub["bleu_pred"].max()) + 0.5
    ax.scatter(sub["bleu"], sub["bleu_pred"], alpha=0.4, s=14)
    ax.plot([lo, hi], [lo, hi], "r--", alpha=0.7, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("BLEU actual")
    ax.set_ylabel("BLEU predicted (back-transformed)")
    ax.set_title(f"BLEU-scale fit ({model})")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"log(BLEU) = a + b log P_s + c log P_t   ({model})", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_model(df_model: pd.DataFrame, model: str,
                  out_dir: Path, img_dir: Path) -> Tuple[Dict, pd.DataFrame]:
    sub = prepare(df_model)
    n_total = len(df_model)
    n_used = len(sub)

    X = np.column_stack([np.ones(n_used), sub["log_src"].to_numpy(),
                         sub["log_tgt"].to_numpy()])
    y = sub["log_bleu"].to_numpy()
    fit = fit_ols(X, y)

    sub["log_bleu_pred"] = X @ fit["beta"]
    sub["bleu_pred"] = np.exp(sub["log_bleu_pred"])

    # Back-transformed R^2 on BLEU scale
    bleu = sub["bleu"].to_numpy()
    bleu_pred = sub["bleu_pred"].to_numpy()
    ss_res_bt = float(((bleu - bleu_pred) ** 2).sum())
    ss_tot_bt = float(((bleu - bleu.mean()) ** 2).sum())
    r2_bt = 1.0 - ss_res_bt / ss_tot_bt if ss_tot_bt > 0 else float("nan")

    sub.to_csv(out_dir / f"predictions_{model}.csv", index=False)

    summary = {
        "model": model,
        "n_total_pairs": n_total,
        "n_used": n_used,
        "r2_log_scale": fit["r2"],
        "adj_r2_log_scale": fit["adj_r2"],
        "r2_bleu_scale_backtransformed": r2_bt,
        "F_stat": fit["F_stat"],
        "F_df1": fit["F_df1"],
        "F_df2": fit["F_df2"],
        "F_p_value": fit["F_p"],
        "coef_intercept": float(fit["beta"][0]),
        "coef_log_src_ppl": float(fit["beta"][1]),
        "coef_log_tgt_ppl": float(fit["beta"][2]),
        "se_log_src_ppl": float(fit["se"][1]),
        "se_log_tgt_ppl": float(fit["se"][2]),
        "t_log_src_ppl": float(fit["t"][1]),
        "t_log_tgt_ppl": float(fit["t"][2]),
        "p_log_src_ppl": float(fit["p"][1]),
        "p_log_tgt_ppl": float(fit["p"][2]),
    }

    coef_rows = []
    for i, term in enumerate(["intercept", "log_src_ppl", "log_tgt_ppl"]):
        coef_rows.append({"model": model, "term": term,
                          "coef": float(fit["beta"][i]), "se": float(fit["se"][i]),
                          "t": float(fit["t"][i]), "p": float(fit["p"][i])})

    plot_predictions(model, sub, img_dir / f"log_multiplicative_{model}.png")

    print(f"\n[{model}] n_used = {n_used}/{n_total}")
    print(f"[{model}] log(BLEU) = {fit['beta'][0]:+.3f} "
          f"+ ({fit['beta'][1]:+.3f}) log P_s "
          f"+ ({fit['beta'][2]:+.3f}) log P_t")
    print(f"[{model}] R^2 (log scale)        = {fit['r2']:.4f}")
    print(f"[{model}] R^2 (BLEU back-trans.)  = {r2_bt:.4f}")
    print(f"[{model}] F({fit['F_df1']},{fit['F_df2']}) = {fit['F_stat']:.2f}, "
          f"p = {fit['F_p']:.3g}")
    print(f"[{model}] per-coef t: log_P_s = {fit['t'][1]:+.2f} (p={fit['p'][1]:.3g})  "
          f"log_P_t = {fit['t'][2]:+.2f} (p={fit['p'][2]:.3g})")

    return summary, pd.DataFrame(coef_rows)


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
    summaries: List[Dict] = []
    coef_frames: List[pd.DataFrame] = []
    for model in sorted(pairs["model"].unique()):
        df_model = pairs[pairs["model"] == model]
        summary, coefs = analyze_model(df_model, model, args.out_dir, args.img_dir)
        summaries.append(summary)
        coef_frames.append(coefs)

    pd.DataFrame(summaries).to_csv(args.out_dir / "fit_summary.csv", index=False)
    pd.concat(coef_frames, ignore_index=True).to_csv(
        args.out_dir / "coefficients.csv", index=False
    )
    print(f"\nWrote: {args.out_dir/'fit_summary.csv'}")
    print(f"Wrote: {args.out_dir/'coefficients.csv'}")


if __name__ == "__main__":
    main()
