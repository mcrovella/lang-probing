"""Compare competence proxies by how well they predict translation BLEU.

Proxies used to predict BLEU(src,tgt) via OLS
    BLEU ~ a + b_src * proxy[src] + b_tgt * proxy[tgt]:

  1. flores_ppl_per_token  (lower = better)  -- FLORES per-token corpus PPL
  2. bits_per_byte         (lower = better)  -- tokenization-invariant fluency
  3. multiblimp_accuracy   (higher = better) -- grammatical acceptability
  4. multiblimp_margin     (higher = better) -- mean correct-minus-wrong logprob margin

NOTE: "perplexity accuracy rate" (1 - PER) is *numerically identical* to
multiblimp_accuracy here (Pearson r = 1.000, max|diff| = 0.0) -- they are the
same grammatical judgment, computed the same way. We therefore report a single
grammatical-accuracy proxy and do not double-count it.

Every figure/table is labeled with the exact language set used.
Outputs in outputs/perplexity_bleu_linear/competence_proxy_comparison/.
"""
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from lang_probing_src.config import OUTPUTS_DIR

BASE = Path(OUTPUTS_DIR) / "perplexity_bleu_linear"
PPL_DIR = BASE / "bleu_and_ppl"
PER_DIR = BASE / "per"
OUT = BASE / "competence_proxy_comparison"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "llama"

PROXY_META = {
    "flores_ppl_per_token": {"higher_better": False, "label": "Per-token perplexity"},
    "bits_per_byte":        {"higher_better": False, "label": "Bits-per-byte"},
    "multiblimp_accuracy":  {"higher_better": True,  "label": "MultiBLiMP accuracy"},
    "multiblimp_margin":    {"higher_better": True,  "label": "MultiBLiMP logprob margin"},
}


def lc(p):
    return list(csv.DictReader(open(p)))


def load_proxies():
    P = {k: {} for k in PROXY_META}
    for r in lc(PPL_DIR / f"perplexity_results_{MODEL}.csv"):
        try:
            P["flores_ppl_per_token"][r["Language"].strip()] = float(r["Perplexity"])
        except ValueError:
            pass
    for r in lc(PPL_DIR / f"bits_per_byte_{MODEL}.csv"):
        P["bits_per_byte"][r["lang"]] = float(r["bits_per_byte"])
    for r in lc(BASE / "shareable_metrics" / "language_metrics.csv"):
        if r["model"] == MODEL and r.get("multiblimp_accuracy"):
            P["multiblimp_accuracy"][r["lang"]] = float(r["multiblimp_accuracy"])
    margin_candidates = sorted((BASE / "multiblimp_margins").glob("*/margins_by_lang.csv"))
    for path in margin_candidates:
        for r in lc(path):
            if r.get("lang") and r.get("margin_total"):
                P["multiblimp_margin"][r["lang"]] = float(r["margin_total"])
    return P


def load_perplexity_accuracy():
    """1 - PER; loaded only to verify identity with multiblimp_accuracy."""
    p = next(PER_DIR.glob("error_rates_by_language_*Llama*multiblimp.json"), None) \
        or next(PER_DIR.glob("error_rates_by_language_*Meta-Llama*multiblimp.json"), None)
    return {k: 1.0 - float(v) for k, v in json.load(open(p)).items()}


def load_bleu():
    out = []
    for r in lc(PPL_DIR / f"bleu_results_{MODEL}.csv"):
        try:
            out.append((r["src"].strip(), r["tgt"].strip(), float(r["bleu"])))
        except (ValueError, KeyError):
            pass
    return out


def fit(pairs, proxy, restrict=None, interaction=False):
    X, y = [], []
    for s, t, b in pairs:
        if restrict and (s not in restrict or t not in restrict):
            continue
        if s not in proxy or t not in proxy:
            continue
        row = [1.0, proxy[s], proxy[t]]
        if interaction:
            row.append(proxy[s] * proxy[t])
        X.append(row); y.append(b)
    X, y = np.asarray(X), np.asarray(y)
    n, p = len(y), X.shape[1] - 1
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ coef
    ss_res = float(((y - yh) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else float("nan")
    return {"n": n, "r2": r2, "adj_r2": adj, "mae": float(np.abs(y - yh).mean()),
            "coef": coef.tolist(), "y": y, "yhat": yh}


def marginal(pairs, proxy, restrict):
    by_t = {}
    for s, t, b in pairs:
        if restrict and (s not in restrict or t not in restrict):
            continue
        by_t.setdefault(t, []).append(b)
    langs = sorted(L for L in by_t if L in proxy)
    xv = np.array([proxy[L] for L in langs]); yv = np.array([np.mean(by_t[L]) for L in langs])
    sp = spearmanr(xv, yv)
    return langs, xv, yv, sp.correlation, sp.pvalue


def main():
    P = load_proxies()
    pairs = load_bleu()
    pacc = load_perplexity_accuracy()

    active_keys = [k for k in PROXY_META if P.get(k)]
    coverage = {k: sorted(P[k]) for k in active_keys}
    common = sorted(set.intersection(*[set(P[k]) for k in active_keys]))
    common_set = set(common)
    n_common_pairs = sum(1 for s, t, _ in pairs if s in common_set and t in common_set)
    common_str = ", ".join(common)
    print(f"coverage: {{k: len}} = {{ {', '.join(f'{k}:{len(v)}' for k,v in coverage.items())} }}")
    print(f"COMMON {len(common)} langs ({n_common_pairs} pairs): {common_str}")

    # --- identity check: perplexity-accuracy (1-PER) == multiblimp_accuracy ---
    shared = [L for L in common if L in pacc and L in P["multiblimp_accuracy"]]
    diff = max(abs(pacc[L] - P["multiblimp_accuracy"][L]) for L in shared)
    print(f"[identity] multiblimp_accuracy vs (1-PER): max|diff| over {len(shared)} langs = {diff:.2e}")

    # ---- per-proxy own vs common R^2 (for fig1 + report) ----
    summary = []
    for k in active_keys:
        meta = PROXY_META[k]
        own = fit(pairs, P[k]); com = fit(pairs, P[k], restrict=common_set)
        c = com["coef"]; want_pos = meta["higher_better"]
        sign_ok = (c[1] > 0) == want_pos and (c[2] > 0) == want_pos
        summary.append({"proxy": k, "label": meta["label"], "n_langs": len(coverage[k]),
                        "own_n_pairs": own["n"], "own_R2": own["r2"],
                        "common_n_pairs": com["n"], "common_R2": com["r2"],
                        "common_MAE": com["mae"], "b_src": c[1], "b_tgt": c[2],
                        "sign_matches_expectation": bool(sign_ok)})
    with (OUT / "proxy_predictive_power.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    # ---- predictive-power table (DRAFT): linear & bilinear MultiBLiMP, perplexity, BPB ----
    #      all on the common subset for an apples-to-apples comparison.
    table_specs = [
        ("MultiBLiMP accuracy -- linear",   "multiblimp_accuracy",   False),
        ("MultiBLiMP accuracy -- bilinear", "multiblimp_accuracy",   True),
        ("MultiBLiMP margin -- linear",     "multiblimp_margin",     False),
        ("Per-token perplexity -- linear",  "flores_ppl_per_token",  False),
        ("Bits-per-byte -- linear",         "bits_per_byte",         False),
    ]
    table = []
    for name, key, inter in table_specs:
        if key not in active_keys:
            continue
        r = fit(pairs, P[key], restrict=common_set, interaction=inter)
        table.append({"model": name, "R2": r["r2"], "adj_R2": r["adj_r2"],
                      "MAE": r["mae"], "n_pairs": r["n"]})
    with (OUT / "predictive_power_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    md = [f"Predictive power for BLEU, common {len(common)} languages "
          f"({common_str}); {n_common_pairs} ordered pairs.",
          "",
          "| Model | R² | adj. R² | MAE (BLEU) | n pairs |",
          "|---|---|---|---|---|"]
    for t in table:
        md.append(f"| {t['model']} | {t['R2']:.3f} | {t['adj_R2']:.3f} | {t['MAE']:.2f} | {t['n_pairs']} |")
    (OUT / "predictive_power_table.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))

    json.dump({"common_langs": common, "n_common_pairs": n_common_pairs,
               "coverage": coverage, "summary": summary, "table": table,
               "perplexity_accuracy_equals_multiblimp_max_diff": diff},
              open(OUT / "proxy_predictive_power.json", "w"), indent=2)

    # ---- Figure 1: R^2 bar chart (3 proxies; perplexity-accuracy dropped as duplicate) ----
    labels = [s["label"] for s in summary]
    own_r2 = [s["own_R2"] for s in summary]; com_r2 = [s["common_R2"] for s in summary]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w/2, own_r2, w, label="own coverage (24 langs / 552 pairs for fluency; 17 / 272 for MultiBLiMP)", color="#9ecae1")
    ax.bar(x + w/2, com_r2, w, label=f"common {len(common)} langs ({n_common_pairs} pairs)", color="#3182bd")
    for i, (o, c) in enumerate(zip(own_r2, com_r2)):
        ax.text(i - w/2, o + .004, f"{o:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, c + .004, f"{c:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("R²  (BLEU ~ proxy_src + proxy_tgt)")
    ax.set_title("Predictive power of competence proxies for translation BLEU\n"
                 "(MultiBLiMP accuracy = perplexity-accuracy = 1−PER; shown once)")
    ax.legend(fontsize=7.5); fig.tight_layout()
    fig.savefig(OUT / "fig1_r2_comparison.png", dpi=140); plt.close(fig)

    # ---- DRAFT figure: ONLY MultiBLiMP accuracy vs mean BLEU into language ----
    langs, xv, yv, rho, pv = marginal(pairs, P["multiblimp_accuracy"], common_set)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(xv, yv, s=40, color="#e6550d", zorder=3)
    for L, a, b in zip(langs, xv, yv):
        ax.annotate(L, (a, b), fontsize=8, xytext=(3, 3), textcoords="offset points")
    # OLS trend line
    m, c0 = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 50)
    ax.plot(xs, m * xs + c0, "k--", lw=1, alpha=.7)
    ax.set_xlabel("MultiBLiMP grammatical-acceptability accuracy")
    ax.set_ylabel("Mean BLEU when translating INTO the language")
    ax.set_title(f"MultiBLiMP accuracy vs translation-into BLEU\n"
                 f"{len(langs)} languages; Spearman ρ={rho:+.2f} (p={pv:.3f})")
    fig.tight_layout(); fig.savefig(OUT / "fig_multiblimp_marginal.png", dpi=150); plt.close(fig)
    print(f"\n[marginal] MultiBLiMP accuracy, {len(langs)} langs ({', '.join(langs)}): "
          f"Spearman rho={rho:+.3f} p={pv:.3f}")

    # ---- Figure 2: predicted vs actual (3 proxies, common subset) ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, k in zip(axes, active_keys[:3]):
        com = fit(pairs, P[k], restrict=common_set)
        ax.scatter(com["y"], com["yhat"], s=14, alpha=.6, color="#3182bd")
        lo, hi = min(com["y"].min(), com["yhat"].min()), max(com["y"].max(), com["yhat"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{PROXY_META[k]['label']}\nR²={com['r2']:.3f}", fontsize=10)
        ax.set_xlabel("actual BLEU"); ax.set_ylabel("predicted BLEU")
    fig.suptitle(f"Predicted vs actual BLEU, common {len(common)} langs ({n_common_pairs} pairs): {common_str}", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig2_pred_vs_actual.png", dpi=140); plt.close(fig)

    # ---- Figure 3: per-language marginal for all 3 proxies (report only) ----
    fig, axes = plt.subplots(1, min(4, len(active_keys)), figsize=(14, 4.6))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, active_keys[:len(axes)]):
        Ls, a, b, r, p = marginal(pairs, P[k], common_set)
        ax.scatter(a, b, s=24, color="#e6550d")
        for L, xx, yy in zip(Ls, a, b):
            ax.annotate(L, (xx, yy), fontsize=7, alpha=.8)
        ax.set_xlabel(PROXY_META[k]["label"]); ax.set_ylabel("mean BLEU into language")
        ax.set_title(f"{k}: Spearman ρ={r:+.2f} (p={p:.3f})", fontsize=10)
    fig.suptitle(f"Per-language proxy vs mean BLEU into language, common {len(common)} langs ({n_common_pairs} pairs)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig3_marginal.png", dpi=140); plt.close(fig)

    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
