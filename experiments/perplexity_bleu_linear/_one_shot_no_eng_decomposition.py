"""One-shot helper: rerun the additive-vs-multiplicative decomposition
comparison on the 23-language no-English subset. Numbers used by Section
(a) of the paper-style writeup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from additive_random_effects import (
    build_bleu_matrix, fit_additive_ols, evaluate_decompositions,
)


def main() -> None:
    df = pd.read_csv("/Users/markcrovella/repos/lang-probing/data/pair_metrics.csv")
    df_no_eng = df[(df["src"] != "eng") & (df["tgt"] != "eng")].copy()

    rows = []
    for model in ["aya", "llama"]:
        df_m = df_no_eng[df_no_eng["model"] == model]
        M, langs = build_bleu_matrix(df_m)
        mu, alphas, betas, _ = fit_additive_ols(df_m, langs)
        metrics = evaluate_decompositions(M, None, mu, alphas, betas, langs)
        metrics["model"] = model
        metrics["n_langs"] = len(langs)
        rows.append(metrics)

    out = pd.DataFrame(rows)
    keep = ["model", "n_langs", "n_obs_cells", "r2_additive_centered",
            "r2_rank1_on_centered_matrix", "r2_ammi_centered",
            "faithfulness_rank1_uncentered"]
    print(out[keep].to_string(index=False))
    out_path = Path("/Users/markcrovella/repos/lang-probing/outputs/perplexity_bleu_linear/additive_random_effects/no_eng_decomposition.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
