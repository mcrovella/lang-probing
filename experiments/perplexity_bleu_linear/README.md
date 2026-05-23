# `perplexity_bleu_linear`

**Hypotheses:** H1 — BLEU predictable from monolingual source and target competence, without a substantive language-pair-specific interaction term.

**Question:** How much of pairwise BLEU variance is explained by per-language competence proxies? Is there a residual language-pair-specific interaction?

**Method:** Two stages. (1) Generate the per-language metric data: compute FLORES corpus perplexity per language per model and Multi-BLiMP accuracy per language per model; join with BLEU. (2) Decompose the BLEU matrix into per-language additive effects $\alpha, \beta$ via a random-effects model; correlate $\alpha, \beta$ with monolingual competence proxies; fit BLEU directly from proxies; test for source–target interaction terms.

## Paper analysis (no GPU required)

This is the analysis layer the [paper draft](paper.pdf) and [research log](research_log.pdf) are built from. It runs locally against the precomputed `data/pair_metrics.csv` joined-metric file and produces all numerical claims in the paper.

### One-shot reproduction

```bash
./experiments/perplexity_bleu_linear/reproduce_paper.sh
```

Runs nine scripts in dependency order in ~10s. Writes to `outputs/perplexity_bleu_linear/<experiment>/` and `img/perplexity_bleu_linear/`. Prints a mapping from paper sections/appendices to the specific CSV result files at the end.

### Paper draft and research log

- **[`paper.tex`](paper.tex) / [`paper.pdf`](paper.pdf)** — 10-page paper-style writeup with main body + appendix.
- **[`research_log.tex`](research_log.tex) / [`research_log.pdf`](research_log.pdf)** — 24-page chronological log of the 10-experiment chain that produced the paper findings.

### Headline findings

- Additive random-effects model `BLEU_ij = μ + α_i + β_j` explains 93% (Llama) / 88% (Aya) of centered BLEU variance on the 23-language no-English subset.
- Multiplicative (rank-1 SVD) decomposition fits comparably on the same parameter budget but does not outperform additive on the apples-to-apples comparison; AMMI shows the multiplicative residual interaction is small (+3–5 pp).
- **Multi-BLiMP accuracy is a 3–6× stronger BLEU predictor than FLORES corpus PPL** on the same 17-language subset (R² up to 0.44 vs 0.13).
- **Pair-specific interaction terms are not significant** under any (subset, predictor, model, transform) tested — supports H1.

### Analysis scripts

In the order `reproduce_paper.sh` runs them:

1. **[`additive_random_effects.py`](additive_random_effects.py)** — fit additive RE model `μ + α_i + β_j` on full 24-lang set; report centered R² vs rank-1 SVD and AMMI.
2. **[`alpha_beta_vs_ppl.py`](alpha_beta_vs_ppl.py)** — correlate recovered α, β with FLORES PPL transforms; emit annotated scatter plots.
3. **[`src_tgt_asymmetry.py`](src_tgt_asymmetry.py)** — variance decomposition of BLEU across src vs tgt axes; per-pair vs per-language correlations.
4. **[`interaction_test.py`](interaction_test.py)** — nested-model F-test for `P_s × P_t` interaction in `BLEU ~ P_s + P_t [+ P_s × P_t]`.
5. **[`log_multiplicative.py`](log_multiplicative.py)** — fit `log BLEU ~ log P_s + log P_t` (the log-additive analog of multiplicative BLEU).
6. **[`drop_english_ablation.py`](drop_english_ablation.py)** — refit additive RE without English; recompute α, β vs PPL correlations; side-by-side comparison with full-set.
7. **[`ppl_to_bleu_no_eng.py`](ppl_to_bleu_no_eng.py)** — BLEU ~ PPL fits on the 23-language no-English subset, plus a chain-implied R² check.
8. **[`bleu_to_multiblimp.py`](bleu_to_multiblimp.py)** — Multi-BLiMP analyses on the 17-language Multi-BLiMP-available subset (and its no-English sub-subset): α, β correlations, direct BLEU prediction, interaction-term tests; PPL baselines on the same subset for fair comparison.
9. **[`_one_shot_no_eng_decomposition.py`](_one_shot_no_eng_decomposition.py)** — additive vs multiplicative vs AMMI decomposition on the 23-lang no-eng subset (the apples-to-apples numbers used in Section 3 of the paper).

Also present but not part of the paper pipeline: **[`rank1_latent_regression.py`](rank1_latent_regression.py)** — early exploration that correlated rank-1 SVD latents with PPL; superseded by `alpha_beta_vs_ppl.py` (the additive coefficients and the SVD latents agree closely when the matrix is mostly additive).

## Data generation pipeline (requires GPU)

The original pipeline that computes the inputs to the analysis above. Used to generate `pair_metrics.csv` (and earlier `combined_results_{model}.csv`). Requires loading Llama-3.1-8B and Aya-23-8B; meant for the BU SCC.

```bash
# 1. corpus perplexity per language per model
python experiments/perplexity_bleu_linear/run_perplexity.py --model_id llama --batch_size 8
python experiments/perplexity_bleu_linear/run_perplexity.py --model_id aya   --batch_size 8

# 2. perplexity error rate on Multi-BLiMP minimal pairs
python experiments/perplexity_bleu_linear/run_per.py --multilang --languages English Spanish French German Turkish Hebrew Hindi Chinese Indonesian

# 3. join BLEU + PPL
python experiments/perplexity_bleu_linear/combine_csvs.py

# 4. fit the linear model
python experiments/perplexity_bleu_linear/run_linear_fit.py --feature-transform raw --include-interaction yes
python experiments/perplexity_bleu_linear/run_linear_fit.py --feature-transform log --include-interaction no

# 5. visualize
python experiments/perplexity_bleu_linear/visualize_correlation.py
python experiments/perplexity_bleu_linear/visualize_error_bar.py

# 6. rank-1 SVD faithfulness of the BLEU matrix (the "88%" claim)
python experiments/perplexity_bleu_linear/rank1_approximation.py --model llama
python experiments/perplexity_bleu_linear/rank1_approximation.py --model aya
```

### Rank-1 SVD result (added 2026-04-22)

- **Llama:** rank-1 faithfulness = **88.31%**
- **Aya:** rank-1 faithfulness = 82.37%

Faithfulness = 1 − ‖M − M₁‖_F / ‖M‖_F where M is the (src, tgt) BLEU matrix. Produces `img/perplexity_bleu_linear/linear_effects_ranks_{model}.png` (error-vs-rank curve) and `linear_effects_{model}.png` (rank-1 predicted vs actual scatter). The rank-1 matrix factors into a src-competence vector outer-producted with a tgt-competence vector, which is the core H1 story despite the in-PER-space linear model's low R².

## Inputs

- FLORES devtest (HuggingFace `gsarti/flores_101`).
- Multi-BLiMP (HuggingFace `jumelet/multiblimp`).
- External BLEU scores (joined per-language-pair).
- Models: Llama-3.1-8B, Aya-23-8B.

## Outputs

- `outputs/perplexity_bleu_linear/bleu_and_ppl/perplexity_results_{model}.csv` — raw PPL per language (tiny pilot files).
- `outputs/perplexity_bleu_linear/bleu_and_ppl/combined_results_{model}.csv` — joined BLEU + PPL (current source of truth, ~28 KB).
- `outputs/perplexity_bleu_linear/per/error_rates_by_language_{model}.json` — PER per language.
- `outputs/perplexity_bleu_linear/per/perplexity_matrices_{model}.npz`.
- `outputs/perplexity_bleu_linear/bleu_and_ppl/linear_models/linear_coeffs_{model}_{raw|log}_{joint|nojoint}.csv`.
- `outputs/perplexity_bleu_linear/bleu_and_ppl/linear_models/linear_predictions_*.csv`.

## Figures

- `img/perplexity_bleu_linear/{aya,llama}_{source,target,joint}_competence*.png` — current H1 figures.
- `img/perplexity_bleu_linear/perplexity_plot_{model}.png` (+ `_sorted`).
- `img/perplexity_bleu_linear/perplexity_vs_bleu_{model}_sorted.png`.

## Known caveats

- The earlier-reported "linear-in-PPL R² is 0.02 (Llama), 0.10 (Aya)" was on the full 24-language set with English included. English is a high-leverage positive outlier in FLORES PPL space: removing it raises Llama's R² to 0.114 (see `ppl_to_bleu_no_eng.py` and `paper.tex` Appendix A). The remaining ~12% ceiling for PPL → BLEU is the proxy quality, not a code defect.
- Multi-BLiMP accuracy is a substantially stronger competence proxy than FLORES PPL (R² up to 0.44 for direct BLEU prediction) — but is only available for 17 of 24 languages.
- `combine_csvs.py` has hardcoded `/projectnb/mcnet/jbrin/...` paths from JB's checkout; will fail on other hosts and should use `OUTPUTS_DIR` from `config`. (Not on the paper pipeline's critical path — `pair_metrics.csv` is read directly.)

## Status

Active. Paper draft and reproducibility script in place ([`paper.tex`](paper.tex), [`reproduce_paper.sh`](reproduce_paper.sh)). See [LEDGER.md](../../LEDGER.md#perplexity_bleu_linear).
