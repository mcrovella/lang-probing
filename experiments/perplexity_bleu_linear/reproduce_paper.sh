#!/usr/bin/env bash
# Reproduce all numerical results in paper.tex by running the analysis
# scripts in dependency order. Outputs land in
# outputs/perplexity_bleu_linear/<experiment>/ and
# img/perplexity_bleu_linear/.
#
# Usage: ./experiments/perplexity_bleu_linear/reproduce_paper.sh
#        (or from this directory: ./reproduce_paper.sh)
#
# Requires: python3 with pandas, numpy, scipy, matplotlib. No GPU or
# model inference; all inputs are precomputed in data/pair_metrics.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

run_step() {
    local n="$1"; local title="$2"; local script="$3"
    echo
    echo "================================================================"
    echo "[$n] $title"
    echo "    python3 $script"
    echo "================================================================"
    python3 "$script"
}

# Order matters: some scripts read outputs written by earlier ones
# (e.g. drop_english_ablation.py reads alpha_beta_vs_ppl/correlations.csv,
# ppl_to_bleu_no_eng.py reads drop_english_ablation/ and the full-set
# interaction_test/log_multiplicative outputs).

run_step "1/9" "Full-set additive RE: alphas, betas, R^2 comparison" \
    experiments/perplexity_bleu_linear/additive_random_effects.py

run_step "2/9" "alpha/beta vs FLORES PPL on the full 24-lang set" \
    experiments/perplexity_bleu_linear/alpha_beta_vs_ppl.py

run_step "3/9" "Source/target asymmetry diagnostics" \
    experiments/perplexity_bleu_linear/src_tgt_asymmetry.py

run_step "4/9" "Full-set BLEU ~ PPL with-vs-without interaction (Exp 1)" \
    experiments/perplexity_bleu_linear/interaction_test.py

run_step "5/9" "Full-set log BLEU ~ log P + log P fit (Exp 3)" \
    experiments/perplexity_bleu_linear/log_multiplicative.py

run_step "6/9" "Drop-English ablation: refit additive RE + correlations" \
    experiments/perplexity_bleu_linear/drop_english_ablation.py

run_step "7/9" "BLEU ~ PPL on no-eng subset + chain check" \
    experiments/perplexity_bleu_linear/ppl_to_bleu_no_eng.py

run_step "8/9" "Multi-BLiMP analyses on 17-lang and 16-lang subsets" \
    experiments/perplexity_bleu_linear/bleu_to_multiblimp.py

run_step "9/9" "Additive vs multiplicative on 23-lang no-eng subset" \
    experiments/perplexity_bleu_linear/_one_shot_no_eng_decomposition.py

echo
echo "================================================================"
echo "All steps complete."
echo "================================================================"
echo
echo "Outputs:  outputs/perplexity_bleu_linear/"
echo "Figures:  img/perplexity_bleu_linear/"
echo
echo "Mapping of paper sections to result files:"
echo
echo "  Section 3 (additive vs multiplicative on 23-lang no-eng):"
echo "    outputs/perplexity_bleu_linear/additive_random_effects/no_eng_decomposition.csv"
echo
echo "  Section 4 (alpha/beta correlations with PPL and Multi-BLiMP):"
echo "    outputs/perplexity_bleu_linear/bleu_to_multiblimp/alpha_beta_vs_mbl.csv"
echo "    outputs/perplexity_bleu_linear/bleu_to_multiblimp/chain_implied_r2.csv"
echo "      (the PPL-predictor rows of chain_implied_r2.csv hold r_alpha and r_beta)"
echo
echo "  Section 5 (direct BLEU ~ proxy fits on 17-lang subset):"
echo "    outputs/perplexity_bleu_linear/bleu_to_multiblimp/observed_fits.csv"
echo
echo "  Section 6 (source/target asymmetry, variance ratios):"
echo "    outputs/perplexity_bleu_linear/src_tgt_asymmetry/diagnostics.csv"
echo "    outputs/perplexity_bleu_linear/bleu_to_multiblimp/alpha_beta_vs_mbl.csv"
echo
echo "  Section 7 (interaction term F-tests):"
echo "    outputs/perplexity_bleu_linear/bleu_to_multiblimp/observed_fits.csv"
echo "      (rows with form containing '*' have F_interaction / p_F_interaction)"
echo
echo "  Appendix A (English-outlier evidence):"
echo "    outputs/perplexity_bleu_linear/alpha_beta_vs_ppl/correlations.csv"
echo "    outputs/perplexity_bleu_linear/drop_english_ablation/comparison_with_full.csv"
echo "    outputs/perplexity_bleu_linear/ppl_to_bleu_no_eng/comparison_with_full.csv"
echo
echo "  Appendix B (full per-language correlation tables on 17-lang subset):"
echo "    outputs/perplexity_bleu_linear/bleu_to_multiblimp/alpha_beta_vs_mbl.csv"
echo "    outputs/perplexity_bleu_linear/bleu_to_multiblimp/chain_implied_r2.csv"
echo
echo "  Appendix D (PPL on 23-lang no-eng subset):"
echo "    outputs/perplexity_bleu_linear/drop_english_ablation/correlations_no_eng.csv"
echo "    outputs/perplexity_bleu_linear/ppl_to_bleu_no_eng/observed_fits.csv"
echo
echo "  Appendix E (per-coefficient details for BLEU ~ proxy fits):"
echo "    outputs/perplexity_bleu_linear/bleu_to_multiblimp/observed_fits.csv"
echo "      (coef_src, coef_tgt, p_coef_src, p_coef_tgt columns)"
