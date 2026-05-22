# PAPER_TODO resolution report

Date: 2026-05-22

This pass prioritized the `PAPER_TODO.md` tags in the order requested: FIG,
P2, APPENDIX, UNCERTAINTY, and REPRO. I regenerated the figures and resolved
the checks that could be settled from existing outputs. Items that require new
GPU compute are queued or explicitly flagged below.

## Executive status

- Figure pass: regenerated H1/rank-1, GCM, head-ablation, and H2 overlap plots
  from the current repo outputs.
- P2 pass: added missing diagnostics for rank-1 residuals/leave-one-language-out,
  GCM signed IE, GCM sparsity, grammar-feature provenance, and H2 aggregate
  overlap.
- Appendix pass: `reports/draft_v5.tex` now includes the mean-vs-zero head
  ablation figure and English-to-Spanish GCM method heatmaps.
- Uncertainty pass: the draft now hedges the H2 overlap claim and the report
  below records the remaining scientific blockers.
- Repro pass: pinned head-ablation reproduction details in this report and
  added/updated scripts so the next reruns are less ambiguous.

## Draft changes

Updated `reports/draft_v5.tex`:

- Replaced the H1 rank-1 figure with the paired raw BLEU/rank-1 heatmap from
  the same 24 by 24 pivot used by the rank-1 computation.
- Added a signed universal-head reuse plot for GCM.
- Added explicit H2 limitation text: 25 usable cells, mean Jaccard@200 = 0.095
  raw and 0.122 absolute.
- Added appendix figures for mean-vs-zero ablation and English-to-Spanish GCM
  heatmaps.
- Shortened a long appendix path note that caused an overfull line in the PDF.

`reports/draft_v5.pdf` was rebuilt successfully.

## A1: linear model / rank-1 BLEU

Changed `experiments/perplexity_bleu_linear/rank1_approximation.py`:

- Added raw BLEU heatmaps over the exact matrix used by the rank-1 SVD.
- Added paired actual-vs-rank1 heatmaps.
- Added residual heatmaps.
- Added leave-one-language-out summaries.
- Added matrix metadata JSON with source CSV, dimensions, missing cells, and
  imputation details.

Generated outputs:

- `img/perplexity_bleu_linear/bleu_matrix_llama.png`
- `img/perplexity_bleu_linear/bleu_matrix_rank1_llama.png`
- `img/perplexity_bleu_linear/rank1_residual_heatmap_llama.png`
- Aya equivalents for the three figures.
- `outputs/perplexity_bleu_linear/bleu_and_ppl/rank1/matrix_summary_llama.json`
- `outputs/perplexity_bleu_linear/bleu_and_ppl/rank1/leave_one_language_out_llama.csv`
- Aya equivalents for summary and leave-one-language-out.

Important numbers:

- Llama rank-1 faithfulness: 0.8831.
- Aya rank-1 faithfulness: 0.8237.
- Both matrices are 24 by 24 with 24 missing/self cells filled by target-column
  means for SVD.
- Dropping English or Turkish slightly raises Llama faithfulness; dropping
  Hebrew gives 0.8818. The high rank-1 score is not driven by Turkish or Hebrew
  as single outliers.

Scientific note:

- The source-of-truth CSV for these fits is now documented as
  `combined_results_{model}.csv`. External BLEU provenance and decoding setup
  are still not recoverable from this repo, so the final paper should either
  cite the upstream generation source or hedge this clearly.
- Mean-imputing the diagonal/self cells can inflate low-rank structure. I added
  diagnostics and metadata, but a true masked-SVD/observed-cell objective is
  still the cleaner final check if time allows.

## A2: GCM heads and SAE features

Changed `experiments/gcm_translation/analyze.py`:

- Added `--include_same_lang`.
- Default analysis now excludes same-language directories, restoring the intended
  56 cross-language directions.

Changed `experiments/gcm_translation/meeting_plots.py`:

- Retitled the universal-head plot to state top-20/K=20.
- Added signed universal-head reuse plot from raw signed mean IE.
- Added SAE real-vs-null/content-floor scatter from the three-way summary.

Added `experiments/gcm_translation/summarize_ie_sparsity.py`:

- Computes per-direction top-head mean absolute IE divided by median-head mean
  absolute IE.

Regenerated:

- `experiments/gcm_translation/img/meeting_direction_by_universal_head.png`
- `experiments/gcm_translation/img/meeting_direction_by_universal_head_signed.png`
- `experiments/gcm_translation/img/meeting_sae_real_vs_null_same.png`
- `experiments/gcm_translation/img/meeting_sae_grammar_overlap.png`
- `experiments/gcm_translation/img/three_way_decomposition_sae_full.png`
- `experiments/gcm_translation/img/head_ie_top_median_ratio.png`
- Per-direction head heatmaps, signed heatmaps, top-head bars, and top-SAE bars.

Important numbers:

- Universal-head counts are computed over 56 cross-language directions.
- L17H21 and L30H25 appear in 56/56 directions under top-20 counting.
- Head-level translation-circuit contributions remain tiny in the null-control
  decomposition: the top head is about +0.025 nat.
- SAE f31444 has a much cleaner circuit signal: real = 0.0741,
  null_cross = 0.0380, null_same = 0.0004, translation-circuit = +0.0361.
- GCM head IE is sparse: top/median mean absolute IE across directions ranges
  from 21.8 to 62.9, median 34.6.
- Grammar-feature provenance: the 218 "grammar features" are the union of the
  seven concept-level top-50 `top_50_by_mean_grad_x_act` lists in
  `outputs/counterfactual_attribution/aggregated_by_concept.json`. The GCM plot
  compares this 218-feature union against the top 50 universal GCM SAE features;
  overlap is 9/50.

Red-team C1-C6 check:

- `gcm_core.py` contains the two-trace pattern, `o_proj` output delta, fp32
  leaves, token-space truncation, and trailing-space response tokenization path.
- `flores_pairs.py` builds prompts ending in `"{tgt_lang}: "`.
- `run.py` has the sm_120 GPU guard and per-component error handling.

Remaining scientific blocker:

- The direct finite-difference linearization test is still marked `xfail`.
- ACP-vs-ATP interventional faithfulness figures are still not run on the sweep.
  Do not claim the gradient ranking is interventional without this, or phrase it
  as a ranking used to choose candidates rather than a validated causal ranking.
- The paper should foreground SAE features, not universal-head counts. The head
  universality result is real as a descriptive statistic, but the null control
  makes the head-level translation-circuit story weak.

## A3: head ablation on Multi-BLiMP

Regenerated:

- `outputs/head_ablation_multiblimp/analysis_summary.csv`
- `experiments/head_ablation_multiblimp/img/cross_lang_sign_split.png`
- `experiments/head_ablation_multiblimp/img/cross_lang_mean_vs_zero.png`
- Per-language bars, mean-vs-zero plots, rank/effect scatters, and sign-split
  plots under `experiments/head_ablation_multiblimp/img/`.

Changed `experiments/head_ablation_multiblimp/visualize.py`:

- Matched-control plots are now only written when matched controls are present.
  This prevents misleading non-English `*_sign_split_matched.png` figures made
  from missing values.

Changed `experiments/head_ablation_multiblimp/submit_v2.qsub`:

- Pinned the job to A100 GPUs.

Added `experiments/head_ablation_multiblimp/submit_v2_missing.qsub`:

- Queues the seven missing languages (`ara deu fra heb hin spa tur`) without
  overwriting the completed English v2 output.
- Uses `--n_ctrl_seeds 5`, `--max_pairs 400`, A100, and gpu compute capability
  8.0.

Cluster status:

- Submitted job array `5772145.1-7` (`habl_v2_missing`).
- As of the last check, task 1 is running on an A100 node and tasks 2-7 are
  queued.

Current data status:

- English has 53 conditions, n=400, and matched POS/NEG controls.
- Arabic, German, French, Spanish, and Turkish currently have 46 conditions,
  n=400, without matched controls.
- Hebrew and Hindi currently have 46 conditions, n=200, without matched
  controls. The queued v2 rerun should bring them to n=400.
- The Hindi random-control anomaly remains present in current outputs.

Important current deltas:

- English: POS +0.0752, NEG -0.1604, matched POS control -0.0280, matched NEG
  control +0.0320.
- Arabic: POS +0.1105, NEG -0.4675.
- German: POS +0.0493, NEG -1.0215.
- Hindi: POS +0.2197, NEG -0.2728, random control +0.2306.

Remaining blocker:

- Cross-language matched controls are not complete until job `5772145` finishes
  and `analyze.py`/`visualize.py` are rerun.
- Dose-response over k = {1, 5, 10, 20} is still not generated. The current
  scripts can run one `--top_k_collective` value per job; a convenience loop or
  multi-k condition emitter should be added before claiming dose response.

Reproduction details to preserve in the final paper:

- Head selection: top-20 heads by aggregated mean absolute GCM IE over the seven
  directions into the target language.
- Sign split: POS/NEG by signed mean IE over those incoming directions.
- Mean-activation cache: position-averaged per target language.
- Ablation mechanism: add `(replacement - original) @ W_O.T` at `o_proj.output`
  for the selected head slice.
- Conditions: 46 baseline/current conditions; 53 when matched controls are
  present.
- Data: `data/multilingual_pairs/{lang}.json`.
- Scoring: final-token-only log-probability margin on Multi-BLiMP minimal pairs.
- Current intended rerun hardware: A100 class GPU, avoiding Blackwell sm_120.

## B2: input/output feature overlap

Changed `src/lang_probing_src/io/effects.py`:

- Added an indexer that selects the latest timestamped output-effect file per
  `(source, target)` pair.
- Added a metadata helper so the overlap visualizer does not load every effect
  tensor into memory.

Changed `experiments/input_output_overlap/visualize.py`:

- Added raw and absolute top-k Jaccard curves per cell.
- Added aggregate raw and absolute Jaccard plots.
- Added summary CSV output.
- Skips missing input vectors and malformed output vectors instead of crashing.

Regenerated:

- `img/input_output_overlap/aggregate_jaccard_topk_raw.png`
- `img/input_output_overlap/aggregate_jaccard_topk_abs.png`
- Per-cell `jaccard_topk.png` and `jaccard_topk_abs.png`.
- `outputs/input_output_overlap/jaccard_summary.csv`.

Important numbers:

- 25 usable cells.
- Mean Jaccard@50: raw 0.0422, absolute 0.0723.
- Mean Jaccard@100: raw 0.0621, absolute 0.0863.
- Mean Jaccard@200: raw 0.0947, absolute 0.1220.
- English Number=Plur Jaccard@200: raw 0.1173, absolute 0.1050.

Scientific note:

- English plural is not uniquely low; the broader input/output overlap signal is
  modest.
- H2 should not be a strong headline claim in the final paper unless the missing
  cells are repaired and the aggregate overlap improves or is reframed.
- Missing/malformed cells found during the pass include missing input diff
  vectors for several Dual/Sing/Tense cells, Arabic Tense missing from output
  effects, and malformed scalar Hindi Tense output vectors.

Updated docs:

- `experiments/input_output_overlap/README.md` now records the aggregate result
  and the weak-claim caveat.
- `experiments/output_features/README.md` now records that the old
  `from src.config` import bug is already fixed, but the active environment is
  missing `torchtyping` despite it being listed in `requirements.txt`.

## Cross-cutting checks

- Interpretability scope: all mechanistic results here are Llama-3.1-8B base
  with one layer-16 SAE. Aya appears only in the linear-model section.
- Counterfactual-attribution anomalies remain open if B1 enters the paper:
  Turkish Number=Sing, Arabic Person=3, f9539 universality, and last-BPE
  approximation issues for multi-token forms.
- Probe caveats remain open: the code trains logistic regression probes, not
  "mass-mean probes"; there is no shuffled-label baseline.
- `input_features` word-level procedure is still not implemented; current
  canonical input-feature outputs are sentence-level.

## Verification

- `python -m py_compile` passed for all modified Python scripts.
- `pytest tests/test_gcm_translation.py -m 'not gpu'` passed: 9 passed,
  6 deselected.
- `pdflatex` rebuilt `reports/draft_v5.pdf` successfully.

## Recommended next steps

1. Let cluster job `5772145` finish, then rerun:
   `python experiments/head_ablation_multiblimp/analyze.py --output_dir outputs/head_ablation_multiblimp`
   and
   `python experiments/head_ablation_multiblimp/visualize.py --output_dir outputs/head_ablation_multiblimp --out_img_dir experiments/head_ablation_multiblimp/img`.
2. Add or loop dose-response jobs for `--top_k_collective` in `{1,5,10,20}`.
3. Run ACP-vs-ATP interventional faithfulness on the GCM sweep before making
   a strong causal-ranking claim.
4. Decide whether to implement a masked/observed-cell rank-1 check for H1, or
   explicitly state the current column-mean imputation caveat.
5. Keep H2 as a cautionary/mechanistic-side result unless missing/malformed
   cells are repaired and the aggregate overlap story gets stronger.
