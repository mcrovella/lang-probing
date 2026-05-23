# draft_v5.tex audit report

Date: 2026-05-22

Scope: I audited `reports/draft_v5.tex` against `reports/v4.tex`, `reports/P1_overview.md`, the experiment reports, and the saved outputs. I focused on scientific, mathematical, and general-claim correctness, not ordinary prose polish.

## Actions taken

- Clarified that the reported OLS coefficients use FLORES corpus perplexity, not Multi-BLiMP PER (`draft_v5.tex` §3).
- Reworded the rank-1 result as source/target separability, not the same additive "no interaction" test as the OLS model.
- Added the 24x24 BLEU-matrix size and the 24 missing/self cells imputed by column mean next to the rank-1 faithfulness number.
- Removed the direct `R^2 = 0.02` vs. `rank-1 = 88%` comparison as if they were the same metric; the draft now says they are not directly comparable.
- Fixed the GCM sign/wording issue: the formula in the code is an orig-minus-cf contribution, not literally the intervention effect of swapping `z_orig -> z_cf`.
- Added the eng->fra 98-successful-pair exception to the GCM sweep description.
- Clarified that the Multi-BLiMP head-ablation score is the correct vs. incorrect final token.
- Softened two overclaiming headings: the section now says "overlap with language-modeling machinery" and the head-ablation subsection now says "signed GCM head groups predict..." rather than implying all influential heads straightforwardly carry grammar.
- Added missing BibTeX entries for FLORES-101, MultiBLiMP, and GCM, then rebuilt `draft_v5.pdf`.

## Findings to decide together

### 1. H1 is still not a clean "monolingual competence predicts BLEU" result

The draft now hedges this, but the underlying issue remains. The measured FLORES-perplexity OLS fit is weak (`R^2 = 0.02` Llama, `0.10` Aya), while the BLEU-only rank-1 decomposition is strong (`88.3%` / `82.4%`). The rank-1 factors show source/target separability in BLEU, but they do not yet prove that measured monolingual competence explains those factors.

Decision needed: either add an analysis tying rank-1 left/right factors back to FLORES PPL, Multi-BLiMP accuracy/PER, or another competence measure, or keep this framed as suggestive low-rank structure rather than a demonstrated competence predictor.

### 2. BLEU provenance is underspecified

The BLEU scores are joined from an external source, but the draft does not state decoding setup, model prompting/evaluation details, whether diagonals/self-pairs are real or missing, or why column-mean imputation is appropriate for the rank-1 matrix.

Decision needed: add a reproducibility sentence or footnote naming the BLEU source and exact generation/evaluation protocol.

### 3. GCM head-level "translation circuit" evidence is weak after controls

The null-control decomposition says head-level translation-circuit contribution is at most `+0.025` nat, at the bf16 noise floor. L30H25 is the clearest warning: `real_cross = +0.48`, but `null_same = +1.21`, so the head looks more like content discrimination than translation-specific computation.

Decision needed: lead with the SAE-feature evidence, not universal-head counts. If head universality stays in the main text, it needs the null-control caveat immediately beside it.

### 4. GCM faithfulness is still incomplete

The finite-difference linearization test is `xfail`, and the ACP-vs-ATP interventional faithfulness check has not been run on the sweep. That means the gradient ranking is not yet validated against direct interventions for the reported components.

Decision needed: either run/fix at least a top-K ACP-vs-ATP check or explicitly mark the GCM rankings as gradient-estimator evidence with incomplete interventional validation.

### 5. Head-ablation sign split is promising but under-controlled

The POS/NEG sign split is directionally consistent in 8/8 languages, but the proper matched-size POS/NEG random controls do not exist on disk. Hebrew and Hindi are at `n=200` while the rest are `n=400`; Hindi also has a random-control anomaly (`+0.23`) comparable to the POS-IE effect (`+0.22`). Several effects are only 2-3x the bf16 noise floor.

Decision needed: rerun the v2 controls on supported GPUs, get Hebrew/Hindi to `n=400`, and decide how much of the sign-split claim can appear before that.

### 6. Grammar-feature overlap needs provenance pinned down

The 9/50 SAE overlap vs. ~0.33 expected by chance is a strong-looking result, but the exact provenance of the 218 "grammar features" is still marked as a TODO. The draft currently calls them independently defined; that is probably fine, but only after the feature-set construction is spelled out.

Decision needed: document the exact file/threshold/procedure defining those 218 features.

### 7. H3 finetuning is external and should stay provisional

The repo has no monolingual-vs-parallel finetuning results. I changed the abstract note to say the collaborator-owned H3 result should be stated once verified, but the full draft still needs real prose/figures from that codebase before H3 can be claimed as a result.

### 8. v5 still has intentional stubs where v4 has real prose

`draft_v5.tex` replaces the v4 Introduction, Related Work, Discussion, and Limitations with `\jb{...}` placeholders. That is not a scientific contradiction, but it is a completeness issue and a source of possible story drift.

Decision needed: copy the stable v4 prose back in and then adapt it around the new empirical sections.

## Verification

- `bibtex draft_v5` now finds all cited database entries.
- `pdflatex -interaction=nonstopmode -halt-on-error draft_v5.tex` succeeds and writes `reports/draft_v5.pdf`.
- No undefined references/citations or overfull boxes remain in the final checked log.

## Citation metadata sources checked

- FLORES-101: https://doi.org/10.1162/tacl_a_00474
- MultiBLiMP TACL version: https://doi.org/10.1162/TACL.a.600
- GCM arXiv record: https://arxiv.org/abs/2602.16080
