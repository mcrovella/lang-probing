# PAPER_TODO

Consolidated open items for the paper draft (*"Read, Translate, Write: LLMs
Translate with Fuzzy Semantic Hubs"*). Compiled from `reports/P1_overview.md`
and the per-section feedback pass. Goal: rough draft. Items are tagged:

- **[FIG]** — a figure to generate/regenerate.
- **[P2]** — deferred to the Phase-2 deep-verification pass (method/code/results check).
- **[UNCERTAINTY]** — a scientific-validity question to resolve or explicitly hedge in the text.
- **[REPRO]** — reproduction detail to pin down so the experiment is write-up-ready.
- **[APPENDIX]** / **[CUT]** — figure-routing decisions.

Figures kept for the main draft per section are listed under "Figure routing".

---

## Agent handoff: where everything lives

Canonical docs to read first (already written — start here, don't re-derive):
- `reports/P1_overview.md` — full impartial method/results/figure account for every experiment, with validity flags. **The single best orientation doc.**
- `LEDGER.md` — curated per-experiment status. `LAB_NOTEBOOK.md` — chronological log.
- `experiments/gcm_translation/REPORT.md` (method §3, control tasks §5.5, findings) and `REDTEAM.md` (the C1–C6 fix log).
- `experiments/head_ablation_multiblimp/REPORT.md` (method, full table, caveats).
- `experiments/perplexity_bleu_linear/README.md` (pipeline + rank-1).

Key scripts / data per item are named inline below. General map:
- GCM outputs: `outputs/gcm_translation/<Src>__<Tgt>/{heads_ie.pt, sae_ie.pt, top_rankings.json, summary.json}`; cross-direction JSON (`universal_heads.json`, `universal_sae.json`, `sae_overlap_with_grammar.json`) written by `analyze.py`; global plots by `meeting_plots.py` / `three_way_decomposition.py` / `visualize.py`.
- Head-ablation outputs: `outputs/head_ablation_multiblimp/<lang>/{summary.json, deltas.json, mean_acts.pt}` + `analysis_summary.csv`; figures by `visualize.py`.
- Linear-model data: `outputs/perplexity_bleu_linear/bleu_and_ppl/combined_results_{model}.csv`; figures at repo-level `img/perplexity_bleu_linear/`.

---

## STATUS as of 2026-05-22 (after the resolution pass + v7)

Current draft is **`reports/draft_v7.tex`** (single-column, readable; compiles
clean, 0 overfull, all figures embed). v5 is the other agent's; v6 was the
intermediate. The per-section detail below is retained; this block is the
current scoreboard.

### Resolved / done
- **Grammar-feature "218" provenance** — RESOLVED. `analyze.py:138–157` defines it as the union of each concept's `top_50_by_mean_grad_x_act` list from `outputs/counterfactual_attribution/aggregated_by_concept.json` (7 concepts → 218 unique features); overlap with GCM top-50 universal SAE = 9. *Caveat below.*
- **BLEU provenance** — DOCUMENTED (was unknown). Generated in `../lang-similarity` (`src/translation_eval.py` + `bleu_similarity.py`): 2-shot `src >> tgt||` prompting (2 preceding FLORES sentences as exemplars), greedy decode to 1.35× gold length, regex-extract 3rd segment, `sacrebleu.corpus_bleu(pred,[gold])` per direction, 24 langs, FLORES-101 `devtest`. Now in v7 §3 footnote.
- **Signed universal-heads figure** — DONE (`meeting_direction_by_universal_head_signed.png`, in v7).
- **rank-1 robustness** — DONE: leave-one-language-out (0.878–0.888, full 0.883) + residual heatmap; in v7 (§3 prose + appendix).
- **GCM SAE real-vs-null figure + sparsity figure** — DONE (`meeting_sae_real_vs_null_same.png`, `head_ie_top_median_ratio.png`), in v7.
- **H2 aggregate overlap quantified** — DONE: 25 cells, mean Jaccard@200 = 0.095 raw / 0.122 abs; in v7 Limitations.
- **English matched POS/NEG controls** — DONE (n=400, 53 conditions): matched −0.028/+0.032 vs sign-split +0.075/−0.160; in v7.
- **Red-team C1–C6** — code-level check passed (two-trace, o_proj.output delta, fp32 leaf, token-space truncation, trailing-space prompt, sm_120 guard present).
- **`output_features` import bug** — already fixed in code (per resolution).
- **Appendix figures (mean-vs-zero, eng→spa heatmaps)** — in v7.

### OPEN — still blocking a confident claim
- **[UNCERTAINTY — new, important] Grammar-overlap chance baseline is naive.** The "≈0.33 expected, so 27×" assumes the GCM top-50 is a *uniform* draw from 32,768. Both sets are top-by-attribution features and likely cluster in a small generic-feature pool (cf. f9539-type "fires on every morphological decision" features, flagged as possible SAE artifacts). Recompute the null by sampling 50 from the *attribution-active* feature pool, and check whether the 9 overlaps are a few generic features. Until then the 27× may be inflated.
- **[UNCERTAINTY] ACP-vs-ATP faithfulness never run.** Confirmed no `acp_faithfulness.json` on disk. Combined with the `xfail` finite-difference test, GCM rankings have no interventional validation. Phrase as *candidate-selection* ranking, or run top-K ACP.
- **[UNCERTAINTY] rank-1 imputation.** 24/576 cells (the diagonal) imputed by column-mean before SVD; recompute faithfulness masked / observed-cells-only to confirm the 88% isn't imputation-inflated. (Only 4% of cells, and LOO is robust, so likely minor — but still the clean check.)
- **[REPRO — new] BLEU split size.** Committed `translation_eval.py` slices FLORES `devtest[:10%]` (~100 sentences/direction), but the user recalls "all sentences." Confirm whether canonical `bleu_results.csv` used 10% or the full split.
- **[FIG/compute] 7-language matched controls** — queued (job `5772145`, A100, n=400). On completion rerun `analyze.py` + `visualize.py`, regenerate `cross_lang_sign_split.png`, update v7 §4b. Brings heb/hin to n=400 too.
- **[FIG/compute] Dose-response over k={1,5,10,20}** — not generated; `run.py --top_k_collective` runs one k per job.
- **[UNCERTAINTY] Hindi random-control anomaly** (+0.23) persists; revisit after n=400 rerun.
- **[UNCERTAINTY] H1 factors↔competence link** unexamined: rank-1 left/right vectors never correlated against PPL/PER. Either add that analysis or keep H1 framed as suggestive low-rank structure.
- **[UNCERTAINTY — RESOLVED diagnosis, 2026-05-22] Why linear-in-PPL R²=0.02.** Audited (`/tmp/ppl_audit*.py`, read-only over `bleu_results_llama.csv` + `perplexity_results_llama.csv`). The fit code (`run_linear_fit.py`) is correct; R²=0.0208 reproduces. Root causes, in order:
  1. **Per-token corpus perplexity is tokenization-confounded and is a poor cross-lingual competence proxy.** `run_perplexity.py` computes `exp(mean NLL per *token*)` on FLORES devtest. Per-token PPL depends on how finely each language is tokenized: fragmented scripts get low per-token PPL regardless of competence. The values are backwards from competence — **eng PPL=21.9 (near highest) but is the best translator; heb PPL=3.67 (lowest) but mid/low BLEU**.
  2. **PPL is empirically ⊥ competence.** PPL vs marginal BLEU: Spearman −0.30 (as src), −0.22 (as tgt), both non-significant. PPL vs the **rank-1 competence factors** (the thing explaining 88% of BLEU): Spearman −0.31 / −0.22, non-significant. So the draft's "PPL is a noisy proxy for the rank-1 factors" is **too charitable** — PPL barely correlates with them. **Soften this claim in v7 §3.**
  3. **Functional form is NOT the issue:** the interaction term changes R² by <0.001 (additive vs multiplicative is not what's breaking it).
  4. **English is a leverage outlier:** dropping eng raises R² 0.021→0.114 (still weak).
  - **Fixes to consider:** (a) use a tokenization-invariant fluency measure — **bits-per-byte** (NLL / UTF-8 byte count) instead of per-token PPL; (b) use **MultiBLiMP accuracy/PER** as the competence proxy (scale-free per-pair accuracy; repo already computes it but the fit didn't use it); (c) stop leaning on the linear fit and present rank-1 separability as the H1 evidence, dropping the "PPL proxy" reconciliation. **Recommend (b)+(a) as a quick re-fit before claiming H1.**
- **[RESOLVED 2026-05-22] Competence-proxy comparison done.** New scripts `compute_bits_per_byte.py` + `compare_competence_proxies.py`; full report `reports/competence_proxy_comparison.md` (3 figs + tables in `outputs/perplexity_bleu_linear/competence_proxy_comparison/`). Findings (Llama, common 17-lang/272-pair subset, verified 3 ways):
  - **MultiBLiMP accuracy = the H1 proxy.** R²=0.250, correct signs, target-competence weighted 3.4× source, marginal Spearman +0.62 (p=0.008). Use this for H1.
  - **MultiBLiMP accuracy ≡ perplexity-accuracy (1−PER) exactly** (r=1.0) — they are the *same* number, not two proxies.
  - **Bits-per-byte does NOT help** (R²=0.012 common, marginal ρ=−0.03, wrong signs). The per-token-PPL failure is **not just tokenization** — monolingual fluency/compression is the wrong *kind* of signal. Per-token PPL R²=0.07 (wrong signs).
  - Interaction term adds <0.001 everywhere (functional form not the bottleneck).
  - **Action for v7/v8 §3:** replace the linear-in-PPL framing with the MultiBLiMP-accuracy fit (R²≈0.25, modest but real, no interaction) + rank-1 separability; drop the "PPL is a noisy proxy" sentence. Caveats: 17-lang coverage (no CJK), R² modest, mild MultiBLiMP/BLEU shared-construct concern, BPB is an exact reconstruction from saved PPL (a direct GPU ΣNLL/Σbytes pass would reproduce it).
- **[UNCERTAINTY] H2 weak + incomplete:** overlap is modest (≤0.12) and several cells are missing/malformed (missing input diff vectors for some Dual/Sing/Tense cells; Arabic Tense missing from output effects; malformed scalar Hindi Tense output vectors). Keep H2 cautionary unless repaired.
- **[REPRO] `output_features` env:** `torchtyping` missing from the active env despite being in `requirements.txt`.
- **[P2] H1 source CSV / external BLEU upstream:** decoding now documented; still cite the upstream generation properly in the paper.
- **[P2] probes:** "mass-mean" wording vs logistic-regression code; no shuffled-label baseline. **[P2] `input_features` word-level** procedure still unimplemented (sentence-level only). **[P2] counterfactual_attribution anomalies** (Turkish Number=Sing, ara Person=3, f9539, last-BPE) open if B1 enters the paper.

### Draft-completeness (not scientific)
- v7 keeps `\jb{}` stubs for Intro / Related / Discussion / Limitations / §5 (finetuning, external/Jannik). Copy stable v4.tex prose back in and adapt around the new empirical sections.

---

## A1 — Linear model (`perplexity_bleu_linear`, §3)

### Figure routing
- **Main:** raw BLEU src×tgt grid + rank-1 approximation (paired); joint competence scatter.
- **[P2]** rank-1 predicted-vs-actual scatter → defer to P2.
- **[P2]** joint competence scatter → defer to P2.
- **[CUT]** competence contour, perplexity-vs-BLEU sorted, source/target competence singletons — probably not needed.

### Items
- **[FIG]** Raw BLEU src×tgt grid **exists** in the sibling repo
  `../lang-similarity`: clean 4-language version `img/bleu_heatmaps.png`
  (eng/spa/tur/jpn, Llama+Aya), and a denser ~17-language version
  `eval_img/heatmaps_lla_aya.png` (Spanish axis labels). **Neither matches the
  exact language set / model used by the rank-1 approximation.** TODO: regenerate
  a clean BLEU src×tgt grid over the *same* languages + model as
  `rank1_approximation.py`, to display directly alongside the rank-1 result.
  *Pointers:* old heatmap code = `../lang-similarity/src/eski/legacy_visual.py`
  ("eski" = legacy). The rank-1 matrix is built in
  `experiments/perplexity_bleu_linear/rank1_approximation.py:load_bleu_matrix`,
  which reads `combined_results_{model}.csv` and pivots
  `df.pivot(index="src", columns="tgt", values="bleu")` — generate the grid from
  that **same** pivot so figure and rank-1 number share one matrix.
- **[UNCERTAINTY]** R²≈0.02 (linear-in-PER, Llama) vs 88.31% (rank-1 SVD) is
  unreconciled. The "PER is a noisy proxy for the true competence vectors"
  explanation is asserted, not demonstrated. Either show it or hedge it in text.
- **[P2/UNCERTAINTY]** Source-of-truth CSV ambiguity: `perplexity_results_*.csv`
  (pilot) vs `combined_results_*.csv` (canonical). Confirm which the fits used.
- **[UNCERTAINTY]** External BLEU provenance/decoding setup is unspecified here —
  state where the BLEU numbers came from and how they were computed.
- **[UNCERTAINTY]** rank-1 faithfulness is mechanically high for small matrices;
  report the matrix dimensions next to the 88% so the number is interpretable.
- **[UNCERTAINTY — important]** The rank-1 matrix is 24×24 with **24 missing/self
  cells imputed by column mean** (per the v5 draft edit). Mean-imputation pushes
  cells toward a separable (low-rank) structure, so it can *inflate* the 88.3%.
  Re-run rank-1 faithfulness (a) on the observed cells only (masked SVD / report
  error over non-imputed entries) and (b) with diagonal excluded, to confirm the
  separability is in the real data, not the imputation.
- **[P2]** No residual plot / leave-one-language-out (Turkish, Hebrew) yet.

---

## A2 — GCM head & feature identification (`gcm_translation`, §4 first half)

### Figure routing
- **Main:** universal-heads heatmap; direction × universal-head matrix;
  three-way decomposition (SAE — the stronger result); SAE↔grammar overlap.
- **[P2]** per-direction eng→spa heatmaps (method illustration).

### Items
- **[FIG]** Generate **more figures of the SAE-feature results** (these separated
  real vs null better than heads — e.g. f31444 with null_same≈0). The SAE
  three-way decomposition should be foregrounded over the head version.
- **[FIG]** Universal-heads reuse plot: make a version that shows **signed IE
  (POS and NEG separately), not `abs(IE)`**, so the direction of each universal
  head's effect is visible across the 56 directions. *Pointers:* signed mean IE
  per direction is `nanmean(IE, dim=0)` over `outputs/gcm_translation/<dir>/heads_ie.pt`;
  the current abs-based reuse matrix is `meeting_plots.py` (Plot at line ~28,
  uses top-20 universal heads). Color by signed IE (RdBu) instead of |IE|.
- **[FIG]** Regenerate the "universal translation heads" / direction×head plot
  with the **actual K stated** in the title/caption. **K = 20** — set by
  `analyze.py --top_k_for_universality` (default 20); `meeting_plots.py` then
  plots the top-20 universal heads (and top-50 SAE features). Just put "top-20"
  in the title.
- **[P2]** Top/median IE ratio (27–45× across directions): pull a small table or
  per-direction summary so the "sparse, structured, not noise" claim is backed.
- **[UNCERTAINTY — most important]** Phase 1 "universal translation heads"
  (L17H21, L30H25 in 56/56) are reclassified as **content heads** by the Phase 2
  null control (head-level translation-circuit contribution ≤ +0.025 nat, at
  noise floor). The head-level H4 story is weak; the SAE-feature story is the
  real one. The paper must not lead with the head-universality counts without
  the null-control caveat.
- **[P2]** Red-team C1–C6 (`gcm_translation/REDTEAM.md`): confirm the committed
  `gcm_core.py` / `run.py` behind the production sweep actually contains the
  claimed fixes (two-trace pattern, o_proj.output delta, fp32 leaf, trailing-
  space prompt, token-space truncation).
- **[UNCERTAINTY]** Finite-difference linearization test is `xfail` — the most
  direct per-pair faithfulness check is not asserted to pass. Decide whether to
  fix/report or omit.
- **[P2/FIG]** ACP-vs-ATP interventional faithfulness figures are **pending**
  (`validate_faithfulness.py` exists, not run on the sweep). Needed to claim the
  gradient ranking matches the true interventional ranking.
- **[P2]** Pin down the provenance of the **218 "grammar features"** used in the
  SAE↔grammar overlap plot. *Pointers:* the overlap is computed in
  `gcm_translation/analyze.py` (writes `sae_overlap_with_grammar.json` with key
  `overlap_features`), cross-referencing against `counterfactual_attribution`'s
  top features. Read `analyze.py` around line 143 to find exactly which
  counterfactual-attribution file/threshold defines the 218-feature set, and
  whether 218 is a fixed top-N or a significance cutoff.

---

## A3 — Head ablation on Multi-BLiMP (`head_ablation_multiblimp`, §4 second half)

### Figure routing
- **Main:** cross-language signed-IE (POS vs NEG) collective ablation — once the
  matched controls exist (below).
- **[APPENDIX]** mean-vs-zero ablation figure.
- **[CUT]** all per-figure charts that are not the cross-language signed-IE
  ablation (per-head bars, scatter rank-vs-effect, correlations) — probably not
  needed in the draft.
- **[P2]** the full cross-language results table.

### Items
- **[FIG]** Add the **matched random-control conditions for the cross-language
  signed-IE collective ablations** (the v2 matched POS/NEG controls). Currently
  these do not exist on disk — `submit_v2.qsub` produced no saved outputs (landed
  on Blackwell sm_120 GPUs / English run ended before save), so the matched-
  control fields in `analysis_summary.csv` are `nan`. This is the proper control
  for the headline sign-split result; rerun on supported A100/H100 GPUs.
  *Pointers:* matched POS/NEG controls + `n_ctrl_seeds` extra controls are
  **already coded** in `head_ablation_multiblimp/run.py` (module docstring lines
  22–26; `build_conditions(..., n_ctrl_seeds=...)` ~line 268) — they failed to
  save only because `submit_v2.qsub` hit sm_120 Blackwell GPUs / the English run
  ended before save. Rerun `submit_v2.qsub` pinned to `gpu_type=A100` (or H200),
  `--n_ctrl_seeds 5`; then `analyze.py` + `visualize.py` repopulate
  `*_sign_split_matched.png`. heb/hin also need n=400 here.
- **[P2/FIG]** Dose-response over collective size **k**: only k=5 (plus the
  sign-split groups) was tested. Generate a table or curve over k=1, 5, 10, 20
  showing where the ablation becomes informative (k=1 is below the bf16 noise
  floor; the signal appears only in correctly-grouped collectives). No such
  chart exists yet. *Pointer:* `run.py` takes `--top_k_collective` (currently 5);
  loop it over {1,5,10,20} or extend `build_conditions` to emit all four.
- **[UNCERTAINTY]** heb + hin only at **n=200** (others n=400), on a different
  sample — cross-language magnitude comparisons mix two sample sizes. Rerun
  heb/hin at n=400 with the memory-fixed loop.
- **[UNCERTAINTY]** Hindi random-control anomaly (+0.23, comparable to its POS-IE
  collective) — unexplained (genuine heterogeneity / n=200 / control correlated
  with POS-IE subset). Resolve or flag.
- **[UNCERTAINTY]** Several reported Δ-change values (eng −0.16, spa −0.12, the
  whole POS-IE row) are within 2–3× of the ~0.06 bf16 noise floor. Consider
  float32 scoring or harder pairs; at minimum state the noise floor in text.
- **[REPRO]** Ensure the write-up has everything to reproduce: head-selection
  rule (top-20 by aggregated mean-abs IE over the 7 directions into L), the
  POS/NEG sign-split definition, mean-activation caching (position-averaged,
  per-language), the `o_proj.output += (r−x)·Wᵀ` ablation mechanism, the 46
  vs 53 condition sets, n per language, seeds, GPU class, Multi-BLiMP source
  (`data/multilingual_pairs/{lang}.json`), and final-token-only scoring.

---

## B2 — Input/output feature overlap (`input_features` / `output_features` / `input_output_overlap`, H2)

### Items
- **[UNCERTAINTY/FIG]** The Jaccard@k for English Number=Plur suggests **low
  input↔output overlap** — at odds with the H2 "feature spaces overlap" framing.
  Investigate: is overlap genuinely low, or is it a top-k / sparsity artifact?
  Quantify across cells before making any H2 overlap claim. *Pointers:* plots
  per cell at `img/input_output_overlap/<Lang>_<Concept>_<Value>/jaccard_topk.png`,
  built by `experiments/input_output_overlap/visualize.py` from `input_features`
  diff vectors and `output_features` effect tensors; compute mean Jaccard@k over
  all 29 cells for the aggregate.
- **[UNCERTAINTY]** H2 is **not reduced to a single headline statistic** — 29
  cells have plots but no aggregate (e.g. mean Jaccard@k across languages/
  concepts). Needed before H2 can be a paper claim.
- **[P2]** `output_features/run.py` has an unfixed import bug
  (`from src.config` → `lang_probing_src.config`) and an undeclared `torchtyping`
  dependency — not runnable as-is. Confirm which saved outputs are valid.
- **[P2]** Word-level input-feature procedure (priority-based negative sampling)
  specified in the draft is **unimplemented**; only sentence-level exists.
  Decide whether the paper claims word-level or sentence-level.

---

## Cross-cutting

- **[UNCERTAINTY]** All interpretability results are Llama-3.1-8B + one L16 SAE,
  base (not instruction-tuned); Aya appears only in the linear model. State scope
  explicitly in Limitations.
- **[P2]** `counterfactual_attribution`: Turkish Number=Sing (+1.05) and
  ara/Person/3 (+0.556) positive-Δ ablation anomalies, and f9539 universality,
  are open. Multi-token last-BPE approximation dominates fra/spa/ara (95%+) and
  may distort those rankings. (Relevant only if B1 enters the draft.)
- **[UNCERTAINTY]** `probes`: "task is easy" caveat (no shuffled-label baseline);
  draft says "mass-mean probes" but code is logistic regression — fix terminology.

---

## Suggested draft-figure shortlist (from the routing above)

1. §3: raw BLEU src×tgt grid + rank-1 approximation (paired); joint competence scatter.
2. §4a: SAE three-way decomposition; SAE↔grammar overlap; signed universal-heads reuse (K stated).
3. §4b: cross-language signed-IE collective ablation **with matched controls**.
4. Appendix: mean-vs-zero ablation; per-direction eng→spa GCM heatmaps.
