# Overnight run results (2026-05-23)

Verification/execution loop over the implementing agent's work. All numbers are
finished results; pending/failed items are flagged. **Working tree is NOT
committed** (48 changed/new files) — review and commit when ready (on `main`, so
branch first).

## Headline results

### Phase 0 — GCM sign fix (root + migration) — VERIFIED
- `gcm_core.py` flipped to `M = logp(r_orig) − logp(r_cf)`; migration negated all
  128 stored direction dirs (idempotent marker). CPU tests pass.
- **Scientific validation (MultiBLiMP head-ablation, Llama, all 8 langs):**
  ablating the top-10 **positive-IE** (correct-favoring) heads **weakens** the
  grammaticality margin (Δ_change −0.05→−0.36); ablating top-10 **negative-IE**
  heads **strengthens** it (+0.12→+0.46); matched control ≈ 0. Sign convention is
  correct. Figure: `experiments/head_ablation_multiblimp/img/cross_lang_sign_split.png`
  (outputs in `outputs/head_ablation_multiblimp_signfix/`).
  - Honest caveat: neg-side clears its control in all 8; pos-side beats control
    cleanly for ara/fra/heb/hin/spa but is ≈control for deu/eng/tur.

### §3 — MultiBLiMP logprob-margin proxy (the saturation fix) — DONE
- Margin = mean(logp(correct) − logp(wrong)), per-item, per-phenomenon, over the
  **18-language** MultiBLiMP∩BLEU intersection (CJK/ind/vie/kor absent).
- **Margin proxy predicts BLEU better than accuracy:** R² **0.301** (common 17)
  / 0.324 (own 18) vs accuracy 0.250; perplexity 0.071; bits-per-byte 0.012.
  Correct sign on both coefficients. Margins spread where accuracy saturates
  (e.g. heb acc 0.88 but pertoken-margin 0.044 vs eng 0.99 / 0.467).
- Outputs: `outputs/perplexity_bleu_linear/multiblimp_margins/{model}/` (per_item.parquet,
  margins_by_lang.csv, margins_by_lang_phenomenon.csv) for Llama and Aya.
- 2-model proxy comparison fit: `compare_competence_proxies.py` is Llama-centric;
  Aya margins are saved but the Aya BLEU∼margin fit was not run (small script
  tweak; off critical path).

### Rank-1 BLEU-matrix faithfulness (masked, observed-only) — DONE
- Replaced column-mean imputation with masked rank-1 ALS (diagonal excluded).
- Llama **88.31% → 89.80%**; Aya **82.37% → 83.42%** (masked came out slightly
  *higher*, no spin). `--legacy_impute` retained.

### Phase 3.3 — Llama BLEU head-ablation — DONE, clean
- Ablate top-10 signed-IE heads (into each target), measure ΔBLEU on FLORES (10%),
  conditions baseline/pos10/neg10/ctrl. **pos10 ablation hurts BLEU more than the
  matched control in all 8 targets** (mean ΔBLEU pos10 **−2.90**, neg10 −0.65,
  ctrl −0.18; strongest deu −4.1, fra −4.0, heb −4.2; eng weakest at −1.78 vs
  ctrl −1.54). Mirrors and is *cleaner* than the MultiBLiMP result.
- This is the convergent cross-task evidence: GCM correct-favoring heads are
  causal for **both** monolingual grammaticality and translation BLEU.
- Outputs: `outputs/head_ablation_bleu/{target}/` (per-condition .jsonl cached
  generations + summary.json), `delta_bleu_summary.csv`,
  `experiments/head_ablation_bleu/img/cross_lang_delta_bleu.png`.

### Aya rerun (2nd model)
- **Aya GCM 56-direction sweep (heads-only, A100-80G): DONE, clean** (95-100%
  pairs/direction). NOTE: a first sweep on L40S-48G was OOM-corrupted (Hindi→Spa
  0%); re-run on A100 fixed it. `outputs/gcm_translation_aya/`.
- **Aya head-ablation diagnostic: PARTIAL reproduction.** neg10 ablation
  strengthens grammaticality in 6/8 langs (robust); pos10<0 in 6/8 but often tiny,
  and matched controls are noisy (deu/heb/spa ctrl ≈ −8 while pos10 ≈ 0). The
  clean Llama pos-side story does NOT cleanly reproduce on Aya. **hin failed
  (OOM at n=200)** — needs a retry at lower n. `outputs/head_ablation_multiblimp_aya/`.
  Aya's MultiBLiMP baseline Δ is very large (+65→+151) vs Llama (+5→+11) — worth
  understanding before drawing conclusions.

## Bugs fixed this session
1. `extract_translation` returned the last `||`-segment → wrong on over-generation
   (greedy almost always over-generates); now takes the first segment.
2. MultiBLiMP config map used `eng_Latn`/`pes_Arab` → 0 coverage; jumelet/multiblimp
   uses bare ISO (`eng`, `arb`, `fas`); fixed → 18 langs.
3. Margin output path: `args.output_dir.replace("/per",...)` mangled "perplexity";
   now uses output_dir directly.
4. `FLORES_CONFIG` used `deu_Latn`-style; gsarti/flores_101 uses bare codes; fixed.
5. BLEU runner: `.model.layers` level mismatch + `_unwrap` returned `LlamaModel`
   (no `.generate`) → now loads HF CausalLM directly + `_decoder_layers` helper.
6. `heads.py` cross-direction aggregation used `.mean` → a fully-failed direction
   NaN'd the whole map; now `nanmean` (Llama unaffected — it had no NaN dirs).
7. Added `--model_id` to `gcm_translation/run.py` and `head_ablation_multiblimp/run.py`
   for Aya; installed `sacrebleu` into the `probes` env.

## Known issues / follow-ups
- Aya head-ablation hin: re-run at n≈100 (OOM at 200).
- Aya pos-side ablation weak + noisy controls — investigate (Aya baseline-Δ scale,
  more control seeds, or matched-size controls).
- 2-model BLEU∼margin linear fit for Aya (script tweak).
- Consider re-running the Aya head-ablation matched control with multiple seeds.

## Deletion report (awaiting go-ahead — nothing deleted)
- `outputs/multiblimp_marginsplexity_bleu_linear/` — empty mangled dir from bug #3
  (Llama margins already moved to the correct path). Safe to remove.
- `outputs/gcm_translation_aya_smoke/` (27K), `outputs/head_ablation_bleu_smoke/` (482K)
  — smoke validation outputs; safe to remove (kept for now).
