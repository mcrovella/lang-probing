# Implementation plan — GCM sign fix, Aya rerun, head ablation, margin proxy

**Audience:** the implementing agent. Construction steps, file layout, and
function signatures. Companion doc `docs/PLAN_VERIFY.md` holds the verification
pass (you do **not** need it; assertion bodies are filled there).

Workstreams: (0) GCM sign fix at the root [hard gate]; (1) parallel no/low-GPU
tracks incl. the new MultiBLiMP **logprob-margin** competence proxy; (2) Aya GCM
heads-only sweep [longest pole]; (3) attention-head ablation (new top-10/top-10
scheme) on MultiBLiMP then BLEU; (4) linear re-fit with the margin proxy + Aya
verify + rank-1 masked fix; (5) paper-text suggestions report (no `.tex` edits).

## Ground rules
- SCC: heavy compute via `qsub`/`qrsh` only; `-l gpu_type=A100` (avoid sm_120
  Blackwell); per-job logs in each experiment's `run/`. Smoke → (pause for
  review) → full sweep. Batch all generation.
- Do **not** edit `reports/*.tex`. Paper changes go to a suggestions report.
- Sign migration is **in place** (git-tracked). No `rm -rf`; leave stray files
  for an end-of-task deletion report.
- Create test *files and fixtures* with the signatures below, but you are not
  told the asserted properties — write the infrastructure and leave assertion
  bodies as `assert False, "TODO: verifier fills this"` placeholders so the
  files import and collect.

## Phase 0 — GCM sign fix at the root (HARD GATE; blocks Phases 2–3)
Adopt **positive IE = component favors the correct/original/grammatical
completion**. New metric `M = logp(r_orig) − logp(r_cf)`.

1. `experiments/gcm_translation/gcm_core.py`:
   - `gcm_attribute_sae`: line 299 `grad = grad_cf − grad_orig` → `grad_orig − grad_cf`;
     line 302 `M_patched = m_cf_patched − m_orig_patched` → `m_orig_patched − m_cf_patched`.
   - `gcm_attribute_heads`: lines 487, 492 — same two flips.
   - Rewrite the docstring sign block (lines 6, 11–19, 298). `metric_orig_clean`
     / `metric_cf_clean` are raw logp values and stay unchanged.
2. `scripts/migrate_gcm_sign.py` (new): for each direction dir in
   `outputs/gcm_translation/` **and** `outputs/gcm_translation_null/`:
   negate `ie` tensors in `heads_ie.pt` and `sae_ie.pt`; negate
   `metric_diff_patched` wherever stored in `per_pair_records.json` /
   `summary.json`; leave `top_rankings.json` (ranked by `mean_abs_ie`) untouched;
   write `"sign_convention": "orig_minus_cf"` into each `summary.json` as an
   idempotency marker (refuse to re-negate a marked dir). Entry point
   `migrate_gcm_sign(dir: Path) -> dict` (returns a per-file change record).
3. Audit consumers (grep `signed`, `mean_signed_ie`, `> 0`, `< 0`,
   `POS`/`NEG`, `RdBu`, red/blue): `head_ablation_multiblimp/{heads.py,run.py,
   analyze.py,visualize.py}`, `gcm_translation/{three_way_decomposition.py,
   bootstrap.py,analyze.py,meeting_plots.py,visualize.py,validate_faithfulness.py}`.
   Ensure `heads.py:72` reads the now-correct sign with **no** compensating
   negation. Reconcile figure colormaps/captions to the new convention.
4. `docs/SIGN_CONVENTION.md` (new): convention, date, one-line math, marker,
   audited-consumer list. Pointer line in `docs/TODO_DONE.md`.
5. Test infra: `tests/test_gcm_sign.py` with fixtures
   `make_synthetic_gcm_dir(tmp_path, *, n_pairs, n_layers, n_heads, sae_dim)->Path`
   and `tiny_causal_lm()` (2-layer random HF CausalLM, CPU); helper
   `run_gcm_pair(model, tokenizer, pair)->dict`. Mark GPU cases
   `@pytest.mark.gpu`. Run: `pytest tests/test_gcm_sign.py -q`.

## Phase 1 — Parallel tracks (start at t0; 1a–1c no GPU, 1d moderate GPU)

### 1a. BLEU translation harness — `experiments/head_ablation_bleu/translate.py`
Port from `lang-similarity` (the pipeline behind the paper's BLEU matrix). No
sign dependency — build/smoke now.
- `build_flores_dataset(lang_keys, split_pct) -> dict[str, list[str]]` (reuse
  `lang-similarity/src/utils.py:create_base_lang_dataset`; `gsarti/flores_101`,
  `devtest[:N%]`; reuse its name→FLORES-config keys).
- `build_prompt(src_sents, tgt_sents, i) -> str` — 2-shot cyclic `>>`/`||`
  (translation_eval.py:44-48).
- `extract_translation(output_text) -> str` — regex (translation_eval.py:59).
- `generate_batch(model, tokenizer, prompts, max_new_tokens, batch_size) -> list[str]`
  — batched, greedy, **left-padded**, `pad_token_id=eos`.
- `corpus_bleu_score(preds, golds) -> float` — `sacrebleu.corpus_bleu(preds,[golds]).score`.
- Test infra: `tests/test_bleu_harness.py` with `recorded_translation_io()`
  fixture (a few cached prompt→output strings) and a `tiny_causal_lm()` reuse for
  the hook-composition path. Run `pytest tests/test_bleu_harness.py -q`.

### 1b. TODO/docs restructure
- Merge `docs/PAPER_TODO.md` + `docs/TODO.md` → **`docs/TODO.md`** (open items
  only; pinned `## [PAPER] — top priority` section on top of a general section).
- New **`docs/TODO_DONE.md`** (append-only; one line + completion date per closed
  item = the lightweight ledger). Fold in & retire `docs/PAPER_TODO_RESOLUTION.md`.
- Leave `docs/LEDGER.md`, `docs/LAB_NOTEBOOK.md` as-is.
- Seed the deferred items list (see end of this file) into `TODO.md`.

### 1c. Rank-1 masked fix (appendix, low priority) — `rank1_approximation.py`
- `masked_rank1_als(M, mask, *, n_iter, seed) -> (u, v)` minimizing
  `||mask⊙(M−uvᵀ)||_F` (rank-1 ALS, closed-form per vector over observed entries).
- `masked_faithfulness(M, mask, u, v) -> float = 1 − ||mask⊙(M−uvᵀ)||_F/||mask⊙M||_F`.
- `mask` = observed (non-NaN) **and** off-diagonal (diagonal always excluded).
- Swap into `main()`; keep old path behind `--legacy_impute` for before/after.
- Test infra: `tests/test_masked_rank1.py` +
  `make_low_rank_with_holes(shape, rank, hole_frac, seed)->(M, mask)`.

### 1d. NEW: MultiBLiMP logprob-margin competence proxy — extend `run_per.py`
Goal: per-item **log-prob margins** on MultiBLiMP, carrying `phenomenon`,
across the MultiBLiMP∩BLEU language intersection, for the §3 linear model.
- **Coverage check first:** enumerate `jumelet/multiblimp` configs; intersect
  with the 24 BLEU langs (ara, rus, fra, zho_simpl, ita, por, ind, heb, pol,
  deu, ell, tur, vie, ukr, spa, fas, kor, hin, jpn, eng, zho_trad, ces, nld,
  ron). Build a `BLEU_CODE → MULTIBLIMP_CONFIG` map (e.g. `eng→eng_Latn`,
  `zho_simpl→zho_Hans`/`zho_trad→zho_Hant`); report which of the 24 are absent.
- Add `sequence_logprobs(model, tokenizer, texts, device, batch_size, max_length)
  -> (logp_total: np.ndarray, n_tokens: np.ndarray)` (mirror
  `perplexity_per_sentence` but return summed token logprob and token counts,
  not exp(avgNLL)). Right-pad, shift, mask.
- New `run_margins_multilang(dataset_path, model_id, language_map, split, ...)`:
  per language, load config, pull `sen`/`wrong_sen` **and the phenomenon field**
  (reuse the extraction in `counterfactual_attribution/build_multiblimp_pairs.py`),
  compute per-item `logp_correct`, `logp_wrong`, `margin_total = logp_correct −
  logp_wrong`, `margin_pertoken` (length-normalized), and accuracy
  `1[logp_correct>logp_wrong]`.
- Save **per-item** `outputs/perplexity_bleu_linear/multiblimp_margins/{model}/
  per_item.parquet` (cols: lang, phenomenon, concept, logp_correct, logp_wrong,
  margin_total, margin_pertoken, correct) + aggregates `margins_by_lang.csv`
  and `margins_by_lang_phenomenon.csv` (mean margin, n, accuracy).
- Run for `llama` and `aya` (Aya is a second forward-only pass; cheap).
- Test infra: extend `tests/` with `tests/test_margins.py` and a
  `synthetic_minimal_pairs()` fixture (tiny list with known token-level logprobs
  via `tiny_causal_lm()`), plus an aggregation helper
  `aggregate_margins(per_item_df) -> (by_lang, by_lang_phenomenon)`.
- GPU: batched fp16, one A100; full 24-lang (or 17) MultiBLiMP is forward-only →
  smoke 2 langs first, then a single full job (not a long pole).

## Phase 2 — Aya GCM sweep (LONGEST POLE; after Phase 0 gate)
`--components heads` (SAE deferred; see deferred list). Born with corrected sign.
- Pre-flight 5-pair smoke (`qrsh`): confirm Cohere arch exposes
  `model.model.layers[i].self_attn.o_proj` + `.config` fields; `setup_model(
  "CohereForAI/aya-23-8B", sae_id=None)` works; the Aya tokenizer passes the
  `tokenize_pair` `LangName: ` seam assert (gcm_core.py:129).
- Submit early: `qsub` array over the 56 ordered directions in {eng, spa, deu,
  fra, tur, ara, hin, heb}; mirror `gcm_translation/run.py` + `configs/`.
- Napkin: read Llama heads-only per-direction wall-time from `logs/`/`qacct`,
  report Aya estimate before committing. Aya null sweep → deferred.

## Phase 3 — Head ablation, new scheme (after Phase 0; Aya parts after Phase 2)
### 3.0 `head_ablation_multiblimp/heads.py`: add
`select_signed_heads(signed_map, abs_map, *, n_per_sign=10)
-> {"pos":[...], "neg":[...]}` — top-n by signed IE on each side; entries
`{layer, head, signed_ie, abs_ie}`.
### 3.1 `build_conditions` core set (both metrics): `baseline`, `pos10_mean`,
`neg10_mean`, `ctrl_matched10_mean` (10 random heads, layer-stratified over
POS∪NEG layers, excluding selected; via `sample_stratified_controls`). Drop the
mixed top-k collective. Mean replacement = position-averaged per-head mean of
`o_proj.input`.
### 3.2 MultiBLiMP diagnostic FIRST (Llama, fast): re-run
`head_ablation_multiblimp/run.py` (8 langs) under the new selection; regenerate
`cross_lang_sign_split.png` with corrected labels. **Pause for review.**
### 3.3 BLEU head ablation (Llama) — `experiments/head_ablation_bleu/run.py`,
reusing `heads.py` + 1a `translate.py`:
- `install_head_ablation(hf_model, ablations, mean_acts) -> handles`: persistent
  forward hook on each target layer's `self_attn.o_proj`. Hook
  `(module, inputs, output)`: `inputs[0]` = `o_proj.input` `[B,S,d_model]`; per
  `(layer,head)` add `(mean[L,h] − input_head) @ W_h.T` to `output` at **all**
  positions (matches MultiBLiMP all-position broadcast); works for prefill and
  S=1 decode. `remove_hooks(handles)` between conditions.
- `compute_mean_activations_translation(model, prompts, ...)` — per-head mean of
  `o_proj.input` over FLORES prompts for target L.
- Reach the raw HF model from the nnsight wrapper (`setup_model`); install hooks;
  HF `.generate()` batched.
- Scope: per target L, translate 7 others → L; 56 directions; conditions from
  3.1; FLORES 10%.
- **Cache generations** (JSONL per target+condition): `{direction, sentence_idx,
  source, gold, generation, sentence_bleu, head_set}`. Decoded text + sentence
  BLEU; no logprobs.
- Metric: ΔBLEU = corpus_BLEU(cond) − corpus_BLEU(baseline) per direction +
  aggregated per target L. Smoke 2 targets/50 sents first → review → full array.
### 3.4 Repeat 3.2 + 3.3 on Aya (after Phase 2).

## Phase 4 — Linear re-fit + Aya verify + rank-1 outputs
- Extend `compare_competence_proxies.py` / `run_linear_fit.py` with a
  `multiblimp_margin` proxy (per-lang mean `margin_total`; option for
  per-phenomenon-then-mean, and a "hard phenomena only" subset toggle). Re-fit
  `BLEU ~ 1 + margin[src] + margin[tgt]` for Llama (+Aya). Emit the same R²/MAE
  table + marginal scatter as the accuracy proxy, side by side.
- Re-verify Aya `perplexity_bleu_linear` numbers (`--model aya`).
- Run masked rank-1 (1c) both models; record old vs new faithfulness.

## Phase 5 — `reports/v9_suggestions.md` (no `.tex` edits)
Itemized, line-anchored suggestions: sign-convention text/captions (eq:metric,
line-223 sentence, §4.2 paragraph, signed-heatmap captions); "noisy channel"
misnomer (v9.tex:96); broken `\jb{}` inside `tab:proxy` (v9.tex:138,141);
undefined `\aaron`; Limitations (two models honestly); new margin-proxy result;
new BLEU-ablation figure; masked rank-1 number.

## Critical files
gcm_core.py; scripts/migrate_gcm_sign.py; head_ablation_multiblimp/{heads,run,
visualize,analyze}.py; experiments/head_ablation_bleu/{translate,run}.py + qsub;
perplexity_bleu_linear/{run_per,rank1_approximation,compare_competence_proxies,
run_linear_fit}.py; counterfactual_attribution/build_multiblimp_pairs.py (reuse
phenomenon extraction); src/lang_probing_src/{config,utils}.py; docs/{TODO,
TODO_DONE,SIGN_CONVENTION}.md; tests/{test_gcm_sign,test_masked_rank1,
test_head_selection,test_bleu_harness,test_margins}.py.

## Deferred to TODO (out of scope; seed into docs/TODO.md)
real_cross−null_cross head selection + high-null/low-real control; alternate
k/threshold schemes; Aya SAE attribution + all SAE downstream experiments; Aya
null GCM sweep; per-phenomenon "hard subset" selection once margins are in.
