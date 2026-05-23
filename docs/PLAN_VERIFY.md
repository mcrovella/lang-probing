# Verification & debugging playbook (overnight loop)

**Audience:** me, looping overnight after the implementing agent finishes initial
setup. Companion to `docs/PLAN_IMPLEMENT.md`. This doc holds the **test
assertions, acceptance gates, scientific sanity checks, and a debug runbook** —
deliberately separate from the implementer's doc so code isn't tuned to the
tests. Work phase by phase; fill the assertion bodies the implementer left as
`assert False, "TODO: verifier fills this"` placeholders.

## Global verification approach
- Prefer `qrsh` for interactive debugging of GPU code; reproduce failures on a
  5-pair smoke before resubmitting arrays. Expect SCC queue latency — submit,
  then validate other phases while waiting.
- For each phase: (a) unit tests green, (b) a tiny end-to-end smoke, (c) a
  scientific sanity check on real outputs.
- Scientific integrity: never assert a pending sweep will confirm a pattern.
  Report finished results; describe pending as pending; investigate divergences.

## Phase 0 — sign fix
- `tests/test_gcm_sign.py` assertions to fill:
  - migration is **exactly negation**: `migrate_gcm_sign` on a synthetic dir
    flips the sign of every `ie` element and leaves `|ie|`, `top_rankings.json`
    bit-identical.
  - migration is **idempotent**: second call is a no-op (marker respected).
  - on `tiny_causal_lm`, the recomputed metric satisfies the new convention:
    a component that raises `logp(r_orig)` yields **positive** IE.
- Real-output sanity: pick `English__Spanish`; confirm post-migration top-|IE|
  head set is identical to pre-migration and signs are flipped; baseline
  MultiBLiMP Δ printout (run.py:530) stays ≫ 0 (sign of the *score*, independent
  of IE convention, must not have regressed).
- **GATE:** all green + marker on every dir under both gcm dirs. Until then, do
  not start Phases 2–3.

## Phase 1a/3 — BLEU harness + ablation
- `tests/test_bleu_harness.py`: `extract_translation` recovers the target span
  from recorded outputs incl. the no-match fallback; `build_prompt` is the exact
  2-shot `>>`/`||` shape; `generate_batch` left-pads (batched output == per-item
  output on `tiny_causal_lm`).
- Hook composition: on `tiny_causal_lm`, `install_head_ablation` with an empty
  ablation list is a no-op (logits unchanged); a mean-ablation hook changes
  logits and `remove_hooks` restores them exactly.
- Scientific sanity (real model, smoke): baseline corpus BLEU for 2 directions
  matches the existing `lang-similarity` matrix within tolerance (confirms the
  prompt/extraction port is faithful); ablation conditions move BLEU.
- **GATE before full BLEU array:** smoke (2 targets, 50 sents) shows distinct
  ΔBLEU for pos10/neg10 vs baseline and a finite matched-control effect.

## Phase 3.0/3.1 — head selection
- `tests/test_head_selection.py`: `select_signed_heads` returns exactly
  `n_per_sign` per side, POS strictly > 0 signed IE and NEG strictly < 0 (or
  fewer with a logged warning if a side is short), disjoint sets, and the
  control sampler never overlaps the selected union.

## Phase 1d/4 — margin proxy + linear fit
- `tests/test_margins.py`: on `synthetic_minimal_pairs` with known token
  logprobs, `sequence_logprobs` returns the exact summed logprob and token
  counts; `margin_total = logp_correct − logp_wrong`; `aggregate_margins`
  groups correctly by (lang, phenomenon) and the per-lang accuracy equals the
  fraction with positive margin.
- Cross-check: per-token margin recomputed from a saved `perplexity_matrices_*.npz`
  (`log(ppl_wrong) − log(ppl_correct)`) matches `margin_pertoken` for the
  overlap languages, within float tolerance (validates the re-run against the
  old artifact).
- Coverage report present: which of the 24 BLEU langs exist in
  `jumelet/multiblimp`; the linear fit runs on the true intersection.
- Scientific sanity: margin proxy R²/MAE table emitted next to the accuracy
  proxy; report numbers with **no** prediction of whether margins beat accuracy
  (this is an open empirical question — record what comes out).

## Phase 1c — masked rank-1
- `tests/test_masked_rank1.py`: on a clean rank-1 matrix with random holes,
  `masked_rank1_als` recovers the factors (faithfulness ≈ 1.0 on observed);
  diagonal is excluded from both fit and faithfulness; `--legacy_impute`
  reproduces the old number.
- Record old (imputed) vs new (masked) faithfulness for llama & aya; no
  prediction of direction.

## Phase 2 — Aya GCM
- Smoke: 5 Aya pairs on one direction complete without arch/shape errors; the
  per-direction `summary.json` carries `sign_convention: orig_minus_cf` and a
  non-degenerate `|IE|` distribution; `metric_orig_clean > metric_cf_clean` on
  average (model prefers the gold translation).
- Array health: monitor for `qw` stalls (qdel+resub per the SCC notes); confirm
  all 56 directions produce `heads_ie.pt`.

## Debug runbook (common GPU failure modes)
- `CUDA error: no kernel image` → landed on sm_120; resubmit with
  `-l gpu_type=A100`.
- nnsight `OutOfOrderError` → touch Envoy hooks in ascending layer order.
- OOM mid-sweep → lower batch / add `gc.collect()+empty_cache()` cadence
  (pattern in head_ablation run.py:236).
- `ExitTracingException` / unbound var in a trace → replace any list
  comprehension inside `model.trace()` with an explicit pre-initialized loop.
- Zero-gradient assert in `gcm_core` → the patch didn't connect; check the
  `o_proj.output` write path and float32 leaf.
