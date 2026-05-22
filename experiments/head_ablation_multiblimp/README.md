# head_ablation_multiblimp

Tests whether the attention heads identified by GCM as causally important for
translation are also important for monolingual grammaticality, by mean-ablating
them on Multi-BLIMP minimal pairs in 8 languages (ara, deu, eng, fra, heb,
hin, spa, tur). For each language, the top-20 GCM heads (aggregated
`mean_abs_ie` across all translation directions into that language) are
ablated individually and collectively on Δ = logp(correct) − logp(incorrect)
at the prefix-final position, and compared against stratified random controls,
sign-split sub-populations (POS-IE vs NEG-IE), and zero-ablation spot checks.

## Inputs

- GCM head rankings: `outputs/gcm_translation/<Src>__<Tgt>/top_rankings.json`
  (top-20 heads by mean-abs IE, aggregated across directions into the target
  language).
- Multi-BLIMP minimal pairs: `data/multilingual_pairs/{lang}.json` (n=400 per
  language; heb/hin at n=200 due to memory constraints).
- Llama-3.1-8B (bf16) loaded via nnsight; intervention point is
  `o_proj.input` (post-attention, per-query-head slice).

## Outputs

`outputs/head_ablation_multiblimp/<lang>/`:
- `summary.json`   — per-condition mean_delta + metadata
- `deltas.json`    — per-pair Δ values for every condition
- `mean_acts.pt`   — cached position-averaged head activations (for reuse)

`outputs/head_ablation_multiblimp/`:
- `analysis_summary.csv` — cross-language summary table (written by `analyze.py`)

`experiments/head_ablation_multiblimp/img/`:
- `<lang>_scatter_rank_vs_effect.png` — GCM rank vs per-head delta_change
- `<lang>_bars_per_head.png`          — paired top/ctrl bars ordered by GCM rank
- `<lang>_sign_split.png`             — POS-IE vs NEG-IE collective effect
- `<lang>_mean_vs_zero.png`           — mean vs zero ablation of top-5 collective
- `cross_lang_correlations.png`       — Spearman r (GCM rank vs effect) per language
- `cross_lang_summary.png`            — top-5 collective effect per language + ctrl overlay
- `cross_lang_sign_split.png`         — POS/NEG signed-IE collectives across languages
- `cross_lang_mean_vs_zero.png`       — top-5 mean-vs-zero ablation across languages
- `cross_lang_per_head_summary.png`   — average single-head top/control effects across languages

Figures are stored inside this experiment folder (not the repo-level `img/`)
to keep the experiment self-contained and portable.

## How to run

Smoke test (fra, 50 pairs, ~30 min):
```
qsub experiments/head_ablation_multiblimp/smoke.qsub
```

Full sweep (8 languages, n=400 each, as 8-task array):
```
qsub experiments/head_ablation_multiblimp/submit.qsub
```

Retry failed tasks from the first array (7 langs on A100-80G):
```
qsub experiments/head_ablation_multiblimp/submit_failed.qsub
```

Retry heb + hin at reduced n=200 (OOM mitigation):
```
qsub experiments/head_ablation_multiblimp/retry_heb_hin.qsub
```

Cross-language analysis (writes `analysis_summary.csv`):
```
python experiments/head_ablation_multiblimp/analyze.py
```

Figures (writes to `experiments/head_ablation_multiblimp/img/`):
```
python experiments/head_ablation_multiblimp/visualize.py
```

## Companion docs

- [REPORT.md](REPORT.md) — method details, full results table, key findings,
  caveats, and suggested headline.

## Status

v0.1 — 8/8 langs analyzed; heb/hin at n=200 pending n=400 retry with memory
fix (nnsight trace accumulation across 46-condition loop OOM'd on 80GB A100 at
n=400; tracked in REPORT.md caveats).
