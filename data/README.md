# `data/`

Small committed datasets. Anything big (model weights, parquet caches, treebanks) lives elsewhere — see `src/lang_probing_src/config.py` paths.

## `grammatical_pairs.json`

Minimal pairs for counterfactual attribution (`experiments/counterfactual_attribution/`).

Currently 30 English pairs spanning 7 concepts: Aspect, Gender, Mood, Number, Person, Polarity, Tense.

### Schema

```jsonc
{
  "id": "number_01",                 // short unique id, used for filenames
  "prefix": "The cats",               // tokens the model conditions on
  "original_token": " sit",           // correct next-token (with leading space)
  "counterfactual_token": " sits",    // alternative next-token with different concept value
  "concept": "Number",                // one of Aspect, Gender, Mood, Number, Person, Polarity, Tense
  "concept_value_orig": "Plur",
  "concept_value_cf": "Sing",
  "note": "Subject-verb number agreement: plural subject expects plural verb"
}
```

Constraints:
- `original_token` and `counterfactual_token` must each be a **single Llama-3.1 token** after tokenization. `counterfactual_attribution/run.py` skips multi-token pairs and lists them in `outputs/counterfactual_attribution/skipped_pairs.json`.
- Pairs must differ on exactly one grammatical dimension; lexical similarity (e.g. `sit` ↔ `sits`) keeps the probe gradient interpretable.

### Multilingual extension (planned)

To add a new language, create a sibling file `grammatical_pairs_{lang}.json` (e.g. `grammatical_pairs_spanish.json`) with the same schema. The counterfactual runner supports a `--data_file` argument and can be invoked per-language:

```bash
python experiments/counterfactual_attribution/run.py \
    --data_file data/grammatical_pairs_spanish.json \
    --output_dir outputs/counterfactual_attribution/spanish
```

Per-concept counts should be >= 5 to make cross-concept comparisons stable (the current English set has Polarity=2, Aspect/Mood=3 — see [TODO.md](../TODO.md)).

### Suggested language seed set

For parity with existing FLORES / Multi-BLiMP work, prioritize:

- English (done)
- Spanish
- French
- German
- Turkish (agglutinative; interesting case/morphology)
- Chinese (no morphological agreement; tests whether features are robust without explicit grammar)

## `language_metrics.csv` and `pair_metrics.csv`

Shareable metric tables produced by `experiments/perplexity_bleu_linear/compile_shareable_metrics.py` (received from Jake, 2026-05-23). These are the canonical joined-metric snapshots feeding the H1 (`perplexity_bleu_linear`) analyses without having to re-run the upstream perplexity / PER / BLEU pipeline.

Both files cover **2 models** (`llama`, `aya`) × **24 languages** (codes match `LANG_CODE_TO_NAME` in `src/lang_probing_src/config.py`).

### `language_metrics.csv` (48 rows = 2 models × 24 langs)

One row per `(model, lang)`. Columns:

| column | meaning |
|---|---|
| `model` | `llama` or `aya` |
| `lang` | language code (e.g. `eng`, `zho_simpl`) |
| `lang_name` | human-readable name from `LANG_CODE_TO_NAME` |
| `flores_perplexity` | corpus PPL on FLORES devtest (from `run_perplexity.py`) |
| `multiblimp_accuracy` | Multi-BLiMP accuracy = 1 − error rate (from `run_per.py`) |
| `avg_bleu_as_source` | mean BLEU across pairs where this lang is source |
| `n_bleu_pairs_as_source` | count of such pairs |
| `avg_bleu_as_target` | mean BLEU across pairs where this lang is target |
| `n_bleu_pairs_as_target` | count of such pairs |

Missing metrics are blank (NaN) — e.g. Arabic has no `multiblimp_accuracy` in some rows.

### `pair_metrics.csv` (1104 rows = 2 models × 552 pairs)

One row per `(model, src, tgt)`. 552 pairs/model = 24×24 − 24 diagonal (no self-pairs). Columns:

| column | meaning |
|---|---|
| `model`, `src`, `tgt` | identifiers |
| `src_lang_name`, `tgt_lang_name` | human-readable |
| `bleu` | BLEU score for the directed pair |
| `src_flores_perplexity`, `tgt_flores_perplexity` | joined from `language_metrics` |
| `src_multiblimp_accuracy`, `tgt_multiblimp_accuracy` | joined from `language_metrics` |

### Use

These are inputs of convenience: anything in `experiments/perplexity_bleu_linear/` that consumes `combined_results_{model}.csv` can be rewritten against `pair_metrics.csv` (filtering by `model`) without needing the BU SCC outputs locally. The accuracy column here is the complement of the error rate stored upstream — convert if comparing directly.
