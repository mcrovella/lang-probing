# `input_output_overlap`

**Hypotheses:** H2 (the central claim).

**Question:** Do the top-K SAE features identified monolingually (input features) overlap with the top-K features driving a probe during translation (output features)? Across how many languages and concepts?

**Method:** Pure consumer. Load input features (`outputs/input_features/…/diff_vector.pt`) and output features (`outputs/output_features/…/effects_{Src}_{Tgt}.pt`); compute Jaccard at top-K, signal plots (top-k magnitude vs complementary-set rank), per-language distributions.

## Run

```bash
python experiments/input_output_overlap/visualize.py
```

## Inputs

- `outputs/input_features/{Lang}/{Concept}/{Value}/diff_vector.pt`
- `outputs/output_features/.../effects_{Src}_{Tgt}.pt`

## Outputs

- `img/input_output_overlap/{Lang}_{Concept}_{Value}/jaccard_topk.png`
- `img/input_output_overlap/{Lang}_{Concept}_{Value}/jaccard_topk_abs.png`
- `img/input_output_overlap/aggregate_jaccard_topk_raw.png`
- `img/input_output_overlap/aggregate_jaccard_topk_abs.png`
- `outputs/input_output_overlap/jaccard_summary.csv`
- `img/input_output_overlap/{Lang}_{Concept}_{Value}/signal_input.png`
- `img/input_output_overlap/{Lang}_{Concept}_{Value}/signal_output.png`

Some signal-plot code paths are commented out in `visualize.py`; TODO to re-enable and verify.

## Status

Active. The paper-todo pass added an aggregate summary. Current successful coverage is 25 target-language/concept/value cells; mean Jaccard@200 is 0.0947 using raw scores and 0.1220 using absolute-magnitude top-k. Several cells are missing input vectors or have malformed scalar output tensors (notably Hindi tense), so H2 should be stated cautiously until those cells are repaired. See [LEDGER.md](../../LEDGER.md#input_output_overlap).
