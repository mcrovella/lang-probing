# `output_features`

**Hypotheses:** H2 — features the model reads *from* when writing a grammatical concept.

**Question:** Which SAE features does a late-layer grammatical concept probe's prediction depend on during translation? (Gradient attribution from probe logit → SAE features.)

**Method:** Load a late-layer (32) concept probe trained by `experiments/probes`. Convert sklearn LR → torch `LinearLayer` (fusing the StandardScaler). Run nnsight trace at layer 32 on FLORES sentence pairs; attribute the probe logit's gradient to SAE features at layer 16. Accumulate effects per (src, tgt).

## Run

```bash
python experiments/output_features/run.py \
    --language English \
    --layer 32 \
    --num_probe_samples 1024 \
    --max_samples 256 \
    --batch_size 8 \
    --output_dir outputs/output_features
```

Batch-submission template: `run/attribution_flores.sh` (currently a one-liner — polish needed).

## Dependencies

- **`torchtyping`** (via `src/lang_probing_src/features/sparse_activations.py`). It is listed in the repo `requirements.txt`; the current `probes` conda env used during the paper-todo pass did **not** have it installed (`importlib.util.find_spec("torchtyping") == None`), so reruns need the environment refreshed.
- Word probes at `outputs/probes/word_probes/`.
- FLORES-101 sentence pairs.
- Llama-3.1-8B + layer-16 SAE.

## Known issues

- The old `from src.config ...` import bug is fixed in `run.py`; it now imports from `lang_probing_src.config`.
- Saved outputs are timestamped and include at least one tiny pilot (`English_French`, 2 examples) plus later 256-example runs. Downstream loading now indexes the latest timestamped file per `(source, target)` pair before loading, to avoid nondeterministic overwrites.

## Outputs

- `outputs/output_features/{timestamp}/effects_{Source}_{Target}.pt` + `config.json`.
- Plan is to collapse to one canonical per (src, tgt) pair in Wave 4.

## Figures

No direct figures; consumed by `experiments/input_output_overlap`.

## Status

Active but hampered by the import bug + missing dependency. See [LEDGER.md](../../LEDGER.md#output_features).
