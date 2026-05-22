# Head Ablation on Multi-BLIMP — Scientific Report

**Question**: Are attention heads that GCM identifies as causally important for *translation* also doing core *monolingual* work? Hypothesis (refined): translation circuitry is largely "hooking up" infrastructure that *borrows* monolingual LM machinery; the LM heads are also GCM-important for translation, but a sub-population of GCM-top heads is translation-specific.

**Status**: v0.1 complete for 8/8 target languages on May 11, 2026. Six
languages ran at n=400; Hebrew and Hindi currently use n=200 because the first
n=400 attempt OOM'd under the original nnsight loop. A v2 rerun path has been
implemented for matched sign-split controls and random-control variance, but it
has not produced saved outputs yet.

## Project context

The top-level project thesis is that multilingual LLMs translate largely by
reusing monolingual computations through shared multilingual features, rather
than through fully separate language-pair-specific translation modules. In the
project ledger, this experiment is mainly about **H4: translation uses the same
monolingual circuits the model already uses for language modeling**.

The immediate upstream experiment is `gcm_translation`. It applies Generative
Causal Mediation to FLORES translation pairs for all 56 ordered cross-language
directions among Arabic, English, French, German, Hebrew, Hindi, Spanish, and
Turkish. For each direction, it writes per-pair indirect-effect tensors for
attention heads:

```text
outputs/gcm_translation/<Source>__<Target>/heads_ie.pt
```

Those tensors have shape `[n_pairs, n_layers, n_heads]`. For Llama-3.1-8B this
means 32 layers x 32 query heads = 1024 head components per direction. This
experiment consumes those GCM head effects and asks whether the heads important
for translating *into* a target language also affect monolingual grammatical
minimal pairs *in that same target language*.

This is intentionally complementary to the SAE-feature counterfactual work
elsewhere in the repo. The SAE work studies sparse features at layer 16; this
experiment intervenes on actual attention-head output channels.

## Research framing

The initial simple hypothesis was: if translation reuses monolingual language
modeling circuits, then GCM-top translation heads should matter more for
Multi-BLIMP grammaticality than same-layer random controls.

The first full sweep pushed this into a more interesting hypothesis. GCM-top
translation heads are not homogeneous. Ranking by `mean_abs_ie` finds heads
that are important for the translation contrast, but the **signed** aggregated
IE separates two populations:

- **NEG-signed IE heads** look like shared LM machinery. Ablating them weakens
  the Multi-BLIMP grammatical preference.
- **POS-signed IE heads** look more translation-specific or routing-like.
  Ablating them does not hurt monolingual grammaticality and often slightly
  improves the Multi-BLIMP score.

So the emerging claim is not "all translation heads are grammar heads." It is
that the GCM-top set decomposes into shared monolingual infrastructure plus a
smaller translation/task-specific population.

## Method (one paragraph)

Take the top 20 attention heads by aggregated `mean_abs_ie` across all 7 `*__L` GCM directions (excluding self-source) per target language L. Score each Multi-BLIMP minimal pair as Δ = logp(correct_token | prefix) − logp(incorrect_token | prefix) at the prefix-final position. Run mean-ablation conditions (replace the head's `o_proj.input` slice with its position-averaged mean over BLIMP-L) and compare against (a) stratified random controls (same layer, head excluded from top-20), (b) sign-split top heads (positive vs negative `mean_signed_ie_agg`), and (c) zero-ablation spot check. n=400 per language. Llama-3.1-8B bf16.

## Method details

### Head selection from GCM

Head selection is implemented in `heads.py`. For each target language `L`, the
script aggregates over the seven cross-language GCM directories ending in that
target and excludes self-source directions. For English, for example, the
contributing directories are:

```text
Arabic__English
French__English
German__English
Hebrew__English
Hindi__English
Spanish__English
Turkish__English
```

For each source-target direction, the aggregation does:

1. Load `heads_ie.pt`.
2. Compute `nanmean(abs(IE), dim=0)` over pairs to get a `[layer, head]`
   absolute-importance map.
3. Compute `nanmean(IE, dim=0)` over pairs to get a signed-IE map.
4. Average both maps across the seven source directions into the target.
5. Select the target-language top 20 heads by aggregated mean absolute IE.

This is more principled than averaging only entries that appear in each
direction's pre-filtered top-k list, because pre-filtering would over-weight
single-direction outliers.

The current POS/NEG split among top-20 heads is:

| lang | POS heads | NEG heads | first 5 top heads by signed IE |
|---|---:|---:|---|
| ara | 5 | 15 | L26H23:-0.53, L17H21:-0.31, L23H28:-0.27, L21H9:-0.22, L31H2:+0.24 |
| deu | 7 | 13 | L30H25:+0.61, L30H27:-0.58, L23H28:-0.26, L15H17:-0.28, L28H10:-0.29 |
| eng | 5 | 15 | L31H25:+0.70, L31H3:-0.63, L23H28:-0.51, L31H24:-0.46, L30H25:+0.41 |
| fra | 4 | 16 | L30H25:+0.82, L30H27:-0.65, L17H21:-0.30, L23H28:-0.29, L28H10:-0.25 |
| heb | 7 | 13 | L17H21:-0.22, L30H24:-0.21, L16H1:-0.14, L31H2:+0.16, L30H25:+0.17 |
| hin | 5 | 15 | L30H25:+0.48, L31H1:-0.36, L30H24:-0.31, L31H2:+0.27, L17H21:-0.20 |
| spa | 6 | 14 | L30H25:+0.41, L30H27:-0.39, L17H21:-0.31, L28H10:-0.27, L23H28:-0.25 |
| tur | 5 | 15 | L30H25:+0.62, L30H27:-0.53, L31H2:+0.31, L31H3:-0.27, L20H1:-0.24 |

### Multi-BLIMP scoring

The pair preparation reuses `prepare_pair` from the
`counterfactual_attribution` pipeline. Each pair is reduced to a final-token
minimal contrast:

```text
Delta = log p(correct_final_token | prefix) - log p(incorrect_final_token | prefix)
```

The code records this as `mean_delta` after averaging over pairs. The main
analysis reports:

```text
delta_change = mean_delta(ablation) - mean_delta(baseline)
```

Interpretation:

- negative `delta_change`: ablation hurt grammaticality preference.
- positive `delta_change`: ablation improved grammaticality preference.

The current saved sample sizes are n=400 for Arabic, German, English, French,
Spanish, and Turkish, and n=200 for Hebrew and Hindi.

### Mean activation cache

Before scoring conditions, `run.py` computes a language-specific mean activation
for every `(layer, query_head)` over the sampled Multi-BLIMP inputs:

```text
mean_acts[layer, head, head_dim]
```

This pass traces each prepared input, saves every layer's
`self_attn.o_proj.input`, reshapes `[batch, seq, hidden]` to
`[batch, seq, n_heads, head_dim]`, sums over all batch/sequence positions, and
divides by the total number of positions. The result is a position-averaged
mean, not a position-conditional mean. It is saved as:

```text
outputs/head_ablation_multiblimp/<lang>/mean_acts.pt
```

### Ablation implementation

The conceptual intervention is to replace one or more head slices of
`self_attn.o_proj.input`:

- **mean ablation**: replace the head slice with its cached language mean.
- **zero ablation**: replace the head slice with zero.

nnsight does not support directly assigning to module inputs, so the code uses
the equivalent output-projection delta. If `x_h` is the original head slice,
`r_h` is the replacement, and `W_h` is the corresponding column slice of
`o_proj.weight`, then:

```text
delta_post = (r_h - x_h) @ W_h.T
self_attn.o_proj.output += delta_post
```

This is mathematically equivalent to replacing the input slice before the
linear output projection. The current implementation computes that delta one
head at a time, avoiding the full `[batch, seq, hidden]` clone that contributed
to earlier OOMs.

### Conditions in current saved outputs

Each current full run has 46 conditions:

- baseline
- 20 individual top-head mean ablations
- 20 individual same-layer control-head mean ablations
- top-5 collective mean ablation
- control-5 collective mean ablation
- POS-signed top-head collective mean ablation
- NEG-signed top-head collective mean ablation
- top-5 collective zero ablation

The random controls are stratified by layer: for each top head, sample one
non-top head from the same layer. This keeps layer distribution fixed between
the top and control sets.

### Implemented v2 conditions not yet in saved outputs

`run.py`, `analyze.py`, and `visualize.py` now support a richer v2 condition
set:

- matched-size random control for the POS-signed collective.
- matched-size random control for the NEG-signed collective.
- extra `ctrl_collective_k5_seedXX_mean` conditions for estimating
  random-control variance.

`submit_v2.qsub` requested `--n_ctrl_seeds 5`, which would create 53 conditions
per language. The current `summary.json` files still contain 46 conditions and
therefore the matched-control fields in `analysis_summary.csv` are `nan`.

## What has been run so far

### Smoke runs

Several French smoke runs were submitted first:

- `habl_smoke.5414624.out`: early 11-condition smoke completed.
- `habl_smoke.5415067.out`: 16-condition smoke hit a traceback.
- `habl_smoke.5415269.out`: 16-condition smoke completed.

These established that model loading, head aggregation, pair preparation, and
basic ablation hooks worked before the full sweep.

### First full array

`submit.qsub` launched an 8-language array at n=400:

```text
qsub experiments/head_ablation_multiblimp/submit.qsub
```

English completed successfully. Several other languages OOM'd on 44GB-class
GPUs, with the trace showing CUDA out-of-memory near the condition loop rather
than during model load.

### Failed-language retry

`submit_failed.qsub` retried the failed languages on A100-80G-style resources:

```text
qsub experiments/head_ablation_multiblimp/submit_failed.qsub
```

Arabic, German, French, Spanish, and Turkish completed at n=400. Hebrew and
Hindi still OOM'd at n=400 under the original 46-condition loop.

### Hebrew/Hindi reduced-n retry

`retry_heb_hin.qsub` reran Hebrew and Hindi with `--max_pairs 200`:

```text
qsub experiments/head_ablation_multiblimp/retry_heb_hin.qsub
```

Both completed. These are the current saved Hebrew/Hindi outputs.

### v2 attempt

`submit_v2.qsub` was added to run the richer 53-condition setup:

```text
qsub experiments/head_ablation_multiblimp/submit_v2.qsub
```

That attempt did not produce saved v2 outputs. Tasks 1, 3, and 4 landed on
Blackwell RTX PRO 6000 GPUs and were rejected by the compute-capability guard
because the current PyTorch build lacks sm_120 kernels. Task 2 for English
started on an sm_90 GPU and reached condition 48/53, but the log ends before
the save step, so it did not overwrite `outputs/head_ablation_multiblimp/eng`.

### Analysis and visualization

The current cross-language table comes from:

```text
python experiments/head_ablation_multiblimp/analyze.py
```

which writes:

```text
outputs/head_ablation_multiblimp/analysis_summary.csv
```

Figures come from:

```text
python experiments/head_ablation_multiblimp/visualize.py
```

and are saved in:

```text
experiments/head_ablation_multiblimp/img/
```

## Cross-language results (all 8 langs)

Δ change from baseline per condition (more negative = ablation hurt grammaticality more):

| lang | n   | baseline | top-5 mean | ctrl-5 mean | top-5 zero | **POS-IE collective** | **NEG-IE collective** |
|------|-----|----------|------------|-------------|------------|------------------------|------------------------|
| eng  | 400 | +5.23    | +0.06      | −0.02       | +0.04      | +0.08 (n=5)            | **−0.16 (n=15)**       |
| ara  | 400 | +7.45    | −0.14      | −0.06       | −0.17      | +0.11 (n=5)            | **−0.47 (n=15)**       |
| deu  | 400 | +10.18   | −0.00      | −0.01       | +0.04      | +0.05 (n=7)            | **−1.02 (n=13)**       |
| fra  | 400 | +9.16    | −0.01      | −0.01       | −0.01      | +0.09 (n=4)            | **−0.28 (n=16)**       |
| spa  | 400 | +9.61    | +0.02      | −0.01       | +0.03      | +0.10 (n=6)            | **−0.12 (n=14)**       |
| tur  | 400 | +9.56    | +0.05      | +0.02       | +0.01      | +0.18 (n=5)            | **−0.65 (n=15)**       |
| heb  | 200 | +11.48   | +0.07      | +0.06       | −0.05      | +0.16 (n=7)            | **−0.28 (n=13)**       |
| hin  | 200 | +6.33    | −0.05      | **+0.23**   | −0.08      | +0.22 (n=5)            | **−0.27 (n=15)**       |

## Key findings

1. **GCM signed-IE direction predicts the direction of the BLIMP ablation effect in all 8 langs.** Ablating the NEG-signed-IE subset of GCM top heads produces a negative Δ change (range −0.12 to −1.02), i.e. grammaticality preference weakens. Ablating the POS-signed-IE subset produces a positive Δ change (range +0.05 to +0.22), i.e. grammaticality slightly improves. Both directions are consistent across 8/8 languages; magnitudes vary substantially.
2. **Top-K mixed and matched random controls produce near-noise in 7/8 langs** (±0.06 typically), because the mixed top-K combines opposing-direction populations and bf16 quantization at Δ ≈ 5–10 has a noise floor of ~0.06. **Hindi is an exception**: its `ctrl_collective_k5` produced +0.23, comparable in magnitude to its POS-IE collective (+0.22). The stratified-by-layer random sample happened to land on heads that, when ablated, increase grammaticality in this language. Reason unclear — may be (a) genuine head-importance heterogeneity in Hindi, (b) the smaller n=200 reducing averaging, or (c) the controls being correlated with the POS-IE subset at the layer level.
3. **Consistent with the refined hypothesis**: the NEG-signed-IE majority (~65–80% of GCM top-20) behaves like shared LM machinery that translation borrows. The smaller POS-signed-IE minority behaves like translation-specific routing; ablating those heads doesn't carry monolingual grammaticality information and in fact slightly improves the BLIMP metric. The hin control divergence is a caveat that warrants a separate look (e.g., scatter of per-head delta_change vs layer index for hin).
4. **Effect-size magnitude varies but is not a simple function of baseline Δ**: deu (baseline +10.2) shows the biggest absolute drop in the NEG ablation (−1.02); eng (baseline +5.2) the smallest (−0.16). Hindi (baseline +6.3) shows a NEG effect (−0.27) similar in magnitude to fra/heb (~−0.28) despite a lower baseline. No clean monotonic relationship.
5. **Per-head k=1 ablations were not informative** — single-head effects are below bf16 noise floor on confident BLIMP pairs. The signal lives in collective ablations of correctly-direction-grouped heads.

## Interpretation

The best current reading is a mixed-circuit story:

1. **Shared monolingual machinery:** The NEG-signed majority of GCM-top heads
   behaves like language-modeling infrastructure used by translation. Removing
   those heads weakens grammatical preferences in the target language.
2. **Translation/task routing:** The POS-signed minority behaves differently.
   Removing those heads slightly improves the monolingual minimal-pair metric,
   suggesting they are not the part of the circuit that carries ordinary
   grammaticality information.

This also explains why the mixed top-5 collective is weak. A top-k set chosen
only by absolute IE can contain both groups, and their effects partially cancel.
For this downstream monolingual test, GCM signed direction is more informative
than GCM absolute rank alone.

The per-head tests are not the main evidence. `analysis_summary.csv` includes
Spearman rank/effect correlations and Welch top-vs-control tests, but those are
mostly unconvincing because individual head effects are small. The robust signal
comes from grouping heads by signed IE.

## Caveats & open questions

- **heb + hin had to be re-run at n=200** (n=400 OOM'd mid-loop even on 80GB A100; nnsight trace operations accumulate memory across the 46-condition loop). All other langs are at n=400.
- **v2 controls are implemented but not complete**. The saved summaries do not
  yet include matched-size POS/NEG controls or extra random-control seeds, so
  the size-controlled sign-split comparison still needs the v2 rerun.
- **Hindi control divergence**: the matched random control (ctrl_collective_k5) produced +0.23 in hin, comparable to its POS-IE collective effect. Not seen in the other 7 langs. Could be a genuine signal about Hindi head-importance distribution, an artifact of n=200, or correlation between the random sample and the POS-IE subset. Worth a follow-up.
- **bf16 quantization** of per-pair Δ (1 ULP ≈ 0.06 at Δ≈10) means single-head and small-effect signals get clipped. Float32 scoring or harder pairs would tighten the analysis.
- **Mean ablation is gentle by construction** — replaces head output with its in-distribution average. Zero ablation gives qualitatively similar pattern; resample ablation deferred.
- **GQA caveat**: ablation reshapes by query heads (32 × 128); the K/V heads (8) are not directly addressable as our intervention point is `o_proj.input` (post-attention, per-query-head concat).
- **Position-averaged mean** could be sharpened to position-conditional (per-token-position mean); deferred.
- **k=5 collective was the only group size tested** outside sign-split. k=10, k=20 collective ablations would map the dose-response curve.
- **Only final-token Multi-BLIMP scoring is used**. This is aligned with the
  prepared minimal-pair setup, but it is narrower than full-continuation scoring.
- **Blackwell nodes are currently unusable with this environment**. The
  compute-capability guard correctly refuses sm_120 GPUs because the installed
  PyTorch build supports up to sm_90.

## Recommended next runs

1. Rerun `submit_v2.qsub` on supported A100/H100-class GPUs, avoiding sm_120
   Blackwell nodes unless the conda environment is rebuilt.
2. Complete Hebrew and Hindi at n=400 using the memory-fixed implementation.
3. Promote matched POS/NEG controls into the main result table once v2 outputs
   exist.
4. Use `--n_ctrl_seeds 5` or higher to estimate random-control variance,
   especially for Hindi.
5. Add a Hindi diagnostic plot: per-head delta change by layer/head for top
   heads and all sampled controls.
6. Add k=10 and k=20 collective ablations to map dose response.
7. Run a float32 or harder-pair spot check to estimate the true per-head noise
   floor.

## Figures

- `experiments/head_ablation_multiblimp/img/<lang>_scatter_rank_vs_effect.png` — GCM rank vs ablation effect per language
- `experiments/head_ablation_multiblimp/img/<lang>_bars_per_head.png` — per-head paired top vs control
- `experiments/head_ablation_multiblimp/img/<lang>_sign_split.png` — POS-IE vs NEG-IE collective effect
- `experiments/head_ablation_multiblimp/img/<lang>_sign_split_matched.png` — POS/NEG plot layout for matched controls; currently limited by missing v2 outputs
- `experiments/head_ablation_multiblimp/img/<lang>_mean_vs_zero.png` — top-5 mean vs zero ablation
- `experiments/head_ablation_multiblimp/img/cross_lang_correlations.png` — Spearman r per language
- `experiments/head_ablation_multiblimp/img/cross_lang_summary.png` — top-5 collective effect per language

## Saved outputs

- `outputs/head_ablation_multiblimp/<lang>/summary.json` — condition-level
  means, metadata, top heads, controls, and pair metadata.
- `outputs/head_ablation_multiblimp/<lang>/deltas.json` — per-pair deltas for
  every condition.
- `outputs/head_ablation_multiblimp/<lang>/mean_acts.pt` — cached mean
  activations used for mean ablation.
- `outputs/head_ablation_multiblimp/analysis_summary.csv` — cross-language
  analysis table.

## Suggested headline for the meeting

> The signed-IE direction in GCM appears to be a meaningful axis. In all 8 languages tested, ablating the NEG-signed-IE subset of GCM top heads produced a negative Δ change on Multi-BLIMP (range 0.12–1.02 logp units), while ablating the POS-signed-IE subset produced a positive Δ change (range 0.05–0.22). The mixed top-K and stratified random controls produced near-noise in 7/8 langs — Hindi was an exception where the random control also shifted positively. The direction effect is consistent with the refined hypothesis that GCM-identified translation heads decompose into shared monolingual LM machinery (NEG-IE majority) and a smaller translation-specific population (POS-IE minority), with Hindi flagging a per-language control-quality question worth follow-up.
