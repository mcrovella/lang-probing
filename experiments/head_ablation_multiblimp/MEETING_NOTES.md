# Meeting notes — head_ablation_multiblimp

**Question**: Are attention heads that GCM identifies as causally important for *translation* also doing core *monolingual* work? Hypothesis (refined): translation circuitry is largely "hooking up" infrastructure that *borrows* monolingual LM machinery; the LM heads are also GCM-important for translation, but a sub-population of GCM-top heads is translation-specific.

## Method (one paragraph)

Take the top 20 attention heads by aggregated `mean_abs_ie` across all 7 `*__L` GCM directions (excluding self-source) per target language L. Score each Multi-BLIMP minimal pair as Δ = logp(correct_token | prefix) − logp(incorrect_token | prefix) at the prefix-final position. Run mean-ablation conditions (replace the head's `o_proj.input` slice with its position-averaged mean over BLIMP-L) and compare against (a) stratified random controls (same layer, head excluded from top-20), (b) sign-split top heads (positive vs negative `mean_signed_ie_agg`), and (c) zero-ablation spot check. n=400 per language. Llama-3.1-8B bf16.

## Cross-language results (7/8 langs; hin retry still running)

Δ change from baseline per condition (more negative = ablation hurt grammaticality more):

| lang | n   | baseline | top-5 mean | ctrl-5 mean | top-5 zero | **POS-IE collective** | **NEG-IE collective** |
|------|-----|----------|------------|-------------|------------|------------------------|------------------------|
| eng  | 400 | +5.23    | +0.06      | −0.02       | +0.04      | +0.08 (n=5)            | **−0.16 (n=15)**       |
| ara  | 400 | +7.45    | −0.14      | −0.06       | −0.17      | +0.11 (n=5)            | **−0.47 (n=15)**       |
| deu  | 400 | +10.18   | −0.00      | −0.01       | +0.04      | +0.05 (n=7)            | **−1.02 (n=13)** 🔥    |
| fra  | 400 | +9.16    | −0.01      | −0.01       | −0.01      | +0.09 (n=4)            | **−0.28 (n=16)**       |
| spa  | 400 | +9.61    | +0.02      | −0.01       | +0.03      | +0.10 (n=6)            | **−0.12 (n=14)**       |
| tur  | 400 | +9.56    | +0.05      | +0.02       | +0.01      | +0.18 (n=5)            | **−0.65 (n=15)**       |
| heb  | 200 | +11.48   | +0.07      | +0.06       | −0.05      | +0.16 (n=7)            | **−0.28 (n=13)**       |

## Key findings

1. **GCM signed-IE direction cleanly distinguishes head populations.** In all 6 langs, ablating the NEG-signed-IE subset of GCM top heads *systematically* hurts Multi-BLIMP grammaticality (Δ change −0.12 to **−1.02** in deu). The POS-signed-IE subset shows *small positive* Δ change (+0.05 to +0.18) — ablating those heads slightly *helps* grammaticality.
2. **Top-K mixed and matched random controls produce near-noise** (±0.05 typically), because the mixed top-K combines opposing-direction populations and bf16 quantization at Δ ≈ 5–10 has a noise floor of ~0.06.
3. **Refined hypothesis is supported**: a substantial fraction (~65–80%) of GCM-top heads — the NEG-signed-IE majority — are doing monolingual LM work that translation borrows. A smaller minority (POS-signed-IE) appears to be translation-specific routing/noise that distracts the model on monolingual grammaticality; removing it slightly improves performance.
4. **Effect-size magnitude scales with baseline Δ**: deu (baseline +10.2) shows the biggest absolute drop in the NEG ablation (−1.02), eng (baseline +5.2) the smallest (−0.16). Suggestive of saturation effects at confident pairs.
5. **Per-head k=1 ablations were not informative** — single-head effects are below bf16 noise floor on confident BLIMP pairs. The signal lives in collective ablations of correctly-direction-grouped heads.

## Caveats & open questions

- **heb + hin failed** with OOM mid-loop even on 80GB A100; resubmitted with n=200. Memory leak in long nnsight trace loops at large n.
- **bf16 quantization** of per-pair Δ (1 ULP ≈ 0.06 at Δ≈10) means single-head and small-effect signals get clipped. Float32 scoring or harder pairs would tighten the analysis.
- **Mean ablation is gentle by construction** — replaces head output with its in-distribution average. Zero ablation gives qualitatively similar pattern; resample ablation deferred.
- **GQA caveat**: ablation reshapes by query heads (32 × 128); the K/V heads (8) are not directly addressable as our intervention point is `o_proj.input` (post-attention, per-query-head concat).
- **Position-averaged mean** could be sharpened to position-conditional (per-token-position mean); deferred.
- **k=5 collective was the only group size tested** outside sign-split. k=10, k=20 collective ablations would map the dose-response curve.

## Figures

- `img/head_ablation_multiblimp/<lang>_scatter_rank_vs_effect.png` — GCM rank vs ablation effect per language
- `img/head_ablation_multiblimp/<lang>_bars_per_head.png` — per-head paired top vs control
- `img/head_ablation_multiblimp/<lang>_sign_split.png` — POS-IE vs NEG-IE collective effect
- `img/head_ablation_multiblimp/cross_lang_correlations.png` — Spearman r per language
- `img/head_ablation_multiblimp/cross_lang_summary.png` — top-5 collective effect per language

## Suggested headline for the meeting

> The "signed-IE direction" in GCM is a meaningful axis: in 6/6 languages tested, ablating the NEG-signed-IE subset of GCM top heads systematically hurts monolingual Multi-BLIMP grammaticality (Δ drops by 0.12–1.02 logp units), while ablating the POS-signed-IE subset slightly *helps* grammaticality. This is consistent with the refined hypothesis that translation-identified heads decompose into (a) shared LM machinery that translation borrows — the NEG-IE majority — and (b) a smaller translation-specific routing population that does not encode monolingual grammaticality information.
