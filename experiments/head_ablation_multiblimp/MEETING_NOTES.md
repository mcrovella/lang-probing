# Meeting notes — head_ablation_multiblimp

**Question**: Are attention heads that GCM identifies as causally important for *translation* also doing core *monolingual* work? Hypothesis (refined): translation circuitry is largely "hooking up" infrastructure that *borrows* monolingual LM machinery; the LM heads are also GCM-important for translation, but a sub-population of GCM-top heads is translation-specific.

## Method (one paragraph)

Take the top 20 attention heads by aggregated `mean_abs_ie` across all 7 `*__L` GCM directions (excluding self-source) per target language L. Score each Multi-BLIMP minimal pair as Δ = logp(correct_token | prefix) − logp(incorrect_token | prefix) at the prefix-final position. Run mean-ablation conditions (replace the head's `o_proj.input` slice with its position-averaged mean over BLIMP-L) and compare against (a) stratified random controls (same layer, head excluded from top-20), (b) sign-split top heads (positive vs negative `mean_signed_ie_agg`), and (c) zero-ablation spot check. n=400 per language. Llama-3.1-8B bf16.

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

## Caveats & open questions

- **heb + hin had to be re-run at n=200** (n=400 OOM'd mid-loop even on 80GB A100; nnsight trace operations accumulate memory across the 46-condition loop). All other langs are at n=400.
- **Hindi control divergence**: the matched random control (ctrl_collective_k5) produced +0.23 in hin, comparable to its POS-IE collective effect. Not seen in the other 7 langs. Could be a genuine signal about Hindi head-importance distribution, an artifact of n=200, or correlation between the random sample and the POS-IE subset. Worth a follow-up.
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

> The signed-IE direction in GCM appears to be a meaningful axis. In all 8 languages tested, ablating the NEG-signed-IE subset of GCM top heads produced a negative Δ change on Multi-BLIMP (range 0.12–1.02 logp units), while ablating the POS-signed-IE subset produced a positive Δ change (range 0.05–0.22). The mixed top-K and stratified random controls produced near-noise in 7/8 langs — Hindi was an exception where the random control also shifted positively. The direction effect is consistent with the refined hypothesis that GCM-identified translation heads decompose into shared monolingual LM machinery (NEG-IE majority) and a smaller translation-specific population (POS-IE minority), with Hindi flagging a per-language control-quality question worth follow-up.
