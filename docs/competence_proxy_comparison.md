# Competence proxies for predicting translation BLEU

Date: 2026-05-22 · Model: Llama-3.1-8B

## Motivation

H1 claims translation BLEU for a pair $(S,T)$ is predictable from monolingual
competence in $S$ and $T$. The original fit used **per-token corpus perplexity**
on FLORES and gave $R^2 = 0.02$ — implausibly low. An earlier audit traced this
to perplexity being tokenization-confounded (English has *higher* per-token
perplexity than Hebrew, the reverse of real competence). This report tests three
alternative competence proxies head-to-head for how well they predict BLEU:

1. **Bits-per-byte (BPB)** — newly implemented; perplexity normalized by raw UTF-8
   bytes instead of tokens, removing the tokenization confound.
2. **MultiBLiMP accuracy** — fraction of MultiBLiMP minimal pairs where the model
   prefers the grammatical sentence (a grammatical-competence measure).
3. **Perplexity accuracy rate** ( = 1 − PER) — the same grammatical preference,
   scored by per-sentence perplexity instead of total log-prob.

Per-token perplexity is kept as the **baseline**.

## Headline result

**Grammatical-judgment accuracy predicts BLEU; fluency/perplexity does not.** On
the common 17-language subset (272 ordered pairs), competence-from-grammar
reaches $R^2 = 0.25$ with the correct sign, while per-token perplexity ($0.07$)
and bits-per-byte ($0.01$) are near-zero **and have the wrong sign**. Removing
the tokenization confound (BPB) did **not** rescue perplexity — so the failure is
not merely tokenization; monolingual *fluency/compression* is simply the wrong
*kind* of signal, whereas monolingual *grammatical acceptability* transfers to
translation quality.

![R^2 of each proxy](../outputs/perplexity_bleu_linear/competence_proxy_comparison/fig1_r2_comparison.png)

*Figure 1. Predictive power ($R^2$ of OLS `BLEU ~ a + b_src·proxy_src + b_tgt·proxy_tgt`) for each proxy: light = the proxy's own coverage (24 langs / 552 pairs for the fluency proxies; 17 langs / 272 pairs for MultiBLiMP), dark = the common 17-language / 272-pair subset. Perplexity-accuracy (1−PER) is omitted because it is identical to MultiBLiMP accuracy. Produced by `compare_competence_proxies.py`.*

## Predictive-power comparison

All four models below are fit on the **same** common 17-language set MultiBLiMP
covers (`ces, deu, ell, eng, fas, fra, heb, hin, ita, nld, pol, por, ron, rus,
spa, tur, ukr`; **272 ordered pairs**), so they are directly comparable.
Per-token perplexity and bits-per-byte are defined for all 24 FLORES languages;
restricting them to these 17 is what makes the comparison apples-to-apples.

| Model | R² | adj. R² | MAE (BLEU) | n pairs |
|---|---|---|---|---|
| **MultiBLiMP accuracy — linear** | **0.250** | 0.244 | 4.86 | 272 |
| MultiBLiMP accuracy — bilinear (interaction) | 0.251 | 0.243 | 4.84 | 272 |
| Per-token perplexity — linear | 0.071 | 0.064 | 5.43 | 272 |
| Bits-per-byte — linear | 0.012 | 0.005 | 5.49 | 272 |

MultiBLiMP accuracy ≡ perplexity-accuracy (1−PER) **exactly** (Pearson r=1.000,
max|diff|=0.0 over these 17 languages), so it is one row, not two. The bilinear
model does not beat the linear one (adj-R² 0.243 vs 0.244) — consistent with
H1's no-interaction prediction.

![MultiBLiMP accuracy vs BLEU](../outputs/perplexity_bleu_linear/competence_proxy_comparison/fig_multiblimp_marginal.png)

*Figure 2. MultiBLiMP grammatical-acceptability accuracy vs mean BLEU when translating INTO each language, over the 17 MultiBLiMP languages (ces, deu, ell, eng, fas, fra, heb, hin, ita, nld, pol, por, ron, rus, spa, tur, ukr). Spearman ρ=+0.62 (p=0.008). This is the figure used in the draft. Produced by `compare_competence_proxies.py`.*

Coefficient signs and marginal correlations (same common 17 languages):

| Proxy | b_src | b_tgt | marginal Spearman ρ (proxy vs mean BLEU into-lang) |
|---|---|---|---|
| Per-token perplexity | +0.07 | +0.36 | +0.23 (p=0.37, ns; **wrong sign**) |
| Bits-per-byte | +2.88 | +3.45 | −0.03 (p=0.91; ~zero) |
| MultiBLiMP accuracy (≡ 1−PER) | +19.3 | +65.1 | +0.62 (p=0.008) |

For context, on each fluency proxy's **full 24-language** coverage (552 pairs):
per-token perplexity R²=0.021, bits-per-byte R²=0.051 — the slightly higher BPB
number there comes only from separating the 7 non-MultiBLiMP Asian languages
(ara, ind, jpn, kor, vie, zho_simpl, zho_trad) by script family, not from
measuring competence within a language set (on the common 17 it is R²=0.012).

![Predicted vs actual](../outputs/perplexity_bleu_linear/competence_proxy_comparison/fig2_pred_vs_actual.png)

*Figure 3. Predicted vs actual BLEU on the common 17-language subset (ces…ukr, 272 pairs). Only grammatical-accuracy shows a real positive relationship; perplexity and bits-per-byte are flat clouds. Produced by `compare_competence_proxies.py`.*

![Per-language marginal, all proxies](../outputs/perplexity_bleu_linear/competence_proxy_comparison/fig3_marginal.png)

*Figure 4. Per-language proxy value vs mean BLEU into language, all three proxies, common 17 languages (ces…ukr). Grammatical accuracy tracks competence (ρ=+0.62, p=0.008); bits-per-byte is uncorrelated (ρ=−0.03). Produced by `compare_competence_proxies.py`.*

## Findings

1. **The two grammatical proxies are the *same number*.** MultiBLiMP accuracy
   (from `shareable_metrics/language_metrics.csv`) equals 1 − PER (from the
   `per/error_rates_by_language_*.json`) **exactly** (Pearson r = 1.000,
   max |diff| = 0.0, over the 17 common languages). So the three requested
   measures are really **two distinct constructs**: a fluency/perplexity family
   (per-token PPL, bits-per-byte) and a single grammatical-acceptability accuracy.
   Perplexity-accuracy is therefore dropped from all tables/figures (it would
   duplicate the MultiBLiMP-accuracy row exactly), not shown as independent
   corroboration.

2. **Grammatical accuracy is a real, correctly-signed competence proxy:**
   $R^2 = 0.250$, both coefficients positive, target-language competence weighted
   $\sim$3.4× the source ($b_{tgt}=65$ vs $b_{src}=19$) — sensible, since
   producing grammatical *target* text is the harder half of translation. The
   marginal correlation (proxy vs mean BLEU into a language) is $\rho = +0.62$,
   $p = 0.008$.

3. **Per-token perplexity is not a competence proxy:** $R^2 = 0.07$ on the common
   subset, **positive** coefficients (wrong direction — higher perplexity would
   "predict" higher BLEU), marginal $\rho=+0.23$ (n.s.). Consistent with the
   tokenization-confound audit (English: high PPL yet best translator).

4. **Bits-per-byte does not fix it.** Despite removing the tokenization confound
   (byte/token ratios are now sensible: heb 1.85, rus 5.93, eng 4.86), BPB has
   essentially **zero** marginal correlation with translation competence
   ($\rho=-0.03$, $p=0.91$) and wrong-signed coefficients. Its slightly higher
   *own-coverage* $R^2$ (0.051 on 24 langs vs 0.021 for PPL) comes only from
   separating the 7 non-MultiBLiMP Asian languages (ara, ind, jpn, kor, vie,
   zho×2) by script family — not from measuring competence within a language set.
   On the European-heavy common 17 it carries no signal. **Takeaway: the problem
   was never just tokenization — monolingual fluency/compression does not capture
   the competence that transfers to translation.**

5. **Functional form is not the bottleneck.** Adding the multiplicative
   interaction term changes $R^2$ by $\le 0.001$ for every proxy. (Note: this is a
   weaker claim than the rank-1 BLEU-matrix result, which finds the BLEU surface
   $\approx f(S)g(T)$ at 88% faithfulness — see `draft_v7.tex` §3. The interaction
   term here is on the *proxy* scale, not the latent-competence scale.)

## Methodology

### Bits-per-byte (exact reconstruction from saved PPL)

`run_perplexity.py` already computed corpus perplexity
$\text{PPL} = \exp(\text{NLL}_\text{total}/N_\text{tok})$ on FLORES devtest, where
$N_\text{tok}$ is the number of *scored* tokens = $\sum_\text{sent}(\,|\text{tokens incl. BOS}|-1)$
(causal shift; no truncation, since FLORES sentences $\ll$ `max_position_embeddings`).
Therefore the corpus NLL is recoverable exactly:
$$\text{NLL}_\text{total} = N_\text{tok}\,\ln(\text{PPL}), \qquad
\text{BPB} = \frac{\text{NLL}_\text{total}}{N_\text{bytes}\,\ln 2}
= \frac{N_\text{tok}\,\ln(\text{PPL})}{N_\text{bytes}\,\ln 2},$$
where $N_\text{bytes}=\sum_\text{sent}|\text{utf-8}(\text{sent})|$. `compute_bits_per_byte.py`
recomputes $N_\text{tok}$ (Llama tokenizer, CPU) and $N_\text{bytes}$ exactly as
`run_perplexity` scored them, and reads PPL from the saved CSV. This is
algebraically identical to a direct $\sum\text{NLL}/\sum\text{bytes}$ GPU pass; a
GPU pass would only re-derive the same PPL we already have. Output:
`outputs/perplexity_bleu_linear/bleu_and_ppl/bits_per_byte_llama.csv`.

### Proxy sources
- `flores_ppl_per_token`: `bleu_and_ppl/perplexity_results_llama.csv` (24 langs).
- `bits_per_byte`: as above (24 langs).
- `multiblimp_accuracy`: `shareable_metrics/language_metrics.csv` (17 langs).
- `perplexity_accuracy` = 1 − PER: `per/error_rates_by_language_*Llama*.json` (17 langs).

### BLEU and regression
- BLEU: `bleu_and_ppl/bleu_results_llama.csv` — 552 ordered pairs over 24 FLORES
  languages (2-shot greedy `src >> tgt||` prompting, sacrebleu corpus BLEU; see
  `draft_v7.tex` §3 footnote).
- For each proxy: OLS `BLEU ~ 1 + proxy[src] + proxy[tgt]` over all pairs where
  both languages have the proxy. Reported on (a) own coverage and (b) the common
  17-language / 272-pair subset for an apples-to-apples comparison.
- Scripts: `experiments/perplexity_bleu_linear/compute_bits_per_byte.py`,
  `experiments/perplexity_bleu_linear/compare_competence_proxies.py`.

## Uncertainties / caveats

- **Only two distinct measures.** MultiBLiMP accuracy ≡ 1−PER exactly; the
  comparison is really fluency-family (PPL, BPB) vs grammatical accuracy.
- **$R^2 = 0.25$ is modest**, MAE ≈ 4.9 BLEU; grammatical accuracy explains a
  quarter of BLEU variance, not most of it. The target coefficient dominates.
- **Coverage mismatch.** MultiBLiMP covers 17 languages (European-heavy, plus
  heb/hin/tur/fas; no CJK). The head-to-head is on those 17; the broader 24-lang
  claim only exists for the fluency proxies.
- **Mild shared-construct concern.** Both MultiBLiMP accuracy and BLEU reward
  producing well-formed text, so some of the $R^2=0.25$ could be shared-construct
  rather than purely "monolingual competence → translation." MultiBLiMP is
  monolingual acceptability and BLEU is cross-lingual overlap, so they are
  distinct, but this is worth stating.
- **BPB is an exact reconstruction**, not an independent GPU measurement. The
  identity is exact algebra; a direct $\sum\text{NLL}/\sum\text{bytes}$ pass would
  reproduce it to floating-point. (Could run as a belt-and-suspenders check.)
- **English leverage** in the PPL/BPB fits (dropping English raised the original
  24-lang PPL $R^2$ from 0.02→0.11) still applies; it does not change the ranking.
- Aya not analyzed here (Aya is missing FLORES PPL for hin and has its own
  coverage gaps); the same pipeline runs for Aya by changing `MODEL`.

## Recommendation for the paper

Use **MultiBLiMP accuracy (= 1−PER)** as the monolingual-competence proxy for H1,
not perplexity or bits-per-byte. It is the only tested measure with a real,
correctly-signed relationship to BLEU. Frame H1 honestly as a *modest but
real* effect ($R^2 \approx 0.25$, no interaction needed, target competence
dominant), and pair it with the rank-1 BLEU-matrix separability result (88%) as
the structural evidence. Report that perplexity/bits-per-byte fail, as a
methodological finding in its own right.
