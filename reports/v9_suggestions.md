# v9 Suggestions

Date: 2026-05-22

- Sign convention: update `eq:metric` and nearby prose so positive IE means
  `logp(r_orig) - logp(r_cf)`, i.e. favors the correct/original completion.
- `reports/v9.tex:96`: reconsider "noisy channel" wording; the current setup is
  a competence-proxy regression, not a channel model.
- `reports/v9.tex:138` and `reports/v9.tex:141`: fix the broken `\jb{}` text in
  `tab:proxy`.
- Define `\aaron` or remove the macro before camera-ready compilation.
- Limitations: state that the core claims are currently checked on two model
  families, with Aya reruns still pending for the newest sign-fixed experiments.
- Add the MultiBLiMP logprob-margin proxy result once the forward-only jobs
  write `margins_by_lang.csv`.
- Add the BLEU head-ablation figure after the smoke/full SCC sweep.
- Replace the imputed rank-1 number with the masked observed-offdiagonal result
  once `rank1_approximation.py` has been run without `--legacy_impute`.
