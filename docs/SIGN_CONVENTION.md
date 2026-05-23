# GCM Sign Convention

Date: 2026-05-22

The repository convention is positive IE = the component favors the original,
correct, or grammatical completion.

Math:

`M = logp(r_orig | p_orig) - logp(r_cf | p_orig)`

Saved migrated GCM direction directories are marked with:

`"sign_convention": "orig_minus_cf"`

Audited consumers:

- `experiments/gcm_translation/gcm_core.py`
- `experiments/gcm_translation/run.py`
- `experiments/gcm_translation/analyze.py`
- `experiments/gcm_translation/bootstrap.py`
- `experiments/gcm_translation/meeting_plots.py`
- `experiments/gcm_translation/visualize.py`
- `experiments/gcm_translation/validate_faithfulness.py`
- `experiments/head_ablation_multiblimp/heads.py`
- `experiments/head_ablation_multiblimp/run.py`
- `experiments/head_ablation_multiblimp/analyze.py`
- `experiments/head_ablation_multiblimp/visualize.py`
