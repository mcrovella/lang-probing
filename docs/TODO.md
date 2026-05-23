# TODO -- lang-probing

> Generated 2026-04-22 by exhaustive file-by-file audit of every source file,
> config, script, test, shell script, output, and doc in the repo.
>
> Priority: **P0** = correctness bug or broken entrypoint (fix first),
> **P1** = important before sharing or publishing,
> **P2** = polish / nice-to-have.

> **2026-04-30 sweep:** items 1-25, 26-36, 41 (partial), 42, 46-47,
> 54-68, 69, 73, 75, 78 (partial), 79-92 implemented in an overnight
> autonomous pass. Items requiring judgement (8, 11, 28-rewrite scope,
> 37, 38, 39, 43-50 docs, 70, 71, 72, 74, 76, 77, 86) are intentionally
> deferred. See git log for the diff.

---

## A. Correctness / real bugs

### P0

1. **`.gitignore:34`** -- `__init__.py` is listed as an ignored pattern. All `__init__.py` files (essential Python package markers) will be missing on any fresh clone. **This breaks the entire package on clone.** Action: remove the `__init__.py` line from `.gitignore`.

2. **`src/lang_probing_src/utils.py:17-18`** -- `sys.path.append('/projectnb/mcnet/jbrin/lang-similarity/src')` + `from sae.utils import setup_autoencoder`. Hard dependency on a separate project not in `requirements.txt` or `pyproject.toml`. Importing `lang_probing_src.utils` crashes without that path on disk. Action: guard with `try/except ImportError` or make it a documented optional dep.

3. **`src/lang_probing_src/probes/word_utils.py:6-10`** -- Top-level `import cupy` and `from cuml.preprocessing import StandardScaler`. CUDA-only libraries not in `requirements.txt`. Since `probes/__init__.py` does `from .word_utils import *`, importing the `probes` subpackage crashes without GPU + cuML. Action: wrap in `try/except` or lazy-import inside functions.

4. **`src/lang_probing_src/data/ud.py:33,95`** -- `from .config import UD_BASE_FOLDER` resolves to `data.config` which does not exist. Should be `from ..config import UD_BASE_FOLDER`. Same bug at line 95 in `get_ud_filepath`. Crashes on any call that leaves `ud_base_folder=None`.

5. **`src/lang_probing_src/features/attribution.py:86`** -- `.tolist().save()` calls `.save()` on a plain Python list (no such method). The nnsight proxy `.save()` must be called *before* `.tolist()`. Raises `AttributeError` at runtime.

6. **`src/lang_probing_src/features/attribution.py:543-553`** -- `attribution_patching_loop` references `DataLoader` and `tqdm` which are never imported. Crashes with `NameError`.

7. **`src/lang_probing_src/features/attribution.py:548`** -- `attribution_patching_loop` passes 5 positional args to `attribution_patching` but the signature expects 6 (`probe_submodule` is missing). Arguments are silently shifted.

8. **`src/lang_probing_src/activations/extraction.py:96`** -- Inline TODO acknowledges this is **incorrect**: `extract_sae_activations()` mean-pools activations *before* SAE encoding. Non-linear encoding must happen before pooling. Produces wrong feature activations. Action: encode per-token first, then pool.

9. **`src/lang_probing_src/config.py:71,74`** -- Duplicate key `"zho_simpl"` in `LANG_CODE_TO_NAME`. Second definition (`"Chinese"`) silently overwrites first (`"Chinese (Simplified)"`). `NAME_TO_LANG_CODE["Chinese (Simplified)"]` will `KeyError`.

10. **`src/lang_probing_src/config.py:101`** -- `LAYERS` list includes index `32`, but a 32-layer model has valid indices 0-31. Using this to index `model.model.layers` raises `IndexError`.

11. **`src/lang_probing_src/features/sparse_activations.py:138-154 vs 242-253`** -- `__gt__`, `__lt__`, `nonzero`, `squeeze` defined **twice** in `SparseActivation`. Second definitions silently override first. Semantics differ. Decide which to keep and delete the dead copy.

12. **`src/lang_probing_src/activations/sae.py:94,106`** -- `from_pretrained` and `from_hub` missing `@staticmethod` decorator. Calling on an instance passes `self` as `path`. Action: add `@staticmethod` or change to `@classmethod`.

13. **`experiments/output_features/run.py:132`** -- `os` is never imported. Script crashes immediately with `NameError`. Action: add `import os`.

14. **`experiments/output_features/run.py:152`** -- `SAE_FILENAME` is never defined or imported. Script crashes with `NameError`. Action: import from config or define locally.

15. **`experiments/input_features/run.py:75-76`** -- `LANGUAGES` assigned twice; effective value is only `["Arabic", "Hindi", "Chinese", "Indonesian"]`. Six languages silently dropped. Looks like leftover debugging. Action: remove the second assignment.

16. **`experiments/activations_collection/run.py:37-38`** -- Silent exception swallowing: if `load_sentences_with_tags()` fails, `sentences` is never assigned; line 42 crashes with `NameError`. Action: add `sentences = []` before try, or `return`/`continue` in except.

17. **`experiments/activations_collection/run.py:112`** -- Hardcoded `for layer in [32]:` contradicts `COLLECTION_LAYERS` (max 31). Filter `df[df['layer'] == 32]` returns empty DataFrame. Action: restore `for layer in COLLECTION_LAYERS:`.

18. **`experiments/counterfactual_attribution/run.py:329`** -- `project_root = Path(__file__).resolve().parent.parent` resolves to `experiments/`, not the repo root. `data_path` at line 331 looks for `data/grammatical_pairs.json` under `experiments/`. Action: use `.parent.parent.parent` or `config.PROJECT_DIR`.

19. **`experiments/probes/run.py:245`** -- Default `--values` is `CONCEPTS_VALUES.values()` (a `dict_values` of lists, not flat strings). Value filtering `if value not in args.values` compares strings against lists-of-strings; always fails. Action: flatten the default or change the comparison.

20. **`experiments/probes/visualize.py:11`** -- `INPUT_CSV_FILE` hardcoded to `all_probe_results_tense.csv`. Actual output filenames have timestamps. Crashes with `FileNotFoundError`. Action: accept filename as CLI arg or glob for latest.

21. **`experiments/token_analysis/configs/other_feats.yaml:22`** -- Empty experiment entry `- name: ""` with no other fields. Runner crashes on `exp["concept"]`. Action: remove the empty entry or fill in fields.

22. **`experiments/perplexity_bleu_linear/visualize_correlation.py:62`** -- X-axis label says "Source Perplexity Error Rate" on the **target competence** plot. Should say "Target". Published figure has wrong label.

23. **`experiments/perplexity_bleu_linear/visualize_correlation.py:99-109`** -- After `plt.close()` at line 109, `plt.tricontourf` is called without `plt.figure()`. Draws on closed figure; corrupt output or error.

24. **`experiments/token_analysis/run.py:377-380`** -- Probe path format `outputs/word_probes/{lang}/{concept}/{value}/probe_layer{N}_n{N}.joblib` does not match canonical format `outputs/probes/word_probes/{Lang}_{Concept}_{Value}_l{N}_n{N}.joblib`. Probe filtering is silently disabled. Action: fix path construction to match writer format.

25. **`experiments/perplexity_bleu_linear/run_per.py:389`** -- Code splits `args.languages` by comma, but argparse receives a single `type=str`. README shows space-separated usage which would fail. Action: use `nargs='+'` or document comma-separated format.

### P1

26. **`src/lang_probing_src/features/sparse_activations.py:224-227`** -- `detach()` mutates `self` in-place *and* returns a new object. If `resc` is set but `res` is `None`, crashes with `AttributeError`. Also drops `resc` from returned object.

27. **`src/lang_probing_src/features/sparse_activations.py:129-132`** -- `__neg__` accesses `self.act` and `self.res` without `None` check. Raises `TypeError` if either is `None`. Ignores `resc`.

28. **`src/lang_probing_src/utils.py:64`** -- `setup_model` hardcodes `model.model.layers[16]` regardless of model architecture. For models with <17 layers: `IndexError`. For any other layer: wrong submodule. Action: accept `layer` as a parameter.

29. **`src/lang_probing_src/probes/word_utils.py:143,156`** -- `grid_search.fit()` called **twice**. Entire hyperparameter search runs redundantly. Intermediate logging at lines 148-149 reports stale results.

30. **`src/lang_probing_src/data/ud.py:349-361`** -- `load_flores_sentences_with_tags` has debug code (`print(sentence)`, `if i > 10: break`) and returns `None`. Effectively a stub. Action: finish implementation or `raise NotImplementedError`.

31. **`src/lang_probing_src/activations/sae.py:98`** -- `torch.load(path)` uses deprecated default `weights_only=False` (security risk, PyTorch >= 2.6 warning). Same at `io/effects.py:42`. Action: add `weights_only=True`.

32. **`experiments/input_features/run.py:48`** -- `reduce(sae_acts, "b s d -> d", "mean")` averages batch and sequence dims simultaneously. Sentences with more tokens contribute disproportionately. Should mean-pool per sentence first, then average across sentences.

33. **`experiments/input_features/run.py:63`** -- Hardcodes `Meta-Llama-3.1-8B-Instruct` instead of using `MODEL_ID` from config. If `MODEL_ID` changes, this script uses a different model.

34. **`experiments/perplexity_bleu_linear/run_linear_fit.py:211`** -- Default `--outputs-dir` uses `perplexity_bleu` but actual data is in `perplexity_bleu_linear/bleu_and_ppl`. Won't find input CSVs.

35. **`experiments/output_features/run.py:146`** -- Uses `float16` while all other scripts use `bfloat16`. Inconsistent dtype may cause numerical differences.

---

## B. Reproducibility

### P1

36. **`src/lang_probing_src/data/ud.py:278,343`** -- `np.random.seed(seed)` sets the global RNG, affecting all downstream numpy randomness. Action: use `np.random.default_rng(seed)` for isolation.

37. **`pyproject.toml`** -- No `python_requires` field and no pinned dependency versions. `requirements.txt` uses `>=` floors only. No lock file. Reproducibility depends entirely on the conda environment.

38. **`.gitignore:20-26`** -- `*.csv`, `*.json`, `*.txt`, `*.tsv` globally ignored. New data files added to `data/` will be silently untracked. Only `requirements.txt` has an exception. Action: add exceptions `!data/**/*.json`, `!data/**/*.txt`, etc.

39. **`outputs/probes/`** -- 14 intermediate CSV checkpoints from Nov 22 grid search retained. Unclear which is canonical. Action: clean up or document which file is authoritative.

40. **`experiments/perplexity_bleu_linear/combine_csvs.py:4-5,26`** -- All paths hardcoded to `llama` model only. No CLI arguments, no `aya` support despite README documenting both.

41. **Multiple hardcoded absolute paths** -- `/projectnb/mcnet/jbrin/lang-probing/...` appears in: `config.py:7,24-25`, `io/effects.py:21`, `experiments/input_features/run.py:27,60`, `experiments/input_features/visualize.py:224,231`, `experiments/input_output_overlap/visualize.py:122,126`, `experiments/perplexity_bleu_linear/run_perplexity.py:147-149`, `tests/test_ablate_batch.py:50`, `tests/test_ablate_edge_cases.py:62`. Action: derive from `config.PROJECT_DIR` or `Path(__file__)`.

### P2

42. **`experiments/input_features/run.py:44`** -- No `TRACER_KWARGS` passed to `model.trace()`, unlike every other script. May cause subtle differences depending on nnsight version.

---

## C. Documentation errors

### P1

43. **`docs/reference_paper.tex:121`** -- Abstract says "TODO". Related Work sections are empty stubs (lines 146-149, 173-179). Limitations section empty (410-413). `\section{Related Work}` appears twice (lines 144 and 172).

44. **`docs/reference_paper.tex:68`** -- Author block still has placeholder "First Author / Second Author / email@domain".

45. **`docs/reference_paper.tex:189`** -- Inline `\jake{...}` TODO note referencing an arXiv paper to integrate.

46. **`src/lang_probing_src/activations/extraction.py:104`** -- Docstring lists `layer_num` as argument but function signature does not accept it.

47. **`src/lang_probing_src/data/ud.py:369-374`** -- `ConlluDataset.__init__` docstring says it takes `conllu_path (str)` but actual params are `(language, treebank, split)`. Same for `ConlluDatasetPooled` (lines 395-400).

48. **`data/README.md:27`** -- States `counterfactual_attribution/run.py` skips multi-token pairs, but `attribute_multilingual.py` handles multi-token via last-BPE. README doesn't mention the multilingual pipeline.

49. **`LEDGER.md:79`** -- Says `attribute_multilingual.py` is "pending relocation" but REPORT.md says it was already committed. Contradictory.

50. **`experiments/perplexity_bleu_linear/README.md:66`** -- States "Rank-1 SVD approximation ... has no code yet" but `rank1_approximation.py` already exists and is documented earlier in the same README.

### P2

51. **`docs/reference_paper.tex:165`** -- `\includegraphics[width=1.2\linewidth]` exceeds page margins. Will overflow in PDF.

52. **`outputs/counterfactual_attribution/REPORT.md:183-187,206,231,260,274-283`** -- Image links use relative paths. Verify all referenced PNGs exist at those paths.

53. **`tests/how_to_run_tests.md`** -- Only 4 bare command lines. No context on which tests require GPU, which are broken, or how to run selectively.

---

## D. Broken entrypoints / missing scaffolding

### P0 -- Stale `scripts/` references (post-restructure)

All shell scripts and test files below reference `scripts/*.py` paths that no longer exist after the `scripts/` -> `experiments/` restructuring. Each one is a broken entrypoint.

54. **`experiments/ablation/run/run_ablate.sh:43`** -- Calls `python scripts/ablate.py`. Should be `experiments/ablation/run.py`.

55. **`experiments/ablation/run/run_ablate.sh:41`** -- Experiment names `multi_random_src`/`multi_random_tgt` don't match `EXP_CONFIGS` keys (`multi_input_random`/`multi_output_random`).

56. **`experiments/counterfactual_attribution/run/run_counterfactual_attribution.sh:48,57`** -- References `scripts/counterfactual_attribution.py` and `scripts/analyze_counterfactual_results.py`.

57. **`experiments/probes/run/run_probes.sh:40`** -- Calls `python scripts/word_probes.py`. Should be `experiments/probes/run.py`.

58. **`experiments/token_analysis/run/run_token_analysis.sh:41,51,59`** -- References `configs/token_analysis_example.yaml`, `scripts/analyze_tokens.py`, `scripts/visualize_tokens.py`.

59. **`experiments/token_analysis/run/run_token_analysis_mood.sh:29,39,49`** -- Same stale `scripts/` and `configs/` paths.

60. **`experiments/token_analysis/run/run_token_analysis_multilang.sh:29,39,49`** -- Same.

61. **`experiments/token_analysis/run/run_token_analysis_other_feats.sh:29,39,49`** -- Same.

62. **`experiments/output_features/run/attribution_flores.sh:1`** -- Contains only `python scripts/attribution_flores.py`. Not a valid batch script (no shebang, no SGE directives, no module loads).

63. **`tests/test_ablate_integration.py:21`** -- `from scripts.ablate import ...` -- `ModuleNotFoundError`.

64. **`tests/test_perplexity_comparison.py:22`** -- `from perplexity_comparison import ...` -- references removed `scripts/perplexity_comparison.py`.

65. **`tests/test_steering_vectors.py:25`** -- `from scripts.generate_steering_vectors import main` -- `scripts/` does not exist.

66. **`tests/test_visualize_tokens.py:18`** -- `from scripts.visualize_tokens import ...` -- `scripts/` does not exist.

67. **`tests/verify_setup.py:46,104,154`** -- `from src.config import ...` and `from src.data import ...` -- package was renamed to `lang_probing_src`.

68. **`tests/check_structure.py:29-60`** -- All paths reference pre-restructure layout (`src/__init__.py`, `scripts/verify_setup.py`, etc.). Every check fails.

### P1

69. **`requirements.txt`** -- Missing `einops` (used in `extraction.py:8`, `ablate.py:18`), `pandas` (used in `utils.py:11`), `datasets` (used in `ud.py:12`). Package fails to install cleanly.

70. **`pyproject.toml`** -- No `dependencies` field. `pip install .` installs the package with zero dependencies.

71. **`src/lang_probing_src/eval/__init__.py`** -- Empty file. The `eval` subpackage is dead scaffolding. Action: remove or populate.

72. **`src/lang_probing_src/viz/__init__.py`** -- Empty file. The `viz` subpackage is dead scaffolding. Action: remove or populate.

73. **`experiments/ablation/run/run_ablate.sh:6`** -- `gpu_memory=32` missing `G` suffix (should be `32G`). Same in `run_counterfactual_attribution.sh:6`.

74. **`experiments/monolingual_ft/evaluate.py:31`** -- Raises `NotImplementedError`. File is a scaffold with no implementation. If someone follows the README they'll hit this. Action: implement or mark README as not-yet-runnable.

---

## E. Code hygiene

### P1

75. **`src/lang_probing_src/interventions/ablate.py:1-10`** -- Module-level docstring is raw LaTeX (`\item`, `\begin{itemize}`). Copy-pasted from paper draft; not valid documentation.

76. **`tests/test_ablate_statistics.py`** -- All "statistical" tests generate synthetic data with `np.random.normal` and test its properties. These validate numpy's RNG, not the ablation pipeline. They always pass regardless of codebase state.

77. **`tests/test_ablate_batch.py:133-134`** -- `test_random_baseline_less_effective` uses `self.model.model.layers[16]` while other tests use `self.submodule` from `setup_model()`. Inconsistent test setup.

### P2

78. **Debug `print()` statements left in production code:**
    - `features/attribution.py:192,218,224`
    - `activations/extraction.py:415`
    - `interventions/ablate.py:327,330,333`
    - `data/ud.py:358`
    - `io/effects.py:38`
    Action: replace with `logging.debug()` or remove.

79. **`src/lang_probing_src/interventions/ablate.py:238-295`** -- ~60 lines of commented-out code. Action: delete (it's in git history).

80. **`src/lang_probing_src/interventions/ablate.py:327-334`** -- `ablate_bleu` ends with debug prints and no `return`. Function is dead code.

81. **`src/lang_probing_src/features/attribution.py:430`** -- `metric_kwargs=dict()` mutable default argument. Should be `metric_kwargs=None` with `if metric_kwargs is None: metric_kwargs = {}`.

82. **`src/lang_probing_src/features/attribution.py:448-451`** -- `is_tuple` detection unconditionally sets `True`, ignoring actual output type. Dead code.

83. **`src/lang_probing_src/__init__.py:3`** -- Docstring in Spanish ("Sistema para entrenar probes lineales..."). Mixed-language docs throughout codebase.

84. **`src/lang_probing_src/data/ud.py:455-456,363`** -- `from datasets import load_dataset` and `from torch.utils.data import Dataset` imported twice in same file. Action: deduplicate.

85. **`src/lang_probing_src/data/ud.py:482`** -- Module-level `collate_fn` shadows the one in `data/dataset.py`. Both exported via `__init__.py`. Import-order-dependent collision.

86. **Backward-compat shim files** -- Six top-level shim modules exist solely for `from .X import *` re-exports: `ablate.py`, `autoencoder.py`, `probe.py`, `sentence_dataset_class.py`, `utils_input_output.py`, `word_probing_utils.py`. Action: grep for old import paths; if unused, delete.

87. **`experiments/ablation/run.py:250,262`** -- `setup_model()` called twice with identical args. First return value overwritten. Wastes GPU memory.

88. **`experiments/ablation/ablate_validate.py:29`** -- `SAE_DIM` imported but mask hardcoded as `torch.ones(32768, ...)`. If `SAE_DIM` changes, mask size will be wrong.

89. **`experiments/output_features/run.py:93`** -- Variable named `dict` shadows the built-in.

90. **`experiments/perplexity_bleu_linear/run_perplexity.py:148`** -- Model name detection `"llama" if "llama" in args.model_id else "aya"` is fragile. Non-llama/non-aya model silently writes to `aya` path.

91. **`experiments/ablation/run.py:440`** -- 13 consecutive `\n` in a print statement. Debug noise.

92. **`experiments/ablation/run.py:1`** -- Four opening quotes (`""""`) instead of three. Produces empty string expression before docstring.

---

## Summary

| Priority | Count |
|----------|-------|
| P0       | ~40   |
| P1       | ~30   |
| P2       | ~22   |

### Top 7 highest-impact P0 issues (fix these first)

1. **`.gitignore` ignoring `__init__.py`** -- breaks package on any fresh clone
2. **`utils.py` hard dep on external `lang-similarity`** -- top-level import crash
3. **`probes/word_utils.py` unconditional `import cupy/cuml`** -- crashes `probes` subpackage without GPU
4. **15 shell scripts + test files reference non-existent `scripts/` paths** -- all broken post-restructure
5. **`output_features/run.py` missing `import os` + undefined `SAE_FILENAME`** -- crashes immediately
6. **`config.py` layer index 32 out of bounds** for 32-layer model
7. **`ud.py` broken relative import** (`from .config` should be `from ..config`)
