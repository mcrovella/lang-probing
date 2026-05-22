"""Compile shareable BLEU, perplexity, and accuracy metric tables.

Creates two collaborator-facing CSVs:

1. language_metrics.csv
   One row per (model, language), with monolingual FLORES perplexity,
   Multi-BLiMP accuracy, and aggregate BLEU as source/as target.

2. pair_metrics.csv
   One row per (model, source language, target language), with pairwise BLEU
   enriched by source/target monolingual perplexity and accuracy.

Missing language-level metrics are left as NaN/blank in the CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from lang_probing_src.config import LANG_CODE_TO_NAME, OUTPUTS_DIR


MODEL_TO_ACCURACY_FILE_STEM = {
    "llama": "error_rates_by_language_meta-llama_Meta-Llama-3_1-8B_jumelet_multiblimp.json",
    "aya": "error_rates_by_language_CohereLabs_aya-23-8B_jumelet_multiblimp.json",
}


def load_bleu(data_dir: Path, model: str) -> pd.DataFrame:
    path = data_dir / f"bleu_results_{model}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing BLEU CSV: {path}")
    df = pd.read_csv(path)
    expected = {"src", "tgt", "bleu"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.loc[:, ["src", "tgt", "bleu"]].copy()
    out.insert(0, "model", model)
    return out


def load_flores_perplexity(data_dir: Path, model: str) -> Dict[str, float]:
    path = data_dir / f"perplexity_results_{model}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    expected = {"Language", "Perplexity"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["Perplexity"] = pd.to_numeric(df["Perplexity"], errors="coerce")
    df = df.dropna(subset=["Language", "Perplexity"])
    return dict(zip(df["Language"].astype(str), df["Perplexity"].astype(float)))


def load_multiblimp_accuracy(perplexity_error_dir: Path, model: str) -> Dict[str, float]:
    filename = MODEL_TO_ACCURACY_FILE_STEM.get(model)
    if filename is None:
        return {}
    path = perplexity_error_dir / filename
    if not path.exists():
        return {}
    with path.open() as f:
        error_rates = json.load(f)
    return {
        str(lang): 1.0 - float(error_rate)
        for lang, error_rate in error_rates.items()
        if error_rate is not None and np.isfinite(float(error_rate))
    }


def build_language_metrics(
    model: str,
    bleu_df: pd.DataFrame,
    flores_perplexity: Dict[str, float],
    multiblimp_accuracy: Dict[str, float],
) -> pd.DataFrame:
    as_source = (
        bleu_df.groupby("src", as_index=False)
        .agg(avg_bleu_as_source=("bleu", "mean"), n_bleu_pairs_as_source=("bleu", "size"))
        .rename(columns={"src": "lang"})
    )
    as_target = (
        bleu_df.groupby("tgt", as_index=False)
        .agg(avg_bleu_as_target=("bleu", "mean"), n_bleu_pairs_as_target=("bleu", "size"))
        .rename(columns={"tgt": "lang"})
    )

    languages = sorted(
        set(bleu_df["src"])
        | set(bleu_df["tgt"])
        | set(flores_perplexity)
        | set(multiblimp_accuracy)
    )
    metrics = pd.DataFrame({"model": model, "lang": languages})
    metrics["lang_name"] = metrics["lang"].map(LANG_CODE_TO_NAME)
    metrics["flores_perplexity"] = metrics["lang"].map(flores_perplexity)
    metrics["multiblimp_accuracy"] = metrics["lang"].map(multiblimp_accuracy)
    metrics = metrics.merge(as_source, on="lang", how="left")
    metrics = metrics.merge(as_target, on="lang", how="left")
    return metrics


def build_pair_metrics(
    bleu_df: pd.DataFrame,
    flores_perplexity: Dict[str, float],
    multiblimp_accuracy: Dict[str, float],
) -> pd.DataFrame:
    pairs = bleu_df.copy()
    pairs["src_lang_name"] = pairs["src"].map(LANG_CODE_TO_NAME)
    pairs["tgt_lang_name"] = pairs["tgt"].map(LANG_CODE_TO_NAME)
    pairs["src_flores_perplexity"] = pairs["src"].map(flores_perplexity)
    pairs["tgt_flores_perplexity"] = pairs["tgt"].map(flores_perplexity)
    pairs["src_multiblimp_accuracy"] = pairs["src"].map(multiblimp_accuracy)
    pairs["tgt_multiblimp_accuracy"] = pairs["tgt"].map(multiblimp_accuracy)
    return pairs[
        [
            "model",
            "src",
            "tgt",
            "src_lang_name",
            "tgt_lang_name",
            "bleu",
            "src_flores_perplexity",
            "tgt_flores_perplexity",
            "src_multiblimp_accuracy",
            "tgt_multiblimp_accuracy",
        ]
    ]


def build_coverage_summary(language_metrics: pd.DataFrame, pair_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(pair_metrics["model"].unique()):
        lang_model = language_metrics[language_metrics["model"] == model]
        pair_model = pair_metrics[pair_metrics["model"] == model]
        rows.append(
            {
                "model": model,
                "n_languages": int(lang_model["lang"].nunique()),
                "n_languages_with_flores_perplexity": int(lang_model["flores_perplexity"].notna().sum()),
                "n_languages_with_multiblimp_accuracy": int(lang_model["multiblimp_accuracy"].notna().sum()),
                "n_bleu_pairs": int(len(pair_model)),
                "n_pairs_with_source_and_target_flores_perplexity": int(
                    pair_model[["src_flores_perplexity", "tgt_flores_perplexity"]].notna().all(axis=1).sum()
                ),
                "n_pairs_with_source_and_target_multiblimp_accuracy": int(
                    pair_model[["src_multiblimp_accuracy", "tgt_multiblimp_accuracy"]].notna().all(axis=1).sum()
                ),
                "n_pairs_with_all_metrics": int(
                    pair_model[
                        [
                            "bleu",
                            "src_flores_perplexity",
                            "tgt_flores_perplexity",
                            "src_multiblimp_accuracy",
                            "tgt_multiblimp_accuracy",
                        ]
                    ]
                    .notna()
                    .all(axis=1)
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def compile_metrics(
    models: Iterable[str],
    data_dir: Path,
    perplexity_error_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    language_frames = []
    pair_frames = []

    for model in models:
        bleu_df = load_bleu(data_dir, model)
        flores_perplexity = load_flores_perplexity(data_dir, model)
        multiblimp_accuracy = load_multiblimp_accuracy(perplexity_error_dir, model)

        language_frames.append(
            build_language_metrics(model, bleu_df, flores_perplexity, multiblimp_accuracy)
        )
        pair_frames.append(
            build_pair_metrics(bleu_df, flores_perplexity, multiblimp_accuracy)
        )

    language_metrics = pd.concat(language_frames, ignore_index=True)
    pair_metrics = pd.concat(pair_frames, ignore_index=True)
    coverage_summary = build_coverage_summary(language_metrics, pair_metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    language_metrics.to_csv(output_dir / "language_metrics.csv", index=False)
    pair_metrics.to_csv(output_dir / "pair_metrics.csv", index=False)
    coverage_summary.to_csv(output_dir / "metric_coverage_summary.csv", index=False)

    return language_metrics, pair_metrics, coverage_summary


def parse_args() -> argparse.Namespace:
    base_dir = Path(OUTPUTS_DIR) / "perplexity_bleu_linear"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["llama", "aya"])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=base_dir / "bleu_and_ppl",
        help="Directory containing BLEU and FLORES perplexity CSVs.",
    )
    parser.add_argument(
        "--perplexity-error-dir",
        type=Path,
        default=base_dir / "per",
        help="Directory containing Multi-BLiMP perplexity error-rate JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "shareable_metrics",
        help="Directory where compiled collaborator-facing CSVs are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    language_metrics, pair_metrics, coverage_summary = compile_metrics(
        models=args.models,
        data_dir=args.data_dir,
        perplexity_error_dir=args.perplexity_error_dir,
        output_dir=args.output_dir,
    )

    print(f"Wrote {len(language_metrics)} language rows")
    print(f"Wrote {len(pair_metrics)} pair rows")
    print(f"Wrote coverage summary for {len(coverage_summary)} models")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
