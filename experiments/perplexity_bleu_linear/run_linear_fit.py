import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from lang_probing_src.config import OUTPUTS_DIR, MODEL_TO_ID


def load_perplexities(csv_path: Path) -> Dict[str, float]:
    """
    Load per-language perplexities from a CSV file with columns:
    Language,Perplexity

    Rows whose Perplexity cannot be parsed as a positive float are skipped.
    """
    lang_to_ppl: Dict[str, float] = {}

    if not csv_path.is_file():
        raise FileNotFoundError(f"Perplexity CSV not found: {csv_path}")

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "Language" not in reader.fieldnames or "Perplexity" not in reader.fieldnames:
            raise ValueError(
                f"Expected columns 'Language' and 'Perplexity' in {csv_path}, "
                f"got {reader.fieldnames!r}"
            )

        for row in reader:
            lang = (row.get("Language") or "").strip()
            val_str = (row.get("Perplexity") or "").strip()
            if not lang or not val_str:
                continue
            try:
                val = float(val_str)
            except (TypeError, ValueError):
                # e.g. "Error" rows
                continue
            if not math.isfinite(val) or val <= 0.0:
                continue
            lang_to_ppl[lang] = val

    if not lang_to_ppl:
        raise ValueError(f"No valid perplexity entries found in {csv_path}")

    return lang_to_ppl


def load_multiblimp_margins(csv_path: Path) -> Dict[str, float]:
    """Load per-language mean `margin_total` from margins_by_lang.csv."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Margin CSV not found: {csv_path}")
    out: Dict[str, float] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = (row.get("lang") or "").strip()
            val = row.get("margin_total")
            if not lang or val is None:
                continue
            try:
                out[lang] = float(val)
            except ValueError:
                continue
    if not out:
        raise ValueError(f"No valid margins found in {csv_path}")
    return out


def build_regression_dataset(
    bleu_csv_path: Path,
    lang_to_ppl: Dict[str, float],
    feature_transform: str,
    include_interaction: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Build design matrix X, target vector y and list of (src, tgt) pairs
    from BLEU and perplexity CSVs.
    """
    if not bleu_csv_path.is_file():
        raise FileNotFoundError(f"BLEU CSV not found: {bleu_csv_path}")

    X_rows: List[List[float]] = []
    y_values: List[float] = []
    pairs: List[Tuple[str, str]] = []

    with bleu_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"src", "tgt", "bleu"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Expected columns {required_cols} in {bleu_csv_path}, "
                f"got {reader.fieldnames!r}"
            )

        for row in reader:
            src = (row.get("src") or "").strip()
            tgt = (row.get("tgt") or "").strip()
            bleu_str = (row.get("bleu") or "").strip()
            if not src or not tgt or not bleu_str:
                continue
            try:
                bleu_val = float(bleu_str)
            except (TypeError, ValueError):
                continue

            Pi = lang_to_ppl.get(src)
            Pj = lang_to_ppl.get(tgt)
            if Pi is None or Pj is None:
                # Skip pairs where we do not have both perplexities
                continue

            if feature_transform == "raw":
                x_src, x_tgt = Pi, Pj
            elif feature_transform == "log":
                if Pi <= 0.0 or Pj <= 0.0:
                    continue
                x_src, x_tgt = math.log(Pi), math.log(Pj)
            else:
                raise ValueError(f"Unknown feature_transform: {feature_transform}")

            row_vec: List[float] = [1.0, x_src, x_tgt]
            if include_interaction:
                row_vec.append(x_src * x_tgt)

            X_rows.append(row_vec)
            y_values.append(bleu_val)
            pairs.append((src, tgt))

    if not X_rows:
        raise ValueError(
            "No valid (src, tgt) pairs found after filtering by perplexity and BLEU."
        )

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_values, dtype=float)
    return X, y, pairs


def get_feature_names(feature_transform: str, include_interaction: bool) -> List[str]:
    if feature_transform == "raw":
        base = "P"
    elif feature_transform == "log":
        base = "logP"
    else:
        raise ValueError(f"Unknown feature_transform: {feature_transform}")

    names: List[str] = ["intercept", f"{base}_src", f"{base}_tgt"]
    if include_interaction:
        names.append(f"{base}_src*{base}_tgt")
    return names


def fit_linear_model(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit ordinary least squares using numpy.linalg.lstsq.

    Returns:
        coeffs: (n_features,) vector of coefficients.
        y_hat: (n_samples,) vector of predictions.
    """
    coeffs, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    return coeffs, y_hat


def compute_metrics(y: np.ndarray, y_hat: np.ndarray) -> Tuple[float, float]:
    """
    Compute MSE and R^2 for the regression.
    """
    if y.shape != y_hat.shape:
        raise ValueError("y and y_hat must have the same shape")

    residuals = y - y_hat
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    mse = ss_res / max(len(y), 1)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return mse, r2


def save_coefficients(
    coeff_names: Sequence[str],
    coeffs: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save regression coefficients to a CSV file with columns: term,coef
    """
    if len(coeff_names) != len(coeffs):
        raise ValueError(
            f"Coefficient name/values length mismatch: "
            f"{len(coeff_names)} vs {len(coeffs)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "coef"])
        for name, val in zip(coeff_names, coeffs):
            writer.writerow([name, float(val)])


def save_predictions(
    pairs: Sequence[Tuple[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save per-pair BLEU true/predicted values and errors.
    """
    if len(pairs) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError(
            "Lengths of pairs, y_true and y_pred must match "
            f"({len(pairs)}, {len(y_true)}, {len(y_pred)})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "tgt", "bleu_true", "bleu_pred", "error"])
        for (src, tgt), bt, bp in zip(pairs, y_true, y_pred):
            err = float(bp - bt)
            writer.writerow([src, tgt, float(bt), float(bp), err])


def parse_args() -> argparse.Namespace:
    default_outputs_dir = Path(OUTPUTS_DIR) / "perplexity_bleu_linear"

    parser = argparse.ArgumentParser(
        description=(
            "Fit a linear model predicting BLEU scores for language pairs "
            "from per-language perplexities."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"Model key, one of: {', '.join(sorted(MODEL_TO_ID.keys()))}",
    )
    parser.add_argument(
        "--feature-transform",
        type=str,
        choices=("raw", "log"),
        default="raw",
        help="Feature transformation to apply to perplexities (default: raw).",
    )
    parser.add_argument(
        "--proxy",
        choices=("perplexity", "multiblimp_margin"),
        default="perplexity",
        help="Per-language feature to use in BLEU ~ src + tgt regression.",
    )
    parser.add_argument(
        "--margin-csv",
        type=str,
        default=None,
        help="Optional margins_by_lang.csv path for --proxy multiblimp_margin.",
    )
    interaction_group = parser.add_mutually_exclusive_group()
    interaction_group.add_argument(
        "--include-interaction",
        dest="include_interaction",
        action="store_true",
        help="Include joint term (Pi*Pj or logPi*logPj) in the regression.",
    )
    interaction_group.add_argument(
        "--no-include-interaction",
        dest="include_interaction",
        action="store_false",
        help="Exclude the joint term from the regression.",
    )
    parser.set_defaults(include_interaction=True)
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=str(default_outputs_dir),
        help=(
            "Base directory for perplexity/BLEU CSVs and linear model outputs. "
            "Defaults to '<OUTPUTS_DIR>/perplexity_bleu'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model not in MODEL_TO_ID:
        raise ValueError(
            f"Unknown model '{args.model}'. "
            f"Available keys: {', '.join(sorted(MODEL_TO_ID.keys()))}"
        )

    outputs_dir = Path(args.outputs_dir)

    perplexity_csv = outputs_dir / f"perplexity_results_{args.model}.csv"
    bleu_csv = outputs_dir / f"bleu_results_{args.model}.csv"

    if args.proxy == "multiblimp_margin":
        margin_csv = Path(args.margin_csv) if args.margin_csv else (
            outputs_dir.parent / "multiblimp_margins" / args.model / "margins_by_lang.csv"
        )
        lang_to_ppl = load_multiblimp_margins(margin_csv)
        feature_transform = "raw"
    else:
        lang_to_ppl = load_perplexities(perplexity_csv)
        feature_transform = args.feature_transform
    X, y, pairs = build_regression_dataset(
        bleu_csv,
        lang_to_ppl=lang_to_ppl,
        feature_transform=feature_transform,
        include_interaction=args.include_interaction,
    )

    coeffs, y_hat = fit_linear_model(X, y)
    mse, r2 = compute_metrics(y, y_hat)

    coeff_names = get_feature_names(
        feature_transform=feature_transform,
        include_interaction=args.include_interaction,
    )

    joint_flag = "joint" if args.include_interaction else "nojoint"
    feature_flag = f"{args.proxy}_{feature_transform}"

    results_dir = outputs_dir / "linear_models"
    coeffs_path = results_dir / f"linear_coeffs_{args.model}_{feature_flag}_{joint_flag}.csv"
    preds_path = results_dir / f"linear_predictions_{args.model}_{feature_flag}_{joint_flag}.csv"

    save_coefficients(coeff_names, coeffs, coeffs_path)
    save_predictions(pairs, y_true=y, y_pred=y_hat, output_path=preds_path)

    print(f"Fitted linear model for {args.model!r}")
    print(f"  samples: {len(y)}")
    print(f"  features: {len(coeffs)}")
    print(f"  MSE: {mse:.4f}")
    print(f"  R^2: {r2:.4f}")
    print("  Coefficients:")
    for name, val in zip(coeff_names, coeffs):
        print(f"    {name:16s}: {float(val): .6f}")

    print(f"\nSaved coefficients to: {coeffs_path}")
    print(f"Saved per-pair predictions to: {preds_path}")


if __name__ == "__main__":
    main()
