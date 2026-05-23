"""
Compare per-sentence perplexity between correct vs incorrect sentences.
Computes error rate (proportion of rows where model prefers the wrong sentence)
and saves distributions for analysis. Supports multi-language datasets where
each language is a dataset config (e.g. FLORES-style).
"""

import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple, Union

# lang_probing_src is installed via pyproject.toml; no path hack needed when
# the package is installed. The inserts below are a safety net for running
# the script from a non-installed checkout.
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lang_probing_src.config import LANG_CODE_TO_NAME


COL_CORRECT = "sen"
COL_WRONG = "wrong_sen"

BLEU_LANGS_24 = [
    "ara", "rus", "fra", "zho_simpl", "ita", "por", "ind", "heb", "pol",
    "deu", "ell", "tur", "vie", "ukr", "spa", "fas", "kor", "hin", "jpn",
    "eng", "zho_trad", "ces", "nld", "ron",
]

# jumelet/multiblimp uses bare ISO-639-3 config names (verified against
# get_dataset_config_names: 101 configs). 18 of the 24 BLEU langs are present;
# Chinese (simpl/trad), Indonesian, Vietnamese, Korean, Japanese have no
# MultiBLiMP config and are reported in the `absent` list by the coverage check.
BLEU_CODE_TO_MULTIBLIMP_CONFIG = {
    "ara": "arb",   # Standard Arabic (config is 'arb', not 'ara')
    "rus": "rus",
    "fra": "fra",
    "ita": "ita",
    "por": "por",
    "heb": "heb",
    "pol": "pol",
    "deu": "deu",
    "ell": "ell",
    "tur": "tur",
    "ukr": "ukr",
    "spa": "spa",
    "fas": "fas",   # Persian (config is 'fas')
    "hin": "hin",
    "eng": "eng",
    "ces": "ces",
    "nld": "nld",
    "ron": "ron",
    # No MultiBLiMP config -> excluded by the `cfg in configs` filter, surfaced
    # in the absent list:
    "zho_simpl": "zho_Hans",
    "zho_trad": "zho_Hant",
    "ind": "ind",
    "vie": "vie",
    "kor": "kor",
    "jpn": "jpn",
}


def _slug(s: str) -> str:
    """Sanitize string for use in filenames (e.g. model_id or dataset path)."""
    return re.sub(r"[^\w\-]", "_", s)[:80]


def perplexity_per_sentence(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list,
    device: str,
    batch_size: int = 64,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Compute per-sentence perplexity for a list of texts.

    Uses right-padding, shift logits/labels, and masks padding tokens.
    Returns one perplexity per input text (exp(avg NLL over non-pad tokens)).

    Args:
        model: Causal LM already on device.
        tokenizer: Tokenizer with padding_side and pad_token set.
        texts: List of input strings.
        device: Device string.
        batch_size: Batch size for forward passes.
        max_length: Max sequence length (default from model config).

    Returns:
        Array of shape (len(texts),) with perplexity per sentence. NaN for empty sequences.
    """
    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 2048)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    all_ppl = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        shift_attention_mask = inputs.attention_mask[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        raw_loss = loss_fct(flat_logits, flat_labels)
        raw_loss = raw_loss.view(shift_labels.size())
        masked_loss = raw_loss * shift_attention_mask

        # Per-sequence: sum NLL and token count, then ppl = exp(avg_nll)
        nll_per_seq = masked_loss.sum(dim=1)
        n_tokens_per_seq = shift_attention_mask.sum(dim=1).float().clamp(min=1e-8)
        avg_nll = (nll_per_seq / n_tokens_per_seq).cpu()
        batch_ppl = np.array([math.exp(x.item()) if n_tokens_per_seq[j].item() >= 1 else np.nan for j, x in enumerate(avg_nll)])
        all_ppl.append(batch_ppl)

        del outputs, logits, shift_logits, inputs, shift_labels, shift_attention_mask, flat_logits, flat_labels, raw_loss, masked_loss, nll_per_seq, n_tokens_per_seq, avg_nll, batch_ppl
        torch.cuda.empty_cache()

    return np.concatenate(all_ppl, axis=0)


def sequence_logprobs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list,
    device: str,
    batch_size: int = 64,
    max_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return summed next-token logprobs and token counts for each text."""
    if max_length is None:
        max_length = getattr(model.config, "max_position_embeddings", 2048)

    all_logp: list[np.ndarray] = []
    all_n_tokens: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()
        shift_attention_mask = inputs.attention_mask[..., 1:].contiguous()
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * shift_attention_mask

        logp_total = token_logp.sum(dim=1).cpu().numpy()
        n_tokens = shift_attention_mask.sum(dim=1).cpu().numpy()
        all_logp.append(logp_total)
        all_n_tokens.append(n_tokens)

        del inputs, logits, shift_logits, shift_labels, shift_attention_mask, log_probs, token_logp
        torch.cuda.empty_cache()

    return np.concatenate(all_logp, axis=0), np.concatenate(all_n_tokens, axis=0)


def aggregate_margins(per_item_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-item MultiBLiMP margins by language and phenomenon."""
    agg_spec = {
        "margin_total": "mean",
        "margin_pertoken": "mean",
        "correct": "mean",
        "logp_correct": "count",
    }
    by_lang = (
        per_item_df.groupby("lang", dropna=False)
        .agg(agg_spec)
        .rename(columns={"logp_correct": "n", "correct": "accuracy"})
        .reset_index()
    )
    by_lang_phen = (
        per_item_df.groupby(["lang", "phenomenon"], dropna=False)
        .agg(agg_spec)
        .rename(columns={"logp_correct": "n", "correct": "accuracy"})
        .reset_index()
    )
    return by_lang, by_lang_phen


def available_multiblimp_language_map(
    dataset_path: str = "jumelet/multiblimp",
    bleu_codes: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Map BLEU language codes to available MultiBLiMP configs."""
    bleu_codes = bleu_codes or BLEU_LANGS_24
    configs = set(get_dataset_config_names(dataset_path))
    language_map = {
        code: cfg
        for code, cfg in BLEU_CODE_TO_MULTIBLIMP_CONFIG.items()
        if code in bleu_codes and cfg in configs
    }
    absent = [code for code in bleu_codes if code not in language_map]
    return language_map, absent


def _first_present(row: dict, names: list[str], default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def run_margins_multilang(
    dataset_path: str,
    model_id: str,
    language_map: Dict[str, str],
    split: str = "train",
    batch_size: int = 64,
    device: str = "cuda",
    max_length: Optional[int] = None,
    output_dir: str = "outputs/perplexity_bleu_linear/multiblimp_margins",
) -> pd.DataFrame:
    """Compute per-item correct-minus-wrong logprob margins across languages."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)
    model.eval()

    rows = []
    for lang, config in tqdm(language_map.items(), desc="Margin languages"):
        ds = load_dataset(dataset_path, name=config, split=split)
        records = list(ds)
        texts_correct = [r[COL_CORRECT] for r in records]
        texts_wrong = [r[COL_WRONG] for r in records]
        logp_correct, n_tok_correct = sequence_logprobs(
            model, tokenizer, texts_correct, device, batch_size, max_length
        )
        logp_wrong, n_tok_wrong = sequence_logprobs(
            model, tokenizer, texts_wrong, device, batch_size, max_length
        )
        denom = np.maximum(n_tok_correct, 1) + np.maximum(n_tok_wrong, 1)
        margin_total = logp_correct - logp_wrong
        margin_pertoken = (logp_correct / np.maximum(n_tok_correct, 1)) - (
            logp_wrong / np.maximum(n_tok_wrong, 1)
        )
        for i, r in enumerate(records):
            rows.append({
                "lang": lang,
                "config": config,
                "row_id": i,
                "phenomenon": _first_present(r, ["phenomenon", "linguistic_phenomenon", "category"]),
                "concept": _first_present(r, ["concept", "field", "subphenomenon"]),
                "logp_correct": float(logp_correct[i]),
                "logp_wrong": float(logp_wrong[i]),
                "n_tokens_correct": int(n_tok_correct[i]),
                "n_tokens_wrong": int(n_tok_wrong[i]),
                "margin_total": float(margin_total[i]),
                "margin_pertoken": float(margin_pertoken[i]),
                "correct": bool(logp_correct[i] > logp_wrong[i]),
            })

    per_item = pd.DataFrame(rows)
    model_dir = os.path.join(output_dir, _slug(model_id))
    os.makedirs(model_dir, exist_ok=True)
    per_item_path = os.path.join(model_dir, "per_item.parquet")
    try:
        per_item.to_parquet(per_item_path, index=False)
    except Exception:
        per_item_path = os.path.join(model_dir, "per_item.csv")
        per_item.to_csv(per_item_path, index=False)
    by_lang, by_lang_phen = aggregate_margins(per_item)
    by_lang.to_csv(os.path.join(model_dir, "margins_by_lang.csv"), index=False)
    by_lang_phen.to_csv(os.path.join(model_dir, "margins_by_lang_phenomenon.csv"), index=False)
    with open(os.path.join(model_dir, "language_map.json"), "w") as f:
        json.dump(language_map, f, indent=2)
    return per_item


def compute_error_rate(
    ppl_correct: np.ndarray, ppl_wrong: np.ndarray
) -> Tuple[float, int, int]:
    """
    Error rate = proportion of rows where model assigns lower perplexity to the wrong sentence.

    Args:
        ppl_correct: Perplexity of correct sentence per row.
        ppl_wrong: Perplexity of wrong sentence per row.

    Returns:
        (error_rate, n_errors, n_total). Rows with NaN in either array are excluded from n_total.
    """
    ppl_correct = np.asarray(ppl_correct, dtype=float)
    ppl_wrong = np.asarray(ppl_wrong, dtype=float)
    valid = np.isfinite(ppl_correct) & np.isfinite(ppl_wrong)
    if valid.sum() == 0:
        return float("nan"), 0, 0
    n_total = int(valid.sum())
    n_errors = int((ppl_wrong[valid] < ppl_correct[valid]).sum())
    return n_errors / n_total, n_errors, n_total


def run_comparison(
    dataset_path: str,
    model_id: str,
    col_correct: str = COL_CORRECT,
    col_wrong: str = COL_WRONG,
    batch_size: int = 64,
    device: str = "cuda",
    max_length: Optional[int] = None,
    output_dir: str = "outputs/perplexity_bleu_linear/per",
    split: str = "train",
    config: Optional[str] = None,
    dataset: Optional[Union["Dataset", "DatasetDict"]] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Load dataset and model, compute per-sentence perplexities for both columns,
    compute error rate, and save results.

    Args:
        dataset_path: HuggingFace dataset path (e.g. "username/dataset_name").
        model_id: HuggingFace model id.
        col_correct: Dataset column for correct sentence.
        col_wrong: Dataset column for wrong sentence.
        batch_size: Batch size for inference.
        device: Device string.
        max_length: Max sequence length (default from model config).
        output_dir: Directory for output files.
        split: Dataset split to use.
        config: Optional dataset config name.
        dataset: Optional pre-loaded Dataset or DatasetDict (for testing); if set, dataset_path is only used for output filenames.

    Returns:
        (error_rate, dataframe with columns ppl_sen, ppl_wrong_sen, row_id).
    """
    # Load dataset
    if dataset is not None:
        ds = dataset[split] if isinstance(dataset, dict) else dataset
    else:
        if config is not None:
            ds = load_dataset(dataset_path, name=config, split=split)
        else:
            ds = load_dataset(dataset_path, split=split)
        if isinstance(ds, dict):
            ds = ds[split]
    texts_correct = list(ds[col_correct])
    texts_wrong = list(ds[col_wrong])
    n = len(texts_correct)
    if len(texts_wrong) != n:
        raise ValueError(
            f"Column lengths differ: {col_correct}={n}, {col_wrong}={len(texts_wrong)}"
        )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)
    model.eval()

    # Per-sentence perplexities
    ppl_correct = perplexity_per_sentence(
        model, tokenizer, texts_correct, device, batch_size, max_length
    )
    ppl_wrong = perplexity_per_sentence(
        model, tokenizer, texts_wrong, device, batch_size, max_length
    )

    error_rate, n_errors, n_total = compute_error_rate(ppl_correct, ppl_wrong)

    # Build distribution dataframe
    dist_df = pd.DataFrame({
        "row_id": np.arange(n),
        "ppl_sen": ppl_correct,
        "ppl_wrong_sen": ppl_wrong,
    })

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model_slug = _slug(model_id)
    dataset_slug = _slug(dataset_path)

    error_row = {
        "error_rate": error_rate,
        "n_errors": n_errors,
        "n_total": n_total,
        "n_samples": n,
        "dataset": dataset_path,
        "model_id": model_id,
        "col_correct": col_correct,
        "col_wrong": col_wrong,
    }
    pd.DataFrame([error_row]).to_csv(
        os.path.join(output_dir, f"error_rate_{model_slug}_{dataset_slug}.csv"),
        index=False,
    )

    dist_path = os.path.join(
        output_dir, f"perplexity_distributions_{model_slug}_{dataset_slug}.csv"
    )
    dist_df.to_csv(dist_path, index=False)

    return error_rate, dist_df


def run_comparison_multilang(
    dataset_path: str,
    model_id: str,
    col_correct: str = COL_CORRECT,
    col_wrong: str = COL_WRONG,
    batch_size: int = 64,
    device: str = "cuda",
    max_length: Optional[int] = None,
    output_dir: str = "outputs/perplexity_bleu_linear/per",
    split: str = "train",
    language_codes: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Run perplexity comparison for each language (each language = one dataset config).
    Loads model once and iterates over languages. Saves (language x sample) matrices
    in NPZ and per-language error rates in JSON.

    Args:
        dataset_path: HuggingFace dataset path (e.g. "username/dataset_name").
        model_id: HuggingFace model id.
        col_correct: Dataset column for correct sentence.
        col_wrong: Dataset column for wrong sentence.
        batch_size: Batch size for inference.
        device: Device string.
        max_length: Max sequence length (default from model config).
        output_dir: Directory for output files.
        split: Dataset split to use.
        language_codes: List of language config names to use; default LANG_CODE_TO_NAME.keys().

    Returns:
        (error_rates_by_lang, ppl_sen_matrix, ppl_wrong_sen_matrix, language_codes_list, n_samples_per_language).
        Matrices are (n_languages, max_samples) with NaN padding.
    """
    if language_codes is None:
        language_codes = list(LANG_CODE_TO_NAME.keys())
    # Deduplicate and preserve order (dict.keys() order)
    language_codes = list(dict.fromkeys(language_codes))

    # Load model and tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)
    model.eval()

    error_rates: Dict[str, float] = {}
    list_ppl_sen: List[np.ndarray] = []
    list_ppl_wrong_sen: List[np.ndarray] = []
    list_n_samples: List[int] = []
    valid_lang_codes: List[str] = []

    for lang_code in tqdm(language_codes, desc="Languages"):
        try:
            ds = load_dataset(dataset_path, name=lang_code, split=split)
            if isinstance(ds, dict):
                ds = ds[split]
        except Exception as e:
            tqdm.write(f"[{lang_code}] Skip: {e}")
            continue
        texts_correct = list(ds[col_correct])
        texts_wrong = list(ds[col_wrong])
        n = len(texts_correct)
        if n == 0 or len(texts_wrong) != n:
            tqdm.write(f"[{lang_code}] Skip: empty or column length mismatch")
            continue

        ppl_correct = perplexity_per_sentence(
            model, tokenizer, texts_correct, device, batch_size, max_length
        )
        ppl_wrong = perplexity_per_sentence(
            model, tokenizer, texts_wrong, device, batch_size, max_length
        )
        err_rate, _, n_total = compute_error_rate(ppl_correct, ppl_wrong)
        error_rates[lang_code] = float(err_rate)
        list_ppl_sen.append(ppl_correct)
        list_ppl_wrong_sen.append(ppl_wrong)
        list_n_samples.append(n_total)
        valid_lang_codes.append(lang_code)
    if not valid_lang_codes:
        raise ValueError("No language config could be loaded from the dataset.")

    # Build (language x sample) matrices with NaN padding
    max_samples = max(list_n_samples)
    n_lang = len(valid_lang_codes)
    ppl_sen_matrix = np.full((n_lang, max_samples), np.nan, dtype=float)
    ppl_wrong_sen_matrix = np.full((n_lang, max_samples), np.nan, dtype=float)
    n_samples_per_language = np.array(list_n_samples, dtype=int)
    for i in range(n_lang):
        n = list_n_samples[i]
        ppl_sen_matrix[i, :n] = list_ppl_sen[i]
        ppl_wrong_sen_matrix[i, :n] = list_ppl_wrong_sen[i]

    # Save NPZ: (language x sample) matrices + metadata
    os.makedirs(output_dir, exist_ok=True)
    model_slug = _slug(model_id)
    dataset_slug = _slug(dataset_path)
    npz_path = os.path.join(
        output_dir, f"perplexity_matrices_{model_slug}_{dataset_slug}.npz"
    )
    np.savez(
        npz_path,
        ppl_sen=ppl_sen_matrix,
        ppl_wrong_sen=ppl_wrong_sen_matrix,
        language_codes=np.array(valid_lang_codes, dtype=object),
        n_samples_per_language=n_samples_per_language,
    )

    # Save JSON: language -> error rate
    json_path = os.path.join(
        output_dir, f"error_rates_by_language_{model_slug}_{dataset_slug}.json"
    )
    with open(json_path, "w") as f:
        json.dump(error_rates, f, indent=2)

    return (
        error_rates,
        ppl_sen_matrix,
        ppl_wrong_sen_matrix,
        valid_lang_codes,
        n_samples_per_language,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare perplexity of correct vs wrong sentences; report error rate and save distributions."
    )
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset path (e.g. username/dataset_name)")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model id")
    parser.add_argument("--col_correct", type=str, default=COL_CORRECT, help="Column name for correct sentence")
    parser.add_argument("--col_wrong", type=str, default=COL_WRONG, help="Column name for wrong sentence")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="outputs/perplexity_bleu_linear/per")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--config", type=str, default=None, help="Optional dataset config name (single-language run)")
    parser.add_argument("--multilang", action="store_true", help="Run for each language (each lang = dataset config)")
    parser.add_argument(
        "--margins_multilang",
        action="store_true",
        help="Compute MultiBLiMP correct-minus-wrong logprob margins for BLEU language coverage.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Config names for multilang. Accepts either space-separated values "
             "(--languages eng fra deu) or a single comma-separated string "
             "(--languages eng,fra,deu). Default: LANG_CODE_TO_NAME.keys()",
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.margins_multilang:
        if args.languages:
            tokens = []
            for chunk in args.languages:
                tokens.extend(s.strip() for s in chunk.split(",") if s.strip())
            bleu_codes = tokens or BLEU_LANGS_24
        else:
            bleu_codes = BLEU_LANGS_24
        language_map, absent = available_multiblimp_language_map(args.dataset, bleu_codes)
        print(f"MultiBLiMP coverage: {len(language_map)}/{len(bleu_codes)} present")
        if absent:
            print(f"Absent BLEU codes: {', '.join(absent)}")
        run_margins_multilang(
            dataset_path=args.dataset,
            model_id=args.model_id,
            language_map=language_map,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
            output_dir=args.output_dir,
        )
        print(f"Margins saved under {args.output_dir}")
    elif args.multilang:
        language_codes = None
        if args.languages:
            # Accept both ["eng", "fra"] and ["eng,fra,deu"] forms.
            tokens = []
            for chunk in args.languages:
                tokens.extend(s.strip() for s in chunk.split(",") if s.strip())
            language_codes = tokens or None
        error_rates, ppl_sen_mat, ppl_wrong_mat, lang_codes, n_per_lang = run_comparison_multilang(
            dataset_path=args.dataset,
            model_id=args.model_id,
            col_correct=args.col_correct,
            col_wrong=args.col_wrong,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
            output_dir=args.output_dir,
            split=args.split,
            language_codes=language_codes,
        )
        print(f"Error rates by language: {len(error_rates)} languages")
        for lang, rate in list(error_rates.items())[:5]:
            print(f"  {lang}: {rate:.4f}")
        if len(error_rates) > 5:
            print(f"  ... and {len(error_rates) - 5} more")
        print(f"NPZ (language x sample matrices) and JSON (error rates) saved to {args.output_dir}")
    else:
        error_rate, dist_df = run_comparison(
            dataset_path=args.dataset,
            model_id=args.model_id,
            col_correct=args.col_correct,
            col_wrong=args.col_wrong,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
            output_dir=args.output_dir,
            split=args.split,
            config=args.config,
        )
        print(f"Error rate: {error_rate:.4f} (n_errors/n_total)")
        print(f"Distributions saved to {args.output_dir}")


if __name__ == "__main__":
    main()

# python perplexity_comparison.py --dataset jumelet/multiblimp --model_id meta-llama/Meta-Llama-3.1-8B --col_correct sen --col_wrong wrong_sen --batch_size 64 --output_dir outputs/perplexity_bleu_linear/per --split devtest --config eng_Latn --max_length 2048 --device cuda

# Todos los idiomas de LANG_CODE_TO_NAME (cada uno = config del dataset)
# python scripts/perplexity_comparison.py --dataset jumelet/multiblimp --model_id meta-llama/Meta-Llama-3.1-8B --multilang --split train --device cuda --max_length 2048 --batch_size 64

# # Solo algunos idiomas o configs (p. ej. FLORES)
# python scripts/perplexity_comparison.py --dataset gsarti/flores_101 --model_id meta-llama/Meta-Llama-3.1-8B --multilang --languages eng_Latn,spa_Latn --split devtest --col_correct sentence --col_wrong wrong_sen
