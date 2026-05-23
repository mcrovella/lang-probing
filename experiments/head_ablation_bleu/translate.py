"""BLEU translation harness for head-ablation experiments."""
from __future__ import annotations

import re
from typing import Iterable

import torch
from datasets import load_dataset


# gsarti/flores_101 configs are bare ISO codes (plus zho_simpl/zho_trad), not
# the FLORES-200 <iso>_<Script> form. Identity map over the BLEU codes.
FLORES_CONFIG = {
    c: c for c in (
        "ara", "rus", "fra", "zho_simpl", "ita", "por", "ind", "heb", "pol",
        "deu", "ell", "tur", "vie", "ukr", "spa", "fas", "kor", "hin", "jpn",
        "eng", "zho_trad", "ces", "nld", "ron",
    )
}


def build_flores_dataset(lang_keys: Iterable[str], split_pct: int = 10) -> dict[str, list[str]]:
    """Load FLORES devtest sentences for language keys, optionally slicing by percent."""
    if split_pct <= 0 or split_pct > 100:
        raise ValueError("split_pct must be in 1..100")
    split = f"devtest[:{split_pct}%]" if split_pct < 100 else "devtest"
    out: dict[str, list[str]] = {}
    for key in lang_keys:
        config = FLORES_CONFIG.get(key, key)
        ds = load_dataset("gsarti/flores_101", config, split=split)
        col = "sentence" if "sentence" in ds.column_names else ds.column_names[0]
        out[key] = list(ds[col])
    return out


def build_prompt(src_sents: list[str], tgt_sents: list[str], i: int) -> str:
    """Build the 2-shot cyclic `src >> tgt || src >>` prompt."""
    if len(src_sents) != len(tgt_sents):
        raise ValueError("source and target sentence lists must align")
    n = len(src_sents)
    if n < 3:
        raise ValueError("need at least three sentence pairs for 2-shot cyclic prompting")
    j0 = (i - 2) % n
    j1 = (i - 1) % n
    return (
        f"{src_sents[j0]} >> {tgt_sents[j0]} ||\n"
        f"{src_sents[j1]} >> {tgt_sents[j1]} ||\n"
        f"{src_sents[i]} >>"
    )


def extract_translation(output_text: str) -> str:
    """Extract the translation from a prompt-stripped generation.

    `generate_batch` returns only the tokens generated after the query's
    trailing ``>>``, so the translation is the FIRST ``||``-delimited segment;
    everything after the first ``||`` is the model continuing the few-shot
    pattern (greedy decoding almost always does this) and must be discarded.
    A stray ``>>`` inside the first segment (model re-echoing the pattern) is
    handled by keeping what follows it.
    """
    seg = re.split(r"\s*\|\|\s*", output_text, maxsplit=1)[0]
    if ">>" in seg:
        seg = seg.rsplit(">>", 1)[1]
    return seg.strip()


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Batched greedy generation with left padding and EOS as pad."""
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    generations: list[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
            )
        prompt_len = inputs.input_ids.shape[1]
        decoded = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
        generations.extend(decoded)
    tokenizer.padding_side = old_padding_side
    return generations


def corpus_bleu_score(preds: list[str], golds: list[str]) -> float:
    """Compute sacreBLEU corpus score."""
    import sacrebleu

    return float(sacrebleu.corpus_bleu(preds, [golds]).score)
