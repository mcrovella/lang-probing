"""Compute per-language bits-per-byte (BPB) on FLORES devtest.

Why BPB: per-token corpus perplexity (`perplexity_results_{model}.csv`) is
tokenization-confounded across languages -- a language fragmented into many
predictable sub-tokens gets a low per-token PPL regardless of competence. BPB
normalizes the model's negative log-likelihood by the *raw UTF-8 byte count* of
the text instead of by token count, which removes the per-token granularity
confound and is comparable across languages.

How (exact reconstruction, no GPU needed): `run_perplexity.py` already computed
corpus PPL = exp(total_NLL / total_tokens) on FLORES devtest, where total_tokens
is the number of *scored* tokens = sum over sentences of (len(tokens_with_BOS) - 1)
with the model's own tokenizer and no truncation (FLORES sentences are far below
max_position_embeddings). Therefore

    total_NLL_nats = total_tokens * ln(PPL)                      [exact identity]
    bits_per_byte  = total_NLL_nats / (total_bytes * ln 2)

We recompute total_tokens (tokenizer, CPU) and total_bytes (UTF-8, CPU) exactly
the way run_perplexity scored them, and read PPL from the saved CSV. This is
algebraically identical to a direct Sum(NLL)/Sum(bytes) GPU pass; see the report
for the verification argument.

Output: outputs/perplexity_bleu_linear/bleu_and_ppl/bits_per_byte_{model}.csv
"""
import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from lang_probing_src.config import LANG_CODE_TO_NAME, OUTPUTS_DIR

LN2 = math.log(2.0)

# Resolve a locally-cached snapshot dir for the tokenizer (offline-safe).
def resolve_tokenizer_dir(model_key: str) -> str:
    hub = os.path.expanduser(
        os.environ.get("HF_HOME", "/projectnb/mcnet/jbrin/.cache/huggingface")
    )
    repo = {
        "llama": "models--meta-llama--Llama-3.1-8B",
        "aya": "models--CohereForAI--aya-23-8B",
    }[model_key]
    snaps = sorted(glob.glob(os.path.join(hub, "hub", repo, "snapshots", "*")))
    if not snaps:
        raise FileNotFoundError(f"No cached snapshot for {repo} under {hub}/hub")
    return snaps[-1]


def load_ppl_table(model_key: str, base: Path) -> dict:
    path = base / f"perplexity_results_{model_key}.csv"
    out = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            v = (row.get("Perplexity") or "").strip()
            try:
                fv = float(v)
            except ValueError:
                continue
            if math.isfinite(fv) and fv > 0:
                out[row["Language"].strip()] = fv
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama", choices=["llama", "aya"])
    ap.add_argument("--max_length", type=int, default=None,
                    help="Cap tokens per sentence (defaults to model max_position_embeddings).")
    args = ap.parse_args()

    base = Path(OUTPUTS_DIR) / "perplexity_bleu_linear" / "bleu_and_ppl"
    tok_dir = resolve_tokenizer_dir(args.model)
    tok = AutoTokenizer.from_pretrained(tok_dir)

    # match run_perplexity's truncation cap
    if args.max_length is None:
        cfg = json.load(open(Path(tok_dir) / "config.json"))
        args.max_length = cfg.get("max_position_embeddings", 131072)

    ppl = load_ppl_table(args.model, base)

    rows = []
    for lang in LANG_CODE_TO_NAME.keys():
        if lang not in ppl:
            print(f"[skip] {lang}: no saved PPL")
            continue
        try:
            ds = load_dataset("gsarti/flores_101", lang, split="devtest")
        except Exception as e:
            print(f"[skip] {lang}: flores load failed: {repr(e)[:120]}")
            continue
        sents = ds["sentence"]
        total_tokens = 0
        total_bytes = 0
        for s in sents:
            ids = tok(s).input_ids  # default add_special_tokens=True -> includes BOS
            if len(ids) > args.max_length:
                ids = ids[: args.max_length]
            total_tokens += max(len(ids) - 1, 0)  # scored tokens (shift by 1)
            total_bytes += len(s.encode("utf-8"))
        if total_tokens == 0 or total_bytes == 0:
            print(f"[skip] {lang}: empty")
            continue
        ln_ppl = math.log(ppl[lang])
        total_nll = total_tokens * ln_ppl
        bpb = total_nll / (total_bytes * LN2)
        rows.append({
            "lang": lang,
            "ppl_per_token": ppl[lang],
            "n_sentences": len(sents),
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
            "bytes_per_token": total_bytes / total_tokens,
            "bits_per_token": ln_ppl / LN2,
            "bits_per_byte": bpb,
        })
        print(f"{lang}: ppl={ppl[lang]:.3f} bpt={ln_ppl/LN2:.3f} "
              f"bytes/tok={total_bytes/total_tokens:.3f} -> BPB={bpb:.4f}")

    out_path = base / f"bits_per_byte_{args.model}.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} languages -> {out_path}")


if __name__ == "__main__":
    main()
