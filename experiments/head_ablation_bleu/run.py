"""BLEU head-ablation runner.

This file provides the reusable hook machinery and a CLI for cached FLORES
translation generations. Heavy runs should be submitted through SCC/qsub.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from experiments.head_ablation_bleu.translate import (
    build_flores_dataset,
    build_prompt,
    corpus_bleu_score,
    extract_translation,
    generate_batch,
)
from experiments.head_ablation_multiblimp.heads import (
    LANG_KEY_TO_NAME,
    aggregate_target_heads,
    sample_stratified_controls,
    select_signed_heads,
)
from experiments.head_ablation_multiblimp.run import MODEL_ID
from lang_probing_src.utils import get_device_info
from transformers import AutoModelForCausalLM, AutoTokenizer


def _unwrap_hf_model(model):
    return getattr(model, "_model", None) or getattr(model, "model", None) or model


def _decoder_layers(hf_model):
    """Return the decoder layer list for either a *ForCausalLM (.model.layers)
    or the inner transformer model (.layers). nnsight unwrap can yield either."""
    inner = getattr(hf_model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    return hf_model.layers


def install_head_ablation(hf_model, ablations, mean_acts):
    """Install persistent o_proj output hooks for all requested heads.

    `ablations` is an iterable of `(layer, head, mode)`. Only `mean` mode is
    used by the BLEU plan. The hook changes the o_proj output by the exact
    projected delta implied by replacing that head's o_proj input with its
    mean activation at every sequence position.
    """
    layers = _decoder_layers(hf_model)
    cfg = hf_model.config
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    by_layer = defaultdict(list)
    for layer, head, mode in ablations:
        if mode != "mean":
            raise ValueError(f"BLEU ablation supports only mean mode, got {mode!r}")
        by_layer[int(layer)].append(int(head))

    handles = []
    for layer, heads in by_layer.items():
        module = layers[layer].self_attn.o_proj

        def hook(mod, inputs, output, *, layer=layer, heads=tuple(heads)):
            o_in = inputs[0]
            W = mod.weight
            delta_out = torch.zeros_like(output)
            for head in heads:
                start = head * head_dim
                stop = start + head_dim
                mean = mean_acts[layer, head].to(device=o_in.device, dtype=o_in.dtype)
                delta_head = mean.view(1, 1, head_dim) - o_in[:, :, start:stop]
                W_h = W[:, start:stop]
                delta_out = delta_out + (delta_head @ W_h.T)
            return output + delta_out

        handles.append(module.register_forward_hook(hook))
    return handles


def remove_hooks(handles) -> None:
    for h in handles:
        h.remove()


def compute_mean_activations_translation(model, prompts, tokenizer, batch_size: int = 8):
    """Compute per-layer/head mean o_proj.input over translation prompts."""
    hf_model = _unwrap_hf_model(model)
    cfg = hf_model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    device = next(hf_model.parameters()).device
    sums = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float32)
    counts = torch.zeros(n_layers, dtype=torch.long)

    buffers = [[] for _ in range(n_layers)]
    handles = []
    decoder_layers = _decoder_layers(hf_model)
    for layer in range(n_layers):
        module = decoder_layers[layer].self_attn.o_proj

        def hook(mod, inputs, output, *, layer=layer):
            x = inputs[0].detach().float().cpu()
            buffers[layer].append(x)

        handles.append(module.register_forward_hook(hook))

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            hf_model(**inputs)
        for layer in range(n_layers):
            x = torch.cat(buffers[layer], dim=0)
            buffers[layer].clear()
            x = x.reshape(-1, n_heads, head_dim)
            sums[layer] += x.sum(dim=0)
            counts[layer] += x.shape[0]
    tokenizer.padding_side = old_padding_side
    remove_hooks(handles)
    return sums / counts.clamp(min=1).view(n_layers, 1, 1)


def _sentence_bleu(pred: str, gold: str) -> float:
    import sacrebleu

    return float(sacrebleu.sentence_bleu(pred, [gold]).score)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target_lang", required=True, choices=list(LANG_KEY_TO_NAME))
    ap.add_argument("--source_langs", nargs="+", default=None)
    ap.add_argument("--split_pct", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--gcm_dir", type=Path, default=Path("outputs/gcm_translation"))
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/head_ablation_bleu"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_id", default=MODEL_ID,
                    help="HF model id. Use CohereForAI/aya-23-8B with --gcm_dir the Aya GCM dir.")
    args = ap.parse_args()

    source_langs = args.source_langs or [k for k in LANG_KEY_TO_NAME if k != args.target_lang]
    langs = sorted(set(source_langs + [args.target_lang]))
    flores = build_flores_dataset(langs, args.split_pct)
    prompts_by_src = {
        src: [
            build_prompt(flores[src], flores[args.target_lang], i)
            for i in range(len(flores[args.target_lang]))
        ]
        for src in source_langs
    }

    # Load the HF CausalLM directly (this experiment uses plain PyTorch forward
    # hooks + HF .generate(), so nnsight is unnecessary and its unwrap is
    # version-fragile). CausalLM gives .generate, .config, and .model.layers.
    device, dtype = get_device_info()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)

    all_prompts = [p for prompts in prompts_by_src.values() for p in prompts]
    mean_acts = compute_mean_activations_translation(
        hf_model, all_prompts, tokenizer, batch_size=args.batch_size
    ).to(next(hf_model.parameters()).device)

    target_name = LANG_KEY_TO_NAME[args.target_lang]
    res = aggregate_target_heads(args.gcm_dir, target_name, top_k=32)
    signed = select_signed_heads(res["signed_map"], res["abs_map"], n_per_sign=10)
    selected = signed["pos"] + signed["neg"]
    ctrl = sample_stratified_controls(
        selected, hf_model.config.num_attention_heads, seed=args.seed
    )
    conditions = [
        ("baseline", []),
        ("pos10_mean", [(h["layer"], h["head"], "mean") for h in signed["pos"]]),
        ("neg10_mean", [(h["layer"], h["head"], "mean") for h in signed["neg"]]),
        ("ctrl_matched10_mean", [(h["layer"], h["head"], "mean") for h in ctrl]),
    ]

    out_dir = args.output_dir / args.target_lang
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for condition_name, ablations in conditions:
        handles = install_head_ablation(hf_model, ablations, mean_acts) if ablations else []
        try:
            rows = []
            for src, prompts in prompts_by_src.items():
                gens = generate_batch(
                    hf_model, tokenizer, prompts, args.max_new_tokens, args.batch_size
                )
                preds = [extract_translation(g) for g in gens]
                golds = flores[args.target_lang]
                for i, (prompt, gen, pred, gold) in enumerate(zip(prompts, gens, preds, golds)):
                    rows.append({
                        "direction": f"{src}->{args.target_lang}",
                        "sentence_idx": i,
                        "source": flores[src][i],
                        "gold": gold,
                        "generation": gen,
                        "prediction": pred,
                        "sentence_bleu": _sentence_bleu(pred, gold),
                        "head_set": condition_name,
                    })
            by_dir = {}
            for src in source_langs:
                dir_rows = [r for r in rows if r["direction"] == f"{src}->{args.target_lang}"]
                by_dir[src] = corpus_bleu_score(
                    [r["prediction"] for r in dir_rows], [r["gold"] for r in dir_rows]
                )
            with (out_dir / f"{condition_name}.jsonl").open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary.append({"condition": condition_name, "bleu_by_source": by_dir})
        finally:
            remove_hooks(handles)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
