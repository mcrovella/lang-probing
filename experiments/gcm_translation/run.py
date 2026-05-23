"""
GCM translation attribution: CLI entry point.

For one (src_lang, tgt_lang) direction, iterates FLORES (orig, cf) pairs and
computes head + SAE attribution per pair. Saves stacked tensors + ranking
JSON + summary metadata.

Bug-fix-aware:
  * Token-space truncation (no decode->retokenize round-trip) — tokenize_pair
    handles via max_response_tokens.
  * Bare response (no leading space) — works for RTL/CJK targets too.
  * Per-component try/except so SAE-success-but-heads-fail doesn't desync
    the stacked tensors (NaN sentinels keep all lists index-aligned).
  * GPU memory released between pairs.
  * Per-token-mean for the sign-sanity stat (not biased by response length).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch

# Safety net for running from a non-installed checkout
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from lang_probing_src.config import (
    MODEL_ID,
    SAE_ID,
    OUTPUTS_DIR,
    SAE_DIM,
)
from lang_probing_src.utils import setup_model, get_device_info

# local imports (this experiment)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flores_pairs import sample_pairs, sample_null_triples, get_shots, make_prompt
from gcm_core import gcm_attribute_sae, gcm_attribute_heads, tokenize_pair, sum_response_logprobs, set_logit_scale
from lang_probing_src.config import TRACER_KWARGS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _compute_clean_metrics(model, tokenizer, prompt_orig, response_orig, response_cf, device, max_response_tokens):
    """
    Compute logp(r_orig | prompt_orig) and logp(r_cf | prompt_orig) in two
    no-grad forward passes. Used by run.py to hoist the clean-metric computation
    out of the per-component attribution functions when --components both, so
    neither gcm_attribute_sae nor gcm_attribute_heads duplicates these passes.

    Both metrics are scored against prompt_orig (not prompt_cf), matching the
    GCM formulation: M = logp(r_orig | p_orig, z) - logp(r_cf | p_orig, z).
    """
    pr_or = tokenize_pair(tokenizer, prompt_orig, response_orig, device, max_response_tokens)
    pr_or_rc = tokenize_pair(tokenizer, prompt_orig, response_cf, device, max_response_tokens)
    with model.trace(pr_or.input_ids, **TRACER_KWARGS), torch.no_grad():
        m_orig_proxy = sum_response_logprobs(
            model.lm_head.output, pr_or.input_ids, pr_or.response_start
        ).save()
    with model.trace(pr_or_rc.input_ids, **TRACER_KWARGS), torch.no_grad():
        m_cf_proxy = sum_response_logprobs(
            model.lm_head.output, pr_or_rc.input_ids, pr_or_rc.response_start
        ).save()
    return float(m_orig_proxy.item()), float(m_cf_proxy.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_lang", required=True, help="e.g. English")
    p.add_argument("--tgt_lang", required=True, help="e.g. Spanish")
    p.add_argument("--n_pairs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="dev", choices=["dev", "devtest"])
    p.add_argument("--output_dir", default=os.path.join(OUTPUTS_DIR, "gcm_translation"))
    p.add_argument("--top_heads_k", type=int, default=20)
    p.add_argument("--top_sae_k", type=int, default=50)
    p.add_argument("--components", default="both", choices=["both", "heads", "sae"])
    p.add_argument("--model_id", default=MODEL_ID,
                   help="HF model id. Use CohereForAI/aya-23-8B for the Aya rerun.")
    p.add_argument("--sae_id", default=SAE_ID,
                   help="SAE repo id. Ignored when --components heads (SAE not loaded).")
    p.add_argument("--max_response_tokens", type=int, default=128,
                   help="Truncate target translation to this many tokens (in token space).")
    p.add_argument("--null_control", action="store_true",
                   help="Phase 2 null mode: sample (A, B, C) triples and score logp(tgt_B|p_A) "
                        "vs logp(tgt_C|p_A) with z patched between z_B and z_C. Neither response "
                        "is the gold translation of A; isolates content-discrimination from the "
                        "translation-circuit anchor.")
    args = p.parse_args()

    direction_key = f"{args.src_lang}__{args.tgt_lang}"
    out_dir = Path(args.output_dir) / direction_key
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== GCM translation: {direction_key} ===")
    logger.info(f"  n_pairs={args.n_pairs}  components={args.components}  out={out_dir}")

    # --- GPU compute-capability sanity check ---
    # Our conda env's PyTorch build does not include sm_120 (Blackwell, e.g.
    # RTX Pro 6000 on scc-b04). If a job lands there it fails mid-run with
    # "no kernel image is available for execution on the device". Detect
    # before model load so the exit is fast and the failure is obvious in
    # the qsub log.
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        cap_float = cap[0] + 0.1 * cap[1]
        if cap_float >= 10.0:
            logger.error(
                f"Refusing to run: GPU compute capability {cap_float:.1f} "
                f"(host={os.environ.get('HOSTNAME', '?')}) is not supported "
                f"by this conda env's PyTorch (no sm_{cap[0]}{cap[1]} kernels). "
                f"Constrain qsub with `-l gpu_type=A100` or `-l gpu_type=H200`."
            )
            sys.exit(2)
        logger.info(f"GPU compute capability: {cap_float:.1f}")

    # --- Setup model ---
    # Heads-only attribution does not need the SAE; skip loading it (the Aya
    # rerun runs --components heads, and no Aya SAE is in scope for this plan).
    effective_sae_id = None if args.components == "heads" else args.sae_id
    logger.info(f"Loading model {args.model_id} and SAE {effective_sae_id}...")
    t_load = time.time()
    model, submodule, autoencoder, tokenizer = setup_model(args.model_id, effective_sae_id)
    device, _ = get_device_info()
    logger.info(f"  loaded in {time.time() - t_load:.1f}s on {device}")

    # Cohere/Aya apply config.logit_scale after lm_head; we read lm_head.output
    # directly, so reapply it (Llama has no logit_scale -> stays 1.0).
    _scale = float(getattr(model.model.config, "logit_scale", 1.0))
    set_logit_scale(_scale)
    logger.info(f"  logit_scale = {_scale}")

    # Freeze all model + autoencoder params: GCM only needs grad w.r.t. z_leaf,
    # not w.r.t. parameters. Without this, every backward() pass accumulates
    # ~16 GB of gradient memory on the 8B-param model and OOMs after pair 2.
    for p in model.parameters():
        p.requires_grad_(False)
    if autoencoder is not None:
        for p in autoencoder.parameters():
            p.requires_grad_(False)

    cfg = model.model.config
    n_layers_cfg = cfg.num_hidden_layers
    n_heads_cfg = cfg.num_attention_heads

    # --- Shots + sampled pairs ---
    shots = get_shots(args.src_lang, args.tgt_lang, split=args.split)
    if args.null_control:
        triples = sample_null_triples(
            args.src_lang, args.tgt_lang, args.n_pairs, seed=args.seed, split=args.split
        )
        logger.info(
            f"NULL mode: sampled {len(triples)} (A, B, C) triples (split={args.split}); "
            f"shots from indices (0,1). Neither tgt_B nor tgt_C is the gold translation of A."
        )
        pairs = triples  # uniform name for the loop below
    else:
        pairs = sample_pairs(args.src_lang, args.tgt_lang, args.n_pairs, seed=args.seed, split=args.split)
        logger.info(f"Sampled {len(pairs)} pairs (split={args.split}); shots from indices (0,1).")

    # --- Per-pair loop ---
    head_ies = []
    sae_ies = []
    pair_records = []

    head_nan_sentinel = torch.full((n_layers_cfg, n_heads_cfg), float("nan"))
    sae_nan_sentinel = torch.full((SAE_DIM,), float("nan"))

    t_start = time.time()
    for i, pair in enumerate(pairs):
        if args.null_control:
            # Null: prompt = p_A; both scored responses are non-A (tgt_B, tgt_C);
            # z_orig := z_B (cached from p_B), z_cf := z_C (cached from p_C).
            prompt_orig = make_prompt(args.src_lang, args.tgt_lang, shots, pair.src_a)
            prompt_cf   = prompt_orig                 # not used for caching in null path
            response_orig = pair.tgt_b
            response_cf   = pair.tgt_c
            cache_prompt_orig = make_prompt(args.src_lang, args.tgt_lang, shots, pair.src_b)
            cache_response_orig = pair.tgt_b
            cache_prompt_cf = make_prompt(args.src_lang, args.tgt_lang, shots, pair.src_c)
            cache_response_cf = pair.tgt_c
            rec = {
                "pair_id": pair.pair_id,
                "a_idx": pair.a_idx,
                "b_idx": pair.b_idx,
                "c_idx": pair.c_idx,
                "sae_ok": False,
                "heads_ok": False,
            }
        else:
            prompt_orig = make_prompt(args.src_lang, args.tgt_lang, shots, pair.src_orig)
            prompt_cf   = make_prompt(args.src_lang, args.tgt_lang, shots, pair.src_cf)
            # No leading space on response; make_prompt's prompt ends with "Spanish: ".
            response_orig = pair.tgt_orig
            response_cf   = pair.tgt_cf
            cache_prompt_orig = None
            cache_response_orig = None
            cache_prompt_cf = None
            cache_response_cf = None
            rec = {
                "pair_id": pair.pair_id,
                "orig_idx": pair.orig_idx,
                "cf_idx": pair.cf_idx,
                "sae_ok": False,
                "heads_ok": False,
            }

        # --- Hoist clean-metric forward passes when running both components ---
        # When --components both, each attribution function would independently
        # run 2 no-grad forward passes for m_orig_clean / m_cf_clean (~20%
        # extra compute). Pre-compute once here and pass in to both.
        hoisted_m_orig_clean = None
        hoisted_m_cf_clean = None
        if args.components == "both":
            try:
                hoisted_m_orig_clean, hoisted_m_cf_clean = _compute_clean_metrics(
                    model, tokenizer, prompt_orig, response_orig, response_cf,
                    device, args.max_response_tokens,
                )
            except Exception as e:
                logger.warning(
                    f"  [{i+1}/{len(pairs)}] clean-metric pre-computation FAILED on "
                    f"{pair.pair_id}: {type(e).__name__}: {e}; "
                    f"both attribution functions will compute internally."
                )

        # --- SAE attribution ---
        if args.components in ("both", "sae"):
            try:
                sae_out = gcm_attribute_sae(
                    model, submodule, autoencoder, tokenizer,
                    prompt_orig, response_orig, prompt_cf, response_cf, device,
                    max_response_tokens=args.max_response_tokens,
                    cache_prompt_orig=cache_prompt_orig,
                    cache_response_orig=cache_response_orig,
                    cache_prompt_cf=cache_prompt_cf,
                    cache_response_cf=cache_response_cf,
                    m_orig_clean=hoisted_m_orig_clean,
                    m_cf_clean=hoisted_m_cf_clean,
                )
                sae_ies.append(sae_out["ie"])
                rec.update({
                    "sae_ok": True,
                    "sae_metric_orig_clean": sae_out["metric_orig_clean"],
                    "sae_metric_cf_clean": sae_out["metric_cf_clean"],
                    "sae_metric_orig_patched": sae_out["metric_orig_patched"],
                    "sae_metric_cf_patched": sae_out["metric_cf_patched"],
                    "sae_metric_diff_patched": sae_out["metric_diff_patched"],
                    "sae_sanity_orig_drift": sae_out["sanity_orig_drift"],
                    "sae_z_orig_norm": sae_out["z_orig_norm"],
                    "sae_z_cf_norm": sae_out["z_cf_norm"],
                    "decoded_last_src_orig": sae_out["decoded_last_src_orig"],
                    "decoded_last_src_cf": sae_out["decoded_last_src_cf"],
                    "n_response_orig": sae_out["n_response_orig"],
                    "n_response_cf": sae_out["n_response_cf"],
                })
                del sae_out
            except Exception as e:
                logger.warning(f"  [{i+1}/{len(pairs)}] SAE FAILED on {pair.pair_id}: {type(e).__name__}: {e}")
                sae_ies.append(sae_nan_sentinel.clone())
                rec["sae_error"] = f"{type(e).__name__}: {e}"

        # Free SAE-trace memory before heads (heads needs ~1 GB more headroom)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Head attribution ---
        if args.components in ("both", "heads"):
            try:
                head_out = gcm_attribute_heads(
                    model, tokenizer,
                    prompt_orig, response_orig, prompt_cf, response_cf, device,
                    max_response_tokens=args.max_response_tokens,
                    cache_prompt_orig=cache_prompt_orig,
                    cache_response_orig=cache_response_orig,
                    cache_prompt_cf=cache_prompt_cf,
                    cache_response_cf=cache_response_cf,
                    m_orig_clean=hoisted_m_orig_clean,
                    m_cf_clean=hoisted_m_cf_clean,
                )
                head_ies.append(head_out["ie"])
                rec.update({
                    "heads_ok": True,
                    "head_metric_orig_clean": head_out["metric_orig_clean"],
                    "head_metric_cf_clean": head_out["metric_cf_clean"],
                    "head_metric_orig_patched": head_out["metric_orig_patched"],
                    "head_metric_cf_patched": head_out["metric_cf_patched"],
                    "head_metric_diff_patched": head_out["metric_diff_patched"],
                    "head_sanity_orig_drift": head_out["sanity_orig_drift"],
                })
                del head_out
            except Exception as e:
                logger.warning(f"  [{i+1}/{len(pairs)}] HEADS FAILED on {pair.pair_id}: {type(e).__name__}: {e}")
                head_ies.append(head_nan_sentinel.clone())
                rec["heads_error"] = f"{type(e).__name__}: {e}"

        pair_records.append(rec)

        # Free GPU memory between pairs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (i + 1) % 5 == 0 or i == len(pairs) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(pairs) - i - 1) / rate if rate > 0 else 0
            n_sae_ok = sum(1 for r in pair_records if r.get("sae_ok"))
            n_heads_ok = sum(1 for r in pair_records if r.get("heads_ok"))
            logger.info(
                f"  [{i+1}/{len(pairs)}] {elapsed/(i+1):.1f}s/pair  ETA {eta/60:.1f}min  "
                f"sae_ok={n_sae_ok}  heads_ok={n_heads_ok}"
            )

    # --- Save tensors ---
    if head_ies:
        head_stack = torch.stack(head_ies)        # [N, n_layers, n_heads]
        torch.save(head_stack, out_dir / "heads_ie.pt")
        logger.info(f"Saved heads_ie.pt  shape={tuple(head_stack.shape)}")

    if sae_ies:
        sae_stack = torch.stack(sae_ies)          # [N, SAE_DIM]
        torch.save(sae_stack, out_dir / "sae_ie.pt")
        logger.info(f"Saved sae_ie.pt  shape={tuple(sae_stack.shape)}")

    # --- Top-K rankings (NaN-aware) ---
    top = {}
    if head_ies:
        # NaN-aware aggregation: failed pairs are NaN sentinels and ignored.
        mean_abs_head = torch.nanmean(head_stack.abs(), dim=0)        # [n_layers, n_heads]
        mean_signed_head = torch.nanmean(head_stack, dim=0)
        flat = mean_abs_head.flatten()
        vals, idxs = flat.topk(min(args.top_heads_k, flat.numel()))
        n_heads = mean_abs_head.shape[1]
        top["heads_top_k_by_mean_abs_ie"] = [
            {
                "layer": int(idx.item() // n_heads),
                "head": int(idx.item() % n_heads),
                "mean_abs_ie": float(v.item()),
                "mean_signed_ie": float(mean_signed_head.flatten()[idx].item()),
            }
            for v, idx in zip(vals, idxs)
        ]

    if sae_ies:
        mean_abs_sae = torch.nanmean(sae_stack.abs(), dim=0)           # [SAE_DIM]
        mean_signed_sae = torch.nanmean(sae_stack, dim=0)
        vals, idxs = mean_abs_sae.topk(min(args.top_sae_k, mean_abs_sae.numel()))
        top["sae_top_k_by_mean_abs_ie"] = [
            {
                "feature_idx": int(idx.item()),
                "mean_abs_ie": float(v.item()),
                "mean_signed_ie": float(mean_signed_sae[idx].item()),
            }
            for v, idx in zip(vals, idxs)
        ]

    with open(out_dir / "top_rankings.json", "w") as f:
        json.dump(top, f, indent=2)

    # --- Summary ---
    summary = {
        "src_lang": args.src_lang,
        "tgt_lang": args.tgt_lang,
        "direction_key": direction_key,
        "null_control": bool(args.null_control),
        "n_pairs_attempted": len(pairs),
        "n_pairs_successful_heads": sum(1 for r in pair_records if r.get("heads_ok")),
        "n_pairs_successful_sae": sum(1 for r in pair_records if r.get("sae_ok")),
        "seed": args.seed,
        "split": args.split,
        "max_response_tokens": args.max_response_tokens,
        "elapsed_seconds": time.time() - t_start,
        "model_id": args.model_id,
        "sae_id": effective_sae_id,
        "components": args.components,
        "sign_convention": "orig_minus_cf",
    }

    # Sign sanity: per-token-mean comparison (length-bias-free)
    sae_clean_pertoken = []
    for r in pair_records:
        if r.get("sae_ok"):
            n_orig = max(r.get("n_response_orig", 1), 1)
            n_cf = max(r.get("n_response_cf", 1), 1)
            o_pt = r["sae_metric_orig_clean"] / n_orig
            c_pt = r["sae_metric_cf_clean"] / n_cf
            sae_clean_pertoken.append((o_pt, c_pt))
    if sae_clean_pertoken:
        n_orig_preferred = sum(1 for o, c in sae_clean_pertoken if o > c)
        summary["sae_clean_frac_orig_preferred_pertoken"] = n_orig_preferred / len(sae_clean_pertoken)
        summary["sae_clean_mean_logp_orig_pertoken"] = sum(o for o, _ in sae_clean_pertoken) / len(sae_clean_pertoken)
        summary["sae_clean_mean_logp_cf_pertoken"] = sum(c for _, c in sae_clean_pertoken) / len(sae_clean_pertoken)

    # Sanity drift summary (smaller is better; tracks bf16 roundoff in patch identity)
    drifts_sae = [r["sae_sanity_orig_drift"] for r in pair_records if "sae_sanity_orig_drift" in r]
    if drifts_sae:
        summary["sae_max_orig_drift"] = max(drifts_sae)
        summary["sae_mean_orig_drift"] = sum(drifts_sae) / len(drifts_sae)
    drifts_head = [r["head_sanity_orig_drift"] for r in pair_records if "head_sanity_orig_drift" in r]
    if drifts_head:
        summary["head_max_orig_drift"] = max(drifts_head)
        summary["head_mean_orig_drift"] = sum(drifts_head) / len(drifts_head)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_pair_records.json", "w") as f:
        json.dump(pair_records, f, indent=2)

    logger.info(f"Done. Summary: {summary}")


if __name__ == "__main__":
    main()
