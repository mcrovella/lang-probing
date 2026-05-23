"""
ACP-vs-ATP faithfulness check for GCM head attribution.

ATP (attribution patching, what `gcm_attribute_heads` computes) approximates the
indirect effect of patching a component via a 1st-order Taylor expansion:

    IE_atp = grad_z M . (z_orig - z_cf)        (paper Eq. 1; one fwd+bwd)

ACP (activation patching) is the ground-truth version: actually run the model
with the component patched and measure the metric directly:

    IE_acp = M(z patched at component=cf) - M(z unpatched)

This script picks the top-K heads from a completed direction's `heads_ie.pt`,
samples M pairs from `per_pair_records.json`, and runs ACP on each (head, pair).
Reports the Pearson correlation between ATP and ACP across the K * M points,
and emits a scatter plot.

Usage:
    python experiments/gcm_translation/validate_faithfulness.py \\
        --direction_dir outputs/gcm_translation/English__Spanish \\
        --top_k 20 --n_pairs 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lang_probing_src.config import MODEL_ID, SAE_ID, TRACER_KWARGS, OUTPUTS_DIR
from lang_probing_src.utils import setup_model, get_device_info

from flores_pairs import sample_pairs, get_shots, make_prompt
from gcm_core import tokenize_pair, sum_response_logprobs


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _acp_score(
    model, tokenizer,
    prompt: str, response: str,
    layer: int, head: int, head_dim: int,
    z_cf_at_layer: torch.Tensor,     # [d_model] o_proj.input value to splice in (head slice only)
    W_o: torch.Tensor,                # [d_model, d_model]
    device, max_response_tokens=None,
) -> float:
    """
    Single forward pass with one (layer, head) patched at the prompt's
    last_src_idx, returning sum_response_logprobs.

    Patch mechanic mirrors gcm_attribute_heads:
      * Read o_proj.input (= post-attention concat of per-q-head outputs).
      * Clone it and replace the (head's) slice at last_src_idx with the
        slice from z_cf_at_layer.
      * Compute the equivalent delta in o_proj.output (= (patched - clean)
        @ W_o.T) and add it to o_proj.output.
    """
    layers = model.model.layers
    s, e = head * head_dim, (head + 1) * head_dim
    pr = tokenize_pair(tokenizer, prompt, response, device, max_response_tokens)

    with model.trace(pr.input_ids, **TRACER_KWARGS), torch.no_grad():
        o_in_clean = layers[layer].self_attn.o_proj.input         # [B, S, d_model]
        o_in_patched = o_in_clean.clone()
        # Replace just the head's slice at last_src_idx
        o_in_patched[:, pr.last_src_idx, s:e] = z_cf_at_layer[s:e].unsqueeze(0).to(o_in_clean.dtype)
        o_out_clean = o_in_clean @ W_o.T
        o_out_patched = o_in_patched @ W_o.T
        layers[layer].self_attn.o_proj.output[:] = (
            layers[layer].self_attn.o_proj.output + (o_out_patched - o_out_clean)
        )
        m_proxy = sum_response_logprobs(
            model.lm_head.output, pr.input_ids, pr.response_start
        ).save()

    return float(m_proxy.item())


def acp_single_head_pair(
    model, tokenizer,
    prompt_orig: str, response_orig: str, response_cf: str,
    z_cf_at_layer: torch.Tensor,
    layer: int, head: int, head_dim: int,
    W_o: torch.Tensor, device, max_response_tokens=None,
) -> dict:
    """Run both (m_orig, m_cf) scoring passes with the same single-head patch."""
    m_orig_p = _acp_score(
        model, tokenizer, prompt_orig, response_orig,
        layer, head, head_dim, z_cf_at_layer, W_o, device, max_response_tokens,
    )
    m_cf_p = _acp_score(
        model, tokenizer, prompt_orig, response_cf,
        layer, head, head_dim, z_cf_at_layer, W_o, device, max_response_tokens,
    )
    return {"m_orig_patched": m_orig_p, "m_cf_patched": m_cf_p}


def cache_o_proj_inputs(model, tokenizer, prompt: str, response: str, device, max_response_tokens=None):
    """Cache o_proj.input at the last_src_idx for every layer."""
    pr = tokenize_pair(tokenizer, prompt, response, device, max_response_tokens)
    layers = model.model.layers
    n_layers = model.model.config.num_hidden_layers
    proxies = []
    with model.trace(pr.input_ids, **TRACER_KWARGS), torch.no_grad():
        for i in range(n_layers):
            proxies.append(
                layers[i].self_attn.o_proj.input[:, pr.last_src_idx, :].clone().save()
            )
    z = torch.stack([p.detach().squeeze(0) for p in proxies])  # [n_layers, d_model]
    return z, pr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--direction_dir", required=True,
                   help="e.g. outputs/gcm_translation/English__Spanish")
    p.add_argument("--src_lang", required=True)
    p.add_argument("--tgt_lang", required=True)
    p.add_argument("--n_pairs", type=int, default=10, help="Number of pairs to sample for ACP check")
    p.add_argument("--top_k", type=int, default=20, help="Number of top-ATP heads to verify with ACP")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="dev")
    p.add_argument("--max_response_tokens", type=int, default=128)
    args = p.parse_args()

    direction_dir = Path(args.direction_dir)
    if not direction_dir.exists():
        raise SystemExit(f"Direction dir not found: {direction_dir}")

    # --- Load ATP results ---
    heads_ie_path = direction_dir / "heads_ie.pt"
    if not heads_ie_path.exists():
        raise SystemExit(f"heads_ie.pt not found at {heads_ie_path} — run run.py first")
    heads_ie = torch.load(heads_ie_path, map_location="cpu").float()  # [N, n_layers, n_heads]
    logger.info(f"Loaded ATP heads_ie  shape={tuple(heads_ie.shape)}")

    n_pairs_total, n_layers, n_heads = heads_ie.shape

    # --- Pick top-K heads by mean-abs ATP ---
    mean_abs = torch.nanmean(heads_ie.abs(), dim=0)  # [n_layers, n_heads]
    flat = mean_abs.flatten()
    vals, idxs = flat.topk(args.top_k)
    top_heads = [(int(i.item() // n_heads), int(i.item() % n_heads)) for i in idxs]
    logger.info(f"Top-{args.top_k} heads by mean-abs ATP IE: {top_heads}")

    # --- Sample pairs from the same FLORES seed ---
    pairs_all = sample_pairs(args.src_lang, args.tgt_lang, n_pairs_total, seed=args.seed, split=args.split)
    rng = random.Random(args.seed + 1)  # different seed for pair-subset selection
    pair_indices = sorted(rng.sample(range(min(len(pairs_all), n_pairs_total)), min(args.n_pairs, n_pairs_total)))
    pairs_sample = [pairs_all[i] for i in pair_indices]
    logger.info(f"Sampling {len(pairs_sample)} pairs at indices {pair_indices}")

    # --- Setup model ---
    logger.info("Loading model...")
    model, _, _, tokenizer = setup_model(MODEL_ID, SAE_ID)
    device, _ = get_device_info()
    layers = model.model.layers
    cfg = model.model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    shots = get_shots(args.src_lang, args.tgt_lang, split=args.split)

    # --- Pre-cache W_o per layer (touched only for the top-K heads' layers) ---
    layers_needed = sorted({l for l, _ in top_heads})
    W_o_cache = {l: layers[l].self_attn.o_proj.weight.detach() for l in layers_needed}

    # --- Per-pair: cache z_orig + z_cf, run ACP on each top-K head ---
    rows = []
    t_start = time.time()
    for pi, (orig_pair_idx, pair) in enumerate(zip(pair_indices, pairs_sample)):
        prompt_orig = make_prompt(args.src_lang, args.tgt_lang, shots, pair.src_orig)
        prompt_cf   = make_prompt(args.src_lang, args.tgt_lang, shots, pair.src_cf)
        response_orig = pair.tgt_orig
        response_cf   = pair.tgt_cf

        # Cache z_orig at o_proj.input for all layers we care about
        z_orig, pr_or = cache_o_proj_inputs(model, tokenizer, prompt_orig, response_orig, device, args.max_response_tokens)
        z_cf, pr_cf = cache_o_proj_inputs(model, tokenizer, prompt_cf, response_cf, device, args.max_response_tokens)

        # Clean (unpatched) metrics for the M baseline
        with model.trace(pr_or.input_ids, **TRACER_KWARGS), torch.no_grad():
            m_orig_clean_proxy = sum_response_logprobs(
                model.lm_head.output, pr_or.input_ids, pr_or.response_start
            ).save()
        # Clean m_cf scored under prompt_orig
        pr_or_rc = tokenize_pair(tokenizer, prompt_orig, response_cf, device, args.max_response_tokens)
        with model.trace(pr_or_rc.input_ids, **TRACER_KWARGS), torch.no_grad():
            m_cf_clean_proxy = sum_response_logprobs(
                model.lm_head.output, pr_or_rc.input_ids, pr_or_rc.response_start
            ).save()
        m_orig_clean = float(m_orig_clean_proxy.item())
        m_cf_clean = float(m_cf_clean_proxy.item())
        M_clean = m_cf_clean - m_orig_clean

        for (layer, head) in top_heads:
            atp_ie = float(heads_ie[orig_pair_idx, layer, head].item())
            try:
                out = acp_single_head_pair(
                    model, tokenizer,
                    prompt_orig, response_orig, response_cf,
                    z_cf_at_layer=z_cf[layer],
                    layer=layer, head=head, head_dim=head_dim,
                    W_o=W_o_cache[layer],
                    device=device,
                    max_response_tokens=args.max_response_tokens,
                )
                M_patched = out["m_orig_patched"] - out["m_cf_patched"]
                acp_ie = M_patched - M_clean
                rows.append({
                    "pair_index_in_full_run": orig_pair_idx,
                    "pair_id": pair.pair_id,
                    "layer": layer,
                    "head": head,
                    "atp_ie": atp_ie,
                    "acp_ie": acp_ie,
                    "M_clean": M_clean,
                    "M_patched": M_patched,
                })
            except Exception as e:
                logger.warning(f"ACP failed for pair={pair.pair_id} (L{layer}H{head}): {type(e).__name__}: {e}")

        elapsed = time.time() - t_start
        eta = elapsed / (pi + 1) * (len(pairs_sample) - pi - 1)
        logger.info(f"  [{pi+1}/{len(pairs_sample)}] elapsed {elapsed:.1f}s  ETA {eta/60:.1f}min")

    # --- Compute correlation ---
    if not rows:
        logger.error("No ACP rows produced — bailing")
        return

    atp = torch.tensor([r["atp_ie"] for r in rows])
    acp = torch.tensor([r["acp_ie"] for r in rows])
    # Pearson r
    atp_c = atp - atp.mean()
    acp_c = acp - acp.mean()
    pearson_r = float(
        (atp_c * acp_c).sum() / (atp_c.norm() * acp_c.norm() + 1e-12)
    )
    spearman_r = None
    try:
        from scipy.stats import spearmanr
        spearman_r = float(spearmanr(atp.numpy(), acp.numpy()).correlation)
    except Exception:
        pass

    # --- Save ---
    out_path = direction_dir / "acp_faithfulness.json"
    summary = {
        "direction": direction_dir.name,
        "top_k": args.top_k,
        "n_pairs": len(pairs_sample),
        "n_rows": len(rows),
        "pearson_r_atp_vs_acp": pearson_r,
        "spearman_r_atp_vs_acp": spearman_r,
        "atp_mean": float(atp.mean().item()),
        "acp_mean": float(acp.mean().item()),
        "atp_std": float(atp.std().item()),
        "acp_std": float(acp.std().item()),
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved {out_path}  pearson_r={pearson_r:.3f}")

    # --- Plot ---
    try:
        import matplotlib.pyplot as plt
        # Experiment-local img dir (keeps the experiment self-contained)
        img_dir = Path(__file__).resolve().parent / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(atp.numpy(), acp.numpy(), alpha=0.5, s=20)
        lim = max(atp.abs().max().item(), acp.abs().max().item()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4, label="y=x")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("ATP IE  (gradient-based estimate)")
        ax.set_ylabel("ACP IE  (actual interventional measurement)")
        ax.set_title(f"{direction_dir.name}: ATP vs ACP  (n={len(rows)}, r={pearson_r:.3f})")
        ax.legend()
        plt.tight_layout()
        out_png = img_dir / f"{direction_dir.name}_acp_vs_atp.png"
        plt.savefig(out_png, dpi=120)
        plt.close(fig)
        logger.info(f"Saved {out_png}")
    except Exception as e:
        logger.warning(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
