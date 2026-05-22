"""
Head ablation on Multi-BLIMP minimal pairs.

For a given target language L:
  1. Aggregate top-K heads from GCM *__L directories (via heads.py).
  2. Sample stratified random control heads (one per top head, same layer).
  3. Compute per-(layer, head) mean of o_proj.input over Multi-BLIMP-L pairs
     (single reference pass, used as the replacement value for mean ablation).
  4. Run scoring conditions and record Δ = logp(orig) - logp(cf) at cf_pos
     for each condition.

Conditions per language (--mode full):
  * baseline (no ablation)
  * 20 per-head k=1 mean-ablations of top heads
  * 20 per-head k=1 mean-ablations of stratified controls
  * 1 collective k=5 mean-ablation of top-5
  * 1 collective k=5 mean-ablation of control-5
  * (1 + n_ctrl_seeds extra) collective k=5 mean of randomly-resampled
    same-layer controls (lets us estimate ctrl variance across random
    samples — needed to interpret single-sample ctrl deviations like
    Hindi's first-pass +0.23)
  * 1 collective mean-ablation of top heads with mean_signed_ie > 0
  * 1 collective mean-ablation of top heads with mean_signed_ie < 0
  * 1 matched-size random control for POS-IE collective (same n)
  * 1 matched-size random control for NEG-IE collective (same n)
  * 1 collective zero-ablation of top-5 (mean-vs-zero spot check)

Smoke mode (--mode smoke):
  * baseline + 5 per-head + 5 per-control + top-5 collective. Fast pipeline test.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# repo imports
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments" / "counterfactual_attribution"))

from lang_probing_src.config import MODEL_ID, TRACER_KWARGS
from lang_probing_src.utils import setup_model, get_device_info
from attribute_multilingual import prepare_pair

# local
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heads import LANG_KEY_TO_NAME, aggregate_target_heads, sample_stratified_controls


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def gpu_capability_guard():
    """Refuse to run on sm_120+ (Blackwell) — our PyTorch lacks those kernels."""
    if not torch.cuda.is_available():
        return
    cap = torch.cuda.get_device_capability(0)
    cap_float = cap[0] + 0.1 * cap[1]
    if cap_float >= 10.0:
        logger.error(
            f"Refusing to run: GPU compute capability {cap_float:.1f} "
            f"(host={os.environ.get('HOSTNAME', '?')}) is not supported "
            f"by this conda env's PyTorch (no sm_{cap[0]}{cap[1]} kernels)."
        )
        sys.exit(2)
    logger.info(f"GPU compute capability: {cap_float:.1f}")


def load_pairs(pairs_path: Path, tokenizer, max_pairs: int, seed: int = 42):
    """Load Multi-BLIMP pairs, prepare for scoring, cap at max_pairs."""
    with open(pairs_path) as f:
        raw = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(raw)
    pairs = []
    n_skipped = 0
    for p in raw:
        if not prepare_pair(p, tokenizer):
            n_skipped += 1
            continue
        pairs.append(p)
        if len(pairs) >= max_pairs:
            break
    logger.info(f"Loaded {len(pairs)} pairs (skipped {n_skipped} degenerate)")
    return pairs


def compute_mean_activations(model, pairs, n_layers, n_heads, head_dim):
    """
    Compute per-(layer, head) mean of o_proj.input over all token positions
    across reference pairs. Position-averaged means.

    Returns: torch.Tensor [n_layers, n_heads, head_dim] on CPU, float32.
    """
    sums = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float32)
    n_positions = 0
    layers = model.model.layers

    for pi, p in enumerate(pairs):
        input_ids = torch.tensor([p["_input_ids"]], device=_device())
        # nnsight 0.5 caveat: pre-init the list OUTSIDE the trace; sys.settrace
        # fires ExitTracingException mid-block and leaves names defined inside
        # unbound. See gcm_core.py:394-403 for the canonical workaround.
        saves = []
        with model.trace(input_ids, **TRACER_KWARGS), torch.no_grad():
            for L in range(n_layers):
                saves.append(layers[L].self_attn.o_proj.input.cpu().save())
        # After trace: extract and accumulate
        for L in range(n_layers):
            o_in = _unwrap(saves[L])  # [1, S, d_model]
            S = o_in.shape[1]
            o_in_h = o_in.view(1, S, n_heads, head_dim).float()
            sums[L] += o_in_h.sum(dim=(0, 1))
            if L == 0:
                n_positions += S
        if (pi + 1) % 25 == 0 or pi == len(pairs) - 1:
            logger.info(f"  mean-acts pass: {pi+1}/{len(pairs)} pairs ({n_positions} positions)")
        del saves
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    means = sums / max(n_positions, 1)
    return means  # [L, H, D]


def _unwrap(saved):
    """Extract value from nnsight Save proxy if needed."""
    return getattr(saved, "value", saved)


def _device():
    d, _ = get_device_info()
    return d


def score_pair(
    model,
    pair,
    ablations,
    mean_acts,
    n_heads,
    head_dim,
):
    """
    Score a single pair with the given ablations installed.

    Args:
        ablations: list of (layer, head, mode) where mode in {"mean", "zero"}.
                   Empty list = baseline.
        mean_acts: [n_layers, n_heads, head_dim] tensor (CPU). Required if any
                   ablation uses mode="mean".

    Returns: float, Δ = logp(orig) - logp(cf) at cf_pos.
    """
    input_ids = torch.tensor([pair["_input_ids"]], device=_device())
    cf_pos = pair["_cf_pos"]
    orig_id = pair["_last_orig_id"]
    cf_id = pair["_last_cf_id"]
    layers = model.model.layers

    with model.trace(input_ids, **TRACER_KWARGS), torch.no_grad():
        if ablations:
            by_layer: dict = {}
            for (L, h, mode) in ablations:
                by_layer.setdefault(L, []).append((h, mode))

            # nnsight requires Envoy hooks be touched in causal order
            # (ascending layer index) — otherwise OutOfOrderError.
            for L, ops in sorted(by_layer.items()):
                self_attn = layers[L].self_attn
                o_in = self_attn.o_proj.input  # [B, S, d_model]
                W_o = self_attn.o_proj.weight  # [d_model, d_model]
                # Per-head delta accumulation. Earlier versions cloned
                # o_in.view(B,S,n_heads,head_dim) — a [B,S,4096] bf16 alloc
                # per layer per pair per condition. We instead compute the
                # delta one head at a time via column-sliced W_o, which
                # avoids the full clone and reduces peak memory enough that
                # heb/hin at n=400 no longer OOM on 80GB.
                for (h, mode) in ops:
                    h_start = h * head_dim
                    h_end = h_start + head_dim
                    head_in = o_in[:, :, h_start:h_end]   # [B, S, head_dim]
                    W_h = W_o[:, h_start:h_end]           # [d_model, head_dim]
                    if mode == "zero":
                        delta_in = -head_in
                    elif mode == "mean":
                        mv = mean_acts[L, h].to(o_in.dtype)
                        delta_in = mv - head_in           # broadcasts [head_dim] over [B,S]
                    else:
                        raise ValueError(f"unknown ablation mode {mode!r}")
                    # Project the per-head delta through W_o's column slice.
                    # Cast to o_in.dtype rather than W_o.dtype to avoid
                    # querying a Parameter proxy under the trace
                    # (canonical pattern: gcm_core.py:434-438).
                    delta_post = torch.matmul(delta_in.to(o_in.dtype), W_h.T)
                    self_attn.o_proj.output[:] = self_attn.o_proj.output + delta_post

        logits = model.lm_head.output  # [1, S, vocab]
        cf_logits = logits[:, cf_pos, :]
        log_probs = F.log_softmax(cf_logits.float(), dim=-1)
        delta = (log_probs[:, orig_id] - log_probs[:, cf_id]).cpu().save()

    val = float(_unwrap(delta).item())
    del delta  # release the nnsight save proxy explicitly
    return val


def run_condition(model, pairs, name, ablations, mean_acts, n_heads, head_dim,
                  baseline_deltas=None, cleanup_every: int = 50):
    """Score all pairs under one ablation condition; return mean+stats.

    If baseline_deltas is provided, also compute disruption metrics
    (mean|diff|, frac_decreased, frac_sign_flip) per-pair relative to baseline.

    cleanup_every: call gc.collect() + torch.cuda.empty_cache() every N pairs
    inside the per-pair score loop. nnsight 0.5 holds onto trace residue
    between calls; without periodic cleanup the per-condition memory grows
    enough that long sweeps (heb/hin, n=400) OOM mid-loop on 80GB.
    """
    t0 = time.time()
    deltas = []
    for i, p in enumerate(pairs):
        deltas.append(score_pair(model, p, ablations, mean_acts, n_heads, head_dim))
        if (i + 1) % cleanup_every == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elapsed = time.time() - t0
    t = torch.tensor(deltas)
    out = {
        "name": name,
        "ablations": [(L, h, m) for (L, h, m) in ablations],
        "n_pairs": len(deltas),
        "mean_delta": float(t.mean().item()),
        "std_delta": float(t.std().item()) if len(deltas) > 1 else 0.0,
        "median_delta": float(t.median().item()),
        "frac_pos": float((t > 0).float().mean().item()),
        "elapsed_s": elapsed,
        "deltas": deltas,
    }
    if baseline_deltas is not None:
        bt = torch.tensor(baseline_deltas)
        diff = t - bt  # signed: negative = ablation hurt grammaticality
        out["mean_diff_from_baseline"] = float(diff.mean().item())
        out["mean_abs_diff_from_baseline"] = float(diff.abs().mean().item())
        out["max_abs_diff"] = float(diff.abs().max().item())
        out["frac_decreased"] = float((diff < 0).float().mean().item())
        out["frac_sign_flip"] = float(((t > 0) != (bt > 0)).float().mean().item())
    return out


def build_conditions(
    mode: str,
    top_heads,
    control_heads,
    top_k_collective: int,
    n_heads_per_layer: int,
    n_ctrl_seeds: int = 0,
    base_seed: int = 42,
):
    """
    Build the list of conditions to run.

    Each condition is a dict with: name, ablations [(L,h,mode), ...].

    Additional controls beyond the original layout:
      * matched-size random controls for POS-IE and NEG-IE collective
        (same layer-stratification, same head count) — addresses the
        "5 vs 15 heads" group-size confound in the sign-split.
      * `n_ctrl_seeds` extra k=5 collective controls with different
        random seeds — gives a distribution of single-sample ctrl effects,
        useful when a single ctrl deviates (e.g. Hindi's +0.23 on the
        first sample).
    """
    conds = []

    conds.append({"name": "baseline", "ablations": []})

    for i, h in enumerate(top_heads):
        conds.append({
            "name": f"top_rank{i+1:02d}_L{h['layer']:02d}H{h['head']:02d}_mean",
            "ablations": [(h["layer"], h["head"], "mean")],
        })
    for i, c in enumerate(control_heads):
        conds.append({
            "name": f"ctrl_rank{i+1:02d}_L{c['layer']:02d}H{c['head']:02d}_mean",
            "ablations": [(c["layer"], c["head"], "mean")],
        })

    if mode == "smoke":
        return conds

    # k=5 collective: top + matched ctrl (existing)
    top_k_subset = top_heads[:top_k_collective]
    ctrl_k_subset = control_heads[:top_k_collective]
    conds.append({
        "name": f"top_collective_k{top_k_collective}_mean",
        "ablations": [(h["layer"], h["head"], "mean") for h in top_k_subset],
    })
    conds.append({
        "name": f"ctrl_collective_k{top_k_collective}_mean",
        "ablations": [(h["layer"], h["head"], "mean") for h in ctrl_k_subset],
    })

    # Extra randomly-resampled k=5 controls (for ctrl variance estimation).
    # Each seed produces a different stratified sample over the same layers.
    for s in range(n_ctrl_seeds):
        alt_ctrl = sample_stratified_controls(
            top_heads[:top_k_collective],
            n_heads_per_layer=n_heads_per_layer,
            seed=base_seed + 100 + s,
        )
        conds.append({
            "name": f"ctrl_collective_k{top_k_collective}_seed{s:02d}_mean",
            "ablations": [(h["layer"], h["head"], "mean") for h in alt_ctrl],
        })

    # Sign-split collective ablations
    pos_heads = [h for h in top_heads if h["mean_signed_ie_agg"] > 0]
    neg_heads = [h for h in top_heads if h["mean_signed_ie_agg"] < 0]
    if pos_heads:
        conds.append({
            "name": f"top_collective_signedPOS_n{len(pos_heads)}_mean",
            "ablations": [(h["layer"], h["head"], "mean") for h in pos_heads],
        })
        # Matched-size random control over the POS-IE layers
        pos_ctrl = sample_stratified_controls(
            pos_heads, n_heads_per_layer=n_heads_per_layer, seed=base_seed + 1,
        )
        conds.append({
            "name": f"ctrl_collective_matched_POS_n{len(pos_heads)}_mean",
            "ablations": [(h["layer"], h["head"], "mean") for h in pos_ctrl],
        })
    if neg_heads:
        conds.append({
            "name": f"top_collective_signedNEG_n{len(neg_heads)}_mean",
            "ablations": [(h["layer"], h["head"], "mean") for h in neg_heads],
        })
        # Matched-size random control over the NEG-IE layers
        neg_ctrl = sample_stratified_controls(
            neg_heads, n_heads_per_layer=n_heads_per_layer, seed=base_seed + 2,
        )
        conds.append({
            "name": f"ctrl_collective_matched_NEG_n{len(neg_heads)}_mean",
            "ablations": [(h["layer"], h["head"], "mean") for h in neg_ctrl],
        })

    # Zero ablation spot check on top-k_collective (mean-vs-zero comparison)
    conds.append({
        "name": f"top_collective_k{top_k_collective}_zero",
        "ablations": [(h["layer"], h["head"], "zero") for h in top_k_subset],
    })

    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang_key", required=True, choices=list(LANG_KEY_TO_NAME))
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--top_k_collective", type=int, default=5)
    ap.add_argument("--max_pairs", type=int, default=400)
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_ctrl_seeds", type=int, default=0,
                    help="Extra k_collective random controls with distinct seeds "
                         "(estimates ctrl variance; useful for langs where the "
                         "first ctrl deviates from near-zero).")
    ap.add_argument("--gcm_dir", default="/projectnb/mcnet/jbrin/lang-probing/outputs/gcm_translation")
    ap.add_argument("--pairs_dir", default="/projectnb/mcnet/jbrin/lang-probing/data/multilingual_pairs")
    ap.add_argument("--output_dir", default="/projectnb/mcnet/jbrin/lang-probing/outputs/head_ablation_multiblimp")
    args = ap.parse_args()

    target_name = LANG_KEY_TO_NAME[args.lang_key]
    logger.info(f"=== head_ablation_multiblimp: target={target_name} ({args.lang_key}) mode={args.mode} ===")
    logger.info(f"    top_k={args.top_k} top_k_collective={args.top_k_collective} max_pairs={args.max_pairs}")

    gpu_capability_guard()

    out_dir = Path(args.output_dir) / args.lang_key
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Head selection ---
    res = aggregate_target_heads(Path(args.gcm_dir), target_name, top_k=args.top_k)
    top_heads = res["top"]
    logger.info(f"Aggregated over {len(res['source_dirs'])} dirs: {res['source_dirs']}")
    logger.info(f"Top-{args.top_k} heads:")
    for i, h in enumerate(top_heads):
        logger.info(
            f"  rank={i+1:2d}  L{h['layer']:2d}.H{h['head']:2d}  "
            f"abs={h['mean_abs_ie_agg']:.4f}  signed={h['mean_signed_ie_agg']:+.4f}"
        )

    # Need to know n_heads_per_layer; defer until model is loaded
    # (will sample controls after model load)

    # --- 2. Load model ---
    logger.info(f"Loading model {MODEL_ID}...")
    t_load = time.time()
    model, _, _, tokenizer = setup_model(MODEL_ID, sae_id=None)
    device, _ = get_device_info()
    logger.info(f"  loaded in {time.time() - t_load:.1f}s on {device}")

    for p in model.parameters():
        p.requires_grad_(False)

    cfg = model.model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads
    logger.info(f"  n_layers={n_layers}  n_heads={n_heads}  head_dim={head_dim}")

    # Controls
    control_heads = sample_stratified_controls(top_heads, n_heads_per_layer=n_heads, seed=args.seed)
    logger.info("Stratified controls:")
    for i, (top, ctrl) in enumerate(zip(top_heads, control_heads)):
        logger.info(f"  rank={i+1:2d}  top=L{top['layer']:2d}.H{top['head']:2d}  "
                    f"ctrl=L{ctrl['layer']:2d}.H{ctrl['head']:2d}")

    # --- 3. Load pairs ---
    pairs_path = Path(args.pairs_dir) / f"{args.lang_key}.json"
    if not pairs_path.exists():
        logger.error(f"Pairs file not found: {pairs_path}. "
                     f"Run build_multiblimp_pairs.py with {args.lang_key} in LANG_SPECS first.")
        sys.exit(3)
    pairs = load_pairs(pairs_path, tokenizer, args.max_pairs, seed=args.seed)
    if not pairs:
        logger.error("No usable pairs after prepare_pair; aborting.")
        sys.exit(4)

    # --- 4. Mean activations ---
    logger.info("Computing per-head mean activations over reference pairs...")
    t0 = time.time()
    mean_acts = compute_mean_activations(model, pairs, n_layers, n_heads, head_dim)
    logger.info(f"  mean activations done in {time.time() - t0:.1f}s; shape={tuple(mean_acts.shape)}")
    # Pre-load to GPU once so per-head ablation hooks don't allocate a new
    # [head_dim] tensor for every pair × every condition (memory churn).
    mean_acts = mean_acts.to(device)

    # --- 5. Conditions ---
    conditions = build_conditions(
        args.mode, top_heads, control_heads, args.top_k_collective,
        n_heads_per_layer=n_heads, n_ctrl_seeds=args.n_ctrl_seeds,
        base_seed=args.seed,
    )
    logger.info(f"Running {len(conditions)} conditions...")

    results = []
    baseline_deltas = None
    t_all = time.time()
    for ci, c in enumerate(conditions):
        t0 = time.time()
        r = run_condition(model, pairs, c["name"], c["ablations"], mean_acts,
                          n_heads, head_dim, baseline_deltas=baseline_deltas)
        results.append(r)
        if c["name"] == "baseline":
            baseline_deltas = r["deltas"]
        # Free per-condition GPU residue before next condition.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elapsed = time.time() - t0
        log_extra = ""
        if "mean_abs_diff_from_baseline" in r:
            log_extra = (f"  |Δ|={r['mean_abs_diff_from_baseline']:.4f}  "
                         f"frac_dec={r['frac_decreased']:.2f}")
        logger.info(
            f"  [{ci+1:3d}/{len(conditions)}] {c['name']:50s}  "
            f"mean_Δ={r['mean_delta']:+.4f}  frac_pos={r['frac_pos']:.2f}{log_extra}  "
            f"{elapsed:.1f}s"
        )
    total_elapsed = time.time() - t_all

    # --- 6. Save ---
    # Per-pair metadata aligned to the pair order used for scoring
    pair_meta = [
        {"id": p["id"], "phenomenon": p.get("phenomenon"),
         "concept": p.get("concept"), "concept_value_orig": p.get("concept_value_orig"),
         "concept_value_cf": p.get("concept_value_cf")}
        for p in pairs
    ]

    summary = {
        "lang_key": args.lang_key,
        "target_lang_name": target_name,
        "mode": args.mode,
        "top_k": args.top_k,
        "top_k_collective": args.top_k_collective,
        "max_pairs": args.max_pairs,
        "n_pairs_used": len(pairs),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "model_id": MODEL_ID,
        "gcm_source_dirs": res["source_dirs"],
        "top_heads": top_heads,
        "control_heads": control_heads,
        "pair_meta": pair_meta,
        "total_elapsed_s": total_elapsed,
    }

    # Don't pickle the deltas tensor in the JSON summary; save separately
    light_results = [{k: v for k, v in r.items() if k != "deltas"} for r in results]
    deltas_by_cond = {r["name"]: r["deltas"] for r in results}

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"meta": summary, "conditions": light_results}, f, indent=2)
    with open(out_dir / "deltas.json", "w") as f:
        json.dump(deltas_by_cond, f, indent=2)
    torch.save(mean_acts, out_dir / "mean_acts.pt")

    logger.info(f"=== Done: {len(results)} conditions in {total_elapsed:.1f}s -> {out_dir} ===")

    # Quick sanity printouts
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    if baseline:
        logger.info(f"BASELINE Δ = {baseline['mean_delta']:+.4f} "
                    f"(frac_pos={baseline['frac_pos']:.2f}) -- should be >> 0")


if __name__ == "__main__":
    main()
