"""
Head selection from GCM translation rankings, aggregated per target language.

The GCM experiment saves per-(source, target) raw indirect-effect tensors at
``outputs/gcm_translation/<Src>__<Tgt>/heads_ie.pt`` (shape [N_pairs, n_layers,
n_heads], dtype float32 with NaN sentinels on failed pairs).

To answer "what heads matter when L is the target?" we aggregate the *full*
[n_layers, n_heads] importance map across all ``*__L`` source directions:

  1. Per dir: compute nanmean of |IE| over pairs → [n_layers, n_heads].
  2. Stack across dirs → [n_dirs, n_layers, n_heads].
  3. Mean across dirs → aggregated importance map.

This is more principled than averaging only entries that appear in each dir's
pre-filtered top-K (which biases toward heads with single-direction outliers).

Same-language source dirs (e.g. ``French__French``) are excluded by default
because they represent a degenerate identity / paraphrase task.

Also provides a stratified random control: for each top head, sample one
random head from the same layer (excluding any top-k head at that layer).
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import torch


LANG_KEY_TO_NAME = {
    "ara": "Arabic", "eng": "English", "fra": "French", "deu": "German",
    "heb": "Hebrew", "hin": "Hindi", "spa": "Spanish", "tur": "Turkish",
}
LANG_NAME_TO_KEY = {v: k for k, v in LANG_KEY_TO_NAME.items()}


def aggregate_target_heads(
    gcm_dir: Path,
    target_lang_name: str,
    top_k: int = 20,
    exclude_self_source: bool = True,
) -> Dict:
    """
    Aggregate per-head IE across all *__<target_lang_name> GCM directories.

    Returns dict with:
      - top: list of {layer, head, mean_abs_ie_agg, mean_signed_ie_agg} sorted desc by abs.
      - abs_map: torch.Tensor [n_layers, n_heads] of aggregated mean_abs_ie.
      - signed_map: torch.Tensor [n_layers, n_heads] of aggregated mean_signed_ie.
      - source_dirs: list of dir names contributing to the aggregation.
    """
    gcm_dir = Path(gcm_dir)

    abs_maps = []
    signed_maps = []
    source_dirs = []

    for d in sorted(gcm_dir.glob(f"*__{target_lang_name}")):
        src_name = d.name.split("__")[0]
        if exclude_self_source and src_name == target_lang_name:
            continue
        ie_path = d / "heads_ie.pt"
        if not ie_path.exists():
            continue
        ie = torch.load(ie_path, map_location="cpu", weights_only=True)  # [N, L, H]
        # nanmean over pairs; treat failed pairs (NaN) as missing
        abs_map = torch.nanmean(ie.abs(), dim=0)
        signed_map = torch.nanmean(ie, dim=0)
        abs_maps.append(abs_map)
        signed_maps.append(signed_map)
        source_dirs.append(d.name)

    if not abs_maps:
        raise FileNotFoundError(
            f"No usable *__{target_lang_name} directories found in {gcm_dir} "
            f"(exclude_self_source={exclude_self_source})."
        )

    # Average across source directions (each dir already mean over its pairs)
    abs_agg = torch.stack(abs_maps).mean(dim=0)        # [L, H]
    signed_agg = torch.stack(signed_maps).mean(dim=0)  # [L, H]

    n_layers, n_heads = abs_agg.shape
    flat = abs_agg.flatten()
    vals, idxs = flat.topk(min(top_k, flat.numel()))
    top = []
    for v, idx in zip(vals.tolist(), idxs.tolist()):
        layer = idx // n_heads
        head = idx % n_heads
        top.append({
            "layer": int(layer),
            "head": int(head),
            "mean_abs_ie_agg": float(v),
            "mean_signed_ie_agg": float(signed_agg[layer, head].item()),
        })

    return {
        "top": top,
        "abs_map": abs_agg,
        "signed_map": signed_agg,
        "source_dirs": source_dirs,
    }


def sample_stratified_controls(
    top_heads: List[Dict],
    n_heads_per_layer: int,
    seed: int = 42,
) -> List[Dict]:
    """
    For each top head, sample one random head from the same layer, excluding
    every (layer, head) that appears in the top set (across all layers).

    Returned list aligns 1:1 with `top_heads` by position; same layer as the
    corresponding top head, different head index.
    """
    rng = random.Random(seed)
    top_set = {(h["layer"], h["head"]) for h in top_heads}
    controls = []
    for h in top_heads:
        layer = h["layer"]
        candidates = [
            j for j in range(n_heads_per_layer)
            if (layer, j) not in top_set
        ]
        if not candidates:
            raise ValueError(
                f"No control candidates at layer {layer}: every head at this "
                f"layer is in the top-k set. Reduce top_k or check head count."
            )
        ctrl_head = rng.choice(candidates)
        controls.append({"layer": layer, "head": ctrl_head})
    return controls


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gcm_dir", default="/projectnb/mcnet/jbrin/lang-probing/outputs/gcm_translation")
    p.add_argument("--lang_key", required=True, choices=list(LANG_KEY_TO_NAME))
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--include_self_source", action="store_true",
                   help="Include same-language source dir (e.g. French__French)")
    args = p.parse_args()

    target_name = LANG_KEY_TO_NAME[args.lang_key]
    res = aggregate_target_heads(
        Path(args.gcm_dir), target_name, args.top_k,
        exclude_self_source=not args.include_self_source,
    )
    print(f"=== Aggregated over {len(res['source_dirs'])} dirs: {res['source_dirs']} ===")
    print(f"=== Top {len(res['top'])} heads for target={target_name} ===")
    for i, h in enumerate(res["top"]):
        print(f"  rank={i+1:2d}  L{h['layer']:2d}.H{h['head']:2d}  "
              f"abs={h['mean_abs_ie_agg']:.4f}  signed={h['mean_signed_ie_agg']:+.4f}")
    ctrls = sample_stratified_controls(res["top"], n_heads_per_layer=32)
    print("\n=== Stratified controls ===")
    for i, (top, ctrl) in enumerate(zip(res["top"], ctrls)):
        print(f"  rank={i+1:2d}  top=L{top['layer']:2d}.H{top['head']:2d}  "
              f"ctrl=L{ctrl['layer']:2d}.H{ctrl['head']:2d}")
