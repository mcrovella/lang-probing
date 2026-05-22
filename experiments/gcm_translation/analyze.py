"""
Aggregate per-direction GCM results across the full sweep.

Reads outputs/gcm_translation/<src>__<tgt>/{heads_ie.pt, sae_ie.pt, top_rankings.json}
and produces:
  - summary_all_directions.json
  - universal_heads.json     (heads in top-K of many directions)
  - universal_sae.json        (SAE features in top-K of many directions)
  - sae_overlap_with_grammar.json (cross-ref vs counterfactual_attribution)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


def load_direction(direction_dir: Path) -> dict:
    out = {"name": direction_dir.name}
    s = direction_dir / "summary.json"
    if s.exists():
        with open(s) as f:
            out["summary"] = json.load(f)
    t = direction_dir / "top_rankings.json"
    if t.exists():
        with open(t) as f:
            out["top"] = json.load(f)
    h = direction_dir / "heads_ie.pt"
    if h.exists():
        out["heads_ie"] = torch.load(h, map_location="cpu")
    s_ie = direction_dir / "sae_ie.pt"
    if s_ie.exists():
        out["sae_ie"] = torch.load(s_ie, map_location="cpu")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", default="outputs/gcm_translation")
    p.add_argument("--out_dir", default="outputs/gcm_translation/_aggregate")
    p.add_argument("--top_k_for_universality", type=int, default=20)
    p.add_argument("--include_same_lang", action="store_true",
                   help="Include same-language addendum dirs in universality counts. "
                        "Default is cross-language only, matching the 56-direction paper claim.")
    p.add_argument("--counterfactual_attribution_aggregate",
                   default="outputs/counterfactual_attribution/aggregated_by_concept.json",
                   help="Optional: cross-reference SAE GCM top-features vs grammar top-features.")
    args = p.parse_args()

    sweep = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    direction_dirs = []
    for d in sorted(sweep.iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        src, tgt = d.name.split("__", 1)
        if src == tgt and not args.include_same_lang:
            continue
        direction_dirs.append(d)
    print(f"Found {len(direction_dirs)} directions under {sweep}")

    summary_all = {}
    head_topk_counter: Counter = Counter()              # (layer, head) -> count
    head_signed_means = defaultdict(list)                # (layer, head) -> [signed mean ie per direction it appears]
    sae_topk_counter: Counter = Counter()                # feature_idx -> count
    sae_signed_means = defaultdict(list)

    for d in direction_dirs:
        loaded = load_direction(d)
        if "top" not in loaded:
            print(f"  skip {d.name}: no top_rankings.json")
            continue

        summary_all[loaded["name"]] = {
            "summary": loaded.get("summary", {}),
            "top_5_heads": loaded["top"].get("heads_top_k_by_mean_abs_ie", [])[:5],
            "top_10_sae": loaded["top"].get("sae_top_k_by_mean_abs_ie", [])[:10],
        }

        for entry in loaded["top"].get("heads_top_k_by_mean_abs_ie", [])[:args.top_k_for_universality]:
            key = (entry["layer"], entry["head"])
            head_topk_counter[key] += 1
            head_signed_means[key].append(entry["mean_signed_ie"])

        for entry in loaded["top"].get("sae_top_k_by_mean_abs_ie", [])[:args.top_k_for_universality]:
            key = entry["feature_idx"]
            sae_topk_counter[key] += 1
            sae_signed_means[key].append(entry["mean_signed_ie"])

    # Universal heads / SAE features (appear in top-K of >=N directions)
    universal_heads = sorted(
        [
            {
                "layer": k[0],
                "head": k[1],
                "n_directions_in_topk": v,
                "mean_signed_ie_across_directions": sum(head_signed_means[k]) / len(head_signed_means[k]),
            }
            for k, v in head_topk_counter.items()
        ],
        key=lambda x: -x["n_directions_in_topk"],
    )
    universal_sae = sorted(
        [
            {
                "feature_idx": k,
                "n_directions_in_topk": v,
                "mean_signed_ie_across_directions": sum(sae_signed_means[k]) / len(sae_signed_means[k]),
            }
            for k, v in sae_topk_counter.items()
        ],
        key=lambda x: -x["n_directions_in_topk"],
    )

    with open(out_dir / "summary_all_directions.json", "w") as f:
        json.dump(summary_all, f, indent=2)
    with open(out_dir / "universal_heads.json", "w") as f:
        json.dump(universal_heads[:50], f, indent=2)
    with open(out_dir / "universal_sae.json", "w") as f:
        json.dump(universal_sae[:50], f, indent=2)

    print(f"Top universal heads (in >= half of directions):")
    threshold = max(1, len(direction_dirs) // 2)
    for h in universal_heads[:20]:
        if h["n_directions_in_topk"] >= threshold:
            print(f"  L{h['layer']:>2} H{h['head']:>2}  in {h['n_directions_in_topk']}/{len(direction_dirs)}  signed_ie_mean={h['mean_signed_ie_across_directions']:+.4f}")

    print(f"\nTop universal SAE features (>= half of directions):")
    for s in universal_sae[:20]:
        if s["n_directions_in_topk"] >= threshold:
            print(f"  f{s['feature_idx']:>5}  in {s['n_directions_in_topk']}/{len(direction_dirs)}  signed_ie_mean={s['mean_signed_ie_across_directions']:+.4f}")

    # Cross-reference SAE GCM top-features vs counterfactual_attribution grammar top-features
    grammar_path = Path(args.counterfactual_attribution_aggregate)
    if grammar_path.exists():
        with open(grammar_path) as f:
            grammar = json.load(f)
        grammar_top_features = set()
        for concept, info in grammar.items():
            for feat in info.get("top_50_by_mean_grad_x_act", []):
                grammar_top_features.add(feat["feature_idx"])
        gcm_top_features = {s["feature_idx"] for s in universal_sae[:50]}
        overlap = grammar_top_features & gcm_top_features
        cross = {
            "n_grammar_top_features": len(grammar_top_features),
            "n_gcm_top_features": len(gcm_top_features),
            "n_overlap": len(overlap),
            "overlap_features": sorted(overlap),
        }
        with open(out_dir / "sae_overlap_with_grammar.json", "w") as f:
            json.dump(cross, f, indent=2)
        print(f"\nSAE GCM-top vs grammar-top overlap: {len(overlap)} / {len(gcm_top_features)} features in both sets")
    else:
        print(f"\nNote: {grammar_path} not found, skipping grammar cross-reference")


if __name__ == "__main__":
    main()
