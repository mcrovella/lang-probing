import os
import glob
import logging
import csv
import gc
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np

import matplotlib.pyplot as plt

from lang_probing_src.config import OUTPUTS_DIR, IMG_DIR
from lang_probing_src.utils_input_output import (
    get_output_features_vector,
    get_input_features_vector,
    index_effects_files,
    get_language_pairs_and_concepts_from_index,
)

def jaccard_topk_scores(
    input_features: np.ndarray,
    output_features: np.ndarray,
    max_k: int | None = None,
    step_k: int = 1,
    score_mode: str = "raw",
):
    """Return k values and Jaccard@k scores for two feature-score vectors."""
    if input_features.shape != output_features.shape:
        raise ValueError(
            f"Shape mismatch: input {input_features.shape} vs output {output_features.shape}"
        )

    n_features = len(input_features)
    if max_k is None:
        max_k = n_features
    max_k = min(max_k, n_features)

    if score_mode == "raw":
        input_scores = input_features
        output_scores = output_features
    elif score_mode == "abs":
        input_scores = np.abs(input_features)
        output_scores = np.abs(output_features)
    else:
        raise ValueError(f"unknown score_mode={score_mode!r}; expected 'raw' or 'abs'")

    input_sorted_indices = np.argsort(input_scores)[::-1]
    output_sorted_indices = np.argsort(output_scores)[::-1]

    k_values = []
    jaccard_scores = []
    for k in range(1, max_k + 1, step_k):
        top_k_input = set(input_sorted_indices[:k])
        top_k_output = set(output_sorted_indices[:k])
        intersection = len(top_k_input.intersection(top_k_output))
        union = len(top_k_input.union(top_k_output))
        jaccard = intersection / union if union > 0 else 0.0
        k_values.append(k)
        jaccard_scores.append(jaccard)
    return k_values, jaccard_scores


def plot_jaccard_topk_similarity(
    input_features: np.ndarray,
    output_features: np.ndarray,
    max_k: int | None = None,
    step_k: int = 1,
    score_mode: str = "raw",
    title: str | None = None,
    save_path: Path | str | None = None,
):
    """
    Plot Jaccard similarity between the top-K features of two feature vectors.

    This function treats the feature vectors as scores over the same feature
    indices. For each K, it takes the indices of the K largest scores in
    input_features and output_features, computes the Jaccard similarity
    between these two index sets, and plots Jaccard(K) vs K.

    Args:
        input_features: 1D numpy array of input feature scores.
        output_features: 1D numpy array of output feature scores.
        max_k: Maximum K to consider. If None, defaults to the length
            of the feature vectors.
        step_k: Step size for K (e.g., 10 plots K = 1, 11, 21, ...).
        title: Optional plot title. If None, a default title is used.
        save_path: Optional path to save the figure. If None, the plot
            is shown instead of saved.
    """

    k_values, jaccard_scores = jaccard_topk_scores(
        input_features, output_features, max_k=max_k, step_k=step_k,
        score_mode=score_mode,
    )

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, jaccard_scores, label='Jaccard Similarity', linewidth=2)
    
    # Styling
    final_title = title if title else f"Top-K Jaccard Similarity ({score_mode}, max K={max_k})"
    plt.title(final_title, fontsize=14)
    plt.xlabel("K (Number of Top Features)", fontsize=12)
    plt.ylabel("Jaccard Similarity", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-0.05, 1.05)  # Jaccard is always between 0 and 1
    
    # 5. Save or Show
    if save_path:
        # Convert string to Path object if necessary and ensure parent dir exists
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_obj, dpi=300, bbox_inches='tight')
        plt.close() # Close to free memory
    else:
        plt.show()


def plot_aggregate_jaccard(curves: list[dict], score_mode: str, save_path: Path):
    mode_curves = [c for c in curves if c["score_mode"] == score_mode]
    if not mode_curves:
        return
    k_values = np.array(mode_curves[0]["k_values"])
    scores = np.array([c["scores"] for c in mode_curves])
    mean = scores.mean(axis=0)
    lo = np.percentile(scores, 25, axis=0)
    hi = np.percentile(scores, 75, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, mean, linewidth=2, label=f"mean over {len(mode_curves)} cells")
    plt.fill_between(k_values, lo, hi, alpha=0.22, label="IQR across cells")
    plt.xlabel("K (number of top features)")
    plt.ylabel("Jaccard similarity")
    plt.title(f"Aggregate input-output feature overlap ({score_mode} top-k)")
    plt.ylim(-0.02, 0.35)
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.legend()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def signal_plot(signal, title, xlabel, ylabel, save_path=None):
    plt.figure(figsize=(15, 6))

    # Use a thin line (linewidth) and transparency (alpha) to handle the density of 32k points
    plt.plot(signal, linewidth=0.5, alpha=0.8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3) # Adds a subtle grid for readability

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    input_features_dir = Path(OUTPUTS_DIR) / "input_features"
    effects_index = index_effects_files()
    language_pairs, concepts = get_language_pairs_and_concepts_from_index(effects_index)

    plots_dir = Path(IMG_DIR) / "input_output_overlap"
    plots_dir.mkdir(exist_ok=True)
    summary_dir = Path(OUTPUTS_DIR) / "input_output_overlap"
    summary_dir.mkdir(exist_ok=True)

    selected_ks = [10, 30, 50, 100, 200]
    summary_rows = []
    aggregate_curves = []

    target_langs = sorted({target for _, target in language_pairs})
    for target_lang in target_langs:
        for concept in concepts:
            for value in concepts[concept]:
                # if target_lang != "English" or concept != "Tense" or value != "Past":
                #     continue
                
                print(f"Getting input/output features for tg:{target_lang} {concept}={value}")
                try:
                    input_path = input_features_dir / target_lang / concept / value / "diff_vector.pt"
                    if not input_path.exists():
                        print(f"Missing input diff vector: {input_path}")
                        continue
                    input_features = get_input_features_vector(input_features_dir, target_lang, concept, value)
                    output_features_list = []

                    # print(input_features[10:])
                    # input_features = np.sort(input_features)
                    # print(input_features[10:])

                    source_langs = [src for src, tgt in language_pairs if tgt == target_lang]
                    for source_lang in source_langs:
                        pair = (source_lang, target_lang)
                        effects = torch.load(effects_index[pair], map_location="cpu", weights_only=False)
                        output_features = get_output_features_vector({pair: effects}, pair, concept, value)
                        if output_features.shape != input_features.shape:
                            print(
                                f"Skipping {pair} {concept}={value}: output shape "
                                f"{output_features.shape} != input {input_features.shape}"
                            )
                            del effects
                            gc.collect()
                            continue
                        output_features_list.append(output_features)
                        del effects
                        gc.collect()
                    if not output_features_list:
                        raise ValueError("No source-language output vectors matched this cell")
                    output_features = np.mean(output_features_list, axis=0)

                    # print(output_features[10:])
                    # output_features = np.sort(output_features)
                    # print(output_features[10:])

                    print("in/out nonzero counts:")
                    print(np.count_nonzero(input_features))
                    print(np.count_nonzero(output_features))

                    save_dir = plots_dir / f"{target_lang}_{concept}_{value}"
                    save_dir.mkdir(exist_ok=True)

                    # input_features = np.sort(input_features)
                    # output_features = np.sort(output_features)

                    # input_order = np.argsort(input_features)[::-1]
                    # output_order = np.argsort(output_features)[::-1]

                    # print("First 10 indices input_order:", input_order[:10])
                    # print("First 10 indices output_order:", output_order[:10])

                    for score_mode, filename in [
                        ("raw", "jaccard_topk.png"),
                        ("abs", "jaccard_topk_abs.png"),
                    ]:
                        k_values, scores = jaccard_topk_scores(
                            input_features,
                            output_features,
                            max_k=200,
                            step_k=1,
                            score_mode=score_mode,
                        )
                        aggregate_curves.append({
                            "target_lang": target_lang,
                            "concept": concept,
                            "value": value,
                            "score_mode": score_mode,
                            "k_values": k_values,
                            "scores": scores,
                        })
                        score_by_k = dict(zip(k_values, scores))
                        for k in selected_ks:
                            if k in score_by_k:
                                summary_rows.append({
                                    "target_lang": target_lang,
                                    "concept": concept,
                                    "value": value,
                                    "score_mode": score_mode,
                                    "k": k,
                                    "jaccard": score_by_k[k],
                                    "input_nonzero": int(np.count_nonzero(input_features)),
                                    "output_nonzero": int(np.count_nonzero(output_features)),
                                    "n_source_langs": len(source_langs),
                                })

                        plot_jaccard_topk_similarity(
                            input_features,
                            output_features,
                            max_k=200,
                            step_k=1,
                            score_mode=score_mode,
                            title=f"Jaccard top-K ({score_mode}) for {target_lang} {concept}={value}",
                            save_path=save_dir / filename,
                        )

                    # signal_plot(input_features, f"Input features for {target_lang} {concept}={value}", "Feature Index", "Value", save_path=save_dir / "signal_input.png")
                    # signal_plot(output_features, f"Output features for {target_lang} {concept}={value}", "Feature Index", "Value", save_path=save_dir / "signal_output.png")

                except Exception as e:
                    print(f"Error getting input/output features for {target_lang} {concept} {value}: {e}")

    if summary_rows:
        csv_path = summary_dir / "jaccard_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        plot_aggregate_jaccard(
            aggregate_curves, "raw", plots_dir / "aggregate_jaccard_topk_raw.png"
        )
        plot_aggregate_jaccard(
            aggregate_curves, "abs", plots_dir / "aggregate_jaccard_topk_abs.png"
        )
        print(f"Wrote aggregate summary to {csv_path}")
        for mode in ["raw", "abs"]:
            vals = [
                r["jaccard"] for r in summary_rows
                if r["score_mode"] == mode and r["k"] == 200
            ]
            if vals:
                print(f"Mean Jaccard@200 ({mode}) over {len(vals)} cells: {np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
