import os
import glob
import logging
import gc
import torch
import numpy as np
from tqdm import tqdm

from ..config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def get_output_features_vector(effects_files, language_pair, concept, value):
    e = effects_files[language_pair][concept][value]
    e = e.cpu().numpy()
    return np.mean(e, axis=0)


def get_input_features_vector(output_dir, language, concept, value):
    save_dir = output_dir / language / concept / value
    save_path = save_dir / "diff_vector.pt"

    return torch.load(save_path, weights_only=True).numpy()


def index_effects_files(out_dir=None):
    # Default to <OUTPUTS_DIR>/output_features so the path stays consistent
    # if OUTPUTS_DIR is overridden.
    if out_dir is None:
        out_dir = os.path.join(OUTPUTS_DIR, "output_features")

    # Timestamped output dirs can contain repeated language pairs, including
    # tiny pilots. Pick the latest timestamp per pair first, then load only one
    # tensor per pair. This avoids nondeterministic overwrites and large memory
    # spikes from loading superseded outputs.
    latest_by_pair = {}
    for subfolder in sorted(os.listdir(out_dir)):
        subdir = os.path.join(out_dir, subfolder)
        if not os.path.isdir(subdir):
            continue
        effects_file = glob.glob(os.path.join(subdir, "effects_*.pt"))
        if len(effects_file) == 0:
            raise ValueError(f"No effects file found for {subfolder}")
        effects_file = effects_file[0]

        # The suffix is a language pair, such as effects_English_French.pt
        source_lang, target_lang = effects_file.split("/")[-1].split("_")[1:]
        target_lang, _ = target_lang.split(".")
        language_pair = (source_lang, target_lang)
        latest_by_pair[language_pair] = effects_file

    return latest_by_pair


def load_effects_files(out_dir=None):
    latest_by_pair = index_effects_files(out_dir)

    effects_files = {}
    for language_pair, effects_file in tqdm(sorted(latest_by_pair.items())):
        logger.debug("Loading effects file for %s from %s", language_pair, effects_file)
        effects_files[language_pair] = torch.load(effects_file, weights_only=False)

    return effects_files


def get_language_pairs_and_concepts(effects_files):
    language_pairs = set()
    concepts = {}
    for language_pair in effects_files:
        language_pairs.add(language_pair)
        for concept_key in effects_files[language_pair]:
            if concept_key not in concepts:
                concepts[concept_key] = set()
            for concept_value in effects_files[language_pair][concept_key]: 
                concepts[concept_key].add(concept_value)
    
    return sorted(language_pairs), concepts


def get_language_pairs_and_concepts_from_index(effects_index):
    """Collect language pairs and concept/value keys without keeping all effects in memory."""
    language_pairs = set()
    concepts = {}
    for language_pair, path in tqdm(sorted(effects_index.items())):
        language_pairs.add(language_pair)
        effects = torch.load(path, map_location="cpu", weights_only=False)
        for concept_key in effects:
            if concept_key not in concepts:
                concepts[concept_key] = set()
            for concept_value in effects[concept_key]:
                concepts[concept_key].add(concept_value)
        del effects
        gc.collect()

    return sorted(language_pairs), concepts
