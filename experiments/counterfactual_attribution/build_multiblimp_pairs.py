"""
Build multilingual minimal-pair JSONs from jumelet/multiblimp.

Outputs: data/multilingual_pairs/{fra,spa,tur,ara}.json with the extended
grammatical_pairs schema (prefix, original_token, counterfactual_token,
concept, concept_value_orig, concept_value_cf, lang_code, source,
phenomenon).

Also writes data_snapshot.json with per-cell pair counts.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "/projectnb/mcnet/jbrin/.cache/huggingface")
from datasets import load_dataset

PHENOMENON_TO_CONCEPT = {
    "SV-#": "Number",
    "SV-P": "Person",
    "SV-G": "Gender",
    "SP-#": "Number",
    "SP-G": "Gender",
}

NUMBER_CODE_TO_VALUE = {
    "SG": "Sing", "PL": "Plur", "DU": "Dual",
    "Sing": "Sing", "Plur": "Plur", "Dual": "Dual",
}
PERSON_CODE_TO_VALUE = {"1": "1", "2": "2", "3": "3"}
GENDER_CODE_TO_VALUE = {
    "M": "Masc", "F": "Fem", "N": "Neut",
    "Masc": "Masc", "Fem": "Fem", "Neut": "Neut",
}

LANG_SPECS = {
    "fra": {"code": "fra", "name": "French"},
    "spa": {"code": "spa", "name": "Spanish"},
    "tur": {"code": "tur", "name": "Turkish"},
    "ara": {"code": "arb", "name": "Arabic"},
    "eng": {"code": "eng", "name": "English"},
    "deu": {"code": "deu", "name": "German"},
    "heb": {"code": "heb", "name": "Hebrew"},
    "hin": {"code": "hin", "name": "Hindi"},
}


def parse_feature(concept, feature_str):
    """Multi-BLiMP grammatical_feature field is like 'Number=Plur' or 'PL'
    or just raw. Extract the value by concept."""
    if feature_str is None:
        return None
    s = str(feature_str).strip()
    # Handle "Key=Val" style
    if "=" in s:
        key, val = s.split("=", 1)
        s = val.strip()

    if concept == "Number":
        return NUMBER_CODE_TO_VALUE.get(s, s)
    if concept == "Person":
        # Could be "1sg", "3pl" etc. Try to extract the digit
        for c in s:
            if c.isdigit():
                return PERSON_CODE_TO_VALUE.get(c, c)
        return PERSON_CODE_TO_VALUE.get(s, s)
    if concept == "Gender":
        return GENDER_CODE_TO_VALUE.get(s[:1], s)
    return s


def build_pairs(lang_key, max_per_phenomenon=500):
    spec = LANG_SPECS[lang_key]
    code = spec["code"]
    try:
        ds = load_dataset("jumelet/multiblimp", name=code, split="train")
    except Exception as e:
        print(f"[{lang_key}] FAILED to load: {type(e).__name__}: {e}", file=sys.stderr)
        return []

    pairs = []
    by_cell = defaultdict(int)
    for i, row in enumerate(ds):
        phen = row.get("phenomenon")
        concept = PHENOMENON_TO_CONCEPT.get(phen)
        if not concept:
            continue

        prefix_raw = row.get("prefix")
        if not prefix_raw:
            continue
        prefix = str(prefix_raw).strip()
        if not prefix:
            continue

        verb = row.get("verb")
        swap = row.get("swap_head")
        if not verb or not swap or verb == swap:
            continue

        value_orig = parse_feature(concept, row.get("grammatical_feature"))
        value_cf = parse_feature(concept, row.get("ungrammatical_feature"))
        if not value_orig or not value_cf or value_orig == value_cf:
            continue

        # For Arabic Dual: phenomenon might be SV-# with grammatical_feature=DU.
        # Rename concept for clarity in ledger.
        if value_orig == "Dual" or value_cf == "Dual":
            concept_tag = "Dual"
        else:
            concept_tag = concept

        # Per-cell cap
        cell = (concept_tag, value_orig)
        if by_cell[cell] >= max_per_phenomenon:
            continue
        by_cell[cell] += 1

        pairs.append({
            "id": f"{lang_key}_{phen}_{i:05d}",
            "prefix": prefix,
            "original_token": " " + verb.lstrip(),
            "counterfactual_token": " " + swap.lstrip(),
            "concept": concept_tag,
            "concept_value_orig": value_orig,
            "concept_value_cf": value_cf,
            "lang_code": lang_key,
            "source": "multiblimp",
            "phenomenon": phen,
        })

    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="data/multilingual_pairs")
    ap.add_argument("--snapshot_out", default="outputs/counterfactual_attribution/data_snapshot.json")
    ap.add_argument("--max_per_phenomenon", type=int, default=500)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_path = project_root / args.snapshot_out

    snapshot = {}
    all_pairs = []
    for lang_key in LANG_SPECS:
        print(f"[{lang_key}] building pairs...")
        pairs = build_pairs(lang_key, args.max_per_phenomenon)
        with open(out_dir / f"{lang_key}.json", "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)

        per_cell = defaultdict(int)
        for p in pairs:
            per_cell[(p["concept"], p["concept_value_orig"])] += 1
        snapshot[lang_key] = {
            "n_pairs": len(pairs),
            "per_cell": {f"{c}|{v}": n for (c, v), n in per_cell.items()},
        }
        print(f"[{lang_key}] {len(pairs)} pairs -> {out_dir / f'{lang_key}.json'}")
        all_pairs.extend(pairs)

    with open(out_dir / "all.json", "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    snap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"snapshot -> {snap_path}")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
