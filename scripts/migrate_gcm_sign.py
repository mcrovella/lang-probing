"""Migrate saved GCM outputs to the orig-minus-cf sign convention.

The new convention is:

    M = logp(r_orig | p_orig) - logp(r_cf | p_orig)

Positive IE therefore favors the original/correct completion. Older saved
outputs used the opposite sign for `ie` tensors and `metric_diff_patched`.
This migration is in-place and idempotent: directories whose `summary.json`
already contains `"sign_convention": "orig_minus_cf"` are refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


MARKER_KEY = "sign_convention"
MARKER_VALUE = "orig_minus_cf"


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _negate_metric_diff_records(obj: Any) -> int:
    """Negate every nested `metric_diff_patched` value in a JSON object."""
    n = 0
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key.endswith("metric_diff_patched") and isinstance(value, (int, float)):
                obj[key] = -float(value)
                n += 1
            else:
                n += _negate_metric_diff_records(value)
    elif isinstance(obj, list):
        for value in obj:
            n += _negate_metric_diff_records(value)
    return n


def migrate_gcm_sign(dir: Path) -> dict:
    """Migrate one GCM direction directory in place.

    Returns a record with changed files and counts. Raises `RuntimeError` if
    the directory is already marked as migrated.
    """
    dir = Path(dir)
    if not dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {dir}")

    summary_path = dir / "summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {}
    if isinstance(summary, dict) and summary.get(MARKER_KEY) == MARKER_VALUE:
        raise RuntimeError(f"{dir} is already marked {MARKER_VALUE}; refusing to re-negate")

    record: dict[str, Any] = {"dir": str(dir), "changed": []}

    for name in ("heads_ie.pt", "sae_ie.pt"):
        path = dir / name
        if not path.exists():
            continue
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        torch.save(-tensor, path)
        record["changed"].append({"file": name, "action": "negated_tensor"})

    for name in ("per_pair_records.json", "summary.json"):
        path = dir / name
        if not path.exists():
            continue
        data = _load_json(path)
        n = _negate_metric_diff_records(data)
        if name == "summary.json":
            if not isinstance(data, dict):
                data = {"summary": data}
            data[MARKER_KEY] = MARKER_VALUE
        if n or name == "summary.json":
            _write_json(path, data)
            record["changed"].append({
                "file": name,
                "action": "negated_metric_diff_patched",
                "n_values": n,
            })

    return record


def iter_direction_dirs(root: Path):
    for parent_name in ("gcm_translation", "gcm_translation_null"):
        parent = root / parent_name
        if not parent.exists():
            continue
        for d in sorted(parent.iterdir()):
            if d.is_dir() and "__" in d.name:
                yield d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root containing gcm_translation/ and gcm_translation_null/.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = []
    for d in iter_direction_dirs(args.outputs_root):
        if args.dry_run:
            summary_path = d / "summary.json"
            marked = False
            if summary_path.exists():
                marked = _load_json(summary_path).get(MARKER_KEY) == MARKER_VALUE
            records.append({"dir": str(d), "would_skip_marked": marked})
            continue
        records.append(migrate_gcm_sign(d))

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
