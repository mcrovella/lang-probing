"""Verification of the GCM sign migration (filled by the overnight verify pass).

The new convention is M = logp(r_orig) - logp(r_cf), so positive IE favors the
correct/original completion. The migration negates `ie` tensors and
`*metric_diff_patched` scalars in place, exactly once (idempotency marker).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
from migrate_gcm_sign import migrate_gcm_sign, MARKER_KEY, MARKER_VALUE  # noqa: E402


def make_synthetic_gcm_dir(tmp_path, *, n_pairs, n_layers, n_heads, sae_dim) -> Path:
    d = Path(tmp_path) / "English__Spanish"
    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save(torch.randn(n_pairs, n_layers, n_heads), d / "heads_ie.pt")
    torch.save(torch.randn(n_pairs, sae_dim), d / "sae_ie.pt")
    (d / "per_pair_records.json").write_text(json.dumps([
        {"head_metric_diff_patched": 1.0, "sae_metric_diff_patched": -2.0, "pair_id": "p0"},
    ]))
    (d / "summary.json").write_text(json.dumps({"n_pairs": n_pairs, "metric_diff_patched": 3.0}))
    return d


def test_gcm_sign_migration_is_exact_negation(tmp_path):
    d = make_synthetic_gcm_dir(tmp_path, n_pairs=3, n_layers=4, n_heads=4, sae_dim=8)
    heads_before = torch.load(d / "heads_ie.pt", weights_only=True).clone()
    sae_before = torch.load(d / "sae_ie.pt", weights_only=True).clone()

    migrate_gcm_sign(d)

    heads_after = torch.load(d / "heads_ie.pt", weights_only=True)
    sae_after = torch.load(d / "sae_ie.pt", weights_only=True)
    # Exact sign flip; magnitudes preserved bit-for-bit.
    assert torch.equal(heads_after, -heads_before)
    assert torch.equal(sae_after, -sae_before)
    assert torch.equal(heads_after.abs(), heads_before.abs())

    # Nested *metric_diff_patched scalars negated.
    recs = json.loads((d / "per_pair_records.json").read_text())
    assert recs[0]["head_metric_diff_patched"] == -1.0
    assert recs[0]["sae_metric_diff_patched"] == 2.0
    summary = json.loads((d / "summary.json").read_text())
    assert summary["metric_diff_patched"] == -3.0
    assert summary[MARKER_KEY] == MARKER_VALUE


def test_gcm_sign_migration_is_idempotent(tmp_path):
    d = make_synthetic_gcm_dir(tmp_path, n_pairs=2, n_layers=2, n_heads=2, sae_dim=4)
    migrate_gcm_sign(d)
    heads_once = torch.load(d / "heads_ie.pt", weights_only=True).clone()
    # A second migration must refuse (marker present) and leave tensors untouched.
    with pytest.raises(RuntimeError):
        migrate_gcm_sign(d)
    assert torch.equal(torch.load(d / "heads_ie.pt", weights_only=True), heads_once)


@pytest.mark.gpu
@pytest.mark.skip(reason="gcm_core targets self_attn.o_proj (Llama/Cohere); GPT2 "
                         "lacks it. Sign-vs-metric direction is validated end-to-end "
                         "by the head_ablation_multiblimp diagnostic instead.")
def test_gcm_pair_sign_matches_metric_direction():
    pass
