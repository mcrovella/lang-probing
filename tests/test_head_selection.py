"""Verification of signed head selection (filled by the overnight verify pass)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "head_ablation_multiblimp"))
from heads import select_signed_heads  # noqa: E402


def test_select_signed_heads_top_n_per_side():
    # signed IE map: (0,0)=+1, (0,1)=-2, (1,0)=+3, (1,1)=-4
    signed = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    abs_map = signed.abs()

    out = select_signed_heads(signed, abs_map, n_per_sign=1)
    assert len(out["pos"]) == 1 and len(out["neg"]) == 1
    # Most positive is (1,0)=+3; most negative is (1,1)=-4.
    assert (out["pos"][0]["layer"], out["pos"][0]["head"]) == (1, 0)
    assert (out["neg"][0]["layer"], out["neg"][0]["head"]) == (1, 1)
    assert out["pos"][0]["signed_ie"] > 0
    assert out["neg"][0]["signed_ie"] < 0


def test_select_signed_heads_disjoint_and_signed():
    signed = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    out = select_signed_heads(signed, signed.abs(), n_per_sign=2)
    pos_keys = {(h["layer"], h["head"]) for h in out["pos"]}
    neg_keys = {(h["layer"], h["head"]) for h in out["neg"]}
    assert pos_keys.isdisjoint(neg_keys)
    assert all(h["signed_ie"] > 0 for h in out["pos"])
    assert all(h["signed_ie"] < 0 for h in out["neg"])
    # pos ranked by descending signed IE.
    assert [h["signed_ie"] for h in out["pos"]] == sorted(
        [h["signed_ie"] for h in out["pos"]], reverse=True
    )
