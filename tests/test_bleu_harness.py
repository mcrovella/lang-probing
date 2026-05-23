"""Verification of the BLEU translation harness (filled by the overnight verify pass).

`generate_batch` returns the prompt-stripped continuation, so `extract_translation`
receives only what the model generated after the query's trailing `>>`. The
translation is the FIRST `||`-segment; later segments are the model continuing
the few-shot pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "head_ablation_bleu"))
from translate import build_prompt, extract_translation  # noqa: E402


@pytest.fixture
def recorded_translation_io():
    # (prompt-stripped generation, expected translation)
    return [
        (" hola ||\ncat >> gato", "hola"),          # over-generated past first ||
        ("perro", "perro"),                          # clean stop
        (" El lunes, los cientificos ||\nx >> y", "El lunes, los cientificos"),
    ]


def test_extract_translation_recorded_io(recorded_translation_io):
    for gen, expected in recorded_translation_io:
        assert extract_translation(gen) == expected


def test_extract_translation_takes_first_segment_not_last():
    # Regression: must NOT return the last hallucinated segment.
    assert extract_translation("perro ||\nz >> w") == "perro"


def test_build_prompt_two_shot_cyclic_shape():
    src = ["s0", "s1", "s2"]
    tgt = ["t0", "t1", "t2"]
    p = build_prompt(src, tgt, 2)
    assert p == "s0 >> t0 ||\ns1 >> t1 ||\ns2 >>"
    assert p.rstrip().endswith(">>")


@pytest.mark.gpu
@pytest.mark.skip(reason="hook composition needs the real Llama o_proj; validated "
                         "in the BLEU head-ablation smoke on a real model.")
def test_generate_batch_hook_composition_path():
    pass
