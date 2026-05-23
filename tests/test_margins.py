"""Verification of MultiBLiMP margin aggregation (filled by the overnight verify pass)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "perplexity_bleu_linear"))
from run_per import aggregate_margins  # noqa: E402


@pytest.fixture
def per_item_df():
    # Two langs, two phenomena; known margins/accuracy.
    return pd.DataFrame([
        {"lang": "eng", "phenomenon": "SV-#", "logp_correct": -1.0, "logp_wrong": -2.0,
         "margin_total": 1.0, "margin_pertoken": 0.5, "correct": True},
        {"lang": "eng", "phenomenon": "SV-#", "logp_correct": -3.0, "logp_wrong": -2.0,
         "margin_total": -1.0, "margin_pertoken": -0.5, "correct": False},
        {"lang": "eng", "phenomenon": "SV-P", "logp_correct": -1.0, "logp_wrong": -5.0,
         "margin_total": 4.0, "margin_pertoken": 2.0, "correct": True},
        {"lang": "deu", "phenomenon": "SV-#", "logp_correct": -2.0, "logp_wrong": -1.0,
         "margin_total": -1.0, "margin_pertoken": -1.0, "correct": False},
    ])


def test_aggregate_margins_by_lang(per_item_df):
    by_lang, by_lang_phen = aggregate_margins(per_item_df)
    eng = by_lang.set_index("lang").loc["eng"]
    # accuracy = fraction with positive margin = 2/3; n = 3.
    assert eng["accuracy"] == pytest.approx(2 / 3)
    assert eng["n"] == 3
    # mean margin_total over the 3 eng rows = (1 - 1 + 4)/3.
    assert eng["margin_total"] == pytest.approx(4 / 3)


def test_aggregate_margins_by_phenomenon(per_item_df):
    _, by_lang_phen = aggregate_margins(per_item_df)
    row = by_lang_phen.set_index(["lang", "phenomenon"]).loc[("eng", "SV-#")]
    assert row["n"] == 2
    assert row["margin_total"] == pytest.approx(0.0)  # (1 + -1)/2
    assert row["accuracy"] == pytest.approx(0.5)
