import polars as pl
import pytest

from rna_age.train.evaluation import evaluate, headline


def test_metrics():
    df = pl.DataFrame(
        {
            "sample_id": list("abcdef"),
            "source": ["x", "x", "x", "gtex", "gtex", "gtex"],
            "split": ["test"] * 6,
            "age_type": ["exact"] * 3 + ["binned"] * 3,
            "age_years": [20.0, 40.0, 60.0, None, None, None],
            "age_min": [20.0, 40.0, 60.0, 50.0, 50.0, 60.0],
            "age_max": [20.0, 40.0, 60.0, 59.0, 59.0, 69.0],
            "pred": [25.0, 40.0, 55.0, 55.0, 62.0, 58.0],
            "is_healthy": [True] * 6,
            "is_pseudobulk": [False] * 6,
            "tissue": ["t"] * 6,
            "cohort": [None] * 6,
        }
    )
    h = headline(evaluate(df, min_group=1))
    assert h["exact_mae"] == pytest.approx(10 / 3)
    assert h["exact_bias_slope"] == pytest.approx(-0.25)
    assert h["binned_interval_hit"] == pytest.approx(1 / 3)
    assert h["binned_interval_dist"] == pytest.approx((0 + 2 + 2) / 3)
