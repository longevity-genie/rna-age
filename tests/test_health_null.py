"""The within-study label shuffle behind the health gap's null (RM1)."""

import numpy as np
import polars as pl

from rna_age.train.health import _category_gaps, permute_within_study


def _table(gap_case: float) -> pl.DataFrame:
    rows = []
    for s in range(6):
        for i in range(12):
            case = i < 5
            rows.append({"study_id": f"S{s}", "group": "case" if case else "control",
                         "category": "cardiometabolic" if case else None, "gap": (gap_case if case else 0.0) + 0.01 * i})  # fmt: skip
    return pl.DataFrame(rows)


def test_shuffle_keeps_each_study_counts_and_values():
    p = _table(5.0)
    q = permute_within_study(p, np.random.default_rng(0))
    count = lambda t: t.group_by("study_id", "group").len().sort("study_id", "group")  # noqa: E731
    assert count(p).equals(count(q))
    assert sorted(p["gap"].to_list()) == sorted(q["gap"].to_list())
    # labels moved only within studies: each study's gap values are unchanged as a set
    for s in p["study_id"].unique():
        assert sorted(p.filter(pl.col("study_id") == s)["gap"]) == sorted(q.filter(pl.col("study_id") == s)["gap"])


def test_shuffle_destroys_a_real_gap():
    p = _table(5.0)
    obs = _category_gaps(p, 5)["all"][0].mean()
    rng = np.random.default_rng(1)
    null = [_category_gaps(permute_within_study(p, rng), 5)["all"][0].mean() for _ in range(200)]
    assert obs > 4.9 and abs(np.mean(null)) < 1.0 and max(null) < obs
