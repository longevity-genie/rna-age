import numpy as np
import polars as pl

from rna_age.train.health import _age_pairs


def _table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source": ["archs4"] * 6 + ["gtex"] * 4,
            "study_id": ["S1", "S1", "S1", "S2", "S2", "S1", "D1", "D2", "D3", "D4"],
            "tissue": ["blood", "blood", "blood", "blood", "blood", "liver", "lung", "lung", "lung", "skin"],
            "target": [20.0, 30.0, 60.0, 25.0, 70.0, 80.0, 25.0, 65.0, 45.0, 75.0],
        }
    )


def test_pairs_stay_within_group_and_gap():
    t = _table()
    p = _age_pairs(t, np.ones(t.height, bool), min_gap=20.0, seed=0)
    assert len(p) > 0
    grp = [
        ("gtex:" + ti) if s == "gtex" else f"{st}:{ti}"
        for s, st, ti in zip(t["source"], t["study_id"], t["tissue"], strict=True)
    ]
    a = t["target"].to_numpy()
    for o, y in p:
        assert grp[o] == grp[y]
        assert a[o] - a[y] >= 20.0
    seen = {(int(o), int(y)) for o, y in p}
    # S1 blood: (60, 20), (60, 30); S2 blood: (70, 25); GTEx lung across donors: (65, 25), (65, 45), (45, 25)
    assert seen <= {(2, 0), (2, 1), (4, 3), (7, 6), (7, 8), (8, 6)}
    assert (7, 6) in seen  # GTEx pairs across donors of one tissue


def test_mask_and_no_pairs():
    t = _table()
    mask = np.zeros(t.height, bool)
    mask[[0, 1]] = True  # 20 and 30: under the minimum gap
    assert _age_pairs(t, mask, min_gap=20.0).shape == (0, 2)
