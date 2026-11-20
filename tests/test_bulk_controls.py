import numpy as np
import polars as pl

from rna_age.sources.ncbi_geo import map_counts
from rna_age.train import bulk_controls
from rna_age.train.bulk_controls import BulkControl, labels, run_predictions, score_control


def test_map_counts_drops_unmapped_sums_shared_and_fills_zero():
    counts = pl.DataFrame({"geneid": [1, 2, 3, 4], "GSM1": [5, 7, 11, 13], "GSM2": [1, 2, 3, 4]})
    ann = pl.DataFrame({"geneid": [1, 2, 3], "gene_id": ["ENSG2", "ENSG2", "ENSG1"]})  # 4 has no Ensembl id
    m = map_counts(counts, ann, ["ENSG1", "ENSG2", "ENSG3"])
    assert m["gene_id"].to_list() == ["ENSG1", "ENSG2", "ENSG3"]
    assert m["GSM1"].to_list() == [11, 12, 0]
    assert m["GSM2"].to_list() == [3, 3, 0]


def _table(rng: np.random.Generator, n: int = 160, shift: float = 4.0) -> pl.DataFrame:
    age = rng.uniform(5, 70, n)
    case = rng.random(n) < 0.5
    pred = 25 + 0.6 * age + shift * case + rng.normal(0, 2, n)
    return pl.DataFrame({
        "gse": "GSE1", "gsm": [f"GSM{i:04d}" for i in range(n)], "donor": [f"d{i}" for i in range(n)],
        "arm": np.where(case, "case", "normal"), "age_years": age,
        "sex": np.where(rng.random(n) < 0.5, "male", "female"), "batch": np.where(rng.random(n) < 0.5, "A", "B"),
        "exclude_reason": None, "pred": pred,
    })  # fmt: skip


def test_score_control_recovers_shift_and_respects_age_subset():
    tab = _table(np.random.default_rng(0))
    spec = BulkControl("GSE1", "trisomy", "blood", "clock", "accelerates", "adults", min_age=18, batch=True)
    out = score_control(spec, tab, n_boot=200)
    (p,) = out["predictions"]
    assert p["counted"] and p["verdict"] == "hold" and p["lo"] < 4.0 < p["hi"]
    (t,) = out["type"]
    assert t["n_donors"] == int((tab["age_years"] >= 18).sum()) and abs(t["slope"] - 0.6) < 0.15
    assert out["contrast"][0]["covariates"].endswith("+batch")


def test_contaminated_series_is_exploratory():
    tab = _table(np.random.default_rng(1))
    spec = BulkControl("GSE1", "sarcopenia", "muscle", "clock", "accelerates")
    (p,) = score_control(spec, tab, n_train=3, n_boot=100)["predictions"]
    assert not p["counted"] and p["prediction"] == "exploratory" and p["contaminated"]


def test_labels_keep_usable_arms_only():
    lab = pl.DataFrame(
        {
            "gse": ["G"] * 4,
            "gsm": list("abcd"),
            "arm": ["case", "normal", None, "case"],
            "exclude_reason": [None, None, None, "single_nucleus"],
            "age_years": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert labels("G", lab)["gsm"].to_list() == ["a", "b"]


def test_series_predictions_average_the_members(tmp_path, monkeypatch):
    monkeypatch.setattr(bulk_controls, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(bulk_controls, "PRED_DIR", tmp_path / "pred")
    monkeypatch.setattr(bulk_controls, "ATLAS_MEMBERS", {"ens": ("m1", "m2")})
    (tmp_path / "pred").mkdir()
    pl.DataFrame(
        {
            "sample_id": ["a", "b", "a", "b", "c"],
            "run": ["m1", "m1", "m2", "m2", "m1"],
            "pred": [10.0, 20.0, 30.0, 40.0, 5.0],
        }
    ).write_parquet(tmp_path / "pred" / "G.parquet")
    out = run_predictions("ens", ["a", "b", "c"], "G").sort("sample_id")
    assert out["sample_id"].to_list() == ["a", "b"] and out["pred"].to_list() == [20.0, 30.0]
    assert run_predictions("ens", ["a"], "OTHER").height == 0
