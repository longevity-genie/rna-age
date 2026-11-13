import polars as pl
import pytest

from rna_age import positive_controls as pc


def _donors(case_ages, ctrl_ages, batch=None, sex="female"):
    n = len(case_ages) + len(ctrl_ages)
    df = pl.DataFrame({
        "donor_id": [f"d{i}" for i in range(n)],
        "arm": ["case"] * len(case_ages) + ["control"] * len(ctrl_ages),
        "age": [float(a) if a is not None else None for a in [*case_ages, *ctrl_ages]],
        "sex": [sex] * n,
        "n_cells": [100] * n,
    }, schema_overrides={"age": pl.Float64})  # fmt: skip
    df = df.with_columns(age_any=pl.col("age"))
    if batch is not None:
        df = df.with_columns(batch=pl.Series(batch, dtype=pl.String))
    return df


def test_match_pairs_caliper_and_optimal():
    assert pc.match_pairs([50, 60], [52, 80]) == 1
    assert pc.match_pairs([10, 20, 30], [14, 24, 34]) == 3
    assert pc.match_pairs([10, 12], [7, 15]) == 2  # greedy-by-case would still find both
    assert pc.match_pairs([], [1, 2]) == 0
    assert pc.match_pairs([None, 40.0], [41.0]) == 1


def test_common_support():
    assert pc.common_support([20, 50, 90], [48, 52]) == (1, 2)


def test_age_from_range_and_stage():
    assert pc.age_from_range(40, 40) == ("exact", 40.0, 40.0)
    assert pc.age_from_range(40, 44) == ("narrow", 42.0, 42.0)
    assert pc.age_from_range(40, 50) == ("binned", None, 45.0)
    assert pc.age_from_range(80, None) == ("censored", None, 80.0)
    assert pc.age_from_range(None, None) == (None, None, None)
    assert pc.age_from_stage("54-year-old stage") == ("exact", 54.0, 54.0)
    assert pc.age_from_stage("sixth decade stage")[0] == "binned"
    assert pc.age_from_stage("Carnegie stage 22") == (None, None, None)
    assert pc.age_from_stage("10th week post-fertilization stage") == (None, None, None)


def test_cancer_and_prenatal():
    assert pc.is_cancer("lung adenocarcinoma") and pc.is_cancer("B-cell non-Hodgkin lymphoma")
    assert not pc.is_cancer("Alzheimer disease") and not pc.is_cancer(None)
    assert pc.is_prenatal("fetal stage") and not pc.is_prenatal("45-year-old stage")


def test_cramers_v_and_batch_check():
    arm = ["case"] * 5 + ["control"] * 5
    assert pc.cramers_v(arm, ["b1"] * 5 + ["b2"] * 5) == pytest.approx(1.0)
    assert pc.batch_check(arm, ["b1"] * 5 + ["b2"] * 5)["status"] == "confounded"
    balanced = ["b1", "b2"] * 5
    r = pc.batch_check(arm, balanced)
    assert r["status"] == "ok" and r["shared"] == 2
    # no level holds both arms, even with V below 0.8 (three batches)
    r = pc.batch_check(arm, ["b1"] * 5 + ["b2", "b2", "b3", "b3", "b3"])
    assert r["status"] == "confounded" and r["shared"] == 0
    assert pc.batch_check(arm, ["b1"] * 10)["status"] == "ok"
    assert pc.batch_check(arm, [None] * 10)["status"] == "none"


def test_verdict_rules():
    assert pc.verdict(20, 20, 12, 12, "ok", True) == ("usable", "")
    v, why = pc.verdict(20, 20, 8, 8, "ok", True)
    assert v == "caveat" and "donors" in why
    v, why = pc.verdict(20, 20, 4, 12, "ok", True)
    assert v == "caveat" and "ages" in why
    v, why = pc.verdict(20, 20, 12, 12, "confounded", True)
    assert v == "not usable" and "batch" in why
    v, why = pc.verdict(20, 20, 12, 12, "none", True)
    assert v == "caveat" and "D9" in why
    assert pc.verdict(20, 20, 12, 12, "none", False)[0] == "usable"  # bulk: no batch needed
    v, why = pc.verdict(20, 20, 8, 8, "borderline", True)
    assert v == "not usable" and "several" in why
    v, why = pc.verdict(5, 40, 3, 3, "ok", False)
    assert v == "not usable" and "too few" in why


def test_score_counts_arms_matching_and_batch():
    case = list(range(40, 60, 2))  # 10 donors
    ctrl = [a + 1 for a in case] + [90, 91]
    batch = (["b1", "b2"] * 11)[: len(case) + len(ctrl)]
    s = pc.score(_donors(case, ctrl, batch), ["batch"], needs_batch=True, condition="Alzheimer disease")
    assert (s["n_case"], s["n_control"]) == (10, 12)
    assert s["n_pairs_matched"] == 10 and s["n_control_support"] == 10
    assert s["batch_status"] == "ok" and s["verdict"] == "usable"
    assert s["female_case"] == 10 and s["male_control"] == 0
    nested = ["b1"] * 10 + ["b2"] * 12
    s = pc.score(_donors(case, ctrl, nested), ["batch"], needs_batch=True, condition="Alzheimer disease")
    assert s["batch_status"] == "confounded" and s["verdict"] == "not usable"


def test_score_skips_id_like_and_synonym_columns():
    d = _donors([50] * 6, [50] * 6).with_columns(
        sample=pl.Series([f"s{i}" for i in range(12)]),
        cohort=pl.Series(["AD cohort"] * 6 + ["control cohort"] * 6),
    )
    s = pc.score(d, ["sample", "cohort"], needs_batch=True, condition="Alzheimer disease")
    assert s["batch_status"] == "none" and "one level per donor" in s["batch_detail"]
    assert "reads like the condition" in s["batch_detail"]


def test_coarse_class_and_target():
    assert pc.coarse_class("CD8-positive, alpha-beta T cell") == "CD8 T"
    assert pc.coarse_class("microglial cell") == "microglia"
    assert pc.coarse_class("type II pneumocyte") == "epithelial"
    assert pc.coarse_class("pancreatic B cell") != "other"
    assert pc.target_relation("Alzheimer disease", "brain_cns")[0] == "direct"
    assert pc.target_relation("Alzheimer disease", "blood")[0] == "indirect"
    assert pc.target_relation("systemic lupus erythematosus", "pbmc")[0] == "direct"
    assert pc.target_relation("something new", "blood")[0] == "indirect"
    assert pc.target_relation("something new", "liver")[0] == "unknown"
    assert pc.system_class("labial gland") == "oral"


def test_cellxgene_candidates_excludes_cancer_and_nonhuman():
    def ds(i, dis, org="Homo sapiens", primary=True):
        return {"dataset_id": i, "organism": [{"label": org}], "is_primary_data": [primary],
                "disease": [{"label": d} for d in dis]}  # fmt: skip

    disc = [ds("a", ["normal", "COVID-19"]), ds("b", ["normal", "lung adenocarcinoma"]),
            ds("c", ["normal", "COVID-19"], org="Mus musculus"), ds("d", ["COVID-19"]),
            ds("e", ["normal", "COVID-19"], primary=False), ds("f", ["normal"])]  # fmt: skip
    assert [x["dataset_id"] for x in pc.cellxgene_candidates(disc)] == ["a"]
    assert [x["dataset_id"] for x in pc.cellxgene_candidates(disc, {"f"})] == ["a", "f"]


def _groups():
    rows = []
    for i in range(12):
        dis = "type 2 diabetes mellitus" if i < 6 else "normal"
        for ct, n in (("type B pancreatic cell", 80), ("CD4-positive, alpha-beta T cell", 20)):
            rows.append({"donor_id": f"d{i}", "disease": dis, "development_stage": f"{40 + i}-year-old stage",
                         "sex": "female" if i % 2 else "male", "tissue": "islet of Langerhans",
                         "tissue_type": "tissue", "assay": "10x 3' v3", "suspension_type": "cell",
                         "cell_type": ct, "is_primary_data": "True", "n_cells": n})  # fmt: skip
    rows.append({**rows[0], "donor_id": "fetal", "development_stage": "fetal stage"})
    rows.append({**rows[0], "donor_id": "tum", "disease": "pancreatic carcinoma"})
    return pl.DataFrame(rows)


def test_score_cellxgene_end_to_end():
    extra = pl.DataFrame({"donor_id": [f"d{i}" for i in range(12)], "column": ["batch"] * 12,
                          "value": ["b1", "b2"] * 6, "n_cells": [100] * 12})  # fmt: skip
    rows = pc._score_cellxgene({"source": "cellxgene", "dataset_id": "x"}, _groups(), extra, {})
    assert len(rows) == 1  # cancer label is never an arm; the fetal donor is dropped
    r = rows[0]
    assert r["condition"] == "type 2 diabetes mellitus" and (r["n_case"], r["n_control"]) == (6, 6)
    assert r["batch_status"] == "ok" and r["target_relation"] == "direct"
    assert "endocrine" in r["coarse_cells"] and r["cells_per_donor_case"] == 100


def test_condition_column_arms_among_normal_donors():
    g = _groups().filter(pl.col("disease") == "normal")
    extra = pl.DataFrame({"donor_id": [f"d{i}" for i in range(6, 12)], "column": ["subject.cmv"] * 6,
                          "value": ["Positive"] * 3 + ["Negative"] * 3, "n_cells": [100] * 6})  # fmt: skip
    rows = pc._score_cellxgene({"source": "cellxgene", "dataset_id": "x"}, g, extra, {})
    assert [r["condition"] for r in rows] == ["subject.cmv=Positive"]
    assert rows[0]["arm_control"] == "subject.cmv=Negative" and rows[0]["n_case"] == 3


def test_bulk_donors_and_scan():
    n = 24
    samples = pl.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)], "source": "archs4", "study_id": "GSE1",
        "donor_id": [f"p{i}" for i in range(n)], "age_years": [float(30 + 2 * (i % 12) + (i >= 12)) for i in range(n)],
        "age_min": None, "age_max": None, "age_type": "exact", "sex": "male", "tissue": "Lung",
        "disease": ["IPF"] * 12 + [None] * 12, "is_healthy": [False] * 12 + [True] * 12,
        "is_tumor": False, "is_cell_line": False, "is_pseudobulk": False, "exclude_reason": None,
        "platform": "HiSeq", "cohort": None,
    }, schema_overrides={"age_min": pl.Float64, "age_max": pl.Float64, "exclude_reason": pl.String,
                         "cohort": pl.String})  # fmt: skip
    extra = samples.head(2).with_columns(sample_id=pl.lit("t"), donor_id=pl.lit("t"), disease=pl.lit("lung cancer"),
                                         is_healthy=pl.lit(False))  # fmt: skip
    rows = pc.scan_bulk(pl.concat([samples, extra]))
    assert len(rows) == 1 and rows[0]["condition"] == "IPF"
    assert rows[0]["n_pairs_matched"] >= 10 and rows[0]["verdict"] == "usable"


def test_conditions_table_and_seed_flag():
    base = {"source": "cellxgene", "covered_by": None, "in_census": None, "scan_status": "scored",
            "tissue_class": "lung", "n_pairs_matched": 12, "n_case": 15, "n_control": 20}  # fmt: skip
    out = pl.DataFrame([
        {**base, "dataset_id": "a", "condition": "COVID-19", "condition_id": "MONDO:0100096", "verdict": "usable"},
        {**base, "dataset_id": "b", "condition": "covid-19", "condition_id": "MONDO:0100096", "verdict": "caveat"},
        {**base, "dataset_id": "c", "condition": "pemphigus", "condition_id": "pemphigus", "verdict": "usable"},
    ])  # fmt: skip
    seed = pl.DataFrame({"condition": ["COVID-19 (acute)"]})
    t = pc.conditions_table(out, seed)
    covid = t.filter(pl.col("condition_id") == "MONDO:0100096").row(0, named=True)
    assert covid["n_studies"] == 2 and covid["n_usable_studies"] == 1 and covid["seed"]
    assert covid["usable_matched_pairs"] == 12 and covid["case_donors"] == 30
    assert not t.filter(pl.col("condition_id") == "pemphigus")["seed"].item()


def test_clean_label():
    assert pc.clean_label("  Cancer (from text) ") == "cancer"
    assert pc.clean_label("Crohn's  disease.") == "crohn disease"
    assert pc.clean_label("SLE") == "systemic lupus erythematosus"
    assert pc.clean_label("rheumatoid arthritis (RA)") == "rheumatoid arthritis"
    assert pc.is_generic("case") and pc.is_generic("Mid") and pc.is_generic("case (TB)")
    assert pc.is_generic("Infected-Vax") and not pc.is_generic("Alzheimer's disease")


def test_batch_check_condition_derived_and_many_small_levels():
    arm = ["case"] * 6 + ["control"] * 6
    assert pc.batch_check(arm, ["d3", "d5", "d7", "d3", "d1", "d2"] + [None] * 6)["status"] == "none"
    # 40 donors in 20 libraries of two, arms alternating within most libraries: not nested
    arm = ["case", "control"] * 20
    libs = [f"L{i // 2}" for i in range(40)]
    r = pc.batch_check(arm, libs)
    assert r["status"] == "ok" and r["mixed_share"] == 1.0
    assert pc.cramers_v(arm, libs) < pc.cramers_v(arm, libs, corrected=False) + 1e-9


def test_named_conditions_are_not_generic():
    assert not pc.is_generic("COVID-19") and not pc.is_generic("type 2 diabetes mellitus")


def test_size_skip_retried_with_larger_limit(tmp_path, monkeypatch):
    import json

    from rna_age import positive_controls as pc

    monkeypatch.setattr(pc, "CACHE", tmp_path)
    monkeypatch.setattr(pc, "DOWNLOAD", tmp_path / "dl")
    calls = []

    def fake_read(url, max_bytes, local):
        calls.append(max_bytes)
        raise pc.ObsTooLarge("big")

    monkeypatch.setattr(pc, "read_obs", fake_read)
    assert pc.cached_obs("d1", "u", max_bytes=10) == "obs too large, needs job"
    assert json.loads((tmp_path / "d1.meta.json").read_text())["max_bytes"] == 10
    assert pc.cached_obs("d1", "u", max_bytes=10) == "obs too large, needs job"  # same limit: marker stands
    assert pc.cached_obs("d1", "u", max_bytes=100) == "obs too large, needs job"  # larger limit: read again
    assert calls == [10, 100]


def test_h5ad_remote_prefers_census_then_discover():
    import pytest

    from rna_age import positive_controls as pc

    discover = [{"dataset_id": "new", "assets": [{"filetype": "RDS", "url": "r"}, {"filetype": "H5AD", "url": "h",
                                                                                  "filesize": 7}]}]
    r = pc.h5ad_remote("old", discover, {"old": (5, "e")})
    assert r.url.endswith("old.h5ad") and r.size == 5 and r.checksum == "s3etag:e"
    r = pc.h5ad_remote("new", discover, {})
    assert (r.url, r.size, r.name) == ("h", 7, "new.h5ad")
    with pytest.raises(KeyError):
        pc.h5ad_remote("gone", discover, {})
