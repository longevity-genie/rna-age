"""Curated labels for the bulk GEO positive controls: every series' rule on real characteristic strings."""

import polars as pl

from rna_age import positive_controls as pc
from rna_age import positive_controls_bulk as pb


def _text(*samples: tuple[str, str, list[str], dict[str, str]]) -> str:
    """GEO targ=gsm form=text records: (gsm, title, characteristics, extra '!Sample_x' fields)."""
    lines = []
    for gsm, title, chars, extra in samples:
        lines += [f"^SAMPLE = {gsm}", f"!Sample_title = {title}", f"!Sample_geo_accession = {gsm}"]
        lines += [f"!Sample_characteristics_ch1 = {c}" for c in chars]
        lines += [f"!Sample_{k} = {v}" for k, v in extra.items()]
    return "\n".join(lines)


SEQ = {"type": "SRA", "library_strategy": "RNA-Seq", "instrument_model": "NextSeq 500"}


def _labels(gse: str, *samples) -> pl.DataFrame:
    return pb.label_series(gse, pb.read_samples(_text(*samples)), "archs4")


def test_read_samples_keeps_fields():
    s = pb.read_samples(_text(("GSM1", "t", ["Sex: 1-Male", "age: 75"], {"platform_id": "GPL14550", "type": "RNA"})))
    assert s[0].get("sex") == "1-Male" and s[0].platform == "GPL14550" and pb.assay_of(s[0]) == "microarray"


def test_gse113957_progeria():
    df = _labels(
        "GSE113957",
        ("GSM1", "197_2yr3mos_F_Unknown",
         ["cell id: HGADFN188", "disease: HGPS", "Sex: Female", "age: 2yr3mos", "ethnicity: Unknown"], SEQ),
        ("GSM2", "14_1yr_Male_Asian", ["cell id: AG08498", "disease: Normal", "Sex: Male", "age: 1",
                                       "ethnicity: Asian"], SEQ),
        ("GSM3", "139_8_Female-HGPS_Caucasian", ["cell id: AG11513", "disease: HGPS", "Sex: Female", "age: 8yr"], SEQ),
    )  # fmt: skip
    assert df["arm"].to_list() == ["case", "normal", "case"]
    assert df["condition"][0] == pb.HGPS and df["condition"][1] == "normal"
    assert df["age_years"].to_list() == [2.25, 1.0, 8.0]
    assert df["batch"].to_list() == ["PRF", "Coriell", "Coriell"]
    assert df["donor"][0] == "HGADFN188" and df["sex"].to_list() == ["female", "male", "female"]


def test_gse111016_sarcopenia():
    df = _labels(
        "GSE111016",
        ("GSM1", "Sample 3 [sss]", ["population: Chinese descent", "sarcopenia status: yes", "Sex: male",
                                    "age (yr): 69", "tissue: vastus lateralis muscle"], SEQ),
        ("GSM2", "Sample 1 [sss]", ["sarcopenia status: no", "Sex: male", "age (yr): 69"], SEQ),
    )  # fmt: skip
    assert df["arm"].to_list() == ["case", "normal"] and df["condition"][0] == pb.SARCOPENIA
    assert df["age_years"].to_list() == [69.0, 69.0] and df["tissue"][0] == "vastus lateralis muscle"


def test_gse167186_groups_and_single_nucleus():
    bulk = {**SEQ, "description": "Bulk RNA-seq"}
    sn = {**SEQ, "description": "Single-nuclei RNA-seq"}
    df = _labels(
        "GSE167186",
        ("GSM1", "X_10", ["group: Sarcopenia", "tissue: Skeletal muscle", "age: 68", "sppb: 12"], bulk),
        ("GSM2", "X_11", ["group: Old Healthy", "tissue: Skeletal muscle", "age: 71"], bulk),
        ("GSM3", "X_12", ["group: Young Healthy", "tissue: Skeletal muscle", "age: 21"], bulk),
        ("GSM4", "OldHM1", ["group: Old", "tissue: Skeletal muscle", "age: 75"], sn),
        ("GSM5", "X_13", ["group: UNCLASSIFIED", "tissue: Skeletal muscle", "age: 72"], bulk),
    )
    assert df["arm"].to_list() == ["case", "normal", "normal", None, None]
    assert df["exclude_reason"].to_list()[2:] == [None, "single_nucleus", "group 'unclassified'"]
    assert df["sex"].null_count() == 5  # GEO gives no sex for this series


def test_gse114007_oa_grade_and_instrument_batch():
    df = _labels(
        "GSE114007",
        ("GSM1", "Normal_Cart_2_2", ["age: 35", "Sex: F", "oa grade: 1"], {**SEQ, "instrument_model": "Illumina HiSeq 2000"}),
        ("GSM2", "OA_03", ["age: 67", "Sex: F", "oa grade: 4"], SEQ),
    )  # fmt: skip
    assert df["arm"].to_list() == ["normal", "case"] and df["condition"][1] == pb.OA
    assert df["batch"].to_list() == ["Illumina HiSeq 2000", "NextSeq 500"] and df["sex"][0] == "female"


def test_gse190125_down_syndrome():
    df = _labels(
        "GSE190125",
        ("GSM1", "Control, HTP0001B2", ["tissue: whole blood", "karyotype: Control", "Sex: Female", "age: 25.8",
                                        "sample source: A", "library type: strand-specific"], SEQ),
        ("GSM2", "T21, HTP0005A3", ["tissue: whole blood", "karyotype: T21", "Sex: Male", "age: 13.7",
                                    "sample source: D"], SEQ),
    )  # fmt: skip
    assert df["arm"].to_list() == ["normal", "case"] and df["condition"][1] == pb.DOWN
    assert df["donor"].to_list() == ["HTP0001", "HTP0005"] and df["batch"].to_list() == ["A", "D"]
    assert df["age_years"].to_list() == [25.8, 13.7]


def test_gse135251_masld_without_ages():
    df = _labels(
        "GSE135251",
        ("GSM1", "Liver patient 97", ["nas score: 4", "fibrosis stage: 2", "group in paper: NASH_F2",
                                      "disease: NAFLD", "Stage: early"], SEQ),
        ("GSM2", "Liver patient 1", ["nas score: 0", "fibrosis stage: 0", "group in paper: control",
                                     "disease: Control", "Stage: control"], SEQ),
    )  # fmt: skip
    assert df["arm"].to_list() == ["case", "normal"] and df["condition"][0] == pb.MASLD
    assert df["age_years"].null_count() == 2 and df["tissue"][0] == "liver"


def test_gse47460_microarray_lung():
    arr = {"type": "RNA", "platform_id": "GPL14550"}
    df = _labels(
        "GSE47460",
        ("GSM1", "LT000842RU_CTRL", ["disease state: Control", "Sex: 1-Male", "age: 75", "gold stage: 0-At Risk"], arr),
        ("GSM2", "LT001600RL_ILD", ["disease state: Interstitial lung disease", "Sex: 2-Female", "age: 62",
                                    "ild subtype: 2-UIP/IPF"], arr),
        ("GSM3", "LT001098RU_COPD", ["disease state: Chronic Obstructive Lung Disease", "Sex: 1-Male", "age: 68"], arr),
    )  # fmt: skip
    assert df["condition"].to_list() == ["normal", pb.ILD, pb.COPD]
    assert df["sex"].to_list() == ["male", "female", "male"] and df["donor"][0] == "LT000842"
    assert set(df["exclude_reason"].to_list()) == {"microarray"} and set(df["assay"].to_list()) == {"microarray"}
    s = pb.read_samples(_text(("GSM1", "LT1_CTRL", [], arr)))
    assert pb.counts_source("GSE47460", s, set(), set()).startswith("none: microarray")


def test_counts_source_order():
    s = pb.read_samples(_text(("GSM1", "t", [], SEQ)))
    assert pb.counts_source("GSE1", s, {"GSE1"}, set()) == "archs4"
    assert pb.counts_source("GSE1", s, set(), {"GSE1"}) == "ncbi_geo"
    assert pb.counts_source("GSE1", s, set(), set()) == "none: no NCBI-generated counts"


def test_scan_bulk_labels_scores_and_marks_missing_counts():
    rows = []
    for i in range(12):
        rows.append(("GSE9", f"GSMc{i}", f"c{i}", "case", "sarcopenia", 60.0 + i, "M", "muscle", "b1" if i % 2 else "b2"))
        rows.append(("GSE9", f"GSMn{i}", f"n{i}", "normal", "normal", 60.5 + i, "F", "muscle", "b2" if i % 2 else "b1"))
    df = pl.DataFrame(rows, schema=["gse", "gsm", "donor", "arm", "condition", "age_years", "sex", "tissue", "batch"],
                      orient="row").with_columns(
        age_type=pl.lit("exact"), sex=pl.col("sex").replace({"M": "male", "F": "female"}), assay=pl.lit("RNA-Seq"),
        counts_source=pl.lit("archs4"), exclude_reason=pl.lit(None, pl.String))  # fmt: skip
    (r,) = pc.scan_bulk_labels(df)
    assert r["verdict"] == "usable" and r["n_pairs_matched"] == 12 and "batch" in r["batch_columns"]
    (r,) = pc.scan_bulk_labels(df.with_columns(counts_source=pl.lit("none: microarray only (no RNA-seq counts)"),
                                               exclude_reason=pl.lit("microarray")))  # fmt: skip
    assert r["verdict"] == "not usable" and r["reason"].startswith("microarray")


def test_add_ncbi_targets_keeps_rows(tmp_path, monkeypatch):
    from rna_age.sources import ncbi_geo

    path = tmp_path / "targets.parquet"
    pl.DataFrame({"gse": ["GSE1", "GSE2"], "reason": ["crosscheck", "not_in_archs4"]}).write_parquet(path)
    monkeypatch.setattr(ncbi_geo, "TARGETS", path)
    out = pb.add_ncbi_targets(["GSE2", "GSE3"])
    assert out.rows() == [("GSE1", "crosscheck"), ("GSE2", "not_in_archs4"), ("GSE3", "positive_control")]
