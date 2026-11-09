"""Curated aging cohorts in GEO: labels, not a separate count source.

The counts of these series come from ARCHS4 or the NCBI-generated GEO counts (``ncbi_geo`` cross-check
targets include them); this module records which GSMs belong to a named cohort and, for the Shokhirev &
Johnson 2021 compilation, the authors' curated age/sex/condition per sample (Mendeley Data
doi:10.17632/92rgnswtn8, v3), so our parsed ages can be audited against an independent curation.

Output: ``data/input/geo_cohorts/cohort_series.parquet`` (series → cohort) and
``data/input/geo_cohorts/shokhirev2021_samples.parquet`` (GSM/SRR → curated labels). The interim merge
fills ``cohort`` on ARCHS4 / NCBI samples from these tables. There is deliberately no ``samples.parquet``
here so the summary does not count these samples twice.
"""

from __future__ import annotations

import io
import logging
import zipfile

import polars as pl

from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir, input_dir

log = logging.getLogger(__name__)

NAME = "geo_cohorts"
MENDELEY = "https://data.mendeley.com/public-files/datasets/92rgnswtn8/files"
REMOTES = (
    Remote(
        url=f"{MENDELEY}/523aa92c-8930-4a36-8739-0645bfc4f068/file_downloaded",
        name="shokhirev2021_RawTables.zip",
        size=57_619_603,
        checksum="sha256:f0fb548ae40bb28909ab049282c990da27c8c6545406e9b8d050f8e912d77f35",
    ),
    Remote(
        url=f"{MENDELEY}/3eb9140a-d390-4c7b-9975-25e5d8469b4d/file_downloaded",
        name="shokhirev2021_Allcounts.zip",
        size=145_443_299,
        checksum="sha256:5c976908b5b85669d7ebbbe34c5434022e2fb2b573830dac61530c1b34bb4e96",
    ),
)

# Verified on GEO 2026-11-09 (RNA-seq, age in characteristics_ch1). Shokhirev & Johnson 2021 series are
# added from their metadata in ``parse``.
COHORTS: dict[str, str] = {
    "GSE113957": "fleischer2018_fibroblasts",
    "GSE134080": "500fg_whole_blood",
    "GSE141910": "magnet_heart_lv",
    "GSE115828": "ratnapriya2019_retina",
    "GSE80655": "ramaker2017_brain",
    "GSE124326": "bipolar_whole_blood",
    "GSE60424": "linsley2014_immune_subsets",
    "GSE121827": "tfh_treg_sorted",
    "GSE102556": "labonte2017_brain",
    "GSE164471": "gestalt_muscle_tumasian2021",
    "GSE95587": "friedman2018_brain_fusiform",
    "GSE104704": "nativio2018_brain_temporal",
    "GSE81089": "mezheyeuski_lung_tumour_normal",
    "GSE51799": "huntington_blood",
    "GSE94438": "tb_blood_cohort",
    "GSE95640": "adipose_cohort",
    "GSE99074": "galatro2017_microglia",
}


def fetch(connections: int = 4) -> None:
    for r in REMOTES:
        _fetch(r, download_dir(NAME), connections=1)


def _read_meta(zip_name: str, member_suffix: str) -> pl.DataFrame:
    with zipfile.ZipFile(download_dir(NAME) / zip_name) as z:
        name = next(n for n in z.namelist() if n.endswith(member_suffix))
        raw = z.read(name)
    return pl.read_csv(io.BytesIO(raw), separator="\t", infer_schema_length=0)


def parse() -> None:
    out = input_dir(NAME)
    used = _read_meta("shokhirev2021_RawTables.zip", "meta_filtered.txt")
    all_meta = _read_meta("shokhirev2021_Allcounts.zip", "Allmeta.txt")
    gsm_col = next(c for c in all_meta.columns if c.upper().startswith("GEO") or c.upper() == "GSM")
    srr_col = next(c for c in all_meta.columns if c.upper().startswith("SRR"))
    used_srr = set(used[next(c for c in used.columns if c.upper().startswith("SRR"))].to_list())
    samples = all_meta.select(
        gsm=pl.col(gsm_col),
        srr=pl.col(srr_col),
        age_years=pl.col("Age").cast(pl.Float64, strict=False),
        sex=pl.col("Gender").str.to_lowercase(),
        condition=pl.col("Condition"),
        tissue=pl.col("Tissue"),
        batch=pl.col("Batch"),
    ).with_columns(in_paper_set=pl.col("srr").is_in(list(used_srr)), cohort=pl.lit("shokhirev2021"))
    samples.write_parquet(out / "shokhirev2021_samples.parquet")
    series = pl.DataFrame({"gse": list(COHORTS), "cohort": list(COHORTS.values())})
    series.write_parquet(out / "cohort_series.parquet")
    log.info(
        "geo_cohorts: %d named series; Shokhirev 2021: %d samples (%d in the paper's filtered set)",
        series.height, samples.height, samples["in_paper_set"].sum(),
    )  # fmt: skip
