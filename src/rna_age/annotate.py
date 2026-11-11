"""Sample-level flags from free-text GEO/BioSample annotations: sex, tissue, disease, health, tumour,
cell line, stem cell / organoid / xenograft, prenatal, single cell. Used by every GEO-based source.

Everything works on a polars frame with a lower-cased ``text`` column (title + source + characteristics) and
the parsed ``fields`` (list of key/value structs), so the rules are vectorised and auditable.
"""

from __future__ import annotations

import polars as pl

TUMOR = (
    r"tumou?r|cancer|carcinoma|melanoma|leuka?emia|lymphoma|sarcoma|gliom|glioblastoma|myeloma|neoplas|"
    r"metasta|adenoma\b|blastoma|\bcarcinoid|mesothelioma|\bhcc\b|\bnsclc\b|\baml\b|\bcll\b|\ball\b(?: patient)|"
    r"\bmds\b|myelodysplas|malignan|meningioma|schwannoma|ependymoma|craniopharyngioma|chordoma|thymoma|"
    r"pheochromocytoma|paraganglioma|insulinoma"
)
TUMOR_NORMAL = r"adjacent|non[- ]?tumou?r|tumou?r[- ]free|normal tissue|matched normal|paired normal|healthy tissue"
CELL_LINE_NAMES = (
    r"\bhela\b|\bhek ?293|\b293t\b|\bmcf[- ]?7\b|\bmcf10a\b|\bk562\b|\bhepg2\b|\ba549\b|\bjurkat\b|\bthp[- ]?1\b|"
    r"\bu2os\b|\bhct[- ]?116\b|\bsh[- ]?sy5y\b|\bimr[- ]?90\b|\bwi[- ]?38\b|\bbj\b(?: fibroblast)|\bhacat\b|"
    r"\bcaco[- ]?2\b|\bhuh[- ]?7\b|\bpc[- ]?3\b|\blncap\b|\bdu145\b|\bmda[- ]?mb|\bt47d\b|\bsk[- ]?n|\bu87\b|"
    r"\bu937\b|\bhl[- ]?60\b|\bmolm|\bnalm|\braji\b|\bgm\d{4,5}\b|\blcl\b|lymphoblastoid|\bht[- ]?29\b|"
    r"\bsw480\b|\bsw620\b|\bh1299\b|\bh460\b|\brpe[- ]?1\b|\bhff\b|immortali[sz]ed|\btert\b"
)
STEM = (
    r"\bipsc|\bips cell|induced pluripotent|\bhesc|\besc\b|embryonic stem|pluripotent|\bh1\b(?: cell| esc| hesc)|"
    r"\bh9\b(?: cell| esc| hesc)|organoid|spheroid|xenograft|\bpdx\b|in vitro differentiat|differentiated from|"
    r"derived from (?:ipsc|hesc|esc|pluripotent)|reprogramm|assembloid|gastruloid"
)
PRENATAL = r"fetal|foetal|fetus|embryo\b|embryos\b|embryonic (?:tissue|day|brain|kidney|liver)|gestation|\bpcw\b|post[- ]?conception|placenta|umbilical|cord blood|amniotic|chorionic|trimester"
SINGLE_CELL = r"single[- ]?cell|scrna|snrna|single[- ]?nucle|10x genomics|chromium|drop-?seq|smart-?seq2? single"
HEALTHY = (
    r"healthy|\bnormal\b|\bcontrols?\b|\bctrls?\b|\bhc\b|\bhcs\b|non[- ]?diseased|\bnone\b|no disease|unaffected|"
    r"\bwt\b|wild[- ]?type|non[- ]?demented|\bnd\b|neurologically normal"
)
# values of disease-like keys that describe study design, not health (cohort / split labels)
NOT_A_DISEASE = (
    r"^(?:discovery|validation|replication|training|test(?:ing)?|train|batch.*|cohort.*|set.*|group ?\d*|[a-z]?\d+)$"
)

DISEASE_KEYS = r"^(?:disease(?: state| status)?|diagnosis|condition|status|phenotype|disease type|group|subject status|clinical status|health status|cohort)$"
SEX_KEYS = r"^(?:sex|gender|donor sex|patient sex|sex of donor|donor gender|biological sex)$"
TISSUE_KEYS = r"^(?:tissue|tissue type|organ|body site|tissue source|tissue/cell type|anatomical site|biopsy site|sampling site|source tissue)$"
CELL_TYPE_KEYS = r"^(?:cell type|cell population|cell subset|sorted population|celltype|cell types)$"
DONOR_KEYS = r"^(?:donor(?: id)?|individual(?: id)?|subject(?: id)?|patient(?: id)?|participant(?: id)?|donor_id|subject_id|patient_id|individual_id)$"
CELL_LINE_KEYS = r"^(?:cell line|cell-line|cellline|cell line name)$"
NA_LIKE = r"^(?:|n/?a|na|none|not applicable|not available|unknown|-|--|primary|primary cells?|no)$"


def field(fields: str, keys: str) -> pl.Expr:
    """First value of the key/value list whose key matches ``keys`` (case-insensitive)."""
    return (
        pl.col(fields)
        .list.eval(pl.element().filter(pl.element().struct.field("key").str.to_lowercase().str.contains(keys)))
        .list.first()
        .struct.field("value")
    )


def normalise_sex(expr: pl.Expr) -> pl.Expr:
    s = expr.str.to_lowercase().str.strip_chars()
    return (
        pl.when(s.str.contains(r"^(?:f|female|woman|women|girl|xx)$"))
        .then(pl.lit("female"))
        .when(s.str.contains(r"^(?:m|male|man|men|boy|xy)$"))
        .then(pl.lit("male"))
    )


def add_flags(df: pl.DataFrame, fields: str = "fields", text: str = "text") -> pl.DataFrame:
    """Add sex, tissue, cell_type, disease, donor_id, is_healthy, is_tumor, is_cell_line and exclude_reason.

    ``exclude_reason`` is set in priority order: non_human, single_cell, cell_line, stem_or_organoid,
    prenatal, tumor, cancer_patient; null = eligible for age training (when the sample also has an age).
    """
    t = pl.col(text)
    disease = field(fields, DISEASE_KEYS)
    disease = (
        pl.when(disease.str.to_lowercase().str.strip_chars().str.contains(NOT_A_DISEASE)).then(None).otherwise(disease)
    )
    cell_line_value = field(fields, CELL_LINE_KEYS)
    df = df.with_columns(
        sex=normalise_sex(field(fields, SEX_KEYS)),
        tissue=field(fields, TISSUE_KEYS),
        cell_type=field(fields, CELL_TYPE_KEYS),
        disease=disease,
        donor_id=field(fields, DONOR_KEYS),
        _cell_line_key=cell_line_value.is_not_null()
        & ~cell_line_value.str.to_lowercase().str.strip_chars().str.contains(NA_LIKE),
        _cancer=t.str.contains(TUMOR),
        _tumor_normal=t.str.contains(TUMOR_NORMAL),
        _stem=t.str.contains(STEM),
        _prenatal=t.str.contains(PRENATAL),
        _sc=t.str.contains(SINGLE_CELL),
        _healthy_text=disease.str.to_lowercase().str.contains(HEALTHY).fill_null(False)
        | t.str.contains(
            r"healthy (?:donor|control|volunteer|subject|individual)|disease state: (?:control|normal|healthy)"
        ),
    )
    df = df.with_columns(
        is_cell_line=pl.col("_cell_line_key") | t.str.contains(CELL_LINE_NAMES),
        is_tumor=pl.col("_cancer") & ~pl.col("_tumor_normal"),
    )
    df = df.with_columns(
        is_healthy=pl.when(pl.col("_cancer"))
        .then(False)
        .when(pl.col("_healthy_text"))
        .then(True)
        .when(pl.col("disease").is_not_null())
        .then(False),
        disease=pl.when(pl.col("disease").is_null() & pl.col("_cancer"))
        .then(pl.lit("cancer (from text)"))
        .otherwise(pl.col("disease")),
    )
    reason = (
        pl.when(pl.col("_sc"))
        .then(pl.lit("single_cell"))
        .when(pl.col("is_cell_line"))
        .then(pl.lit("cell_line"))
        .when(pl.col("_stem"))
        .then(pl.lit("stem_or_organoid"))
        .when(pl.col("_prenatal"))
        .then(pl.lit("prenatal"))
        .when(pl.col("is_tumor"))
        .then(pl.lit("tumor"))
        .when(pl.col("_cancer"))
        .then(pl.lit("cancer_patient"))
    )
    if "exclude_reason" in df.columns:
        reason = pl.coalesce(pl.col("exclude_reason"), reason)
    return df.with_columns(exclude_reason=reason).drop(
        "_cell_line_key", "_cancer", "_tumor_normal", "_stem", "_prenatal", "_sc", "_healthy_text"
    )
