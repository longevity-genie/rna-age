"""Genome-scale CD4 T-cell Perturb-seq (RM17): Zhu, Dann … Pritchard, Marson, Cell 2026,
doi:10.1016/j.cell.2026.08.002; CZI Virtual Cells Platform, MIT licence; the public S3 bucket
``genome-scale-tcell-perturb-seq`` (us-east-1).

Primary CD4 T cells of 4 donors, CRISPRi of ~12,700 genes, Rest / Stim8hr / Stim48hr, 10x Flex probes (18,129
genes). Only the pseudobulk file is fetched (one profile per guide × donor × condition, 44.6 GB), not the per-cell
files (~120-170 GB per donor × condition), with the readme and the paper's two aging tables (its CD4 T-cell aging
signature from OneK1K, and the regulators' predicted aging effects: our comparator).

Not an age source: no samples enter the age table. Primary immune cells from donors, so within the no-cancer scope.
"""

from __future__ import annotations

from rna_age.download import Remote
from rna_age.download import fetch as _fetch
from rna_age.paths import download_dir

NAME = "cd4_perturbseq"
BASE = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/"
FILES = (
    Remote(url=BASE + "GWCD4i.pseudobulk_merged.h5ad", name="GWCD4i.pseudobulk_merged.h5ad", size=44_566_657_140,
           checksum="s3etag:010c14e0af0dccbc2524529d28ca517e-5313"),
    Remote(url=BASE + "data_sharing_readme.md", name="data_sharing_readme.md", size=26_759,
           checksum="md5:0919489793f206e7cc23418362a1acc8"),
    Remote(url=BASE + "suppl_tables/CD4T_aging_signature_DE_results_full.suppl_table.csv",
           name="CD4T_aging_signature_DE_results_full.suppl_table.csv", size=1_988_893,
           checksum="md5:b8e372ecc73bfe68ee8699f5c9b1d12b"),
    Remote(url=BASE + "suppl_tables/aging_prediction_condition_comparison_regulator_coefficients.csv",
           name="aging_prediction_condition_comparison_regulator_coefficients.csv", size=1_224_277,
           checksum="md5:9f89f82e4ea620a1e8108780df933533"),
)  # fmt: skip


def fetch(connections: int = 8) -> None:
    for remote in FILES:
        _fetch(remote, download_dir(NAME), connections=connections)


def parse() -> None:
    raise NotImplementedError("cd4_perturbseq: scored by rna-age perturb-score (RM17), not parsed into the age table")
