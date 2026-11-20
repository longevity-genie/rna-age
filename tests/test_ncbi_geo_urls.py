from rna_age.sources import ncbi_geo


def test_urls_use_ncbi_ftp_copy():
    assert ncbi_geo.counts_url("GSE164471") == (
        "https://ftp.ncbi.nlm.nih.gov/geo/rnaseq_counts/GSE164nnn/GSE164471_raw_counts_GRCh38.p13_NCBI.tsv.gz"
    )
    assert ncbi_geo.ANNOT_URL.startswith("https://ftp.ncbi.nlm.nih.gov/geo/rnaseq_counts/annot/")
