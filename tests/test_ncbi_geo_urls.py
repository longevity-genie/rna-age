from rna_age.sources import ncbi_geo


def test_urls_use_ncbi_ftp_copy():
    assert ncbi_geo.counts_url("GSE164471") == (
        "https://ftp.ncbi.nlm.nih.gov/geo/rnaseq_counts/GSE164nnn/GSE164471_raw_counts_GRCh38.p13_NCBI.tsv.gz"
    )
    assert ncbi_geo.ANNOT_URL.startswith("https://ftp.ncbi.nlm.nih.gov/geo/rnaseq_counts/annot/")


def test_polite_fetch_skips_pause_for_fetched(tmp_path, monkeypatch):
    import json

    from rna_age.download import Remote
    from rna_age.sources import ncbi_geo

    (tmp_path / "x.txt").write_text("abc")
    (tmp_path / "x.txt.meta.json").write_text(json.dumps({"size": 3, "checksum": "c"}))
    slept, fetched = [], []
    monkeypatch.setattr(ncbi_geo.time, "sleep", slept.append)
    monkeypatch.setattr(ncbi_geo, "_fetch", lambda *a, **k: fetched.append(a))
    assert ncbi_geo._polite_fetch(Remote(url="https://x/x.txt", name="x.txt"), tmp_path) == tmp_path / "x.txt"
    assert slept == [] and fetched == []
