"""RM17: reading an AnnData pseudobulk with h5py, and shifts against non-targeting guides."""

import h5py
import numpy as np
import polars as pl

from rna_age.train.perturb import check, per_gene, read_screen, shifts


def _write(path, X, csr: bool):
    with h5py.File(path, "w") as h:
        obs = h.create_group("obs")
        g = obs.create_group("donor_id")
        g["codes"] = np.array([0, 1, 0], dtype=np.int8)
        g["categories"] = np.array([b"D1", b"D2"])
        obs["n_cells"] = np.array([30, 40, 50])
        var = h.create_group("var")
        var["gene_ids"] = np.array([b"ENSG1.1", b"ENSG2", b"ENSG3"])
        if csr:
            from scipy.sparse import csr_matrix

            m = csr_matrix(X)
            x = h.create_group("X")
            x["data"], x["indices"], x["indptr"] = m.data, m.indices, m.indptr
        else:
            h["X"] = X


def test_read_screen_dense_and_csr_agree(tmp_path):
    X = np.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype=np.float32)
    for csr in (False, True):
        p = tmp_path / f"s{csr}.h5ad"
        _write(p, X, csr)
        obs, c = read_screen(p, ("donor_id", "n_cells"), ["ENSG3", "ENSG9", "ENSG1"], chunk=2)
        assert obs["donor_id"].to_list() == ["D1", "D2", "D1"]
        assert c.tolist() == [[3, 0, 1], [0, 0, 0], [9, 0, 7]]  # reordered, absent gene zero, version stripped


def test_shift_is_relative_to_the_same_donor_and_condition():
    obs = pl.DataFrame({"donor_id": ["a", "a", "b", "b", "a"], "culture_condition": ["Rest"] * 5,
                        "guide_type": ["non-targeting", "targeting", "non-targeting", "targeting", "non-targeting"],
                        "guide_id": ["n1", "g1", "n1", "g1", "n2"], "perturbed_gene_name": [None, "X", None, "X", None]})  # fmt: skip
    s = shifts(obs, np.array([40.0, 45.0, 60.0, 58.0, 42.0]))
    t = s.filter(pl.col("guide_type") == "targeting").sort("donor_id")["shift"].to_list()
    assert t == [4.0, -2.0]  # donor a: 45 - mean(40, 42); donor b: 58 - 60


def test_per_gene_p_from_non_targeting_guides():
    rng = np.random.default_rng(0)
    rows = [{"culture_condition": "Rest", "guide_type": "non-targeting", "guide_id": f"n{i}", "perturbed_gene_name": None,
             "shift": float(rng.normal(0, 0.5))} for i in range(200)]  # fmt: skip
    rows += [{"culture_condition": "Rest", "guide_type": "targeting", "guide_id": "g", "perturbed_gene_name": "BIG",
              "shift": 5.0}, {"culture_condition": "Rest", "guide_type": "targeting", "guide_id": "h",
              "perturbed_gene_name": "NONE", "shift": 0.0}]  # fmt: skip
    g = per_gene(pl.DataFrame(rows, infer_schema_length=None))
    p = dict(zip(g["perturbed_gene_name"].to_list(), g["p"].to_list(), strict=True))
    assert p["BIG"] < 0.01 and p["NONE"] > 0.5


def test_check_reads_agreement_per_condition_and_ignores_k562():
    rng = np.random.default_rng(1)
    names = [f"G{i}" for i in range(300)]
    coef = rng.permutation(np.linspace(-1, 1, 300))  # symmetric, so -|coef| has rho 0 with coef
    rows_g, rows_c = [], []
    for cond, sign in (("Rest", 1.0), ("Stim8hr", 0.0)):
        shift = coef + rng.normal(0, 0.5, 300) if sign else -np.abs(coef)  # Stim8hr: tied to |coef|, not to its sign
        rows_g += [
            {"culture_condition": cond, "perturbed_gene_name": n, "shift": float(s)}
            for n, s in zip(names, shift, strict=True)
        ]
        rows_c += [{"celltype": cond, "regulator": n, "coef_mean": float(c)} for n, c in zip(names, coef, strict=True)]
    rows_c += [{"celltype": "K562", "regulator": n, "coef_mean": 0.0} for n in names]
    out = {r["culture_condition"]: r for r in check(pl.DataFrame(rows_g), pl.DataFrame(rows_c)).to_dicts()}
    assert set(out) == {"Rest", "Stim8hr"}
    assert out["Rest"]["validated"] and out["Rest"]["rho"] > 0.8 and 59 <= out["Rest"]["n_top"] <= 61
    assert not out["Stim8hr"]["validated"]


def test_hits_removes_the_cell_loss_trend_and_keeps_a_real_shift():
    from rna_age.train.perturb import hits

    rng = np.random.default_rng(2)
    rows = []
    for cond in ("Rest", "Stim8hr"):
        for d in ("a", "b", "c"):
            rows += [{"donor_id": d, "culture_condition": cond, "guide_id": f"n{i}", "perturbed_gene_name": None,
                      "guide_type": "non-targeting", "n_cells": 100, "shift": float(rng.normal(0, 1))} for i in range(300)]  # fmt: skip
            for gi in range(400):  # essential-like genes: fewer cells, a shift that follows cell loss
                cells = int(rng.integers(10, 100))
                trend = 3 * np.log(100 / cells)
                extra = 4.0 if gi == 0 else 0.0  # REAL: shifted beyond its cell loss
                for k in range(2):
                    rows.append({"donor_id": d, "culture_condition": cond, "guide_id": f"g{gi}_{k}", "perturbed_gene_name": f"G{gi}",
                                 "guide_type": "targeting", "n_cells": cells, "shift": float(trend + extra + rng.normal(0, 1))})  # fmt: skip
    g = hits(pl.DataFrame(rows, infer_schema_length=None))
    rep = set(g.filter(pl.col("replicated"))["perturbed_gene_name"].to_list())
    assert "G0" in rep
    assert len(rep) <= 3  # the cell-loss trend alone does not make hits
