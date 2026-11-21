import numpy as np
import torch

from rna_age.train import cell_loss as cl


def test_pool_units_mean_and_weighted():
    mu = torch.tensor([1.0, 3.0, 10.0])
    lv = torch.tensor([0.0, np.log(3.0), 0.0], dtype=torch.float32)
    seg = torch.tensor([0, 0, 1])
    assert torch.allclose(cl.pool_units(mu, lv, seg, 2, weighted=False), torch.tensor([2.0, 10.0]))
    # weights 1 and 1/3: (1 + 1) / (4/3) = 1.5
    assert torch.allclose(cl.pool_units(mu, lv, seg, 2, weighted=True), torch.tensor([1.5, 10.0]))


def test_ccc_and_rank_loss():
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert abs(float(cl.ccc(y, y)) - 1.0) < 1e-5
    assert float(cl.ccc(y * 0.5, y)) < 1.0  # shrunk predictions lose concordance despite r = 1
    good, bad = cl.rank_loss(y, y, 1.0), cl.rank_loss(-y, y, 1.0)
    assert float(good) < float(bad)
    assert float(cl.rank_loss(y, torch.zeros(4), 1.0)) == 0.0  # no pair far enough apart


def test_model_name():
    assert cl.model_name("mse", 0.0, 15) == "cell-mse-pc15"
    assert cl.model_name("nll", 0.1, 1) == "cell-nll-w0.1-cells"


def test_fit_fold_learns_a_linear_signal():
    rng = np.random.default_rng(0)
    donors = [f"d{i}" for i in range(80)]
    age = rng.uniform(20, 80, 80)
    unit_donor = np.repeat(donors, 6)
    x = rng.normal(size=(len(unit_donor), 20)).astype(np.float32)
    x[:, 0] += (np.repeat(age, 6) - 50) / 15  # one gene carries age
    tr, va, te = donors[:50], donors[50:60], donors[60:]
    for loss, w in (("mse", 0.0), ("nll", 0.1), ("ccc", 0.5), ("rank", 0.5)):
        p = cl.fit_fold(x.astype(np.float16), unit_donor, tr, age[:50], va, age[50:60], te, loss, w, epochs=40,
                        donors_per_batch=16, seed=0)  # fmt: skip
        assert np.corrcoef(p, age[60:])[0, 1] > 0.8, loss
