import torch

from rna_age.train.sc_jepa import JepaHeads, effective_rank, jepa_loss, sigreg
from rna_age.train.sc_pretrain import ScMAE, sample_mask


def test_sigreg_small_for_gaussian_large_for_collapse():
    g = torch.Generator().manual_seed(0)
    gauss = torch.randn(4096, 32, generator=g)
    collapsed = torch.zeros(4096, 32) + 0.01 * torch.randn(4096, 32, generator=g)
    shifted = torch.randn(4096, 32, generator=g) * 3 + 2
    s_gauss = float(sigreg(gauss, generator=torch.Generator().manual_seed(1)))
    assert s_gauss < 2.0  # finite-sample noise, scaled by N, is O(1)
    assert float(sigreg(collapsed, generator=torch.Generator().manual_seed(1))) > 50 * s_gauss
    assert float(sigreg(shifted, generator=torch.Generator().manual_seed(1))) > 50 * s_gauss


def test_effective_rank():
    g = torch.Generator().manual_seed(0)
    assert effective_rank(torch.randn(2000, 16, generator=g)) > 14
    assert effective_rank(torch.randn(2000, 1, generator=g) @ torch.randn(1, 16, generator=g)) < 1.5


def test_jepa_step_learns_and_keeps_rank():
    torch.manual_seed(0)
    n, genes = 512, 64
    base = torch.randn(n, 4) @ torch.randn(4, genes)  # a low-dimensional signal shared by every gene view
    x = base + 0.1 * torch.randn(n, genes)
    model, heads = ScMAE(genes, hidden=64, emb=16, depth=1, dropout=0.0), JepaHeads(16, proj=8, hidden=64, dropout=0.0)
    opt = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=3e-3)
    first = None
    for _ in range(150):
        m1, m2 = sample_mask(n, genes, 0.3, 0.6), sample_mask(n, genes, 0.3, 0.6)
        loss, parts = jepa_loss(model, heads, x, m1, m2, lam=0.05)
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first
    with torch.no_grad():
        assert effective_rank(model.encode(x)) > 2
