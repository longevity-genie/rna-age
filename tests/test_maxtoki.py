"""RM23: MaxToki weights mapping, rank encoding, prompts, the cached time read-out and the statistics, on synthetic data."""

import numpy as np
import polars as pl
import pytest

from rna_age.train.maxtoki import CONFIG, to_hf
from rna_age.train.maxtoki_cd4 import build_prompts, compare, rank_encode


def test_rank_encode_orders_by_median_normalised_value_and_truncates():
    counts = np.array([10.0, 10.0, 5.0, 3.0])
    cols = np.array([0, 1, 2, 3])
    token = np.array([100, 101, 102, -1])  # column 3 is not in the vocabulary
    median = np.array([2.0, 1.0, 0.1, 1.0])  # normalised: 5, 10, 50
    assert rank_encode(counts, cols, 28.0, token, median, 10, 2, 3) == [2, 102, 101, 100, 3]
    assert rank_encode(counts, cols, 28.0, token, median, 2, 2, 3) == [2, 102, 101, 3]


def test_to_hf_splits_grouped_qkv_and_gate_up():
    h, nh = CONFIG["hidden_size"], CONFIG["num_attention_heads"]
    hd = h // nh
    L = 2
    # label every row of linear_qkv by (layer, group, part): part 0 = q, 1 = k, 2 = v
    qkv = np.zeros((L, nh * 3 * hd, h), dtype=np.float32)
    for i in range(L):
        for g in range(nh):
            for part in range(3):
                r0 = (g * 3 + part) * hd
                qkv[i, r0 : r0 + hd] = 100 * i + 10 * g + part
    import torch

    fc1 = torch.cat([torch.zeros(L, 4, h), torch.ones(L, 4, h)], dim=1)
    s = {"module.embedding.word_embeddings.weight": torch.zeros(5, h),
         "module.decoder.final_layernorm.weight": torch.ones(h), "module.output_layer.weight": torch.zeros(5, h),
         "module.decoder.layers.self_attention.linear_qkv.weight": torch.from_numpy(qkv),
         "module.decoder.layers.self_attention.linear_proj.weight": torch.zeros(L, h, h),
         "module.decoder.layers.self_attention.linear_qkv.layer_norm_weight": torch.ones(L, h),
         "module.decoder.layers.mlp.linear_fc1.layer_norm_weight": torch.ones(L, h),
         "module.decoder.layers.mlp.linear_fc1.weight": fc1,
         "module.decoder.layers.mlp.linear_fc2.weight": torch.zeros(L, h, 4)}  # fmt: skip
    out = to_hf(s)
    q = out["model.layers.1.self_attn.q_proj.weight"]
    assert q.shape == (h, h)
    assert float(q[3 * hd, 0]) == 100 + 30 + 0  # layer 1, head 3, q
    assert float(out["model.layers.1.self_attn.k_proj.weight"][5 * hd, 0]) == 100 + 50 + 1
    assert float(out["model.layers.0.self_attn.v_proj.weight"][7 * hd + 1, 0]) == 70 + 2
    assert float(out["model.layers.0.mlp.gate_proj.weight"].sum()) == 0
    assert float(out["model.layers.0.mlp.up_proj.weight"].min()) == 1


def test_build_prompts_stay_within_type_and_sex_and_use_each_query_once():
    rng = np.random.default_rng(0)
    n = 60
    cells = pl.DataFrame({"cell": range(n), "donor_id": [f"d{i}" for i in range(n)],
                          "age": rng.integers(20, 90, n).tolist(), "sex": ["f", "m"] * (n // 2),
                          "ctype": ["naive"] * (n // 2) + ["cm"] * (n // 2), "decade": [0] * n})  # fmt: skip
    toks = [[2, 1000 + i, 3] for i in range(n)]
    vocab = {str(t): 5000 + t for t in range(-1500, 1500)}
    ps = build_prompts(cells, toks, vocab, 20, seed=1)
    assert len(ps) == 20
    assert len({p.query_cell for p in ps}) == 20
    for p in ps:
        a, dt, b = p.context[1] - 1000, p.context[3] - 5000, p.context[5] - 1000
        q = p.query_cell
        assert {cells["sex"][a], cells["sex"][b]} == {cells["sex"][q]}
        assert {cells["ctype"][a], cells["ctype"][b]} == {cells["ctype"][q]}
        assert dt == 12 * (cells["age"][b] - cells["age"][a]) > 0
        assert p.t_true == 12 * (cells["age"][q] - cells["age"][b])


def test_compare_finds_signal_and_not_noise():
    rng = np.random.default_rng(0)
    y = rng.normal(size=400)
    good = compare(y + rng.normal(size=400), y, n_boot=200, n_perm=200, seed=0)
    assert good["rho"] > 0.5 and good["ci_lo"] < good["rho"] < good["ci_hi"] and good["p_perm"] < 0.01
    assert good["sign_agree_top"] > good["sign_agree_top_null"]
    bad = compare(rng.normal(size=400), y, n_boot=200, n_perm=200, seed=0)
    assert bad["p_perm"] > 0.01 and abs(bad["null_mean"]) < 0.05


def test_time_reader_cache_matches_full_forward():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from rna_age.train.maxtoki_cd4 import TimeReader

    vocab = {"<pad>": 0, "<mask>": 1, "<bos>": 2, "<eos>": 3, **{f"g{i}": 4 + i for i in range(20)},
             **{str(t): 24 + t + 5 for t in range(-5, 5)}, "<boq>": 34, "<eoq>": 35}  # fmt: skip
    cfg = transformers.LlamaConfig(vocab_size=36, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128)  # fmt: skip
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).eval()
    reader = TimeReader(model, vocab, batch=2)
    context = [2, 5, 6, 7, 3, 30, 2, 8, 9, 3]
    queries = [[2, 10, 11, 12, 3], [2, 11, 12, 3], [2, 10, 12, 3], [2, 10, 11, 3]]
    t, m = reader(context, queries)
    for q, tq, mq in zip(queries, t, m, strict=True):
        ids = torch.tensor([context + [34, *q, 35]])
        with torch.no_grad():
            logits = model(input_ids=ids).logits[:, -1]
        tf, mf = reader.expected_time(logits)
        assert abs(float(tf[0]) - tq) < 1e-4 and abs(float(mf[0]) - mq) < 1e-5
    p = torch.full((1, 36), -1e9)
    p[0, 24 + 3 + 5] = 0.0  # all mass on the time token "3"
    tt, mm = reader.expected_time(p)
    assert abs(float(tt[0]) - 3.0) < 1e-6 and float(mm[0]) > 0.999
