"""MaxToki-217M (Theodoris lab, 2026; Apache-2.0) as a plain ``transformers`` Llama: weights, vocabulary, time decode.

The released temporal 217M checkpoint is the BioNeMo one (``MaxToki-217M-bionemo``: Megatron ``torch_dist``, vocabulary
23,277 = 20,275 gene/special tokens + time tokens -1500..1499 + ``<boq>``/``<eoq>``); the ``-HF`` folder holds the
stage-1 model only (vocabulary 20,275, no time tokens). The NVIDIA code needs NeMo, Megatron and TransformerEngine on
an A100; here the checkpoint is read without them and mapped to ``LlamaForCausalLM``:

- the ``.metadata`` pickle is read with an allow-list unpickler (torch DCP metadata classes only; megatron's save plan
  is replaced by a stub), each tensor chunk with ``torch.load(weights_only=True)``;
- Megatron → HF names as NeMo's exporter maps them (``linear_qkv`` is grouped per query group: q, k, v);
- the vocabulary comes from the checkpoint's ``context/io.json`` (plain JSON).

The time answer is MaxToki's headless decode: at ``<eoq>``, softmax over the vocabulary restricted to the time tokens
and renormalised, expected value (``MaxTokiFineTuneModel._headless_timelapse``).
"""

from __future__ import annotations

import io
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from rna_age.paths import INTERIM, download_dir

log = logging.getLogger(__name__)

CHECKPOINT = "hf/MaxToki-217M-bionemo"
REPO_RESOURCES = "repo/resources"
CONFIG = dict(hidden_size=1232, intermediate_size=2464, num_hidden_layers=11, num_attention_heads=8,
              num_key_value_heads=8, rms_norm_eps=1e-6, rope_theta=10000.0)  # fmt: skip


def maxtoki_dir() -> Path:
    return download_dir("maxtoki")


class _Stub:
    """Stands in for any class outside the allow-list that the metadata pickle names (megatron's save plan)."""

    def __init__(self, *a, **k) -> None:
        pass

    def __setstate__(self, state) -> None:
        self.__dict__["state"] = state


class _MetadataUnpickler(pickle.Unpickler):
    ALLOWED_MODULES = ("torch.distributed.checkpoint.",)
    ALLOWED = {("torch", "Size"), ("torch.serialization", "_get_layout"), ("collections", "OrderedDict"),
               ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "slice")}  # fmt: skip

    def find_class(self, module: str, name: str):
        import torch

        if module.startswith(self.ALLOWED_MODULES) or (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        if module == "torch" and isinstance(getattr(torch, name, None), torch.dtype):
            return getattr(torch, name)
        if module.startswith("megatron."):
            return _Stub
        raise pickle.UnpicklingError(f"refusing {module}.{name} in checkpoint metadata")


def load_dcp(weights: Path) -> dict:
    """Full tensors of a torch_dist checkpoint directory, assembled from their chunks (no megatron, no pickle code)."""
    import torch
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    with open(weights / ".metadata", "rb") as f:
        meta = _MetadataUnpickler(f).load()
    out: dict[str, torch.Tensor] = {}
    files: dict[str, bytes] = {}
    for idx, info in meta.storage_data.items():
        md = meta.state_dict_metadata[idx.fqn]
        if not isinstance(md, TensorStorageMetadata):
            continue  # _extra_state bytes (TransformerEngine fp8 state), not weights
        if info.relative_path not in files:
            files[info.relative_path] = (weights / info.relative_path).read_bytes()
        raw = files[info.relative_path][info.offset : info.offset + info.length]
        chunk = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
        full = out.setdefault(idx.fqn, torch.empty(tuple(md.size), dtype=md.properties.dtype))
        sl = tuple(slice(o, o + s) for o, s in zip(idx.offset, chunk.shape, strict=True))
        full[sl] = chunk
    return out


def to_hf(state: dict) -> dict:
    """Megatron GPT (stacked layers) → HF Llama state dict."""
    h, nh = CONFIG["hidden_size"], CONFIG["num_attention_heads"]
    hd, ng = h // nh, CONFIG["num_key_value_heads"]
    per = nh // ng
    s = {k.removeprefix("module."): v for k, v in state.items()}
    out = {"model.embed_tokens.weight": s["embedding.word_embeddings.weight"],
           "model.norm.weight": s["decoder.final_layernorm.weight"],
           "lm_head.weight": s["output_layer.weight"]}  # fmt: skip
    L = s["decoder.layers.self_attention.linear_qkv.weight"].shape[0]
    for i in range(L):
        p = f"model.layers.{i}."
        qkv = s["decoder.layers.self_attention.linear_qkv.weight"][i].reshape(ng, (per + 2) * hd, h)
        out[p + "self_attn.q_proj.weight"] = qkv[:, : per * hd].reshape(-1, h)
        out[p + "self_attn.k_proj.weight"] = qkv[:, per * hd : (per + 1) * hd].reshape(-1, h)
        out[p + "self_attn.v_proj.weight"] = qkv[:, (per + 1) * hd :].reshape(-1, h)
        out[p + "self_attn.o_proj.weight"] = s["decoder.layers.self_attention.linear_proj.weight"][i]
        out[p + "input_layernorm.weight"] = s["decoder.layers.self_attention.linear_qkv.layer_norm_weight"][i]
        out[p + "post_attention_layernorm.weight"] = s["decoder.layers.mlp.linear_fc1.layer_norm_weight"][i]
        gate, up = s["decoder.layers.mlp.linear_fc1.weight"][i].chunk(2, dim=0)
        out[p + "mlp.gate_proj.weight"] = gate
        out[p + "mlp.up_proj.weight"] = up
        out[p + "mlp.down_proj.weight"] = s["decoder.layers.mlp.linear_fc2.weight"][i]
    return {k: v.contiguous() for k, v in out.items()}


def token_dictionary() -> dict[str, int]:
    """The temporal model's vocabulary (token → id), from the checkpoint's io.json (JSON, read as data)."""
    cache = INTERIM / "maxtoki" / "token_dictionary_temporal.json"
    if cache.exists():
        return json.loads(cache.read_text())
    d = json.loads((maxtoki_dir() / CHECKPOINT / "context" / "io.json").read_text())["objects"]
    best = max((v for v in d.values() if isinstance(v, dict) and v.get("type", {}).get("name") == "dict"),
               key=lambda v: len(v.get("items", [])))  # fmt: skip
    vocab = {k[len("Key(key='") : -2]: int(v["value"]) for k, v in best["items"]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(vocab))
    return vocab


def gene_medians() -> dict[str, float]:
    return json.loads((maxtoki_dir() / REPO_RESOURCES / "gene_median_dictionary_v1.json").read_text())


def time_tokens(vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """(token ids, values) of the numeric time tokens."""
    pairs = []
    for k, v in vocab.items():
        try:
            pairs.append((v, float(int(k))))
        except ValueError:
            continue
    pairs.sort()
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def converted_weights() -> Path:
    """The checkpoint as an HF safetensors file under interim/ (converted once)."""
    out = INTERIM / "maxtoki" / "maxtoki-217M-temporal.safetensors"
    if not out.exists():
        from safetensors.torch import save_file

        state = to_hf(load_dcp(maxtoki_dir() / CHECKPOINT / "weights"))
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, str(out))
        log.info("converted %d tensors → %s", len(state), out)
    return out


def load_model(device: str = "cuda", dtype: str = "float16", rope_factor: float = 1.0):
    """LlamaForCausalLM with the temporal weights. ``rope_factor`` > 1 applies Llama-3 RoPE scaling (low 1, high 4,
    original 4096), as the BioNeMo config's ``scale_factor`` would; the checkpoint itself records 1.0."""
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaConfig, LlamaForCausalLM

    vocab = token_dictionary()
    rope = None
    if rope_factor != 1.0:
        rope = {"rope_type": "llama3", "factor": rope_factor, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 4096}  # fmt: skip
    cfg = LlamaConfig(vocab_size=len(vocab), max_position_embeddings=16384, rope_scaling=rope, tie_word_embeddings=False,
                      attention_bias=False, mlp_bias=False, pad_token_id=vocab["<pad>"], **CONFIG)  # fmt: skip
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    model.load_state_dict(load_file(str(converted_weights())), assign=True, strict=True)
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    model.model.rotary_emb = LlamaRotaryEmbedding(config=cfg)  # its buffers were created on meta: rebuild them
    return model.to(device=device, dtype=getattr(torch, dtype)).eval()
