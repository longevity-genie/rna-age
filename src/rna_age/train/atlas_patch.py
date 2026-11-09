"""Memory fix for BulkFormer-Atlas attention on 11 GB cards.

Atlas passes an explicit boolean key mask to ``scaled_dot_product_attention`` even when every position is
valid, which forces the full-matrix math kernel (~2 GB per layer at batch 4 × 4,097 tokens). With a fixed
panel all positions are valid, so we pass ``attn_mask=None`` in that case: mathematically identical, and SDPA
can use its memory-efficient kernel. Masks with padding keep the original path.
"""

from __future__ import annotations

import torch.nn.functional as F


def patch_atlas_attention() -> None:
    from bulkformer import atlas

    block_cls = next(
        c
        for c in vars(atlas).values()
        if isinstance(c, type) and hasattr(c, "forward") and "qkv" in c.__init__.__code__.co_names
    )
    if getattr(block_cls, "_rna_age_patched", False):
        return
    original = block_cls.forward

    def forward(self, x, attention_mask):
        if not bool(attention_mask.all()):
            return original(self, x, attention_mask)
        batch, length, _ = x.shape
        residual = x
        qkv = self.qkv(self.attn_norm(x)).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.dim)
        x = residual + self.dropout(self.out_proj(attended))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))

    block_cls.forward = forward
    block_cls._rna_age_patched = True
