"""Tiny language model: token embed -> N x (linear-attn + HOLA cache) block -> head.

The block sums the compressive readout (o_linear) and the exact cache readout
(o_cache) before the output projection — the "complementary learning systems"
combination of a lossy fast state and an exact slow memory.
"""

import torch
import torch.nn as nn

from .backbones import BACKBONES
from .cache import HOLACache


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, backbone, cache_mode, w, chunk):
        super().__init__()
        self.dh = d_head
        self.h = n_heads
        inner = n_heads * d_head
        self.ln1 = nn.LayerNorm(d_model)
        self.mixer = BACKBONES[backbone](d_model, n_heads, d_head)
        self.cache = HOLACache(d_head, n_heads, w=w, chunk=chunk,
                               mode=(cache_mode or "none"))
        self.o_proj = nn.Linear(inner, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        o_lin, score, q, k, v = self.mixer(self.ln1(x))
        self.last_score = score.detach()          # stashed for the forget-probe
        self.last_k = k.detach()
        o_cache = self.cache(q, k, v, score)      # zeros if cache_mode == 'none'
        o = o_lin + o_cache                        # CLS combination
        B, H, L, dh = o.shape
        o = o.transpose(1, 2).reshape(B, L, H * dh)
        x = x + self.o_proj(o)
        x = x + self.mlp(self.ln2(x))
        return x


class HOLALM(nn.Module):
    def __init__(self, vocab, d_model=128, n_layers=2, n_heads=4, d_head=32,
                 backbone="gdn", cache_mode="importance", w=16, chunk=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_head, backbone, cache_mode, w, chunk)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.cache_mode = cache_mode
        self.backbone = backbone

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
