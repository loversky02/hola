"""HOLA cache — the 'hippocampus': a bounded, exact key-value memory.

Two ideas from the paper, both implemented here:

1. IMPORTANCE-BASED SELECTION.  Instead of keeping the most *recent* tokens
   (a sliding window), keep the top-`w` tokens by surprise score beta*||e|| — the
   items the compressed state could *not* predict.  `mode='recency'` is the
   ablation baseline (position-based eviction) the paper compares against.

2. DECOUPLED RMSNorm-gamma READ.  The cache read normalises q and k with a
   learnable-gamma RMSNorm so that ||q~|| ~ ||k~|| ~ sqrt(d).  Logits then scale
   like sqrt(d)*cos instead of ~0.83*cos, turning a near-uniform soft average into
   *near-argmax* retrieval.  Crucially this sharpening acts ONLY on the cache read
   and is decoupled from the state-update path (which keeps unit-L2 q/k).

Read set per chunk = {top-w cache from the past} ∪ {current chunk, causal} ∪ {sink},
matching the paper's "w + C + 1" bounded read.  The sink is a learnable null logit
with zero value: when nothing matches, weight flows to the sink and the cache
contributes ~0, deferring to the recurrent state instead of hallucinating a match.
"""

import torch
import torch.nn as nn


class HOLACache(nn.Module):
    def __init__(self, d_head, n_heads, w=16, chunk=16, mode="importance", eps=1e-6):
        super().__init__()
        assert mode in ("importance", "recency", "none")
        self.dh = d_head
        self.h = n_heads
        self.w = w
        self.chunk = chunk
        self.mode = mode
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_heads, d_head))  # RMSNorm-gamma
        self.sink_logit = nn.Parameter(torch.zeros(n_heads))     # null attention sink

    def _rmsnorm_g(self, x):
        # x: B H T dh  ->  scaled to ||.|| ~ sqrt(dh) * gamma  (the sharpening)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.gamma.view(1, self.h, 1, self.dh)

    def _select_past(self, score, k, v, s):
        """Return (K_cache, V_cache) chosen from tokens strictly before index s."""
        B, H = score.shape[:2]
        m = min(self.w, s)
        if self.mode == "importance":
            idx = score[:, :, :s].topk(m, dim=-1).indices           # B H m
        else:  # recency: the m most recent tokens before s
            idx = torch.arange(s - m, s, device=score.device).view(1, 1, m).expand(B, H, m)
        idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, self.dh)
        return torch.gather(k[:, :, :s], 2, idx_e), torch.gather(v[:, :, :s], 2, idx_e)

    def forward(self, q, k, v, score):
        """q,k,v: B H L dh ; score: B H L  ->  o_cache: B H L dh."""
        if self.mode == "none":
            return torch.zeros_like(q)
        B, H, L, dh = q.shape
        qn, kn = self._rmsnorm_g(q), self._rmsnorm_g(k)
        out = torch.zeros_like(q)
        C = self.chunk
        for s in range(0, L, C):
            e = min(s + C, L)
            Tq = e - s
            q_c = qn[:, :, s:e]
            k_in, v_in = kn[:, :, s:e], v[:, :, s:e]
            if s > 0:
                k_cache, v_cache = self._select_past(score, kn, v, s)
            else:
                k_cache = kn[:, :, :0]
                v_cache = v[:, :, :0]
            mc = k_cache.shape[2]
            K = torch.cat([k_cache, k_in], dim=2)   # B H (mc+Tq) dh
            V = torch.cat([v_cache, v_in], dim=2)
            logits = torch.einsum("bhqd,bhkd->bhqk", q_c, K)  # sqrt(d)*cos, no 1/sqrt(d)
            # cache entries always visible; within-chunk keys are causal.
            causal = torch.triu(q.new_full((Tq, Tq), float("-inf")), diagonal=1)
            mask = q.new_zeros(Tq, mc + Tq)
            mask[:, mc:] = causal
            logits = logits + mask
            sink = self.sink_logit.view(1, H, 1, 1).expand(B, H, Tq, 1)
            weights = torch.cat([logits, sink], dim=-1).softmax(dim=-1)
            out[:, :, s:e] = torch.einsum("bhqk,bhkd->bhqd", weights[..., :-1], V)
        return out
