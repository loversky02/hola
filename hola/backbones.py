"""Linear-attention backbones (the compressive 'neocortex' state).

Each backbone is a recurrent scan that maintains a bounded state S and, for every
token, exposes a *surprise* score used by the HOLA cache to decide what to keep:

    score_t = beta_t * || e_t ||_2 ,   e_t = v_t - (state's prediction of v_t)

This is the delta-rule write magnitude from the paper (Sec. on prediction
residuals): "a token's entire effect on S is this one rank-1 matrix" of Frobenius
norm beta*||e||. It measures how much a token *changed* the compressed state, i.e.
how surprising / hard-to-predict it was.

All scans are plain PyTorch loops over the sequence — slow but exact, CPU/MPS
friendly, and enough for the small associative-recall regime we study.
"""

import torch
import torch.nn as nn


def l2norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class LinearAttnBase(nn.Module):
    """Shared projections + forward wrapper. Subclasses implement `_scan`."""

    def __init__(self, d_model, n_heads, d_head=None, conv_size=4):
        super().__init__()
        self.h = n_heads
        self.dh = d_head or (d_model // n_heads)
        inner = self.h * self.dh
        # Depthwise causal short conv: lets each position mix the last few tokens
        # so a value position can 'see' its key and bind the association. Without
        # this, linear-attn models famously cannot solve MQAR (Based / Zoology).
        self.conv_size = conv_size
        self.conv = nn.Conv1d(d_model, d_model, conv_size, groups=d_model,
                              padding=conv_size - 1, bias=True)
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads)  # decay-gate logits
        self.b_proj = nn.Linear(d_model, n_heads)  # write-strength logits
        # Init decay ~1 (near-lossless state) — essential for recall: a per-step
        # decay of 0.88 would erase the state as 0.88^t across the sequence.
        nn.init.constant_(self.a_proj.bias, 6.0)  # sigmoid(6) ~ 0.9975
        nn.init.constant_(self.b_proj.bias, 0.0)

    def _project(self, x):
        B, L, D = x.shape
        H, dh = self.h, self.dh
        x = self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2)  # causal short conv
        q = l2norm(self.q_proj(x).view(B, L, H, dh)).transpose(1, 2)  # B H L dh
        k = l2norm(self.k_proj(x).view(B, L, H, dh)).transpose(1, 2)  # unit L2
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)
        a = torch.sigmoid(self.a_proj(x)).transpose(1, 2)  # B H L in (0,1)
        b = torch.sigmoid(self.b_proj(x)).transpose(1, 2)  # B H L in (0,1)
        return q, k, v, a, b

    def _scan(self, q, k, v, a, b):
        raise NotImplementedError

    def forward(self, x):
        q, k, v, a, b = self._project(x)
        o, score = self._scan(q, k, v, a, b)
        # q,k are unit-L2 (the cache re-normalises with its own RMSNorm-gamma);
        # v is raw. The token's own query q addresses the exact cache.
        return o, score, q, k, v


class GatedDeltaNet(LinearAttnBase):
    """Gated DeltaNet (the paper's backbone, "All experiments use GDN").

        v_hat_t = alpha_t * (k_t^T S_{t-1})            # state predicts v
        e_t     = v_t - v_hat_t                         # prediction residual
        S_t     = alpha_t * S_{t-1} + beta_t * k_t (x) e_t
        o_t     = q_t^T S_t
    """

    def _scan(self, q, k, v, a, b):
        B, H, L, dh = q.shape
        S = q.new_zeros(B, H, dh, dh)  # keys x values
        outs, scores = [], []
        for t in range(L):
            kt, vt = k[:, :, t], v[:, :, t]          # B H dh
            at = a[:, :, t, None, None]              # B H 1 1
            bt = b[:, :, t, None, None]
            vhat = at[..., 0] * torch.einsum("bhk,bhkv->bhv", kt, S)
            e = vt - vhat                            # residual, B H dh
            S = at * S + bt * torch.einsum("bhk,bhv->bhkv", kt, e)
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
            scores.append(b[:, :, t] * e.norm(dim=-1))  # beta * ||e||
        return torch.stack(outs, 2), torch.stack(scores, 2)


class DeltaNet(LinearAttnBase):
    """DeltaNet: Gated DeltaNet with the decay gate pinned to 1 (no forgetting)."""

    def _scan(self, q, k, v, a, b):
        B, H, L, dh = q.shape
        S = q.new_zeros(B, H, dh, dh)
        outs, scores = [], []
        for t in range(L):
            kt, vt = k[:, :, t], v[:, :, t]
            bt = b[:, :, t, None, None]
            vhat = torch.einsum("bhk,bhkv->bhv", kt, S)  # alpha = 1
            e = vt - vhat
            S = S + bt * torch.einsum("bhk,bhv->bhkv", kt, e)
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
            scores.append(b[:, :, t] * e.norm(dim=-1))
        return torch.stack(outs, 2), torch.stack(scores, 2)


class GLA(LinearAttnBase):
    """Gated Linear Attention: additive write (no delta rule), scalar decay gate.

        S_t = alpha_t * S_{t-1} + k_t (x) v_t
        o_t = q_t^T S_t
    The surprise score reuses ||v - alpha*k^T S|| as a backbone-agnostic proxy for
    "what the compressed state failed to predict" (there is no delta residual here,
    so this tests whether HOLA's importance signal transfers beyond delta rules).
    """

    def _scan(self, q, k, v, a, b):
        B, H, L, dh = q.shape
        S = q.new_zeros(B, H, dh, dh)
        outs, scores = [], []
        for t in range(L):
            kt, vt = k[:, :, t], v[:, :, t]
            at = a[:, :, t, None, None]
            pred = at[..., 0] * torch.einsum("bhk,bhkv->bhv", kt, S)
            e = vt - pred
            S = at * S + torch.einsum("bhk,bhv->bhkv", kt, vt)
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
            scores.append(e.norm(dim=-1))
        return torch.stack(outs, 2), torch.stack(scores, 2)


BACKBONES = {"gdn": GatedDeltaNet, "deltanet": DeltaNet, "gla": GLA}
