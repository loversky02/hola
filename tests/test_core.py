"""Offline unit checks for the HOLA core (no training, no GPU, seconds to run).

Run:  .m15venv/bin/python hola/tests/test_core.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from hola.backbones import BACKBONES
from hola.cache import HOLACache
from hola.model import HOLALM

torch.manual_seed(0)
PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  ok  {name}")


# ---------------------------------------------------------------- backbones
def test_backbones():
    B, L, D, H, dh = 2, 12, 32, 4, 8
    x = torch.randn(B, L, D)
    for key, cls in BACKBONES.items():
        m = cls(D, H, dh)
        o, score, q, k, v = m(x)
        check(f"{key}: readout shape", o.shape == (B, H, L, dh))
        check(f"{key}: score shape", score.shape == (B, H, L))
        check(f"{key}: score >= 0", bool((score >= 0).all()))
        check(f"{key}: keys unit-L2", torch.allclose(k.norm(dim=-1),
              torch.ones(B, H, L), atol=1e-3))


# ------------------------------------------------- cache: the core mechanism
def test_cache_importance_beats_recency():
    """A high-surprise key stored early must remain retrievable long after, which
    a recency window would have evicted. This is the paper's central claim,
    isolated at the module level with hand-built tensors (no training)."""
    B, H, dh, L = 1, 1, 4, 6
    e1 = torch.tensor([1., 0, 0, 0])
    q = torch.zeros(B, H, L, dh)
    k = torch.zeros(B, H, L, dh)
    v = torch.zeros(B, H, L, dh)
    score = torch.zeros(B, H, L)

    vA = torch.tensor([9., 9, 9, 9])
    k[0, 0, 0] = e1;             v[0, 0, 0] = vA;   score[0, 0, 0] = 10.0   # early, surprising
    k[0, 0, 1] = torch.tensor([0., 1, 0, 0]); v[0, 0, 1] = 0.1; score[0, 0, 1] = 0.1
    k[0, 0, 2] = torch.tensor([0., 0, 1, 0]); v[0, 0, 2] = 0.1; score[0, 0, 2] = 0.1
    k[0, 0, 3] = torch.tensor([0., 0, 0, 1]); v[0, 0, 3] = 0.1; score[0, 0, 3] = 0.1
    q[0, 0, 5] = e1                                                        # ask for the early key

    imp = HOLACache(dh, H, w=1, chunk=2, mode="importance")
    rec = HOLACache(dh, H, w=1, chunk=2, mode="recency")
    with torch.no_grad():
        o_imp = imp(q, k, v, score)[0, 0, 5].mean().item()
        o_rec = rec(q, k, v, score)[0, 0, 5].mean().item()
    check("importance retrieves early surprising value", o_imp > 4.0)
    check("recency evicts it (fails)", o_rec < 1.0)
    check("importance >> recency", o_imp > 5 * max(o_rec, 1e-3))


def test_cache_sharpening_and_sink():
    """RMSNorm-gamma read = near-argmax: a query picks the single matching value,
    not a soft blend. And a high sink logit drains weight from the cache."""
    B, H, dh, L = 1, 1, 4, 6
    q = torch.zeros(B, H, L, dh); k = torch.zeros(B, H, L, dh)
    v = torch.zeros(B, H, L, dh); score = torch.zeros(B, H, L)
    k[0, 0, 0] = torch.tensor([1., 0, 0, 0]); v[0, 0, 0] = torch.tensor([9., 9, 9, 9]); score[0, 0, 0] = 10.
    k[0, 0, 1] = torch.tensor([0., 1, 0, 0]); v[0, 0, 1] = torch.tensor([-9., -9, -9, -9]); score[0, 0, 1] = 10.
    q[0, 0, 5] = torch.tensor([1., 0, 0, 0])   # matches key 0 -> want +9, not a blend (~0)
    c = HOLACache(dh, H, w=2, chunk=2, mode="importance")
    with torch.no_grad():
        near_argmax = c(q, k, v, score)[0, 0, 5].mean().item()
    check("near-argmax retrieval (not soft blend)", near_argmax > 6.0)
    with torch.no_grad():
        c.sink_logit.data.fill_(20.0)          # sink dominates -> cache stays quiet
        with_sink = c(q, k, v, score)[0, 0, 5].abs().mean().item()
    check("high sink logit drains cache output", with_sink < 0.1)


# --------------------------------------------------------- end-to-end grad
def test_model_overfits_tiny_batch():
    vocab, B, L = 12, 4, 16
    torch.manual_seed(1)
    idx = torch.randint(0, vocab, (B, L))
    tgt = torch.roll(idx, -1, dims=1)
    m = HOLALM(vocab, d_model=64, n_layers=2, n_heads=4, d_head=16,
               backbone="gdn", cache_mode="importance", w=8, chunk=8)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    lossf = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(40):
        opt.zero_grad()
        out = m(idx)
        loss = lossf(out.reshape(-1, vocab), tgt.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    check("model forward shape", True)
    check(f"loss drops ({losses[0]:.2f} -> {losses[-1]:.2f})", losses[-1] < 0.5 * losses[0])
    grads = [p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()]
    check("all grads finite", all(bool(g) for g in grads))


if __name__ == "__main__":
    print("== backbones =="); test_backbones()
    print("== cache mechanism =="); test_cache_importance_beats_recency()
    print("== cache sharpening + sink =="); test_cache_sharpening_and_sink()
    print("== end-to-end =="); test_model_overfits_tiny_batch()
    print(f"\nALL PASS ({len(PASS)} checks)")
