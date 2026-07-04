"""Beyond the paper: does the importance cache generalise past Gated DeltaNet?

The paper tests only GDN. Here we run the same mechanistic recall benchmark as
`exp2` but drive the state with three different linear-attn recurrences — GDN
(gated delta), DeltaNet (delta, no decay), GLA (additive, no delta rule) — and ask:
does bolting the exact importance cache on top lift recall over the state alone,
for *every* backbone? Since the surprise score `‖v − k·S‖` is defined the same way
for all three, the cache should transfer. Deterministic, no training, seconds.

Usage:
  .m15venv/bin/python hola/experiments/exp3_anybackbone.py
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def unit(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def scan(backbone, K, V, alpha):
    """Return (final state S, per-token surprise score) for one backbone."""
    T, d = K.shape
    S = torch.zeros(d, d)
    score = torch.zeros(T)
    for t in range(T):
        kt, vt = K[t], V[t]
        a = 1.0 if backbone == "deltanet" else alpha
        pred = a * (kt @ S)                        # state's guess for v
        e = vt - pred
        if backbone == "gla":
            S = alpha * S + torch.outer(kt, vt)    # additive write (no delta correction)
        else:
            S = a * S + torch.outer(kt, e)         # (gated) delta write
        score[t] = e.norm()
    return S, score


def trial(backbone, n_facts, n_noise, w, d, codebook, alpha, g):
    M = codebook.shape[0]
    fk = unit(torch.randn(n_facts, d, generator=g))
    labels = torch.randint(M, (n_facts,), generator=g)
    fv = codebook[labels]
    nk = unit(torch.randn(n_noise, d, generator=g))
    nv = 0.03 * torch.randn(n_noise, d, generator=g)
    K = torch.cat([fk, nk], 0)
    V = torch.cat([fv, nv], 0)
    S, score = scan(backbone, K, V, alpha)

    def decode(o):
        return (unit(o) @ codebook.T).argmax(-1)

    state_acc = (decode(fk @ S) == labels).float().mean().item()
    idx = score.topk(min(w, K.shape[0])).indices     # importance: top-w by surprise
    Kc, Vc = unit(K[idx]), V[idx]
    pick = (unit(fk) @ Kc.T).argmax(-1)
    cache_acc = (decode(Vc[pick]) == labels).float().mean().item()
    return state_acc, cache_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", nargs="+", default=["gdn", "deltanet", "gla"])
    p.add_argument("--facts", type=int, default=32)
    p.add_argument("--noise", type=int, default=64)
    p.add_argument("--w", type=int, default=32)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.98)
    p.add_argument("--codebook", type=int, default=64)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "results"))
    args = p.parse_args()
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(args.seed)
    codebook = unit(torch.randn(args.codebook, args.d, generator=g))
    results = {}
    for bk in args.backbones:
        accs = np.array([trial(bk, args.facts, args.noise, args.w, args.d, codebook, args.alpha, g)
                         for _ in range(args.trials)])
        m = accs.mean(0)
        results[bk] = {"state_only": float(m[0]), "with_cache": float(m[1]),
                       "lift": float(m[1] - m[0])}
        print(f"  {bk:>9} | state {m[0]:.3f} | +importance cache {m[1]:.3f} | lift {m[1]-m[0]:+.3f}")

    (outdir / "exp3_results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    plot(results, args, outdir)
    print(f"\nsaved -> {outdir}/exp3_results.json  and  exp3_anybackbone.png")


def plot(results, args, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bks = list(results.keys())
    state = [results[b]["state_only"] for b in bks]
    cache = [results[b]["with_cache"] for b in bks]
    x = np.arange(len(bks)); wd = 0.36
    plt.figure(figsize=(6.6, 4.3))
    plt.bar(x - wd / 2, state, wd, label="state only", color="#888")
    plt.bar(x + wd / 2, cache, wd, label="+ HOLA importance cache", color="#2a7de1")
    for i in range(len(bks)):
        plt.text(x[i] - wd / 2, state[i] + 0.01, f"{state[i]:.2f}", ha="center", fontsize=9)
        plt.text(x[i] + wd / 2, cache[i] + 0.01, f"{cache[i]:.2f}", ha="center", fontsize=9)
    plt.xticks(x, [b.upper() for b in bks]); plt.ylim(0, 1.05)
    plt.ylabel(f"exact recall ({args.facts} facts, w={args.w})")
    plt.title("The importance cache lifts recall across linear-attn backbones")
    plt.legend(); plt.grid(axis="y", alpha=0.25); plt.tight_layout()
    plt.savefig(outdir / "exp3_anybackbone.png", dpi=150)


if __name__ == "__main__":
    main()
