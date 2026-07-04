"""Mechanistic money plot (NO training, seconds, deterministic).

Isolates the memory MECHANISM from the confound of "did SGD learn to use it".
We write N random facts (high-surprise key->value pairs) into a delta-rule state,
padded by a haystack of low-surprise distractor tokens, then query every fact
(all non-recent). We compare exact-recall of three readouts on the SAME tokens:

  * state-only   : o = q^T S           (the compressive recurrent readout)
  * HOLA importance : top-w by surprise beta*||e||   (the paper's cache)
  * HOLA recency    : last-w tokens                  (a sliding window)

Expected and shown: the compressive state degrades with load (interference); the
recency window fails because the facts are early; the importance cache holds every
fact until the number of facts exceeds its capacity w. This is exactly the paper's
claim, measured directly.
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def unit(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def run_trial(n_facts, n_noise, w, d, codebook, g):
    """Return (state_acc, imp_acc, rec_acc) for one random trial."""
    M = codebook.shape[0]
    # facts: novel keys + codebook values (high surprise). noise: tiny-norm values.
    fk = unit(torch.randn(n_facts, d, generator=g))
    labels = torch.randint(M, (n_facts,), generator=g)
    fv = codebook[labels]
    nk = unit(torch.randn(n_noise, d, generator=g))
    nv = 0.03 * torch.randn(n_noise, d, generator=g)          # ~predictable -> low residual
    K = torch.cat([fk, nk], 0)                                 # facts first (early), noise after
    Vv = torch.cat([fv, nv], 0)
    T = K.shape[0]

    # delta-rule scan (alpha=1, beta=1): S_t = S_{t-1} + k_t (x) e_t ; score = ||e||
    S = torch.zeros(d, d)
    score = torch.zeros(T)
    for t in range(T):
        kt, vt = K[t], Vv[t]
        e = vt - kt @ S                                        # residual
        S = S + torch.outer(kt, e)
        score[t] = e.norm()

    def decode(o):                                             # nearest codebook value -> label
        return (unit(o) @ codebook.T).argmax(-1)

    # 1) state-only readout for each fact query
    o_state = fk @ S                                           # (n_facts, d)
    state_acc = (decode(o_state) == labels).float().mean().item()

    # 2/3) exact caches: pick w tokens, near-argmax retrieve
    def cache_acc(idx):
        Kc, Vc = unit(K[idx]), Vv[idx]                         # (w,d)
        logits = unit(fk) @ Kc.T                               # (n_facts, w) ~ cosine (near-argmax)
        pick = logits.argmax(-1)
        return (decode(Vc[pick]) == labels).float().mean().item()

    imp_idx = score.topk(min(w, T)).indices                   # importance: top-w by surprise
    rec_idx = torch.arange(T - min(w, T), T)                   # recency: last-w tokens
    return state_acc, cache_acc(imp_idx), cache_acc(rec_idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--facts", type=int, nargs="+", default=[2, 4, 8, 16, 24, 32, 48])
    p.add_argument("--noise", type=int, default=64)           # haystack size
    p.add_argument("--w", type=int, default=16)               # cache capacity
    p.add_argument("--d", type=int, default=32)               # state dim (per-head capacity ~ d)
    p.add_argument("--codebook", type=int, default=64)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "results"))
    args = p.parse_args()
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(args.seed)
    codebook = unit(torch.randn(args.codebook, args.d, generator=g))
    rows = {"state": [], "importance": [], "recency": []}
    for n in args.facts:
        accs = np.array([run_trial(n, args.noise, args.w, args.d, codebook, g)
                         for _ in range(args.trials)])
        m = accs.mean(0)
        rows["state"].append(m[0]); rows["importance"].append(m[1]); rows["recency"].append(m[2])
        print(f"  facts={n:>3} | state {m[0]:.3f} | importance {m[1]:.3f} | recency {m[2]:.3f}")

    out = {"args": vars(args), "facts": args.facts, "results": rows}
    (outdir / "exp2_results.json").write_text(json.dumps(out, indent=2))
    plot(args.facts, rows, args, outdir)
    print(f"\nsaved -> {outdir}/exp2_results.json  and  exp2_money_plot.png")


def plot(facts, rows, args, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.4, 4.4))
    plt.plot(facts, rows["importance"], "s-", color="#2a7de1", lw=2, ms=7, label="HOLA importance cache")
    plt.plot(facts, rows["state"], "o-", color="#888", lw=2, ms=6, label="compressive state only")
    plt.plot(facts, rows["recency"], "^-", color="#e06c00", lw=2, ms=7, label="HOLA recency window")
    plt.axvline(args.w, ls=":", c="#2a7de1", alpha=0.6)
    plt.text(args.w + 0.4, 0.05, f"cache size w={args.w}", color="#2a7de1", fontsize=8)
    plt.xlabel(f"# facts written (needles), haystack={args.noise} distractors")
    plt.ylabel("exact recall accuracy (all queries non-recent)")
    plt.title("What the recurrent state forgets, the importance cache keeps")
    plt.ylim(0, 1.05); plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(outdir / "exp2_money_plot.png", dpi=150)


if __name__ == "__main__":
    main()
