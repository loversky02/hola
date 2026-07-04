"""Money plot: recall vs memory load for GDN vs HOLA(importance) vs HOLA(recency).

Reproduces the paper's central claim on MQAR at tiny scale ($0, CPU/MPS):
importance-based exact memory holds early, surprising facts that both a
compressive state (GDN) and a recency window (HOLA+recency) lose.

Usage:
  .m15venv/bin/python hola/experiments/exp1_importance_vs_recency.py \
      --steps 1500 --npairs 8 16 32 --device cpu
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from hola.data import vocab_size
from hola.model import HOLALM
from hola.train import train_mqar

CONFIGS = [
    ("GDN (no cache)",     None),
    ("HOLA (importance)",  "importance"),
    ("HOLA (recency)",     "recency"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--npairs", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--nquery", type=int, default=4)
    p.add_argument("--gap", type=int, default=16)
    p.add_argument("--nkeys", type=int, default=64)
    p.add_argument("--nvalues", type=int, default=32)
    p.add_argument("--w", type=int, default=32)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--dmodel", type=int, default=64)
    p.add_argument("--nheads", type=int, default=4)
    p.add_argument("--dhead", type=int, default=8)   # inner=nheads*dhead=32 (learns); capacity ~inner
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "results"))
    args = p.parse_args()

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    V = vocab_size(args.nkeys, args.nvalues)
    results = {name: [] for name, _ in CONFIGS}

    for npairs in args.npairs:
        task = dict(n_pairs=npairs, n_query=min(args.nquery, npairs), gap=args.gap,
                    n_keys=args.nkeys, n_values=args.nvalues)
        seqlen = 2 * npairs + args.gap + 2 * min(args.nquery, npairs)
        print(f"\n=== n_pairs={npairs}  (seqlen={seqlen}, w={args.w}) ===")
        for name, mode in CONFIGS:
            torch.manual_seed(args.seed)
            model = HOLALM(V, d_model=args.dmodel, n_layers=2, n_heads=args.nheads,
                           d_head=args.dhead, backbone="gdn", cache_mode=mode,
                           w=args.w, chunk=args.chunk)
            print(f"  [{name}]  params={model.num_params()/1e3:.0f}k")
            r = train_mqar(model, task, steps=args.steps, batch=args.batch,
                           device=args.device, seed=args.seed)
            results[name].append({"n_pairs": npairs, "acc": r["final_acc"]})
            print(f"  -> recall {r['final_acc']:.3f}")

    (outdir / "exp1_results.json").write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=2))
    plot(results, args, outdir)
    print(f"\nsaved -> {outdir}/exp1_results.json  and  exp1_money_plot.png")


def plot(results, args, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.2, 4.3))
    styles = {"GDN (no cache)": ("o", "#888888"),
              "HOLA (importance)": ("s", "#2a7de1"),
              "HOLA (recency)": ("^", "#e06c00")}
    for name, pts in results.items():
        xs = [d["n_pairs"] for d in pts]; ys = [d["acc"] for d in pts]
        mk, c = styles.get(name, ("o", None))
        plt.plot(xs, ys, marker=mk, color=c, label=name, linewidth=2, markersize=7)
    plt.axhline(1.0, ls=":", c="#bbb", lw=1)
    plt.xlabel("# key-value pairs to remember (memory load)")
    plt.ylabel("MQAR recall accuracy")
    plt.title("HOLA: importance-based exact memory holds what the state forgets")
    plt.ylim(0, 1.05); plt.legend(); plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "exp1_money_plot.png", dpi=150)


if __name__ == "__main__":
    main()
