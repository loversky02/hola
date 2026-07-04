"""Forget-probe: does surprise (beta*||e||) mark what the recurrent state forgets,
and does an importance cache cover those failures better than a recency window?

The paper *asserts* the cache should store "surprising items the compressed state
couldn't predict." We test it directly on a trained plain-GDN (no cache):

  1. For every queried fact, read its write-time surprise score and whether GDN
     alone recalls it. -> AUC(surprise -> forgotten).
  2. For the SAME failures, check which the two caches would have retained.
     -> failure coverage: P(retained | forgotten) for importance vs recency.

This needs only one small model ($0). A clean win for importance here is the
mechanistic 'why' behind exp1.
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hola.data import vocab_size, make_mqar_batch
from hola.model import HOLALM
from hola.train import train_mqar


def auc(pos, neg):
    """P(a random 'pos' scores above a random 'neg'); pos=forgotten, neg=kept."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def collect(model, task, layer, w, chunk, device, n_batches=16, batch=64, seed=4242):
    g = torch.Generator().manual_seed(seed)
    rows = []  # (surprise, forgotten, imp_retained, rec_retained)
    model.eval()
    for _ in range(n_batches):
        idx, lab, meta = make_mqar_batch(batch, generator=g, return_meta=True, **task)
        out = model(idx.to(device))
        pred = out.argmax(-1).cpu()
        score = model.blocks[layer].last_score.mean(1).cpu()  # B L  (mean over heads)
        svp, qkp = meta["study_val_pos"], meta["query_key_pos"]
        for b in range(batch):
            sc = score[b]
            for j in range(svp.shape[1]):
                sp, qp = int(svp[b, j]), int(qkp[b, j])
                forgotten = int(pred[b, qp].item() != lab[b, qp].item())
                s0 = (qp // chunk) * chunk                      # cache 'past' for this query
                surprise = float(sc[sp])
                # importance retains sp iff it is top-w by surprise among tokens < s0
                past = sc[:s0]
                imp_ret = int(sp < s0 and (past >= surprise).sum().item() <= w)
                rec_ret = int(s0 - w <= sp < s0)
                rows.append((surprise, forgotten, imp_ret, rec_ret))
    model.train()
    return np.array(rows, float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--npairs", type=int, default=40)   # > state capacity -> real forgetting
    p.add_argument("--nquery", type=int, default=6)
    p.add_argument("--gap", type=int, default=16)
    p.add_argument("--nkeys", type=int, default=96)
    p.add_argument("--nvalues", type=int, default=32)
    p.add_argument("--w", type=int, default=24)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--dmodel", type=int, default=64)
    p.add_argument("--nheads", type=int, default=4)  # inner=32 learns; npairs>capacity -> forgetting
    p.add_argument("--dhead", type=int, default=8)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "results"))
    args = p.parse_args()
    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    task = dict(n_pairs=args.npairs, n_query=args.nquery, gap=args.gap,
                n_keys=args.nkeys, n_values=args.nvalues)
    V = vocab_size(args.nkeys, args.nvalues)
    torch.manual_seed(args.seed)
    model = HOLALM(V, d_model=args.dmodel, n_layers=2, n_heads=args.nheads,
                   d_head=args.dhead, backbone="gdn", cache_mode=None,
                   w=args.w, chunk=args.chunk)
    print(f"Training plain GDN ({model.num_params()/1e3:.0f}k params) on a "
          f"deliberately hard MQAR (npairs={args.npairs}, gap={args.gap})...")
    train_mqar(model, task, steps=args.steps, batch=args.batch, device=args.device, seed=args.seed)

    rows = collect(model, task, args.layer, args.w, args.chunk, args.device)
    surprise, forgot, imp, rec = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    kept_mask = forgot == 0

    a = auc(surprise[forgot == 1], surprise[forgot == 0])
    cov_imp = imp[forgot == 1].mean() if (forgot == 1).any() else float("nan")
    cov_rec = rec[forgot == 1].mean() if (forgot == 1).any() else float("nan")
    ret_imp_kept = imp[kept_mask].mean()

    summary = {
        "n_facts": int(len(rows)),
        "gdn_forget_rate": float(forgot.mean()),
        "auc_surprise_predicts_forgetting": float(a),
        "failure_coverage_importance": float(cov_imp),
        "failure_coverage_recency": float(cov_rec),
        "importance_retention_on_remembered": float(ret_imp_kept),
        "args": vars(args),
    }
    (outdir / "probe_results.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FORGET-PROBE ===")
    print(f"  GDN forgets {summary['gdn_forget_rate']*100:.1f}% of queried facts")
    print(f"  AUC(surprise -> forgotten)     : {a:.3f}   "
          f"({'surprise predicts forgetting' if a>0.55 else 'weak/None' if a>0.45 else 'INVERSE'})")
    print(f"  failure coverage  importance   : {cov_imp*100:.1f}%")
    print(f"  failure coverage  recency      : {cov_rec*100:.1f}%")
    print(f"  (importance retains {ret_imp_kept*100:.1f}% of already-remembered facts too)")
    plot(surprise, forgot, cov_imp, cov_rec, outdir)
    print(f"\nsaved -> {outdir}/probe_results.json  and  probe_plot.png")


def plot(surprise, forgot, cov_imp, cov_rec, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(surprise[forgot == 0], bins=30, alpha=0.6, label="remembered", color="#2a7de1", density=True)
    ax[0].hist(surprise[forgot == 1], bins=30, alpha=0.6, label="forgotten", color="#e0442b", density=True)
    ax[0].set_xlabel("write-time surprise  beta*||e||"); ax[0].set_ylabel("density")
    ax[0].set_title("Is what GDN forgets more surprising?"); ax[0].legend()
    ax[1].bar(["importance", "recency"], [cov_imp, cov_rec], color=["#2a7de1", "#e06c00"])
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("P(fact retained | GDN forgot it)")
    ax[1].set_title("Which cache covers the state's failures?")
    for i, v in enumerate([cov_imp, cov_rec]):
        ax[1].text(i, v + 0.02, f"{v*100:.0f}%", ha="center")
    plt.tight_layout(); plt.savefig(outdir / "probe_plot.png", dpi=150)


if __name__ == "__main__":
    main()
