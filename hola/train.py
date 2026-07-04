"""Training / eval loop for MQAR. Fresh random samples each step (learn the
algorithm, not memorise a fixed set)."""

import torch

from .data import make_mqar_batch, recall_accuracy


@torch.no_grad()
def evaluate(model, task, device, n_batches=6, batch=64, seed=9999):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    accs = []
    for _ in range(n_batches):
        idx, lab = make_mqar_batch(batch, generator=g, **task)
        out = model(idx.to(device))
        accs.append(recall_accuracy(out.cpu(), lab))
    model.train()
    return sum(accs) / len(accs)


def train_mqar(model, task, steps=1200, lr=3e-3, batch=32, device="cpu",
               seed=0, eval_every=300, verbose=True):
    model.to(device)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = torch.nn.CrossEntropyLoss(ignore_index=-100)
    hist = []
    for step in range(1, steps + 1):
        idx, lab = make_mqar_batch(batch, generator=g, **task)
        idx, lab = idx.to(device), lab.to(device)
        out = model(idx)
        loss = lossf(out.reshape(-1, out.size(-1)), lab.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            acc = evaluate(model, task, device)
            hist.append({"step": step, "loss": loss.item(), "acc": acc})
            if verbose:
                print(f"    step {step:>4} | loss {loss.item():.3f} | recall {acc:.3f}")
    return {"final_acc": hist[-1]["acc"], "history": hist}
