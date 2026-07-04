"""MQAR — Multi-Query Associative Recall (Zoology / Based benchmark).

The clean synthetic test for whether a sequence model can *recall* stored
associations — exactly what HOLA targets. Each sample:

    study:  k1 v1 k2 v2 ... kN vN          (N key->value pairs, introduced once)
    gap:    0 0 0 ...                        (filler, pushes the pairs into the past)
    query:  kq1 [vq1] kq2 [vq2] ...          (ask for a random subset; predict values)

Loss/accuracy are measured ONLY at the bracketed value slots. Because the answers
were introduced early and asked late, a *recency* window evicts them while an
*importance* cache (which keeps the surprising study pairs) can still answer —
the setup that separates HOLA from a sliding window.
"""

import torch


def vocab_size(n_keys, n_values):
    return 1 + n_keys + n_values  # id 0 = filler


def make_mqar_batch(batch, n_pairs, n_query, gap, n_keys, n_values, generator,
                    return_meta=False):
    assert n_pairs <= n_keys and n_query <= n_pairs
    L = 2 * n_pairs + gap + 2 * n_query
    idx = torch.zeros(batch, L, dtype=torch.long)
    labels = torch.full((batch, L), -100, dtype=torch.long)
    study_val_pos = torch.zeros(batch, n_query, dtype=torch.long)   # where the assoc is bound (k~key, v~value)
    query_key_pos = torch.zeros(batch, n_query, dtype=torch.long)   # where it is asked
    g = generator
    qpos_start = 2 * n_pairs + gap
    for b in range(batch):
        keys = (torch.randperm(n_keys, generator=g)[:n_pairs] + 1)
        vals = (torch.randint(n_values, (n_pairs,), generator=g) + 1 + n_keys)
        study = torch.stack([keys, vals], dim=1).reshape(-1)           # k v k v ...
        qsel = torch.randperm(n_pairs, generator=g)[:n_query]
        qkeys, qvals = keys[qsel], vals[qsel]
        query = torch.stack([qkeys, qvals], dim=1).reshape(-1)
        idx[b] = torch.cat([study, torch.zeros(gap, dtype=torch.long), query])
        for j in range(n_query):                                       # predict value from key pos
            labels[b, qpos_start + 2 * j] = qvals[j]
            study_val_pos[b, j] = 2 * qsel[j] + 1   # the value token binds key->value (via short conv)
            query_key_pos[b, j] = qpos_start + 2 * j
    if return_meta:
        return idx, labels, {"study_val_pos": study_val_pos, "query_key_pos": query_key_pos}
    return idx, labels


@torch.no_grad()
def recall_accuracy(logits, labels):
    """Accuracy over the answered value slots only."""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    pred = logits.argmax(-1)
    return (pred[mask] == labels[mask]).float().mean().item()
