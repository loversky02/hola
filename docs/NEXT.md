# HOLA — resume / next steps

Reproduction of arXiv:2607.02303 (HOLA — Hippocampus for Linear Attention).
Pure-PyTorch, $0 on Mac (venv `.m15venv`, torch 2.10 + MPS). Run tests/experiments
with `.m15venv/bin/python ...`.

## Done
- **M0 core + offline proof** (`hola/`, `tests/test_core.py`, 20 checks PASS):
  GatedDeltaNet / DeltaNet / GLA recurrences, HOLA cache (importance top-w + RMSNorm-γ
  near-argmax read + null sink). Module-level proof that importance beats recency and
  the read is near-argmax — before any training.
- **Two fixes that make small-scale MQAR actually learn** (both baked into
  `backbones.py`):
  1. decay init ≈ 1 (`a_proj.bias=6.0`) — else `α^t` erases the state across the gap;
  2. depthwise causal **short conv** on the block input — else linear-attn cannot bind
     key→value across positions (Based/Zoology result). DeltaNet npairs=8 recall went
     0.27 → 0.997 after adding it.

## Done (results in `results/`)
- **exp2 mechanistic money plot** — the headline. importance = perfect until facts > w,
  then ~w/N; state degrades with load; recency ≈ 0. `results/exp2_money_plot.png`.
- **exp3 any-backbone** (rewritten mechanistic) — cache lifts recall to ~0.98 on
  GDN/DeltaNet/GLA. `results/exp3_anybackbone.png`.
- **forget-probe** — honest mixed result: AUC(surprise→forgotten) ≈ 0.48 (surprise
  doesn't rank *which* fact is forgotten in MQAR — all equally surprising); coverage
  importance 23% > recency 6% (from a weak trained GDN). `results/probe_plot.png`.

## Open / optional
- **exp1 trained money plot** (`exp1_importance_vs_recency.py`) — still flaky: tiny-scale
  MQAR grokking is unreliable at the capacity crossover (learnable regime
  `inner=n_heads·d_head ≳ 32` AND `npairs > capacity` fight each other at this size).
  Best observed: HOLA groks faster than GDN at npairs=16 (0.99@600 vs ~1200+). To make
  it clean: more steps (3–5k), or `inner≈64` with `npairs∈[48,96]` (slow, seqlen ~210).
  Mechanism already shown by exp2/exp3 — this is an honest reproduction lesson.
- `git init` + push to github.com/loversky02 (repo `hola`). **Omit** Co-Authored-By
  trailers (paper-repo convention). Update the profile pins if it ships.
- Optional GPU (RunPod ~$1-2): real 340M/SlimPajama Wikitext-PPL point, or load a
  pretrained fla Gated-DeltaNet and run the forget-probe on a strong model.

## Resume phrase
"tiếp tục HOLA" — read this file + `results/*.json`, then continue from the first
unfinished item above.

## Watch-outs
- Backbone scan is a Python loop over L — fine for MQAR (seqlen<160), slow past that.
- Cache only helps when facts > state capacity (~n_heads·d_head). Sweep across it.
- `w` must be ≥ npairs for importance to hold every fact; else it degrades too (honest).
