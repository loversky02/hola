# Empirical review — HOLA — exact KV-cache hippocampus on a compressive recurrent state

arXiv [2607.02303](https://arxiv.org/abs/2607.02303) · paper-forge empirical-review lane (heuristic)

> Unlike a static manuscript review (e.g. PAT), this verdict comes from *executed* evidence: the reproduction ran, and each central claim is checked against the numbers it produced — including whether the claim's own risk regime was ever exercised.

**Hypothesis under test:** The recall lift is shown on a synthetic, untrained MQAR task; the cache is perfect only until facts exceed its capacity w, and the benefit shrinks when the backbone already recalls well. Whether the lift transfers to a trained LM at larger scale is untested.

**Summary:** 3 claim(s) — 🟢 2 hold · 🔴 0 break · 🟡 0 mixed · ⚪ 1 inconclusive

## Claims

### 🟢 HOLDS — An exact importance KV-cache lifts associative recall far above the bare compressive recurrent state.
**Measured:** Δ +0.47 (47%) · regime *synthetic MQAR, 8 facts within cache capacity w=16, no training, 200 trials* · confidence **medium** · ⚠ risk regime untested

- **Why:** MQAR exact recall (importance cache vs state-only) moved 0.53→1 (47%) in the claimed 'higher' direction. But the hypothesis names a failure regime this evidence never touched, so the claim is confirmed only in the tested regime.
- **Honest finding:** Reproduced: MQAR exact recall (importance cache vs state-only) moved 0.53→1 (47%) in the claimed direction under 'synthetic MQAR, 8 facts within cache capacity w=16, no training, 200 trials'. But the failure regime named in the hypothesis was never exercised, so this holds only in the tested regime — treat it as regime-local, not a general confirmation.

### 🟢 HOLDS — The importance cache generalizes past Gated DeltaNet to other linear-attention backbones.
**Measured:** Δ +0.899 (90%) · regime *synthetic MQAR, 32 facts, no training, backbones {gdn, deltanet, gla}* · confidence **medium** · ⚠ risk regime untested

- **Why:** MQAR recall with cache vs state-only on the Gated DeltaNet backbone (32 facts) moved 0.099→0.998 (90%) in the claimed 'higher' direction. But the hypothesis names a failure regime this evidence never touched, so the claim is confirmed only in the tested regime.
- **Honest finding:** Reproduced: MQAR recall with cache vs state-only on the Gated DeltaNet backbone (32 facts) moved 0.099→0.998 (90%) in the claimed direction under 'synthetic MQAR, 32 facts, no training, backbones {gdn, deltanet, gla}'. But the failure regime named in the hypothesis was never exercised, so this holds only in the tested regime — treat it as regime-local, not a general confirmation.

### ⚪ INCONCLUSIVE — Surprise (the paper's importance signal beta*||e||) predicts which facts the recurrent state forgets.
**Measured:** Δ -0.0245043 (5%) · regime *forget-probe on 6144 facts, Gated DeltaNet layer 0* · confidence **low**

- **Why:** AUC of surprise predicting forgetting (>0.5 = predictive; the paper only asserts this) moved only 4.9% (0.5→0.475496) — within noise.
- **Honest finding:** Inconclusive: the evidence for AUC of surprise predicting forgetting (>0.5 = predictive; the paper only asserts this) (0.5→0.475496) is too flat or too thin to confirm or refute — a near-flat number is not a finding.
