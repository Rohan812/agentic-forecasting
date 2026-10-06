# Results: exploratory gate and the one-time 2026 evaluation

Run 2026-10-06 exactly as pre-registered in `preregistration.md` (commit b4c810b). Gate A
results were committed before any 2026 score (commit e372c5b). Spend: $12.17 for training
labels (discovery ledger $28.41 of $30) and $9.81 for the 2026 evaluation (of a $20 cap).

## Gate A

One of fifteen candidates graduated: **X-1, NVIDIA reports quarterly results after the
close** (training lift 3.17, p 0.030, 11 matches, precision 0.381; 2025 lift 3.58).
Hyperscaler capex caution, the strongest news lead, matched one training window and it was
not a shock (training lift 0); product launches (training lift 1.27, p 0.29) and export
controls (0.69, p 0.82) also failed training. Full table: `gate_a_results.yaml`.

## 2026 evaluation

182 after-close origins (every 2026 session whose next weekday is a session), 7 shocks.
X-1 fired on 3 origins (2026-02-25, 05-20, 08-26); 2 were followed by shocks.

| Forecaster | Brier | Skill vs climatology | 90% interval on Brier difference | Separation |
|---|---|---|---|---|
| Climatology | 0.045 | — | — | 0.000 |
| Climatology + patterns | **0.041** | +0.078 | −0.008 to +0.001 | +0.072 |
| Agent (after close) | 0.055 | −0.229 | **+0.003 to +0.017** (worse) | +0.039 |
| Agent + patterns | 0.044 | +0.001 | −0.009 to +0.007 | **+0.114** |

Agent + patterns against agent alone: Brier −0.010, 90% interval **−0.017 to −0.005**.

On the three earnings days:

| Origin | Next session | Climatology | Climatology + patterns | Agent | Agent + patterns |
|---|---|---|---|---|---|
| 2026-02-25 | shock | 0.13 | 0.38 | 0.50 | 0.55 |
| 2026-05-20 | calm | 0.13 | 0.38 | 0.45 | 0.48 |
| 2026-08-26 | shock | 0.12 | 0.38 | 0.14 | 0.50 |

## Against the pre-registered rule

A forecaster *helps* if its skill against climatology is above zero and the interval on its
Brier difference lies entirely below zero.

- **Climatology + patterns: does not meet the rule.** Skill +0.078, but the interval reaches
  +0.001: it changes only three forecasts, and three days cannot establish an effect.
- **Agent: worse than climatology**, measurably. It raised P above 0.3 on ten calm days.
- **Agent + patterns: ties climatology** (skill +0.001, interval spans zero).
- **The patterns do help the agent.** Against the agent alone, the interval lies entirely
  below zero. It caught the 2026-08-26 earnings reaction (0.14 → 0.50) and raised P above
  0.3 on three calm days instead of ten.

## How to read this

The one pattern that a loosened gate let through is earnings, and the disclosure in the
pre-registration applies: the 2026 shocks after earnings had been seen before the gate was
chosen. Two of three 2026 earnings reactions were shocks, which is close to the training
precision (0.38) times three, so the result is consistent with the pattern rather than
an extreme draw. The clearest finding is the agent comparison: with the earnings evidence
in its input, the agent stopped issuing high probabilities on ordinary days. Even so, no
forecaster beat plain climatology by the pre-registered standard on 182 days with 7 shocks.

The protected 2026 window has now been used and is no longer blind for any later work.
