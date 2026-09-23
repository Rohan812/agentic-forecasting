# Phase 0 demo script (2 minutes)

The end-of-day demo for Phase 0: NVDA plumbed through the harness, and the agent layer
retargeted from WTI. Two speakers, one hand-off. Have every file listed under "Show" open in a
tab before starting, and **run nothing live**.

## 0:00–0:20 · Framing (lead)

> "We're building a forecaster that learns which news predicts large NVDA moves. It only keeps
> a pattern if the pattern clears a statistical gate. Today was foundation: data, baselines,
> and the contracts the agents code against. No LLM agent ran today, on purpose."

**Show:** the Phase 0 row of the roadmap. Nothing else.

## 0:20–1:00 · Team Signals: data and baselines

> "NVDA's split-adjusted daily closes go through the same cutoff-safe `DataService` as every
> other series in the repo. Two numerical baselines ran over 51 weekly origins in 2025."

**Show:** `01_leaderboard_and_calibration.ipynb`, first the leaderboard and then the
coverage-vs-sharpness chart.

- AutoARIMA leads on CRPS: **10.03 vs 10.67**.
- **But** its MAE is worse than the random walk, and its 80% intervals cover only **14.5%**.
  "The floor is sharp and wrong. An agent with honestly wide intervals would beat it on
  coverage for free, so we won't claim that. We'll claim CRPS."
- `signals.py`: "The contract between the two teams. Five functions with fixed signatures and a
  five-part gate."

**Hand-off line:** "The gate only means something if the model can't remember the answers. So
first, where does the model's memory end?"

## 1:00–1:45 · Team Agents: cutoffs and the agent layer

**Show:** `LLM_CUTOFFS.md`, the results table.

> "We measured the cutoff instead of assuming it. The lite model remembers NVDA's price
> through January 2025, and the advanced model through November 2024. After that, both refuse.
> On 2024 shock windows they said they knew all of them and got about 64% right. They're
> confident and wrong. So 2024 is contaminated, and 2025 and 2026 are clean."

**Show:** `tasks.py`, `TASK_SHOCK_SPEC`.

> "The shock question is two-sided: a 7% move either way. The anchors are measured on
> 2020–2024. About 4% of sessions are shocks, but 40% of the sessions right after earnings
> are. The agent starts from those numbers, not from a feeling."

**Show:** the analyst instruction in `analyst_agent/agent.py`, and the rendered
`NvdaStrategyState` table (`adaptive_agent/nvda_strategy_state.py`).

> "The prompts now talk about capex, export controls and CoWoS instead of OPEC. The master
> strategy file holds one kind of entry, a graduated pattern, and every pattern carries the
> train and holdout numbers that got it through the gate."

## 1:45–2:00 · Close (lead)

> "Tomorrow: the stateless news agent on the same 2025 origins as these baselines, and the
> first real `signals.py` bodies. The claim to beat is CRPS 10.03."

## If asked

- *Why 7% and not 10%?* At 10% there are 13 events in 2020–2024 and none in 2026, so the gate
  could never pass anything.
- *Doesn't the gate catch memorised patterns?* No. Memorisation raises in-sample precision.
  Only a holdout after the cutoff catches it (`LLM_CUTOFFS.md`).
- *Why is AutoARIMA so overconfident?* We haven't diagnosed it yet. A $3.91 five-day interval
  is far too narrow for a stock with 3.4% daily volatility. The pipeline scores it correctly,
  so this is a finding about the baseline, and we'll investigate it.
