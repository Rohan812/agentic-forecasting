# Pre-registration: exploratory gate and the one-time 2026 evaluation

Written 2026-10-06, before any training labels for the candidates below, before any
exploratory pattern was graduated, and before any 2026 forecast was scored. The commit that
adds this file is the timestamp.

## Question

Does a pattern that passes a **loosened** gate improve out-of-time forecasts of NVDA's
next-session ±5% moves? The strict gate graduated nothing from fifteen candidates; this
experiment measures what a looser standard buys.

## Gate A (exploratory)

A candidate graduates to the exploratory tier when all of these hold:

| Split | Criterion | Strict gate | Gate A |
|---|---|---|---|
| Training (2020 – Jan 2025) | matches | ≥ 5 | ≥ 5 |
| | lift | ≥ 2.0 | ≥ 1.5 |
| | p-value | < 0.05 | < 0.10 |
| | lift interval lower end | > 1 | not required |
| Holdout (Feb – Dec 2025) | lift | ≥ 1.5 | ≥ 1.0 |
| | p-value | < 0.10 | not required |

Holdout design as in the strict experiments: matched, three controls per shock for news
rules; every 2025 session for calendar and price rules. In simulation a pattern with no
effect graduates under Gate A 3–5% of the time per candidate (0.1–0.2% under the strict
gate), so across fifteen candidates 0.4–0.75 false graduations are expected. Exploratory
patterns live in `experiments/exploratory/patterns.yaml`, never in the strict master
strategy file.

## Candidates

All fifteen existing candidates, unchanged. Three news candidates have a holdout lift of at
least 1.0 but no training labels: exp02 q1 (export controls), exp03 q2 (hyperscaler capex
caution, down), exp05 q1 (NVIDIA product launches, up). Their 238 training windows will be
labelled (about $4 each) within the existing $30 discovery cap. No question is edited and no
new candidate is added.

## Forecasters (shock task, 2026)

Origins: an after-close origin (20:00) on every 2026 session D whose next session is in the
price cache when the run starts. The forecast is P(the session after D closes at least 5%
away from D's close). After-close origins match the discovery windows, which use news
published by the end of D.

1. **Climatology**: `HistoricalFrequencyPredictor`.
2. **Climatology + patterns** (no LLM): climatology, except on a day when an exploratory
   pattern fires, where the forecast is that pattern's training precision (the highest one if
   several fire).
3. **Agent**: the after-close news-grounded shock agent (`agent_close`), unchanged.
4. **Agent + patterns**: the same agent, with each exploratory pattern's cue, evidence and
   whether it fires on D added to its input.

Pattern firing on D uses only information available by the end of D: the earnings calendar
(published in advance) and news labels made with the fenced search at cutoff D.

## Metrics and decision rule

Brier score on all origins, Brier skill against climatology, a 90% paired-bootstrap
interval on the Brier difference, separation (mean P before shocks minus before calm
sessions), and the same on the origins where a pattern fired. A forecaster **helps** if its
Brier skill against climatology is above zero and the interval on the Brier difference lies
entirely below zero. Arm 2 against arm 1 isolates the patterns; arm 4 against arm 3 measures
whether the agent uses them.

## Budget

Training labels within the remaining discovery budget ($13.77 at writing). The 2026
evaluation (2026 news labels for graduated news patterns plus two agent arms) has its own
cap of **$20**, enforced by the same ledger mechanism.

## One time only

The protected 2026 window is used once, for this experiment. No forecaster, pattern or
threshold will be changed after seeing 2026 results. If a bug is found, it is fixed and
reported together with the result it changed.

## Disclosure

The 2026 price data was not blind. On 2026-09-24, while checking the price cache, the list of
2026 shock sessions was printed. Two of them, 2026-02-26 and 2026-08-27, are the sessions
after NVIDIA's earnings (2026-02-25 and 2026-08-26). Gate A was chosen on its simulated
false-pass rate, not on which candidates it lets through, but the earnings candidate passes
it, and the 2026 result for an earnings pattern should be read knowing this.
