# Pre-registration: the four-stock AI basket (stage 0, no LLM spend)

Written 2026-10-08, before any basket result below was computed by the code this plan
describes. The commit that adds this file is the timestamp.

## Why

One stock and one year give 13 post-cutoff shocks, too few to confirm rare patterns. Pooling
NVDA, AMD, MSFT and GOOG gives about four times as many. A price-only feasibility check
(not committed) motivated this plan; see the disclosure.

## Shock definition (volatility-relative)

For each ticker, a session is a **shock** when its close-to-close return is at least **2
times** the standard deviation of that ticker's previous 21 daily returns (known at the prior
close). This makes shocks comparable across tickers of different volatility. Consecutive
shock sessions merge into one event, as in `signals.flag_shock_windows`.

## Windows

Per ticker: every shock event from 2020-01-01, each with matched controls drawn by
`signals.sample_matched_controls` on that ticker's standardised returns (seed 0): one
control per shock in training, three in the holdout. Split by date exactly as before:
training to 2025-01-31, holdout 2025-02-01 to 2025-12-31. Windows are pooled across tickers.
**All stage 0 analysis truncates prices at 2025-12-31.**

## Candidates (calendar rules, no labelling cost)

A rule fires for a window when the session after its `as_of` is a scheduled reaction session:

1. **Earnings reaction**: the session after that ticker's own results (all four report after
   the close). Dates from yfinance.
2. **Fed decision day**: scheduled FOMC decision days, hand-entered and cross-checked
   against FRED's target-rate changes.
3. **CPI release day** and 4. **jobs report day**: FRED release dates.

Direction "either" only. No other candidate is tested in stage 0.

## Gate

The **strict** gate, unchanged (`signals.gate_reasons`: training lift ≥ 2, p < 0.05, lift
interval above 1, ≥ 5 matches; holdout lift ≥ 1.5, p < 0.10), computed on the pooled matched
windows, **plus a dependence check**. Pooled ticker-days are not independent (a Fed day hits
every ticker), so Fisher's p-value overstates the evidence. Each rule must also pass a
**calendar-shift test**: the rule's dates are shifted by a random number of sessions (the
same shift for every ticker, 2,000 shifts of between 5 and 120 sessions either way) and the
number of matched shocks recomputed; the p-value is the share of shifts with at least as many
matched shocks as observed. Required: shift p < 0.05 in training and < 0.10 in the holdout.

## Baseline model (idea 1)

A logistic regression for P(shock next session), fitted on every pooled ticker-day from
2020-01-01 to 2025-01-31, with these features known at the close: the ratio of 5-session to
63-session realised volatility, the absolute standardised return of the current session,
and four flags for the next session (earnings reaction, Fed day, CPI day, jobs day).
scikit-learn defaults (L2, C = 1). Scored on every pooled 2025 holdout ticker-day against
climatology (the training shock rate): Brier, Brier skill, and a 90% bootstrap interval on
the Brier difference resampled **by date**. It *beats climatology* if skill is above zero
and the interval lies entirely below zero. This model, frozen, is the baseline the agent
must beat in the later blind test.

## Later stages (not run here)

Agent forecasts on AMD, MSFT and GOOG, and blind 2026 tests on those three tickers, are
planned under separate caps ($50 in total) and will be pre-registered before they run. The
2026 outcomes of AMD, MSFT and GOOG have not been looked at; NVDA's 2026 window is used.

## Disclosure

A feasibility check on 2026-10-08 already computed, for this basket and shock definition,
the pooled earnings result on all sessions (not matched windows): training 47 of 81 earnings
reactions were shocks (lift 8.2); 2025 holdout 7 of 15 (lift 7.0). It also showed Fed days
with lift 2.3 to 3.3 in training for AMD, MSFT and GOOG and none of 7 meetings each in the
holdout. So the 2025 holdout is **not blind** for these rules, and a pass here is a
confirmation on matched windows with a dependence check, not a first look. The blind test
is 2026 on AMD, MSFT and GOOG.
