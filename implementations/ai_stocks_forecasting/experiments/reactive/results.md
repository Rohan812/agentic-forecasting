# Reactive next-day ranges: results

Pre-registered in [`preregistration.md`](preregistration.md) (commit `bb5b022`, amendment 1 in `081fdfc`), run with
`uv run python -m ai_stocks_forecasting.reactive score` from `implementations/`. Full numbers:
[`results.yaml`](results.yaml).

**Status (2026-11-18):** the four no-LLM arms are scored on 2026. The three agent arms have not run: the proxy key
expired before the 2025 wiring check, so comparisons 1–4 are pending. Their prompts are frozen as committed.

## Setup in one paragraph

Every 2026 session from January 2 to September 30 for AMD, Microsoft and Alphabet: 187 sessions each, 561 forecasts
per arm, none of it read by any earlier analysis. Each forecast is made at midnight before the session and is the
session's close as 11 quantiles. The primary score is the interval score of the 80% range (width plus 10 × any
miss, in % of the previous close; lower is better).

## The no-LLM arms

| Arm | Interval score | Coverage of the 80% range | Mean width | CRPS |
|---|---|---|---|---|
| `vol21` (21-day realised volatility) | 10.79 | 80.6% | 7.09% | 1.613 |
| `ewma` (RiskMetrics, decay 0.94) | 10.62 | 83.2% | 7.24% | 1.603 |
| `vol21_earnings` (vol21, ×3.13 on earnings-reaction days) | **10.30** | 81.1% | 7.31% | 1.588 |
| `vxn` (Nasdaq-100 implied volatility, scaled per stock) | 10.50 | 78.4% | 6.01% | **1.580** |

The earnings factor (3.13) is the standard deviation of 2020–2025 earnings-reaction returns in units of the previous
21-day volatility, pooled over the three stocks. The VXN scales (AMD 2.12, MSFT 1.17, GOOG 1.28) are each stock's
2020–2025 daily volatility over the index's implied daily volatility.

## Pre-registered comparisons decided so far

| # | Comparison | Mean difference | 90% interval | Verdict |
|---|---|---|---|---|
| 5 | `vol21_earnings` − `vol21`: does the earnings calendar help a volatility model? | **−0.49** | **−0.94 to −0.10** | **Improves** |
| 6 | `vxn` − `ewma`: does implied volatility beat a realised-volatility model? | −0.11 | −0.40 to +0.19 | No demonstrated difference |

- **The earnings calendar measurably improves next-day ranges.** It is the first comparison in this project to
  meet a pre-registered standard. The whole gain comes from 9 earnings-reaction sessions (three per stock): there
  the plain volatility model covered 1 of 9 closes (11%) with an interval score of 75.5, and the calendar-scaled one
  covered 4 of 9 (44%) with 44.9. On every other session the two arms are identical. Nine sessions is a small base;
  the date-resampled interval accounts for that and still excludes zero.
- **Implied volatility is not demonstrably better than EWMA.** It gives the narrowest ranges (6.0% against 7.2%) and
  the best CRPS, with coverage nearest the 80% target, but the interval-score gain is inside the noise. Like every
  arm without the calendar, it missed the earnings reactions: VXN is an index's volatility and does not know a single
  company's report date.

## Illustration, not a claim: volatility targeting

Each arm's forecast sets a position aiming at 1% daily risk (capped at 1.5×). The arm whose positions came closest
to the target was the calendar-scaled one (realised daily volatility 1.03–1.05% across the three stocks, against
1.08–1.40% for the others). Drawdowns were far smaller than buy-and-hold's (−21% to −27%), but mostly because a 1%
risk target holds well under one share's worth of these stocks, so that comparison says little. It assumes trading
at the previous close, which flatters every arm equally.

## What it means for the project

It confirms the presentation's direction with a pre-registered test on unseen data: **scheduled information helps
size the next move.** Whether reading the news adds anything beyond that is comparisons 1–4, which need the agent
arms (estimated US$27, capped at US$36).
