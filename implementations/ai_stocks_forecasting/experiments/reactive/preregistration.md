# Reactive next-day ranges: pre-registration

Written and committed before any 2026 forecast for AMD, Microsoft or Alphabet is made or scored. The commit that adds
this file is the timestamp. Anything changed after it is listed under *Deviations* at the end of the results.

## Why

Every earlier result points the same way: news that is public before a forecast is already in the price, so the agent
cannot *anticipate* a shock (NVDA 2025 backtest, discovery loop, 2026 shock test). This experiment asks a question an
agent might actually answer: after reading the day's news, can it set a better **80% range for the next session's
close**? Here the job is sizing the next move, not calling its direction. Volatility is predictable in a way direction
is not, and the one positive result so far (the earnings calendar cut the 2026 shock agent's false alarms from 10 to
3) is exactly this kind of information.

## Data

- **Tickers: AMD, MSFT, GOOG.** Their 2026 prices have never been read by any analysis in this project (basket stage 0
  stopped at 2025-12-31). NVDA is excluded from the test because its 2026 shock days were inspected earlier.
- **Test window:** every session from 2026-01-02 to 2026-09-30 for which the previous session is also in the price
  cache (the cache was fetched on 2026-10-08). About 185 sessions per ticker, about 555 forecasts per arm.
- **Prices:** split- and dividend-adjusted daily closes (yfinance), the same cache as the basket study.
- **Earnings calendar:** `calendars/earnings.yaml` (yfinance; all three report after the close, so the reaction is the
  next session).

## Forecast timing

Each forecast is made at **00:00 on the target session's date**, before it opens. The agent sees closes through the
previous session, and fenced web search admits only news published before the target date, so it includes the whole
previous day, after-close earnings releases included, and nothing from the session being forecast. The forecast is
the next close as 11 quantiles (5%–95%); the 80% range is the 10%–90% quantiles.

## Arms

| Arm | Uses an LLM | News | Earnings calendar |
|---|---|---|---|
| `vol21` | no | no | no |
| `ewma` | no | no | no |
| `vol21_earnings` | no | no | yes |
| `agent_prices` | yes | no | no |
| `agent_calendar` | yes | no | yes |
| `agent_news` | yes | yes | (only what search finds) |

- `vol21`: normal distribution on the log price, centred on the last close, with the standard deviation of the last
  21 daily log returns.
- `ewma`: the same with an exponentially weighted variance (decay 0.94, RiskMetrics).
- `vol21_earnings`: `vol21`, with the standard deviation multiplied on earnings-reaction sessions by a factor measured
  on 2020–2025 data only: the standard deviation of earnings-reaction returns divided by the previous 21-day
  volatility, pooled over the three tickers.
- The three agent arms share one instruction and model (Gemini 3.1 Flash Lite). `agent_calendar` adds one payload
  field saying whether the previous session was an earnings report (so the target session is the reaction).
  `agent_news` adds fenced web search (checker: Gemini 3.5 Flash, plus the date rule in `search_web`) with two named
  topics: the company's news from the previous day, and scheduled events for the target session.

Prompts are frozen in the commit that starts the 2026 runs, after a wiring check of at most 10 sessions per agent arm
on **2025** dates. That check is not scored for any claim.

## Scoring

All scores are in percent of the previous close, so the tickers can be pooled.

- **Primary: the interval score of the 80% range** (Gneiting and Raftery): width + (2 / 0.2) × the distance by which
  the close falls outside the range. Lower is better. It rewards narrow ranges and penalises misses, so it can't be
  gamed by widening.
- **Secondary:** coverage (target 80%), mean width, CRPS on the 11 quantiles.
- **Subsets, reported but not used for the verdict:** earnings-reaction sessions; sessions after a move of 2 or more
  standard deviations.

## Comparisons and the decision rule

Pre-registered comparisons (A minus B, pooled over tickers):

1. `agent_news` vs `agent_prices`: does reading the news help?
2. `agent_news` vs `ewma`: does the news agent beat a standard volatility model?
3. `agent_calendar` vs `agent_prices`: does knowing the earnings date help?
4. `agent_calendar` vs `vol21_earnings`: does the agent add anything beyond a calendar-scaled volatility model?

An improvement is claimed only if the mean interval-score difference is below zero **and** its 90% paired bootstrap
interval (2,000 resamples of dates, all tickers of a date drawn together) lies entirely below zero. Otherwise the
result is reported as no demonstrated difference.

## Illustration, not a claim: volatility targeting

To show what the ranges are for, each arm's forecast standard deviation (80% width ÷ 2.563) sets a position that
targets 1% daily risk, capped at 1.5× the stock: position = min(1.5, 1% ÷ forecast sd). Reported per arm: realised
daily volatility of the position, its worst drawdown, and the same for buy-and-hold. The position is applied to the
target session's close-to-close return, which assumes trading at the previous close; news after that close makes
this optimistic, so it is an illustration only.

## Budget

A separate ledger per ticker (`experiments/reactive/budget_<ticker>.yaml`), US$12 each, US$36 in total. An arm does
not start on a ticker unless its worst-case cost fits in what is left. Costs are metered on every call, including
US$0.014 per search.

## Known limitations

- Three tickers, one model, one run per forecast.
- Six arms and four comparisons. The four are fixed here; anything else is exploratory.
- `agent_news` can find the earnings date through search, so comparison 1 mixes news with calendar knowledge.
  Comparison 3 isolates the calendar.
