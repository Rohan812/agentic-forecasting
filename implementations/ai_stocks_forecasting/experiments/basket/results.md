# Stage 0 results: the four-stock basket

Run 2026-10-08 as pre-registered in `preregistration.md` (commit 5f0bb3b). No LLM calls.
Reproduce with `uv run python -m ai_stocks_forecasting.basket` from `implementations/`,
after `uv run python scripts/fetch_ai_basket.py`. Numbers: `stage0_results.yaml`.

## Windows

Shock: a return of at least 2 standard deviations of that ticker's previous 21 returns.
Prices truncated at 2025-12-31.

| Split | Windows | Shock events | NVDA / AMD / MSFT / GOOG | Shock rate, all ticker-days |
|---|---|---|---|---|
| Training (2020 – Jan 2025) | 660 | 330 | 79 / 86 / 81 / 84 | 7.1% |
| Holdout (Feb – Dec 2025) | 236 | 59 | 15 / 13 / 16 / 15 | 6.6% |

The holdout has 59 shock events against 13 for NVDA alone at ±5%.

## Calendar rules against the strict gate plus the calendar-shift test

| Rule | Training: matches, shocks, lift (95% CI), Fisher p, shift p | Holdout: matches, shocks, lift, Fisher p, shift p | Verdict |
|---|---|---|---|
| **Earnings reaction** | 48, 47, 11.0 (7.2–14.1), 3e-14, 0.0005 | 8, 7, 9.0, 0.0003, 0.0005 | **Graduates** |
| Fed decision day | 34, 26, 2.8 (1.6–6.1), 0.001, 0.015 | 7, 1, 0.5, 0.87, 0.79 | Fails the holdout |
| CPI release day | 37, 22, 1.4 (0.8–2.6), 0.15, 0.24 | 9, 2, 0.9, 0.70, 0.64 | Fails training and holdout |
| Jobs report day | 26, 14, 1.2 (0.6–2.4), 0.42, 0.69 | 10, 4, 1.9, 0.22, 0.22 | Fails training |

**Earnings is the first pattern to clear the strict gate**, with no threshold loosened, on
matched windows, and with the dependence check passed in both periods. On NVDA alone it could
not pass because one stock reports four times a year; the basket gives 15 earnings reactions
in the holdout year. Across every ticker-day (not only matched windows), 47 of 81 earnings
reactions were shocks in training (58%) and 7 of 15 in the holdout (47%), against about 7%
of ordinary days.

**Fed decision days were real before the model cutoff and not after**: lift 2.8 over
2020 to January 2025, then 1 shock in 7 meetings in 2025. The same shape as NVDA earnings
in the single-stock study, and the reason the gate has two halves.

The 2025 holdout was not blind for these rules (see the pre-registration's disclosure), so
this is a confirmation on matched windows with a dependence check, not a first look.

## Baseline model

Logistic regression on volatility and calendar features, fitted on 2020 to January 2025
and scored on all 916 holdout ticker-days (61 shocks).

| | Brier | Skill vs climatology | 90% interval on Brier difference, resampled by date |
|---|---|---|---|
| Climatology (7.1%) | 0.0622 | — | — |
| Baseline model | 0.0599 | +3.7% | −0.0060 to +0.0010 |

Coefficients: earnings next 2.78, CPI next 0.41, Fed next 0.40, |z| today 0.15, jobs next
−0.14, volatility ratio −0.08. **It does not meet the pre-registered rule** (the interval
reaches just above zero): almost all of its gain comes from the 15 earnings ticker-days, and
with relative shocks the volatility features add little, because the shock definition has
already absorbed volatility. Frozen, it is still the baseline an agent must beat in the blind
2026 test.

## What this changes

- The basket fixes the sample-size problem that stopped the single-stock study.
- The pattern that survives everything is a **calendar date**, not a news category.
- The next question is whether an agent adds anything beyond "it is earnings tomorrow",
  tested blind on AMD, MSFT and GOOG in 2026 (not yet looked at).
