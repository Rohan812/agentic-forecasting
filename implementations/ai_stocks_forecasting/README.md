# AI Stocks Forecasting (NVDA)

> **Status: scaffold in progress** on the `ai-stocks-poc` branch. This directory was created by copying the Python modules of [`energy_oil_forecasting/`](../energy_oil_forecasting/) and rewiring the package name. The **data path, specs, shock definition, and numerical baselines are NVDA**; **the agent layer is still WTI-targeted** — prompt strings, task specs, and skills all describe crude oil. Retargeting them is the work of the tasks below.

The goal of this implementation is a **news-grounded equity forecaster with a learning loop**: an agent that discovers which news patterns precede large NVDA moves, validates each candidate pattern against a statistical gate, and reuses only the graduated patterns when it forecasts.

Energy/oil is the parent implementation because it is the repo's other **daily, news-driven, shock-prone** series — the same stateless-agent and adaptive-agent machinery transfers directly.

---

## What is here after the scaffold step

38 files copied from the energy implementation with every `energy_oil_forecasting` import rewritten to `ai_stocks_forecasting`:

| Path | Copied from energy | Retarget status |
|------|--------------------|-----------------|
| `data.py` | WTI `DataService` wiring (`CL=F` via `YFinanceDailyAdapter`) | **done** — `NVDA_SERIES_ID` + `build_nvda_service()`; covariate panel removed |
| `paths.py` | Cache paths, colour palette, `SHOCK_THRESHOLD` / `SHOCK_HORIZON` | **done** — shock is `\|1-day return\| >= 7%`, both directions. Cache-path constants are still energy-named |
| `tasks.py` | Trajectory / shock / scenario task specs and prompt builders | **pending** — NVDA calibration anchors |
| `analysis.py`, `viz.py`, `prophet_baseline.py` | Shared analysis, plotting, Prophet baseline | domain-neutral; `score_backtest_results` fixed to honour `mae_horizon` (see below) |
| `baselines.py` | *new* — not from energy | **done** — runs the two numerical baselines through a spec |
| `analyst_agent/` | Stateless news-grounded analyst + its skills | **pending** — semiconductor / AI-capex / export-control instructions |
| `adaptive_agent/` | Curriculum-trained analyst, `WtiStrategyState`, skill mutation tools | **pending** — `NvdaStrategyState` with a `NewsPattern` field |
| `starter_agent/` | Hackable "build your own" agent | **pending** |

Deliberately **not** copied: the energy notebooks, the committed WTI prediction YAMLs under `data/`, the 52 cached curriculum news files, the trained `wti-strategy-trained/` skill state, and the oil forecast animation. Those are WTI results, not scaffolding — the equivalents are produced here from NVDA runs. The energy `specs/` were not copied either; the NVDA specs below were written fresh rather than edited down from WTI ones.

`adaptive_agent/skills/wti-strategy/` keeps its WTI name until the strategy-state schema is retargeted, so that the directory name never disagrees with its contents.

---

## Remaining scaffold work

| Step | Output |
|------|--------|
| Add the statistics module | `signals.py` — shock-window flagging, matched negative controls, train/holdout split, pattern metrics, and the graduation gate |
| Retarget the agent layer | NVDA analyst instructions, shock task spec, strategy-state schema |

---

## Data

`build_nvda_service()` registers NVDA's daily **split- and dividend-adjusted** close (Yahoo Finance `NVDA`) as series `nvda_stock_price`, through the core library's `YFinanceDailyAdapter`. History starts at the 1999-01-22 listing; the adapter caches to `data/yfinance/nvda_adj_close_1d.parquet` at the repository root (gitignored) and fetches transparently on a cache miss.

The adjusted close is not a cosmetic choice for this ticker. NVDA split 10-for-1 in June 2024 and 4-for-1 in July 2021, so the raw `Close` series contains ~90% and ~75% single-day "drops" that are pure bookkeeping — a shock detector keyed on large returns would flag them as the largest events in the sample. The tradeoff is that back-adjustment rewrites price *levels* slightly whenever a new dividend accrues; returns, which is what the shock definition and the statistical gate operate on, are unaffected.

There is no covariate panel here, unlike the energy/oil parent. The NVDA forecaster's non-price signal is news; a numerical covariate panel (VIX, peer semiconductor returns, hyperscaler equities) belongs to the later fleet phase, added as a separate `build_nvda_multivariate_service` rather than by widening `build_nvda_service`.

Populate the cache before running anything, from the repository root:

```bash
uv run python scripts/fetch_nvda.py
```

It is idempotent, overwrites the cache with a fresh download, and prints the shock count at the committed threshold so the base rate is visible without recomputing it.

## Shock definition

A **shock is a single-day move of at least ±7%, in either direction** — `SHOCK_THRESHOLD = 7.0`, `SHOCK_HORIZON = 1` in [`paths.py`](paths.py). One day rather than five, because a single-day move isolates the reaction to a discrete news event; a multi-day window blends several events together and makes pattern attribution ambiguous.

The threshold is a balance between how unusual an event is and how many of them exist to learn from, measured over 2020–2024:

| Threshold (1 day, both directions) | 2020–24 events | % of days | 2025 | 2026 YTD |
|---|---|---|---|---|
| ±5% | 151 (87 up / 64 down) | 12.0% | 19 | 7 |
| **±7%** (committed) | **52 (31 up / 21 down)** | **4.1%** | **7** | **2** |
| ±8% | 29 (19 up / 10 down) | 2.3% | 5 | 1 |
| ±10% | 13 (10 up / 3 down) | 1.0% | 2 | **0** |

NVDA's daily return standard deviation over 2020–24 is 3.39%, so ±7% is a 2.1σ day and ±10% a 2.9σ day. ±10% is the more intuitive "shock" but leaves 13 events to discover from and none at all in the 2026 evaluation window — the graduation gate would reject essentially everything, and a shock task scored on 2026 would have no positives. ±5% is the fallback if the gate still turns out to be starved of positives; switching is a one-line change to `SHOCK_THRESHOLD` plus a re-run.

Shocks are **asymmetric**: upside outnumbers downside roughly 3:2 at ±7% and 3:1 at ±10%. Pattern precision should be compared against a direction-aware base rate rather than a pooled one.

## Specs

| Spec | Window | Origins | Purpose |
|---|---|---|---|
| [`specs/nvda_backtest.yaml`](specs/nvda_backtest.yaml) | 2025-01-06 → 2025-12-22, weekly | 51 | Model selection. 7 shock days (2 up / 5 down), clustered Jan–Apr. |
| [`specs/nvda_eval.yaml`](specs/nvda_eval.yaml) | 2026-01-05 → 2026-08-17, weekly | 33 | Protected prospective evaluation. 1 shock day (2026-02-06, +7.9%). |

Both use `task_id: nvda_price_forecast`, `target_series_id: nvda_stock_price`, horizons `[5, 10, 21]` business days, `warmup: 250`, and load as `MultiTargetBacktestSpec` (matching the energy/oil specs, not the single-task `BacktestSpec`).

The eval `end` is the latest origin whose 21-business-day horizon still resolves against cached data. It must stay at least 21 business days behind the most recent cached price, so extend it as newer data accumulates.

## Baselines

```bash
uv run python -m ai_stocks_forecasting.baselines                   # 2025 backtest
uv run python -m ai_stocks_forecasting.baselines --spec nvda_eval  # 2026 protected eval
```

[`baselines.py`](baselines.py) runs `LastValuePredictor` and `DartsAutoARIMAPredictor(num_samples=100)` through a spec and writes one YAML per predictor per task to `data/predictions/<spec_id>/`. Re-running is cheap — `cached_multi_backtest` loads whatever is already on disk and only computes what is missing; pass `--force` to recompute.

Committed reference outputs for the 2025 backtest live in [`data/predictions/nvda_backtest/`](data/predictions/nvda_backtest/), so the numbers below can be re-read without a re-run.

### 2025 backtest results — 51 weekly origins

| Predictor | Mean CRPS | MAE h=21 | 80% coverage |
|---|---|---|---|
| Naive (last value) | 10.67 | 13.44 | 0.0% (degenerate intervals) |
| AutoARIMA | **10.03** | 16.46 | **14.5%** |

Per horizon (USD/share):

| Horizon | CRPS naive | CRPS ARIMA | MAE naive | MAE ARIMA | ARIMA coverage | ARIMA 80% width |
|---|---|---|---|---|---|---|
| 5 bd | 8.23 | **7.45** | **8.23** | 8.56 | 10.6% | $3.91 |
| 10 bd | 10.09 | **8.79** | **10.09** | 10.50 | 17.0% | $7.02 |
| 21 bd | **13.44** | 13.54 | **13.44** | 16.46 | 15.7% | $12.74 |

**Read these two findings before building on top of them.**

*AutoARIMA wins on CRPS but loses on point accuracy.* Its MAE is worse than the naive random walk at every horizon. It only leads on CRPS because the naive predictor emits degenerate (zero-width) intervals, so CRPS collapses to absolute error for it. On a liquid equity the random walk is a genuinely strong point forecast, and beating it is the real bar.

*AutoARIMA is severely overconfident.* Its 80% intervals contain the outcome **14.5%** of the time, against a nominal 80%. A $3.91 interval at a 5-day horizon is far too narrow for a stock whose daily return standard deviation is 3.39%. This is the headline for the coverage-vs-sharpness chart: the numerical baseline sits at the extreme sharp-and-wrong corner, so a news-grounded agent with honestly wide intervals can beat it on coverage almost trivially. Claiming that as a win would be misleading — the honest comparison is CRPS, which penalises both miscalibration and vagueness.

### A fix carried in this implementation

`score_backtest_results` in [`analysis.py`](analysis.py) accepted a `mae_horizon` argument and never used it, returning MAE pooled across all horizons under the key `mae_h21`. It is fixed here to filter by horizon. The energy/oil copy still has the original behaviour, so `mae_h21` values are **not** comparable between the two implementations — on NVDA the pooled and h=21 figures differ substantially (10.67 vs 13.44 for the naive baseline).

### Known gap: US market holidays

145 of an expected 153 predictions (51 origins x 3 horizons) are scored. The 8 missing ones are horizons whose target date lands on a Monday US market holiday — MLK Day, Presidents' Day, Memorial Day, Labor Day 2025. Pandas' `BDay` counts those as business days, the NYSE does not trade, so there is no actual to score against and the harness drops the prediction. Only the 5- and 10-day horizons are affected; 21 business days from a Monday always lands on a Tuesday. This is correct behaviour, not a bug, but it means horizon counts are uneven (5 bd: 47, 10 bd: 47, 21 bd: 51) and per-horizon means are not over identical origin sets.

## Cutoff discipline

Both proxy models (`gemini-3.1-flash-lite-preview`, `gemini-3.5-flash`) have a training cutoff around January 2025. Any agent scored on pre-cutoff origins is being measured on recall, not forecasting. This implementation follows the same discipline as energy and S&P 500: LLM-inclusive comparisons run on a 2025 backtest and a protected 2026 evaluation, and pre-cutoff windows stay numerical-only. Pattern discovery over 2024 data is allowed precisely because every candidate pattern has to clear a statistical gate — a memorised narrative that does not actually predict fails the test.

## Relationship to the other implementations

- [`energy_oil_forecasting/`](../energy_oil_forecasting/) — the parent. Same daily cadence, same news-shock structure, same stateless and adaptive agent tracks.
- [`sp500_forecasting/`](../sp500_forecasting/) — the other equity-market use case, focused on a leak-safe numerical covariate panel rather than news grounding.
