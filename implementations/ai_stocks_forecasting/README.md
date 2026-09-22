# AI Stocks Forecasting (NVDA)

> **Status: scaffold in progress** on the `ai-stocks-poc` branch. This directory was created by copying the Python modules of [`energy_oil_forecasting/`](../energy_oil_forecasting/) and rewiring the package name. The **data path is now NVDA** (`build_nvda_service()` fetches and caches real NVDA history); **everything else is still WTI-targeted** — prompt strings, task specs, shock constants, and skills all describe crude oil. Retargeting them is the work of the tasks below.

The goal of this implementation is a **news-grounded equity forecaster with a learning loop**: an agent that discovers which news patterns precede large NVDA moves, validates each candidate pattern against a statistical gate, and reuses only the graduated patterns when it forecasts.

Energy/oil is the parent implementation because it is the repo's other **daily, news-driven, shock-prone** series — the same stateless-agent and adaptive-agent machinery transfers directly.

---

## What is here after the scaffold step

38 files copied from the energy implementation with every `energy_oil_forecasting` import rewritten to `ai_stocks_forecasting`:

| Path | Copied from energy | Retarget status |
|------|--------------------|-----------------|
| `data.py` | WTI `DataService` wiring (`CL=F` via `YFinanceDailyAdapter`) | **done** — `NVDA_SERIES_ID` + `build_nvda_service()`; covariate panel removed |
| `paths.py` | Cache paths, colour palette, `SHOCK_THRESHOLD` / `SHOCK_HORIZON` | **pending** — NVDA cache names, 10% / 5-business-day shock definition (currently WTI's 5% / 5 bd) |
| `tasks.py` | Trajectory / shock / scenario task specs and prompt builders | **pending** — NVDA calibration anchors |
| `analysis.py`, `viz.py`, `prophet_baseline.py` | Shared analysis, plotting, Prophet baseline | mostly domain-neutral |
| `analyst_agent/` | Stateless news-grounded analyst + its skills | **pending** — semiconductor / AI-capex / export-control instructions |
| `adaptive_agent/` | Curriculum-trained analyst, `WtiStrategyState`, skill mutation tools | **pending** — `NvdaStrategyState` with a `NewsPattern` field |
| `starter_agent/` | Hackable "build your own" agent | **pending** |

Deliberately **not** copied: the energy notebooks, the committed WTI prediction YAMLs under `data/`, the 52 cached curriculum news files, the trained `wti-strategy-trained/` skill state, and the oil forecast animation. Those are WTI results, not scaffolding — the equivalents are produced here from NVDA runs. The energy `specs/` were not copied either; NVDA specs are written fresh rather than edited down from WTI ones.

`adaptive_agent/skills/wti-strategy/` keeps its WTI name until the strategy-state schema is retargeted, so that the directory name never disagrees with its contents.

---

## Remaining scaffold work

| Step | Output |
|------|--------|
| Add the fetch script | `scripts/fetch_nvda.py`, caching NVDA daily closes under `data/yfinance/` |
| Write the specs | `specs/nvda_backtest.yaml`, `specs/nvda_eval.yaml` (horizons 5 / 10 / 21 business days) |
| Add the statistics module | `signals.py` — shock-window flagging, matched negative controls, train/holdout split, pattern metrics, and the graduation gate |
| Run the baselines | `LastValuePredictor` and `DartsAutoARIMAPredictor` through `backtest()` |
| Retarget the agent layer | NVDA analyst instructions, shock task spec, strategy-state schema |

---

## Data

`build_nvda_service()` registers NVDA's daily **split- and dividend-adjusted** close (Yahoo Finance `NVDA`) as series `nvda_stock_price`, through the core library's `YFinanceDailyAdapter`. History starts at the 1999-01-22 listing; the adapter caches to `data/yfinance/nvda_adj_close_1d.parquet` at the repository root (gitignored) and fetches transparently on a cache miss.

The adjusted close is not a cosmetic choice for this ticker. NVDA split 10-for-1 in June 2024 and 4-for-1 in July 2021, so the raw `Close` series contains ~90% and ~75% single-day "drops" that are pure bookkeeping — a shock detector keyed on large returns would flag them as the largest events in the sample. The tradeoff is that back-adjustment rewrites price *levels* slightly whenever a new dividend accrues; returns, which is what the shock definition and the statistical gate operate on, are unaffected.

There is no covariate panel here, unlike the energy/oil parent. The NVDA forecaster's non-price signal is news; a numerical covariate panel (VIX, peer semiconductor returns, hyperscaler equities) belongs to the later fleet phase, added as a separate `build_nvda_multivariate_service` rather than by widening `build_nvda_service`.

`scripts/fetch_nvda.py` — the explicit cache-population step every other implementation ships — does not exist yet.

## Cutoff discipline

Both proxy models (`gemini-3.1-flash-lite-preview`, `gemini-3.5-flash`) have a training cutoff around January 2025. Any agent scored on pre-cutoff origins is being measured on recall, not forecasting. This implementation follows the same discipline as energy and S&P 500: LLM-inclusive comparisons run on a 2025 backtest and a protected 2026 evaluation, and pre-cutoff windows stay numerical-only. Pattern discovery over 2024 data is allowed precisely because every candidate pattern has to clear a statistical gate — a memorised narrative that does not actually predict fails the test.

## Relationship to the other implementations

- [`energy_oil_forecasting/`](../energy_oil_forecasting/) — the parent. Same daily cadence, same news-shock structure, same stateless and adaptive agent tracks.
- [`sp500_forecasting/`](../sp500_forecasting/) — the other equity-market use case, focused on a leak-safe numerical covariate panel rather than news grounding.
