# AI Stocks Forecasting (NVDA)

> **Status: scaffold in progress** on the `ai-stocks-poc` branch. This directory was created by copying the Python modules of [`energy_oil_forecasting/`](../energy_oil_forecasting/) and rewiring the package name. The **data path, specs, shock definition, numerical baselines, statistics contract, analyst prompt strings, shock task spec, and master-strategy schema are NVDA**. Still WTI-targeted: the config-factory and prompt-builder *names* (`build_wti_*`), the scenario task, the adaptive agent's own instructions and skills, and the starter agent.

The goal of this implementation is a **news-grounded equity forecaster with a learning loop**: an agent that discovers which news patterns precede large NVDA moves, validates each candidate pattern against a statistical gate, and reuses only the graduated patterns when it forecasts.

Energy/oil is the parent implementation because it is the repo's other **daily, news-driven, shock-prone** series — the same stateless-agent and adaptive-agent machinery transfers directly.

---

## What is here after the scaffold step

38 files copied from the energy implementation with every `energy_oil_forecasting` import rewritten to `ai_stocks_forecasting`:

| Path | Copied from energy | Retarget status |
|------|--------------------|-----------------|
| `data.py` | WTI `DataService` wiring (`CL=F` via `YFinanceDailyAdapter`) | **done** — `NVDA_SERIES_ID` + `build_nvda_service()`; covariate panel removed |
| `paths.py` | Cache paths, colour palette, `SHOCK_THRESHOLD` / `SHOCK_HORIZON` | **done** — shock is `\|1-day return\| >= 7%`, both directions. Cache-path constants are still energy-named |
| `tasks.py` | Trajectory / shock / scenario task specs and prompt builders | **trajectory and shock done** — two-sided ±7% shock spec with measured anchors (see [Agent layer](#agent-layer)); scenario task still WTI |
| `shock_anchors.py` | *new* — not from energy | **done** — reproduces the shock-spec calibration anchors from 2020–2024 |
| `analysis.py` | Shared scoring helpers | **fixed** — true 80% interval (q10–q90) and `mae_horizon` honoured; see *Two scoring fixes* |
| `viz.py` | Plotly charts for the WTI notebooks | **not retargeted and unused** — still WTI-labelled (oil futures curve, US–Iran war annotations). The NVDA charts live in `charts.py` |
| `prophet_baseline.py` | Prophet baseline | domain-neutral, unused so far |
| `baselines.py` | *new* — not from energy | **done** — runs naive and log-return AutoARIMA through a spec |
| `signals.py` | *new* — not from energy | **contract fixed, bodies pending** — the statistics gate (see below) |
| `charts.py` + `01_leaderboard_and_calibration.ipynb` | *new* — not from energy | **done** — leaderboard, coverage-vs-sharpness, every prediction in full, predicted-vs-actual line chart |
| `02_arima_root_cause.ipynb` | *new* — not from energy | **done** — diagnosis of the raw-price AutoARIMA −77% forecast |
| `analyst_agent/` | Stateless news-grounded analyst + its skills | **prompts done** — analyst role, retrieval supplement and search sub-agent rewritten for NVDA (see [Agent layer](#agent-layer)); `build_wti_*` factory names still WTI |
| `adaptive_agent/` | Curriculum-trained analyst, `WtiStrategyState`, skill mutation tools | **schema drafted** — `NvdaStrategyState` with `NewsPattern` in `nvda_strategy_state.py`; the agent's own instructions and skills still WTI |
| `starter_agent/` | Hackable "build your own" agent | **pending** |

Deliberately **not** copied: the energy notebooks, the committed WTI prediction YAMLs under `data/`, the 52 cached curriculum news files, the trained `wti-strategy-trained/` skill state, and the oil forecast animation. Those are WTI results, not scaffolding — the equivalents are produced here from NVDA runs. The energy `specs/` were not copied either; the NVDA specs below were written fresh rather than edited down from WTI ones.

`adaptive_agent/skills/wti-strategy/` keeps its WTI name until the strategy-state schema is retargeted, so that the directory name never disagrees with its contents.

---

## Remaining scaffold work

| Step | Output |
|------|--------|
| Implement `signals.py` | flagging, control sampling, splits, Fisher's exact + bootstrap, the gate, regime labels — each with tests |
| Finish the agent layer | `build_nvda_news_config` and an NVDA prompt builder (renaming the `build_wti_*` factories); wire the trajectory and shock `AgentPredictor`s; finalise `NvdaStrategyState` and move the adaptive agent onto it |

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

[`baselines.py`](baselines.py) runs `LastValuePredictor` and `DartsAutoARIMAPredictor(num_samples=1000, log_transform=True)` through a spec and writes one YAML per predictor per task to `data/predictions/<spec_id>/`. Re-running is cheap: `cached_multi_backtest` loads whatever is already on disk and only computes what is missing. Pass `--force` to recompute.

Committed reference outputs for the 2025 backtest live in [`data/predictions/nvda_backtest/`](data/predictions/nvda_backtest/), so the numbers below can be re-read without a re-run.

### Why AutoARIMA is fitted on log prices

With `log_transform=True` the model is fitted on `log(price)` and the samples are converted back with `exp`. AutoARIMA selects **`(0,1,0)` with drift at every origin**, and one difference of a log price is the log return. So this is a model of daily log returns: a random walk with a drift of about +0.12% a day. Its point forecast is the naive one plus that drift. What it adds is **intervals whose width scales with the price**, which makes it the naive forecast *as a distribution*. The zero-width naive predictor can't play that role.

There are three reasons not to fit on raw dollars. NVDA rose about 3,600x across the training window. On raw prices AutoARIMA selected second differencing, which extends recent slopes in a straight line. A fixed-dollar error treats a $1 move in 2003 the same as one in 2025. And raw-level Gaussian intervals put probability on negative prices: one committed forecast had a 10th percentile of −$44.

1,000 Monte Carlo samples, not 100, because the point forecast is the *median of the samples*. At 100 samples the 21-day median jittered by about 2.3% from origin to origin, enough to push the log model's MAE above naive's (11.17 against 10.67). At 1,000 the jitter drops to about 0.84% and MAE falls below naive.

### 2025 backtest results: 51 weekly origins

| Predictor | Mean CRPS | MAE | Median abs. error | 80% interval coverage | 80% interval width |
|---|---|---|---|---|---|
| Naive (last value) | 10.67 | 10.67 | 8.85 | 0% (zero-width) | $0 |
| AutoARIMA (log returns) | **8.17** | **10.37** | **7.90** | 90.3% | $49.82 |

Per horizon (USD/share):

| Horizon | CRPS naive | CRPS ARIMA | MAE naive | MAE ARIMA | ARIMA 80% coverage | ARIMA 80% width |
|---|---|---|---|---|---|---|
| 5 bd | 8.23 | **6.25** | 8.23 | **7.90** | 83.0% | $32.83 |
| 10 bd | 10.09 | **7.60** | 10.09 | **9.70** | 89.4% | $46.53 |
| 21 bd | 13.44 | **10.48** | 13.44 | **13.26** | 98.0% | $68.51 |

**Read these findings before building on top of them.**

*The floor is roughly 23% better than naive on CRPS, and almost all of that comes from having intervals at all.* The point forecasts differ only by the drift, and the point-error gap is small (10.37 against 10.67). The CRPS gap is large because the naive predictor emits a point mass, so its CRPS is just its absolute error.

*Its intervals are too wide, not too narrow.* The 80% interval covers 90% of outcomes overall and 98% at 21 days. The likely cause is that volatility is estimated from all history since 1999, including the dot-com era, which was far more volatile than 2025. **So there is room to move left on the coverage chart:** a forecaster can tighten these intervals, keep coverage near 80%, and improve CRPS. Widening them can't win. Fitting the baseline on a shorter window is the cheap numerical version of that move, and is worth doing first, so the agents are measured against the best honest floor.

*The headroom is in shock windows.* The floor's CRPS is 7.36 in quiet windows and 11.45 in windows that contain a ±7% day. Anticipating those moves is what news grounding is meant to buy.

*The drift is also fitted on the whole history.* Every forecast leans about +2.5% upward over 21 days, reflecting NVDA's long-run average return, and outcomes land below the forecast median 43% of the time. A shorter window addresses this too.

### The raw-price baseline, and why it was replaced

An earlier version fitted AutoARIMA on raw prices, and every calendar gap reached the model as `NaN`. Its results are kept in [`data/predictions/archive/`](data/predictions/archive/) as evidence. They are not a baseline, and the chart loader doesn't pick them up. [`02_arima_root_cause.ipynb`](02_arima_root_cause.ipynb) walks through the diagnosis. In short: NYSE was closed on 2025-01-09 for a national day of mourning. That put a `NaN` in the second-to-last training row, which flipped the selected model and produced a **−77% forecast** ($31.32 against an actual $132.45) where the gap-free series gives +2.7%. The three worst forecasts in the backtest were exactly the three origins with a holiday in that position.

| Variant | Mean CRPS | MAE | Largest miss | 80% interval coverage |
|---|---|---|---|---|
| Raw prices, `NaN` gaps (original) | 10.03 | 11.97 | $101.13 | 21.4% |
| Raw prices, forward-filled | 9.06 | 10.48 | $30.96 | 20.0% |
| **Log prices, forward-filled (shipped)** | **8.17** | **10.37** | $31.07 | 90.3% |

Forward-filling removes the blowups but does nothing for calibration: both raw-price variants are badly overconfident. The log fit flips the calibration error from far too narrow to somewhat too wide. That's the better side to be on, since CRPS rewards it, but it isn't the finished state. Both fixes are now in the core `DartsAutoARIMAPredictor`: forward-filling is on by default for every caller, and `log_transform` is opt-in.

### Two scoring fixes carried in this implementation

Both are in [`analysis.py`](analysis.py), inherited from the energy/oil copy, and both produced plausible-looking numbers, which is why they went unnoticed:

- **The "80% interval" was a 60% interval.** Coverage and interval width were computed from the 20th and 80th percentiles. A central 80% interval runs from the 10th to the 90th. Every coverage figure was measured against the wrong target. Fixed in both `score_backtest_results` and `predictions_to_frame`.
- **`mae_horizon` was ignored.** `score_backtest_results` accepted the argument and never used it, so `mae_h21` was MAE pooled across all horizons (10.67 against the true 13.44 for the naive baseline). Fixed.

Both are pinned by tests in `implementations/tests/ai_stocks_forecasting/test_analysis.py`, confirmed to fail against the original code. **The energy/oil copy still has both bugs**, so its coverage and `mae_h21` figures are not comparable with these.

### Known gap: US market holidays

145 of an expected 153 predictions (51 origins x 3 horizons) are scored. The 8 missing ones are horizons whose target date lands on a Monday US market holiday — MLK Day, Presidents' Day, Memorial Day, Labor Day 2025. Pandas' `BDay` counts those as business days, the NYSE does not trade, so there is no actual to score against and the harness drops the prediction. Only the 5- and 10-day horizons are affected; 21 business days from a Monday always lands on a Tuesday. This is correct behaviour, not a bug, but it means horizon counts are uneven (5 bd: 47, 10 bd: 47, 21 bd: 51) and per-horizon means are not over identical origin sets. The same calendar mismatch on the *fitting* side, holidays reaching the model as `NaN`, was a real bug, fixed as described above.

## The statistics contract (`signals.py`)

[`signals.py`](signals.py) is the interface between the two halves of the system. The
discovery agent proposes candidate news patterns; this module — deterministic, LLM-free —
decides whether a candidate has earned the right to influence a forecast. The agent side
imports these functions and nothing else from the statistics layer.

```python
flag_shock_windows(prices_df, threshold_pct=7.0, horizon_days=1) -> list[Window]
sample_matched_controls(windows, prices_df, n_each=1, seed=None) -> list[Window]
train_holdout_split(windows, holdout_fraction=0.5)               -> tuple[list[Window], list[Window]]
evaluate_pattern(pattern_matches, shock_labels, base_rate=None)  -> PatternMetrics
gate_pass(metrics, holdout_metrics)                              -> bool
label_regimes(prices_df, window_days=21, ...)                    -> pd.DataFrame
```

`Window` and `PatternMetrics` are frozen dataclasses. **Signatures and gate thresholds are
final; the bodies raise `NotImplementedError`** and are implemented next, each with tests.

A pattern graduates to the master strategy file only if *all five* criteria hold:

| Criterion | Threshold | Why |
|---|---|---|
| `n_matches` | ≥ 5 (`MIN_MATCHES`) | below this the confidence interval is too wide for a high precision to mean anything |
| `lift` (train) | ≥ 2.0 (`MIN_LIFT`) | must at least double the shock probability over the base rate |
| `p_value` (train) | < 0.05 (`MAX_P_VALUE`) | Fisher's exact, not chi-squared — cell counts are small by construction |
| `ci_low` | > 1.0 | the bootstrap interval on lift must exclude "no effect" |
| `lift` (holdout) | ≥ 1.5 (`MIN_HOLDOUT_LIFT`) | some shrinkage is honest; a pattern that exists only where it was found is not |

The conjunction is the point — each criterion alone is gameable. This is also the reason the
shock threshold is ±7% and not ±10%: at ±10% there are 13 shock events in 2020–2024, so
almost nothing could clear `MIN_MATCHES` and `MAX_P_VALUE` together.

Callers should record the **reason** for a rejection into the per-experiment file, not just
the boolean. "Rejected: lift 2.4 but holdout lift 0.9" tells the next study session
something; `False` does not.

## Agent layer

**Analyst instructions** ([`analyst_agent/agent.py`](analyst_agent/agent.py)). The analyst role,
context-retrieval supplement, and search sub-agent instruction are rewritten for NVDA. The WTI
drivers (OPEC+, Gulf shipping, the SPR) are replaced by the AI-accelerator demand cycle,
hyperscaler capex, TSMC/CoWoS and HBM supply, US export controls, and competitor launches. The
recommended search queries are cut to three, because each `search_web` call also runs a
leakage-verifier call. Payload keys dropped the oil unit: they are now `origin_price_usd` and
`last_close_usd`.

**Shock task** ([`tasks.py`](tasks.py) `TASK_SHOCK_SPEC`). This asks for P(|next-session return|
≥ 7%) in either direction, matching `paths.SHOCK_THRESHOLD`. The WTI version was a one-sided
upside question with guessed anchors. The NVDA anchors are measured over 2020–2024 by
[`shock_anchors.py`](shock_anchors.py):

| Condition (trailing vol = std of prior 21 daily returns) | Sessions | Shocks | Rate |
|---|---|---|---|
| All sessions | 1258 | 52 | 4.1% |
| No earnings, calm (vol < 2.5%) | 474 | 4 | 0.8% |
| No earnings, normal (2.5–3.5%) | 303 | 11 | 3.6% |
| No earnings, elevated (vol > 3.5%) | 461 | 29 | 6.3% |
| Session after a ≥7% move | 52 | 4 | 7.7% |
| **Earnings reaction session** | 20 | 8 | **40%** |

Earnings are by far the largest known source of shocks: a reaction session is about ten times
as likely to be a shock as an average one. A news pattern therefore has to beat the earnings
calendar, not just the unconditional rate. Re-run `uv run python -m
ai_stocks_forecasting.shock_anchors` after any change to the threshold, and update the spec
string to match.

**Master strategy schema**
([`adaptive_agent/nvda_strategy_state.py`](adaptive_agent/nvda_strategy_state.py), draft).
`NvdaStrategyState` extends `AdaptiveSkillState`. Its main field is `news_patterns: list[NewsPattern]`.
Each pattern stores the train and holdout `PatternEvidence` it was graduated on, copied
field-for-field from `signals.PatternMetrics` via `PatternEvidence.from_metrics`, along with its
`source_experiment`. Only graduated patterns go in this file; candidates and rejections belong
in the per-experiment trail. The WTI `skill_state.py` stays in place until the adaptive agent
moves over to the new schema.

## Charts

[`01_leaderboard_and_calibration.ipynb`](01_leaderboard_and_calibration.ipynb) holds the
CRPS leaderboard and the coverage-vs-sharpness chart. Both load by globbing
`data/predictions/<spec_id>/`, so a new predictor appears the moment its YAML lands —
re-running the notebook is the entire update path, and `SPEC_ID` switches between the 2025
backtest and the protected 2026 window.

The coverage chart plots realised coverage of the 80% interval against its mean width, one
panel per horizon. On the nominal-80% line is honest, below is overconfident, above is vague;
further left on the line is better. The log-return AutoARIMA floor sits **above** the line (its
intervals are too wide), so an agent cannot win by widening. It can win by moving *left*:
the same ~80% coverage from a narrower interval, which also shows up as lower CRPS.

The notebook also prints **every prediction** (section 3). `load_scored_frame` returns one row
per predictor, origin and horizon, with the forecast date, point forecast, percentiles, actual
price, error and CRPS, ready to display or save with `to_csv`. And it draws
**`predicted_vs_actual`** (section 4): the actual daily close as a line, each predictor's
forecasts plotted at the date they were forecasting, with 80% bands, one panel per horizon.
On the 21-day panel the naive forecast visibly lags every turn.

[`02_arima_root_cause.ipynb`](02_arima_root_cause.ipynb) is the root-cause analysis of the
original raw-price baseline's −77% forecast, described above. It reads the archived results
and deliberately refits with the old code path, so the failure stays reproducible.

Colour encodes the predictor **family**, marker shape the individual predictor. The
categorical palette is only validated for colourblind separation up to three series on a
scatter, so a fourth hue would fail the check; and tying colour to family rather than rank
means adding a predictor never repaints the existing ones between phases.

## Cutoff discipline

The cutoff was measured, not assumed; see [`LLM_CUTOFFS.md`](../../LLM_CUTOFFS.md). `gemini-3.1-flash-lite-preview` recalls NVDA prices through 2025-01 and `gemini-3.5-flash` through 2024-11. Both refuse on every later window. Any agent scored on pre-cutoff origins is measured on recall, not forecasting. LLM comparisons therefore run on the 2025 backtest (from February, since January sits on the lite model's boundary) and on the protected 2026 evaluation. Pre-cutoff windows stay numerical-only.

Pattern discovery over 2024 data is allowed, but that data is **contaminated**, and the statistical gate does not clean it. A pattern recalled from memory scores *well* in-sample. What protects a graduated pattern is a holdout split that falls after the cutoff.

## Relationship to the other implementations

- [`energy_oil_forecasting/`](../energy_oil_forecasting/) — the parent. Same daily cadence, same news-shock structure, same stateless and adaptive agent tracks.
- [`sp500_forecasting/`](../sp500_forecasting/) — the other equity-market use case, focused on a leak-safe numerical covariate panel rather than news grounding.
