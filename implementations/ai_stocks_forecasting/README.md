# AI Stocks Forecasting (NVDA)

> **Status: scaffold in progress** on the `ai-stocks-poc` branch. This directory was created by copying the Python modules of [`energy_oil_forecasting/`](../energy_oil_forecasting/) and rewiring the package name. The **data path, specs, shock definition, numerical baselines, statistics contract, analyst prompt strings, news-grounded agent configs (`build_nvda_news_config`, `build_nvda_multitask_news_config`), prompt builders, shock task and its `AgentPredictor`, and master-strategy schema are NVDA**. Still WTI-targeted: the other config-factory *names* (`build_wti_*`), the scenario task, the adaptive agent's own instructions and skills, and the starter agent.

The goal of this implementation is a **news-grounded equity forecaster with a learning loop**: an agent that discovers which news patterns precede large NVDA moves, validates each candidate pattern against a statistical gate, and reuses only the graduated patterns when it forecasts.

Energy/oil is the parent implementation because it is the repo's other **daily, news-driven, shock-prone** series — the same stateless-agent and adaptive-agent machinery transfers directly.

---

## What is here after the scaffold step

38 files copied from the energy implementation with every `energy_oil_forecasting` import rewritten to `ai_stocks_forecasting`:

| Path | Copied from energy | Retarget status |
|------|--------------------|-----------------|
| `data.py` | WTI `DataService` wiring (`CL=F` via `YFinanceDailyAdapter`) | **done** — `NVDA_SERIES_ID` + `build_nvda_service()`; covariate panel removed |
| `paths.py` | Cache paths, colour palette, `SHOCK_THRESHOLD` / `SHOCK_HORIZON` | **done** — shock is `\|1-day return\| >= 5%`, both directions. Cache-path constants are still energy-named |
| `tasks.py` | Trajectory / shock / scenario task specs and prompt builders | **trajectory and shock done** — two-sided ±5% shock spec with measured anchors, `nvda_shock_task()`, and NVDA-named predictors via `build_nvda_news_predictor` (see [Agent layer](#agent-layer)); scenario task still WTI |
| `shock_anchors.py` | *new* — not from energy | **done** — reproduces the shock-spec calibration anchors from 2020–2024 |
| `analysis.py` | Shared scoring helpers | **fixed** — true 80% interval (q10–q90) and `mae_horizon` honoured; see *Two scoring fixes* |
| `viz.py` | Plotly charts for the WTI notebooks | **not retargeted and unused** — still WTI-labelled (oil futures curve, US–Iran war annotations). The NVDA charts live in `charts.py` |
| `prophet_baseline.py` | Prophet baseline | domain-neutral, unused so far |
| `baselines.py` | *new* — not from energy | **done** — runs naive and log-return AutoARIMA through a spec |
| `signals.py` | *new* — not from energy | **done** — flagging, control sampling, the train/holdout split, pattern scoring, the gate and regime labels, all tested (see below) |
| `charts.py` + `01_leaderboard_and_calibration.ipynb` | *new* — not from energy | **done** — leaderboard, coverage-vs-sharpness, every prediction in full, predicted-vs-actual line chart |
| `02_arima_root_cause.ipynb` | *new* — not from energy | **done** — diagnosis of the raw-price AutoARIMA −77% forecast |
| `analyst_agent/` | Stateless news-grounded analyst + its skills | **prompts done** — analyst role, retrieval supplement and search sub-agent rewritten for NVDA (see [Agent layer](#agent-layer)); news-grounded factories are `build_nvda_news_config` and `build_nvda_multitask_news_config`; prompt builder is `NvdaPriceForecastPromptBuilder`; the basic, code-execution and tool factories are still `build_wti_*` |
| `adaptive_agent/` | Curriculum-trained analyst, `WtiStrategyState`, skill mutation tools | **schema done** — `NvdaStrategyState` with `NewsPattern` in `nvda_strategy_state.py`, gate-enforced on construction and on load; the agent's own instructions and skills still WTI |
| `starter_agent/` | Hackable "build your own" agent | **pending** |

Deliberately **not** copied: the energy notebooks, the committed WTI prediction YAMLs under `data/`, the 52 cached curriculum news files, the trained `wti-strategy-trained/` skill state, and the oil forecast animation. Those are WTI results, not scaffolding — the equivalents are produced here from NVDA runs. The energy `specs/` were not copied either; the NVDA specs below were written fresh rather than edited down from WTI ones.

`adaptive_agent/skills/wti-strategy/` keeps its WTI name until the strategy-state schema is retargeted, so that the directory name never disagrees with its contents.

---

## Remaining scaffold work

| Step | Output |
|------|--------|
| Package `signals.py` for the sandbox | a self-contained copy the discovery agent can import inside E2B. Its only third-party dependencies are numpy and pandas; the import of the shock constants from `paths.py` would need inlining |
| Finish the agent layer | renaming the remaining `build_wti_*` factories; find evidence that news separates shocks from calm days: no shock-agent variant has yet (see *After-close origins*). `nvda_shock_fresh` is closed to further iterations; move the adaptive agent onto `NvdaStrategyState` |

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

A **shock is a single-day move of at least ±5%, in either direction**: `SHOCK_THRESHOLD = 5.0`, `SHOCK_HORIZON = 1` in [`paths.py`](paths.py). One day rather than five, because a single-day move isolates the reaction to a discrete news event; a multi-day window blends several events together and makes pattern attribution ambiguous.

The threshold was first set at ±7% and **lowered to ±5% to give the holdout enough events.** Both proxy models remember prices through about January 2025 ([`LLM_CUTOFFS.md`](../../LLM_CUTOFFS.md)), so a pattern only counts as evidence once it holds up after the cutoff. The clean window before the protected 2026 evaluation is Feb–Dec 2025, and after merging consecutive-day clusters it holds:

| Threshold (1 day, both directions) | 2020–24 shock days | % of days | 2020–24 events | **Feb–Dec 2025 events (clean holdout)** | 2026 YTD events |
|---|---|---|---|---|---|
| **±5%** (committed) | **151 (87 up / 64 down)** | **12.0%** | **117** | **13** | **7** |
| ±6% | 94 (53 up / 41 down) | 7.5% | 79 | 6 | 4 |
| ±7% (previous) | 52 (31 up / 21 down) | 4.1% | 48 | 4 | 2 |
| ±8% | 29 (19 up / 10 down) | 2.3% | 26 | 3 | 1 |
| ±10% | 13 (10 up / 3 down) | 1.0% | 10 | 1 | 0 |

"Events" merge shocks on consecutive trading days (see *Shock windows* below). Only ±5% gives a double-digit holdout. At ±7%, four events could not support a holdout-lift criterion, since a single hit swings the lift enormously.

**The cost is a milder "shock".** NVDA's daily return standard deviation over 2020–24 is 3.39%, so ±5% is about a 1.5σ day, and 12% of sessions qualify, against 4% at ±7%. The discovery loop is looking for news that precedes an *unusually large* day, not only a rare one.

Shocks are **asymmetric**: upside outnumbers downside about 4:3 at ±5% and 3:1 at ±10%. Pattern precision should be compared against a direction-aware base rate rather than a pooled one.

**Changing the threshold ripples out**, and the order matters:
1. Re-run `uv run python -m ai_stocks_forecasting.shock_anchors` and update the anchors in `tasks.TASK_SHOCK_SPEC`.
2. Re-check that `signals.sample_matched_controls` still finds volatility-matched controls. At ±5% its exclusion buffer had to shrink from 5 sessions to 2; see *Matched controls* below.

## Specs

| Spec | Window | Origins | Purpose |
|---|---|---|---|
| [`specs/nvda_backtest.yaml`](specs/nvda_backtest.yaml) | 2025-01-06 → 2025-12-22, weekly | 51 | Model selection. 19 shock days at ±5% (8 up / 11 down), 15 events after merging. The first origins sit on the model-cutoff boundary. |
| [`specs/nvda_eval.yaml`](specs/nvda_eval.yaml) | 2026-01-05 → 2026-08-17, weekly | 33 | Protected prospective evaluation. 6 shock days at ±5% (4 up / 2 down). |
| [`specs/nvda_shock_smoke.yaml`](specs/nvda_shock_smoke.yaml) | Feb–Dec 2025, fixed | 10 | Shock-agent smoke: 5 holdout shocks + 5 volatility-matched controls, each with its expected outcome frozen. See *Shock smoke backtest*. |
| [`specs/nvda_shock_fresh.yaml`](specs/nvda_shock_fresh.yaml) | Feb–Dec 2025, fixed | 16 | Search-topics A/B on every non-holdout 2025 session that follows a shock (3 continued). See *Fresh re-score*. |
| [`specs/nvda_shock_fresh_close.yaml`](specs/nvda_shock_fresh_close.yaml) | Feb–Dec 2025, fixed, 20:00 origins | 16 | The same sessions for the after-close agent. The second and last iteration on this set. |

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

*The headroom is in shock windows.* The floor's CRPS is 6.83 in quiet windows and 10.31 in the 56 of 145 forecast windows that contain a ±5% day. Anticipating those moves is what news grounding is meant to buy.

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
sample_matched_controls(windows, prices_df, n_each=1, seed=None, *, threshold_pct=7.0) -> list[Window]
train_holdout_split(windows)                                     -> tuple[list[Window], list[Window]]
evaluate_pattern(pattern_matches, shock_labels, base_rate)       -> PatternMetrics
shock_base_rate(prices_df, start, end, threshold_pct=5.0)        -> float
gate_pass(metrics, holdout_metrics)                              -> bool
gate_reasons(metrics, holdout_metrics)                           -> list[str]
label_regimes(prices_df, window_days=21, ...)                    -> pd.DataFrame
```

`Window` and `PatternMetrics` are frozen dataclasses. **Every function in the contract is implemented and tested.**

**Shock windows.** A session is a shock when its simple return reaches ±5%, the same definition
the shock task's prompt anchors use, so "shock" means one thing everywhere. Shocks on
consecutive trading days are merged into **one event** (`SHOCK_CLUSTER_GAP_DAYS = 1`). The COVID
crash, nine consecutive shock sessions from 9 to 19 March 2020, therefore counts once, not nine
times. A merged window is anchored on its largest move, but its news cutoff (`as_of`) is the
session before the episode's *first* shock. Anchoring the cutoff on the largest move would let
the agent read coverage of a crash already under way. On 2020–2024 this gives **117 events**
(71 up / 46 down) from 151 shock days. Wider merge gaps are more conservative about
independence but leave the gate fewer events: 93 at a 2-day gap, 53 at 5, 29 at 10.

**Matched controls.** Each shock's controls are ordinary sessions that are:

1. **At least two sessions away from *every* shock in the price history**, including shocks the
   caller didn't pass in. That matters when sampling controls for a training split while the
   holdout's shocks are still in the data.
2. **Within about six months of the shock.**
3. **Closest to it in trailing 21-day volatility.**
4. **On the same side of the holdout boundary**, and never straddling it (see below).

The volatility rule matters most. Shocks cluster in volatile markets, so random controls come
disproportionately from calm ones, and any headline that merely tracks volatility would then
look predictive. On 2020–2024 all 117 events get a control. The matched controls' median
trailing volatility is 3.51% against the shocks' 3.53%. Random non-shock days sit at 2.95%,
with a far calmer lower quartile (2.20% against 3.05%). Draws are seeded and never reuse a
session.

**The buffer is two sessions because shocks are common at ±5%.** A one-week buffer excluded
nearly every session in volatile stretches. That left only calm days to choose from, so the
volatility match collapsed and 41 of 117 events got no control. A small buffer also errs on
the safe side. A control near a shock's news occasionally matches a pattern and *lowers*
measured lift, which makes the gate conservative. Calm controls *inflate* the lift of anything
volatility-correlated, which makes it permissive.

**Train / holdout split.** `train_holdout_split(windows)` splits by date, not by fraction. The
boundary is `HOLDOUT_START = 2025-02-01`, the first month neither proxy model remembers
([`LLM_CUTOFFS.md`](../../LLM_CUTOFFS.md)):

- **Train** is everything before 2025-02-01. January 2025 counts as train, because it sits on
  the cutoff boundary.
- **The holdout** is Feb–Dec 2025, and a window must lie wholly inside it, its news cutoff
  included.
- **2026 is in neither.** It's the protected evaluation.

The original plan split 2024 in half. That fails for a subtle reason: both models remember
2024, and memorisation *raises* in-sample precision. A remembered pattern would pass a 2024
holdout just as easily as its training data. Only data the models cannot remember tests a
pattern.

On the real pipeline (flag → sample → split, 2020 onward) the split gives:

| Split | Shocks | Controls | Base rate |
|---|---|---|---|
| Train (2020 – Jan 2025) | 119 | 119 | 0.50 |
| Holdout (Feb–Dec 2025) | 13 | 13 | 0.50 |

The 2026 windows are excluded. Any split with fewer than `MIN_SPLIT_SHOCKS = 10` shocks is
**refused with an error** rather than returned: at ±7% the holdout would hold 4, and the
gate's holdout criterion would be decided by noise.

Two details worth knowing:

- **Controls stay with their shock's period.** That's rule 4 above. Without it, controls for
  2025 shocks drifted into 2024 and 2026, and the holdout ended up with 13 shocks but only 5
  controls.
- **The holdout's volatility match is looser than train's**, 3.28% against the shocks' 3.76%
  (train: 3.51% against 3.52%). Its 13 shocks cluster in the volatile spring of 2025, and a short
  window offers few volatile quiet days. Matching still closes most of the gap: random quiet
  days from the same months sit at 2.21%.

**One caveat for anyone reading 2025 results: the holdout overlaps the 2025 backtest spec.** A
pattern graduated because it held up in Feb–Dec 2025 will flatter any 2025 backtest of a
forecaster that uses it. The honest measure of a pattern-using forecaster is the protected
2026 evaluation.

**Scoring a pattern.** `evaluate_pattern(matches, labels, base_rate)` returns precision, lift,
a one-sided Fisher's exact p-value, and a 95% bootstrap interval on lift.

**`base_rate` is required, and it has to be the *market* rate, not the sample's.** Our windows
are a matched sample, one control per shock, so they are 50% shocks by design. Lift measured
inside that mix is precision ÷ 0.5, which can never exceed 2.0, and `MIN_LIFT = 2.0` would then
demand a perfect pattern. So lift is rebuilt from how often the pattern appears among shocks
versus among controls, weighted by the real rate. That rate comes from
`shock_base_rate(prices, start, end)`: 12.0% of sessions in 2020–2024, and 7.0% in the Feb–Dec
2025 holdout, so compute it per split.

The p-value needs no correction, because the test compares shocks with controls directly. It's
computed as an exact sum rather than through scipy, which keeps `signals.py` dependency-light
for the sandbox, and it's cross-checked against scipy in the tests. The bootstrap resamples
shocks and controls separately, with a fixed seed, so the same inputs always give the same
interval.

On the real training split, two stand-in patterns show the statistics doing their job:

| Stand-in pattern | Scored against | Lift | 95% CI | p |
|---|---|---|---|---|
| Elevated volatility beforehand (trailing vol > 3.5%) | our matched controls | **1.02** | 0.82 – 1.25 | 0.5 |
| Elevated volatility beforehand | random controls | 1.91 | 1.46 – 2.54 | 0.000008 |
| Earnings reaction session | our matched controls | **4.63** | 1.98 – 8.33 | 0.009 |

Against our controls the volatility proxy correctly shows no effect. Against random controls it
looks highly significant, which is the false discovery the matching exists to prevent. The
earnings calendar clears every training criterion of the gate, consistent with the shock
anchors (50% of reaction sessions against 12% overall).

A pattern graduates to the master strategy file only if *all six* criteria hold:

| Criterion | Threshold | Why |
|---|---|---|
| `n_matches` | ≥ 5 (`MIN_MATCHES`) | below this the confidence interval is too wide for a high precision to mean anything |
| `lift` (train) | ≥ 2.0 (`MIN_LIFT`) | must at least double the shock probability over the base rate |
| `p_value` (train) | < 0.05 (`MAX_P_VALUE`) | Fisher's exact, not chi-squared — cell counts are small by construction |
| `ci_low` | > 1.0 | the bootstrap interval on lift must exclude "no effect" |
| `lift` (holdout) | ≥ 1.5 (`MIN_HOLDOUT_LIFT`) | some shrinkage is honest; a pattern that exists only where it was found is not |
| `p_value` (holdout) | < 0.10 (`MAX_HOLDOUT_P_VALUE`) | a big holdout lift from one or two lucky matches is not evidence; see below |

The conjunction is the point — each criterion alone is gameable. Event counts are also why the
shock threshold is ±5%: the clean post-cutoff holdout (Feb–Dec 2025) holds 13 events at ±5%
but only 4 at ±7%.

Callers should record the **reason** for a rejection into the per-experiment file, not just
the boolean. `gate_reasons(train, holdout)` returns one plain-English reason per failed
criterion, for example `"holdout lift 1.00 is below 1.5"`, and an empty list when the pattern
graduates. An undefined value fails its criterion rather than slipping through: a `nan`
compares false against everything, so a naive `lift < 2` check would wave it past. A pattern
that matched no holdout window is rejected as never tested on post-cutoff data.

**Why the holdout needs its own significance test.** A memorised pattern passes the four
training criteria by construction ([`LLM_CUTOFFS.md`](../../LLM_CUTOFFS.md)), so the holdout is
the only real protection against memorisation. With 13 shocks and 13 controls, lift from a
handful of matches is mostly luck: one lucky hit gives a lift of 1 ÷ base rate, about 14, on no
evidence at all. Simulated on that holdout:

| Holdout rule | No effect, rarely fires | No effect, often fires | Real, strong | Real, weaker |
|---|---|---|---|---|
| lift ≥ 1.5 alone (original spec) | 33% pass | 23% | 95% | 68% |
| + at least 5 holdout matches | 2% | 21% | 91% | 63% |
| **+ holdout p < 0.10 (adopted)** | **1%** | **4%** | **71%** | **26%** |
| + holdout CI low > 1 | 18% | 3% | 64% | 19% |

"Strong" means the pattern appears in half the shocks and a tenth of the controls, like the
earnings calendar. "Weaker" means 40% against 20%.

With the lift rule alone, roughly one memorised pattern in three or four would have graduated.
The holdout p-value rule cuts that to about 1 in 30, **at a deliberate cost in power**: strong
real patterns now pass 71% of the time, and weaker ones about a quarter of the time. A graduated
pattern is only useful if it can be trusted, so rejecting some real patterns is the better
error. The level is 0.10, looser than training's 0.05, because 0.05 on a 13-shock holdout
would reject most real patterns too.

**Regimes.** `label_regimes(prices)` adds two columns:

- `realized_vol`: the standard deviation of the last 21 daily returns, known at the close.
- `regime`: `low`, `normal` or `high`, relative to the 33rd and 67th percentiles of all
  volatility *up to that day*.

Because the percentiles only ever look backwards, a day's label never changes when later data
arrives. Labels up to 2020 are identical whether or not 2021–2026 exists; a whole-series cut
point would silently relabel the past. There is no label until a year of volatility history
exists.

Pass the **full history since 1999**. The extreme dot-com years and the calm 2004–2019 years
balance out, so by the end of 2024 the cut points are **2.27% / 3.54%** daily volatility,
almost exactly the 2.5% / 3.5% bands the shock-task anchors use. Over 2020–2024:

| Regime | Share of days | Chance the next session is a ±5% shock | Shock events starting here |
|---|---|---|---|
| Low | 23% | 5.5% | 14 |
| Normal | 41% | 9.7% | 45 |
| High | 36% | 18.8% | 58 |

Starting the history in 2015 instead would label half of 2020–2024 "high", because 2015–2019
was unusually calm. The gate doesn't use regimes. They're for spotting degradation: a pattern
that works in calm markets and stops working in volatile ones shows up in a per-regime split
long before it shows up in pooled precision. `Window.regime` is left `None` by the flagging and
sampling functions; look it up at `as_of` when a split is needed.

## Agent layer

**Analyst instructions** ([`analyst_agent/agent.py`](analyst_agent/agent.py)). The analyst role,
context-retrieval supplement, and search sub-agent instruction are rewritten for NVDA. The WTI
drivers (OPEC+, Gulf shipping, the SPR) are replaced by the AI-accelerator demand cycle,
hyperscaler capex, TSMC/CoWoS and HBM supply, US export controls, and competitor launches. The
recommended search queries are cut to three, because each `search_web` call also runs a
leakage-verifier call. Payload keys dropped the oil unit: they are now `origin_price_usd` and
`last_close_usd`.

**News-grounded config** ([`analyst_agent/agent.py`](analyst_agent/agent.py) `build_nvda_news_config`).
It combines the NVDA analyst instruction with cutoff-fenced web search. Every `search_web` call
searches only up to the origin's `as_of`, and an independent leakage verifier checks the result;
it runs on the advanced model, so it doesn't share the search model's blind spots. When
verification fails, the analyst proceeds on price history rather than fill the gap from memory.
The agent is named `nvda_analyst_news`, which matters because `AgentPredictor` builds its
predictor id, and so the prediction's filename in the registry, from that name. The inherited
WTI factory would have filed NVDA forecasts as `agent_predictor_wti_analyst_news_…`. Each origin
costs about three searches plus three to nine verifier calls. Tests in
`implementations/tests/ai_stocks_forecasting/test_analyst_agent.py` pin down the identity and the
fence, and run offline.

**Prompt builder** ([`analyst_agent/agent.py`](analyst_agent/agent.py) `NvdaPriceForecastPromptBuilder`).
It serialises the task and the price history into the JSON payload that the analyst instruction's
forecasting contract describes: `task`, `as_of`, `horizons`, `standard_quantiles`, a
`target_summary` (last close, 52-week range) and `target_history_csv`. A test fails if the
instruction promises a key the payload doesn't carry. The history is daily for the last six months
and weekly averages for the **five years** before that (`WEEKLY_HISTORY_YEARS`). It used to reach
back to NVDA's 1999 listing. Split-adjusted, those years are a few cents a share, and 900 of 1,300
weekly rows printed as `0.0x`. Bounding them cut a 2025 payload from about 26K to 7.5K characters,
on every agent turn. `target_summary.last_date` is usually the session before `as_of`, because the
adapter releases each close one business day later.

**Shock task** ([`tasks.py`](tasks.py) `TASK_SHOCK_SPEC`). This asks for P(|next-session return|
≥ 5%) in either direction, matching `paths.SHOCK_THRESHOLD`. The WTI version was a one-sided
upside question with guessed anchors. The NVDA anchors are measured over 2020–2024 by
[`shock_anchors.py`](shock_anchors.py):

| Condition (trailing vol = std of prior 21 daily returns) | Sessions | Shocks | Rate |
|---|---|---|---|
| All sessions | 1258 | 151 | 12.0% |
| No earnings, calm (vol < 2.5%) | 474 | 18 | 3.8% |
| No earnings, normal (2.5–3.5%) | 303 | 38 | 12.5% |
| No earnings, elevated (vol > 3.5%) | 461 | 85 | 18.4% |
| Session after a ≥5% move | 151 | 34 | 22.5% |
| **Earnings reaction session** | 20 | 10 | **50%** |

These were regenerated at ±5% with `shock_anchors.py`; at ±7% the same script reproduces the
original 4% / 1% / 4% / 6% / 8% / 40%. The prompt's rule that a probability needs a named
catalyst moved from above ~15% to above ~30%. The old cap sat below several ordinary anchors
at ±5%, so it now sits above the highest non-earnings anchor (22%).

Earnings are by far the largest known source of shocks: a reaction session is about four times
as likely to be a shock as an average one (50% against 12%). A news pattern therefore has to beat the earnings
calendar, not just the unconditional rate. Re-run `uv run python -m
ai_stocks_forecasting.shock_anchors` after any change to the threshold, and update the spec
string to match.

**Shock predictor** ([`tasks.py`](tasks.py)). `build_nvda_news_predictor("shock")` pairs the
multitask news identity (`nvda_analyst_multitask`, a task-agnostic instruction) with
`TASK_SHOCK_SPEC` and `DiscreteAgentForecastOutput`, so a forecast comes back as a
`BinaryForecast` filed as `agent_predictor_nvda_analyst_multitask_<model>_discrete`.
`nvda_shock_task()` is the matching single-horizon task (`nvda_shock_1d`). **Mind the timing.**
The adapter releases each close one business day late, so at a Monday `as_of` the history ends
on Friday. The question, like `signals.Window`, is Tuesday's close against Monday's, and the
prediction's `forecast_date` is Tuesday. So the agent has not seen the `as_of` session's own
close. That matters most for the "session after a ≥5% move" anchor, since the move it depends on
is exactly the missing session. The payload carries `last_close_date`, and the spec tells the
agent to judge the `as_of` session from news rather than assume it was quiet. `forecast_date` is
pandas `BDay`, not the NYSE calendar: an origin followed by a market holiday has no session to
resolve against, so choose shock origins with a trading day after them. Offline tests in
`implementations/tests/ai_stocks_forecasting/test_tasks.py` pin the id, the output schema, and
the payload against the instruction's input list. Writing that test turned up an undocumented
`task` key, which the instruction now lists.

**Scoring shock forecasts.** The harness resolves a binary forecast by reading the task's target
series at `forecast_date`, so the outcome is a series of its own. `register_shock_series(service)`
adds `nvda_shock_1d`, which is 1 on a session whose close moved at least 5% from the prior close.
That is the `signals.flag_shock_windows` definition before cluster merging, and it reproduces the
151 shock days of 2020–2024. Each value is released with its close, one business day late, and a
test checks that an outcome is hidden on its own session. `nvda_shock_task()` targets this series
with `payload_type="binary"`, and the prompt builder reads price history from `price_series_id`
rather than from the task's target.

**Shock smoke backtest** ([`shock_smoke.py`](shock_smoke.py), `uv run python -m
ai_stocks_forecasting.shock_smoke`). The news-grounded shock agent and `HistoricalFrequencyPredictor`
run on the ten origins in `specs/nvda_shock_smoke.yaml`: five holdout shocks and five of their
volatility-matched controls from `signals`. Ten *random* origins would not work: only 2% of the
sessions after the weekly 2025 origins were shocks, so a random ten almost surely holds none. Both
predictors are scored on the intersection of their origins, and the report prints how many were
dropped. The agent loses an origin whenever the proxy fails after retries, and the harness does not
line those up across predictors. Resolved outcomes are checked against the labels frozen in the
spec before scoring. Cost and latency are read back from each prediction's Langfuse trace into
`data/predictions/costs/nvda_shock_smoke.yaml`. First run (lite model, 2026-09-24):

| Predictor | Brier (50% shock sample) | Mean P before shock | Mean P before control |
|---|---|---|---|
| Shock agent (news-grounded) | 0.353 | 0.174 | 0.154 |
| Climatology | 0.388 | 0.128 | 0.129 |

No origins dropped. **$0.104 in total, $0.0104 per origin** (maximum $0.0126), 12 s mean latency.
The leakage verifier on the advanced model is about 60% of each origin's cost. Read these numbers
as a pipeline and cost check. Ten origins in a sample built to be half shocks can't rank
predictors, and that Brier is not a market Brier. What they do show is that **the agent barely
separates shocks from volatile quiet days.** It answered 0.14, 0.15 or 0.18 at every origin, which
is the trailing-volatility anchor from the task spec. It gave 0.18 the day before the +18.7%
tariff-pause rally. The traces show why news added little: **every origin ran exactly one search.**
The multitask instruction says to search but, unlike `build_nvda_news_config`, names no topics.

**Search topics added (after the run above).** `TASK_SHOCK_SPEC` now names three fenced queries.
The first asks how NVDA moved on the `as_of` session, which is the session its price history does
not show. The second covers scheduled catalysts for the next session: earnings, CPI, jobs, FOMC,
export-control rulings. The third covers unscheduled ones: export controls, tariffs, hyperscaler
capex, competition. The topics live in the task spec, not the shared multitask instruction, because
the trajectory task uses that instruction too. A one-origin mechanics check on 2025-07-14, which is
not a holdout window, ran three searches instead of one, for $0.0092. The table above predates the
change and was **deliberately not re-run** on the same ten origins: they are gate holdout windows,
and re-scoring a prompt change on them would tune on the holdout. The smoke spec therefore declares
its arms as `climatology` and `agent_notopics`. The earlier agent results were relabelled to the
no-topics predictor id (`nvda_analyst_multitask_notopics`), because the old id now means "with
topics". `TASK_SHOCK_SPEC_NO_TOPICS` reproduces the earlier prompt byte for byte.

**Fresh re-score** (`--spec nvda_shock_fresh`, 2026-09-24). Fresh origins turned out to be
scarce. They have to fall after both models' cutoff, before the protected 2026 window, and outside
the gate holdout, and the holdout already uses every independent 2025 shock event. Only three 2025
shock sessions lie outside it, and all three are the second day of an episode. So the fresh set
conditions on "the previous session was a shock", the spec's ~22% anchor. It takes every such
session that is not a holdout or smoke target: **16 origins, 3 continued** (19%, not enriched).
Every origin's `as_of` is a shock session the price history does not show, which is the case
search topic 1 was written for. `run` refuses to score the `agent` arm on a gate-holdout shock
origin or a smoke origin, and a test enforces this.

| Arm | Brier | Mean P, move continued | Mean P, calmed down | Cost / origin |
|---|---|---|---|---|
| Agent, with search topics | 0.166 | 0.167 | 0.192 | $0.0215 |
| Agent, no topics (earlier prompt) | **0.138** | 0.270 | 0.190 | $0.0100 |
| Climatology | 0.156 | 0.129 | 0.129 | — |

No origins dropped, no 503s. **The topics did not help.** With three positives this can't show they
hurt either. Almost all of the Brier gap is one origin: the day after the +18.7% tariff-pause rally,
where the no-topics arm said 0.45 (correct, −5.9% next) and the topics arm said 0.14. The topics
arm's probability was *lower* before continuations than before calm days. The topics also doubled
the cost.

**Why: web search is not a reliable source for a price move.** Every origin ran three searches
with no verification failures, but topic 1, "how did NVDA move on `as_of`", was often wrong or
empty. On 2025-04-09 and 2025-11-10 it returned only source URLs and no summary, passed the
verifier, and reached the agent as a successful result. That is the soft-failure mode where
retrieval returns something that isn't news. On 04-09 it cost the agent the continuation. On
03-06 it reported "down ~3%" for a −5.7% day, on 03-10 "closed higher" for a −5.1% day, and on
04-10 "a significant rally" for a −5.9% day. The move is already in our price data, one session
late. The fix is to let the shock agent see the `as_of` close, for example a price series for this
task stamped at the 16:00 close with origins after the close, rather than asking search for it.
Search results with no summary body should be rejected as unusable, not passed on as news. These 16
origins have now been scored, so the next prompt change needs origins of its own.

**After-close origins (the fix).** The root cause turned out to be the fence itself. The harness
fences `search_web` to information published *strictly before* the `as_of` date. Asking how NVDA
moved *on* `as_of` therefore asked for fenced-out news, and the verifier stripped it. Two changes
follow.

- **Prices, not search, for the `as_of` session.** `register_shock_series` now also registers
  `nvda_stock_price_at_close`. It holds the same adjusted closes, each released at 16:00 on its own
  session (`MARKET_CLOSE`) rather than a business day later, and the shock indicator is released at
  the close too. An origin at 20:00 (`after_close(day)`) sees that day's close.
  `build_nvda_shock_predictor_after_close()` reads it with `TASK_SHOCK_SPEC_AFTER_CLOSE`, which is
  the no-topics prompt with its timing paragraph replaced. It is filed as
  `nvda_analyst_multitask_close`. The news fence does not move: the agent still sees only news
  published before `as_of`, so news after that day's close stays out of reach. Checked on
  2025-04-09, where the history now ends 96.06 → 114.04, the +18.7% session that was invisible
  before.
- **Every arm is wrapped in `SessionDatePredictor`.** Predictors stamp `forecast_date = as_of + h`,
  so a 20:00 origin forecasts 20:00 on the target session. The harness matches outcomes by exact
  timestamp, so that forecast goes unscored: with every origin lost the harness raises, but a spec
  mixing origin times would lose the after-close ones silently. The wrapper pins the date to the
  session and changes nothing else. A test shows both behaviours. The existing specs reproduce
  exactly with it on.
- **Empty search results are retried, not served** (core `search_web`). A clean verdict on an empty
  summary now spends an attempt, and exhausting the attempts returns `[SEARCH_VERIFICATION_FAILED]`,
  which the agent is already told how to handle.

`shock_smoke.py` accepts `origin_time: after_close` in a spec and an `agent_close` arm. It refuses
`agent_close`, like `agent`, on holdout or smoke origins, and refuses it at midnight origins. A live
check on 2025-07-14 ran end to end for $0.01.

**Scored on the same 16 sessions** (`--spec nvda_shock_fresh_close`, 20:00 origins, 2026-09-24).
This is the second and last iteration on that set, so it is weaker evidence than a first look.
Every origin was scored, with no drops and no 503s, for $0.0104 per origin, the same as the
no-topics arm.

| Arm (same 16 sessions, 3 continued) | Sees `as_of` close | Brier | Mean P, continued | Mean P, calmed |
|---|---|---|---|---|
| Agent, after close | yes | 0.154 | 0.227 | 0.212 |
| Agent, no topics (midnight) | no | **0.138** | 0.270 | 0.190 |
| Agent, search topics (midnight) | no | 0.166 | 0.167 | 0.192 |
| Climatology (either origin time) | — | 0.156 | 0.129 | 0.129 |

**The mechanism works; the forecast doesn't improve.** Every after-close rationale quotes the
`as_of` move correctly (−8.5%, −8.7%, +18.7%, +5.4%) and applies the "after a ≥5% move" anchor.
The "no catalysts on March 3" error is gone. But the agent lifts every origin to about 0.2 without
separating the sessions that continued from those that calmed down (0.227 against 0.212), and
lands on climatology's Brier. The no-topics arm's lead rests largely on one origin, 0.45 on the day
after the tariff-pause rally. The rally was not in its price history, but a later review of the
traces showed it saw the rally through search anyway (see the fence leak below). With three
positives none of these differences is evidence. **The honest summary for this task: across three variants,
no shock agent has shown it can tell a continuing shock from a calm-down better than climatology.**
`nvda_shock_fresh`'s 16 sessions are now closed to further iterations. The next measurement is the
protected 2026 run.

**Search fence leak (found in review, fixed in core `search_web`).** At midnight origins the
agent is not meant to see anything from the `as_of` session. Langfuse traces show the LLM leakage
verifier sometimes passed that session anyway, marked clean at confidence 9–10. Examples: "On
February 4, 2025, NVIDIA (NVDA) stock ... closing the session at $118.34" for a 2025-02-04 origin,
and "On April 3, 2025, NVIDIA (NVDA) stock experienced a decline, closing at $101.54" for
2025-04-03. One midnight origin per agent arm quotes the exact `as_of` close in its forecast
record: smoke 2025-02-04, topics 2025-04-03, and no-topics 2025-04-09. The 2025-04-09 rationale
cites the $96.06 → $114.04 rally and the H20 export notice, which was not public until 2025-04-15.
That origin is the one that gave the no-topics arm its lead, so **the midnight-arm Brier scores
above are contaminated and should not be read as agent skill.** The after-close arm sees the
`as_of` close legitimately and is unaffected, although its fence had the same gap. `search_web` now
drops, after the verifier passes a result, every sentence that names a day or month that is not
entirely before the cutoff. The verifier is also told that facts about the cutoff day itself fail.
Undated leaks still depend on the verifier. These arms were not re-run: their origins are closed to
further iterations, so a clean re-run would measure the fence, not a new prompt.

**Master strategy schema**
([`adaptive_agent/nvda_strategy_state.py`](adaptive_agent/nvda_strategy_state.py)).
`NvdaStrategyState` extends `AdaptiveSkillState`. Its main field is `news_patterns: list[NewsPattern]`.
Each pattern stores the train and holdout `PatternEvidence` it was graduated on, copied
field-for-field from `signals.PatternMetrics` via `PatternEvidence.from_metrics`, along with its
`source_experiment`. Only graduated patterns go in this file; candidates and rejections belong
in the per-experiment trail. The WTI `skill_state.py` stays in place until the adaptive agent
moves over to the new schema.

**The schema enforces the gate itself.** A `NewsPattern` re-runs `signals.gate_reasons` on its own
train and holdout evidence when it is built, and refuses to exist if any criterion fails; the error
lists which ones. `AdaptiveSkillStore.load` validates `skill_state.yaml` through the same model, so a
hand-edited file with weakened evidence fails to load instead of reaching the forecaster. Tightening
a gate constant in `signals.py` therefore makes any previously graduated pattern that no longer
clears it fail to load, by name. That is deliberate: every pattern in the master file meets the
current standard, and a change to the standard shows up rather than being grandfathered in. Pattern
ids (`P-<n>`) must be unique, because a forecast names the pattern it matched by id, and
agent-written text is escaped so a `|` in a cue can't shift the `SKILL.md` table. Tests are in
`implementations/tests/ai_stocks_forecasting/test_nvda_strategy_state.py`.

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
