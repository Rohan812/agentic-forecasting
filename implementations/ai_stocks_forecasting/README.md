# AI Stocks Forecasting (NVDA)

> **Status: scaffold in progress** on the `ai-stocks-poc` branch. This directory was created by copying the Python modules of [`energy_oil_forecasting/`](../energy_oil_forecasting/) and rewiring the package name. The **data path, specs, shock definition, numerical baselines, statistics contract, analyst prompt strings, news-grounded agent config (`build_nvda_news_config`), shock task spec, and master-strategy schema are NVDA**. Still WTI-targeted: the other config-factory and prompt-builder *names* (`build_wti_*`), the scenario task, the adaptive agent's own instructions and skills, and the starter agent.

The goal of this implementation is a **news-grounded equity forecaster with a learning loop**: an agent that discovers which news patterns precede large NVDA moves, validates each candidate pattern against a statistical gate, and reuses only the graduated patterns when it forecasts.

Energy/oil is the parent implementation because it is the repo's other **daily, news-driven, shock-prone** series — the same stateless-agent and adaptive-agent machinery transfers directly.

---

## What is here after the scaffold step

38 files copied from the energy implementation with every `energy_oil_forecasting` import rewritten to `ai_stocks_forecasting`:

| Path | Copied from energy | Retarget status |
|------|--------------------|-----------------|
| `data.py` | WTI `DataService` wiring (`CL=F` via `YFinanceDailyAdapter`) | **done** — `NVDA_SERIES_ID` + `build_nvda_service()`; covariate panel removed |
| `paths.py` | Cache paths, colour palette, `SHOCK_THRESHOLD` / `SHOCK_HORIZON` | **done** — shock is `\|1-day return\| >= 5%`, both directions. Cache-path constants are still energy-named |
| `tasks.py` | Trajectory / shock / scenario task specs and prompt builders | **trajectory and shock done** — two-sided ±5% shock spec with measured anchors (see [Agent layer](#agent-layer)); scenario task still WTI |
| `shock_anchors.py` | *new* — not from energy | **done** — reproduces the shock-spec calibration anchors from 2020–2024 |
| `analysis.py` | Shared scoring helpers | **fixed** — true 80% interval (q10–q90) and `mae_horizon` honoured; see *Two scoring fixes* |
| `viz.py` | Plotly charts for the WTI notebooks | **not retargeted and unused** — still WTI-labelled (oil futures curve, US–Iran war annotations). The NVDA charts live in `charts.py` |
| `prophet_baseline.py` | Prophet baseline | domain-neutral, unused so far |
| `baselines.py` | *new* — not from energy | **done** — runs naive and log-return AutoARIMA through a spec |
| `signals.py` | *new* — not from energy | **flagging, control sampling, the train/holdout split and pattern scoring done**; the gate and regime labels pending (see below) |
| `charts.py` + `01_leaderboard_and_calibration.ipynb` | *new* — not from energy | **done** — leaderboard, coverage-vs-sharpness, every prediction in full, predicted-vs-actual line chart |
| `02_arima_root_cause.ipynb` | *new* — not from energy | **done** — diagnosis of the raw-price AutoARIMA −77% forecast |
| `analyst_agent/` | Stateless news-grounded analyst + its skills | **prompts done** — analyst role, retrieval supplement and search sub-agent rewritten for NVDA (see [Agent layer](#agent-layer)); news-grounded factory is `build_nvda_news_config`; the basic, multitask, code-execution and tool factories and the prompt builder are still `build_wti_*` / `Wti*` |
| `adaptive_agent/` | Curriculum-trained analyst, `WtiStrategyState`, skill mutation tools | **schema drafted** — `NvdaStrategyState` with `NewsPattern` in `nvda_strategy_state.py`; the agent's own instructions and skills still WTI |
| `starter_agent/` | Hackable "build your own" agent | **pending** |

Deliberately **not** copied: the energy notebooks, the committed WTI prediction YAMLs under `data/`, the 52 cached curriculum news files, the trained `wti-strategy-trained/` skill state, and the oil forecast animation. Those are WTI results, not scaffolding — the equivalents are produced here from NVDA runs. The energy `specs/` were not copied either; the NVDA specs below were written fresh rather than edited down from WTI ones.

`adaptive_agent/skills/wti-strategy/` keeps its WTI name until the strategy-state schema is retargeted, so that the directory name never disagrees with its contents.

---

## Remaining scaffold work

| Step | Output |
|------|--------|
| Finish `signals.py` | the gate (`gate_pass`) and regime labels — each with tests |
| Finish the agent layer | an NVDA prompt builder, and renaming the remaining `build_wti_*` factories; wire the trajectory and shock `AgentPredictor`s; finalise `NvdaStrategyState` and move the adaptive agent onto it |

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
label_regimes(prices_df, window_days=21, ...)                    -> pd.DataFrame
```

`Window` and `PatternMetrics` are frozen dataclasses. **`flag_shock_windows`,
`sample_matched_controls`, `train_holdout_split` and `evaluate_pattern` are implemented and tested**; the other
functions keep their final signatures and raise `NotImplementedError` until their own tasks land.

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

A pattern graduates to the master strategy file only if *all five* criteria hold:

| Criterion | Threshold | Why |
|---|---|---|
| `n_matches` | ≥ 5 (`MIN_MATCHES`) | below this the confidence interval is too wide for a high precision to mean anything |
| `lift` (train) | ≥ 2.0 (`MIN_LIFT`) | must at least double the shock probability over the base rate |
| `p_value` (train) | < 0.05 (`MAX_P_VALUE`) | Fisher's exact, not chi-squared — cell counts are small by construction |
| `ci_low` | > 1.0 | the bootstrap interval on lift must exclude "no effect" |
| `lift` (holdout) | ≥ 1.5 (`MIN_HOLDOUT_LIFT`) | some shrinkage is honest; a pattern that exists only where it was found is not |

The conjunction is the point — each criterion alone is gameable. Event counts are also why the
shock threshold is ±5%: the clean post-cutoff holdout (Feb–Dec 2025) holds 13 events at ±5%
but only 4 at ±7%.

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
