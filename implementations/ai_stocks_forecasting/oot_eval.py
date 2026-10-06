"""The one-time 2026 evaluation of the exploratory tier (pre-registered in ``experiments/exploratory/``).

Four shock forecasters at an after-close origin on every 2026 session whose next
session is in the price cache:

1. climatology (``HistoricalFrequencyPredictor``)
2. climatology + patterns (:class:`~ai_stocks_forecasting.exploratory.PatternClimatologyPredictor`)
3. the after-close agent
4. the after-close agent + patterns

News patterns need a fenced label for every 2026 origin first.  Everything
that calls an LLM is metered against a separate $20 ledger
(``experiments/exploratory/oot_budget.yaml``); an arm does not start unless
its worst-case estimate fits in what is left.

Usage, from ``implementations/``::

    uv run python -m ai_stocks_forecasting.oot_eval
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import litellm
import pandas as pd
import yaml
from ai_stocks_forecasting.baselines import PREDICTIONS_DIR, SPECS_DIR
from ai_stocks_forecasting.data import NVDA_SERIES_ID, build_nvda_service
from ai_stocks_forecasting.exploratory import (
    EXPLORATORY_DIR,
    ExploratoryPattern,
    PatternClimatologyPredictor,
    build_agent_close_with_patterns,
    load_patterns,
)
from ai_stocks_forecasting.news_labels import SEARCH_FEE_USD, BudgetLedger, CostMeter, label_windows
from ai_stocks_forecasting.paths import SHOCK_THRESHOLD
from ai_stocks_forecasting.shock_smoke import check_expected_outcomes, load_smoke_spec, scored_frame
from ai_stocks_forecasting.tasks import (
    SessionDatePredictor,
    build_nvda_shock_predictor_after_close,
    register_shock_series,
)
from aieng.forecasting.evaluation.artifacts import cached_backtest
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods.baselines.historical_frequency import HistoricalFrequencyPredictor


SPEC_ID = "nvda_shock_oot_2026"
OOT_BUDGET_USD = 20.0
OOT_LEDGER = EXPLORATORY_DIR / "oot_budget.yaml"
AGENT_COST_PER_ORIGIN_WORST = 0.0104 * 1.5 + 2 * SEARCH_FEE_USD
"""Worst-case agent cost per origin: 1.5x the measured token cost plus two searches."""
LABEL_COST_PER_WINDOW_WORST = 0.03


def write_oot_spec(prices: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the 2026 spec: every 2026 session whose next weekday is a session in the cache."""
    close = prices.set_index(pd.to_datetime(prices["timestamp"]).dt.normalize())["value"].sort_index()
    ret = close.pct_change() * 100
    sessions = close.loc["2026-01-01":].index
    origins = []
    for day in sessions:
        nxt = day + pd.offsets.BDay(1)
        if nxt in close.index:
            origins.append({"as_of": str(day.date()), "outcome": int(abs(ret.loc[nxt]) >= SHOCK_THRESHOLD)})
    spec = {
        "spec_id": SPEC_ID,
        "warmup": 250,
        "summary": f"One-time 2026 evaluation of the exploratory tier: {len(origins)} after-close origins, "
        f"{sum(o['outcome'] for o in origins)} shocks",
        "origin_time": "after_close",
        "arms": ["climatology", "agent_close"],
        "origins": origins,
    }
    path = path or SPECS_DIR / f"{SPEC_ID}.yaml"
    header = (
        "# One-time 2026 evaluation of the exploratory tier.\n"
        "# Pre-registered in experiments/exploratory/preregistration.md; used once.\n"
        "# Origins: after-close (20:00) on every 2026 session whose next weekday is a session.\n"
        "# Run with: uv run python -m ai_stocks_forecasting.oot_eval\n\n"
    )
    path.write_text(header + yaml.safe_dump(spec, sort_keys=False))
    return path


def label_2026(patterns: list[ExploratoryPattern], days: list[pd.Timestamp], ledger: BudgetLedger) -> None:
    """Make the fenced news label for every 2026 origin for each news pattern."""
    for p in patterns:
        if not p.rule.startswith("news:"):
            continue
        qid = p.rule.removeprefix("news:")
        if len(days) * LABEL_COST_PER_WINDOW_WORST > ledger.remaining():
            raise RuntimeError(f"Labelling {qid} for 2026 could exceed the ${ledger.cap_usd:.0f} evaluation cap.")
        label_windows(qid, p.cue, days, ledger, run_cap_usd=ledger.remaining())


def _run_arm(predictor: Predictor, spec, service, ledger: BudgetLedger, uses_llm: bool) -> BacktestResult:  # noqa: ANN001
    n = len(spec.expected)
    if uses_llm and n * AGENT_COST_PER_ORIGIN_WORST > ledger.remaining():
        raise RuntimeError(
            f"{predictor.predictor_id} could cost ${n * AGENT_COST_PER_ORIGIN_WORST:.2f}; "
            f"only ${ledger.remaining():.2f} is left under the evaluation cap."
        )
    meter = CostMeter()
    litellm.callbacks = [*litellm.callbacks, meter]
    try:
        result = cached_backtest(predictor, spec.backtest_spec(), spec.spec_id, service, PREDICTIONS_DIR)
    finally:
        litellm.callbacks = [c for c in litellm.callbacks if c is not meter]
        if meter.usd:
            ledger.record(predictor.predictor_id, meter.usd, n, meter.searches)
    return result


def run() -> pd.DataFrame:
    """Run the four arms (loading any already cached) and return the scored frame."""
    service = register_shock_series(build_nvda_service())
    prices = service.get_series(NVDA_SERIES_ID, as_of=datetime.now())
    spec_path = SPECS_DIR / f"{SPEC_ID}.yaml"
    if not spec_path.exists():
        write_oot_spec(prices, spec_path)
    spec = load_smoke_spec(SPEC_ID)
    check_expected_outcomes(spec, service)
    ledger = BudgetLedger(path=OOT_LEDGER, cap_usd=OOT_BUDGET_USD)
    patterns = load_patterns()
    print(
        f"{SPEC_ID}: {len(spec.expected)} origins, {sum(spec.expected.values())} shocks; "
        f"{len(patterns)} exploratory pattern(s); evaluation spend so far ${ledger.spent_usd:.2f} of ${ledger.cap_usd:.0f}"
    )
    label_2026(patterns, list(spec.expected), ledger)

    arms: list[tuple[Predictor, bool]] = [
        (HistoricalFrequencyPredictor(), False),
        (PatternClimatologyPredictor(patterns), False),
        (build_nvda_shock_predictor_after_close(), True),
        (build_agent_close_with_patterns(patterns), True),
    ]
    results = {}
    for predictor, uses_llm in arms:
        wrapped = SessionDatePredictor(predictor)
        results[wrapped.predictor_id] = _run_arm(wrapped, spec, service, ledger, uses_llm)
        print(f"  {wrapped.predictor_id}: done; evaluation spend ${ledger.spent_usd:.2f}")
    frame, dropped = scored_frame(results, spec)
    fired = {d for d in spec.expected if any(p.fires_on(d) for p in patterns)}
    frame["pattern_fired"] = frame["as_of"].isin(fired)
    print(
        f"{dropped} origin(s) dropped by the identical-origins intersection; patterns fired on {len(fired)} origin(s)"
    )
    return frame


if __name__ == "__main__":
    run()
