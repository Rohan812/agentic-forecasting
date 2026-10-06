"""The exploratory tier's no-LLM forecaster: pattern precision on firing days, climatology otherwise."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from ai_stocks_forecasting.exploratory import ExploratoryPattern, PatternClimatologyPredictor, gate_a_reasons
from ai_stocks_forecasting.signals import PatternMetrics
from ai_stocks_forecasting.tasks import nvda_shock_task


def _context(day: str) -> SimpleNamespace:
    history = pd.DataFrame({"timestamp": pd.bdate_range("2025-01-01", periods=100), "value": [0.0] * 90 + [1.0] * 10})
    return SimpleNamespace(as_of=pd.Timestamp(day) + pd.Timedelta(hours=20), get_series=lambda _sid: history)


def test_forecast_is_pattern_precision_on_a_firing_day_and_climatology_otherwise() -> None:
    """2026-02-25 is an earnings date, so the earnings pattern fires; the next day it does not."""
    earnings = ExploratoryPattern("X-1", "earnings", "earnings_after_close", "either", 0.38, 3.2, 3.6)
    predictor = PatternClimatologyPredictor([earnings])
    task = nvda_shock_task()

    on_earnings = predictor.predict(task, _context("2026-02-25"))[0]
    next_day = predictor.predict(task, _context("2026-02-26"))[0]

    assert on_earnings.payload.probability == pytest.approx(0.38)
    assert on_earnings.metadata["patterns_fired"] == ["X-1"]
    assert next_day.payload.probability == pytest.approx(0.10)
    assert next_day.metadata["patterns_fired"] == []


def test_gate_a_drops_the_interval_and_holdout_p_but_keeps_training_significance() -> None:
    """Earnings' real evidence passes Gate A; a training p of 0.29 does not, however strong the holdout."""

    def metrics(n, lift, p, ci_low):  # noqa: ANN001, ANN202
        return PatternMetrics(
            n_windows=238,
            n_matches=n,
            n_hits=n // 2,
            precision=0.3,
            base_rate=0.12,
            lift=lift,
            p_value=p,
            ci_low=ci_low,
            ci_high=9.0,
        )

    assert gate_a_reasons(metrics(11, 3.17, 0.03, 1.21), metrics(4, 3.58, 0.25, 0.0)) == []
    assert gate_a_reasons(metrics(31, 1.27, 0.29, 0.6), metrics(5, 6.8, 0.01, 2.0)) != []
