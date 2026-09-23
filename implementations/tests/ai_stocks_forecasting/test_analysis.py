"""Tests for the two scoring bugs fixed in ``ai_stocks_forecasting.analysis``.

Both were inherited from the energy/oil copy and both silently produced
plausible-looking numbers, which is why they are pinned here: an "80% coverage"
computed from the 20th/80th percentiles (a 60% interval), and a ``mae_horizon``
argument that was accepted but never applied.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from ai_stocks_forecasting.analysis import predictions_to_frame, score_backtest_results
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask


ORIGIN = datetime(2025, 1, 6)  # a Monday
QUANTILES = {0.1: 90.0, 0.2: 95.0, 0.5: 100.0, 0.8: 105.0, 0.9: 110.0}


def _result(horizons: list[int], points: dict[int, float]) -> BacktestResult:
    task = ForecastingTask(task_id="t", target_series_id="px", horizons=horizons, frequency="B", description="test")
    preds = [
        Prediction(
            predictor_id="p",
            task_id="t",
            issued_at=ORIGIN,
            as_of=ORIGIN,
            forecast_date=(pd.Timestamp(ORIGIN) + pd.offsets.BDay(h)).to_pydatetime(),
            payload=ContinuousForecast(point_forecast=points[h], quantiles=QUANTILES),
        )
        for h in horizons
    ]
    spec = BacktestSpec(task=task, start=ORIGIN, end=datetime(2025, 1, 7), stride=5, warmup=0)
    return BacktestResult(
        spec=spec,
        predictor_id="p",
        predictions=preds,
        scores=[1.0] * len(preds),
        mean_score=1.0,
        ran_at=ORIGIN,
        skipped_origins=0,
    )


def _service(actuals: dict[int, float]) -> DataService:
    """Actual prices on each forecast date, available long before the scoring cutoff."""
    dates = [pd.Timestamp(ORIGIN) + pd.offsets.BDay(h) for h in actuals]
    frame = pd.DataFrame({"timestamp": dates, "value": list(actuals.values()), "released_at": dates})
    svc = DataService()
    svc.register(
        "px",
        StaticFrameAdapter(frame),
        SeriesMetadata(series_id="px", description="px", source="test", units="USD", frequency="B"),
    )
    return svc


def test_80_percent_interval_runs_from_q10_to_q90() -> None:
    """An outcome between q10 and q20 is inside the 80% interval.

    It sits outside q20..q80, so the old 60%-interval code scored it as a miss
    and reported the interval as half its real width.
    """
    result = _result([5], {5: 100.0})
    svc = _service({5: 92.0})

    row = predictions_to_frame({"p": {"t": result}}, svc).iloc[0]
    assert row["inside80"] == 1.0
    assert row["width80"] == pytest.approx(20.0)
    assert (row["q20"], row["q80"]) == (95.0, 105.0), "q20/q80 columns must hold the real percentiles."

    assert score_backtest_results({"t": result}, svc)["coverage_80"] == pytest.approx(100.0)


def test_mae_is_computed_at_the_requested_horizon_only() -> None:
    """``mae_h21`` must not be diluted by the 5-day error."""
    result = _result([5, 21], {5: 100.0, 21: 100.0})
    svc = _service({5: 101.0, 21: 130.0})  # errors: 1 at h=5, 30 at h=21

    scores = score_backtest_results({"t": result}, svc, mae_horizon=21)
    assert scores["mae_h21"] == pytest.approx(30.0), "Pooled across horizons it would be 15.5."
