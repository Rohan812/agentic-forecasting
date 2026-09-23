"""Tests for ``aieng.forecasting.methods.numerical.darts_arima``.

Covers the two behaviours that are easy to get wrong and expensive when they
are: calendar gaps must not reach the model as ``NaN``, and the log transform
must round-trip to a strictly positive forecast.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical import DartsAutoARIMAPredictor


AS_OF = datetime(2024, 1, 16)


class _InMemoryAdapter(BaseAdapter):
    """Adapter that returns a supplied DataFrame unchanged."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def fetch(self) -> pd.DataFrame:
        """Return the supplied DataFrame."""
        return self._df.copy()


def _price_path(
    n: int = 500, start: float = 100.0, drift: float = 0.0005, daily_vol: float = 0.02, seed: int = 0
) -> pd.DataFrame:
    """Geometric random walk on business days, ending the day before ``AS_OF``."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp(AS_OF) - pd.offsets.BDay(1), periods=n)
    values = start * np.exp(np.cumsum(rng.normal(drift, daily_vol, n)))
    return pd.DataFrame({"timestamp": dates, "value": values})


def _service(df: pd.DataFrame) -> DataService:
    service = DataService()
    service.register(
        "price",
        _InMemoryAdapter(df),
        SeriesMetadata(series_id="price", description="Synthetic price", source="test", units="USD", frequency="B"),
    )
    return service


def _task(horizons: list[int]) -> ForecastingTask:
    return ForecastingTask(
        task_id="synthetic_price",
        target_series_id="price",
        horizons=horizons,
        frequency="B",
        description="Synthetic business-day price forecast for unit tests.",
    )


def test_calendar_gap_near_the_end_is_forward_filled() -> None:
    """A missing business day behaves exactly as if the market had not moved that day.

    Regression test for a production failure: an unscheduled exchange closure in
    the second-to-last row reached statsforecast as ``NaN``, flipped the selected
    order, and produced a -76% forecast on NVDA where the filled series gives
    +2.7%.  With ``num_samples=1`` the forecast is the deterministic mean path, so
    the gapped and pre-filled series must give identical forecasts.

    ``seed=1`` is deliberate: on this path the ``NaN`` flips AutoARIMA from
    ``(1,2,2)`` to ``(0,2,0)`` and moves the 21-day forecast by about 20%, so the
    test fails without the forward-fill.  Many seeds do not expose the bug at all
    — when AutoARIMA settles on a plain random walk the forecast is the last
    value and a gap one row back cannot move it.
    """
    full = _price_path(seed=1)
    gap_day = full["timestamp"].iloc[-2]

    gapped = full[full["timestamp"] != gap_day]
    prefilled = full.copy()
    prefilled.loc[prefilled["timestamp"] == gap_day, "value"] = full["value"].iloc[-3]

    # Precondition: the gap must sit inside the window the model sees. A gap at
    # or past the final visible row is never materialised, and the test would
    # pass vacuously.
    gapped_ctx = _service(gapped).context(AS_OF)
    assert gapped_ctx.get_series("price")["timestamp"].max() > gap_day

    task = _task([5, 21])
    predictor = DartsAutoARIMAPredictor(num_samples=1)
    from_gapped = predictor.predict(task, gapped_ctx)
    from_prefilled = predictor.predict(task, _service(prefilled).context(AS_OF))

    for a, b in zip(from_gapped, from_prefilled, strict=True):
        assert a.payload.point_forecast == pytest.approx(b.payload.point_forecast, rel=1e-9)


def test_log_transform_keeps_a_collapsed_stock_positive() -> None:
    """A stock that fell ~98% gets a strictly positive distribution only in log space.

    The series falls from about $55 to $1.27.  A raw-level fit estimates its
    error variance in dollars over the whole history, so near the bottom it puts
    its 5th percentile around -$10 — a large share of probability on an
    impossible price.  The first assertion pins that down so the test keeps
    proving the scenario is live; the second is the actual contract.
    """
    df = _price_path(n=500, start=50.0, drift=-0.006, daily_vol=0.05, seed=0)
    task, ctx = _task([21]), _service(df).context(AS_OF)

    raw = DartsAutoARIMAPredictor(num_samples=300).predict(task, ctx)[0].payload.quantiles
    assert raw[0.05] < 0, "Scenario no longer exercises the failure log_transform exists to prevent."

    predictor = DartsAutoARIMAPredictor(num_samples=300, log_transform=True)
    logged = predictor.predict(task, ctx)[0].payload.quantiles
    assert predictor.predictor_id == "darts_autoarima_log"
    assert min(logged.values()) > 0, "log_transform must never forecast a non-positive price."
    assert logged[0.95] > logged[0.05], "Degenerate (point) distribution."


def test_log_transform_rejects_non_positive_values() -> None:
    """A zero or negative value fails loudly, naming the series, not as ``nan``."""
    df = _price_path()
    df.loc[df.index[10], "value"] = 0.0
    with pytest.raises(ValueError, match="'price'.*<= 0"):
        DartsAutoARIMAPredictor(log_transform=True).predict(_task([5]), _service(df).context(AS_OF))
