"""Shock backtest contracts: identical origins across predictors, and no agent scoring on holdout windows.  Offline."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from ai_stocks_forecasting.data import NVDA_SERIES_ID
from ai_stocks_forecasting.shock_smoke import SmokeSpec, check_agent_origins_are_fresh, scored_frame
from aieng.forecasting.data import DataService
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction


def _result(spec: SmokeSpec, predictor_id: str, as_ofs: list[pd.Timestamp]) -> BacktestResult:
    preds = [
        Prediction(
            predictor_id=predictor_id,
            task_id="nvda_shock_1d",
            issued_at=datetime(2026, 9, 24),
            as_of=a.to_pydatetime(),
            forecast_date=(a + pd.offsets.BDay(1)).to_pydatetime(),
            payload=BinaryForecast(probability=0.2),
        )
        for a in as_ofs
    ]
    scores = [(0.2 - spec.expected[a]) ** 2 for a in as_ofs]
    return BacktestResult(
        spec=spec.backtest_spec(),
        predictor_id=predictor_id,
        predictions=preds,
        scores=scores,
        mean_score=sum(scores) / len(scores),
        ran_at=datetime(2026, 9, 24),
    )


def test_an_origin_one_predictor_skipped_is_dropped_for_all() -> None:
    """If the agent loses an origin (a proxy 503 after retries), the baseline is not scored on it either."""
    origins = [pd.Timestamp("2025-02-04"), pd.Timestamp("2025-02-07"), pd.Timestamp("2025-03-05")]
    spec = SmokeSpec(spec_id="t", warmup=0, expected=dict(zip(origins, [1, 0, 1], strict=True)))
    results = {
        "baseline": _result(spec, "baseline", origins),
        "agent": _result(spec, "agent", origins[:2]),
    }
    frame, dropped = scored_frame(results, spec)
    assert dropped == 1
    assert frame.groupby("predictor_id")["as_of"].apply(set).tolist() == [set(origins[:2])] * 2


def test_agent_arm_is_refused_on_a_holdout_shock_window() -> None:
    """A spec that scores the with-topics agent on a gate-holdout shock origin must not run."""
    dates = pd.bdate_range("2020-01-02", "2025-06-30")
    values = pd.Series(100.0, index=dates)
    values[values.index >= pd.Timestamp("2025-03-04")] = 110.0  # a +10% shock on 2025-03-04
    service = DataService()
    service.register(
        NVDA_SERIES_ID,
        StaticFrameAdapter(
            pd.DataFrame({"timestamp": dates, "value": values.to_numpy(), "released_at": dates + pd.offsets.BDay(1)})
        ),
        SeriesMetadata(series_id=NVDA_SERIES_ID, description="t", source="t", units="USD/share", frequency="B"),
    )
    shock_origin = pd.Timestamp("2025-03-03")  # the window's as_of: the session before the shock
    fresh_origin = pd.Timestamp("2025-04-01")

    check_agent_origins_are_fresh(SmokeSpec("t", 0, {fresh_origin: 0}, ("agent",)), service)
    check_agent_origins_are_fresh(SmokeSpec("t", 0, {shock_origin: 1}, ("agent_notopics",)), service)
    with pytest.raises(ValueError, match="2025-03-03"):
        check_agent_origins_are_fresh(SmokeSpec("t", 0, {shock_origin: 1}, ("agent",)), service)
