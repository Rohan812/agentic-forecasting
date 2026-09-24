"""The shock smoke report scores every predictor on the same origins.  Offline."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from ai_stocks_forecasting.shock_smoke import SmokeSpec, scored_frame
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
