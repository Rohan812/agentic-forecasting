"""Darts AutoARIMA predictor — probabilistic forecast via Monte Carlo sampling.

``DartsAutoARIMAPredictor`` wraps Darts ``AutoARIMA`` on the target series only
(univariate). Darts' ``AutoARIMA`` implementation used here does not support
exogenous covariates; this class does not expose any covariate parameters.

The probabilistic forecast is produced via Monte Carlo sampling (``num_samples``
draws from the predictive distribution).  Point forecast is the median;
quantiles use :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`
levels.

For multi-horizon tasks, the model is fitted once to ``n = max(task.horizons)``
and samples are extracted at each requested horizon index from the resulting
trajectory. This is more efficient than fitting once per horizon.

Missing dates are forward-filled before fitting.  A business-day (``"B"``)
calendar has a slot for every weekday, so every exchange holiday arrives as a
gap; forward-filling reads it as a no-change day, which is what a closed market
is.  Leaving the gaps as ``NaN`` is not safe: statsforecast's Kalman filter
accepts them silently, and a single gap near the end of the series can flip the
selected model order and corrupt the terminal state the forecast extrapolates
from.  On NVDA, one unscheduled closure in the second-to-last row turned a +2.7%
forecast into a -76% one.

Pass ``log_transform=True`` for strictly positive, multiplicative series such
as equity prices.  The model is then fitted on ``log(value)`` and samples are
mapped back with ``exp``.  AutoARIMA's differencing turns a log-price fit into a
model of log returns (it selects ``d = 1``), errors become proportional to the
price level rather than fixed in currency units, and forecasts cannot go
negative.  Because ``exp`` is monotonic, quantiles and the median transform
exactly — no bias correction is needed.

Usage::

    from aieng.forecasting.methods.darts_arima import DartsAutoARIMAPredictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = DartsAutoARIMAPredictor()
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"Mean CRPS: {result.mean_score:.4f}")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class DartsAutoARIMAPredictor(Predictor):
    """Probabilistic predictor wrapping Darts AutoARIMA (univariate).

    Fits AutoARIMA on the target series history available at the forecast
    origin, then generates a probabilistic trajectory via Monte Carlo sampling.
    One :class:`~aieng.forecasting.evaluation.prediction.Prediction` is
    returned per horizon step declared in ``task.horizons``.

    Parameters
    ----------
    num_samples : int
        Number of Monte Carlo samples used to build the predictive distribution.
        Higher values give smoother quantile estimates at the cost of compute.
        Default: 500.
    log_transform : bool
        Fit on ``log(value)`` and exponentiate the samples back.  Use for
        prices and other strictly positive series whose moves scale with their
        level.  Raises :class:`ValueError` if the series contains a
        non-positive value.  Default: ``False``, so existing callers are
        unchanged.  Changes :attr:`predictor_id`, so log and raw results never
        collide in the prediction registry.

    Notes
    -----
    - **Darts AutoARIMA** requires ``statsforecast`` (already a project
      dependency).  No additional install is needed.
    - AutoARIMA can be slow (seconds to tens of seconds per origin). For rapid
      iteration use
      :class:`~aieng.forecasting.methods.darts_regression.DartsLinearRegressionPredictor`
      instead.
    """

    def __init__(self, num_samples: int = 500, log_transform: bool = False) -> None:
        self._num_samples = num_samples
        self._log_transform = log_transform

    @property
    def predictor_id(self) -> str:
        """Return a stable string identifier for this predictor."""
        return "darts_autoarima_log" if self._log_transform else "darts_autoarima"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic AutoARIMA forecasts for every horizon in the task.

        Parameters
        ----------
        task : ForecastingTask
            Defines the target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.  All series returned respect
            ``context.as_of``.

        Returns
        -------
        list[Prediction]
            One ``ContinuousForecast`` per horizon step in ``task.horizons``,
            with ``point_forecast`` equal to the median of the predictive
            sample at that step.
        """
        from darts import TimeSeries  # noqa: PLC0415
        from darts.models import AutoARIMA  # noqa: PLC0415  # type: ignore[import-untyped]

        series_df = context.get_series(task.target_series_id)

        if self._log_transform:
            if (series_df["value"] <= 0).any():
                raise ValueError(
                    f"log_transform=True requires strictly positive values, but "
                    f"{task.target_series_id!r} contains a value <= 0."
                )
            series_df = series_df.assign(value=np.log(series_df["value"]))

        ts = TimeSeries.from_dataframe(
            series_df,
            time_col="timestamp",
            value_cols="value",
            fill_missing_dates=True,
            freq=task.frequency,
        )
        # fill_missing_dates inserts NaN for every calendar slot with no
        # observation (exchange holidays, on a "B" calendar). Forward-fill them:
        # see the module docstring for why leaving them in is dangerous.
        ts = TimeSeries.from_series(ts.to_series().ffill(), freq=task.frequency)

        model = AutoARIMA()
        model.fit(ts)

        # Fit once to max horizon; extract samples at each requested step.
        # all_values() shape: (n_steps, n_components, n_samples), 0-indexed.
        forecast_ts: Any = model.predict(
            n=task.horizon,
            num_samples=self._num_samples,
        )

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []

        for h in task.horizons:
            samples: np.ndarray = forecast_ts.all_values()[h - 1, 0, :]
            if self._log_transform:
                samples = np.exp(samples)
            payload = ContinuousForecast(
                point_forecast=float(np.median(samples)),
                quantiles={q: float(np.quantile(samples, q)) for q in STANDARD_QUANTILES},
            )
            forecast_date: datetime = (pd.Timestamp(context.as_of) + offset * h).to_pydatetime()
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=forecast_date,
                    payload=payload,
                )
            )

        return predictions
