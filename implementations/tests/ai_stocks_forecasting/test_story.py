"""Timing contracts for the notebook analyses in ``story.py``.

A midnight origin has seen closes only up to the session before it.  Both
splits below must respect that, in opposite directions: the origin's own
session is *inside* the forecast window, and *outside* what the regime label
may use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ai_stocks_forecasting.story import failure_group, regime_crps_table, shock_window_crps


def _prices(returns_pct: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    days = pd.bdate_range(start, periods=len(returns_pct) + 1)
    close = 100 * np.cumprod([1.0] + [1 + r / 100 for r in returns_pct])
    return pd.DataFrame({"timestamp": days, "value": close})


def test_shock_on_the_origin_session_counts_as_inside_the_window() -> None:
    """A ±5% move on the origin's own session happens after a midnight forecast, so it is in the window."""
    prices = _prices([0.1] * 10 + [-8.0] + [0.1] * 10)
    shock_day = prices["timestamp"].iloc[11]
    frame = pd.DataFrame(
        {
            "predictor": ["A", "A"],
            "horizon": [5, 5],
            "as_of": [shock_day, shock_day + pd.offsets.BDay(1)],
            "forecast_date": [shock_day + pd.offsets.BDay(5), shock_day + pd.offsets.BDay(6)],
            "crps": [9.0, 1.0],
        }
    )
    table = shock_window_crps(frame, prices)
    assert table.loc[("A", 5), "shock in window"] == 9.0
    assert table.loc[("A", 5), "no shock"] == 1.0


def test_regime_label_comes_from_the_session_before_the_origin() -> None:
    """The label a midnight origin gets must not reflect its own session's move.

    The history is calm for over a year, then the origin's own session is a
    large move.  Read on that session, the label would already be ``high``.
    """
    rng = np.random.default_rng(0)
    calm = list(rng.normal(0, 1.0, 400))
    prices = _prices(calm + [12.0, 0.5])
    origin = prices["timestamp"].iloc[-2]  # the 12% session
    frame = pd.DataFrame({"predictor": ["A"], "as_of": [origin], "crps": [1.0]})
    table = regime_crps_table(frame, prices)
    assert not any(col.startswith("high") for col in table.columns)


def test_failure_groups_follow_the_evidence_on_each_half() -> None:
    """Each recorded candidate lands in the group its two halves of evidence imply."""
    strong_train = {"n_matches": 11, "lift": 3.2, "p_value": 0.03, "ci_low": 1.2}
    weak_train = {"n_matches": 49, "lift": 0.77, "p_value": 0.9, "ci_low": 0.46}
    strong_holdout = {"n_matches": 5, "n_hits": 4, "lift": 6.8, "p_value": 0.011}
    weak_holdout = {"n_matches": 4, "n_hits": 1, "lift": 3.6, "p_value": 0.25}
    single_hit = {"n_matches": 1, "n_hits": 1, "lift": 25.6, "p_value": 0.135}

    assert failure_group({"train": strong_train, "holdout": weak_holdout}) == "real before the cutoff, not after"
    assert failure_group({"train": weak_train, "holdout": strong_holdout}) == "real after the cutoff, not before"
    assert failure_group({"holdout": single_hit}) == "one event"
    assert (
        failure_group({"holdout": {"n_matches": 0, "n_hits": 0, "lift": float("nan"), "p_value": 1.0}}) == "no effect"
    )
