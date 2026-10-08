"""Contracts for the basket study: relative shocks, calendar alignment and the shift test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ai_stocks_forecasting import signals
from ai_stocks_forecasting.basket import (
    SHOCK_SIGMAS,
    BasketStudy,
    BasketWindow,
    Calendar,
    _standardised_prices,
    calendar_shift_p,
    standardised_returns,
)


def _close(seed: int = 0, n: int = 600) -> pd.Series:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-06-03", periods=n)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, n)), index=days)


def test_a_percent_threshold_on_the_standardised_series_flags_exactly_the_relative_shocks() -> None:
    """``signals`` thresholds percent returns, so the synthetic series must return exactly z percent.

    Then a threshold of ``SHOCK_SIGMAS`` percent flags the sessions whose real
    return reached that many trailing standard deviations, no more and no fewer.
    """
    close = _close()
    z = standardised_returns(close)
    synthetic = _standardised_prices(close).set_index("timestamp")["value"]
    synthetic_pct = synthetic.pct_change().dropna() * 100
    assert np.allclose(synthetic_pct.to_numpy(), z.iloc[1:].to_numpy())
    assert ((synthetic_pct.abs() >= SHOCK_SIGMAS) == (z.iloc[1:].abs() >= SHOCK_SIGMAS)).all()


def test_earnings_react_the_next_session_and_a_weekend_fed_move_the_next_session_too() -> None:
    """Earnings come after the close; the March 2020 Sunday Fed cut is first reflected on Monday."""
    sessions = {"NVDA": pd.bdate_range("2020-01-01", "2020-12-31")}
    calendar = Calendar(sessions)

    earnings = calendar.reaction_sessions("earnings", "NVDA")
    assert pd.Timestamp("2020-02-14") in earnings  # announced 2020-02-13 after the close
    assert pd.Timestamp("2020-02-13") not in earnings

    fomc = calendar.reaction_sessions("fomc", "NVDA")
    assert pd.Timestamp("2020-03-16") in fomc  # Sunday 2020-03-15 cut
    assert pd.Timestamp("2020-01-29") in fomc  # an ordinary decision day reacts the same day

    shifted = calendar.reaction_sessions("fomc", "NVDA", shift=3)
    assert pd.Timestamp("2020-02-03") in shifted  # three sessions after 2020-01-29


def test_shift_test_separates_real_alignment_from_no_alignment() -> None:
    """Shocks placed on Fed days give a small shift p-value; shocks placed elsewhere do not."""
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    close = {"NVDA": pd.Series(100.0, index=idx)}
    study = BasketStudy(close=close, train=[], holdout=[])
    fed_days = list(study.calendar.reaction_sessions("fomc", "NVDA"))

    def windows(shock_days: list[pd.Timestamp]) -> list[BasketWindow]:
        out = []
        for day in shock_days:
            as_of = idx[idx.get_loc(day) - 1]
            out.append(
                BasketWindow(
                    "NVDA", signals.Window(as_of=as_of, event_date=day, is_shock=True, return_pct=5.0, direction="up")
                )
            )
        return out

    aligned = calendar_shift_p(study, windows(fed_days[:20]), "fomc")
    misaligned = calendar_shift_p(study, windows([idx[idx.get_loc(d) + 7] for d in fed_days[:20]]), "fomc")
    assert aligned < 0.01
    assert misaligned > 0.2


def test_a_sessions_yardstick_excludes_its_own_return() -> None:
    """A shock is measured against the 21 returns before it; its own move must not inflate the yardstick."""
    close = _close()
    spiked = close.copy()
    spiked.iloc[-1] = spiked.iloc[-2] * 1.30
    ret = spiked.pct_change()
    expected = ret.iloc[-1] / ret.iloc[-22:-1].std()
    assert np.isclose(standardised_returns(spiked).iloc[-1], expected)
