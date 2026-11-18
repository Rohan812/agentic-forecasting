"""Tests for the reactive next-day-range experiment: no look-ahead, the interval score, and the decision rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ai_stocks_forecasting import reactive


@pytest.fixture
def synthetic_close(monkeypatch: pytest.MonkeyPatch) -> pd.Series:
    """Patch a random-walk close, one earnings report and a flat VXN into the module."""
    idx = pd.bdate_range("2019-01-01", "2026-03-31")
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx)))), index=idx)
    monkeypatch.setattr(reactive, "load_close", lambda ticker: close.copy())
    monkeypatch.setattr(reactive, "earnings_reports", lambda ticker: {pd.Timestamp("2026-02-03")})
    vxn = pd.Series(25.0, index=idx)
    monkeypatch.setattr(reactive, "load_vxn", vxn.copy)
    return close


@pytest.mark.parametrize("arm", reactive.BASELINE_ARMS)
def test_baseline_ignores_the_target_session_and_after(
    synthetic_close: pd.Series, monkeypatch: pytest.MonkeyPatch, arm: reactive.Arm
) -> None:
    """Changing the close on or after the target session must not change that session's forecast."""
    day = pd.Timestamp("2026-02-04")  # the earnings reaction, so vol21_earnings is exercised
    before = reactive.baseline_forecasts("AMD", arm, [day], factor=3.0)[0]["quantiles"]
    shocked = synthetic_close.copy()
    shocked.loc[day:] *= 1.5
    monkeypatch.setattr(reactive, "load_close", lambda ticker: shocked.copy())
    after = reactive.baseline_forecasts("AMD", arm, [day], factor=3.0)[0]["quantiles"]
    assert after == pytest.approx(before)


def test_earnings_factor_widens_only_the_reaction_session(synthetic_close: pd.Series) -> None:
    """The calendar arm equals vol21 except on the session after a report."""
    days = [pd.Timestamp("2026-02-03"), pd.Timestamp("2026-02-04")]
    plain = reactive.baseline_forecasts("AMD", "vol21", days, factor=3.0)
    calendar = reactive.baseline_forecasts("AMD", "vol21_earnings", days, factor=3.0)
    width = lambda row: row["quantiles"]["0.9"] - row["quantiles"]["0.1"]  # noqa: E731
    assert width(calendar[0]) == pytest.approx(width(plain[0]))
    assert width(calendar[1]) > 2.5 * width(plain[1])


def test_interval_score_charges_ten_times_the_miss() -> None:
    """Width plus (2 / 0.2) x the distance outside the 80% range, in percent of the previous close."""
    quantiles = {str(q): 100.0 for q in reactive.STANDARD_QUANTILES} | {"0.1": 98.0, "0.9": 102.0}
    row = {"prev_close": 100.0, "quantiles": quantiles}
    inside = reactive.score_row(row, actual=101.0)
    below = reactive.score_row(row, actual=97.0)
    assert inside["interval_score"] == pytest.approx(4.0)
    assert inside["covered"] == 1.0
    assert below["interval_score"] == pytest.approx(4.0 + 10 * 1.0)
    assert below["covered"] == 0.0


def _paired_frame(diffs: np.ndarray) -> pd.DataFrame:
    days = pd.bdate_range("2026-01-02", periods=len(diffs))
    rows = []
    for day, d in zip(days, diffs, strict=True):
        rows.append({"target_session": day, "ticker": "AMD", "arm": "b", "interval_score": 5.0})
        rows.append({"target_session": day, "ticker": "AMD", "arm": "a", "interval_score": 5.0 + d})
    return pd.DataFrame(rows)


def test_improvement_needs_the_whole_interval_below_zero() -> None:
    """A consistent gain is claimed; a noisy one with a slightly negative mean is not."""
    steady = reactive.paired_difference(_paired_frame(np.full(120, -0.5)), "a", "b")
    noisy = reactive.paired_difference(_paired_frame(np.random.default_rng(1).normal(-0.05, 2.0, 120)), "a", "b")
    assert steady["improves"]
    assert not noisy["improves"]
