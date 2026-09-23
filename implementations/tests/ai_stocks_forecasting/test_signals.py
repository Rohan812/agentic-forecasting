"""Tests for shock flagging and matched-control sampling in ``ai_stocks_forecasting.signals``.

Synthetic price paths with known answers.  Each test pins a property that is
easy to get subtly wrong and that would bias the graduation gate without
raising any error.  Every magnitude is expressed relative to ``SHOCK_THRESHOLD``
(``T``), so the tests keep meaning the same thing if the threshold changes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from ai_stocks_forecasting.paths import SHOCK_THRESHOLD
from ai_stocks_forecasting.signals import (
    CONTROL_EXCLUSION_DAYS,
    VOL_WINDOW_DAYS,
    Window,
    flag_shock_windows,
    sample_matched_controls,
)


T = SHOCK_THRESHOLD
CALM_VOL = 0.1 * T  # daily return std, in percent
VOLATILE_VOL = 0.35 * T


def _prices(returns_pct: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    """Business-day price path from a vector of daily simple returns in percent."""
    dates = pd.bdate_range(start, periods=len(returns_pct) + 1)
    values = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns_pct / 100.0]))
    return pd.DataFrame({"timestamp": dates, "value": values})


def test_shocks_are_flagged_both_ways_with_the_prior_session_as_cutoff() -> None:
    """Moves past the threshold count in both directions, and the cutoff is the prior *session*.

    Not tested at exactly ``T``: a return rebuilt from prices lands a few ulps
    either side of it, so an exact-boundary test would only test float rounding.
    """
    r = np.zeros(30)
    r[9] = 1.5 * T
    r[19] = -1.6 * T
    r[4] = 0.98 * T  # just below: not a shock
    prices = _prices(r)

    windows = flag_shock_windows(prices)
    assert [(w.direction, round(w.return_pct, 6)) for w in windows] == [
        ("up", round(1.5 * T, 6)),
        ("down", round(-1.6 * T, 6)),
    ]

    dates = pd.DatetimeIndex(prices["timestamp"])
    for w in windows:
        i = dates.get_loc(w.event_date)
        assert w.as_of == dates[i - 1], "as_of must be the previous trading session, not the previous calendar day."


def test_a_cluster_is_one_event_whose_cutoff_precedes_the_whole_episode() -> None:
    """Consecutive shocks merge, anchored on the largest move, with ``as_of`` before the first one.

    Anchoring the cutoff on the largest move instead would put coverage of the
    episode's opening days inside the agent's news window.
    """
    r = np.zeros(40)
    r[10], r[11], r[12] = 1.2 * T, -2.2 * T, 1.3 * T  # one three-day episode
    r[16] = 1.4 * T  # three sessions later: a separate event
    prices = _prices(r)
    dates = pd.DatetimeIndex(prices["timestamp"])

    windows = flag_shock_windows(prices)
    assert len(windows) == 2

    episode = windows[0]
    assert episode.event_date == dates[12], "Anchor on the largest move."
    assert (episode.direction, round(episode.return_pct, 6)) == ("down", round(-2.2 * T, 6))
    assert episode.as_of == dates[10], "Cutoff is the session before the episode's first shock."
    assert windows[1].event_date == dates[17]


def _regime_path(seed: int = 0) -> tuple[pd.DataFrame, list[int]]:
    """Alternating 45-session calm and volatile blocks, with shocks planted inside volatile blocks.

    Ordinary returns are clipped below the threshold, so the only shocks are the
    planted ones.  Returns the prices and the return indices of those shocks.
    """
    rng = np.random.default_rng(seed)
    blocks = [CALM_VOL, VOLATILE_VOL] * 3 + [CALM_VOL]
    r = np.concatenate([np.clip(rng.normal(0.0, vol, 45), -0.8 * T, 0.8 * T) for vol in blocks])
    shock_idx = [45 + 30, 135 + 30]  # 30 sessions into volatile blocks, so trailing vol is pure
    r[shock_idx[0]], r[shock_idx[1]] = 1.8 * T, -1.8 * T
    return _prices(r), shock_idx


def test_controls_avoid_shocks_and_their_neighbours() -> None:
    """No control is a shock or within the exclusion buffer; none repeats; a seed reproduces the draw."""
    prices, shock_idx = _regime_path()
    shocks = flag_shock_windows(prices)
    assert len(shocks) == len(shock_idx), "Precondition: only the planted shocks exist."

    controls = sample_matched_controls(shocks, prices, n_each=3, seed=11)
    assert len(controls) == 3 * len(shocks)
    assert len({c.event_date for c in controls}) == len(controls), "No session may be used twice."
    assert all(not c.is_shock and c.direction is None for c in controls)

    dates = pd.DatetimeIndex(prices["timestamp"])
    shock_pos = [dates.get_loc(w.event_date) for w in shocks]
    for c in controls:
        gap = min(abs(dates.get_loc(c.event_date) - p) for p in shock_pos)
        assert gap > CONTROL_EXCLUSION_DAYS, f"Control {c.event_date.date()} is {gap} sessions from a shock."

    assert controls == sample_matched_controls(shocks, prices, n_each=3, seed=11)


def test_controls_avoid_shocks_the_caller_did_not_pass_in() -> None:
    """A control may not sit next to *any* shock in the prices, not just the ones supplied.

    Callers pass subsets: controls for a training split are drawn while the
    holdout's shocks still sit in the price history.  A control beside one of
    those would carry that shock's news into the base rate.  Asking for far
    more controls than exist makes the sampler take every eligible session, so
    the check is deterministic.
    """
    rng = np.random.default_rng(1)
    r = np.clip(rng.normal(0.0, 0.3 * T, 200), -0.8 * T, 0.8 * T)
    r[60], r[90] = 1.8 * T, -1.8 * T  # 30 sessions apart: the second lies inside the first's candidate window
    prices = _prices(r)
    dates = pd.DatetimeIndex(prices["timestamp"])
    shocks = flag_shock_windows(prices)
    assert len(shocks) == 2, "Precondition: only the planted shocks exist."
    left_out = dates.get_loc(shocks[1].event_date)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # asking for more than exist warns, by design
        controls = sample_matched_controls(shocks[:1], prices, n_each=500, seed=0)

    assert controls, "Precondition: eligible sessions exist."
    for c in controls:
        gap = abs(dates.get_loc(c.event_date) - left_out)
        assert gap > CONTROL_EXCLUSION_DAYS, f"Control {c.event_date.date()} is {gap} sessions from an omitted shock."


def test_controls_come_from_the_shocks_volatility_regime() -> None:
    """Shocks sit in volatile blocks, so their controls must too, even though half the candidates are calm.

    This is the property that stops a pattern which merely tracks volatility
    from looking predictive of shocks.
    """
    prices, _ = _regime_path()
    shocks = flag_shock_windows(prices)
    controls = sample_matched_controls(shocks, prices, n_each=4, seed=3)

    close = prices.set_index("timestamp")["value"]
    vol_before = (close.pct_change() * 100).rolling(VOL_WINDOW_DAYS).std().shift(1)
    control_vol = vol_before.loc[[c.event_date for c in controls]]
    midpoint = (CALM_VOL + VOLATILE_VOL) / 2
    assert (control_vol > midpoint).all(), f"Calm-regime controls drawn: {control_vol[control_vol <= midpoint]}"


def test_controls_reject_non_shock_windows() -> None:
    """Passing controls back in as shocks is a caller error, not something to sample around."""
    prices, _ = _regime_path()
    control = Window(
        as_of=pd.Timestamp("2024-02-01"), event_date=pd.Timestamp("2024-02-02"), is_shock=False, return_pct=0.1
    )
    try:
        sample_matched_controls([control], prices)
    except ValueError as exc:
        assert "shock windows only" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for a non-shock window.")
