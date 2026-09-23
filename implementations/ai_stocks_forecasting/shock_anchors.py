"""Calibration anchors for the NVDA shock task, measured on 2020-2024.

The shock task asks the agent for ``P(|next-session return| >= SHOCK_THRESHOLD)``.
An LLM left to itself anchors that probability on a vague sense of "big move",
so :data:`~ai_stocks_forecasting.tasks.TASK_SHOCK_SPEC` gives it explicit
base-rate anchors per market condition.  This module is where those anchors
come from; re-run it after changing the threshold and update the spec string
from its output.

The window is 2020-2024 on purpose: it is the discovery window, strictly before
both scored windows (the 2025 backtest and the protected 2026 evaluation), so
anchoring the prompt on it leaks nothing about the outcomes being scored.

Usage, from the repository root::

    uv run python -m ai_stocks_forecasting.shock_anchors
"""

from __future__ import annotations

import pandas as pd
from ai_stocks_forecasting.data import NVDA_SERIES_ID, build_nvda_service
from ai_stocks_forecasting.paths import SHOCK_HORIZON, SHOCK_THRESHOLD


CALIBRATION_START = "2020-01-01"
CALIBRATION_END = "2024-12-31"

EARNINGS_ANNOUNCEMENTS_2020_2024: tuple[str, ...] = (
    "2020-02-13", "2020-05-21", "2020-08-19", "2020-11-18",
    "2021-02-24", "2021-05-26", "2021-08-18", "2021-11-17",
    "2022-02-16", "2022-05-25", "2022-08-24", "2022-11-16",
    "2023-02-22", "2023-05-24", "2023-08-23", "2023-11-21",
    "2024-02-21", "2024-05-22", "2024-08-28", "2024-11-20",
)  # fmt: skip
"""NVDA quarterly results dates.  NVDA reports after the close, so the price
reaction is the **next** session, which is what :func:`anchor_table` scores."""

CALM_VOL_PCT = 2.5
"""Trailing 21-session daily return std (percent) below which the tape is "calm"."""

ELEVATED_VOL_PCT = 3.5
"""Trailing 21-session daily return std (percent) above which volatility is elevated."""


def anchor_table(prices: pd.DataFrame, threshold_pct: float = SHOCK_THRESHOLD) -> pd.DataFrame:
    """Return the shock rate per market condition over the calibration window.

    Parameters
    ----------
    prices
        NVDA history with ``timestamp`` and ``value`` columns.
    threshold_pct
        Absolute 1-session move, in percent, that counts as a shock.

    Returns
    -------
    pd.DataFrame
        One row per condition with ``sessions``, ``shocks`` and ``rate``.

    Notes
    -----
    Trailing volatility is computed from returns **strictly before** each
    session (``shift(1)``), so a condition is something the agent could know at
    the origin.  The conditions overlap (an earnings reaction can also be an
    elevated-vol day); each row is a marginal rate, not a partition.
    """
    if SHOCK_HORIZON != 1:
        raise ValueError("anchor_table assumes a 1-session shock horizon.")
    series = prices.set_index(pd.to_datetime(prices["timestamp"]))["value"].astype(float)
    returns = series.pct_change() * 100.0
    trailing_vol = returns.rolling(21).std().shift(1)

    window = returns.loc[CALIBRATION_START:CALIBRATION_END]
    vol = trailing_vol.loc[window.index]
    shock = window.abs() >= threshold_pct
    after_shock = shock.shift(1, fill_value=False)

    sessions = window.index
    reactions = pd.DatetimeIndex(
        [sessions[sessions.searchsorted(pd.Timestamp(d), side="right")] for d in EARNINGS_ANNOUNCEMENTS_2020_2024]
    )
    is_reaction = sessions.isin(reactions)

    conditions: dict[str, pd.Series] = {
        "all sessions": pd.Series(True, index=sessions),
        f"no earnings, calm (vol < {CALM_VOL_PCT}%)": ~is_reaction & (vol < CALM_VOL_PCT),
        f"no earnings, normal ({CALM_VOL_PCT}-{ELEVATED_VOL_PCT}%)": ~is_reaction
        & (vol >= CALM_VOL_PCT)
        & (vol <= ELEVATED_VOL_PCT),
        f"no earnings, elevated (vol > {ELEVATED_VOL_PCT}%)": ~is_reaction & (vol > ELEVATED_VOL_PCT),
        "day after a shock": after_shock,
        "earnings reaction session": pd.Series(is_reaction, index=sessions),
    }
    rows = []
    for name, mask in conditions.items():
        hits = shock[mask]
        rows.append({"condition": name, "sessions": int(mask.sum()), "shocks": int(hits.sum()), "rate": hits.mean()})
    table = pd.DataFrame(rows).set_index("condition")
    up = int((window >= threshold_pct).sum())
    down = int((window <= -threshold_pct).sum())
    table.attrs["direction_split"] = f"{up} up / {down} down"
    return table


def main() -> None:
    """Print the anchor table for the committed threshold."""
    # Each close is released the next day, so this cutoff admits every close through CALIBRATION_END.
    as_of = (pd.Timestamp(CALIBRATION_END) + pd.Timedelta(days=1)).to_pydatetime()
    prices = build_nvda_service().get_series(NVDA_SERIES_ID, as_of)
    table = anchor_table(prices)
    print(f"NVDA |1-session return| >= {SHOCK_THRESHOLD}%, {CALIBRATION_START} -> {CALIBRATION_END}")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"direction: {table.attrs['direction_split']}")


if __name__ == "__main__":
    main()
