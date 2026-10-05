"""Contracts for the discovery engine: statistics are computed here, and direction is respected."""

from __future__ import annotations

import pandas as pd
import pytest
from ai_stocks_forecasting import signals
from ai_stocks_forecasting.discovery import StudySet, evaluate_candidate


def _window(day: str, shock: bool, direction: str | None = None) -> signals.Window:
    as_of = pd.Timestamp(day)
    return signals.Window(
        as_of=as_of, event_date=as_of + pd.offsets.BDay(1), is_shock=shock, return_pct=0.0, direction=direction
    )


def _study(train: list[signals.Window]) -> StudySet:
    days = pd.bdate_range("2020-01-01", "2025-12-31")
    prices = pd.DataFrame({"timestamp": days, "value": 100.0})
    study = StudySet(
        prices=prices,
        train=train,
        holdout=train,
        train_period=(pd.Timestamp("2020-01-01"), pd.Timestamp("2025-01-31")),
        holdout_period=(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-12-31")),
    )
    for split in ("train", "holdout"):
        study._base_rates[(split, "either")] = 0.12
        study._base_rates[(split, "up")] = 0.06
        study._base_rates[(split, "down")] = 0.06
    return study


def test_a_one_direction_pattern_gets_no_credit_for_moves_the_other_way() -> None:
    """A rule that fires before every down shock looks perfect pooled, and like nothing for up shocks.

    Scored as an "up" pattern, its matches before down shocks must count as
    misses, not hits, against the up-only base rate.
    """
    shocks_down = [_window(f"2021-03-{d:02d}", True, "down") for d in (1, 2, 3, 4, 5, 8, 9, 10)]
    shocks_up = [_window(f"2022-03-{d:02d}", True, "up") for d in (1, 2, 3, 4, 7, 8, 9, 10)]
    controls = [_window(f"2023-03-{d:02d}", False) for d in (1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 20, 21, 22)]
    study = _study(shocks_down + shocks_up + controls)
    fires_before_down = {w.as_of for w in shocks_down}

    def rule(w: signals.Window, s: StudySet) -> bool:
        return w.as_of in fires_before_down

    pooled = evaluate_candidate(study, "P-1", "cue", rule, "either")
    up_only = evaluate_candidate(study, "P-1", "cue", rule, "up")

    assert pooled.train.n_hits == 8
    assert pooled.train.lift > 2
    assert up_only.train.n_hits == 0
    assert up_only.train.lift == pytest.approx(0.0)
    assert up_only.train.base_rate == pytest.approx(0.06)
