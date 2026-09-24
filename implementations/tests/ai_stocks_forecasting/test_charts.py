"""Contracts for the shock leaderboard behind the Phase 1 presentation charts."""

from __future__ import annotations

import pandas as pd
import pytest
from ai_stocks_forecasting.charts import CLIMATOLOGY_ARM, shock_leaderboard


def _row(spec: str, arm: str, day: str, p: float, y: int, leaked: bool = False) -> dict:
    return {
        "spec": spec,
        "arm": arm,
        "as_of": pd.Timestamp(day),
        "probability": p,
        "outcome": y,
        "brier": (p - y) ** 2,
        "quotes_unseen_close": leaked,
    }


def test_leaked_origin_leaves_every_arm_and_skill_uses_its_own_specs_climatology() -> None:
    """A leak drops the origin from all arms of that spec only, and skill is scored against that spec's climatology.

    Dropping it from the leaking arm alone would compare the agent and
    climatology on different origins.  Here the leak is the agent's one good
    call, so excluding it must turn its skill from positive to negative.
    """
    frame = pd.DataFrame(
        [
            _row("a", "Agent", "2025-03-03", 0.9, 1, leaked=True),
            _row("a", CLIMATOLOGY_ARM, "2025-03-03", 0.2, 1),
            _row("a", "Agent", "2025-03-04", 0.4, 0),
            _row("a", CLIMATOLOGY_ARM, "2025-03-04", 0.2, 0),
            _row("a", "Agent", "2025-03-05", 0.4, 0),
            _row("a", CLIMATOLOGY_ARM, "2025-03-05", 0.2, 0),
            # Another spec on the same date, not leaked: it must keep its origin.
            _row("b", "Agent", "2025-03-03", 0.1, 0),
            _row("b", CLIMATOLOGY_ARM, "2025-03-03", 0.5, 0),
        ]
    )

    everything = shock_leaderboard(frame)
    clean = shock_leaderboard(frame, exclude_leaked=True)

    assert everything.loc[("a", "Agent"), "brier_skill"] > 0
    assert clean.loc[("a", "Agent"), "n"] == clean.loc[("a", CLIMATOLOGY_ARM), "n"] == 2
    assert clean.loc[("a", "Agent"), "brier_skill"] == pytest.approx(1 - 0.16 / 0.04, abs=1e-3)
    assert clean.loc[("b", "Agent"), "n"] == 1
    assert clean.loc[("b", "Agent"), "brier_skill"] == pytest.approx(1 - 0.01 / 0.25, abs=1e-3)
