"""The discovery engine: test a candidate pattern against labelled windows and graduate it only through the gate.

A **candidate** is a rule that answers yes or no for one window, using only what
was knowable at that window's ``as_of``: the price history up to that session,
the published earnings calendar, or (later) a fenced news label.  The engine
evaluates the rule on the discovery windows (2020-2024 shocks and their matched
controls) and on the post-cutoff holdout (Feb-Dec 2025), then runs the gate in
:mod:`~ai_stocks_forecasting.signals`.

The statistics are always computed here, never accepted from a caller.  A study
agent proposes rules; it cannot hand in its own lift or p-value.  Every
candidate, passed or rejected, is written to the experiment's research trail,
and only a passing one reaches the master strategy file, whose schema re-checks
the gate on load (:class:`~ai_stocks_forecasting.adaptive_agent.nvda_strategy_state.NvdaStrategyState`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from ai_stocks_forecasting import signals
from ai_stocks_forecasting.paths import SHOCK_THRESHOLD


EARNINGS_ANNOUNCEMENTS: tuple[str, ...] = (
    "2020-02-13", "2020-05-21", "2020-08-19", "2020-11-18",
    "2021-02-24", "2021-05-26", "2021-08-18", "2021-11-17",
    "2022-02-16", "2022-05-25", "2022-08-24", "2022-11-16",
    "2023-02-22", "2023-05-24", "2023-08-23", "2023-11-21",
    "2024-02-21", "2024-05-22", "2024-08-28", "2024-11-20",
    "2025-02-26", "2025-05-28", "2025-08-27", "2025-11-19",
)  # fmt: skip
"""NVDA quarterly results dates, 2020-2025 (yfinance ``get_earnings_dates``).

NVDA reports after the close, so the price reaction is the **next** session.
The dates are scheduled weeks in advance, so using them at a window's
``as_of`` is not look-ahead."""

STUDY_START = "2020-01-01"
"""First session of the discovery history, matching the shock-anchor calibration window."""

STUDY_SEED = 0
"""Seed for matched-control sampling, fixed so every experiment scores against the same windows."""

Direction = Literal["up", "down", "either"]
Rule = Callable[[signals.Window, "StudySet"], bool]
"""A candidate pattern: given a window and the study context, did its signature appear by ``as_of``?"""

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


@dataclass
class StudySet:
    """The labelled windows every experiment is scored on, plus the context rules may read.

    Build it once with :func:`build_study_set`; it is deterministic for a given
    price history and :data:`STUDY_SEED`.
    """

    prices: pd.DataFrame
    train: list[signals.Window]
    holdout: list[signals.Window]
    train_period: tuple[pd.Timestamp, pd.Timestamp]
    holdout_period: tuple[pd.Timestamp, pd.Timestamp]
    threshold_pct: float = SHOCK_THRESHOLD
    _base_rates: dict[tuple[str, Direction], float] = field(default_factory=dict)

    @cached_property
    def close(self) -> pd.Series:
        """Close indexed by session date."""
        return self.prices.set_index(pd.to_datetime(self.prices["timestamp"]).dt.normalize())["value"].sort_index()

    @cached_property
    def regimes(self) -> pd.Series:
        """Volatility regime per session, known at that session's close (no look-ahead)."""
        labels = signals.label_regimes(self.prices)
        return labels.set_index(pd.to_datetime(labels["timestamp"]).dt.normalize())["regime"].sort_index()

    def base_rate(self, split: Literal["train", "holdout"], direction: Direction = "either") -> float:
        """Share of sessions in the split's period that were shocks in ``direction``."""
        key = (split, direction)
        if key not in self._base_rates:
            start, end = self.train_period if split == "train" else self.holdout_period
            ret = self.close.pct_change().loc[start:end] * 100
            hit = ret.abs() >= self.threshold_pct
            if direction == "up":
                hit &= ret > 0
            elif direction == "down":
                hit &= ret < 0
            self._base_rates[key] = float(hit.mean())
        return self._base_rates[key]


def build_study_set(prices: pd.DataFrame, seed: int = STUDY_SEED, n_each: int = 1) -> StudySet:
    """Flag shocks from :data:`STUDY_START`, sample matched controls, and split train / holdout."""
    history = prices[pd.to_datetime(prices["timestamp"]) >= pd.Timestamp(STUDY_START)].reset_index(drop=True)
    shocks = signals.flag_shock_windows(history)
    controls = signals.sample_matched_controls(shocks, history, n_each=n_each, seed=seed)
    train, holdout = signals.train_holdout_split(shocks + controls)
    holdout_start = pd.Timestamp(signals.HOLDOUT_START)
    return StudySet(
        prices=history,
        train=train,
        holdout=holdout,
        train_period=(pd.Timestamp(STUDY_START), holdout_start - pd.Timedelta(days=1)),
        holdout_period=(holdout_start, pd.Timestamp(signals.HOLDOUT_END)),
    )


def _labels(windows: list[signals.Window], direction: Direction) -> list[bool]:
    if direction == "either":
        return [w.is_shock for w in windows]
    return [w.is_shock and w.direction == direction for w in windows]


@dataclass(frozen=True)
class CandidateResult:
    """One candidate's evidence on both splits and the gate's verdict."""

    pattern_id: str
    cue: str
    direction: Direction
    train: signals.PatternMetrics
    holdout: signals.PatternMetrics
    reasons: list[str]

    @property
    def graduates(self) -> bool:
        """True when the gate found nothing wrong."""
        return not self.reasons


def evaluate_candidate(
    study: StudySet, pattern_id: str, cue: str, rule: Rule, direction: Direction = "either"
) -> CandidateResult:
    """Score a rule on the train and holdout windows and run the gate.

    For a one-directional pattern, shocks in the other direction count as
    non-shocks and the base rate is the one-directional shock rate, so a rule
    cannot borrow strength from moves it does not predict.
    """
    metrics = {}
    for split, windows in (("train", study.train), ("holdout", study.holdout)):
        matches = [bool(rule(w, study)) for w in windows]
        metrics[split] = signals.evaluate_pattern(
            matches, _labels(windows, direction), study.base_rate(split, direction)
        )
    reasons = signals.gate_reasons(metrics["train"], metrics["holdout"])
    return CandidateResult(pattern_id, cue, direction, metrics["train"], metrics["holdout"], reasons)


# ── Built-in rules: knowable at as_of from prices and the calendar ────────────

_EARNINGS = frozenset(pd.Timestamp(d) for d in EARNINGS_ANNOUNCEMENTS)


def earnings_after_close(window: signals.Window, study: StudySet) -> bool:
    """Match when NVDA reported results after the close on ``as_of`` (next session = earnings reaction)."""
    return window.as_of.normalize() in _EARNINGS


def high_volatility_regime(window: signals.Window, study: StudySet) -> bool:
    """Match when trailing 21-session volatility at ``as_of`` is in the top third of earlier history."""
    regime = study.regimes.get(window.as_of.normalize())
    return regime == "high"


def low_volatility_regime(window: signals.Window, study: StudySet) -> bool:
    """Match when trailing 21-session volatility at ``as_of`` is in the bottom third of earlier history."""
    return study.regimes.get(window.as_of.normalize()) == "low"


BUILTIN_RULES: dict[str, Rule] = {
    "earnings_after_close": earnings_after_close,
    "high_volatility_regime": high_volatility_regime,
    "low_volatility_regime": low_volatility_regime,
}


# ── The research trail ─────────────────────────────────────────────────────────


def _metrics_dict(m: signals.PatternMetrics) -> dict[str, float | int]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.__dict__.items()}


def record_candidate(
    experiment_id: str, result: CandidateResult, rule_name: str, experiments_dir: Path | None = None
) -> Path:
    """Append one candidate's evidence and verdict to ``experiments/<id>/trail.yaml``."""
    path = (experiments_dir or EXPERIMENTS_DIR) / experiment_id / "trail.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    trail = yaml.safe_load(path.read_text()) if path.exists() else {"experiment_id": experiment_id, "candidates": []}
    trail["candidates"] = [c for c in trail["candidates"] if c["pattern_id"] != result.pattern_id]
    trail["candidates"].append(
        {
            "pattern_id": result.pattern_id,
            "cue": result.cue,
            "rule": rule_name,
            "direction": result.direction,
            "evaluated_on": date.today().isoformat(),
            "graduated": result.graduates,
            "rejection_reasons": result.reasons,
            "train": _metrics_dict(result.train),
            "holdout": _metrics_dict(result.holdout),
        }
    )
    path.write_text(yaml.safe_dump(trail, sort_keys=False, allow_unicode=True))
    return path
