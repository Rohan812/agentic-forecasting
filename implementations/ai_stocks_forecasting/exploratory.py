"""The exploratory tier: Gate A, its patterns, and the forecasters that use them in 2026.

Pre-registered in ``experiments/exploratory/preregistration.md``.  Gate A is
looser than the strict gate in :mod:`~ai_stocks_forecasting.signals`: a
pattern with no effect passes it 3-5% of the time per candidate instead of
0.1-0.2%.  Its patterns therefore live in their own file
(``experiments/exploratory/patterns.yaml``) and never reach the strict master
strategy file, whose schema still enforces the strict gate.

Two forecasters read the patterns:

- :class:`PatternClimatologyPredictor`: climatology, except on a day a pattern
  fires, where it forecasts that pattern's training precision.  No LLM, so it
  measures the patterns alone.
- :func:`build_agent_close_with_patterns`: the after-close shock agent, told
  each pattern's cue, evidence and whether it fires today.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from ai_stocks_forecasting import signals
from ai_stocks_forecasting.analyst_agent import build_nvda_multitask_news_config
from ai_stocks_forecasting.discovery import _EARNINGS, EXPERIMENTS_DIR
from ai_stocks_forecasting.news_labels import load_labels
from ai_stocks_forecasting.tasks import NVDA_CLOSE_SERIES_ID, TASK_SHOCK_SPEC_AFTER_CLOSE, NvdaMultitaskPromptBuilder
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentPredictor
from aieng.forecasting.methods.agentic.outputs import DiscreteAgentForecastOutput
from aieng.forecasting.methods.baselines.historical_frequency import HistoricalFrequencyPredictor
from aieng.forecasting.models import LITE_MODEL
from pydantic import Field


GATE_A_MIN_MATCHES = 5
GATE_A_MIN_LIFT = 1.5
GATE_A_MAX_P_VALUE = 0.10
GATE_A_MIN_HOLDOUT_LIFT = 1.0
"""Gate A thresholds, as pre-registered.  No lift-interval and no holdout p-value requirement."""

EXPLORATORY_DIR = EXPERIMENTS_DIR / "exploratory"
PATTERNS_PATH = EXPLORATORY_DIR / "patterns.yaml"


def gate_a_reasons(train: signals.PatternMetrics, holdout: signals.PatternMetrics) -> list[str]:
    """Return every Gate A criterion a candidate fails; an empty list means it enters the exploratory tier."""
    reasons = []
    if not train.n_matches >= GATE_A_MIN_MATCHES:
        reasons.append(f"training matches {train.n_matches} is below {GATE_A_MIN_MATCHES}")
    if not train.lift >= GATE_A_MIN_LIFT:
        reasons.append(f"training lift {train.lift:.2f} is below {GATE_A_MIN_LIFT}")
    if not train.p_value < GATE_A_MAX_P_VALUE:
        reasons.append(f"training p-value {train.p_value:.3g} is not below {GATE_A_MAX_P_VALUE}")
    if not holdout.lift >= GATE_A_MIN_HOLDOUT_LIFT:
        reasons.append(f"holdout lift {holdout.lift:.2f} is below {GATE_A_MIN_HOLDOUT_LIFT}")
    return reasons


@dataclass(frozen=True)
class ExploratoryPattern:
    """A pattern that passed Gate A, with what a forecaster needs to use it."""

    pattern_id: str
    cue: str
    rule: str
    """``earnings_after_close``, or ``news:<question_id>`` for a fenced news label."""
    direction: str
    train_precision: float
    train_lift: float
    holdout_lift: float

    def fires_on(self, day: pd.Timestamp) -> bool:
        """Return True if the pattern's signature is present by the end of ``day``."""
        day = pd.Timestamp(day).normalize()
        if self.rule == "earnings_after_close":
            return day in _EARNINGS
        if self.rule.startswith("news:"):
            label = load_labels(self.rule.removeprefix("news:")).get(day.date().isoformat())
            if label is None:
                raise KeyError(
                    f"{self.pattern_id} has no news label for {day.date()}; label 2026 days before forecasting."
                )
            return label["answer"] == "yes"
        raise ValueError(f"Unknown rule {self.rule!r}")


def save_patterns(patterns: list[ExploratoryPattern], path: Path = PATTERNS_PATH) -> None:
    """Write the exploratory tier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "gate": "A (exploratory; see preregistration.md)",
                "written_on": date.today().isoformat(),
                "patterns": [p.__dict__ for p in patterns],
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )


def load_patterns(path: Path = PATTERNS_PATH) -> list[ExploratoryPattern]:
    """Read the exploratory tier."""
    return [ExploratoryPattern(**p) for p in yaml.safe_load(path.read_text())["patterns"]]


def _session_day(context: ForecastContext) -> pd.Timestamp:
    return pd.Timestamp(context.as_of).normalize()


class PatternClimatologyPredictor(Predictor):
    """Climatology, replaced by a pattern's training precision on days a pattern fires."""

    def __init__(self, patterns: list[ExploratoryPattern]) -> None:
        self._patterns = patterns
        self._base = HistoricalFrequencyPredictor()

    @property
    def predictor_id(self) -> str:
        """Stable id for the registry."""
        return "climatology_patterns"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Forecast climatology, or the highest training precision among patterns firing today."""
        fired = [p for p in self._patterns if p.fires_on(_session_day(context))]
        out = []
        for pred in self._base.predict(task, context):
            meta = {**pred.metadata, "patterns_fired": [p.pattern_id for p in fired]}
            payload = pred.payload
            if fired:
                payload = payload.model_copy(update={"probability": max(p.train_precision for p in fired)})
            out.append(
                pred.model_copy(update={"predictor_id": self.predictor_id, "payload": payload, "metadata": meta})
            )
        return out


_PATTERN_SPEC_ADDENDUM = """

## Exploratory patterns

The payload's `exploratory_patterns` lists news or calendar patterns found by a discovery
study. They passed a **loosened** statistical gate, so treat them as evidence to weigh, not
rules: about one in twenty such patterns can be noise. For each pattern you get its cue, how
much more likely a shock was when it was present in 2020-2024 (`train_lift`), its share of
shocks when present (`train_precision`), its 2025 lift, and `fires_today`. When a pattern
fires today, weigh its evidence against your other reasoning; when none fires, they say
nothing about today."""


class PatternAwarePromptBuilder(NvdaMultitaskPromptBuilder):
    """The multitask prompt plus each exploratory pattern and whether it fires on the origin's session."""

    patterns: list[dict[str, Any]] = Field(default_factory=list)

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Build the usual payload and add ``exploratory_patterns``."""
        payload = json.loads(super().__call__(task=task, context=context))
        day = _session_day(context)
        payload["exploratory_patterns"] = [
            {
                "id": p["pattern_id"],
                "cue": p["cue"],
                "direction": p["direction"],
                "train_lift": round(p["train_lift"], 2),
                "train_precision": round(p["train_precision"], 3),
                "holdout_lift": round(p["holdout_lift"], 2),
                "fires_today": ExploratoryPattern(**p).fires_on(day),
            }
            for p in self.patterns
        ]
        return json.dumps(payload, indent=2)


def build_agent_close_with_patterns(patterns: list[ExploratoryPattern], model: str = LITE_MODEL) -> AgentPredictor:
    """Build the after-close shock agent with the exploratory tier in its input (``..._close_patterns``)."""
    config = build_nvda_multitask_news_config(model=model)
    return AgentPredictor(
        agent_config=config.model_copy(update={"name": f"{config.name}_close_patterns"}),
        prompt_builder=PatternAwarePromptBuilder(
            task_spec=TASK_SHOCK_SPEC_AFTER_CLOSE + _PATTERN_SPEC_ADDENDUM,
            price_series_id=NVDA_CLOSE_SERIES_ID,
            patterns=[p.__dict__ for p in patterns],
        ),
        output_schema=DiscreteAgentForecastOutput,
    )
