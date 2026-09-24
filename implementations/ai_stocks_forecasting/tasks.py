"""Task specifications and agent predictor wiring for the NVDA experiment.

Implements the "one agent, three tasks" pattern: a single :class:`AgentConfig`
identity with task-specific prompt builders and output schemas supplied via
:class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`.  The
trajectory and shock tasks are NVDA; the scenario task is still WTI.

**Which session the shock task asks about.**  ``YFinanceDailyAdapter`` releases
each close one business day after the session, so a context at ``as_of`` ends
at the *previous* session's close.  The shock question, like
:class:`~ai_stocks_forecasting.signals.Window`, is about the session after
``as_of``: its close against the ``as_of`` close, which the agent has not seen.
News fenced at ``as_of`` may describe the ``as_of`` session, so the payload
says so rather than let the agent assume its price history is current.  See
:func:`nvda_shock_task`.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, ClassVar, Literal

import pandas as pd
from ai_stocks_forecasting.analyst_agent import (
    build_nvda_multitask_news_config,
    compress_history,
)
from ai_stocks_forecasting.data import NVDA_SERIES_ID
from ai_stocks_forecasting.paths import SHOCK_HORIZON, SHOCK_THRESHOLD
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, BinaryForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    DiscreteAgentForecastOutput,
)
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from aieng.forecasting.models import LITE_MODEL
from pydantic import BaseModel, Field


TaskKind = Literal["trajectory", "shock", "scenario"]


class NvdaMultitaskPromptBuilder(BaseModel):
    """Prompt builder for task-spec-driven agent calls (NB3).

    The system instruction is task-agnostic; the ask lives in ``task_spec``.
    The payload also includes ``horizons`` and ``standard_quantiles`` so
    trajectory (and any horizon-aware) tasks can read them without baking the
    forecasting contract into the system prompt.
    """

    task_spec: str

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        df = context.get_series(task.target_series_id)
        last_row = df.iloc[-1]
        payload: dict[str, Any] = {
            "task": task.task_id,
            "task_spec": self.task_spec,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "origin_price_usd": float(last_row["value"]),
            "last_close_date": str(pd.Timestamp(last_row["timestamp"]).date()),
            "target_history_csv": compress_history(df),
        }
        return json.dumps(payload, indent=2)


class ScenarioCard(BaseModel):
    """One scenario card from Task C agent output."""

    model_config = {"extra": "ignore"}

    name: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    wti_range_60d: list[float]
    point_estimate_60d: float
    key_drivers: list[str] = Field(default_factory=list)


class ScenarioAgentForecastOutput(AgentForecastOutput):
    """Track 2 scenario analysis output for the energy case study."""

    modality: ClassVar[Literal["continuous", "discrete"]] = "discrete"

    model_config = {"extra": "ignore"}

    scenarios: list[ScenarioCard]
    base_case: str
    reasoning: str = ""

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a JSON template for use in agent instruction strings.

        Returns
        -------
        str
            Indented JSON string showing the exact structure the agent must
            pass to ``set_model_response``.
        """
        template: dict[str, object] = {
            "scenarios": [
                {
                    "name": "<string>",
                    "description": "<string>",
                    "probability": "<float in [0, 1]>",
                    "wti_range_60d": ["<float_low>", "<float_high>"],
                    "point_estimate_60d": "<float>",
                    "key_drivers": ["<driver 1>", "<driver 2>"],
                }
            ],
            "base_case": "<scenario name>",
            "reasoning": "<paragraph>",
        }
        return json.dumps(template, indent=2)

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Convert scenario output to a metadata-rich prediction (Track 2 display)."""
        if len(task.horizons) != 1:
            raise ValueError("Scenario agent output expects exactly one task horizon.")

        horizon = task.horizons[0]
        issued_at = datetime.utcnow()
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        base_prob = float(sum(s.probability for s in self.scenarios))
        prediction_metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        prediction_metadata["scenarios"] = [s.model_dump() for s in self.scenarios]
        prediction_metadata["base_case"] = self.base_case
        if self.reasoning.strip():
            prediction_metadata["rationale"] = self.reasoning

        return [
            Prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                payload=BinaryForecast(probability=min(base_prob, 1.0)),
                metadata=prediction_metadata,
            )
        ]


# Task specification strings embedded in user prompts for NB3.
# Defined after the output classes so each spec can reference the
# corresponding prompt_schema_json() classmethod — single source of truth.
# Notebook 03 copies these into editable cells; the factory uses these defaults.

TASK_TRAJECTORY_SPEC = (
    "Forecast NVDA's split-adjusted closing share price (USD) at each horizon "
    "listed in the payload "
    "(`horizons`, business days ahead).\n\n"
    "Rules:\n"
    "  - Produce one forecast for each horizon in `horizons`.\n"
    "  - Use exactly the quantile levels from `standard_quantiles` — "
    "no additions, no omissions.\n"
    "  - `point_forecast` must exactly equal the 0.50 quantile value.\n"
    "  - Quantile values must be strictly non-decreasing as quantile levels increase.\n"
    "  - Document your reasoning in the `rationale` fields.\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + ContinuousAgentForecastOutput.prompt_schema_json()
)

TASK_SHOCK_SPEC = (
    f"Estimate P(shock) — the probability that NVDA's close {SHOCK_HORIZON} "
    f"trading day(s) after `as_of` differs from the `as_of` close by at least "
    f"{SHOCK_THRESHOLD:g}% in EITHER direction (|return| >= {SHOCK_THRESHOLD:g}%).\n\n"
    "Timing: the price history ends at `last_close_date`, normally the session "
    "BEFORE `as_of` (each close is published the next day). So you have not seen "
    "the `as_of` session's own close or return; `origin_price_usd` is the "
    "`last_close_date` close. Retrieved news may describe the `as_of` session — "
    "use it to judge whether that session was itself a large move. Do not assume "
    "it was quiet because the history does not show it.\n\n"
    "This is a two-sided magnitude question: a large drop counts exactly like a "
    "large rally. Report which way you lean in `direction_bias`, but the "
    "probability is for a move of either sign.\n\n"
    "Calibration anchors (NVDA, 2020-2024, share of sessions with a move this large;\n"
    "trailing vol = std of daily returns over the previous 21 sessions):\n"
    "  - Any session, unconditional                     -> ~12%\n"
    "  - Calm tape (trailing vol < 2.5%), no earnings    -> ~4%\n"
    "  - Normal tape (trailing vol 2.5-3.5%), no earnings -> ~12%\n"
    "  - Elevated vol (trailing vol > 3.5%), no earnings -> ~18%\n"
    f"  - Session right after a >={SHOCK_THRESHOLD:g}% move                -> ~22%\n"
    "  - Next session is the reaction to NVDA's own quarterly results "
    "(reported after the close) -> ~50%\n\n"
    "Start from the anchor that matches the price history and the calendar, then "
    "move away from it only for a specific, dated catalyst inside the horizon "
    "(e.g. an announced export-control ruling, a hyperscaler capex guidance "
    "change, a major competitor launch). Generic AI enthusiasm or a strong "
    "trend is not a catalyst. Probabilities above ~30% outside an earnings "
    "reaction need a named event. Upside shocks were more common than downside "
    "(87 up vs 64 down).\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + DiscreteAgentForecastOutput.prompt_schema_json()
)
"""Two-sided NVDA shock question with base-rate anchors.

The anchors come from :func:`ai_stocks_forecasting.shock_anchors.anchor_table`
over 2020-2024 at the committed ``SHOCK_THRESHOLD`` / ``SHOCK_HORIZON``; re-run
that module and update these numbers if either constant changes.
"""

TASK_SCENARIOS_SPEC = (
    "Identify the three scenarios that oil market analysts and experts are most "
    "actively debating for WTI crude over the next 60 days, given the current "
    "market context and price history.\n\n"
    "For each scenario:\n"
    "  - Give it a concise name (3-6 words)\n"
    "  - Describe it in 1-2 sentences\n"
    "  - Assign a probability (all three must sum to <= 1.0)\n"
    "  - Provide an expected WTI price range at the 60-day horizon as [low, high]\n"
    "  - Give your point estimate for WTI at 60 days under this scenario\n"
    "  - List 1-2 key drivers that would cause this scenario to materialise\n\n"
    "Also identify which scenario is the base case and provide an overall "
    "one-paragraph reasoning summary.\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + ScenarioAgentForecastOutput.prompt_schema_json()
)

TASK_SPECS: dict[TaskKind, str] = {
    "trajectory": TASK_TRAJECTORY_SPEC,
    "shock": TASK_SHOCK_SPEC,
    "scenario": TASK_SCENARIOS_SPEC,
}


TASK_OUTPUT_SCHEMAS: dict[TaskKind, type[AgentForecastOutput]] = {
    "trajectory": ContinuousAgentForecastOutput,
    "shock": DiscreteAgentForecastOutput,
    "scenario": ScenarioAgentForecastOutput,
}


def build_nvda_news_predictor(
    task: TaskKind,
    model: str = LITE_MODEL,
) -> AgentPredictor:
    """Build a news-grounded agent predictor for the given task kind.

    All three task kinds share the same multitask news identity
    (:func:`~ai_stocks_forecasting.analyst_agent.build_nvda_multitask_news_config`);
    only the user-payload ``task_spec`` and output schema change.

    Parameters
    ----------
    task : TaskKind
        One of ``"trajectory"``, ``"shock"``, or ``"scenario"``.
    model : str
        Model identifier passed through to the underlying
        :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`.
        Defaults to the lite model (``"gemini-3.1-flash-lite-preview"``); pass the
        advanced model (``"gemini-3.5-flash"``) when more capability is needed.
    """
    return AgentPredictor(
        agent_config=build_nvda_multitask_news_config(model=model),
        prompt_builder=NvdaMultitaskPromptBuilder(task_spec=TASK_SPECS[task]),
        output_schema=TASK_OUTPUT_SCHEMAS[task],
    )


def build_nvda_agent_predictor_for_task(config: AgentConfig, task: TaskKind) -> AgentPredictor:
    """Wire any NVDA agent config to a task-specific predictor.

    Uses the multitask prompt builder for every task kind so the ask rides in
    ``task_spec`` rather than in the system instruction.
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=NvdaMultitaskPromptBuilder(task_spec=TASK_SPECS[task]),
        output_schema=TASK_OUTPUT_SCHEMAS[task],
    )


SHOCK_TASK_ID = "nvda_shock_1d"


def nvda_shock_task() -> ForecastingTask:
    """Return the binary shock task the shock predictor answers: one horizon of ``SHOCK_HORIZON`` sessions.

    ``DiscreteAgentForecastOutput`` requires exactly one horizon and stamps
    ``forecast_date = as_of + SHOCK_HORIZON`` business days: the session whose
    move against the ``as_of`` close decides the outcome, the same event
    :func:`~ai_stocks_forecasting.signals.flag_shock_windows` labels.  That is
    pandas ``BDay``, not the NYSE calendar: an origin whose next business day
    is a market holiday has no session to resolve against, so choose shock
    origins with a trading day after them.
    """
    return ForecastingTask(
        task_id=SHOCK_TASK_ID,
        target_series_id=NVDA_SERIES_ID,
        horizons=[SHOCK_HORIZON],
        frequency="B",
        description=(
            f"P(|NVDA close-to-close return| >= {SHOCK_THRESHOLD:g}%) over the {SHOCK_HORIZON} session(s) after as_of"
        ),
    )


__all__ = [
    "SHOCK_TASK_ID",
    "TASK_SCENARIOS_SPEC",
    "TASK_SHOCK_SPEC",
    "TASK_SPECS",
    "TASK_TRAJECTORY_SPEC",
    "ScenarioAgentForecastOutput",
    "ScenarioCard",
    "TaskKind",
    "NvdaMultitaskPromptBuilder",
    "build_nvda_agent_predictor_for_task",
    "build_nvda_news_predictor",
    "nvda_shock_task",
]
