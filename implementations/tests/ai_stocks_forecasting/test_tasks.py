"""Wiring contracts for the NVDA shock predictor.  Offline: no LLM calls."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import pytest
from ai_stocks_forecasting.analyst_agent.agent import _NVDA_MULTITASK_ANALYST_INSTRUCTION
from ai_stocks_forecasting.tasks import (
    TASK_SHOCK_SPEC,
    NvdaMultitaskPromptBuilder,
    build_nvda_news_predictor,
    nvda_shock_task,
)
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.methods.agentic import DiscreteAgentForecastOutput


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build against bare model strings so the ids below don't depend on the environment."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_shock_predictor_is_nvda_and_binary() -> None:
    """Shock forecasts are filed under an NVDA name and parsed as a binary probability."""
    predictor = build_nvda_news_predictor("shock")
    assert predictor.predictor_id == "agent_predictor_nvda_analyst_multitask_gemini-3.1-flash-lite-preview_discrete"
    assert predictor.output_schema is DiscreteAgentForecastOutput


def test_shock_payload_states_the_one_session_lag(nvda_like_context: Callable[[str], ForecastContext]) -> None:
    """On a Monday origin the history ends Friday; the payload must say so and carry what the instruction lists.

    The shock is the session after ``as_of`` against the ``as_of`` close, which the
    agent has not seen.  Without ``last_close_date`` it would read Friday's close
    as the origin and could not know the ``as_of`` session is missing.
    """
    shock_task = nvda_shock_task()
    payload = json.loads(
        NvdaMultitaskPromptBuilder(task_spec=TASK_SHOCK_SPEC)(task=shock_task, context=nvda_like_context("2025-03-03"))
    )
    assert payload["as_of"] == "2025-03-03"
    assert payload["last_close_date"] == "2025-02-28"
    assert "`last_close_date`" in payload["task_spec"]

    promised = set(re.findall(r"^- `(\w+)`", _NVDA_MULTITASK_ANALYST_INSTRUCTION, flags=re.MULTILINE))
    assert promised == set(payload), f"instruction lists {sorted(promised)}, payload has {sorted(payload)}"
