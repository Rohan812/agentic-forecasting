"""Wiring contracts for the NVDA news-grounded analyst config.

Both run offline: building the ADK agent and the predictor makes no LLM call.
They pin the properties a refactor could break silently: the registry
identity, because the agent's name becomes the predictor id and so the
prediction filename; the leakage fence around web search; and the prompt
payload, which must carry what the instruction promises and nothing unbounded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import pytest
from ai_stocks_forecasting.analyst_agent import (
    NvdaPriceForecastPromptBuilder,
    build_nvda_news_config,
    build_wti_agent_predictor,
)
from ai_stocks_forecasting.analyst_agent.agent import _NVDA_ANALYST_INSTRUCTION
from ai_stocks_forecasting.data import NVDA_SERIES_ID
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import build_adk_agent


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build against bare model strings so the ids below don't depend on the environment."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_predictions_are_filed_under_an_nvda_name() -> None:
    """The predictor id, which is the registry filename key, must not say WTI."""
    predictor_id = build_wti_agent_predictor(build_nvda_news_config()).predictor_id
    assert predictor_id == "agent_predictor_nvda_analyst_news_gemini-3.1-flash-lite-preview_continuous"


def test_web_search_is_fenced_and_independently_verified() -> None:
    """The agent gets ``search_web``, audited by a verifier on a different model than the search.

    The instruction must also pin ``cutoff_date`` to ``as_of`` and forbid
    filling a failed verification from memory.  The cutoff probe in
    ``LLM_CUTOFFS.md`` found both models state wrong pre-cutoff outcomes with
    confidence, which is why the second rule matters.
    """
    config = build_nvda_news_config()
    retrieval = config.context_retrieval
    assert retrieval.enabled
    assert retrieval.verifier_model != retrieval.search_model, "Verifier must not share the search model's blind spots."

    agent = build_adk_agent(config)
    tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in agent.tools]
    assert "search_web" in tool_names

    assert "The ``cutoff_date`` MUST always equal ``as_of``" in config.instruction
    assert "[SEARCH_VERIFICATION_FAILED]" in config.instruction
    assert "Do not use your own background knowledge to fill the gap" in config.instruction


def test_payload_delivers_the_keys_the_instruction_promises(
    nvda_like_context: Callable[[str], ForecastContext],
) -> None:
    """Every backticked payload key in the forecasting contract must exist in the payload.

    The instruction and the builder live in different places; a renamed key
    leaves the agent reasoning about a field it never receives, with no error.
    """
    task = ForecastingTask(
        task_id="nvda_price_forecast",
        target_series_id=NVDA_SERIES_ID,
        horizons=[5, 10, 21],
        frequency="B",
        description="NVDA close",
    )
    payload = json.loads(NvdaPriceForecastPromptBuilder()(task=task, context=nvda_like_context("2025-03-03")))

    contract = _NVDA_ANALYST_INSTRUCTION.split("## Forecasting contract")[1].split("Rules:")[0]
    promised = set(re.findall(r"^- `(\w+)`", contract, flags=re.MULTILINE))
    assert promised == set(payload), f"instruction promises {sorted(promised)}, payload has {sorted(payload)}"


def test_payload_history_is_bounded_not_back_to_1999(nvda_like_context: Callable[[str], ForecastContext]) -> None:
    """Weekly history stops five years before the daily window; 1999 penny prices never reach the prompt."""
    task = ForecastingTask(
        task_id="nvda_price_forecast", target_series_id=NVDA_SERIES_ID, horizons=[5], frequency="B", description="x"
    )
    payload = json.loads(NvdaPriceForecastPromptBuilder()(task=task, context=nvda_like_context("2025-03-03")))
    first_date = payload["target_history_csv"].splitlines()[1].split(",")[0]
    assert first_date >= "2019-08-01", first_date
