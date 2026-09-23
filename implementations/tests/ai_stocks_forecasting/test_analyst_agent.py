"""Wiring contracts for the NVDA news-grounded analyst config.

Both run offline: building the ADK agent and the predictor makes no LLM call.
They pin the two properties a refactor could break silently. One is the
registry identity, because the agent's name becomes the predictor id and so
the prediction filename. The other is the leakage fence around web search.
"""

from __future__ import annotations

import pytest
from ai_stocks_forecasting.analyst_agent import build_nvda_news_config, build_wti_agent_predictor
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
