"""Wiring contracts for the NVDA shock predictor and its outcome series.  Offline: no LLM calls."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import pandas as pd
import pytest
from ai_stocks_forecasting.analyst_agent.agent import _NVDA_MULTITASK_ANALYST_INSTRUCTION
from ai_stocks_forecasting.data import NVDA_SERIES_ID
from ai_stocks_forecasting.tasks import (
    NVDA_SHOCK_SERIES_ID,
    TASK_SHOCK_SPEC,
    NvdaMultitaskPromptBuilder,
    build_nvda_news_predictor,
    nvda_shock_task,
    register_shock_series,
    shock_indicator_frame,
)
from aieng.forecasting.data import DataService
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.data.models import SeriesMetadata
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


def _prices(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2025-03-03", periods=len(closes))
    return pd.DataFrame({"timestamp": dates, "value": closes, "released_at": dates + pd.offsets.BDay(1)})


def test_shock_indicator_matches_the_signals_definition_at_the_boundary() -> None:
    """Exactly ±5% is a shock in both directions (``>=``, simple returns), 4.9% is not, and day one is unlabelled."""
    frame = shock_indicator_frame(_prices([100.0, 105.0, 99.75, 104.6376, 100.0]))
    # 100->105 = +5.00%, 105->99.75 = -5.00%, 99.75->104.6376 = +4.90%, 104.6376->100 = -4.43%
    assert frame["value"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2025-03-04")


def test_shock_outcome_is_not_visible_on_its_own_session() -> None:
    """At ``as_of`` = the shock session, its outcome is hidden; it appears the next business day with the close."""
    service = DataService()
    service.register(
        NVDA_SERIES_ID,
        StaticFrameAdapter(_prices([100.0, 110.0, 110.0])),
        SeriesMetadata(series_id=NVDA_SERIES_ID, description="t", source="t", units="USD/share", frequency="B"),
    )
    register_shock_series(service)
    shock_day = pd.Timestamp("2025-03-04")

    seen_on_the_day = service.context(as_of=shock_day).get_series(NVDA_SHOCK_SERIES_ID)
    assert shock_day not in set(pd.to_datetime(seen_on_the_day["timestamp"]))

    seen_next_day = service.context(as_of=pd.Timestamp("2025-03-05")).get_series(NVDA_SHOCK_SERIES_ID)
    assert seen_next_day.set_index("timestamp").loc[shock_day, "value"] == 1.0


def test_shock_spec_names_fenced_search_topics() -> None:
    """The shock ask must name its search topics, each fenced at ``as_of``.

    The multitask instruction only says to search.  Without topics in the spec,
    the first smoke backtest ran one search per origin and the agent never left
    the volatility anchor.
    """
    queries = re.findall(r"search_web\(query=\"[^\"]+\", cutoff_date=<as_of>\)", TASK_SHOCK_SPEC)
    assert len(queries) == 3, queries
    assert "[SEARCH_VERIFICATION_FAILED]" in TASK_SHOCK_SPEC
