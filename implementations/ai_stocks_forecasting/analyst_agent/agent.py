"""NVDA equity analyst agent configurations and prompt builder.

The instruction strings are NVDA-targeted (semiconductor cycle, hyperscaler AI
capex, export controls, supply chain).  Three
:class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`
factories define the capability levels used here:

1. :func:`build_nvda_basic_config`: the LLM reasons from price history alone (no tools).
2. :func:`build_nvda_news_config`: adds bounded Google Search via a
   :class:`~aieng.forecasting.methods.agentic.agent_factory.ContextRetrievalConfig`
   sub-agent with strict temporal cutoffs and an independent leakage verifier.
3. :func:`build_nvda_multitask_news_config`: the same news grounding with a
   task-agnostic instruction, so one identity answers the trajectory and shock tasks.

Also provides:

- :class:`NvdaPriceForecastPromptBuilder`: Pydantic ``BaseModel`` that serialises
  the task and history into a structured JSON payload for the agent.
- :func:`build_nvda_agent_predictor`: convenience factory that wires a config to
  an :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`.

The energy/oil parent also has code-execution and forecast-tool capability
levels; they were not carried over, because no NVDA experiment uses them.

Module-level ``__getattr__`` exposes ``root_agent`` lazily so ``adk web`` can
load this module for interactive (schema-free) use without importing the full
predictor stack.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# System prompt (root analyst agent)
# ---------------------------------------------------------------------------

_NVDA_MULTITASK_ANALYST_INSTRUCTION = """\
## Role

You are an expert equity analyst covering NVIDIA (NASDAQ: NVDA) and the AI \
semiconductor complex.

## Input

You will receive a JSON payload containing:
- `task`: the task identifier
- `task_spec`: the exact question and required JSON output schema
- `as_of`: the forecast origin date (temporal cutoff)
- `horizons`: integer horizon steps (business days ahead)
- `standard_quantiles`: quantile levels for continuous forecasts (when applicable)
- `origin_price_usd`: NVDA split-adjusted close on `last_close_date`, USD per share
- `last_close_date`: date of the latest close in the history — normally the \
session before `as_of`, because each close is published the next day
- `target_history_csv`: compressed NVDA daily close history (split-adjusted)

When context retrieval is enabled, call ``search_web`` BEFORE answering.

## Output contract

Read the data (and briefing, if retrieved) carefully, then execute the task \
in `task_spec` precisely.

If a `set_model_response` tool is available, call it with your complete JSON \
as `json_response` — the exact schema is described in `task_spec`. Otherwise \
return the JSON directly as plain text with no preamble.\
"""


def _build_nvda_analyst_instruction() -> str:
    """Build the tool-free NVDA analyst instruction (price history only).

    Tool-specific guidance (web search, code exec, forecast tool) is appended
    by the config factories that enable those capabilities — never reference a
    tool here that ``build_nvda_basic_config`` does not attach.

    Using a function instead of a static string ensures the ``## Output schema``
    block is always in sync with ``ContinuousAgentForecastOutput`` —
    no manual JSON to maintain.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        "You are an expert equity analyst covering NVIDIA (NVDA) and the AI "
        "semiconductor complex. You produce calibrated probabilistic forecasts of "
        "NVDA's split-adjusted share price, grounded in the AI-accelerator demand "
        "cycle, hyperscaler capital expenditure, supply-chain capacity, export-control "
        "policy, and NVDA's own historical return dynamics.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last close price (USD/share), 52-week range, and observation count\n"
        "- `target_history_csv`: NVDA split-adjusted daily close history (recent 6 months daily, "
        "older history as weekly averages)\n\n"
        "Rules:\n"
        "1. Produce one forecast for each horizon listed in `horizons`.\n"
        "2. Use exactly the quantile levels from `standard_quantiles` — no additions, no omissions.\n"
        "3. `point_forecast` must exactly equal the 0.50 quantile value.\n"
        "4. Quantile values must be strictly non-decreasing as quantile levels increase.\n"
        "5. Document your reasoning in the `rationale` fields.\n\n"
        "## Output schema\n\n"
        "If a `set_model_response` tool is available, call it with a "
        "`json_response` string matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        "Otherwise return that JSON directly as plain text with no preamble.\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_days"`). '
        '`"quantiles"` is a **list** of `{"quantile": <level>, "value": <price>}` '
        "objects — not a dict. Omit any field not shown above.\n\n"
        "## Analysis discipline\n\n"
        "Document your key assumptions (data-center demand and hyperscaler capex "
        "guidance, product-cycle timing, TSMC/CoWoS supply, export-control exposure, "
        "earnings-date proximity, and the broad risk-on/risk-off tone of the market) "
        "in the `rationale` fields of your forecast output. NVDA routinely moves 3-4% "
        "in a day and 7%+ around earnings, so do not produce intervals narrower than "
        "its realised volatility justifies. "
        "Reason only from the payload (and any tools listed in later sections of "
        "this instruction — if none are listed, you have no tools)."
    )


_NVDA_ANALYST_INSTRUCTION = _build_nvda_analyst_instruction()

# Appended only by configs that enable ContextRetrievalConfig (news / code / tool).
_CONTEXT_RETRIEVAL_SUPPLEMENT = """

## Context retrieval

Call ``search_web`` to gather market intelligence BEFORE producing forecasts.

Call ``search_web`` with ``query`` and ``cutoff_date`` (set to the ``as_of`` \
date from the payload). The ``cutoff_date`` MUST always equal ``as_of`` — \
this is the temporal fence that prevents post-origin information from \
contaminating historical backtests.

If ``search_web`` returns a result beginning with \
``[SEARCH_VERIFICATION_FAILED]``, treat it as no verified news context for \
that query. Do not use your own background knowledge to fill the gap or \
speculate about what the news might have said — proceed with price-history \
and other available signals only, and note the gap in your rationale.

Recommended queries (call ``search_web`` once per topic):
- ``search_web(query="NVIDIA stock news, analyst revisions and next earnings date", cutoff_date=<as_of>)``
- ``search_web(query="hyperscaler AI capex guidance and TSMC CoWoS AI accelerator supply", cutoff_date=<as_of>)``
- ``search_web(query="US AI chip export controls China and AMD or custom-silicon competition with NVIDIA", cutoff_date=<as_of>)``

Each ``search_web`` call runs a search plus an independent leakage-verifier \
call, so keep to these three unless a result points at a specific event \
(e.g. an earnings date inside the forecast horizon) that needs one follow-up.
"""

# ---------------------------------------------------------------------------
# Context retrieval instruction (sub-agent)
# ---------------------------------------------------------------------------

_NVDA_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are a semiconductor and AI-infrastructure equity intelligence specialist \
with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering relevant aspects of:
- NVDA share price level, recent trend, and any unusually large recent moves
- NVDA earnings: the next report date, guidance, and data-center revenue trend
- Hyperscaler AI capital expenditure (Microsoft, Alphabet, Amazon, Meta, \
Oracle) and signs of acceleration or pull-back
- US export controls and trade policy affecting AI accelerators, especially \
sales to China
- Supply chain: TSMC advanced-node and CoWoS packaging capacity, HBM supply, \
product-ramp timing (e.g. new GPU generations)
- Competitive launches from AMD, Intel, and hyperscaler custom silicon
- Published analyst forecasts or unusual price-target revisions

Ground your summary in the search results you actually retrieve. \
When a cutoff date is specified, do not report or speculate about events \
that occurred after that date.

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, \
never from a source's claimed publish date or byline timestamp — those are \
frequently stale or updated after original publication; (2) discard \
anything you cannot confidently place before the cutoff date; (3) only then \
write your summary. Do not supplement the search results with your own \
background/training knowledge — if the results are insufficient, say so \
explicitly rather than filling gaps from memory.\
"""

# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------


WEEKLY_HISTORY_YEARS = 5
"""Years of weekly averages kept before the six-month daily window.

NVDA's split-adjusted history starts in 1999 at about $0.04 a share. Unbounded,
the weekly section was ~1,300 rows, 900 of them printing as ``0.0x`` at two
decimals: about two thirds of a ~26K-character payload carrying no information,
paid for on every agent turn.  Five years still spans a full semiconductor
cycle (the 2022 drawdown and the 2023-24 AI rally).
"""


def compress_history(df: pd.DataFrame, weekly_years: int | None = WEEKLY_HISTORY_YEARS) -> str:
    """Compress NVDA daily history to stay within context limits.

    Returns daily bars for the most recent 6 months and weekly averages for
    the ``weekly_years`` before that.  The CSV header is ``date,close``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``timestamp`` and ``value``.
    weekly_years : int or None, default=WEEKLY_HISTORY_YEARS
        How many years of weekly averages to keep before the daily window.
        ``None`` keeps the full history.

    Returns
    -------
    str
        CSV string with header ``date,close``.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.DateOffset(months=6)

    recent = df[df["timestamp"] >= cutoff].copy()
    old = df[df["timestamp"] < cutoff].copy()
    if weekly_years is not None:
        old = old[old["timestamp"] >= cutoff - pd.DateOffset(years=weekly_years)]

    rows: list[str] = ["date,close"]

    if not old.empty:
        old_indexed = old.set_index("timestamp")["value"]
        weekly: pd.Series = old_indexed.resample("W").mean().dropna()
        for date, val in weekly.items():
            rows.append(f"{date.date()},{val:.2f}")

    for _, row in recent.iterrows():
        rows.append(f"{row['timestamp'].date()},{row['value']:.2f}")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class NvdaPriceForecastPromptBuilder(BaseModel):
    """Prompt builder for NVDA share-price forecasting tasks.

    Produces a structured JSON payload for the analyst agent containing the
    task specification, compressed price history, and a data summary.
    The payload includes ``standard_quantiles`` explicitly so the agent knows
    the exact grid it must produce.  Its keys are the ones the analyst
    instruction's forecasting contract lists; a test pins the two together.

    ``last_date`` in ``target_summary`` can precede ``as_of``: the adapter
    releases a session's close the next day, so an origin sees the prior
    session's close as its latest point.

    Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol (structural typing — no explicit inheritance required).
    """

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Serialise the task and context into a JSON string for the agent.

        Parameters
        ----------
        task : ForecastingTask
            The forecasting task — supplies ``task_id``, ``horizons``.
        context : ForecastContext
            The information state at forecast time.

        Returns
        -------
        str
            JSON-serialised payload with task metadata, compressed history, and
            the standard quantile grid the agent must populate.
        """
        df = context.get_series(task.target_series_id)
        compressed = compress_history(df)

        last_row = df.iloc[-1]
        last_close = float(last_row["value"])
        last_date = str(pd.Timestamp(last_row["timestamp"]).date())
        trailing_252 = df["value"].tail(252)

        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_close_usd": last_close,
                "last_date": last_date,
                "n_trading_days": int(len(df)),
                "52w_high": float(trailing_252.max()),
                "52w_low": float(trailing_252.min()),
            },
            "target_history_csv": compressed,
        }

        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# AgentConfig factories
# ---------------------------------------------------------------------------


def build_nvda_basic_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build an :class:`AgentConfig` with no tools.

    The agent reasons purely from the price history in the prompt payload.
    Useful as a low-cost baseline or starting point when comparing capability
    levels.

    Parameters
    ----------
    model : str
        Gemini model identifier.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="nvda_analyst_basic",
        model=model,
        instruction=_NVDA_ANALYST_INSTRUCTION,
    )


def build_nvda_multitask_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """News-grounded config for the one-agent-three-tasks demo (NB3).

    Uses a task-agnostic analyst instruction; the task schema is supplied in
    the user prompt payload via :class:`~ai_stocks_forecasting.tasks.NvdaMultitaskPromptBuilder`.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.
    """
    return AgentConfig(
        name="nvda_analyst_multitask",
        model=model,
        instruction=_NVDA_MULTITASK_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_NVDA_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_nvda_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build the news-grounded NVDA analyst: price history plus cutoff-fenced web search.

    The analyst reads the JSON payload from the prompt builder and, before
    forecasting, calls ``search_web`` on three NVDA topics: the stock and its
    next earnings date, hyperscaler capex and TSMC/CoWoS supply, and export
    controls and competition.  Each call goes to a
    :class:`~aieng.forecasting.methods.agentic.agent_factory.ContextRetrievalConfig`
    sub-agent that searches only up to ``cutoff_date`` (the origin's ``as_of``)
    and then passes the result through an **independent leakage verifier**.
    The verifier rejects anything it cannot confidently place before the
    cutoff, retrying up to ``verifier_max_attempts`` times.  When every attempt
    fails, the analyst receives ``[SEARCH_VERIFICATION_FAILED]``, and its
    instruction tells it to proceed on price history alone rather than fill the
    gap from memory.  Live origins (``as_of`` of today or later) skip the fence.

    **The agent's name is part of the result's identity.**
    :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor` builds
    its ``predictor_id`` as ``agent_predictor_<name>_<model>_<modality>``, and
    that id is the filename key in the prediction registry and the label on the
    charts.  The WTI factory this replaces named its agent ``wti_analyst_news``,
    which would have filed NVDA forecasts under a WTI name.

    **Cost.** Each ``search_web`` call is a search plus at least one verifier
    call on ``verifier_model``, so an origin costs roughly three searches and
    three to nine verifier calls on top of the analyst's own turn.  That is why
    the instruction caps the recommended queries at three.

    **Where results can be scored.** Both proxy models remember NVDA prices
    through about January 2025 (see ``LLM_CUTOFFS.md``).  Scores on origins
    before February 2025 measure recall, not forecasting.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to the
        lite model independently of ``model``, so Gemini handles Google Search
        even when the analyst runs on a different model.
    verifier_model : str
        Model for the independent temporal-leakage verifier. Defaults to the
        advanced model, so it doesn't share ``search_model``'s blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before returning the
        ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="nvda_analyst_news",
        model=model,
        instruction=_NVDA_ANALYST_INSTRUCTION + _CONTEXT_RETRIEVAL_SUPPLEMENT,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_NVDA_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_nvda_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap an :class:`AgentConfig` in an :class:`AgentPredictor`.

    Uses :class:`NvdaPriceForecastPromptBuilder` and
    :class:`~aieng.forecasting.methods.agentic.outputs.ContinuousAgentForecastOutput`
    as the output schema.

    Parameters
    ----------
    config : AgentConfig
        Any of the configs produced by :func:`build_nvda_basic_config`,
        :func:`build_nvda_news_config` or :func:`build_nvda_multitask_news_config`.

    Returns
    -------
    AgentPredictor
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=NvdaPriceForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(
            build_nvda_multitask_news_config(
                model=ADVANCED_MODEL, search_model=ADVANCED_MODEL, verifier_model=ADVANCED_MODEL
            )
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
