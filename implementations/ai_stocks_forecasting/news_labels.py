"""Fenced news labels for discovery windows, under a hard spending cap.

A news candidate is a yes/no question ("Was a new US restriction on AI-chip
sales to China reported?").  :func:`label_windows` answers it for each window
from a web search fenced at that window's own cutoff, so a 2021 window only
ever sees news published by its ``as_of``.  The engine in
:mod:`~ai_stocks_forecasting.discovery` then scores the answers like any other
rule.

Two things keep the cost bounded:

- :class:`CostMeter` prices every LLM call's tokens as it returns (Langfuse's
  per-token prices for the two proxy models) and adds a conservative per-query
  charge for Google Search grounding, which those prices leave out.
- :class:`BudgetLedger` keeps the stage's cumulative spend in
  ``experiments/budget.yaml``.  Labelling stops before any window that would
  start past the cap, and every answer is cached, so a re-run never pays twice.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from ai_stocks_forecasting.discovery import EXPERIMENTS_DIR
from aieng.forecasting.methods.agentic.agent_factory import ContextRetrievalConfig, _build_search_tool
from aieng.forecasting.models import LITE_MODEL
from litellm.integrations.custom_logger import CustomLogger


PRICES_PER_TOKEN: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite-preview": (0.25e-6, 1.50e-6),
    "gemini-3.5-flash": (1.50e-6, 9.00e-6),
}
"""(input, output) USD per token, from the Langfuse model price table (2026-10-05)."""

SEARCH_FEE_USD = 0.014
"""Assumed Google Search grounding charge per search call.  The Langfuse prices for
these two models include none; 0.014 is the per-query rate Langfuse lists for newer
Gemini Flash models, used here so the meter errs high."""

STAGE_BUDGET_USD = 30.0
"""Hard cap on cumulative labelling spend for the Phase 2 news experiments."""

LOOKBACK_DAYS = 3
"""Calendar days of news before and including ``as_of`` that a question asks about."""

LABELS_DIR = EXPERIMENTS_DIR / "labels"
LEDGER_PATH = EXPERIMENTS_DIR / "budget.yaml"

_JUDGE_INSTRUCTION = """\
You label one news briefing for a statistical study. Answer the question using ONLY \
the briefing text. Answer "yes" only if the briefing clearly reports what the question \
asks about, within the stated dates. If it is absent, vague, outside the dates, or the \
briefing says the search failed, answer "no". Return JSON: \
{"answer": "yes" | "no", "evidence": "<one short quote or 'none'>"}"""


class CostMeter(CustomLogger):
    """Accumulate the cost of every successful LiteLLM call while registered."""

    def __init__(self) -> None:
        super().__init__()
        self.tokens_usd = 0.0
        self.searches = 0
        self.unpriced: set[str] = set()

    @property
    def usd(self) -> float:
        """Token cost plus the assumed grounding charge."""
        return self.tokens_usd + self.searches * SEARCH_FEE_USD

    def _record(self, kwargs: dict[str, Any], response: Any) -> None:
        model = str(kwargs.get("model", "")).removeprefix("openai/")
        usage = getattr(response, "usage", None)
        prices = PRICES_PER_TOKEN.get(model)
        if prices is None:
            self.unpriced.add(model)
            prices = max(PRICES_PER_TOKEN.values())  # unknown model: charge at the dearest rate
        if usage is not None:
            self.tokens_usd += (usage.prompt_tokens or 0) * prices[0] + (usage.completion_tokens or 0) * prices[1]
        tools = kwargs.get("tools") or (kwargs.get("optional_params") or {}).get("tools") or []
        if any("googleSearch" in t for t in tools if isinstance(t, dict)):
            self.searches += 1

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:  # noqa: ANN001, D102
        self._record(kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:  # noqa: ANN001, D102
        self._record(kwargs, response_obj)


@dataclass
class BudgetLedger:
    """The stage's cumulative spend, persisted so the cap holds across runs."""

    path: Path = LEDGER_PATH
    cap_usd: float = STAGE_BUDGET_USD
    entries: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path.exists():
            self.entries = yaml.safe_load(self.path.read_text()).get("entries", [])

    @property
    def spent_usd(self) -> float:
        """Total recorded spend."""
        return round(sum(e["usd"] for e in self.entries), 4)

    def remaining(self) -> float:
        """Budget left under the cap."""
        return self.cap_usd - self.spent_usd

    def record(self, what: str, usd: float, windows: int, searches: int) -> None:
        """Append one labelling run's measured cost and save."""
        self.entries.append({"what": what, "usd": round(usd, 4), "windows": windows, "searches": searches})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(
                {"cap_usd": self.cap_usd, "spent_usd": self.spent_usd, "entries": self.entries}, sort_keys=False
            )
        )


def _cache_path(question_id: str) -> Path:
    return LABELS_DIR / f"{question_id}.yaml"


def load_labels(question_id: str) -> dict[str, dict[str, str]]:
    """Return cached answers for a question, keyed by ISO ``as_of``."""
    path = _cache_path(question_id)
    return yaml.safe_load(path.read_text())["labels"] if path.exists() else {}


def _save_labels(question_id: str, question: str, labels: dict[str, dict[str, str]]) -> None:
    path = _cache_path(question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {"question_id": question_id, "question": question, "labels": dict(sorted(labels.items()))},
            sort_keys=False,
            allow_unicode=True,
        )
    )


async def _label_one(search: Any, question: str, as_of: pd.Timestamp) -> dict[str, str]:
    import litellm  # noqa: PLC0415

    start = (as_of - timedelta(days=LOOKBACK_DAYS - 1)).date()
    cutoff = (as_of + timedelta(days=1)).date().isoformat()  # search keeps news published strictly before this
    briefing = await search(
        query=f"{question} Consider only news published from {start} to {as_of.date()} inclusive.",
        cutoff_date=cutoff,
    )
    resp = await litellm.acompletion(
        model=f"openai/{LITE_MODEL}",
        api_base=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        messages=[
            {"role": "system", "content": _JUDGE_INSTRUCTION},
            {
                "role": "user",
                "content": f"Question: {question}\nDates: {start} to {as_of.date()}\n\nBriefing:\n{briefing[:6000]}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=200,
        timeout=60.0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```"))
        answer = "yes" if str(parsed.get("answer", "")).lower().startswith("y") else "no"
        evidence = str(parsed.get("evidence", ""))[:300]
    except json.JSONDecodeError:
        answer, evidence = "no", "unparseable judge output"
    failed = briefing.startswith("[SEARCH_VERIFICATION_FAILED]")
    return {"answer": answer, "evidence": evidence, "search": "failed" if failed else "ok"}


def label_windows(
    question_id: str,
    question: str,
    as_of_dates: list[pd.Timestamp],
    ledger: BudgetLedger,
    *,
    concurrency: int = 5,
    run_cap_usd: float | None = None,
) -> tuple[dict[str, dict[str, str]], float]:
    """Answer ``question`` for every date not yet cached, stopping before the budget is exceeded.

    Returns the full label map (cached plus new) and this run's measured cost.
    Windows are labelled in batches of ``concurrency``; before each batch the
    run checks that a batch at the most expensive per-window cost seen so far
    still fits under both the ledger's remaining budget and ``run_cap_usd``.
    """
    import litellm  # noqa: PLC0415

    labels = load_labels(question_id)
    todo = sorted(
        {pd.Timestamp(d).normalize() for d in as_of_dates if pd.Timestamp(d).date().isoformat() not in labels}
    )
    if not todo:
        return labels, 0.0
    config = ContextRetrievalConfig(enabled=True, verifier_max_attempts=2)
    search = _build_search_tool(
        config, openai_base_url=os.environ["OPENAI_BASE_URL"], openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    meter = CostMeter()
    litellm.callbacks = [*litellm.callbacks, meter]
    limit = min(ledger.remaining(), run_cap_usd if run_cap_usd is not None else float("inf"))
    done = 0
    try:
        done = asyncio.run(_label_all(search, question_id, question, todo, labels, meter, limit, concurrency))
    finally:
        litellm.callbacks = [c for c in litellm.callbacks if c is not meter]
        ledger.record(question_id, meter.usd, done, meter.searches)
    if meter.unpriced:
        print(f"  note: unpriced models charged at the dearest rate: {sorted(meter.unpriced)}")
    return labels, meter.usd


async def _label_all(
    search: Any,
    question_id: str,
    question: str,
    todo: list[pd.Timestamp],
    labels: dict[str, dict[str, str]],
    meter: CostMeter,
    limit: float,
    concurrency: int,
) -> int:
    """Label ``todo`` in batches inside one event loop, checking the budget before each batch."""
    worst_per_window = 0.0
    done = 0

    async def one(day: pd.Timestamp) -> dict[str, str]:
        try:
            return await _label_one(search, question, day)
        except Exception as exc:  # noqa: BLE001 — one failed window must not lose the batch
            return {"answer": "no", "evidence": f"error: {type(exc).__name__}", "search": "error"}

    for i in range(0, len(todo), concurrency):
        batch = todo[i : i + concurrency]
        if meter.usd + worst_per_window * len(batch) > limit:
            print(
                f"  budget stop: ${meter.usd:.2f} spent this run, limit ${limit:.2f}; {len(todo) - done} windows unlabelled"
            )
            break
        before = meter.usd
        results = await asyncio.gather(*(one(d) for d in batch))
        for day, result in zip(batch, results, strict=True):
            labels[day.date().isoformat()] = result
        done += len(batch)
        worst_per_window = max(worst_per_window, (meter.usd - before) / len(batch))
        _save_labels(question_id, question, labels)
    return done
