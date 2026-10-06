"""The labelling budget must stop spending before the cap, not after it."""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from ai_stocks_forecasting import news_labels
from ai_stocks_forecasting.news_labels import CostMeter, _label_all


def test_labelling_stops_before_a_batch_that_would_cross_the_limit(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    """Each window costs $0.10; with a $0.35 limit and batches of 2, only the first batch may run.

    After it, $0.20 is spent and the next batch would reach $0.40, so the run
    must stop with 2 windows labelled rather than overshoot to $0.40.
    """
    meter = CostMeter()

    async def fake_label(search, question, day):  # noqa: ANN001, ANN202
        meter.tokens_usd += 0.10
        return {"answer": "no", "evidence": "none", "search": "ok"}

    monkeypatch.setattr(news_labels, "_label_one", fake_label)
    monkeypatch.setattr(news_labels, "LABELS_DIR", tmp_path)
    days = list(pd.bdate_range("2021-01-04", periods=10))
    labels: dict[str, dict[str, str]] = {}

    done = asyncio.run(_label_all(None, "q", "question", days, labels, meter, limit=0.35, concurrency=2))

    assert done == 2
    assert meter.usd == pytest.approx(0.20)
    assert len(labels) == 2
