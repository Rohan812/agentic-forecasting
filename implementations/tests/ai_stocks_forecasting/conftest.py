"""Shared fixtures for the NVDA implementation tests."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from ai_stocks_forecasting.data import NVDA_SERIES_ID
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.store import SeriesStore


@pytest.fixture
def nvda_like_context() -> Callable[[str], ForecastContext]:
    """Build a context over a split-adjusted-looking NVDA series: cents in 1999, ~$130 in 2025.

    Closes are released one business day after the session, as
    ``YFinanceDailyAdapter`` does, so the context's last close precedes ``as_of``.
    """
    dates = pd.bdate_range("1999-01-22", "2025-03-07")
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "value": np.geomspace(0.04, 130.0, len(dates)),
            "released_at": dates + pd.offsets.BDay(1),
        }
    )
    store = SeriesStore()
    store.put(
        NVDA_SERIES_ID,
        frame,
        SeriesMetadata(
            series_id=NVDA_SERIES_ID, description="synthetic", source="test", units="USD/share", frequency="B"
        ),
    )
    return lambda as_of: ForecastContext(store, as_of=datetime.fromisoformat(as_of))
