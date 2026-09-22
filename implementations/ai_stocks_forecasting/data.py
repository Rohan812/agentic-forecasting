"""Data-service setup for the NVDA (AI stocks) forecasting experiment.

:func:`build_nvda_service` registers NVIDIA's daily split- and dividend-adjusted
close (Yahoo Finance ticker ``NVDA``) under the canonical
:data:`NVDA_SERIES_ID`.  Both the reference YAML specs under
``implementations/ai_stocks_forecasting/specs/`` and the notebooks here
reference the same ``series_id`` via this module.

**Why the adjusted close, not the raw close.**  NVDA has split repeatedly — most
recently 10-for-1 in June 2024 and 4-for-1 in July 2021.  The raw ``Close``
series therefore contains ~90% and ~75% single-day "drops" that are pure
bookkeeping, and a shock detector keyed on large returns would flag them as the
biggest events in the sample.  ``Adj Close`` is back-adjusted for splits and
dividends, so its returns are the returns an investor actually experienced.  The
tradeoff is that back-adjustment rewrites history when a new dividend accrues,
which shifts price *levels* slightly between fetches; returns are unaffected,
which is what the shock definition and the statistical gate operate on.

This module registers the **target series only**.  Unlike the energy/oil parent
implementation, there is no covariate panel here: the NVDA forecaster's
non-price signal comes from news, and a numerical covariate panel (VIX, peer
semiconductor returns, hyperscaler equities) belongs to the later fleet phase
where covariate-bearing predictors are introduced.  Add it as a separate
``build_nvda_multivariate_service`` at that point rather than widening this
function, so the single-series path stays the simple one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter


def naive_utc_now() -> datetime:
    """Return current UTC time as a timezone-naive :class:`datetime`.

    :class:`~aieng.forecasting.data.service.DataService` and
    :class:`~aieng.forecasting.data.cutoff.CutoffEnforcer` require naive
    ``as_of`` values — tz-aware timestamps raise on comparison with cached
    series timestamps.
    """
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


NVDA_SERIES_ID = "nvda_stock_price"
"""Canonical series ID for the NVDA daily adjusted close price."""

NVDA_TICKER = "NVDA"
"""Yahoo Finance symbol for NVIDIA Corporation common stock."""

DEFAULT_CACHE_DIR = Path("data/yfinance")
"""Default yfinance parquet cache directory (resolved relative to CWD at call time)."""

NVDA_HISTORY_START = "1999-01-01"
"""Earliest date requested from yfinance.

NVIDIA listed on 1999-01-22, so this asks for the full available history.
Setting an explicit start matters: without it yfinance returns only a 30-day
window, which is not enough history for the shock base rates the discovery loop
compares candidate patterns against.
"""


def build_nvda_service(cache_dir: Path | None = None) -> DataService:
    """Return a :class:`DataService` with the NVDA daily adjusted close registered.

    Parameters
    ----------
    cache_dir : Path or None
        yfinance parquet cache directory.  Defaults to ``data/yfinance``
        relative to the current working directory, which is where
        ``scripts/fetch_nvda.py`` writes.  Notebooks typically run from their
        own directory, so the adapter will transparently fetch from yfinance if
        the cache is absent or stale, then persist the result for later runs.

    Returns
    -------
    DataService
        A data service with the NVDA series registered, ready to be handed
        to :func:`~aieng.forecasting.evaluation.backtest.backtest` /
        :func:`~aieng.forecasting.evaluation.backtest.cached_multi_backtest` /
        :func:`~aieng.forecasting.evaluation.eval.evaluate`.
    """
    resolved_cache_dir: Path = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    svc = DataService()
    svc.register(
        NVDA_SERIES_ID,
        # field defaults to "Adj Close" — matches the cache key nvda_adj_close_1d.parquet
        # produced by scripts/fetch_nvda.py, and is split/dividend adjusted (see module
        # docstring for why that matters for NVDA specifically).
        YFinanceDailyAdapter(ticker=NVDA_TICKER, start=NVDA_HISTORY_START, cache_dir=resolved_cache_dir),
        SeriesMetadata(
            series_id=NVDA_SERIES_ID,
            description="NVIDIA Corporation daily split- and dividend-adjusted close (Yahoo Finance NVDA)",
            source="yfinance",
            units="USD/share",
            frequency="B",
        ),
    )
    return svc


__all__ = [
    "DEFAULT_CACHE_DIR",
    "NVDA_HISTORY_START",
    "NVDA_SERIES_ID",
    "NVDA_TICKER",
    "build_nvda_service",
    "naive_utc_now",
]
