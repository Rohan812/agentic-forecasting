"""Fetch and cache daily price history for the AI-stock basket (NVDA, AMD, MSFT, GOOG).

Downloads each ticker's split- and dividend-adjusted daily close via yfinance into
``data/yfinance/<ticker>_adj_close_1d.parquet``, the cache
:func:`~ai_stocks_forecasting.basket.load_basket_prices` reads.

Usage
-----
    uv run python scripts/fetch_ai_basket.py

Idempotent: each run overwrites the cache with a fresh download.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_stocks_forecasting.basket import BASKET, HISTORY_START
from ai_stocks_forecasting.data import NVDA_HISTORY_START
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter


CACHE_DIR = Path("data/yfinance")


def main() -> None:
    """Fetch every basket ticker and print a one-line summary each."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in BASKET:
        # NVDA shares its cache file with the single-stock study, which needs the full history
        # (regime labels use every earlier session), so it is always fetched from its own start.
        start = NVDA_HISTORY_START if ticker == "NVDA" else HISTORY_START
        adapter = YFinanceDailyAdapter(ticker=ticker, start=start, cache_dir=CACHE_DIR, refresh=True)
        df = adapter.fetch()
        print(f"{ticker}: {len(df):,} sessions, {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")


if __name__ == "__main__":
    main()
