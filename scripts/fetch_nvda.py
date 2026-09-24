"""Fetch and cache NVIDIA (NVDA) daily price history from Yahoo Finance.

Downloads NVDA's split- and dividend-adjusted daily close via yfinance and
stores it as ``data/yfinance/nvda_adj_close_1d.parquet``.  The local parquet
cache is what :func:`~ai_stocks_forecasting.data.build_nvda_service` reads;
running this script once before a notebook session avoids live yfinance
requests during forecasting or backtesting.

The adjusted close matters for this ticker: NVDA split 10-for-1 in June 2024
and 4-for-1 in July 2021, so the raw ``Close`` series contains ~90% and ~75%
single-day "drops" that are pure bookkeeping.  A shock detector keyed on large
daily returns would rank those as the biggest events in the sample.

Usage
-----
    uv run python scripts/fetch_nvda.py

The script is idempotent and safe to re-run — it overwrites the cache with a
fresh download each time (``refresh=True``).
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_stocks_forecasting.paths import SHOCK_THRESHOLD
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter


CACHE_DIR = Path("data/yfinance")
TICKER = "NVDA"
HISTORY_START = "1999-01-01"  # NVDA listed 1999-01-22; ask for the full history.


def main() -> None:
    """Fetch NVDA history and print a brief summary."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    adapter = YFinanceDailyAdapter(ticker=TICKER, start=HISTORY_START, cache_dir=CACHE_DIR, refresh=True)
    print(f"Fetching {TICKER} (Adj Close) → {adapter.cache_path}")
    df = adapter.fetch()
    print(f"  {len(df):,} trading days  |  {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Latest close: ${df['value'].iloc[-1]:.2f}")

    # Shock counts are the headline property of this series for the discovery
    # loop, so surface them here rather than making everyone recompute them.
    returns = df.set_index("timestamp")["value"].pct_change()
    threshold = SHOCK_THRESHOLD / 100.0
    recent = returns.loc["2020-01-01":]
    up = int((recent >= threshold).sum())
    down = int((recent <= -threshold).sum())
    print(
        f"  Shocks since 2020 at ±{SHOCK_THRESHOLD:g}% in 1 day: "
        f"{up + down} ({up} up / {down} down, {100 * (up + down) / recent.notna().sum():.1f}% of days)"
    )
    print("Done.")


if __name__ == "__main__":
    main()
