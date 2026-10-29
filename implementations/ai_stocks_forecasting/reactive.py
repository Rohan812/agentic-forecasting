"""Reactive next-day ranges: test whether reading the news sets a better 80% range for tomorrow's close.

Pre-registered in ``experiments/reactive/preregistration.md``.  Each forecast is
made at 00:00 on the target session's date: the agent sees closes through the
previous session, and fenced search admits news published before the target
date (the whole previous day, after-close earnings included).  The forecast is
the target session's close as 11 quantiles; the 80% range is the 10%-90% pair.

Six arms, three without an LLM (``vol21``, ``ewma``, ``vol21_earnings``) and three
agent arms sharing one instruction (``agent_prices``, ``agent_calendar``,
``agent_news``).  Agent forecasts are cached one JSON line per session under
``data/predictions/reactive/``, so an interrupted run resumes where it stopped.

Usage, from ``implementations/``::

    uv run python -m ai_stocks_forecasting.reactive run --ticker AMD --arm agent_news
    uv run python -m ai_stocks_forecasting.reactive run --ticker AMD --arm agent_news --dev   # 2025 wiring check
    uv run python -m ai_stocks_forecasting.reactive score
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import litellm
import numpy as np
import pandas as pd
import properscoring
import yaml
from ai_stocks_forecasting.analyst_agent import compress_history
from ai_stocks_forecasting.news_labels import SEARCH_FEE_USD, BudgetLedger, CostMeter
from ai_stocks_forecasting.paths import DATA_DIR
from ai_stocks_forecasting.tasks import MARKET_CLOSE
from aieng.forecasting.data import DataService
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentPredictor, ContinuousAgentForecastOutput
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig, ContextRetrievalConfig
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from pydantic import BaseModel


TICKERS: tuple[str, ...] = ("AMD", "MSFT", "GOOG")
COMPANY = {"AMD": "Advanced Micro Devices", "MSFT": "Microsoft", "GOOG": "Alphabet (Google)"}
HISTORY_START = "2018-06-01"
TEST_START, TEST_END = "2026-01-02", "2026-09-30"
CALIBRATION_START, CALIBRATION_END = "2020-01-01", "2025-12-31"
DEV_SESSIONS: tuple[str, ...] = (
    "2025-08-04",
    "2025-08-05",
    "2025-08-06",
    "2025-08-07",
    "2025-11-04",
    "2025-11-05",
    "2025-11-06",
    "2025-12-01",
)
"""AMD 2025 wiring-check sessions; 2025-08-06 and 2025-11-05 are earnings reactions."""

VOL_WINDOW = 21
EWMA_DECAY = 0.94
TARGET_RISK_PCT = 1.0
MAX_POSITION = 1.5
Z80 = NormalDist().inv_cdf(0.9) * 2  # width of a central 80% normal interval, in sds (2.563)

Arm = Literal["vol21", "ewma", "vol21_earnings", "agent_prices", "agent_calendar", "agent_news"]
BASELINE_ARMS: tuple[Arm, ...] = ("vol21", "ewma", "vol21_earnings")
AGENT_ARMS: tuple[Arm, ...] = ("agent_prices", "agent_calendar", "agent_news")
ARMS: tuple[Arm, ...] = BASELINE_ARMS + AGENT_ARMS

REACTIVE_DIR = Path(__file__).parent / "experiments" / "reactive"
PREDICTIONS_DIR = Path(__file__).parent / "data" / "predictions" / "reactive"
EARNINGS_PATH = Path(__file__).parent / "calendars" / "earnings.yaml"
TICKER_BUDGET_USD = 12.0
DEV_BUDGET_USD = 2.0
WORST_COST_PER_ORIGIN = {"agent_prices": 0.01, "agent_calendar": 0.01, "agent_news": 0.012 * 1.5 + 3 * SEARCH_FEE_USD}
BATCH = 10

COMPARISONS: tuple[tuple[Arm, Arm, str], ...] = (
    ("agent_news", "agent_prices", "does reading the news help?"),
    ("agent_news", "ewma", "does the news agent beat a standard volatility model?"),
    ("agent_calendar", "agent_prices", "does knowing the earnings date help?"),
    ("agent_calendar", "vol21_earnings", "does the agent add anything beyond calendar-scaled volatility?"),
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def close_series_id(ticker: str) -> str:
    """Series id of a ticker's close, released at the 16:00 close of its own session."""
    return f"{ticker.lower()}_close_at_close"


class _AtCloseAdapter(BaseAdapter):
    """A ticker's adjusted close from the yfinance cache, each value released at its session's close."""

    def __init__(self, ticker: str) -> None:
        self._inner = YFinanceDailyAdapter(ticker=ticker, start=HISTORY_START, cache_dir=DATA_DIR / "yfinance")

    def fetch(self) -> pd.DataFrame:
        frame = self._inner.fetch()
        frame["released_at"] = pd.to_datetime(frame["timestamp"]).dt.normalize() + MARKET_CLOSE
        return frame[["timestamp", "value", "released_at"]]


def build_ticker_service(ticker: str) -> DataService:
    """Return a service holding one ticker's close, released at the close."""
    svc = DataService()
    svc.register(
        close_series_id(ticker),
        _AtCloseAdapter(ticker),
        SeriesMetadata(
            series_id=close_series_id(ticker),
            description=f"{COMPANY[ticker]} daily adjusted close (Yahoo Finance {ticker}), released at 16:00",
            source="yfinance",
            units="USD/share",
            frequency="B",
        ),
    )
    return svc


def load_close(ticker: str) -> pd.Series:
    """Return the full adjusted-close series, indexed by session date."""
    df = _AtCloseAdapter(ticker).fetch()
    return df.set_index(pd.to_datetime(df["timestamp"]).dt.normalize())["value"].sort_index()


def earnings_reports(ticker: str) -> set[pd.Timestamp]:
    """Report dates (all after the close), so the reaction is the next session."""
    raw = yaml.safe_load(EARNINGS_PATH.read_text())
    return {pd.Timestamp(d) for d in raw[ticker]}


def target_sessions(close: pd.Series, start: str = TEST_START, end: str = TEST_END) -> list[pd.Timestamp]:
    """Every session in the window whose previous session is in the series."""
    idx = close.index
    return [d for d in idx[(idx >= start) & (idx <= end)] if idx.get_loc(d) > 0]


def previous_session(close: pd.Series, day: pd.Timestamp) -> pd.Timestamp:
    """Return the session before ``day``."""
    return close.index[close.index.get_loc(day) - 1]


# ---------------------------------------------------------------------------
# Baselines (no LLM)
# ---------------------------------------------------------------------------


def _normal_quantiles(prev_close: float, sd: float) -> dict[float, float]:
    return {q: prev_close * math.exp(NormalDist().inv_cdf(q) * sd) for q in STANDARD_QUANTILES}


def earnings_factor(tickers: tuple[str, ...] = TICKERS) -> float:
    """Sd of earnings-reaction log returns over the previous 21-day vol, pooled, 2020-2025 only."""
    ratios = []
    for t in tickers:
        close = load_close(t).loc[:CALIBRATION_END]
        logret = np.log(close).diff()
        vol = logret.rolling(VOL_WINDOW).std().shift(1)
        for report in earnings_reports(t):
            if not (pd.Timestamp(CALIBRATION_START) <= report <= pd.Timestamp(CALIBRATION_END)):
                continue
            after = close.index[close.index > report]
            if len(after) and not math.isnan(vol.loc[after[0]]):
                ratios.append(logret.loc[after[0]] / vol.loc[after[0]])
    return float(np.std(ratios, ddof=1))


def baseline_forecasts(ticker: str, arm: Arm, sessions: list[pd.Timestamp], factor: float) -> list[dict[str, Any]]:
    """Forecasts for one baseline arm; each uses only closes before the target session."""
    close = load_close(ticker)
    logret = np.log(close).diff()
    vol21 = logret.rolling(VOL_WINDOW).std()
    ewma_var = (logret**2).ewm(alpha=1 - EWMA_DECAY, adjust=False).mean()
    reactions = {close.index[close.index > r][0] for r in earnings_reports(ticker) if (close.index > r).any()}
    rows = []
    for day in sessions:
        prev = previous_session(close, day)
        sd = math.sqrt(ewma_var.loc[prev]) if arm == "ewma" else float(vol21.loc[prev])
        if arm == "vol21_earnings" and day in reactions:
            sd *= factor
        rows.append(_row(ticker, arm, day, prev, float(close.loc[prev]), _normal_quantiles(float(close.loc[prev]), sd)))
    return rows


def _row(ticker: str, arm: str, day: pd.Timestamp, prev: pd.Timestamp, prev_close: float, quantiles: dict[float, float],
         **extra: Any) -> dict[str, Any]:  # fmt: skip
    return {
        "ticker": ticker,
        "arm": arm,
        "target_session": str(day.date()),
        "last_close_date": str(prev.date()),
        "prev_close": prev_close,
        "quantiles": {str(q): v for q, v in quantiles.items()},
        **extra,
    }


# ---------------------------------------------------------------------------
# Agent arms
# ---------------------------------------------------------------------------

_INSTRUCTION = """\
## Role

You are an expert equity analyst covering {company} ({ticker}).

## Input

You will receive a JSON payload containing:
- `task_spec`: the exact question and required JSON output schema
- `ticker`, `company`
- `target_session`: the trading session to forecast; it has not opened yet
- `last_close_date`: the most recent close in the history, the session before `target_session`
- `origin_price_usd`: the split-adjusted close on `last_close_date`, USD per share
- `standard_quantiles`: the quantile levels to forecast
- `target_history_csv`: compressed daily close history (split-adjusted)
- `earnings_reaction_session` (only when provided): true if the company reported results after the close on \
`last_close_date`, so `target_session` is the market's first reaction to them

When context retrieval is enabled, call ``search_web`` BEFORE answering.

## Output contract

Read the data (and briefing, if retrieved) carefully, then execute the task in `task_spec` precisely.

If a `set_model_response` tool is available, call it with your complete JSON as `json_response`; the exact schema \
is described in `task_spec`. Otherwise return the JSON directly as plain text with no preamble.\
"""

_RETRIEVAL_INSTRUCTION = """\
You are an equity news specialist with access to web search.

Search for information relevant to the query and return a concise structured markdown summary (2-4 paragraphs) \
covering what bears on the company's next trading session:
- company news, analyst actions and guidance changes
- earnings results, the earnings call, and the next report date
- scheduled events for the next session: US CPI, jobs report, FOMC decisions
- sector or market-wide news that moved the stock

Ground your summary in the search results you actually retrieve. When a cutoff date is specified, do not report or \
speculate about events that occurred after that date.

Before finalizing your summary, reason step by step: (1) for each candidate fact, judge its actual recency from the \
substance of the result itself, never from a source's claimed publish date or byline timestamp; (2) discard anything \
you cannot confidently place before the cutoff date; (3) only then write your summary. Do not supplement the search \
results with your own background knowledge; if the results are insufficient, say so explicitly.\
"""

_TASK_SPEC = """\
Forecast {ticker}'s split-adjusted closing share price (USD) for `target_session`, the next trading session after \
`last_close_date`. Produce exactly one forecast, with `horizon` 1.

Rules:
  - Use exactly the quantile levels from `standard_quantiles`, no additions, no omissions.
  - `point_forecast` must exactly equal the 0.50 quantile value.
  - Quantile values must be strictly non-decreasing as quantile levels increase.
  - The 0.10 and 0.90 quantiles form an 80% range that should contain the close 80% of the time. Size it to the move \
you expect for this one session: wider when a large move is likely, narrower when it is not.
  - Document your reasoning in the `rationale` fields.

{research}If a `set_model_response` tool is available, call it with your complete JSON as `json_response`. Otherwise \
return the JSON directly as plain text.

Required JSON format:
"""

_RESEARCH = """\
Research before answering. Call `search_web` once per topic below, always with `cutoff_date` equal to \
`target_session`:
  1. `search_web(query="{company} {ticker} stock news, analyst actions and any earnings results on <last_close_date>", \
cutoff_date=<target_session>)`
  2. `search_web(query="scheduled events on <target_session> that could move {ticker}: earnings, US CPI, jobs report, \
FOMC", cutoff_date=<target_session>)`
If a result begins with `[SEARCH_VERIFICATION_FAILED]`, treat that topic as unknown and do not fill it from memory.

"""


def task_spec(ticker: str, arm: Arm) -> str:
    """Return the user-payload task spec for an agent arm."""
    research = _RESEARCH.format(company=COMPANY[ticker], ticker=ticker) if arm == "agent_news" else ""
    return _TASK_SPEC.format(ticker=ticker, research=research) + ContinuousAgentForecastOutput.prompt_schema_json()


class ReactivePromptBuilder(BaseModel):
    """Payload for the next-session range task."""

    ticker: str
    arm: str
    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        df = context.get_series(close_series_id(self.ticker))
        last = df.iloc[-1]
        last_date = pd.Timestamp(last["timestamp"]).normalize()
        payload: dict[str, Any] = {
            "task_spec": task_spec(self.ticker, self.arm),  # type: ignore[arg-type]
            "ticker": self.ticker,
            "company": COMPANY[self.ticker],
            "target_session": str(pd.Timestamp(context.as_of).date()),
            "last_close_date": str(last_date.date()),
            "origin_price_usd": float(last["value"]),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_history_csv": compress_history(df),
        }
        if self.arm == "agent_calendar":
            payload["earnings_reaction_session"] = last_date in earnings_reports(self.ticker)
        return json.dumps(payload, indent=2)


def build_agent(ticker: str, arm: Arm, model: str = LITE_MODEL) -> AgentPredictor:
    """One of the three agent arms for a ticker; same instruction and model, different inputs."""
    retrieval = ContextRetrievalConfig(
        enabled=arm == "agent_news",
        instruction=_RETRIEVAL_INSTRUCTION,
        search_model=LITE_MODEL,
        verifier_model=ADVANCED_MODEL,
        verifier_max_attempts=3,
        verifier_confidence_threshold=8,
    )
    config = AgentConfig(
        name=f"{ticker.lower()}_reactive_{arm.removeprefix('agent_')}",
        model=model,
        instruction=_INSTRUCTION.format(company=COMPANY[ticker], ticker=ticker),
        context_retrieval=retrieval,
    )
    return AgentPredictor(
        agent_config=config,
        prompt_builder=ReactivePromptBuilder(ticker=ticker, arm=arm),
        output_schema=ContinuousAgentForecastOutput,
    )


def reactive_task(ticker: str) -> ForecastingTask:
    """One-session-ahead close; the origin is 00:00 on the target session, so horizon 1 means that session."""
    return ForecastingTask(
        task_id=f"{ticker.lower()}_close_next_session",
        target_series_id=close_series_id(ticker),
        horizons=[1],
        frequency="B",
        description=f"{ticker} close of the session that opens after as_of (00:00 on that session)",
    )


def predictions_path(ticker: str, arm: str, dev: bool = False) -> Path:
    """JSON-lines cache for one ticker and arm."""
    return PREDICTIONS_DIR / ("dev" if dev else "test") / f"{ticker.lower()}_{arm}.jsonl"


def _cached(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["target_session"]: r for r in rows if "quantiles" in r}


def run_agent(ticker: str, arm: Arm, sessions: list[pd.Timestamp], dev: bool = False) -> None:
    """Forecast every session not yet cached, metering cost against the ticker's ledger."""
    path = predictions_path(ticker, arm, dev)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _cached(path)
    todo = [d for d in sessions if str(d.date()) not in done]
    ledger = BudgetLedger(
        path=REACTIVE_DIR / ("budget_dev.yaml" if dev else f"budget_{ticker.lower()}.yaml"),
        cap_usd=DEV_BUDGET_USD if dev else TICKER_BUDGET_USD,
    )
    print(
        f"{ticker} {arm}: {len(done)} cached, {len(todo)} to run; ledger ${ledger.spent_usd:.2f} of ${ledger.cap_usd:.0f}"
    )
    close = load_close(ticker)
    service = build_ticker_service(ticker)
    predictor = build_agent(ticker, arm)
    task = reactive_task(ticker)
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        if len(batch) * WORST_COST_PER_ORIGIN[arm] > ledger.remaining():
            raise RuntimeError(f"{ticker} {arm}: the next batch could exceed the ${ledger.cap_usd:.0f} cap; stopping.")
        meter = CostMeter()
        litellm.callbacks = [*litellm.callbacks, meter]
        try:
            for day in batch:
                prev = previous_session(close, day)
                row = _forecast_one(predictor, task, service, ticker, arm, day, prev, float(close.loc[prev]))
                with path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
        finally:
            litellm.callbacks = [c for c in litellm.callbacks if c is not meter]
            if meter.usd:
                ledger.record(f"{arm} {batch[0].date()}..{batch[-1].date()}", meter.usd, len(batch), meter.searches)
        print(f"  {ticker} {arm}: {min(start + BATCH, len(todo))}/{len(todo)}; ledger ${ledger.spent_usd:.2f}")


def _forecast_one(predictor: AgentPredictor, task: ForecastingTask, service: DataService, ticker: str, arm: Arm,
                  day: pd.Timestamp, prev: pd.Timestamp, prev_close: float) -> dict[str, Any]:  # fmt: skip
    context = service.context(as_of=day.to_pydatetime())
    seen = context.get_series(close_series_id(ticker))
    if pd.Timestamp(seen["timestamp"].iloc[-1]).normalize() != prev:
        raise RuntimeError(
            f"{ticker} {day.date()}: context ends at {seen['timestamp'].iloc[-1]}, expected {prev.date()}"
        )
    error = ""
    for _ in range(3):
        try:
            pred = predictor.predict(task, context)[0]
            q = {float(k): float(v) for k, v in pred.payload.quantiles.items()}
            return _row(ticker, arm, day, prev, prev_close, q,
                        rationale=str(pred.metadata.get("rationale", ""))[:2000],
                        issued_at=datetime.now(tz=timezone.utc).isoformat())  # fmt: skip
        except litellm.AuthenticationError:
            raise  # an expired or missing key fails every origin; stop instead of caching errors
        except Exception as exc:  # noqa: BLE001 - one failed origin must not stop the run
            error = f"{type(exc).__name__}: {exc}"[:500]
    return {"ticker": ticker, "arm": arm, "target_session": str(day.date()), "error": error}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_row(row: dict[str, Any], actual: float) -> dict[str, float]:
    """Interval score, coverage, width and CRPS, all in percent of the previous close."""
    q = {float(k): v for k, v in row["quantiles"].items()}
    lo, hi, prev = q[0.1], q[0.9], row["prev_close"]
    miss = max(0.0, lo - actual) + max(0.0, actual - hi)
    values = np.array([q[k] for k in sorted(q)])
    return {
        "interval_score": ((hi - lo) + (2 / 0.2) * miss) / prev * 100,
        "covered": float(lo <= actual <= hi),
        "width": (hi - lo) / prev * 100,
        "crps": float(properscoring.crps_ensemble(actual, values)) / prev * 100,
        "sd_pct": (hi - lo) / prev * 100 / Z80,
        "return_pct": (actual / prev - 1) * 100,
    }


def load_frame(dev: bool = False, factor: float | None = None) -> pd.DataFrame:
    """Every arm's scored forecasts for every test session (baselines computed on the fly)."""
    factor = earnings_factor() if factor is None else factor
    rows = []
    for t in TICKERS:
        close = load_close(t)
        sessions = [pd.Timestamp(d) for d in DEV_SESSIONS] if dev else target_sessions(close)
        reactions = {close.index[close.index > r][0] for r in earnings_reports(t) if (close.index > r).any()}
        for arm in ARMS:
            arm_rows = (
                baseline_forecasts(t, arm, sessions, factor)
                if arm in BASELINE_ARMS
                else list(_cached(predictions_path(t, arm, dev)).values())
            )
            for r in arm_rows:
                day = pd.Timestamp(r["target_session"])
                rows.append({**r, **score_row(r, float(close.loc[day])), "earnings_reaction": day in reactions})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["target_session"] = pd.to_datetime(frame["target_session"])
    return frame


def paired_difference(frame: pd.DataFrame, a: str, b: str, resamples: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean interval-score difference a - b on shared (ticker, session) pairs, with a date-block bootstrap."""
    pivot = (
        frame[frame["arm"].isin([a, b])]
        .pivot_table(index=["target_session", "ticker"], columns="arm", values="interval_score")
        .dropna()
    )
    diff = (pivot[a] - pivot[b]).rename("d").reset_index()
    by_date = diff.groupby("target_session")["d"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(by_date), size=(resamples, len(by_date)))
    sums, counts = by_date["sum"].to_numpy()[idx].sum(1), by_date["count"].to_numpy()[idx].sum(1)
    boot = sums / counts
    mean = float(diff["d"].mean())
    lo, hi = np.quantile(boot, [0.05, 0.95])
    return {
        "n": int(len(diff)),
        "mean_diff": mean,
        "lo": float(lo),
        "hi": float(hi),
        "improves": bool(mean < 0 and hi < 0),
    }


def vol_targeting(frame: pd.DataFrame) -> pd.DataFrame:
    """Illustration only: size a position to 1% forecast daily risk, capped at 1.5x, per arm and ticker."""
    out = []
    for (arm, t), group in frame.groupby(["arm", "ticker"]):
        g = group.sort_values("target_session")
        pos = np.minimum(MAX_POSITION, TARGET_RISK_PCT / g["sd_pct"].clip(lower=1e-6))
        pnl = pos * g["return_pct"]
        out.append({"arm": arm, "ticker": t, "realised_vol_pct": pnl.std(), "max_drawdown_pct": _max_drawdown(pnl),
                    "buy_hold_vol_pct": g["return_pct"].std(), "buy_hold_drawdown_pct": _max_drawdown(g["return_pct"])})  # fmt: skip
    return pd.DataFrame(out)


def _max_drawdown(returns_pct: pd.Series) -> float:
    wealth = (1 + returns_pct / 100).cumprod()
    return float((wealth / wealth.cummax() - 1).min() * 100)


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-arm means, pooled over tickers, on the sessions every arm has."""
    complete = frame.groupby(["target_session", "ticker"])["arm"].transform("nunique") == frame["arm"].nunique()
    shared = frame[complete]
    table = shared.groupby("arm")[["interval_score", "covered", "width", "crps"]].mean()
    table["n"] = shared.groupby("arm").size()
    table["covered"] *= 100
    return table.reindex([a for a in ARMS if a in table.index])


def main() -> None:
    """Command-line entry: ``run`` one ticker and arm, or ``score`` everything."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--ticker", choices=TICKERS, required=True)
    run.add_argument("--arm", choices=AGENT_ARMS, required=True)
    run.add_argument("--dev", action="store_true", help="2025 wiring check on DEV_SESSIONS (AMD only)")
    score = sub.add_parser("score")
    score.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    if args.cmd == "run":
        close = load_close(args.ticker)
        sessions = [pd.Timestamp(d) for d in DEV_SESSIONS] if args.dev else target_sessions(close)
        run_agent(args.ticker, args.arm, sessions, dev=args.dev)
        return
    factor = earnings_factor()
    frame = load_frame(dev=args.dev, factor=factor)
    print(f"earnings factor (2020-2025): {factor:.2f}\n")
    print(summary(frame).round(3).to_string(), "\n")
    for a, b, question in COMPARISONS:
        if {a, b} <= set(frame["arm"]):
            r = paired_difference(frame, a, b)
            print(f"{a} - {b} ({question}): n={r['n']} diff={r['mean_diff']:+.3f} "
                  f"90% [{r['lo']:+.3f}, {r['hi']:+.3f}] improves={r['improves']}")  # fmt: skip


if __name__ == "__main__":
    main()
