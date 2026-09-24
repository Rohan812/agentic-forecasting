"""Shock smoke backtest: the news-grounded shock agent against climatology on ten fixed origins.

Runs :func:`~ai_stocks_forecasting.tasks.build_nvda_news_predictor` (``"shock"``)
and :class:`~aieng.forecasting.methods.baselines.historical_frequency.HistoricalFrequencyPredictor`
over the origins frozen in ``specs/nvda_shock_smoke.yaml``, then reports:

- **Brier on identical origins.** The harness skips an origin when a predictor
  raises after its retries (a proxy 503, a schema failure). Nothing lines those
  skips up across predictors, so both are scored on the intersection and the
  number of dropped origins is printed.
- **Mean probability before shocks vs. before controls.** The controls are
  volatility-matched, so a gap here is something beyond trailing volatility.
- **Cost and latency per origin**, read back from each prediction's Langfuse
  trace.  The leakage verifier runs on the advanced model and is most of the
  cost, so a per-origin figure that leaves it out is wrong.

Before scoring, every origin's resolved outcome is checked against the label
frozen in the spec, so a change in the price data can't quietly change the test.

Usage, from the repository root (spends proxy credit; about $0.01-0.03 per origin)::

    uv run python -m ai_stocks_forecasting.shock_smoke            # run or load cached
    uv run python -m ai_stocks_forecasting.shock_smoke --force    # recompute

Predictions are cached under ``data/predictions/nvda_shock_smoke/``, and costs go
to ``data/predictions/costs/nvda_shock_smoke.yaml``. Re-running reads the cache and makes no LLM calls.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from ai_stocks_forecasting.baselines import PREDICTIONS_DIR, SPECS_DIR
from ai_stocks_forecasting.data import build_nvda_service
from ai_stocks_forecasting.tasks import (
    NVDA_SHOCK_SERIES_ID,
    build_nvda_news_predictor,
    nvda_shock_task,
    register_shock_series,
)
from aieng.forecasting.data import DataService
from aieng.forecasting.evaluation.artifacts import cached_backtest
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods.baselines.historical_frequency import HistoricalFrequencyPredictor


SPEC_NAME = "nvda_shock_smoke"


@dataclass(frozen=True)
class SmokeSpec:
    """The frozen origins and the outcome each one is expected to resolve to."""

    spec_id: str
    warmup: int
    expected: dict[pd.Timestamp, int]

    def backtest_spec(self) -> BacktestSpec:
        """Build the harness spec for :func:`nvda_shock_task` over these origins."""
        origins = sorted(self.expected)
        return BacktestSpec(
            task=nvda_shock_task(),
            start=origins[0].to_pydatetime(),
            end=origins[-1].to_pydatetime(),
            origin_dates=[o.to_pydatetime() for o in origins],
            warmup=self.warmup,
            description="NVDA shock smoke: 5 holdout shocks + 5 matched controls",
        )


def load_smoke_spec(path: Path | None = None) -> SmokeSpec:
    """Read ``specs/nvda_shock_smoke.yaml``."""
    raw = yaml.safe_load((path or SPECS_DIR / f"{SPEC_NAME}.yaml").read_text())
    expected = {pd.Timestamp(row["as_of"]): int(row["outcome"]) for row in raw["origins"]}
    return SmokeSpec(spec_id=raw["spec_id"], warmup=int(raw["warmup"]), expected=expected)


def check_expected_outcomes(spec: SmokeSpec, service: DataService) -> None:
    """Raise if any origin's next-session outcome differs from the label frozen in the spec."""
    series = service.get_series(NVDA_SHOCK_SERIES_ID, as_of=datetime.now()).set_index("timestamp")["value"]
    mismatches = []
    for as_of, label in spec.expected.items():
        resolved = series.get(as_of + pd.offsets.BDay(1))
        if resolved is None or int(resolved) != label:
            mismatches.append(f"{as_of.date()}: spec says {label}, data says {resolved}")
    if mismatches:
        raise ValueError("Shock smoke origins no longer resolve as frozen: " + "; ".join(mismatches))


def scored_frame(results: dict[str, BacktestResult], spec: SmokeSpec) -> tuple[pd.DataFrame, int]:
    """One row per (predictor, origin) on origins every predictor was scored on, plus the drop count.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Columns ``predictor_id, as_of, probability, outcome, brier``, and the
        number of spec origins missing from at least one predictor.
    """
    rows = []
    for predictor_id, result in results.items():
        for pred, score in zip(result.predictions, result.scores, strict=True):
            as_of = pd.Timestamp(pred.as_of)
            rows.append(
                {
                    "predictor_id": predictor_id,
                    "as_of": as_of,
                    "probability": float(pred.payload.probability),  # type: ignore[union-attr]
                    "outcome": spec.expected[as_of],
                    "brier": float(score),
                    "trace_id": pred.metadata.get("langfuse_trace_id"),
                }
            )
    frame = pd.DataFrame(rows)
    per_predictor = [set(frame.loc[frame.predictor_id == pid, "as_of"]) for pid in results]
    common = set.intersection(*per_predictor) if per_predictor else set()
    dropped = len(spec.expected) - len(common)
    return frame[frame["as_of"].isin(common)].reset_index(drop=True), dropped


def trace_costs(trace_ids: list[str]) -> list[dict[str, Any]]:
    """Read total cost (USD) and latency (s) for each trace from Langfuse; ``None`` where unavailable."""
    from aieng.forecasting.evaluation.langfuse_traces import fetch_trace_with_wait  # noqa: PLC0415

    out = []
    for trace_id in trace_ids:
        trace = fetch_trace_with_wait(trace_id, ready=lambda t: getattr(t, "total_cost", None) is not None)
        out.append(
            {
                "trace_id": trace_id,
                "cost_usd": getattr(trace, "total_cost", None),
                "latency_s": getattr(trace, "latency", None),
            }
        )
    return out


def run(force: bool = False, store_dir: Path | None = None) -> None:
    """Run (or load) both predictors on the smoke origins and print the report."""
    spec = load_smoke_spec()
    service = register_shock_series(build_nvda_service())
    check_expected_outcomes(spec, service)
    store = store_dir or PREDICTIONS_DIR

    predictors: list[Predictor] = [HistoricalFrequencyPredictor(), build_nvda_news_predictor("shock")]
    results = {
        p.predictor_id: cached_backtest(p, spec.backtest_spec(), spec.spec_id, service, store, force_refresh=force)
        for p in predictors
    }
    frame, dropped = scored_frame(results, spec)

    print(f"\n{spec.spec_id}: {len(spec.expected)} origins, {dropped} dropped by the identical-origins intersection")
    summary = frame.groupby("predictor_id").agg(
        n=("brier", "size"),
        brier=("brier", "mean"),
        p_before_shock=("probability", lambda s: s[frame.loc[s.index, "outcome"] == 1].mean()),
        p_before_control=("probability", lambda s: s[frame.loc[s.index, "outcome"] == 0].mean()),
    )
    print(summary.round(3).to_string())

    agent_rows = frame[frame["trace_id"].notna()].sort_values("as_of")
    print("\nAgent forecasts:")
    print(agent_rows[["as_of", "outcome", "probability", "brier"]].round(3).to_string(index=False))

    # Beside, not inside, the spec directory: loaders parse every YAML there as a BacktestResult.
    cost_path = store / "costs" / f"{spec.spec_id}.yaml"
    stale = not cost_path.exists() or any(c["cost_usd"] is None for c in yaml.safe_load(cost_path.read_text()))
    if force or stale:
        costs = trace_costs(agent_rows["trace_id"].tolist())
        for row, as_of in zip(costs, agent_rows["as_of"], strict=True):
            row["as_of"] = str(as_of.date())
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        cost_path.write_text(yaml.safe_dump(costs, sort_keys=False))
    costs = yaml.safe_load(cost_path.read_text())
    known = [c["cost_usd"] for c in costs if c["cost_usd"] is not None]
    latencies = [c["latency_s"] for c in costs if c["latency_s"] is not None]
    if known:
        print(
            f"\nCost: ${sum(known):.3f} over {len(known)} traced origins, ${sum(known) / len(known):.4f} per origin"
            f" (max ${max(known):.4f}); mean latency {sum(latencies) / max(len(latencies), 1):.0f}s"
        )
    missing = len(costs) - len(known)
    if missing:
        print(f"{missing} trace(s) had no cost in Langfuse yet; re-run without --force to read them again.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--force", action="store_true", help="recompute instead of loading cached predictions")
    run(force=parser.parse_args().force)


if __name__ == "__main__":
    main()
