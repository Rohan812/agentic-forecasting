"""Small shock backtests: news-grounded shock agents against climatology on fixed origins.

Each spec (``specs/nvda_shock_*.yaml``) freezes its origins, the outcome each
resolves to, and the **arms** it runs:

- ``climatology``: :class:`~aieng.forecasting.methods.baselines.historical_frequency.HistoricalFrequencyPredictor`
- ``agent``: :func:`~ai_stocks_forecasting.tasks.build_nvda_shock_predictor`, with search topics
- ``agent_notopics``: the same agent without the named search topics (the ablation arm)

Specs:

- ``nvda_shock_smoke``: 5 gate-holdout shocks and 5 matched controls.  Runs
  ``climatology`` and ``agent_notopics`` only.  Its results were produced before
  the search topics existed, and the agent arm must not be re-run there,
  because that would tune on the holdout.
- ``nvda_shock_fresh``: every post-cutoff 2025 session that follows a shock
  session, none of them a holdout or smoke target.  Runs all three arms.

For each spec it reports:

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

Usage, from the repository root (spends proxy credit; about $0.01-0.03 per agent origin)::

    uv run python -m ai_stocks_forecasting.shock_smoke                          # nvda_shock_smoke
    uv run python -m ai_stocks_forecasting.shock_smoke --spec nvda_shock_fresh
    uv run python -m ai_stocks_forecasting.shock_smoke --spec nvda_shock_fresh --force   # recompute

Predictions are cached under ``data/predictions/<spec_id>/`` and costs go to
``data/predictions/costs/<spec_id>.yaml``.  Re-running reads the cache and makes no LLM calls.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from ai_stocks_forecasting import signals
from ai_stocks_forecasting.baselines import PREDICTIONS_DIR, SPECS_DIR
from ai_stocks_forecasting.data import NVDA_SERIES_ID, build_nvda_service
from ai_stocks_forecasting.tasks import (
    NVDA_SHOCK_SERIES_ID,
    build_nvda_shock_predictor,
    nvda_shock_task,
    register_shock_series,
)
from aieng.forecasting.data import DataService
from aieng.forecasting.evaluation.artifacts import cached_backtest
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods.baselines.historical_frequency import HistoricalFrequencyPredictor


SPEC_NAME = "nvda_shock_smoke"

Arm = Literal["climatology", "agent", "agent_notopics"]


def build_arm(arm: Arm) -> Predictor:
    """Build the predictor for one arm name used in the spec YAMLs."""
    if arm == "climatology":
        return HistoricalFrequencyPredictor()
    if arm == "agent":
        return build_nvda_shock_predictor()
    if arm == "agent_notopics":
        return build_nvda_shock_predictor(search_topics=False)
    raise ValueError(f"Unknown arm {arm!r}; expected climatology, agent or agent_notopics.")


@dataclass(frozen=True)
class SmokeSpec:
    """The frozen origins, the outcome each one is expected to resolve to, and the arms to run."""

    spec_id: str
    warmup: int
    expected: dict[pd.Timestamp, int]
    arms: tuple[Arm, ...] = ("climatology", "agent")
    description: str = ""

    def backtest_spec(self) -> BacktestSpec:
        """Build the harness spec for :func:`nvda_shock_task` over these origins."""
        origins = sorted(self.expected)
        return BacktestSpec(
            task=nvda_shock_task(),
            start=origins[0].to_pydatetime(),
            end=origins[-1].to_pydatetime(),
            origin_dates=[o.to_pydatetime() for o in origins],
            warmup=self.warmup,
            description=self.description or f"NVDA shock backtest {self.spec_id}",
        )


def load_smoke_spec(name: str = SPEC_NAME) -> SmokeSpec:
    """Read ``specs/<name>.yaml``."""
    raw = yaml.safe_load((SPECS_DIR / f"{name}.yaml").read_text())
    expected = {pd.Timestamp(row["as_of"]): int(row["outcome"]) for row in raw["origins"]}
    return SmokeSpec(
        spec_id=raw["spec_id"],
        warmup=int(raw["warmup"]),
        expected=expected,
        arms=tuple(raw["arms"]),
        description=str(raw.get("summary", "")),
    )


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


def check_agent_origins_are_fresh(spec: SmokeSpec, service: DataService) -> None:
    """Refuse to run the ``agent`` arm on a gate-holdout shock window or a smoke origin.

    The search topics were written after looking at the smoke results, so
    scoring them there, or on any other holdout window, would tune the prompt
    on the holdout.  Shock windows are deterministic (``signals.flag_shock_windows``
    from 2020); the matched controls depend on a sampling seed, so they are not
    checked here.  Spec authors exclude them, as ``nvda_shock_fresh.yaml`` records.
    """
    if "agent" not in spec.arms:
        return
    prices = service.get_series(NVDA_SERIES_ID, as_of=datetime.now())
    prices = prices[pd.to_datetime(prices["timestamp"]) >= pd.Timestamp("2020-01-01")]
    windows = signals.flag_shock_windows(prices)
    holdout_start = pd.Timestamp(signals.HOLDOUT_START)
    forbidden = {w.as_of for w in windows if w.as_of >= holdout_start}
    if spec.spec_id != SPEC_NAME:
        forbidden |= set(load_smoke_spec(SPEC_NAME).expected)
    clashes = sorted(a for a in spec.expected if a in forbidden)
    if clashes:
        raise ValueError(
            f"{spec.spec_id} runs the `agent` arm on holdout or smoke origins: "
            + ", ".join(str(a.date()) for a in clashes)
        )


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


def run(spec_name: str = SPEC_NAME, force: bool = False, store_dir: Path | None = None) -> None:
    """Run (or load) the spec's arms on its origins and print the report."""
    spec = load_smoke_spec(spec_name)
    service = register_shock_series(build_nvda_service())
    check_expected_outcomes(spec, service)
    check_agent_origins_are_fresh(spec, service)
    store = store_dir or PREDICTIONS_DIR

    predictors = [build_arm(arm) for arm in spec.arms]
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

    agent_rows = frame[frame["trace_id"].notna()].sort_values(["as_of", "predictor_id"])
    print("\nAgent forecasts (one column per agent arm):")
    wide = agent_rows.pivot(index=["as_of", "outcome"], columns="predictor_id", values="probability")
    wide.columns = [c.replace("agent_predictor_nvda_analyst_", "").split("_gemini")[0] for c in wide.columns]
    print(wide.round(3).to_string())

    # Beside, not inside, the spec directory: loaders parse every YAML there as a BacktestResult.
    cost_path = store / "costs" / f"{spec.spec_id}.yaml"
    stale = not cost_path.exists() or any(c["cost_usd"] is None for c in yaml.safe_load(cost_path.read_text()))
    if force or stale:
        costs = trace_costs(agent_rows["trace_id"].tolist())
        for row, (_, agent_row) in zip(costs, agent_rows.iterrows(), strict=True):
            row["as_of"] = str(agent_row["as_of"].date())
            row["predictor_id"] = agent_row["predictor_id"]
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        cost_path.write_text(yaml.safe_dump(costs, sort_keys=False))
    costs = yaml.safe_load(cost_path.read_text())
    print()
    for predictor_id in sorted({c.get("predictor_id", "agent") for c in costs}):
        arm_costs = [c for c in costs if c.get("predictor_id", "agent") == predictor_id]
        known = [c["cost_usd"] for c in arm_costs if c["cost_usd"] is not None]
        latencies = [c["latency_s"] for c in arm_costs if c["latency_s"] is not None]
        if known:
            print(
                f"Cost {predictor_id}: ${sum(known):.3f} over {len(known)} origins, "
                f"${sum(known) / len(known):.4f} per origin (max ${max(known):.4f}); "
                f"mean latency {sum(latencies) / max(len(latencies), 1):.0f}s"
            )
    missing = sum(c["cost_usd"] is None for c in costs)
    if missing:
        print(f"{missing} trace(s) had no cost in Langfuse yet; re-run without --force to read them again.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--spec", default=SPEC_NAME, help="spec name under specs/, e.g. nvda_shock_fresh")
    parser.add_argument("--force", action="store_true", help="recompute instead of loading cached predictions")
    args = parser.parse_args()
    run(args.spec, force=args.force)


if __name__ == "__main__":
    main()
