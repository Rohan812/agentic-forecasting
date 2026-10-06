"""Run the news-grounded analyst agent on the price-trajectory task through a backtest spec.

This is the agent row of the trajectory leaderboard: the same multitask
identity as the shock agent (:func:`~ai_stocks_forecasting.tasks.build_nvda_news_predictor`),
asked for 5/10/21-day price quantiles at every origin of the spec.  It runs on
the same weekly 2025 origins as the numerical baselines in
:mod:`~ai_stocks_forecasting.baselines`, so CRPS compares on identical origins.

Results land in this implementation's artefact store next to the baselines
(``data/predictions/<spec_id>/``), and cost and latency per origin are read
back from each prediction's Langfuse trace into
``data/predictions/costs/<spec_id>_trajectory_agent.yaml``.

Usage
-----
From ``implementations/`` (spends proxy credit; about $0.01-0.03 per origin)::

    uv run python -m ai_stocks_forecasting.agent_backtest                  # 2025 backtest
    uv run python -m ai_stocks_forecasting.agent_backtest --max-origins 2 --store /tmp/x  # wiring check

Re-running loads the cached result; pass ``--force`` to recompute.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from ai_stocks_forecasting.baselines import PREDICTIONS_DIR, load_spec
from ai_stocks_forecasting.data import build_nvda_service
from ai_stocks_forecasting.shock_smoke import trace_costs
from ai_stocks_forecasting.tasks import build_nvda_news_predictor
from aieng.forecasting.evaluation.artifacts import cached_multi_backtest
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.models import LITE_MODEL


def run_trajectory_agent(
    spec_name: str = "nvda_backtest",
    *,
    model: str = LITE_MODEL,
    store_dir: Path | None = None,
    max_origins: int | None = None,
    force_refresh: bool = False,
) -> dict[str, BacktestResult]:
    """Backtest the news-grounded trajectory agent and persist its results.

    Parameters
    ----------
    spec_name : str
        Spec under ``specs/``.
    model : str
        Agent model; the lite model by default.
    store_dir : Path or None
        Artefact store root.  Defaults to :data:`~ai_stocks_forecasting.baselines.PREDICTIONS_DIR`.
        A ``max_origins`` check run should pass a scratch directory, so the
        partial result is not cached as the full one.
    max_origins : int or None
        Keep only the first ``max_origins`` origins (at least two), for a cheap wiring check.
    force_refresh : bool
        Recompute even when a cached result exists.
    """
    spec = load_spec(spec_name)
    if max_origins is not None:
        # The spec has no origin list, so shorten its window instead; start < end forces two origins minimum.
        weeks = max(max_origins, 2) - 1
        spec = spec.model_copy(
            update={"end": (pd.Timestamp(spec.start) + pd.offsets.BDay(weeks * spec.stride)).to_pydatetime()}
        )
    return cached_multi_backtest(
        build_nvda_news_predictor("trajectory", model=model),
        spec,
        build_nvda_service(),
        store_dir=store_dir if store_dir is not None else PREDICTIONS_DIR,
        force_refresh=force_refresh,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--spec", default="nvda_backtest", help="spec name under specs/")
    parser.add_argument("--max-origins", type=int, default=None, help="run only the first N origins")
    parser.add_argument("--store", type=Path, default=None, help="artefact store (default: the committed one)")
    parser.add_argument("--force", action="store_true", help="recompute instead of loading cached predictions")
    args = parser.parse_args()

    results = run_trajectory_agent(
        args.spec, store_dir=args.store, max_origins=args.max_origins, force_refresh=args.force
    )
    for task_id, result in results.items():
        print(f"{task_id}: {len(result.predictions)} predictions")
        # The three horizons of one origin share a trace, so cost is read once per origin.
        first_per_trace = {
            p.metadata["langfuse_trace_id"]: p for p in result.predictions if p.metadata.get("langfuse_trace_id")
        }
        if args.store is None and args.max_origins is None:
            costs = trace_costs(list(first_per_trace))
            for row, pred in zip(costs, first_per_trace.values(), strict=True):
                row["as_of"] = str(pred.as_of)[:10]
            path = PREDICTIONS_DIR / "costs" / f"{load_spec(args.spec).spec_id}_trajectory_agent.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(costs, sort_keys=False))
            known = [c["cost_usd"] for c in costs if c["cost_usd"] is not None]
            if known:
                print(f"Cost: ${sum(known):.3f} over {len(known)} origins, ${sum(known) / len(known):.4f} per origin")


if __name__ == "__main__":
    _main()
