"""Numerical baselines for the NVDA forecasting experiment.

The two predictors here are the floor every later model has to beat and the
honest comparison ground for the news-grounded agents:

- :class:`~aieng.forecasting.methods.baselines.naive.LastValuePredictor`
  (``last_value_naive``) — the random-walk floor.  For a liquid equity this is
  a genuinely strong baseline, not a straw man.
- :class:`~aieng.forecasting.methods.numerical.darts_arima.DartsAutoARIMAPredictor`
  (``darts_autoarima``) — the conventional statistical baseline.

Both are cutoff-safe and carry no LLM, so they can be scored on *any* window,
including pre-cutoff ones where LLM rows would only be measuring recall.

Results are written to this implementation's own artefact store
(:data:`PREDICTIONS_DIR`) rather than the repository-root ``data/``, matching
the energy/oil implementation, so the reference baseline outputs can be
committed and re-read without a re-run.

Usage
-----
From the repository root::

    uv run python -m ai_stocks_forecasting.baselines                  # 2025 backtest
    uv run python -m ai_stocks_forecasting.baselines --spec nvda_eval # 2026 protected eval

Re-running is cheap: :func:`~aieng.forecasting.evaluation.artifacts.cached_multi_backtest`
loads any result already on disk and only computes what is missing.  Pass
``--force`` to recompute.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ai_stocks_forecasting.data import build_nvda_service
from aieng.forecasting.data import DataService
from aieng.forecasting.evaluation.artifacts import cached_multi_backtest
from aieng.forecasting.evaluation.backtest import BacktestResult, MultiTargetBacktestSpec
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods import LastValuePredictor
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor


SPECS_DIR = Path(__file__).parent / "specs"
"""Directory holding this implementation's backtest and eval spec YAMLs."""

PREDICTIONS_DIR = Path(__file__).parent / "data" / "predictions"
"""Artefact store for prediction YAMLs, keyed by ``spec_id`` then predictor."""

DEFAULT_ARIMA_SAMPLES = 100
"""Monte Carlo sample count for AutoARIMA.

Kept modest so a full 51-origin backtest finishes in a coffee break.  The
sampling is what makes AutoARIMA probabilistic, so this number does affect
CRPS slightly — raise it for a final scored comparison if the margin between
predictors is thin.
"""


def load_spec(spec_name: str) -> MultiTargetBacktestSpec:
    """Load a spec YAML from :data:`SPECS_DIR` by name.

    Parameters
    ----------
    spec_name : str
        Spec filename with or without the ``.yaml`` suffix, e.g. ``"nvda_backtest"``.

    Returns
    -------
    MultiTargetBacktestSpec
        The parsed spec.  These specs are multi-target (a ``tasks`` list) to
        match the energy/oil ones, even though NVDA currently has a single
        task — it keeps the shape stable when shock and scenario tasks are
        added alongside the trajectory task.
    """
    path = SPECS_DIR / (spec_name if spec_name.endswith(".yaml") else f"{spec_name}.yaml")
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in SPECS_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"No spec {path.name!r} in {SPECS_DIR}. Available: {available}")
    return MultiTargetBacktestSpec.model_validate(yaml.safe_load(path.read_text()))


def build_baseline_predictors(num_samples: int = DEFAULT_ARIMA_SAMPLES) -> list[Predictor]:
    """Return the two numerical baselines, in leaderboard order.

    Parameters
    ----------
    num_samples : int
        Monte Carlo sample count handed to AutoARIMA.
    """
    return [LastValuePredictor(), DartsAutoARIMAPredictor(num_samples=num_samples)]


def run_baselines(
    spec_name: str = "nvda_backtest",
    *,
    data_service: DataService | None = None,
    store_dir: Path | None = None,
    num_samples: int = DEFAULT_ARIMA_SAMPLES,
    force_refresh: bool = False,
) -> dict[str, dict[str, BacktestResult]]:
    """Run both numerical baselines through a spec and persist the results.

    Parameters
    ----------
    spec_name : str
        Spec to run, resolved against :data:`SPECS_DIR`.
    data_service : DataService or None
        Pre-populated service.  When ``None``, a cache-backed NVDA service is
        built via :func:`~ai_stocks_forecasting.data.build_nvda_service`.
    store_dir : Path or None
        Artefact store root.  Defaults to :data:`PREDICTIONS_DIR`.
    num_samples : int
        Monte Carlo sample count for AutoARIMA.
    force_refresh : bool
        Recompute even when a cached result file already exists.

    Returns
    -------
    dict[str, dict[str, BacktestResult]]
        Results keyed by ``predictor_id`` then ``task_id``.
    """
    spec = load_spec(spec_name)
    service = data_service if data_service is not None else build_nvda_service()
    store = store_dir if store_dir is not None else PREDICTIONS_DIR

    results: dict[str, dict[str, BacktestResult]] = {}
    for predictor in build_baseline_predictors(num_samples=num_samples):
        results[predictor.predictor_id] = cached_multi_backtest(
            predictor,
            spec,
            service,
            store_dir=store,
            force_refresh=force_refresh,
        )
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", default="nvda_backtest", help="spec name under specs/ (default: nvda_backtest)")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_ARIMA_SAMPLES, help="AutoARIMA Monte Carlo samples")
    parser.add_argument("--force", action="store_true", help="recompute even if cached results exist")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    print(f"Spec {spec.spec_id}: {spec.start.date()} → {spec.end.date()}, stride {spec.stride}, warmup {spec.warmup}")
    results = run_baselines(args.spec, num_samples=args.num_samples, force_refresh=args.force)

    for predictor_id, per_task in results.items():
        for task_id, result in per_task.items():
            path = PREDICTIONS_DIR / spec.spec_id / f"{predictor_id}__{task_id}.yaml"
            print(f"  {predictor_id:<18} {task_id}: {len(result.predictions)} predictions → {path}")
    print("Done.")


if __name__ == "__main__":
    _main()
