"""Presentation charts for the NVDA experiment.

The headline chart is :func:`coverage_sharpness` — the plot that answers "is
this forecaster honest?", which is a different question from "is it accurate?"
and the one a probabilistic leaderboard is prone to hide.

Everything here loads from the prediction registry by globbing, so a chart picks
up new predictors the moment their YAMLs land in the store.  Re-running the
notebook after a study session is the whole update path; no code changes.

Design notes worth keeping if these charts are edited
-----------------------------------------------------
*Colour encodes the predictor **family**, not the individual predictor.*  There
are three families (baseline / numerical ML / LLM-agent) and the categorical
palette is only validated for colourblind separation up to three series on a
scatter, where every pair of points is compared against every other.  A fourth
hue would fail that check.  Individual predictors within a family are separated
by marker shape instead, so the chart scales to a dozen predictors without
inventing colours — and, because colour is tied to the family rather than to
rank, adding a predictor never repaints the existing ones between phases.

*Every point is directly labelled and a table is printed alongside.*  The aqua
family colour sits below 3:1 contrast on a white surface, so identity may not
rest on colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ai_stocks_forecasting.analysis import predictions_to_frame
from aieng.forecasting.data import DataService
from aieng.forecasting.evaluation.backtest import BacktestResult


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


PREDICTIONS_DIR = Path(__file__).parent / "data" / "predictions"

# ── Palette ───────────────────────────────────────────────────────────────────
# Categorical slots 1-3 of the validated default palette. Checked with the
# palette validator in all-pairs mode (scatter): worst CVD deltaE 9.2, worst
# normal-vision deltaE 24.0, both clear. Aqua warns on contrast (2.74:1), which
# is why direct labels and the table view are not optional here.
FAMILY_COLORS: dict[str, str] = {
    "Baseline": "#2a78d6",  # blue
    "Numerical ML": "#eb6834",  # orange
    "LLM / Agent": "#1baf7a",  # aqua
    "Other": "#52514e",  # de-emphasis grey, never a generated hue
}

# Marker shape separates predictors *within* a family. Assigned from a fixed
# list so a predictor keeps its shape as the registry grows.
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#fcfcfb"
GRID = "#e3e2dd"

NOMINAL_COVERAGE = 80.0
"""The coverage an 80% prediction interval claims. The line every point is read against."""


def load_scored_frame(spec_id: str, data_service: DataService) -> pd.DataFrame:
    """Load every committed prediction for a spec into one tidy scored frame.

    Globs ``data/predictions/<spec_id>/*.yaml``, so predictors added in later
    phases appear automatically.

    Parameters
    ----------
    spec_id
        Directory key under the prediction store, e.g. ``"nvda_backtest"``.
    data_service
        Used to look up realised values for error and coverage columns.

    Returns
    -------
    pd.DataFrame
        One row per scored prediction, with ``predictor``, ``family``,
        ``horizon``, ``crps``, ``abs_error``, ``width80`` and ``inside80``.
        Empty if the spec directory holds no results yet.
    """
    spec_dir = PREDICTIONS_DIR / spec_id
    if not spec_dir.exists():
        return pd.DataFrame()

    results: dict[str, dict[str, BacktestResult]] = {}
    for path in sorted(spec_dir.glob("*.yaml")):
        predictor_id, _, task_id = path.stem.partition("__")
        result = BacktestResult.model_validate(yaml.safe_load(path.read_text()))
        results.setdefault(_display_name(predictor_id), {})[task_id or "default"] = result
    return predictions_to_frame(results, data_service)


def _display_name(predictor_id: str) -> str:
    """Turn a registry predictor id into a readable chart label."""
    pretty = {
        "last_value_naive": "Naive (last value)",
        "darts_autoarima": "AutoARIMA",
        "darts_autoarima_log": "AutoARIMA (log returns)",
        "darts_lightgbm": "LightGBM",
        "prophet_daily": "Prophet",
    }
    return pretty.get(predictor_id, predictor_id.replace("_", " "))


def coverage_sharpness_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the numbers behind :func:`coverage_sharpness`.

    This is the table view the contrast-relief rule requires, and it is also the
    thing to read when two points overlap on the plot.
    """
    if frame.empty:
        return pd.DataFrame()
    out = frame.groupby(["family", "predictor", "horizon"]).agg(
        coverage_80=("inside80", lambda s: 100 * s.mean()),
        mean_width=("width80", "mean"),
        mean_crps=("crps", "mean"),
        n=("crps", "size"),
    )
    out["miscalibration"] = (out["coverage_80"] - NOMINAL_COVERAGE).round(1)
    return out.round(2)


def coverage_sharpness(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 10, 21),
    title: str = "Are the intervals honest? Coverage vs sharpness, NVDA",
) -> tuple[Figure, Any]:
    """Plot realised coverage against interval width, one panel per horizon.

    **How to read it.**  The dashed line is the 80% coverage an 80% interval
    promises.  A predictor *on* the line is honest; the further left it sits on
    the line, the more useful it is, because it achieves that honesty with a
    narrower interval.  **Below** the line is overconfident — the interval is
    too narrow and the forecaster is claiming more certainty than it has.
    **Above** the line is under-confident: correct but vague.

    Horizons are faceted rather than pooled because interval width is in dollars
    and is not comparable across them: a $12 interval at 21 days is not
    "worse" than a $4 interval at 5 days, it is answering a harder question.

    Parameters
    ----------
    frame
        Scored frame from :func:`load_scored_frame`.
    horizons
        Business-day horizons to facet over, one panel each.
    title
        Figure title.

    Returns
    -------
    tuple[Figure, Any]
        The figure and its array of axes, so a caller can annotate further.
    """
    summary = coverage_sharpness_table(frame)
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 5.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes_list = list(axes) if len(horizons) > 1 else [axes]

    markers = _marker_map(frame)
    for ax, horizon in zip(axes_list, horizons, strict=False):
        _panel(ax, summary, horizon, markers)

    axes_list[0].set_ylabel("Realised coverage of the 80% interval (%)", color=INK_SECONDARY, fontsize=10)
    fig.suptitle(title, fontsize=14, color=INK_PRIMARY, fontweight="600", y=0.99)
    fig.text(
        0.5,
        0.005,
        "On the dashed line = honest · below = overconfident · above = vague. Further left on the line is better.",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        # A legend is present whenever there are >= 2 series, even though every
        # point is also directly labelled — identity must never rest on colour.
        fig.legend(
            handles,
            labels,
            loc="upper right",
            frameon=False,
            fontsize=9,
            labelcolor=INK_SECONDARY,
            bbox_to_anchor=(0.995, 0.955),
        )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return fig, axes


def _marker_map(frame: pd.DataFrame) -> dict[str, str]:
    """Assign a stable marker shape per predictor, grouped by family."""
    order = sorted(frame["predictor"].unique()) if not frame.empty else []
    return {name: _MARKERS[i % len(_MARKERS)] for i, name in enumerate(order)}


def _panel(ax: Axes, summary: pd.DataFrame, horizon: int, markers: dict[str, str]) -> None:
    """Draw one horizon's panel."""
    ax.set_facecolor(SURFACE)
    ax.axhline(NOMINAL_COVERAGE, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)
    # Reference label on the left: well-calibrated predictors have wide intervals
    # and sit on the line at the right, while the left end of the line is where
    # the degenerate zero-width baseline lives, far below it.
    ax.text(
        0.01,
        NOMINAL_COVERAGE + 1.5,
        "nominal 80%",
        transform=ax.get_yaxis_transform(),
        ha="left",
        fontsize=8.5,
        color=INK_MUTED,
    )

    if summary.empty:
        ax.text(0.5, 0.5, "no predictions yet", transform=ax.transAxes, ha="center", color=INK_MUTED, fontsize=11)
        ax.set_xlim(-0.5, 10)
    else:
        levels = summary.index.get_level_values("horizon")
        rows = summary.xs(horizon, level="horizon") if horizon in levels else summary.iloc[0:0]
        for (family, predictor), row in rows.iterrows():
            ax.scatter(
                row["mean_width"],
                row["coverage_80"],
                s=170,
                color=FAMILY_COLORS.get(family, FAMILY_COLORS["Other"]),
                marker=markers.get(predictor, "o"),
                edgecolor=SURFACE,  # 2px surface ring so overlapping marks stay readable
                linewidth=2,
                zorder=3,
                label=f"{predictor} · {family}",
            )
            # A point near the nominal line would put a label above it straight
            # onto the dashed line, so near the line the label goes underneath.
            near_line = abs(row["coverage_80"] - NOMINAL_COVERAGE) < 12
            ax.annotate(
                predictor,
                (row["mean_width"], row["coverage_80"]),
                textcoords="offset points",
                xytext=(0, -18 if near_line else 14),
                ha="center",
                va="top" if near_line else "baseline",
                fontsize=9,
                color=INK_SECONDARY,  # text wears ink tokens, never the series colour
                zorder=4,
            )

    ax.set_title(f"h = {horizon} business days", fontsize=11, color=INK_PRIMARY, pad=10)
    ax.set_xlabel("Mean 80% interval width (USD/share)  →  vaguer", color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(-5, 105)
    if not summary.empty and len(rows):
        # Margin scales with the data so the direct label on a point sitting at
        # x=0 (a zero-width, degenerate interval) is not clipped by the spine.
        span = max(float(rows["mean_width"].max()), 1.0)
        ax.set_xlim(-0.22 * span, 1.28 * span)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


__all__ = [
    "FAMILY_COLORS",
    "NOMINAL_COVERAGE",
    "PREDICTIONS_DIR",
    "coverage_sharpness",
    "coverage_sharpness_table",
    "load_scored_frame",
]
