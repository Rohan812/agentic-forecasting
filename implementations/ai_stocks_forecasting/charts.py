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

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ai_stocks_forecasting.analysis import predictions_to_frame
from ai_stocks_forecasting.data import NVDA_SERIES_ID
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


def predicted_vs_actual(
    frame: pd.DataFrame,
    data_service: DataService,
    horizons: tuple[int, ...] = (5, 10, 21),
    predictors: list[str] | None = None,
    series_id: str = NVDA_SERIES_ID,
    title: str = "NVDA: forecasts vs actual price",
) -> tuple[Figure, Any]:
    """Line chart of each predictor's forecasts against the actual daily price.

    One panel per horizon, sharing the time axis.  In each panel the black line
    is the actual close, and each predictor's point forecast is drawn at the
    date it was *forecasting* (``forecast_date``, not the origin), so a perfect
    forecaster would sit exactly on the black line.  The shaded band is the
    central 80% interval (10th to 90th percentile); predictors with zero-width
    intervals, such as the naive baseline, get no band.

    **How to read it.**  The naive forecast is the actual line shifted right by
    the horizon, so on a 21-day panel it visibly lags every turn.  A model that
    adds information tracks turns sooner than that; a model whose band keeps
    missing the black line is overconfident; one whose band is far wider than
    the line's wiggles is vague.

    Parameters
    ----------
    frame
        Scored frame from :func:`load_scored_frame`.
    data_service
        Source of the actual daily price for the black line.
    horizons
        Business-day horizons, one panel each.
    predictors
        Subset of ``frame['predictor']`` to draw.  ``None`` draws all of them.
    series_id
        Target series for the actual-price line.
    title
        Figure title.

    Returns
    -------
    tuple[Figure, Any]
        The figure and its array of axes.
    """
    names = sorted(frame["predictor"].unique()) if predictors is None else predictors
    markers = _marker_map(frame)
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    actual = data_service.get_series(series_id, as_of=now).set_index("timestamp")["value"]

    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 3.4 * len(horizons)), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes_list = list(axes) if len(horizons) > 1 else [axes]

    start = pd.Timestamp(frame["forecast_date"].min()) - pd.Timedelta(days=5)
    end = pd.Timestamp(frame["forecast_date"].max())
    window = actual.loc[start:end]

    for ax, horizon in zip(axes_list, horizons, strict=False):
        ax.set_facecolor(SURFACE)
        # Each panel shows the actual price only as far as its own forecasts reach,
        # so the line never runs on underneath the direct labels at the right.
        panel_end = pd.Timestamp(frame.loc[frame["horizon"] == horizon, "forecast_date"].max())
        panel_actual = window.loc[:panel_end]
        ax.plot(panel_actual.index, panel_actual.values, color=INK_PRIMARY, linewidth=2, label="Actual close", zorder=4)
        ends: list[tuple[str, pd.Timestamp, float]] = []
        for name in names:
            sub = frame[(frame["predictor"] == name) & (frame["horizon"] == horizon)].sort_values("forecast_date")
            if sub.empty:
                continue
            family = sub["family"].iloc[0]
            color = FAMILY_COLORS.get(family, FAMILY_COLORS["Other"])
            dates = pd.to_datetime(sub["forecast_date"])
            if (sub["q90"] - sub["q10"]).max() > 1e-9:
                ax.fill_between(dates, sub["q10"], sub["q90"], color=color, alpha=0.14, linewidth=0, zorder=1)
            ax.plot(
                dates,
                sub["point"],
                color=color,
                linewidth=2,
                marker=markers.get(name, "o"),
                markersize=6,
                markeredgecolor=SURFACE,  # surface ring keeps overlapping markers readable
                markeredgewidth=1,
                label=f"{name} · {family}",
                zorder=3,
            )
            ends.append((name, dates.iloc[-1], float(sub["point"].iloc[-1])))
        _stacked_end_labels(ax, ends)
        ax.set_title(f"h = {horizon} business days ahead", fontsize=10.5, color=INK_PRIMARY, loc="left")
        ax.set_ylabel("USD / share", color=INK_SECONDARY, fontsize=9.5)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        # Room on the right for the direct labels.
        ax.set_xlim(start, end + pd.Timedelta(days=(end - start).days * 0.13))

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        ncol=len(labels),
        bbox_to_anchor=(0.995, 0.985),
    )
    fig.suptitle(title, fontsize=14, color=INK_PRIMARY, fontweight="600", x=0.01, ha="left", y=0.99)
    fig.text(
        0.01,
        0.005,
        "Each forecast is plotted at the date it forecast. Shaded band: central 80% interval.",
        fontsize=9,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return fig, axes


def _stacked_end_labels(ax: Axes, ends: list[tuple[str, pd.Timestamp, float]], gap_pt: float = 11.0) -> None:
    """Direct-label each line at its last point, pushing labels apart vertically.

    Forecasts that end at nearly the same price — common, since most models sit
    close to the random walk — would otherwise print their labels on top of each
    other.  Labels are ordered by end value and spread ``gap_pt`` points apart
    around the cluster's centre; the text wears ink, the colour stays in the line.
    """
    if not ends:
        return
    ordered = sorted(ends, key=lambda e: e[2])
    centre = (len(ordered) - 1) / 2
    for rank, (name, x, y) in enumerate(ordered):
        ax.annotate(
            name,
            (x, y),
            xytext=(8, (rank - centre) * gap_pt),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )


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
            # Near the nominal line, a label on the wrong side lands on the dashed
            # line, so put it on the side facing away from the line.
            near_line = abs(row["coverage_80"] - NOMINAL_COVERAGE) < 12
            below = near_line and row["coverage_80"] < NOMINAL_COVERAGE
            ax.annotate(
                predictor,
                (row["mean_width"], row["coverage_80"]),
                textcoords="offset points",
                xytext=(0, -18 if below else 14),
                ha="center",
                va="top" if below else "baseline",
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
    "predicted_vs_actual",
]
