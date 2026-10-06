"""Charts for the NVDA notebook series (01-04), built for presenting.

Every chart reads committed artefacts (prices from the cache, predictions from
the registry), so the notebooks run in seconds and spend no proxy credit.  The
look follows :mod:`~ai_stocks_forecasting.charts`: the same validated
three-family palette, ink tokens for all text, recessive grids, a legend on
every multi-series chart, and shape as well as colour for every distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai_stocks_forecasting import signals
from ai_stocks_forecasting.analysis import predictor_family
from ai_stocks_forecasting.charts import (
    FAMILY_COLORS,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
)
from ai_stocks_forecasting.paths import SHOCK_THRESHOLD
from matplotlib.lines import Line2D


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


UP = FAMILY_COLORS["Baseline"]  # blue
DOWN = FAMILY_COLORS["Numerical ML"]  # orange
PRICE_LINE = INK_PRIMARY


@dataclass(frozen=True)
class MarketEvent:
    """A dated, publicly documented event behind a large NVDA move.

    ``label_at`` places the label: a calendar-day offset from the event and a
    price level, chosen by hand so that the 2025 labels sit in empty space.
    """

    day: str
    label: str
    label_at: tuple[int, float] = (0, 0.0)


EVENTS_2025: tuple[MarketEvent, ...] = (
    MarketEvent("2025-01-27", "DeepSeek R1 sell-off", (-10, 182)),
    MarketEvent("2025-02-27", "Day after Q4 FY25 earnings", (-12, 80)),
    MarketEvent("2025-04-03", "'Liberation Day' tariffs", (-22, 163)),
    MarketEvent("2025-04-09", "90-day tariff pause", (10, 200)),
    MarketEvent("2025-04-16", "H20 China export licence,\n$5.5B charge", (28, 78)),
    MarketEvent("2025-05-12", "US-China tariff truce", (26, 176)),
)
"""Events labelled on the case-study chart.  Every other shock day is marked but unlabelled."""


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9.5)


def price_series(prices: pd.DataFrame) -> pd.Series:
    """Return the close as a Series indexed by session date."""
    return prices.set_index(pd.to_datetime(prices["timestamp"]).dt.normalize())["value"].sort_index()


def shock_days(close: pd.Series, start: str, end: str, threshold_pct: float = SHOCK_THRESHOLD) -> pd.DataFrame:
    """Every session in ``[start, end]`` whose close moved at least ``threshold_pct`` from the prior close."""
    ret = close.pct_change() * 100
    window = ret.loc[start:end]
    hits = window[window.abs() >= threshold_pct]
    return pd.DataFrame({"return_pct": hits.round(1), "close": close.loc[hits.index].round(2)})


def price_story(
    prices: pd.DataFrame,
    start: str = "2025-01-01",
    end: str = "2025-12-31",
    events: tuple[MarketEvent, ...] = EVENTS_2025,
    title: str = "NVDA in 2025: a year of one-day shocks",
) -> tuple[Figure, Axes]:
    """Plot the close with every ±5% session marked and the documented events labelled.

    Up shocks are blue triangles pointing up, down shocks orange triangles
    pointing down, so the direction never rests on colour alone.
    """
    close = price_series(prices)
    window = close.loc[start:end]
    shocks = shock_days(close, start, end)

    fig, ax = plt.subplots(figsize=(13, 5.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(window.index, window.values, color=PRICE_LINE, linewidth=1.6, zorder=2)
    for sign, marker, color, label in (
        (1, "^", UP, "up ≥5% in one session"),
        (-1, "v", DOWN, "down ≥5% in one session"),
    ):
        sub = shocks[np.sign(shocks["return_pct"]) == sign]
        ax.scatter(
            sub.index,
            sub["close"],
            marker=marker,
            s=90,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
            label=label,
        )

    for event in events:
        day = pd.Timestamp(event.day)
        if day not in close.index or not (pd.Timestamp(start) <= day <= pd.Timestamp(end)):
            continue
        move = (close.pct_change() * 100).loc[day]
        dx, y = event.label_at
        ax.annotate(
            f"{event.label}\n{day:%b %d}: {move:+.1f}%",
            (day, close.loc[day]),
            xytext=(day + pd.Timedelta(days=dx), y or close.loc[day]),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
            arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8, "shrinkA": 4, "shrinkB": 6},
            zorder=4,
        )
    ax.set_ylim(min(70, window.min() * 0.8), max(215, window.max() * 1.05))

    ax.set_ylabel("Split-adjusted close (USD)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(title, loc="left", fontsize=14, color=INK_PRIMARY, fontweight="600", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)
    n_up = int((shocks["return_pct"] > 0).sum())
    fig.text(
        0.01,
        0.01,
        f"{len(shocks)} sessions moved ≥5% ({n_up} up, {len(shocks) - n_up} down). Source: yfinance adjusted close.",
        fontsize=9,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, ax


def origin_fan(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
    origin: str,
    predictors: list[str] | None = None,
    history_days: int = 60,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Show what each predictor forecast from one origin against what happened.

    Each predictor's median and 80% interval (10th-90th percentile) are drawn
    at the 5, 10 and 21-session horizons, slightly offset so they do not
    overlap.  The black line is the realised close: history up to the origin,
    then the path the forecasts were trying to hit.
    """
    close = price_series(prices)
    at = frame[frame["as_of"] == pd.Timestamp(origin)]
    names = sorted(at["predictor"].unique()) if predictors is None else predictors
    last_target = pd.Timestamp(at["forecast_date"].max())
    # A midnight origin has seen closes up to the session before it; the dashed line picks up from there.
    hist = close.loc[: pd.Timestamp(origin) - pd.Timedelta(days=1)].iloc[-history_days:]
    after = close.loc[hist.index[-1] : last_target + pd.Timedelta(days=5)]

    fig, ax = plt.subplots(figsize=(12, 5.4))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(hist.index, hist.values, color=PRICE_LINE, linewidth=1.6, zorder=2)
    ax.plot(after.index, after.values, color=PRICE_LINE, linewidth=1.6, linestyle=(0, (4, 2)), zorder=2)
    ax.axvline(pd.Timestamp(origin), color=INK_MUTED, linewidth=1, linestyle=":", zorder=1)
    ax.text(pd.Timestamp(origin), ax.get_ylim()[1], " forecast origin", va="top", fontsize=9, color=INK_MUTED)

    markers = ["o", "s", "D", "^", "v"]
    span = len(names)
    handles = [
        Line2D([], [], color=PRICE_LINE, linewidth=1.6, label="close the forecaster had seen"),
        Line2D([], [], color=PRICE_LINE, linewidth=1.6, linestyle=(0, (4, 2)), label="what happened next"),
    ]
    for i, name in enumerate(names):
        rows = at[at["predictor"] == name].sort_values("forecast_date")
        color = FAMILY_COLORS.get(predictor_family(name), FAMILY_COLORS["Other"])
        shift = pd.Timedelta(hours=(i - (span - 1) / 2) * 20)
        xs = pd.to_datetime(rows["forecast_date"]) + shift
        ax.vlines(xs, rows["q10"], rows["q90"], color=color, linewidth=2.2, alpha=0.85, zorder=3)
        ax.scatter(
            xs,
            rows["q50"],
            marker=markers[i % len(markers)],
            s=70,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=4,
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=markers[i % len(markers)],
                markersize=8,
                linewidth=2.2,
                label=f"{name}: median, 80% band",
            )
        )
    ax.set_ylabel("Close (USD)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        title or f"What each forecaster said on {pd.Timestamp(origin):%b %d, %Y}",
        loc="left",
        fontsize=14,
        color=INK_PRIMARY,
        fontweight="600",
        pad=12,
    )
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig, ax


def regime_crps_table(frame: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Mean CRPS per predictor in each volatility regime at the forecast origin.

    The regime is :func:`ai_stocks_forecasting.signals.label_regimes` on the
    last session before ``as_of``, the latest label a midnight origin can know:
    expanding percentiles of 21-session realised volatility, so a label never
    uses data after its own date.
    """
    labels = signals.label_regimes(prices)
    regimes = labels.set_index(pd.to_datetime(labels["timestamp"]).dt.normalize())["regime"].sort_index()
    # A label is known at its session's close, so a midnight origin reads the session before it.
    origins = pd.to_datetime(frame["as_of"]).dt.normalize()
    at_origin = {o: regimes.loc[: o - pd.Timedelta(days=1)].iloc[-1] for o in origins.unique()}
    labelled = frame.assign(regime=origins.map(at_origin))
    table = labelled.pivot_table(index="predictor", columns="regime", values="crps", aggfunc="mean")
    counts = labelled.drop_duplicates("as_of")["regime"].value_counts()
    table = table.reindex(columns=[c for c in ("low", "normal", "high") if c in table.columns])
    table.columns = [f"{c} ({counts.get(c, 0)} origins)" for c in table.columns]
    return table.round(2)


def regime_crps_chart(
    table: pd.DataFrame, title: str = "Where the error comes from: CRPS by volatility regime"
) -> tuple[Figure, Axes]:
    """Dot chart of :func:`regime_crps_table`: one row per regime, one marker per predictor."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    markers = ["o", "s", "D", "^", "v"]
    regimes = list(table.columns)
    for i, (name, row) in enumerate(table.iterrows()):
        color = FAMILY_COLORS.get(predictor_family(str(name)), FAMILY_COLORS["Other"])
        ys = np.arange(len(regimes)) + (i - (len(table) - 1) / 2) * 0.18
        ax.scatter(
            row.values,
            ys,
            marker=markers[i % len(markers)],
            s=80,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
            label=str(name),
        )
    ax.set_yticks(range(len(regimes)), regimes, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean CRPS (USD/share), lower is better", color=INK_SECONDARY, fontsize=10)
    ax.set_title(title, loc="left", fontsize=13, color=INK_PRIMARY, fontweight="600", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.text(
        0.01,
        0.01,
        "* scored on every 2025 session, which is not volatility-matched; see the notes.",
        fontsize=8.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, ax


def describe_shocks(prices: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    """Headline counts for a window: sessions, shock sessions, up and down."""
    close = price_series(prices)
    shocks = shock_days(close, start, end)
    sessions = int(close.loc[start:end].shape[0])
    return {
        "sessions": sessions,
        "shock_sessions": len(shocks),
        "up": int((shocks["return_pct"] > 0).sum()),
        "down": int((shocks["return_pct"] < 0).sum()),
        "share_pct": round(100 * len(shocks) / max(sessions, 1), 1),
    }


def shock_window_crps(
    frame: pd.DataFrame, prices: pd.DataFrame, threshold_pct: float = SHOCK_THRESHOLD
) -> pd.DataFrame:
    """Mean CRPS per predictor and horizon, split by whether a shock fell inside the forecast window.

    A forecast's window runs from its origin session (a midnight origin has
    not seen that session's close) up to and including its target date.  The split shows how much of each predictor's error comes
    from the few windows that contain a ±5% session.
    """
    close = price_series(prices)
    ret = close.pct_change() * 100
    shock_dates = ret.index[ret.abs() >= threshold_pct]

    def _has_shock(as_of: pd.Timestamp, target: pd.Timestamp) -> bool:
        # A midnight origin forecasts from the previous close, so its own session's move is inside the window.
        return bool(((shock_dates >= as_of.normalize()) & (shock_dates <= target)).any())

    flagged = frame.assign(
        window=[
            "shock in window" if _has_shock(pd.Timestamp(a), pd.Timestamp(t)) else "no shock"
            for a, t in zip(frame["as_of"], frame["forecast_date"], strict=True)
        ]
    )
    table = flagged.pivot_table(index=["predictor", "horizon"], columns="window", values="crps", aggfunc="mean")
    share = flagged.groupby("horizon")["window"].apply(lambda s: round(100 * (s == "shock in window").mean()))
    table["windows with a shock (%)"] = table.index.get_level_values("horizon").map(share)
    return table.round(2)


def agent_rationale(predictor_prefix: str, origin: str, spec_id: str = "nvda_backtest") -> str:
    """Return the trajectory agent's rationale at one origin, from the committed predictions."""
    import yaml  # noqa: PLC0415
    from ai_stocks_forecasting.charts import PREDICTIONS_DIR  # noqa: PLC0415

    for path in sorted((PREDICTIONS_DIR / spec_id).glob(f"{predictor_prefix}*.yaml")):
        for pred in yaml.safe_load(path.read_text())["predictions"]:
            if str(pred["as_of"])[:10] == origin and pred["metadata"].get("rationale"):
                return str(pred["metadata"]["rationale"]).strip()
    return ""


def shock_forecast_records(
    day: str, spec_ids: tuple[str, ...] = ("nvda_shock_fresh", "nvda_shock_fresh_close")
) -> pd.DataFrame:
    """Every agent shock forecast for one session date, with its probability and rationale."""
    import yaml  # noqa: PLC0415
    from ai_stocks_forecasting.charts import PREDICTIONS_DIR, _shock_arm  # noqa: PLC0415

    rows = []
    for spec_id in spec_ids:
        for path in sorted((PREDICTIONS_DIR / spec_id).glob("agent_*.yaml")):
            for pred in yaml.safe_load(path.read_text())["predictions"]:
                if str(pred["as_of"])[:10] == day:
                    rows.append(
                        {
                            "spec": spec_id,
                            "arm": _shock_arm(path.stem),
                            "origin": str(pred["as_of"])[:16],
                            "probability": pred["payload"]["probability"],
                            "rationale": str(pred["metadata"].get("rationale", "")).strip(),
                        }
                    )
    return pd.DataFrame(rows)


# ── Discovery loop (notebook 05) ──────────────────────────────────────────────


def _train_passes(t: dict[str, Any] | None) -> bool:
    if not t:
        return False
    return (
        t["n_matches"] >= signals.MIN_MATCHES
        and t["lift"] >= signals.MIN_LIFT
        and t["p_value"] < signals.MAX_P_VALUE
        and t["ci_low"] > 1
    )


def _holdout_passes(h: dict[str, Any]) -> bool:
    lift = h["lift"]
    return not np.isnan(lift) and lift >= signals.MIN_HOLDOUT_LIFT and h["p_value"] < signals.MAX_HOLDOUT_P_VALUE


def failure_group(entry: dict[str, Any]) -> str:
    """Classify why a candidate did not graduate, from its recorded evidence."""
    t, h = entry.get("train"), entry["holdout"]
    if entry.get("graduated"):
        return "graduated"
    if _train_passes(t) and not _holdout_passes(h):
        return "real before the cutoff, not after"
    if _holdout_passes(h) and not _train_passes(t):
        return "real after the cutoff, not before"
    if h["n_matches"] == 1 and h["n_hits"] == 1:
        return "one event"
    return "no effect"


def discovery_results(experiments_dir: Any = None) -> pd.DataFrame:
    """Every candidate's current evidence from the experiment trails, one row each."""
    import yaml  # noqa: PLC0415
    from ai_stocks_forecasting.discovery import EXPERIMENTS_DIR  # noqa: PLC0415

    rows = []
    for path in sorted((experiments_dir or EXPERIMENTS_DIR).glob("*/trail.yaml")):
        trail = yaml.safe_load(path.read_text())
        for c in trail["candidates"]:
            t, h = c.get("train") or {}, c["holdout"]
            rows.append(
                {
                    "experiment": trail["experiment_id"],
                    "candidate": c["pattern_id"].replace(f"{trail['experiment_id']}_", ""),
                    "cue": c["cue"],
                    "direction": c["direction"],
                    "holdout_design": c.get("holdout_design", ""),
                    "train_lift": t.get("lift", np.nan),
                    "train_p": t.get("p_value", np.nan),
                    "holdout_matches": h["n_matches"],
                    "holdout_hits": h["n_hits"],
                    "holdout_lift": h["lift"],
                    "holdout_p": h["p_value"],
                    "why_not": failure_group(c),
                }
            )
    return pd.DataFrame(rows)


SHORT_NAMES: dict[tuple[str, str], str] = {
    ("exp01_earnings", "P-1"): "earnings",
    ("exp01_earnings", "P-3"): "high volatility*",
    ("exp03_hyperscaler_capex", "q2"): "capex caution (down)",
    ("exp07_semis_peers", "q1"): "peer results",
}
"""Readable labels for the candidates worth naming on the chart; the rest are counted in the legend."""


def two_halves_chart(results: pd.DataFrame, title: str = "Both halves of the gate must pass") -> tuple[Figure, Axes]:
    """Plot every candidate by its holdout p-value (x) and training p-value (y), one marker per failure group.

    The pass region is the bottom-left corner: training p < 0.05 and holdout
    p < 0.10.  Candidates screened out on the holdout have no training test;
    they sit on a strip along the top, at "not tested".
    """
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    untested_y = 1.6
    ax.axvspan(
        1e-3, signals.MAX_HOLDOUT_P_VALUE, ymin=0, ymax=0.48, color=FAMILY_COLORS["LLM / Agent"], alpha=0.08, zorder=0
    )
    ax.axvline(signals.MAX_HOLDOUT_P_VALUE, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.axhline(signals.MAX_P_VALUE, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(1.1e-3, 1.2e-3 * 1.4, "graduates here\n(nothing yet)", fontsize=9.5, color=INK_SECONDARY, va="bottom")
    styles = {
        "no effect": ("o", FAMILY_COLORS["Other"]),
        "one event": ("D", FAMILY_COLORS["Numerical ML"]),
        "real before the cutoff, not after": ("s", FAMILY_COLORS["Baseline"]),
        "real after the cutoff, not before": ("^", FAMILY_COLORS["LLM / Agent"]),
        "graduated": ("*", INK_PRIMARY),
    }
    jitter = np.random.default_rng(3)
    for group, (marker, color) in styles.items():
        sub = results[results["why_not"] == group]
        if sub.empty:
            continue
        xs = sub["holdout_p"].clip(lower=1.5e-3).to_numpy() * np.exp(jitter.uniform(-0.08, 0.08, len(sub)))
        ys = sub["train_p"].fillna(untested_y).clip(lower=1.5e-3).to_numpy() * np.exp(
            jitter.uniform(-0.12, 0.12, len(sub))
        )
        ax.scatter(
            xs,
            ys,
            marker=marker,
            s=90,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
            label=f"{group} ({len(sub)})",
        )
        for x, y, (_, row) in zip(xs, ys, sub.iterrows(), strict=True):
            if group != "no effect":
                ax.annotate(
                    f"{row['experiment'][:5]} {row['candidate']}",
                    (x, y),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=INK_SECONDARY,
                )
    ax.set_xlim(1e-3, 2.5)
    ax.set_ylim(1e-3, 3)
    ax.set_yticks(
        [1e-3, 1e-2, signals.MAX_P_VALUE, 0.3, 1, untested_y], ["0.001", "0.01", "0.05", "0.3", "1", "not tested"]
    )
    ax.set_xticks([1e-3, 1e-2, signals.MAX_HOLDOUT_P_VALUE, 0.3, 1], ["0.001", "0.01", "0.10", "0.3", "1"])
    ax.set_xlabel("Holdout p-value (2025, after the model cutoff)  →  weaker", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Training p-value (2020–Jan 2025)  →  weaker", color=INK_SECONDARY, fontsize=10)
    ax.set_title(title, loc="left", fontsize=13.5, color=INK_PRIMARY, fontweight="600", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.text(
        0.01,
        0.01,
        "* scored on every 2025 session, which is not volatility-matched; see the notes.",
        fontsize=8.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, ax
