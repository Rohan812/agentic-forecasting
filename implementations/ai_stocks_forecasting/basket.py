"""The four-stock AI basket: volatility-relative shocks, pooled windows, calendar rules and a baseline model.

Pre-registered in ``experiments/basket/preregistration.md``.  One stock and one
year give too few post-cutoff shocks to confirm rare patterns, so the study
pools NVDA, AMD, MSFT and GOOG.  Three things differ from the single-stock
engine in :mod:`~ai_stocks_forecasting.discovery`:

- **Shocks are relative.**  A session is a shock when its return is at least
  :data:`SHOCK_SIGMAS` times the standard deviation of that ticker's previous
  :data:`VOL_WINDOW` returns, so a 3% move in MSFT and a 6% move in NVDA count
  alike.  Each ticker's returns are standardised and passed through the tested
  ``signals`` functions with a threshold of :data:`SHOCK_SIGMAS`.
- **Windows are pooled.**  Each carries its ticker.
- **Dependence is checked.**  Pooled ticker-days are not independent (a Fed
  day hits every ticker), so :func:`calendar_shift_p` tests each calendar rule
  against the same calendar shifted in time.

``end`` defaults to the end of the 2025 holdout everywhere: stage 0 must not
read the 2026 outcomes of AMD, MSFT and GOOG, which are reserved for a blind
test.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from ai_stocks_forecasting import signals
from ai_stocks_forecasting.paths import DATA_DIR
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter


BASKET: tuple[str, ...] = ("NVDA", "AMD", "MSFT", "GOOG")
HISTORY_START = "2018-06-01"
STUDY_START = "2020-01-01"
STAGE0_END = "2025-12-31"
"""Last date stage 0 may read: the end of the 2025 holdout."""

SHOCK_SIGMAS = 2.0
VOL_WINDOW = 21
SEED = 0
HOLDOUT_CONTROLS_PER_SHOCK = 3
SHIFT_RESAMPLES = 2000
SHIFT_RANGE = (5, 120)

CALENDARS_DIR = Path(__file__).parent / "calendars"
EventKind = Literal["earnings", "fomc", "cpi", "jobs"]
EVENT_KINDS: tuple[EventKind, ...] = ("earnings", "fomc", "cpi", "jobs")


def load_basket_prices(cache_dir: Path | None = None, end: str = STAGE0_END) -> dict[str, pd.Series]:
    """Load the adjusted close per ticker from the yfinance cache, truncated at ``end``."""
    cache = cache_dir or DATA_DIR / "yfinance"
    out = {}
    for ticker in BASKET:
        df = YFinanceDailyAdapter(ticker=ticker, start=HISTORY_START, cache_dir=cache).fetch()
        close = df.set_index(pd.to_datetime(df["timestamp"]).dt.normalize())["value"].sort_index()
        out[ticker] = close.loc[:end]
    return out


def standardised_returns(close: pd.Series) -> pd.Series:
    """Return each session's return divided by the std of the previous ``VOL_WINDOW`` returns."""
    ret = close.pct_change()
    vol = ret.rolling(VOL_WINDOW).std().shift(1)
    return (ret / vol).dropna()


def _standardised_prices(close: pd.Series) -> pd.DataFrame:
    """Build a synthetic price frame whose percent returns are the standardised returns.

    ``signals`` flags shocks by a percent threshold; on this frame a threshold
    of ``SHOCK_SIGMAS`` percent flags exactly the sessions whose real return
    reached ``SHOCK_SIGMAS`` trailing standard deviations.
    """
    z = standardised_returns(close)
    value = 100 * np.cumprod(1 + z.clip(lower=-90) / 100)
    return pd.DataFrame({"timestamp": z.index, "value": value.to_numpy()})


@dataclass(frozen=True)
class BasketWindow:
    """A study window and the ticker it belongs to."""

    ticker: str
    window: signals.Window

    @property
    def is_shock(self) -> bool:
        """True for a shock window, False for a matched control."""
        return self.window.is_shock


class Calendar:
    """Scheduled reaction sessions per ticker: the session whose move reflects the event."""

    def __init__(self, sessions: dict[str, pd.DatetimeIndex], calendars_dir: Path = CALENDARS_DIR) -> None:
        self._sessions = sessions
        earnings = yaml.safe_load((calendars_dir / "earnings.yaml").read_text())
        macro = yaml.safe_load((calendars_dir / "macro.yaml").read_text())
        self._announcements = {t: pd.to_datetime(earnings[t]) for t in sessions}
        self._macro = {k: pd.to_datetime(macro[k]) for k in ("fomc", "cpi", "jobs")}

    def reaction_sessions(self, kind: EventKind, ticker: str, shift: int = 0) -> pd.DatetimeIndex:
        """Sessions on which ``kind`` is reflected for ``ticker``, optionally shifted by ``shift`` sessions.

        Earnings are announced after the close, so the reaction is the next
        session.  A macro release is reflected on its own day, or on the next
        session if that day is not one (the March 2020 Sunday Fed cut).
        """
        idx = self._sessions[ticker]
        if kind == "earnings":
            pos = idx.searchsorted(self._announcements[ticker], side="right")
        else:
            pos = idx.searchsorted(self._macro[kind], side="left")
        pos = pos + shift
        return idx[pos[(pos >= 0) & (pos < len(idx))]]


@dataclass
class BasketStudy:
    """Pooled train and holdout windows for the basket, with what calendar rules need."""

    close: dict[str, pd.Series]
    train: list[BasketWindow]
    holdout: list[BasketWindow]

    @cached_property
    def sessions(self) -> dict[str, pd.DatetimeIndex]:
        """Session index per ticker."""
        return {t: c.index for t, c in self.close.items()}

    @cached_property
    def calendar(self) -> Calendar:
        """The scheduled-event calendar aligned to each ticker's sessions."""
        return Calendar(self.sessions)

    @cached_property
    def shock_table(self) -> pd.DataFrame:
        """One row per ticker-session from ``STUDY_START``: standardised return and shock flag."""
        rows = []
        for ticker, close in self.close.items():
            z = standardised_returns(close).loc[STUDY_START:]
            rows.append(pd.DataFrame({"ticker": ticker, "z": z, "shock": z.abs() >= SHOCK_SIGMAS}))
        return pd.concat(rows).rename_axis("date").reset_index()

    def base_rate(self, split: Literal["train", "holdout"]) -> float:
        """Pooled share of ticker-sessions in the split's period that were shocks."""
        t = self.shock_table
        start, end = (
            (pd.Timestamp(STUDY_START), pd.Timestamp(signals.HOLDOUT_START) - pd.Timedelta(days=1))
            if split == "train"
            else (pd.Timestamp(signals.HOLDOUT_START), pd.Timestamp(signals.HOLDOUT_END))
        )
        return float(t.loc[(t["date"] >= start) & (t["date"] <= end), "shock"].mean())

    def next_session(self, bw: BasketWindow) -> pd.Timestamp | None:
        """Return the session after a window's ``as_of``: the first session its shock could fall on."""
        idx = self.sessions[bw.ticker]
        pos = idx.searchsorted(bw.window.as_of.normalize(), side="right")
        return idx[pos] if pos < len(idx) else None


def build_basket_study(close: dict[str, pd.Series] | None = None, seed: int = SEED) -> BasketStudy:
    """Flag relative shocks per ticker, sample matched controls, split by date and pool."""
    close = close if close is not None else load_basket_prices()
    holdout_start, holdout_end = pd.Timestamp(signals.HOLDOUT_START), pd.Timestamp(signals.HOLDOUT_END)
    train: list[BasketWindow] = []
    holdout: list[BasketWindow] = []
    for ticker, series in close.items():
        prices = _standardised_prices(series)
        prices = prices[prices["timestamp"] >= pd.Timestamp(STUDY_START)].reset_index(drop=True)
        shocks = signals.flag_shock_windows(prices, threshold_pct=SHOCK_SIGMAS)
        controls = signals.sample_matched_controls(shocks, prices, n_each=1, seed=seed, threshold_pct=SHOCK_SIGMAS)
        tr, ho = signals.train_holdout_split(shocks + controls)
        ho_shocks = [w for w in ho if w.is_shock]
        extra = signals.sample_matched_controls(
            ho_shocks, prices, n_each=HOLDOUT_CONTROLS_PER_SHOCK, seed=seed, threshold_pct=SHOCK_SIGMAS
        )
        ho = ho_shocks + [w for w in extra if w.as_of >= holdout_start and w.event_date <= holdout_end]
        train += [BasketWindow(ticker, w) for w in tr]
        holdout += [BasketWindow(ticker, w) for w in ho]
    return BasketStudy(close=close, train=train, holdout=holdout)


def rule_matches(study: BasketStudy, windows: list[BasketWindow], kind: EventKind, shift: int = 0) -> list[bool]:
    """Flag each window whose next session is a (shifted) reaction session of ``kind``."""
    reaction = {t: set(study.calendar.reaction_sessions(kind, t, shift)) for t in study.close}
    return [study.next_session(bw) in reaction[bw.ticker] for bw in windows]


def calendar_shift_p(study: BasketStudy, windows: list[BasketWindow], kind: EventKind, seed: int = SEED) -> float:
    """Share of time-shifted calendars that match at least as many shocks as the real one.

    Every ticker's dates move by the same number of sessions, so the placebo
    calendar keeps the real one's spacing and its cross-ticker coincidences.
    A small value means the real dates line up with shocks in a way arbitrary
    dates with the same structure do not.
    """
    shocks = [bw.is_shock for bw in windows]

    def hits(shift: int) -> int:
        return sum(m and s for m, s in zip(rule_matches(study, windows, kind, shift), shocks, strict=True))

    observed = hits(0)
    rng = np.random.default_rng(seed)
    low, high = SHIFT_RANGE
    shifts = rng.integers(low, high + 1, SHIFT_RESAMPLES) * rng.choice([-1, 1], SHIFT_RESAMPLES)
    cache: dict[int, int] = {}
    count = 0
    for raw in shifts:
        shift = int(raw)
        if shift not in cache:
            cache[shift] = hits(shift)
        count += cache[shift] >= observed
    return (count + 1) / (SHIFT_RESAMPLES + 1)


@dataclass(frozen=True)
class BasketResult:
    """One calendar rule's evidence on the pooled windows and its verdict."""

    kind: str
    train: signals.PatternMetrics
    holdout: signals.PatternMetrics
    train_shift_p: float
    holdout_shift_p: float
    reasons: list[str]

    @property
    def graduates(self) -> bool:
        """True when the strict gate and the dependence check both pass."""
        return not self.reasons


def evaluate_rule(study: BasketStudy, kind: EventKind) -> BasketResult:
    """Score a calendar rule on the pooled windows: the strict gate plus the calendar-shift test."""
    metrics, shift_p = {}, {}
    for split, windows in (("train", study.train), ("holdout", study.holdout)):
        matches = rule_matches(study, windows, kind)
        metrics[split] = signals.evaluate_pattern(matches, [bw.is_shock for bw in windows], study.base_rate(split))  # type: ignore[arg-type]
        shift_p[split] = calendar_shift_p(study, windows, kind)
    reasons = signals.gate_reasons(metrics["train"], metrics["holdout"])
    if not shift_p["train"] < signals.MAX_P_VALUE:
        reasons.append(f"training calendar-shift p {shift_p['train']:.3g} is not below {signals.MAX_P_VALUE}")
    if not shift_p["holdout"] < signals.MAX_HOLDOUT_P_VALUE:
        reasons.append(f"holdout calendar-shift p {shift_p['holdout']:.3g} is not below {signals.MAX_HOLDOUT_P_VALUE}")
    return BasketResult(kind, metrics["train"], metrics["holdout"], shift_p["train"], shift_p["holdout"], reasons)


# ── Baseline model: calendar and volatility (pre-registered idea 1) ───────────

FEATURES: tuple[str, ...] = ("vol_ratio", "abs_z", "earnings_next", "fomc_next", "cpi_next", "jobs_next")


def feature_table(study: BasketStudy) -> pd.DataFrame:
    """One row per ticker-session: features known at its close, and whether the next session was a shock.

    ``vol_ratio`` is the 5-session over 63-session realised volatility,
    ``abs_z`` the absolute standardised return of the session itself, and the
    four flags say whether the **next** session is a scheduled reaction session.
    """
    rows = []
    for ticker, close in study.close.items():
        ret = close.pct_change()
        z = standardised_returns(close)
        frame = pd.DataFrame(
            {
                "ticker": ticker,
                "vol_ratio": (ret.rolling(5).std() / ret.rolling(63).std()),
                "abs_z": z.abs(),
                "shock_next": (z.abs() >= SHOCK_SIGMAS).shift(-1),
            }
        )
        idx = close.index
        for kind in EVENT_KINDS:
            reaction = study.calendar.reaction_sessions(kind, ticker)
            frame[f"{kind}_next"] = pd.Series(idx.isin(reaction), index=idx).shift(-1)
        rows.append(frame.dropna().loc[STUDY_START:])
    table = pd.concat(rows).rename_axis("date").reset_index()
    table["shock_next"] = table["shock_next"].astype(int)
    for kind in EVENT_KINDS:
        table[f"{kind}_next"] = table[f"{kind}_next"].astype(int)
    return table


@dataclass(frozen=True)
class BaselineScore:
    """The baseline model against climatology on the holdout ticker-days."""

    n: int
    shocks: int
    brier_model: float
    brier_climatology: float
    skill: float
    diff_low: float
    diff_high: float
    coefficients: dict[str, float]

    @property
    def beats_climatology(self) -> bool:
        """Pre-registered rule: skill above zero and the interval entirely below zero."""
        return self.skill > 0 and self.diff_high < 0


def fit_baseline(table: pd.DataFrame):  # noqa: ANN201
    """Fit the logistic baseline on every training ticker-day (scikit-learn defaults)."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    train = table[table["date"] < pd.Timestamp(signals.HOLDOUT_START)]
    return LogisticRegression(max_iter=1000).fit(train[list(FEATURES)], train["shock_next"])


def score_baseline(table: pd.DataFrame, seed: int = SEED, resamples: int = 2000) -> BaselineScore:
    """Score the frozen baseline on the 2025 holdout, with an interval resampled by date."""
    model = fit_baseline(table)
    train = table[table["date"] < pd.Timestamp(signals.HOLDOUT_START)]
    hold = table[
        (table["date"] >= pd.Timestamp(signals.HOLDOUT_START)) & (table["date"] <= pd.Timestamp(signals.HOLDOUT_END))
    ].copy()
    hold["p"] = model.predict_proba(hold[list(FEATURES)])[:, 1]
    climatology = float(train["shock_next"].mean())
    hold["diff"] = (hold["p"] - hold["shock_next"]) ** 2 - (climatology - hold["shock_next"]) ** 2
    by_date = hold.groupby("date")["diff"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(by_date), size=(resamples, len(by_date)))
    sums, sizes = by_date["sum"].to_numpy()[picks].sum(axis=1), by_date["size"].to_numpy()[picks].sum(axis=1)
    low, high = np.quantile(sums / sizes, [0.05, 0.95])
    brier_model = float(((hold["p"] - hold["shock_next"]) ** 2).mean())
    brier_clim = float(((climatology - hold["shock_next"]) ** 2).mean())
    return BaselineScore(
        n=len(hold),
        shocks=int(hold["shock_next"].sum()),
        brier_model=brier_model,
        brier_climatology=brier_clim,
        skill=1 - brier_model / brier_clim,
        diff_low=float(low),
        diff_high=float(high),
        coefficients=dict(zip(FEATURES, (round(float(c), 3) for c in model.coef_[0]), strict=True)),
    )


# ── Stage 0 report ────────────────────────────────────────────────────────────

RESULTS_PATH = Path(__file__).parent / "experiments" / "basket" / "stage0_results.yaml"


def _metrics(m: signals.PatternMetrics) -> dict[str, float | int]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.__dict__.items()}


def run_stage0(path: Path = RESULTS_PATH) -> dict:
    """Run the pre-registered stage 0 (no LLM calls) and write its results file."""
    study = build_basket_study()
    report: dict = {
        "shock_definition": f"|return| >= {SHOCK_SIGMAS} x std of the previous {VOL_WINDOW} returns",
        "data_through": STAGE0_END,
        "windows": {
            split: {
                "windows": len(ws),
                "shock_events": sum(w.is_shock for w in ws),
                "by_ticker": {t: sum(1 for w in ws if w.ticker == t and w.is_shock) for t in BASKET},
                "base_rate": round(study.base_rate(split), 4),  # type: ignore[arg-type]
            }
            for split, ws in (("train", study.train), ("holdout", study.holdout))
        },
        "rules": [],
    }
    for kind in EVENT_KINDS:
        r = evaluate_rule(study, kind)
        report["rules"].append(
            {
                "rule": kind,
                "graduated": r.graduates,
                "rejection_reasons": r.reasons,
                "train": {**_metrics(r.train), "calendar_shift_p": round(r.train_shift_p, 4)},
                "holdout": {**_metrics(r.holdout), "calendar_shift_p": round(r.holdout_shift_p, 4)},
            }
        )
    score = score_baseline(feature_table(study))
    report["baseline_model"] = {
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in score.__dict__.items()},
        "beats_climatology": score.beats_climatology,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(report, sort_keys=False))
    return report


if __name__ == "__main__":
    out = run_stage0()
    for rule in out["rules"]:
        print(
            f"{rule['rule']:9s} {'GRADUATES' if rule['graduated'] else 'fails: ' + '; '.join(rule['rejection_reasons'])[:110]}"
        )
    print("baseline model:", out["baseline_model"])
