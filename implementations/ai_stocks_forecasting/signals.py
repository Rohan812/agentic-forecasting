"""Statistical substrate for the NVDA discovery loop — the gate that patterns must clear.

This module is **the contract between the two teams**.  Everything here is
deterministic and LLM-free: the discovery agent proposes candidate news
patterns, and this module decides whether a candidate has earned the right to
influence a forecast.  The agent side imports these five functions and nothing
else from the statistics layer.

The pipeline the functions compose into::

    prices ──flag_shock_windows──> shock windows
                                        │
                                        ├──sample_matched_controls──> + control windows
                                        │
                                        └──train_holdout_split──> (train, holdout)
                                                                        │
    agent says "pattern P matched these windows" ───────────────────────┤
                                                                        v
                                                        evaluate_pattern(train)  ─┐
                                                        evaluate_pattern(holdout) ─┴─> gate_pass
                                                                                          │
                                                                          True ──> written to the
                                                                                   master strategy file

Why a gate at all: a pattern is only evidence if it predicts on data it was
not found on.  Note what the gate does *not* do: both proxy models remember
2024 (see ``LLM_CUTOFFS.md`` at the repository root), and a pattern recalled
from memory scores *well* in-sample, so train-split significance alone rewards
leakage rather than catching it.  The holdout criterion is what protects the
gate, and it only does so when the holdout windows fall after the model
cutoff (February 2025 onward).

Status
------
**Partly implemented.**  :func:`flag_shock_windows` and
:func:`sample_matched_controls` are implemented and tested.  The remaining
functions keep their final signatures and raise :class:`NotImplementedError`
until their own tasks land.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from ai_stocks_forecasting.paths import SHOCK_HORIZON, SHOCK_THRESHOLD


Direction = Literal["up", "down"]
"""Which way a shock went.  Kept explicit because NVDA's shocks are asymmetric."""

Regime = Literal["low", "normal", "high"]
"""Volatility regime bucket, assigned by :func:`label_regimes`."""


# ── Gate thresholds ───────────────────────────────────────────────────────────
# A candidate pattern must clear every one of these to be written to the master
# strategy file.  They are module constants rather than arguments so that every
# graduated pattern in the repository was held to the same standard, and so that
# changing the standard is a visible, reviewable diff.

MIN_LIFT = 2.0
"""Minimum precision-over-base-rate ratio on the training split.

A pattern must at least double the probability of a shock relative to the
unconditional rate.  Anything less is not worth the forecaster's attention even
if it is statistically detectable.
"""

MIN_HOLDOUT_LIFT = 1.5
"""Minimum lift on the held-out split.

Deliberately lower than :data:`MIN_LIFT`.  Some shrinkage between train and
holdout is expected and honest; what the gate rejects is a pattern that only
existed in the data it was discovered on.
"""

MAX_P_VALUE = 0.05
"""Maximum Fisher's exact p-value on the training split."""

MIN_MATCHES = 5
"""Minimum number of windows a pattern must match to be assessable.

With fewer matches the confidence interval is so wide that a high precision
carries no information.  Event counts drive the choice of shock threshold for
the same reason.  At ±5% the clean post-cutoff window (Feb-Dec 2025) holds 13
shock events.  At ±7% it held 4, too few for any holdout criterion to mean
anything.
"""


# ── Window construction ───────────────────────────────────────────────────────
# How shocks are grouped into independent events and how their controls are
# drawn.  Module constants for the same reason as the gate thresholds: every
# committed result was built the same way, and changing the rules is a diff.

SHOCK_CLUSTER_GAP_DAYS = 1
"""Shocks this many trading days apart or closer are one event, not several.

The default merges consecutive trading days.  Measured on 2020-2024 at ±5%, it
turns 151 shock days into 117 events.  23 of them are multi-day clusters; the
longest is the COVID crash, nine consecutive shock sessions from 2020-03-09 to
2020-03-19, which counts once.  Wider gaps merge more aggressively:

====  ======
gap   events
====  ======
0     151
1     117
2     93
3     74
5     53
10    29
====  ======

Wider is more conservative about independence, but leaves the gate fewer events.
At a 5-day gap, for example, the 2025-04-03/04 tariff crash would absorb the
separate +18.7% relief rally on 2025-04-09.
"""

VOL_WINDOW_DAYS = 21
"""Trailing-volatility window, in sessions, used to match controls to shocks.

The same definition as :mod:`ai_stocks_forecasting.shock_anchors`: the standard
deviation of the 21 daily returns *strictly before* the session, so it is
something the agent could know at the origin.
"""

CONTROL_WINDOW_DAYS = 126
"""Controls are drawn within this many trading days (about six months) of their shock.

This keeps each control in the same market era as its shock, the "seasonal"
half of the matching.  It is wider than a quarter because at ±5% shocks are
dense enough in volatile stretches that a quarter sometimes holds too few
eligible, volatility-matched sessions: 2 of 117 shocks came up short at 63.
"""

CONTROL_EXCLUSION_DAYS = 2
"""No control within this many trading days of any shock, or of a window's span.

The buffer keeps controls out of a shock's immediate news cycle.  It is
deliberately small because shocks are common at ±5%, 12% of sessions.  A
one-week buffer (5) excluded almost every session in volatile periods.  Only
calm sessions were left to be controls, the volatility match collapsed
(control median trailing vol 2.64% against the shocks' 3.53%, no better than
random), and 41 of 117 shocks got no control.  At 2 every shock gets its
control and the match holds (3.51% against 3.53%).

The two errors are not symmetric.  A control near a shock's news occasionally
matches a pattern, which lowers measured lift: the gate gets more
*conservative*.  Calm controls inflate the lift of anything that tracks
volatility: the gate gets more *permissive*.  A small buffer is the safer side
to err on.
"""

CONTROL_POOL_MULTIPLE = 5
"""Each shock's controls are sampled from its ``n_each * 5`` best-matched candidates.

Taking only the nearest matches would make the draw deterministic and could
re-use the same handful of days.  Sampling from a small pool keeps the match
tight while leaving the seed meaningful.
"""


@dataclass(frozen=True)
class Window:
    """One labelled observation window — either a shock or a matched control.

    Windows are the unit the whole discovery loop operates on.  A candidate
    pattern is evaluated by asking, for each window, "did this pattern's news
    signature appear in the lead-up to this window?", then comparing the answers
    against :attr:`is_shock`.

    Attributes
    ----------
    as_of
        Information cutoff for the window: the last day whose news the agent may
        look at.  Everything the agent is allowed to see ends here.
    event_date
        The day the move happened (for a shock), or the day the control window
        is anchored on.  Always strictly after :attr:`as_of`.
    is_shock
        ``True`` for a flagged shock, ``False`` for a matched control.  This is
        the label the gate scores patterns against.
    direction
        Sign of the move, for shocks only.  ``None`` for controls.  Present
        because NVDA's shocks skew upward (87 up vs 64 down shock days at ±5%
        over 2020-2024), so pooling directions inflates the apparent base rate
        for a pattern that only predicts one of them.
    return_pct
        The realised move over the window, in percent, signed.
    regime
        Volatility regime on :attr:`as_of`, from :func:`label_regimes`.  ``None``
        until the regime labeller is wired in.  The gate does not use it, but
        per-regime precision splits are how the evaluation agent later decides
        whether a pattern is degrading.
    """

    as_of: pd.Timestamp
    event_date: pd.Timestamp
    is_shock: bool
    return_pct: float
    direction: Direction | None = None
    regime: Regime | None = None


@dataclass(frozen=True)
class PatternMetrics:
    """Scored evidence for one candidate pattern on one split.

    Attributes
    ----------
    n_windows
        Total windows the pattern was evaluated over.
    n_matches
        Windows where the pattern's signature was present.
    n_hits
        Windows where the pattern matched **and** a shock occurred.
    precision
        ``n_hits / n_matches`` — P(shock | pattern matched).  ``nan`` when the
        pattern matched nothing.
    base_rate
        P(shock) over the same windows, the thing precision has to beat.
    lift
        ``precision / base_rate``.  The headline number.
    p_value
        Fisher's exact test on the 2x2 table of matched/unmatched against
        shock/no-shock.  One-sided, testing precision above base rate.
    ci_low, ci_high
        Bootstrap confidence interval on :attr:`lift`.  The gate requires this
        interval to exclude 1.0, which is a stricter and more informative
        statement than the p-value alone.
    """

    n_windows: int
    n_matches: int
    n_hits: int
    precision: float
    base_rate: float
    lift: float
    p_value: float
    ci_low: float
    ci_high: float


def _close_series(prices_df: pd.DataFrame) -> pd.Series:
    """Return the price history as a float series indexed by trading day."""
    series = prices_df.set_index(pd.to_datetime(prices_df["timestamp"]))["value"].astype(float)
    return series.sort_index()


def flag_shock_windows(
    prices_df: pd.DataFrame,
    threshold_pct: float = SHOCK_THRESHOLD,
    horizon_days: int = SHOCK_HORIZON,
) -> list[Window]:
    """Find every independent shock event in a price history.

    A session is a shock when its simple return over ``horizon_days`` reaches
    ``threshold_pct`` in absolute value.  Simple returns in percent match
    :data:`~ai_stocks_forecasting.paths.SHOCK_THRESHOLD` and the calibration in
    :mod:`ai_stocks_forecasting.shock_anchors`, so "shock" means the same thing
    in the gate as in the shock task's prompt anchors.  Log returns would shift
    the threshold asymmetrically and produce a different event set.

    Parameters
    ----------
    prices_df
        Price history with ``timestamp`` and ``value`` columns, as returned by
        :meth:`~aieng.forecasting.data.service.DataService.get_series`.
    threshold_pct
        Absolute move, in percent, that counts as a shock.  Defaults to
        :data:`~ai_stocks_forecasting.paths.SHOCK_THRESHOLD` (5.0).  Applied in
        **both directions**.
    horizon_days
        Trading-day span the move is measured over.  Defaults to
        :data:`~ai_stocks_forecasting.paths.SHOCK_HORIZON` (1).

    Returns
    -------
    list[Window]
        One window per event, with ``is_shock=True`` and a populated
        ``direction``, ordered by ``event_date``.

    Notes
    -----
    *Clusters are one event.*  Shocks within :data:`SHOCK_CLUSTER_GAP_DAYS`
    trading days of each other are merged, which is also what stops overlapping
    multi-day windows being double-counted when ``horizon_days > 1``.  Counting a
    four-day crash as four independent events would let a single news pattern
    score four hits for one piece of news.

    *A merged window is anchored on its largest move* (``event_date``,
    ``return_pct`` and ``direction`` come from it).  Its ``as_of`` is the session
    before the cluster's *first* shock, not before the largest one.  Otherwise
    the agent's news window would include coverage of a crash that had already
    started.
    """
    close = _close_series(prices_df)
    dates = close.index
    returns = (close.pct_change(horizon_days) * 100.0).to_numpy()

    shock_pos = np.flatnonzero(np.abs(np.nan_to_num(returns)) >= threshold_pct)
    if shock_pos.size == 0:
        return []

    # Overlapping multi-day windows (gap < horizon_days) are always one event.
    max_gap = SHOCK_CLUSTER_GAP_DAYS + horizon_days - 1
    clusters = np.split(shock_pos, np.flatnonzero(np.diff(shock_pos) > max_gap) + 1)

    windows: list[Window] = []
    for cluster in clusters:
        anchor = int(cluster[np.argmax(np.abs(returns[cluster]))])
        move = float(returns[anchor])
        windows.append(
            Window(
                as_of=dates[int(cluster[0]) - horizon_days],
                event_date=dates[anchor],
                is_shock=True,
                return_pct=move,
                direction="up" if move > 0 else "down",
            )
        )
    return windows


def sample_matched_controls(
    windows: list[Window],
    prices_df: pd.DataFrame,
    n_each: int = 1,
    seed: int | None = None,
    *,
    threshold_pct: float = SHOCK_THRESHOLD,
) -> list[Window]:
    """Draw non-shock control sessions matched to the given shock windows.

    Without controls, precision cannot be compared against anything: a pattern
    that matches every window whatsoever would look perfect on shocks alone.

    Parameters
    ----------
    windows
        Shock windows from :func:`flag_shock_windows`.
    prices_df
        The same price history those windows were derived from.
    n_each
        Controls to draw per shock window.  Raising this tightens the base-rate
        estimate at the cost of more news retrieval per experiment, which is the
        dominant cost in a study session.
    seed
        Seed for the sampler.  Pass one whenever a result will be committed —
        a graduated pattern's evidence has to be reproducible.
    threshold_pct
        The shock threshold the windows were flagged with.  Any session at or
        above it is excluded as a control, together with its neighbours.  Pass
        the same value given to :func:`flag_shock_windows`.

    Returns
    -------
    list[Window]
        One-session control windows with ``is_shock=False`` and
        ``direction=None``, ordered by ``event_date``.  No session is used twice.

    Raises
    ------
    ValueError
        If ``n_each < 1``, or if ``windows`` contains a non-shock window.

    Notes
    -----
    Each shock's controls come from sessions that are:

    1. **Not near any shock.**  Every session whose return reaches
       ``threshold_pct``, and every window's ``as_of``-to-``event_date`` span,
       is excluded along with :data:`CONTROL_EXCLUSION_DAYS` sessions either side.
    2. **In the same era.**  Within :data:`CONTROL_WINDOW_DAYS` of the shock.
    3. **In the same volatility regime.**  Candidates are ranked by how close
       their trailing volatility is to the shock's, measured over the
       :data:`VOL_WINDOW_DAYS` returns before the episode began.  The controls
       are then sampled from the best :data:`CONTROL_POOL_MULTIPLE` ``* n_each``.

    The volatility match is the one that matters most.  Shocks cluster in
    volatile markets, so controls drawn at random come mostly from calm ones.
    Any pattern that merely tracks volatility, such as "analysts expect
    turbulence", would then look predictive of shocks when it only predicts
    the regime they happen in.

    Shocks are visited in a seeded random order, so no period systematically
    gets first pick of the candidates.  If a shock has fewer eligible candidates
    than ``n_each``, it gets what is available and a warning is raised, rather
    than silently drawing unmatched controls.
    """
    if n_each < 1:
        raise ValueError(f"n_each must be at least 1, got {n_each}.")
    if any(not w.is_shock for w in windows):
        raise ValueError("sample_matched_controls expects shock windows only; got a window with is_shock=False.")

    close = _close_series(prices_df)
    dates = close.index
    n = len(dates)
    position = {d: i for i, d in enumerate(dates)}
    returns = close.pct_change() * 100.0
    vol_before = returns.rolling(VOL_WINDOW_DAYS).std().shift(1).to_numpy()
    ret = returns.to_numpy()

    excluded = np.isnan(vol_before)  # also covers the first session, which has no prior day
    for p in np.flatnonzero(np.abs(np.nan_to_num(ret)) >= threshold_pct):
        excluded[max(0, p - CONTROL_EXCLUSION_DAYS) : p + CONTROL_EXCLUSION_DAYS + 1] = True
    for w in windows:
        start, end = position[w.as_of], position[w.event_date]
        excluded[max(0, start - CONTROL_EXCLUSION_DAYS) : end + CONTROL_EXCLUSION_DAYS + 1] = True

    rng = np.random.default_rng(seed)
    used = np.zeros(n, dtype=bool)
    chosen_all: list[int] = []
    for i in rng.permutation(len(windows)):
        w = windows[i]
        event = position[w.event_date]
        candidates = np.arange(max(0, event - CONTROL_WINDOW_DAYS), min(n, event + CONTROL_WINDOW_DAYS + 1))
        candidates = candidates[~excluded[candidates] & ~used[candidates]]
        if candidates.size < n_each:
            warnings.warn(
                f"Only {candidates.size} eligible control(s) near the shock on {w.event_date.date()}; wanted {n_each}.",
                stacklevel=2,
            )
        if candidates.size == 0:
            continue

        # Volatility known at the close before the episode's first session.
        target_vol = vol_before[position[w.as_of] + 1]
        if np.isnan(target_vol):
            pool = candidates
        else:
            order = np.argsort(np.abs(vol_before[candidates] - target_vol), kind="stable")
            pool = candidates[order[: CONTROL_POOL_MULTIPLE * n_each]]
        picked = rng.choice(pool, size=min(n_each, pool.size), replace=False)
        used[picked] = True
        chosen_all.extend(int(c) for c in picked)

    return [
        Window(as_of=dates[c - 1], event_date=dates[c], is_shock=False, return_pct=float(ret[c]))
        for c in sorted(chosen_all)
    ]


def train_holdout_split(
    windows: list[Window],
    holdout_fraction: float = 0.5,
) -> tuple[list[Window], list[Window]]:
    """Split windows into a discovery set and a validation set.

    Parameters
    ----------
    windows
        Shock and control windows combined.
    holdout_fraction
        Share of windows reserved for validation.

    Returns
    -------
    tuple[list[Window], list[Window]]
        ``(train, holdout)``.

    Notes
    -----
    The split must be **chronological, not random**.  A random split lets a
    pattern discovered from one half of an earnings cycle validate on the other
    half of the same cycle, which measures memorisation rather than
    generalisation. Splitting on time is the only version that answers the
    question the gate is actually asking.

    Both halves need enough shocks to be assessable — see :data:`MIN_MATCHES`.
    At ±5% there are 117 shock events in 2020-2024 and 13 in Feb-Dec 2025, the
    clean window after the model cutoff.  ``LLM_CUTOFFS.md`` explains why a
    holdout inside the remembered period cannot catch memorised patterns.  The
    implementation should refuse, loudly, rather than return a holdout too
    small to validate anything.
    """
    raise NotImplementedError("Phase 1, Team Signals T2")


def evaluate_pattern(
    pattern_matches: list[bool],
    shock_labels: list[bool],
    base_rate: float | None = None,
) -> PatternMetrics:
    """Score one candidate pattern against labelled windows.

    Parameters
    ----------
    pattern_matches
        Per window, whether the pattern's news signature was present.
    shock_labels
        Per window, whether a shock occurred.  Same length and order as
        ``pattern_matches``.
    base_rate
        P(shock) to compare precision against.  When ``None``, computed from
        ``shock_labels``.  Pass an explicit value to score a directional pattern
        against a directional base rate — pooling up and down shocks flatters a
        pattern that only predicts one direction, and NVDA's shocks are
        asymmetric enough for this to matter.

    Returns
    -------
    PatternMetrics
        Precision, lift, Fisher's exact p-value, and a bootstrap CI on lift.

    Notes
    -----
    Fisher's exact rather than a chi-squared test because the cell counts are
    small by construction: a pattern matching 8 windows of which 5 are shocks is
    a typical case, and chi-squared is unreliable there.

    The implementation must handle ``n_matches == 0`` (return ``nan`` precision
    and lift, ``p_value = 1.0``) rather than dividing by zero — the discovery
    agent will propose patterns that match nothing, and that is a normal,
    informative outcome to record in the per-experiment file.
    """
    raise NotImplementedError("Phase 1, Team Signals T3")


def gate_pass(metrics: PatternMetrics, holdout_metrics: PatternMetrics) -> bool:
    """Decide whether a candidate pattern graduates to the master strategy file.

    This is the single function that separates a hypothesis from a finding.
    Everything upstream is measurement; this is the judgement.

    A pattern graduates only if **all** of the following hold:

    - ``metrics.n_matches >=`` :data:`MIN_MATCHES`
    - ``metrics.lift >=`` :data:`MIN_LIFT`
    - ``metrics.p_value <`` :data:`MAX_P_VALUE`
    - ``metrics.ci_low > 1.0`` — the confidence interval excludes "no effect"
    - ``holdout_metrics.lift >=`` :data:`MIN_HOLDOUT_LIFT`

    Parameters
    ----------
    metrics
        Scored on the training split.
    holdout_metrics
        Scored on the held-out split, same pattern.

    Returns
    -------
    bool
        ``True`` if every criterion is met.

    Notes
    -----
    The conjunction is the point.  Each criterion alone is gameable: lift is
    high on tiny samples, p-values fall as windows are added, and a training
    result says nothing about a later period.  Requiring all five together is
    what makes graduation mean something.

    Callers should record the **reason for rejection**, not just the boolean,
    into the per-experiment file.  "Rejected: lift 2.4 but holdout lift 0.9"
    tells the next study session something; ``False`` does not.
    """
    raise NotImplementedError("Phase 1, Team Signals T4")


def label_regimes(
    prices_df: pd.DataFrame,
    window_days: int = 21,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> pd.DataFrame:
    """Attach a volatility regime label to each day of a price series.

    Parameters
    ----------
    prices_df
        Price history with ``timestamp`` and ``value`` columns.
    window_days
        Rolling window for realised volatility.
    low_quantile, high_quantile
        Cut points, as quantiles of the realised-volatility distribution,
        separating ``"low"`` / ``"normal"`` / ``"high"``.

    Returns
    -------
    pd.DataFrame
        ``prices_df`` plus ``realized_vol`` and ``regime`` columns.

    Notes
    -----
    Quantiles must be computed on an **expanding** basis, not over the whole
    series: bucketing today against the full-sample distribution uses future
    volatility to label the past, which leaks into anything scored per regime.

    This is realised volatility from the price series itself, not VIX. A VIX
    covariate is a later addition and would be registered as its own series
    rather than derived here.

    Not used by :func:`gate_pass`.  It exists because per-regime precision is
    how a degrading pattern gets caught: a cue that worked in a calm market and
    stopped working in a volatile one shows up as a regime split long before it
    shows up in pooled precision.
    """
    raise NotImplementedError("Phase 1, Team Signals T5")


__all__ = [
    "CONTROL_EXCLUSION_DAYS",
    "CONTROL_POOL_MULTIPLE",
    "CONTROL_WINDOW_DAYS",
    "MAX_P_VALUE",
    "MIN_HOLDOUT_LIFT",
    "MIN_LIFT",
    "MIN_MATCHES",
    "SHOCK_CLUSTER_GAP_DAYS",
    "VOL_WINDOW_DAYS",
    "Direction",
    "PatternMetrics",
    "Regime",
    "Window",
    "evaluate_pattern",
    "flag_shock_windows",
    "gate_pass",
    "label_regimes",
    "sample_matched_controls",
    "train_holdout_split",
]
