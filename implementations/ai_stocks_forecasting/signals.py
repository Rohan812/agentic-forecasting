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
**Partly implemented.**  :func:`flag_shock_windows`,
:func:`sample_matched_controls`, :func:`train_holdout_split` and
:func:`evaluate_pattern` are implemented and tested.  The remaining
functions keep their final signatures and raise :class:`NotImplementedError`
until their own tasks land.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import comb
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


# ── Holdout ───────────────────────────────────────────────────────────────────
# Where train stops and validation starts.  Module constants for the same reason
# as the gate thresholds: they define what "validated" means for every
# graduated pattern.

HOLDOUT_START = pd.Timestamp("2025-02-01")
"""First day of the holdout: the first month neither proxy model remembers.

``LLM_CUTOFFS.md`` measured it.  The lite model still recalls the 2025-01-02
close and first fails on 2025-02-03; the advanced model's recall ends in
2024-11.  A holdout inside the remembered period cannot catch a memorised
pattern.  Memorisation *raises* in-sample precision, so a remembered pattern
passes a remembered holdout just as easily as its training split.
"""

HOLDOUT_END = pd.Timestamp("2025-12-31")
"""Last day of the holdout: the day before the protected 2026 evaluation.

The 2026 window is kept for the final forecasting claim.  No shock or control
from it may enter pattern discovery or validation.
"""

MIN_SPLIT_SHOCKS = 10
"""Fewest shock windows either split may hold before :func:`train_holdout_split` refuses.

At the committed ±5% threshold the holdout holds 13 shock events; at the earlier
±7% it held 4, and that is the kind of configuration this floor exists to reject.
With a handful of shocks, one extra hit swings holdout lift enormously, and the
``MIN_HOLDOUT_LIFT`` criterion would be passed or failed by chance.  10 leaves a
little room below 13 for events a caller filters out.
"""


# ── Scoring ───────────────────────────────────────────────────────────────────

BOOTSTRAP_RESAMPLES = 2000
"""Bootstrap resamples behind the confidence interval on lift."""

CI_LEVEL = 0.95
"""Two-sided coverage of the interval on lift; ``ci_low`` is its lower end."""

BOOTSTRAP_SEED = 0
"""Fixed so that a graduated pattern's interval is reproducible from its inputs."""


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
        P(shock | pattern matched) **in the population**, not in the sample.
        The windows are a matched sample, one or more controls per shock, so
        their own shock share is set by the design, not by the market.
        Precision is therefore rebuilt from how often the pattern appears among
        shocks and among controls, weighted by :attr:`base_rate`.  It equals
        ``n_hits / n_matches`` only when ``base_rate`` equals the sample's own
        shock share.  ``nan`` when the pattern matched nothing.
    base_rate
        The population shock rate the pattern has to beat: the share of all
        sessions in the split's period that were shocks.  See
        :func:`shock_base_rate`.
    lift
        ``precision / base_rate``: how many times more likely a shock is when
        the pattern is present.  The headline number, and what
        :data:`MIN_LIFT` and :data:`MIN_HOLDOUT_LIFT` are thresholds on.
    p_value
        One-sided Fisher's exact test on the 2x2 table of matched/unmatched
        against shock/control: is the pattern more common among shocks than
        among controls?  Unlike precision it needs no correction, because the
        odds ratio it tests does not depend on how many controls were sampled.
    ci_low, ci_high
        Stratified bootstrap confidence interval (:data:`CI_LEVEL`) on
        :attr:`lift`.  The gate requires ``ci_low > 1``, a stricter and more
        informative statement than the p-value alone.
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
    4. **On the same side of the holdout boundary.**  Before :data:`HOLDOUT_START`,
       inside the holdout, or after :data:`HOLDOUT_END` (protected), the same as
       the shock, and never straddling a boundary.  Without this rule,
       controls for 2025 holdout shocks drift into 2024 or 2026.  On the real
       data, the holdout ended up with 13 shocks but only 5 controls, and its
       base rate no longer matched train's.  The sampler was also reading
       protected-period sessions.

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

    # Each session's period: 0 before the holdout, 1 inside it, 2 protected. A
    # control must share its shock's period and must not straddle a boundary
    # (its news cutoff, the previous session, on the other side).
    period = np.where(dates < HOLDOUT_START, 0, np.where(dates <= HOLDOUT_END, 1, 2))
    straddles = np.concatenate([[True], period[1:] != period[:-1]])

    excluded = np.isnan(vol_before) | straddles  # also covers the first session, which has no prior day
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
        candidates = candidates[~excluded[candidates] & ~used[candidates] & (period[candidates] == period[event])]
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


def train_holdout_split(windows: list[Window]) -> tuple[list[Window], list[Window]]:
    """Split windows into a discovery set and a post-cutoff validation set.

    Parameters
    ----------
    windows
        Shock and control windows, from :func:`flag_shock_windows` and
        :func:`sample_matched_controls`.

    Returns
    -------
    tuple[list[Window], list[Window]]
        ``(train, holdout)``, each ordered by ``event_date``.  ``train`` holds
        every window whose ``event_date`` is before :data:`HOLDOUT_START`.
        ``holdout`` holds every window that lies wholly inside
        [:data:`HOLDOUT_START`, :data:`HOLDOUT_END`], its ``as_of`` included.
        Windows after :data:`HOLDOUT_END` belong to the protected evaluation and
        are in neither.  So is a window that straddles :data:`HOLDOUT_START`,
        with its news cutoff before the boundary and its move after it.

    Raises
    ------
    ValueError
        If either split holds fewer than :data:`MIN_SPLIT_SHOCKS` shock windows.
        The gate would still run, but its holdout criterion would be decided by
        noise, so refusing is the honest answer.

    Notes
    -----
    **The split is by date, not by fraction, and certainly not random.**  A
    random split lets a pattern found in one half of an earnings cycle validate
    on the other half of the same cycle.  A chronological split inside
    2020-2024, as first planned, still fails: both proxy models remember that
    period, so a memorised pattern would pass its holdout as easily as its
    training data.  Only data the models cannot remember tests a pattern, and
    that means a holdout that starts after the measured cutoff.

    **Controls stay with their shock's split.**  :func:`sample_matched_controls`
    draws every control from the same period as its shock, so each split keeps
    its shock-to-control ratio.  Windows built some other way are split by
    their own dates, like everything else.  :func:`evaluate_pattern` computes
    each split's base rate from that split's own windows either way.

    **The holdout overlaps the 2025 backtest spec.**  A pattern graduated
    because it held up in Feb-Dec 2025 will flatter any 2025 backtest of a
    forecaster that uses it.  The pattern was selected partly for working
    there.  The honest measure of a pattern-using forecaster is the protected
    2026 evaluation.

    This function sets no lower bound on ``train``.  Pass windows from the
    discovery period (2020 onward) rather than the whole price history.
    """

    def by_date(w: Window) -> pd.Timestamp:
        return w.event_date

    train = sorted((w for w in windows if w.event_date < HOLDOUT_START), key=by_date)
    holdout = sorted(
        (w for w in windows if w.as_of >= HOLDOUT_START and w.event_date <= HOLDOUT_END),
        key=by_date,
    )

    for name, part in (("train", train), ("holdout", holdout)):
        n_shocks = sum(w.is_shock for w in part)
        if n_shocks < MIN_SPLIT_SHOCKS:
            hint = (
                f"The holdout is {HOLDOUT_START:%Y-%m-%d} to {HOLDOUT_END:%Y-%m-%d}, the clean window after the "
                "models' training cutoff. A lower SHOCK_THRESHOLD yields more events; at ±7% this window "
                "holds only 4."
                if name == "holdout"
                else "Pass windows from the discovery period (2020 onward), before the holdout."
            )
            raise ValueError(
                f"The {name} split holds {n_shocks} shock window(s); at least {MIN_SPLIT_SHOCKS} are needed "
                f"for its lift to mean anything. {hint}"
            )
    return train, holdout


def shock_base_rate(
    prices_df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    threshold_pct: float = SHOCK_THRESHOLD,
) -> float:
    """Return the share of sessions in ``[start, end]`` whose move reached the shock threshold.

    This is the ``base_rate`` :func:`evaluate_pattern` needs: the rate a pattern
    has to beat in the market, not in the matched sample.  Compute it per
    split, because the periods differ.  At ±5% it is 12.0% of sessions in
    2020-2024 and 7.0% in the Feb-Dec 2025 holdout.

    It counts sessions, not merged events, because it answers "how often is the
    next session a shock?", the question a forecaster using the pattern faces.
    It is the same quantity as the shock task's unconditional anchor.
    """
    returns = _close_series(prices_df).pct_change() * 100.0
    window = returns.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
    if window.empty:
        raise ValueError(f"No sessions between {start} and {end}.")
    return float((window.abs() >= threshold_pct).mean())


def _fisher_greater(hits: int, n_shocks: int, n_controls: int, n_matches: int) -> float:
    """One-sided Fisher's exact p-value: P(at least ``hits`` shocks among ``n_matches`` matched windows).

    The hypergeometric upper tail, summed exactly in integers, so there is no
    scipy dependency to carry into the sandbox.  ``math.comb`` returns 0 for
    impossible terms, so the sum needs no bounds bookkeeping.
    """
    total = comb(n_shocks + n_controls, n_matches)
    tail = sum(comb(n_shocks, k) * comb(n_controls, n_matches - k) for k in range(hits, n_matches + 1))
    return tail / total


def _lift(sensitivity: np.ndarray, false_match_rate: np.ndarray, base_rate: float) -> np.ndarray:
    """Compute population lift from match rates: ``P(match | shock) / P(match)``.

    ``P(match) = sensitivity * base_rate + false_match_rate * (1 - base_rate)``.
    It is ``nan`` where the pattern would match nothing at all.
    """
    matched = sensitivity * base_rate + false_match_rate * (1.0 - base_rate)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(matched > 0, sensitivity / matched, np.nan)


def evaluate_pattern(
    pattern_matches: list[bool],
    shock_labels: list[bool],
    base_rate: float,
) -> PatternMetrics:
    """Score one candidate pattern against labelled windows.

    Parameters
    ----------
    pattern_matches
        Per window, whether the pattern's news signature was present.
    shock_labels
        Per window, whether it is a shock (``True``) or a control (``False``).
        Same length and order as ``pattern_matches``.
    base_rate
        The population shock rate for the windows' period, from
        :func:`shock_base_rate`.  For a directional pattern, pass directional
        labels and the directional rate.  **Required, and not the sample's own
        shock share**: with one control per shock that share is 50%, lift can
        then never exceed 2.0, and ``MIN_LIFT = 2.0`` would demand a perfect
        pattern.

    Returns
    -------
    PatternMetrics
        Population precision and lift, Fisher's exact p-value, and a bootstrap
        interval on lift.  A pattern that matched nothing scores ``nan``
        precision and lift, ``p_value = 1.0`` and a ``nan`` interval.  That is a
        normal, informative outcome for the per-experiment file, not an error.

    Raises
    ------
    ValueError
        If the inputs differ in length, ``base_rate`` is outside (0, 1), or
        the windows lack either shocks or controls.

    Notes
    -----
    **Why precision is rebuilt instead of counted.**  The windows are a matched
    sample: every shock plus a chosen number of controls.  Counting
    ``n_hits / n_matches`` measures precision in that artificial mix.  What a
    forecaster needs is precision in the market.  So the pattern's rate among
    shocks (``n_hits / n_shocks``) and among controls are combined with the
    real base rate.  This is the standard correction for a case-control
    design, and it recovers ``n_hits / n_matches`` when ``base_rate`` equals
    the sample's shock share.

    **Why Fisher's exact test, and why it isn't corrected.**  Cell counts are
    small by construction; a holdout of 13 shocks is typical, and chi-squared
    is unreliable there.  The test asks whether the pattern is more common
    among shocks than among controls.  Its odds ratio does not depend on how
    many controls were sampled, so the p-value is valid for this design as it
    stands, and it is the same whatever ``base_rate`` is passed.

    **Why a stratified bootstrap.**  The sample fixes how many shocks and how
    many controls there are, so each resample draws shocks from shocks and
    controls from controls.  Resamples in which the pattern matches nothing
    have no lift and are left out.  The seed is fixed, so the same inputs
    always give the same interval.
    """
    matches = np.asarray(pattern_matches, dtype=bool)
    shocks = np.asarray(shock_labels, dtype=bool)
    if matches.ndim != 1 or matches.shape != shocks.shape:
        raise ValueError(
            f"pattern_matches and shock_labels must be flat and equal in length; got {matches.shape} and {shocks.shape}."
        )
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must lie strictly between 0 and 1; got {base_rate}.")
    n_shocks, n_controls = int(shocks.sum()), int((~shocks).sum())
    if n_shocks == 0 or n_controls == 0:
        raise ValueError(
            f"A pattern needs both shocks and controls to be scored; got {n_shocks} shock(s) and "
            f"{n_controls} control(s). Draw controls with sample_matched_controls."
        )

    hits = int((matches & shocks).sum())
    n_matches = int(matches.sum())
    # A pattern that matches nothing needs no special case: the tail sum is 1,
    # _lift is nan when nothing matches, and so is every bootstrap resample.
    p_value = _fisher_greater(hits, n_shocks, n_controls, n_matches)

    sensitivity = np.array(hits / n_shocks)
    false_match_rate = np.array((n_matches - hits) / n_controls)
    lift = float(_lift(sensitivity, false_match_rate, base_rate))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    shock_matches, control_matches = matches[shocks], matches[~shocks]
    sens_b = shock_matches[rng.integers(0, n_shocks, (BOOTSTRAP_RESAMPLES, n_shocks))].mean(axis=1)
    fmr_b = control_matches[rng.integers(0, n_controls, (BOOTSTRAP_RESAMPLES, n_controls))].mean(axis=1)
    lift_b = _lift(sens_b, fmr_b, base_rate)
    tail = 100 * (1 - CI_LEVEL) / 2
    if np.isnan(lift_b).all():
        ci_low = ci_high = np.nan
    else:
        ci_low, ci_high = (float(v) for v in np.nanpercentile(lift_b, [tail, 100 - tail]))

    return PatternMetrics(
        n_windows=len(matches),
        n_matches=n_matches,
        n_hits=hits,
        precision=lift * base_rate,
        base_rate=base_rate,
        lift=lift,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
    )


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
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CI_LEVEL",
    "CONTROL_EXCLUSION_DAYS",
    "CONTROL_POOL_MULTIPLE",
    "CONTROL_WINDOW_DAYS",
    "HOLDOUT_END",
    "HOLDOUT_START",
    "MAX_P_VALUE",
    "MIN_HOLDOUT_LIFT",
    "MIN_LIFT",
    "MIN_MATCHES",
    "MIN_SPLIT_SHOCKS",
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
    "shock_base_rate",
    "train_holdout_split",
]
