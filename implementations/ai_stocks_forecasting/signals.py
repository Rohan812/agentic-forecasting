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
**Skeleton.**  Signatures, return types, and the gate thresholds are fixed so
the agent side can code against them immediately.  Bodies raise
:class:`NotImplementedError` and are implemented next, each with its own tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
carries no information.  This is the main reason the shock threshold was set at
±7% rather than ±10%: at ±10% there are only 13 shock events in 2020-2024, so
almost nothing could clear both this and :data:`MAX_P_VALUE`.
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
        because NVDA's shocks skew upward (31 up vs 21 down at ±7% over
        2020-2024), so pooling directions inflates the apparent base rate for a
        pattern that only predicts one of them.
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


def flag_shock_windows(
    prices_df: pd.DataFrame,
    threshold_pct: float = SHOCK_THRESHOLD,
    horizon_days: int = SHOCK_HORIZON,
) -> list[Window]:
    """Find every window whose realised move clears the shock threshold.

    Parameters
    ----------
    prices_df
        Price history with ``timestamp`` and ``value`` columns, as returned by
        :meth:`~aieng.forecasting.data.service.DataService.get_series`.
    threshold_pct
        Absolute move, in percent, that counts as a shock.  Defaults to
        :data:`~ai_stocks_forecasting.paths.SHOCK_THRESHOLD` (7.0).  Applied in
        **both directions**.
    horizon_days
        Business-day span the move is measured over.  Defaults to
        :data:`~ai_stocks_forecasting.paths.SHOCK_HORIZON` (1).

    Returns
    -------
    list[Window]
        One window per shock, each with ``is_shock=True`` and a populated
        ``direction``, ordered by ``event_date``.

    Notes
    -----
    Two decisions the implementation has to make explicitly:

    *Overlapping shocks.*  At ``horizon_days > 1`` consecutive qualifying
    windows overlap and would double-count a single event.  Merge them into one
    window anchored on the largest move.  At the committed ``horizon_days=1``
    this cannot arise, but the parameter is exposed so the choice must not be
    silently wrong if someone changes it.

    *Adjacent-day clusters.*  Even at one day, shocks cluster — 2025-01-27
    (-17.0%) and 2025-01-28 (+8.9%) are one earnings narrative, not two
    independent events.  Treating them as independent inflates the sample the
    gate thinks it has.  Decide and document a minimum separation.
    """
    raise NotImplementedError("Phase 1, Team Signals T1")


def sample_matched_controls(
    windows: list[Window],
    prices_df: pd.DataFrame,
    n_each: int = 1,
    seed: int | None = None,
) -> list[Window]:
    """Draw non-shock control windows matched to the given shock windows.

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

    Returns
    -------
    list[Window]
        Control windows with ``is_shock=False`` and ``direction=None``.

    Notes
    -----
    Matching is what makes the comparison fair, and what to match on is a real
    decision, not a formality. Matching on calendar proximity controls for the
    market regime but risks landing inside the same news cycle as the shock;
    matching on volatility regime controls for the fact that shocks cluster in
    high-volatility periods, so an unmatched control set is quietly drawn from
    calmer markets and any pattern correlated with volatility will look
    predictive. Controls must also exclude the shock windows themselves and
    their immediate neighbours.
    """
    raise NotImplementedError("Phase 1, Team Signals T1")


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
    With 52 shocks across 2020-2024 an even split leaves roughly 26 each, which
    is workable but not generous; the implementation should refuse, loudly,
    rather than return a holdout too small to validate anything.
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
    "MAX_P_VALUE",
    "MIN_HOLDOUT_LIFT",
    "MIN_LIFT",
    "MIN_MATCHES",
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
