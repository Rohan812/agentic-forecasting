"""Tests for shock flagging and matched-control sampling in ``ai_stocks_forecasting.signals``.

Synthetic price paths with known answers.  Each test pins a property that is
easy to get subtly wrong and that would bias the graduation gate without
raising any error.  Every magnitude is expressed relative to ``SHOCK_THRESHOLD``
(``T``), so the tests keep meaning the same thing if the threshold changes.
"""

from __future__ import annotations

import warnings
from math import comb

import numpy as np
import pandas as pd
import pytest
from ai_stocks_forecasting.paths import SHOCK_THRESHOLD
from ai_stocks_forecasting.signals import (
    CONTROL_EXCLUSION_DAYS,
    HOLDOUT_END,
    HOLDOUT_START,
    MAX_HOLDOUT_P_VALUE,
    MAX_P_VALUE,
    MIN_HOLDOUT_LIFT,
    MIN_LIFT,
    MIN_MATCHES,
    VOL_WINDOW_DAYS,
    PatternMetrics,
    Window,
    evaluate_pattern,
    flag_shock_windows,
    gate_pass,
    gate_reasons,
    sample_matched_controls,
    train_holdout_split,
)


T = SHOCK_THRESHOLD
CALM_VOL = 0.1 * T  # daily return std, in percent
VOLATILE_VOL = 0.35 * T


def _prices(returns_pct: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    """Business-day price path from a vector of daily simple returns in percent."""
    dates = pd.bdate_range(start, periods=len(returns_pct) + 1)
    values = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + returns_pct / 100.0]))
    return pd.DataFrame({"timestamp": dates, "value": values})


def test_shocks_are_flagged_both_ways_with_the_prior_session_as_cutoff() -> None:
    """Moves past the threshold count in both directions, and the cutoff is the prior *session*.

    Not tested at exactly ``T``: a return rebuilt from prices lands a few ulps
    either side of it, so an exact-boundary test would only test float rounding.
    """
    r = np.zeros(30)
    r[9] = 1.5 * T
    r[19] = -1.6 * T
    r[4] = 0.98 * T  # just below: not a shock
    prices = _prices(r)

    windows = flag_shock_windows(prices)
    assert [(w.direction, round(w.return_pct, 6)) for w in windows] == [
        ("up", round(1.5 * T, 6)),
        ("down", round(-1.6 * T, 6)),
    ]

    dates = pd.DatetimeIndex(prices["timestamp"])
    for w in windows:
        i = dates.get_loc(w.event_date)
        assert w.as_of == dates[i - 1], "as_of must be the previous trading session, not the previous calendar day."


def test_a_cluster_is_one_event_whose_cutoff_precedes_the_whole_episode() -> None:
    """Consecutive shocks merge, anchored on the largest move, with ``as_of`` before the first one.

    Anchoring the cutoff on the largest move instead would put coverage of the
    episode's opening days inside the agent's news window.
    """
    r = np.zeros(40)
    r[10], r[11], r[12] = 1.2 * T, -2.2 * T, 1.3 * T  # one three-day episode
    r[16] = 1.4 * T  # three sessions later: a separate event
    prices = _prices(r)
    dates = pd.DatetimeIndex(prices["timestamp"])

    windows = flag_shock_windows(prices)
    assert len(windows) == 2

    episode = windows[0]
    assert episode.event_date == dates[12], "Anchor on the largest move."
    assert (episode.direction, round(episode.return_pct, 6)) == ("down", round(-2.2 * T, 6))
    assert episode.as_of == dates[10], "Cutoff is the session before the episode's first shock."
    assert windows[1].event_date == dates[17]


def _regime_path(seed: int = 0) -> tuple[pd.DataFrame, list[int]]:
    """Alternating 45-session calm and volatile blocks, with shocks planted inside volatile blocks.

    Ordinary returns are clipped below the threshold, so the only shocks are the
    planted ones.  Returns the prices and the return indices of those shocks.
    """
    rng = np.random.default_rng(seed)
    blocks = [CALM_VOL, VOLATILE_VOL] * 3 + [CALM_VOL]
    r = np.concatenate([np.clip(rng.normal(0.0, vol, 45), -0.8 * T, 0.8 * T) for vol in blocks])
    shock_idx = [45 + 30, 135 + 30]  # 30 sessions into volatile blocks, so trailing vol is pure
    r[shock_idx[0]], r[shock_idx[1]] = 1.8 * T, -1.8 * T
    return _prices(r), shock_idx


def test_controls_avoid_shocks_and_their_neighbours() -> None:
    """No control is a shock or within the exclusion buffer; none repeats; a seed reproduces the draw."""
    prices, shock_idx = _regime_path()
    shocks = flag_shock_windows(prices)
    assert len(shocks) == len(shock_idx), "Precondition: only the planted shocks exist."

    controls = sample_matched_controls(shocks, prices, n_each=3, seed=11)
    assert len(controls) == 3 * len(shocks)
    assert len({c.event_date for c in controls}) == len(controls), "No session may be used twice."
    assert all(not c.is_shock and c.direction is None for c in controls)

    dates = pd.DatetimeIndex(prices["timestamp"])
    shock_pos = [dates.get_loc(w.event_date) for w in shocks]
    for c in controls:
        gap = min(abs(dates.get_loc(c.event_date) - p) for p in shock_pos)
        assert gap > CONTROL_EXCLUSION_DAYS, f"Control {c.event_date.date()} is {gap} sessions from a shock."

    assert controls == sample_matched_controls(shocks, prices, n_each=3, seed=11)


def test_controls_avoid_shocks_the_caller_did_not_pass_in() -> None:
    """A control may not sit next to *any* shock in the prices, not just the ones supplied.

    Callers pass subsets: controls for a training split are drawn while the
    holdout's shocks still sit in the price history.  A control beside one of
    those would carry that shock's news into the base rate.  Asking for far
    more controls than exist makes the sampler take every eligible session, so
    the check is deterministic.
    """
    rng = np.random.default_rng(1)
    r = np.clip(rng.normal(0.0, 0.3 * T, 200), -0.8 * T, 0.8 * T)
    r[60], r[90] = 1.8 * T, -1.8 * T  # 30 sessions apart: the second lies inside the first's candidate window
    prices = _prices(r)
    dates = pd.DatetimeIndex(prices["timestamp"])
    shocks = flag_shock_windows(prices)
    assert len(shocks) == 2, "Precondition: only the planted shocks exist."
    left_out = dates.get_loc(shocks[1].event_date)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # asking for more than exist warns, by design
        controls = sample_matched_controls(shocks[:1], prices, n_each=500, seed=0)

    assert controls, "Precondition: eligible sessions exist."
    for c in controls:
        gap = abs(dates.get_loc(c.event_date) - left_out)
        assert gap > CONTROL_EXCLUSION_DAYS, f"Control {c.event_date.date()} is {gap} sessions from an omitted shock."


def test_controls_come_from_the_shocks_volatility_regime() -> None:
    """Shocks sit in volatile blocks, so their controls must too, even though half the candidates are calm.

    This is the property that stops a pattern which merely tracks volatility
    from looking predictive of shocks.
    """
    prices, _ = _regime_path()
    shocks = flag_shock_windows(prices)
    controls = sample_matched_controls(shocks, prices, n_each=4, seed=3)

    close = prices.set_index("timestamp")["value"]
    vol_before = (close.pct_change() * 100).rolling(VOL_WINDOW_DAYS).std().shift(1)
    control_vol = vol_before.loc[[c.event_date for c in controls]]
    midpoint = (CALM_VOL + VOLATILE_VOL) / 2
    assert (control_vol > midpoint).all(), f"Calm-regime controls drawn: {control_vol[control_vol <= midpoint]}"


def test_controls_reject_non_shock_windows() -> None:
    """Passing controls back in as shocks is a caller error, not something to sample around."""
    prices, _ = _regime_path()
    control = Window(
        as_of=pd.Timestamp("2024-02-01"), event_date=pd.Timestamp("2024-02-02"), is_shock=False, return_pct=0.1
    )
    try:
        sample_matched_controls([control], prices)
    except ValueError as exc:
        assert "shock windows only" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for a non-shock window.")


# ── Train / holdout split ─────────────────────────────────────────────────────


def _window(event: str, is_shock: bool = True) -> Window:
    """Build a one-session window whose news cutoff is the previous business day."""
    event_date = pd.Timestamp(event)
    return Window(
        as_of=event_date - pd.offsets.BDay(1),
        event_date=event_date,
        is_shock=is_shock,
        return_pct=6.0 if is_shock else 0.5,
        direction="up" if is_shock else None,
    )


def test_split_puts_the_holdout_after_the_model_cutoff() -> None:
    """Train is everything before the cutoff, holdout is Feb-Dec 2025, and 2026 is in neither.

    January 2025 goes to train: it sits on the cutoff boundary, so it may be
    remembered.  A window whose news cutoff falls before the boundary and whose
    move falls after it belongs to neither split.  The input is shuffled to
    check each split comes back in date order.
    """
    train_shocks = [f"2024-{m:02d}-15" for m in range(1, 12)] + ["2025-01-15"]
    holdout_shocks = [f"2025-{m:02d}-14" for m in range(2, 13)]
    windows = [_window(d) for d in train_shocks + holdout_shocks]
    windows += [_window("2024-12-10", is_shock=False), _window("2025-06-10", is_shock=False)]
    straddler = _window("2025-02-03")  # a Monday: its as_of is Friday 2025-01-31
    protected = [_window("2026-02-06"), _window("2026-03-10", is_shock=False)]
    windows += [straddler, *protected]
    windows = [windows[i] for i in np.random.default_rng(0).permutation(len(windows))]

    train, holdout = train_holdout_split(windows)

    assert [w.event_date for w in train] == sorted(pd.Timestamp(d) for d in [*train_shocks, "2024-12-10"])
    assert [w.event_date for w in holdout] == sorted(pd.Timestamp(d) for d in [*holdout_shocks, "2025-06-10"])
    assert straddler not in train + holdout
    assert not any(w in train + holdout for w in protected), "The protected 2026 evaluation must not be touched."


def test_split_refuses_a_holdout_too_small_to_judge() -> None:
    """Four holdout shocks, like the old ±7% threshold gave, is refused with a message naming the fix."""
    windows = [_window(f"2024-{m:02d}-15") for m in range(1, 13)]
    windows += [_window(f"2025-{m:02d}-14") for m in (3, 5, 7, 9)]
    try:
        train_holdout_split(windows)
    except ValueError as exc:
        message = str(exc)
        assert "holdout split holds 4 shock window(s)" in message
        assert "SHOCK_THRESHOLD" in message
    else:
        raise AssertionError("Expected a ValueError for a four-shock holdout.")


def test_controls_stay_on_their_shocks_side_of_the_holdout_boundary() -> None:
    """A shock just before the cutoff gets only pre-cutoff controls; holdout shocks get only holdout ones.

    Each shock's candidate window reaches far across a boundary, so without the
    rule controls would cross it.  The March 2025 shock's window reaches back
    to the holdout's first session (2025-02-03).  That session's own news
    cutoff is 2025-01-31, so it straddles the boundary and must be skipped.
    The December shock's window reaches into protected 2026.  Asking for more
    controls than exist takes every eligible session, so the check is
    deterministic.
    """
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2024-06-03", "2026-03-31")
    r = np.clip(rng.normal(0.0, 0.3 * T, len(dates) - 1), -0.8 * T, 0.8 * T)
    for day, move in (("2025-01-22", 1.8 * T), ("2025-03-12", 1.8 * T), ("2025-12-22", -1.8 * T)):
        r[dates.get_loc(pd.Timestamp(day)) - 1] = move
    prices = _prices(r, start="2024-06-03")
    shocks = flag_shock_windows(prices)
    assert [w.event_date for w in shocks] == [pd.Timestamp(d) for d in ("2025-01-22", "2025-03-12", "2025-12-22")]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = sample_matched_controls(shocks[:1], prices, n_each=500, seed=0)
        inside = sample_matched_controls(shocks[1:], prices, n_each=500, seed=0)

    assert before and all(c.event_date < HOLDOUT_START for c in before)
    assert inside and all(c.as_of >= HOLDOUT_START and c.event_date <= HOLDOUT_END for c in inside)


# ── Pattern scoring ───────────────────────────────────────────────────────────


def _labelled(hits: int, n_shocks: int, false_matches: int, n_controls: int) -> tuple[list[bool], list[bool]]:
    """Build match/label vectors for a pattern hitting ``hits`` shocks and ``false_matches`` controls."""
    matches = (
        [True] * hits + [False] * (n_shocks - hits) + [True] * false_matches + [False] * (n_controls - false_matches)
    )
    labels = [True] * n_shocks + [False] * n_controls
    return matches, labels


def test_p_value_is_an_exact_one_sided_fisher_test() -> None:
    """Hand-computed on one table, and identical to scipy's implementation on many random ones.

    ``evaluate_pattern`` sums the hypergeometric tail itself so ``signals.py``
    carries no scipy dependency; scipy here is an independent reference.
    """
    matches, labels = _labelled(hits=6, n_shocks=10, false_matches=2, n_controls=10)
    by_hand = (comb(10, 6) * comb(10, 2) + comb(10, 7) * comb(10, 1) + comb(10, 8)) / comb(20, 8)
    assert evaluate_pattern(matches, labels, base_rate=0.12).p_value == pytest.approx(by_hand, rel=1e-12)

    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_s, n_c = int(rng.integers(3, 40)), int(rng.integers(3, 40))
        hits, false = int(rng.integers(0, n_s + 1)), int(rng.integers(0, n_c + 1))
        if hits + false == 0:
            continue
        matches, labels = _labelled(hits, n_s, false, n_c)
        expected = stats.fisher_exact([[hits, false], [n_s - hits, n_c - false]], alternative="greater").pvalue
        assert evaluate_pattern(matches, labels, base_rate=0.12).p_value == pytest.approx(expected, rel=1e-9)


def test_lift_is_measured_against_the_population_rate_not_the_sample() -> None:
    """The same pattern scores its market lift, not the lift inside a 50/50 matched sample.

    With one control per shock the sample's shock share is 50%, so even a perfect
    pattern would score lift 2.0 in-sample.  ``MIN_LIFT = 2.0`` would then demand
    perfection.  Against the real rate the same perfect pattern scores 1 / rate.
    The p-value tests shocks against controls and must not depend on the rate.
    """
    matches, labels = _labelled(hits=6, n_shocks=10, false_matches=2, n_controls=10)
    in_sample = evaluate_pattern(matches, labels, base_rate=0.5)
    in_market = evaluate_pattern(matches, labels, base_rate=0.12)
    assert (in_sample.precision, in_sample.lift) == pytest.approx((6 / 8, 1.5))
    sens, false_rate = 0.6, 0.2
    expected_lift = sens / (sens * 0.12 + false_rate * 0.88)
    assert (in_market.precision, in_market.lift) == pytest.approx((expected_lift * 0.12, expected_lift))
    assert in_market.p_value == in_sample.p_value

    perfect, labels = _labelled(hits=10, n_shocks=10, false_matches=0, n_controls=10)
    assert evaluate_pattern(perfect, labels, base_rate=0.5).lift == pytest.approx(2.0)
    assert evaluate_pattern(perfect, labels, base_rate=0.12).lift == pytest.approx(1 / 0.12)


def test_a_pattern_matching_nothing_is_no_evidence_not_an_error() -> None:
    """Proposals that match no window are normal; they score ``nan`` lift and ``p_value = 1``."""
    matches, labels = _labelled(hits=0, n_shocks=8, false_matches=0, n_controls=8)
    metrics = evaluate_pattern(matches, labels, base_rate=0.12)
    assert metrics.n_matches == 0 and metrics.p_value == 1.0
    assert np.isnan(metrics.lift) and np.isnan(metrics.precision) and np.isnan(metrics.ci_low)


def test_interval_separates_a_real_pattern_from_a_null_one() -> None:
    """A pattern five times more common among shocks clears 1; one equally common in both does not.

    The interval must also bracket the point estimate and be reproducible.
    """
    real = evaluate_pattern(*_labelled(hits=30, n_shocks=60, false_matches=6, n_controls=60), base_rate=0.12)
    null = evaluate_pattern(*_labelled(hits=18, n_shocks=60, false_matches=18, n_controls=60), base_rate=0.12)

    assert real.ci_low > 1.0 and real.ci_low <= real.lift <= real.ci_high
    assert null.ci_low < 1.0 < null.ci_high and null.lift == pytest.approx(1.0)
    again = evaluate_pattern(*_labelled(hits=30, n_shocks=60, false_matches=6, n_controls=60), base_rate=0.12)
    assert (again.ci_low, again.ci_high) == (real.ci_low, real.ci_high)


@pytest.mark.parametrize(
    ("matches", "labels", "base_rate", "message"),
    [
        ([True, False], [True, False, False], 0.12, "equal in length"),
        ([True, False], [True, False], 0.0, "strictly between 0 and 1"),
        ([True, False], [True, False], 1.2, "strictly between 0 and 1"),
        ([True, False], [True, True], 0.12, "both shocks and controls"),
    ],
)
def test_scoring_rejects_malformed_input(
    matches: list[bool], labels: list[bool], base_rate: float, message: str
) -> None:
    """Malformed input fails with a message that says what is wrong."""
    with pytest.raises(ValueError, match=message):
        evaluate_pattern(matches, labels, base_rate=base_rate)


# ── The gate ──────────────────────────────────────────────────────────────────


def _metrics(**overrides: float) -> PatternMetrics:
    """Build metrics sitting exactly on every passing threshold, with ``overrides`` applied.

    Built directly rather than via ``evaluate_pattern``, so the thresholds can
    be tested exactly, with no floating-point noise.
    """
    values: dict[str, float] = {
        "n_windows": 40,
        "n_matches": MIN_MATCHES,
        "n_hits": MIN_MATCHES,
        "precision": 0.3,
        "base_rate": 0.12,
        "lift": MIN_LIFT,
        "p_value": MAX_P_VALUE / 2,
        "ci_low": 1.01,
        "ci_high": 5.0,
    }
    values.update(overrides)
    return PatternMetrics(**values)  # type: ignore[arg-type]


def test_gate_passes_a_pattern_on_every_inclusive_threshold() -> None:
    """``n >= 5``, ``lift >= 2`` and ``holdout lift >= 1.5`` are inclusive: sitting exactly on them passes."""
    train, holdout = _metrics(), _metrics(lift=MIN_HOLDOUT_LIFT)
    assert gate_pass(train, holdout)
    assert gate_reasons(train, holdout) == []


@pytest.mark.parametrize(
    ("train_overrides", "holdout_overrides", "reason"),
    [
        ({"n_matches": MIN_MATCHES - 1}, {}, "training window(s); needs at least"),
        ({"lift": MIN_LIFT - 0.01}, {}, "training lift"),
        ({"p_value": MAX_P_VALUE}, {}, "training p-value"),  # the p-value bound is strict
        ({"ci_low": 1.0}, {}, "does not exclude 1"),  # so is the interval's
        ({}, {"lift": MIN_HOLDOUT_LIFT - 0.01}, "holdout lift"),
        ({}, {"p_value": MAX_HOLDOUT_P_VALUE}, "holdout p-value"),  # strict, like the training bound
    ],
)
def test_failing_any_single_criterion_blocks_graduation(
    train_overrides: dict[str, float], holdout_overrides: dict[str, float], reason: str
) -> None:
    """The conjunction is the point: each criterion alone vetoes, and the reason names it."""
    train = _metrics(**train_overrides)
    holdout = _metrics(**{"lift": MIN_HOLDOUT_LIFT, **holdout_overrides})
    assert not gate_pass(train, holdout)
    reasons = gate_reasons(train, holdout)
    assert len(reasons) == 1 and reason in reasons[0], reasons


def test_one_lucky_holdout_hit_does_not_graduate_a_memorised_pattern() -> None:
    """A pattern that aces training and then matches a single holdout shock must not graduate.

    This is the memorised-pattern case the holdout exists to catch.  One lucky
    hit on a 13-shock holdout gives an enormous lift (1 / base rate) on no
    evidence at all.  The lift rule alone waved it through; the holdout p-value
    rule stops it.
    """
    train = evaluate_pattern(*_labelled(hits=30, n_shocks=60, false_matches=6, n_controls=60), base_rate=0.12)
    holdout = evaluate_pattern(*_labelled(hits=1, n_shocks=13, false_matches=0, n_controls=13), base_rate=0.07)
    assert holdout.lift >= MIN_HOLDOUT_LIFT, "Precondition: the lift rule alone would pass it."
    assert not gate_pass(train, holdout)
    assert any("holdout p-value" in r for r in gate_reasons(train, holdout))


def test_undefined_values_fail_instead_of_slipping_through() -> None:
    """A ``nan`` compares false against everything, so a naive ``lift < 2`` check would wave it through.

    A pattern that never matched a holdout window was never tested on data the
    models cannot remember, and must say so.
    """
    nan = float("nan")
    assert not gate_pass(_metrics(lift=nan), _metrics(lift=MIN_HOLDOUT_LIFT))

    never_tested = _metrics(n_matches=0, n_hits=0, precision=nan, lift=nan, p_value=1.0, ci_low=nan, ci_high=nan)
    assert not gate_pass(_metrics(), never_tested)
    assert gate_reasons(_metrics(), never_tested) == [
        "matched no holdout windows, so it was never tested on data the models cannot remember"
    ]
