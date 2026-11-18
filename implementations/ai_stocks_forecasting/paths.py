"""Shared paths and the shock definition for the NVDA experiment."""

from __future__ import annotations

from pathlib import Path


def repo_data_dir() -> Path:
    """Return ``data/`` at the repository root (walk up from CWD if needed)."""
    cwd = Path.cwd().resolve()
    root = cwd
    while not (root / "pyproject.toml").exists():
        if root.parent == root:
            return cwd / "data"
        root = root.parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


DATA_DIR = repo_data_dir()

SHOCK_THRESHOLD = 5.0
"""Single-day move, in percent, that counts as an NVDA shock.

Applies in **both directions**: a shock is ``abs(1-day return) >= 5%``.

Why 5% and not the earlier 7%: **the holdout needs events.**  Both proxy models
remember prices through about January 2025 (``LLM_CUTOFFS.md``), so a pattern
only counts as evidence once it holds up on a holdout after the cutoff, and the
clean window before the protected 2026 evaluation is Feb-Dec 2025.  After
merging consecutive-day clusters, that window holds **13 shock events at 5%
but only 4 at 7%**.  Four events cannot support a holdout-lift criterion, since
a single hit swings the lift enormously.

The cost is a milder definition of "shock".  Over 2020-2024 it fires on 151
days (87 up, 64 down), **12% of trading days**, roughly a 1.5-sigma day against
NVDA's 3.39% daily return standard deviation.  At 7% it fired on 4%.  The
discovery loop is therefore looking for news that precedes an unusually large
day, not only a rare one.

Changing this constant ripples out.  Re-run
``python -m ai_stocks_forecasting.shock_anchors`` and update the anchors in
``tasks.TASK_SHOCK_SPEC``, and re-check that ``signals.sample_matched_controls``
still finds volatility-matched controls.  At 5% its exclusion buffer had to
shrink from 5 sessions to 2.
"""

SHOCK_HORIZON = 1
"""Horizon of the shock definition, in business days.

One day, not five. A single-day move isolates the reaction to a discrete news
event, which is what the discovery loop is trying to attribute patterns to; a
5-day window blends several events together and makes attribution ambiguous.
"""

__all__ = ["DATA_DIR", "SHOCK_HORIZON", "SHOCK_THRESHOLD", "repo_data_dir"]
