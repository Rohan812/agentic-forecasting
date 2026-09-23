"""NVDA master-strategy state — the schema behind the ``nvda-strategy`` adaptive skill.

**Draft.**  The field set is fixed enough for the mutation tool
(``graduate_news_pattern``) and the study prompt to code against; the rendered
``SKILL.md`` layout and the per-experiment research-trail files are finalised
when the adaptive agent is retargeted from WTI.  The WTI schema in
:mod:`~ai_stocks_forecasting.adaptive_agent.skill_state` stays in place until
then so the inherited adaptive agent keeps loading.

How this differs from the WTI strategy
--------------------------------------
The WTI agent graduates a hypothesis after a *count* of confirming resolutions.
Here a news pattern graduates only after clearing the statistical gate in
:mod:`ai_stocks_forecasting.signals` — so every :class:`NewsPattern` carries the
train and holdout :class:`PatternEvidence` that ``gate_pass`` was called with.
The master file is the operational path (the forecasting agent reads only
this); it holds graduated patterns only.  Candidates, rejections, and queries
belong in the per-experiment files, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillState
from pydantic import BaseModel


if TYPE_CHECKING:
    from ai_stocks_forecasting.signals import PatternMetrics


SKILL_NAME = "nvda-strategy"

PatternDirection = Literal["up", "down", "either"]
"""Which shocks a pattern predicts.  Scored against a direction-aware base rate."""


class PatternEvidence(BaseModel):
    """The statistics for one pattern on one split — mirrors :class:`~ai_stocks_forecasting.signals.PatternMetrics`.

    A Pydantic copy rather than the frozen dataclass itself so the state
    validates on load from YAML.  Build it with :meth:`from_metrics`.
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

    @classmethod
    def from_metrics(cls, metrics: PatternMetrics) -> PatternEvidence:
        """Copy a ``signals.PatternMetrics`` into a persistable model."""
        return cls(
            n_windows=metrics.n_windows,
            n_matches=metrics.n_matches,
            n_hits=metrics.n_hits,
            precision=metrics.precision,
            base_rate=metrics.base_rate,
            lift=metrics.lift,
            p_value=metrics.p_value,
            ci_low=metrics.ci_low,
            ci_high=metrics.ci_high,
        )


class NewsPattern(BaseModel):
    """One graduated news pattern: a cue the forecaster matches against today's news.

    Attributes
    ----------
    id
        Stable identifier, ``P-<n>``.
    cue
        The news signature in plain language, specific enough that a forecaster
        can say yes/no for a given day's briefing (e.g. "a US Commerce rule
        restricting AI-accelerator sales to China announced within 2 sessions").
    direction
        Which shocks the pattern predicts.
    source_experiment
        Experiment ID that discovered it (e.g. ``exp01_earnings``) — the pointer
        into the per-experiment research trail.
    train, holdout
        The evidence ``gate_pass`` accepted.
    graduated_on
        ISO date of graduation.
    forecast_guidance
        How the forecaster should use a match — e.g. "raise P(shock) toward the
        holdout precision", never a bare number detached from the evidence.
    """

    id: str
    cue: str
    direction: PatternDirection
    source_experiment: str
    train: PatternEvidence
    holdout: PatternEvidence
    graduated_on: str
    forecast_guidance: str = ""


class StrategyNote(BaseModel):
    """A dated observation that is not (yet) a graduated pattern."""

    date: str
    note: str


class StrategyVersion(BaseModel):
    """One row of the version history table."""

    date: str
    description: str


class NvdaStrategyState(AdaptiveSkillState):
    """Master strategy for the NVDA forecaster: approach plus graduated news patterns."""

    approach_narrative: str
    news_patterns: list[NewsPattern] = []
    notes: list[StrategyNote] = []
    version_history: list[StrategyVersion] = []

    def build_markdown(self, skill_name: str | None = None) -> str:
        """Render ``SKILL.md`` — the file the forecasting agent reads before every origin."""
        lines = [
            "---",
            f"name: {skill_name or SKILL_NAME}",
            "description: >-",
            "  The NVDA forecaster's master strategy: graduated news patterns that",
            "  cleared the statistical gate, with their evidence. Load this before",
            "  every forecast. Generated — change it through the mutation tools.",
            "---",
            "",
            "# NVDA Forecasting Strategy",
            "",
            "## Approach",
            "",
            self.approach_narrative.strip(),
            "",
            "## Graduated news patterns",
            "",
        ]
        if self.news_patterns:
            lines += [
                "| ID | Cue | Direction | Holdout precision | Holdout lift | Train lift (CI) | p (train) | Source |",
                "|----|-----|-----------|-------------------|--------------|-----------------|-----------|--------|",
            ]
            for p in self.news_patterns:
                t, h = p.train, p.holdout
                lines.append(
                    f"| {p.id} | {p.cue} | {p.direction} | {h.precision:.2f} ({h.n_hits}/{h.n_matches}) "
                    f"| {h.lift:.1f}x | {t.lift:.1f}x ({t.ci_low:.1f}-{t.ci_high:.1f}) | {t.p_value:.3f} "
                    f"| {p.source_experiment} |"
                )
            lines.append("")
            guided = [p for p in self.news_patterns if p.forecast_guidance.strip()]
            if guided:
                lines += ["### How to use a match", ""]
                lines += [f"- **{p.id}**: {p.forecast_guidance.strip()}" for p in guided]
                lines.append("")
        else:
            lines += [
                "*(No graduated patterns yet. Forecast from the base-rate anchors in the task spec.)*",
                "",
            ]

        if self.notes:
            lines += ["## Notes", "", "| Date | Note |", "|------|------|"]
            lines += [f"| {n.date} | {n.note} |" for n in self.notes]
            lines.append("")

        lines += ["## Version history", "", "| Date | Change |", "|------|--------|"]
        lines += [f"| {v.date} | {v.description} |" for v in self.version_history]
        lines.append("")
        return "\n".join(lines)


__all__ = [
    "SKILL_NAME",
    "NewsPattern",
    "NvdaStrategyState",
    "PatternDirection",
    "PatternEvidence",
    "StrategyNote",
    "StrategyVersion",
]
