"""NVDA master-strategy state — the schema behind the ``nvda-strategy`` adaptive skill.

The field set is final for the mutation tool (``graduate_news_pattern``) and
the study prompt to code against.  The per-experiment research-trail files are
a separate schema, written when the discovery agent is built.  The WTI schema
in :mod:`~ai_stocks_forecasting.adaptive_agent.skill_state` stays in place
until the adaptive agent is retargeted, so the inherited agent keeps loading.

How this differs from the WTI strategy
--------------------------------------
The WTI agent graduates a hypothesis after a *count* of confirming resolutions.
Here a news pattern graduates only after clearing the statistical gate in
:mod:`ai_stocks_forecasting.signals` — so every :class:`NewsPattern` carries the
train and holdout :class:`PatternEvidence` that ``gate_pass`` was called with.
The master file is the operational path (the forecasting agent reads only
this); it holds graduated patterns only.  Candidates, rejections, and queries
belong in the per-experiment files, not here.

The gate is enforced by the schema, not by convention
------------------------------------------------------
:class:`NewsPattern` re-runs :func:`~ai_stocks_forecasting.signals.gate_reasons`
on its own evidence when it is constructed, and refuses to exist if any
criterion fails.  :meth:`AdaptiveSkillStore.load` validates the YAML through
the same model, so this also covers the file on disk: a hand-edited
``skill_state.yaml`` with weakened evidence fails to load rather than reaching
the forecaster.  The consequence to know about: **tightening a gate constant in
``signals.py`` makes previously graduated patterns that no longer clear it fail
to load**, naming each one.  That is deliberate.  A pattern in the master file
meets the current standard, and changing the standard is a visible event rather
than a silent grandfathering.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from ai_stocks_forecasting.signals import PatternMetrics, gate_reasons
from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillState
from pydantic import BaseModel, Field, field_validator, model_validator


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

    def to_metrics(self) -> PatternMetrics:
        """Rebuild the ``signals.PatternMetrics`` this evidence was copied from."""
        return PatternMetrics(**self.model_dump())


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

    id: str = Field(pattern=r"^P-\d+$")
    cue: str = Field(min_length=1)
    direction: PatternDirection
    source_experiment: str = Field(min_length=1)
    train: PatternEvidence
    holdout: PatternEvidence
    graduated_on: str
    forecast_guidance: str = ""

    @field_validator("graduated_on")
    @classmethod
    def _iso_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _cleared_the_gate(self) -> NewsPattern:
        """Refuse to exist unless the stored evidence passes ``signals.gate_pass``."""
        reasons = gate_reasons(self.train.to_metrics(), self.holdout.to_metrics())
        if reasons:
            raise ValueError(
                f"Pattern {self.id} did not clear the gate and cannot enter the master strategy: " + "; ".join(reasons)
            )
        return self


class StrategyNote(BaseModel):
    """A dated observation that is not (yet) a graduated pattern."""

    date: str
    note: str


class StrategyVersion(BaseModel):
    """One row of the version history table."""

    date: str
    description: str


def _cell(text: str) -> str:
    """Make agent-written text safe inside a markdown table cell (no pipes, no line breaks)."""
    return " ".join(text.split()).replace("|", "\\|")


class NvdaStrategyState(AdaptiveSkillState):
    """Master strategy for the NVDA forecaster: approach plus graduated news patterns."""

    approach_narrative: str
    news_patterns: list[NewsPattern] = []
    notes: list[StrategyNote] = []
    version_history: list[StrategyVersion] = []

    @field_validator("news_patterns")
    @classmethod
    def _unique_ids(cls, patterns: list[NewsPattern]) -> list[NewsPattern]:
        """Pattern ids are how a forecast names the pattern it matched, so they must be unique."""
        ids = [p.id for p in patterns]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate pattern id(s): {', '.join(duplicates)}")
        return patterns

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
                    f"| {p.id} | {_cell(p.cue)} | {p.direction} | {h.precision:.2f} ({h.n_hits}/{h.n_matches}) "
                    f"| {h.lift:.1f}x | {t.lift:.1f}x ({t.ci_low:.1f}-{t.ci_high:.1f}) | {t.p_value:.3f} "
                    f"| {_cell(p.source_experiment)} |"
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
            lines += [f"| {n.date} | {_cell(n.note)} |" for n in self.notes]
            lines.append("")

        lines += ["## Version history", "", "| Date | Change |", "|------|--------|"]
        lines += [f"| {v.date} | {_cell(v.description)} |" for v in self.version_history]
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
