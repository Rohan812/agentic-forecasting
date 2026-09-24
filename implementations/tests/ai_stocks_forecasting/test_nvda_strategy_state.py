"""The master strategy file admits only patterns that clear the statistical gate.

These pin the contract that makes the file trustworthy: a ``NewsPattern``
re-checks its own evidence against ``signals.gate_reasons``, both when built in
code and when loaded from ``skill_state.yaml``.  Evidence comes from
``signals.evaluate_pattern`` on synthetic windows rather than hand-typed
numbers, so the tests move with the gate if its statistics change.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from ai_stocks_forecasting.adaptive_agent.nvda_strategy_state import (
    NewsPattern,
    NvdaStrategyState,
    PatternEvidence,
)
from ai_stocks_forecasting.signals import evaluate_pattern
from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillStore
from pydantic import ValidationError


# Train: present before half of 119 shocks and a tenth of 119 controls (earnings-calendar strength).
_TRAIN = evaluate_pattern([True] * 60 + [False] * 59 + [True] * 12 + [False] * 107, [True] * 119 + [False] * 119, 0.12)
# Holdout: 8 of 13 shocks, 1 of 13 controls.  Clears lift 1.5 and p < 0.10.
_HOLDOUT = evaluate_pattern([True] * 8 + [False] * 5 + [True] * 1 + [False] * 12, [True] * 13 + [False] * 13, 0.07)
# Holdout where the pattern appears as often before controls as before shocks: memorised, not predictive.
_HOLDOUT_DECAYED = evaluate_pattern(
    [True] * 4 + [False] * 9 + [True] * 4 + [False] * 9, [True] * 13 + [False] * 13, 0.07
)


def _pattern(pattern_id: str = "P-1", holdout: PatternEvidence | None = None) -> NewsPattern:
    return NewsPattern(
        id=pattern_id,
        cue="US Commerce rule restricting AI-accelerator sales to China | announced within 2 sessions",
        direction="either",
        source_experiment="exp01_export_controls",
        train=PatternEvidence.from_metrics(_TRAIN),
        holdout=holdout or PatternEvidence.from_metrics(_HOLDOUT),
        graduated_on="2026-09-24",
    )


def test_pattern_that_fails_the_gate_cannot_be_constructed() -> None:
    """Strong training evidence is not enough; a decayed holdout is refused, and the error says why."""
    with pytest.raises(ValidationError, match=r"did not clear the gate.*holdout lift 1\.00 is below 1\.5"):
        _pattern(holdout=PatternEvidence.from_metrics(_HOLDOUT_DECAYED))


def test_tampered_state_file_fails_to_load(tmp_path: Path) -> None:
    """Weakening a graduated pattern's evidence by hand on disk is caught at load, not served to the forecaster."""
    store = AdaptiveSkillStore(tmp_path, NvdaStrategyState)
    store.save(NvdaStrategyState(approach_narrative="Start from the anchors.", news_patterns=[_pattern()]))
    assert store.load().news_patterns[0].id == "P-1"

    state = yaml.safe_load(store.state_path.read_text())
    state["news_patterns"][0]["holdout"]["p_value"] = 0.4
    store.state_path.write_text(yaml.safe_dump(state))

    with pytest.raises(ValidationError, match="holdout p-value 0.4 is not below"):
        store.load()


def test_duplicate_pattern_ids_are_refused() -> None:
    """A forecast names the pattern it matched by id, so two patterns cannot share one."""
    with pytest.raises(ValidationError, match="Duplicate pattern id"):
        NvdaStrategyState(approach_narrative="x", news_patterns=[_pattern("P-1"), _pattern("P-1")])


def test_agent_written_pipes_do_not_break_the_pattern_table() -> None:
    """Agent-written cells containing ``|`` or a line break must stay one cell on one row.

    Both the cue and ``source_experiment`` are free text from the agent.  An
    unescaped ``|`` shifts every column after it, and a line break splits the
    row, leaving a stray fragment in SKILL.md.
    """
    pattern = _pattern().model_copy(update={"source_experiment": "exp01 | export controls\nrerun"})
    markdown = NvdaStrategyState(approach_narrative="x", news_patterns=[pattern]).build_markdown()
    lines = markdown.splitlines()
    row = next(line for line in lines if line.startswith("| P-1 "))
    assert row.replace("\\|", "").count("|") == 9  # 8 columns
    assert "rerun" in row, "The source must stay on its row, not spill onto the next line."
    assert not any(line.startswith("rerun") for line in lines)
