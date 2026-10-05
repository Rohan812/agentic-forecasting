"""Run a news experiment: propose yes/no questions, screen them on the holdout, then test survivors in full.

For each experiment in ``experiments/<id>/focus.yaml``:

1. **Propose.** One lite-model call turns the focus into at most
   ``max_questions`` yes/no news questions, saved to ``questions.yaml`` before
   any labelling.  They are never edited after labels are seen.
2. **Screen on the holdout.** Each question is labelled on the holdout
   windows first (13 shocks with three matched controls each, about $0.85).  The gate needs holdout lift >= 1.5 and
   holdout p < 0.10 *as well as* the training criteria, so a question that
   fails these cannot graduate whatever its training result, and its 238
   training windows are never paid for.
3. **Test in full.** A survivor is labelled on the training windows and
   scored by :func:`~ai_stocks_forecasting.discovery.evaluate_candidate`.

Every question, screened out or tested, goes to the experiment's trail.
Spend is capped per experiment and for the whole stage
(:data:`~ai_stocks_forecasting.news_labels.STAGE_BUDGET_USD`).

Usage, from ``implementations/``::

    uv run python -m ai_stocks_forecasting.news_study --experiment exp02_export_controls
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from typing import Any

import yaml
from ai_stocks_forecasting import signals
from ai_stocks_forecasting.data import NVDA_SERIES_ID, build_nvda_service
from ai_stocks_forecasting.discovery import (
    EXPERIMENTS_DIR,
    HOLDOUT_CONTROLS_PER_SHOCK,
    StudySet,
    _labels,
    build_study_set,
    evaluate_candidate,
    record_candidate,
    upsert_candidate,
)
from ai_stocks_forecasting.news_labels import BudgetLedger, label_windows
from aieng.forecasting.models import LITE_MODEL


EXPERIMENT_BUDGET_USD = 10.0
"""Cap on one experiment's labelling spend (two full questions at about $4.70 each, screens included)."""

_PROPOSER_INSTRUCTION = """\
You design candidate news signals for a statistical study of NVIDIA (NVDA) stock shocks \
(a move of at least 5% in one session). Given a research focus, write yes/no questions \
that a reader could answer from a news briefing covering the 3 days before a date.

Rules:
- Ask whether a specific kind of EVENT was reported, never about NVDA's stock price, its \
move, or market reaction.
- Make each question concrete enough to answer yes or no from news alone.
- Do not name specific dates or past episodes; the question is asked about many dates.
- Questions must differ from each other.

Return JSON: {"questions": [{"question": "...", "direction": "either" | "up" | "down", \
"rationale": "one sentence on why it could precede a shock"}]}"""


def propose_questions(experiment_id: str, max_questions: int) -> list[dict[str, str]]:
    """Return the experiment's questions, proposing and saving them on first use."""
    import litellm  # noqa: PLC0415

    folder = EXPERIMENTS_DIR / experiment_id
    path = folder / "questions.yaml"
    if path.exists():
        return yaml.safe_load(path.read_text())["questions"]
    focus = yaml.safe_load((folder / "focus.yaml").read_text())["focus"]
    resp = litellm.completion(
        model=f"openai/{LITE_MODEL}",
        api_base=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        messages=[
            {"role": "system", "content": _PROPOSER_INSTRUCTION},
            {"role": "user", "content": f"Research focus:\n{focus}\n\nWrite at most {max_questions} questions."},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=800,
    )
    proposed = json.loads(resp.choices[0].message.content or "{}")["questions"][:max_questions]
    questions = [
        {
            "question_id": f"{experiment_id}_q{i}",
            "question": q["question"].strip(),
            "direction": q.get("direction", "either") if q.get("direction") in ("either", "up", "down") else "either",
            "rationale": q.get("rationale", ""),
        }
        for i, q in enumerate(proposed, start=1)
    ]
    path.write_text(
        yaml.safe_dump(
            {"experiment_id": experiment_id, "proposed_on": date.today().isoformat(), "questions": questions},
            sort_keys=False,
            allow_unicode=True,
        )
    )
    return questions


def _rule_from(labels: dict[str, dict[str, str]]) -> Any:
    def rule(window: signals.Window, study: StudySet) -> bool:
        return labels.get(window.as_of.date().isoformat(), {}).get("answer") == "yes"

    return rule


def holdout_screen(
    study: StudySet, labels: dict[str, dict[str, str]], direction: str
) -> tuple[signals.PatternMetrics, list[str]]:
    """Score the holdout alone and return the holdout criteria it fails."""
    windows = study.holdout
    rule = _rule_from(labels)
    metrics = signals.evaluate_pattern(
        [rule(w, study) for w in windows],
        _labels(windows, direction),
        study.base_rate("holdout", direction),  # type: ignore[arg-type]
    )
    reasons = []
    if not metrics.lift >= signals.MIN_HOLDOUT_LIFT:
        reasons.append(f"holdout lift {metrics.lift:.2f} is below {signals.MIN_HOLDOUT_LIFT}")
    if not metrics.p_value < signals.MAX_HOLDOUT_P_VALUE:
        reasons.append(f"holdout p-value {metrics.p_value:.3g} is not below {signals.MAX_HOLDOUT_P_VALUE}")
    return metrics, reasons


def _record_screened_out(
    experiment_id: str, q: dict[str, str], metrics: signals.PatternMetrics, reasons: list[str]
) -> None:
    path = EXPERIMENTS_DIR / experiment_id / "trail.yaml"
    trail = yaml.safe_load(path.read_text()) if path.exists() else {"experiment_id": experiment_id, "candidates": []}
    upsert_candidate(
        trail,
        {
            "pattern_id": q["question_id"],
            "cue": q["question"],
            "rule": f"news label: labels/{q['question_id']}.yaml",
            "direction": q["direction"],
            "evaluated_on": date.today().isoformat(),
            "graduated": False,
            "stage": "screened out on the holdout; training windows not labelled",
            "rejection_reasons": reasons,
            "holdout_design": f"matched, {HOLDOUT_CONTROLS_PER_SHOCK} controls per shock",
            "holdout": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.__dict__.items()},
        },
    )
    path.write_text(yaml.safe_dump(trail, sort_keys=False, allow_unicode=True))


def run_experiment(experiment_id: str, max_questions: int = 2, budget_usd: float = EXPERIMENT_BUDGET_USD) -> None:
    """Propose, screen and test an experiment's questions within its budget."""
    ledger = BudgetLedger()
    study = build_study_set(build_nvda_service().get_series(NVDA_SERIES_ID, as_of=datetime.now()))
    spent = 0.0
    print(
        f"{experiment_id}: stage spend so far ${ledger.spent_usd:.2f} of ${ledger.cap_usd:.2f}; experiment cap ${budget_usd:.2f}"
    )
    for q in propose_questions(experiment_id, max_questions):
        print(f"\n{q['question_id']} ({q['direction']}): {q['question']}")
        labels, usd = label_windows(
            q["question_id"], q["question"], [w.as_of for w in study.holdout], ledger, run_cap_usd=budget_usd - spent
        )
        spent += usd
        if any(w.as_of.date().isoformat() not in labels for w in study.holdout):
            print("  stopped: holdout not fully labelled within budget")
            return
        metrics, reasons = holdout_screen(study, labels, q["direction"])
        print(
            f"  holdout: {metrics.n_matches} matches, {metrics.n_hits} hits, lift {metrics.lift:.2f}, p {metrics.p_value:.3g} (${usd:.2f})"
        )
        if reasons:
            _record_screened_out(experiment_id, q, metrics, reasons)
            print("  screened out: " + "; ".join(reasons))
            continue
        labels, usd = label_windows(
            q["question_id"], q["question"], [w.as_of for w in study.train], ledger, run_cap_usd=budget_usd - spent
        )
        spent += usd
        if any(w.as_of.date().isoformat() not in labels for w in study.train):
            print("  stopped: training windows not fully labelled within budget")
            return
        result = evaluate_candidate(study, q["question_id"], q["question"], _rule_from(labels), q["direction"])  # type: ignore[arg-type]
        record_candidate(experiment_id, result, f"news label: labels/{q['question_id']}.yaml")
        t = result.train
        print(
            f"  train: {t.n_matches} matches, {t.n_hits} hits, lift {t.lift:.2f}, p {t.p_value:.3g}, CI {t.ci_low:.2f}-{t.ci_high:.2f} (${usd:.2f})"
        )
        print("  GRADUATES" if result.graduates else "  rejected: " + "; ".join(result.reasons))
    print(
        f"\n{experiment_id}: spent ${spent:.2f}; stage total ${BudgetLedger().spent_usd:.2f} of ${ledger.cap_usd:.2f}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--experiment", required=True, help="experiment id under experiments/")
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--budget", type=float, default=EXPERIMENT_BUDGET_USD, help="USD cap for this experiment")
    args = parser.parse_args()
    run_experiment(args.experiment, args.max_questions, args.budget)


if __name__ == "__main__":
    _main()
