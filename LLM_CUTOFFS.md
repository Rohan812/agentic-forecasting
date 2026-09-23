# LLM training cutoffs

The project uses two Vector-proxy models, defined in `aieng.forecasting.models`:

| Constant | Model |
|---|---|
| `LITE_MODEL` (default) | `gemini-3.1-flash-lite-preview` |
| `ADVANCED_MODEL` | `gemini-3.5-flash` |

The working assumption has been that both models are trained **through about January 2025**. That
number decides which backtest windows can be scored honestly, so it was **measured** rather than
taken from a model card. The measurement roughly confirms the assumption.

## What was measured

A cutoff probe was run with search disabled, so it measures only what the model has memorised.
It asks each model three kinds of question:

1. **Self-report.** "What is your training cutoff?"
2. **Price-recall ladder.** NVDA's close on the first session of each month from 2024-06 to
   2026-09. An answer counts as recalled if it is within 10% of the actual close. A "don't know"
   is recorded as a refusal.
3. **Memorisation probe.** For a balanced sample of real shock and non-shock windows, did the
   stock make a large move that week? The sample is balanced, so chance is 50%.

| Model | Self-reported | Last correct recall | First failure or refusal | Claims to know holdout/protected windows |
|---|---|---|---|---|
| `gemini-3.1-flash-lite-preview` | 2024-01 | **2025-01-02** | 2025-02-03: wrong by 16% | 0%, refuses |
| `gemini-3.5-flash` | 2025-01 | **2024-11-01** | 2024-12-02: refuses | 0%, refuses |

Across both models, recall of pre-cutoff shock outcomes was **31/48 (p = 0.030)** against a
coin flip. Both models also said they knew **100%** of those windows but got only about 64%
right. **They answer confidently and are often wrong.**

The self-reported cutoffs don't agree with each other or with the measured ones. A model's
self-report is not evidence.

The probe itself (`cutoff_probe.py`, plus its full transcript) is part of the signal-discovery
work package and has not been merged into this branch yet. The numbers above come from that
transcript.

## What it means for each window

| Window | Relation to cutoff | What you may claim |
|---|---|---|
| **2020–2024: discovery and calibration** | Before the cutoff. The models remember it. | Numerical baselines and base-rate calibration are fine. LLM forecasts scored here measure **recall, not forecasting**. Treat any pattern the discovery agent proposes from this window as possibly remembered, not discovered. |
| **January 2025** | On the lite model's boundary: it still recalls the 2025-01-02 close to within 2%. | Exclude it from LLM scoring, or report it separately. `specs/nvda_backtest.yaml` starts at 2025-01-06, so the first few origins sit on this boundary. |
| **Feb–Dec 2025: backtest** (`specs/nvda_backtest.yaml`) | After the cutoff. Both models refuse. | A clean window for comparing LLM forecasters with numerical ones and for choosing a model. |
| **2026: protected eval** (`specs/nvda_eval.yaml`) | Fully after the cutoff. | Clean. The agent knows only what it retrieves through cutoff-fenced search, which is the condition we want to test. |

## A correction to an earlier assumption

An earlier version of the plan said memorised 2024 patterns are harmless because they "fail the
statistical gate". **That is backwards.** Suppose a model remembers that NVDA dropped after a
given headline. The pattern it proposes will then score well on exactly the data it remembers.
Memorisation raises in-sample precision; the gate does not catch leakage, it rewards it.

What does protect the gate:

- The **holdout** split must lie after the cutoff (Feb 2025 or later). A pattern discovered in
  2024 only counts once it holds up on data the model cannot remember.
- The discovery agent must reason from **retrieved or archived news**, never from its own
  memory. The probe shows the models state false outcomes with confidence, so this is an
  evidence-based requirement, not just a precaution.
- Every write-up that uses 2024 discovery data must say the window is contaminated.

## What to brief the team on

1. The January 2025 cutoff is roughly correct: the lite model's recall ends at 2025-01 and the
   advanced model's at 2024-11.
2. Discovery on 2024 data is allowed, but it is contaminated. The gate does not clean it; only a
   post-cutoff holdout does.
3. The 2025 backtest (from February on) and the 2026 eval are clean. Any headline LLM result
   must come from these windows.
4. Re-run the probe when the proxy's model versions change. Cutoffs change without notice, and a
   later cutoff would push the backtest window inside the training data.
