"""NVDA equity analyst agent module.

Exports the :class:`AgentConfig` factories, prompt builder, and predictor
convenience factory for the NVDA implementation.
"""

from ai_stocks_forecasting.analyst_agent.agent import (
    NvdaPriceForecastPromptBuilder,
    build_nvda_agent_predictor,
    build_nvda_basic_config,
    build_nvda_multitask_news_config,
    build_nvda_news_config,
    compress_history,
)


__all__ = [
    "NvdaPriceForecastPromptBuilder",
    "build_nvda_agent_predictor",
    "build_nvda_basic_config",
    "build_nvda_multitask_news_config",
    "build_nvda_news_config",
    "compress_history",
]
