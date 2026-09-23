"""NVDA equity analyst agent module.

Exports the :class:`AgentConfig` factories, prompt builder, and predictor
convenience factory for the NVDA implementation.  Only the news-grounded factory
is NVDA-named so far (:func:`build_nvda_news_config`); the rest still carry the
``wti`` prefix inherited from the energy/oil parent.
"""

from ai_stocks_forecasting.analyst_agent.agent import (
    WtiPriceForecastPromptBuilder,
    build_nvda_news_config,
    build_wti_agent_predictor,
    build_wti_basic_config,
    build_wti_code_exec_config,
    build_wti_multitask_news_config,
    build_wti_tool_config,
    compress_history,
)


__all__ = [
    "WtiPriceForecastPromptBuilder",
    "build_nvda_news_config",
    "build_wti_agent_predictor",
    "build_wti_basic_config",
    "build_wti_code_exec_config",
    "build_wti_multitask_news_config",
    "build_wti_tool_config",
    "compress_history",
]
