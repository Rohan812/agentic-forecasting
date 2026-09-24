"""NVDA equity analyst agent module.

Exports the :class:`AgentConfig` factories, prompt builder, and predictor
convenience factory for the NVDA implementation.  The news-grounded factory
(:func:`build_nvda_news_config`) and the prompt builder
(:class:`NvdaPriceForecastPromptBuilder`) are NVDA-named; the other factories
still carry the ``wti`` prefix inherited from the energy/oil parent.
"""

from ai_stocks_forecasting.analyst_agent.agent import (
    NvdaPriceForecastPromptBuilder,
    build_nvda_news_config,
    build_wti_agent_predictor,
    build_wti_basic_config,
    build_wti_code_exec_config,
    build_wti_multitask_news_config,
    build_wti_tool_config,
    compress_history,
)


__all__ = [
    "NvdaPriceForecastPromptBuilder",
    "build_nvda_news_config",
    "build_wti_agent_predictor",
    "build_wti_basic_config",
    "build_wti_code_exec_config",
    "build_wti_multitask_news_config",
    "build_wti_tool_config",
    "compress_history",
]
