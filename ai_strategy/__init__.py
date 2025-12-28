"""
TrendCortex AI Strategy Framework

A comprehensive backtestable trading system combining:
- Technical indicators (EMA, ATR, RSI)
- Machine Learning models (RandomForest, XGBoost)
- LLM decision gating for final approval
- Risk management and position sizing
- Full backtest capabilities

This framework is designed to be:
1. Backtestable offline (no API keys required for training)
2. Integrable with live trading (WEEX API support)
3. Explainable via LLM reasoning
4. Auditable with structured logging
"""

__version__ = "1.0.0"
__author__ = "TrendCortex Team"

from ai_strategy.config import AIStrategyConfig
from ai_strategy.model_engine import ModelEngine
from ai_strategy.llm_gate import LLMGate
# from ai_strategy.ai_backtester import AIBacktester  # Coming soon

__all__ = [
    "AIStrategyConfig",
    "ModelEngine",
    "LLMGate",
    # "AIBacktester",  # Coming soon
]
