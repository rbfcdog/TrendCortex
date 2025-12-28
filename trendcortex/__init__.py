"""
TrendCortex: Hybrid AI-Assisted Crypto Trading System

A sophisticated cryptocurrency trading system combining technical analysis,
machine learning, and LLM-powered decision making for the WEEX AI Wars hackathon.
"""

__version__ = "1.0.0"
__author__ = "TrendCortex Team"
__license__ = "MIT"

from trendcortex.config import Config
from trendcortex.logger import setup_logging

__all__ = [
    "Config",
    "setup_logging",
]
