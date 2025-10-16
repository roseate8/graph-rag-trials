"""
Calculate-metrics package for synthetic evaluation.

Provides batch async retrieval evaluation with comprehensive metrics.
"""

from .config import EvalConfig
from .evaluator import Evaluator
from .metrics import IRMetrics
from .retriever_for_evals import EvalRetriever, RetrievalResult
from .reporter import Reporter

__all__ = [
    'EvalConfig',
    'Evaluator',
    'IRMetrics',
    'EvalRetriever',
    'RetrievalResult',
    'Reporter'
]

