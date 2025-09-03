"""
Re-ranking package for improving document retrieval relevance.
"""

try:
    from .base import BaseReRanker, ReRankResult, ReRankingError, ModelLoadError
    from .config import ReRankerConfig
    from .reranker_model import BGEReRanker, create_bge_reranker
except ImportError:
    # Handle direct execution case
    from base import BaseReRanker, ReRankResult, ReRankingError, ModelLoadError
    from config import ReRankerConfig
    from reranker_model import BGEReRanker, create_bge_reranker

__all__ = [
    # Base classes
    'BaseReRanker',
    'ReRankResult', 
    'ReRankingError',
    'ModelLoadError',
    
    # Configuration
    'ReRankerConfig',
    
    # BGE implementation
    'BGEReRanker',
    'create_bge_reranker'
]
