"""
Base classes and data structures for re-ranking implementations.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class ReRankingError(Exception):
    """Base exception for re-ranking errors."""
    pass


class ModelLoadError(ReRankingError):
    """Raised when a re-ranking model fails to load."""
    pass


@dataclass
class ReRankResult:
    """Result from re-ranking operation."""
    chunk_id: str
    original_rank: int
    rerank_score: float
    final_rank: int
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'chunk_id': self.chunk_id,
            'original_rank': self.original_rank,
            'rerank_score': self.rerank_score,
            'final_rank': self.final_rank,
            'content': self.content,
            'metadata': self.metadata
        }


class BaseReRanker(ABC):
    """Abstract base class for document re-rankers."""
    
    def __init__(self, model_name: str):
        """Initialize the re-ranker.
        
        Args:
            model_name: Name/path of the model to use
        """
        self.model_name = model_name
        self.is_loaded = False
        self._load_time = None
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the re-ranking model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        top_k: int = 10
    ) -> List[ReRankResult]:
        """Re-rank chunks based on relevance to query.
        
        Args:
            query: The user query
            chunks: List of chunk dictionaries with content and metadata
            top_k: Number of top chunks to return after re-ranking
            
        Returns:
            List[ReRankResult]: Re-ranked chunks with scores and metadata
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dict containing model information
        """
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "load_time": self._load_time,
            "class": self.__class__.__name__
        }
    
    def ensure_loaded(self) -> bool:
        """Ensure the model is loaded, loading if necessary.
        
        Returns:
            bool: True if model is loaded successfully
        """
        if not self.is_loaded:
            start_time = time.time()
            success = self.load_model()
            if success:
                self._load_time = time.time() - start_time
            return success
        return True
    
    def __str__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}({self.model_name}) - {status}"
