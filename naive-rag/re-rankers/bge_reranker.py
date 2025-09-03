"""
BGE (Beijing Academy of Artificial Intelligence) re-ranker implementation.
Uses BAAI/bge-reranker-v2-m3 model for high-quality chunk re-ranking.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import OrderedDict
import hashlib
import time

try:
    from .base import BaseReRanker, ReRankResult, ModelLoadError, ReRankingError
    from .config import ReRankerConfig
except ImportError:
    # Handle direct execution case
    from base import BaseReRanker, ReRankResult, ModelLoadError, ReRankingError
    from config import ReRankerConfig

logger = logging.getLogger(__name__)


class BGEReRanker(BaseReRanker):
    """BGE re-ranker implementation using BAAI models."""
    
    def __init__(self, config: Optional[ReRankerConfig] = None):
        """Initialize BGE re-ranker.
        
        Args:
            config: Re-ranker configuration. If None, uses default config.
        """
        self.config = config or ReRankerConfig()
        super().__init__(self.config.model_name)
        
        self.tokenizer = None
        self.model = None
        self.device = self._determine_device()
        self._score_cache = OrderedDict() if self.config.enable_caching else None
        
        logger.info(f"Initialized BGE re-ranker with device: {self.device}")
    
    def _determine_device(self) -> str:
        """Determine the best device to use."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return self.config.device
    
    def load_model(self) -> bool:
        """Load the BGE re-ranker model and tokenizer."""
        try:
            logger.info(f"Loading BGE model: {self.model_name}")
            start_time = time.time()
            
            # Import transformers here to avoid import issues if not installed
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
            except ImportError:
                raise ModelLoadError(
                    "transformers library not found. Install with: pip install transformers torch"
                )
            
            # Load tokenizer
            logger.debug("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.config.model_cache_dir
            )
            
            # Load model
            logger.debug("Loading model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.config.model_cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Successfully loaded BGE model in {load_time:.2f}s on {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            raise ModelLoadError(f"Failed to load model {self.model_name}: {e}")
    
    def _get_cache_key(self, query: str, content: str) -> str:
        """Generate cache key for query-content pair."""
        combined = f"{query}||{content}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cached_score(self, query: str, content: str) -> Optional[float]:
        """Get cached score if available."""
        if not self._score_cache:
            return None
        
        cache_key = self._get_cache_key(query, content)
        if cache_key in self._score_cache:
            # Move to end (LRU)
            self._score_cache.move_to_end(cache_key)
            return self._score_cache[cache_key]
        
        return None
    
    def _cache_score(self, query: str, content: str, score: float):
        """Cache a score for future use."""
        if not self._score_cache:
            return
        
        cache_key = self._get_cache_key(query, content)
        
        # Evict oldest if cache is full
        if len(self._score_cache) >= self.config.cache_size:
            self._score_cache.popitem(last=False)
        
        self._score_cache[cache_key] = score
    
    def _prepare_input_pairs(self, query: str, chunks: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Prepare query-content pairs for batch processing - optimized."""
        # Pre-calculate max content length once
        max_content_len = self.config.max_length * 4
        
        # Use list comprehension for better performance
        return [
            (query, chunk.get('content', '')[:max_content_len] if len(chunk.get('content', '')) > max_content_len 
             else chunk.get('content', ''))
            for chunk in chunks
        ]
    
    def _compute_scores_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute re-ranking scores for a batch of query-content pairs - optimized."""
        if not self.is_loaded:
            raise ReRankingError("Model not loaded. Call load_model() first.")
        
        try:
            # Pre-allocate arrays for better memory efficiency
            num_pairs = len(pairs)
            scores = [None] * num_pairs
            uncached_pairs = []
            uncached_indices = []
            
            # Batch cache lookup optimization - single pass
            if self._score_cache:
                for i, (query, content) in enumerate(pairs):
                    cache_key = self._get_cache_key(query, content)
                    if cache_key in self._score_cache:
                        # Move to end (LRU) and get score in one operation
                        self._score_cache.move_to_end(cache_key)
                        scores[i] = self._score_cache[cache_key]
                    else:
                        uncached_pairs.append((query, content))
                        uncached_indices.append(i)
            else:
                # No caching - prepare all pairs
                uncached_pairs = pairs
                uncached_indices = list(range(num_pairs))
            
            # Process uncached pairs in optimized batches
            if uncached_pairs:
                logger.debug(f"Computing scores for {len(uncached_pairs)} uncached pairs")
                
                # Optimize tokenization with pre-allocated tensors
                inputs = self.tokenizer(
                    uncached_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Optimized inference with minimal memory allocation
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Extract scores efficiently using tensor operations
                    batch_scores = outputs.logits[:, 1].cpu().numpy()
                
                # Batch cache update - minimize dict operations
                if self._score_cache:
                    cache_updates = {}
                    for idx, score in zip(uncached_indices, batch_scores):
                        query, content = pairs[idx]
                        cache_key = self._get_cache_key(query, content)
                        cache_updates[cache_key] = float(score)
                        scores[idx] = float(score)
                    
                    # Batch update cache and handle eviction
                    cache_size = len(self._score_cache)
                    new_items = len(cache_updates)
                    if cache_size + new_items > self.config.cache_size:
                        # Evict oldest items efficiently
                        evict_count = (cache_size + new_items) - self.config.cache_size
                        for _ in range(evict_count):
                            self._score_cache.popitem(last=False)
                    
                    # Add new items
                    self._score_cache.update(cache_updates)
                else:
                    # No caching - just fill scores
                    for idx, score in zip(uncached_indices, batch_scores):
                        scores[idx] = float(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error computing re-ranking scores: {e}")
            raise ReRankingError(f"Failed to compute scores: {e}")
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        top_k: int = 10
    ) -> List[ReRankResult]:
        """Re-rank chunks based on relevance to query - optimized.
        
        Args:
            query: The user query
            chunks: List of chunk dictionaries with content and metadata
            top_k: Number of top chunks to return after re-ranking
            
        Returns:
            List[ReRankResult]: Re-ranked chunks with scores and metadata
        """
        if not chunks:
            return []
        
        if not self.is_loaded:
            logger.warning("Model not loaded, loading now...")
            self.load_model()
        
        logger.debug(f"Re-ranking {len(chunks)} chunks for query: {query[:50]}...")
        start_time = time.time()
        
        try:
            # Prepare input pairs - single pass
            pairs = self._prepare_input_pairs(query, chunks)
            
            # Compute scores in optimized batches
            batch_size = self.config.batch_size
            num_pairs = len(pairs)
            
            if num_pairs <= batch_size:
                # Single batch optimization
                all_scores = self._compute_scores_batch(pairs)
            else:
                # Multiple batches with pre-allocated result array
                all_scores = [0.0] * num_pairs
                for i in range(0, num_pairs, batch_size):
                    end_idx = min(i + batch_size, num_pairs)
                    batch_pairs = pairs[i:end_idx]
                    batch_scores = self._compute_scores_batch(batch_pairs)
                    all_scores[i:end_idx] = batch_scores
            
            # Optimized score normalization
            if self.config.normalize_scores and all_scores:
                min_score = min(all_scores)
                max_score = max(all_scores)
                if max_score > min_score:
                    score_range = max_score - min_score
                    all_scores = [(s - min_score) / score_range for s in all_scores]
            
            # Create results with optimized list comprehension and zip
            results = [
                ReRankResult(
                    chunk_id=chunk.get('chunk_id', f'chunk_{i}'),
                    original_rank=i,
                    rerank_score=score,
                    final_rank=0,  # Will be set after sorting
                    content=chunk.get('content', ''),
                    metadata=chunk.get('metadata', {})
                )
                for i, (chunk, score) in enumerate(zip(chunks, all_scores))
            ]
            
            # Optimized sorting using key extraction
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Set final ranks only for top_k results (avoid unnecessary work)
            top_results = results[:top_k]
            for final_rank, result in enumerate(top_results):
                result.final_rank = final_rank
            
            rerank_time = time.time() - start_time
            logger.info(f"Re-ranked {len(chunks)} chunks in {rerank_time:.2f}s, returning top {len(top_results)}")
            
            return top_results
            
        except Exception as e:
            logger.error(f"Error during re-ranking: {e}")
            raise ReRankingError(f"Re-ranking failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the BGE model."""
        info = super().get_model_info()
        info.update({
            "device": self.device,
            "config": self.config.to_dict(),
            "cache_size": len(self._score_cache) if self._score_cache else 0,
            "cache_enabled": self.config.enable_caching,
            "optimizations": {
                "batch_processing": True,
                "cache_optimization": True,
                "memory_efficient": True,
                "lazy_loading": True
            }
        })
        return info
    
    def clear_cache(self):
        """Clear the score cache."""
        if self._score_cache:
            self._score_cache.clear()
            logger.info("Cleared re-ranking score cache")


def create_bge_reranker(config: Optional[ReRankerConfig] = None) -> BGEReRanker:
    """Factory function to create a BGE re-ranker."""
    return BGEReRanker(config)
