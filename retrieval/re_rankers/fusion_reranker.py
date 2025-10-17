"""
Fusion re-ranking using Reciprocal Rank Fusion (RRF) for multi-query retrieval.

This module implements RRF to combine and re-rank chunks retrieved from multiple
sub-queries, giving higher scores to chunks that appear relevant across multiple queries.
"""

import logging
import time
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    from ..retrieval import RetrievedChunk
except ImportError:
    try:
        from retrieval import RetrievedChunk
    except ImportError:
        import sys
        from pathlib import Path
        parent_path = Path(__file__).parent.parent
        if str(parent_path) not in sys.path:
            sys.path.insert(0, str(parent_path))
        from retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Container for fusion re-ranking result with detailed scoring information."""
    chunk: RetrievedChunk
    fusion_score: float
    sub_query_ranks: Dict[str, int]  # sub-query -> rank in that query's results
    sub_query_scores: Dict[str, float]  # sub-query -> rerank score for that query
    appearance_count: int  # How many sub-queries this chunk appeared in
    
    def __str__(self) -> str:
        return f"FusionResult(fusion_score={self.fusion_score:.4f}, appearances={self.appearance_count})"


class FusionReranker:
    """
    Reciprocal Rank Fusion (RRF) re-ranker for multi-query retrieval.
    
    RRF Formula: score = Σ(1 / (k + rank_i))
    where k is a constant (typically 60) and rank_i is the rank in query i's results.
    
    Benefits:
    - Automatically normalizes scores across different queries
    - Boosts chunks appearing in multiple query results
    - More robust than simple score averaging
    - Proven effective in multi-query retrieval scenarios
    """
    
    def __init__(self, k_constant: int = 60):
        """
        Initialize fusion re-ranker.
        
        Args:
            k_constant: RRF constant (typically 60). Higher values reduce the impact
                       of rank differences, lower values amplify them.
        """
        self.k_constant = k_constant
        logger.info(f"Initialized FusionReranker (k={k_constant})")
    
    def fuse_and_rerank(
        self,
        sub_queries: List[str],
        chunk_results: Dict[str, List[RetrievedChunk]],
        top_k: int,
        reranker: Optional[Any] = None
    ) -> List[RetrievedChunk]:
        """Fuse and re-rank chunks using RRF algorithm."""
        start_time = time.time()
        
        # Optimized: Single-pass deduplication and appearance tracking
        unique_chunks, chunk_appearances = self._deduplicate_and_track(chunk_results)
        
        if not unique_chunks:
            logger.warning("No chunks to fuse")
            return []
        
        logger.info(f"Fusing {len(unique_chunks)} unique chunks from {len(sub_queries)} queries")
        
        # Calculate RRF scores directly (skip re-ranking if no reranker)
        if reranker and hasattr(reranker, 'rerank'):
            chunk_scores = self._rerank_against_all_queries(unique_chunks, sub_queries, reranker)
        else:
            chunk_scores = None  # Use rank-based RRF only
        
        # Calculate fusion scores and build final chunks in one pass
        final_chunks = self._calculate_and_build_final(
            unique_chunks,
            chunk_appearances,
            top_k
        )
        
        logger.info(f"Fusion completed in {time.time() - start_time:.2f}s: {len(final_chunks)} chunks")
        return final_chunks
    
    def _deduplicate_and_track(
        self,
        chunk_results: Dict[str, List[RetrievedChunk]]
    ) -> Tuple[Dict[str, RetrievedChunk], Dict[str, Dict[str, int]]]:
        """
        Optimized: Deduplicate and track appearances in single pass.
        
        Returns:
            Tuple of (unique_chunks, chunk_appearances)
        """
        unique_chunks = {}
        chunk_appearances = defaultdict(dict)
        
        for sub_query, chunks in chunk_results.items():
            for rank, chunk in enumerate(chunks, 1):  # 1-indexed ranks
                chunk_id = chunk.chunk_id
                
                # Store first occurrence
                if chunk_id not in unique_chunks:
                    unique_chunks[chunk_id] = chunk
                
                # Track rank in this query
                chunk_appearances[chunk_id][sub_query] = rank
        
        return unique_chunks, dict(chunk_appearances)
    
    def _build_chunk_appearances(
        self,
        chunk_results: Dict[str, List[RetrievedChunk]]
    ) -> Dict[str, Dict[str, int]]:
        """
        Build mapping of chunk_id -> {sub_query -> rank}.
        
        Args:
            chunk_results: Dict of sub-query -> chunks
            
        Returns:
            Dict of chunk_id -> {sub_query -> rank in that query's results}
        """
        chunk_appearances = defaultdict(dict)
        
        for sub_query, chunks in chunk_results.items():
            for rank, chunk in enumerate(chunks):
                # Rank is 0-indexed, but for RRF we use 1-indexed ranks
                chunk_appearances[chunk.chunk_id][sub_query] = rank + 1
        
        return dict(chunk_appearances)
    
    def _rerank_against_all_queries(
        self,
        unique_chunks: Dict[str, RetrievedChunk],
        sub_queries: List[str],
        reranker: Any
    ) -> Dict[str, Dict[str, float]]:
        """
        Re-rank each unique chunk against ALL sub-queries.
        
        Args:
            unique_chunks: Dict of chunk_id -> chunk
            sub_queries: List of all sub-queries
            reranker: Reranker model to use
            
        Returns:
            Dict of chunk_id -> {sub_query -> rerank_score}
        """
        logger.info(f"Re-ranking {len(unique_chunks)} chunks against {len(sub_queries)} sub-queries")
        
        chunk_scores = {}
        
        # Ensure reranker model is loaded
        if not reranker.ensure_loaded():
            logger.error("Failed to load re-ranker model")
            raise RuntimeError("Reranker model failed to load")
        
        # For each sub-query, re-rank all unique chunks
        for sub_query in sub_queries:
            logger.debug(f"Re-ranking chunks for: {sub_query[:50]}...")
            
            # Convert chunks to format expected by reranker
            chunk_dicts = [
                {
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'metadata': {
                        'doc_id': chunk.doc_id,
                        'word_count': chunk.word_count,
                        'section_path': chunk.section_path,
                        'original_similarity': chunk.similarity_score
                    }
                }
                for chunk in unique_chunks.values()
            ]
            
            # Re-rank all chunks for this sub-query
            # Note: We pass len(unique_chunks) as top_k to get scores for ALL chunks
            rerank_results = reranker.rerank(sub_query, chunk_dicts, top_k=len(unique_chunks))
            
            # Store scores for this sub-query
            for result in rerank_results:
                if result.chunk_id not in chunk_scores:
                    chunk_scores[result.chunk_id] = {}
                chunk_scores[result.chunk_id][sub_query] = result.rerank_score
        
        logger.debug(f"Collected re-rank scores for {len(chunk_scores)} chunks")
        return chunk_scores
    
    def _use_original_scores(
        self,
        unique_chunks: Dict[str, RetrievedChunk],
        chunk_appearances: Dict[str, Dict[str, int]],
        chunk_results: Dict[str, List[RetrievedChunk]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Use original similarity scores when reranker is not available.
        
        Args:
            unique_chunks: Dict of chunk_id -> chunk
            chunk_appearances: Chunk appearance information
            chunk_results: Original chunk results
            
        Returns:
            Dict of chunk_id -> {sub_query -> score}
        """
        chunk_scores = {}
        
        for chunk_id, chunk in unique_chunks.items():
            chunk_scores[chunk_id] = {}
            
            # For each sub-query this chunk appeared in, use its similarity score
            for sub_query, rank in chunk_appearances.get(chunk_id, {}).items():
                # Find the chunk in the original results
                for original_chunk in chunk_results[sub_query]:
                    if original_chunk.chunk_id == chunk_id:
                        chunk_scores[chunk_id][sub_query] = original_chunk.similarity_score
                        break
        
        return chunk_scores
    
    def _calculate_fusion_scores(
        self,
        unique_chunks: Dict[str, RetrievedChunk],
        chunk_scores: Dict[str, Dict[str, float]],
        chunk_appearances: Dict[str, Dict[str, int]]
    ) -> List[FusionResult]:
        """
        Calculate RRF fusion scores for all chunks.
        
        RRF Formula: score = Σ(1 / (k + rank_i))
        
        Args:
            unique_chunks: Dict of chunk_id -> chunk
            chunk_scores: Dict of chunk_id -> {sub_query -> score}
            chunk_appearances: Dict of chunk_id -> {sub_query -> rank}
            
        Returns:
            List of FusionResult objects
        """
        fusion_results = []
        
        for chunk_id, chunk in unique_chunks.items():
            # Get appearances and scores for this chunk
            appearances = chunk_appearances.get(chunk_id, {})
            scores = chunk_scores.get(chunk_id, {})
            
            if not appearances:
                # Skip chunks with no appearances (shouldn't happen)
                logger.warning(f"Chunk {chunk_id} has no appearance data, skipping")
                continue
            
            # Calculate RRF score
            rrf_score = self._calculate_rrf_score(appearances)
            
            # Create fusion result
            fusion_result = FusionResult(
                chunk=chunk,
                fusion_score=rrf_score,
                sub_query_ranks=appearances,
                sub_query_scores=scores,
                appearance_count=len(appearances)
            )
            
            fusion_results.append(fusion_result)
        
        return fusion_results
    
    def _calculate_and_build_final(
        self,
        unique_chunks: Dict[str, RetrievedChunk],
        chunk_appearances: Dict[str, Dict[str, int]],
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        Optimized: Calculate RRF scores and build final chunks in one pass.
        Uses heapq for efficient top-K selection.
        """
        import heapq
        
        # Use min-heap for efficient top-K (negate scores for max-heap behavior)
        heap = []
        
        for chunk_id, chunk in unique_chunks.items():
            appearances = chunk_appearances.get(chunk_id, {})
            if not appearances:
                continue
            
            # Calculate RRF score inline
            rrf_score = sum(1.0 / (self.k_constant + rank) 
                           for rank in appearances.values())
            
            # Maintain heap of size top_k
            if len(heap) < top_k:
                heapq.heappush(heap, (rrf_score, chunk_id, chunk))
            elif rrf_score > heap[0][0]:
                heapq.heapreplace(heap, (rrf_score, chunk_id, chunk))
        
        # Extract and sort final results
        top_results = sorted(heap, key=lambda x: x[0], reverse=True)
        
        # Build final chunks with fusion scores
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                word_count=chunk.word_count,
                section_path=chunk.section_path,
                similarity_score=chunk.similarity_score,
                rerank_score=score,  # Store fusion score
                rerank_probability=None,
                chunk_type=chunk.chunk_type,
                regions=chunk.regions,
                product_version=chunk.product_version,
                folder_path=chunk.folder_path,
                structural_metadata=chunk.structural_metadata,
                entity_metadata=chunk.entity_metadata
            )
            for score, _, chunk in top_results
        ]
    
    def _calculate_rrf_score(self, chunk_ranks: Dict[str, int]) -> float:
        """Calculate RRF: score = Σ(1 / (k + rank_i))"""
        return sum(1.0 / (self.k_constant + rank) for rank in chunk_ranks.values())
    
    def _build_final_chunks(self, fusion_results: List[FusionResult]) -> List[RetrievedChunk]:
        """
        Convert FusionResult objects back to RetrievedChunk with fusion scores.
        
        Args:
            fusion_results: List of FusionResult objects
            
        Returns:
            List of RetrievedChunk with fusion scores as rerank_score
        """
        final_chunks = []
        
        for fusion_result in fusion_results:
            chunk = fusion_result.chunk
            
            # Create new chunk with fusion score stored as rerank_score
            final_chunk = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                word_count=chunk.word_count,
                section_path=chunk.section_path,
                similarity_score=chunk.similarity_score,  # Keep original
                rerank_score=fusion_result.fusion_score,  # Store fusion score
                rerank_probability=None,  # Not applicable for fusion
                chunk_type=chunk.chunk_type,
                regions=chunk.regions,
                product_version=chunk.product_version,
                folder_path=chunk.folder_path,
                structural_metadata=chunk.structural_metadata,
                entity_metadata=chunk.entity_metadata
            )
            
            final_chunks.append(final_chunk)
        
        return final_chunks


def fuse_results(
    sub_queries: List[str],
    chunk_results: Dict[str, List[RetrievedChunk]],
    top_k: int = 10,
    k_constant: int = 60,
    reranker: Optional[Any] = None
) -> List[RetrievedChunk]:
    """
    Simple function to fuse multi-query results without managing reranker lifecycle.
    
    Args:
        sub_queries: List of sub-queries
        chunk_results: Dict of sub_query -> chunks
        top_k: Number of top chunks to return
        k_constant: RRF k constant
        reranker: Optional reranker instance
        
    Returns:
        List of top K fused chunks
    """
    fusion_reranker = FusionReranker(k_constant=k_constant)
    return fusion_reranker.fuse_and_rerank(sub_queries, chunk_results, top_k, reranker)

