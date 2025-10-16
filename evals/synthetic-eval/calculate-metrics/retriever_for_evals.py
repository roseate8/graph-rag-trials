"""
Retriever integration for evaluation - uses existing retrieval system via RAGSystem.
Optimized for async batch processing without caching.

This module wraps the main RAGSystem from retrieval/core.py, ensuring that any
changes to the retrieval pipeline (query decomposition, re-ranking, etc.) automatically
flow into evaluation without requiring code changes here.
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from retrieval.core import RAGSystem

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval for a single query."""
    query_id: str
    query_text: str
    query_type: str  # single_hop or multi_hop
    retrieved_docs: List[Dict[str, Any]]  # List of {chunk_id, score, rank}
    success: bool
    error: Optional[str] = None


class EvalRetriever:
    """
    Wrapper around RAGSystem for batch async evaluation.
    
    Uses the complete retrieval pipeline from retrieval/core.py, ensuring
    all features (query decomposition, re-ranking, etc.) are automatically included.
    """
    
    def __init__(self, config):
        """
        Initialize retriever with RAGSystem.

        Args:
            config: EvalConfig instance
        """
        self.config = config
        self.rag_system = None

        # Extract params for logging
        params = config.rag_system_params
        logger.info(f"Initializing EvalRetriever with RAGSystem:")
        logger.info(f"  Collection: {params.get('collection_name', 'N/A')}")
        logger.info(f"  Embedding: {params.get('embedding_model', 'N/A')}")
        logger.info(f"  Re-ranking: {params.get('enable_reranking', False)}")
        logger.info(f"  Query decomposition: {params.get('enable_query_decomposition', False)}")
    
    def connect(self) -> bool:
        """Connect to Milvus using RAGSystem."""
        try:
            # Initialize RAGSystem with **kwargs pass-through
            # This ensures 100% dependency on retrieval/core.py
            # Any new parameters added to RAGSystem automatically work here!
            self.rag_system = RAGSystem(**self.config.rag_system_params)

            success = self.rag_system.connect()
            if success:
                logger.info("✓ Connected to Milvus via RAGSystem")

                # Get collection stats
                stats = self.rag_system.get_system_stats()
                retriever_stats = stats.get('retriever_stats', {})
                logger.info(f"  Total documents: {retriever_stats.get('num_entities', 'N/A')}")
            else:
                logger.error("✗ Failed to connect to Milvus")

            return success
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Milvus."""
        if self.rag_system:
            self.rag_system.disconnect()
            logger.info("Disconnected from Milvus")

    def _create_failed_result(self, query: Dict[str, Any], error_msg: str) -> RetrievalResult:
        """
        Helper to create a failed RetrievalResult. Reduces code duplication.

        Args:
            query: Query dict
            error_msg: Error message

        Returns:
            RetrievalResult with success=False
        """
        return RetrievalResult(
            query_id=query.get('_id', ''),
            query_text=query.get('text', ''),
            query_type=query.get('metadata', {}).get('query_type', 'unknown'),
            retrieved_docs=[],
            success=False,
            error=error_msg
        )

    async def retrieve_single(self, query: Dict[str, Any], top_k: int) -> RetrievalResult:
        """
        Retrieve for a single query using RAGSystem.
        
        This method uses the complete RAG pipeline including:
        - Query decomposition (if enabled)
        - Multi-query retrieval with fusion (if decomposition enabled)
        - Re-ranking (if enabled)
        
        Args:
            query: Query dict with _id, text, metadata
            top_k: Number of documents to retrieve
            
        Returns:
            RetrievalResult with retrieved documents
        """
        query_id = query.get('_id', '')
        query_text = query.get('text', '')
        query_type = query.get('metadata', {}).get('query_type', 'unknown')
        
        try:
            # Call RAGSystem.query which handles the complete retrieval pipeline
            # Note: RAGSystem.query is synchronous, wrap in executor for async
            loop = asyncio.get_event_loop()
            rag_result = await loop.run_in_executor(
                None,
                self.rag_system.query,
                query_text,
                top_k,
                0.0  # min_similarity (retrieve all top-k)
            )
            
            # Convert RAGResult.retrieved_chunks to evaluation format
            # Prefer rerank_score when available (from re-ranking), otherwise use similarity_score
            retrieved_docs = [
                {
                    'chunk_id': chunk.chunk_id,
                    'score': chunk.rerank_score if chunk.rerank_score is not None else chunk.similarity_score,
                    'rank': rank,
                    # Include additional metadata for analysis
                    'similarity_score': chunk.similarity_score,
                    'rerank_score': chunk.rerank_score,
                    'rerank_probability': chunk.rerank_probability
                }
                for rank, chunk in enumerate(rag_result.retrieved_chunks, start=1)
            ]
            
            return RetrievalResult(
                query_id=query_id,
                query_text=query_text,
                query_type=query_type,
                retrieved_docs=retrieved_docs,
                success=True
            )
        
        except Exception as e:
            logger.error(f"Error retrieving for query {query_id}: {e}")
            return self._create_failed_result(query, str(e))
    
    async def retrieve_batch(
        self,
        queries: List[Dict[str, Any]],
        top_k: int,
        show_progress: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve for multiple queries with async batching.

        Args:
            queries: List of query dicts
            top_k: Number of documents to retrieve per query
            show_progress: Show progress bar

        Returns:
            List of RetrievalResult objects
        """
        logger.info(f"Processing {len(queries)} queries (batch_size={self.config.batch_size}, top_k={top_k})")

        # Suppress noisy loggers during batch processing
        old_levels = {}
        noisy_loggers = [
            'embeddings.milvus_store',
            'retrieval.retrieval',     # RAGSystem uses MilvusRetriever internally
            'retrieval.core',
            'retrieval.decomposer',    # Query decomposer (if enabled)
            'pymilvus',
            'handler'
        ]
        for logger_name in noisy_loggers:
            noisy_logger = logging.getLogger(logger_name)
            old_levels[logger_name] = noisy_logger.level
            noisy_logger.setLevel(logging.CRITICAL)  # Suppress everything except critical

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def retrieve_with_semaphore(query):
            async with semaphore:
                return await self.retrieve_single(query, top_k)

        # Create tasks
        tasks = [retrieve_with_semaphore(query) for query in queries]

        # Execute with simple progress - gather all at once for cleaner output
        if show_progress:
            from tqdm import tqdm
            results = []

            # Use tqdm with manual updates for cleaner async handling
            with tqdm(total=len(queries), desc="Retrieving", unit="query", ncols=80) as pbar:
                # Process in batches to avoid overwhelming the progress bar
                for i in range(0, len(tasks), self.config.batch_size):
                    batch = tasks[i:i + self.config.batch_size]
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)

                    # Optimized: handle exceptions with helper function
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            query_idx = i + j
                            query = queries[query_idx]
                            results.append(self._create_failed_result(query, str(result)))
                        else:
                            results.append(result)

                    pbar.update(len(batch))

                    # Update postfix with current stats
                    success_count = sum(1 for r in results if r.success)
                    failed_count = len(results) - success_count
                    pbar.set_postfix({"ok": success_count, "fail": failed_count}, refresh=False)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Optimized: handle exceptions with list comprehension
            results = [
                self._create_failed_result(queries[i], str(result))
                if isinstance(result, Exception)
                else result
                for i, result in enumerate(results)
            ]

        # Restore logger levels
        for logger_name, old_level in old_levels.items():
            logging.getLogger(logger_name).setLevel(old_level)

        # Summary
        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count

        logger.info(f"Retrieval complete: {success_count} success, {failed_count} failed")

        return results

