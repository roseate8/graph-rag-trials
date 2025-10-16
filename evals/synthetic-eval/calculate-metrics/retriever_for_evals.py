"""
Retriever integration for evaluation - uses existing retrieval system.
Optimized for async batch processing without caching.
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm.asyncio import tqdm as async_tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from retrieval.retrieval import MilvusRetriever

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
    Wrapper around existing MilvusRetriever for batch async evaluation.
    
    Integrates with your existing retrieval system without modification.
    """
    
    def __init__(self, config):
        """
        Initialize retriever with existing system.
        
        Args:
            config: EvalConfig instance
        """
        self.config = config
        self.retriever = None
        
        logger.info(f"Initializing EvalRetriever with:")
        logger.info(f"  Collection: {config.collection_name}")
        logger.info(f"  Embedding: {config.embedding_model}")
        logger.info(f"  Re-ranking: {config.enable_reranking}")
    
    def connect(self) -> bool:
        """Connect to Milvus using existing retriever."""
        try:
            self.retriever = MilvusRetriever(
                embedding_model=self.config.embedding_model,
                milvus_profile=self.config.milvus_profile,
                collection_name=self.config.collection_name,
                enable_reranking=self.config.enable_reranking
            )
            
            success = self.retriever.connect()
            if success:
                logger.info("✓ Connected to Milvus")
                
                # Get collection stats
                stats = self.retriever.get_collection_stats()
                logger.info(f"  Total documents: {stats.get('num_entities', 'N/A')}")
            else:
                logger.error("✗ Failed to connect to Milvus")
            
            return success
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Milvus."""
        if self.retriever:
            self.retriever.disconnect()
            logger.info("Disconnected from Milvus")
    
    async def retrieve_single(self, query: Dict[str, Any], top_k: int) -> RetrievalResult:
        """
        Retrieve for a single query using existing retrieval system.
        
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
            # Call existing retrieval system
            # Note: retriever.retrieve is synchronous, wrap in executor for async
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.retriever.retrieve,
                query_text,
                top_k,
                0.0  # min_similarity (retrieve all top-k)
            )
            
            # Convert to evaluation format - FIX: use similarity_score not similarity
            retrieved_docs = [
                {
                    'chunk_id': result.chunk_id,
                    'score': result.similarity_score,
                    'rank': rank
                }
                for rank, result in enumerate(results, start=1)
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
            return RetrievalResult(
                query_id=query_id,
                query_text=query_text,
                query_type=query_type,
                retrieved_docs=[],
                success=False,
                error=str(e)
            )
    
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
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def retrieve_with_semaphore(query):
            async with semaphore:
                return await self.retrieve_single(query, top_k)
        
        # Create tasks
        tasks = [retrieve_with_semaphore(query) for query in queries]
        
        # Execute with progress bar
        if show_progress:
            results = []
            pbar = async_tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Retrieving",
                unit="query",
                ncols=100
            )
            for coro in pbar:
                result = await coro
                results.append(result)
                # Update progress with success/failure counts
                success_count = sum(1 for r in results if r.success)
                pbar.set_postfix({"success": success_count, "failed": len(results) - success_count})
        else:
            results = await asyncio.gather(*tasks)
        
        # Summary
        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count
        
        logger.info(f"✓ Retrieval complete: {success_count} success, {failed_count} failed")
        
        return results

