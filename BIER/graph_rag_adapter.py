"""
BEIR adapter for GraphRAG pipeline integration.

This adapter wraps the existing MilvusRetriever to make it compatible with BEIR's
evaluation framework while maintaining all existing functionality.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / "naive-rag"))
sys.path.append(str(project_root / "vector-ingest" / "src"))

try:
    from retrieval import MilvusRetriever, RetrievedChunk
except ImportError as e:
    raise ImportError(f"Could not import MilvusRetriever: {e}. Ensure naive-rag module is available.")

logger = logging.getLogger(__name__)


class GraphRAGAdapter:
    """
    BEIR adapter for GraphRAG pipeline retrieval system.
    
    This adapter implements the interface expected by BEIR evaluation framework
    while using our existing MilvusRetriever backend.
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        milvus_profile: str = "production", 
        collection_name: str = "document_chunks",
        enable_reranking: bool = True,
        reranker_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize GraphRAG adapter.
        
        Args:
            embedding_model: SentenceTransformer model for embeddings
            milvus_profile: Milvus configuration profile
            collection_name: Milvus collection name
            enable_reranking: Whether to enable re-ranking
            reranker_config: Re-ranker configuration
            **kwargs: Additional arguments passed to MilvusRetriever
        """
        self.embedding_model = embedding_model
        self.milvus_profile = milvus_profile
        self.collection_name = collection_name
        self.enable_reranking = enable_reranking
        
        # Initialize the underlying retriever
        logger.info(f"Initializing GraphRAG adapter with model: {embedding_model}")
        self.retriever = MilvusRetriever(
            embedding_model=embedding_model,
            milvus_profile=milvus_profile,
            collection_name=collection_name,
            enable_reranking=enable_reranking,
            reranker_config=reranker_config,
            **kwargs
        )
        
        logger.info("GraphRAG adapter initialized successfully")
    
    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str], 
        top_k: int,
        score_function: str = "dot",
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Search interface compatible with BEIR framework.
        
        Args:
            corpus: BEIR corpus (not used - we search our own vector store)
            queries: Dict mapping query_id -> query_text
            top_k: Number of results to return per query
            score_function: Scoring function (ignored - we use our own similarity)
            **kwargs: Additional search parameters
            
        Returns:
            Dict mapping query_id -> {doc_id: score}
        """
        logger.info(f"Starting BEIR search for {len(queries)} queries, top_k={top_k}")
        
        results = {}
        
        for query_id, query_text in queries.items():
            try:
                # Use our existing retrieval pipeline
                chunks = self.retriever.retrieve(
                    query=query_text,
                    top_k=top_k,
                    min_similarity=kwargs.get('min_similarity', 0.0),
                    retrieval_multiplier=kwargs.get('retrieval_multiplier', 10)
                )
                
                # Convert to BEIR format: {doc_id: score}
                query_results = {}
                for chunk in chunks:
                    # Use rerank score if available, otherwise similarity score
                    score = chunk.rerank_score if chunk.rerank_score is not None else chunk.similarity_score
                    
                    # BEIR expects positive scores, convert if needed
                    if score < 0:
                        score = 1.0 / (1.0 + abs(score))  # Convert to positive similarity
                    
                    query_results[chunk.chunk_id] = float(score)
                
                results[query_id] = query_results
                logger.debug(f"Query {query_id}: Retrieved {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing query {query_id}: {e}")
                results[query_id] = {}  # Empty results on error
        
        logger.info(f"Completed BEIR search for {len(queries)} queries")
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[RetrievedChunk]:
        """
        Direct retrieval interface (non-BEIR).
        
        Args:
            query: Query text
            top_k: Number of results to return
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of RetrievedChunk objects
        """
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            **kwargs
        )
    
    def get_corpus_info(self) -> Dict[str, int]:
        """
        Get information about the indexed corpus.
        
        Returns:
            Dict with corpus statistics
        """
        try:
            # Try to get collection statistics from Milvus
            store = self.retriever.vector_store
            if hasattr(store, 'get_collection_stats'):
                stats = store.get_collection_stats()
                return {
                    'num_documents': stats.get('num_entities', 0),
                    'collection_name': self.collection_name
                }
            else:
                return {
                    'num_documents': 0,
                    'collection_name': self.collection_name
                }
        except Exception as e:
            logger.warning(f"Could not get corpus info: {e}")
            return {
                'num_documents': 0,
                'collection_name': self.collection_name
            }
    
    def close(self):
        """Clean up resources."""
        try:
            if hasattr(self.retriever, 'close'):
                self.retriever.close()
            logger.info("GraphRAG adapter closed successfully")
        except Exception as e:
            logger.warning(f"Error closing adapter: {e}")


class BEIRCompatibleAdapter:
    """
    Alternative adapter that provides more direct BEIR compatibility.
    
    This version focuses on exact BEIR interface compliance for evaluation.
    """
    
    def __init__(self, graph_rag_adapter: GraphRAGAdapter):
        """Initialize with a GraphRAG adapter instance."""
        self.adapter = graph_rag_adapter
    
    def search(
        self, 
        corpus: Dict[str, Dict[str, str]], 
        queries: Dict[str, str], 
        top_k: int, 
        score_function: str = "dot", 
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """BEIR-compatible search interface."""
        return self.adapter.search(corpus, queries, top_k, score_function, **kwargs)