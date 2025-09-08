"""
Document retrieval with optimized caching - combines retriever.py + cache.py functionality.
"""

import sys
import logging
import hashlib
import time
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Setup vector-ingest imports
vector_ingest_path = Path(__file__).parent.parent / "vector-ingest" / "src"
sys.path.append(str(vector_ingest_path))

from embeddings.embedding_service import EmbeddingService
from embeddings.milvus_config import get_config
from embeddings.milvus_store import MilvusVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Container for retrieved chunk with metadata."""
    chunk_id: str
    doc_id: str
    content: str
    word_count: int
    section_path: str
    similarity_score: float
    rerank_score: Optional[float] = None  # Re-ranking logit score
    rerank_probability: Optional[float] = None  # Re-ranking probability
    
    def __str__(self) -> str:
        return f"[{self.doc_id}] {self.content[:100]}..."


class EmbeddingCache:
    """Optimized O(1) cache for query embeddings using OrderedDict."""
    
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()  # O(1) operations
        self.max_size = max_size
        self._hash_cache = {}  # Cache hash computations
    
    def _get_query_hash(self, query: str) -> str:
        """Get cached hash or compute new one."""
        if query not in self._hash_cache:
            # Limit hash cache size to prevent memory leaks
            if len(self._hash_cache) >= self.max_size * 2:
                self._hash_cache.clear()
            self._hash_cache[query] = hashlib.md5(query.encode('utf-8')).hexdigest()
        return self._hash_cache[query]
    
    def get(self, query: str) -> Optional[List[float]]:
        """Get cached embedding for query - O(1) with hash caching."""
        query_hash = self._get_query_hash(query)
        
        if query_hash in self.cache:
            # Move to end (most recently used) - O(1)
            self.cache.move_to_end(query_hash)
            return self.cache[query_hash]
        
        return None
    
    def put(self, query: str, embedding: List[float]):
        """Cache embedding for query - O(1) with optimized eviction."""
        query_hash = self._get_query_hash(query)
        
        if query_hash in self.cache:
            # Update existing and move to end - O(1)
            self.cache[query_hash] = embedding
            self.cache.move_to_end(query_hash)
        else:
            # Evict oldest if needed - O(1)
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # Remove oldest
            self.cache[query_hash] = embedding
    
    def clear(self):
        """Clear all cached embeddings and hashes."""
        self.cache.clear()
        self._hash_cache.clear()


def optimize_numpy_to_list(arr) -> List[float]:
    """Optimized conversion from numpy array to list with lazy imports."""
    # Lazy import for better startup performance
    try:
        import numpy as np
        
        if isinstance(arr, np.ndarray):
            # Optimized conversion based on data type
            if arr.dtype == np.float32:
                return arr.astype(np.float64).tolist()  # More efficient conversion
            elif arr.dtype in (np.float64, np.int32, np.int64):
                return arr.tolist()
            else:
                # Fallback for other types
                return [float(x) for x in arr]
        elif hasattr(arr, 'tolist'):
            return arr.tolist()
        else:
            return list(arr)
    except ImportError:
        # Fallback if numpy not available
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        else:
            return list(arr)


class MilvusRetriever:
    """Optimized retriever that searches Milvus for relevant document chunks with optional re-ranking."""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        milvus_profile: str = "production",
        collection_name: str = "document_chunks",
        enable_reranking: bool = False,
        reranker_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the retriever."""
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.milvus_config = get_config(milvus_profile)
        self.milvus_config.collection_name = collection_name
        self.milvus_store = MilvusVectorStore(self.milvus_config)
        self.connected = False
        self._cache = EmbeddingCache()
        
        # Re-ranking configuration
        self.enable_reranking = enable_reranking
        self.reranker = None
        if enable_reranking:
            self._init_reranker(reranker_config)
        
        logger.info(f"Initialized MilvusRetriever for collection: {collection_name} (re-ranking: {enable_reranking})")
    
    def _init_reranker(self, reranker_config: Optional[Dict[str, Any]] = None):
        """Initialize the re-ranker."""
        try:
            # Fix import path issue - use direct sys.path approach
            import sys
            from pathlib import Path
            re_rankers_path = Path(__file__).parent / "re-rankers"
            if str(re_rankers_path) not in sys.path:
                sys.path.insert(0, str(re_rankers_path))
            from reranker_model import create_reranker
            from config import ReRankerConfig
            
            # Create config from provided dict or use defaults
            if reranker_config:
                config = ReRankerConfig.from_dict(reranker_config)
            else:
                config = ReRankerConfig()
            
            # Create the re-ranker
            self.reranker = create_reranker(config)
            logger.info("Initialized CrossEncoder re-ranker")
            
        except ImportError as e:
            logger.error(f"Failed to import re-ranker: {e}")
            self.enable_reranking = False
            self.reranker = None
        except Exception as e:
            logger.error(f"Failed to initialize re-ranker: {e}")
            self.enable_reranking = False
            self.reranker = None
    
    def connect(self) -> bool:
        """Connect to Milvus database."""
        try:
            if self.milvus_store.connect():
                # Initialize collection to ensure it exists and is loaded
                if self.milvus_store.create_collection():
                    self.connected = True
                    logger.info("Successfully connected to Milvus")
                    return True
                else:
                    logger.error("Failed to initialize collection")
                    return False
            else:
                logger.error("Failed to connect to Milvus")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Milvus."""
        if self.connected:
            self.milvus_store.disconnect()
            self.connected = False
            logger.info("Disconnected from Milvus")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        retrieval_multiplier: int = 10
    ) -> List[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query with optional re-ranking.
        
        Args:
            query: The user query
            top_k: Final number of chunks to return
            min_similarity: Minimum similarity threshold
            retrieval_multiplier: Multiplier for initial retrieval when re-ranking is enabled
                                (e.g., retrieval_multiplier=10 means retrieve 10*top_k chunks first)
        """
        if not self.connected:
            logger.error("Not connected to Milvus. Call connect() first.")
            return []
        
        try:
            # Determine initial retrieval count
            if self.enable_reranking and self.reranker:
                initial_top_k = top_k * retrieval_multiplier
                logger.debug(f"Re-ranking enabled: retrieving {initial_top_k} chunks for re-ranking to top {top_k}")
            else:
                initial_top_k = top_k
                logger.debug(f"Re-ranking disabled: retrieving {initial_top_k} chunks directly")
            
            # Generate embedding for the query (with caching)
            logger.debug(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self._get_query_embedding(query)
            
            # Search Milvus for similar chunks
            search_results = self.milvus_store.search_similar(
                query_embedding=query_embedding,
                top_k=initial_top_k,
                output_fields=["chunk_id", "doc_id", "content", "word_count", "section_path"]
            )
            
            # Check for duplicates in Milvus search results
            milvus_ids = [result.get("chunk_id") for result in search_results if result.get("chunk_id")]
            unique_milvus_ids = set(milvus_ids)
            if len(milvus_ids) != len(unique_milvus_ids):
                duplicate_count = len(milvus_ids) - len(unique_milvus_ids)
                logger.warning(f"Milvus returned {duplicate_count} duplicate chunk IDs out of {len(milvus_ids)} results")
            
            # Convert to RetrievedChunk objects and filter by similarity - Optimized O(k)
            retrieved_chunks = []
            for result in search_results:
                # Fast similarity check first (avoid multiple dict lookups)
                similarity_score = result.get("similarity_score", 0)
                if similarity_score >= min_similarity:
                    try:
                        # Single-pass extraction with defaults - fewer dict lookups
                        chunk = RetrievedChunk(
                            chunk_id=result.get("chunk_id") or "",
                            doc_id=result.get("doc_id") or "", 
                            content=result.get("content") or "",
                            word_count=result.get("word_count") or 0,
                            section_path=result.get("section_path") or "",
                            similarity_score=similarity_score
                        )
                        retrieved_chunks.append(chunk)
                    except (KeyError, TypeError):
                        # More specific exception handling
                        logger.warning(f"Malformed result: {result}, skipping")
                        continue
            
            # Apply re-ranking if enabled
            if self.enable_reranking and self.reranker and retrieved_chunks:
                retrieved_chunks = self._apply_reranking(query, retrieved_chunks, top_k)
            
            # Check for duplicates and deduplicate
            chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
            unique_ids = set(chunk_ids)
            if len(chunk_ids) != len(unique_ids):
                duplicate_count = len(chunk_ids) - len(unique_ids)
                logger.warning(f"Found {duplicate_count} duplicate chunks in retrieval results")
                # Log which chunks are duplicated
                from collections import Counter
                id_counts = Counter(chunk_ids)
                duplicates = [chunk_id for chunk_id, count in id_counts.items() if count > 1]
                logger.warning(f"Duplicate chunk IDs: {duplicates[:5]}...")  # Show first 5
                
                # Deduplicate by keeping first occurrence of each chunk_id
                seen_ids = set()
                deduplicated_chunks = []
                for chunk in retrieved_chunks:
                    if chunk.chunk_id not in seen_ids:
                        deduplicated_chunks.append(chunk)
                        seen_ids.add(chunk.chunk_id)
                
                retrieved_chunks = deduplicated_chunks
                logger.info(f"Deduplicated to {len(retrieved_chunks)} unique chunks")
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query ({len(set(chunk.chunk_id for chunk in retrieved_chunks))} unique)")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding with caching and optimized conversion."""
        # Check cache first
        cached_embedding = self._cache.get(query)
        if cached_embedding is not None:
            logger.debug("Using cached embedding")
            return cached_embedding
        
        # Generate new embedding with optimized conversion
        raw_embedding = self.embedding_service.model.encode(query, convert_to_tensor=False)
        
        # Optimize conversion from numpy to list
        query_embedding = optimize_numpy_to_list(raw_embedding)
        
        # Cache for future use
        self._cache.put(query, query_embedding)
        
        return query_embedding
    
    def _apply_reranking(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Apply re-ranking to retrieved chunks - optimized."""
        if not self.reranker:
            logger.warning("Re-ranker not available, returning original chunks")
            return chunks[:top_k]
        
        try:
            logger.debug(f"Re-ranking {len(chunks)} chunks to top {top_k}")
            start_time = time.time()
            
            # Optimized chunk conversion using list comprehension
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
                for chunk in chunks
            ]
            
            # Create O(1) lookup dictionary to eliminate nested loop
            chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
            
            # Ensure re-ranker model is loaded
            if not self.reranker.ensure_loaded():
                logger.error("Failed to load re-ranker model")
                return chunks[:top_k]
            
            # Perform re-ranking
            rerank_results = self.reranker.rerank(query, chunk_dicts, top_k)
            
            # Optimized conversion using dictionary lookup O(1) instead of O(n²)
            reranked_chunks = []
            for result in rerank_results:
                original_chunk = chunk_lookup.get(result.chunk_id)
                if original_chunk:
                    # Create new chunk preserving original similarity + adding re-ranking scores
                    # Convert logit score to probability using sigmoid
                    import math
                    rerank_probability = 1 / (1 + math.exp(-result.rerank_score))
                    
                    reranked_chunk = RetrievedChunk(
                        chunk_id=original_chunk.chunk_id,
                        doc_id=original_chunk.doc_id,
                        content=original_chunk.content,
                        word_count=original_chunk.word_count,
                        section_path=original_chunk.section_path,
                        similarity_score=original_chunk.similarity_score,  # Preserve original
                        rerank_score=result.rerank_score,  # Add re-ranking logit score
                        rerank_probability=rerank_probability  # Add re-ranking probability
                    )
                    reranked_chunks.append(reranked_chunk)
            
            rerank_time = time.time() - start_time
            
            # Check for duplicates in re-ranked results
            reranked_ids = [chunk.chunk_id for chunk in reranked_chunks]
            unique_reranked = set(reranked_ids)
            if len(reranked_ids) != len(unique_reranked):
                duplicate_count = len(reranked_ids) - len(unique_reranked)
                logger.warning(f"Re-ranking introduced {duplicate_count} duplicates!")
            
            logger.info(f"Re-ranked {len(chunks)} chunks to top {len(reranked_chunks)} in {rerank_time:.2f}s")
            
            return reranked_chunks
            
        except Exception as e:
            logger.error(f"Error during re-ranking: {e}")
            # Fall back to original chunks if re-ranking fails
            return chunks[:top_k]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Milvus collection."""
        if not self.connected:
            return {"error": "Not connected to Milvus"}
        
        return self.milvus_store.get_stats()
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Factory functions for backward compatibility
def create_retriever(
    collection_name: str = "document_chunks",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    enable_reranking: bool = False,
    reranker_config: Optional[Dict[str, Any]] = None
) -> MilvusRetriever:
    """Factory function to create a configured retriever."""
    return MilvusRetriever(
        embedding_model=embedding_model,
        collection_name=collection_name,
        enable_reranking=enable_reranking,
        reranker_config=reranker_config
    )


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.0,
    enable_reranking: bool = False,
    retrieval_multiplier: int = 10
) -> List[RetrievedChunk]:
    """Simple function to retrieve chunks without managing retriever lifecycle."""
    with create_retriever(enable_reranking=enable_reranking) as retriever:
        return retriever.retrieve(query, top_k, min_similarity, retrieval_multiplier)