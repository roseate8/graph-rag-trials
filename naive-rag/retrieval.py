"""
Document retrieval with optimized caching - combines retriever.py + cache.py functionality.
"""

import sys
import logging
import hashlib
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
    
    def __str__(self) -> str:
        return f"[{self.doc_id}] {self.content[:100]}..."


class EmbeddingCache:
    """O(1) cache for query embeddings using OrderedDict."""
    
    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()  # O(1) operations
        self.max_size = max_size
    
    def get(self, query: str) -> Optional[List[float]]:
        """Get cached embedding for query - O(1)."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self.cache:
            # Move to end (most recently used) - O(1)
            self.cache.move_to_end(query_hash)
            return self.cache[query_hash]
        
        return None
    
    def put(self, query: str, embedding: List[float]):
        """Cache embedding for query - O(1)."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self.cache:
            # Update existing and move to end - O(1)
            self.cache.move_to_end(query_hash)
        else:
            # Evict oldest if needed - O(1)
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # Remove oldest
            self.cache[query_hash] = embedding
    
    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()


def optimize_numpy_to_list(arr) -> List[float]:
    """Optimized conversion from numpy array to list."""
    import numpy as np
    
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.float32:
            return [float(x) for x in arr]
        return arr.tolist()
    elif hasattr(arr, 'tolist'):
        return arr.tolist()
    else:
        return list(arr)


class MilvusRetriever:
    """Optimized retriever that searches Milvus for relevant document chunks."""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        milvus_profile: str = "production",
        collection_name: str = "document_chunks"
    ):
        """Initialize the retriever."""
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.milvus_config = get_config(milvus_profile)
        self.milvus_config.collection_name = collection_name
        self.milvus_store = MilvusVectorStore(self.milvus_config)
        self.connected = False
        self._cache = EmbeddingCache()
        
        logger.info(f"Initialized MilvusRetriever for collection: {collection_name}")
    
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
        min_similarity: float = 0.0
    ) -> List[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query."""
        if not self.connected:
            logger.error("Not connected to Milvus. Call connect() first.")
            return []
        
        try:
            # Generate embedding for the query (with caching)
            logger.debug(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self._get_query_embedding(query)
            
            # Search Milvus for similar chunks
            logger.debug(f"Searching Milvus for top {top_k} similar chunks...")
            search_results = self.milvus_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                output_fields=["chunk_id", "doc_id", "content", "word_count", "section_path"]
            )
            
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
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query")
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
    embedding_model: str = "BAAI/bge-small-en-v1.5"
) -> MilvusRetriever:
    """Factory function to create a configured retriever."""
    return MilvusRetriever(
        embedding_model=embedding_model,
        collection_name=collection_name
    )


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.0
) -> List[RetrievedChunk]:
    """Simple function to retrieve chunks without managing retriever lifecycle."""
    with create_retriever() as retriever:
        return retriever.retrieve(query, top_k, min_similarity)