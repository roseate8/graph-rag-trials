"""Embeddings manager that coordinates embedding generation and optional Milvus storage."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    # Try relative imports first
    from .embedding_service import EmbeddingService
    from .milvus_config import MilvusConfig
    from .milvus_store import MilvusVectorStore
except ImportError:
    # Fallback to absolute imports
    from embedding_service import EmbeddingService
    from milvus_config import MilvusConfig
    from milvus_store import MilvusVectorStore

try:
    from ..chunking.models import Chunk
except ImportError:
    try:
        from chunking.models import Chunk
    except ImportError:
        # Create a simple Chunk class as fallback
        from dataclasses import dataclass
        from typing import Dict, Any
        
        @dataclass
        class Chunk:
            text: str
            metadata: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Manages embedding generation with optional Milvus vector storage."""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        enable_milvus: bool = False,
        milvus_config: Optional[MilvusConfig] = None
    ):
        """
        Initialize embeddings manager.
        
        Args:
            embedding_model: Name of the embedding model to use
            enable_milvus: Whether to enable Milvus vector storage
            milvus_config: Milvus configuration. If None, uses default config.
        """
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.enable_milvus = enable_milvus
        self.milvus_store = None
        self.connected = False
        
        if self.enable_milvus:
            config = milvus_config or MilvusConfig.default()
            self.milvus_store = MilvusVectorStore(config)
            logger.info(f"Milvus storage enabled with collection: {config.collection_name}")
        else:
            logger.info("Milvus storage disabled - embeddings will only be generated")
    
    def connect_milvus(self) -> bool:
        """Connect to Milvus (if enabled)."""
        if not self.enable_milvus or not self.milvus_store:
            return True  # No-op if Milvus disabled
        
        try:
            if self.milvus_store.connect():
                self.connected = True
                logger.info("Connected to Milvus successfully")
                return True
            else:
                logger.error("Failed to connect to Milvus")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {e}")
            return False
    
    def initialize_milvus(self, drop_existing: bool = False) -> bool:
        """Initialize Milvus collection and index (if enabled)."""
        if not self.enable_milvus or not self.connected:
            return True  # No-op if Milvus disabled or not connected
        
        try:
            # Create collection
            if not self.milvus_store.create_collection(drop_if_exists=drop_existing):
                logger.error("Failed to create Milvus collection")
                return False
            
            # Create index
            if not self.milvus_store.create_index():
                logger.error("Failed to create Milvus index")
                return False
            
            logger.info("Milvus collection and index initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Milvus: {e}")
            return False
    
    def process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Process chunks: generate embeddings and optionally store in Milvus.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with embeddings
        """
        if not chunks:
            return chunks
        
        logger.info(f"Processing {len(chunks)} chunks with embeddings manager")
        
        # Generate embeddings for all chunks
        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        
        # Store in Milvus if enabled
        if self.enable_milvus and self.connected:
            if self._store_in_milvus(embedded_chunks):
                logger.info(f"Stored {len(embedded_chunks)} chunks in Milvus")
            else:
                logger.warning("Failed to store some chunks in Milvus")
        
        return embedded_chunks
    
    def _store_in_milvus(self, chunks: List[Chunk]) -> bool:
        """Store embedded chunks in Milvus with optimized batch processing."""
        if not self.milvus_store or not self.connected:
            return False
        
        try:
            # Pre-filter chunks with embeddings - single pass O(n)
            chunks_with_embeddings = [chunk for chunk in chunks if chunk.embedding]
            
            if not chunks_with_embeddings:
                logger.warning("No chunks with embeddings to store")
                return True
            
            # Pre-allocate Milvus chunks list for better memory efficiency
            chunk_count = len(chunks_with_embeddings)
            milvus_chunks = []
            milvus_chunks.extend([None] * chunk_count)  # Reserve capacity
            
            # Convert chunks to Milvus format - single optimized pass
            for i, chunk in enumerate(chunks_with_embeddings):
                milvus_chunks[i] = {
                    "chunk_id": chunk.metadata.chunk_id,
                    "doc_id": chunk.metadata.doc_id,
                    "content": chunk.content,
                    "word_count": chunk.metadata.word_count,
                    "section_path": str(chunk.metadata.section_path or ""),
                    "embedding": chunk.embedding
                }
            
            # Optimized batch processing with larger batches for better throughput
            batch_size = 200  # Increased from 100 for better performance
            success_count = 0
            total_batches = (chunk_count + batch_size - 1) // batch_size  # Ceiling division
            
            # Process all but last batch without flush for speed
            for i in range(0, chunk_count - batch_size, batch_size):
                batch = milvus_chunks[i:i + batch_size]
                
                # Disable flush for intermediate batches (faster)
                if self.milvus_store.insert_chunks(batch, flush=False):
                    success_count += len(batch)
                else:
                    logger.error(f"Failed to store batch {i//batch_size + 1}/{total_batches}")
            
            # Process final batch with flush
            if chunk_count > 0:
                final_batch_start = (total_batches - 1) * batch_size
                final_batch = milvus_chunks[final_batch_start:]
                
                if self.milvus_store.insert_chunks(final_batch, flush=True):
                    success_count += len(final_batch)
                else:
                    logger.error(f"Failed to store final batch {total_batches}/{total_batches}")
            
            logger.info(f"Successfully stored {success_count}/{chunk_count} chunks in {total_batches} batches")
            return success_count == chunk_count
            
        except Exception as e:
            logger.error(f"Error storing chunks in Milvus: {e}")
            return False
    
    def search_similar(
        self,
        query_text: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using text query (requires Milvus).
        
        Args:
            query_text: Text to search for similar chunks
            top_k: Number of similar chunks to return
            
        Returns:
            List of similar chunks with metadata and scores
        """
        if not self.enable_milvus or not self.connected:
            logger.error("Milvus not enabled or not connected")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.embed_text(query_text)
            
            # Search in Milvus
            results = self.milvus_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            logger.info(f"Found {len(results)} similar chunks for query: {query_text[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def get_milvus_stats(self) -> Dict[str, Any]:
        """Get Milvus collection statistics (if enabled)."""
        if not self.enable_milvus or not self.connected:
            return {"error": "Milvus not enabled or not connected"}
        
        return self.milvus_store.get_stats()
    
    def disconnect(self):
        """Disconnect from Milvus and clean up resources."""
        try:
            if self.milvus_store and self.connected:
                self.milvus_store.disconnect()
                self.connected = False
                logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        if self.enable_milvus:
            self.connect_milvus()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def create_embeddings_manager(
    enable_milvus: bool = False,
    milvus_profile: str = "production"
) -> EmbeddingsManager:
    """
    Factory function to create embeddings manager with optional Milvus.
    
    Args:
        enable_milvus: Whether to enable Milvus vector storage
        milvus_profile: Milvus configuration profile (development, production, testing)
        
    Returns:
        Configured EmbeddingsManager instance
    """
    milvus_config = None
    if enable_milvus:
        from .milvus_config import get_config
        milvus_config = get_config(milvus_profile)
    
    return EmbeddingsManager(
        enable_milvus=enable_milvus,
        milvus_config=milvus_config
    )