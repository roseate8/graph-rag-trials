"""Embeddings manager that coordinates embedding generation and vector storage."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .embedding_service import EmbeddingService
from .vector_store import MilvusVectorStore
from ..chunking.models import Chunk

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Manages embedding generation and storage in vector database."""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        milvus_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize embeddings manager.
        
        Args:
            embedding_model: Name of the embedding model to use
            milvus_config: Configuration for Milvus connection
        """
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        
        # Default Milvus configuration
        default_config = {
            "collection_name": "document_chunks",
            "host": "localhost",
            "port": "19530",
            "user": "",
            "password": "",
            "embedding_dim": 384
        }
        
        if milvus_config:
            default_config.update(milvus_config)
            
        self.vector_store = MilvusVectorStore(**default_config)
        self.connected = False
        
    def initialize_vector_store(self) -> bool:
        """Initialize vector store connection and collection."""
        try:
            # Connect to Milvus
            if not self.vector_store.connect():
                logger.error("Failed to connect to Milvus")
                return False
            
            # Create collection if it doesn't exist
            if not self.vector_store.create_collection():
                logger.error("Failed to create collection")
                return False
            
            # Create index for efficient search
            if not self.vector_store.create_index():
                logger.error("Failed to create index")
                return False
            
            self.connected = True
            logger.info("Vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            return False
    
    def process_and_store_chunks(self, chunks: List[Chunk]) -> bool:
        """
        Generate embeddings for chunks and store in vector database.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Generate embeddings if not already present
            chunks_with_embeddings = []
            chunks_needing_embeddings = []
            
            for chunk in chunks:
                if hasattr(chunk, 'embedding') and chunk.embedding:
                    chunks_with_embeddings.append(chunk)
                else:
                    chunks_needing_embeddings.append(chunk)
            
            # Generate embeddings for chunks that need them
            if chunks_needing_embeddings:
                logger.info(f"Generating embeddings for {len(chunks_needing_embeddings)} chunks")
                embedded_chunks = self.embedding_service.embed_chunks(chunks_needing_embeddings)
                chunks_with_embeddings.extend(embedded_chunks)
            
            # Convert chunks to format suitable for Milvus
            chunk_data = []
            for chunk in chunks_with_embeddings:
                chunk_dict = {
                    "chunk_id": chunk.metadata.chunk_id,
                    "doc_id": chunk.metadata.doc_id,
                    "content": chunk.content,
                    "word_count": chunk.metadata.word_count,
                    "section_path": chunk.metadata.section_path or [],
                    "embedding": chunk.embedding
                }
                chunk_data.append(chunk_dict)
            
            # Store in vector database
            if self.vector_store.insert_chunks(chunk_data):
                logger.info(f"Successfully stored {len(chunk_data)} chunks in vector database")
                return True
            else:
                logger.error("Failed to store chunks in vector database")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process and store chunks: {str(e)}")
            return False
    
    def search_similar_chunks(
        self,
        query_text: str,
        top_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using text query.
        
        Args:
            query_text: Text to search for similar chunks
            top_k: Number of similar chunks to return
            search_params: Additional search parameters
            
        Returns:
            List of similar chunks with metadata and scores
        """
        if not self.connected:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Generate embedding for query text
            query_embedding = self.embedding_service.model.encode([query_text])
            query_embedding = query_embedding[0].tolist()
            
            # Search for similar chunks
            similar_chunks = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                search_params=search_params
            )
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def search_similar_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using embedding vector.
        
        Args:
            query_embedding: Embedding vector to search with
            top_k: Number of similar chunks to return
            search_params: Additional search parameters
            
        Returns:
            List of similar chunks with metadata and scores
        """
        if not self.connected:
            logger.error("Vector store not initialized")
            return []
        
        return self.vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            search_params=search_params
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.connected:
            return {"error": "Vector store not initialized"}
        
        return self.vector_store.get_collection_stats()
    
    def export_embeddings(self, output_path: Path) -> bool:
        """
        Export embeddings to a file for backup or analysis.
        
        Args:
            output_path: Path to save the embeddings
            
        Returns:
            True if successful, False otherwise
        """
        # This would implement export functionality
        # For now, just log the intention
        logger.info(f"Export embeddings functionality would save to: {output_path}")
        return True
    
    def cleanup(self):
        """Clean up resources and disconnect from vector store."""
        try:
            if self.connected:
                self.vector_store.disconnect()
                self.connected = False
                logger.info("Embeddings manager cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize_vector_store()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()