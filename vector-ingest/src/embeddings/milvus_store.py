"""Milvus standalone vector store implementation."""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

try:
    from .milvus_config import MilvusConfig
except ImportError:
    from milvus_config import MilvusConfig

logger = logging.getLogger(__name__)

# Optional import - will provide helpful error if not available
try:
    from pymilvus import (
        connections, Collection, CollectionSchema, DataType, FieldSchema, 
        utility, MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError as e:
    MILVUS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Create dummy classes to prevent import errors
    connections = Collection = CollectionSchema = DataType = FieldSchema = utility = None
    MilvusException = Exception


class MilvusVectorStore:
    """Vector store implementation using Milvus standalone."""
    
    def __init__(self, config: Optional[MilvusConfig] = None):
        """
        Initialize Milvus vector store.
        
        Args:
            config: Milvus configuration. If None, uses default config.
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                f"pymilvus is required for MilvusVectorStore. "
                f"Install it with: pip install pymilvus>=2.3.0\n"
                f"Original error: {IMPORT_ERROR}"
            )
        
        self.config = config or MilvusConfig.default()
        self.collection: Optional[Collection] = None
        self.connected = False
        
        logger.info(f"Initialized MilvusVectorStore with config: {self.config.collection_name}")
    
    def connect(self) -> bool:
        """Connect to Milvus standalone instance."""
        try:
            # Use default alias for simplicity
            alias = "default"
            
            # Check if already connected
            if connections.has_connection(alias):
                connections.remove_connection(alias)
            
            # Connect to Milvus
            connections.connect(
                alias=alias,
                host=self.config.host,
                port=self.config.port
            )
            
            self.connected = True
            logger.info(f"Connected to Milvus at {self.config.host}:{self.config.port}")
            return True
            
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Milvus: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Milvus."""
        try:
            alias = "default"
            if connections.has_connection(alias):
                connections.remove_connection(alias)
            
            self.connected = False
            self.collection = None
            logger.info("Disconnected from Milvus")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")
    
    def create_collection(self, drop_if_exists: bool = False) -> bool:
        """
        Create collection with schema optimized for document chunks.
        
        Args:
            drop_if_exists: If True, drop existing collection before creating new one.
        """
        if not self.connected:
            logger.error("Not connected to Milvus")
            return False
        
        try:
            collection_name = self.config.collection_name
            
            # Drop existing collection if requested
            if drop_if_exists and utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"Dropped existing collection: {collection_name}")
            
            # Check if collection already exists
            if utility.has_collection(collection_name):
                self.collection = Collection(collection_name)
                logger.info(f"Using existing collection: {collection_name}")
                return True
            
            # Define schema for document chunks with ALL metadata fields
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                    description="Auto-generated primary key"
                ),
                FieldSchema(
                    name="chunk_id",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                    description="Unique chunk identifier"
                ),
                FieldSchema(
                    name="doc_id", 
                    dtype=DataType.VARCHAR,
                    max_length=256,
                    description="Document identifier"
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=65535,  # Max text length
                    description="Chunk content text"
                ),
                FieldSchema(
                    name="word_count",
                    dtype=DataType.INT32,
                    description="Number of words in chunk"
                ),
                FieldSchema(
                    name="section_path",
                    dtype=DataType.VARCHAR,
                    max_length=1024,
                    description="Section hierarchy path"
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.config.embedding_dim,
                    description="Dense embedding vector"
                ),
                
                # Additional metadata fields that we compute - nullable
                FieldSchema(
                    name="chunk_type",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                    nullable=True,
                    description="Type of chunk (text, table, etc.)"
                ),
                FieldSchema(
                    name="product_version",
                    dtype=DataType.VARCHAR,
                    max_length=32,
                    nullable=True,
                    description="Product version"
                ),
                FieldSchema(
                    name="source",
                    dtype=DataType.VARCHAR,
                    max_length=256,
                    nullable=True,
                    description="Source document"
                ),
                FieldSchema(
                    name="chunk_index",
                    dtype=DataType.INT32,
                    nullable=True,
                    description="Index of chunk within document"
                ),
                FieldSchema(
                    name="page",
                    dtype=DataType.INT32,
                    nullable=True,
                    description="Page number"
                ),
                
                # Entity extraction metadata (stored as JSON strings) - nullable
                FieldSchema(
                    name="regions",
                    dtype=DataType.VARCHAR,
                    max_length=2048,
                    nullable=True,
                    description="JSON array of extracted regions"
                ),
                FieldSchema(
                    name="metrics",
                    dtype=DataType.VARCHAR,
                    max_length=2048,
                    nullable=True,
                    description="JSON array of extracted metrics"
                ),
                FieldSchema(
                    name="time_periods",
                    dtype=DataType.VARCHAR,
                    max_length=2048,
                    nullable=True,
                    description="JSON array of extracted time periods"
                ),
                FieldSchema(
                    name="dates",
                    dtype=DataType.VARCHAR,
                    max_length=2048,
                    nullable=True,
                    description="JSON array of extracted dates"
                ),
                FieldSchema(
                    name="orgs",
                    dtype=DataType.VARCHAR,
                    max_length=2048,
                    nullable=True,
                    description="JSON array of extracted organizations"
                ),
                
                # SpaCy extraction results - nullable
                FieldSchema(
                    name="spacy_extraction",
                    dtype=DataType.VARCHAR,
                    max_length=4096,
                    nullable=True,
                    description="JSON object with spaCy extraction results"
                ),
                
                # Structural metadata - nullable
                FieldSchema(
                    name="structural_metadata",
                    dtype=DataType.VARCHAR,
                    max_length=4096,
                    nullable=True,
                    description="JSON object with structural metadata"
                ),
                
                # Time context - nullable
                FieldSchema(
                    name="time_context",
                    dtype=DataType.VARCHAR,
                    max_length=1024,
                    nullable=True,
                    description="JSON object with time context information"
                ),
                
                # Folder path - nullable
                FieldSchema(
                    name="folder_path",
                    dtype=DataType.VARCHAR,
                    max_length=2048,
                    nullable=True,
                    description="JSON array of folder path components"
                )
            ]
            
            # Create schema
            schema = CollectionSchema(
                fields=fields,
                description="Document chunks with embeddings for RAG system",
                enable_dynamic_field=True  # Allow additional fields
            )
            
            # Create collection
            self.collection = Collection(
                name=collection_name,
                schema=schema
            )
            
            logger.info(f"Created collection '{collection_name}' with {self.config.embedding_dim}D embeddings")
            return True
            
        except MilvusException as e:
            logger.error(f"Failed to create collection: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating collection: {e}")
            return False
    
    def create_index(self) -> bool:
        """Create index on embedding field for efficient similarity search."""
        if not self.collection:
            logger.error("Collection not initialized")
            return False
        
        try:
            index_params = {
                "index_type": self.config.index_type,
                "metric_type": self.config.metric_type,
                "params": self.config.index_params
            }
            
            # Create index on embedding field
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"Created {self.config.index_type} index with {self.config.metric_type} metric")
            return True
            
        except MilvusException as e:
            logger.error(f"Failed to create index: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating index: {e}")
            return False
    
    def insert_chunks(self, chunks: List[Dict[str, Any]], flush: bool = True) -> bool:
        """
        Insert document chunks with embeddings into collection (optimized).
        
        Args:
            chunks: List of chunk dictionaries
            flush: Whether to flush after insert (can be disabled for batch operations)
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return False
        
        if not chunks:
            logger.warning("No chunks to insert")
            return True
        
        try:
            # Ensure collection is loaded (required for insert operations in some Milvus versions)
            try:
                self.collection.load()
            except Exception:
                pass  # Collection might already be loaded or not need loading
            
            # Insert chunks directly
            logger.info(f"Inserting {len(chunks)} chunks...")
            insert_result = self.collection.insert(chunks)
            logger.info(f"Insert completed, result: {type(insert_result)}")
            
            if flush:
                logger.info("Flushing collection...")
                try:
                    # Add timeout to flush operation
                    import threading
                    import time
                    
                    def flush_with_timeout():
                        self.collection.flush()
                    
                    flush_thread = threading.Thread(target=flush_with_timeout)
                    flush_thread.daemon = True
                    flush_thread.start()
                    flush_thread.join(timeout=10.0)  # 10 second timeout
                    
                    if flush_thread.is_alive():
                        logger.warning("Flush operation timed out, but data should still be inserted")
                    else:
                        logger.info("Flush completed successfully")
                except Exception as e:
                    logger.warning(f"Flush failed: {e}, but data should still be inserted")
                    # Continue anyway - flush is not critical for data persistence
            
            logger.info(f"Successfully inserted {len(chunks)} chunks into collection")
            return True
            
        except MilvusException as e:
            logger.error(f"Failed to insert chunks: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error inserting chunks: {e}")
            return False
    
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on embedding similarity (optimized).
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            output_fields: Fields to return in results
            
        Returns:
            List of similar chunks with metadata and similarity scores
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return []
        
        try:
            # Load collection for search if not loaded (cached operation)
            self.collection.load()
            
            # Pre-defined default output fields to avoid repeated list creation
            if output_fields is None:
                output_fields = ["chunk_id", "doc_id", "content", "word_count", "section_path"]
            
            # Fast dimension check - O(1)
            if len(query_embedding) != self.config.embedding_dim:
                raise ValueError(
                    f"Query embedding dimension mismatch: "
                    f"expected {self.config.embedding_dim}, got {len(query_embedding)}"
                )
            
            # Pre-built search parameters (cached)
            search_params = {
                "metric_type": self.config.metric_type,
                "params": self.config.search_params
            }
            
            # Perform search - Milvus handles the heavy lifting
            search_results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )
            
            # Optimized result processing - pre-allocate result list
            results = []
            for hits in search_results:
                
                for hit in hits:
                    # Direct dictionary construction (faster than incremental building)
                    result = {
                        "id": hit.id,
                        "similarity_score": hit.score,
                        **{field: hit.entity.get(field) for field in output_fields}
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except MilvusException as e:
            logger.error(f"Search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:
            # Load collection to get accurate stats
            self.collection.load()
            
            stats = {
                "collection_name": self.collection.name,
                "num_entities": self.collection.num_entities,
                "index_type": self.config.index_type,
                "metric_type": self.config.metric_type,
                "embedding_dim": self.config.embedding_dim,
                "schema": {
                    field.name: {
                        "type": str(field.dtype),
                        "description": field.description
                    }
                    for field in self.collection.schema.fields
                }
            }
            
            return stats
            
        except MilvusException as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error getting stats: {e}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the collection completely."""
        try:
            collection_name = self.config.collection_name
            
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            else:
                logger.info(f"Collection {collection_name} does not exist")
            
            self.collection = None
            return True
            
        except MilvusException as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting collection: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            return utility.has_collection(self.config.collection_name)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False
    
    def get_entity_count(self) -> int:
        """Get the number of entities in the collection."""
        try:
            if not self.collection:
                self.collection = Collection(self.config.collection_name)
            
            # Ensure collection is loaded and flushed for accurate count
            self.collection.load()
            try:
                self.collection.flush()
            except Exception:
                pass  # Flush might timeout, but load should show current data
            
            return self.collection.num_entities
        except Exception as e:
            logger.error(f"Error getting entity count: {e}")
            return 0
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection (keeping structure)."""
        try:
            if not self.collection:
                self.collection = Collection(self.config.collection_name)
            
            # Load collection to ensure it's accessible
            self.collection.load()
            
            # Use a broader expression to match all entities
            # In Milvus, we need to use a filter that matches all records
            expr = "id >= 0"  # Assuming ID field exists and all IDs are >= 0
            
            # Try different deletion approaches
            try:
                self.collection.delete(expr)
                logger.info(f"Cleared all data from collection: {self.config.collection_name}")
                return True
            except Exception as delete_error:
                # If direct delete fails, try dropping and recreating collection
                logger.warning(f"Direct delete failed: {delete_error}")
                logger.info("Attempting to drop and recreate collection...")
                
                # Drop the collection completely
                utility.drop_collection(self.config.collection_name)
                logger.info(f"Dropped collection: {self.config.collection_name}")
                
                # Recreate it
                success = self.create_collection()
                if success:
                    logger.info(f"Recreated collection: {self.config.collection_name}")
                    return True
                else:
                    logger.error("Failed to recreate collection after drop")
                    return False
                    
        except MilvusException as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error clearing collection: {e}")
            return False
    
    def drop_collection(self) -> bool:
        """Drop the entire collection (alias for delete_collection)."""
        return self.delete_collection()
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the Milvus connection and collection."""
        health_status = {
            "connected": self.connected,
            "collection_exists": False,
            "collection_loaded": False,
            "num_entities": 0,
            "errors": []
        }
        
        try:
            # Check connection
            if not self.connected:
                health_status["errors"].append("Not connected to Milvus")
                return health_status
            
            # Check collection existence
            if utility.has_collection(self.config.collection_name):
                health_status["collection_exists"] = True
                
                # Try to load collection
                if not self.collection:
                    self.collection = Collection(self.config.collection_name)
                
                # Check if loaded
                try:
                    self.collection.load()
                    health_status["collection_loaded"] = True
                    health_status["num_entities"] = self.collection.num_entities
                except Exception as e:
                    health_status["errors"].append(f"Collection load failed: {e}")
            else:
                health_status["errors"].append("Collection does not exist")
            
        except Exception as e:
            health_status["errors"].append(f"Health check failed: {e}")
        
        return health_status
    
    def __enter__(self):
        """Context manager entry."""
        if not self.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()