"""Milvus vector store implementation for storing and retrieving embeddings."""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional import for Milvus - will raise informative error if not available
try:
    from pymilvus import connections, Collection, CollectionSchema, DataType, FieldSchema, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    connections = Collection = CollectionSchema = DataType = FieldSchema = utility = None


class MilvusVectorStore:
    """Milvus vector store for embedding storage and similarity search."""
    
    def __init__(
        self,
        collection_name: str = "document_chunks",
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        embedding_dim: int = 384,
        uri: str = "",
        token: str = "",
        secure: bool = False,
        database: str = "default"
    ):
        """
        Initialize Milvus vector store.
        
        Args:
            collection_name: Name of the Milvus collection
            host: Milvus server host (for local/self-hosted)
            port: Milvus server port (for local/self-hosted)
            user: Username for authentication
            password: Password for authentication
            embedding_dim: Dimension of embedding vectors (BGE-small-en-v1.5 uses 384)
            uri: Full URI for cloud connections (e.g., Zilliz Cloud)
            token: API token for cloud authentication
            secure: Use secure connection (HTTPS/TLS)
            database: Database name to connect to
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is required for MilvusVectorStore. "
                "Install it with: pip install pymilvus"
            )
            
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.embedding_dim = embedding_dim
        self.uri = uri
        self.token = token
        self.secure = secure
        self.database = database
        self.collection = None
        self._connected = False
        
    def connect(self) -> bool:
        """Connect to Milvus server (local or cloud)."""
        try:
            connect_params = {
                "alias": "default"
            }
            
            # Cloud connection (token-based)
            if self.uri and self.token:
                connect_params.update({
                    "uri": self.uri,
                    "token": self.token
                })
                logger.info(f"Connecting to Milvus Cloud: {self.uri}")
            
            # Local/self-hosted connection
            else:
                connect_params.update({
                    "host": self.host,
                    "port": self.port
                })
                
                if self.user and self.password:
                    connect_params.update({
                        "user": self.user,
                        "password": self.password
                    })
                
                logger.info(f"Connecting to Milvus at {self.host}:{self.port}")
            
            # Add database if specified
            if self.database != "default":
                connect_params["db_name"] = self.database
            
            connections.connect(**connect_params)
            self._connected = True
            logger.info("Successfully connected to Milvus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Milvus server."""
        try:
            connections.disconnect("default")
            self._connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {str(e)}")
    
    def create_collection(self) -> bool:
        """Create collection with schema for document chunks."""
        if not self._connected:
            logger.error("Not connected to Milvus")
            return False
        
        try:
            # Check if collection already exists
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                self.collection = Collection(self.collection_name)
                return True
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="word_count", dtype=DataType.INT64),
                FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Document chunks with embeddings for RAG system"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            logger.info(f"Created collection '{self.collection_name}' with {self.embedding_dim}D embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False
    
    def create_index(self, index_type: str = "IVF_FLAT", metric_type: str = "IP", nlist: int = 1024) -> bool:
        """
        Create index on embedding field for faster similarity search.
        
        Args:
            index_type: Type of index (IVF_FLAT, IVF_SQ8, HNSW)
            metric_type: Similarity metric (IP for inner product, L2 for Euclidean)
            nlist: Number of cluster units (for IVF indexes)
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return False
        
        try:
            index_params = {
                "index_type": index_type,
                "metric_type": metric_type,
                "params": {"nlist": nlist}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"Created {index_type} index with {metric_type} metric")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False
    
    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Insert document chunks with embeddings into Milvus.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return False
        
        try:
            # Prepare data for insertion
            data = [
                [chunk["chunk_id"] for chunk in chunks],
                [chunk["doc_id"] for chunk in chunks],
                [chunk["content"] for chunk in chunks],
                [chunk["word_count"] for chunk in chunks],
                [str(chunk.get("section_path", [])) for chunk in chunks],
                [chunk["embedding"] for chunk in chunks]
            ]
            
            # Insert data
            mr = self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Inserted {len(chunks)} chunks into collection")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert chunks: {str(e)}")
            return False
    
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top similar results to return
            search_params: Search parameters for the index
            
        Returns:
            List of similar chunks with metadata and similarity scores
        """
        if not self.collection:
            logger.error("Collection not initialized")
            return []
        
        try:
            # Load collection for search
            self.collection.load()
            
            # Default search params
            if search_params is None:
                search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["chunk_id", "doc_id", "content", "word_count", "section_path"]
            )
            
            # Format results
            similar_chunks = []
            for hits in results:
                for hit in hits:
                    similar_chunks.append({
                        "id": hit.id,
                        "chunk_id": hit.entity.get("chunk_id"),
                        "doc_id": hit.entity.get("doc_id"),
                        "content": hit.entity.get("content"),
                        "word_count": hit.entity.get("word_count"),
                        "section_path": hit.entity.get("section_path"),
                        "similarity_score": hit.score
                    })
            
            logger.info(f"Found {len(similar_chunks)} similar chunks")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        try:
            self.collection.load()
            stats = {
                "name": self.collection.name,
                "num_entities": self.collection.num_entities,
                "schema": str(self.collection.schema),
                "indexes": [str(index) for index in self.collection.indexes]
            }
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'")
                return True
            else:
                logger.info(f"Collection '{self.collection_name}' does not exist")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()