"""Milvus standalone configuration for vector storage."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class MilvusConfig:
    """Configuration for Milvus standalone instance."""
    
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "document_chunks"
    embedding_dim: int = 384  # BGE-small-en-v1.5 dimension
    index_type: str = "HNSW"
    metric_type: str = "IP"  # Inner Product for normalized embeddings
    
    # Index parameters
    index_params: Dict[str, Any] = None
    search_params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set optimized parameters after initialization."""
        if self.index_params is None:
            # Optimized HNSW parameters for better performance
            if self.index_type == "HNSW":
                self.index_params = {
                    "M": 32,           # Increased for better recall (16->32)
                    "efConstruction": 128  # Increased for better index quality (64->128)
                }
            elif self.index_type == "IVF_FLAT":
                self.index_params = {
                    "nlist": 2048  # Optimized for ~100K vectors
                }
            elif self.index_type == "IVF_SQ8":
                self.index_params = {
                    "nlist": 2048  # Optimized for ~100K vectors
                }
            else:
                self.index_params = {}
        
        if self.search_params is None:
            # Optimized search parameters
            if self.index_type == "HNSW":
                self.search_params = {
                    "ef": 128  # Increased for better search quality (64->128)
                }
            elif self.index_type in ["IVF_FLAT", "IVF_SQ8"]:
                self.search_params = {
                    "nprobe": 32  # Optimized probe count
                }
            else:
                self.search_params = {}
    
    @classmethod
    def from_env(cls) -> 'MilvusConfig':
        """Create config from environment variables."""
        return cls(
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=int(os.getenv("MILVUS_PORT", 19530)),
            collection_name=os.getenv("MILVUS_COLLECTION", "document_chunks"),
            embedding_dim=int(os.getenv("MILVUS_DIM", 384)),
            index_type=os.getenv("MILVUS_INDEX_TYPE", "HNSW"),
            metric_type=os.getenv("MILVUS_METRIC_TYPE", "IP")
        )
    
    @classmethod
    def default(cls) -> 'MilvusConfig':
        """Create default configuration for standalone Milvus."""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "collection_name": self.collection_name,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "index_params": self.index_params,
            "search_params": self.search_params
        }


# Predefined configurations for different scenarios
CONFIGS = {
    "development": MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="dev_document_chunks",
        index_type="IVF_FLAT",  # Simpler index for development
        index_params={"nlist": 128}
    ),
    
    "production": MilvusConfig(
        host="localhost", 
        port=19530,
        collection_name="document_chunks",
        index_type="HNSW",  # Best performance for production
        index_params={"M": 16, "efConstruction": 64}
    ),
    
    "testing": MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="test_document_chunks",
        index_type="FLAT",  # No index for small test datasets
        index_params={}
    )
}


def get_config(profile: str = "production") -> MilvusConfig:
    """Get configuration for specified profile."""
    if profile in CONFIGS:
        return CONFIGS[profile]
    
    # Try to load from environment if profile not found
    return MilvusConfig.from_env()