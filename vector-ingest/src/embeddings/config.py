"""Configuration settings for embeddings and vector storage."""

from typing import Dict, Any
import os


class MilvusConfig:
    """Configuration for Milvus vector database."""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default Milvus configuration."""
        return {
            "collection_name": "document_chunks",
            "host": os.getenv("MILVUS_HOST", "localhost"),
            "port": os.getenv("MILVUS_PORT", "19530"),
            "user": os.getenv("MILVUS_USER", ""),
            "password": os.getenv("MILVUS_PASSWORD", ""),
            "embedding_dim": 384  # BGE-small-en-v1.5 dimension
        }
    
    @staticmethod
    def get_cloud_config() -> Dict[str, Any]:
        """Get configuration for Milvus cloud (Zilliz Cloud)."""
        # Support both token and username/password authentication
        endpoint = os.getenv("MILVUS_ENDPOINT", os.getenv("ZILLIZ_ENDPOINT", ""))
        token = os.getenv("MILVUS_TOKEN", "")
        
        config = {
            "collection_name": os.getenv("MILVUS_COLLECTION_NAME", "document_chunks"),
            "embedding_dim": 384,
            "secure": True,
            "database": os.getenv("MILVUS_DATABASE", "default")
        }
        
        if token:
            # Token-based authentication (recommended)
            config.update({
                "uri": endpoint,
                "token": token
            })
        else:
            # Username/password authentication
            config.update({
                "host": endpoint.replace("https://", "").replace("http://", ""),
                "port": os.getenv("ZILLIZ_PORT", "443"),
                "user": os.getenv("ZILLIZ_USERNAME", ""),
                "password": os.getenv("ZILLIZ_PASSWORD", "")
            })
        
        return config
    
    @staticmethod
    def get_custom_config(
        host: str,
        port: str = "19530",
        user: str = "",
        password: str = "",
        collection_name: str = "document_chunks"
    ) -> Dict[str, Any]:
        """Get custom Milvus configuration."""
        return {
            "collection_name": collection_name,
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "embedding_dim": 384
        }


class EmbeddingConfig:
    """Configuration for embedding models."""
    
    # Available embedding models and their dimensions
    MODELS = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768
    }
    
    @staticmethod
    def get_default_model() -> str:
        """Get default embedding model."""
        return "BAAI/bge-small-en-v1.5"
    
    @staticmethod
    def get_model_dimension(model_name: str) -> int:
        """Get embedding dimension for a model."""
        return EmbeddingConfig.MODELS.get(model_name, 384)
    
    @staticmethod
    def validate_model(model_name: str) -> bool:
        """Check if model is supported."""
        return model_name in EmbeddingConfig.MODELS


# Search parameters for different index types
SEARCH_PARAMS = {
    "IVF_FLAT": {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    },
    "IVF_SQ8": {
        "metric_type": "IP", 
        "params": {"nprobe": 10}
    },
    "HNSW": {
        "metric_type": "IP",
        "params": {"ef": 64}
    }
}

# Index parameters for different index types
INDEX_PARAMS = {
    "IVF_FLAT": {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 1024}
    },
    "IVF_SQ8": {
        "index_type": "IVF_SQ8",
        "metric_type": "IP",
        "params": {"nlist": 1024}
    },
    "HNSW": {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 64}
    }
}