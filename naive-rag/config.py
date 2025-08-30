"""
Simple configuration and error handling - combines common/config.py + errors.py functionality.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from contextlib import contextmanager


# Simple Error Classes
class RAGError(Exception):
    """Base exception for RAG system errors."""
    
    def __init__(self, message: str, component: Optional[str] = None):
        super().__init__(message)
        self.component = component
        self.message = message
    
    def __str__(self):
        if self.component:
            return f"[{self.component}] {self.message}"
        return self.message


class ConnectionError(RAGError):
    """Errors related to database/service connections."""
    pass


class RetrievalError(RAGError):
    """Errors during document retrieval."""
    pass


class LLMError(RAGError):
    """Errors during LLM interaction."""
    pass


@contextmanager
def handle_rag_errors(component: str, operation: str):
    """Simple context manager for consistent error handling."""
    try:
        yield
    except RAGError:
        # Re-raise RAG errors as-is
        raise
    except Exception as e:
        error_msg = f"Error during {operation}: {str(e)}"
        logging.getLogger(__name__).error(f"{component}: {error_msg}")
        raise RAGError(error_msg, component)


@dataclass
class RAGConfig:
    """Simple RAG system configuration with environment variable support."""
    
    # Retriever settings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    collection_name: str = "document_chunks"
    
    # Context formatter settings
    max_context_tokens: int = 4000
    include_metadata: bool = True
    include_scores: bool = False
    
    # LLM settings
    llm_type: str = "openai"  # "openai" or "mock"
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.1
    
    # Performance settings
    cache_size: int = 100
    
    @classmethod
    def from_env(cls, prefix: str = "RAG_") -> 'RAGConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Simple environment variable loading
        env_mappings = {
            'embedding_model': f'{prefix}EMBEDDING_MODEL',
            'collection_name': f'{prefix}COLLECTION_NAME',
            'max_context_tokens': f'{prefix}MAX_CONTEXT_TOKENS',
            'llm_type': f'{prefix}LLM_TYPE',
            'llm_model': f'{prefix}LLM_MODEL',
            'max_tokens': f'{prefix}MAX_TOKENS',
            'temperature': f'{prefix}TEMPERATURE',
            'cache_size': f'{prefix}CACHE_SIZE'
        }
        
        for field, env_var in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Simple type conversion
                try:
                    field_type = type(getattr(config, field))
                    if field_type == bool:
                        value = value.lower() in ('true', '1', 'yes')
                    elif field_type == int:
                        value = int(value)
                    elif field_type == float:
                        value = float(value)
                    
                    setattr(config, field, value)
                except (ValueError, TypeError) as e:
                    raise RAGError(f"Invalid value for {env_var}: {value}", "ConfigManager")
        
        return config
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise RAGError(f"Unknown configuration field: {key}", "ConfigManager")


# Global config instance
_config = None


def get_config() -> RAGConfig:
    """Get global configuration, loading from environment if needed."""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    return _config


def update_config(**kwargs):
    """Update global configuration."""
    global _config
    if _config is None:
        _config = RAGConfig()
    _config.update(**kwargs)