"""
Advanced RAG Implementation with Query Decomposition and Fusion Re-ranking.

Structure:
- retrieval.py: Document retrieval + multi-query support
- formatting.py: Context formatting + token utils
- llm.py: Secure LLM client
- core.py: RAG orchestration + decomposition pipeline
- config.py: Configuration + decomposition parameters
- decomposer/: Query decomposition module
- re-rankers/: Re-ranking models + fusion re-ranking
"""

# Core components
from .retrieval import MilvusRetriever, RetrievedChunk, create_retriever, retrieve_chunks
from .formatting import ContextFormatter, RAGPrompt, format_simple_context, create_formatter, count_tokens
from .llm import SecureOpenAIClient, MockLLMClient, RAGResponse, create_llm_client, generate_rag_response
from .core import RAGSystem, RAGResult, create_rag_system, ask_rag
from .config import RAGConfig, RAGError, ConnectionError, RetrievalError, LLMError, get_config, update_config

# Query decomposition (new)
from .decomposer import QueryDecomposer, DecomposedQuery, decompose_query_simple

# Fusion re-ranking (new)
from .re_rankers.fusion_reranker import FusionReranker, FusionResult, fuse_results

__all__ = [
    # Core classes
    "MilvusRetriever", "RetrievedChunk", "ContextFormatter", "RAGPrompt",
    "SecureOpenAIClient", "MockLLMClient", "RAGResponse", "RAGSystem", "RAGResult",
    "QueryDecomposer", "DecomposedQuery", "FusionReranker", "FusionResult",
    
    # Configuration
    "RAGConfig", "get_config", "update_config",
    
    # Error handling
    "RAGError", "ConnectionError", "RetrievalError", "LLMError",
    
    # Factory functions
    "create_retriever", "create_formatter", "create_llm_client", "create_rag_system",
    
    # Simple function interfaces
    "retrieve_chunks", "format_simple_context", "generate_rag_response", "ask_rag",
    "decompose_query_simple", "fuse_results",
    
    # Utilities
    "count_tokens"
]
