"""
Optimized Naive RAG Implementation - Streamlined for performance and simplicity.

Consolidated structure:
- retrieval.py: Document retrieval + caching (was retriever.py + cache.py)
- formatting.py: Context formatting + token utils (was context_formatter.py + tokens.py)  
- llm.py: LLM client (was utils/llm_client.py)
- core.py: Main orchestration + CLI (was rag_system.py + main.py)
- config.py: Configuration + error handling (was common/config.py + errors.py)
"""

# Core components
from .retrieval import MilvusRetriever, RetrievedChunk, create_retriever, retrieve_chunks
from .formatting import ContextFormatter, RAGPrompt, format_simple_context, create_formatter, count_tokens
from .llm import SecureOpenAIClient, MockLLMClient, RAGResponse, create_llm_client, generate_rag_response
from .core import RAGSystem, RAGResult, create_rag_system, ask_rag
from .config import RAGConfig, RAGError, ConnectionError, RetrievalError, LLMError, get_config, update_config

__all__ = [
    # Core classes
    "MilvusRetriever", "RetrievedChunk", "ContextFormatter", "RAGPrompt",
    "SecureOpenAIClient", "MockLLMClient", "RAGResponse", "RAGSystem", "RAGResult",
    
    # Configuration
    "RAGConfig", "get_config", "update_config",
    
    # Error handling
    "RAGError", "ConnectionError", "RetrievalError", "LLMError",
    
    # Factory functions
    "create_retriever", "create_formatter", "create_llm_client", "create_rag_system",
    
    # Simple function interfaces
    "retrieve_chunks", "format_simple_context", "generate_rag_response", "ask_rag",
    
    # Utilities
    "count_tokens"
]