"""
Core RAG system orchestration and main interface - combines rag_system.py + main.py functionality.
"""

import sys
import logging
import warnings
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from .retrieval import MilvusRetriever, RetrievedChunk
    from .formatting import ContextFormatter, RAGPrompt
    from .llm import SecureOpenAIClient, MockLLMClient, RAGResponse, create_llm_client
except ImportError:
    from retrieval import MilvusRetriever, RetrievedChunk
    from formatting import ContextFormatter, RAGPrompt
    from llm import SecureOpenAIClient, MockLLMClient, RAGResponse, create_llm_client


@dataclass
class RAGResult:
    """Complete RAG result with full pipeline metadata."""
    query: str
    response: str
    retrieved_chunks: List[RetrievedChunk]
    context_token_count: int
    response_tokens: Optional[int] = None
    model_used: Optional[str] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None


class RAGSystem:
    """Optimized RAG system that orchestrates retrieval and generation."""
    
    def __init__(
        self,
        retriever: Optional[MilvusRetriever] = None,
        formatter: Optional[ContextFormatter] = None,
        llm_client: Optional[Any] = None,
        # Retriever parameters
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        collection_name: str = "document_chunks",
        # Context parameters
        max_context_tokens: int = 4000,
        include_scores: bool = False,
        # LLM parameters
        llm_type: str = "mock",
        llm_model: str = "gpt-4o-mini"
    ):
        """Initialize complete RAG system."""
        # Initialize components
        self.retriever = retriever or MilvusRetriever(
            embedding_model=embedding_model,
            collection_name=collection_name
        )
        
        self.formatter = formatter or ContextFormatter(
            max_context_tokens=max_context_tokens,
            include_scores=include_scores
        )
        
        self.llm_client = llm_client or create_llm_client(
            client_type=llm_type,
            model=llm_model
        )
        
        self.connected = False
        logger.info("Initialized RAG system")
    
    def connect(self) -> bool:
        """Connect to required services (Milvus)."""
        if not self.retriever.connect():
            logger.error("Failed to connect retriever")
            return False
        
        self.connected = True
        logger.info("RAG system connected")
        return True
    
    def disconnect(self):
        """Disconnect from services."""
        if self.connected:
            self.retriever.disconnect()
            self.connected = False
            logger.info("RAG system disconnected")
    
    def query(
        self,
        user_query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        system_prompt: Optional[str] = None
    ) -> RAGResult:
        """Execute complete RAG pipeline for a user query."""
        try:
            self._validate_connection(user_query)
            chunks, retrieval_time = self._retrieve_chunks(user_query, top_k, min_similarity)
            rag_prompt = self._format_context(user_query, chunks, system_prompt)
            llm_response, generation_time = self._generate_response(rag_prompt)
            return self._build_result(user_query, chunks, llm_response, retrieval_time, generation_time)
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return self._create_error_result(user_query, str(e))
    
    def _validate_connection(self, user_query: str) -> None:
        """Validate RAG system connection."""
        if not self.connected:
            logger.error("RAG system not connected. Call connect() first.")
            raise ConnectionError("System not connected")
    
    def _retrieve_chunks(self, user_query: str, top_k: int, min_similarity: float) -> Tuple[List, float]:
        """Retrieve relevant chunks with timing."""
        import time
        logger.debug(f"Retrieving chunks for query: {user_query[:50]}...")
        
        start_time = time.time()
        chunks = self.retriever.retrieve(
            query=user_query,
            top_k=top_k,
            min_similarity=min_similarity
        )
        retrieval_time = time.time() - start_time
        
        logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        return chunks, retrieval_time
    
    def _format_context(self, user_query: str, chunks: List, system_prompt: Optional[str]) -> Any:
        """Format chunks into LLM context."""
        logger.debug("Formatting context...")
        return self.formatter.format_context(
            query=user_query,
            chunks=chunks,
            system_prompt=system_prompt
        )
    
    def _generate_response(self, rag_prompt: Any) -> Tuple[Any, float]:
        """Generate LLM response with timing."""
        import time
        logger.debug("Generating LLM response...")
        
        start_time = time.time()
        llm_response = self.llm_client.generate_response(rag_prompt)
        generation_time = time.time() - start_time
        
        logger.info(f"Generated response in {generation_time:.2f}s")
        return llm_response, generation_time
    
    def _build_result(self, user_query: str, chunks: List, llm_response: Any, 
                     retrieval_time: float, generation_time: float) -> RAGResult:
        """Build final RAG result."""
        return RAGResult(
            query=user_query,
            response=llm_response.response,
            retrieved_chunks=chunks,
            context_token_count=llm_response.context_token_count,
            response_tokens=llm_response.response_tokens,
            model_used=llm_response.model_used,
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )
    
    def _create_error_result(self, query: str, error: str) -> RAGResult:
        """Create error result."""
        return RAGResult(
            query=query,
            response=f"Error: {error}",
            retrieved_chunks=[],
            context_token_count=0,
            response_tokens=0,
            model_used="error"
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        stats = {
            "connected": self.connected,
            "retriever_stats": {},
            "formatter_config": {
                "max_context_tokens": self.formatter.max_context_tokens,
                "include_metadata": self.formatter.include_metadata,
                "include_scores": self.formatter.include_scores
            },
            "llm_config": {
                "type": type(self.llm_client).__name__,
                "model": getattr(self.llm_client, 'model', 'unknown')
            }
        }
        
        if self.connected:
            stats["retriever_stats"] = self.retriever.get_collection_stats()
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def create_rag_system(
    llm_type: str = "mock",
    collection_name: str = "document_chunks",
    **kwargs
) -> RAGSystem:
    """Factory function to create configured RAG system."""
    return RAGSystem(
        llm_type=llm_type,
        collection_name=collection_name,
        **kwargs
    )


def ask_rag(
    query: str,
    top_k: int = 5,
    llm_type: str = "mock",
    **kwargs
) -> str:
    """Simple function to ask RAG system without managing lifecycle."""
    with create_rag_system(llm_type=llm_type, **kwargs) as rag:
        result = rag.query(query, top_k=top_k)
        return result.response


# Main CLI interface functionality
logger = logging.getLogger(__name__)


def setup_clean_logging():
    """Setup minimal logging for CLI."""
    # Suppress all logs except critical errors
    logging.basicConfig(level=logging.CRITICAL)
    
    # Suppress specific noisy loggers
    logging.getLogger('pymilvus').setLevel(logging.CRITICAL)
    logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL)
    logging.getLogger('transformers').setLevel(logging.CRITICAL)
    logging.getLogger('torch').setLevel(logging.CRITICAL)
    
    # Suppress warnings
    warnings.filterwarnings('ignore')


def main():
    """Main RAG interface with clean startup flow."""
    setup_clean_logging()
    
    print("=== NAIVE RAG SYSTEM ===")
    print()
    
    # Step 1: Initialize system components
    print("1. Initializing RAG components...")
    rag = create_rag_system(llm_type="openai")  # Use mock for testing
    
    # Step 2: Connect to Milvus
    print("2. Connecting to Milvus vector database...")
    if not rag.connect():
        print("ERROR: Could not connect to Milvus.")
        print("Make sure Milvus is running: docker-compose up -d")
        return 1
    
    # Step 3: Check data availability
    stats = rag.get_system_stats()
    num_entities = stats.get('retriever_stats', {}).get('num_entities', 0)
    print(f"3. Connected to collection with {num_entities} document chunks")
    
    if num_entities == 0:
        print("ERROR: No documents found. Upload data first.")
        return 1
    
    # Step 4: Initialize LLM (this will prompt for API key)
    print("4. Initializing OpenAI LLM client...")
    print("   (API key will be requested securely)")
    
    # Step 5: Ready for query
    print("5. RAG system ready!")
    print()
    
    try:
        # Get user query
        query = input("Enter your query: ").strip()
        
        if not query:
            print("No query provided. Exiting.")
            return 1
        
        # Process query through RAG pipeline
        print("Searching documents and generating response...")
        result = rag.query(query, top_k=5)
        
        # Display response
        print(f"\nResponse:")
        print("-" * 50)
        print(result.response)
        print("-" * 50)
        print(f"Sources: {len(result.retrieved_chunks)} chunks | Model: {result.model_used}")
        print()
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        rag.disconnect()
        print("RAG system disconnected.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())