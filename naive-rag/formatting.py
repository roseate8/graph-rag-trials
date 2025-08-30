"""
Context formatting with optimized token counting - combines context_formatter.py + tokens.py functionality.
"""

import hashlib
from functools import lru_cache
from typing import List, Optional
from dataclasses import dataclass

try:
    from .retrieval import RetrievedChunk
except ImportError:
    from retrieval import RetrievedChunk


class TokenEstimator:
    """Accurate token estimation using proper tokenization."""
    
    def __init__(self):
        self._tokenizer = None
        self._token_cache = {}
        self._max_cache_size = 1000
    
    def _get_tokenizer(self):
        """Lazy load tokenizer to avoid import overhead."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
            except ImportError:
                # Fallback to rough estimation if tiktoken not available
                self._tokenizer = None
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text with caching."""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._token_cache:
            return self._token_cache[text_hash]
        
        # Evict oldest entries if cache is full
        if len(self._token_cache) >= self._max_cache_size:
            # Simple FIFO eviction (remove 25% of entries)
            to_remove = list(self._token_cache.keys())[:self._max_cache_size // 4]
            for key in to_remove:
                del self._token_cache[key]
        
        # Count tokens
        tokenizer = self._get_tokenizer()
        if tokenizer:
            # Accurate count using proper tokenizer
            token_count = len(tokenizer.encode(text))
        else:
            # Fallback: improved approximation
            # Better than simple char/4 - accounts for punctuation, whitespace
            words = len(text.split())
            chars = len(text)
            # Empirically derived formula for better accuracy
            token_count = int(words * 1.3 + chars * 0.1)
        
        # Cache result
        self._token_cache[text_hash] = token_count
        return token_count
    
    def clear_cache(self):
        """Clear token cache."""
        self._token_cache.clear()


# Global token estimator instance
_token_estimator = TokenEstimator()


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return _token_estimator.count_tokens(text)


def estimate_tokens_fast(text: str) -> int:
    """Fast token estimation without tokenization (for when accuracy isn't critical)."""
    # Improved fast estimation
    words = len(text.split())
    return int(words * 1.3)  # Better approximation than char/4


@lru_cache(maxsize=128)
def count_tokens_cached(text: str) -> int:
    """Cached token counting using functools.lru_cache."""
    return count_tokens(text)


def clear_token_cache():
    """Clear all token caches."""
    _token_estimator.clear_cache()
    count_tokens_cached.cache_clear()


@dataclass
class RAGPrompt:
    """Container for formatted RAG prompt with system and user messages."""
    system_message: str
    user_message: str
    context_token_count: int
    chunk_count: int
    
    def to_openai_format(self) -> List[dict]:
        """Convert to OpenAI API message format."""
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message}
        ]
    
    def to_single_prompt(self) -> str:
        """Convert to single prompt string for models that don't use chat format."""
        return f"{self.system_message}\n\n{self.user_message}"


class ContextFormatter:
    """Optimized formatter for retrieved chunks into structured context for LLM consumption."""
    
    def __init__(
        self,
        max_context_tokens: int = 4000,
        include_metadata: bool = True,
        include_scores: bool = False
    ):
        """Initialize context formatter."""
        self.max_context_tokens = max_context_tokens
        self.include_metadata = include_metadata
        self.include_scores = include_scores
    
    def format_context(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        system_prompt: Optional[str] = None
    ) -> RAGPrompt:
        """Format retrieved chunks into a structured RAG prompt."""
        if not chunks:
            return self._create_no_context_prompt(query, system_prompt)
        
        # Build context from chunks
        context_parts = []
        total_tokens = 0
        used_chunks = 0
        
        # Pre-format all chunks and estimate tokens in batch - more efficient
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_text = self._format_chunk(chunk, i + 1)
            tokens = count_tokens(chunk_text)
            formatted_chunks.append((chunk_text, tokens))
        
        # Now efficiently select chunks that fit within token limit
        for chunk_text, estimated_tokens in formatted_chunks:
            if total_tokens + estimated_tokens > self.max_context_tokens:
                break
            
            context_parts.append(chunk_text)
            total_tokens += estimated_tokens
            used_chunks += 1
        
        # Build the complete context
        context = "\n\n".join(context_parts)
        
        # Create system message
        system_message = system_prompt or self._get_default_system_prompt()
        
        # Create user message with context and query
        user_message = self._build_user_message(query, context)
        
        return RAGPrompt(
            system_message=system_message,
            user_message=user_message,
            context_token_count=total_tokens,
            chunk_count=used_chunks
        )
    
    def _format_chunk(self, chunk: RetrievedChunk, index: int) -> str:
        """Format a single chunk for context."""
        parts = [f"Document {index}:"]
        
        # Add metadata if requested
        if self.include_metadata:
            metadata_parts = []
            
            if chunk.doc_id:
                metadata_parts.append(f"Source: {chunk.doc_id}")
            
            if chunk.section_path:
                metadata_parts.append(f"Section: {chunk.section_path}")
            
            if self.include_scores:
                metadata_parts.append(f"Relevance: {chunk.similarity_score:.3f}")
            
            if metadata_parts:
                parts.append("(" + " | ".join(metadata_parts) + ")")
        
        # Add content
        parts.append(chunk.content.strip())
        
        return "\n".join(parts)
    
    def _build_user_message(self, query: str, context: str) -> str:
        """Build the user message with context and query."""
        return f"""Based on the following documents, please answer the question.

CONTEXT:
{context}

QUESTION: {query}

Please provide a comprehensive answer based on the information in the documents above. If the documents don't contain enough information to answer the question, please say so."""
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for RAG."""
        return """You are a helpful AI assistant that answers questions based on provided documents. 

Your instructions:
1. Answer questions using ONLY the information provided in the context documents
2. Be accurate and cite specific information from the documents when possible
3. If the provided documents don't contain enough information to answer the question, clearly state this
4. Provide clear, well-structured answers that directly address the user's question
5. Do not make up information that isn't in the provided context"""
    
    def _create_no_context_prompt(self, query: str, system_prompt: Optional[str]) -> RAGPrompt:
        """Create prompt when no chunks were retrieved."""
        system_message = system_prompt or self._get_default_system_prompt()
        
        user_message = f"""QUESTION: {query}

I don't have any relevant documents to help answer your question. Please let me know if you'd like to rephrase your question or if you can provide more context."""
        
        return RAGPrompt(
            system_message=system_message,
            user_message=user_message,
            context_token_count=0,
            chunk_count=0
        )


def format_simple_context(
    query: str,
    chunks: List[RetrievedChunk],
    max_tokens: int = 4000
) -> str:
    """Simple function to format context without full prompt structure."""
    formatter = ContextFormatter(max_context_tokens=max_tokens)
    prompt = formatter.format_context(query, chunks)
    return prompt.user_message


def create_formatter(
    template: str = "factual",
    max_tokens: int = 4000,
    include_scores: bool = False
) -> ContextFormatter:
    """Factory function to create formatted with pre-defined templates."""
    return ContextFormatter(
        max_context_tokens=max_tokens,
        include_metadata=True,
        include_scores=include_scores
    )


# Pre-defined prompt templates for different use cases
PROMPT_TEMPLATES = {
    "factual": """You are a factual assistant that provides accurate information based on documents.

Instructions:
- Answer questions using only the provided documents
- Be precise and factual
- Quote specific information when helpful
- State when information is not available in the documents""",
    
    "analytical": """You are an analytical assistant that helps users understand complex information.

Instructions:
- Analyze the provided documents to answer questions
- Explain relationships and patterns in the information
- Provide insights based on the document content
- Clearly distinguish between facts in the documents and your analysis""",
    
    "conversational": """You are a helpful conversational assistant.

Instructions:
- Answer questions in a friendly, conversational tone
- Use the provided documents as your knowledge source
- Explain concepts clearly for better understanding
- Be honest about limitations in the available information"""
}