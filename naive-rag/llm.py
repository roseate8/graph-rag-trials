"""
Optimized LLM client with secure API handling - simplified from utils/llm_client.py.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import requests
import json

# Setup LLM utils imports
vector_ingest_path = Path(__file__).parent.parent / "vector-ingest" / "src"
sys.path.append(str(vector_ingest_path))

from chunking.processors.llm_utils import get_openai_api_key, has_openai_api_key
try:
    from .formatting import RAGPrompt
except ImportError:
    from formatting import RAGPrompt

logger = logging.getLogger(__name__)


@dataclass 
class RAGResponse:
    """Container for RAG system response with metadata."""
    response: str
    context_used: bool
    chunk_count: int
    context_token_count: int
    response_tokens: Optional[int] = None
    model_used: Optional[str] = None


class SecureOpenAIClient:
    """Direct OpenAI API client with secure key management."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1000,
        temperature: float = 0.1
    ):
        """Initialize secure OpenAI client with direct API calls."""
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        logger.info(f"Initialized direct OpenAI API client with model: {model}")
    
    def generate_response(self, prompt: RAGPrompt) -> RAGResponse:
        """Generate response using direct OpenAI API calls."""
        try:
            response_data = self._make_api_call(prompt)
            response_text, response_tokens = self._parse_api_response(response_data)
            return self._build_rag_response(prompt, response_text, response_tokens)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._create_error_response(str(e))
    
    def _make_api_call(self, prompt: RAGPrompt) -> dict:
        """Execute OpenAI API call."""
        api_key = get_openai_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": prompt.to_openai_format(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        
        logger.debug(f"Making API call to {self.model}")
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            error_msg = f"OpenAI API error {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        return response.json()
    
    def _parse_api_response(self, response_data: dict) -> tuple:
        """Parse API response and extract key information."""
        response_text = response_data["choices"][0]["message"]["content"]
        usage = response_data.get("usage", {})
        response_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        
        logger.info(f"Generated response with {response_tokens} tokens (total: {total_tokens})")
        return response_text, response_tokens
    
    def _build_rag_response(self, prompt: RAGPrompt, response_text: str, response_tokens: int) -> RAGResponse:
        """Build RAGResponse from components."""
        return RAGResponse(
            response=response_text,
            context_used=prompt.chunk_count > 0,
            chunk_count=prompt.chunk_count,
            context_token_count=prompt.context_token_count,
            response_tokens=response_tokens,
            model_used=self.model
        )
    
    def _create_error_response(self, error_msg: str) -> RAGResponse:
        """Create error RAGResponse."""
        return RAGResponse(
            response=f"Sorry, I encountered an error generating a response: {error_msg}",
            context_used=False,
            chunk_count=0,
            context_token_count=0,
            model_used=self.model
        )
    
    def can_generate(self) -> bool:
        """Check if client can generate without prompting for API key."""
        return has_openai_api_key()


class MockLLMClient:
    """Mock LLM client for testing without API dependencies."""
    
    def __init__(self, model: str = "mock-gpt-4o-mini"):
        self.model = model
        logger.info("Initialized Mock LLM client")
    
    def generate_response(self, prompt: RAGPrompt) -> RAGResponse:
        """Generate a mock response based on the prompt."""
        
        if prompt.chunk_count == 0:
            response_text = "I don't have any relevant information to answer your question. Please provide more context or rephrase your query."
        else:
            response_text = f"""Based on the provided documents, I can answer your question using information from {prompt.chunk_count} relevant sources.

[This is a mock response that would normally be generated by GPT-4o mini (nano). In a real implementation, this would contain the actual AI-generated answer based on the retrieved document chunks.]

The information comes from the context documents, and I've used approximately {prompt.context_token_count} tokens worth of context to generate this response."""
        
        return RAGResponse(
            response=response_text,
            context_used=prompt.chunk_count > 0,
            chunk_count=prompt.chunk_count,
            context_token_count=prompt.context_token_count,
            response_tokens=len(response_text.split()) * 4,  # Rough token estimate
            model_used=self.model
        )
    
    def can_generate(self) -> bool:
        """Mock client can always generate."""
        return True


def create_llm_client(
    client_type: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs
):
    """Factory function to create LLM client."""
    if client_type == "openai":
        return SecureOpenAIClient(model=model, **kwargs)
    elif client_type == "mock":
        return MockLLMClient(model=model)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


def generate_rag_response(
    prompt: RAGPrompt,
    client_type: str = "mock",
    **client_kwargs
) -> RAGResponse:
    """Simple function to generate RAG response without managing client lifecycle."""
    client = create_llm_client(client_type, **client_kwargs)
    return client.generate_response(prompt)