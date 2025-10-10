"""
Query decomposition using LLM for intelligent multi-view retrieval.

This module uses LLM to analyze user queries and generate 1-5 optimal sub-queries
including paraphrases, expansions, compressions, and sub-questions.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Setup LLM utils imports for secure API key handling
vector_ingest_path = Path(__file__).parent.parent / "vector-ingest" / "src"
sys.path.append(str(vector_ingest_path))

from chunking.processors.llm_utils import get_openai_api_key, has_openai_api_key

logger = logging.getLogger(__name__)


@dataclass
class DecomposedQuery:
    """Container for decomposed query with metadata."""
    original_query: str
    sub_queries: List[str]
    decomposition_reasoning: str
    decomposition_type: str
    query_count: int
    
    def __str__(self) -> str:
        return f"DecomposedQuery(count={self.query_count}, type={self.decomposition_type})"


class QueryDecomposer:
    """
    Intelligent query decomposition using LLM.
    
    Analyzes user queries and generates 1-5 optimal sub-queries based on:
    - Complexity: Does it need breaking into sub-questions?
    - Ambiguity: Does it need paraphrasing?
    - Terseness: Does it need expansion?
    - Verbosity: Does it need compression?
    """
    
    def __init__(
        self,
        max_sub_queries: int = 5,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3
    ):
        """
        Initialize query decomposer.
        
        Args:
            max_sub_queries: Maximum number of sub-queries to generate (1-5)
            model: LLM model to use for decomposition
            temperature: Temperature for LLM (0.3 for more focused, deterministic)
        """
        self.max_sub_queries = max(1, min(max_sub_queries, 5))  # Clamp to [1, 5]
        self.model = model
        self.temperature = temperature
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        logger.info(f"Initialized QueryDecomposer (max_sub_queries={self.max_sub_queries}, model={model})")
    
    def decompose_query(self, query: str) -> DecomposedQuery:
        """
        Decompose user query into optimal sub-queries for retrieval.
        
        Args:
            query: Original user query
            
        Returns:
            DecomposedQuery with 1-5 sub-queries
        """
        if not query or not query.strip():
            logger.warning("Empty query provided, returning original")
            return DecomposedQuery(
                original_query=query,
                sub_queries=[query],
                decomposition_reasoning="Empty query, no decomposition needed",
                decomposition_type="original-only",
                query_count=1
            )
        
        try:
            # Check if LLM is available
            if not has_openai_api_key():
                logger.warning("No OpenAI API key available, using original query only")
                return self._create_fallback_decomposition(query)
            
            logger.debug(f"Decomposing query: {query[:100]}...")
            
            # Build decomposition prompt
            prompt = self._build_decomposition_prompt(query)
            
            # Call LLM
            llm_response = self._call_llm(prompt)
            
            # Parse response
            decomposed = self._parse_llm_response(query, llm_response)
            
            logger.info(f"Decomposed into {decomposed.query_count} sub-queries (type: {decomposed.decomposition_type})")
            
            return decomposed
            
        except Exception as e:
            logger.error(f"Error during query decomposition: {e}")
            logger.info("Falling back to original query only")
            return self._create_fallback_decomposition(query)
    
    def _build_decomposition_prompt(self, query: str) -> str:
        """Build prompt for LLM query decomposition."""
        prompt = f"""You are a query analysis expert for document retrieval systems. Your task is to analyze a user's query and generate optimal sub-queries that will improve retrieval quality.

ORIGINAL QUERY: "{query}"

ANALYSIS GUIDELINES:
1. **Assess query complexity**: Is it simple (1 concept) or complex (multiple concepts)?
2. **Identify decomposition needs**:
   - Sub-questions: Break multi-part queries into separate questions
   - Paraphrases: Generate alternative phrasings for ambiguous terms
   - Expansion: Add context to very short/terse queries
   - Compression: Extract core intent from verbose queries
   - Original: Always consider including the original query

3. **Generate 1-{self.max_sub_queries} sub-queries** (quality over quantity):
   - Simple queries → 1-2 sub-queries
   - Moderate queries → 2-3 sub-queries
   - Complex queries → 3-5 sub-queries

EXAMPLES:

Example 1 - Simple query:
Query: "What is revenue?"
Analysis: Simple, clear query - needs minimal decomposition
Sub-queries: ["What is revenue?", "revenue definition"]
Type: "paraphrasing"

Example 2 - Complex query:
Query: "What are the revenue and expense trends, and how do they impact profitability?"
Analysis: Multiple concepts - break into sub-questions
Sub-queries: [
  "What are the revenue trends?",
  "What are the expense trends?",
  "How do revenue and expenses impact profitability?",
  "What is the profitability trend?",
  "What are the revenue and expense trends?"
]
Type: "sub-questions"

Example 3 - Terse query:
Query: "EPS"
Analysis: Too short - needs expansion
Sub-queries: [
  "EPS",
  "What is earnings per share?",
  "earnings per share metrics"
]
Type: "expansion"

Example 4 - Verbose query:
Query: "I would like to understand the detailed financial performance metrics specifically related to revenue generation and cost structures over the past fiscal year"
Analysis: Verbose - extract core intent
Sub-queries: [
  "revenue performance last fiscal year",
  "cost structure last fiscal year",
  "financial performance metrics revenue and costs",
  "What are the revenue and cost trends?"
]
Type: "compression"

OUTPUT FORMAT (JSON only, no markdown):
{{
  "sub_queries": ["query1", "query2", ...],
  "reasoning": "Brief explanation of why these sub-queries were generated",
  "decomposition_type": "original-only|sub-questions|paraphrasing|expansion|compression|mixed"
}}

Now analyze the query and provide your JSON response:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API with secure key handling."""
        import requests
        
        api_key = get_openai_api_key()
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a query analysis expert. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": 500,
            "response_format": {"type": "json_object"}  # Ensure JSON response
        }
        
        logger.debug(f"Calling LLM for query decomposition (model={self.model})")
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            error_msg = f"LLM API error {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        response_data = response.json()
        llm_output = response_data["choices"][0]["message"]["content"]
        
        logger.debug(f"LLM decomposition response: {llm_output[:200]}...")
        
        return llm_output
    
    def _parse_llm_response(self, original_query: str, llm_response: str) -> DecomposedQuery:
        """Parse LLM response into DecomposedQuery."""
        try:
            # Parse JSON response
            data = json.loads(llm_response)
            
            # Extract sub-queries
            sub_queries = data.get("sub_queries", [original_query])
            
            # Ensure we have valid sub-queries
            if not sub_queries or not isinstance(sub_queries, list):
                logger.warning("Invalid sub_queries in LLM response, using original")
                sub_queries = [original_query]
            
            # Clean and validate sub-queries
            sub_queries = [q.strip() for q in sub_queries if q and isinstance(q, str) and q.strip()]
            
            # Ensure original is included and limit to max
            if original_query not in sub_queries:
                sub_queries.insert(0, original_query)
            
            # Limit to max_sub_queries
            if len(sub_queries) > self.max_sub_queries:
                logger.info(f"Truncating {len(sub_queries)} sub-queries to {self.max_sub_queries}")
                sub_queries = sub_queries[:self.max_sub_queries]
            
            # Extract metadata
            reasoning = data.get("reasoning", "LLM-generated decomposition")
            decomposition_type = data.get("decomposition_type", "mixed")
            
            return DecomposedQuery(
                original_query=original_query,
                sub_queries=sub_queries,
                decomposition_reasoning=reasoning,
                decomposition_type=decomposition_type,
                query_count=len(sub_queries)
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {llm_response[:500]}")
            return self._create_fallback_decomposition(original_query)
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_decomposition(original_query)
    
    def _create_fallback_decomposition(self, query: str) -> DecomposedQuery:
        """Create fallback decomposition when LLM is unavailable or fails."""
        return DecomposedQuery(
            original_query=query,
            sub_queries=[query],
            decomposition_reasoning="Fallback: LLM unavailable or decomposition failed",
            decomposition_type="original-only",
            query_count=1
        )
    
    def can_decompose(self) -> bool:
        """Check if decomposer can make LLM calls."""
        return has_openai_api_key()


def decompose_query_simple(
    query: str,
    max_sub_queries: int = 5
) -> DecomposedQuery:
    """
    Simple function to decompose a query without managing decomposer lifecycle.
    
    Args:
        query: User query to decompose
        max_sub_queries: Maximum sub-queries to generate
        
    Returns:
        DecomposedQuery with sub-queries
    """
    decomposer = QueryDecomposer(max_sub_queries=max_sub_queries)
    return decomposer.decompose_query(query)

