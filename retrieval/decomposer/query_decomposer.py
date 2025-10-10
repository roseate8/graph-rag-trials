"""
Intelligent query decomposition using LLM to generate optimal sub-queries.

This module analyzes user queries and generates 1-5 sub-queries based on:
- Sub-questions: Breaking complex multi-part queries
- Paraphrases: Alternative phrasings for ambiguous terms
- Expansions: Adding context for terse queries
- Compressions: Extracting core intent from verbose queries
- Original: Always considering the original query
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

# Import secure LLM utilities
vector_ingest_path = Path(__file__).parent.parent / "vector-ingest" / "src"
sys.path.append(str(vector_ingest_path))

from chunking.processors.llm_utils import get_openai_api_key, has_openai_api_key
import requests

logger = logging.getLogger(__name__)


@dataclass
class DecomposedQuery:
    """Container for decomposed query results."""
    original_query: str
    sub_queries: List[str]
    decomposition_reasoning: str
    query_count: int
    decomposition_type: str  # "sub-questions", "paraphrasing", "expansion", etc.
    
    def __str__(self) -> str:
        return f"DecomposedQuery(original='{self.original_query[:50]}...', count={self.query_count})"


class QueryDecomposer:
    """
    LLM-powered intelligent query decomposition.
    
    Analyzes queries and generates 1-5 sub-queries based on complexity and need.
    Uses secure OpenAI API access via llm_utils.
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
            model: OpenAI model to use for decomposition
            temperature: LLM temperature (lower = more focused)
        """
        if not 1 <= max_sub_queries <= 5:
            raise ValueError("max_sub_queries must be between 1 and 5")
        
        self.max_sub_queries = max_sub_queries
        self.model = model
        self.temperature = temperature
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        logger.info(f"Initialized QueryDecomposer (max_sub_queries={max_sub_queries}, model={model})")
    
    def decompose_query(self, query: str) -> DecomposedQuery:
        """
        Decompose a query into optimal sub-queries.
        
        Args:
            query: The user's original query
            
        Returns:
            DecomposedQuery with 1-5 sub-queries and reasoning
        """
        logger.info(f"Decomposing query: {query[:100]}...")
        
        try:
            # Build decomposition prompt
            prompt = self._build_decomposition_prompt(query)
            
            # Call LLM
            response_data = self._call_llm(prompt)
            
            # Parse response
            decomposed = self._parse_decomposition_response(query, response_data)
            
            logger.info(f"Decomposed into {decomposed.query_count} sub-queries: {decomposed.decomposition_type}")
            return decomposed
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            # Fallback: return original query only
            return self._create_fallback_decomposition(query)
    
    def _build_decomposition_prompt(self, query: str) -> str:
        """Build the prompt for LLM query decomposition."""
        prompt = f"""You are an expert at analyzing search queries and generating optimal sub-queries for retrieval systems.

Analyze the following user query and generate 1-{self.max_sub_queries} sub-queries that will help retrieve the most relevant information.

QUERY: "{query}"

Your task:
1. Determine if the query needs decomposition
2. Generate sub-queries based on these strategies:
   - **Sub-questions**: Break complex multi-part queries into individual questions
   - **Paraphrases**: Create alternative phrasings for ambiguous or technical terms
   - **Expansion**: Add context or detail if the query is too terse
   - **Compression**: Extract core intent if the query is verbose or rambling
   - **Original**: Always consider including the original query

Rules:
- Generate ONLY 1-{self.max_sub_queries} sub-queries
- Each sub-query must be a complete, standalone question or search phrase
- Avoid redundant or overlapping sub-queries
- If the query is already clear and focused, return ONLY the original query
- Sub-queries should cover different aspects or interpretations of the original query

Return ONLY valid JSON with this exact structure (no markdown, no code blocks):
{{
  "sub_queries": ["query1", "query2", ...],
  "reasoning": "brief explanation of why these sub-queries were generated",
  "decomposition_type": "one of: original-only|sub-questions|paraphrasing|expansion|compression|mixed"
}}

Examples:

Example 1 - Complex multi-part query:
Query: "What are the revenue and expenses for Q3 2023?"
Response:
{{
  "sub_queries": [
    "What are the revenue and expenses for Q3 2023?",
    "What was the total revenue in Q3 2023?",
    "What were the operating expenses in Q3 2023?"
  ],
  "reasoning": "Query has multiple aspects (revenue AND expenses), so breaking into focused sub-questions helps retrieve specific information for each component",
  "decomposition_type": "sub-questions"
}}

Example 2 - Ambiguous technical query:
Query: "What is EPS?"
Response:
{{
  "sub_queries": [
    "What is EPS?",
    "What is earnings per share?",
    "How is EPS calculated?",
    "EPS definition and meaning"
  ],
  "reasoning": "Acronym EPS is ambiguous - generating paraphrases with full term and related questions to capture different explanations",
  "decomposition_type": "paraphrasing"
}}

Example 3 - Already optimal query:
Query: "What is the company's mission statement?"
Response:
{{
  "sub_queries": [
    "What is the company's mission statement?"
  ],
  "reasoning": "Query is already clear, focused, and unambiguous - no decomposition needed",
  "decomposition_type": "original-only"
}}

Example 4 - Terse query needing expansion:
Query: "profits"
Response:
{{
  "sub_queries": [
    "profits",
    "What are the company's profits?",
    "What is the net profit margin?",
    "What were the profit trends over time?"
  ],
  "reasoning": "Query is too terse - expanding with context to capture different profit-related information",
  "decomposition_type": "expansion"
}}

Now analyze the given query and generate the JSON response:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> dict:
        """Call OpenAI API securely using llm_utils."""
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
            "response_format": {"type": "json_object"}  # Force JSON output
        }
        
        logger.debug(f"Calling {self.model} for query decomposition")
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            error_msg = f"OpenAI API error {response.status_code}: {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        return response.json()
    
    def _parse_decomposition_response(self, original_query: str, response_data: dict) -> DecomposedQuery:
        """Parse LLM response and extract decomposition."""
        try:
            content = response_data["choices"][0]["message"]["content"]
            
            # Parse JSON
            data = json.loads(content)
            
            # Extract fields with validation
            sub_queries = data.get("sub_queries", [original_query])
            reasoning = data.get("reasoning", "No reasoning provided")
            decomposition_type = data.get("decomposition_type", "unknown")
            
            # Validate sub_queries
            if not isinstance(sub_queries, list) or len(sub_queries) == 0:
                logger.warning("Invalid sub_queries format, using original query")
                sub_queries = [original_query]
            
            # Enforce max_sub_queries limit
            if len(sub_queries) > self.max_sub_queries:
                logger.warning(f"LLM generated {len(sub_queries)} sub-queries, limiting to {self.max_sub_queries}")
                sub_queries = sub_queries[:self.max_sub_queries]
            
            # Clean sub-queries
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
            
            # Ensure at least original query is included
            if not sub_queries:
                sub_queries = [original_query]
            
            return DecomposedQuery(
                original_query=original_query,
                sub_queries=sub_queries,
                decomposition_reasoning=reasoning,
                query_count=len(sub_queries),
                decomposition_type=decomposition_type
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._create_fallback_decomposition(original_query)
        except Exception as e:
            logger.error(f"Error parsing decomposition response: {e}")
            return self._create_fallback_decomposition(original_query)
    
    def _create_fallback_decomposition(self, query: str) -> DecomposedQuery:
        """Create fallback decomposition with only the original query."""
        logger.info("Using fallback decomposition (original query only)")
        return DecomposedQuery(
            original_query=query,
            sub_queries=[query],
            decomposition_reasoning="Fallback to original query due to decomposition error",
            query_count=1,
            decomposition_type="fallback"
        )
    
    def can_decompose(self) -> bool:
        """Check if decomposer can function (has valid API key)."""
        return has_openai_api_key()


def decompose_query_simple(query: str, max_sub_queries: int = 5) -> List[str]:
    """
    Simple function to decompose a query without managing decomposer lifecycle.
    
    Args:
        query: The user's query
        max_sub_queries: Maximum sub-queries to generate
        
    Returns:
        List of sub-queries (1-5)
    """
    decomposer = QueryDecomposer(max_sub_queries=max_sub_queries)
    result = decomposer.decompose_query(query)
    return result.sub_queries

