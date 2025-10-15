"""
Atomic fact extractor using regex patterns and LLM.
"""

import sys
import json
import logging
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
vector_ingest_path = project_root / "vector-ingest" / "src"
sys.path.insert(0, str(vector_ingest_path))

from chunking.processors.llm_utils import SecureAPIKeyManager

# Import utils from current package
import utils
from utils import (
    extract_dates, extract_numbers, extract_currencies,
    find_answer_span, normalize_text
)

logger = logging.getLogger(__name__)


@dataclass
class AtomicFact:
    """Represents an atomic fact extracted from a chunk."""
    fact_id: str
    chunk_id: str
    fact_type: str  # number, date, currency, triple, key_value
    fact_text: str  # The fact content
    answer_span: str  # Exact answer text
    answer_start: int  # Char offset in chunk
    answer_end: int  # Char offset end
    entities: List[str]  # For multi-hop linking
    metadata: Dict[str, Any]  # Additional info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FactExtractor:
    """
    Extracts atomic facts from chunks using regex + LLM.
    
    Facts include:
    - Numbers, dates, currencies (regex)
    - Subject-relation-object triples (LLM)
    - Key-value pairs from tables (LLM)
    """
    
    def __init__(self, config, llm_manager: SecureAPIKeyManager):
        """
        Initialize fact extractor.

        Args:
            config: SyntheticEvalConfig instance
            llm_manager: SecureAPIKeyManager for LLM calls
        """
        self.config = config
        self.llm_manager = llm_manager
        self.fact_counter = 0
        self._async_client = None

        logger.info(f"Initialized FactExtractor with model: {config.model_name}")

    def extract_facts_batch(self, chunks: List[Dict[str, Any]], concurrency: int = 5, progress_callback=None) -> List[List[AtomicFact]]:
        """
        Extract facts from multiple chunks in parallel (async).

        Args:
            chunks: List of chunk dictionaries
            concurrency: Number of concurrent LLM calls
            progress_callback: Optional callback to report progress (called with chunk count)

        Returns:
            List of fact lists (one per chunk)
        """
        # Create new event loop for this batch
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._extract_facts_batch_async(chunks, concurrency, progress_callback))
            return result
        finally:
            # Clean up properly
            loop.run_until_complete(self._cleanup_async_client())
            loop.close()

    async def _cleanup_async_client(self):
        """Clean up async client."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

    async def _get_async_client(self):
        """Get or create async OpenAI client."""
        if self._async_client is None:
            import openai
            api_key = self.llm_manager.get_api_key()
            self._async_client = openai.AsyncOpenAI(api_key=api_key)
        return self._async_client

    async def _extract_facts_batch_async(self, chunks: List[Dict[str, Any]], concurrency: int, progress_callback=None) -> List[List[AtomicFact]]:
        """Async batch processing of chunks."""
        semaphore = asyncio.Semaphore(concurrency)
        completed_count = 0

        async def process_chunk(chunk):
            nonlocal completed_count
            async with semaphore:
                try:
                    result = await self._extract_facts_async(chunk)
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count)
                    return result
                except Exception as e:
                    logger.warning(f"Error processing chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count)
                    return []

        tasks = [process_chunk(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _extract_facts_async(self, chunk: Dict[str, Any]) -> List[AtomicFact]:
        """Async version of extract_facts."""
        chunk_id = chunk.get('chunk_id', '')
        content = chunk.get('content', '')

        if not content:
            return []

        facts = []

        # 1. Extract structured data with regex (synchronous)
        facts.extend(self._extract_date_facts(chunk_id, content))
        facts.extend(self._extract_number_facts(chunk_id, content))
        facts.extend(self._extract_currency_facts(chunk_id, content))

        # 2. Extract semantic facts with async LLM call
        semantic_facts = await self._extract_semantic_facts_llm_async(chunk_id, content)
        facts.extend(semantic_facts)

        return facts

    def extract_facts(self, chunk: Dict[str, Any]) -> List[AtomicFact]:
        """
        Extract all atomic facts from a chunk.

        Args:
            chunk: Chunk dictionary with 'chunk_id' and 'content'

        Returns:
            List of AtomicFact objects
        """
        chunk_id = chunk.get('chunk_id', '')
        content = chunk.get('content', '')

        if not content:
            logger.warning(f"Empty content for chunk {chunk_id}")
            return []

        logger.debug(f"Extracting facts from chunk {chunk_id}")

        facts = []

        # 1. Extract structured data with regex
        facts.extend(self._extract_date_facts(chunk_id, content))
        facts.extend(self._extract_number_facts(chunk_id, content))
        facts.extend(self._extract_currency_facts(chunk_id, content))

        # 2. Extract semantic facts with unified LLM call (optimized)
        facts.extend(self._extract_semantic_facts_llm(chunk_id, content))

        logger.debug(f"Extracted {len(facts)} facts from chunk {chunk_id}")

        return facts
    
    def _generate_fact_id(self, chunk_id: str) -> str:
        """Generate unique fact ID."""
        self.fact_counter += 1
        return f"{chunk_id}_fact_{self.fact_counter}"
    
    def _extract_date_facts(self, chunk_id: str, content: str) -> List[AtomicFact]:
        """Extract date facts using regex - optimized."""
        facts = []
        dates = extract_dates(content)
        
        # Limit dates to avoid too many facts
        for date in dates[:5]:  # Max 5 dates per chunk
            start, end = find_answer_span(date, content)
            if start == -1:  # Skip if not found
                continue
            
            # Optimized context extraction
            context_start = max(0, start - 30)  # Reduced context size
            context_end = min(len(content), end + 30)
            context = content[context_start:context_end].strip()
            
            fact = AtomicFact(
                fact_id=self._generate_fact_id(chunk_id),
                chunk_id=chunk_id,
                fact_type="date",
                fact_text=f"Date: {date}",  # Simplified fact text
                answer_span=date,
                answer_start=start,
                answer_end=end,
                entities=[date],
                metadata={"pattern": "regex", "context": context}
            )
            facts.append(fact)
        
        return facts
    
    def _extract_number_facts(self, chunk_id: str, content: str) -> List[AtomicFact]:
        """Extract numeric facts using regex - optimized."""
        facts = []
        numbers = extract_numbers(content)
        
        # Pre-filter and limit in one pass for efficiency
        significant_numbers = []
        for number in numbers:
            if self._is_significant_number(number):
                significant_numbers.append(number)
                if len(significant_numbers) >= 5:  # Reduced limit
                    break
        
        for number in significant_numbers:
            start, end = find_answer_span(number, content)
            if start == -1:
                continue
            
            # Optimized context extraction
            context_start = max(0, start - 30)
            context_end = min(len(content), end + 30)
            context = content[context_start:context_end].strip()
            
            fact = AtomicFact(
                fact_id=self._generate_fact_id(chunk_id),
                chunk_id=chunk_id,
                fact_type="number",
                fact_text=f"Number: {number}",
                answer_span=number,
                answer_start=start,
                answer_end=end,
                entities=[number],
                metadata={"pattern": "regex", "context": context}
            )
            facts.append(fact)
        
        return facts
    
    def _extract_currency_facts(self, chunk_id: str, content: str) -> List[AtomicFact]:
        """Extract currency amounts using regex."""
        facts = []
        currencies = extract_currencies(content)
        
        for currency in currencies:
            start, end = find_answer_span(currency, content)
            
            # Get context
            context_start = max(0, start - 50)
            context_end = min(len(content), end + 50)
            context = content[context_start:context_end]
            
            fact = AtomicFact(
                fact_id=self._generate_fact_id(chunk_id),
                chunk_id=chunk_id,
                fact_type="currency",
                fact_text=f"Currency amount: {currency} (context: {context})",
                answer_span=currency,
                answer_start=start,
                answer_end=end,
                entities=[currency],
                metadata={"pattern": "regex", "context": context}
            )
            facts.append(fact)
        
        return facts
    
    def _is_significant_number(self, number_str: str) -> bool:
        """Check if number is significant (not year, page number, etc.)."""
        # Remove formatting
        cleaned = re.sub(r'[,$€£¥\s]', '', number_str)
        
        # Filter out years (4-digit numbers between 1900-2100)
        if re.match(r'^(19|20)\d{2}$', cleaned):
            return False
        
        # Filter out small integers (likely page numbers, etc.)
        try:
            val = float(cleaned.lower().replace('k', '').replace('m', '').replace('b', ''))
            if 0 < val < 100 and '.' not in cleaned:
                return False
        except:
            pass
        
        return True
    
    async def _extract_semantic_facts_llm_async(self, chunk_id: str, content: str) -> List[AtomicFact]:
        """
        Async version: Extract both triples and key-value pairs in a single unified LLM call.

        Args:
            chunk_id: Chunk identifier
            content: Chunk content

        Returns:
            List of AtomicFact objects for both triples and key-values
        """
        # Limit content length for LLM
        max_content_len = 2000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        prompt = f"""Extract atomic facts from this text. Identify BOTH:
1. Subject-Relation-Object triples (factual statements)
2. Key-Value pairs (metrics, attributes from tables/structured data)

Text: {content}

For each fact, provide:
- fact_type: "triple" or "key_value"
- For triples: subject, relation, object
- For key-values: key, value
- answer_span: the exact text to extract
- entities: key entities mentioned
- fact_text: natural language description

Output JSON format (array combining both types):
[
  {{
    "fact_type": "triple",
    "triple": ["Elastic N.V.", "fiscal_year_revenue", "$1.2B"],
    "answer_span": "$1.2B",
    "entities": ["Elastic N.V.", "2024"],
    "fact_text": "Elastic N.V. reported fiscal year revenue of $1.2B"
  }},
  {{
    "fact_type": "key_value",
    "key": "EBITDA",
    "value": "$400M",
    "answer_span": "$400M",
    "entities": ["2024", "Q1"],
    "fact_text": "Q1 2024 EBITDA was $400M"
  }}
]

Extract 5-8 key facts total. Output ONLY valid JSON, no other text."""

        try:
            # Use shared async client
            client = await self._get_async_client()

            llm_params = self.config.get_llm_params({
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": "You are a fact extraction assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            })

            logger.debug(f"Calling LLM for chunk {chunk_id}...")
            response = await client.chat.completions.create(**llm_params)
            logger.debug(f"LLM response received for chunk {chunk_id}")

            response_text = response.choices[0].message.content.strip()

            # Parse JSON response
            if response_text.startswith("```"):
                response_text = re.sub(r'```json?\n?', '', response_text)
                response_text = re.sub(r'```\n?$', '', response_text)

            facts_data = json.loads(response_text)

            # Convert to AtomicFact objects
            facts = []
            for fact_data in facts_data:
                fact_type = fact_data.get('fact_type', 'triple')
                answer_span = fact_data.get('answer_span', '')

                # Skip if None or empty
                if answer_span is None:
                    continue
                # Convert to string if needed
                if isinstance(answer_span, (list, dict)):
                    answer_span = json.dumps(answer_span)
                elif not isinstance(answer_span, str):
                    answer_span = str(answer_span)
                if not answer_span:  # Skip empty values
                    continue

                start, end = find_answer_span(answer_span, content)

                # Build metadata based on fact type
                if fact_type == "key_value":
                    metadata = {
                        "key": fact_data.get('key', ''),
                        "value": fact_data.get('value', ''),
                        "source": "llm"
                    }
                else:  # triple
                    metadata = {
                        "triple": fact_data.get('triple', []),
                        "source": "llm"
                    }

                fact = AtomicFact(
                    fact_id=self._generate_fact_id(chunk_id),
                    chunk_id=chunk_id,
                    fact_type=fact_type,
                    fact_text=fact_data.get('fact_text', ''),
                    answer_span=answer_span,
                    answer_start=start,
                    answer_end=end,
                    entities=fact_data.get('entities', []),
                    metadata=metadata
                )
                facts.append(fact)

            return facts

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text[:500] if 'response_text' in locals() else 'NO RESPONSE'}")
            return []
        except Exception as e:
            logger.error(f"Error extracting semantic facts with LLM: {e}")
            return []

    def _extract_semantic_facts_llm(self, chunk_id: str, content: str) -> List[AtomicFact]:
        """
        Extract both triples and key-value pairs in a single unified LLM call (optimized).

        Args:
            chunk_id: Chunk identifier
            content: Chunk content

        Returns:
            List of AtomicFact objects for both triples and key-values
        """
        # Limit content length for LLM
        max_content_len = 2000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        prompt = f"""Extract atomic facts from this text. Identify BOTH:
1. Subject-Relation-Object triples (factual statements)
2. Key-Value pairs (metrics, attributes from tables/structured data)

Text: {content}

For each fact, provide:
- fact_type: "triple" or "key_value"
- For triples: subject, relation, object
- For key-values: key, value
- answer_span: the exact text to extract
- entities: key entities mentioned
- fact_text: natural language description

Output JSON format (array combining both types):
[
  {{
    "fact_type": "triple",
    "triple": ["Elastic N.V.", "fiscal_year_revenue", "$1.2B"],
    "answer_span": "$1.2B",
    "entities": ["Elastic N.V.", "2024"],
    "fact_text": "Elastic N.V. reported fiscal year revenue of $1.2B"
  }},
  {{
    "fact_type": "key_value",
    "key": "EBITDA",
    "value": "$400M",
    "answer_span": "$400M",
    "entities": ["2024", "Q1"],
    "fact_text": "Q1 2024 EBITDA was $400M"
  }}
]

Extract 5-8 key facts total. Output ONLY valid JSON, no other text."""

        try:
            api_key = self.llm_manager.get_api_key()

            import openai
            client = openai.OpenAI(api_key=api_key)

            llm_params = self.config.get_llm_params({
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": "You are a fact extraction assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            })
            response = client.chat.completions.create(**llm_params)

            response_text = response.choices[0].message.content.strip()

            # Parse JSON response
            if response_text.startswith("```"):
                response_text = re.sub(r'```json?\n?', '', response_text)
                response_text = re.sub(r'```\n?$', '', response_text)

            facts_data = json.loads(response_text)

            # Convert to AtomicFact objects
            facts = []
            for fact_data in facts_data:
                fact_type = fact_data.get('fact_type', 'triple')
                answer_span = fact_data.get('answer_span', '')

                # Skip if None or empty
                if answer_span is None:
                    continue
                # Convert to string if needed
                if isinstance(answer_span, (list, dict)):
                    answer_span = json.dumps(answer_span)
                elif not isinstance(answer_span, str):
                    answer_span = str(answer_span)
                if not answer_span:  # Skip empty values
                    continue

                start, end = find_answer_span(answer_span, content)

                # Build metadata based on fact type
                if fact_type == "key_value":
                    metadata = {
                        "key": fact_data.get('key', ''),
                        "value": fact_data.get('value', ''),
                        "source": "llm"
                    }
                else:  # triple
                    metadata = {
                        "triple": fact_data.get('triple', []),
                        "source": "llm"
                    }

                fact = AtomicFact(
                    fact_id=self._generate_fact_id(chunk_id),
                    chunk_id=chunk_id,
                    fact_type=fact_type,
                    fact_text=fact_data.get('fact_text', ''),
                    answer_span=answer_span,
                    answer_start=start,
                    answer_end=end,
                    entities=fact_data.get('entities', []),
                    metadata=metadata
                )
                facts.append(fact)

            return facts

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text[:500] if 'response_text' in locals() else 'NO RESPONSE'}")
            return []
        except Exception as e:
            logger.error(f"Error extracting semantic facts with LLM: {e}", exc_info=True)
            return []

    def _extract_triples_llm(self, chunk_id: str, content: str) -> List[AtomicFact]:
        """
        Extract (subject, relation, object) triples using LLM.
        
        Args:
            chunk_id: Chunk identifier
            content: Chunk content
            
        Returns:
            List of AtomicFact objects for triples
        """
        # Limit content length for LLM
        max_content_len = 2000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        
        prompt = f"""Extract atomic facts as (subject, relation, object) triples from this text. Focus on concrete, factual statements.

Text: {content}

For each fact, provide:
1. The triple as [subject, relation, object]
2. The exact answer span (the object/value)
3. Key entities mentioned

Output JSON format (array of facts):
[
  {{
    "triple": ["Elastic N.V.", "fiscal_year_revenue", "$1.2B"],
    "answer_span": "$1.2B",
    "entities": ["Elastic N.V.", "fiscal_year", "2024"],
    "fact_text": "Elastic N.V. reported fiscal year revenue of $1.2B"
  }}
]

Extract 3-5 key facts. Output ONLY valid JSON, no other text."""
        
        try:
            api_key = self.llm_manager.get_api_key()
            
            # Call OpenAI API
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            llm_params = self.config.get_llm_params({
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": "You are a fact extraction assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            })
            response = client.chat.completions.create(**llm_params)
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = re.sub(r'```json?\n?', '', response_text)
                response_text = re.sub(r'```\n?$', '', response_text)
            
            triples_data = json.loads(response_text)
            
            # Convert to AtomicFact objects
            facts = []
            for triple_data in triples_data:
                answer_span = triple_data.get('answer_span', '')
                # Skip if None or empty
                if answer_span is None:
                    continue
                # Convert to string if needed
                if isinstance(answer_span, (list, dict)):
                    answer_span = json.dumps(answer_span)
                elif not isinstance(answer_span, str):
                    answer_span = str(answer_span)
                if not answer_span:  # Skip empty values
                    continue
                start, end = find_answer_span(answer_span, content)
                
                fact = AtomicFact(
                    fact_id=self._generate_fact_id(chunk_id),
                    chunk_id=chunk_id,
                    fact_type="triple",
                    fact_text=triple_data.get('fact_text', ''),
                    answer_span=answer_span,
                    answer_start=start,
                    answer_end=end,
                    entities=triple_data.get('entities', []),
                    metadata={
                        "triple": triple_data.get('triple', []),
                        "source": "llm"
                    }
                )
                facts.append(fact)
            
            return facts
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text[:500] if 'response_text' in locals() else 'NO RESPONSE'}")
            return []
        except Exception as e:
            logger.error(f"Error extracting triples with LLM: {e}", exc_info=True)
            return []
    
    def _extract_key_values_llm(self, chunk_id: str, content: str) -> List[AtomicFact]:
        """
        Extract key-value pairs (especially from tables) using LLM.
        
        Args:
            chunk_id: Chunk identifier
            content: Chunk content
            
        Returns:
            List of AtomicFact objects for key-value pairs
        """
        # Only process if chunk looks like it has structured data
        if not self._has_structured_data(content):
            return []
        
        # Limit content length
        max_content_len = 2000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        
        prompt = f"""Extract key-value pairs from this text, especially from tables or structured data.

Text: {content}

For each key-value pair, provide:
1. The key (metric/field name)
2. The value
3. Any associated entities (e.g., year, product)

Output JSON format (array of key-value facts):
[
  {{
    "key": "EBITDA",
    "value": "$400M",
    "entities": ["2024", "Q1"],
    "fact_text": "Q1 2024 EBITDA was $400M"
  }}
]

Extract 3-5 key pairs. Output ONLY valid JSON, no other text."""
        
        try:
            api_key = self.llm_manager.get_api_key()
            
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            llm_params = self.config.get_llm_params({
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": "You are a data extraction assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            })
            response = client.chat.completions.create(**llm_params)
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            if response_text.startswith("```"):
                response_text = re.sub(r'```json?\n?', '', response_text)
                response_text = re.sub(r'```\n?$', '', response_text)
            
            kv_data = json.loads(response_text)
            
            # Convert to AtomicFact objects
            facts = []
            for kv in kv_data:
                value = kv.get('value', '')
                # Skip if value is None or empty
                if value is None:
                    continue
                # Convert value to string if it's not already
                if isinstance(value, (list, dict)):
                    value = json.dumps(value)
                elif not isinstance(value, str):
                    value = str(value)
                if not value:  # Skip empty values
                    continue
                start, end = find_answer_span(value, content)
                
                fact = AtomicFact(
                    fact_id=self._generate_fact_id(chunk_id),
                    chunk_id=chunk_id,
                    fact_type="key_value",
                    fact_text=kv.get('fact_text', ''),
                    answer_span=value,
                    answer_start=start,
                    answer_end=end,
                    entities=kv.get('entities', []),
                    metadata={
                        "key": kv.get('key', ''),
                        "value": value,
                        "source": "llm"
                    }
                )
                facts.append(fact)
            
            return facts
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text[:500] if 'response_text' in locals() else 'NO RESPONSE'}")
            return []
        except Exception as e:
            logger.error(f"Error extracting key-values with LLM: {e}")
            return []
    
    def _has_structured_data(self, content: str) -> bool:
        """Check if content likely has structured data (tables, lists)."""
        # Look for table markers
        table_indicators = ['|', '---', 'Table', 'Row', 'Column']
        
        for indicator in table_indicators:
            if indicator in content:
                return True
        
        # Look for repeated patterns (lists, etc.)
        lines = content.split('\n')
        if len(lines) > 5:
            # Check for bullet points or numbered lists
            bullet_count = sum(1 for line in lines if line.strip().startswith(('-', '*', '•', '1.', '2.')))
            if bullet_count > 3:
                return True
        
        return False

