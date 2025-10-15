"""
Semantic proposition extractor for generating evaluation datasets.

Extracts self-contained semantic propositions (claims) from business documents
rather than low-level atomic data types. This approach produces higher-quality
facts suitable for question generation.
"""

import sys
import json
import logging
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
vector_ingest_path = project_root / "vector-ingest" / "src"
sys.path.insert(0, str(vector_ingest_path))

from chunking.processors.llm_utils import SecureAPIKeyManager
from utils import find_answer_span

logger = logging.getLogger(__name__)


@dataclass
class AtomicFact:
    """
    Represents a semantic proposition extracted from a chunk.

    A proposition is a complete, self-contained statement that can be questioned.
    Examples:
    - "Elastic N.V. acquired Build Security Ltd. in December 2021"
    - "Q1 2024 total revenue was $400 million, representing 20% year-over-year growth"
    """
    fact_id: str
    chunk_id: str
    fact_type: str  # factual_claim, temporal_event, or comparative_statement
    fact_text: str  # Complete semantic proposition
    answer_span: str  # Key piece that would answer a question
    answer_start: int  # Character offset in chunk
    answer_end: int  # Character offset end
    entities: List[str]  # Semantic entities for multi-hop linking
    metadata: Dict[str, Any]  # Additional info

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class FactExtractor:
    """
    Extracts semantic propositions from business documents using LLM.

    Proposition types:
    1. factual_claim: Complete statements about entities, metrics, or events
    2. temporal_event: Events anchored to specific times
    3. comparative_statement: Comparisons, changes, or trends over time
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

        logger.info(f"Initialized FactExtractor (semantic propositions) with model: {config.model_name}")

    async def _get_async_client(self):
        """Get or create async OpenAI client (reused across requests)."""
        if self._async_client is None:
            import openai
            api_key = self.llm_manager.get_api_key()
            self._async_client = openai.AsyncOpenAI(api_key=api_key)
        return self._async_client

    def extract_facts_batch(
        self,
        chunks: List[Dict[str, Any]],
        concurrency: int = 5,
        progress_callback=None
    ) -> List[List[AtomicFact]]:
        """
        Extract facts from multiple chunks in parallel using asyncio.

        Args:
            chunks: List of chunk dictionaries
            concurrency: Number of concurrent LLM calls
            progress_callback: Optional callback(completed_count) for progress tracking

        Returns:
            List of fact lists (one per chunk), in same order as input chunks
        """
        return asyncio.run(self._extract_batch_async(chunks, concurrency, progress_callback))

    async def _extract_batch_async(
        self,
        chunks: List[Dict[str, Any]],
        concurrency: int,
        progress_callback=None
    ) -> List[List[AtomicFact]]:
        """Async implementation of batch extraction with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency)

        async def extract_with_semaphore(chunk):
            async with semaphore:
                try:
                    result = await self._extract_facts_async(chunk)
                    if progress_callback:
                        progress_callback(1)  # Report completion of one chunk
                    return result
                except Exception as e:
                    logger.error(f"Error extracting facts from chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                    if progress_callback:
                        progress_callback(1)
                    return e  # Return exception to caller

        tasks = [extract_with_semaphore(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def _extract_facts_async(self, chunk: Dict[str, Any]) -> List[AtomicFact]:
        """
        Async fact extraction from a single chunk.

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

        logger.debug(f"Extracting semantic propositions from chunk {chunk_id}")

        # Single LLM call for all proposition types
        facts = await self._extract_propositions_llm_async(chunk_id, content)

        logger.debug(f"Extracted {len(facts)} propositions from chunk {chunk_id}")
        return facts

    def extract_facts(self, chunk: Dict[str, Any]) -> List[AtomicFact]:
        """
        Synchronous fact extraction (for single-chunk use cases).

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

        logger.debug(f"Extracting semantic propositions from chunk {chunk_id}")

        # Single LLM call for all proposition types
        facts = self._extract_propositions_llm_sync(chunk_id, content)

        logger.debug(f"Extracted {len(facts)} propositions from chunk {chunk_id}")
        return facts

    def _generate_fact_id(self, chunk_id: str) -> str:
        """Generate unique fact ID."""
        self.fact_counter += 1
        return f"{chunk_id}_fact_{self.fact_counter}"

    def _is_valid_entity_set(self, entities: List[str], fact_text: str) -> bool:
        """
        Validate entity set quality - rejects facts with only bare numbers.

        Args:
            entities: List of entity strings
            fact_text: The full fact text for context

        Returns:
            True if entities are semantically valid, False otherwise
        """
        if not entities:
            return False  # Must have at least one entity

        # Check if ALL entities are just bare numbers (indicates low quality)
        all_bare_numbers = True
        for entity in entities:
            entity_clean = entity.strip()
            if not entity_clean:
                continue
            # Check if it's JUST a number (optionally with currency/percent symbols)
            if not re.match(r'^[$€£¥]?\s*[\d,\.]+\s*[%kmbt]?$', entity_clean, re.IGNORECASE):
                all_bare_numbers = False
                break

        if all_bare_numbers:
            logger.debug(f"Skipping fact with bare number entities: {entities}")
            return False

        # Require minimum fact_text length (meaningful propositions need context)
        if len(fact_text.split()) < 8:
            logger.debug(f"Skipping fact with insufficient context: {fact_text[:50]}...")
            return False

        return True

    def _build_extraction_prompt(self, content: str) -> str:
        """
        Build the LLM prompt for semantic proposition extraction.

        Args:
            content: Chunk content to extract from

        Returns:
            Formatted prompt string
        """
        return f"""Extract self-contained semantic propositions (claims) from this business document. Each proposition should be a complete, meaningful statement that can stand alone and be questioned.

**Proposition Types:**

1. **factual_claim**: Complete semantic statements about entities, metrics, or events
   - "Elastic N.V. acquired Build Security Ltd. in Q1 2024"
   - "The company's Q1 2024 revenue was $400 million"
   - "EBITDA margin improved to 25% in Q1 2024"
   - "More than 50% of Fortune 500 companies use Elastic"

2. **temporal_event**: Events anchored to specific times
   - "The fiscal quarter ended on January 31, 2024"
   - "The Form 10-K was filed on March 15, 2024"
   - "The acquisition closed in Q1 2024"

3. **comparative_statement**: Comparisons, changes, or trends over time
   - "Deferred revenue increased 24% year-over-year"
   - "Employee count grew from 1,200 to 1,500 in FY2024"
   - "Revenue growth accelerated compared to prior quarter"

**CRITICAL RULES:**
- Extract ONLY facts explicitly stated in the text
- Each fact MUST be a complete, self-contained statement with full context
- Do NOT extract: HTML tags, CSS classes, JavaScript code, technical IDs, colspan/rowspan, isolated numbers without context
- Focus on business substance: financial metrics, corporate events, strategic initiatives, market position
- Entities MUST be semantic (company names, product names, metric names, time periods) - NOT bare numbers
- Extract 5-10 propositions depending on content richness
- If content is primarily HTML/CSS/code, return EMPTY array []

**Text to analyze:**
{content}

**Output format (JSON array):**
[
  {{
    "fact_type": "factual_claim",
    "fact_text": "Elastic N.V. acquired Build Security Ltd. in December 2021",
    "answer_span": "Build Security Ltd.",
    "entities": ["Elastic N.V.", "Build Security Ltd.", "December 2021", "acquisition"]
  }},
  {{
    "fact_type": "factual_claim",
    "fact_text": "Q1 2024 total revenue was $400 million, representing 20% year-over-year growth",
    "answer_span": "$400 million",
    "entities": ["Q1 2024", "total revenue", "$400 million", "20% growth"]
  }},
  {{
    "fact_type": "temporal_event",
    "fact_text": "The fiscal quarter ended on January 31, 2024",
    "answer_span": "January 31, 2024",
    "entities": ["fiscal quarter", "January 31, 2024"]
  }},
  {{
    "fact_type": "comparative_statement",
    "fact_text": "Deferred revenue increased 24% year-over-year to $536 million",
    "answer_span": "24%",
    "entities": ["deferred revenue", "24% increase", "year-over-year", "$536 million"]
  }}
]

**Required for each fact:**
- fact_type: "factual_claim", "temporal_event", or "comparative_statement"
- fact_text: Full semantic proposition with complete context (minimum 10 words)
- answer_span: The key piece of information that would answer a question (verbatim from text)
- entities: Semantic entities - company names, products, metrics, time periods (NOT bare numbers like "200" or "4.2")

Extract 5-10 semantic propositions. If text is mostly HTML/code, return []. Output ONLY valid JSON."""

    def _parse_and_validate_facts(
        self,
        response_text: str,
        chunk_id: str,
        content: str
    ) -> List[AtomicFact]:
        """
        Parse LLM JSON response and validate/filter facts.

        Args:
            response_text: Raw LLM response
            chunk_id: Chunk identifier
            content: Original chunk content

        Returns:
            List of validated AtomicFact objects
        """
        # Clean JSON markers if present
        if response_text.startswith("```"):
            response_text = re.sub(r'```json?\n?', '', response_text)
            response_text = re.sub(r'```\n?$', '', response_text)

        facts_data = json.loads(response_text)

        # Convert to AtomicFact objects with quality filtering
        facts = []
        seen_facts = set()  # Deduplication: (fact_text, answer_span)

        for fact_data in facts_data:
            fact_type = fact_data.get('fact_type', 'factual_claim')
            answer_span = fact_data.get('answer_span', '')
            fact_text = fact_data.get('fact_text', '')
            entities = fact_data.get('entities', [])

            # Skip if None or empty
            if not answer_span or not fact_text:
                continue

            # Convert answer_span to string if needed
            if isinstance(answer_span, (list, dict)):
                answer_span = json.dumps(answer_span)
            elif not isinstance(answer_span, str):
                answer_span = str(answer_span)

            # Quality filter: Check entity validity
            if not self._is_valid_entity_set(entities, fact_text):
                continue

            # Deduplication
            dedup_key = (fact_text.lower().strip(), answer_span.lower().strip())
            if dedup_key in seen_facts:
                continue
            seen_facts.add(dedup_key)

            # Find answer span location in content
            start, end = find_answer_span(answer_span, content)

            # Build metadata
            metadata = {
                "source": "llm",
                "proposition_type": fact_type
            }

            fact = AtomicFact(
                fact_id=self._generate_fact_id(chunk_id),
                chunk_id=chunk_id,
                fact_type=fact_type,
                fact_text=fact_text,
                answer_span=answer_span,
                answer_start=start,
                answer_end=end,
                entities=entities,
                metadata=metadata
            )
            facts.append(fact)

        return facts

    async def _extract_propositions_llm_async(
        self,
        chunk_id: str,
        content: str
    ) -> List[AtomicFact]:
        """
        Extract semantic propositions using LLM (async).

        Args:
            chunk_id: Chunk identifier
            content: Chunk content

        Returns:
            List of AtomicFact objects
        """
        # Limit content length for LLM
        max_content_len = 2000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        prompt = self._build_extraction_prompt(content)

        try:
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

            return self._parse_and_validate_facts(response_text, chunk_id, content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON for chunk {chunk_id}: {e}")
            logger.error(f"Response was: {response_text[:500] if 'response_text' in locals() else 'NO RESPONSE'}")
            return []
        except Exception as e:
            logger.error(f"Error extracting propositions from chunk {chunk_id}: {e}")
            return []

    def _extract_propositions_llm_sync(
        self,
        chunk_id: str,
        content: str
    ) -> List[AtomicFact]:
        """
        Extract semantic propositions using LLM (synchronous).

        Args:
            chunk_id: Chunk identifier
            content: Chunk content

        Returns:
            List of AtomicFact objects
        """
        # Limit content length for LLM
        max_content_len = 2000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        prompt = self._build_extraction_prompt(content)

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

            return self._parse_and_validate_facts(response_text, chunk_id, content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON for chunk {chunk_id}: {e}")
            logger.error(f"Response was: {response_text[:500] if 'response_text' in locals() else 'NO RESPONSE'}")
            return []
        except Exception as e:
            logger.error(f"Error extracting propositions from chunk {chunk_id}: {e}", exc_info=True)
            return []
