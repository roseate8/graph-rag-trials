"""
Query generator for single-hop and multi-hop questions.
"""

import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
vector_ingest_path = project_root / "vector-ingest" / "src"
sys.path.insert(0, str(vector_ingest_path))

from chunking.processors.llm_utils import SecureAPIKeyManager
from fact_extractor import AtomicFact
from utils import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """Represents a generated query."""
    query_id: str
    query_text: str
    answer: str
    gold_chunk_ids: List[str]  # 1 for single-hop, 2+ for multi-hop
    query_type: str  # single_hop, multi_hop
    question_style: str  # wh_question, cloze, keyword
    metadata: Dict[str, Any]  # Additional info (facts used, reasoning, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class QueryGenerator:
    """
    Generates diverse queries from atomic facts.
    
    Supports:
    - Single-hop: 3-5 paraphrased questions per fact
    - Multi-hop: Questions requiring multiple facts
    """
    
    def __init__(self, config, llm_manager: SecureAPIKeyManager):
        """
        Initialize query generator.
        
        Args:
            config: SyntheticEvalConfig instance
            llm_manager: SecureAPIKeyManager for LLM calls
        """
        self.config = config
        self.llm_manager = llm_manager
        self.query_counter = 0
        
        logger.info(f"Initialized QueryGenerator with model: {config.model_name}")
    
    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        self.query_counter += 1
        return f"q{self.query_counter:04d}"
    
    def generate_single_hop(self, fact: AtomicFact) -> List[Query]:
        """
        Generate 3-5 diverse single-hop questions for a fact.
        
        Args:
            fact: AtomicFact to generate questions for
            
        Returns:
            List of Query objects
        """
        prompt = f"""Generate {self.config.queries_per_fact_min}-{self.config.queries_per_fact_max} diverse questions for this fact. 
Do NOT reuse the exact phrasing from the text.

Fact: {fact.fact_text}
Answer: {fact.answer_span}
Fact Type: {fact.fact_type}

Generate different question styles:
1. A "who/what/when/where/why/how" question
2. A cloze-style question (fill in the blank: "The X was ___")
3. A keyword-based query (simpler, more search-like)
4. Additional paraphrased variations

Requirements:
- Each question must be answerable with: {fact.answer_span}
- Use different phrasing than the original fact
- Make questions natural and diverse

Output JSON format (array of questions):
[
  {{
    "question": "What was the revenue in fiscal year 2024?",
    "style": "wh_question"
  }},
  {{
    "question": "The fiscal year 2024 revenue was ___",
    "style": "cloze"
  }},
  {{
    "question": "fiscal year 2024 revenue amount",
    "style": "keyword"
  }}
]

Output ONLY valid JSON, no other text."""
        
        try:
            api_key = self.llm_manager.get_api_key()
            
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            llm_params = self.config.get_llm_params({
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": "You are a question generation assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            })
            response = client.chat.completions.create(**llm_params)
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            if response_text.startswith("```"):
                response_text = re.sub(r'```json?\n?', '', response_text)
                response_text = re.sub(r'```\n?$', '', response_text)
            
            questions_data = json.loads(response_text)
            
            # Convert to Query objects
            queries = []
            for q_data in questions_data:
                query = Query(
                    query_id=self._generate_query_id(),
                    query_text=q_data.get('question', ''),
                    answer=fact.answer_span,
                    gold_chunk_ids=[fact.chunk_id],
                    query_type="single_hop",
                    question_style=q_data.get('style', 'unknown'),
                    metadata={
                        "fact_id": fact.fact_id,
                        "fact_type": fact.fact_type,
                        "entities": fact.entities
                    }
                )
                queries.append(query)
            
            logger.debug(f"Generated {len(queries)} single-hop queries for fact {fact.fact_id}")
            return queries
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response_text[:500] if 'response_text' in locals() else 'NO RESPONSE'}")
            return []
        except Exception as e:
            logger.error(f"Error generating single-hop queries: {e}")
            return []
    
    def generate_multi_hop(self, fact_pairs: List[Tuple[AtomicFact, AtomicFact]]) -> List[Query]:
        """
        Generate multi-hop questions from fact pairs.
        
        Args:
            fact_pairs: List of (fact1, fact2) tuples that share entities
            
        Returns:
            List of Query objects
        """
        queries = []
        
        for fact1, fact2 in fact_pairs:
            # Find shared entity
            shared_entities = set(fact1.entities) & set(fact2.entities)
            if not shared_entities:
                continue
            
            shared_entity = list(shared_entities)[0]
            
            prompt = f"""Create a question that requires BOTH facts to answer. The question should compare, combine, or reason across both facts.

Fact 1 (from chunk {fact1.chunk_id}): {fact1.fact_text}
Answer 1: {fact1.answer_span}

Fact 2 (from chunk {fact2.chunk_id}): {fact2.fact_text}
Answer 2: {fact2.answer_span}

Shared entity: {shared_entity}

Generate a multi-hop question that:
- Requires information from BOTH facts
- Could involve comparison, calculation, or reasoning
- Has a clear answer

Examples:
- "What changed in X between year1 and year2?"
- "What is the difference between X and Y?"
- "How did Z evolve from doc1 to doc2?"

Output JSON format:
{{
  "question": "Your multi-hop question here",
  "answer": "Combined answer from both facts",
  "reasoning": "Why this requires both chunks"
}}

Output ONLY valid JSON, no other text."""
            
            try:
                api_key = self.llm_manager.get_api_key()
                
                import openai
                client = openai.OpenAI(api_key=api_key)
                
                llm_params = self.config.get_llm_params({
                    "model": self.config.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a multi-hop question generation assistant. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ]
                })
                response = client.chat.completions.create(**llm_params)
                
                response_text = response.choices[0].message.content.strip()
                
                # Parse JSON
                if response_text.startswith("```"):
                    response_text = re.sub(r'```json?\n?', '', response_text)
                    response_text = re.sub(r'```\n?$', '', response_text)
                
                q_data = json.loads(response_text)
                
                query = Query(
                    query_id=self._generate_query_id(),
                    query_text=q_data.get('question', ''),
                    answer=q_data.get('answer', f"{fact1.answer_span}; {fact2.answer_span}"),
                    gold_chunk_ids=[fact1.chunk_id, fact2.chunk_id],
                    query_type="multi_hop",
                    question_style="comparison",
                    metadata={
                        "fact_ids": [fact1.fact_id, fact2.fact_id],
                        "shared_entity": shared_entity,
                        "reasoning": q_data.get('reasoning', ''),
                        "fact1_type": fact1.fact_type,
                        "fact2_type": fact2.fact_type
                    }
                )
                queries.append(query)
                
                logger.debug(f"Generated multi-hop query: {query.query_id}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                continue
            except Exception as e:
                logger.error(f"Error generating multi-hop query: {e}")
                continue
        
        logger.info(f"Generated {len(queries)} multi-hop queries from {len(fact_pairs)} fact pairs")
        return queries
    
    def find_linkable_facts(self, all_facts: List[AtomicFact]) -> List[Tuple[AtomicFact, AtomicFact]]:
        """
        Find pairs of facts that share entities across different chunks.
        
        Args:
            all_facts: List of all extracted facts
            
        Returns:
            List of (fact1, fact2) tuples that can be linked
        """
        logger.info(f"Finding linkable facts from {len(all_facts)} total facts...")
        
        # Group facts by entities
        entity_to_facts = defaultdict(list)
        
        for fact in all_facts:
            for entity in fact.entities:
                # Normalize entity for matching
                normalized_entity = normalize_text(entity)
                if normalized_entity:
                    entity_to_facts[normalized_entity].append(fact)
        
        # Find fact pairs with shared entities across different chunks
        fact_pairs = []
        processed_pairs = set()
        
        for entity, facts in entity_to_facts.items():
            if len(facts) < 2:
                continue
            
            # Find facts from different chunks
            for i, fact1 in enumerate(facts):
                for fact2 in facts[i+1:]:
                    # Must be from different chunks
                    if fact1.chunk_id == fact2.chunk_id:
                        continue
                    
                    # Avoid duplicate pairs
                    pair_key = tuple(sorted([fact1.fact_id, fact2.fact_id]))
                    if pair_key in processed_pairs:
                        continue
                    
                    processed_pairs.add(pair_key)
                    fact_pairs.append((fact1, fact2))
        
        logger.info(f"Found {len(fact_pairs)} linkable fact pairs")
        
        # Limit to reasonable number based on multi_hop_ratio
        target_multi_hop = int(self.config.target_questions * self.config.multi_hop_ratio)
        if len(fact_pairs) > target_multi_hop:
            import random
            random.seed(42)
            fact_pairs = random.sample(fact_pairs, target_multi_hop)
            logger.info(f"Sampled {len(fact_pairs)} fact pairs for multi-hop queries")
        
        return fact_pairs
    
    def generate_all_queries(self, all_facts: List[AtomicFact]) -> Tuple[List[Query], Dict[str, Any]]:
        """
        Generate both single-hop and multi-hop queries from all facts.
        
        Args:
            all_facts: List of all extracted facts
            
        Returns:
            Tuple of (all_queries, generation_stats)
        """
        logger.info(f"Generating queries from {len(all_facts)} facts...")
        
        all_queries = []

        # 1. Generate single-hop queries
        logger.info("Generating single-hop queries...")
        single_hop_count = 0

        for fact in tqdm(all_facts, desc="Generating single-hop queries", unit="fact"):
            queries = self.generate_single_hop(fact)
            all_queries.extend(queries)
            single_hop_count += len(queries)

            # Stop if we have enough queries
            if len(all_queries) >= self.config.target_questions:
                logger.info(f"Reached target of {self.config.target_questions} queries")
                break
        
        # 2. Generate multi-hop queries if we haven't reached target
        multi_hop_count = 0
        if len(all_queries) < self.config.target_questions:
            logger.info("Generating multi-hop queries...")
            
            fact_pairs = self.find_linkable_facts(all_facts)
            multi_hop_queries = self.generate_multi_hop(fact_pairs)
            
            all_queries.extend(multi_hop_queries)
            multi_hop_count = len(multi_hop_queries)
        
        # Trim to target size
        if len(all_queries) > self.config.target_questions:
            all_queries = all_queries[:self.config.target_questions]
        
        # Generate stats
        stats = {
            'total_queries': len(all_queries),
            'single_hop': single_hop_count,
            'multi_hop': multi_hop_count,
            'facts_used': len(all_facts),
            'query_styles': self._count_styles(all_queries)
        }
        
        logger.info(f"Generated {len(all_queries)} total queries ({single_hop_count} single-hop, {multi_hop_count} multi-hop)")
        
        return all_queries, stats
    
    def _count_styles(self, queries: List[Query]) -> Dict[str, int]:
        """Count query styles."""
        style_counts = defaultdict(int)
        for query in queries:
            style_counts[query.question_style] += 1
        return dict(style_counts)

