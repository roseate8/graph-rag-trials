"""
Silver labeler for assigning graded relevance (0-3) to chunks.
"""

import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
vector_ingest_path = project_root / "vector-ingest" / "src"
sys.path.insert(0, str(vector_ingest_path))

from chunking.processors.llm_utils import SecureAPIKeyManager
from retrieval.retrieval import MilvusRetriever
from query_generator import Query
from utils import (
    compute_token_f1, compute_token_f1_sentences,
    has_exact_match, normalize_text
)

logger = logging.getLogger(__name__)


class SilverLabeler:
    """
    Assigns graded relevance labels (0-3) to chunks for queries.
    
    Relevance scale:
    - 3: Contains exact answer or paraphrased answer
    - 2: Provides supporting context but not direct answer
    - 1: Same topic but minimal relevance
    - 0: Not relevant
    """
    
    def __init__(
        self,
        config,
        llm_manager: SecureAPIKeyManager,
        retriever: MilvusRetriever
    ):
        """
        Initialize silver labeler.
        
        Args:
            config: SyntheticEvalConfig instance
            llm_manager: SecureAPIKeyManager for LLM calls
            retriever: MilvusRetriever for semantic similarity
        """
        self.config = config
        self.llm_manager = llm_manager
        self.retriever = retriever
        
        # Cache for document groupings
        self.doc_id_cache = {}
        
        logger.info("Initialized SilverLabeler with graded relevance (0-3)")
    
    def label_all_chunks(
        self,
        query: Query,
        all_chunks: List[Dict[str, Any]],
        chunk_to_doc: Dict[str, str]
    ) -> Dict[str, int]:
        """
        Assign relevance labels to all chunks for a query.

        Args:
            query: Query object
            all_chunks: List of all chunks
            chunk_to_doc: Pre-built mapping of chunk_id -> doc_id

        Returns:
            Dictionary mapping chunk_id -> relevance_label (0-3)
        """
        logger.debug(f"Labeling chunks for query {query.query_id}")

        labels = {}

        for chunk in all_chunks:
            chunk_id = chunk.get('chunk_id', '')

            # Gold chunks always get rel=3
            if chunk_id in query.gold_chunk_ids:
                labels[chunk_id] = 3
                continue

            # Compute relevance for non-gold chunks
            relevance = self._compute_relevance(query, chunk, chunk_to_doc)
            labels[chunk_id] = relevance

        # Log label distribution
        label_counts = defaultdict(int)
        for label in labels.values():
            label_counts[label] += 1

        logger.debug(f"Query {query.query_id} label distribution: {dict(label_counts)}")

        return labels
    
    def _compute_relevance(
        self,
        query: Query,
        chunk: Dict[str, Any],
        chunk_to_doc: Dict[str, str]
    ) -> int:
        """
        Compute relevance score for a chunk.
        
        Args:
            query: Query object
            chunk: Chunk dictionary
            chunk_to_doc: Mapping of chunk_id -> doc_id
            
        Returns:
            Relevance score (0-3)
        """
        content = chunk.get('content', '')
        chunk_id = chunk.get('chunk_id', '')
        
        if not content:
            return 0
        
        # 1. Check for exact match (rel=3)
        if has_exact_match(query.answer, content, self.config.exact_match_threshold):
            logger.debug(f"Exact match found in chunk {chunk_id}")
            return 3
        
        # 2. Compute token-F1 with answer
        f1_score = compute_token_f1(query.answer, content)
        
        # High F1 indicates answer present (rel=3)
        if f1_score >= self.config.token_f1_high:
            logger.debug(f"High token-F1 ({f1_score:.2f}) in chunk {chunk_id}")
            return 3
        
        # Mid F1 indicates supporting context (rel=2)
        if f1_score >= self.config.token_f1_mid:
            logger.debug(f"Mid token-F1 ({f1_score:.2f}) in chunk {chunk_id}")
            return 2
        
        # 3. Check if same document as gold chunk
        gold_doc_ids = set()
        for gold_chunk_id in query.gold_chunk_ids:
            doc_id = chunk_to_doc.get(gold_chunk_id, '')
            if doc_id:
                gold_doc_ids.add(doc_id)
        
        chunk_doc_id = chunk_to_doc.get(chunk_id, '')
        
        if chunk_doc_id and chunk_doc_id in gold_doc_ids:
            # Same document - check semantic similarity
            semantic_sim = self._compute_semantic_similarity(query.query_text, content)
            
            if semantic_sim >= self.config.semantic_similarity_threshold:
                logger.debug(f"High semantic similarity ({semantic_sim:.2f}) in same doc for chunk {chunk_id}")
                return 2
            else:
                logger.debug(f"Low semantic similarity ({semantic_sim:.2f}) in same doc for chunk {chunk_id}")
                return 1
        
        # 4. Use LLM judge for ambiguous cases
        if self.config.enable_llm_judge and self.config.token_f1_low <= f1_score < self.config.token_f1_mid:
            logger.debug(f"Using LLM judge for ambiguous case (F1={f1_score:.2f}) in chunk {chunk_id}")
            return self._llm_judge(query, chunk)
        
        # 5. Default to not relevant
        return 0
    
    def _compute_semantic_similarity(self, query_text: str, chunk_content: str) -> float:
        """
        Compute semantic similarity between query and chunk using embeddings.
        
        Args:
            query_text: Query text
            chunk_content: Chunk content
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        try:
            # Get embeddings
            query_embedding = self.retriever._get_query_embedding(query_text)
            chunk_embedding = self.retriever._get_query_embedding(chunk_content[:500])  # Limit length
            
            # Compute cosine similarity
            import numpy as np
            
            query_vec = np.array(query_embedding)
            chunk_vec = np.array(chunk_embedding)
            
            # Normalize
            query_norm = query_vec / np.linalg.norm(query_vec)
            chunk_norm = chunk_vec / np.linalg.norm(chunk_vec)
            
            # Cosine similarity
            similarity = float(np.dot(query_norm, chunk_norm))
            
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def _llm_judge(self, query: Query, chunk: Dict[str, Any]) -> int:
        """
        Use LLM to judge relevance for ambiguous cases.
        
        Args:
            query: Query object
            chunk: Chunk dictionary
            
        Returns:
            Relevance score (0-3)
        """
        content = chunk.get('content', '')
        
        # Limit content length
        max_content_len = 1000
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        
        prompt = f"""Grade the relevance of this chunk to the query on a 0-3 scale.

Query: {query.query_text}
Expected answer: {query.answer}

Chunk: {content}

Rubric:
- 3: Contains exact answer or clearly paraphrased answer
- 2: Provides supporting context that helps answer the query, but not the direct answer
- 1: Same topic/domain but minimal relevance to the specific query
- 0: Not relevant to the query

Think carefully about whether the chunk contains the answer or just related information.

Output ONLY a single number (0, 1, 2, or 3). No explanation."""
        
        try:
            api_key = self.llm_manager.get_api_key()
            
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            llm_params = self.config.get_llm_params({
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": "You are a relevance grading assistant. Output only a single digit."},
                    {"role": "user", "content": prompt}
                ]
            })
            # Override token limit for this specific call
            if "max_completion_tokens" in llm_params:
                llm_params["max_completion_tokens"] = 10
            else:
                llm_params["max_tokens"] = 10
            response = client.chat.completions.create(**llm_params)
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract number from response
            match = re.search(r'\b([0-3])\b', response_text)
            if match:
                score = int(match.group(1))
                logger.debug(f"LLM judge score: {score}")
                return score
            else:
                logger.warning(f"Could not parse LLM judge response: {response_text}")
                return 1  # Default to minimal relevance
            
        except Exception as e:
            logger.error(f"Error in LLM judge: {e}")
            return 1  # Default to minimal relevance
    
    def batch_label_queries(
        self,
        queries: List[Query],
        all_chunks: List[Dict[str, Any]],
        batch_size: int = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Label all chunks for multiple queries in batches.

        Args:
            queries: List of Query objects
            all_chunks: List of all chunks
            batch_size: Batch size (default from config)

        Returns:
            Dictionary mapping query_id -> {chunk_id: relevance_label}
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        logger.info(f"Batch labeling {len(queries)} queries...")

        # Build chunk_to_doc mapping ONCE for all queries (optimization)
        logger.info(f"Building chunk-to-doc mapping for {len(all_chunks)} chunks...")
        chunk_to_doc = {chunk['chunk_id']: chunk.get('doc_id', '') for chunk in all_chunks}

        all_qrels = {}

        pbar = tqdm(queries, desc="Labeling queries", unit="query", ncols=100)
        for query in pbar:
            qrels = self.label_all_chunks(query, all_chunks, chunk_to_doc)
            all_qrels[query.query_id] = qrels
        pbar.close()

        logger.info(f"Completed labeling for {len(queries)} queries")

        return all_qrels
    
    def compute_label_statistics(
        self,
        all_qrels: Dict[str, Dict[str, int]]
    ) -> Dict[str, Any]:
        """
        Compute statistics about label distribution.
        
        Args:
            all_qrels: Dictionary of query_id -> {chunk_id: relevance}
            
        Returns:
            Statistics dictionary
        """
        logger.info("Computing label statistics...")
        
        # Overall label distribution
        all_labels = []
        for qrels in all_qrels.values():
            all_labels.extend(qrels.values())
        
        label_counts = defaultdict(int)
        for label in all_labels:
            label_counts[label] += 1
        
        total_labels = len(all_labels)
        label_percentages = {
            label: count / total_labels * 100
            for label, count in label_counts.items()
        }
        
        # Per-query statistics
        avg_relevant_per_query = {}
        for query_id, qrels in all_qrels.items():
            relevant_count = sum(1 for label in qrels.values() if label > 0)
            avg_relevant_per_query[query_id] = relevant_count
        
        avg_relevant = sum(avg_relevant_per_query.values()) / len(avg_relevant_per_query)
        
        stats = {
            'total_labels': total_labels,
            'total_queries': len(all_qrels),
            'label_distribution': dict(label_counts),
            'label_percentages': label_percentages,
            'avg_relevant_per_query': avg_relevant,
            'queries_with_no_relevant': sum(1 for c in avg_relevant_per_query.values() if c == 0)
        }
        
        logger.info(f"Label distribution: {dict(label_counts)}")
        logger.info(f"Average relevant chunks per query: {avg_relevant:.1f}")
        
        return stats

