"""
Silver labeler for assigning neighbor-based relevance labels (0-2) to chunks.

This implementation uses a simple proximity-based approach:
- Score 2: Gold chunks (source of the query)
- Score 1: Immediate neighbors (±1 chunk_index in same document)
- Score 0: Everything else (not labeled, implicit)
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
vector_ingest_path = project_root / "vector-ingest" / "src"
sys.path.insert(0, str(vector_ingest_path))

from retrieval.retrieval import MilvusRetriever
from query_generator import Query

logger = logging.getLogger(__name__)


class SilverLabeler:
    """
    Assigns neighbor-based relevance labels (0-2) to chunks for queries.

    Relevance scale:
    - 2: Gold chunk (contains the fact/answer used to generate query)
    - 1: Neighbor chunk (±1 chunk_index in same document)
    - 0: Not relevant (implicit, not stored)
    """

    def __init__(self, config, retriever: MilvusRetriever):
        """
        Initialize silver labeler.

        Args:
            config: SyntheticEvalConfig instance
            retriever: MilvusRetriever for accessing Milvus collection
        """
        self.config = config
        self.retriever = retriever
        self.collection = retriever.vector_store.collection

        logger.info("Initialized SilverLabeler with neighbor-based approach (0-2 scale)")

    def label_query(self, query: Query) -> Dict[str, int]:
        """
        Assign relevance labels to chunks for a single query.

        Args:
            query: Query object with gold_chunk_ids

        Returns:
            Dictionary mapping chunk_id -> relevance_label (1-2)
            Note: Only returns labeled chunks (score > 0)
        """
        labels = {}

        for gold_chunk_id in query.gold_chunk_ids:
            # Get gold chunk metadata from Milvus
            try:
                gold_results = self.collection.query(
                    expr=f'chunk_id == "{gold_chunk_id}"',
                    output_fields=['chunk_id', 'doc_id', 'chunk_index'],
                    limit=1
                )

                if not gold_results:
                    logger.warning(f"Gold chunk {gold_chunk_id} not found in Milvus for query {query.query_id}")
                    continue

                gold_info = gold_results[0]
                doc_id = gold_info['doc_id']
                chunk_idx = gold_info['chunk_index']

                # Score 2: Gold chunk itself
                labels[gold_chunk_id] = 2

                # Score 1: ±1 neighbors (same doc, adjacent chunk_index)
                neighbor_results = self.collection.query(
                    expr=f'doc_id == "{doc_id}" && chunk_index >= {chunk_idx - self.config.neighbor_window} && chunk_index <= {chunk_idx + self.config.neighbor_window}',
                    output_fields=['chunk_id', 'chunk_index'],
                    limit=10  # Max neighbors expected
                )

                for neighbor in neighbor_results:
                    neighbor_id = neighbor['chunk_id']
                    if neighbor_id != gold_chunk_id:  # Don't override gold chunk
                        if neighbor_id not in labels:  # Don't override if already labeled from another gold chunk
                            labels[neighbor_id] = 1

            except Exception as e:
                logger.error(f"Error labeling gold chunk {gold_chunk_id} for query {query.query_id}: {e}")
                continue

        return labels

    def batch_label_queries(self, queries: List[Query]) -> Dict[str, Dict[str, int]]:
        """
        Label chunks for multiple queries.

        Args:
            queries: List of Query objects

        Returns:
            Dictionary mapping query_id -> {chunk_id: relevance_label}
        """
        logger.info(f"Labeling {len(queries)} queries using neighbor-based approach...")

        all_qrels = {}

        pbar = tqdm(queries, desc="Labeling queries", unit="query", ncols=100)
        for query in pbar:
            qrels = self.label_query(query)
            all_qrels[query.query_id] = qrels
            pbar.set_postfix({"labels": len(qrels)})
        pbar.close()

        logger.info(f"Completed labeling for {len(queries)} queries")

        return all_qrels

    def compute_label_statistics(self, all_qrels: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
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
            label: count / total_labels * 100 if total_labels > 0 else 0
            for label, count in label_counts.items()
        }

        # Per-query statistics
        labels_per_query = {}
        for query_id, qrels in all_qrels.items():
            labels_per_query[query_id] = len(qrels)

        avg_labels = sum(labels_per_query.values()) / len(labels_per_query) if labels_per_query else 0

        # Count by relevance level
        score_2_count = sum(1 for labels in all_qrels.values() for score in labels.values() if score == 2)
        score_1_count = sum(1 for labels in all_qrels.values() for score in labels.values() if score == 1)

        stats = {
            'total_labels': total_labels,
            'total_queries': len(all_qrels),
            'label_distribution': dict(label_counts),
            'label_percentages': label_percentages,
            'avg_labels_per_query': avg_labels,
            'score_2_count': score_2_count,
            'score_1_count': score_1_count,
            'queries_with_no_labels': sum(1 for c in labels_per_query.values() if c == 0)
        }

        logger.info(f"Label distribution: {dict(label_counts)}")
        logger.info(f"Average labels per query: {avg_labels:.1f}")
        logger.info(f"  Score 2 (gold): {score_2_count}")
        logger.info(f"  Score 1 (neighbors): {score_1_count}")

        return stats
