"""
Silver labeler for assigning neighbor-based relevance labels (0-2) to chunks.

Optimized implementation with batch Milvus queries for high performance:
- Score 2: Gold chunks (source of the query)
- Score 1: Immediate neighbors (±N chunk_index in same document)
- Score 0: Everything else (not labeled, implicit)

Performance: O(Q) Milvus queries where Q = number of queries (not Q*G where G = gold chunks)
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
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

    Optimized for batch processing with minimal Milvus queries.

    Relevance scale:
    - 2: Gold chunk (contains the fact/answer used to generate query)
    - 1: Neighbor chunk (±N chunk_index in same document)
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
        self.collection = retriever.milvus_store.collection
        self.neighbor_window = config.neighbor_window

        logger.info(f"Initialized SilverLabeler (neighbor_window=±{self.neighbor_window})")

    def _build_gold_chunks_expr(self, chunk_ids: Set[str]) -> str:
        """Build Milvus expr for batch querying gold chunks."""
        if not chunk_ids:
            return ""

        # Escape quotes and build IN expression
        escaped_ids = [cid.replace('"', '\\"') for cid in chunk_ids]
        expr = 'chunk_id in [' + ', '.join(f'"{cid}"' for cid in escaped_ids) + ']'
        return expr

    def _batch_fetch_gold_metadata(self, gold_chunk_ids: Set[str]) -> Dict[str, Tuple[str, int]]:
        """
        Batch fetch metadata for all gold chunks in one query.

        Returns:
            Dict mapping chunk_id -> (doc_id, chunk_index)
        """
        if not gold_chunk_ids:
            return {}

        expr = self._build_gold_chunks_expr(gold_chunk_ids)

        try:
            results = self.collection.query(
                expr=expr,
                output_fields=['chunk_id', 'doc_id', 'chunk_index'],
                limit=len(gold_chunk_ids)
            )

            metadata = {
                r['chunk_id']: (r['doc_id'], r['chunk_index'])
                for r in results
            }

            # Log missing chunks
            missing = gold_chunk_ids - set(metadata.keys())
            if missing:
                logger.warning(f"Missing {len(missing)} gold chunks in Milvus: {list(missing)[:3]}...")

            return metadata

        except Exception as e:
            logger.error(f"Error batch fetching gold metadata: {e}")
            return {}

    def _build_neighbor_expr(self, doc_chunks: Dict[str, List[int]]) -> str:
        """
        Build optimized Milvus expr for batch neighbor queries.

        Args:
            doc_chunks: Dict mapping doc_id -> [chunk_indices]

        Returns:
            Milvus filter expression
        """
        if not doc_chunks:
            return ""

        # Build (doc_id == X && chunk_index in [Y, Z]) OR ...
        conditions = []
        for doc_id, indices in doc_chunks.items():
            escaped_doc_id = doc_id.replace('"', '\\"')

            # Expand each index to include neighbors
            neighbor_indices = set()
            for idx in indices:
                for offset in range(-self.neighbor_window, self.neighbor_window + 1):
                    neighbor_indices.add(idx + offset)

            indices_str = ', '.join(str(i) for i in sorted(neighbor_indices))
            conditions.append(f'(doc_id == "{escaped_doc_id}" && chunk_index in [{indices_str}])')

        return ' || '.join(conditions)

    def label_query(self, query: Query, gold_metadata: Dict[str, Tuple[str, int]]) -> Dict[str, int]:
        """
        Assign relevance labels to chunks for a single query using pre-fetched metadata.

        Args:
            query: Query object with gold_chunk_ids
            gold_metadata: Pre-fetched metadata mapping chunk_id -> (doc_id, chunk_index)

        Returns:
            Dictionary mapping chunk_id -> relevance_label (1-2)
        """
        labels = {}

        # Group gold chunks by document for efficient neighbor lookup
        doc_chunks = {}  # doc_id -> [chunk_indices]

        for gold_chunk_id in query.gold_chunk_ids:
            # Score 2: Gold chunk itself
            labels[gold_chunk_id] = 2

            # Get metadata
            if gold_chunk_id not in gold_metadata:
                continue

            doc_id, chunk_idx = gold_metadata[gold_chunk_id]

            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(chunk_idx)

        # Batch query for all neighbors across all docs
        if doc_chunks:
            try:
                neighbor_expr = self._build_neighbor_expr(doc_chunks)

                neighbor_results = self.collection.query(
                    expr=neighbor_expr,
                    output_fields=['chunk_id'],
                    limit=len(doc_chunks) * (2 * self.neighbor_window + 1) * len(query.gold_chunk_ids)
                )

                # Score 1: Neighbors (but don't override gold chunks)
                for neighbor in neighbor_results:
                    neighbor_id = neighbor['chunk_id']
                    if neighbor_id not in labels:
                        labels[neighbor_id] = 1

            except Exception as e:
                logger.error(f"Error labeling neighbors for query {query.query_id}: {e}")

        return labels

    def batch_label_queries(self, queries: List[Query]) -> Dict[str, Dict[str, int]]:
        """
        Label chunks for multiple queries with optimized batch processing.

        Args:
            queries: List of Query objects

        Returns:
            Dictionary mapping query_id -> {chunk_id: relevance_label}
        """
        logger.info(f"Labeling {len(queries)} queries...")

        # Step 1: Collect all unique gold chunk IDs across all queries
        all_gold_ids = set()
        for query in queries:
            all_gold_ids.update(query.gold_chunk_ids)

        logger.info(f"  Fetching metadata for {len(all_gold_ids)} unique gold chunks...")

        # Step 2: Batch fetch all gold metadata in ONE query
        gold_metadata = self._batch_fetch_gold_metadata(all_gold_ids)

        logger.info(f"  Retrieved metadata for {len(gold_metadata)} chunks")

        # Step 3: Label each query using pre-fetched metadata
        all_qrels = {}

        pbar = tqdm(queries, desc="Labeling", unit="q", ncols=80, leave=False)
        for query in pbar:
            qrels = self.label_query(query, gold_metadata)
            all_qrels[query.query_id] = qrels
            pbar.set_postfix({"labels": len(qrels)}, refresh=False)
        pbar.close()

        logger.info(f"Labeling complete: {len(all_qrels)} queries processed")

        return all_qrels

    def compute_label_statistics(self, all_qrels: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Compute statistics about label distribution (optimized).

        Args:
            all_qrels: Dictionary of query_id -> {chunk_id: relevance}

        Returns:
            Statistics dictionary
        """
        if not all_qrels:
            return {
                'total_labels': 0,
                'total_queries': 0,
                'label_distribution': {},
                'label_percentages': {},
                'avg_labels_per_query': 0,
                'score_2_count': 0,
                'score_1_count': 0,
                'queries_with_no_labels': 0
            }

        # Single-pass statistics computation
        label_counts = Counter()
        labels_per_query_counts = []
        queries_with_no_labels = 0

        for qrels in all_qrels.values():
            qrels_len = len(qrels)
            labels_per_query_counts.append(qrels_len)

            if qrels_len == 0:
                queries_with_no_labels += 1

            label_counts.update(qrels.values())

        total_labels = sum(label_counts.values())
        total_queries = len(all_qrels)

        # Compute percentages
        label_percentages = {
            label: (count / total_labels * 100) if total_labels > 0 else 0
            for label, count in label_counts.items()
        }

        # Average labels per query
        avg_labels = sum(labels_per_query_counts) / total_queries if total_queries > 0 else 0

        stats = {
            'total_labels': total_labels,
            'total_queries': total_queries,
            'label_distribution': dict(label_counts),
            'label_percentages': label_percentages,
            'avg_labels_per_query': round(avg_labels, 2),
            'score_2_count': label_counts.get(2, 0),
            'score_1_count': label_counts.get(1, 0),
            'queries_with_no_labels': queries_with_no_labels
        }

        logger.info(f"Statistics: {total_labels} labels across {total_queries} queries (avg={avg_labels:.1f})")
        logger.info(f"  Gold chunks (2): {label_counts.get(2, 0)}, Neighbors (1): {label_counts.get(1, 0)}")

        return stats
