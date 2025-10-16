"""
Metrics calculation for information retrieval evaluation.

Supports graded relevance (0-3) for NDCG and binary relevance for other metrics.
"""

import math
import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class IRMetrics:
    """Information Retrieval metrics calculator with graded relevance support."""
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calculate Recall@K: relevant_retrieved@K / total_relevant
        
        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of relevant document IDs (any relevance > 0)
            k: Cut-off rank
            
        Returns:
            Recall@K score [0, 1]
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_retrieved = retrieved_at_k & relevant_ids
        
        return len(relevant_retrieved) / len(relevant_ids)
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calculate Precision@K: relevant_retrieved@K / K
        
        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of relevant document IDs (any relevance > 0)
            k: Cut-off rank
            
        Returns:
            Precision@K score [0, 1]
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_retrieved = retrieved_at_k & relevant_ids
        
        return len(relevant_retrieved) / k
    
    @staticmethod
    def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Calculate Average Precision (AP) for a single query.
        
        AP = (1/total_relevant) * Σ(Precision@k * rel(k))
        where rel(k) = 1 if doc at rank k is relevant, 0 otherwise
        
        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of relevant document IDs
            
        Returns:
            Average Precision score [0, 1]
        """
        if not relevant_ids:
            return 0.0
        
        precision_sum = 0.0
        relevant_count = 0
        
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                relevant_count += 1
                precision_at_rank = relevant_count / rank
                precision_sum += precision_at_rank
        
        return precision_sum / len(relevant_ids)
    
    @staticmethod
    def mean_average_precision(queries_results: List[Tuple[List[str], Set[str]]]) -> float:
        """
        Calculate Mean Average Precision (MAP) across multiple queries.
        
        Args:
            queries_results: List of (retrieved_ids, relevant_ids) tuples
            
        Returns:
            MAP score [0, 1]
        """
        if not queries_results:
            return 0.0
        
        ap_scores = [
            IRMetrics.average_precision(retrieved, relevant)
            for retrieved, relevant in queries_results
        ]
        
        return sum(ap_scores) / len(ap_scores)
    
    @staticmethod
    def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Calculate Reciprocal Rank (RR) - 1 / rank_of_first_relevant.
        
        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of relevant document IDs
            
        Returns:
            RR score [0, 1]
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(queries_results: List[Tuple[List[str], Set[str]]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) across multiple queries.
        
        Args:
            queries_results: List of (retrieved_ids, relevant_ids) tuples
            
        Returns:
            MRR score [0, 1]
        """
        if not queries_results:
            return 0.0
        
        rr_scores = [
            IRMetrics.reciprocal_rank(retrieved, relevant)
            for retrieved, relevant in queries_results
        ]
        
        return sum(rr_scores) / len(rr_scores)
    
    @staticmethod
    def hits_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """
        Calculate Hits@K - binary indicator if ANY relevant doc in top-K.
        
        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: Set of relevant document IDs
            k: Cut-off rank
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        retrieved_at_k = set(retrieved_ids[:k])
        return 1.0 if (retrieved_at_k & relevant_ids) else 0.0
    
    @staticmethod
    def dcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, int], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at K with graded relevance.
        
        DCG@K = Σ(i=1 to k) (rel_i / log2(i + 1))
        
        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevance_scores: Dict mapping doc_id -> relevance grade (0-3)
            k: Cut-off rank
            
        Returns:
            DCG@K score
        """
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            relevance = relevance_scores.get(doc_id, 0)
            # DCG formula: rel / log2(rank + 1)
            dcg += relevance / math.log2(rank + 1)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K with graded relevance.
        
        NDCG@K = DCG@K / IDCG@K
        where IDCG@K is the ideal DCG (perfect ranking)
        
        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevance_scores: Dict mapping doc_id -> relevance grade (0-3)
            k: Cut-off rank
            
        Returns:
            NDCG@K score [0, 1]
        """
        # Calculate DCG@K for actual ranking
        dcg = IRMetrics.dcg_at_k(retrieved_ids, relevance_scores, k)
        
        # Calculate IDCG@K (ideal ranking - sort by relevance desc)
        sorted_relevances = sorted(relevance_scores.values(), reverse=True)
        ideal_dcg = 0.0
        for rank, relevance in enumerate(sorted_relevances[:k], start=1):
            ideal_dcg += relevance / math.log2(rank + 1)
        
        # Avoid division by zero
        if ideal_dcg == 0.0:
            return 0.0
        
        return dcg / ideal_dcg
    
    @staticmethod
    def calculate_all_metrics(
        query_id: str,
        retrieved_ids: List[str],
        relevance_scores: Dict[str, int],
        k_values: List[int]
    ) -> Dict[str, any]:
        """
        Calculate all metrics for a single query across multiple K values.
        
        Args:
            query_id: Query identifier
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevance_scores: Dict mapping doc_id -> relevance grade (0-3)
            k_values: List of K values to evaluate
            
        Returns:
            Dict with all metrics
        """
        # Get binary relevant set (any relevance > 0)
        relevant_ids = {doc_id for doc_id, rel in relevance_scores.items() if rel > 0}
        
        metrics = {
            'query_id': query_id,
            'total_relevant': len(relevant_ids),
            'total_retrieved': len(retrieved_ids)
        }
        
        # Calculate metrics at each K
        for k in k_values:
            k_str = f"@{k}"
            
            metrics[f"recall{k_str}"] = IRMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"precision{k_str}"] = IRMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"ndcg{k_str}"] = IRMetrics.ndcg_at_k(retrieved_ids, relevance_scores, k)
            metrics[f"hits{k_str}"] = IRMetrics.hits_at_k(retrieved_ids, relevant_ids, k)
        
        # Calculate rank-independent metrics
        metrics['average_precision'] = IRMetrics.average_precision(retrieved_ids, relevant_ids)
        metrics['reciprocal_rank'] = IRMetrics.reciprocal_rank(retrieved_ids, relevant_ids)
        
        return metrics
    
    @staticmethod
    def aggregate_metrics(all_query_metrics: List[Dict], k_values: List[int]) -> Dict[str, float]:
        """
        Aggregate metrics across all queries.
        
        Args:
            all_query_metrics: List of per-query metrics
            k_values: List of K values
            
        Returns:
            Dict with aggregated metrics
        """
        if not all_query_metrics:
            return {}
        
        aggregated = {
            'num_queries': len(all_query_metrics)
        }
        
        # Aggregate each metric
        metric_keys = []
        for k in k_values:
            k_str = f"@{k}"
            metric_keys.extend([
                f"recall{k_str}",
                f"precision{k_str}",
                f"ndcg{k_str}",
                f"hits{k_str}"
            ])
        metric_keys.extend(['average_precision', 'reciprocal_rank'])
        
        for metric_key in metric_keys:
            values = [m[metric_key] for m in all_query_metrics if metric_key in m]
            if values:
                aggregated[metric_key] = sum(values) / len(values)
        
        # Add MAP and MRR explicitly
        aggregated['MAP'] = aggregated.get('average_precision', 0.0)
        aggregated['MRR'] = aggregated.get('reciprocal_rank', 0.0)
        
        return aggregated

