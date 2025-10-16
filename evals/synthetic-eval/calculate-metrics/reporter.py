"""
Reporter generates human-readable evaluation reports.

Optimized for efficient string formatting and minimal memory overhead.
Creates detailed text reports with tables and summaries.
"""

import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class Reporter:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, config):
        """
        Initialize reporter.
        
        Args:
            config: EvalConfig instance
        """
        self.config = config
    
    def generate_report(
        self,
        overall_metrics: Dict,
        by_type_metrics: Dict,
        by_k_metrics: Dict,
        per_query_metrics: List[Dict],
        failed_queries: List
    ):
        """
        Generate comprehensive text report.
        
        Args:
            overall_metrics: Aggregated metrics across all queries
            by_type_metrics: Metrics broken down by query type
            by_k_metrics: Metrics organized by K value
            per_query_metrics: List of per-query metrics
            failed_queries: List of failed query results
        """
        report_path = Path(self.config.detailed_report_file)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("SYNTHETIC EVALUATION - METRICS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Collection: {self.config.rag_system_params['collection_name']}\n")
            f.write(f"Embedding Model: {self.config.rag_system_params['embedding_model']}\n")
            f.write(f"Re-ranking: {'Enabled' if self.config.rag_system_params['enable_reranking'] else 'Disabled'}\n")
            f.write("\n")
            
            # Summary statistics
            f.write("=" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total queries evaluated: {overall_metrics.get('num_queries', 0)}\n")
            f.write(f"Failed queries: {len(failed_queries)}\n")
            f.write(f"Success rate: {(1 - len(failed_queries)/max(overall_metrics.get('num_queries', 1), 1))*100:.1f}%\n")
            f.write("\n")
            
            # Query type breakdown
            f.write("Query type breakdown:\n")
            for qtype, metrics in by_type_metrics.items():
                f.write(f"  - {qtype}: {metrics.get('num_queries', 0)} queries\n")
            f.write("\n")
            
            # Overall metrics
            f.write("=" * 80 + "\n")
            f.write("OVERALL METRICS (ALL QUERIES)\n")
            f.write("=" * 80 + "\n")
            f.write("\n")
            
            self._write_metrics_table(f, overall_metrics, self.config.k_values)
            
            # By query type
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("METRICS BY QUERY TYPE\n")
            f.write("=" * 80 + "\n")
            
            for qtype, metrics in by_type_metrics.items():
                f.write(f"\n{qtype.upper()} ({metrics.get('num_queries', 0)} queries)\n")
                f.write("-" * 80 + "\n")
                self._write_metrics_table(f, metrics, self.config.k_values)
            
            # Top and worst performing queries - Optimized: single sort
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("TOP PERFORMING QUERIES (by NDCG@10)\n")
            f.write("=" * 80 + "\n")

            # Sort once by NDCG@10
            sorted_queries = sorted(
                per_query_metrics,
                key=lambda x: x.get('ndcg@10', 0),
                reverse=True
            )

            # Top 10 queries
            for i, query in enumerate(sorted_queries[:10], start=1):
                f.write(f"\n{i}. Query: {query['query_id']}\n")
                f.write(f"   Type: {query.get('query_type', 'unknown')}\n")
                f.write(f"   NDCG@10: {query.get('ndcg@10', 0):.4f}\n")
                f.write(f"   Recall@10: {query.get('recall@10', 0):.4f}\n")
                f.write(f"   Precision@10: {query.get('precision@10', 0):.4f}\n")

            # Worst performing queries
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("WORST PERFORMING QUERIES (by NDCG@10)\n")
            f.write("=" * 80 + "\n")

            # Reuse sorted list - no need to re-sort or reverse
            worst_queries = sorted_queries[-10:] if len(sorted_queries) >= 10 else sorted_queries
            for i, query in enumerate(reversed(worst_queries), start=1):
                f.write(f"\n{i}. Query: {query['query_id']}\n")
                f.write(f"   Type: {query.get('query_type', 'unknown')}\n")
                f.write(f"   NDCG@10: {query.get('ndcg@10', 0):.4f}\n")
                f.write(f"   Recall@10: {query.get('recall@10', 0):.4f}\n")
                f.write(f"   Precision@10: {query.get('precision@10', 0):.4f}\n")
            
            # Failed queries
            if failed_queries:
                f.write("\n")
                f.write("=" * 80 + "\n")
                f.write(f"FAILED QUERIES ({len(failed_queries)})\n")
                f.write("=" * 80 + "\n")
                
                for i, failed in enumerate(failed_queries[:20], start=1):  # Show first 20
                    f.write(f"\n{i}. Query: {failed.query_id}\n")
                    f.write(f"   Error: {failed.error}\n")
                
                if len(failed_queries) > 20:
                    f.write(f"\n... and {len(failed_queries) - 20} more (see failed_queries.jsonl)\n")
            
            # Footer
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"✓ Generated detailed report: {report_path}")
    
    def _write_metrics_table(self, f, metrics: Dict, k_values: List[int]):
        """Write a formatted metrics table."""
        # Header
        f.write(f"{'Metric':<20} {'Value':>10}\n")
        f.write("-" * 32 + "\n")
        
        # MAP and MRR first
        f.write(f"{'MAP':<20} {metrics.get('MAP', 0.0):>10.4f}\n")
        f.write(f"{'MRR':<20} {metrics.get('MRR', 0.0):>10.4f}\n")
        f.write("\n")
        
        # Metrics by K
        for k in k_values:
            f.write(f"--- K = {k} ---\n")
            f.write(f"{'  Recall@' + str(k):<20} {metrics.get(f'recall@{k}', 0.0):>10.4f}\n")
            f.write(f"{'  Precision@' + str(k):<20} {metrics.get(f'precision@{k}', 0.0):>10.4f}\n")
            f.write(f"{'  NDCG@' + str(k):<20} {metrics.get(f'ndcg@{k}', 0.0):>10.4f}\n")
            f.write(f"{'  Hits@' + str(k):<20} {metrics.get(f'hits@{k}', 0.0):>10.4f}\n")
            f.write("\n")

