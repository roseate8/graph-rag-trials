"""
Configuration for synthetic evaluation metrics calculation.
Optimized with efficient defaults and validation.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics calculation. Optimized with dataclass field defaults."""

    # Retrieval settings (must match your existing system)
    collection_name: str = "elastic_embeddings_m3"
    embedding_model: str = "BAAI/bge-m3"
    milvus_profile: str = "production"
    enable_reranking: bool = True

    # Evaluation K values - Use field() for mutable defaults
    # IMPORTANT: Max K limited to 50 to avoid Milvus ef parameter issues (ef=128 in milvus_config)
    # With re-ranking multiplier of 10, max retrieval is 50*10=500 < ef(128) limit
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 50])

    # Async processing settings
    batch_size: int = 15  # Number of concurrent retrieval requests
    max_concurrent: int = 15  # Maximum concurrent async operations

    # Input/output paths (relative to calculate-metrics/)
    queries_file: str = "../output/queries.jsonl"
    qrels_file: str = "../output/qrels.tsv"
    corpus_file: str = "../output/corpus.jsonl"

    # Output paths
    results_dir: str = "results"
    retrieval_results_file: str = "results/retrieval_results.jsonl"
    metrics_overall_file: str = "results/metrics_overall.json"
    metrics_by_type_file: str = "results/metrics_by_type.json"
    metrics_by_k_file: str = "results/metrics_by_k.json"
    detailed_report_file: str = "results/detailed_report.txt"
    failed_queries_file: str = "results/failed_queries.jsonl"

    # Graded relevance mapping (from qrels) - Use field() for dict default
    relevance_levels: dict = field(default_factory=lambda: {
        0: "irrelevant",
        1: "partially_relevant",
        2: "relevant",
        3: "highly_relevant"
    })

    def get_max_k(self) -> int:
        """Get maximum K value for retrieval. Optimized: no error handling needed with defaults."""
        return max(self.k_values)

