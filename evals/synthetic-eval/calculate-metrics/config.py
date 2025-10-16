"""
Configuration for synthetic evaluation metrics calculation.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics calculation."""
    
    # Retrieval settings (must match your existing system)
    collection_name: str = "elastic_embeddings_m3"
    embedding_model: str = "BAAI/bge-m3"
    milvus_profile: str = "production"
    enable_reranking: bool = True
    
    # Evaluation K values
    k_values: List[int] = None
    
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
    
    # Graded relevance mapping (from qrels)
    relevance_levels: dict = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10, 20, 50, 100]
        
        if self.relevance_levels is None:
            # Graded relevance: 0=irrelevant, 1=partially, 2=relevant, 3=highly_relevant
            self.relevance_levels = {
                0: "irrelevant",
                1: "partially_relevant",
                2: "relevant",
                3: "highly_relevant"
            }
    
    def get_max_k(self) -> int:
        """Get maximum K value for retrieval."""
        return max(self.k_values)

