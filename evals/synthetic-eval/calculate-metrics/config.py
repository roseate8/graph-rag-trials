"""
Configuration for synthetic evaluation metrics calculation.
Optimized with efficient defaults and validation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class EvalConfig:
    """
    Configuration for evaluation metrics calculation.

    Architecture: Uses **kwargs pass-through pattern for RAGSystem parameters.
    This ensures 100% dependency on retrieval/core.py - any new parameters added
    to RAGSystem automatically work here without code changes.
    """

    # =============================================================================
    # RAGSystem Parameters (passed via **kwargs to retrieval/core.py)
    # =============================================================================
    # Any parameters here are passed directly to RAGSystem.__init__()
    # If you add new parameters to RAGSystem, just add them to this dict.
    # No changes needed in retriever_for_evals.py!

    rag_system_params: Dict[str, Any] = field(default_factory=lambda: {
        # Retriever parameters
        'embedding_model': "BAAI/bge-m3",
        'collection_name': "elastic_embeddings_m3",

        # Re-ranking parameters
        'enable_reranking': True,
        'reranker_config': None,  # Custom re-ranker configuration
        'retrieval_multiplier': 10,  # Multiplier for initial retrieval when re-ranking

        # Query decomposition parameters
        'enable_query_decomposition': True,  # Enable multi-query retrieval with fusion
        'max_sub_queries': 5,  # Maximum number of sub-queries to generate
        'fusion_k_constant': 60,  # K constant for reciprocal rank fusion

        # Context formatting parameters
        'max_context_tokens': 4000,  # Maximum tokens for context
        'include_scores': False,  # Include similarity scores in formatted context

        # LLM parameters (mock for evaluation - no actual generation needed)
        'llm_type': "mock",
        'llm_model': "gpt-4o-mini",

        # History parameters (disabled for evaluation)
        'enable_history': False,
        'history_file': "conversation_history.json"
    })

    # =============================================================================
    # Evaluation-Specific Parameters (not passed to RAGSystem)
    # =============================================================================

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

