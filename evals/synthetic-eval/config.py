"""
Configuration for synthetic evaluation dataset generation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SyntheticEvalConfig:
    """Configuration for synthetic evaluation dataset generation."""
    
    # Sampling parameters
    total_chunks: int = 15000
    target_sample_size: int = 400
    num_clusters: int = 20  # Topic clusters for stratification
    
    # Query generation parameters
    target_questions: int = 400  # Target: 300-800 range
    queries_per_fact_min: int = 3  # Minimum queries per fact
    queries_per_fact_max: int = 5  # Maximum queries per fact
    multi_hop_ratio: float = 0.2  # 20% multi-hop queries
    
    # LLM parameters
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 2000
    # Note: gpt-4o-mini may not support temperature parameter
    
    # Milvus parameters
    collection_name: str = "elastic_embeddings_m3"
    embedding_model: str = "BAAI/bge-m3"
    milvus_profile: str = "production"
    
    # Silver labeling thresholds
    exact_match_threshold: float = 0.9  # For rel=3
    token_f1_high: float = 0.7  # For rel=3
    token_f1_mid: float = 0.4   # For rel=2
    token_f1_low: float = 0.3   # For LLM judge
    semantic_similarity_threshold: float = 0.75  # For rel=2
    
    # Processing parameters
    batch_size: int = 10  # Batch size for LLM calls
    enable_llm_judge: bool = True  # Use LLM for ambiguous cases
    
    # Output parameters
    output_dir: str = "evals/synthetic-eval/output"
    save_intermediate: bool = True  # Save intermediate results
    
    # Validation parameters
    validate_retrieval: bool = False  # Validate that gold chunks are retrievable
    validation_top_k: int = 10
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (300 <= self.target_questions <= 800):
            raise ValueError("target_questions must be between 300 and 800")
        
        if not (0.0 <= self.multi_hop_ratio <= 1.0):
            raise ValueError("multi_hop_ratio must be between 0.0 and 1.0")
        
        if self.queries_per_fact_min > self.queries_per_fact_max:
            raise ValueError("queries_per_fact_min must be <= queries_per_fact_max")

