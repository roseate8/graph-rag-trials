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
    target_sample_size: int = 100  # Reduced for faster testing
    num_clusters: int = 10  # Reduced clusters for faster K-means
    
    # Query generation parameters
    target_questions: int = 200  # Updated to generate more queries
    queries_per_fact_min: int = 1  # Reduced to use more diverse facts
    queries_per_fact_max: int = 2  # Reduced to use more diverse facts
    multi_hop_ratio: float = 0.4  # 40% multi-hop, 60% single-hop
    
    # LLM parameters
    model_name: str = "gpt-4.1-nano"
    max_completion_tokens: int = 2000  # Used for gpt-5 models
    max_tokens: int = 2000  # Used for other models
    
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
    batch_size: int = 5  # Smaller batches for memory efficiency
    enable_llm_judge: bool = False  # Disabled for faster processing
    max_facts_per_chunk: int = 10  # Limit facts per chunk
    
    # Output parameters
    output_dir: str = "output"  # Relative to synthetic-eval directory
    save_intermediate: bool = True  # Save intermediate results
    
    # Validation parameters
    validate_retrieval: bool = False  # Validate that gold chunks are retrievable
    validation_top_k: int = 10
    
    def get_llm_params(self, base_params: dict = None) -> dict:
        """
        Get LLM parameters with correct token parameter based on model.

        Args:
            base_params: Base parameters to extend (optional)

        Returns:
            Dictionary with appropriate token parameter for the model
        """
        params = base_params or {}

        # GPT-5 models use max_completion_tokens
        if self.model_name.startswith("gpt-5"):
            params["max_completion_tokens"] = self.max_completion_tokens
        else:
            params["max_tokens"] = self.max_tokens

        return params

    def __post_init__(self):
        """Validate configuration parameters."""
        if not (10 <= self.target_questions <= 800):  # Allow smaller values for testing
            raise ValueError("target_questions must be between 10 and 800")

        if not (0.0 <= self.multi_hop_ratio <= 1.0):
            raise ValueError("multi_hop_ratio must be between 0.0 and 1.0")

        if self.queries_per_fact_min > self.queries_per_fact_max:
            raise ValueError("queries_per_fact_min must be <= queries_per_fact_max")

