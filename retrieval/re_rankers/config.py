"""
Configuration classes for re-ranking implementations.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ReRankerConfig:
    """Configuration for re-ranking models."""
    
    # Model configuration
    model_name: str = "cross-encoder/ms-marco-MiniLM-L12-v2"
    model_cache_dir: Optional[str] = None
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    # Processing parameters
    batch_size: int = 8
    max_length: int = 512
    normalize_scores: bool = True
    
    # Caching configuration
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Performance tuning
    use_fp16: bool = True  # Use half precision for CUDA
    num_threads: Optional[int] = None  # For CPU inference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ReRankerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        validations = [
            (self.batch_size <= 0, "batch_size must be positive"),
            (self.max_length <= 0, "max_length must be positive"),
            (self.cache_size <= 0, "cache_size must be positive"),
            (self.device not in ("auto", "cpu", "cuda", "mps"), f"Invalid device: {self.device}")
        ]
        
        for condition, message in validations:
            if condition:
                raise ValueError(message)


# Predefined configurations for different use cases
DEFAULT_CONFIG = ReRankerConfig()

FAST_CONFIG = ReRankerConfig(
    batch_size=16,
    max_length=256,
    enable_caching=True,
    cache_size=500
)

ACCURATE_CONFIG = ReRankerConfig(
    batch_size=4,
    max_length=1024,
    normalize_scores=True,
    enable_caching=True,
    cache_size=2000
)

CPU_CONFIG = ReRankerConfig(
    device="cpu",
    batch_size=4,
    max_length=512,
    use_fp16=False,
    num_threads=4
)

GPU_CONFIG = ReRankerConfig(
    device="cuda",
    batch_size=16,
    max_length=512,
    use_fp16=True
)
