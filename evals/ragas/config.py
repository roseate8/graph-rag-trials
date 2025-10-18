"""Configuration for Ragas synthetic test data generation."""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Add vector-ingest to path for llm_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vector-ingest" / "src" / "chunking" / "processors"))

# Elasticsearch Configuration
ELASTICSEARCH_CONFIG = {
    "url": "https://1600c6e333fd4bdb8c8e9b9dec5c5fef.us-west-2.aws.found.io:443",
    "username": "elastic",
    "password": "XI6rIccvUKLCgVnX11QPI8CV",
    "index_name": "embeddings_index_fixed",  # Using available index with 5498 docs
    "verify_certs": True,
    "timeout": 30,
}

# Ragas Test Generation Configuration
RAGAS_CONFIG = {
    "testset_size": 100,
    "llm_provider": "openai",
    "generator_model": "gpt-4o-mini",
    "critic_model": "gpt-4o-mini",
    "embeddings_model": "text-embedding-3-small",
    "distributions": {"simple": 0.4, "reasoning": 0.3, "multi_context": 0.2, "conditional": 0.1},
    "max_documents": 500,
    "sample_strategy": "random",
}

# OpenAI Configuration - Uses secure llm_utils
try:
    from llm_utils import get_openai_api_key, has_openai_api_key
    _has_llm_utils = True
except ImportError:
    _has_llm_utils = False
    get_openai_api_key = None
    has_openai_api_key = None

def get_api_key():
    """Get OpenAI API key using secure llm_utils."""
    if _has_llm_utils and get_openai_api_key:
        return get_openai_api_key()
    return None

OPENAI_CONFIG = {
    "get_api_key": get_api_key,  # Function to get key securely
    "timeout": 60,
    "max_retries": 3,
}

# Output Configuration
OUTPUT_CONFIG = {
    "output_dir": "output",
    "testset_csv": "testset.csv",
    "testset_json": "testset.json",
    "report_txt": "generation_report.txt",
    "stats_json": "generation_stats.json",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "log_file": "ragas_generation.log",
}


@dataclass
class ElasticsearchConfig:
    """Elasticsearch connection configuration."""
    url: str
    username: str
    password: str
    index_name: str
    verify_certs: bool = True
    timeout: int = 30


@dataclass
class RagasGenerationConfig:
    """Ragas test generation configuration."""
    testset_size: int
    generator_model: str
    critic_model: str
    embeddings_model: str
    max_documents: int
    distributions: dict = field(default_factory=dict)


def validate_config() -> bool:
    """Validate configuration settings."""
    errors = []
    
    # Check if we can get API key using secure method
    if RAGAS_CONFIG["llm_provider"] == "openai":
        if not _has_llm_utils:
            errors.append("llm_utils not available - cannot manage OpenAI API key securely")
        elif has_openai_api_key and not has_openai_api_key():
            # Don't error - llm_utils will prompt user when needed
            pass
    
    dist_sum = sum(RAGAS_CONFIG["distributions"].values())
    if not (0.99 <= dist_sum <= 1.01):
        errors.append(f"Distribution values must sum to 1.0, got {dist_sum}")
    
    if RAGAS_CONFIG["testset_size"] < 1:
        errors.append("testset_size must be at least 1")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

