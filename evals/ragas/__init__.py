"""Ragas-based synthetic test data generation for RAG evaluation."""

__version__ = "1.0.0"

from .config import (
    ELASTICSEARCH_CONFIG,
    RAGAS_CONFIG,
    OPENAI_CONFIG,
    OUTPUT_CONFIG,
    validate_config,
)

from .elasticsearch_loader import (
    ElasticsearchDocumentLoader,
    load_documents_for_ragas,
)

from .generate_testset import RagasTestsetGenerator

from .evaluate_rag import RagasEvaluator

__all__ = [
    "ELASTICSEARCH_CONFIG",
    "RAGAS_CONFIG",
    "OPENAI_CONFIG",
    "OUTPUT_CONFIG",
    "validate_config",
    "ElasticsearchDocumentLoader",
    "load_documents_for_ragas",
    "RagasTestsetGenerator",
    "RagasEvaluator",
]
