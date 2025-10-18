"""Ragas-based synthetic test data generation for RAG evaluation."""

__version__ = "1.0.0"

from .config import (
    MILVUS_CONFIG,
    RAGAS_CONFIG,
    OPENAI_CONFIG,
    OUTPUT_CONFIG,
    validate_config,
)

from .milvus_loader import (
    MilvusDocumentLoader,
    load_documents_for_ragas,
)

from .generate_testset import RagasTestsetGenerator

from .evaluate_rag import RagasEvaluator

__all__ = [
    "MILVUS_CONFIG",
    "RAGAS_CONFIG",
    "OPENAI_CONFIG",
    "OUTPUT_CONFIG",
    "validate_config",
    "MilvusDocumentLoader",
    "load_documents_for_ragas",
    "RagasTestsetGenerator",
    "RagasEvaluator",
]
