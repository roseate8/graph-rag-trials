"""
Metadata enrichment package for advanced entity extraction and financial analysis.

This package provides independent metadata extraction capabilities using various
NLP models and techniques for enhanced document understanding.
"""

from .spacy_extractor import SpacyMetadataExtractor, create_spacy_extractor
from .core_handler import enrich_chunks_with_structure

__all__ = [
    'SpacyMetadataExtractor',
    'create_spacy_extractor',
    'enrich_chunks_with_structure'
]