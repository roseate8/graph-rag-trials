"""
Query decomposer module for intelligent multi-view query generation.
"""

from .query_decomposer import QueryDecomposer, DecomposedQuery, decompose_query_simple

__all__ = [
    "QueryDecomposer",
    "DecomposedQuery",
    "decompose_query_simple"
]

