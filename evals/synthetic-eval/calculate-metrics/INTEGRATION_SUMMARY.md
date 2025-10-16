# RAGSystem Integration Summary

## Overview

The evaluation system has been refactored to use `retrieval/core.py` (RAGSystem) as its retrieval engine instead of directly using `MilvusRetriever`. This ensures that any changes to the retrieval pipeline automatically flow into evaluation without requiring code changes.

## Changes Made

### 1. Configuration Extensions (`config.py`)

Added new parameters to `EvalConfig` to support all RAGSystem features:

```python
# Re-ranking configuration
retrieval_multiplier: int = 10
reranker_config: Optional[Dict[str, Any]] = None

# Query decomposition configuration
enable_query_decomposition: bool = False
max_sub_queries: int = 5
fusion_k_constant: int = 60

# Context formatting
max_context_tokens: int = 4000
include_scores: bool = False

# LLM configuration (mock for evaluation)
llm_type: str = "mock"
llm_model: str = "gpt-4o-mini"

# History disabled for evaluation
enable_history: bool = False
```

### 2. Retriever Refactoring (`retriever_for_evals.py`)

**Before:**
```python
from retrieval.retrieval import MilvusRetriever

self.retriever = MilvusRetriever(
    embedding_model=config.embedding_model,
    collection_name=config.collection_name,
    enable_reranking=config.enable_reranking
)

results = self.retriever.retrieve(query_text, top_k, min_similarity)
```

**After:**
```python
from retrieval.core import RAGSystem

self.rag_system = RAGSystem(
    # All retrieval pipeline parameters
    embedding_model=config.embedding_model,
    collection_name=config.collection_name,
    enable_reranking=config.enable_reranking,
    retrieval_multiplier=config.retrieval_multiplier,
    enable_query_decomposition=config.enable_query_decomposition,
    max_sub_queries=config.max_sub_queries,
    fusion_k_constant=config.fusion_k_constant,
    # ... and more
)

rag_result = self.rag_system.query(query_text, top_k, min_similarity)
# Use rag_result.retrieved_chunks
```

### 3. Score Handling

The evaluation system now intelligently handles both similarity and re-ranking scores:

```python
retrieved_docs = [
    {
        'chunk_id': chunk.chunk_id,
        'score': chunk.rerank_score if chunk.rerank_score is not None else chunk.similarity_score,
        'rank': rank,
        # Additional metadata for analysis
        'similarity_score': chunk.similarity_score,
        'rerank_score': chunk.rerank_score,
        'rerank_probability': chunk.rerank_probability
    }
    for rank, chunk in enumerate(rag_result.retrieved_chunks, start=1)
]
```

This ensures metrics reflect the actual ranking used in production (re-ranked if enabled, otherwise vector similarity).

### 4. Documentation (`README.md`)

- Added prominent warning about `retrieval/core.py` dependency
- Added dependency flow diagram
- Documented RAGSystem integration benefits
- Explained score handling logic
- Updated configuration examples

## Benefits

### 1. Automatic Pipeline Updates
Any changes to `retrieval/core.py` or its dependencies automatically flow into evaluation:
- Query decomposition strategies
- Re-ranking algorithms
- Fusion methods
- Bug fixes and optimizations

### 2. Feature Parity
Evaluations use the exact same retrieval logic as production:
- ✅ Query decomposition (multi-query retrieval)
- ✅ Reciprocal rank fusion
- ✅ Cross-encoder re-ranking
- ✅ All optimizations and caching

### 3. Single Source of Truth
Only edit `retrieval/` folder for retrieval changes. The evaluation system automatically inherits those changes.

### 4. Configuration Flexibility
All RAG system parameters are configurable via `config.py` without modifying evaluation code.

## Usage Examples

### Enable Query Decomposition
```python
# In config.py
enable_query_decomposition = True
max_sub_queries = 5
fusion_k_constant = 60
```

Then run:
```bash
python main.py
```

The evaluation will automatically use query decomposition!

### Adjust Re-ranking
```python
# In config.py
enable_reranking = True
retrieval_multiplier = 20  # Retrieve 20x more for better re-ranking
```

### Compare Configurations
```bash
# With query decomposition
python main.py --output-dir results_with_decomposition

# Edit config.py to disable it, then:
python main.py --output-dir results_without_decomposition

# Compare the metrics!
```

## Migration Notes

### No Breaking Changes
The evaluation system still produces the same output files and metrics. The only difference is:
- It now uses RAGSystem internally
- Scores may differ if re-ranking is enabled (this is expected and desired)

### Testing
To verify the integration:
1. Run a small evaluation: `python main.py --num-queries 10`
2. Check that retrieval results are populated
3. Verify metrics are calculated correctly

### Troubleshooting

If you see import errors:
```
ModuleNotFoundError: No module named 'retrieval.core'
```

Solution: Ensure the project root is in your Python path (the code handles this automatically, but just in case).

If you see connection issues:
```
Failed to connect to Milvus
```

Solution: Ensure Milvus is running and the configuration in `retrieval/core.py` is correct.

## Future Improvements

This integration enables:
1. **A/B testing**: Compare different retrieval strategies by simply editing `retrieval/core.py`
2. **Automated benchmarking**: Run evaluations on every retrieval pipeline change
3. **Continuous monitoring**: Track retrieval performance over time as you make improvements

## Summary

✅ **All evaluation code now depends on `retrieval/core.py`**  
✅ **No need to update evaluation code when changing retrieval logic**  
✅ **100% consistency between production and evaluation**  
✅ **All RAG features (decomposition, re-ranking, fusion) available in evaluation**  

The evaluation system is now truly synchronized with your production retrieval pipeline!

