# Changes Made: RAGSystem Integration

## Summary
Refactored the evaluation system to use `retrieval/core.py` (RAGSystem) instead of directly using `MilvusRetriever`, ensuring 100% consistency between production retrieval and evaluation metrics.

## Files Modified

### 1. `config.py`
**Changes:**
- Added `retrieval_multiplier: int = 10` for re-ranking configuration
- Added `reranker_config: Optional[Dict[str, Any]] = None` for custom re-ranker settings
- Added `enable_query_decomposition: bool = False` to enable multi-query retrieval
- Added `max_sub_queries: int = 5` for decomposition
- Added `fusion_k_constant: int = 60` for reciprocal rank fusion
- Added `max_context_tokens: int = 4000` for context formatting
- Added `include_scores: bool = False` for score inclusion in context
- Added `llm_type: str = "mock"` for LLM configuration (mock for evaluation)
- Added `llm_model: str = "gpt-4o-mini"` for model selection
- Added `enable_history: bool = False` to disable history for evaluation

**Impact:** EvalConfig now mirrors all RAGSystem parameters, allowing full control over the retrieval pipeline.

### 2. `retriever_for_evals.py`
**Changes:**
- Changed import from `from retrieval.retrieval import MilvusRetriever` to `from retrieval.core import RAGSystem`
- Updated class docstring to explain RAGSystem dependency
- Refactored `__init__` to mention RAGSystem integration
- Refactored `connect()` to instantiate RAGSystem with all config parameters
- Refactored `retrieve_single()` to:
  - Call `rag_system.query()` instead of `retriever.retrieve()`
  - Extract chunks from `RAGResult.retrieved_chunks`
  - Prefer `rerank_score` when available, fallback to `similarity_score`
  - Include additional metadata (similarity_score, rerank_score, rerank_probability)
- Added `retrieval.core` to suppressed loggers during batch processing

**Impact:** All retrieval now goes through RAGSystem, automatically including query decomposition, re-ranking, and fusion when enabled.

### 3. `README.md`
**Changes:**
- Added prominent warning at the top about `retrieval/core.py` dependency
- Added "Full RAG pipeline integration" to overview features
- Added "Dependency Flow" diagram showing RAGSystem as main dependency
- Rewrote "Integration with Existing System" section to explain RAGSystem usage
- Added "Why RAGSystem Integration?" section with 4 key benefits
- Added "Retrieval Score Handling" explanation
- Updated code examples to show RAGSystem configuration

**Impact:** Documentation now clearly explains the dependency and its benefits.

### 4. `INTEGRATION_SUMMARY.md` (New file)
**Content:**
- Detailed explanation of changes
- Before/after code comparisons
- Benefits of RAGSystem integration
- Usage examples
- Migration notes
- Troubleshooting guide

**Impact:** Comprehensive reference for understanding the integration.

## Key Changes Summary

### Before
```python
# Direct MilvusRetriever usage
from retrieval.retrieval import MilvusRetriever

retriever = MilvusRetriever(
    embedding_model=config.embedding_model,
    collection_name=config.collection_name,
    enable_reranking=config.enable_reranking
)

results = retriever.retrieve(query_text, top_k, min_similarity)
```

### After
```python
# RAGSystem integration
from retrieval.core import RAGSystem

rag_system = RAGSystem(
    embedding_model=config.embedding_model,
    collection_name=config.collection_name,
    enable_reranking=config.enable_reranking,
    retrieval_multiplier=config.retrieval_multiplier,
    enable_query_decomposition=config.enable_query_decomposition,
    max_sub_queries=config.max_sub_queries,
    fusion_k_constant=config.fusion_k_constant,
    max_context_tokens=config.max_context_tokens,
    include_scores=config.include_scores,
    llm_type=config.llm_type,
    enable_history=config.enable_history
)

rag_result = rag_system.query(query_text, top_k, min_similarity)
chunks = rag_result.retrieved_chunks  # Full pipeline results
```

## Benefits Achieved

✅ **Automatic pipeline updates**: Changes to `retrieval/core.py` flow into evaluation automatically  
✅ **Feature parity**: Evaluations use exact same retrieval as production  
✅ **Single source of truth**: Only edit `retrieval/` for retrieval changes  
✅ **Configuration flexibility**: All RAG parameters configurable via `config.py`  
✅ **Better score handling**: Prefers re-ranking scores when available  
✅ **Query decomposition support**: Can enable multi-query retrieval in evaluation  
✅ **Fusion re-ranking support**: Can test reciprocal rank fusion strategies  

## Testing Checklist

- [x] No linter errors in modified files
- [x] Configuration accepts all new parameters
- [x] RAGSystem imported correctly
- [x] All config parameters passed to RAGSystem
- [x] Score handling prefers rerank_score over similarity_score
- [x] Documentation updated to explain dependency
- [x] Dependency flow diagram added
- [ ] Run test evaluation: `python main.py --num-queries 10`
- [ ] Verify retrieval results populated correctly
- [ ] Verify metrics calculated correctly
- [ ] Test with query decomposition enabled
- [ ] Test with query decomposition disabled

## Next Steps

1. **Test the integration**: Run `python main.py --num-queries 10` to verify everything works
2. **Compare results**: Run with/without re-ranking to ensure scores are handled correctly
3. **Enable query decomposition**: Set `enable_query_decomposition = True` in config and re-run
4. **Benchmark**: Compare evaluation metrics before/after to ensure consistency

## Rollback Plan

If issues arise, you can temporarily revert by:
1. Change import back to `from retrieval.retrieval import MilvusRetriever`
2. Update `connect()` to use `MilvusRetriever` instead of `RAGSystem`
3. Update `retrieve_single()` to call `retriever.retrieve()` instead of `rag_system.query()`

However, this is not recommended as it breaks the dependency on `retrieval/core.py`.

