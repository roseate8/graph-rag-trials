# Query Decomposition with Multi-View Retrieval and Fusion Re-ranking

## Implementation Plan

### Overview
Implement an intelligent query decomposition system that uses LLM to analyze user queries and generate optimal sub-queries (1-5) including paraphrases, expansions, compressions, and sub-questions. Each sub-query retrieves K chunks, all K*J chunks are then fusion re-ranked against all sub-queries to produce final top-K results.

---

## Strategy & Approach

### Query Decomposition Strategy
The LLM will intelligently analyze the query and decide:
- **Sub-questions**: Does it need breaking into sub-questions? (complex multi-part queries)
- **Paraphrasing**: Does it need paraphrasing? (ambiguous terms, multiple interpretations)
- **Expansion**: Does it need expansion? (short queries that need context)
- **Compression**: Does it need compression? (verbose queries with core intent)
- **Original**: Should we use the original? (always included as baseline)

This produces **1-5 queries dynamically** based on query complexity.

### Fusion Re-ranking Strategy (Reciprocal Rank Fusion)

**Pipeline:**
1. **Retrieve** K chunks for each of J sub-queries → total K*J chunks (with deduplication)
2. **Re-rank** each unique chunk against ALL sub-queries:
   - Compute relevance score against EACH sub-query using re-ranker
   - Apply Reciprocal Rank Fusion (RRF): `score = Σ(1/(k + rank_i))` across all sub-queries
   - This gives higher weight to chunks relevant to multiple sub-queries
3. **Sort** by fusion score and select top K chunks
4. **Generate** final answer using top K chunks

**Why Reciprocal Rank Fusion (RRF)?**
- **Score normalization**: Handles score normalization across different sub-queries automatically
- **Multi-faceted relevance**: Chunks appearing in multiple sub-query results get boosted
- **Robustness**: More robust than simple score averaging (handles score scale differences)
- **Proven effectiveness**: Industry-standard approach in multi-query retrieval scenarios
- **No hyperparameters**: Only k constant (typically 60) needs tuning

**Mathematical Formula:**
```
RRF_score(chunk) = Σ [1 / (k + rank_i)]
                   for i in all sub-queries where chunk appears

where:
- k = constant (typically 60)
- rank_i = rank position of chunk in sub-query i results (1-indexed)
```

---

## Architecture

### Component Hierarchy
```
RAGSystem (core.py)
    ├── QueryDecomposer (query_decomposer.py) [NEW]
    │   └── Uses llm_utils for secure LLM calls
    ├── MilvusRetriever (retrieval.py)
    │   └── retrieve_multi_query() [NEW METHOD]
    └── FusionReranker (fusion_reranker.py) [NEW]
        └── Uses existing reranker from retrieval.py
```

### Data Flow
```
User Query
    ↓
[QueryDecomposer] → Generates 1-5 sub-queries
    ↓
[MilvusRetriever.retrieve_multi_query()] → K chunks per sub-query
    ↓
[Deduplicate] → Unique chunks from all results
    ↓
[FusionReranker] → Re-rank each chunk against ALL sub-queries
    ↓
[RRF Score Calculation] → Aggregate scores
    ↓
[Sort & Select Top K] → Final K chunks
    ↓
[Context Formatting & LLM Generation] → Final answer
```

---

## Files to Create/Modify

### 1. Create: `retrieval/query_decomposer.py` (NEW)
**Purpose:** LLM-powered intelligent query decomposition

**Key Components:**
```python
@dataclass
class DecomposedQuery:
    original_query: str
    sub_queries: List[str]
    decomposition_reasoning: str
    query_count: int

class QueryDecomposer:
    def __init__(self, max_sub_queries: int = 5)
    def decompose_query(self, query: str) -> DecomposedQuery
    def _build_decomposition_prompt(self, query: str) -> str
    def _parse_llm_response(self, response: str) -> DecomposedQuery
```

**LLM Prompt Structure:**
```
Analyze the following user query and generate optimal sub-queries for retrieval.

QUERY: {query}

Your task:
1. Determine if the query needs decomposition
2. Generate 1-5 sub-queries based on:
   - Sub-questions: Break complex multi-part queries
   - Paraphrases: Alternative phrasings for ambiguous terms
   - Expanded: Add context if query is too terse
   - Compressed: Extract core intent if verbose
   - Original: Always consider including original

Return JSON with structure...
```

### 2. Create: `retrieval/fusion_reranker.py` (NEW)
**Purpose:** Reciprocal Rank Fusion across multiple sub-queries

**Key Components:**
```python
@dataclass
class FusionResult:
    chunk: RetrievedChunk
    fusion_score: float
    sub_query_ranks: Dict[str, int]
    sub_query_scores: Dict[str, float]

class FusionReranker:
    def __init__(self, k_constant: int = 60)
    def fuse_and_rerank(
        sub_queries: List[str],
        chunk_results: Dict[str, List[RetrievedChunk]],
        top_k: int,
        reranker
    ) -> List[RetrievedChunk]
    def _calculate_rrf_score(chunk_ranks: Dict[str, int]) -> float
```

**RRF Algorithm Implementation:**
```python
# For each unique chunk:
rrf_score = 0
for sub_query, rank_in_results in chunk_appearances.items():
    rrf_score += 1 / (k_constant + rank_in_results)

# Chunks appearing in multiple sub-query results get higher scores
```

### 3. Modify: `retrieval/retrieval.py`
**Changes:** Add multi-query retrieval capability

**New Method (line ~450):**
```python
def retrieve_multi_query(
    self,
    queries: List[str],
    top_k_per_query: int = 10,
    min_similarity: float = 0.0
) -> Dict[str, List[RetrievedChunk]]:
    """Retrieve chunks for multiple queries efficiently."""
```

### 4. Modify: `retrieval/core.py`
**Changes:** Integrate decomposition + fusion into RAG pipeline

**A. Update `__init__` (line ~47):**
- Add `enable_query_decomposition` parameter
- Add `max_sub_queries` parameter
- Add `fusion_k_constant` parameter
- Initialize QueryDecomposer and FusionReranker when enabled

**B. Update `query` method (line ~284):**
- Add conditional logic for decomposition vs direct retrieval
- Route to new `_retrieve_with_decomposition` when enabled

**C. Add new method (line ~367):**
```python
def _retrieve_with_decomposition(
    self, user_query: str, top_k: int, min_similarity: float
) -> Tuple[List, float]:
    """Execute multi-query retrieval with fusion re-ranking."""
```

### 5. Modify: `retrieval/config.py`
**Changes:** Add configuration parameters (line ~56)

```python
# Query decomposition settings
enable_query_decomposition: bool = False
max_sub_queries: int = 5
fusion_k_constant: int = 60
```

### 6. Update: `retrieval/__init__.py`
**Changes:** Export new components

```python
from .query_decomposer import QueryDecomposer, DecomposedQuery
from .fusion_reranker import FusionReranker, FusionResult
```

### 7. Create: `retrieval/QUERY_DECOMPOSITION.md` (NEW)
**Purpose:** Comprehensive user documentation

**Contents:**
- Architecture overview with diagrams
- Query decomposition strategy explanation
- Fusion re-ranking algorithm details
- Configuration options
- Usage examples
- Performance considerations
- Troubleshooting guide

---

## Configuration & Usage

### Enable Query Decomposition

**Method 1: Factory Function**
```python
from retrieval import create_rag_system

rag = create_rag_system(
    llm_type="openai",
    enable_query_decomposition=True,
    max_sub_queries=5,
    fusion_k_constant=60,
    enable_reranking=True  # Required for fusion
)

result = rag.query("What are the revenue and expense trends?", top_k=10)
```

**Method 2: Direct Instantiation**
```python
from retrieval import RAGSystem

rag = RAGSystem(
    enable_query_decomposition=True,
    max_sub_queries=5,
    fusion_k_constant=60,
    enable_reranking=True
)
```

**Method 3: Via Configuration**
```python
from retrieval import RAGConfig, RAGSystem

config = RAGConfig(
    enable_query_decomposition=True,
    max_sub_queries=5,
    fusion_k_constant=60
)

rag = RAGSystem(**config.__dict__)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_query_decomposition` | bool | False | Enable query decomposition and fusion re-ranking |
| `max_sub_queries` | int | 5 | Maximum number of sub-queries to generate (1-5) |
| `fusion_k_constant` | int | 60 | RRF k constant for score calculation |
| `enable_reranking` | bool | True | Must be True for decomposition to work |

---

## Testing Strategy

### 1. Unit Tests
- **QueryDecomposer**: Test with various query types
  - Simple queries → should generate 1-2 queries
  - Complex queries → should generate 3-5 queries
  - Multi-part queries → should break into sub-questions
  - Ambiguous queries → should generate paraphrases

- **FusionReranker**: Test RRF calculation
  - Single query result → should match original ranking
  - Multiple query results → should boost shared chunks
  - Edge cases: empty results, duplicate chunks

### 2. Integration Tests
- Full pipeline with decomposition enabled
- Compare with traditional retrieval
- Verify chunk deduplication
- Validate scoring aggregation

### 3. Performance Benchmarks
- Measure latency overhead
- Token usage comparison
- Memory footprint
- Cache effectiveness

---

## Performance Considerations

### Latency
- **LLM decomposition call**: ~1-2 seconds (one-time per query)
- **Multiple retrievals**: Parallelizable (K*J embeddings)
- **Fusion re-ranking**: Linear complexity O(K*J*num_queries)

### Optimization Strategies
1. **Cache decomposed queries** for repeated queries
2. **Batch embedding generation** for sub-queries
3. **Parallel retrieval** for multiple sub-queries
4. **Early stopping** in re-ranking if scores converge

### Token Usage
- Decomposition prompt: ~200-300 tokens
- Decomposition response: ~100-200 tokens
- Total overhead: ~300-500 tokens per query

---

## Backward Compatibility

- **Default behavior unchanged**: Decomposition is opt-in
- **Existing APIs preserved**: No breaking changes
- **Configuration backward compatible**: New fields have defaults
- **Can be disabled**: Set `enable_query_decomposition=False`

---

## Scope Limitations

✅ **Included:**
- All changes confined to `retrieval/` folder
- New components: query_decomposer.py, fusion_reranker.py
- Modified components: retrieval.py, core.py, config.py, __init__.py
- Comprehensive documentation

❌ **Excluded:**
- No modifications to `rag-ui/`, `backup-rag-ui/`
- No modifications to `evals/`
- No modifications to `vector-ingest/`
- UI integration (can be added later)

---

## Future Enhancements

1. **Query decomposition caching**: Cache LLM decomposition results
2. **Adaptive sub-query count**: Learn optimal J from query patterns
3. **Custom fusion strategies**: Beyond RRF (weighted fusion, learned fusion)
4. **Query performance analytics**: Track which decompositions work best
5. **UI controls**: Expose decomposition toggle in web interface

---

## References

- Reciprocal Rank Fusion: Cormack et al. (2009)
- Multi-query retrieval strategies in IR literature
- Query expansion and reformulation techniques
- Fusion methods for information retrieval

---

**Implementation Date**: 2025-10-10  
**Version**: 1.0  
**Author**: Graph RAG Trials Team

