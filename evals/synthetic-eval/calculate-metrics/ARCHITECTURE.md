# Calculate-Metrics Architecture

## System Overview

A production-ready evaluation system that calculates comprehensive IR metrics for synthetic evaluation datasets using batch async processing and your existing retrieval infrastructure.

## Design Principles

1. **No Duplication**: Uses existing `MilvusRetriever` - no recreation
2. **Performance**: Batch async processing (15 concurrent) for 4-6x speedup
3. **Graded Relevance**: Proper NDCG calculation with 0-3 relevance levels
4. **Query-Type Aware**: Separate metrics for single-hop vs multi-hop
5. **Comprehensive**: 6 key metrics (Recall, Precision, MAP, MRR, NDCG, Hits)
6. **Production Ready**: Error handling, logging, progress tracking

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         MAIN.PY                              │
│                    (CLI Entry Point)                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                      EVALUATOR.PY                            │
│                 (Pipeline Orchestrator)                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Load Data (queries.jsonl, qrels.tsv)             │   │
│  └──────────────────────────────────────────────────────┘   │
│                      │                                       │
│                      ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 2. Batch Async Retrieval                            │   │
│  │    ┌────────────────────────────────────┐           │   │
│  │    │  retriever_for_evals.py            │           │   │
│  │    │  - Connect to existing Milvus      │           │   │
│  │    │  - 15 concurrent queries           │           │   │
│  │    │  - Progress tracking               │           │   │
│  │    └────────────────────────────────────┘           │   │
│  └──────────────────────────────────────────────────────┘   │
│                      │                                       │
│                      ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 3. Calculate Metrics                                │   │
│  │    ┌────────────────────────────────────┐           │   │
│  │    │  metrics.py                        │           │   │
│  │    │  - Recall@K, Precision@K          │           │   │
│  │    │  - MAP, MRR                        │           │   │
│  │    │  - NDCG@K (graded relevance)      │           │   │
│  │    │  - Hits@K                          │           │   │
│  │    └────────────────────────────────────┘           │   │
│  └──────────────────────────────────────────────────────┘   │
│                      │                                       │
│                      ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 4. Aggregate Results                                │   │
│  │    - Overall (all queries)                          │   │
│  │    - By Type (single-hop / multi-hop)              │   │
│  │    - By K value                                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                      │                                       │
│                      ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 5. Generate Reports                                 │   │
│  │    ┌────────────────────────────────────┐           │   │
│  │    │  reporter.py                       │           │   │
│  │    │  - JSON outputs                    │           │   │
│  │    │  - Human-readable report           │           │   │
│  │    └────────────────────────────────────┘           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    RESULTS DIRECTORY                         │
│  - retrieval_results.jsonl (raw)                            │
│  - metrics_overall.json (aggregated)                        │
│  - metrics_by_type.json (single/multi breakdown)            │
│  - metrics_by_k.json (for visualization)                    │
│  - detailed_report.txt (human-readable)                     │
│  - failed_queries.jsonl (errors)                            │
│  - evaluation.log (execution log)                           │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Input
```
output/
├── queries.jsonl       # 182 queries with metadata
├── qrels.tsv          # Graded relevance (0-3)
└── corpus.jsonl       # Optional document content
```

### Processing
```
Query Batch (15 concurrent)
    ↓
Milvus Retrieval (BGE-M3 + Re-ranking)
    ↓
Top-K Documents (K=100 max)
    ↓
Metrics Calculation (per query)
    ↓
Aggregation (overall + by type)
    ↓
Report Generation
```

### Output
```
results/
├── retrieval_results.jsonl   # Raw retrieval: query → [docs]
├── metrics_overall.json       # Aggregated metrics
├── metrics_by_type.json       # single_hop vs multi_hop
├── metrics_by_k.json          # By K value (for plots)
├── detailed_report.txt        # Human-readable
├── failed_queries.jsonl       # Any failures
└── evaluation.log             # Execution log
```

## Key Technical Decisions

### 1. Async Batch Processing
**Why**: 182 queries × 2s each = ~6 min sequential → 15 concurrent = ~30s per batch
**Implementation**: `asyncio.Semaphore(15)` for concurrency control

### 2. Graded Relevance for NDCG
**Why**: Qrels have 0-3 scale, binary NDCG would lose information
**Implementation**: `DCG = Σ(rel_i / log2(i + 1))` with actual relevance grades

### 3. Integration Not Recreation
**Why**: Leverage existing retrieval cache, configuration, optimization
**Implementation**: Direct import of `MilvusRetriever` with async wrapper

### 4. Per-Query-Type Breakdown
**Why**: Single-hop and multi-hop queries have different characteristics
**Implementation**: Filter and aggregate separately, compare performance

### 5. Multiple K Values
**Why**: Different applications care about different cut-offs
**Implementation**: K ∈ [1, 3, 5, 10, 20, 50, 100] calculated efficiently

## Performance Characteristics

### Time Complexity
- **Retrieval**: O(Q/B × T) where Q=queries, B=batch_size, T=time_per_query
  - Sequential: 182 × 2s = 364s (~6 min)
  - Batch (15): 182/15 × 2s = 24s + overhead = ~1-2 min
  - **Speedup: 4-6x**

- **Metrics**: O(Q × K) where K=max_k_value
  - 182 × 100 = 18,200 relevance lookups
  - **Time: <1 second** (in-memory operations)

### Space Complexity
- **Memory**: O(Q × K) for storing retrieval results
  - 182 queries × 100 docs × 200 bytes ≈ 3.6 MB
  - **Very manageable**

### Scalability
| Queries | Sequential | Batch (15) | Memory |
|---------|------------|------------|--------|
| 100     | ~3 min     | ~30-45 sec | ~2 MB  |
| 182     | ~6 min     | ~1-2 min   | ~4 MB  |
| 500     | ~17 min    | ~4-6 min   | ~10 MB |
| 1000    | ~33 min    | ~8-12 min  | ~20 MB |

## Error Handling Strategy

### Graceful Degradation
```python
try:
    result = await retrieve_single(query, k)
    results.append(result)  # Success
except Exception as e:
    results.append(RetrievalResult(
        query_id=query_id,
        success=False,
        error=str(e)
    ))  # Failed but continue
```

### Levels
1. **Query Level**: Individual query failure doesn't stop batch
2. **Batch Level**: One batch failure doesn't stop evaluation
3. **System Level**: Connection errors abort gracefully

### Recovery
- Failed queries logged to `failed_queries.jsonl`
- Metrics calculated for successful queries only
- Success rate reported in summary

## Metrics Implementation Details

### Binary Relevance Metrics

**Recall@K**:
```python
recall@K = |retrieved@K ∩ relevant| / |relevant|
```
Example: Retrieved 7 of 10 relevant docs → Recall@10 = 0.7

**Precision@K**:
```python
precision@K = |retrieved@K ∩ relevant| / K
```
Example: 7 relevant in top-10 → Precision@10 = 0.7

**Average Precision (AP)**:
```python
AP = (1/|relevant|) × Σ(Precision@k × rel(k))
```
Example: Relevant at ranks [2, 5, 8]
- P@2 = 1/2 → 0.5
- P@5 = 2/5 → 0.4
- P@8 = 3/8 → 0.375
- AP = (0.5 + 0.4 + 0.375) / 3 = 0.425

**MAP (Mean Average Precision)**:
```python
MAP = (1/Q) × Σ(AP_q) for all queries Q
```

**MRR (Mean Reciprocal Rank)**:
```python
RR = 1 / rank_first_relevant
MRR = (1/Q) × Σ(RR_q)
```
Example: First relevant at rank 3 → RR = 0.333

**Hits@K**:
```python
Hits@K = 1 if any relevant in top-K, else 0
```

### Graded Relevance Metric

**NDCG@K (Normalized Discounted Cumulative Gain)**:

```python
# Discounted Cumulative Gain
DCG@K = Σ(i=1 to K) [rel_i / log2(i + 1)]

# Ideal DCG (perfect ranking)
IDCG@K = Σ(i=1 to K) [ideal_rel_i / log2(i + 1)]

# Normalized
NDCG@K = DCG@K / IDCG@K
```

Example with graded relevance (0-3):
```
Retrieved: [3, 1, 2, 0, 3, 1, ...]
Positions:  1  2  3  4  5  6

DCG@5 = 3/log2(2) + 1/log2(3) + 2/log2(4) + 0/log2(5) + 3/log2(6)
      = 3.0 + 0.63 + 1.0 + 0.0 + 1.16
      = 5.79

IDCG@5 (sorted: [3, 3, 2, 1, 1])
       = 3/log2(2) + 3/log2(3) + 2/log2(4) + 1/log2(5) + 1/log2(6)
       = 3.0 + 1.89 + 1.0 + 0.43 + 0.39
       = 6.71

NDCG@5 = 5.79 / 6.71 = 0.863
```

**Why NDCG with Grading?**
- Binary NDCG treats all relevant docs equally
- Graded NDCG rewards highly relevant docs more
- Matches qrels structure (0=irrelevant, 1=partial, 2=relevant, 3=highly)

## Configuration Management

### Default Configuration
```python
collection_name = "elastic_embeddings_m3"
embedding_model = "BAAI/bge-m3"
milvus_profile = "production"
enable_reranking = True
batch_size = 15
max_concurrent = 15
k_values = [1, 3, 5, 10, 20, 50, 100]
```

### Customization Points
1. **K Values**: Focus on specific cut-offs
2. **Batch Size**: Trade-off between speed and load
3. **Concurrency**: System-dependent optimization
4. **Relevance Levels**: Adapt to different grading schemes

## Integration Points

### With Retrieval System
```python
# Direct import - no modification needed
from retrieval.retrieval import MilvusRetriever

retriever = MilvusRetriever(
    embedding_model=config.embedding_model,
    milvus_profile=config.milvus_profile,
    collection_name=config.collection_name,
    enable_reranking=config.enable_reranking
)
```

### With Synthetic Eval
```
synthetic-eval/
├── output/
│   ├── queries.jsonl     ← Input
│   ├── qrels.tsv         ← Input
│   └── corpus.jsonl      ← Input (optional)
└── calculate-metrics/
    ├── [evaluation code]
    └── results/          → Output
```

## Future Enhancements

### Potential Additions
1. **Per-query-type K optimization**: Different K for single vs multi-hop
2. **Confidence intervals**: Bootstrap sampling for metric uncertainty
3. **Error analysis**: Automatic failure pattern detection
4. **A/B testing**: Compare different retrieval configurations
5. **Visualization**: Auto-generate plots from metrics_by_k.json
6. **Streaming evaluation**: Process queries in streaming fashion

### Extensibility Points
1. **Custom metrics**: Add to `metrics.py`
2. **Custom aggregations**: Extend `evaluator.py`
3. **Custom reports**: Modify `reporter.py`
4. **Custom filters**: Pre-process queries in `main.py`

## Testing Strategy

### Unit Tests (TODO)
- `test_metrics.py`: Validate metric calculations
- `test_retriever.py`: Test async batch logic
- `test_evaluator.py`: Pipeline integration

### Integration Tests (TODO)
- End-to-end with small test dataset
- Performance benchmarks
- Error handling validation

### Validation
- Compare metrics against reference implementation
- Verify NDCG calculations with known examples
- Cross-check with BEIR evaluation toolkit

## Monitoring and Observability

### Logging Levels
- **INFO**: Progress, summaries, key milestones
- **WARNING**: Failed queries, missing data
- **ERROR**: Critical failures, connection issues
- **DEBUG**: Detailed execution trace (disabled by default)

### Progress Tracking
- Real-time tqdm progress bars
- Per-batch success/failure counts
- Time estimates for remaining work

### Output Files
All results timestamped and versioned for reproducibility

