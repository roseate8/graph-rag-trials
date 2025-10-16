# Metrics Calculation for Synthetic Evaluation

Comprehensive evaluation system for information retrieval metrics using batch async processing.

> **Important**: This evaluation system uses `retrieval/core.py` as its retrieval engine. Any changes you make to the retrieval pipeline (query decomposition, re-ranking, fusion, etc.) will automatically be reflected in evaluation runs. This ensures 100% consistency between production retrieval and evaluation metrics.

## Overview

This system evaluates retrieval performance on synthetic evaluation datasets with:
- **Batch async retrieval** (15 concurrent queries) for speed
- **Graded relevance support** (0-3 scale) for NDCG
- **Query-type breakdowns** (single-hop vs multi-hop)
- **Comprehensive metrics**: Recall, Precision, MAP, MRR, NDCG, Hits
- **Full RAG pipeline integration**: Uses the complete retrieval stack from `retrieval/core.py`

## Architecture

```
calculate-metrics/
├── config.py              # Configuration (BGE-M3, batch settings, RAG params)
├── retriever_for_evals.py # Async wrapper around RAGSystem from retrieval/core.py
├── metrics.py             # Metrics calculation (with graded relevance)
├── evaluator.py           # Pipeline orchestration
├── reporter.py            # Report generation
├── main.py                # CLI entry point
└── results/               # Output directory
    ├── retrieval_results.jsonl
    ├── metrics_overall.json
    ├── metrics_by_type.json
    ├── metrics_by_k.json
    ├── detailed_report.txt
    ├── failed_queries.jsonl
    └── evaluation.log
```

### Dependency Flow

```
main.py
  └─> evaluator.py
      └─> retriever_for_evals.py (EvalRetriever)
          └─> retrieval/core.py (RAGSystem)  ← MAIN DEPENDENCY
              ├─> retrieval/retrieval.py (MilvusRetriever)
              │   ├─> Vector search in Milvus
              │   └─> Cross-encoder re-ranking (if enabled)
              ├─> retrieval/decomposer/query_decomposer.py (if enabled)
              ├─> retrieval/re_rankers/fusion_reranker.py (if enabled)
              ├─> retrieval/formatting.py (ContextFormatter)
              └─> retrieval/llm.py (MockLLMClient for eval)
```

**Key Design**: The evaluation system depends on `retrieval/core.py` as the single source of truth for retrieval behavior. Any changes to the retrieval pipeline automatically propagate to evaluation runs.

## Usage

### Prerequisites

1. Generate synthetic evaluation dataset first:
   ```bash
   cd evals/synthetic-eval
   python -m main
   ```

2. Ensure output files exist:
   - `output/queries.jsonl` (182 queries)
   - `output/qrels.tsv` (relevance judgments)
   - `output/corpus.jsonl` (optional)

### Run Evaluation

```bash
cd evals/synthetic-eval/calculate-metrics
python main.py
```

### Expected Output

```
Starting synthetic evaluation metrics calculation...
[Step 1/5] Loading data...
✓ Loaded 182 queries
  Query types: {'single_hop': 120, 'multi_hop': 62}
✓ Loaded qrels for 182 queries

[Step 2/5] Running batch retrieval...
✓ Connected to Milvus
  Total documents: 491
Retrieving: 100%|████████████| 182/182 [05:23<00:00, 0.56query/s]
✓ Retrieval complete: 182 success, 0 failed

[Step 3/5] Calculating metrics...
✓ Calculated metrics for 182 queries

[Step 4/5] Aggregating results...
✓ Aggregated overall metrics
✓ Aggregated metrics for single_hop: 120 queries
✓ Aggregated metrics for multi_hop: 62 queries

[Step 5/5] Saving results...
✓ Saved retrieval results: results/retrieval_results.jsonl
✓ Saved overall metrics: results/metrics_overall.json
✓ Saved by-type metrics: results/metrics_by_type.json
✓ Saved by-k metrics: results/metrics_by_k.json
✓ Generated detailed report: results/detailed_report.txt

✓ Evaluation completed successfully!
Results saved to: results/
```

## Configuration

Edit `config.py` to customize:

```python
# Retrieval settings
collection_name = "elastic_embeddings_m3"
embedding_model = "BAAI/bge-m3"
enable_reranking = True

# K values for metrics
k_values = [1, 3, 5, 10, 20, 50, 100]

# Async settings
batch_size = 15          # Concurrent queries
max_concurrent = 15      # Max async operations
```

## Metrics Explained

### Binary Relevance Metrics

**Recall@K**: `relevant_retrieved@K / total_relevant`
- Measures coverage: What fraction of relevant docs are in top-K?

**Precision@K**: `relevant_retrieved@K / K`
- Measures accuracy: What fraction of top-K are relevant?

**MAP (Mean Average Precision)**:
- Average of precision at each relevant document position
- Rewards ranking relevant docs higher

**MRR (Mean Reciprocal Rank)**:
- `1 / rank_of_first_relevant`
- Rewards finding any relevant doc quickly

**Hits@K**: Binary indicator
- 1.0 if any relevant doc in top-K, else 0.0

### Graded Relevance Metric

**NDCG@K (Normalized Discounted Cumulative Gain)**:
- Uses graded relevance (0=irrelevant, 1=partial, 2=relevant, 3=highly)
- Formula: `DCG@K / IDCG@K`
- DCG: `Σ(rel_i / log2(i + 1))`
- Rewards highly relevant docs at top positions

## Output Files

### 1. `retrieval_results.jsonl`
Raw retrieval results for each query:
```json
{
  "query_id": "q0001",
  "query_text": "What was Q1 2024 revenue?",
  "query_type": "single_hop",
  "retrieved_docs": [
    {"chunk_id": "doc123", "score": 0.95, "rank": 1},
    ...
  ],
  "success": true
}
```

### 2. `metrics_overall.json`
Aggregated metrics across all queries:
```json
{
  "num_queries": 182,
  "MAP": 0.4523,
  "MRR": 0.6234,
  "recall@10": 0.7845,
  "precision@10": 0.6234,
  "ndcg@10": 0.8123,
  "hits@10": 0.9450
}
```

### 3. `metrics_by_type.json`
Breakdown by query type:
```json
{
  "single_hop": {
    "num_queries": 120,
    "MAP": 0.5123,
    "ndcg@10": 0.8456,
    ...
  },
  "multi_hop": {
    "num_queries": 62,
    "MAP": 0.3234,
    "ndcg@10": 0.7234,
    ...
  }
}
```

### 4. `metrics_by_k.json`
Metrics organized by K value (for plotting):
```json
{
  "k=1": {"k": 1, "recall": 0.23, "precision": 0.45, ...},
  "k=5": {"k": 5, "recall": 0.56, "precision": 0.32, ...},
  ...
}
```

### 5. `detailed_report.txt`
Human-readable report with:
- Summary statistics
- Query type breakdown
- Full metrics tables
- Top/worst performing queries
- Failed queries (if any)

## Performance

### Timing Estimates

| Queries | Sequential | Batch Async (15 concurrent) | Speedup |
|---------|------------|----------------------------|---------|
| 182     | ~30 min    | ~5-8 min                   | **4-6x** |
| 500     | ~80 min    | ~15-20 min                 | **4-5x** |

### Optimization Features

- **Async batch processing**: 15 concurrent retrieval requests
- **Connection reuse**: Single Milvus connection
- **Embedding cache**: Leverages existing retrieval system cache
- **Progress tracking**: tqdm progress bars
- **Error resilience**: Failed queries don't stop evaluation

## Integration with Existing System

This system **uses your complete RAG pipeline** via `retrieval/core.py`:

```python
from retrieval.core import RAGSystem

# Direct integration with full pipeline
rag_system = RAGSystem(
    embedding_model="BAAI/bge-m3",
    collection_name="elastic_embeddings_m3",
    enable_reranking=True,
    retrieval_multiplier=10,
    enable_query_decomposition=False,  # Optional
    max_sub_queries=5,
    fusion_k_constant=60,
    llm_type="mock",  # No LLM generation needed for evaluation
    enable_history=False  # Disabled for evaluation
)
```

### Why RAGSystem Integration?

The evaluation system now depends on `retrieval/core.py` instead of directly using `MilvusRetriever`. This design ensures:

1. **Automatic Pipeline Updates**: Any changes you make to the retrieval pipeline (query decomposition, re-ranking strategies, fusion algorithms) automatically flow into evaluation without requiring updates to evaluation code.

2. **Feature Parity**: Evaluations use the exact same retrieval logic as production, including:
   - Query decomposition (if enabled)
   - Multi-query retrieval with reciprocal rank fusion
   - Cross-encoder re-ranking (if enabled)
   - All retrieval optimizations and bug fixes

3. **Configuration Flexibility**: All RAG system parameters are configurable via `config.py`:
   ```python
   # Query decomposition (for complex queries)
   enable_query_decomposition = False
   max_sub_queries = 5
   fusion_k_constant = 60
   
   # Re-ranking (for better quality)
   enable_reranking = True
   retrieval_multiplier = 10
   
   # Context formatting
   max_context_tokens = 4000
   include_scores = False
   ```

4. **Single Source of Truth**: The `retrieval/` folder is the only place you need to modify retrieval behavior. The evaluation system will automatically pick up those changes.

### Retrieval Score Handling

The evaluation system intelligently handles both similarity and re-ranking scores:

```python
# Prefers rerank_score when available (from cross-encoder re-ranking)
# Falls back to similarity_score (from vector search) otherwise
score = chunk.rerank_score if chunk.rerank_score is not None else chunk.similarity_score
```

This ensures metrics reflect the actual ranking used in production.

**No modifications needed** to your retrieval code! Just edit `retrieval/core.py` or `retrieval/retrieval.py` and the evaluation system will use the updated logic.

## Troubleshooting

### Connection Issues
```
Error: Failed to connect to Milvus
```
**Solution**: Ensure Milvus is running and accessible:
```bash
docker ps | grep milvus
```

### Missing Input Files
```
Error: Queries file not found
```
**Solution**: Run synthetic evaluation generator first:
```bash
cd evals/synthetic-eval
python -m main
```

### Slow Performance
```
Taking >10 min for 182 queries
```
**Solution**: Increase batch size in `config.py`:
```python
batch_size = 20  # From 15
max_concurrent = 20
```

### Out of Memory
```
Error: MemoryError during retrieval
```
**Solution**: Decrease batch size:
```python
batch_size = 10  # From 15
max_concurrent = 10
```

## Advanced Usage

### CLI Arguments (NEW!)

The evaluation system now supports comprehensive CLI arguments for flexible experimentation:

```bash
# Show all options
python main.py --help

# Quick test with 10 queries
python main.py --num-queries 10

# Filter by query type
python main.py --query-type single_hop
python main.py --query-type multi_hop

# Custom K values
python main.py --k-values 1 5 10 20

# Disable re-ranking (faster)
python main.py --no-reranking

# High concurrency
python main.py --batch-size 25 --max-concurrent 25

# Custom output directory
python main.py --output-dir results_experiment_1

# Verbose logging
python main.py --verbose

# Dry run (validate without running)
python main.py --dry-run

# Pagination (process in batches)
python main.py --skip-queries 50 --num-queries 50

# Specific query IDs
python main.py --query-ids q0001 q0002 q0003
```

### Common Workflows

**A/B Testing Re-ranking:**
```bash
python main.py -o results_with_rerank
python main.py --no-reranking -o results_no_rerank
```

**Single vs Multi-hop Comparison:**
```bash
python main.py --query-type single_hop -o results_single
python main.py --query-type multi_hop -o results_multi
```

**Fast Iteration (Development):**
```bash
python main.py -n 20 --no-reranking -b 20 -mc 20
```

**Production Evaluation:**
```bash
python main.py --batch-size 10 --max-concurrent 10
```

### Custom K Values

Via CLI:
```bash
python main.py --k-values 1 5 10 20 50
```

Or edit `config.py`:
```python
k_values = [1, 5, 10, 20, 50]  # Focus on specific K values
```

### Query Filtering

Via CLI (recommended):
```bash
# First 50 queries only
python main.py --num-queries 50

# Skip first 100, evaluate next 50
python main.py --skip-queries 100 --num-queries 50

# Single-hop queries only
python main.py --query-type single_hop
```

Or modify `evaluator.py`:
```python
# Only evaluate single-hop queries
self.queries = [q for q in self.queries
                if q['metadata']['query_type'] == 'single_hop']
```

### Custom Metrics

Add to `metrics.py`:
```python
@staticmethod
def custom_metric(retrieved_ids, relevance_scores, k):
    # Your custom metric logic
    pass
```

## Citation

If you use this evaluation system, please cite:
```
Synthetic Evaluation Framework
Graph-RAG-Trials Project, 2024
```

