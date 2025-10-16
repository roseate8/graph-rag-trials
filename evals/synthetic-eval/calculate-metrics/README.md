# Calculate Metrics - Synthetic Evaluation

Evaluation system for information retrieval metrics using batch async processing.

## Quick Start

```bash
cd evals/synthetic-eval/calculate-metrics
python main.py
```

## Architecture

**100% Dependent on `retrieval/core.py`** - Any changes to retrieval automatically flow to evaluation.

```
main.py → evaluator.py → retriever_for_evals.py → retrieval/core.py (RAGSystem)
```

## CLI Usage

```bash
# Basic
python main.py

# Quick test (10 queries)
python main.py --num-queries 10

# Custom K values
python main.py --k-values 1 3 5 10

# Disable re-ranking
python main.py --no-reranking

# Filter by type
python main.py --query-type single_hop
python main.py --query-type multi_hop

# High concurrency
python main.py --batch-size 20 --max-concurrent 20

# Dry run (validate)
python main.py --dry-run

# All options
python main.py --help
```

## Configuration

**Default Settings** (in `config.py`):
- Collection: `elastic_embeddings_m3`
- Embedding: `BAAI/bge-m3`
- Re-ranking: Enabled
- K values: `[1, 3, 5, 10, 20, 50]`
- Batch size: 15 concurrent queries
- CUDA: Single-threaded executor (thread-safe)

**Config Architecture** (NEW):
```python
# All RAGSystem params in one dict - pass-through pattern
rag_system_params = {
    'collection_name': 'elastic_embeddings_m3',
    'embedding_model': 'BAAI/bge-m3',
    'enable_reranking': True,
    'enable_query_decomposition': False,
    ...
}
# Any new RAGSystem params automatically work!
```

## Metrics

**Binary Relevance:**
- **Recall@K**: Coverage - what fraction of relevant docs in top-K
- **Precision@K**: Accuracy - what fraction of top-K are relevant
- **MAP**: Mean Average Precision - rewards ranking relevant docs higher
- **MRR**: Mean Reciprocal Rank - rewards finding relevant doc quickly
- **Hits@K**: Binary - any relevant doc in top-K

**Graded Relevance:**
- **NDCG@K**: Normalized DCG with 0-3 relevance scale

## Output Files

```
results/
├── retrieval_results.jsonl  # Raw results per query
├── metrics_overall.json     # Aggregated across all queries
├── metrics_by_type.json     # Breakdown by query type
├── metrics_by_k.json        # Organized by K value
├── detailed_report.txt      # Human-readable report
└── failed_queries.jsonl     # Failed queries (if any)
```

## Performance

- **182 queries**: ~5-8 minutes (batch async)
- **Threading**: Single-threaded executor for CUDA thread-safety
- **Re-ranking**: Batched operations (500 chunks at once)

## Key Features

✅ **100% dependency on retrieval/core.py** - automatic feature updates
✅ **Thread-safe CUDA** - single-threaded executor prevents meta tensor errors
✅ **Async batch processing** - 15 concurrent queries
✅ **Graded relevance** - 0-3 scale for NDCG
✅ **CLI flexibility** - 18+ command-line options
✅ **No caching** - fresh calculations every time
✅ **Optimized code** - O(n) algorithms, generator expressions

## Troubleshooting

**CUDA errors**: Fixed with single-threaded executor (automatic)
**Slow performance**: Increase `--batch-size` and `--max-concurrent`
**Out of memory**: Decrease `--batch-size`
**Connection issues**: Ensure Milvus is running
**Missing files**: Run `python -m main` in `evals/synthetic-eval` first

## Documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for complete details on:
- Architecture & design decisions
- CUDA thread-safety fix explanation
- Performance refactoring details
- Time complexity analysis
