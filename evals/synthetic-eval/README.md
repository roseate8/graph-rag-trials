

# Synthetic Evaluation Dataset Generator

Generates high-quality Q&A pairs with graded relevance labels from document chunks using LLM-powered fact extraction, stratified sampling, and multi-hop query generation.

## Overview

This system creates **300-800 evaluation questions** from your ~15K document chunks in the Milvus collection (`elastic_embeddings_m3`) with:

- **Stratified sampling**: Topic-based clustering ensures diverse coverage
- **Atomic fact extraction**: Regex + LLM extracts answerable units (numbers, dates, triples, key-values)
- **Query diversity**: Single-hop (3-5 paraphrases) + multi-hop (cross-chunk reasoning)
- **Graded relevance**: 0-3 scale with heuristic + LLM judging
- **BEIR format**: Standard format for retrieval evaluation

## Architecture

### Pipeline Stages

```
1. Chunk Sampling (stratified by topic clusters) → ~400 chunks
2. Atomic Fact Extraction (regex + LLM) → facts with answer spans
3. Query Generation (single-hop + multi-hop) → 3-5 queries per fact
4. Silver Label Assignment (graded 0-3) → qrels with relevance scores
5. Output (BEIR format) → queries.jsonl, qrels.tsv, corpus.jsonl
```

### Technology Stack

- **LLM**: gpt-4o-mini via secure API key management (`llm_utils.py`)
- **Embeddings**: BAAI/bge-m3 for clustering
- **Clustering**: scikit-learn K-means
- **Vector DB**: Read-only access to `elastic_embeddings_m3` collection

## Installation

```bash
# Install additional dependencies
cd evals/synthetic-eval
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to customize parameters:

```python
class SyntheticEvalConfig:
    # Sampling
    target_sample_size: int = 400  # Chunks to sample
    num_clusters: int = 20         # Topic clusters
    
    # Query generation
    target_questions: int = 400     # Total questions (300-800)
    queries_per_fact_min: int = 3  # Min queries per fact
    queries_per_fact_max: int = 5  # Max queries per fact
    multi_hop_ratio: float = 0.2   # 20% multi-hop
    
    # LLM
    model_name: str = "gpt-4o-mini"
    
    # Silver labeling thresholds
    exact_match_threshold: float = 0.9   # rel=3
    token_f1_high: float = 0.7           # rel=3
    token_f1_mid: float = 0.4            # rel=2
    semantic_similarity_threshold: float = 0.75  # rel=2
```

## Usage

### Basic Usage

```bash
# Run from project root
python -m evals.synthetic-eval.main
```

You'll be prompted for your OpenAI API key (stored temporarily, never saved to disk).

### Output

All files are written to `evals/synthetic-eval/output/`:

```
output/
├── queries.jsonl              # Generated questions
├── qrels.tsv                  # Graded relevance (0-3)
├── corpus.jsonl               # All chunks
├── generation_stats.json      # Detailed statistics
└── generation_report.txt      # Human-readable summary
```

### Output Formats

**queries.jsonl**:
```json
{
  "_id": "q0001",
  "text": "What was the fiscal year 2024 revenue?",
  "metadata": {
    "answer": "$1.2B",
    "gold_chunk_ids": ["chunk_123"],
    "query_type": "single_hop",
    "question_style": "wh_question"
  }
}
```

**qrels.tsv**:
```
query-id    corpus-id    score
q0001       chunk_123    3
q0001       chunk_124    2
q0001       chunk_125    1
```

**corpus.jsonl**:
```json
{
  "_id": "chunk_123",
  "title": "Financial Results > Revenue",
  "text": "Elastic N.V. reported fiscal year 2024 revenue of $1.2B...",
  "metadata": {
    "doc_id": "annual-report-2024",
    "word_count": 250
  }
}
```

## Relevance Scale

- **3**: Contains exact answer or clearly paraphrased answer
- **2**: Provides supporting context that helps answer the query
- **1**: Same topic/domain but minimal relevance
- **0**: Not relevant

## Features

### Fact Extraction

**Regex patterns** extract:
- Dates (YYYY-MM-DD, Month DD YYYY, Q1 2024, FY2024)
- Numbers (with K/M/B/T suffixes)
- Currency amounts ($, €, £, ¥)

**LLM extraction** identifies:
- Subject-relation-object triples
- Key-value pairs from tables
- Named entities for multi-hop linking

### Query Generation

**Single-hop queries** (3-5 per fact):
- Who/what/when/where/why/how questions
- Cloze-style ("The FY2023 EBITDA was ___")
- Keyword-based (simpler, search-like)
- Paraphrased variations

**Multi-hop queries**:
- Automatically link facts sharing entities across chunks
- Generate comparison/reasoning questions
- Example: "What changed in warranty terms between 2023 and 2024?"

### Silver Labeling

**Heuristic-based** (fast):
- Exact string match (normalized)
- Token-F1 score
- Semantic similarity (embedding-based)
- Document co-occurrence

**LLM judge** (for ambiguous cases):
- Invoked when token-F1 is in the ambiguous range (0.3-0.4)
- Grades 0-3 with strict rubric

## Advanced Options

### Enable Retrieval Validation

Test that gold chunks are retrievable:

```python
# In config.py
validate_retrieval: bool = True
validation_top_k: int = 10
```

This runs each query through the retrieval system and reports recall@10.

### Save Intermediate Files

```python
# In config.py
save_intermediate: bool = True
```

Saves:
- `intermediate_facts.jsonl`: All extracted facts
- `intermediate_queries.jsonl`: All generated queries (before labeling)

### Adjust Sampling Strategy

```python
# More/fewer clusters for finer/coarser topic granularity
num_clusters: int = 30

# Sample more chunks for broader coverage
target_sample_size: int = 600
```

## File Structure

```
evals/synthetic-eval/
├── __init__.py              # Package init
├── main.py                  # Main orchestrator
├── config.py                # Configuration
├── chunk_sampler.py         # Stratified sampling
├── fact_extractor.py        # Atomic fact mining
├── query_generator.py       # Query generation
├── silver_labeler.py        # Relevance labeling
├── output_formatter.py      # BEIR format writer
├── utils.py                 # Helper functions
├── requirements.txt         # Dependencies
├── README.md                # This file
└── output/                  # Generated files
```

## Dependencies

### Project Dependencies (Already Installed)

- `retrieval/`: MilvusRetriever for vector DB access
- `vector-ingest/src/chunking/processors/llm_utils.py`: Secure API key management
- `vector-ingest/src/embeddings/`: Embedding service

### New Dependencies

- `scikit-learn`: K-means clustering
- `numpy`: Numerical operations
- `openai`: LLM API

## Examples

### Example Facts Extracted

```json
{
  "fact_id": "chunk_123_fact_1",
  "fact_type": "currency",
  "fact_text": "Elastic N.V. reported fiscal year 2024 revenue of $1.2B",
  "answer_span": "$1.2B",
  "entities": ["Elastic N.V.", "fiscal year 2024", "$1.2B"]
}
```

### Example Single-Hop Queries

```json
[
  {
    "question": "What was Elastic N.V.'s revenue in fiscal year 2024?",
    "style": "wh_question"
  },
  {
    "question": "The fiscal year 2024 revenue for Elastic N.V. was ___",
    "style": "cloze"
  },
  {
    "question": "elastic fiscal 2024 revenue amount",
    "style": "keyword"
  }
]
```

### Example Multi-Hop Query

```json
{
  "question": "What was the change in Elastic's revenue between fiscal year 2023 and 2024?",
  "answer": "Increased from $1.0B to $1.2B",
  "gold_chunk_ids": ["chunk_100", "chunk_123"],
  "reasoning": "Requires data from both FY2023 and FY2024 chunks"
}
```

## Performance Considerations

### LLM API Costs

Estimated costs for 400 queries (gpt-4o-mini):
- **Fact extraction**: ~400 chunks × $0.0003 = $0.12
- **Query generation**: ~400 facts × $0.0005 = $0.20
- **Multi-hop generation**: ~80 pairs × $0.0005 = $0.04
- **Silver labeling (LLM judge)**: ~50 ambiguous × $0.0002 = $0.01
- **Total**: ~$0.37

### Runtime

- **Sampling**: 1-2 minutes (K-means clustering)
- **Fact extraction**: 10-15 minutes (400 chunks, batched LLM calls)
- **Query generation**: 15-20 minutes (single-hop + multi-hop)
- **Silver labeling**: 5-10 minutes (mostly heuristic-based)
- **Total**: ~30-50 minutes

### Optimization Tips

1. **Reduce sample size**: Start with 100 chunks for testing
2. **Limit fact extraction**: Skip LLM extraction for tables if not needed
3. **Disable LLM judge**: Set `enable_llm_judge: False` for faster labeling
4. **Batch processing**: Increase `batch_size` for more concurrent LLM calls

## Troubleshooting

### Connection Errors

```
Error: Failed to connect to Milvus
```

**Solution**: Ensure Milvus is running and `elastic_embeddings_m3` collection exists.

### API Key Issues

```
Error: OpenAI API key invalid
```

**Solution**: Ensure your API key starts with `sk-` or `sk-proj-` and has sufficient credits.

### Low Recall

```
Warning: Low recall detected! Average Recall@10: 0.35
```

**Solution**: 
- Increase `validation_top_k` to test with more results
- Review query generation (queries may be too specific)
- Check if chunks are properly embedded

### Memory Issues

```
Error: Out of memory during clustering
```

**Solution**:
- Reduce `num_clusters` 
- Reduce `target_sample_size`
- Process in smaller batches

## Citation

If you use this dataset generator in your research, please cite:

```bibtex
@software{synthetic_eval_generator,
  title={Synthetic Evaluation Dataset Generator},
  year={2025},
  description={LLM-powered Q&A generation with graded relevance labels}
}
```

## License

This project follows the same license as the parent repository.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review generation logs in console output
3. Examine intermediate files (if `save_intermediate: True`)
4. Check `generation_report.txt` for statistics

---

**Happy Evaluating! 🚀**

