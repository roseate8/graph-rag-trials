# Ragas vs Custom Synthetic-Eval Comparison

A detailed comparison between the Ragas-based approach (this implementation) and the existing custom `synthetic-eval` implementation.

## Executive Summary

| Aspect | Custom synthetic-eval | Ragas (This) |
|--------|----------------------|--------------|
| **Setup Time** | Complex (custom code) | Simple (framework) |
| **Code Complexity** | ~2000 lines | ~500 lines |
| **Maintenance** | Manual | Framework-maintained |
| **Query Diversity** | Template-based | Knowledge graph + evolution |
| **Metrics** | Custom | 7+ standard metrics |
| **Community Support** | Internal only | Active OSS community |
| **Learning Curve** | Steep | Moderate |
| **Flexibility** | Maximum | High |

**Recommendation**: Use Ragas for standard RAG evaluation, keep custom synthetic-eval for specific domain requirements.

## Detailed Comparison

### 1. Architecture

#### Custom Synthetic-Eval
```
evals/synthetic-eval/
├── chunk_sampler.py          # Custom sampling logic
├── fact_extractor.py          # LLM-based fact extraction
├── query_generator.py         # Template-based generation
├── silver_labeler.py          # Relevance labeling
├── calculate-metrics/         # Custom metric calculation
│   ├── evaluator.py
│   ├── metrics.py
│   └── retriever_for_evals.py
└── ~2000 lines total code
```

**Pros**:
- Full control over every step
- Domain-specific customization
- Custom metrics for specific needs

**Cons**:
- High maintenance burden
- Requires deep understanding
- Manual updates for improvements

#### Ragas
```
evals/ragas/
├── config.py                  # Configuration
├── elasticsearch_loader.py    # Document loading
├── generate_testset.py        # Generation wrapper
├── evaluate_rag.py           # Evaluation wrapper
└── ~500 lines wrapper code
```

**Pros**:
- Battle-tested framework
- Automatic updates/improvements
- Standard industry metrics
- Lower maintenance

**Cons**:
- Less control over internals
- Framework dependencies
- May not fit all edge cases

### 2. Generation Process

#### Custom Synthetic-Eval Process

1. **Chunk Sampling**: Load 400 chunks from corpus
2. **Fact Extraction**: Extract facts using LLM
3. **Query Generation**: Generate queries from facts
4. **Silver Labeling**: Re-rank and label relevant docs
5. **Metrics Calculation**: Custom retrieval metrics

```python
# Multi-step process
chunks = sample_chunks(corpus, n=400)
facts = extract_facts(chunks)  # LLM call
queries = generate_queries(facts)  # LLM call
labels = silver_label(queries, corpus)  # Re-ranking
metrics = calculate_metrics(queries, labels)  # Custom
```

**Query Generation**:
- Template-based with LLM
- Focused on retrieval evaluation
- Explicit fact → query mapping

#### Ragas Process

1. **Document Loading**: Load documents from Elasticsearch
2. **KG Construction**: Build knowledge graph from docs
3. **Evolution**: Apply transformations for diverse queries
4. **Generation**: Create question-answer-context triplets
5. **Evaluation**: Standard metrics suite

```python
# Streamlined process
documents = load_documents(elasticsearch)
testset = generator.generate(documents)  # KG + evolution
results = evaluate(testset, rag_system)  # Standard metrics
```

**Query Generation**:
- Knowledge graph-based
- Evolutionary transformations
- Multiple complexity levels

### 3. Query Types

#### Custom Synthetic-Eval

Primarily generates **retrieval-focused queries**:

```
Examples:
- "What is X?"
- "Describe Y"
- "How does Z work?"
```

**Characteristics**:
- Single-hop queries
- Fact-based
- Direct retrieval testing
- Good for chunk-level evaluation

#### Ragas

Generates **diverse query types**:

**Simple** (40%):
```
"What was the revenue in Q4 2024?"
```

**Reasoning** (30%):
```
"How did profit margins change from Q3 to Q4 2024, and what factors contributed?"
```

**Multi-context** (20%):
```
"Compare cloud service performance across AWS, Azure, and GCP based on 2024 reports."
```

**Conditional** (10%):
```
"If the current growth rate continues, what would be the projected revenue for 2025?"
```

**Characteristics**:
- Multi-hop reasoning
- Cross-document queries
- Conditional logic
- Better end-to-end testing

### 4. Metrics

#### Custom Synthetic-Eval Metrics

**Retrieval Metrics**:
- Precision@k
- Recall@k
- MRR (Mean Reciprocal Rank)
- NDCG (Normalized Discounted Cumulative Gain)

**Focus**: Document retrieval quality

**Implementation**: Custom, ~300 lines

**Example Output**:
```json
{
  "precision@1": 0.85,
  "recall@10": 0.92,
  "mrr": 0.78,
  "ndcg@10": 0.83
}
```

#### Ragas Metrics

**RAG-Specific Metrics**:
- **Faithfulness**: Answer factuality (0-1)
- **Answer Relevancy**: Question-answer relevance (0-1)
- **Context Recall**: Ground truth coverage (0-1)
- **Context Precision**: Relevant context ratio (0-1)
- **Context Relevancy**: Context-question relevance (0-1)
- **Answer Similarity**: Semantic similarity (0-1)
- **Answer Correctness**: F1 score (0-1)

**Focus**: End-to-end RAG quality

**Implementation**: Framework-provided

**Example Output**:
```json
{
  "faithfulness": 0.89,
  "answer_relevancy": 0.92,
  "context_recall": 0.87,
  "context_precision": 0.85,
  "context_relevancy": 0.90,
  "answer_similarity": 0.88,
  "answer_correctness": 0.86
}
```

### 5. Data Flow

#### Custom Synthetic-Eval

```
Source Corpus (JSONL)
    ↓
Chunk Sampling (400 chunks)
    ↓
Fact Extraction (LLM)
    ↓
Query Generation (LLM)
    ↓
Silver Labeling (Re-ranking)
    ↓
Evaluation (Retrieval metrics)
    ↓
Results (Custom format)
```

#### Ragas

```
Elasticsearch Collection
    ↓
Document Loading (500 docs)
    ↓
Knowledge Graph Construction
    ↓
Evolutionary Generation
    ↓
Testset (Q-A-Context triplets)
    ↓
RAG System Integration
    ↓
Evaluation (RAG metrics)
    ↓
Results (Standard format)
```

### 6. Use Cases

#### When to Use Custom Synthetic-Eval

✅ **Best for**:
- Pure retrieval system evaluation
- Chunk-level performance testing
- Domain-specific requirements
- Full control needed
- Custom metric requirements
- Understanding internals is priority

❌ **Not ideal for**:
- End-to-end RAG evaluation
- Diverse query types
- Quick iteration
- Standard benchmarking

#### When to Use Ragas

✅ **Best for**:
- End-to-end RAG evaluation
- Diverse query complexity
- Standard benchmarking
- Quick setup and iteration
- Industry comparison
- Production RAG systems

❌ **Not ideal for**:
- Pure retrieval evaluation
- Highly specialized domains
- Custom metric requirements
- Learning system internals

### 7. Performance

#### Custom Synthetic-Eval

**Generation Time** (400 chunks → queries):
- ~30-60 minutes
- Multiple LLM calls per query
- Silver labeling overhead

**Cost** (estimated):
- ~$2-5 per 100 queries
- Depends on fact extraction depth

**Scalability**:
- Linear with chunk count
- Manual optimization needed

#### Ragas

**Generation Time** (500 docs → 100 queries):
- ~5-10 minutes
- Optimized LLM calls
- Efficient KG construction

**Cost** (estimated):
- ~$0.50-1.00 per 100 queries
- gpt-4o-mini is efficient

**Scalability**:
- Well-optimized
- Framework handles batching

### 8. Output Format

#### Custom Synthetic-Eval

```json
{
  "query_id": "q_001",
  "query": "What is X?",
  "relevant_chunks": ["chunk_123", "chunk_456"],
  "retrieved_chunks": ["chunk_123", "chunk_789"],
  "metrics": {
    "precision@1": 1.0,
    "recall@10": 0.5
  }
}
```

**Format**: Custom JSONL
**Focus**: Retrieval results

#### Ragas

```json
{
  "question": "What is X?",
  "ground_truth": "X is defined as...",
  "contexts": ["Context 1...", "Context 2..."],
  "evolution_type": "simple",
  "answer": "Generated answer...",
  "metrics": {
    "faithfulness": 0.89,
    "answer_relevancy": 0.92
  }
}
```

**Format**: Standard CSV/JSON
**Focus**: Question-Answer pairs with context

### 9. Integration

#### Custom Synthetic-Eval Integration

```python
# Load custom eval results
from evals.synthetic_eval.calculate_metrics import evaluator

results = evaluator.evaluate(
    queries_file="queries.jsonl",
    corpus_file="corpus.jsonl",
    qrels_file="qrels.tsv"
)
```

**Integration Points**:
- Custom retriever needed
- Specific format required
- Manual metric calculation

#### Ragas Integration

```python
# Standard integration
from ragas import evaluate

testset = pd.read_csv("testset.csv")
results = evaluate(
    dataset=testset,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings
)
```

**Integration Points**:
- Standard format
- Framework handles complexity
- Easy pipeline integration

### 10. Maintenance

#### Custom Synthetic-Eval

**Maintenance Tasks**:
- Update LLM prompts manually
- Fix generation issues
- Optimize performance
- Update metrics logic
- Handle edge cases
- Document changes

**Estimated Effort**: 4-8 hours/month

#### Ragas

**Maintenance Tasks**:
- Update framework version
- Adjust configuration
- Monitor deprecations

**Estimated Effort**: 1-2 hours/month

## Side-by-Side Example

### Same Task: Generate 100 Test Queries

#### Custom Synthetic-Eval
```bash
cd evals/synthetic-eval
python chunk_sampler.py --n-chunks 400
python fact_extractor.py --input corpus.jsonl
python query_generator.py --facts intermediate_facts.jsonl
python silver_labeler.py --queries intermediate_queries.jsonl
cd calculate-metrics
python main.py --queries ../queries.jsonl
```

**Steps**: 5 separate commands
**Time**: ~45 minutes
**Code**: Custom implementation

#### Ragas
```bash
cd evals/ragas
python generate_testset.py --testset-size 100
```

**Steps**: 1 command
**Time**: ~7 minutes
**Code**: Framework wrapper

## Recommendations

### Use Custom Synthetic-Eval When:

1. **Pure Retrieval Focus**
   - Testing chunk retrieval specifically
   - Optimizing embedding models
   - Comparing retrieval strategies

2. **Custom Requirements**
   - Specialized domain needs
   - Unique metric requirements
   - Custom evaluation pipeline

3. **Learning Goal**
   - Understanding eval internals
   - Building expertise
   - Research purposes

### Use Ragas When:

1. **End-to-End RAG**
   - Testing complete RAG pipeline
   - Answer quality evaluation
   - Production system validation

2. **Standard Benchmarking**
   - Industry comparison
   - Best practices
   - Reproducible results

3. **Quick Iteration**
   - Rapid experimentation
   - MVP/proof-of-concept
   - Time constraints

### Hybrid Approach

**Best of both worlds**:

1. Use **Ragas** for end-to-end RAG evaluation
2. Use **Custom Synthetic-Eval** for retrieval optimization
3. Compare results to identify bottlenecks

```bash
# Generate testsets with both
cd evals/ragas
python generate_testset.py --testset-size 100

cd ../synthetic-eval
python main.py

# Compare retrieval vs RAG metrics
python compare_results.py
```

## Migration Path

### From Custom → Ragas

If you want to adopt Ragas:

1. **Start Small**: Generate 20 samples, compare quality
2. **Run Parallel**: Keep custom eval, add Ragas
3. **Validate**: Ensure Ragas meets requirements
4. **Transition**: Gradually shift to Ragas for new work
5. **Maintain**: Keep custom eval for specialized needs

### From Ragas → Custom

If you need more control:

1. **Identify Gaps**: What Ragas doesn't provide
2. **Extend Ragas**: Try customizing framework first
3. **Build Custom**: Only if extension isn't feasible
4. **Learn**: Study custom eval implementation
5. **Implement**: Build what you need

## Conclusion

Both approaches have merit:

- **Custom Synthetic-Eval**: Deep control, retrieval focus
- **Ragas**: Standard framework, end-to-end RAG

**Our recommendation**: 
- **Start with Ragas** for most use cases
- **Fall back to Custom** for specialized needs
- **Use both** for comprehensive evaluation

The Ragas implementation provides 80% of what you need with 20% of the effort. Use the custom implementation when that 20% matters for your specific use case.

