# Ragas Synthetic Test Data Generation

Generate synthetic test datasets from your Elasticsearch M3 collection using the [Ragas framework](https://github.com/explodinggradients/ragas) for comprehensive RAG evaluation.

## Overview

This implementation uses Ragas to automatically generate high-quality synthetic test data from your existing document corpus in Elasticsearch. Ragas uses knowledge graph construction and evolutionary generation to create diverse question-answer pairs that test various aspects of your RAG system.

### Key Features

- **Knowledge Graph Approach**: Ragas constructs knowledge graphs from your documents and generates questions through transformations
- **Diverse Query Types**: Automatically generates simple, reasoning, multi-context, and conditional queries
- **Standard Framework**: Industry-standard evaluation tool with active development
- **Elasticsearch Integration**: Direct integration with your `rudram-embeddings` index
- **Comprehensive Metrics**: Built-in evaluation metrics for faithfulness, relevancy, precision, and more

## Architecture

```
evals/ragas/
├── config.py                    # Configuration for Elasticsearch, LLM, and generation params
├── elasticsearch_loader.py      # Load documents from Elastic M3 collection
├── generate_testset.py          # Main testset generation script
├── evaluate_rag.py             # Run evaluation with generated testset
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── output/                     # Generated datasets and reports
    ├── testset.csv
    ├── testset.json
    ├── generation_report.txt
    └── generation_stats.json
```

## Setup

### 1. Install Dependencies

```bash
cd evals/ragas
pip install -r requirements.txt
```

### 2. Configure Environment

Set your OpenAI API key (required for LLM-based generation):

```bash
export OPENAI_API_KEY="your-api-key-here"
```

For Azure OpenAI:
```bash
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="your-azure-endpoint"
```

### 3. Configure Settings

Edit `config.py` to customize:

- **Elasticsearch Configuration**: Connection details (pre-configured)
- **Testset Size**: Number of samples to generate (default: 100)
- **LLM Models**: Generator and critic models (default: gpt-4o-mini)
- **Query Distributions**: Mix of simple/reasoning/multi-context queries
- **Sampling Strategy**: How to sample documents from Elasticsearch

## Usage

### Generate Synthetic Testset

**Basic usage** (uses defaults from `config.py`):
```bash
python generate_testset.py
```

**Custom testset size**:
```bash
python generate_testset.py --testset-size 200
```

**Limit source documents**:
```bash
python generate_testset.py --max-documents 1000
```

**Representative sampling** (diverse document coverage):
```bash
python generate_testset.py --sample-strategy representative
```

**Full example**:
```bash
python generate_testset.py \
  --testset-size 150 \
  --max-documents 500 \
  --sample-strategy random \
  --output-dir output
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--testset-size` | Number of test samples to generate | 100 |
| `--max-documents` | Maximum documents to load from Elasticsearch | 500 |
| `--sample-strategy` | Sampling strategy: `random` or `representative` | random |
| `--output-dir` | Output directory for results | output |

## Output Files

After generation, you'll find:

### 1. `testset.csv` / `testset.json`
Main testset with questions, answers, and contexts:
```csv
question,ground_truth,contexts,evolution_type
"What is the revenue...","The revenue was...",['context1','context2'],simple
```

### 2. `generation_report.txt`
Human-readable report with:
- Generation metadata (time, models used)
- Dataset statistics
- Question/answer length statistics
- Sample questions preview

### 3. `generation_stats.json`
Detailed statistics in JSON format for programmatic analysis

## Query Types Generated

Ragas generates different evolution types:

### Simple (Single-hop)
Direct questions answerable from a single context:
```
Q: What is the company's revenue for Q4 2024?
```

### Reasoning (Multi-hop)
Questions requiring reasoning across multiple facts:
```
Q: How did the company's profit margin change between Q3 and Q4 2024?
```

### Multi-context
Questions requiring information from multiple documents:
```
Q: Compare the cloud services performance across AWS, Azure, and GCP in 2024.
```

### Conditional
Complex queries with conditional logic:
```
Q: If the revenue growth continues at the current rate, what would be the projected revenue for 2025?
```

## Configuration

### Elasticsearch Connection

Pre-configured in `config.py`:
```python
ELASTICSEARCH_CONFIG = {
    "url": "https://1600c6e333fd4bdb8c8e9b9dec5c5fef.us-west-2.aws.found.io:443",
    "username": "elastic",
    "password": "XI6rIccvUKLCgVnX11QPI8CV",
    "index_name": "rudram-embeddings",
}
```

### Generation Parameters

Customize in `config.py`:
```python
RAGAS_CONFIG = {
    "testset_size": 100,
    "generator_model": "gpt-4o-mini",
    "critic_model": "gpt-4o-mini",
    "embeddings_model": "text-embedding-3-small",
    
    # Distribution of query types (must sum to 1.0)
    "distributions": {
        "simple": 0.4,
        "reasoning": 0.3,
        "multi_context": 0.2,
        "conditional": 0.1,
    },
    
    "max_documents": 500,
    "sample_strategy": "random",
}
```

### LLM Provider Options

**OpenAI** (default):
```python
RAGAS_CONFIG["llm_provider"] = "openai"
```

**Azure OpenAI**:
```python
RAGAS_CONFIG["llm_provider"] = "azure"
AZURE_CONFIG = {
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "deployment_name": "gpt-4",
}
```

## Evaluation

### Evaluate Your RAG System

Once you have a generated testset, evaluate your RAG system:

```bash
python evaluate_rag.py --testset output/testset.csv
```

### Integration Example

```python
from evaluate_rag import RagasEvaluator
import pandas as pd

# Load testset
evaluator = RagasEvaluator()
testset_df = evaluator.load_testset("output/testset.csv")

# Generate answers with your RAG system
rag_responses = []
for _, row in testset_df.iterrows():
    question = row["question"]
    
    # YOUR RAG SYSTEM HERE
    answer, contexts = your_rag_system.query(question)
    
    rag_responses.append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
    })

# Prepare and evaluate
eval_dataset = evaluator.prepare_evaluation_dataset(testset_df, rag_responses)
results = evaluator.evaluate(eval_dataset)

# Save results
evaluator.save_results(results, "output/evaluation")
evaluator.generate_report(results, "output/evaluation")
```

### Evaluation Metrics

Ragas provides comprehensive metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| **Faithfulness** | Factual accuracy based on retrieved contexts | 0-1 |
| **Answer Relevancy** | How relevant the answer is to the question | 0-1 |
| **Context Recall** | How much ground truth is in retrieved contexts | 0-1 |
| **Context Precision** | Precision of retrieved contexts | 0-1 |
| **Context Relevancy** | Relevance of contexts to question | 0-1 |
| **Answer Similarity** | Semantic similarity to ground truth | 0-1 |
| **Answer Correctness** | Overall correctness (F1 score) | 0-1 |

## Comparison with Existing `synthetic-eval`

### Advantages of Ragas

| Aspect | synthetic-eval (Custom) | Ragas (This Implementation) |
|--------|------------------------|----------------------------|
| **Code Complexity** | ~2000 lines custom code | ~500 lines wrapper code |
| **Query Diversity** | LLM-based with templates | Evolutionary generation + KG |
| **Framework** | Custom implementation | Standard industry framework |
| **Maintenance** | Manual updates needed | Framework updates automatic |
| **Metrics** | Custom metrics | 7+ built-in standard metrics |
| **Community** | Internal only | Active open-source community |

### When to Use Each

**Use `synthetic-eval`** when:
- You need full control over generation logic
- You have specific domain requirements
- You want to understand every detail of the process

**Use Ragas** when:
- You want industry-standard evaluation
- You need diverse query types automatically
- You want to compare with other RAG systems
- You prefer battle-tested frameworks

## Advanced Usage

### Custom Document Sampling

Load specific documents:
```python
from elasticsearch_loader import ElasticsearchDocumentLoader

loader = ElasticsearchDocumentLoader()

# Custom query
custom_query = {
    "bool": {
        "must": [
            {"match": {"category": "financial-reports"}},
            {"range": {"date": {"gte": "2024-01-01"}}}
        ]
    }
}

documents = loader.load_documents(
    max_documents=200,
    query=custom_query
)
```

### Representative Sampling

Get diverse documents across categories:
```python
documents = loader.load_representative_sample(
    max_documents=500,
    metadata_field="source"  # Stratify by source
)
```

### Custom Distributions

Adjust query type mix in `config.py`:
```python
# More complex queries
"distributions": {
    "simple": 0.2,
    "reasoning": 0.4,
    "multi_context": 0.3,
    "conditional": 0.1,
}

# Simpler queries
"distributions": {
    "simple": 0.7,
    "reasoning": 0.2,
    "multi_context": 0.1,
    "conditional": 0.0,
}
```

## Testing Document Loader

Test Elasticsearch connection and document loading:

```bash
python elasticsearch_loader.py
```

This will:
- Connect to Elasticsearch
- Display index statistics
- Load 10 sample documents
- Show document preview

## Troubleshooting

### Connection Issues

**Error**: `Failed to connect to Elasticsearch`

**Solution**: Check your network and credentials:
```python
# Test connection
from elasticsearch import Elasticsearch

client = Elasticsearch(
    "https://1600c6e333fd4bdb8c8e9b9dec5c5fef.us-west-2.aws.found.io:443",
    basic_auth=("elastic", "XI6rIccvUKLCgVnX11QPI8CV"),
    verify_certs=True,
)

print(client.ping())  # Should return True
```

### LLM Rate Limits

**Error**: `Rate limit exceeded`

**Solution**: Reduce batch size or add retry logic:
```python
OPENAI_CONFIG = {
    "timeout": 120,  # Increase timeout
    "max_retries": 5,  # More retries
}
```

Or generate in smaller batches:
```bash
python generate_testset.py --testset-size 50
```

### Out of Memory

**Error**: `MemoryError` or OOM

**Solution**: Reduce documents loaded:
```bash
python generate_testset.py --max-documents 100
```

### Empty Documents

**Error**: `No documents loaded`

**Solution**: Check your index has data:
```python
from elasticsearch_loader import ElasticsearchDocumentLoader

loader = ElasticsearchDocumentLoader()
stats = loader.get_index_stats()
print(f"Document count: {stats['document_count']}")
```

## Best Practices

### 1. Start Small
Begin with small testset for validation:
```bash
python generate_testset.py --testset-size 20 --max-documents 50
```

### 2. Validate Quality
Review generated questions before scaling:
- Check `generation_report.txt` for sample questions
- Verify ground truth answers are reasonable
- Ensure contexts are relevant

### 3. Iterate on Distribution
Adjust query distributions based on your RAG use case:
- Customer support: More simple queries
- Research: More reasoning/multi-context queries
- Analysis: More conditional queries

### 4. Representative Sampling
Use representative sampling for better coverage:
```bash
python generate_testset.py --sample-strategy representative
```

### 5. Version Control
Track testsets in version control:
```bash
git add evals/ragas/output/testset_v1.csv
git commit -m "Add Ragas testset v1 (100 samples)"
```

## Next Steps

After generating your testset:

1. **Validate Quality**: Review sample questions in the report
2. **Compare Approaches**: Compare with `synthetic-eval` outputs
3. **Evaluate Your RAG**: Use testset to evaluate your RAG system
4. **Iterate**: Adjust parameters based on quality and results
5. **Scale Up**: Generate larger testsets once validated

## Resources

- [Ragas Documentation](https://docs.ragas.io/)
- [Ragas GitHub](https://github.com/explodinggradients/ragas)
- [RAG Evaluation Best Practices](https://docs.ragas.io/en/latest/concepts/metrics/)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Ragas documentation
3. Check existing `synthetic-eval` implementation for reference
4. Open an issue in your project repository

