# Quick Start Guide

Get started with Ragas synthetic test data generation in 5 minutes.

## Prerequisites

- Python 3.8+
- OpenAI API key
- Access to Elasticsearch cluster (pre-configured)

## Installation

```bash
cd evals/ragas
pip install -r requirements.txt
```

## Set API Key

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Generate Your First Testset

### Option 1: Quick Test (20 samples)

```bash
python generate_testset.py --testset-size 20 --max-documents 50
```

This will:
- Load 50 documents from Elasticsearch
- Generate 20 synthetic test questions
- Save results to `output/testset.csv`
- Create a generation report

**Expected time**: 2-5 minutes

### Option 2: Standard Testset (100 samples)

```bash
python generate_testset.py
```

**Expected time**: 5-10 minutes

### Option 3: Large Testset (500 samples)

```bash
python generate_testset.py --testset-size 500 --max-documents 1000
```

**Expected time**: 20-30 minutes

## Review Results

Check the generated files:

```bash
# View generation report
cat output/generation_report.txt

# View testset (first 10 lines)
head -n 10 output/testset.csv

# View statistics
cat output/generation_stats.json
```

## Example Output

### Testset Preview

```csv
question,ground_truth,contexts,evolution_type
"What was the total revenue reported in Q4 2024?","The total revenue for Q4 2024 was $52.9 billion...","['In the fourth quarter of 2024...']",simple
"How did cloud revenue growth compare to the previous quarter?","Cloud revenue grew 24% compared to 18% in Q3...","['Cloud services revenue...', 'Previous quarter...']",reasoning
```

### Generation Report

```
================================================================================
Ragas Synthetic Testset Generation Report
================================================================================

Generated: 2024-10-18T16:00:00
Generation Time: 187.45 seconds

Models:
  Generator: gpt-4o-mini
  Critic: gpt-4o-mini
  Embeddings: text-embedding-3-small

Dataset:
  Total Samples: 100
  Source Documents: 500

Question Statistics:
  Average Length: 87.3 chars
  Min Length: 42 chars
  Max Length: 245 chars
```

## Next Steps

### 1. Validate Quality

Review sample questions in `output/generation_report.txt`:
- Are questions relevant to your domain?
- Are ground truth answers accurate?
- Are contexts appropriate?

### 2. Adjust Configuration

Edit `config.py` to customize:

```python
RAGAS_CONFIG = {
    "testset_size": 200,  # Change size
    
    # Adjust query distribution
    "distributions": {
        "simple": 0.5,      # More simple queries
        "reasoning": 0.3,
        "multi_context": 0.15,
        "conditional": 0.05,
    },
}
```

### 3. Evaluate Your RAG System

```python
from evaluate_rag import RagasEvaluator

evaluator = RagasEvaluator()
testset_df = evaluator.load_testset("output/testset.csv")

# Generate answers with your RAG system
for _, row in testset_df.iterrows():
    question = row["question"]
    answer, contexts = your_rag_system.query(question)
    # Collect results...

# Evaluate
results = evaluator.evaluate(eval_dataset)
```

## Common Issues

### Rate Limits

If you hit OpenAI rate limits:

```bash
# Generate smaller batches
python generate_testset.py --testset-size 25
```

### Connection Issues

Test Elasticsearch connection:

```bash
python elasticsearch_loader.py
```

Should show:
```
Successfully connected to Elasticsearch
Index Statistics:
  Total documents: 1234
  Index size: 5678900 bytes
```

### Empty Results

If no questions are generated:
- Check that documents have text content
- Verify OpenAI API key is valid
- Review logs in `ragas_generation.log`

## Tips for Best Results

### 1. Start Small
```bash
python generate_testset.py --testset-size 10 --max-documents 20
```
Review quality before scaling up.

### 2. Use Representative Sampling
```bash
python generate_testset.py --sample-strategy representative
```
Gets diverse documents across categories.

### 3. Monitor Costs
- ~$0.50-1.00 per 100 questions with gpt-4o-mini
- Scale up after validating quality

### 4. Iterate on Distribution
Adjust based on your use case:
- **Customer Support**: More simple queries
- **Research/Analysis**: More reasoning queries
- **Multi-document QA**: More multi-context queries

## Full Example Workflow

```bash
# 1. Install
cd evals/ragas
pip install -r requirements.txt

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Test connection
python elasticsearch_loader.py

# 4. Generate small testset
python generate_testset.py --testset-size 20 --max-documents 50

# 5. Review results
cat output/generation_report.txt

# 6. If quality is good, generate larger testset
python generate_testset.py --testset-size 200 --max-documents 500

# 7. Use testset to evaluate your RAG
# (integrate with your RAG system)
```

## Success Criteria

You'll know it's working when:
- ✓ Generation completes without errors
- ✓ Questions are relevant to your domain
- ✓ Ground truth answers are accurate
- ✓ Contexts contain relevant information
- ✓ Query types are diverse (simple, reasoning, multi-context)

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Review troubleshooting section
- Check logs: `ragas_generation.log`
- Test components individually with test scripts

## Comparison Mode

Want to compare with existing `synthetic-eval`?

```bash
# Generate Ragas testset
cd evals/ragas
python generate_testset.py --testset-size 100

# Compare with existing
cd ../synthetic-eval
python main.py  # Your existing eval

# Compare outputs manually or create comparison script
```

---

**Ready to generate?** Run:
```bash
python generate_testset.py --testset-size 50
```

Your testset will be ready in `output/testset.csv` in about 3-5 minutes! 🚀

