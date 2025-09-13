# BIER - Optimized BEIR Evaluation for Naive RAG

Streamlined evaluation of your naive RAG retrieval pipeline using BEIR benchmarks.

## Quick Start

1. **Install:** `pip install -r requirements.txt`
2. **Run HotpotQA Evaluation:** `python hotpotqa_evaluator.py`

## Files

- **`hotpotqa_evaluator.py`** - Main evaluation script for HotpotQA dataset
- **`graph_rag_adapter.py`** - Optimized adapter for naive RAG integration
- **`config_utils.py`** - Configuration utilities
- **`run_evaluation.py`** - Advanced evaluation with multiple options
- **`config/eval_config.yaml`** - Configuration settings
- **`data/hotpotqa/hotpotqa.zip`** - HotpotQA dataset (preserved)
- **`results/`** - Evaluation results appear here

## Usage

### HotpotQA Evaluation
```bash
python hotpotqa_evaluator.py
```

### Custom Evaluation
```bash
python run_evaluation.py --dataset hotpotqa --collection-name your_collection
```

## Features

- ✅ Optimized for performance and efficiency
- ✅ Batch processing for faster evaluation
- ✅ Comprehensive BEIR metrics (NDCG, MAP, Precision, Recall)
- ✅ Automatic fallback to default collection
- ✅ Minimal dependencies and clean codebase