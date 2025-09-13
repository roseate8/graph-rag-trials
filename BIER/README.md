# BIER - BEIR Evaluation for GraphRAG

Simple evaluation of your retrieval pipeline using BEIR benchmarks.

## Quick Start

1. **Install:** `pip install -r requirements.txt`
2. **Run:** `python quick_start.py`

## Files

- **`quick_start.py`** - Run this first! Simple evaluation script
- **`run_evaluation.py`** - Advanced evaluation with options
- **`requirements.txt`** - Dependencies to install
- **`config/eval_config.yaml`** - Settings (datasets, models, etc.)
- **`results/`** - Your results appear here

## Core Files (You Don't Need to Edit These)

- **`graph_rag_adapter.py`** - Connects to your existing pipeline  
- **`beir_evaluator.py`** - BEIR evaluation logic
- **`config_utils.py`** - Configuration utilities

That's it! Just run `python quick_start.py` to get started.