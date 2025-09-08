# RAG Evaluation Framework

This directory contains evaluation tools and frameworks for assessing the performance of the RAG (Retrieval-Augmented Generation) system.

## Structure

```
evals/
├── deepeval/          # DeepEval framework setup and tests
│   ├── metrics/       # Custom evaluation metrics
│   ├── test_suites/   # Organized test suites
│   └── configs/       # Configuration files
├── ragas/             # RAGAS framework (placeholder)
└── shared/            # Shared utilities and datasets
```

## Evaluation Frameworks

### DeepEval
- **Purpose**: Comprehensive LLM application testing
- **Focus**: RAG system evaluation, customer support chatbot testing
- **Features**: Custom metrics, automated testing, performance benchmarking

### RAGAS (Coming Soon)
- **Purpose**: RAG-specific evaluation metrics
- **Focus**: Retrieval and generation quality assessment

## Quick Start

### DeepEval Setup
```bash
cd evals/deepeval
pip install -r requirements.txt
python setup_deepeval.py
```

### Environment Configuration
Copy `.env.example` to `.env` and configure your API keys and settings.

## Integration

The evaluation framework is designed to be:
- **Self-contained**: All evaluation code remains in this directory
- **Callable**: Can be invoked from other parts of the system (e.g., naive-rag)
- **Modular**: Easy to add new evaluation frameworks and metrics
