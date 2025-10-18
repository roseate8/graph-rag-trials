# Ragas Implementation Architecture

Visual overview of the Ragas-based synthetic data generation system.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ragas Test Generation System                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Data Source     │
│                  │
│  Elasticsearch   │
│  M3 Collection   │
│                  │
│  rudram-         │
│  embeddings      │
│  (768-dim)       │
└────────┬─────────┘
         │
         │ HTTPS Connection
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  elasticsearch_loader.py                                        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  ElasticsearchDocumentLoader                             │ │
│  │  • Connect to Elasticsearch                              │ │
│  │  • Load documents (random/representative sampling)       │ │
│  │  • Convert to LangChain Document format                  │ │
│  │  • Preserve metadata                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────────────────────┘
         │
         │ List[Document]
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  generate_testset.py                                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  RagasTestsetGenerator                                   │ │
│  │                                                          │ │
│  │  ┌─────────────────┐       ┌─────────────────┐        │ │
│  │  │  Generator LLM  │       │   Critic LLM    │        │ │
│  │  │  (gpt-4o-mini) │       │  (gpt-4o-mini)  │        │ │
│  │  └─────────────────┘       └─────────────────┘        │ │
│  │           │                          │                  │ │
│  │           v                          v                  │ │
│  │  ┌──────────────────────────────────────────────┐     │ │
│  │  │     Ragas TestsetGenerator Core              │     │ │
│  │  │  • Build Knowledge Graph from docs           │     │ │
│  │  │  • Apply evolutionary transformations        │     │ │
│  │  │  • Generate Q-A-Context triplets             │     │ │
│  │  │  • Filter with critic model                  │     │ │
│  │  └──────────────────────────────────────────────┘     │ │
│  │                                                          │ │
│  │  Query Types Generated:                                 │ │
│  │  • Simple (40%) - Single-hop queries                   │ │
│  │  • Reasoning (30%) - Multi-hop reasoning               │ │
│  │  • Multi-context (20%) - Cross-document queries        │ │
│  │  • Conditional (10%) - Complex logic                   │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────────────────────┘
         │
         │ Testset (Q-A-Context triplets)
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  Output Generation                                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  output/                                                 │ │
│  │  ├── testset.csv          # Main testset (CSV)          │ │
│  │  ├── testset.json         # Main testset (JSON)         │ │
│  │  ├── generation_report.txt # Human-readable report      │ │
│  │  └── generation_stats.json # Detailed statistics        │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────────────────────┘
         │
         │ Load Testset
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  evaluate_rag.py                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  RagasEvaluator                                          │ │
│  │                                                          │ │
│  │  Input: Testset + Your RAG System Responses             │ │
│  │                                                          │ │
│  │  Metrics:                                                │ │
│  │  • Faithfulness         (Factual accuracy)              │ │
│  │  • Answer Relevancy     (Question-answer relevance)     │ │
│  │  • Context Recall       (Ground truth coverage)         │ │
│  │  • Context Precision    (Relevant context ratio)        │ │
│  │  • Context Relevancy    (Context-question relevance)    │ │
│  │  • Answer Similarity    (Semantic similarity)           │ │
│  │  • Answer Correctness   (F1 score)                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────┬───────────────────────────────────────────────────────┘
         │
         │ Evaluation Results
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  Evaluation Output                                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  output/evaluation/                                      │ │
│  │  ├── evaluation_results.json  # Metrics scores           │ │
│  │  └── evaluation_report.txt    # Human-readable report   │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Elasticsearch          LangChain             Ragas            Output
Documents       →     Documents       →    Testset      →    CSV/JSON
  (JSON)              (Objects)          (Triplets)         (Files)

┌─────────┐        ┌───────────┐       ┌──────────┐      ┌──────────┐
│ {       │        │ Document( │       │ question │      │ testset  │
│  text:  │   →    │  content, │  →    │ answer   │  →   │ .csv     │
│  meta   │        │  metadata │       │ contexts │      │ .json    │
│ }       │        │ )         │       │ type     │      │          │
└─────────┘        └───────────┘       └──────────┘      └──────────┘
```

## Component Interactions

```
┌─────────────┐
│  config.py  │ ─────────────┐
└─────────────┘              │
       │                     │ Configuration
       │                     │
       v                     v
┌──────────────────┐  ┌──────────────────┐
│ elasticsearch_   │  │ generate_        │
│ loader.py        │→ │ testset.py       │
└──────────────────┘  └──────────────────┘
                             │
                             │ Testset
                             │
                             v
                      ┌──────────────────┐
                      │ evaluate_        │
                      │ rag.py           │
                      └──────────────────┘
```

## File Responsibilities

```
┌────────────────────────────────────────────────────────────────┐
│  Core Implementation                                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  config.py                                                      │
│  • Elasticsearch connection config                             │
│  • LLM and embeddings config                                   │
│  • Generation parameters                                       │
│  • Configuration validation                                    │
│                                                                 │
│  elasticsearch_loader.py                                        │
│  • Connect to Elasticsearch                                    │
│  • Load and sample documents                                   │
│  • Convert to LangChain format                                 │
│  • Test connection utilities                                   │
│                                                                 │
│  generate_testset.py                                            │
│  • Initialize Ragas generator                                  │
│  • Generate synthetic testset                                  │
│  • Save results (CSV/JSON)                                     │
│  • Generate reports and statistics                             │
│  • CLI interface                                               │
│                                                                 │
│  evaluate_rag.py                                                │
│  • Load testset                                                │
│  • Prepare evaluation dataset                                  │
│  • Run Ragas evaluation metrics                                │
│  • Save and report results                                     │
│  • Mock evaluation mode                                        │
│                                                                 │
│  __init__.py                                                    │
│  • Package initialization                                      │
│  • Clean public API                                            │
│  • Version management                                          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Configuration Flow

```
┌──────────────┐
│ Environment  │
│ Variables    │
│              │
│ OPENAI_      │
│ API_KEY      │
└──────┬───────┘
       │
       v
┌──────────────────────────────────────────────────────────────┐
│  config.py                                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  ELASTICSEARCH_CONFIG                                  │ │
│  │  • url, username, password, index_name                 │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  RAGAS_CONFIG                                          │ │
│  │  • testset_size, models, distributions                 │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  OPENAI_CONFIG / AZURE_CONFIG                          │ │
│  │  • api_key, organization, timeout                      │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  OUTPUT_CONFIG                                         │ │
│  │  • output_dir, filenames                               │ │
│  └────────────────────────────────────────────────────────┘ │
└────────┬─────────────────────────────────────────────────────┘
         │
         │ Import
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  All Modules                                                    │
└────────────────────────────────────────────────────────────────┘
```

## Testset Generation Pipeline

```
Step 1: Load Documents
┌─────────────────────────────────────────┐
│ Elasticsearch → Documents (500)         │
│ • Random or representative sampling     │
│ • Metadata preservation                 │
└─────────────┬───────────────────────────┘
              │
              v
Step 2: Build Knowledge Graph
┌─────────────────────────────────────────┐
│ Documents → Knowledge Graph             │
│ • Extract entities and relationships    │
│ • Build document connections            │
└─────────────┬───────────────────────────┘
              │
              v
Step 3: Generate Questions
┌─────────────────────────────────────────┐
│ KG → Candidate Questions                │
│ • Apply evolutionary transformations    │
│ • Simple, reasoning, multi-context      │
│ • Generate ground truth answers         │
└─────────────┬───────────────────────────┘
              │
              v
Step 4: Filter with Critic
┌─────────────────────────────────────────┐
│ Candidates → Filtered Questions (100)   │
│ • Quality assessment                    │
│ • Relevance filtering                   │
│ • Diversity selection                   │
└─────────────┬───────────────────────────┘
              │
              v
Step 5: Save Testset
┌─────────────────────────────────────────┐
│ Questions → CSV/JSON + Reports          │
│ • testset.csv / testset.json            │
│ • generation_report.txt                 │
│ • generation_stats.json                 │
└─────────────────────────────────────────┘
```

## Evaluation Pipeline

```
Step 1: Load Testset
┌─────────────────────────────────────────┐
│ testset.csv → DataFrame                 │
└─────────────┬───────────────────────────┘
              │
              v
Step 2: Generate RAG Responses
┌─────────────────────────────────────────┐
│ Questions → Your RAG System             │
│ • For each question                     │
│ • Generate answer + retrieve contexts   │
└─────────────┬───────────────────────────┘
              │
              v
Step 3: Prepare Evaluation Dataset
┌─────────────────────────────────────────┐
│ Responses + Ground Truth → Dataset      │
│ • question, answer, contexts, truth     │
└─────────────┬───────────────────────────┘
              │
              v
Step 4: Run Ragas Metrics
┌─────────────────────────────────────────┐
│ Dataset → Metric Scores                 │
│ • Faithfulness                          │
│ • Answer Relevancy                      │
│ • Context Recall/Precision              │
│ • Answer Similarity/Correctness         │
└─────────────┬───────────────────────────┘
              │
              v
Step 5: Generate Report
┌─────────────────────────────────────────┐
│ Scores → evaluation_report.txt          │
│ • Metric summaries                      │
│ • Recommendations                       │
└─────────────────────────────────────────┘
```

## Query Type Evolution

```
Source Document
       │
       ├─→ Simple Evolution
       │   └→ "What is X?"
       │
       ├─→ Reasoning Evolution
       │   └→ "How does X relate to Y?"
       │
       ├─→ Multi-Context Evolution
       │   └→ "Compare X across documents A, B, C"
       │
       └─→ Conditional Evolution
           └→ "If X, then what would be Y?"
```

## Integration Points

```
┌────────────────────────────────────────────────────────────────┐
│  External Systems Integration                                   │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Elasticsearch (rudram-embeddings)                             │
│  • Pre-configured connection                                   │
│  • 768-dim embeddings                                          │
│  • HTTPS with auth                                             │
│                                                                 │
│  OpenAI API                                                    │
│  • gpt-4o-mini for generation                                  │
│  • text-embedding-3-small for embeddings                       │
│  • Configurable models                                         │
│                                                                 │
│  Your RAG System (to be integrated)                            │
│  • Load testset                                                │
│  • Query for each question                                     │
│  • Return answer + contexts                                    │
│  • Evaluate with Ragas metrics                                 │
│                                                                 │
│  LangChain                                                     │
│  • Document format                                             │
│  • LLM interfaces                                              │
│  • Embeddings interfaces                                       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Scalability

```
Documents      Questions      Time         Cost
─────────────────────────────────────────────────
   50      →     20       ~  2 min     ~  $0.10
  100      →     50       ~  4 min     ~  $0.25
  500      →    100       ~  8 min     ~  $0.60
 1000      →    200       ~ 15 min     ~  $1.20
 2000      →    500       ~ 35 min     ~  $3.00
```

## Error Handling Flow

```
┌─────────────────┐
│ Configuration   │
│ Validation      │
└────────┬────────┘
         │ ✓
         v
┌─────────────────┐     ✗      ┌──────────────────┐
│ Elasticsearch   │─────────────→ Connection Error │
│ Connection      │              │ • Check network  │
└────────┬────────┘              │ • Check creds    │
         │ ✓                     └──────────────────┘
         v
┌─────────────────┐     ✗      ┌──────────────────┐
│ Document        │─────────────→ Loading Error    │
│ Loading         │              │ • Check index    │
└────────┬────────┘              │ • Check query    │
         │ ✓                     └──────────────────┘
         v
┌─────────────────┐     ✗      ┌──────────────────┐
│ Testset         │─────────────→ Generation Error │
│ Generation      │              │ • Check API key  │
└────────┬────────┘              │ • Check rate     │
         │ ✓                     └──────────────────┘
         v
┌─────────────────┐
│ Success         │
│ • Testset saved │
│ • Report gen    │
└─────────────────┘
```

## Deployment Architecture

```
Development Environment
┌─────────────────────────────────────────┐
│  evals/ragas/                           │
│  ├── Python 3.8+                        │
│  ├── Dependencies (requirements.txt)    │
│  ├── Config (config.py)                 │
│  └── Scripts (*.py)                     │
└─────────────┬───────────────────────────┘
              │
              │ HTTPS
              │
              v
┌─────────────────────────────────────────┐
│  External Services                      │
│  ┌───────────────────────────────────┐ │
│  │ Elasticsearch M3 (Cloud)          │ │
│  │ • rudram-embeddings index         │ │
│  └───────────────────────────────────┘ │
│  ┌───────────────────────────────────┐ │
│  │ OpenAI API                        │ │
│  │ • gpt-4o-mini                     │ │
│  │ • text-embedding-3-small          │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Summary

The architecture follows a modular, pipeline-based design:

1. **Document Loading**: Flexible sampling from Elasticsearch
2. **Generation**: Ragas framework for diverse queries
3. **Evaluation**: Standard metrics for RAG quality
4. **Reporting**: Comprehensive reports and statistics

All components are loosely coupled through clear interfaces, making the system:
- **Maintainable**: Each module has single responsibility
- **Extensible**: Easy to add new features
- **Testable**: Each component can be tested independently
- **Configurable**: Behavior controlled through config.py
- **Reusable**: Components can be imported as library

The system integrates seamlessly with existing Elasticsearch infrastructure and provides a standard evaluation framework for RAG systems.

