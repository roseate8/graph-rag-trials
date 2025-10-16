# Calculate-Metrics Architecture Documentation

## TRUE 100% Dependency on retrieval/core.py

**Status**: ✅ ACHIEVED

This document explains how the calculate-metrics evaluation system achieves TRUE 100% dependency on `retrieval/core.py`, meaning adding new features to the RAG system automatically works in evaluations without code changes.

---

## Architecture Overview

### **kwargs Pass-Through Pattern

The system uses a **kwargs pass-through pattern to ensure that ANY parameters added to `RAGSystem.__init__()` automatically flow through to the evaluation system.

```
┌─────────────────────────────────────────────────────────────────┐
│                     retrieval/core.py                           │
│  RAGSystem.__init__(embedding_model, enable_reranking, ...)    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ **kwargs pass-through
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              calculate-metrics/config.py                        │
│  EvalConfig.rag_system_params = {                              │
│      'embedding_model': 'BAAI/bge-m3',                         │
│      'enable_reranking': True,                                 │
│      'enable_verification_agent': True,  # NEW FEATURE         │
│      ...                                                        │
│  }                                                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      │ Pass dict to retriever
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│         calculate-metrics/retriever_for_evals.py                │
│  RAGSystem(**config.rag_system_params)                         │
│  # Automatically passes ALL parameters, including new ones!     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files and Changes

### 1. config.py

**Old Approach** (explicit parameters):
```python
@dataclass
class EvalConfig:
    collection_name: str = "elastic_embeddings_m3"
    embedding_model: str = "BAAI/bge-m3"
    enable_reranking: bool = True
    # ... 15+ individual parameters
```

**New Approach** (dict-based pass-through):
```python
@dataclass
class EvalConfig:
    # All RAGSystem parameters in a single dict
    rag_system_params: Dict[str, Any] = field(default_factory=lambda: {
        'collection_name': 'elastic_embeddings_m3',
        'embedding_model': 'BAAI/bge-m3',
        'enable_reranking': True,
        'enable_query_decomposition': False,
        # ... all other RAGSystem params
    })

    # Evaluation-specific params (NOT passed to RAGSystem)
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    batch_size: int = 15
    max_concurrent: int = 15
```

**Benefits**:
- Adding new RAGSystem params requires ONE line in the dict
- Clear separation between RAG params and eval params
- Self-documenting via comments in the dict

### 2. retriever_for_evals.py

**Old Approach** (explicit parameter passing):
```python
def connect(self) -> bool:
    self.rag_system = RAGSystem(
        embedding_model=self.config.embedding_model,
        collection_name=self.config.collection_name,
        enable_reranking=self.config.enable_reranking,
        # ... 15+ individual parameters
    )
```

**New Approach** (kwargs pass-through):
```python
def connect(self) -> bool:
    # **kwargs pass-through - automatically includes ALL params!
    self.rag_system = RAGSystem(**self.config.rag_system_params)
```

**Benefits**:
- ZERO maintenance when RAGSystem signature changes
- No risk of forgetting to pass a parameter
- Automatic support for new features

### 3. main.py

**Old Approach** (modify attributes):
```python
config.collection_name = args.collection
config.embedding_model = args.embedding_model
config.enable_reranking = not args.no_reranking
```

**New Approach** (modify dict):
```python
# Update RAGSystem parameters via dict
config.rag_system_params['collection_name'] = args.collection
config.rag_system_params['embedding_model'] = args.embedding_model
config.rag_system_params['enable_reranking'] = not args.no_reranking
```

**Benefits**:
- Consistent with overall architecture
- Clear that these params go to RAGSystem
- CLI flags automatically work with new params

---

## Future Workflow: Adding New Features

### Scenario: You add a new verification agent to RAGSystem

**Step 1**: Add parameter to RAGSystem in retrieval/core.py
```python
class RAGSystem:
    def __init__(
        self,
        # ... existing params
        enable_verification_agent: bool = False,  # NEW
        verification_threshold: float = 0.8       # NEW
    ):
        self.enable_verification_agent = enable_verification_agent
        self.verification_threshold = verification_threshold
        # ... implementation
```

**Step 2** (Optional): Add to config defaults in config.py
```python
rag_system_params: Dict[str, Any] = field(default_factory=lambda: {
    # Existing params...
    'enable_reranking': True,

    # NEW - one line to add!
    'enable_verification_agent': True,
    'verification_threshold': 0.8,
})
```

**Step 3** (Optional): Add CLI flag in main.py
```python
eval_group.add_argument(
    '--enable-verification',
    action='store_true',
    help='Enable verification agent for retrieved results'
)

# In create_config_from_args:
config.rag_system_params['enable_verification_agent'] = args.enable_verification
```

**Step 4**: **NO CHANGES NEEDED** in retriever_for_evals.py
- The **kwargs pattern automatically passes new params
- Feature works immediately in evaluations!

### What Changed from Before

**Before** (explicit parameter passing):
- Had to modify retriever_for_evals.py to pass new param
- Easy to forget parameters
- Tedious maintenance

**After** (kwargs pass-through):
- NO changes needed in retriever_for_evals.py
- Automatic parameter discovery
- Zero maintenance

---

## Testing

Run the test suite to verify the architecture:

```bash
cd evals/synthetic-eval/calculate-metrics
python test_passthrough.py
```

Expected output:
```
[OK] Config uses rag_system_params dict
[OK] All existing parameters present
[OK] New parameters can be added to dict
[OK] **kwargs passes all parameters to RAGSystem
[OK] CLI overrides work via dict modification
```

---

## Benefits Summary

1. **Zero Maintenance**: Adding new RAGSystem features requires no changes to retriever wrapper code
2. **Automatic Discovery**: All parameters automatically flow through via **kwargs
3. **Type Safety**: Parameters are validated by RAGSystem's type hints
4. **Clear Separation**: RAG params vs evaluation params clearly separated
5. **CLI Flexibility**: Optional CLI flags for runtime control
6. **Self-Documenting**: Comments in config dict document each parameter's purpose

---

## Example: Current RAGSystem Parameters

As of this refactoring, the following parameters are passed to RAGSystem:

**Retriever Parameters**:
- `embedding_model`: Embedding model for semantic search
- `collection_name`: Milvus collection name

**Re-ranking Parameters**:
- `enable_reranking`: Enable two-stage retrieval with re-ranking
- `reranker_config`: Custom re-ranker configuration
- `retrieval_multiplier`: Initial retrieval multiplier (default: 10)

**Query Decomposition Parameters**:
- `enable_query_decomposition`: Enable multi-query retrieval with fusion
- `max_sub_queries`: Maximum number of sub-queries to generate
- `fusion_k_constant`: K constant for reciprocal rank fusion

**Context Formatting Parameters**:
- `max_context_tokens`: Maximum tokens for context
- `include_scores`: Include similarity scores in formatted context

**LLM Parameters**:
- `llm_type`: LLM client type (mock for evaluation)
- `llm_model`: Model name

**History Parameters**:
- `enable_history`: Enable conversation history
- `history_file`: History file path

**Any new parameters added to RAGSystem automatically work here!**

---

## Architecture Decision Record

**Date**: 2025-10-16

**Decision**: Use **kwargs pass-through pattern for RAGSystem parameters

**Context**: User wanted TRUE 100% dependency on retrieval/core.py such that adding new features (e.g., verification agents, rethink functions) would automatically work in evaluations without modifying calculate-metrics code.

**Alternatives Considered**:
1. **Explicit parameter passing** (original approach)
   - Requires code changes in retriever wrapper
   - Easy to forget parameters

2. **Configuration inheritance via inspect module**
   - Too complex
   - Breaks when RAGSystem signature changes

3. **Dict-based kwargs pass-through** (chosen approach)
   - Simple and explicit
   - Zero maintenance for new features
   - Self-documenting via dict structure

**Consequences**:
- Adding new RAGSystem features requires zero changes to retriever wrapper
- Clear separation between RAG and evaluation parameters
- Breaking change to config structure (migration required)
- Loss of IDE autocomplete for config attributes (trade-off accepted)

**Status**: Implemented and tested

---

## Related Documentation

- REDUNDANCIES_FOUND.md - Original redundancy analysis
- REFACTORING_COMPLETE.md - First refactoring documentation
- ../../../retrieval/README.md - RAG system documentation
